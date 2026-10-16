"""Smoke test: the PyO3 binding round-trips a small plan.

Requires `vecio_planner` to be built in the active environment.
"""
from __future__ import annotations

import pytest

import vecio_planner as vp  # type: ignore[import-not-found]


@pytest.mark.parametrize(
    "strategy", ["separate", "greedy", "serial", "bottleneck", "main_solution"]
)
def test_plan_covers_every_request(strategy):
    requests = [
        (0, 0, 1024),
        (1, 1024, 1024),
        (2, 5_000_000, 1024),
    ]
    kwargs = dict(
        latency_per_request=27.0,
        latency_per_byte=0.0,
        concurrent_parallelism=8.0,
    )
    plan = vp.plan(requests, strategy, **kwargs)
    assert "merged_requests" in plan
    assert "assignments" in plan
    assert len(plan["assignments"]) == len(requests)
    for (off, length) in plan["merged_requests"]:
        assert length > 0
        assert off >= 0

    metrics = vp.evaluate_plan(requests, plan, **kwargs)
    assert metrics["useful_bytes"] == sum(r[2] for r in requests)
    assert metrics["total_bytes"] >= metrics["useful_bytes"]
    assert metrics["request_count"] == len(plan["merged_requests"])


def test_unknown_strategy_raises():
    with pytest.raises(ValueError):
        vp.plan([(0, 0, 1024)], "not_a_strategy")


def test_greedy_knobs_round_trip():
    # Three adjacent clusters with 2048-byte gaps between them.
    requests = [
        (0, 0, 1024),
        (1, 3072, 1024),
        (2, 6144, 1024),
    ]
    base_kwargs = dict(latency_per_request=27.0, latency_per_byte=0.0)

    # Zero gap threshold forbids any merge → same as Separate.
    plan_cut = vp.plan(requests, "greedy", greedy_min_gap_bytes=0, **base_kwargs)
    assert len(plan_cut["merged_requests"]) == len(requests)

    # Large gap threshold allows every merge → one merged request.
    plan_join = vp.plan(
        requests, "greedy", greedy_min_gap_bytes=1 << 20, **base_kwargs
    )
    assert len(plan_join["merged_requests"]) == 1

    # Span cap forces a split even when the gap is acceptable.
    plan_span = vp.plan(
        requests,
        "greedy",
        greedy_min_gap_bytes=1 << 20,
        greedy_max_span_bytes=3500,
        **base_kwargs,
    )
    assert len(plan_span["merged_requests"]) >= 2


@pytest.mark.parametrize(
    "strategy", ["separate", "serial", "bottleneck", "main_solution"]
)
def test_greedy_kwargs_rejected_on_non_greedy_strategy(strategy):
    with pytest.raises(ValueError, match="greedy_"):
        vp.plan(
            [(0, 0, 1024)],
            strategy,
            latency_per_request=27.0,
            latency_per_byte=0.0,
            greedy_min_gap_bytes=4096,
        )


@pytest.mark.parametrize(
    ("strategy", "kwargs"),
    [
        ("greedy", {"max_requests": 1}),
        ("greedy", {"exact_requests": True}),
        ("separate", {"max_requests": 1}),
        ("separate", {"exact_requests": True}),
    ],
)
def test_shared_constraints_rejected_on_unsupported_strategy(strategy, kwargs):
    with pytest.raises(ValueError, match="max_requests|exact_requests"):
        vp.plan(
            [(0, 0, 1024)],
            strategy,
            latency_per_request=27.0,
            latency_per_byte=0.0,
            **kwargs,
        )


@pytest.mark.parametrize("strategy", ["separate", "greedy"])
def test_explicit_default_shared_kwargs_accepted(strategy):
    # `max_requests=0` and `exact_requests=False` are the neutral
    # "no constraint" values; passing them explicitly must not error
    # even for strategies that do not honour the constraint.
    vp.plan(
        [(0, 0, 1024)],
        strategy,
        latency_per_request=27.0,
        latency_per_byte=0.0,
        max_requests=0,
        exact_requests=False,
    )


def test_legacy_max_merge_span_kwarg_raises():
    with pytest.raises(ValueError, match="max_merge_span"):
        vp.plan(
            [(0, 0, 1024)],
            "main_solution",
            latency_per_request=27.0,
            max_merge_span=1 << 20,
        )


def test_evaluate_plan_rejects_impossible_plan():
    with pytest.raises(ValueError, match="plan:"):
        vp.evaluate_plan(
            [(0, 0, 1024)],
            {"merged_requests": [(2048, 1024)], "assignments": [0]},
        )


def test_negative_max_requests_raises():
    with pytest.raises(ValueError, match="max_requests must be >= 0, got -1"):
        vp.plan([(0, 0, 1024)], "main_solution", max_requests=-1)


def test_exact_requests_requires_positive_cap():
    with pytest.raises(ValueError, match="exact_requests=true requires max_requests > 0"):
        vp.plan([(0, 0, 1024)], "main_solution", exact_requests=True)


def test_plan_rejects_malformed_piecewise_cost():
    with pytest.raises(ValueError, match="cost model:"):
        vp.plan(
            [(0, 0, 1024)],
            "main_solution",
            latency_model="piecewise_linear",
            latency_pw_breaks_bytes=[1024],
            latency_pw_overheads_ms=[27.0],
            latency_pw_rates_ms_per_byte=[0.0, 0.0],
        )


def test_plan_accepts_continuity_adjusted_piecewise_overhead():
    plan = vp.plan(
        [(0, 0, 1024), (1, 8_000_000, 1024)],
        "main_solution",
        latency_model="piecewise_linear",
        latency_pw_breaks_bytes=[4_194_304],
        latency_pw_overheads_ms=[10.0, -31.5236096],
        latency_pw_rates_ms_per_byte=[1e-7, 1e-5],
        concurrent_parallelism=4.0,
    )
    assert len(plan["assignments"]) == 2


def test_calibrator_roundtrip():
    cal = vp.Calibrator()
    # Synthetic samples: latency = 27 + 1e-5 * bytes.
    for b in (1024, 4096, 65536, 262144, 1048576, 4194304):
        cal.record(b, 27.0 + 1e-5 * b)
    assert cal.sample_count() == 6

    o, r = cal.fit_linear()
    assert 26.0 < o < 28.0, f"overhead {o} out of expected band"
    assert 0.9e-5 < r < 1.1e-5, f"rate {r} out of expected band"

    cm = cal.to_cost_model(concurrent_parallelism=8.0)
    assert cm["concurrent_parallelism"] == 8.0
    assert cm["latency_per_request"] > 0


@pytest.mark.skipif(vp is None, reason="vecio_planner not built; run `maturin develop`")
def test_metrics_include_t_ub():
    requests = [(0, 0, 1024), (1, 1024, 1024)]
    plan = vp.plan(requests, "separate")
    kwargs = dict(latency_per_request=10.0, latency_per_byte=0.0)

    m = vp.evaluate_plan(requests, plan, concurrent_parallelism=4.0, **kwargs)
    w, b = m["predicted_latency_ms_serial"], m["predicted_latency_ms_concurrent"]
    assert m["predicted_latency_ms_t_ub"] == pytest.approx(w / 4.0 + 0.75 * b)

    # P <= 0 means full parallelism: T_UB degenerates to the bottleneck.
    m0 = vp.evaluate_plan(requests, plan, concurrent_parallelism=0.0, **kwargs)
    assert m0["predicted_latency_ms_t_ub"] == pytest.approx(b)
