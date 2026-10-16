# vecio-planner (Python)

PyO3 bindings over the `warp-planner` Rust crate in this repository.
The package is pre-release and is not yet distributed on PyPI.

## Install

After the first release, installation will be:

```bash
pip install vecio-planner
```

For now, build locally (requires Rust 1.82+ and
[maturin](https://www.maturin.rs/)):

```bash
# 1. Install maturin into a venv (maturin can't write to system Python
#    and needs a venv active to install the built extension into).
python3 -m venv python/.venv
source python/.venv/bin/activate
pip install maturin pytest

# 2. Build + install the extension.
cd python
maturin develop --release
```

## API

```python
import vecio_planner as vp

requests = [(0, 0, 8192), (1, 8192, 8192), (2, 5_000_000, 8192)]
cost_kwargs = dict(
    latency_per_request=27.0,
    latency_per_byte=1.0e-5,
    concurrent_parallelism=16.0,
)
plan_kwargs = dict(
    max_requests=16,
)

# Plan. Strategies:
#   "separate", "greedy", "serial", "bottleneck", "main_solution"
plan = vp.plan(requests, "main_solution", **cost_kwargs, **plan_kwargs)
# plan == {"merged_requests": [(offset, length), ...],
#          "assignments":     [int, ...]}

# Score any plan against the same cost model.
metrics = vp.evaluate_plan(requests, plan, **cost_kwargs)
```

Cost-model keyword arguments mirror the Rust `CostModel` struct:

| kwarg | meaning |
|---|---|
| `latency_model` | `"linear"` or `"piecewise_linear"` |
| `latency_per_request`, `latency_per_byte` | linear-model latency |
| `latency_pw_breaks_bytes`, `latency_pw_overheads_ms`, `latency_pw_rates_ms_per_byte` | piecewise-model latency (vectors) |
| `concurrent_parallelism` | expected number of concurrent range reads used by `MainSolution` |
| `cost_per_request`, `cost_per_byte`, `cost_per_wasted_byte` | dollar costs (e.g. AWS S3 `$4e-7` per GET); used by your own scoring |

Planning kwargs are separate from the cost model:

| kwarg | meaning |
|---|---|
| `max_requests`, `exact_requests` | shared plan-shape constraint for `serial` / `bottleneck` / `main_solution`; passing them to `separate` / `greedy` raises `ValueError` |
| `greedy_min_gap_bytes`, `greedy_max_span_bytes` | `Greedy`-only knobs; omitted → Rust defaults; passing them to other strategies raises `ValueError` |
| `pre_merge_max_gap_bytes`, `pre_merge_max_merged_size_bytes` | `PreMerge`-only knobs. Default `(0, 0)` → no merging; `PreMerge` is opt-in. **Recommended preset for opt-in callers**: `pre_merge_max_gap_bytes=32*1024, pre_merge_max_merged_size_bytes=1024*1024` (32 KiB gap, 1 MiB cap), useful as a pre-processor before `serial` / `main_solution` when the input has more than 128 ranges. Passing these kwargs to other strategies raises `ValueError`. |

The full strategy list accepted by `strategy=` is `"separate"`,
`"greedy"`, `"pre_merge"`, `"serial"`, `"bottleneck"`,
`"main_solution"`. `max_merge_span` is no longer a `CostModel` field.
Unknown kwargs raise `ValueError`.

## Cost-model calibration

Feed `(bytes, latency_ms)` samples you measured externally into a
`Calibrator`:

```python
cal = vp.Calibrator()
for bytes_, ms in my_samples:
    cal.record(bytes_, ms)

cost_model = cal.to_cost_model(
    concurrent_parallelism=16.0,
)
# cost_model is a plain dict you can spread into vp.plan / vp.evaluate_plan.
plan = vp.plan(requests, "main_solution", **cost_model)
```

`Calibrator.fit_linear()` → `(overhead_ms, ms_per_byte)`.
`Calibrator.fit_piecewise()` → `dict | None` with `break_bytes`,
`overheads_ms`, `rates_ms_per_byte`.

The `probe` feature (Rust-only, async, built on `object_store`) is
**not** exposed to Python — network probing lives in Rust. If you
want it, use the Rust API and read the fitted `(overhead, rate)` back
into Python.

## End-to-end demo

`examples/faiss_end_to_end.py` wires FAISS + vecio-planner end to
end: train an IVF index, pack clusters into one blob, run a query,
plan the I/O, compare the merged plan against the naive baseline.
See [`examples/README.md`](examples/README.md).
