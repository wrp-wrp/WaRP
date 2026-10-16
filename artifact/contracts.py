"""Public, JSON-serializable reproduction contracts for paper experiments.

This module is the reader-facing index of resolved paper parameters.  It does
not contain credentials, private endpoints, or private host identities.
Infrastructure owned by the reproducer is described separately under
``reader_inputs``.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Final

from .campaigns import (
    FIGURE11_MATCHED_DATASETS,
    FIGURE11_REQUIRED_CAPABILITIES,
    LANCE_BASE_COMMIT,
    MATCHED_DATASETS,
    PAPER_CAMPAIGNS,
)


S3_INPUTS: Final[dict[str, dict[str, Any]]] = {
    "VECIO_S3_BUCKET": {
        "kind": "infrastructure",
        "required": True,
        "description": "Writable S3 bucket owned by the reproducer.",
    },
    "VECIO_S3_REGION": {
        "kind": "infrastructure",
        "required": True,
        "description": "AWS region containing the reproducer's bucket.",
    },
}
LANCE_INPUT: Final[dict[str, dict[str, Any]]] = {
    "VECIO_LANCE_ROOT": {
        "kind": "companion-checkout",
        "required": True,
        "description": "Existing checkout of the pinned public Lance companion.",
    }
}
PAPER_HOST: Final[dict[str, Any]] = {
    "provider": "AWS EC2",
    "instance_type": "c7gn.8xlarge",
    "vcpus": 32,
    "memory_gib": 64,
    "network_gbps": 100,
    "region": "ap-northeast-1",
    "object_store_location": "same-region S3",
}
LANCE_PIECEWISE_MODEL: Final[dict[str, Any]] = {
    "kind": "piecewise_linear",
    "breaks_bytes": [3_145_728],
    "overheads_ms": [28.849557293569053, 6.266035367002434],
    "rates_ms_per_byte": [2.824569398517584e-06, 1.0003677676972243e-05],
    "max_waste_ratio": 99.0,
}
PLANNER_OVERHEAD_MODEL: Final[dict[str, Any]] = {
    "kind": "piecewise_linear",
    "breaks_bytes": [1_048_576],
    "overheads_ms": [26.05, 15.06],
    "rates_ms_per_byte": [0.0, 1.0486458329132897e-05],
    "concurrent_parallelism": 8.215944927495842,
    "max_waste_ratio": 1.0,
    "max_merge_span_bytes": 0,
}


def _inputs(*groups: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    for group in groups:
        merged.update(deepcopy(group))
    return merged


def _contract(
    experiment: str,
    paper_parameters: dict[str, Any],
    *,
    reader_inputs: dict[str, dict[str, Any]] | None = None,
    sources: list[str] | None = None,
    custom_parameter_policy: str = "not-paper-reproduction",
    execution_environment: dict[str, Any] | None = None,
    availability: str = "ready",
) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "availability": availability,
        "paper_parameters": paper_parameters,
        "parameter_sources": sources or [],
        "reader_inputs": reader_inputs or {},
        "custom_parameter_policy": custom_parameter_policy,
        "execution_environment": execution_environment or {},
    }


def _matched_dataset_payload(key: str) -> dict[str, Any]:
    dataset = MATCHED_DATASETS[key]
    return {
        "rows": dataset.rows,
        "dimension": dataset.dimension,
        "nlist": dataset.nlist,
        "pq_num_sub_vectors": dataset.num_sub_vectors,
        "pq_bits_per_subcode": 8,
        "nprobe": dataset.nprobe,
        "refine_factor": dataset.refine_factor,
        "layout_seed": dataset.random_seed,
    }


FIGURE7 = PAPER_CAMPAIGNS["figure7-main"]
FIGURE8_QUERY = PAPER_CAMPAIGNS["figure8-query-concurrency"]
FIGURE8_IO = PAPER_CAMPAIGNS["figure8-io-limit"]
FIGURE11_SIFT = PAPER_CAMPAIGNS["figure11-sift-transfer"]
FIGURE11_MSONG = PAPER_CAMPAIGNS["figure11-msong-transfer"]
FIGURE11_INDEX = FIGURE11_MATCHED_DATASETS["msong-figure11"]


CONTRACTS: Final[dict[str, dict[str, Any]]] = {
    "figure-01-s3-range-get-latency": _contract(
        "figure-01-s3-range-get-latency",
        {
            "mode": "frozen measured decomposition",
            "payload_kb": [1, 10, 50, 100, 500, 1000, 5000, 10000],
            "components": ["request_overhead_ms", "transfer_time_ms"],
        },
        sources=["runners/figure-01-s3-range-get-latency/run_data.py"],
        custom_parameter_policy="fixed-frozen-evidence",
    ),
    "table-01-cluster-coarsening": _contract(
        "table-01-cluster-coarsening",
        {
            "dataset": "sift1m",
            "fine_nlist": 4096,
            "coarse_nlist": 1024,
            "fine_nprobes": [128, 160, 192, 224, 256],
            "coarse_nprobes": [32, 48, 64, 80, 96, 112, 128],
            "queries": 300,
            "query_seed": 12345,
            "train_size": 100000,
            "recall_k": 10,
        },
        sources=["artifact/registry.py"],
    ),
    "s3-piecewise-calibration-curves": _contract(
        "s3-piecewise-calibration-curves",
        {
            "object_size_mib": 256,
            "request_sizes_mib": {"min": 0.5, "max": 16.0, "step": 0.5},
            "serial_warmup": 5,
            "serial_samples": 20,
            "serial_rounds": 1,
            "between_rounds_sleep_ms": 200,
            "concurrency_probe": False,
            "max_workers": 64,
            "seed": 7,
        },
        reader_inputs=_inputs(S3_INPUTS),
        sources=["runners/s3-piecewise-calibration-curves/run_data.py"],
        execution_environment=PAPER_HOST,
    ),
    "real-s3-tiny-request-fanout-sanity": _contract(
        "real-s3-tiny-request-fanout-sanity",
        {
            "file_size_bytes": 512000000,
            "nlist": 4096,
            "cluster_bytes": 1024,
            "fanout_levels": "1..128 inclusive",
            "query_pool_size": 128,
            "query_concurrency": 1,
            "full_queries": 512,
            "full_warmup_queries": 64,
            "single_queries": 128,
            "single_warmup_queries": 16,
            "per_iop_queries": 8192,
            "per_iop_warmup_queries": 128,
            "seed": 42,
        },
        reader_inputs=_inputs(S3_INPUTS, LANCE_INPUT),
        sources=["runners/real-s3-tiny-request-fanout-sanity/run_data.py"],
        execution_environment=PAPER_HOST,
    ),
    "figure-04-object-store-io": _contract(
        "figure-04-object-store-io",
        {
            "request_sizes_mib": [0.5, 1, 1.5, 2, 2.5, 3, 4, 6, 8, 12, 16, 24, 32],
            "object_size_mib": 64,
            "serial_warmup": 20,
            "serial_rounds": 1,
            "seed": 7,
            "topologies": ["ap_same_region", "us_same_region", "cross_region", "public_internet"],
            "samples_per_topology": {
                "ap_same_region": 30,
                "us_same_region": 300,
                "cross_region": 300,
                "public_internet": 300,
            },
        },
        reader_inputs={
            "--topology": {
                "kind": "infrastructure",
                "required": True,
                "description": (
                    "Four reviewer-owned TARGET,BUCKET,REGION definitions; pass the "
                    "contract-owned sample count as the fifth CLI field."
                ),
            }
        },
        sources=["runners/figure-04-object-store-io/run_data.py"],
        execution_environment=PAPER_HOST,
    ),
    "figure-05-planning-frontier": _contract(
        "figure-05-planning-frontier",
        {"mode": "deterministic explanatory construction", "measurement": False},
        sources=["experiments/figure-05-planning-frontier/render_planning_frontier.py"],
        custom_parameter_policy="not-applicable",
    ),
    "figure-09-qps-cost": _contract(
        "figure-09-qps-cost",
        {
            "datasets": ["sift1m", "glove", "deep1m", "openimage", "msong", "random1m128"],
            "query_concurrency": 4,
            "io_limit": 64,
            "queries": 120,
            "warmup_queries": 20,
            "repeats": 3,
            "query_subset": {"mode": "random", "seed": 12345},
            "nlist": 4096,
            "nprobe": 128,
            "train_size": "all base vectors",
            "warp_p": 16,
            "request_budgets": [8, 16, 32, 64, 96, 128],
            "greedy_gap_kib": [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
            "greedy_span_mib": [1, 2, 4, 8, 16, 32],
            "reorder_method": "pca_nd_zorder",
            "balanced_exact_requests": False,
            "max_iop_size_bytes": 549_755_813_888,
        },
        reader_inputs=_inputs(S3_INPUTS, LANCE_INPUT),
        sources=["runners/figure-09-qps-cost/run_data.py"],
        execution_environment=PAPER_HOST,
    ),
    "figure-10-p-sweep-closedloop": _contract(
        "figure-10-p-sweep-closedloop",
        {
            "preset": "env3",
            "datasets": ["sift1m", "glove", "deep1m", "openimage", "msong", "random1m128"],
            "nlist": 4096,
            "nprobe": 128,
            "queries": 120,
            "warmup_queries": 20,
            "query_subset": {"mode": "random", "seed": 12345},
            "environments": ["q1-io64", "q4-io64", "q4-io8"],
            "repeats": 3,
            "balanced_p": [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            "request_budgets": [16, 64],
            "greedy_gap_kib": [64, 256, 1024, 4096],
            "greedy_span_mib": [4, 8, 16],
            "reorder_method": "pca_nd_zorder",
            "train_size": "all base vectors",
            "balanced_exact_requests": False,
            "max_iop_size_bytes": 549_755_813_888,
        },
        reader_inputs=_inputs(S3_INPUTS, LANCE_INPUT),
        sources=["artifact/registry.py", "runners/figure-10-p-sweep-closedloop/run_data.py"],
        execution_environment=PAPER_HOST,
    ),
    "figure-11-greedy-locality-shift": _contract(
        "figure-11-greedy-locality-shift",
        {
            "sift": {
                "dataset": "sift1m",
                "rows": 1_000_000,
                "dimension": 128,
                "nlist": 4096,
                "pq_num_sub_vectors": 64,
                "pq_bits_per_subcode": 8,
                "nprobe_levels": [32, 256],
                "layout_seed": 20260805,
                "measured_queries": FIGURE11_SIFT.measured_queries,
                "warmup_queries": FIGURE11_SIFT.warmup_queries,
                "repeats": FIGURE11_SIFT.repeats,
                "query_concurrency": FIGURE11_SIFT.query_concurrency,
                "io_limit": FIGURE11_SIFT.io_limit,
                "calibration": {"population": "tight", "nprobe": 32, "cells": 54},
                "evaluation_populations": ["tight", "loose"],
                "evaluation_cells": 12,
                "ivf_training_iterations": 20,
                "pq_training_iterations": 20,
            },
            "msong": {
                "rows": FIGURE11_INDEX.rows,
                "dimension": FIGURE11_INDEX.dimension,
                "nlist": FIGURE11_INDEX.nlist,
                "pq_num_sub_vectors": FIGURE11_INDEX.num_sub_vectors,
                "nprobe": FIGURE11_INDEX.nprobe,
                "refine_factor": FIGURE11_INDEX.refine_factor,
                "measured_queries": FIGURE11_MSONG.measured_queries,
                "warmup_queries": FIGURE11_MSONG.warmup_queries,
                "repeats": FIGURE11_MSONG.repeats,
                "query_concurrency": FIGURE11_MSONG.query_concurrency,
                "io_limit": FIGURE11_MSONG.io_limit,
                "trace_queries": 420,
                "trace_query_concurrency": 1,
                "evaluation_arms": [
                    "sift_frozen_greedy",
                    "msong_tuned_greedy",
                    "warp_p32_k128",
                ],
            },
            "warp_policy": {"P": 32, "K": 128},
            "greedy_gap_kib": [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
            "greedy_span_mib": [1, 2, 4, 8, 16, 32],
            "required_companion_capabilities": FIGURE11_REQUIRED_CAPABILITIES,
            "lance_base_commit": LANCE_BASE_COMMIT,
            "threads": 32,
            "latency_model": "runners/figure-11-greedy-locality-shift/paper_latency_model.json",
        },
        reader_inputs=_inputs(S3_INPUTS, LANCE_INPUT),
        sources=["artifact/campaigns.py", "runners/figure-11-greedy-locality-shift/run_data.py"],
        execution_environment=PAPER_HOST,
    ),
    "figure-12-planner-overhead": _contract(
        "figure-12-planner-overhead",
        {
            "datasets": ["sift1m", "glove"],
            "nlist": 1000,
            "nprobe_levels": [128, 256, 512],
            "request_budgets": [16, 32, 64, 128, 256, 512],
            "queries": 120,
            "warmup_queries": 20,
            "train_size": 200000,
            "jobs": 1,
            "cost_model": deepcopy(PLANNER_OVERHEAD_MODEL),
        },
        reader_inputs=_inputs(LANCE_INPUT),
        sources=["runners/figure-12-planner-overhead/run_data.py"],
        execution_environment=PAPER_HOST,
    ),
    "locality-aware-coaccess-gap": _contract(
        "locality-aware-coaccess-gap",
        {
            "datasets": ["sift1m", "glove", "deep1m", "openimage", "msong", "random1m128"],
            "methods": ["identity", "random_proj", "pca", "pca_zorder", "hilbert", "spectral_cold", "spectral_log"],
            "nlist": 4096,
            "nprobe": 128,
            "train_size": 100000,
            "queries": "full public query split",
        },
        sources=["runners/locality-aware-coaccess-gap/run_data.py"],
    ),
    "reorder-cost": _contract(
        "reorder-cost",
        {
            "dataset": "sift1m",
            "cluster_counts": [512, 1024, 2048, 4096, 8192],
            "graph_degree": 32,
            "repeats": 3,
            "seed": 12345,
            "methods": ["identity", "random_proj", "pca", "pca_zorder", "hilbert", "spectral_cold", "spectral_log"],
        },
        sources=["runners/reorder-cost/run_data.py"],
    ),
    "figure-13-reordering-methods": _contract(
        "figure-13-reordering-methods",
        {
            "mode": "deterministic composition",
            "inputs": ["reorder-cost", "locality-aware-coaccess-gap"],
        },
        sources=["runners/figure-13-reordering-methods/run_data.py"],
        custom_parameter_policy="not-applicable",
    ),
    "figure-14-sift1m-sensitivity": _contract(
        "figure-14-sift1m-sensitivity",
        {
            "dataset": "sift1m",
            "nprobe_levels": [32, 64, 128, 256, 512],
            "nlist_levels": [1024, 2048, 4096],
            "queries": 240,
            "warmup_queries": 20,
            "query_subset": {"mode": "random", "seed": 12345},
            "query_concurrency": 4,
            "process_io_limit": 64,
            "cloud_io_parallelism": 64,
            "repeats": 3,
            "settings": [
                {"nlist": 4096, "nprobe": 32, "request_budgets": [8, 16]},
                {"nlist": 4096, "nprobe": 64, "request_budgets": [16, 32]},
                {"nlist": 4096, "nprobe": 128, "request_budgets": [32, 64]},
                {"nlist": 4096, "nprobe": 256, "request_budgets": [64, 128]},
                {"nlist": 4096, "nprobe": 512, "request_budgets": [128, 256]},
                {"nlist": 1024, "nprobe": 32, "request_budgets": [8, 16]},
                {"nlist": 2048, "nprobe": 64, "request_budgets": [16, 32]},
                {"nlist": 4096, "nprobe": 128, "request_budgets": [32, 64]},
            ],
            "balanced_p": 16,
            "greedy_gap_kib": [64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384],
            "greedy_span_mib": [1, 2, 4, 8, 16, 32],
            "train_size": 200000,
            "faiss_threads": 16,
            "max_iop_size_bytes": 1 << 60,
        },
        reader_inputs=_inputs(S3_INPUTS, LANCE_INPUT),
        sources=["runners/figure-14-sift1m-sensitivity/run_data.py"],
        execution_environment=PAPER_HOST,
    ),
    "figure-07-real-lance-metrics": _contract(
        "figure-07-real-lance-metrics",
        {
            "datasets": {key: _matched_dataset_payload(key) for key in MATCHED_DATASETS},
            "methods": ["Stock Lance", "WaRP", "WaRP + Reorder"],
            "measured_queries": FIGURE7.measured_queries,
            "warmup_queries": FIGURE7.warmup_queries,
            "repeats": FIGURE7.repeats,
            "query_concurrency": FIGURE7.query_concurrency,
            "io_limit": FIGURE7.io_limit,
            "cache_census_mib": 24576,
            "cache_fractions_percent": [10, 50],
            "k": 10,
            "max_merge_span_bytes": 524288,
            "warp_planner_p": 64,
            "latency_model": deepcopy(LANCE_PIECEWISE_MODEL),
            "performance_cells": 63,
            "recall_cells": 21,
            "cache_census_cells": 2,
            "cache_cells": 18,
            "lance_base_commit": LANCE_BASE_COMMIT,
        },
        reader_inputs=_inputs(S3_INPUTS, LANCE_INPUT),
        sources=["artifact/campaigns.py", "runners/figure-07-real-lance-metrics/run_data.py"],
        execution_environment=PAPER_HOST,
    ),
    "figure-08-sift100m-concurrency-scaling": _contract(
        "figure-08-sift100m-concurrency-scaling",
        {
            "dataset": "sift100m",
            **_matched_dataset_payload("sift100m"),
            "query_refine_factor": MATCHED_DATASETS["sift100m"].refine_factor,
            "io_refine_factor": 8,
            "query_io_limit": 2048,
            "io_query_concurrency": 32,
            "k": 10,
            "max_merge_span_bytes": 524288,
            "warp_planner_p": 64,
            "latency_model": deepcopy(LANCE_PIECEWISE_MODEL),
            "query_concurrency_levels": [16, 32, 48, 64, 96, 128],
            "measured_queries": FIGURE8_QUERY.measured_queries,
            "io_measured_queries": FIGURE8_IO.measured_queries,
            "warmup_queries": FIGURE8_QUERY.warmup_queries,
            "repeats": FIGURE8_QUERY.repeats,
            "io_limit_campaigns": {
                "ladder": [
                    [64, 2048],
                    [128, 2048],
                    [256, 2048],
                    [512, 2048],
                    [1024, 2048],
                    [2048, 2048],
                ],
                "2048-4096": [[2048, 2048], [4096, 4096]],
                "4096-8192": [[4096, 4096], [8192, 8192]],
            },
            "methods": ["Stock Lance", "WaRP", "WaRP + Reorder"],
            "lance_base_commit": LANCE_BASE_COMMIT,
        },
        reader_inputs=_inputs(S3_INPUTS, LANCE_INPUT),
        sources=[
            "artifact/campaigns.py",
            "runners/figure-08-sift100m-concurrency-scaling/run_data.py",
            "runners/shared/full_pipeline_campaign.py",
            "runners/shared/lance_build_receipt.py",
        ],
        execution_environment=PAPER_HOST,
    ),
}


def contract_payload(experiment_id: str) -> dict[str, Any]:
    try:
        return deepcopy(CONTRACTS[experiment_id])
    except KeyError as exc:
        raise KeyError(f"no public reproduction contract for {experiment_id}") from exc
