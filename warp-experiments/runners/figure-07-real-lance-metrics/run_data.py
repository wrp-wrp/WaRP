#!/usr/bin/env python3
"""Rerun the complete Figure 7 real-Lance campaign.

The runner expects the matched Random/Reordered IVF-PQ datasets to already be
present below ``--s3-root``. It executes the 63 cache-off performance cells,
21 recall exports, two bounded-cache residency censuses, and 18 cache-on cells,
then publishes the inputs consumed by the Figure 7 figure and table renderers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.shared.full_pipeline_campaign import (
    AGGREGATE_METRICS,
    Cell,
    Method,
    aggregate,
    observed_index_identity,
    run_cell,
    sha256,
)
from artifact.campaigns import MATCHED_DATASETS, PAPER_CAMPAIGNS


OUT = ROOT / "experiments" / HERE.name / "data" / "latest"
DATA_ROOT = ROOT / "data" / "processed"

METHODS = {
    "stock_lance": Method("stock_lance", "random", "native"),
    "warp": Method("warp", "random", "balanced_ub", 64),
    "warp_reorder": Method("warp_reorder", "reorder", "balanced_ub", 64),
}

DATASETS = {
    key: (
        spec.random_object,
        spec.reordered_object,
        spec.nprobe,
        spec.refine_factor,
    )
    for key, spec in MATCHED_DATASETS.items()
}
FIGURE7_MAIN = PAPER_CAMPAIGNS["figure7-main"]

DATASET_ORDERS = (
    ("sift1m", "glove", "openimage", "msong", "gist1m", "sift100m", "deep100m"),
    ("deep100m", "sift100m", "gist1m", "msong", "openimage", "glove", "sift1m"),
    ("openimage", "gist1m", "sift1m", "deep100m", "glove", "msong", "sift100m"),
)
METHOD_ORDERS = (
    ("stock_lance", "warp", "warp_reorder"),
    ("stock_lance", "warp_reorder", "warp"),
    ("warp", "stock_lance", "warp_reorder"),
    ("warp", "warp_reorder", "stock_lance"),
    ("warp_reorder", "stock_lance", "warp"),
    ("warp_reorder", "warp", "stock_lance"),
)


def _default_binary() -> Path:
    env_root = str(os.environ.get("VECIO_LANCE_ROOT", "")).strip()
    roots = ([Path(env_root).expanduser()] if env_root else []) + [
        ROOT.parent / "lance-s3-io-optimization",
        ROOT / "third_party" / "lance",
        ROOT.parent / "lance",
    ]
    paths = [path / "target/release/examples/full_pipeline_real_s3" for path in roots]
    return next((path for path in paths if path.is_file()), paths[0])


def _default_recall_binary() -> Path:
    return _default_binary().with_name("full_pipeline_recall_s3")


def _default_s3_root() -> str:
    explicit = str(os.environ.get("VECIO_REAL_LANCE_S3_ROOT", "")).strip()
    if explicit:
        return explicit.rstrip("/")
    bucket = str(os.environ.get("VECIO_S3_BUCKET", "")).strip()
    return f"s3://{bucket}/real-lance-format-sanity" if bucket else ""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lance-bin", default=str(_default_binary()))
    ap.add_argument("--recall-bin", default=str(_default_recall_binary()))
    ap.add_argument("--data-root", default=str(DATA_ROOT))
    ap.add_argument("--s3-root", default=_default_s3_root())
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument(
        "--index-manifest-root",
        default=str(OUT / "dataset-builds"),
        help="Local managed-build manifests that pin the physical Lance indices.",
    )
    ap.add_argument("--num-queries", type=int, default=FIGURE7_MAIN.measured_queries)
    ap.add_argument("--warmup-queries", type=int, default=FIGURE7_MAIN.warmup_queries)
    ap.add_argument("--repeats", type=int, default=FIGURE7_MAIN.repeats)
    ap.add_argument("--query-concurrency", type=int, default=FIGURE7_MAIN.query_concurrency)
    ap.add_argument("--io-limit", type=int, default=FIGURE7_MAIN.io_limit)
    ap.add_argument("--census-cache-mb", type=int, default=24_576)
    ap.add_argument("--timeout-seconds", type=int, default=14_400)
    ap.add_argument("--cooldown-seconds", type=float, default=8.0)
    ap.add_argument(
        "--network-interface",
        default=str(os.environ.get("VECIO_NETWORK_INTERFACE", "")).strip(),
        help="Interface whose whole-process RX/TX counters are recorded; auto-detected if omitted.",
    )
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _uri(s3_root: str, dataset: str, layout: str) -> str:
    random_name, reorder_name, _, _ = DATASETS[dataset]
    return f"{s3_root.rstrip('/')}/{random_name if layout == 'random' else reorder_name}"


def _cell(
    args: argparse.Namespace,
    dataset: str,
    method: Method,
    repeat: int,
    *,
    cache_mb: int | None = None,
    measured_passes: int = 1,
    experiment: str = "figure7-performance",
) -> Cell:
    _, _, nprobe, refine = DATASETS[dataset]
    return Cell(
        experiment=experiment,
        dataset=dataset,
        method=method,
        repeat=repeat,
        uri=_uri(args.s3_root, dataset, method.layout),
        queries_npy=Path(args.data_root).resolve() / dataset / "queries.npy",
        index_manifest=(
            Path(args.index_manifest_root).expanduser().resolve() / dataset / "manifest.json"
        ),
        query_concurrency=args.query_concurrency,
        io_limit=args.io_limit,
        nprobe=nprobe,
        refine_factor=refine,
        k=10,
        num_queries=args.num_queries,
        warmup_queries=args.warmup_queries,
        index_cache_mb=cache_mb,
        measured_passes=measured_passes,
        disable_partition_cache=cache_mb is None,
        network_interface=args.network_interface or None,
    )


def _environment() -> dict[str, str]:
    env = dict(os.environ)
    env.update(
        {
            "LANCE_IO_MERGE_LATENCY_MODEL": "piecewise_linear",
            "LANCE_IO_MERGE_LATENCY_PW_BREAKS_BYTES": "3145728",
            "LANCE_IO_MERGE_LATENCY_PW_OVERHEADS_MS": "28.849557293569053,6.266035367002434",
            "LANCE_IO_MERGE_LATENCY_PW_RATES_MS_PER_BYTE": "2.824569398517584e-06,1.0003677676972243e-05",
            "LANCE_IO_MERGE_MAX_WASTE_RATIO": "99.0",
        }
    )
    return env


def _detect_network_interface() -> str:
    """Return the Linux interface carrying the default route."""
    route = Path("/proc/net/route")
    if route.is_file():
        for line in route.read_text(encoding="ascii").splitlines()[1:]:
            fields = line.split()
            if len(fields) >= 4 and fields[1] == "00000000" and int(fields[3], 16) & 0x2:
                interface = fields[0]
                if (Path("/sys/class/net") / interface / "statistics").is_dir():
                    return interface
    raise SystemExit(
        "cannot auto-detect the benchmark network interface; pass --network-interface"
    )


def _validate_inputs(args: argparse.Namespace) -> tuple[Path, Path, str]:
    binary = Path(args.lance_bin).expanduser().resolve()
    recall_binary = Path(args.recall_bin).expanduser().resolve()
    errors: list[str] = []
    for label, path in (("full_pipeline_real_s3", binary), ("full_pipeline_recall_s3", recall_binary)):
        if not path.is_file() or not os.access(path, os.X_OK):
            errors.append(f"{label} is missing or not executable: {path}")
    if not args.s3_root.startswith("s3://"):
        errors.append("--s3-root or VECIO_REAL_LANCE_S3_ROOT must be an s3:// URI")
    for dataset in DATASETS:
        query, ground_truth, query_origin = _recall_assets(args, dataset)
        for label, path in (("queries", query), ("ground truth", ground_truth)):
            if not path.is_file():
                errors.append(f"{dataset} {label} is missing: {path}")
        if query_origin is not None and not query_origin.is_file():
            errors.append(f"{dataset} query origin is missing: {query_origin}")
        manifest = (
            Path(args.index_manifest_root).expanduser().resolve() / dataset / "manifest.json"
        )
        for layout in ("random", "reorder"):
            try:
                observed_index_identity(
                    manifest,
                    dataset=dataset,
                    layout=layout,
                    uri=_uri(args.s3_root, dataset, layout),
                )
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                errors.append(f"{dataset} {layout} index identity is invalid: {exc}")
    paper_contract = {
        "num_queries": (args.num_queries, FIGURE7_MAIN.measured_queries),
        "warmup_queries": (args.warmup_queries, FIGURE7_MAIN.warmup_queries),
        "repeats": (args.repeats, FIGURE7_MAIN.repeats),
        "query_concurrency": (args.query_concurrency, FIGURE7_MAIN.query_concurrency),
        "io_limit": (args.io_limit, FIGURE7_MAIN.io_limit),
    }
    changed = [name for name, (actual, expected) in paper_contract.items() if actual != expected]
    if changed:
        errors.append("the paper Figure 7 contract requires the defaults for: " + ", ".join(changed))
    interface = args.network_interface or _detect_network_interface()
    if not (Path("/sys/class/net") / interface / "statistics").is_dir():
        errors.append(f"network interface does not expose Linux counters: {interface}")
    if errors:
        raise SystemExit("Figure 7 preflight failed:\n  - " + "\n  - ".join(errors))
    return binary, recall_binary, interface


def _recall_environment(method: Method) -> dict[str, str]:
    env = _environment()
    for name in (
        "LANCE_IVF_BATCH_PREFETCH",
        "LANCE_IVF_STOCK_EXECUTION",
        "LANCE_IO_MERGE_CONCURRENT_PARALLELISM",
    ):
        env.pop(name, None)
    env.update(
        {
            "LANCE_IO_THREADS": "2048",
            "LANCE_PROCESS_IO_THREADS_LIMIT": "2048",
            "LANCE_IO_MERGE_STRATEGY": method.strategy,
            "LANCE_IO_MERGE_MAX_SPAN_BYTES": "524288",
            "LANCE_DISABLE_PARTITION_CACHE": "1",
            "RAYON_NUM_THREADS": "1",
        }
    )
    if method.planner_p is None:
        env["LANCE_IVF_STOCK_EXECUTION"] = "1"
    else:
        env["LANCE_IVF_BATCH_PREFETCH"] = "1"
        env["LANCE_IO_MERGE_CONCURRENT_PARALLELISM"] = str(method.planner_p)
    return env


def _recall_assets(args: argparse.Namespace, dataset: str) -> tuple[Path, Path, Path | None]:
    root = Path(args.data_root).expanduser().resolve() / dataset
    query = root / "queries.npy"
    ground_truth = root / "gt_I.npy"
    query_origin = root / "query_origin_10k.npy" if dataset == "gist1m" else None
    return query, ground_truth, query_origin


def _analyze_recall(returned_path: Path, ground_truth_path: Path, query_origin: Path | None) -> dict:
    records = [
        json.loads(line)
        for line in returned_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    if len(records) != 10_000 or [int(row["query_index"]) for row in records] != list(range(10_000)):
        raise ValueError("returned-ID records are incomplete or out of order")
    returned = np.asarray([row["ids"] for row in records], dtype=np.int64)
    ground_truth = np.asarray(np.load(ground_truth_path, mmap_mode="r")[:, :10], dtype=np.int64)
    if returned.shape != (10_000, 10) or ground_truth.shape != (10_000, 10):
        raise ValueError(
            f"recall shape mismatch: returned={returned.shape} ground_truth={ground_truth.shape}"
        )
    overlaps = np.asarray(
        [len(set(row) & set(truth)) for row, truth in zip(returned, ground_truth)],
        dtype=np.int64,
    )
    payload = {
        "schema_version": 1,
        "num_requests": 10_000,
        "unique_query_vectors": 10_000,
        "k": 10,
        "recall_at_10": float(overlaps.mean() / 10.0),
        "recall_at_1": float(np.mean(returned[:, 0] == ground_truth[:, 0])),
        "true_nn_hit_at_10": float(
            np.mean([truth[0] in row for row, truth in zip(returned, ground_truth)])
        ),
        "exact_ordered_top10_fraction": float(np.mean(np.all(returned == ground_truth, axis=1))),
        "mean_top10_overlap_count": float(overlaps.mean()),
        "min_top10_overlap_count": int(overlaps.min()),
    }
    if query_origin is not None:
        origin = np.asarray(np.load(query_origin), dtype=np.int64)
        if origin.shape != (10_000,) or set(origin.tolist()) != set(range(1_000)):
            raise ValueError("invalid GIST query-origin array")
        per_unique = np.asarray([overlaps[origin == index].mean() for index in range(1_000)])
        payload.update(
            {
                "unique_query_vectors": 1_000,
                "unique_query_recall_at_10": float(per_unique.mean() / 10.0),
                "uniform_repetitions_per_unique_query": 10,
                "query_origin_sha256": sha256(query_origin),
            }
        )
    return payload


def _recall_metadata(
    args: argparse.Namespace,
    dataset: str,
    method: Method,
    binary: Path,
) -> dict:
    query, ground_truth, query_origin = _recall_assets(args, dataset)
    _, _, nprobe, refine = DATASETS[dataset]
    payload = {
        "dataset": dataset,
        "method": method.name,
        "uri": _uri(args.s3_root, dataset, method.layout),
        "nprobe": nprobe,
        "refine_factor": refine,
        "num_queries": 10_000,
        "warmup_queries": 200,
        "query_concurrency": 32,
        "io_limit": 2048,
        "binary": str(binary),
        "binary_sha256": sha256(binary),
        "queries": str(query),
        "queries_sha256": sha256(query),
        "ground_truth": str(ground_truth),
        "ground_truth_sha256": sha256(ground_truth),
        "index_identity": observed_index_identity(
            Path(args.index_manifest_root).expanduser().resolve() / dataset / "manifest.json",
            dataset=dataset,
            layout=method.layout,
            uri=_uri(args.s3_root, dataset, method.layout),
        ),
    }
    if query_origin is not None:
        payload["query_origin"] = str(query_origin)
        payload["query_origin_sha256"] = sha256(query_origin)
    return payload


def _validate_recall_cell(
    args: argparse.Namespace,
    dataset: str,
    method: Method,
    binary: Path,
    cell_dir: Path,
) -> dict:
    files = ("meta.json", "returned.jsonl", "recall.json", "stdout", "stderr")
    recorded = json.loads((cell_dir / "sha256.json").read_text(encoding="utf-8"))
    if any(recorded.get(name) != sha256(cell_dir / name) for name in files):
        raise ValueError(f"recall checksum mismatch: {dataset}/{method.name}")
    meta = json.loads((cell_dir / "meta.json").read_text(encoding="utf-8"))
    if meta != _recall_metadata(args, dataset, method, binary):
        raise ValueError(f"recall metadata mismatch: {dataset}/{method.name}")
    recall = json.loads((cell_dir / "recall.json").read_text(encoding="utf-8"))
    value = float(recall["recall_at_10"])
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"invalid Recall@10: {dataset}/{method.name}={value}")
    return {"dataset": dataset, "method": method.name, **recall}


def _run_recall_cell(
    args: argparse.Namespace,
    dataset: str,
    method: Method,
    binary: Path,
    cells_dir: Path,
) -> dict:
    cell_dir = cells_dir / f"{dataset}_{method.name}"
    if cell_dir.is_dir() and not args.force:
        try:
            row = _validate_recall_cell(args, dataset, method, binary, cell_dir)
            print(f"[skip] valid recall cell {dataset}/{method.name}", flush=True)
            return row
        except (KeyError, OSError, ValueError, json.JSONDecodeError):
            raise ValueError(
                f"{cell_dir} exists but is not a valid completed recall cell; "
                "inspect it or rerun with --force"
            )
    cell_dir.mkdir(parents=True, exist_ok=True)
    names = ("meta.json", "returned.jsonl", "recall.json", "stdout", "stderr", "sha256.json")
    for name in names:
        path = cell_dir / name
        if path.exists():
            path.unlink()
    meta = _recall_metadata(args, dataset, method, binary)
    (cell_dir / "meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")
    command = [
        str(binary),
        "--uri", str(meta["uri"]),
        "--queries-npy", str(meta["queries"]),
        "--out-jsonl", str(cell_dir / "returned.jsonl"),
        "--num-queries", "10000",
        "--warmup-queries", "200",
        "--query-concurrency", "32",
        "--nprobe", str(meta["nprobe"]),
        "--k", "10",
        "--refine-factor", str(meta["refine_factor"]),
    ]
    print(f"[run] recall {dataset}/{method.name}", flush=True)
    with (cell_dir / "stdout").open("w") as stdout, (cell_dir / "stderr").open("w") as stderr:
        subprocess.run(
            command,
            check=True,
            env=_recall_environment(method),
            stdout=stdout,
            stderr=stderr,
            timeout=args.timeout_seconds,
        )
    _, ground_truth, query_origin = _recall_assets(args, dataset)
    recall = _analyze_recall(cell_dir / "returned.jsonl", ground_truth, query_origin)
    (cell_dir / "recall.json").write_text(
        json.dumps(recall, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    checksums = {name: sha256(cell_dir / name) for name in names[:-1]}
    (cell_dir / "sha256.json").write_text(
        json.dumps(checksums, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return _validate_recall_cell(args, dataset, method, binary, cell_dir)


def _run(args: argparse.Namespace, cell: Cell, cells_dir: Path) -> dict:
    return run_cell(
        cell,
        binary=Path(args.lance_bin),
        cells_dir=cells_dir,
        base_env=_environment(),
        timeout_seconds=args.timeout_seconds,
        cooldown_seconds=args.cooldown_seconds,
        force=args.force,
    )


def _performance_schedule(args: argparse.Namespace) -> list[Cell]:
    schedule: list[Cell] = []
    for repeat in range(1, args.repeats + 1):
        datasets = DATASET_ORDERS[(repeat - 1) % len(DATASET_ORDERS)]
        for dataset_index, dataset in enumerate(datasets):
            method_order = METHOD_ORDERS[((repeat - 1) * len(DATASETS) + dataset_index) % 6]
            schedule.extend(_cell(args, dataset, METHODS[name], repeat) for name in method_order)
    return schedule


def _cache_budgets(args: argparse.Namespace, cells_dir: Path) -> dict[str, dict[int, int]]:
    budgets: dict[str, dict[int, int]] = {}
    for layout, method_name in (("random", "warp"), ("reorder", "warp_reorder")):
        census = _cell(
            args,
            "sift100m",
            METHODS[method_name],
            0,
            cache_mb=args.census_cache_mb,
            measured_passes=2,
            experiment="figure7-cache-census",
        )
        result = _run(args, census, cells_dir)["result"]
        working_set_bytes = int(result.get("cache_size_bytes", 0))
        if working_set_bytes <= 0:
            raise ValueError(f"{layout} cache census did not report cache_size_bytes")
        budgets[layout] = {
            fraction: max(1, math.floor(working_set_bytes * fraction / 100 / 2**20))
            for fraction in (10, 50)
        }
    return budgets


def _cache_summary(performance: dict, cache_rows: list[dict], budgets: dict[str, dict[int, int]]) -> dict:
    base = {(row["dataset"], row["method"]): row for row in performance["aggregate"]}
    groups: list[dict] = []
    for name, method in METHODS.items():
        metrics = base[("sift100m", name)]["metrics"]
        group = {
            "config": name,
            "layout": method.layout,
            "cache_fraction_percent": 0,
            "cache_mb": 0,
            "repeats": 3,
            "partition_cache": "disabled",
        }
        for metric in ("qps", "p50_ms", "p95_ms", "p99_ms", "iops_per_q"):
            group[f"{metric}_mean"] = metrics[metric]["mean"]
            group[f"{metric}_sample_sd"] = metrics[metric]["sample_sd"]
        groups.append(group)

    for fraction in (10, 50):
        selected = [row for row in cache_rows if int(row["meta"]["cache_fraction_percent"]) == fraction]
        aggregated = aggregate(selected, group_fields=("dataset", "method"), expected_repeats=3)
        for row in aggregated:
            name = row["method"]
            method = METHODS[str(name)]
            group = {
                "config": name,
                "layout": method.layout,
                "cache_fraction_percent": fraction,
                "cache_mb": budgets[method.layout][fraction],
                "repeats": 3,
                "partition_cache": "enabled",
            }
            for metric in ("qps", "p50_ms", "p95_ms", "p99_ms", "iops_per_q"):
                group[f"{metric}_mean"] = row["metrics"][metric]["mean"]
                group[f"{metric}_sample_sd"] = row["metrics"][metric]["sample_sd"]
            groups.append(group)
    return {
        "schema_version": 1,
        "status": "complete",
        "campaign": "figure7-cache",
        "cells": 27,
        "new_cache_cells": cache_rows,
        "cache_budgets_mib": budgets,
        "failure_violations": [],
        "groups": groups,
    }


def _write_source_csv(path: Path, performance: dict, cache: dict, recall: dict) -> None:
    rows = {(row["dataset"], row["method"]): row["metrics"] for row in performance["aggregate"]}
    recalls = {
        (row["dataset"], row["method"]): float(row["recall_at_10"])
        for row in recall["rows"]
    }
    for group in cache["groups"]:
        fraction = int(group["cache_fraction_percent"])
        if fraction == 0:
            continue
        key = (f"sift100m_cache{fraction}", group["config"])
        rows[key] = {
            metric: {
                "mean": group[f"{metric}_mean"],
                "sample_sd": group[f"{metric}_sample_sd"],
            }
            for metric in ("qps", "p50_ms", "p95_ms", "p99_ms", "iops_per_q")
        }
    workloads = (*DATASETS.keys(), "sift100m_cache10", "sift100m_cache50")
    fields = (
        "dataset", "method", "recall_at_10", "qps_mean", "qps_sample_sd", "p50_ms_mean",
        "p50_ms_sample_sd", "p95_ms_mean", "p95_ms_sample_sd", "p99_ms_mean",
        "p99_ms_sample_sd", "get_per_q_mean", "get_per_q_sample_sd",
        "bytes_per_q_mean", "payload_gbps_mean", "rx_gbps_mean", "tx_gbps_mean",
    )
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for dataset in workloads:
            for method in METHODS:
                metrics = rows[(dataset, method)]
                writer.writerow(
                    {
                        "dataset": dataset,
                        "method": method,
                        "recall_at_10": recalls[("sift100m" if "_cache" in dataset else dataset, method)],
                        "qps_mean": metrics["qps"]["mean"],
                        "qps_sample_sd": metrics["qps"]["sample_sd"],
                        "p50_ms_mean": metrics["p50_ms"]["mean"],
                        "p50_ms_sample_sd": metrics["p50_ms"]["sample_sd"],
                        "p95_ms_mean": metrics["p95_ms"]["mean"],
                        "p95_ms_sample_sd": metrics["p95_ms"]["sample_sd"],
                        "p99_ms_mean": metrics["p99_ms"]["mean"],
                        "p99_ms_sample_sd": metrics["p99_ms"]["sample_sd"],
                        "get_per_q_mean": metrics["iops_per_q"]["mean"],
                        "get_per_q_sample_sd": metrics["iops_per_q"]["sample_sd"],
                        "bytes_per_q_mean": metrics.get("bytes_per_q", {}).get("mean", ""),
                        "payload_gbps_mean": metrics.get("measured_payload_gbps", {}).get("mean", ""),
                        "rx_gbps_mean": metrics.get("rx_gbps", {}).get("mean", ""),
                        "tx_gbps_mean": metrics.get("tx_gbps", {}).get("mean", ""),
                    }
                )


def main() -> int:
    args = parse_args()
    binary, recall_binary, interface = _validate_inputs(args)
    args.network_interface = interface
    if args.check_only:
        print(f"[plan] binary={binary}")
        print(f"[plan] recall_binary={recall_binary}")
        print(f"[plan] s3_root={args.s3_root}")
        print(f"[plan] data_root={Path(args.data_root).expanduser().resolve()}")
        print(f"[plan] index_manifest_root={Path(args.index_manifest_root).expanduser().resolve()}")
        print(f"[plan] network_interface={args.network_interface}")
        schedule = _performance_schedule(args)
        print(f"[plan] performance_cells={len(schedule)} cache_census_cells=2 cache_cells={args.repeats * 6}")
        print("[plan] recall_cells=21")
        print(
            "[plan] outputs=figure7_performance_summary.json,figure7_cache_summary.json,"
            "figure7_recall_summary.json,figure7_source.csv"
        )
        return 0
    schedule = _performance_schedule(args)

    index_identities = {
        dataset: {
            layout: observed_index_identity(
                Path(args.index_manifest_root).expanduser().resolve()
                / dataset
                / "manifest.json",
                dataset=dataset,
                layout=layout,
                uri=_uri(args.s3_root, dataset, layout),
            )
            for layout in ("random", "reorder")
        }
        for dataset in DATASETS
    }

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    performance_rows = [_run(args, cell, out_dir / "cells" / "performance") for cell in schedule]
    performance = {
        "schema_version": 1,
        "status": "complete",
        "campaign": "figure7-main",
        "index_identities": index_identities,
        "cells": performance_rows,
        "aggregate": aggregate(
            performance_rows,
            group_fields=("dataset", "method"),
            expected_repeats=3,
            metrics=(*AGGREGATE_METRICS, "rx_gbps", "tx_gbps"),
        ),
    }
    performance_path = out_dir / "figure7_performance_summary.json"
    performance_path.write_text(json.dumps(performance, indent=2, sort_keys=True) + "\n")

    budgets = _cache_budgets(args, out_dir / "cells" / "cache_census")
    cache_rows: list[dict] = []
    for repeat in range(1, 4):
        for fraction in (10, 50):
            order = METHOD_ORDERS[((repeat - 1) * 2 + (0 if fraction == 10 else 1)) % 6]
            for name in order:
                method = METHODS[name]
                cell = _cell(
                    args,
                    "sift100m",
                    method,
                    repeat,
                    cache_mb=budgets[method.layout][fraction],
                    measured_passes=2,
                    experiment="figure7-cache",
                )
                row = _run(args, cell, out_dir / "cells" / "cache")
                row["meta"]["cache_fraction_percent"] = fraction
                cache_rows.append(row)
    cache = _cache_summary(performance, cache_rows, budgets)
    cache["index_identities"] = {"sift100m": index_identities["sift100m"]}
    (out_dir / "figure7_cache_summary.json").write_text(
        json.dumps(cache, indent=2, sort_keys=True) + "\n"
    )
    recall_rows = [
        _run_recall_cell(args, dataset, METHODS[method], recall_binary, out_dir / "cells" / "recall")
        for dataset in DATASETS
        for method in METHODS
    ]
    recall = {
        "schema_version": 1,
        "status": "complete",
        "campaign": "figure7-recall",
        "gist1m_unique_query_vectors": 1_000,
        "index_identities": index_identities,
        "rows": recall_rows,
    }
    (out_dir / "figure7_recall_summary.json").write_text(
        json.dumps(recall, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_source_csv(out_dir / "figure7_source.csv", performance, cache, recall)
    print(f"[ok] Figure 7 fresh evidence: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
