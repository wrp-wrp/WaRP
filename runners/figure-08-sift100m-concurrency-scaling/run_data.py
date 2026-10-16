#!/usr/bin/env python3
"""Rerun the complete real-Lance concurrency scaling experiment (Figure 8)."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runners.shared.full_pipeline_campaign import (
    Cell,
    Method,
    aggregate,
    observed_index_identity,
    run_cell,
    sha256,
)
from runners.shared.lance_build_receipt import validate_receipt
from artifact.campaigns import LANCE_BASE_COMMIT, MATCHED_DATASETS, PAPER_CAMPAIGNS
from artifact.lance_toolchain import companion_root_for_binary, validate_lance_source


OUT = ROOT / "experiments" / HERE.name / "data" / "latest"
METHODS = {
    "stock_lance": Method("stock_lance", "random", "native"),
    "warp": Method("warp", "random", "balanced_ub", 64),
    "warp_reorder": Method("warp_reorder", "reorder", "balanced_ub", 64),
}
Q_LEVELS = (16, 32, 48, 64, 96, 128)
IO_CAMPAIGNS = {
    "ladder": (64, 128, 256, 512, 1024, 2048),
    "2048-4096": (2048, 4096),
    "4096-8192": (4096, 8192),
}
IO_SCHEDULES = {
    "ladder": {
        1: ((64, 2048), (128, 2048), (256, 2048), (512, 2048), (1024, 2048), (2048, 2048)),
        2: ((2048, 2048), (1024, 2048), (512, 2048), (256, 2048), (128, 2048), (64, 2048)),
        3: ((256, 2048), (64, 2048), (1024, 2048), (128, 2048), (2048, 2048), (512, 2048)),
    },
    "2048-4096": {
        1: ((2048, 2048), (4096, 4096)),
        2: ((4096, 4096), (2048, 2048)),
        3: ((2048, 2048), (4096, 4096)),
    },
    "4096-8192": {
        1: ((4096, 4096), (8192, 8192)),
        2: ((8192, 8192), (4096, 4096)),
        3: ((4096, 4096), (8192, 8192)),
    },
}
IO_CAMPAIGN_IDS = {
    "ladder": "figure8-io-limit-ladder",
    "2048-4096": "figure8-io-limit-2048-4096",
    "4096-8192": "figure8-io-limit-4096-8192",
}
QUERY_SWEEP = PAPER_CAMPAIGNS["figure8-query-concurrency"]
IO_SWEEP = PAPER_CAMPAIGNS["figure8-io-limit"]
IO_SUMMARY_METRICS = (
    "elapsed_s",
    "iops_per_q",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "qps",
)
REQUIRED_BINARY_FLAGS = (
    "--reuse-warmup-queries",
    "--out-latencies-jsonl",
)


def _default_binary() -> Path:
    env_root = str(os.environ.get("VECIO_LANCE_ROOT", "")).strip()
    roots = ([Path(env_root).expanduser()] if env_root else []) + [
        ROOT.parent / "lance-s3-io-optimization",
        ROOT / "third_party" / "lance",
        ROOT.parent / "lance",
    ]
    paths = [root / "target/release/examples/full_pipeline_real_s3" for root in roots]
    return next((path for path in paths if path.is_file()), paths[0])


def _default_s3_root() -> str:
    explicit = str(os.environ.get("VECIO_REAL_LANCE_S3_ROOT", "")).strip()
    if explicit:
        return explicit.rstrip("/")
    bucket = str(os.environ.get("VECIO_S3_BUCKET", "")).strip()
    return f"s3://{bucket}/real-lance-format-sanity" if bucket else ""


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--lance-bin", default=str(_default_binary()))
    ap.add_argument("--queries-npy", default=str(ROOT / "data/processed/sift100m/queries.npy"))
    ap.add_argument("--s3-root", default=_default_s3_root())
    ap.add_argument("--out-dir", default=str(OUT))
    ap.add_argument(
        "--index-manifest-root",
        default=str(
            ROOT
            / "experiments/figure-07-real-lance-metrics/data/latest/dataset-builds"
        ),
        help="Local managed-build manifests that pin the physical Lance indices.",
    )
    ap.add_argument("--num-queries", type=int, default=QUERY_SWEEP.measured_queries)
    ap.add_argument("--io-num-queries", type=int, default=IO_SWEEP.measured_queries)
    ap.add_argument("--warmup-queries", type=int, default=QUERY_SWEEP.warmup_queries)
    ap.add_argument("--repeats", type=int, default=QUERY_SWEEP.repeats)
    ap.add_argument("--timeout-seconds", type=int, default=14_400)
    ap.add_argument("--cooldown-seconds", type=float, default=8.0)
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--force", action="store_true")
    return ap.parse_args()


def _env() -> dict[str, str]:
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


def _validate_inputs(args: argparse.Namespace) -> Path:
    binary = Path(args.lance_bin).expanduser().resolve()
    errors: list[str] = []
    if not binary.is_file() or not os.access(binary, os.X_OK):
        errors.append(f"full_pipeline_real_s3 is missing or not executable: {binary}")
    else:
        try:
            _companion_identity(binary)
        except (
            OSError,
            RuntimeError,
            ValueError,
            json.JSONDecodeError,
            subprocess.CalledProcessError,
        ) as exc:
            errors.append(str(exc))
    if not args.s3_root.startswith("s3://"):
        errors.append("--s3-root or VECIO_REAL_LANCE_S3_ROOT must be an s3:// URI")
    queries = Path(args.queries_npy).expanduser().resolve()
    if not queries.is_file():
        errors.append(f"SIFT100M queries are missing: {queries}")
    manifest = (
        Path(args.index_manifest_root).expanduser().resolve()
        / "sift100m"
        / "manifest.json"
    )
    for layout in ("random", "reorder"):
        try:
            observed_index_identity(
                manifest,
                dataset="sift100m",
                layout=layout,
                uri=_uri(args, layout),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"SIFT100M {layout} index identity is invalid: {exc}")
    paper_contract = {
        "num_queries": (args.num_queries, QUERY_SWEEP.measured_queries),
        "io_num_queries": (args.io_num_queries, IO_SWEEP.measured_queries),
        "warmup_queries": (args.warmup_queries, QUERY_SWEEP.warmup_queries),
        "repeats": (args.repeats, QUERY_SWEEP.repeats),
    }
    changed = [name for name, (actual, expected) in paper_contract.items() if actual != expected]
    if changed:
        errors.append("the paper Figure 8 contract requires the defaults for: " + ", ".join(changed))
    if errors:
        raise SystemExit("Figure 8 preflight failed:\n  - " + "\n  - ".join(errors))
    return binary


def _uri(args: argparse.Namespace, layout: str) -> str:
    dataset = MATCHED_DATASETS["sift100m"]
    name = dataset.random_object if layout == "random" else dataset.reordered_object
    return f"{args.s3_root.rstrip('/')}/{name}"


def _cell(
    args: argparse.Namespace,
    *,
    experiment: str,
    method: Method,
    repeat: int,
    q: int,
    io_limit: int,
    process_io_limit: int | None = None,
    num_queries: int | None = None,
    refine_factor: int = 8,
    order_ordinal: int | None = None,
) -> Cell:
    return Cell(
        experiment=experiment,
        dataset="sift100m",
        method=method,
        repeat=repeat,
        uri=_uri(args, method.layout),
        queries_npy=Path(args.queries_npy).expanduser().resolve(),
        index_manifest=(
            Path(args.index_manifest_root).expanduser().resolve()
            / "sift100m"
            / "manifest.json"
        ),
        query_concurrency=q,
        io_limit=io_limit,
        process_io_limit=process_io_limit,
        nprobe=MATCHED_DATASETS["sift100m"].nprobe,
        refine_factor=refine_factor,
        k=10,
        num_queries=args.num_queries if num_queries is None else num_queries,
        warmup_queries=args.warmup_queries,
        order_ordinal=order_ordinal,
        runner_sha256=sha256(Path(__file__).resolve()),
        campaign_contract_sha256=sha256(ROOT / "artifact" / "campaigns.py"),
    )


def _run(args: argparse.Namespace, cell: Cell, cells_dir: Path) -> dict:
    return run_cell(
        cell,
        binary=Path(args.lance_bin).expanduser().resolve(),
        cells_dir=cells_dir,
        base_env=_env(),
        timeout_seconds=args.timeout_seconds,
        cooldown_seconds=args.cooldown_seconds,
        force=args.force,
    )


def _paper_cell(row: dict) -> dict:
    meta = row["meta"]
    method = meta["method"]["name"]
    return {
        "cell": row["cell"],
        "meta": {
            "dataset": "sift100m",
            "experiment": str(meta["experiment"]),
            "method": method,
            "index": "IVF65536-PQ16",
            "compression": "32x",
            "query_concurrency": str(meta["query_concurrency"]),
            "io_limits": f"{meta['io_limit']}/{meta['io_limit']}",
            "nprobe": "256",
            "refine_factor": "8",
            "k": "10",
            "num_queries": str(meta["num_queries"]),
            "warmup_queries": str(meta["warmup_queries"]),
            "query_sha256": str(meta["queries_sha256"]),
            "binary_sha256": str(meta["binary_sha256"]),
            "repeat": str(meta["repeat"]),
            "planner_p": "na" if method == "stock_lance" else "64",
            "request_budget": "na" if method == "stock_lance" else "unbounded",
            "layout": "random" if method != "warp_reorder" else "pca_nd_zorder",
        },
        "result": row["result"],
    }


def _companion_root(binary: Path) -> Path:
    return companion_root_for_binary(binary)


def _validate_binary_capabilities(binary: Path) -> None:
    completed = subprocess.run(
        [str(binary), "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )
    help_text = completed.stdout + completed.stderr
    missing = [flag for flag in REQUIRED_BINARY_FLAGS if flag not in help_text]
    if completed.returncode != 0 or missing:
        detail = ", ".join(missing) if missing else f"exit {completed.returncode}"
        raise ValueError(f"paper Lance binary lacks required Figure 8 flags: {detail}")


def _validate_companion_checkout(binary: Path) -> str:
    root = _companion_root(binary)
    try:
        return validate_lance_source(root)["lance_commit"]
    except SystemExit as exc:
        raise RuntimeError(str(exc)) from exc


def _companion_identity(binary: Path) -> dict[str, object]:
    _validate_binary_capabilities(binary)
    root = _companion_root(binary)
    head = _validate_companion_checkout(binary)
    identity = validate_receipt(
        root,
        binary,
        package="lance",
        example="full_pipeline_real_s3",
    )
    if identity["lance_commit"] != head:
        raise ValueError("paper Lance build receipt commit differs from checkout")
    return identity


def _io_summary(
    args: argparse.Namespace,
    binary: Path,
    campaign: str,
    rows: list[dict],
    limits: tuple[int, ...],
    index_identity: dict[str, object],
) -> dict:
    by_limit: dict[str, dict[str, dict[str, object]]] = {}
    for limit in limits:
        matches = sorted(
            (row for row in rows if int(row["meta"]["io_limit"]) == limit),
            key=lambda row: int(row["meta"]["repeat"]),
        )
        if [int(row["meta"]["repeat"]) for row in matches] != [1, 2, 3]:
            raise ValueError(f"{campaign}: repeats are incomplete for I/O limit {limit}")
        metrics: dict[str, dict[str, object]] = {}
        for metric in IO_SUMMARY_METRICS:
            values = [float(row["result"][metric]) for row in matches]
            metrics[metric] = {
                "values": values,
                "mean": statistics.fmean(values),
                "sample_sd": statistics.stdev(values),
            }
        by_limit[str(limit)] = metrics
    if set(by_limit) != {str(limit) for limit in limits}:
        raise ValueError(f"{campaign}: incomplete I/O-limit matrix")
    first_meta = rows[0]["meta"]
    process_limits = [2048 if campaign == "ladder" else limit for limit in limits]
    companion_identity = _companion_identity(binary)
    return {
        "schema_version": 2,
        "status": "complete",
        "campaign": IO_CAMPAIGN_IDS[campaign],
        "measurement_contract": {
            "dataset": "sift100m",
            "layout": "random",
            "method": "stock_lance",
            "query_concurrency": 32,
            "num_queries": args.io_num_queries,
            "warmup_queries": args.warmup_queries,
            "warmup_reused": True,
            "nlist": MATCHED_DATASETS["sift100m"].nlist,
            "num_sub_vectors": MATCHED_DATASETS["sift100m"].num_sub_vectors,
            "bits_per_subcode": 8,
            "nprobe": MATCHED_DATASETS["sift100m"].nprobe,
            "refine_factor": 8,
            "k": 10,
            "repeats": 3,
            "counterbalanced": True,
            "limit_pairs": [
                {"file": file_limit, "process": process_limit}
                for file_limit, process_limit in zip(limits, process_limits)
            ],
            "query_sha256": first_meta["queries_sha256"],
            "binary_sha256": first_meta["binary_sha256"],
            "lance_commit": companion_identity["lance_commit"],
            "binary_build_receipt_sha256": companion_identity["receipt_sha256"],
            "cargo_lock_sha256": companion_identity["cargo_lock_sha256"],
            "index_identity": index_identity,
            "source_hashes": {
                "runner": sha256(Path(__file__).resolve()),
                "shared_harness": sha256(
                    ROOT / "runners" / "shared" / "full_pipeline_campaign.py"
                ),
                "build_receipt": sha256(
                    ROOT / "runners" / "shared" / "lance_build_receipt.py"
                ),
                "campaign_contract": sha256(ROOT / "artifact" / "campaigns.py"),
            },
        },
        "groups": by_limit,
    }


def _query_summary(
    args: argparse.Namespace,
    binary: Path,
    rows: list[dict],
    index_identities: dict[str, object],
) -> dict:
    if not rows:
        raise ValueError("Figure 8 query summary cannot be built from zero cells")
    expected_source_hashes = {
        "runner_sha256": sha256(Path(__file__).resolve()),
        "shared_harness_sha256": sha256(
            ROOT / "runners" / "shared" / "full_pipeline_campaign.py"
        ),
        "campaign_contract_sha256": sha256(ROOT / "artifact" / "campaigns.py"),
    }
    first_meta = rows[0]["meta"]
    binary_sha256 = first_meta["binary_sha256"]
    query_sha256 = first_meta["queries_sha256"]
    expected_cells = {
        (method, q, repeat)
        for method in METHODS
        for q in Q_LEVELS
        for repeat in (1, 2, 3)
    }
    actual_cells: set[tuple[str, int, int]] = set()
    ordinals: dict[int, set[int]] = {repeat: set() for repeat in (1, 2, 3)}
    for row in rows:
        meta = row["meta"]
        method = str(meta["method"]["name"])
        q = int(meta["query_concurrency"])
        repeat = int(meta["repeat"])
        actual_cells.add((method, q, repeat))
        ordinals.setdefault(repeat, set()).add(int(meta["order_ordinal"]))
        expected_scientific_contract = {
            "dataset": "sift100m",
            "experiment": "figure8-query-concurrency",
            "io_limit": 2048,
            "nprobe": MATCHED_DATASETS["sift100m"].nprobe,
            "refine_factor": 8,
            "k": 10,
            "num_queries": args.num_queries,
            "warmup_queries": args.warmup_queries,
            "max_merge_span_bytes": 524_288,
        }
        changed = [
            key
            for key, expected in expected_scientific_contract.items()
            if meta.get(key) != expected
        ]
        if changed:
            raise ValueError(
                "Figure 8 query cell has a mismatched contract: "
                + ", ".join(changed)
            )
        if meta.get("binary_sha256") != binary_sha256:
            raise ValueError("Figure 8 query cells were produced by different binaries")
        if meta.get("queries_sha256") != query_sha256:
            raise ValueError("Figure 8 query cells used different query arrays")
        for key, expected in expected_source_hashes.items():
            if meta.get(key) != expected:
                raise ValueError(f"Figure 8 query cell has stale {key}")
        layout = "reorder" if method == "warp_reorder" else "random"
        if meta.get("index_identity") != index_identities[layout]:
            raise ValueError(
                f"Figure 8 query cell has the wrong {layout} physical index identity"
            )
    if actual_cells != expected_cells or len(rows) != len(expected_cells):
        raise ValueError("Figure 8 query cells do not form the complete 54-cell matrix")
    expected_ordinals = set(range(1, len(METHODS) * len(Q_LEVELS) + 1))
    if any(ordinals.get(repeat) != expected_ordinals for repeat in (1, 2, 3)):
        raise ValueError("Figure 8 query cells do not retain the counterbalanced order")
    grouped = aggregate(
        rows,
        group_fields=("method", "query_concurrency", "io_limit"),
        expected_repeats=3,
    )
    companion_identity = _companion_identity(binary)
    return {
        "schema_version": 1,
        "status": "complete",
        "campaign": "figure8-query-concurrency",
        "measurement_contract": {
            "dataset": "sift100m",
            "index": "IVF65536-PQ16",
            "nlist": MATCHED_DATASETS["sift100m"].nlist,
            "num_sub_vectors": MATCHED_DATASETS["sift100m"].num_sub_vectors,
            "bits_per_subcode": 8,
            "nprobe": MATCHED_DATASETS["sift100m"].nprobe,
            "refine_factor": 8,
            "k": 10,
            "methods": list(METHODS),
            "query_concurrency_levels": list(Q_LEVELS),
            "io_limit": 2048,
            "num_queries": args.num_queries,
            "warmup_queries": args.warmup_queries,
            "warmup_reused": True,
            "repeats": 3,
            "counterbalanced": True,
            "warp_planner_p": 64,
            "request_budget": "unbounded",
            "max_merge_span_bytes": 524_288,
            "query_sha256": query_sha256,
            "binary_sha256": binary_sha256,
            "lance_commit": companion_identity["lance_commit"],
            "binary_build_receipt_sha256": companion_identity["receipt_sha256"],
            "cargo_lock_sha256": companion_identity["cargo_lock_sha256"],
            "index_identities": index_identities,
            "source_hashes": {
                "runner": sha256(Path(__file__).resolve()),
                "shared_harness": sha256(
                    ROOT / "runners" / "shared" / "full_pipeline_campaign.py"
                ),
                "build_receipt": sha256(
                    ROOT / "runners" / "shared" / "lance_build_receipt.py"
                ),
                "campaign_contract": sha256(ROOT / "artifact" / "campaigns.py"),
            },
        },
        "cells": [_paper_cell(row) for row in rows],
        "aggregate": [
            {
                "dataset": "sift100m",
                "method": row["method"],
                "q": row["query_concurrency"],
                "io_cap": row["io_limit"],
                "repeats": row["repeats"],
                "metrics": row["metrics"],
            }
            for row in grouped
        ],
    }


def main() -> int:
    args = parse_args()
    binary = _validate_inputs(args)
    q_cells = len(Q_LEVELS) * len(METHODS) * args.repeats
    io_cells = sum(len(limits) for limits in IO_CAMPAIGNS.values()) * args.repeats
    if args.check_only:
        print(f"[plan] binary={binary}")
        print(f"[plan] queries={Path(args.queries_npy).expanduser().resolve()}")
        print(f"[plan] s3_root={args.s3_root}")
        print(f"[plan] index_manifest_root={Path(args.index_manifest_root).expanduser().resolve()}")
        print(f"[plan] query_ladder_cells={q_cells} io_ladder_cells={io_cells}")
        print(f"[plan] query_num_queries={args.num_queries} io_num_queries={args.io_num_queries}")
        print("[plan] outputs=query_summary.json,io_summaries/{ladder,2048-4096,4096-8192}.json")
        return 0

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = (
        Path(args.index_manifest_root).expanduser().resolve()
        / "sift100m"
        / "manifest.json"
    )
    index_identities = {
        layout: observed_index_identity(
            manifest,
            dataset="sift100m",
            layout=layout,
            uri=_uri(args, layout),
        )
        for layout in ("random", "reorder")
    }
    q_rows: list[dict] = []
    for repeat in range(1, 4):
        order_ordinal = 0
        method_order = list(METHODS)
        method_order = method_order[(repeat - 1) :] + method_order[: repeat - 1]
        levels = Q_LEVELS if repeat % 2 else tuple(reversed(Q_LEVELS))
        for q in levels:
            for name in method_order:
                order_ordinal += 1
                q_rows.append(
                    _run(
                        args,
                        _cell(
                            args,
                            experiment="figure8-query-concurrency",
                            method=METHODS[name],
                            repeat=repeat,
                            q=q,
                            io_limit=2048,
                            order_ordinal=order_ordinal,
                        ),
                        out_dir / "cells" / "query_ladder",
                    )
                )
    query_summary = _query_summary(args, binary, q_rows, index_identities)
    (out_dir / "query_summary.json").write_text(
        json.dumps(query_summary, indent=2, sort_keys=True) + "\n"
    )

    io_out = out_dir / "io_summaries"
    io_out.mkdir(parents=True, exist_ok=True)
    for campaign, limits in IO_CAMPAIGNS.items():
        rows: list[dict] = []
        for repeat in range(1, 4):
            for order_ordinal, (io_limit, process_io_limit) in enumerate(
                IO_SCHEDULES[campaign][repeat],
                start=1,
            ):
                rows.append(
                    _run(
                        args,
                        _cell(
                            args,
                            experiment=f"figure8-io-{campaign}",
                            method=METHODS["stock_lance"],
                            repeat=repeat,
                            q=32,
                            io_limit=io_limit,
                            process_io_limit=process_io_limit,
                            num_queries=args.io_num_queries,
                            refine_factor=8,
                            order_ordinal=order_ordinal,
                        ),
                        out_dir / "cells" / f"io_{campaign}",
                    )
                )
        payload = _io_summary(
            args,
            binary,
            campaign,
            rows,
            limits,
            index_identities["random"],
        )
        (io_out / f"{campaign}.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n"
        )
    print(f"[ok] Figure 8 fresh evidence: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
