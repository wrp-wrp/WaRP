#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
OUTPUT_DIR = ROOT / "experiments" / HERE.name / "data" / "latest"
OUTPUT_JSON = OUTPUT_DIR / "planner_input.json"


def _default_lance_bin() -> Path:
    candidates: list[Path] = []
    env_root = str(os.environ.get("VECIO_LANCE_ROOT", "")).strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.extend(
        [
            ROOT.parent / "lance-s3-io-optimization",
            ROOT / "third_party" / "lance",
            ROOT.parent / "lance",
        ]
    )
    binaries = [root / "target" / "release" / "examples" / "planner_overhead_real" for root in candidates]
    return next((path for path in binaries if path.is_file()), binaries[0])


DEFAULT_LANCE_BIN = _default_lance_bin()

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATASETS = {
    "sift1m": {"label": "SIFT1M", "dim": 128, "nlist": 1000},
    "glove": {"label": "GLOVE", "dim": 100, "nlist": 1000},
}

PIECEWISE_COST_MODEL = {
    "kind": "piecewise_linear",
    "breaks_bytes": [1048576],
    "overheads_ms": [26.05, 15.06],
    "rates_ms_per_byte": [0.0, 1.0486458329132897e-05],
    "concurrent_parallelism": 8.215944927495842,
    "max_waste_ratio": 1.0,
    "max_merge_span": 0,
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Export planner-overhead benchmark input for the Rust Lance backend."
    )
    ap.add_argument("--datasets", default="sift1m,glove")
    ap.add_argument("--nprobes", default="128,256,512")
    ap.add_argument("--ks", default="16,32,64,128,256,512")
    ap.add_argument("--num-queries", type=int, default=120)
    ap.add_argument("--warmup-queries", type=int, default=20)
    ap.add_argument("--train-size", type=int, default=200000)
    ap.add_argument("--out-json", default=str(OUTPUT_JSON))
    ap.add_argument(
        "--lance-bin",
        default=str(DEFAULT_LANCE_BIN),
        help=(
            "Required Lance Rust benchmark binary. Defaults to the binary under "
            "VECIO_LANCE_ROOT or a discovered sibling/third_party Lance checkout."
        ),
    )
    ap.add_argument(
        "--benchmark-out-json",
        default=str(OUTPUT_DIR / "planner_benchmark_output.json"),
        help="Output JSON path for the Lance benchmark rows.",
    )
    ap.add_argument(
        "--benchmark-out-csv",
        default=str(OUTPUT_DIR / "planner_benchmark_raw.csv"),
        help="Raw detailed CSV path for the Lance benchmark rows.",
    )
    ap.add_argument(
        "--merged-summary-csv",
        default=str(OUTPUT_DIR / "merged_summary.csv"),
        help="Canonical four-way CSV consumed by the figure renderer.",
    )
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Parallel job count passed to the Lance Rust benchmark.",
    )
    ap.add_argument("--check-only", action="store_true")
    return ap.parse_args()


def _parse_int_csv(raw: str) -> list[int]:
    out = [int(tok.strip()) for tok in str(raw).split(",") if tok.strip()]
    if not out:
        raise ValueError("empty integer csv")
    return sorted(set(out))


def _parse_dataset_csv(raw: str) -> list[str]:
    out = [tok.strip().lower() for tok in str(raw).split(",") if tok.strip()]
    if not out:
        raise ValueError("empty dataset csv")
    unknown = [name for name in out if name not in DATASETS]
    if unknown:
        raise ValueError(f"unknown datasets: {unknown}")
    return out


def _build_index_and_layout(
    base: np.ndarray, *, dim: int, nlist: int, train_size: int
) -> tuple[Any, list[int], list[int]]:
    import faiss

    from src.index_builder import configure_faiss_threads

    threads = configure_faiss_threads()
    print(
        f"[build] training FAISS IVF with {threads} threads "
        f"(nlist={int(nlist)} train_size={int(train_size)})",
        flush=True,
    )
    quantizer = faiss.IndexFlatL2(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, int(nlist), faiss.METRIC_L2)
    train_n = min(len(base), int(train_size))
    index.train(np.ascontiguousarray(base[:train_n].astype("float32")))
    index.add(np.ascontiguousarray(base.astype("float32")))

    offsets: list[int] = []
    sizes: list[int] = []
    current = 0
    for cluster_id in range(int(nlist)):
        byte_size = int(index.invlists.list_size(cluster_id)) * int(dim) * 4
        offsets.append(int(current))
        sizes.append(int(byte_size))
        current += int(byte_size)
    return index, offsets, sizes


def _real_probe_lists(index: Any, queries: np.ndarray, *, max_nprobe: int) -> list[list[int]]:
    q = np.ascontiguousarray(queries.astype("float32"))
    _dist, cluster_ids = index.quantizer.search(q, int(max_nprobe))
    out: list[list[int]] = []
    for row in cluster_ids:
        out.append([int(cid) for cid in row if int(cid) >= 0])
    return out


def _dataset_payload(dataset_key: str, *, max_nprobe: int, num_queries: int, train_size: int) -> dict:
    cfg = DATASETS[dataset_key]
    base = np.load(str(DATA_DIR / dataset_key / "base.npy"), mmap_mode="r")
    queries = np.load(str(DATA_DIR / dataset_key / "queries.npy"))[: int(num_queries)]

    print(
        f"[build] dataset={dataset_key} nlist={cfg['nlist']} dim={cfg['dim']} "
        f"base={tuple(base.shape)} queries={tuple(queries.shape)} max_nprobe={max_nprobe}",
        flush=True,
    )
    index, offsets, sizes = _build_index_and_layout(
        base,
        dim=int(cfg["dim"]),
        nlist=int(cfg["nlist"]),
        train_size=int(train_size),
    )
    probe_lists = _real_probe_lists(index, queries, max_nprobe=int(max_nprobe))

    return {
        "dataset": dataset_key,
        "label": str(cfg["label"]),
        "nlist": int(cfg["nlist"]),
        "max_nprobe": int(max_nprobe),
        "cluster_offsets": offsets,
        "cluster_sizes": sizes,
        "queries": [
            {
                "query_idx": int(idx),
                "cluster_ids": [int(cid) for cid in cluster_ids],
            }
            for idx, cluster_ids in enumerate(probe_lists)
        ],
    }


def _run_lance_benchmark(
    *,
    lance_bin: Path,
    planner_input_json: Path,
    benchmark_out_json: Path,
    benchmark_out_csv: Path,
    jobs: int,
) -> None:
    cmd = [
        str(lance_bin),
        "--input-json",
        str(planner_input_json),
        "--out-json",
        str(benchmark_out_json),
        "--out-csv",
        str(benchmark_out_csv),
        "--jobs",
        str(max(1, int(jobs))),
    ]
    print("[run] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


FOUR_WAY_FIELDS = {
    "dataset",
    "nprobe",
    "K",
    "plain_timed_out",
    "plain_p95_ms",
    "dc_timed_out",
    "dc_p95_ms",
    "adaptive_dc_timed_out",
    "adaptive_dc_p95_ms",
    "wqs_timed_out",
    "wqs_p95_ms",
    "wqs_over_adaptive_dc_p95",
    "adaptive_dc_err_avg_pct",
    "adaptive_dc_err_max_pct",
    "wqs_err_avg_pct",
    "wqs_err_max_pct",
}


def _publish_four_way_summary(benchmark_csv: Path, merged_summary_csv: Path) -> None:
    with benchmark_csv.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        rows = list(reader)
    missing = sorted(FOUR_WAY_FIELDS - fields)
    if missing:
        raise SystemExit(
            "planner benchmark output is not the four-way paper schema; "
            f"missing columns: {', '.join(missing)}"
        )
    if not rows:
        raise SystemExit("planner benchmark produced an empty CSV")
    selected = [
        row
        for row in rows
        if int(row["K"])
        in {int(row["nprobe"]) // 4, int(row["nprobe"]) // 2, int(row["nprobe"])}
    ]
    if len(selected) != 18:
        raise SystemExit(
            "four-way paper summary requires exactly 18 endpoint rows; "
            f"selected {len(selected)} from {benchmark_csv}"
        )
    for row in selected:
        for timeout_field in ("adaptive_dc_timed_out", "wqs_timed_out"):
            if str(row[timeout_field]).strip().lower() == "true":
                raise SystemExit(
                    f"{timeout_field} is true for {row['dataset']}/{row['nprobe']}/{row['K']}; "
                    "the paper summary schema cannot represent this incomplete endpoint"
                )
    summary_fields = [
        "dataset",
        "nprobe",
        "K",
        "plain_timed_out",
        "plain_p95_ms",
        "dc_timed_out",
        "dc_p95_ms",
        "adaptive_dc_p95_ms",
        "wqs_p95_ms",
        "wqs_over_adaptive_dc_p95",
        "adaptive_dc_err_avg_pct",
        "adaptive_dc_err_max_pct",
        "wqs_err_avg_pct",
        "wqs_err_max_pct",
    ]
    selected.sort(
        key=lambda row: (str(row["dataset"]).upper(), int(row["nprobe"]), int(row["K"]))
    )
    merged_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    with merged_summary_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    field: str(row[field]).strip().lower()
                    if field.endswith("_timed_out")
                    else str(row[field]).upper()
                    if field == "dataset"
                    else row[field]
                    for field in summary_fields
                }
            )
    print(f"[ok] four_way_summary={merged_summary_csv}")


def main() -> int:
    args = parse_args()
    datasets = _parse_dataset_csv(args.datasets)
    nprobes = _parse_int_csv(args.nprobes)
    ks = _parse_int_csv(args.ks)
    max_nprobe = max(nprobes)
    if any(k > max_nprobe for k in ks):
        raise SystemExit("all K values must be <= max(nprobes)")

    out_json = Path(args.out_json).resolve()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    benchmark_out_json = Path(args.benchmark_out_json).resolve()
    benchmark_out_json.parent.mkdir(parents=True, exist_ok=True)
    benchmark_out_csv = Path(args.benchmark_out_csv).resolve()
    benchmark_out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged_summary_csv = Path(args.merged_summary_csv).resolve()
    merged_summary_csv.parent.mkdir(parents=True, exist_ok=True)
    lance_bin = Path(args.lance_bin).expanduser().resolve()
    missing_inputs = [
        path
        for dataset_key in datasets
        for path in (
            DATA_DIR / dataset_key / "base.npy",
            DATA_DIR / dataset_key / "queries.npy",
        )
        if not path.is_file()
    ]
    preflight_errors = []
    if not lance_bin.is_file() or not os.access(lance_bin, os.X_OK):
        preflight_errors.append(f"planner_overhead_real is missing or not executable: {lance_bin}")
    preflight_errors.extend(f"dataset input is missing: {path}" for path in missing_inputs)
    if preflight_errors:
        raise SystemExit("planner-overhead preflight failed:\n  - " + "\n  - ".join(preflight_errors))

    payload = {
        "version": "v1",
        "benchmark_name": "planner_overhead_real_piecewise",
        "latency_model": dict(PIECEWISE_COST_MODEL),
        "benchmark_grid": {
            "nprobes": nprobes,
            "ks": ks,
            "num_queries": int(args.num_queries),
            "warmup_queries": int(args.warmup_queries),
        },
        "datasets": [],
    }

    if args.check_only:
        for dataset_key in datasets:
            cfg = DATASETS[dataset_key]
            print(
                f"[plan] dataset={dataset_key} label={cfg['label']} nlist={cfg['nlist']} "
                f"dim={cfg['dim']} max_nprobe={max_nprobe}"
            )
        print(f"[plan] out_json={out_json}")
        print(f"[plan] lance_bin={lance_bin}")
        print(f"[plan] benchmark_out_json={benchmark_out_json}")
        print(f"[plan] benchmark_out_csv={benchmark_out_csv}")
        print(f"[plan] merged_summary_csv={merged_summary_csv}")
        print(f"[plan] jobs={max(1, int(args.jobs))}")
        return 0

    for dataset_key in datasets:
        payload["datasets"].append(
            _dataset_payload(
                dataset_key,
                max_nprobe=max_nprobe,
                num_queries=int(args.num_queries),
                train_size=int(args.train_size),
            )
        )

    out_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[ok] planner_input={out_json}")
    _run_lance_benchmark(
        lance_bin=lance_bin,
        planner_input_json=out_json,
        benchmark_out_json=benchmark_out_json,
        benchmark_out_csv=benchmark_out_csv,
        jobs=int(args.jobs),
    )
    _publish_four_way_summary(benchmark_out_csv, merged_summary_csv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
