#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PROCESSED = ROOT / "data" / "processed"
DATA_ROOT = ROOT / "experiments" / HERE.name / "data" / "latest"

DEFAULT_DATASETS = "sift1m,glove,deep1m,openimage,msong,random1m128"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Prepare extra 1M-scale datasets for the Figure 7 style experiment.")
    ap.add_argument("--datasets", default=DEFAULT_DATASETS)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-unavailable", action="store_true")
    ap.add_argument("--stats-json", default=str(DATA_ROOT / "dataset_stats.json"))
    ap.add_argument("--stats-csv", default=str(DATA_ROOT / "dataset_stats.csv"))
    ap.add_argument("--sample-rows", type=int, default=4096)
    return ap.parse_args()


def _run(cmd: list[str]) -> None:
    print("$ " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)


def _sample_stats(npy_path: Path, sample_rows: int) -> dict[str, float]:
    arr = np.load(npy_path, mmap_mode="r")
    rows = int(arr.shape[0])
    take = min(int(sample_rows), rows)
    idx = np.linspace(0, rows - 1, num=take, dtype=np.int64)
    sample = np.asarray(arr[idx], dtype=np.float32)
    norms = np.linalg.norm(sample, axis=1)
    return {
        "sample_rows": int(take),
        "value_mean": float(sample.mean()),
        "value_std": float(sample.std()),
        "norm_mean": float(norms.mean()),
        "norm_std": float(norms.std()),
        "finite": bool(np.isfinite(sample).all()),
    }


def _collect_stats(dataset: str, sample_rows: int) -> dict:
    out_dir = PROCESSED / dataset
    meta_path = out_dir / "meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    row = {
        "dataset": dataset,
        "status": "ready",
        "processed_dir": str(out_dir),
        "metric": meta.get("metric"),
        "base_rows": int(meta["base"]["rows"]),
        "dim": int(meta["base"]["dim"]),
        "query_rows": int(meta["queries"]["rows"]),
    }
    row.update({f"base_{k}": v for k, v in _sample_stats(out_dir / "base.npy", sample_rows).items()})
    row.update({f"query_{k}": v for k, v in _sample_stats(out_dir / "queries.npy", sample_rows).items()})
    return row


def _is_ready(dataset: str) -> bool:
    out_dir = PROCESSED / dataset
    return (
        (out_dir / "meta.json").exists()
        and (out_dir / "base.npy").exists()
        and (out_dir / "queries.npy").exists()
    )


def main() -> int:
    args = parse_args()
    datasets = [d.strip().lower() for d in args.datasets.split(",") if d.strip()]
    stats: list[dict] = []

    for dataset in datasets:
        try:
            if args.force or not _is_ready(dataset):
                cmd = [sys.executable, "-m", "artifact", "prepare-dataset", "--dataset", dataset]
                if args.download:
                    cmd.append("--download")
                if args.force:
                    cmd.append("--force")
                _run(cmd)
            stats.append(_collect_stats(dataset, int(args.sample_rows)))
        except Exception as e:
            if not args.skip_unavailable:
                raise
            stats.append(
                {
                    "dataset": dataset,
                    "status": "unavailable",
                    "error": str(e),
                }
            )

    stats_json = Path(args.stats_json).expanduser().resolve()
    stats_csv = Path(args.stats_csv).expanduser().resolve()
    stats_json.parent.mkdir(parents=True, exist_ok=True)
    stats_json.write_text(json.dumps({"rows": stats}, indent=2) + "\n", encoding="utf-8")

    fieldnames: list[str] = []
    for row in stats:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with stats_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in stats:
            writer.writerow(row)

    ready = [r["dataset"] for r in stats if r.get("status") == "ready"]
    print(f"[ready] {','.join(ready)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
