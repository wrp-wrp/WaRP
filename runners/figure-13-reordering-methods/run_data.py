#!/usr/bin/env python3
"""Publish the two measured Figure 13 stage outputs for fresh rendering."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCES = {
    "reorder_cost.csv": (
        ROOT
        / "experiments/figure-13-reordering-methods/stages/reorder-cost/data/latest"
        / "figure_reorder_cost_cluster_sweep.csv"
    ),
    "locality_gap.csv": (
        ROOT
        / "experiments/figure-13-reordering-methods/stages/locality-aware-coaccess-gap/data/latest"
        / "figure_locality_aware_coaccess_gap.csv"
    ),
}
REQUIRED_COLUMNS = {
    "reorder_cost.csv": {"num_clusters", "method_label", "reorder_time_ms"},
    "locality_gap.csv": {"dataset_label", "method_label", "weighted_gap_ratio"},
}
DEFAULT_OUT = ROOT / "experiments/figure-13-reordering-methods/data/latest/stages"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--check-only", action="store_true")
    return parser.parse_args()


def _validate(path: Path, required: set[str]) -> None:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        fields = set(reader.fieldnames or ())
        first = next(reader, None)
    missing = sorted(required - fields)
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    if first is None:
        raise ValueError(f"{path}: empty CSV")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if args.check_only:
        for name, source in SOURCES.items():
            print(f"[plan] {source} -> {out_dir / name}")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, source in SOURCES.items():
        if not source.is_file():
            raise SystemExit(f"missing registered stage output: {source}")
        _validate(source, REQUIRED_COLUMNS[name])
        shutil.copy2(source, out_dir / name)
    print(f"[ok] Figure 13 fresh render inputs: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
