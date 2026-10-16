#!/usr/bin/env python3
"""Prepare Figure 11's SIFT1M/MSong arrays; supports ``--check-only`` and ``--download-public-data``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from artifact.prepare_dataset import PREPARERS


DATASETS = ("sift1m", "msong")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data" / "processed")
    parser.add_argument("--download-public-data", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    plans = [(key, args.data_root.expanduser().resolve() / key) for key in DATASETS]
    for key, output in plans:
        print(f"[dataset] {key} output={output}")
    if args.check_only:
        print("[ok] static Figure 11 public-data plan contains 2 datasets")
        return 0
    for key, output in plans:
        base = output / "base.npy"
        queries = output / "queries.npy"
        if base.is_file() and queries.is_file():
            print(f"[skip] prepared public dataset {key}")
            continue
        if base.exists() or queries.exists():
            raise SystemExit(f"partial processed dataset for {key}: {output}")
        if not args.download_public_data:
            raise SystemExit(
                f"processed dataset {key} is missing; rerun with --download-public-data"
            )
        PREPARERS[key](output, chunk_rows=65_536, download=True)
    print("[ok] Figure 11 public arrays are ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
