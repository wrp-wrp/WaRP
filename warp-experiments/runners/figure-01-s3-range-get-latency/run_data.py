#!/usr/bin/env python3
"""Write the frozen intro-latency decomposition CSV.

The numeric values are baked into this script: each row is a single
payload size with the measured request-overhead and transfer-time
components. The experiment-owned `render_intro_latency_composition.py`
reads the resulting `data.csv` and produces the stacked bar PDF for
`fig:intro-latency-composition`.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA_OUT = ROOT / "experiments" / "figure-01-s3-range-get-latency" / "data" / "latest" / "data.csv"

ROWS = [
    {"payload_kb": 1, "request_overhead_ms": 26.56, "transfer_time_ms": 0.00336},
    {"payload_kb": 10, "request_overhead_ms": 26.56, "transfer_time_ms": 0.03272},
    {"payload_kb": 50, "request_overhead_ms": 26.56, "transfer_time_ms": 0.16884},
    {"payload_kb": 100, "request_overhead_ms": 26.56, "transfer_time_ms": 0.32851},
    {"payload_kb": 500, "request_overhead_ms": 26.56, "transfer_time_ms": 1.68920},
    {"payload_kb": 1000, "request_overhead_ms": 26.56, "transfer_time_ms": 3.25510},
    {"payload_kb": 5000, "request_overhead_ms": 26.56, "transfer_time_ms": 16.84980},
    {"payload_kb": 10000, "request_overhead_ms": 26.56, "transfer_time_ms": 32.74160},
]


def _write_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["payload_kb", "request_overhead_ms", "transfer_time_ms"])
        writer.writeheader()
        writer.writerows(ROWS)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true", help="Print the target path without writing.")
    args = ap.parse_args()
    if args.check_only:
        print(f"[figure1] would write {DATA_OUT}")
        return 0
    _write_csv(DATA_OUT)
    print(f"[ok] csv={DATA_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
