#!/usr/bin/env python3
"""Validate one SIFT S3 benchmark result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GAPS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
SPANS = tuple(value * 1024 * 1024 for value in (1, 2, 4, 8, 16, 32))


def fail(message: str) -> None:
    raise SystemExit(f"[error] {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--expected-nlist", type=int, default=4096)
    parser.add_argument("--expected-nprobe", type=int, required=True)
    parser.add_argument("--expected-cells", type=int, required=True)
    parser.add_argument("--expected-gap-kib", type=int)
    parser.add_argument("--expected-span-mib", type=int)
    args = parser.parse_args()
    doc = json.loads(args.result.read_text(encoding="utf-8"))
    expected = {
        "dataset": "sift1m",
        "nlist": args.expected_nlist,
        "nprobe": args.expected_nprobe,
        "num_queries": 200,
        "query_concurrency": 4,
        "process_io_threads_limit": 64,
        "cloud_io_parallelism": 64,
    }
    for key, value in expected.items():
        if doc.get(key) != value:
            fail(f"unexpected {key}: {doc.get(key)!r}, expected {value!r}")
    if [float(value) for value in doc.get("balanced_ps", [])] != [32.0]:
        fail(f"unexpected WaRP P declaration: {doc.get('balanced_ps')!r}")

    greedy = [
        row
        for row in doc.get("greedy_unbounded_candidates", [])
        if row.get("layout") == "reordered"
    ]
    actual_keys = {(int(row["gap_kb"]), int(row["span_cap_bytes"])) for row in greedy}
    if args.expected_cells == 54:
        expected_keys = {(gap, span) for gap in GAPS for span in SPANS}
    elif args.expected_cells == 1:
        if args.expected_gap_kib is None or args.expected_span_mib is None:
            fail("fixed-policy validation requires expected gap and span")
        expected_keys = {
            (args.expected_gap_kib, args.expected_span_mib * 1024 * 1024)
        }
    else:
        fail(f"unsupported cell count: {args.expected_cells}")
    if len(greedy) != args.expected_cells or actual_keys != expected_keys:
        fail(f"unexpected Greedy grid: {len(greedy)} cells")

    warp = [
        row
        for row in doc.get("rows", [])
        if row.get("layout") == "reordered"
        and row.get("method") == "warp_reorder"
        and int(row.get("budget_k", -1)) == 128
    ]
    if len(warp) != 1 or warp[0].get("status") != "ok":
        fail("missing valid WaRP P32/K128 row")
    if abs(float(warp[0].get("coverage", 0.0)) - 1.0) > 1e-12:
        fail("WaRP coverage is not complete")
    print(
        json.dumps(
            {
                "result": str(args.result),
                "nprobe": args.expected_nprobe,
                "greedy_cells": len(greedy),
                "warp_qps": float(warp[0]["qps"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
