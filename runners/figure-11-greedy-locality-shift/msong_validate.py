#!/usr/bin/env python3
"""Validate one MSong strict-S3 result without selecting or interpreting it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GAPS = (64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384)
SPANS = tuple(value * 1048576 for value in (1, 2, 4, 8, 16, 32))


def fail(message: str) -> None:
    raise SystemExit(f"[error] {message}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    parser.add_argument("--arm", choices=("calibration", "greedy", "warp"), required=True)
    parser.add_argument("--gap-kib", type=int)
    parser.add_argument("--span-bytes", type=int)
    args = parser.parse_args()
    document = json.loads(args.result.read_text(encoding="utf-8"))
    expected = {
        "dataset": "msong",
        "nlist": 4096,
        "nprobe": 64,
        "num_queries": 200,
        "warmup_queries": 20,
        "query_concurrency": 4,
        "process_io_threads_limit": 64,
        "cloud_io_parallelism": 64,
    }
    for key, value in expected.items():
        if document.get(key) != value:
            fail(f"unexpected {key}: {document.get(key)!r}, expected {value!r}")
    if [float(value) for value in document.get("balanced_ps", [])] != [32.0]:
        fail("WaRP P is not exactly 32")

    greedy = [
        row
        for row in document.get("greedy_unbounded_candidates", [])
        if row.get("layout") == "reordered"
    ]
    warp = [
        row
        for row in document.get("rows", [])
        if row.get("layout") == "reordered"
        and row.get("method") == "warp_reorder"
        and int(row.get("budget_k", -1)) == 128
    ]
    if args.arm == "calibration":
        keys = {(int(row["gap_kb"]), int(row["span_cap_bytes"])) for row in greedy}
        if document.get("only_method") != "greedy_reorder":
            fail("calibration method filter mismatch")
        if len(greedy) != 54 or keys != {(gap, span) for gap in GAPS for span in SPANS}:
            fail("calibration Greedy grid is incomplete")
        if warp:
            fail("calibration unexpectedly executed WaRP")
    elif args.arm == "greedy":
        if document.get("only_method") != "greedy_reorder":
            fail("Greedy method filter mismatch")
        if args.gap_kib is None or args.span_bytes is None:
            fail("fixed Greedy validation requires policy parameters")
        if len(greedy) != 1:
            fail("expected one Greedy result")
        actual = (int(greedy[0]["gap_kb"]), int(greedy[0]["span_cap_bytes"]))
        if actual != (args.gap_kib, args.span_bytes):
            fail(f"Greedy policy mismatch: {actual}")
        if warp:
            fail("Greedy-only invocation unexpectedly executed WaRP")
    else:
        if document.get("only_method") != "warp_reorder":
            fail("WaRP method filter mismatch")
        if greedy:
            fail("WaRP-only invocation unexpectedly executed Greedy")
        if len(warp) != 1 or warp[0].get("status") != "ok":
            fail("missing valid WaRP P32/K128 result")
        if abs(float(warp[0].get("coverage", 0)) - 1.0) > 1e-12:
            fail("WaRP coverage is incomplete")
    print(json.dumps({"result": str(args.result), "arm": args.arm}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
