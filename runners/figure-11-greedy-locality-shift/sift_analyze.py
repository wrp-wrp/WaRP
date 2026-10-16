#!/usr/bin/env python3
"""Freeze calibration and summarize the SIFT 2x2 fixed-policy experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


METRICS = ("qps", "p95_latency_ms", "avg_num_requests", "avg_planned_bytes")
CELLS = (("tight", 32), ("loose", 32), ("tight", 256), ("loose", 256))
PROTOCOL_ID = "figure11-sift-transfer-v1"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def greedy_rows(doc: dict) -> list[dict]:
    return [
        row
        for row in doc.get("greedy_unbounded_candidates", [])
        if row.get("layout") == "reordered"
    ]


def warp_row(doc: dict) -> dict:
    rows = [
        row
        for row in doc.get("rows", [])
        if row.get("layout") == "reordered"
        and row.get("method") == "warp_reorder"
        and int(row.get("budget_k", -1)) == 128
    ]
    if len(rows) != 1:
        raise ValueError("missing unique WaRP P32/K128 row")
    return rows[0]


def summarize(rows: list[dict]) -> dict:
    output = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        output[f"{metric}_values"] = values
        output[f"median_{metric}"] = float(statistics.median(values))
    output["median_avg_planned_mib"] = output["median_avg_planned_bytes"] / (1024 * 1024)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    freeze = sub.add_parser("freeze")
    freeze.add_argument("--calibration", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    final = sub.add_parser("final")
    final.add_argument("--results-root", type=Path, required=True)
    final.add_argument("--freeze", type=Path, required=True)
    final.add_argument("--input-manifest", type=Path, required=True)
    final.add_argument("--geometry", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "freeze":
        doc = load(args.calibration)
        if int(doc.get("nprobe", -1)) != 32:
            raise ValueError("calibration is not nprobe=32")
        candidates = greedy_rows(doc)
        keys = {(int(row["gap_kb"]), int(row["span_cap_bytes"])) for row in candidates}
        if len(candidates) != 54 or len(keys) != 54:
            raise ValueError("calibration is not a complete 54-cell grid")
        selected = max(
            candidates,
            key=lambda row: (
                float(row["qps"]),
                -float(row["avg_num_requests"]),
                -int(row["gap_kb"]),
                -int(row["span_cap_bytes"]),
            ),
        )
        payload = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "selection_cell": {"population": "tight", "nprobe": 32},
            "greedy": {
                "rule": "maximum QPS in one complete 54-cell calibration surface",
                "gap_kib": int(selected["gap_kb"]),
                "span_cap_bytes": int(selected["span_cap_bytes"]),
                "selection_row": selected,
            },
            "warp": {
                "rule": "P=32 fixed by explicit user direction before shifted cells",
                "p": 32,
                "k": 128,
            },
            "calibration_sha256": sha256(args.calibration),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 0

    frozen = load(args.freeze)
    gap = int(frozen["greedy"]["gap_kib"])
    span = int(frozen["greedy"]["span_cap_bytes"])
    populations = []
    for population, nprobe in CELLS:
        paths = [
            args.results_root / f"{population}_nprobe{nprobe:04d}_repeat_{repeat:02d}.json"
            for repeat in (1, 2, 3)
        ]
        if not all(path.is_file() for path in paths):
            raise FileNotFoundError(f"incomplete cell: {population}/{nprobe}")
        greedy = []
        warp = []
        for path in paths:
            doc = load(path)
            if int(doc.get("nprobe", -1)) != nprobe:
                raise ValueError(f"{path}: wrong nprobe")
            candidates = greedy_rows(doc)
            if len(candidates) != 1:
                raise ValueError(f"{path}: expected one Greedy row")
            row = candidates[0]
            if (int(row["gap_kb"]), int(row["span_cap_bytes"])) != (gap, span):
                raise ValueError(f"{path}: Greedy policy differs from freeze")
            greedy.append(row)
            warp.append(warp_row(doc))
        greedy_summary = summarize(greedy)
        warp_summary = summarize(warp)
        populations.append(
            {
                "population": population,
                "nprobe": nprobe,
                "greedy_frozen_on_tight_nprobe32": greedy_summary,
                "warp_p32_k128": warp_summary,
                "warp_over_greedy_qps": warp_summary["median_qps"] / greedy_summary["median_qps"],
                "source_sha256": {path.name: sha256(path) for path in paths},
            }
        )
    baseline = populations[0]
    for cell in populations:
        cell["greedy_qps_retention_vs_tight_nprobe32"] = (
            cell["greedy_frozen_on_tight_nprobe32"]["median_qps"]
            / baseline["greedy_frozen_on_tight_nprobe32"]["median_qps"]
        )
        cell["warp_qps_retention_vs_tight_nprobe32"] = (
            cell["warp_p32_k128"]["median_qps"]
            / baseline["warp_p32_k128"]["median_qps"]
        )
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "aggregation": "median across three fixed-policy repeats per cell",
        "frozen_policies": frozen,
        "input_manifest_sha256": sha256(args.input_manifest),
        "geometry_sha256": sha256(args.geometry),
        "cells": populations,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
