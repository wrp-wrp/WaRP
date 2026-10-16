#!/usr/bin/env python3
"""Freeze MSong Greedy tuning and summarize the cross-dataset transfer experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


PROTOCOL_ID = "figure11-msong-transfer-v1"
ARMS = ("sift_frozen_greedy", "msong_tuned_greedy", "warp_p32_k128")
EXPECTED_ORDERS = {
    1: ("sift_frozen_greedy", "msong_tuned_greedy", "warp_p32_k128"),
    2: ("msong_tuned_greedy", "warp_p32_k128", "sift_frozen_greedy"),
    3: ("warp_p32_k128", "sift_frozen_greedy", "msong_tuned_greedy"),
}
METRICS = ("qps", "p95_latency_ms", "avg_num_requests", "avg_planned_bytes")


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def greedy_rows(document: dict) -> list[dict]:
    return [
        row
        for row in document.get("greedy_unbounded_candidates", [])
        if row.get("layout") == "reordered"
    ]


def warp_rows(document: dict) -> list[dict]:
    return [
        row
        for row in document.get("rows", [])
        if row.get("layout") == "reordered"
        and row.get("method") == "warp_reorder"
        and int(row.get("budget_k", -1)) == 128
    ]


def summarize(rows: list[dict]) -> dict:
    output = {}
    for metric in METRICS:
        values = [float(row[metric]) for row in rows]
        output[f"{metric}_values"] = values
        output[f"median_{metric}"] = float(statistics.median(values))
    output["median_avg_planned_mib"] = output["median_avg_planned_bytes"] / 1048576
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--calibration", type=Path, required=True)
    freeze.add_argument("--sift-policy", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    final = subparsers.add_parser("final")
    final.add_argument("--results-root", type=Path, required=True)
    final.add_argument("--schedule", type=Path, required=True)
    final.add_argument("--freeze", type=Path, required=True)
    final.add_argument("--input-identity", type=Path, required=True)
    final.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.command == "freeze":
        calibration = load(args.calibration)
        candidates = greedy_rows(calibration)
        keys = {(int(row["gap_kb"]), int(row["span_cap_bytes"])) for row in candidates}
        if len(candidates) != 54 or len(keys) != 54:
            raise ValueError("MSong calibration is not a complete 54-cell surface")
        selected = max(
            candidates,
            key=lambda row: (
                float(row["qps"]),
                -float(row["avg_num_requests"]),
                -int(row["gap_kb"]),
                -int(row["span_cap_bytes"]),
            ),
        )
        sift = load(args.sift_policy)
        if sift.get("protocol_id") != "figure11-sift-transfer-v1":
            raise ValueError("unexpected SIFT policy identity")
        if (int(sift["greedy"]["gap_kib"]), int(sift["greedy"]["span_cap_bytes"])) != (
            128,
            1048576,
        ):
            raise ValueError("SIFT SIFT-frozen policy changed")
        payload = {
            "schema_version": 1,
            "protocol_id": PROTOCOL_ID,
            "sift_frozen_greedy": {
                "gap_kib": 128,
                "span_cap_bytes": 1048576,
                "source_sha256": sha256(args.sift_policy),
            },
            "msong_tuned_greedy": {
                "rule": "maximum measured QPS in one complete 54-cell calibration surface",
                "gap_kib": int(selected["gap_kb"]),
                "span_cap_bytes": int(selected["span_cap_bytes"]),
                "selection_row": selected,
                "calibration_sha256": sha256(args.calibration),
            },
            "warp": {"p": 32, "k": 128, "retuned_on_msong": False},
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return 0

    frozen = load(args.freeze)
    if frozen.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("frozen-policy protocol mismatch")
    schedule_rows = [line.split("\t") for line in args.schedule.read_text().splitlines()[1:]]
    observed_orders: dict[int, list[str]] = {repeat: [] for repeat in (1, 2, 3)}
    sources = {}
    arm_rows: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    for repeat_text, order_text, arm, filename in schedule_rows:
        repeat, order = int(repeat_text), int(order_text)
        observed_orders[repeat].append(arm)
        path = args.results_root / filename
        document = load(path)
        if arm == "warp_p32_k128":
            rows = warp_rows(document)
        else:
            rows = greedy_rows(document)
        if len(rows) != 1:
            raise ValueError(f"{path}: expected one reported row for {arm}")
        arm_rows[arm].append(rows[0])
        sources[f"repeat_{repeat}_order_{order}_{arm}"] = sha256(path)
    for repeat, expected in EXPECTED_ORDERS.items():
        if tuple(observed_orders[repeat]) != expected:
            raise ValueError(f"repeat {repeat} method order mismatch")
    if any(len(rows) != 3 for rows in arm_rows.values()):
        raise ValueError("formal result matrix is incomplete")

    summaries = {arm: summarize(rows) for arm, rows in arm_rows.items()}
    sift_qps = summaries["sift_frozen_greedy"]["median_qps"]
    msong_qps = summaries["msong_tuned_greedy"]["median_qps"]
    warp_qps = summaries["warp_p32_k128"]["median_qps"]
    retention = sift_qps / msong_qps
    payload = {
        "schema_version": 1,
        "status": "complete",
        "protocol_id": PROTOCOL_ID,
        "aggregation": "median across three order-balanced fixed-policy repeats",
        "frozen_policies": frozen,
        "input_identity_sha256": sha256(args.input_identity),
        "method_order": {str(key): list(value) for key, value in EXPECTED_ORDERS.items()},
        "arms": summaries,
        "primary": {
            "greedy_cross_dataset_transfer_retention": retention,
            "greedy_cross_dataset_transfer_loss": 1.0 - retention,
            "warp_over_sift_frozen_greedy_qps": warp_qps / sift_qps,
            "warp_over_msong_tuned_greedy_qps": warp_qps / msong_qps,
        },
        "source_sha256": sources,
        "failure_violations": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
