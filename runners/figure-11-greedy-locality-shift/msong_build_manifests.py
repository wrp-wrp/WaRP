#!/usr/bin/env python3
"""Turn an exact-Lance payload trace into MSong strict-S3 query manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROTOCOL_ID = "figure11-msong-transfer-v1"
WARMUPS = 20
CALIBRATION = 200
EVALUATION = 200


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_ranges(queries: list, object_size: int) -> None:
    if len(queries) != WARMUPS + CALIBRATION + EVALUATION:
        raise ValueError(f"expected 420 traced queries, got {len(queries)}")
    for query_index, ranges in enumerate(queries):
        if not ranges:
            raise ValueError(f"query {query_index} has no payload ranges")
        previous_start = -1
        for value in ranges:
            if not isinstance(value, list) or len(value) != 2:
                raise ValueError(f"query {query_index}: malformed range {value!r}")
            start, end = map(int, value)
            if start < 0 or end <= start or end > object_size:
                raise ValueError(f"query {query_index}: out-of-bounds range {value!r}")
            if start < previous_start:
                raise ValueError(f"query {query_index}: ranges are not ordered")
            previous_start = start


def manifest(uri: str, warmups: list, measured: list) -> dict:
    return {
        "uri": uri,
        "nlist": 4096,
        "nprobe": 64,
        "queries": measured,
        "warmup_queries": warmups,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--dataset-uri", required=True)
    parser.add_argument("--expected-index-uuid", required=True)
    parser.add_argument("--object-size", type=int, required=True)
    parser.add_argument("--object-etag", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    trace = json.loads(args.trace.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if selection.get("protocol_id") != PROTOCOL_ID:
        raise ValueError("selection protocol identity mismatch")
    observed = sorted(set(map(str, trace.get("index_uuids", []))))
    if observed != [args.expected_index_uuid]:
        raise ValueError(f"unexpected selected index UUID(s): {observed}")
    queries = trace.get("queries")
    if not isinstance(queries, list):
        raise ValueError("trace has no query list")
    validate_ranges(queries, args.object_size)

    prefix = args.dataset_uri.rstrip("/")
    object_uri = f"{prefix}/_indices/{args.expected_index_uuid}/auxiliary.idx"
    warmups = queries[:WARMUPS]
    calibration = queries[WARMUPS : WARMUPS + CALIBRATION]
    evaluation = queries[WARMUPS + CALIBRATION :]
    args.output_root.mkdir(parents=True, exist_ok=True)
    calibration_path = args.output_root / "calibration.json"
    evaluation_path = args.output_root / "evaluation.json"
    calibration_path.write_text(
        json.dumps(manifest(object_uri, warmups, calibration), separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    evaluation_path.write_text(
        json.dumps(manifest(object_uri, warmups, evaluation), separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    identity = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "dataset_uri": args.dataset_uri,
        "selected_index_uuid": args.expected_index_uuid,
        "payload_object_uri": object_uri,
        "payload_object_size": args.object_size,
        "payload_object_etag": args.object_etag,
        "trace_sha256": sha256(args.trace),
        "selection_sha256": sha256(args.selection),
        "calibration_manifest_sha256": sha256(calibration_path),
        "evaluation_manifest_sha256": sha256(evaluation_path),
        "warmup_queries": WARMUPS,
        "calibration_queries": CALIBRATION,
        "evaluation_queries": EVALUATION,
        "nlist": 4096,
        "nprobe": 64,
        "layout": "pca_nd_zorder",
    }
    (args.output_root / "input_identity.json").write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
