#!/usr/bin/env python3
"""Select a preregistered, disjoint MSong warmup/calibration/evaluation split."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SEED = 20260806
WARMUPS = 20
CALIBRATION = 200
EVALUATION = 200
TOTAL = WARMUPS + CALIBRATION + EVALUATION
EXPECTED_SOURCE_SHA256 = (
    "ea146b3dc0ef1dafc2fccdad363e7e584ffff2919430c8ba4f625c47c4938409"
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--output-npy", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    source_sha = sha256(args.queries)
    if source_sha != EXPECTED_SOURCE_SHA256:
        raise ValueError(f"unexpected MSong query SHA-256: {source_sha}")
    queries = np.load(args.queries, mmap_mode="r")
    if queries.ndim != 2 or queries.shape[0] != 10_000 or queries.shape[1] != 420:
        raise ValueError(f"unexpected MSong query shape: {queries.shape}")

    ids = np.random.default_rng(SEED).permutation(queries.shape[0])[:TOTAL]
    if len(set(map(int, ids))) != TOTAL:
        raise AssertionError("query selection is not unique")
    subset = np.asarray(queries[ids], dtype=np.float32)

    args.output_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output_npy, subset, allow_pickle=False)
    groups = {
        "warmup": [int(value) for value in ids[:WARMUPS]],
        "calibration": [int(value) for value in ids[WARMUPS : WARMUPS + CALIBRATION]],
        "evaluation": [int(value) for value in ids[WARMUPS + CALIBRATION :]],
    }
    document = {
        "schema_version": 1,
        "protocol_id": "figure11-msong-transfer-v1",
        "selection_algorithm": "numpy.default_rng(seed).permutation(10000)[:420]",
        "seed": SEED,
        "source_query_sha256": source_sha,
        "source_shape": list(queries.shape),
        "subset_query_sha256": sha256(args.output_npy),
        "subset_shape": list(subset.shape),
        "groups": groups,
    }
    args.output_json.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
