#!/usr/bin/env python3
"""Build SIFT real-query 2x2 locality and nprobe-shift inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import faiss
import numpy as np
from scipy.optimize import linear_sum_assignment


NLIST = 4096
DIMENSION = 128
NPROBES = (32, 256)
NUM_MEASURED = 200
NUM_WARMUPS = 20
SPLIT_SEED = 20260807
WARMUP_SEED = 20260809
TAIL_FRACTION = 0.10
PROTOCOL_ID = "figure11-sift-transfer-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-root", type=Path, required=True)
    parser.add_argument("--queries-npy", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=32)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def ranges_for_ids(
    ids: Sequence[int], offsets: Sequence[int], sizes: Sequence[int]
) -> list[list[int]]:
    output = []
    for cluster_id in sorted(set(int(value) for value in ids)):
        size = int(sizes[cluster_id])
        if size:
            start = int(offsets[cluster_id])
            output.append([start, start + size])
    return output


def geometry(ranges: Sequence[Sequence[int]]) -> dict[str, float | int]:
    ordered = sorted((int(start), int(stop)) for start, stop in ranges)
    required = sum(stop - start for start, stop in ordered)
    span = ordered[-1][1] - ordered[0][0]
    gaps = sum(
        max(0, ordered[index + 1][0] - ordered[index][1])
        for index in range(len(ordered) - 1)
    )
    return {
        "required_bytes": int(required),
        "covered_span_bytes": int(span),
        "fragmentation": float(span / required),
        "density": float(required / span),
        "total_gap_bytes": int(gaps),
        "nonempty_lists": len(ordered),
    }


def percentile_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks / max(1, len(values) - 1)


def summarize(rows: Sequence[dict[str, float | int]]) -> dict[str, Any]:
    output: dict[str, Any] = {"count": len(rows)}
    for metric in (
        "required_bytes",
        "covered_span_bytes",
        "fragmentation",
        "density",
        "total_gap_bytes",
        "nonempty_lists",
    ):
        values = np.asarray([float(row[metric]) for row in rows])
        output[metric] = {
            "min": float(np.min(values)),
            "p10": float(np.percentile(values, 10)),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "max": float(np.max(values)),
        }
    return output


def symmetric_relative(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return np.abs(left - right) / ((left + right) / 2.0)


def make_doc(
    *,
    population: str,
    nprobe: int,
    layout: str,
    source: dict,
    routed_ids: np.ndarray,
    measured_ids: Sequence[int],
    warmup_ids: Sequence[int],
) -> dict[str, Any]:
    offsets = source["cluster_offsets"]
    sizes = source["cluster_sizes"]
    measured_clusters = [routed_ids[index, :nprobe].astype(int).tolist() for index in measured_ids]
    warmup_clusters = [routed_ids[index, :nprobe].astype(int).tolist() for index in warmup_ids]
    measured_ranges = [ranges_for_ids(row, offsets, sizes) for row in measured_clusters]
    warmup_ranges = [ranges_for_ids(row, offsets, sizes) for row in warmup_clusters]
    return {
        "uri": source["uri"],
        "dataset": "sift1m",
        "nlist": NLIST,
        "nprobe": nprobe,
        "dimension": DIMENSION,
        "num_queries": NUM_MEASURED,
        "query_indices": [int(value) for value in measured_ids],
        "warmup_query_indices": [int(value) for value in warmup_ids],
        "reorder_method": layout,
        "payload_representation": "real_faiss_residual_pq64x8_codes",
        "payload_bytes_per_vector": 64,
        "cluster_blob_total_bytes": int(sum(sizes)),
        "cluster_offsets": [int(value) for value in offsets],
        "cluster_sizes": [int(value) for value in sizes],
        "query_cluster_ids": measured_clusters,
        "warmup_query_cluster_ids": warmup_clusters,
        "queries": measured_ranges,
        "warmup_queries": warmup_ranges,
        "query_population": population,
        "locality_summary": summarize([geometry(row) for row in measured_ranges]),
        "construction_seeds": {"split": SPLIT_SEED, "warmup": WARMUP_SEED},
    }


def main() -> int:
    args = parse_args()
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    (root / "status.txt").write_text("routing_queries\n", encoding="utf-8")
    base_manifest = json.loads((args.base_root / "input_manifest.json").read_text())
    index_path = args.base_root / "model" / f"sift1m_ivf{NLIST}_pq64x8.faiss"
    if sha256(index_path) != base_manifest["index"]["sha256"]:
        raise ValueError("base Faiss index hash mismatch")
    if sha256(args.queries_npy) != base_manifest["queries_npy"]["sha256"]:
        raise ValueError("official SIFT query hash mismatch")

    queries = np.load(args.queries_npy, mmap_mode="r")
    if queries.shape != (10_000, DIMENSION) or queries.dtype != np.float32:
        raise ValueError("unexpected query array")
    faiss.omp_set_num_threads(int(args.threads))
    index = faiss.read_index(str(index_path))
    _, raw_ids = index.quantizer.search(np.asarray(queries), max(NPROBES))
    if raw_ids.shape != (10_000, max(NPROBES)) or np.any(raw_ids < 0):
        raise ValueError("unexpected IVF routing output")

    old_to_new = np.load(args.base_root / "model" / "pca_nd_zorder_old_to_new.npy")
    if sorted(int(value) for value in old_to_new.tolist()) != list(range(NLIST)):
        raise ValueError("invalid physical reorder map")
    reordered_ids = old_to_new[raw_ids]
    source_inputs = args.base_root / "inputs_matched_v2"
    if not source_inputs.is_dir():
        source_inputs = args.base_root / "inputs"
    original_source = json.loads((source_inputs / "medium" / "original.json").read_text())
    reordered_source = json.loads((source_inputs / "medium" / "reordered.json").read_text())

    all_geometry: dict[int, list[dict[str, float | int]]] = {}
    for nprobe in NPROBES:
        all_geometry[nprobe] = [
            geometry(
                ranges_for_ids(
                    reordered_ids[query_id, :nprobe],
                    reordered_source["cluster_offsets"],
                    reordered_source["cluster_sizes"],
                )
            )
            for query_id in range(10_000)
        ]

    split = np.random.default_rng(SPLIT_SEED).permutation(10_000)
    selection_pool = np.sort(split[:5000])
    heldout_pool = np.sort(split[5000:])

    residuals: dict[int, np.ndarray] = {}
    regression_coefficients: dict[str, list[float]] = {}
    for nprobe in NPROBES:
        required_values = np.asarray(
            [row["required_bytes"] for row in all_geometry[nprobe]], dtype=np.float64
        )
        fragmentation_values = np.asarray(
            [row["fragmentation"] for row in all_geometry[nprobe]], dtype=np.float64
        )
        x = np.log(required_values)
        design = np.column_stack((np.ones(len(x)), x, x * x))
        coefficients, _, _, _ = np.linalg.lstsq(
            design, np.log(fragmentation_values), rcond=None
        )
        residuals[nprobe] = np.log(fragmentation_values) - design @ coefficients
        regression_coefficients[str(nprobe)] = coefficients.astype(float).tolist()

    def scores(pool: np.ndarray) -> np.ndarray:
        endpoint_ranks = []
        for nprobe in NPROBES:
            values = residuals[nprobe][pool]
            endpoint_ranks.append(percentile_ranks(values))
        return np.mean(np.stack(endpoint_ranks), axis=0)

    selection_scores = scores(selection_pool)
    heldout_scores = scores(heldout_pool)
    tail_size = int(round(len(selection_pool) * TAIL_FRACTION))
    tight_candidates = selection_pool[np.argsort(selection_scores, kind="stable")[:tail_size]]
    loose_candidates = heldout_pool[np.argsort(heldout_scores, kind="stable")[-tail_size:]]

    tight_required = np.asarray(
        [[all_geometry[nprobe][query_id]["required_bytes"] for nprobe in NPROBES] for query_id in tight_candidates],
        dtype=np.float64,
    )
    loose_required = np.asarray(
        [[all_geometry[nprobe][query_id]["required_bytes"] for nprobe in NPROBES] for query_id in loose_candidates],
        dtype=np.float64,
    )
    cost = np.sum(
        np.abs(np.log(tight_required[:, None, :] / loose_required[None, :, :])), axis=2
    )
    tight_assignment, loose_assignment = linear_sum_assignment(cost)
    paired_tight = tight_candidates[tight_assignment]
    paired_loose = loose_candidates[loose_assignment]
    relative = symmetric_relative(
        tight_required[tight_assignment], loose_required[loose_assignment]
    )
    pair_order = np.argsort(np.max(relative, axis=1), kind="stable")
    needed = NUM_WARMUPS + NUM_MEASURED
    chosen = pair_order[:needed]
    if len(chosen) != needed:
        raise ValueError("not enough extreme matched pairs")
    chosen = np.random.default_rng(WARMUP_SEED).permutation(chosen)
    tight_ids = paired_tight[chosen]
    loose_ids = paired_loose[chosen]
    pair_relative = relative[chosen]
    tight_warmup = tight_ids[:NUM_WARMUPS].astype(int).tolist()
    loose_warmup = loose_ids[:NUM_WARMUPS].astype(int).tolist()
    tight_measured = tight_ids[NUM_WARMUPS:].astype(int).tolist()
    loose_measured = loose_ids[NUM_WARMUPS:].astype(int).tolist()
    if set(tight_ids.tolist()) & set(loose_ids.tolist()):
        raise ValueError("Tight and Loose IDs overlap")

    outputs = {}
    for nprobe in NPROBES:
        for population, measured, warmup in (
            ("tight", tight_measured, tight_warmup),
            ("loose", loose_measured, loose_warmup),
        ):
            for layout, source, ids, filename in (
                ("identity", original_source, raw_ids, "original.json"),
                ("pca_nd_zorder", reordered_source, reordered_ids, "reordered.json"),
            ):
                path = root / "inputs" / f"nprobe_{nprobe:04d}" / population / filename
                write_json(
                    path,
                    make_doc(
                        population=population,
                        nprobe=nprobe,
                        layout=layout,
                        source=source,
                        routed_ids=ids,
                        measured_ids=measured,
                        warmup_ids=warmup,
                    ),
                )
                outputs[str(path.relative_to(root))] = sha256(path)

    geometry_summary = {"populations": {}, "pair_matching": {}}
    for nprobe in NPROBES:
        geometry_summary["populations"][str(nprobe)] = {
            "tight": summarize([all_geometry[nprobe][idx] for idx in tight_measured]),
            "loose": summarize([all_geometry[nprobe][idx] for idx in loose_measured]),
        }
        endpoint = NPROBES.index(nprobe)
        values = pair_relative[NUM_WARMUPS:, endpoint]
        geometry_summary["pair_matching"][str(nprobe)] = {
            "median_relative_required_bytes_difference": float(np.median(values)),
            "p90_relative_required_bytes_difference": float(np.percentile(values, 90)),
            "max_relative_required_bytes_difference": float(np.max(values)),
        }
        tight_median = geometry_summary["populations"][str(nprobe)]["tight"]["required_bytes"]["median"]
        loose_median = geometry_summary["populations"][str(nprobe)]["loose"]["required_bytes"]["median"]
        aggregate_relative = abs(tight_median - loose_median) / ((tight_median + loose_median) / 2)
        geometry_summary["pair_matching"][str(nprobe)][
            "aggregate_median_required_bytes_relative_difference"
        ] = float(aggregate_relative)
        if aggregate_relative > 0.05:
            write_json(root / "analysis" / "geometry_diagnostic.json", geometry_summary)
            raise ValueError(f"nprobe={nprobe}: aggregate payload differs by more than 5%")
        if not (
            geometry_summary["populations"][str(nprobe)]["tight"]["fragmentation"]["median"]
            < geometry_summary["populations"][str(nprobe)]["loose"]["fragmentation"]["median"]
        ):
            raise ValueError(f"nprobe={nprobe}: extreme locality ordering failed")

    selection_manifest = {
        "schema_version": 1,
        "selection_is_performance_blind": True,
        "split_seed": SPLIT_SEED,
        "warmup_seed": WARMUP_SEED,
        "tail_fraction": TAIL_FRACTION,
        "tail_size": tail_size,
        "nprobe_endpoints": list(NPROBES),
        "conditional_fragmentation_model": {
            "response": "log(covered_span_bytes / required_bytes)",
            "predictors": ["1", "log(required_bytes)", "log(required_bytes)^2"],
            "coefficients_by_nprobe": regression_coefficients,
        },
        "tight_measured_ids": tight_measured,
        "tight_warmup_ids": tight_warmup,
        "loose_measured_ids": loose_measured,
        "loose_warmup_ids": loose_warmup,
    }
    selection_path = root / "analysis" / "selection_manifest.json"
    geometry_path = root / "analysis" / "geometry_summary.json"
    write_json(selection_path, selection_manifest)
    write_json(geometry_path, geometry_summary)
    manifest = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "base_input_manifest_sha256": sha256(args.base_root / "input_manifest.json"),
        "faiss_index_sha256": sha256(index_path),
        "queries_npy_sha256": sha256(args.queries_npy),
        "selection_manifest_sha256": sha256(selection_path),
        "geometry_summary_sha256": sha256(geometry_path),
        "outputs": outputs,
    }
    write_json(root / "input_manifest.json", manifest)
    (root / "status.txt").write_text("prepared_waiting_geometry_review\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
