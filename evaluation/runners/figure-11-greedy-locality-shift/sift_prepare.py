#!/usr/bin/env python3
"""Build real SIFT1M IVF-PQ payloads and locality-stratified query ranges."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import boto3
import faiss
import numpy as np
from botocore.exceptions import ClientError


SEED = 20260805
NLIST = 4096
NPROBE = 128
DIMENSION = 128
PQ_M = 64
PQ_BITS = 8
PQ_CODE_BYTES = 64
PROTOCOL_ID = "figure11-sift-transfer-v1"
PROTOCOL_SLUG = "figure11-sift-transfer"
NUM_MEASURED = 200
NUM_WARMUPS = 20
POOL_SIZE = 5000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-npy", type=Path, required=True)
    parser.add_argument("--queries-npy", type=Path, required=True)
    parser.add_argument("--vecio-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--bucket", default=os.environ.get("VECIO_S3_BUCKET", ""))
    parser.add_argument("--region", default=os.environ.get("VECIO_S3_REGION", "ap-northeast-1"))
    parser.add_argument(
        "--prefix",
        default="figure11/sift-transfer-inputs-v1",
    )
    parser.add_argument("--threads", type=int, default=32)
    parser.add_argument("--ivf-iters", type=int, default=20)
    parser.add_argument("--pq-iters", type=int, default=20)
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


def upload_payload(
    path: Path,
    digest: str,
    *,
    bucket: str,
    region: str,
    prefix: str,
    layout: str,
) -> str:
    key = f"{prefix.rstrip('/')}/{layout}-pq64-{digest[:16]}.bin"
    client = boto3.client("s3", region_name=region)
    metadata = {
        "sha256": digest,
        "protocol": PROTOCOL_SLUG,
        "layout": layout,
        "record-bytes": str(PQ_CODE_BYTES),
    }
    try:
        head = client.head_object(Bucket=bucket, Key=key)
    except ClientError as error:
        code = str(error.response.get("Error", {}).get("Code", ""))
        if code not in {"404", "NoSuchKey", "NotFound"}:
            raise
        client.upload_file(str(path), bucket, key, ExtraArgs={"Metadata": metadata})
        head = client.head_object(Bucket=bucket, Key=key)
    if int(head["ContentLength"]) != path.stat().st_size:
        raise ValueError(f"{layout}: uploaded payload size mismatch")
    if head.get("Metadata", {}).get("sha256") != digest:
        raise ValueError(f"{layout}: uploaded payload digest metadata mismatch")
    return f"s3://{bucket}/{key}"


def load_arrays(base_path: Path, query_path: Path) -> tuple[np.ndarray, np.ndarray]:
    base = np.load(base_path, mmap_mode="r")
    queries = np.load(query_path, mmap_mode="r")
    if base.shape != (1_000_000, DIMENSION) or base.dtype != np.float32:
        raise ValueError(f"unexpected base array: {base.shape} {base.dtype}")
    if queries.shape != (10_000, DIMENSION) or queries.dtype != np.float32:
        raise ValueError(f"unexpected query array: {queries.shape} {queries.dtype}")
    return base, queries


def build_index(
    base: np.ndarray,
    *,
    threads: int,
    ivf_iters: int,
    pq_iters: int,
) -> faiss.IndexIVFPQ:
    faiss.omp_set_num_threads(int(threads))
    quantizer = faiss.IndexFlatL2(DIMENSION)
    index = faiss.IndexIVFPQ(
        quantizer,
        DIMENSION,
        NLIST,
        PQ_M,
        PQ_BITS,
        faiss.METRIC_L2,
    )
    index.cp.niter = int(ivf_iters)
    index.cp.seed = SEED
    index.cp.min_points_per_centroid = 1
    index.cp.max_points_per_centroid = 32
    index.cp.verbose = True
    index.pq.cp.niter = int(pq_iters)
    index.pq.cp.seed = SEED
    index.pq.cp.verbose = True
    index.verbose = True

    source = np.asarray(base, dtype=np.float32)
    logging.info("training IVF%d,PQ64x8 on %s", NLIST, source.shape)
    started = time.time()
    index.train(source)
    logging.info("training completed in %.2f seconds", time.time() - started)
    if not index.is_trained or int(index.code_size) != PQ_CODE_BYTES:
        raise ValueError("invalid trained IVF-PQ index")

    started = time.time()
    index.add(source)
    logging.info("adding vectors completed in %.2f seconds", time.time() - started)
    if int(index.ntotal) != len(base):
        raise ValueError(f"unexpected index total: {index.ntotal}")
    return index


def extract_centroids(index: faiss.IndexIVFPQ) -> np.ndarray:
    centroids = np.asarray(index.quantizer.reconstruct_n(0, NLIST), dtype=np.float32)
    if centroids.shape != (NLIST, DIMENSION):
        raise ValueError(f"unexpected centroid shape: {centroids.shape}")
    return centroids


def write_layout_payload(
    index: faiss.IndexIVFPQ,
    new_order: np.ndarray,
    path: Path,
) -> tuple[list[int], list[int], int]:
    if sorted(int(value) for value in new_order.tolist()) != list(range(NLIST)):
        raise ValueError("layout is not a permutation")
    invlists = index.invlists
    offsets: list[int] = []
    sizes: list[int] = []
    current = 0
    total_codes = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        for old_id in new_order.tolist():
            list_size = int(invlists.list_size(int(old_id)))
            byte_size = list_size * PQ_CODE_BYTES
            offsets.append(current)
            sizes.append(byte_size)
            if byte_size:
                pointer = invlists.get_codes(int(old_id))
                codes = np.array(
                    faiss.rev_swig_ptr(pointer, byte_size),
                    dtype=np.uint8,
                    copy=True,
                )
                if codes.size != byte_size:
                    raise ValueError(f"list {old_id}: code byte count mismatch")
                stream.write(codes.tobytes())
            current += byte_size
            total_codes += list_size
    if total_codes != 1_000_000 or path.stat().st_size != current:
        raise ValueError("payload code cardinality or size mismatch")
    return offsets, sizes, total_codes


def ranges_for_ids(
    ids: Sequence[int],
    offsets: Sequence[int],
    sizes: Sequence[int],
) -> list[list[int]]:
    output = []
    for cluster_id in sorted(set(int(value) for value in ids)):
        size = int(sizes[cluster_id])
        if size > 0:
            start = int(offsets[cluster_id])
            output.append([start, start + size])
    return output


def geometry(ranges: Sequence[Sequence[int]]) -> dict[str, float | int]:
    ordered = sorted((int(start), int(stop)) for start, stop in ranges)
    if not ordered:
        return {
            "required_bytes": 0,
            "covered_span_bytes": 0,
            "density": 0.0,
            "total_gap_bytes": 0,
            "nonempty_lists": 0,
        }
    required = sum(stop - start for start, stop in ordered)
    span = ordered[-1][1] - ordered[0][0]
    total_gap = sum(
        max(0, ordered[i + 1][0] - ordered[i][1])
        for i in range(len(ordered) - 1)
    )
    return {
        "required_bytes": int(required),
        "covered_span_bytes": int(span),
        "density": float(required / span) if span else 0.0,
        "total_gap_bytes": int(total_gap),
        "nonempty_lists": len(ordered),
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"count": len(rows)}
    for metric in (
        "required_bytes",
        "covered_span_bytes",
        "density",
        "total_gap_bytes",
        "nonempty_lists",
    ):
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        output[metric] = {
            "min": float(np.min(values)),
            "p10": float(np.percentile(values, 10)),
            "median": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "max": float(np.max(values)),
        }
    return output


def select_population(
    candidates: np.ndarray,
    *,
    seed: int,
) -> tuple[list[int], list[int]]:
    needed = NUM_WARMUPS + NUM_MEASURED
    if len(candidates) < needed:
        raise ValueError(f"only {len(candidates)} population candidates, need {needed}")
    rng = np.random.default_rng(seed)
    chosen = rng.choice(candidates, size=needed, replace=False).astype(np.int64)
    return chosen[NUM_WARMUPS:].tolist(), chosen[:NUM_WARMUPS].tolist()


def make_query_doc(
    *,
    population: str,
    layout: str,
    uri: str,
    offsets: Sequence[int],
    sizes: Sequence[int],
    remapped_ids: np.ndarray,
    measured_ids: Sequence[int],
    warmup_ids: Sequence[int],
) -> dict[str, Any]:
    measured_cluster_ids = [remapped_ids[index].astype(int).tolist() for index in measured_ids]
    warmup_cluster_ids = [remapped_ids[index].astype(int).tolist() for index in warmup_ids]
    measured_ranges = [ranges_for_ids(row, offsets, sizes) for row in measured_cluster_ids]
    warmup_ranges = [ranges_for_ids(row, offsets, sizes) for row in warmup_cluster_ids]
    selected_geometry = [geometry(row) for row in measured_ranges]
    return {
        "uri": uri,
        "dataset": "sift1m",
        "nlist": NLIST,
        "nprobe": NPROBE,
        "dimension": DIMENSION,
        "num_queries": NUM_MEASURED,
        "query_indices": [int(value) for value in measured_ids],
        "warmup_query_indices": [int(value) for value in warmup_ids],
        "reorder_method": layout,
        "payload_representation": "real_faiss_residual_pq64x8_codes",
        "payload_bytes_per_vector": PQ_CODE_BYTES,
        "cluster_blob_total_bytes": int(sum(sizes)),
        "cluster_offsets": [int(value) for value in offsets],
        "cluster_sizes": [int(value) for value in sizes],
        "query_cluster_ids": measured_cluster_ids,
        "warmup_query_cluster_ids": warmup_cluster_ids,
        "queries": measured_ranges,
        "warmup_queries": warmup_ranges,
        "query_population": population,
        "locality_summary": summarize(selected_geometry),
        "construction_seed": SEED,
    }


def main() -> int:
    args = parse_args()
    if not str(args.bucket).strip():
        raise SystemExit("--bucket or VECIO_S3_BUCKET is required")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    status_path = root / "prepare_status.txt"
    status_path.write_text("building_index\n", encoding="utf-8")

    base, queries = load_arrays(args.base_npy, args.queries_npy)
    index = build_index(
        base,
        threads=args.threads,
        ivf_iters=args.ivf_iters,
        pq_iters=args.pq_iters,
    )
    index_path = root / "model" / f"sift1m_ivf{NLIST}_pq64x8.faiss"
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    index_digest = sha256(index_path)

    centroids = extract_centroids(index)
    centroid_path = root / "model" / "ivf_centroids.npy"
    np.save(centroid_path, centroids)
    sys.path.insert(0, str(args.vecio_root))
    from src.cluster_reorder import ClusterReorder

    identity_order = np.arange(NLIST, dtype=np.int64)
    reorder = ClusterReorder(centroids, random_state=SEED)
    reordered_order = np.asarray(reorder.reorder("pca_nd_zorder"), dtype=np.int64)
    identity_map = np.arange(NLIST, dtype=np.int64)
    reordered_map = np.asarray(reorder.reorder_map, dtype=np.int64)
    np.save(root / "model" / "pca_nd_zorder_new_to_old.npy", reordered_order)
    np.save(root / "model" / "pca_nd_zorder_old_to_new.npy", reordered_map)

    status_path.write_text("materializing_payloads\n", encoding="utf-8")
    layouts: dict[str, dict[str, Any]] = {}
    for layout, order, old_to_new in (
        ("identity", identity_order, identity_map),
        ("pca_nd_zorder", reordered_order, reordered_map),
    ):
        payload_path = root / "staging" / f"{layout}.bin"
        offsets, sizes, total_codes = write_layout_payload(index, order, payload_path)
        digest = sha256(payload_path)
        uri = upload_payload(
            payload_path,
            digest,
            bucket=args.bucket,
            region=args.region,
            prefix=args.prefix,
            layout=layout,
        )
        layouts[layout] = {
            "uri": uri,
            "sha256": digest,
            "size_bytes": payload_path.stat().st_size,
            "total_codes": total_codes,
            "offsets": offsets,
            "sizes": sizes,
            "old_to_new": old_to_new,
        }

    status_path.write_text("routing_queries\n", encoding="utf-8")
    faiss.omp_set_num_threads(int(args.threads))
    _, raw_query_ids = index.quantizer.search(np.asarray(queries), NPROBE)
    if raw_query_ids.shape != (10_000, NPROBE) or np.any(raw_query_ids < 0):
        raise ValueError(f"unexpected routed query IDs: {raw_query_ids.shape}")
    remapped = {
        name: info["old_to_new"][raw_query_ids]
        for name, info in layouts.items()
    }

    geometry_rows = []
    reordered_info = layouts["pca_nd_zorder"]
    for query_id in range(len(queries)):
        ranges = ranges_for_ids(
            remapped["pca_nd_zorder"][query_id],
            reordered_info["offsets"],
            reordered_info["sizes"],
        )
        row = {"query_id": query_id, **geometry(ranges)}
        geometry_rows.append(row)

    geometry_path = root / "analysis" / "all_query_geometry.csv"
    geometry_path.parent.mkdir(parents=True, exist_ok=True)
    with geometry_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(geometry_rows[0].keys()))
        writer.writeheader()
        writer.writerows(geometry_rows)

    split_rng = np.random.default_rng(SEED)
    permutation = split_rng.permutation(len(queries))
    validation_pool = np.sort(permutation[:POOL_SIZE])
    heldout_pool = np.sort(permutation[POOL_SIZE:])
    densities = np.asarray([row["density"] for row in geometry_rows], dtype=np.float64)
    validation_density = densities[validation_pool]
    heldout_density = densities[heldout_pool]
    medium_low, medium_high = np.quantile(validation_density, [0.40, 0.60])
    loose_max = float(np.quantile(heldout_density, 0.10))
    medium_candidates = validation_pool[
        (validation_density >= medium_low) & (validation_density <= medium_high)
    ]
    loose_candidates = heldout_pool[heldout_density <= loose_max]
    medium_measured, medium_warmup = select_population(
        medium_candidates,
        seed=SEED + 1,
    )
    loose_measured, loose_warmup = select_population(
        loose_candidates,
        seed=SEED + 2,
    )

    selected_sets = {
        "medium_measured": set(medium_measured),
        "medium_warmup": set(medium_warmup),
        "loose_measured": set(loose_measured),
        "loose_warmup": set(loose_warmup),
    }
    names = list(selected_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            if selected_sets[left] & selected_sets[right]:
                raise ValueError(f"selected query sets overlap: {left}, {right}")
    if not selected_sets["medium_measured"].issubset(set(validation_pool.tolist())):
        raise ValueError("Medium measured IDs escaped validation pool")
    if not selected_sets["loose_measured"].issubset(set(heldout_pool.tolist())):
        raise ValueError("Loose measured IDs escaped held-out pool")

    outputs: dict[str, str] = {}
    for population, measured_ids, warmup_ids in (
        ("medium", medium_measured, medium_warmup),
        ("loose", loose_measured, loose_warmup),
    ):
        for layout_name, output_name in (
            ("identity", "original"),
            ("pca_nd_zorder", "reordered"),
        ):
            info = layouts[layout_name]
            doc = make_query_doc(
                population=population,
                layout=layout_name,
                uri=info["uri"],
                offsets=info["offsets"],
                sizes=info["sizes"],
                remapped_ids=remapped[layout_name],
                measured_ids=measured_ids,
                warmup_ids=warmup_ids,
            )
            path = root / "inputs" / population / f"{output_name}.json"
            write_json(path, doc)
            outputs[str(path.relative_to(root))] = sha256(path)

    medium_rows = [geometry_rows[index] for index in medium_measured]
    loose_rows = [geometry_rows[index] for index in loose_measured]
    geometry_summary = {
        "all_queries": summarize(geometry_rows),
        "validation_pool": summarize([geometry_rows[index] for index in validation_pool]),
        "heldout_pool": summarize([geometry_rows[index] for index in heldout_pool]),
        "medium_measured": summarize(medium_rows),
        "loose_measured": summarize(loose_rows),
    }
    if not (
        geometry_summary["medium_measured"]["density"]["median"]
        > geometry_summary["loose_measured"]["density"]["median"]
    ):
        raise ValueError("Medium density does not exceed Loose density")

    selection = {
        "seed": SEED,
        "validation_pool_size": len(validation_pool),
        "heldout_pool_size": len(heldout_pool),
        "medium_density_quantiles": [0.40, 0.60],
        "medium_density_bounds": [float(medium_low), float(medium_high)],
        "loose_density_quantile": 0.10,
        "loose_density_max": loose_max,
        "medium_candidate_count": len(medium_candidates),
        "loose_candidate_count": len(loose_candidates),
        "medium_measured_ids": medium_measured,
        "medium_warmup_ids": medium_warmup,
        "loose_measured_ids": loose_measured,
        "loose_warmup_ids": loose_warmup,
    }
    write_json(root / "analysis" / "selection_manifest.json", selection)
    write_json(root / "analysis" / "geometry_summary.json", geometry_summary)

    manifest = {
        "protocol_id": PROTOCOL_ID,
        "seed": SEED,
        "base_npy": {
            "path": str(args.base_npy),
            "sha256": sha256(args.base_npy),
            "shape": list(base.shape),
            "dtype": str(base.dtype),
        },
        "queries_npy": {
            "path": str(args.queries_npy),
            "sha256": sha256(args.queries_npy),
            "shape": list(queries.shape),
            "dtype": str(queries.dtype),
        },
        "faiss_version": getattr(faiss, "__version__", "unknown"),
        "index": {
            "type": "IndexIVFPQ",
            "metric": "L2",
            "nlist": NLIST,
            "nprobe": NPROBE,
            "pq_m": PQ_M,
            "pq_bits": PQ_BITS,
            "code_size_bytes": int(index.code_size),
            "ntotal": int(index.ntotal),
            "sha256": index_digest,
            "ivf_iters": args.ivf_iters,
            "pq_iters": args.pq_iters,
        },
        "layouts": {
            name: {
                "uri": info["uri"],
                "sha256": info["sha256"],
                "size_bytes": info["size_bytes"],
                "total_codes": info["total_codes"],
            }
            for name, info in layouts.items()
        },
        "selection_manifest_sha256": sha256(root / "analysis" / "selection_manifest.json"),
        "geometry_summary_sha256": sha256(root / "analysis" / "geometry_summary.json"),
        "all_query_geometry_sha256": sha256(geometry_path),
        "outputs": outputs,
    }
    write_json(root / "input_manifest.json", manifest)
    status_path.write_text("complete\n", encoding="utf-8")
    print(root / "input_manifest.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
