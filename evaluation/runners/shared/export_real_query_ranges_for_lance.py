#!/usr/bin/env python3
"""Export real query byte-ranges for Lance scheduler QPS benchmark.

This script builds a real IVF index from dataset vectors, optionally applies
cluster reorder, uploads the merged cluster blob to S3, and emits a JSON file
containing per-query byte ranges:

{
  "uri": "s3://bucket/key",
  "dataset": "sift1m",
  "nlist": 1000,
  "nprobe": 128,
  "queries": [[[start, end], ...], ...],
  "warmup_queries": [[[start, end], ...], ...]
}
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import boto3
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cluster_reorder import ClusterReorder
from src.datasets import get_dataset
from src.index_builder import IVFIndexBuilder

logger = logging.getLogger("export_real_query_ranges_for_lance")

REORDER_METHOD_CHOICES = [
    "identity",
    "random",
    "pca",
    "pca_zorder",
    "pca_nd_zorder",
    "lpp",
    "tsne",
    "hilbert",
]


def _subset_queries(
    queries: np.ndarray,
    num_queries: int,
    mode: str,
    seed: int,
) -> Tuple[np.ndarray, np.ndarray]:
    if num_queries <= 0 or num_queries >= len(queries):
        idx = np.arange(len(queries), dtype=np.int64)
        return queries, idx
    if mode in {"random", "rand", "sample"}:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(queries), size=int(num_queries), replace=False)
        idx = np.sort(idx)
        return queries[idx], idx
    idx = np.arange(int(num_queries), dtype=np.int64)
    return queries[:num_queries], idx


def _to_ranges_for_query(
    cluster_ids: Sequence[int],
    offsets: Dict[int, int],
    sizes: Dict[int, int],
) -> List[List[int]]:
    out: List[List[int]] = []
    for cid in sorted(set(int(x) for x in cluster_ids)):
        size = int(sizes.get(cid, 0))
        if size <= 0:
            continue
        start = int(offsets[cid])
        out.append([start, start + size])
    return out


def _summarize_locality(
    reorder: ClusterReorder,
    remapped_cluster_ids: Sequence[Sequence[int]],
) -> Dict[str, Dict[str, float]]:
    keys = ["span", "density", "avg_gap", "max_gap", "num_segments"]
    cols: Dict[str, List[float]] = {k: [] for k in keys}
    for row in remapped_cluster_ids:
        m = reorder.compute_locality_metrics(np.asarray(row, dtype=np.int64))
        for k in keys:
            cols[k].append(float(m.get(k, 0.0)))

    out: Dict[str, Dict[str, float]] = {}
    for k in keys:
        arr = np.asarray(cols[k], dtype=np.float64)
        if arr.size == 0:
            out[k] = {"mean": 0.0, "median": 0.0, "p95": 0.0, "min": 0.0, "max": 0.0}
            continue
        out[k] = {
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
        }
    return out


def _upload_merged_clusters(
    cluster_vectors: Dict[int, np.ndarray],
    new_order: np.ndarray,
    bucket: str,
    key: str,
    region: str,
) -> Tuple[Dict[int, int], Dict[int, int], int]:
    s3 = boto3.client("s3", region_name=region)
    offsets: Dict[int, int] = {}
    sizes: Dict[int, int] = {}
    current = 0

    with tempfile.NamedTemporaryFile(prefix="lance_real_ranges_", suffix=".bin", delete=False) as tmp:
        tmp_path = Path(tmp.name)
        for new_id, old_id in enumerate(new_order.tolist()):
            vecs = cluster_vectors[int(old_id)]
            if len(vecs) > 0:
                payload = np.asarray(vecs, dtype=np.float32).tobytes()
            else:
                payload = b""
            offsets[int(new_id)] = current
            sizes[int(new_id)] = len(payload)
            if payload:
                tmp.write(payload)
            current += len(payload)

    logger.info(
        "Uploading merged cluster blob to s3://%s/%s (%.2f MB)",
        bucket,
        key,
        current / 1024.0 / 1024.0,
    )
    s3.upload_file(str(tmp_path), bucket, key)
    tmp_path.unlink(missing_ok=True)
    return offsets, sizes, current


def _compute_offsets_and_sizes(
    cluster_vectors: Dict[int, np.ndarray],
    new_order: np.ndarray,
) -> Tuple[Dict[int, int], Dict[int, int], int]:
    offsets: Dict[int, int] = {}
    sizes: Dict[int, int] = {}
    current = 0
    for new_id, old_id in enumerate(new_order.tolist()):
        vecs = cluster_vectors[int(old_id)]
        byte_size = int(np.asarray(vecs, dtype=np.float32).nbytes) if len(vecs) > 0 else 0
        offsets[int(new_id)] = current
        sizes[int(new_id)] = byte_size
        current += byte_size
    return offsets, sizes, current


def _parse_extra_reorder_outputs(raw_items: Sequence[str]) -> List[Tuple[str, Path]]:
    specs: List[Tuple[str, Path]] = []
    allowed = set(REORDER_METHOD_CHOICES)
    for raw in raw_items:
        if "=" not in raw:
            raise SystemExit(
                "--extra-reorder-output must use METHOD=PATH, "
                f"got {raw!r}"
            )
        method, path = raw.split("=", 1)
        method = method.strip().lower()
        if method not in allowed:
            raise SystemExit(
                f"unsupported reorder method in --extra-reorder-output: {method!r}"
            )
        path = path.strip()
        if not path:
            raise SystemExit("--extra-reorder-output path cannot be empty")
        specs.append((method, Path(path)))
    return specs


def _safe_s3_component(raw: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in raw)


def _write_layout_output(
    *,
    dataset_key: str,
    dim: int,
    nlist: int,
    nprobe: int,
    query_idx: np.ndarray,
    raw_nprobe_results: Sequence[np.ndarray],
    cluster_centers: np.ndarray,
    cluster_vectors: Dict[int, np.ndarray],
    reorder_method: str,
    bucket: str,
    region: str,
    s3_key: str,
    existing_uri: str,
    out_path: Path,
    warmup_queries: int,
) -> None:
    reorder = ClusterReorder(cluster_centers)
    new_order = reorder.reorder(reorder_method)
    reorder_map = reorder.reorder_map

    if existing_uri:
        offsets, sizes, total_bytes = _compute_offsets_and_sizes(
            cluster_vectors=cluster_vectors,
            new_order=new_order,
        )
        uri = existing_uri
        logger.info("Reusing existing merged cluster blob: %s", uri)
    else:
        offsets, sizes, total_bytes = _upload_merged_clusters(
            cluster_vectors=cluster_vectors,
            new_order=new_order,
            bucket=bucket,
            key=s3_key,
            region=region,
        )
        uri = f"s3://{bucket}/{s3_key}"

    nprobe_results: List[List[int]] = []
    for cid in raw_nprobe_results:
        remapped = reorder_map[np.asarray(cid, dtype=np.int64)]
        nprobe_results.append(remapped.tolist())

    query_ranges = [_to_ranges_for_query(cids, offsets, sizes) for cids in nprobe_results]

    warm_n = min(max(0, int(warmup_queries)), len(query_ranges))
    warmup_ranges = query_ranges[:warm_n]

    out = {
        "uri": uri,
        "dataset": dataset_key,
        "nlist": nlist,
        "nprobe": nprobe,
        "dimension": dim,
        "num_queries": int(len(query_ranges)),
        "query_indices": [int(x) for x in query_idx.tolist()],
        "reorder_method": reorder_method,
        "cluster_blob_total_bytes": int(total_bytes),
        "cluster_offsets": [int(offsets.get(i, 0)) for i in range(nlist)],
        "cluster_sizes": [int(sizes.get(i, 0)) for i in range(nlist)],
        "query_cluster_ids": [[int(x) for x in row] for row in nprobe_results],
        "locality_summary": _summarize_locality(reorder, nprobe_results),
        "queries": query_ranges,
        "warmup_queries": warmup_ranges,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    logger.info("Wrote query ranges file: %s", out_path)
    logger.info("URI: %s", uri)
    logger.info(
        "Exported %d queries (warmup=%d), avg ranges/query=%.2f",
        len(query_ranges),
        len(warmup_ranges),
        float(np.mean([len(q) for q in query_ranges])) if query_ranges else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export real query ranges for Lance QPS benchmark")
    parser.add_argument("--dataset", required=True, help="Dataset name, e.g. sift1m/glove/coco")
    parser.add_argument("--nlist", type=int, required=True, help="IVF cluster count")
    parser.add_argument("--nprobe", type=int, required=True, help="nprobe per query")
    parser.add_argument("--num-queries", type=int, default=200, help="Number of queries to export")
    parser.add_argument("--warmup-queries", type=int, default=20, help="Warmup query count")
    parser.add_argument(
        "--query-subset-mode",
        type=str,
        default="random",
        choices=["random", "first"],
        help="How to pick query subset",
    )
    parser.add_argument("--query-subset-seed", type=int, default=12345, help="Seed for random query subset")
    parser.add_argument("--train-size", type=int, default=0, help="Optional IVF train_size (0 = all)")
    parser.add_argument(
        "--reorder-method",
        type=str,
        default="pca_nd_zorder",
        choices=REORDER_METHOD_CHOICES,
        help="Cluster reorder method",
    )
    parser.add_argument(
        "--extra-reorder-output",
        action="append",
        default=[],
        help=(
            "Additional layout output as METHOD=PATH. The IVF index is built "
            "once, then each requested layout is emitted from the same clusters."
        ),
    )
    parser.add_argument("--s3-bucket", required=True, help="S3 bucket")
    parser.add_argument("--s3-region", default="ap-northeast-1", help="S3 region")
    parser.add_argument(
        "--s3-key",
        default="",
        help="S3 object key; if empty use pipeline/<dataset>/<ts>/clusters_merged.bin",
    )
    parser.add_argument(
        "--existing-uri",
        default="",
        help="Reuse an existing merged cluster blob URI instead of uploading a new object.",
    )
    parser.add_argument("--out", required=True, help="Output JSON path")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    dataset_key = str(args.dataset).strip().lower()
    ds = get_dataset(dataset_key)
    ds.prepare()
    base = ds.get_base_vectors()
    queries_all = ds.get_queries()
    dim = int(base.shape[1])

    selected_queries, query_idx = _subset_queries(
        queries_all,
        int(args.num_queries),
        str(args.query_subset_mode).strip().lower(),
        int(args.query_subset_seed),
    )

    logger.info(
        "Dataset=%s base=%s queries=%s selected=%s dim=%d",
        dataset_key,
        tuple(base.shape),
        tuple(queries_all.shape),
        tuple(selected_queries.shape),
        dim,
    )

    primary_method = str(args.reorder_method).strip().lower()
    layout_specs = [(primary_method, Path(args.out))]
    layout_specs.extend(_parse_extra_reorder_outputs(args.extra_reorder_output))
    if len(layout_specs) > 1 and str(args.s3_key).strip():
        raise SystemExit("--s3-key is only supported for single-layout export")

    builder = IVFIndexBuilder(dimension=dim, num_clusters=int(args.nlist))
    train_size = int(args.train_size) if int(args.train_size) > 0 else None
    builder.build(base, train_size=train_size)
    _ = builder.extract_inverted_lists()
    cluster_vectors = builder.get_cluster_vectors(base)

    existing_uri = str(args.existing_uri).strip()
    raw_nprobe_results: List[np.ndarray] = []
    for q in selected_queries:
        cid = builder.find_nearest_clusters(np.asarray(q).reshape(1, -1), int(args.nprobe))
        raw_nprobe_results.append(np.asarray(cid, dtype=np.int64))

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    multi_layout = len(layout_specs) > 1
    for reorder_method, out_path in layout_specs:
        if args.s3_key:
            key = str(args.s3_key).strip()
        elif multi_layout:
            key = (
                f"pipeline/{dataset_key}/{ts}/"
                f"{_safe_s3_component(reorder_method)}/clusters_merged.bin"
            )
        else:
            key = f"pipeline/{dataset_key}/{ts}/clusters_merged.bin"
        _write_layout_output(
            dataset_key=dataset_key,
            dim=dim,
            nlist=int(args.nlist),
            nprobe=int(args.nprobe),
            query_idx=query_idx,
            raw_nprobe_results=raw_nprobe_results,
            cluster_centers=builder.cluster_centers,
            cluster_vectors=cluster_vectors,
            reorder_method=reorder_method,
            bucket=str(args.s3_bucket),
            region=str(args.s3_region),
            s3_key=key,
            existing_uri=existing_uri,
            out_path=out_path,
            warmup_queries=int(args.warmup_queries),
        )


if __name__ == "__main__":
    main()
