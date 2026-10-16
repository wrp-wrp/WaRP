#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets import get_dataset
from src.index_builder import IVFIndexBuilder


def _subset_queries(queries: np.ndarray, num_queries: int, mode: str, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if num_queries <= 0 or num_queries >= len(queries):
        idx = np.arange(len(queries), dtype=np.int64)
        return queries, idx
    if mode == "random":
        rng = np.random.default_rng(int(seed))
        idx = np.sort(rng.choice(len(queries), size=int(num_queries), replace=False))
        return queries[idx], idx
    idx = np.arange(int(num_queries), dtype=np.int64)
    return queries[:num_queries], idx


def _exact_topk_l2_chunked(base: np.ndarray, queries: np.ndarray, *, k: int, chunk_rows: int = 100_000) -> np.ndarray:
    num_queries = int(queries.shape[0])
    best_dists = np.full((num_queries, int(k)), np.inf, dtype=np.float32)
    best_ids = np.full((num_queries, int(k)), -1, dtype=np.int64)
    q = np.ascontiguousarray(queries.astype("float32"))
    q_norm = np.sum(q * q, axis=1, dtype=np.float32)[:, None]
    total = int(base.shape[0])

    for start in range(0, total, int(chunk_rows)):
        end = min(total, start + int(chunk_rows))
        xb = np.ascontiguousarray(np.asarray(base[start:end], dtype=np.float32))
        xb_norm = np.sum(xb * xb, axis=1, dtype=np.float32)[None, :]
        dists = q_norm + xb_norm - 2.0 * (q @ xb.T)
        local_k = min(int(k), int(xb.shape[0]))
        local_idx = np.argpartition(dists, kth=local_k - 1, axis=1)[:, :local_k]
        local_dists = np.take_along_axis(dists, local_idx, axis=1)
        local_ids = local_idx.astype(np.int64) + int(start)

        merged_dists = np.concatenate([best_dists, local_dists.astype(np.float32)], axis=1)
        merged_ids = np.concatenate([best_ids, local_ids], axis=1)
        pick = np.argpartition(merged_dists, kth=int(k) - 1, axis=1)[:, : int(k)]
        best_dists = np.take_along_axis(merged_dists, pick, axis=1)
        best_ids = np.take_along_axis(merged_ids, pick, axis=1)

    order = np.argsort(best_dists, axis=1)
    return np.take_along_axis(best_ids, order, axis=1)


def _load_or_compute_gt_subset(ds, *, dataset: str, base: np.ndarray, query_idx: np.ndarray, selected_queries: np.ndarray, k: int) -> np.ndarray:
    gt_I, _gt_D = ds.get_groundtruth(k=int(k))
    if gt_I is not None:
        return np.asarray(gt_I, dtype=np.int64)[query_idx][:, : int(k)]
    return _exact_topk_l2_chunked(base, selected_queries, k=int(k))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute IVF recall@10 for a selected query subset.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--nlist", type=int, required=True)
    ap.add_argument("--nprobe", type=int, required=True)
    ap.add_argument("--num-queries", type=int, default=240)
    ap.add_argument("--query-subset-mode", default="random", choices=["random", "first"])
    ap.add_argument("--query-subset-seed", type=int, default=12345)
    ap.add_argument("--train-size", type=int, default=0)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--out", required=True)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    ds = get_dataset(str(args.dataset))
    ds.prepare()
    base = np.ascontiguousarray(ds.get_base_vectors().astype("float32"))
    queries_all = np.ascontiguousarray(ds.get_queries().astype("float32"))
    selected_queries, query_idx = _subset_queries(
        queries_all,
        int(args.num_queries),
        str(args.query_subset_mode),
        int(args.query_subset_seed),
    )
    gt_I = _load_or_compute_gt_subset(
        ds,
        dataset=str(args.dataset),
        base=base,
        query_idx=query_idx,
        selected_queries=selected_queries,
        k=int(args.k),
    )
    builder = IVFIndexBuilder(dimension=int(base.shape[1]), num_clusters=int(args.nlist))
    train_size = int(args.train_size) if int(args.train_size) > 0 else None
    builder.build(base, train_size=train_size)
    _dist, pred = builder.search(selected_queries, k=int(args.k), nprobe=int(args.nprobe))
    hits = 0
    total = 0
    for gt_row, pred_row in zip(gt_I, pred):
        gt_set = {int(x) for x in np.asarray(gt_row).tolist() if int(x) >= 0}
        pred_set = {int(x) for x in np.asarray(pred_row).tolist() if int(x) >= 0}
        hits += len(gt_set & pred_set)
        total += len(gt_set)
    recall = float(hits) / float(total) if total > 0 else 0.0
    payload = {
        "dataset": str(args.dataset),
        "nlist": int(args.nlist),
        "nprobe": int(args.nprobe),
        "train_size": int(args.train_size),
        "num_queries": int(len(selected_queries)),
        "query_indices": [int(x) for x in np.asarray(query_idx).tolist()],
        "recall_at_10": recall,
    }
    out = Path(str(args.out)).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
