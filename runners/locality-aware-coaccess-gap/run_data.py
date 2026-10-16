#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cluster_reorder import ClusterReorder
from src.datasets import get_dataset
from src.index_builder import IVFIndexBuilder

HERE = Path(__file__).resolve().parent
DATA_ROOT = ROOT / "experiments/figure-13-reordering-methods/stages" / HERE.name / "data/latest"


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Compute normalized weighted co-access gap from full dataset query splits.")
    ap.add_argument("--datasets", default="sift1m,glove,deep1m,openimage,msong,random1m128")
    ap.add_argument(
        "--methods",
        default="identity,random_proj,pca,pca_zorder,hilbert,spectral_cold,spectral_log",
    )
    ap.add_argument("--nlist", type=int, default=4096)
    ap.add_argument("--nprobe", type=int, default=128)
    ap.add_argument("--train-size", type=int, default=100000)
    ap.add_argument("--max-queries", type=int, default=0, help="Optional cap for quick runs; 0 means use full query split.")
    ap.add_argument("--out-dir", default=str(DATA_ROOT))
    ap.add_argument("--force", action="store_true")
    ap.add_argument(
        "--check-only",
        action="store_true",
        help="Print the planned output paths without running the pipeline.",
    )
    return ap.parse_args()


def _parse_csv_list(raw: str) -> list[str]:
    out = [tok.strip().lower() for tok in str(raw).split(",") if tok.strip()]
    if not out:
        raise SystemExit("empty dataset list")
    return out


def _parse_methods(raw: str) -> list[str]:
    methods = _parse_csv_list(raw)
    supported = {
        "identity",
        "random_proj",
        "pca",
        "pca_zorder",
        "hilbert",
        "spectral_cold",
        "spectral_log",
    }
    bad = [method for method in methods if method not in supported]
    if bad:
        raise SystemExit(f"unsupported reorder methods: {bad}")
    return methods


def _dataset_label(name: str) -> str:
    return {
        "sift1m": "SIFT1M",
        "glove": "GloVe",
        "deep1m": "DEEP1M",
        "openimage": "OpenImage",
        "random1m128": "Random-1M",
        "msong": "MSong",
    }.get(str(name).lower(), str(name))


def _method_label(name: str) -> str:
    return {
        "identity": "Identity",
        "random_proj": "Random proj.",
        "pca": "PCA-1D",
        "pca_zorder": "PCA-2D + Z-order",
        "hilbert": "PCA-2D + Hilbert",
        "spectral_cold": "Spectral (cold)",
        "spectral_log": "Spectral (log)",
    }[name]


def _ensure_dir(path: Path, *, force: bool = False) -> None:
    if force and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _coaccess_weights(cluster_ids: np.ndarray, num_clusters: int) -> sparse.csr_matrix:
    if cluster_ids.ndim != 2:
        raise ValueError("expected 2D cluster_ids")
    num_queries, width = cluster_ids.shape
    if width <= 1 or num_queries <= 0:
        return sparse.csr_matrix((num_clusters, num_clusters), dtype=np.float64)

    tri_i, tri_j = np.triu_indices(width, k=1)
    weights = np.zeros((num_clusters, num_clusters), dtype=np.float32)
    ids = cluster_ids.astype(np.int64, copy=False)
    for row in ids:
        src = row[tri_i]
        dst = row[tri_j]
        np.add.at(weights, (src, dst), 1.0)
        np.add.at(weights, (dst, src), 1.0)
    weights /= float(num_queries)
    return sparse.csr_matrix(weights)


def _weighted_gap_value(weights: sparse.csr_matrix, positions: np.ndarray) -> float:
    upper = sparse.triu(weights, k=1).tocoo()
    if upper.nnz == 0:
        return 0.0
    pos = positions.astype(np.int64, copy=False)
    gaps = np.abs(pos[upper.row] - pos[upper.col]).astype(np.float64, copy=False)
    return float(np.dot(upper.data.astype(np.float64, copy=False), gaps))


def _run_dataset(
    dataset: str,
    *,
    methods: list[str],
    nlist: int,
    nprobe: int,
    train_size: int,
    max_queries: int,
) -> tuple[list[dict], dict]:
    ds = get_dataset(dataset)
    ds.prepare()
    base = np.asarray(ds.get_base_vectors(), dtype=np.float32)
    queries = np.asarray(ds.get_queries(), dtype=np.float32)
    if int(max_queries) > 0 and len(queries) > int(max_queries):
        queries = queries[: int(max_queries)]

    builder = IVFIndexBuilder(dimension=int(base.shape[1]), num_clusters=int(nlist))
    builder.build(base, train_size=int(train_size) if int(train_size) > 0 else None)
    _dist, cluster_ids = builder.index.quantizer.search(np.ascontiguousarray(queries), int(nprobe))
    coaccess_weights = _coaccess_weights(cluster_ids, int(nlist))
    identity_positions = np.arange(int(nlist), dtype=np.int64)
    baseline_gap = _weighted_gap_value(coaccess_weights, identity_positions)
    if baseline_gap <= 0.0:
        baseline_gap = 1.0

    rows: list[dict] = []
    for method in methods:
        reorder = ClusterReorder(builder.cluster_centers, coaccess_weights=coaccess_weights)
        t0 = time.perf_counter()
        reorder.reorder(method)
        reorder_time_ms = float((time.perf_counter() - t0) * 1000.0)
        positions = np.asarray(reorder.reorder_map, dtype=np.int64)
        weighted_gap = _weighted_gap_value(coaccess_weights, positions)
        rows.append(
            {
                "dataset": dataset,
                "dataset_label": _dataset_label(dataset),
                "method": method,
                "method_label": _method_label(method),
                "dimension": int(base.shape[1]),
                "num_queries": int(len(queries)),
                "nlist": int(nlist),
                "nprobe": int(nprobe),
                "reorder_time_ms": reorder_time_ms,
                "weighted_gap_value": weighted_gap,
                "weighted_gap_ratio": float(weighted_gap / baseline_gap),
            }
        )

    meta = {
        "dataset": dataset,
        "dataset_label": _dataset_label(dataset),
        "num_base": int(base.shape[0]),
        "num_queries": int(queries.shape[0]),
        "dimension": int(base.shape[1]),
        "nlist": int(nlist),
        "nprobe": int(nprobe),
        "train_size": int(train_size),
        "query_source": "dataset-provided query split",
        "query_selection": "full query split" if int(max_queries) <= 0 else f"first {int(max_queries)} queries",
        "baseline_gap_value": baseline_gap,
    }
    return rows, meta


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir).resolve()
    data_dir = out_dir

    datasets = _parse_csv_list(args.datasets)
    methods = _parse_methods(args.methods)

    if bool(args.check_only):
        print(f"[plan] datasets={datasets}")
        print(f"[plan] methods={methods}")
        print(f"[plan] out_dir={data_dir}")
        print(f"[plan] csv -> {data_dir / 'figure_locality_aware_coaccess_gap.csv'}")
        print(f"[plan] aggregated -> {data_dir / 'aggregated_results.json'}")
        return 0

    _ensure_dir(data_dir, force=False)
    all_rows: list[dict] = []
    meta_rows: list[dict] = []
    for dataset in datasets:
        rows, meta = _run_dataset(
            dataset,
            methods=methods,
            nlist=int(args.nlist),
            nprobe=int(args.nprobe),
            train_size=int(args.train_size),
            max_queries=int(args.max_queries),
        )
        all_rows.extend(rows)
        meta_rows.append(meta)

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["dataset_label", "method"]).reset_index(drop=True)
    (data_dir / "figure_locality_aware_coaccess_gap.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    (data_dir / "aggregated_results.json").write_text(
        json.dumps(
            {
                "figure": "figure_locality_aware_coaccess_gap",
                "query_source": "dataset-provided query split",
                "metric": "weighted_gap_ratio",
                "methods": methods,
                "datasets": meta_rows,
                "rows": df.to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[ok] data={data_dir / 'aggregated_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
