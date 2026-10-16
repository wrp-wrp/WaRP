from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.neighbors import NearestNeighbors


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.cluster_reorder import ClusterReorder  # noqa: E402
from src.datasets import get_dataset  # noqa: E402


HERE = Path(__file__).resolve().parent
OUT_DIR = ROOT / "experiments/figure-13-reordering-methods/stages" / HERE.name / "data/latest"
SUMMARY_CSV = OUT_DIR / "figure_reorder_cost_cluster_sweep.csv"
RAW_CSV = OUT_DIR / "figure_reorder_cost_cluster_sweep_raw.csv"
META_JSON = OUT_DIR / "figure_reorder_cost_cluster_sweep_data.json"
METHODS = [
    "identity",
    "random_proj",
    "pca",
    "pca_zorder",
    "hilbert",
    "spectral_cold",
    "spectral_log",
]
METHOD_LABELS = {
    "identity": "Identity",
    "random_proj": "Random proj.",
    "pca": "PCA-1D",
    "pca_zorder": "PCA-2D + Z-order",
    "hilbert": "PCA-2D + Hilbert",
    "spectral_cold": "Spectral (cold)",
    "spectral_log": "Spectral (log)",
}


def _parse_counts(raw: str) -> list[int]:
    counts = [int(tok.strip()) for tok in str(raw).split(",") if tok.strip()]
    if not counts:
        raise SystemExit("empty --cluster-counts")
    bad = [value for value in counts if value <= 1]
    if bad:
        raise SystemExit(f"cluster counts must be > 1, got {bad}")
    return counts


def _sample_centers(base: np.ndarray, num_clusters: int, rng: np.random.Generator) -> np.ndarray:
    indices = rng.choice(int(base.shape[0]), size=int(num_clusters), replace=False)
    return np.asarray(base[indices], dtype=np.float32)


def _build_surrogate_log_graph(centers: np.ndarray, degree: int) -> sparse.csr_matrix:
    n = int(centers.shape[0])
    if n <= 1:
        return sparse.csr_matrix((n, n), dtype=np.float64)

    k = min(max(2, int(degree)), n - 1)
    X = centers.astype(np.float64, copy=False)
    nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
    nn.fit(X)
    dists, idx = nn.kneighbors(X, return_distance=True)
    neigh_idx = idx[:, 1:]
    neigh_dists = dists[:, 1:]

    positive = neigh_dists[neigh_dists > 0]
    sigma = float(np.median(positive)) if positive.size else 1.0
    if not np.isfinite(sigma) or sigma <= 0.0:
        sigma = 1.0
    denom = 2.0 * sigma * sigma + 1e-12
    weights = np.exp(-(neigh_dists ** 2) / denom)

    row_idx = np.repeat(np.arange(n), k)
    col_idx = neigh_idx.reshape(-1)
    data = weights.reshape(-1)
    graph = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
    graph = 0.5 * (graph + graph.T)
    graph.setdiag(0.0)
    graph.eliminate_zeros()
    return graph


def main() -> int:
    ap = argparse.ArgumentParser(description="Generate Figure 9 left-panel reorder timing vs cluster count.")
    ap.add_argument("--dataset", default="sift1m")
    ap.add_argument("--cluster-counts", default="512,1024,2048,4096,8192")
    ap.add_argument("--graph-degree", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--check-only", action="store_true",
                    help="Print planned outputs and exit without running the benchmark.")
    args = ap.parse_args()

    counts = _parse_counts(args.cluster_counts)
    if args.check_only:
        print(f"[plan] dataset={args.dataset}")
        print(f"[plan] cluster_counts={counts}")
        print(f"[plan] graph_degree={args.graph_degree} repeats={args.repeats}")
        print(f"[plan] summary_csv -> {SUMMARY_CSV}")
        print(f"[plan] raw_csv     -> {RAW_CSV}")
        return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    ds = get_dataset(str(args.dataset))
    ds.prepare()
    base = ds.get_base_vectors()
    dimension = int(base.shape[1])

    rng = np.random.default_rng(int(args.seed))
    raw_rows: list[dict] = []
    for num_clusters in counts:
        for repeat in range(int(args.repeats)):
            centers = _sample_centers(base, int(num_clusters), rng)
            log_graph = _build_surrogate_log_graph(centers, degree=int(args.graph_degree))
            for method in METHODS:
                reorder = ClusterReorder(centers, coaccess_weights=log_graph, random_state=int(args.seed) + repeat)
                t0 = time.perf_counter()
                reorder.reorder(method)
                reorder_time_ms = float((time.perf_counter() - t0) * 1000.0)
                raw_rows.append(
                    {
                        "dataset": str(args.dataset),
                        "sample_source": "sift1m_base_random_without_replacement",
                        "dimension": dimension,
                        "num_clusters": int(num_clusters),
                        "graph_degree": int(args.graph_degree),
                        "repeat": int(repeat),
                        "method": method,
                        "method_label": METHOD_LABELS[method],
                        "reorder_time_ms": reorder_time_ms,
                    }
                )

    raw_df = pd.DataFrame(raw_rows).sort_values(["num_clusters", "repeat", "method"]).reset_index(drop=True)
    raw_df.to_csv(RAW_CSV, index=False)

    summary_df = (
        raw_df.groupby(["dataset", "sample_source", "dimension", "num_clusters", "graph_degree", "method", "method_label"], as_index=False)["reorder_time_ms"]
        .median()
        .sort_values(["num_clusters", "method"])
        .reset_index(drop=True)
    )
    summary_df.to_csv(SUMMARY_CSV, index=False)

    META_JSON.write_text(
        json.dumps(
            {
                "figure": "figure_reorder_cost_cluster_sweep",
                "dataset": str(args.dataset),
                "sample_source": "random SIFT1M base vectors used as simulated centroids",
                "dimension": dimension,
                "cluster_counts": counts,
                "graph_degree": int(args.graph_degree),
                "repeats": int(args.repeats),
                "seed": int(args.seed),
                "spectral_log_surrogate": "symmetric kNN graph over sampled centroids; graph construction excluded from reorder timing",
                "summary_csv": str(SUMMARY_CSV),
                "raw_csv": str(RAW_CSV),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"[ok] wrote {SUMMARY_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
