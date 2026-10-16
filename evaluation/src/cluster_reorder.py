"""
Cluster reordering algorithms used by the paper artifact.

This module intentionally preserves the method names referenced by the
submission-facing experiment scripts, including historical aliases such as
`random_proj`, `pca_zorder`, `spectral_cold`, and `spectral_log`.
The default paper layout should use `pca_nd_zorder`; `hilbert` is kept as
the fixed 2-D Hilbert ablation.
"""

from __future__ import annotations

import logging
import os

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

logger = logging.getLogger(__name__)


class ClusterReorder:
    """Reorder clusters to improve spatial locality."""

    def __init__(
        self,
        cluster_centers: np.ndarray,
        coaccess_weights: sparse.spmatrix | np.ndarray | None = None,
        random_state: int | None = None,
    ):
        self.cluster_centers = np.asarray(cluster_centers, dtype=np.float32)
        if self.cluster_centers.ndim != 2:
            raise ValueError("cluster_centers must be a 2D array")

        self.num_clusters = int(self.cluster_centers.shape[0])
        self.dimension = int(self.cluster_centers.shape[1])
        self.coaccess_weights = self._prepare_graph(coaccess_weights)
        self.random_state = random_state
        self.rng = np.random.default_rng(random_state)
        self.reorder_map = None  # Maps old_id -> new_id
        self.inverse_map = None  # Maps new_id -> old_id

    def reorder(self, method: str = "pca") -> np.ndarray:
        """Return an array mapping new cluster ids to original ids."""
        logger.info("Reordering clusters using method: %s", method)

        if method == "random":
            new_order = self._reorder_random()
        elif method == "random_proj":
            new_order = self._reorder_random_proj()
        elif method == "pca":
            new_order = self._reorder_pca()
        elif method == "pca_zorder":
            new_order = self._reorder_pca_zorder()
        elif method in {"pca_nd_zorder", "pca_kd_zorder", "adaptive_pca_zorder"}:
            new_order = self._reorder_pca_nd_zorder()
        elif method == "lpp":
            new_order = self._reorder_lpp()
        elif method == "tsne":
            new_order = self._reorder_tsne()
        elif method == "hilbert":
            new_order = self._reorder_hilbert()
        elif method == "spectral_cold":
            new_order = self._reorder_spectral(use_log_weights=False)
        elif method == "spectral_log":
            new_order = self._reorder_spectral(use_log_weights=True)
        elif method == "identity":
            new_order = self._reorder_identity()
        else:
            raise ValueError(
                "Unknown reorder method: "
                f"{method}. Supported: 'identity', 'random', 'random_proj', "
                "'pca', 'pca_zorder', 'pca_nd_zorder', 'lpp', 'tsne', 'hilbert', "
                "'spectral_cold', 'spectral_log'"
            )

        self.inverse_map = np.asarray(new_order, dtype=np.int64)  # new_id -> old_id
        self.reorder_map = np.argsort(self.inverse_map)  # old_id -> new_id
        logger.info("Reordering complete. Method: %s", method)
        return self.inverse_map

    def _prepare_graph(self, weights: sparse.spmatrix | np.ndarray | None) -> sparse.csr_matrix | None:
        if weights is None:
            return None
        graph = sparse.csr_matrix(weights, dtype=np.float64)
        expected = (self.num_clusters, self.num_clusters)
        if graph.shape != expected:
            raise ValueError(f"coaccess_weights must have shape {expected}, got {graph.shape}")
        graph = 0.5 * (graph + graph.T)
        graph.setdiag(0.0)
        graph.eliminate_zeros()
        return graph.tocsr()

    def _reorder_random(self) -> np.ndarray:
        """Random shuffle baseline."""
        return self.rng.permutation(self.num_clusters)

    def _reorder_random_proj(self) -> np.ndarray:
        """Sort by projection onto a random direction."""
        if self.num_clusters <= 1:
            return np.arange(self.num_clusters, dtype=np.int64)
        direction = self.rng.normal(size=self.dimension).astype(np.float64, copy=False)
        norm = float(np.linalg.norm(direction))
        if not np.isfinite(norm) or norm <= 0.0:
            return self._reorder_random()
        scores = self.cluster_centers.astype(np.float64, copy=False) @ (direction / norm)
        return np.argsort(scores, kind="mergesort")

    def _reorder_identity(self) -> np.ndarray:
        """Identity ordering."""
        return np.arange(self.num_clusters, dtype=np.int64)

    def _reorder_pca(self) -> np.ndarray:
        """Project cluster centers to 1D with PCA and sort."""
        logger.info("Applying PCA to reduce to 1D...")
        if self.num_clusters <= 1:
            return np.arange(self.num_clusters, dtype=np.int64)
        pca = PCA(n_components=1)
        coords_1d = pca.fit_transform(self.cluster_centers)
        logger.info("PCA explained variance: %.4f", float(pca.explained_variance_ratio_[0]))
        return np.argsort(coords_1d.reshape(-1), kind="mergesort")

    def _reorder_pca_zorder(self) -> np.ndarray:
        """Reduce to 2D with PCA and sort by Morton/Z-order code."""
        logger.info("Applying PCA + Z-order reordering...")
        coords_int = self._project_to_uint2d()
        z_indices = np.fromiter(
            (self._xy_to_morton(int(x), int(y), 16) for x, y in coords_int),
            dtype=np.uint64,
            count=self.num_clusters,
        )
        return np.argsort(z_indices, kind="mergesort")

    def _reorder_pca_nd_zorder(
        self,
        *,
        variance_threshold: float = 0.35,
        max_components: int = 8,
    ) -> np.ndarray:
        """Adaptive PCA n-D projection followed by Morton/Z-order sort.

        This mirrors the Lance fork's deployed centroid reorder: choose the
        smallest top-k PCA projection covering at least 35% cumulative variance,
        cap k at 8 so the Morton key fits in 64 bits, and sort by the n-D
        Morton code. The old `pca_zorder` method remains the fixed 2-D ablation.
        """
        logger.info(
            "Applying adaptive PCA n-D + Z-order reordering "
            "(variance_threshold=%.2f, max_components=%d)...",
            variance_threshold,
            max_components,
        )
        if self.num_clusters <= 1:
            return np.arange(self.num_clusters, dtype=np.int64)

        k_max = min(int(max_components), self.dimension, self.num_clusters)
        if k_max <= 0:
            return np.arange(self.num_clusters, dtype=np.int64)

        pca = PCA(n_components=k_max)
        coords_all = pca.fit_transform(self.cluster_centers)
        ratios = np.asarray(pca.explained_variance_ratio_, dtype=np.float64)
        cum = np.cumsum(ratios)
        idx = int(np.searchsorted(cum, float(variance_threshold), side="left"))
        k = min(max(idx + 1, min(2, k_max)), k_max)
        logger.info(
            "Adaptive PCA selected k=%d/%d, cumulative variance=%.4f",
            k,
            k_max,
            float(cum[k - 1]) if cum.size else 0.0,
        )

        bits_per_axis = min(32, max(1, 64 // k))
        coords_int = self._project_to_uint_nd(coords_all[:, :k], bits_per_axis)
        z_indices = np.fromiter(
            (self._morton_nd(row, bits_per_axis) for row in coords_int),
            dtype=np.uint64,
            count=self.num_clusters,
        )
        return np.argsort(z_indices, kind="mergesort")

    def _reorder_lpp(self, k_neighbors: int = 10) -> np.ndarray:
        """
        LPP (Locality Preserving Projection) based reordering.

        Steps:
        1) Build a kNN graph on cluster centers.
        2) Compute LPP projection via generalized eigenvalue problem:
           X^T L X a = lambda X^T D X a
        3) Sort by the 1D projection.
        """
        logger.info("Applying LPP to reduce to 1D (k=%s)...", k_neighbors)

        X = self.cluster_centers.astype(np.float64, copy=False)
        n, d = X.shape
        if n <= 2:
            return np.arange(n, dtype=np.int64)

        k = min(max(2, k_neighbors), n - 1)
        n_jobs_env = os.getenv("VECIO_LPP_NJOBS")
        n_jobs = None
        if n_jobs_env is not None:
            try:
                n_jobs = int(n_jobs_env)
            except Exception:
                n_jobs = None

        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto", n_jobs=n_jobs)
        nn.fit(X)
        dists, idx = nn.kneighbors(X, return_distance=True)

        neigh_idx = idx[:, 1:]
        neigh_dists = dists[:, 1:]
        sigma = float(np.median(neigh_dists))
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = float(np.mean(neigh_dists)) if float(np.mean(neigh_dists)) > 0.0 else 1.0
        sigma2 = sigma * sigma
        weights = np.exp(-(neigh_dists**2) / (sigma2 + 1e-12))

        row_idx = np.repeat(np.arange(n), k)
        col_idx = neigh_idx.reshape(-1)
        data = weights.reshape(-1)
        W = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
        W = 0.5 * (W + W.T)

        dvals = np.asarray(W.sum(axis=1)).reshape(-1)
        D = sparse.diags(dvals)
        L = D - W

        LX = L.dot(X)
        DX = D.dot(X)
        A = X.T @ LX
        B = X.T @ DX

        reg = 1e-6 * np.trace(B) / max(1, d)
        A = A + reg * np.eye(d)
        B = B + reg * np.eye(d)

        try:
            evals, evecs = eigh(A, B)
        except Exception as exc:
            logger.warning("LPP eigen solve failed: %s; fallback to PCA", exc)
            return self._reorder_pca()

        order = np.argsort(evals)
        vec = None
        for idx_e in order:
            if evals[idx_e] > 1e-9:
                vec = evecs[:, idx_e]
                break
        if vec is None:
            logger.warning("LPP eigenvalues all near-zero; fallback to PCA")
            return self._reorder_pca()

        coords_1d = X @ vec
        if not np.all(np.isfinite(coords_1d)):
            logger.warning("LPP produced non-finite coordinates; fallback to PCA")
            return self._reorder_pca()
        return np.argsort(coords_1d.reshape(-1), kind="mergesort")

    def _reorder_tsne(self) -> np.ndarray:
        """Use t-SNE to reduce cluster centers to 1D and sort."""
        logger.info("Applying t-SNE to reduce to 1D...")
        if self.num_clusters <= 1:
            return np.arange(self.num_clusters, dtype=np.int64)
        tsne = TSNE(n_components=1, random_state=self.random_state or 42, perplexity=30)
        coords_1d = tsne.fit_transform(self.cluster_centers)
        return np.argsort(coords_1d.reshape(-1), kind="mergesort")

    def _reorder_hilbert(self) -> np.ndarray:
        """Reduce to 2D with PCA and sort by Hilbert curve index."""
        logger.info("Applying Hilbert curve reordering...")
        coords_int = self._project_to_uint2d()
        hilbert_indices = np.fromiter(
            (self._xy_to_hilbert(int(x), int(y), 16) for x, y in coords_int),
            dtype=np.uint64,
            count=self.num_clusters,
        )
        logger.info("Hilbert curve reordering complete (resolution: 16)")
        return np.argsort(hilbert_indices, kind="mergesort")

    def _reorder_spectral(self, *, use_log_weights: bool) -> np.ndarray:
        """Sort nodes by the first non-trivial Laplacian eigenvector."""
        variant = "spectral_log" if use_log_weights else "spectral_cold"
        logger.info("Applying %s reordering...", variant)

        graph = self._spectral_graph(use_log_weights=use_log_weights)
        if graph is None or graph.nnz == 0 or self.num_clusters <= 2:
            logger.warning("%s has no usable graph; fallback to PCA", variant)
            return self._reorder_pca()

        degrees = np.asarray(graph.sum(axis=1)).reshape(-1)
        laplacian = sparse.diags(degrees) - graph

        try:
            k = min(8, self.num_clusters - 1)
            evals, evecs = eigsh(laplacian, k=k, which="SM", tol=1e-3)
            order = np.argsort(evals)
            vec = None
            for idx_e in order:
                if evals[idx_e] > 1e-9:
                    vec = evecs[:, idx_e]
                    break
            if vec is None:
                raise ValueError("no non-trivial spectral component")
        except Exception as exc:
            logger.warning("%s eigen solve failed: %s; fallback to PCA", variant, exc)
            return self._reorder_pca()

        if not np.all(np.isfinite(vec)):
            logger.warning("%s produced non-finite coordinates; fallback to PCA", variant)
            return self._reorder_pca()
        return np.argsort(vec.reshape(-1), kind="mergesort")

    def _spectral_graph(self, *, use_log_weights: bool) -> sparse.csr_matrix | None:
        graph = self.coaccess_weights
        if graph is None or graph.nnz == 0:
            graph = self._build_knn_graph()
        if graph is None or graph.nnz == 0:
            return None
        graph = graph.tocsr().astype(np.float64, copy=True)
        if use_log_weights:
            graph.data = np.log1p(graph.data)
        graph = 0.5 * (graph + graph.T)
        graph.setdiag(0.0)
        graph.eliminate_zeros()
        return graph.tocsr()

    def _build_knn_graph(self, k_neighbors: int = 16) -> sparse.csr_matrix | None:
        n = self.num_clusters
        if n <= 1:
            return None

        k = min(max(2, k_neighbors), n - 1)
        X = self.cluster_centers.astype(np.float64, copy=False)
        nn = NearestNeighbors(n_neighbors=k + 1, algorithm="auto")
        nn.fit(X)
        dists, idx = nn.kneighbors(X, return_distance=True)
        neigh_idx = idx[:, 1:]
        neigh_dists = dists[:, 1:]

        positive = neigh_dists[neigh_dists > 0]
        sigma = float(np.median(positive)) if positive.size else 1.0
        if not np.isfinite(sigma) or sigma <= 0.0:
            sigma = 1.0
        weights = np.exp(-(neigh_dists**2) / (2.0 * sigma * sigma + 1e-12))

        row_idx = np.repeat(np.arange(n), k)
        col_idx = neigh_idx.reshape(-1)
        data = weights.reshape(-1)
        graph = sparse.csr_matrix((data, (row_idx, col_idx)), shape=(n, n))
        graph = 0.5 * (graph + graph.T)
        graph.setdiag(0.0)
        graph.eliminate_zeros()
        return graph

    def _project_to_uint2d(self, resolution: int = 16) -> np.ndarray:
        coords_2d = self._project_pca_2d()
        if coords_2d.size == 0:
            return np.zeros((self.num_clusters, 2), dtype=np.int32)

        mins = coords_2d.min(axis=0)
        spans = coords_2d.max(axis=0) - mins
        spans[spans <= 1e-12] = 1.0
        coords_norm = (coords_2d - mins) / spans
        max_coord = (1 << resolution) - 1
        coords_int = np.clip(np.rint(coords_norm * max_coord), 0, max_coord).astype(np.int32)
        return coords_int

    def _project_to_uint_nd(self, coords: np.ndarray, bits_per_axis: int) -> np.ndarray:
        coords = np.asarray(coords, dtype=np.float64)
        if coords.size == 0:
            return np.zeros((self.num_clusters, 0), dtype=np.uint64)

        mins = coords.min(axis=0)
        spans = coords.max(axis=0) - mins
        spans[spans <= 1e-12] = 1.0
        coords_norm = (coords - mins) / spans
        max_coord = (1 << int(bits_per_axis)) - 1
        return np.clip(np.rint(coords_norm * max_coord), 0, max_coord).astype(np.uint64)

    def _project_pca_2d(self) -> np.ndarray:
        if self.num_clusters == 0:
            return np.zeros((0, 2), dtype=np.float64)
        if self.num_clusters == 1:
            return np.zeros((1, 2), dtype=np.float64)

        n_components = 2 if self.dimension >= 2 else 1
        pca = PCA(n_components=n_components)
        coords = pca.fit_transform(self.cluster_centers)
        if coords.shape[1] == 1:
            coords = np.column_stack([coords[:, 0], np.zeros(coords.shape[0], dtype=coords.dtype)])
        return np.asarray(coords[:, :2], dtype=np.float64)

    def _xy_to_morton(self, x: int, y: int, order: int) -> int:
        code = 0
        for bit in range(order):
            code |= ((x >> bit) & 1) << (2 * bit)
            code |= ((y >> bit) & 1) << (2 * bit + 1)
        return code

    def _morton_nd(self, point: np.ndarray, bits_per_axis: int) -> int:
        code = 0
        axes = [int(v) for v in np.asarray(point).reshape(-1)]
        for bit in range(int(bits_per_axis)):
            for axis, value in enumerate(axes):
                code |= ((value >> bit) & 1) << (bit * len(axes) + axis)
        return code

    def _xy_to_hilbert(self, x: int, y: int, order: int) -> int:
        """Convert 2D coordinates to Hilbert curve index."""
        d = 0
        s = 1 << (order - 1)

        while s > 0:
            rx = 1 if (x & s) > 0 else 0
            ry = 1 if (y & s) > 0 else 0
            d += s * s * ((3 * rx) ^ ry)

            if ry == 0:
                if rx == 1:
                    x = s - 1 - x
                    y = s - 1 - y
                x, y = y, x

            s >>= 1

        return d

    def apply_reorder(self, cluster_ids: np.ndarray) -> np.ndarray:
        """Apply the computed old_id -> new_id map."""
        if self.reorder_map is None:
            raise ValueError("Must call reorder() first")
        return self.reorder_map[cluster_ids]

    def reverse_reorder(self, reordered_ids: np.ndarray) -> np.ndarray:
        """Map reordered ids back to original ids."""
        if self.inverse_map is None:
            raise ValueError("Must call reorder() first")
        return self.inverse_map[reordered_ids]

    def compute_locality_metrics(self, cluster_ids_accessed: np.ndarray) -> dict:
        """Compute simple locality diagnostics for a reordered access trace."""
        if len(cluster_ids_accessed) == 0:
            return {
                "span": 0,
                "density": 0,
                "avg_gap": 0,
                "max_gap": 0,
                "num_segments": 0,
            }

        sorted_ids = np.sort(cluster_ids_accessed)
        span = sorted_ids[-1] - sorted_ids[0] if len(sorted_ids) > 1 else 0
        density = len(sorted_ids) / (span + 1) if span > 0 else 1.0

        if len(sorted_ids) > 1:
            gaps = np.diff(sorted_ids)
            avg_gap = np.mean(gaps)
            max_gap = np.max(gaps)
            num_segments = np.sum(gaps > 10) + 1
        else:
            avg_gap = 0
            max_gap = 0
            num_segments = 1

        return {
            "span": int(span),
            "density": float(density),
            "avg_gap": float(avg_gap),
            "max_gap": int(max_gap),
            "num_segments": int(num_segments),
        }


if __name__ == "__main__":
    print("Testing Cluster Reordering...")

    num_clusters = 256
    dimension = 32
    cluster_centers = np.random.randn(num_clusters, dimension).astype(np.float32)
    graph = sparse.random(num_clusters, num_clusters, density=0.02, format="csr", random_state=7)
    graph = 0.5 * (graph + graph.T)

    for method in [
        "identity",
        "random",
        "random_proj",
        "pca",
        "pca_zorder",
        "pca_nd_zorder",
        "hilbert",
        "spectral_cold",
        "spectral_log",
    ]:
        print(f"\nTesting method: {method}")
        reorder = ClusterReorder(cluster_centers, coaccess_weights=graph, random_state=42)
        new_order = reorder.reorder(method)
        assert len(new_order) == num_clusters

        accessed_clusters = np.random.choice(num_clusters, size=50, replace=False)
        reordered_accessed = reorder.apply_reorder(accessed_clusters)
        metrics = reorder.compute_locality_metrics(reordered_accessed)
        print(f"Locality metrics: {metrics}")

    print("\nTest passed!")
