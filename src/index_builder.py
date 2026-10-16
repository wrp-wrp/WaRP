"""
IVF Index Builder

Builds IVF (Inverted File) index for vector search.
"""

import json
import logging
import os
import pickle
from pathlib import Path
from typing import Tuple, Dict, Any

import faiss
import numpy as np

logger = logging.getLogger(__name__)


def _resolve_faiss_threads() -> int:
    raw = os.environ.get("VECIO_FAISS_THREADS") or os.environ.get("OMP_NUM_THREADS")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except ValueError:
            logger.warning("Ignoring invalid FAISS thread count: %r", raw)
    cpu_count = os.cpu_count() or 1
    return max(1, int(cpu_count))


def configure_faiss_threads() -> int:
    threads = _resolve_faiss_threads()
    faiss.omp_set_num_threads(int(threads))
    return int(threads)


class IVFIndexBuilder:
    """Build and manage IVF index for vector search."""

    def __init__(self, dimension: int, num_clusters: int):
        """
        Initialize IVF index builder.

        Args:
            dimension: Vector dimension
            num_clusters: Number of clusters (IVF lists)
        """
        self.dimension = dimension
        self.num_clusters = num_clusters
        self.index = None
        self.cluster_centers = None
        self.inverted_lists = None

    def build(self, vectors: np.ndarray, train_size: int = None) -> None:
        """
        Build IVF index from vectors.

        Args:
            vectors: Training vectors, shape (n, dimension)
            train_size: Number of vectors to use for training (None = use all)
        """
        threads = configure_faiss_threads()
        logger.info(
            "Building IVF index with %d clusters using %d FAISS threads...",
            self.num_clusters,
            threads,
        )

        # Create quantizer (for finding nearest clusters)
        quantizer = faiss.IndexFlatL2(self.dimension)

        # Create IVF index
        self.index = faiss.IndexIVFFlat(
            quantizer,
            self.dimension,
            self.num_clusters,
            faiss.METRIC_L2
        )

        # Train the index
        train_vectors = vectors[:train_size] if train_size else vectors
        logger.info(f"Training on {len(train_vectors)} vectors...")
        self.index.train(train_vectors)

        # Add all vectors
        logger.info(f"Adding {len(vectors)} vectors...")
        self.index.add(vectors)

        # Extract cluster centers
        # Reconstruct all cluster centers
        self.cluster_centers = np.zeros((self.num_clusters, self.dimension), dtype=np.float32)
        for i in range(self.num_clusters):
            self.cluster_centers[i] = self.index.quantizer.reconstruct(i)

        logger.info("Index built successfully!")

    def extract_inverted_lists(self) -> Dict[int, np.ndarray]:
        """
        Extract inverted lists from the index.

        Returns:
            Dictionary mapping cluster_id -> vector_ids in that cluster
        """
        logger.info("Extracting inverted lists...")

        inverted_lists = {}

        for cluster_id in range(self.num_clusters):
            # Get list size
            list_size = self.index.invlists.list_size(cluster_id)

            if list_size > 0:
                # Get vector IDs in this cluster
                vector_ids = faiss.rev_swig_ptr(
                    self.index.invlists.get_ids(cluster_id),
                    list_size
                )
                inverted_lists[cluster_id] = np.array(vector_ids)
            else:
                inverted_lists[cluster_id] = np.array([], dtype=np.int64)

        self.inverted_lists = inverted_lists
        logger.info(f"Extracted {len(inverted_lists)} inverted lists")

        return inverted_lists

    def get_cluster_vectors(self, vectors: np.ndarray) -> Dict[int, np.ndarray]:
        """
        Get vectors organized by cluster.

        Args:
            vectors: All vectors, shape (n, dimension)

        Returns:
            Dictionary mapping cluster_id -> vectors in that cluster
        """
        if self.inverted_lists is None:
            self.extract_inverted_lists()

        cluster_vectors = {}

        for cluster_id, vector_ids in self.inverted_lists.items():
            if len(vector_ids) > 0:
                cluster_vectors[cluster_id] = vectors[vector_ids]
            else:
                cluster_vectors[cluster_id] = np.array([]).reshape(0, self.dimension)

        return cluster_vectors

    def save_metadata(self, output_path: Path) -> None:
        """
        Save index metadata.

        Args:
            output_path: Path to save metadata
        """
        metadata = {
            'dimension': self.dimension,
            'num_clusters': self.num_clusters,
            'num_vectors': self.index.ntotal,
            'cluster_sizes': {
                cid: len(vids)
                for cid, vids in self.inverted_lists.items()
            }
        }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Metadata saved to {output_path}")

    def save_cluster_centers(self, output_path: Path) -> None:
        """
        Save cluster centers as binary file.

        Args:
            output_path: Path to save cluster centers
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.cluster_centers.astype(np.float32).tofile(output_path)
        logger.info(f"Cluster centers saved to {output_path}")

    def search(self, queries: np.ndarray, k: int = 100, nprobe: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for nearest neighbors.

        Args:
            queries: Query vectors, shape (n_queries, dimension)
            k: Number of nearest neighbors to return
            nprobe: Number of clusters to probe

        Returns:
            distances: shape (n_queries, k)
            indices: shape (n_queries, k)
        """
        configure_faiss_threads()
        self.index.nprobe = nprobe
        distances, indices = self.index.search(queries, k)
        return distances, indices

    def find_nearest_clusters(self, query: np.ndarray, nprobe: int) -> np.ndarray:
        """
        Find nearest clusters for a query vector.

        Args:
            query: Query vector, shape (dimension,)
            nprobe: Number of nearest clusters to find

        Returns:
            Array of cluster IDs, shape (nprobe,)
        """
        query = query.reshape(1, -1).astype(np.float32)
        _, cluster_ids = self.index.quantizer.search(query, nprobe)
        return cluster_ids[0]


def load_sift1m(data_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load SIFT1M dataset.

    Args:
        data_path: Path to SIFT1M data directory

    Returns:
        base_vectors: Training vectors, shape (1000000, 128)
        query_vectors: Query vectors, shape (10000, 128)
        ground_truth: Ground truth nearest neighbors, shape (10000, 100)
    """
    logger.info(f"Loading SIFT1M from {data_path}...")

    from src.datasets import load_fvecs, load_ivecs

    base_vectors = load_fvecs(str(data_path / "sift_base.fvecs"))
    query_vectors = load_fvecs(str(data_path / "sift_query.fvecs"))
    ground_truth = load_ivecs(str(data_path / "sift_groundtruth.ivecs"))

    logger.info(f"Loaded {len(base_vectors)} base vectors, "
                f"{len(query_vectors)} queries, "
                f"{len(ground_truth)} ground truth")

    return base_vectors, query_vectors, ground_truth


if __name__ == "__main__":
    # Test the index builder
    print("Testing IVF Index Builder...")

    # Create synthetic data
    dimension = 128
    num_vectors = 10000
    num_clusters = 100

    vectors = np.random.randn(num_vectors, dimension).astype(np.float32)

    # Build index
    builder = IVFIndexBuilder(dimension, num_clusters)
    builder.build(vectors)

    # Extract inverted lists
    inverted_lists = builder.extract_inverted_lists()

    # Test search
    queries = np.random.randn(10, dimension).astype(np.float32)
    distances, indices = builder.search(queries, k=10, nprobe=10)

    print(f"Search results shape: {indices.shape}")
    print(f"Number of clusters: {len(inverted_lists)}")
    print("Test passed!")
