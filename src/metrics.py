"""
Metrics Utilities

Provides recall computation for vector search evaluation.
"""

import numpy as np


def compute_recall(retrieved: np.ndarray, ground_truth: np.ndarray, k: int = 10) -> float:
    """
    Compute recall@k.

    Args:
        retrieved: Retrieved indices, shape (n_queries, n_retrieved)
        ground_truth: Ground truth indices, shape (n_queries, n_gt)
        k: Number of top results to consider

    Returns:
        Recall@k score
    """
    n_queries = len(retrieved)
    recalls = []

    for i in range(n_queries):
        retrieved_k = set(retrieved[i, :k])
        gt_k = set(ground_truth[i, :k])

        if len(gt_k) > 0:
            recall = len(retrieved_k & gt_k) / len(gt_k)
            recalls.append(recall)

    return np.mean(recalls) if recalls else 0.0
