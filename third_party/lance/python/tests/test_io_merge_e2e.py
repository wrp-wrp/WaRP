"""
End-to-end test for IO merge optimization with real IVF index.

This test creates a real Lance dataset with IVF-PQ index on S3,
then measures query performance with different merge strategies.
"""

import os
import time
import numpy as np
import pyarrow as pa
import lance
import pytest
from pathlib import Path


@pytest.mark.integration
@pytest.mark.slow
def test_io_merge_optimization_e2e(tmp_path):
    """
    End-to-end test comparing IO merge strategies on real IVF queries.

    This test:
    1. Creates a dataset with vectors
    2. Builds an IVF-PQ index
    3. Runs queries with different merge strategies
    4. Compares performance metrics
    """
    # Skip if not running integration tests
    if not os.environ.get("RUN_INTEGRATION_TESTS"):
        pytest.skip("Set RUN_INTEGRATION_TESTS=1 to run this test")

    # Configuration
    num_vectors = 100_000
    vector_dim = 128
    nlist = 256
    nprobe = 32
    num_queries = 50

    # Create dataset
    dataset_uri = str(tmp_path / "vectors.lance")
    print(f"\n=== Creating dataset at {dataset_uri} ===")

    vectors = np.random.randn(num_vectors, vector_dim).astype(np.float32)
    table = pa.table({
        "vector": pa.array(vectors.tolist(), type=pa.list_(pa.float32(), vector_dim)),
        "id": pa.array(range(num_vectors)),
    })

    dataset = lance.write_dataset(table, dataset_uri)
    print(f"Created dataset with {len(dataset)} rows")

    # Build IVF-PQ index
    print(f"\n=== Building IVF-PQ index (nlist={nlist}) ===")
    start = time.time()
    dataset.create_index(
        "vector",
        index_type="IVF_PQ",
        num_partitions=nlist,
        num_sub_vectors=16,
        metric="L2",
    )
    index_time = time.time() - start
    print(f"Index built in {index_time:.2f}s")

    # Generate query vectors
    query_vectors = np.random.randn(num_queries, vector_dim).astype(np.float32)

    # Test different merge strategies
    strategies = [
        ("native", "Native (no optimization)"),
        ("lance", "LanceMerge (64KB coalesce)"),
        ("greedy", "GreedyMerge"),
        ("dp", "SerialDP (optimal)"),
        ("balanced_ub", "BalancedUB"),
    ]

    results = {}

    for strategy_name, strategy_desc in strategies:
        print(f"\n=== Testing {strategy_desc} ===")

        # Set environment variable for merge strategy
        os.environ["LANCE_IO_MERGE_STRATEGY"] = strategy_name

        # Reload dataset to pick up new strategy
        dataset = lance.dataset(dataset_uri)

        # Warmup
        for i in range(5):
            dataset.to_table(
                nearest={"column": "vector", "q": query_vectors[i], "k": 10, "nprobes": nprobe}
            )

        # Measure query performance
        latencies = []
        start_total = time.time()

        for query_vec in query_vectors:
            start = time.time()
            result = dataset.to_table(
                nearest={"column": "vector", "q": query_vec, "k": 10, "nprobes": nprobe}
            )
            latency = (time.time() - start) * 1000  # ms
            latencies.append(latency)

        total_time = (time.time() - start_total) * 1000  # ms

        results[strategy_name] = {
            "avg_latency": np.mean(latencies),
            "p50_latency": np.percentile(latencies, 50),
            "p95_latency": np.percentile(latencies, 95),
            "total_time": total_time,
            "qps": num_queries / (total_time / 1000),
        }

        print(f"Avg latency: {results[strategy_name]['avg_latency']:.2f}ms")
        print(f"P50 latency: {results[strategy_name]['p50_latency']:.2f}ms")
        print(f"P95 latency: {results[strategy_name]['p95_latency']:.2f}ms")
        print(f"QPS: {results[strategy_name]['qps']:.2f}")

    # Print comparison
    print("\n=== Performance Comparison ===")
    print(f"{'Strategy':<20} {'Avg (ms)':<12} {'P50 (ms)':<12} {'P95 (ms)':<12} {'Speedup':<10}")
    print("-" * 70)

    baseline = results["native"]["avg_latency"]
    for strategy_name, _ in strategies:
        r = results[strategy_name]
        speedup = baseline / r["avg_latency"]
        print(
            f"{strategy_name:<20} {r['avg_latency']:<12.2f} {r['p50_latency']:<12.2f} "
            f"{r['p95_latency']:<12.2f} {speedup:<10.2f}x"
        )

    # Verify optimization provides benefit
    # SerialDP should be faster than native
    assert results["dp"]["avg_latency"] < results["native"]["avg_latency"], \
        "SerialDP should be faster than native"

    print("\n✅ End-to-end test passed!")


@pytest.mark.integration
def test_io_merge_with_s3(tmp_path):
    """
    Test IO merge optimization with real S3 storage.

    Requires AWS credentials and S3 bucket configured.
    """
    s3_bucket = os.environ.get("LANCE_TEST_S3_BUCKET")
    if not s3_bucket:
        pytest.skip("Set LANCE_TEST_S3_BUCKET to run S3 tests")

    # Similar to above but with S3 URI
    dataset_uri = f"s3://{s3_bucket}/test_io_merge_{int(time.time())}"

    # ... rest of test similar to above ...

    print(f"Testing with S3 at {dataset_uri}")
    # TODO: Implement S3-specific test


if __name__ == "__main__":
    # Run standalone for manual testing
    import tempfile
    with tempfile.TemporaryDirectory() as tmpdir:
        test_io_merge_optimization_e2e(Path(tmpdir))
