// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Benchmark for I/O merge strategies
//!
//! This benchmark compares different merge strategies on simulated
//! IVF-style workloads to measure their effectiveness.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use lance_io::scheduler::{
    ConcurrentMerge, CostModel, GreedyMerge, GreedyMergeCapped, LanceMerge, MergeStrategy, SerialDP,
};
use std::collections::HashMap;

/// Generate a simulated IVF workload
fn generate_ivf_workload(
    num_clusters: usize,
    cluster_size: i64,
    gap_size: i64,
) -> (Vec<i32>, HashMap<i32, i64>, HashMap<i32, i64>) {
    let mut cluster_ids = Vec::new();
    let mut cluster_offsets = HashMap::new();
    let mut cluster_sizes = HashMap::new();

    let mut current_offset = 0i64;
    for i in 0..num_clusters {
        let cid = i as i32;
        cluster_ids.push(cid);
        cluster_offsets.insert(cid, current_offset);
        cluster_sizes.insert(cid, cluster_size);
        current_offset += cluster_size + gap_size;
    }

    (cluster_ids, cluster_offsets, cluster_sizes)
}

fn bench_merge_strategies(c: &mut Criterion) {
    let mut group = c.benchmark_group("merge_strategies");

    // Test different workload sizes
    for num_clusters in [16, 32, 64, 128, 256] {
        let (cluster_ids, cluster_offsets, cluster_sizes) =
            generate_ivf_workload(num_clusters, 250_000, 50_000); // 250KB clusters, 50KB gaps

        let mut cost_model = CostModel::default();
        cost_model.latency_per_request = 169.3; // S3 typical latency
        cost_model.max_waste_ratio = 0.5;
        cost_model.max_merge_span = 16 * 1024 * 1024; // 16MB

        // Benchmark LanceMerge (baseline)
        group.bench_with_input(
            BenchmarkId::new("LanceMerge", num_clusters),
            &num_clusters,
            |b, _| {
                let strategy = LanceMerge::new();
                b.iter(|| {
                    strategy.generate_plan(
                        black_box(&cluster_ids),
                        black_box(&cluster_offsets),
                        black_box(&cluster_sizes),
                        black_box(&cost_model),
                    )
                });
            },
        );

        // Benchmark GreedyMerge
        group.bench_with_input(
            BenchmarkId::new("GreedyMerge", num_clusters),
            &num_clusters,
            |b, _| {
                let strategy = GreedyMerge::new();
                b.iter(|| {
                    strategy.generate_plan(
                        black_box(&cluster_ids),
                        black_box(&cluster_offsets),
                        black_box(&cluster_sizes),
                        black_box(&cost_model),
                    )
                });
            },
        );

        // Benchmark GreedyMergeCapped
        group.bench_with_input(
            BenchmarkId::new("GreedyMergeCapped", num_clusters),
            &num_clusters,
            |b, _| {
                let strategy = GreedyMergeCapped::new();
                b.iter(|| {
                    strategy.generate_plan(
                        black_box(&cluster_ids),
                        black_box(&cluster_offsets),
                        black_box(&cluster_sizes),
                        black_box(&cost_model),
                    )
                });
            },
        );

        // Benchmark SerialDP
        group.bench_with_input(
            BenchmarkId::new("SerialDP", num_clusters),
            &num_clusters,
            |b, _| {
                let strategy = SerialDP::new();
                b.iter(|| {
                    strategy.generate_plan(
                        black_box(&cluster_ids),
                        black_box(&cluster_offsets),
                        black_box(&cluster_sizes),
                        black_box(&cost_model),
                    )
                });
            },
        );

        // Benchmark ConcurrentMerge
        group.bench_with_input(
            BenchmarkId::new("ConcurrentMerge", num_clusters),
            &num_clusters,
            |b, _| {
                let strategy = ConcurrentMerge::new();
                b.iter(|| {
                    strategy.generate_plan(
                        black_box(&cluster_ids),
                        black_box(&cluster_offsets),
                        black_box(&cluster_sizes),
                        black_box(&cost_model),
                    )
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_merge_strategies);
criterion_main!(benches);
