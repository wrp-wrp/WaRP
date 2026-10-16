// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Simple test program to compare merge strategies

use lance_io::scheduler::{
    ConcurrentMerge, CostModel, GreedyMerge, GreedyMergeCapped, LanceMerge, MergeStrategy, SerialDP,
};
use std::collections::HashMap;

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

fn main() {
    println!("=== S3 I/O Merge Strategy Comparison ===\n");

    // Simulate IVF workload: 128 clusters, 250KB each, 50KB gaps
    let (cluster_ids, cluster_offsets, cluster_sizes) = generate_ivf_workload(128, 250_000, 50_000);

    let mut cost_model = CostModel::default();
    cost_model.latency_per_request = 169.3; // S3 typical TTFB
    cost_model.latency_per_byte = 0.0001; // ~10 Gbps
    cost_model.max_waste_ratio = 0.5;
    cost_model.max_merge_span = 16 * 1024 * 1024; // 16MB

    println!(
        "Workload: {} clusters, 250KB each, 50KB gaps",
        cluster_ids.len()
    );
    println!(
        "Cost model: {:.1}ms overhead, waste ratio <= {:.0}%\n",
        cost_model.latency_per_request,
        cost_model.max_waste_ratio * 100.0
    );

    let strategies: Vec<(&str, Box<dyn MergeStrategy>)> = vec![
        ("Baseline (No merge)", Box::new(LanceMerge::new())),
        ("LanceMerge (64KB)", Box::new(LanceMerge::new())),
        ("GreedyMerge", Box::new(GreedyMerge::new())),
        ("GreedyMergeCapped", Box::new(GreedyMergeCapped::new())),
        ("SerialDP", Box::new(SerialDP::new())),
        ("ConcurrentMerge", Box::new(ConcurrentMerge::new())),
    ];

    println!(
        "{:<20} {:>10} {:>12} {:>12} {:>10}",
        "Strategy", "Requests", "Total (MB)", "Waste %", "Latency (ms)"
    );
    println!("{}", "-".repeat(70));

    for (name, strategy) in strategies.iter() {
        let plan =
            strategy.generate_plan(&cluster_ids, &cluster_offsets, &cluster_sizes, &cost_model);

        let total_mb = plan.metrics.total_bytes as f64 / (1024.0 * 1024.0);
        let waste_pct = plan.metrics.waste_ratio * 100.0;

        println!(
            "{:<20} {:>10} {:>12.2} {:>12.1} {:>10.1}",
            name, plan.metrics.num_requests, total_mb, waste_pct, plan.metrics.total_latency
        );
    }

    println!("\n=== Detailed Analysis ===\n");

    // Compare SerialDP vs LanceMerge
    let lance_plan = LanceMerge::new().generate_plan(
        &cluster_ids,
        &cluster_offsets,
        &cluster_sizes,
        &cost_model,
    );

    let serial_plan =
        SerialDP::new().generate_plan(&cluster_ids, &cluster_offsets, &cluster_sizes, &cost_model);

    let request_reduction = (1.0
        - serial_plan.metrics.num_requests as f64 / lance_plan.metrics.num_requests as f64)
        * 100.0;
    let latency_reduction =
        (1.0 - serial_plan.metrics.total_latency / lance_plan.metrics.total_latency) * 100.0;

    println!("SerialDP vs LanceMerge:");
    println!("  Request reduction: {:.1}%", request_reduction);
    println!("  Latency reduction: {:.1}%", latency_reduction);
    println!(
        "  Cost savings: ${:.4} per query (at $0.0004/1000 requests)",
        (lance_plan.metrics.num_requests as f64 - serial_plan.metrics.num_requests as f64)
            * 0.0000004
    );
}
