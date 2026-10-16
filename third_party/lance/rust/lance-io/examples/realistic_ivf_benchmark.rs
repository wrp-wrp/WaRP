// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Realistic IVF workload simulation

use lance_io::scheduler::{
    ConcurrentMerge, CostModel, GreedyMerge, LanceMerge, MergeStrategy, SerialDP,
};
use std::collections::HashMap;

fn generate_realistic_ivf_workload(
    nprobe: usize,
    total_clusters: usize,
    avg_cluster_size: i64,
) -> (Vec<i32>, HashMap<i32, i64>, HashMap<i32, i64>) {
    use rand::Rng;
    let mut rng = rand::rng();

    let mut cluster_ids = Vec::new();
    let mut cluster_offsets = HashMap::new();
    let mut cluster_sizes = HashMap::new();

    // Randomly select nprobe clusters from total_clusters
    let mut selected: Vec<usize> = (0..total_clusters).collect();
    for i in 0..nprobe {
        let j = rng.random_range(i..total_clusters);
        selected.swap(i, j);
    }
    selected.truncate(nprobe);
    selected.sort();

    // Assign offsets and sizes
    for &idx in &selected {
        let cid = idx as i32;
        cluster_ids.push(cid);

        // Cluster offset: simulate file layout
        let offset = (idx as i64) * (avg_cluster_size + 100_000); // 100KB gaps
        cluster_offsets.insert(cid, offset);

        // Cluster size: add some variance (±20%)
        let variance = rng.random_range(-20..=20);
        let size = avg_cluster_size + (avg_cluster_size * variance / 100);
        cluster_sizes.insert(cid, size.max(10_000));
    }

    (cluster_ids, cluster_offsets, cluster_sizes)
}

fn main() {
    println!("=== Realistic IVF Query Simulation ===\n");

    // Typical IVF parameters
    let scenarios = vec![
        ("Small (nprobe=16)", 16, 4096, 250_000),
        ("Medium (nprobe=64)", 64, 4096, 250_000),
        ("Large (nprobe=128)", 128, 4096, 250_000),
        ("XLarge (nprobe=256)", 256, 8192, 200_000),
    ];

    for (name, nprobe, total_clusters, avg_cluster_size) in scenarios {
        println!("=== {} ===", name);
        println!(
            "nprobe={}, total_clusters={}, avg_cluster_size={}KB\n",
            nprobe,
            total_clusters,
            avg_cluster_size / 1024
        );

        // Generate 10 random queries and average results
        let mut results: HashMap<&str, Vec<f64>> = HashMap::new();

        for _ in 0..10 {
            let (cluster_ids, cluster_offsets, cluster_sizes) =
                generate_realistic_ivf_workload(nprobe, total_clusters, avg_cluster_size);

            let mut cost_model = CostModel::default();
            cost_model.latency_per_request = 169.3; // S3 TTFB
            cost_model.latency_per_byte = 0.00001; // ~100 Gbps
            cost_model.max_waste_ratio = 0.5;
            cost_model.max_merge_span = 16 * 1024 * 1024;

            let strategies: Vec<(&str, Box<dyn MergeStrategy>)> = vec![
                ("LanceMerge", Box::new(LanceMerge::new())),
                ("GreedyMerge", Box::new(GreedyMerge::new())),
                ("SerialDP", Box::new(SerialDP::new())),
                ("ConcurrentMerge", Box::new(ConcurrentMerge::new())),
            ];

            for (strategy_name, strategy) in strategies {
                let plan = strategy.generate_plan(
                    &cluster_ids,
                    &cluster_offsets,
                    &cluster_sizes,
                    &cost_model,
                );

                results
                    .entry(strategy_name)
                    .or_insert_with(Vec::new)
                    .push(plan.metrics.total_latency);
            }
        }

        // Print averaged results
        println!(
            "{:<20} {:>15} {:>15} {:>15}",
            "Strategy", "Avg Latency (ms)", "vs Lance (%)", "Speedup"
        );
        println!("{}", "-".repeat(70));

        let lance_avg: f64 = results["LanceMerge"].iter().sum::<f64>() / 10.0;

        for strategy_name in ["LanceMerge", "GreedyMerge", "SerialDP", "ConcurrentMerge"] {
            let avg: f64 = results[strategy_name].iter().sum::<f64>() / 10.0;
            let vs_lance = ((avg - lance_avg) / lance_avg) * 100.0;
            let speedup = lance_avg / avg;

            println!(
                "{:<20} {:>15.1} {:>15.1} {:>15.2}x",
                strategy_name, avg, vs_lance, speedup
            );
        }
        println!();
    }

    println!("\n=== Cost Analysis (1M queries/month) ===\n");

    let (cluster_ids, cluster_offsets, cluster_sizes) =
        generate_realistic_ivf_workload(128, 4096, 250_000);

    let mut cost_model = CostModel::default();
    cost_model.cost_per_request = 0.0000004; // $0.0004 per 1000 requests
    cost_model.latency_per_request = 169.3;
    cost_model.max_waste_ratio = 0.5;

    let lance_plan = LanceMerge::new().generate_plan(
        &cluster_ids,
        &cluster_offsets,
        &cluster_sizes,
        &cost_model,
    );
    let serial_plan =
        SerialDP::new().generate_plan(&cluster_ids, &cluster_offsets, &cluster_sizes, &cost_model);

    let queries_per_month = 1_000_000.0;
    let lance_cost = lance_plan.metrics.request_cost * queries_per_month;
    let serial_cost = serial_plan.metrics.request_cost * queries_per_month;
    let savings = lance_cost - serial_cost;

    println!(
        "LanceMerge: {} requests/query, ${:.2}/month",
        lance_plan.metrics.num_requests, lance_cost
    );
    println!(
        "SerialDP:   {} requests/query, ${:.2}/month",
        serial_plan.metrics.num_requests, serial_cost
    );
    println!(
        "Savings:    ${:.2}/month ({:.1}% reduction)",
        savings,
        (savings / lance_cost) * 100.0
    );
}
