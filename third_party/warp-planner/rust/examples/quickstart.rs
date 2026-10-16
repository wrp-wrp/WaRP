//! Hello, warp-planner.
//!
//! The smallest possible call: build three `IoRequest`s, construct a
//! `CostModel`, run a strategy, inspect the merged plan, score it.
//!
//! ```bash
//! cargo run --release --example quickstart
//! ```

use warp_planner::{evaluate, plan, CostModel, IoRequest, MainSolution, PlanConstraints};

fn main() {
    // 1. What you want to fetch — three small ranges in one cloud object.
    let requests = vec![
        IoRequest {
            cluster_id: 0,
            offset: 0,
            length: 8_192,
        },
        IoRequest {
            cluster_id: 1,
            offset: 8_192,
            length: 8_192,
        },
        IoRequest {
            cluster_id: 2,
            offset: 5_000_000,
            length: 8_192,
        },
    ];

    // 2. Describe the storage backend's cost.
    let cost = CostModel {
        latency_per_request: 27.0, // ms per GET
        latency_per_byte: 1.0e-5,  // ms per byte
        concurrent_parallelism: 16.0,
        ..CostModel::default()
    };
    let constraints = PlanConstraints::default();

    // 3. Plan: turn the three reads into a merged plan.
    let plan = plan(&MainSolution, &requests, &cost, &constraints)
        .expect("the example inputs and cost model are valid");

    // 4. Inspect.
    println!("Merged into {} request(s):", plan.merged_requests.len());
    for (i, m) in plan.merged_requests.iter().enumerate() {
        println!("  [{i}] offset={:>9} length={:>6}", m.offset, m.length);
    }
    println!("Per-input assignment: {:?}", plan.assignments);

    // 5. Score it against the same cost model.
    let metrics = evaluate(&requests, &plan, &cost);
    println!(
        "\nPredicted latency: {:.2} ms",
        metrics.t_ub_ms(cost.concurrent_parallelism),
    );
    println!(
        "Bytes:     {} useful + {} wasted = {} fetched",
        metrics.useful_bytes, metrics.wasted_bytes, metrics.total_bytes,
    );
}
