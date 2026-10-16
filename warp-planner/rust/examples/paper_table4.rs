//! Current-host planner-overhead spot check using the paper Table 4
//! `(nprobe, K)` shapes.
//!
//! `MainSolution` picks its inner solver from the cost shape and
//! request cap: linear fast path, divide-and-conquer, WQS, or plain DP.
//! This example uses deterministic synthetic request ranges, not the
//! paper's exact workload or hardware. Published p95 numbers are printed
//! only as a historical reference; this is not a Table 4 reproduction.
//!
//! Run with `cargo run --release --example paper_table4`.

use std::time::Instant;

use rand::{rngs::StdRng, Rng, SeedableRng};
use warp_planner::{
    strategy::{MainSolution, MergeStrategy},
    CostModel, IoRequest, PlanConstraints,
};

fn paper_cost() -> CostModel {
    CostModel {
        latency_model: "piecewise_linear".to_string(),
        latency_pw_breaks_bytes: vec![3_145_728],
        latency_pw_overheads_ms: vec![28.849557293569053, 6.266035367002434],
        latency_pw_rates_ms_per_byte: vec![2.824569398517584e-06, 1.0003677676972243e-05],
        latency_per_request: 28.849557293569053,
        latency_per_byte: 2.824569398517584e-06,
        concurrent_parallelism: 16.0,
        ..CostModel::default()
    }
}

fn realistic_inputs(n: usize, seed: u64) -> Vec<IoRequest> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut offset: i64 = 0;
    (0..n)
        .map(|i| {
            let length = 256 + rng.gen_range(0..2048);
            let r = IoRequest {
                cluster_id: i as i32,
                offset,
                length,
            };
            offset = r.end() + rng.gen_range(0..8192);
            r
        })
        .collect()
}

fn p95(mut samples: Vec<f64>) -> f64 {
    samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (samples.len() as f64 * 0.95) as usize;
    samples[idx.min(samples.len() - 1)]
}

fn measure(n: usize, k: i32, repeats: usize) -> f64 {
    let cost = paper_cost();
    let constraints = PlanConstraints {
        max_requests: k,
        exact_requests: false,
    };
    let mut samples = Vec::with_capacity(repeats);
    for r in 0..repeats {
        let inputs = realistic_inputs(n, 0xCAFE + r as u64);
        let start = Instant::now();
        let _plan = MainSolution.generate_plan(&inputs, &cost, &constraints);
        let dt_ms = start.elapsed().as_secs_f64() * 1_000.0;
        samples.push(dt_ms);
    }
    p95(samples)
}

fn main() {
    let cells = [
        (128, 32),
        (128, 64),
        (128, 128),
        (256, 64),
        (256, 128),
        (256, 256),
        (512, 128),
        (512, 256),
        (512, 512),
    ];
    println!(
        "  {:>6}   {:>6}   {:>10}   {:>16}",
        "nprobe", "K", "current p95", "published p95"
    );
    println!("  {}", "-".repeat(50));
    let paper_p95: &[f64] = &[0.87, 0.54, 0.50, 1.90, 1.72, 2.17, 4.72, 7.17, 9.90];
    for ((n, k), &paper) in cells.iter().zip(paper_p95.iter()) {
        let ours = measure(*n, *k, 50);
        println!("  {:>6}   {:>6}   {:>10.2}   {:>16.2}", n, k, ours, paper);
    }
}
