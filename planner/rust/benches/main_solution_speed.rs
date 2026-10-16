//! Plan-time scaling for `MainSolution`.
//!
//! Three regimes, named by which inner solver gets dispatched:
//!
//! - **`linear`** — genuinely linear cost. Dispatches the monotonic-
//!   deque `O(n·k)` fast path.
//! - **`piecewise_small_k`** — piecewise convex cost with
//!   `max_requests = 16`. `k_cap = 16 < n/4`, so dispatches DC
//!   `O(n·k·log n)` per layer. This is the **paper deployment regime**
//!   (small concurrency budget per query); paper Table 4 columns
//!   K=32, 64 fall here at n ≥ 256.
//! - **`piecewise_large_k`** — piecewise convex cost without a
//!   `max_requests` cap (`k_cap = n`, so `k_cap ≥ n/4`). Dispatches
//!   WQS Lagrangian. This is the regime paper Table 4 K=n cells use.
//!
//! Plus a parallelism sweep at fixed `n` to expose the adaptive-hull
//! early-exit behaviour.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use warp_planner::{
    strategy::{MainSolution, MergeStrategy},
    CostModel, IoRequest, PlanConstraints,
};

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

fn fast_path_cost(p: f64) -> CostModel {
    CostModel {
        latency_per_request: 28.849557293569053,
        latency_per_byte: 2.824569398517584e-06,
        concurrent_parallelism: p,
        ..CostModel::default()
    }
}

fn piecewise_cost(p: f64) -> CostModel {
    CostModel {
        latency_model: "piecewise_linear".to_string(),
        latency_pw_breaks_bytes: vec![3_145_728],
        latency_pw_overheads_ms: vec![28.849557293569053, 6.266035367002434],
        latency_pw_rates_ms_per_byte: vec![2.824569398517584e-06, 1.0003677676972243e-05],
        latency_per_request: 28.849557293569053,
        latency_per_byte: 2.824569398517584e-06,
        concurrent_parallelism: p,
        ..CostModel::default()
    }
}

/// Small request-count cap (= concurrency budget). At `k_cap < n/4`
/// the DC inner solver wins over WQS; this matches paper Table 4
/// K = 32 / 64 cells and the q4-io64 deployment target.
fn piecewise_small_k_constraints(max_requests: i32) -> PlanConstraints {
    PlanConstraints {
        max_requests,
        exact_requests: false,
    }
}

fn bench_size_scaling(c: &mut Criterion) {
    let p = 16.0;
    for &n in &[64usize, 128, 256, 512, 1024, 2048, 4096] {
        let inputs = realistic_inputs(n, 0xCAFE);
        {
            let cost = fast_path_cost(p);
            let constraints = PlanConstraints::default();
            let mut g = c.benchmark_group("main_solution/linear");
            g.bench_with_input(BenchmarkId::from_parameter(n), &inputs, |b, inputs| {
                b.iter(|| {
                    MainSolution.generate_plan(
                        black_box(inputs),
                        black_box(&cost),
                        black_box(&constraints),
                    )
                })
            });
            g.finish();
        }
        if n <= 1024 {
            // No request cap → k_cap = n ≥ n/4 → WQS dispatches.
            let cost = piecewise_cost(p);
            let constraints = PlanConstraints::default();
            let mut g = c.benchmark_group("main_solution/piecewise_large_k");
            g.bench_with_input(BenchmarkId::from_parameter(n), &inputs, |b, inputs| {
                b.iter(|| {
                    MainSolution.generate_plan(
                        black_box(inputs),
                        black_box(&cost),
                        black_box(&constraints),
                    )
                })
            });
            g.finish();
        }
        {
            // max_requests=16 → k_cap = 16 < n/4 → DC dispatches.
            // This is the paper's deployment regime (small per-query
            // concurrency budget).
            let cost = piecewise_cost(p);
            let constraints = piecewise_small_k_constraints(16);
            let mut g = c.benchmark_group("main_solution/piecewise_small_k");
            g.bench_with_input(BenchmarkId::from_parameter(n), &inputs, |b, inputs| {
                b.iter(|| {
                    MainSolution.generate_plan(
                        black_box(inputs),
                        black_box(&cost),
                        black_box(&constraints),
                    )
                })
            });
            g.finish();
        }
    }
}

fn bench_parallelism_sweep(c: &mut Criterion) {
    let n = 512usize;
    let inputs = realistic_inputs(n, 0xCAFE);
    for &p in &[1.0f64, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0] {
        let cost = piecewise_cost(p);
        let constraints = PlanConstraints::default();
        let mut g = c.benchmark_group("main_solution/parallelism_sweep_n512");
        g.bench_with_input(
            BenchmarkId::from_parameter(p as u32),
            &inputs,
            |b, inputs| {
                b.iter(|| {
                    MainSolution.generate_plan(
                        black_box(inputs),
                        black_box(&cost),
                        black_box(&constraints),
                    )
                })
            },
        );
        g.finish();
    }
}

criterion_group!(benches, bench_size_scaling, bench_parallelism_sweep);
criterion_main!(benches);
