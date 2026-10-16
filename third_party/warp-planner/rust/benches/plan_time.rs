//! Plan-time sweep: strategy × cluster count × gap distribution.
//!
//! The three distributions approximate the layouts the paper
//! discusses:
//!
//! - `uniform_contiguous` — all clusters packed back-to-back, zero
//!   gaps. Upper bound on how well merging can help.
//! - `exp_gaps`            — exponential cluster sizes and gaps;
//!   approximates the natural FAISS IVFFlat layout after a spatial
//!   reorder.
//! - `heavy_tail`          — 90 % tiny gap, 10 % huge gap;
//!   approximates an unreordered IVF layout.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use warp_planner::{
    strategy::{Bottleneck, Greedy, MainSolution, MergeStrategy, Separate, Serial},
    CostModel, IoRequest, PlanConstraints,
};

fn uniform_contiguous(n: usize) -> Vec<IoRequest> {
    (0..n)
        .map(|i| IoRequest {
            cluster_id: i as i32,
            offset: (i as i64) * 512,
            length: 512,
        })
        .collect()
}

fn exp_gaps(n: usize, seed: u64) -> Vec<IoRequest> {
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

fn heavy_tail(n: usize, seed: u64) -> Vec<IoRequest> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut offset: i64 = 0;
    (0..n)
        .map(|i| {
            let length = 128 + rng.gen_range(0..256);
            let r = IoRequest {
                cluster_id: i as i32,
                offset,
                length,
            };
            let gap = if rng.gen::<f32>() < 0.9 {
                rng.gen_range(0..128)
            } else {
                rng.gen_range(32_000..128_000)
            };
            offset = r.end() + gap;
            r
        })
        .collect()
}

fn paper_cost() -> CostModel {
    CostModel {
        latency_model: "piecewise_linear".to_string(),
        latency_pw_breaks_bytes: vec![3_145_728],
        latency_pw_overheads_ms: vec![26.56, 4.01],
        latency_pw_rates_ms_per_byte: vec![3.40e-6, 1.03e-5],
        latency_per_request: 26.56,
        concurrent_parallelism: 8.22,
        ..CostModel::default()
    }
}

fn bench_sweep(c: &mut Criterion) {
    let cost = paper_cost();

    type Generator = Box<dyn Fn(usize) -> Vec<IoRequest>>;
    let distributions: Vec<(&str, Generator)> = vec![
        ("uniform_contiguous", Box::new(uniform_contiguous)),
        ("exp_gaps", Box::new(|n| exp_gaps(n, 0xC0FFEE))),
        ("heavy_tail", Box::new(|n| heavy_tail(n, 0xC0FFEE))),
    ];

    // Keep this cross-strategy matrix compact. MainSolution's dedicated
    // n-scaling and solver-regime coverage lives in
    // `main_solution_speed.rs`; this suite stops at n=128 for it.
    let cluster_counts = [64usize, 128, 256];
    let main_sol_counts = [64usize, 128];

    for (dist_name, gen) in &distributions {
        let mut group = c.benchmark_group(format!("plan_time/{dist_name}"));
        for &n in &cluster_counts {
            let inputs = gen(n);
            let default_constraints = PlanConstraints::default();
            group.bench_with_input(BenchmarkId::new("separate", n), &inputs, |b, inputs| {
                b.iter(|| {
                    Separate.generate_plan(
                        black_box(inputs),
                        black_box(&cost),
                        black_box(&default_constraints),
                    )
                })
            });
            group.bench_with_input(BenchmarkId::new("greedy", n), &inputs, |b, inputs| {
                b.iter(|| {
                    Greedy::default().generate_plan(
                        black_box(inputs),
                        black_box(&cost),
                        black_box(&default_constraints),
                    )
                })
            });
            group.bench_with_input(BenchmarkId::new("serial", n), &inputs, |b, inputs| {
                b.iter(|| {
                    Serial.generate_plan(
                        black_box(inputs),
                        black_box(&cost),
                        black_box(&default_constraints),
                    )
                })
            });
            let cap_constraints = PlanConstraints {
                max_requests: (n as i32 / 4).max(1),
                exact_requests: false,
            };
            group.bench_with_input(
                BenchmarkId::new("bottleneck_k=n/4", n),
                &inputs,
                |b, inputs| {
                    b.iter(|| {
                        Bottleneck.generate_plan(
                            black_box(inputs),
                            black_box(&cost),
                            black_box(&cap_constraints),
                        )
                    })
                },
            );
            if main_sol_counts.contains(&n) {
                group.bench_with_input(
                    BenchmarkId::new("main_solution", n),
                    &inputs,
                    |b, inputs| {
                        b.iter(|| {
                            MainSolution.generate_plan(
                                black_box(inputs),
                                black_box(&cost),
                                black_box(&default_constraints),
                            )
                        })
                    },
                );
            }
        }
        group.finish();
    }
}

criterion_group!(benches, bench_sweep);
criterion_main!(benches);
