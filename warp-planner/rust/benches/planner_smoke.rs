//! Short performance smoke for the checked public API.
//!
//! This intentionally has no absolute wall-clock assertion: shared CI
//! runners are too noisy for a stable latency gate. Criterion records
//! estimates for three representative solver regimes and will fail on
//! crashes or invalid output. Use `main_solution_speed` for the full
//! scaling study.

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion, Throughput};
use warp_planner::{plan, CostModel, IoRequest, MainSolution, PlanConstraints};

fn inputs(n: usize) -> Vec<IoRequest> {
    let mut offset = 0i64;
    (0..n)
        .map(|i| {
            let length = 512 + ((i * 104_729) % 2048) as i64;
            let request = IoRequest {
                cluster_id: i as i32,
                offset,
                length,
            };
            offset = request.end() + ((i * 7_919) % 8192) as i64;
            request
        })
        .collect()
}

fn linear_cost() -> CostModel {
    CostModel {
        latency_per_request: 28.85,
        latency_per_byte: 2.82e-6,
        concurrent_parallelism: 16.0,
        ..CostModel::default()
    }
}

fn piecewise_cost() -> CostModel {
    CostModel {
        latency_model: "piecewise_linear".to_string(),
        latency_pw_breaks_bytes: vec![3_145_728],
        latency_pw_overheads_ms: vec![28.85, 6.27],
        latency_pw_rates_ms_per_byte: vec![2.82e-6, 1.00e-5],
        latency_per_request: 28.85,
        latency_per_byte: 2.82e-6,
        concurrent_parallelism: 16.0,
        ..CostModel::default()
    }
}

fn bench_case(
    c: &mut Criterion,
    label: &str,
    n: usize,
    cost: CostModel,
    constraints: PlanConstraints,
) {
    let requests = inputs(n);
    let mut group = c.benchmark_group(format!("planner_smoke/{label}"));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(100));
    group.measurement_time(Duration::from_millis(500));
    group.throughput(Throughput::Elements(n as u64));
    group.bench_with_input(BenchmarkId::from_parameter(n), &requests, |b, requests| {
        b.iter(|| {
            plan(
                black_box(&MainSolution),
                black_box(requests),
                black_box(&cost),
                black_box(&constraints),
            )
            .expect("benchmark input must produce a valid plan")
        })
    });
    group.finish();
}

fn bench_public_api(c: &mut Criterion) {
    bench_case(c, "linear", 128, linear_cost(), PlanConstraints::default());
    bench_case(
        c,
        "piecewise_small_cap",
        512,
        piecewise_cost(),
        PlanConstraints {
            max_requests: 16,
            exact_requests: false,
        },
    );
    bench_case(
        c,
        "piecewise_large_cap",
        512,
        piecewise_cost(),
        PlanConstraints::default(),
    );
}

criterion_group!(benches, bench_public_api);
criterion_main!(benches);
