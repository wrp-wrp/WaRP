//! Planner-quality sweep on synthetic inputs.
//!
//! Criterion measures time; this bench piggybacks on it to also report
//! the predicted-latency ratio each strategy achieves against the
//! `Separate` baseline across three synthetic distributions. The
//! interesting output is the quality table printed to stderr; the
//! Criterion rows capture plan-time at the same configuration.

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use rand::{rngs::StdRng, Rng, SeedableRng};
use warp_planner::{
    evaluate,
    strategy::{Bottleneck, Greedy, MainSolution, MergeStrategy, Separate, Serial},
    CostModel, IoRequest, PlanConstraints, PlanMetrics,
};

// ---------------------------------------------------------------------
// Synthetic distributions (same set as plan_time.rs, for comparability).
// ---------------------------------------------------------------------

fn uniform_contiguous(n: usize, _seed: u64) -> Vec<IoRequest> {
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

fn paper_cost(parallelism: f64) -> CostModel {
    CostModel {
        latency_model: "piecewise_linear".to_string(),
        latency_pw_breaks_bytes: vec![3_145_728],
        latency_pw_overheads_ms: vec![28.849557293569053, 6.266035367002434],
        latency_pw_rates_ms_per_byte: vec![2.824569398517584e-06, 1.0003677676972243e-05],
        latency_per_request: 28.849557293569053,
        latency_per_byte: 2.824569398517584e-06,
        concurrent_parallelism: parallelism,
        ..CostModel::default()
    }
}

fn predicted_latency(m: &PlanMetrics, p: f64) -> f64 {
    m.t_ub_ms(p)
}

// ---------------------------------------------------------------------
// Quality table.
// ---------------------------------------------------------------------

fn quality_table(name: &str, inputs: &[IoRequest], cost: &CostModel) {
    let p = cost.concurrent_parallelism.max(1.0);
    let constraints = PlanConstraints::default();
    let strats: Vec<Box<dyn MergeStrategy>> = vec![
        Box::new(Separate),
        Box::new(Greedy::default()),
        Box::new(Serial),
        Box::new(Bottleneck),
        Box::new(MainSolution),
    ];
    eprintln!("\n[quality] {} (n={}, P={:.1})", name, inputs.len(), p);
    eprintln!(
        "  {:14} {:>9} {:>9} {:>11} {:>10} {:>7} {:>8}",
        "strategy", "serial", "largest", "latency", "lat_saved", "waste%", "req",
    );
    let baseline_plan = Separate.generate_plan(inputs, cost, &constraints);
    let baseline_m = evaluate(inputs, &baseline_plan, cost);
    let baseline_latency = predicted_latency(&baseline_m, p).max(1e-12);

    for strat in &strats {
        let plan = strat.generate_plan(inputs, cost, &constraints);
        let m = evaluate(inputs, &plan, cost);
        let latency = predicted_latency(&m, p);
        let waste = if m.total_bytes > 0 {
            100.0 * m.wasted_bytes as f64 / m.total_bytes as f64
        } else {
            0.0
        };
        eprintln!(
            "  {:14} {:>9.2} {:>9.2} {:>11.2} {:>9.1}% {:>6.1}% {:>8}",
            strat.name(),
            m.predicted_latency_ms_serial,
            m.predicted_latency_ms_concurrent,
            latency,
            100.0 * (1.0 - latency / baseline_latency),
            waste,
            m.request_count,
        );
    }
}

// ---------------------------------------------------------------------
// Bench entry: print the quality table and also time each strategy.
// ---------------------------------------------------------------------

fn bench_plan_quality(c: &mut Criterion) {
    type Gen = fn(usize, u64) -> Vec<IoRequest>;
    let distributions: Vec<(&str, Gen)> = vec![
        ("uniform_contiguous", uniform_contiguous),
        ("exp_gaps", exp_gaps),
        ("heavy_tail", heavy_tail),
    ];
    let n = 1024usize;
    let cost = paper_cost(16.0);
    let constraints = PlanConstraints::default();

    for (name, gen) in &distributions {
        let inputs = gen(n, 0xC0FFEE);
        quality_table(name, &inputs, &cost);

        let mut group = c.benchmark_group(format!("plan_quality/{name}"));
        let strats: [(&str, Box<dyn MergeStrategy>); 4] = [
            ("greedy", Box::new(Greedy::default())),
            ("serial", Box::new(Serial)),
            ("bottleneck", Box::new(Bottleneck)),
            ("main_solution", Box::new(MainSolution)),
        ];
        for (tag, strat) in strats {
            group.bench_with_input(BenchmarkId::from_parameter(tag), &inputs, |b, inputs| {
                b.iter(|| {
                    let plan = strat.generate_plan(
                        black_box(inputs),
                        black_box(&cost),
                        black_box(&constraints),
                    );
                    black_box(evaluate(inputs, &plan, &cost));
                })
            });
        }
        group.finish();
    }
}

criterion_group!(benches, bench_plan_quality);
criterion_main!(benches);
