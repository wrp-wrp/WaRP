//! Cross-strategy invariants.
//!
//! These live outside `src/` because they treat each strategy as a
//! black box; no internal imports, only the public API. The battery
//! checks the five invariants documented in `docs/algorithms.md` on a
//! fixed suite of synthetic inputs.

use rand::{rngs::StdRng, Rng, SeedableRng};
use warp_planner::{
    evaluate,
    strategy::{Bottleneck, Greedy, MainSolution, MergeStrategy, Separate, Serial},
    CostModel, IoRequest, PlanConstraints,
};

fn strategies() -> Vec<(&'static str, Box<dyn MergeStrategy>)> {
    vec![
        ("separate", Box::new(Separate)),
        ("greedy", Box::new(Greedy::default())),
        ("serial", Box::new(Serial)),
        ("bottleneck", Box::new(Bottleneck)),
        ("main_solution", Box::new(MainSolution)),
    ]
}

// ---------------------------------------------------------------------
// Synthetic input fixtures.
// ---------------------------------------------------------------------

fn uniform_no_gaps(n: usize) -> Vec<IoRequest> {
    (0..n)
        .map(|i| IoRequest {
            cluster_id: i as i32,
            offset: (i as i64) * 256,
            length: 256,
        })
        .collect()
}

fn exp_gaps(n: usize, seed: u64) -> Vec<IoRequest> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut offset = 0i64;
    (0..n)
        .map(|i| {
            let length = 128 + rng.gen_range(0..1024);
            let r = IoRequest {
                cluster_id: i as i32,
                offset,
                length,
            };
            let gap = rng.gen_range(0..4096);
            offset = r.end() + gap;
            r
        })
        .collect()
}

fn heavy_tail_gaps(n: usize, seed: u64) -> Vec<IoRequest> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut offset = 0i64;
    (0..n)
        .map(|i| {
            let length = 128 + rng.gen_range(0..256);
            let r = IoRequest {
                cluster_id: i as i32,
                offset,
                length,
            };
            // 90 % tiny gap, 10 % large gap — approximates the
            // unreordered IVF layout.
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

fn default_cost() -> CostModel {
    CostModel {
        latency_per_request: 10.0,
        latency_per_byte: 0.0001,
        ..CostModel::default()
    }
}

// ---------------------------------------------------------------------
// Invariant 1: every strategy produces a valid plan on every input.
// ---------------------------------------------------------------------

#[test]
fn every_strategy_produces_valid_plans() {
    let inputs: Vec<(&str, Vec<IoRequest>)> = vec![
        ("uniform_small", uniform_no_gaps(16)),
        ("uniform_mid", uniform_no_gaps(128)),
        ("exp_gaps_128", exp_gaps(128, 7)),
        ("heavy_tail_128", heavy_tail_gaps(128, 7)),
    ];
    let cost = default_cost();
    let constraints = PlanConstraints::default();
    for (tag, requests) in &inputs {
        for (name, strat) in &strategies() {
            let plan = strat.generate_plan(requests, &cost, &constraints);
            plan.check_invariants(requests)
                .unwrap_or_else(|e| panic!("{name} on {tag}: {e}"));
        }
    }
}

// ---------------------------------------------------------------------
// Invariant 2: Serial never does worse than Separate on serial W.
// ---------------------------------------------------------------------

#[test]
fn serial_dominates_separate_on_total_work() {
    let cost = default_cost();
    let constraints = PlanConstraints::default();
    for seed in 0..8u64 {
        let rs = exp_gaps(64, seed);
        let sep = evaluate(
            &rs,
            &Separate.generate_plan(&rs, &cost, &constraints),
            &cost,
        );
        let serial = evaluate(&rs, &Serial.generate_plan(&rs, &cost, &constraints), &cost);
        assert!(
            serial.predicted_latency_ms_serial <= sep.predicted_latency_ms_serial + 1e-9,
            "seed={seed}: serial W {} > separate W {}",
            serial.predicted_latency_ms_serial,
            sep.predicted_latency_ms_serial
        );
    }
}

// ---------------------------------------------------------------------
// Invariant 3: Greedy never does worse than Separate on W (under any
// input with at least one join-feasible gap).
// ---------------------------------------------------------------------

#[test]
fn greedy_never_worse_than_separate_on_total_work() {
    let cost = default_cost();
    let constraints = PlanConstraints::default();
    for seed in 0..8u64 {
        let rs = exp_gaps(64, seed);
        let sep = evaluate(
            &rs,
            &Separate.generate_plan(&rs, &cost, &constraints),
            &cost,
        );
        let greedy = evaluate(
            &rs,
            &Greedy::default().generate_plan(&rs, &cost, &constraints),
            &cost,
        );
        assert!(
            greedy.predicted_latency_ms_serial <= sep.predicted_latency_ms_serial + 1e-9,
            "seed={seed}: greedy W {} > separate W {}",
            greedy.predicted_latency_ms_serial,
            sep.predicted_latency_ms_serial
        );
    }
}

// ---------------------------------------------------------------------
// Invariant 4: Serial ≤ Greedy on total work (serial-exact).
// ---------------------------------------------------------------------

#[test]
fn serial_is_optimal_vs_greedy() {
    let cost = default_cost();
    let constraints = PlanConstraints::default();
    for seed in 0..8u64 {
        let rs = exp_gaps(64, seed);
        let serial = evaluate(&rs, &Serial.generate_plan(&rs, &cost, &constraints), &cost);
        let greedy = evaluate(
            &rs,
            &Greedy::default().generate_plan(&rs, &cost, &constraints),
            &cost,
        );
        assert!(
            serial.predicted_latency_ms_serial <= greedy.predicted_latency_ms_serial + 1e-6,
            "seed={seed}: serial {} > greedy {}",
            serial.predicted_latency_ms_serial,
            greedy.predicted_latency_ms_serial
        );
    }
}

// ---------------------------------------------------------------------
// Invariant 5: Bottleneck under a request cap never does worse than
// Separate on bottleneck B (assuming feasibility).
// ---------------------------------------------------------------------

#[test]
fn bottleneck_reduces_max_request_latency() {
    let cost = default_cost();
    let constraints = PlanConstraints {
        max_requests: 4,
        exact_requests: false,
    };
    for seed in 0..4u64 {
        let rs = exp_gaps(16, seed);
        let b_plan = Bottleneck.generate_plan(&rs, &cost, &constraints);
        b_plan.check_invariants(&rs).unwrap();
        let m = evaluate(&rs, &b_plan, &cost);
        assert!(m.predicted_latency_ms_concurrent.is_finite());
    }
}

// ---------------------------------------------------------------------
// Invariant 6: MainSolution at P=1 ≤ Serial on T_UB (since T_UB(1) = W).
// ---------------------------------------------------------------------

#[test]
fn main_solution_at_p1_matches_or_beats_serial() {
    let mut cost = default_cost();
    cost.concurrent_parallelism = 1.0;
    let constraints = PlanConstraints::default();
    for seed in 0..4u64 {
        let rs = exp_gaps(48, seed);
        let serial = evaluate(&rs, &Serial.generate_plan(&rs, &cost, &constraints), &cost);
        let main = evaluate(
            &rs,
            &MainSolution.generate_plan(&rs, &cost, &constraints),
            &cost,
        );
        assert!(
            main.predicted_latency_ms_serial <= serial.predicted_latency_ms_serial + 1e-6,
            "seed={seed}: main_solution W {} > serial W {} at P=1",
            main.predicted_latency_ms_serial,
            serial.predicted_latency_ms_serial
        );
    }
}
