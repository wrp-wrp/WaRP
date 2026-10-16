//! Differential fuzz tests: every planner vs brute-force enumeration.
//!
//! Complements `main_solution_optimality.rs` (which pins a few seeds at
//! P = 4 with default constraints) by sweeping the config axes that
//! select different internal solver paths:
//!
//! - parallelism P ∈ {0 (bottleneck), 1 (serial), 2, 4, 1000}
//! - max_requests ∈ {0 (uncapped), 1, 2, tight, n} × exact_requests
//! - cost shapes: linear, zero-slope, piecewise-convex (DC / WQS),
//!   piecewise-NON-convex (PlainDP fallback)
//! - adjacency extremes: zero gaps, huge gaps, length-1 requests
//!
//! Brute force enumerates all `2^(n-1)` contiguous partitions and
//! evaluates them with the public `CostModel::request_latency_ms`, so
//! any planner/evaluator divergence (e.g. in the duplicated piecewise
//! continuity stitching) also fails these tests.

use rand::{rngs::StdRng, Rng, SeedableRng};
use warp_planner::{
    evaluate,
    strategy::{Bottleneck, MainSolution, MergeStrategy, Serial},
    CostModel, IoRequest, PlanConstraints,
};

/// Mirror of MainSolution's `compute_objective_coefficients`.
fn coeffs(p: f64) -> (f64, f64) {
    if p <= 1e-9 {
        return (0.0, 1.0);
    }
    let p = p.max(1.0);
    (1.0 / p, 1.0 - 1.0 / p)
}

struct BruteBest {
    t_ub: f64,
    w: f64,
    b: f64,
}

/// Enumerate all contiguous partitions satisfying the (clamped)
/// constraints; return the minimum T_UB and the min-W / min-B values.
fn brute_force(
    requests: &[IoRequest],
    cost: &CostModel,
    constraints: &PlanConstraints,
) -> BruteBest {
    let n = requests.len();
    let (cw, cb) = coeffs(cost.concurrent_parallelism);
    // Planner semantics: the cap is clamped to [1, n].
    let target = if constraints.max_requests > 0 {
        (constraints.max_requests as usize).clamp(1, n)
    } else {
        n
    };
    let mut best = BruteBest {
        t_ub: f64::INFINITY,
        w: f64::INFINITY,
        b: f64::INFINITY,
    };
    for mask in 0u32..(1u32 << (n - 1)) {
        let groups = 1 + mask.count_ones() as usize;
        if constraints.max_requests > 0 {
            if constraints.exact_requests {
                if groups != target {
                    continue;
                }
            } else if groups > target {
                continue;
            }
        }
        let mut w_sum = 0.0f64;
        let mut b_max = 0.0f64;
        let mut lo = 0usize;
        for i in 0..n {
            let is_cut = i + 1 == n || (mask >> i) & 1 == 1;
            if is_cut {
                let span = requests[i].end() - requests[lo].offset;
                let w = cost.request_latency_ms(span);
                w_sum += w;
                b_max = b_max.max(w);
                lo = i + 1;
            }
        }
        let t = cw * w_sum + cb * b_max;
        if t < best.t_ub {
            best.t_ub = t;
        }
        if w_sum < best.w {
            best.w = w_sum;
        }
        if b_max < best.b {
            best.b = b_max;
        }
    }
    best
}

fn t_ub_of(
    requests: &[IoRequest],
    plan: &warp_planner::IoPlan,
    cost: &CostModel,
) -> (f64, f64, f64) {
    let m = evaluate(requests, plan, cost);
    let (cw, cb) = coeffs(cost.concurrent_parallelism);
    (
        cw * m.predicted_latency_ms_serial + cb * m.predicted_latency_ms_concurrent,
        m.predicted_latency_ms_serial,
        m.predicted_latency_ms_concurrent,
    )
}

fn gen_requests(rng: &mut StdRng, n: usize, adjacency: u8) -> Vec<IoRequest> {
    let mut offset = 0i64;
    (0..n as i32)
        .map(|i| {
            let length = match adjacency {
                0 => 1 + rng.gen_range(0..8),          // tiny
                1 => 50 + rng.gen_range(0..500),       // mid
                _ => 1024 + rng.gen_range(0..100_000), // large
            };
            let r = IoRequest {
                cluster_id: i,
                offset,
                length,
            };
            let gap = match adjacency {
                0 => 0, // fully adjacent
                1 => rng.gen_range(0..5_000),
                _ => rng.gen_range(0..2_000_000), // huge gaps
            };
            offset = r.end() + gap;
            r
        })
        .collect()
}

fn gen_cost(rng: &mut StdRng, shape: u8, p: f64) -> CostModel {
    match shape {
        // linear, random overhead/rate
        0 => CostModel {
            latency_per_request: 1.0 + rng.gen_range(0.0..50.0),
            latency_per_byte: rng.gen_range(0.0..0.01),
            concurrent_parallelism: p,
            ..CostModel::default()
        },
        // zero slope
        1 => CostModel {
            latency_per_request: 1.0 + rng.gen_range(0.0..50.0),
            latency_per_byte: 0.0,
            concurrent_parallelism: p,
            ..CostModel::default()
        },
        // piecewise convex (rates non-decreasing) → DC / WQS paths
        2 => {
            let r0 = rng.gen_range(0.0..1e-4);
            let r1 = r0 + rng.gen_range(0.0..1e-3);
            CostModel {
                latency_model: "piecewise_linear".into(),
                latency_pw_breaks_bytes: vec![1 + rng.gen_range(100..500_000)],
                latency_pw_overheads_ms: vec![
                    1.0 + rng.gen_range(0.0..40.0),
                    rng.gen_range(0.0..40.0),
                ],
                latency_pw_rates_ms_per_byte: vec![r0, r1],
                concurrent_parallelism: p,
                ..CostModel::default()
            }
        }
        // piecewise NON-convex (rates decreasing) → PlainDP fallback
        _ => {
            let r1 = rng.gen_range(0.0..1e-4);
            let r0 = r1 + rng.gen_range(1e-5..1e-3);
            CostModel {
                latency_model: "piecewise_linear".into(),
                latency_pw_breaks_bytes: vec![1 + rng.gen_range(100..500_000)],
                latency_pw_overheads_ms: vec![
                    1.0 + rng.gen_range(0.0..40.0),
                    rng.gen_range(0.0..40.0),
                ],
                latency_pw_rates_ms_per_byte: vec![r0, r1],
                concurrent_parallelism: p,
                ..CostModel::default()
            }
        }
    }
}

const PARALLELISMS: [f64; 5] = [0.0, 1.0, 2.0, 4.0, 1000.0];

#[test]
fn main_solution_fuzz_vs_brute_force() {
    let mut checked = 0usize;
    for seed in 0..8000u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let n = rng.gen_range(1..=9usize);
        let adjacency = (seed % 3) as u8;
        let shape = (seed % 4) as u8;
        let p = PARALLELISMS[seed as usize % PARALLELISMS.len()];
        let rs = gen_requests(&mut rng, n, adjacency);
        let cost = gen_cost(&mut rng, shape, p);

        for (max_requests, exact) in [
            (0i32, false),
            (1, false),
            (2, false),
            (n as i32 / 2, false),
            (n as i32, false),
            (2, true),
            (n as i32 / 2, true),
        ] {
            if max_requests < 0 || (exact && max_requests == 0) {
                continue;
            }
            let constraints = PlanConstraints {
                max_requests,
                exact_requests: exact,
            };
            let plan = MainSolution.generate_plan(&rs, &cost, &constraints);
            plan.check_invariants(&rs)
                .unwrap_or_else(|e| panic!("seed={seed} n={n} shape={shape} p={p}: invariant {e}"));
            // Constraint compliance (the planner clamps the cap to [1, n]).
            if max_requests > 0 {
                let cap = (max_requests as usize).clamp(1, n);
                assert!(
                    plan.merged_requests.len() <= cap,
                    "seed={seed}: {} requests > cap {cap}",
                    plan.merged_requests.len()
                );
                if exact {
                    assert_eq!(
                        plan.merged_requests.len(),
                        cap,
                        "seed={seed} n={n} shape={shape} p={p}: exact_requests violated"
                    );
                }
            }
            let (got, _, _) = t_ub_of(&rs, &plan, &cost);
            let brute = brute_force(&rs, &cost, &constraints);
            // Sanity: cannot beat the true optimum.
            assert!(
                got >= brute.t_ub - 1e-6,
                "seed={seed} n={n} shape={shape} p={p} cap={max_requests} exact={exact}: \
                 planner {got} beats brute optimum {} — evaluator/planner divergence?",
                brute.t_ub
            );
            // n ≤ 6 keeps candidate count < ADAPTIVE_MIN_CANDIDATES=24 →
            // dense + exact search; larger n may invoke the documented
            // 5 % adaptive-hull tolerance.
            let tol = if n <= 6 {
                1e-6
            } else {
                brute.t_ub * 0.05 + 1e-6
            };
            assert!(
                got <= brute.t_ub + tol,
                "seed={seed} n={n} shape={shape} p={p} cap={max_requests} exact={exact}: \
                 planner T_UB {got} > brute optimum {} + tol {tol}",
                brute.t_ub
            );
            checked += 1;
        }
    }
    assert!(checked > 20000, "only {checked} cases checked");
}

#[test]
fn serial_fuzz_vs_brute_force_min_w() {
    for seed in 0..3000u64 {
        let mut rng = StdRng::seed_from_u64(1_000_000 + seed);
        let n = rng.gen_range(1..=9usize);
        let rs = gen_requests(&mut rng, n, (seed % 3) as u8);
        let cost = gen_cost(&mut rng, (seed % 4) as u8, 1.0);
        for (max_requests, exact) in [(0i32, false), (2, false), (n as i32 / 2, true)] {
            if exact && max_requests == 0 {
                continue;
            }
            let constraints = PlanConstraints {
                max_requests,
                exact_requests: exact,
            };
            let plan = Serial.generate_plan(&rs, &cost, &constraints);
            plan.check_invariants(&rs).unwrap();
            let m = evaluate(&rs, &plan, &cost);
            let brute = brute_force(&rs, &cost, &constraints);
            assert!(
                (m.predicted_latency_ms_serial - brute.w).abs() <= 1e-6,
                "seed={seed} n={n} cap={max_requests} exact={exact}: \
                 Serial W {} != brute min-W {}",
                m.predicted_latency_ms_serial,
                brute.w
            );
        }
    }
}

#[test]
fn bottleneck_fuzz_vs_brute_force_min_b() {
    for seed in 0..3000u64 {
        let mut rng = StdRng::seed_from_u64(2_000_000 + seed);
        let n = rng.gen_range(1..=9usize);
        let rs = gen_requests(&mut rng, n, (seed % 3) as u8);
        let cost = gen_cost(&mut rng, (seed % 4) as u8, 0.0);
        for (max_requests, exact) in [(1i32, false), (2, false), (n as i32 / 2, false), (2, true)] {
            if max_requests <= 0 {
                continue;
            }
            let constraints = PlanConstraints {
                max_requests,
                exact_requests: exact,
            };
            let plan = Bottleneck.generate_plan(&rs, &cost, &constraints);
            plan.check_invariants(&rs).unwrap();
            let m = evaluate(&rs, &plan, &cost);
            let brute = brute_force(&rs, &cost, &constraints);
            // Binary-search TOL is 1e-6 absolute; allow a hair more.
            assert!(
                m.predicted_latency_ms_concurrent <= brute.b + 1e-4,
                "seed={seed} n={n} cap={max_requests} exact={exact}: \
                 Bottleneck B {} > brute min-B {} + 1e-4",
                m.predicted_latency_ms_concurrent,
                brute.b
            );
        }
    }
}

/// Greedy policy compliance: inside a merged group every bridged gap is
/// ≤ min_gap and the span respects the cap; across a group boundary the
/// join must have been impossible.
#[test]
fn greedy_fuzz_policy_compliance() {
    use warp_planner::strategy::Greedy;
    for seed in 0..3000u64 {
        let mut rng = StdRng::seed_from_u64(3_000_000 + seed);
        let n = rng.gen_range(1..=12usize);
        let rs = gen_requests(&mut rng, n, (seed % 3) as u8);
        let min_gap = rng.gen_range(0..1_000_000i64);
        let max_span = if seed % 2 == 0 {
            0
        } else {
            rng.gen_range(1_000..3_000_000i64)
        };
        let g = Greedy::new(min_gap, max_span);
        let plan = g.generate_plan(&rs, &CostModel::default(), &PlanConstraints::default());
        plan.check_invariants(&rs).unwrap();
        for (i, win) in rs.windows(2).enumerate() {
            let same_group = plan.assignments[i] == plan.assignments[i + 1];
            let gap = win[1].offset - win[0].end();
            if same_group {
                assert!(
                    gap <= min_gap,
                    "seed={seed}: merged across gap {gap} > {min_gap}"
                );
            }
        }
        if max_span > 0 {
            // The cap constrains *merging* only: a single request longer
            // than the cap forms its own (legal) over-cap group, so the
            // assertion applies to multi-member groups alone.
            let mut members = vec![0usize; plan.merged_requests.len()];
            for &a in &plan.assignments {
                members[a] += 1;
            }
            for (gi, m) in plan.merged_requests.iter().enumerate() {
                if members[gi] >= 2 {
                    assert!(
                        m.length <= max_span,
                        "seed={seed}: merged span {} > cap {max_span}",
                        m.length
                    );
                }
            }
        }
    }
}

/// Large-n smoke: exercises the sparse candidate enumeration +
/// adaptive-hull path that small-n brute-force tests never reach.
/// Asserts invariants, constraint compliance, and that the planner
/// beats Separate — plus a loose wall-clock sanity bound.
#[test]
fn main_solution_large_n_smoke() {
    use std::time::Instant;
    use warp_planner::strategy::Separate;
    let mut rng = StdRng::seed_from_u64(42);
    let rs = gen_requests(&mut rng, 2000, 2);
    for shape in [0u8, 2, 3] {
        let cost = gen_cost(&mut rng, shape, 16.0);
        for max_requests in [16i32, 0] {
            let constraints = PlanConstraints {
                max_requests,
                exact_requests: false,
            };
            let t0 = Instant::now();
            let plan = MainSolution.generate_plan(&rs, &cost, &constraints);
            let dt = t0.elapsed();
            plan.check_invariants(&rs).unwrap();
            if max_requests > 0 {
                assert!(plan.merged_requests.len() <= max_requests as usize);
            }
            let (got, _, _) = t_ub_of(&rs, &plan, &cost);
            let sep = Separate.generate_plan(&rs, &cost, &constraints);
            let (sep_t, _, _) = t_ub_of(&rs, &sep, &cost);
            if max_requests == 0 {
                assert!(
                    got <= sep_t + 1e-6,
                    "shape={shape}: {got} > separate {sep_t}"
                );
            }
            assert!(
                dt.as_millis() < 30_000,
                "shape={shape} cap={max_requests}: plan took {dt:?}"
            );
            eprintln!("n=2000 shape={shape} cap={max_requests}: T_UB {got:.2} ms (sep {sep_t:.2}), plan time {dt:?}");
        }
    }
}
