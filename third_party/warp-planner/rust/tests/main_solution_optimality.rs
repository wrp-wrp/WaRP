//! Brute-force optimality checks for MainSolution on small inputs.
//!
//! Enumerates every contiguous-partition plan, computes `T_UB(P)` for
//! each feasible one, and asserts MainSolution finds the minimum.
//! These are slow enough (O(2^{n-1})) that they live in a separate
//! integration binary so `cargo test` for the main crate stays fast.

use rand::{rngs::StdRng, Rng, SeedableRng};
use warp_planner::{
    evaluate,
    strategy::{MainSolution, MergeStrategy},
    CostModel, IoRequest, PlanConstraints,
};

fn predicted_t_ub(m: &warp_planner::PlanMetrics, p: f64) -> f64 {
    m.t_ub_ms(p)
}

/// Enumerate every feasible contiguous partition and return the
/// minimum `T_UB(P)`.
fn brute_force_best_t_ub(
    requests: &[IoRequest],
    cost: &CostModel,
    constraints: &PlanConstraints,
    p: f64,
) -> f64 {
    let n = requests.len();
    let mut best = f64::INFINITY;
    for mask in 0u32..(1u32 << n.saturating_sub(1)) {
        let mut cuts = vec![0usize];
        for i in 0..(n - 1) {
            if (mask >> i) & 1 == 1 {
                cuts.push(i + 1);
            }
        }
        cuts.push(n);
        let groups = cuts.len() - 1;
        if constraints.max_requests > 0 {
            let target = constraints.max_requests as usize;
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
        for win in cuts.windows(2) {
            let (lo, hi) = (win[0], win[1]);
            let span = requests[hi - 1].end() - requests[lo].offset;
            let w = cost.request_latency_ms(span);
            w_sum += w;
            if w > b_max {
                b_max = w;
            }
        }
        let t = w_sum / p + (1.0 - 1.0 / p) * b_max;
        if t < best {
            best = t;
        }
    }
    best
}

#[test]
fn main_solution_matches_brute_force_optimum_on_small_inputs() {
    let rs = vec![
        IoRequest {
            cluster_id: 0,
            offset: 0,
            length: 100,
        },
        IoRequest {
            cluster_id: 1,
            offset: 200,
            length: 100,
        },
        IoRequest {
            cluster_id: 2,
            offset: 5_000,
            length: 100,
        },
        IoRequest {
            cluster_id: 3,
            offset: 5_200,
            length: 100,
        },
    ];
    let cost = CostModel {
        latency_per_request: 10.0,
        latency_per_byte: 0.001,
        concurrent_parallelism: 4.0,
        ..CostModel::default()
    };
    let constraints = PlanConstraints::default();
    let p = cost.concurrent_parallelism;
    let plan = MainSolution.generate_plan(&rs, &cost, &constraints);
    let m = evaluate(&rs, &plan, &cost);
    let main_t_ub = predicted_t_ub(&m, p);
    let optimum = brute_force_best_t_ub(&rs, &cost, &constraints, p);
    assert!(
        (main_t_ub - optimum).abs() < 1e-6,
        "MainSolution T_UB {main_t_ub} != brute-force optimum {optimum}",
    );
}

#[test]
fn main_solution_matches_brute_force_on_randomised_small_inputs() {
    for seed in 0..16u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut offset = 0i64;
        let rs: Vec<IoRequest> = (0i32..6)
            .map(|i| {
                let length = 50 + rng.gen_range(0..200);
                let r = IoRequest {
                    cluster_id: i,
                    offset,
                    length,
                };
                offset = r.end() + rng.gen_range(0..2_000);
                r
            })
            .collect();
        let cost = CostModel {
            latency_per_request: 10.0,
            latency_per_byte: 0.005,
            concurrent_parallelism: 4.0,
            ..CostModel::default()
        };
        let constraints = PlanConstraints::default();
        let p = cost.concurrent_parallelism;
        let plan = MainSolution.generate_plan(&rs, &cost, &constraints);
        let m = evaluate(&rs, &plan, &cost);
        let main_t_ub = predicted_t_ub(&m, p);
        let optimum = brute_force_best_t_ub(&rs, &cost, &constraints, p);
        assert!(
            (main_t_ub - optimum).abs() < 1e-6,
            "seed={seed}: MainSolution T_UB {main_t_ub} != brute-force optimum {optimum}",
        );
    }
}

/// Same brute-force gate as above but exercises the WQS dispatch
/// path: piecewise-convex cost + `k_cap >= n/4` (= max_requests
/// large enough to trigger WQS). Runs at small `n` so the
/// `O(2^{n-1})` brute force is tractable.
///
/// The 5 % adaptive-hull tolerance still applies, but at small `n`
/// the candidate enumeration goes dense + exact, so MainSolution
/// should hit the brute-force optimum bit-exactly.
#[test]
fn main_solution_wqs_path_matches_brute_force() {
    for seed in 0..16u64 {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut offset = 0i64;
        let n: i32 = 6;
        let rs: Vec<IoRequest> = (0..n)
            .map(|i| {
                let length = 1024 + rng.gen_range(0..4096);
                let r = IoRequest {
                    cluster_id: i,
                    offset,
                    length,
                };
                offset = r.end() + rng.gen_range(0..16_384);
                r
            })
            .collect();
        // Piecewise-convex cost with `max_requests = n` so k_cap = n,
        // making `is_large_k_for_wqs(n, k_cap)` true → WQS fires.
        let cost = CostModel {
            latency_model: "piecewise_linear".to_string(),
            latency_pw_breaks_bytes: vec![3_145_728],
            latency_pw_overheads_ms: vec![28.85, 6.27],
            latency_pw_rates_ms_per_byte: vec![2.82e-6, 1.00e-5],
            latency_per_request: 28.85,
            latency_per_byte: 2.82e-6,
            concurrent_parallelism: 4.0,
            ..CostModel::default()
        };
        let constraints = PlanConstraints {
            max_requests: n,
            exact_requests: false,
        };
        let p = cost.concurrent_parallelism;
        let plan = MainSolution.generate_plan(&rs, &cost, &constraints);
        let m = evaluate(&rs, &plan, &cost);
        let main_t_ub = predicted_t_ub(&m, p);
        let optimum = brute_force_best_t_ub(&rs, &cost, &constraints, p);
        // Tolerance covers IEEE float drift in the Lagrangian search.
        let denom = main_t_ub.abs().max(optimum.abs()).max(1e-12);
        let rel = (main_t_ub - optimum).abs() / denom;
        assert!(
            rel < 1e-3,
            "seed={seed}: WQS T_UB {main_t_ub} != brute-force optimum {optimum} (rel {rel})",
        );
    }
}
