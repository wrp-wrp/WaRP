//! Serial — minimises total serial work via dynamic programming.
//!
//! Scenario: the plan will be executed one merged request at a time,
//! so the wall-clock latency is `W = Σ w(merged.length)`. Serial
//! returns the plan that minimises `W` subject to an optional merged-
//! request count cap.
//!
//! Optimal among strategies that do not consider parallelism; see
//! [`crate::strategy::Greedy`] for the fast heuristic and
//! [`crate::strategy::MainSolution`] for the finite-parallel case that
//! also considers the bottleneck term.
//!
//! # Two modes
//!
//! - *Unbounded* (`constraints.max_requests == 0`): 1-D DP over prefix
//!   indices.
//!   `dp[i]` = minimum W covering the first `i` requests.
//! - *Bounded* (`constraints.max_requests > 0`): 2-D DP over
//!   `(prefix, request_count)`. Respects `constraints.exact_requests`.
//!
//! The DP is otherwise free to pick any contiguous merge pattern that
//! minimises `W` under the cost model.
//!
//! Complexity:
//!
//! - Unbounded: `O(n²)`.
//! - Bounded: `O(n² · K)`.

use crate::cost_model::CostModel;
use crate::internal::RequestPrefixSums;
use crate::io_plan::{IoPlan, IoRequest, MergedRequest};
use crate::plan_constraints::PlanConstraints;
use crate::strategy::{MergeStrategy, Separate};

/// Serial-execution planner. Minimises total work `W` via dynamic
/// programming; no strategy-private tunables.
#[derive(Debug, Clone, Copy, Default)]
pub struct Serial;

impl MergeStrategy for Serial {
    fn name(&self) -> &'static str {
        "serial"
    }

    fn supports_request_constraints(&self) -> bool {
        true
    }

    fn generate_plan(
        &self,
        requests: &[IoRequest],
        cost: &CostModel,
        constraints: &PlanConstraints,
    ) -> IoPlan {
        if requests.is_empty() {
            return IoPlan::empty();
        }

        let prefix = RequestPrefixSums::new(requests);

        let plan_cuts = if constraints.max_requests > 0 {
            run_bounded_dp(&prefix, requests, cost, constraints)
        } else {
            run_unbounded_dp(&prefix, requests, cost)
        };

        match plan_cuts {
            Some(cuts) => plan_from_cuts(&cuts, requests, &prefix),
            None => Separate.generate_plan(requests, cost, constraints),
        }
    }
}

// ---------------------------------------------------------------------
// Unbounded: dp[i] = min W to cover first i requests.
// ---------------------------------------------------------------------

fn run_unbounded_dp(
    prefix: &RequestPrefixSums,
    requests: &[IoRequest],
    cost: &CostModel,
) -> Option<Vec<usize>> {
    let n = requests.len();
    let mut dp = vec![f64::INFINITY; n + 1];
    let mut parent = vec![usize::MAX; n + 1];
    dp[0] = 0.0;

    for i in 1..=n {
        let mut best_prev = f64::INFINITY;
        let mut best_j = usize::MAX;
        for j in (0..i).rev() {
            let w = cost.request_latency_ms(prefix.span(j, i));
            let cand = dp[j] + w;
            if cand < best_prev {
                best_prev = cand;
                best_j = j;
            }
        }
        if best_j != usize::MAX {
            dp[i] = best_prev;
            parent[i] = best_j;
        }
    }

    if !dp[n].is_finite() {
        return None;
    }
    Some(backtrack(&parent, n))
}

// ---------------------------------------------------------------------
// Bounded: dp[i][k] = min W to cover first i requests using exactly k
// (or ≤ k under !exact_requests) merged requests.
// ---------------------------------------------------------------------

fn run_bounded_dp(
    prefix: &RequestPrefixSums,
    requests: &[IoRequest],
    cost: &CostModel,
    constraints: &PlanConstraints,
) -> Option<Vec<usize>> {
    let n = requests.len();
    let k_max = constraints.request_cap(n);
    // Flat 2D tables: index (i * (k_max+1) + k)
    let stride = k_max + 1;
    let mut dp = vec![f64::INFINITY; (n + 1) * stride];
    let mut parent = vec![usize::MAX; (n + 1) * stride];
    dp[0] = 0.0; // dp[0][0]

    for i in 1..=n {
        let k_cap = i.min(k_max);
        for k in 1..=k_cap {
            let mut best_prev = f64::INFINITY;
            let mut best_j = usize::MAX;
            for j in (k - 1..i).rev() {
                let prev = dp[j * stride + (k - 1)];
                if !prev.is_finite() {
                    continue;
                }
                let w = cost.request_latency_ms(prefix.span(j, i));
                let cand = prev + w;
                if cand < best_prev {
                    best_prev = cand;
                    best_j = j;
                }
            }
            if best_j != usize::MAX {
                dp[i * stride + k] = best_prev;
                parent[i * stride + k] = best_j;
            }
        }
    }

    // Pick the best terminal state under the exact/at-most policy.
    let mut best_cost = f64::INFINITY;
    let mut best_k = 0usize;
    let range: Box<dyn Iterator<Item = usize>> = if constraints.exact_requests {
        Box::new(std::iter::once(k_max))
    } else {
        Box::new(1..=k_max)
    };
    for k in range {
        let c = dp[n * stride + k];
        if c < best_cost {
            best_cost = c;
            best_k = k;
        }
    }
    if !best_cost.is_finite() {
        return None;
    }

    let mut cuts = Vec::with_capacity(best_k);
    let mut i = n;
    let mut k = best_k;
    while i > 0 && k > 0 {
        let j = parent[i * stride + k];
        cuts.push(j);
        i = j;
        k -= 1;
    }
    cuts.reverse();
    Some(cuts)
}

// ---------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------

fn backtrack(parent: &[usize], n: usize) -> Vec<usize> {
    let mut cuts = Vec::new();
    let mut i = n;
    while i > 0 {
        let j = parent[i];
        cuts.push(j);
        i = j;
    }
    cuts.reverse();
    cuts
}

/// Build an `IoPlan` from the DP's cut points.
///
/// `cuts[k]` = starting index of merged request `k`; terminator is `n`
/// (implicit).
fn plan_from_cuts(cuts: &[usize], requests: &[IoRequest], prefix: &RequestPrefixSums) -> IoPlan {
    let n = requests.len();
    let mut boundaries = Vec::with_capacity(cuts.len() + 1);
    boundaries.extend_from_slice(cuts);
    boundaries.push(n);

    let mut merged_requests = Vec::with_capacity(cuts.len());
    let mut assignments = vec![0usize; n];
    for (k, window) in boundaries.windows(2).enumerate() {
        let lo = window[0];
        let hi = window[1];
        let span = prefix.span(lo, hi);
        merged_requests.push(MergedRequest {
            offset: prefix.start_offset(lo),
            length: span,
        });
        assignments[lo..hi].fill(k);
    }
    IoPlan {
        merged_requests,
        assignments,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io_plan::evaluate;
    use crate::plan_constraints::PlanConstraints;
    use crate::strategy::{Greedy, Separate};

    fn req(id: i32, offset: i64, length: i64) -> IoRequest {
        IoRequest {
            cluster_id: id,
            offset,
            length,
        }
    }

    fn cost() -> CostModel {
        CostModel {
            // non-zero overhead so merging is strictly better than separate
            latency_per_request: 10.0,
            latency_per_byte: 0.0001,
            ..CostModel::default()
        }
    }

    #[test]
    fn empty_input_gives_empty_plan() {
        let plan = Serial.generate_plan(&[], &CostModel::default(), &PlanConstraints::default());
        assert!(plan.merged_requests.is_empty());
    }

    #[test]
    fn adjacent_clusters_are_merged() {
        let rs = vec![req(0, 0, 100), req(1, 100, 100)];
        let plan = Serial.generate_plan(&rs, &cost(), &PlanConstraints::default());
        assert_eq!(plan.merged_requests.len(), 1);
        plan.check_invariants(&rs).unwrap();
    }

    #[test]
    fn dp_beats_or_ties_greedy_and_separate() {
        // Construct a mixed-gap input where greedy is suboptimal: two
        // small clusters, large-but-tolerable gap, two small clusters.
        let rs = vec![
            req(0, 0, 100),
            req(1, 100, 100),
            req(2, 800, 100),
            req(3, 900, 100),
        ];
        let cost = cost();
        let constraints = PlanConstraints::default();
        let dp_plan = Serial.generate_plan(&rs, &cost, &constraints);
        let greedy_plan = Greedy::default().generate_plan(&rs, &cost, &constraints);
        let sep_plan = Separate.generate_plan(&rs, &cost, &constraints);

        let dp_m = evaluate(&rs, &dp_plan, &cost);
        let greedy_m = evaluate(&rs, &greedy_plan, &cost);
        let sep_m = evaluate(&rs, &sep_plan, &cost);

        dp_plan.check_invariants(&rs).unwrap();
        assert!(dp_m.predicted_latency_ms_serial <= greedy_m.predicted_latency_ms_serial + 1e-9);
        assert!(dp_m.predicted_latency_ms_serial <= sep_m.predicted_latency_ms_serial + 1e-9);
    }

    #[test]
    fn bounded_dp_respects_max_requests() {
        let rs = vec![req(0, 0, 100), req(1, 200, 100), req(2, 400, 100)];
        let constraints = PlanConstraints {
            max_requests: 2,
            exact_requests: false,
        };
        let plan = Serial.generate_plan(&rs, &cost(), &constraints);
        assert!(plan.merged_requests.len() <= 2);
        plan.check_invariants(&rs).unwrap();
    }

    #[test]
    fn bounded_exact_requests_enforces_count() {
        let rs = vec![req(0, 0, 100), req(1, 200, 100), req(2, 400, 100)];
        let constraints = PlanConstraints {
            max_requests: 2,
            exact_requests: true,
        };
        let plan = Serial.generate_plan(&rs, &cost(), &constraints);
        assert_eq!(plan.merged_requests.len(), 2);
    }

    #[test]
    fn no_request_cap_merges_when_beneficial() {
        let rs = vec![req(0, 0, 100), req(1, 200, 100)];
        let plan = Serial.generate_plan(&rs, &cost(), &PlanConstraints::default());
        assert_eq!(plan.merged_requests.len(), 1);
    }
}
