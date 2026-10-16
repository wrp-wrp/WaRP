//! Bottleneck — minimises the slowest merged request.
//!
//! Scenario: the plan will be issued with enough worker parallelism
//! that every merged request fires simultaneously, so the plan's
//! wall-clock latency degenerates to
//!
//! ```text
//!     T_bottleneck = max over merged requests m of w(m.length)
//! ```
//!
//! i.e. the slowest single request dominates. Minimising this bottleneck
//! under a request-count cap `K` is a classic "min-max partition" problem
//! on a contiguous sequence, solvable by binary search.
//!
//! # Algorithm
//!
//! Binary-search the bottleneck value `B` over the range
//! `[max(w(single)), w(merge_everything)]`. For each candidate `B`:
//!
//! 1. Greedily pack the sorted requests into the fewest possible
//!    contiguous groups such that each group's `w(len(group))` ≤ `B`
//!    .
//! 2. If the resulting group count is ≤ `K`, `B` is feasible; shrink the
//!    upper half. Otherwise grow the lower half.
//!
//! Returns the plan produced by the smallest feasible `B`.
//!
//! Complexity: `O(n · log B_range)` where `B_range` is the span of
//! latency values (tiny in practice).

use crate::cost_model::CostModel;
use crate::internal::RequestPrefixSums;
use crate::io_plan::{IoPlan, IoRequest, MergedRequest};
use crate::plan_constraints::PlanConstraints;
use crate::strategy::{MergeStrategy, Separate};

/// Bottleneck planner — minimises `max w(len(group))` under the cost
/// problem's `max_requests` cap. If the cap is unset, degenerates into
/// Separate (single-cluster groups always give the smallest bottleneck).
#[derive(Debug, Clone, Copy, Default)]
pub struct Bottleneck;

impl MergeStrategy for Bottleneck {
    fn name(&self) -> &'static str {
        "bottleneck"
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
        let n = requests.len();
        if n == 0 {
            return IoPlan::empty();
        }
        let k_cap = constraints.request_cap(n);
        if k_cap >= n {
            // No meaningful cap → Separate is bottleneck-optimal.
            return Separate.generate_plan(requests, cost, constraints);
        }

        let prefix = RequestPrefixSums::new(requests);

        // Latency range to binary-search over.
        let lower = bottleneck_lower_bound(requests, cost);
        let upper = cost.request_latency_ms(prefix.span(0, n));

        let (_final_b, cuts) = match binary_search_bottleneck(&prefix, cost, k_cap, lower, upper) {
            Some(found) => found,
            None => return Separate.generate_plan(requests, cost, constraints),
        };

        let cuts = if constraints.exact_requests && cuts.len() != k_cap {
            // The minimum feasible bottleneck may admit fewer than K
            // groups. Splitting any merged group at a cluster boundary
            // only shortens spans, so the bottleneck and hard
            // constraints remain satisfied; refine the cut list until
            // it uses exactly K groups.
            match refine_cuts_to_exact(&cuts, n, k_cap) {
                Some(refined) => refined,
                None => return Separate.generate_plan(requests, cost, constraints),
            }
        } else {
            cuts
        };

        plan_from_cuts(&cuts, requests, &prefix)
    }
}

/// Lower bound on the bottleneck: the max of per-cluster latencies. No
/// plan can do better than this (a single cluster's latency is a lower
/// bound on any group containing it).
fn bottleneck_lower_bound(requests: &[IoRequest], cost: &CostModel) -> f64 {
    requests
        .iter()
        .map(|r| cost.request_latency_ms(r.length))
        .fold(0.0f64, f64::max)
}

/// Binary-search the feasible `B`. Returns `(B, cuts)` where `cuts[k]`
/// is the starting index of group `k` and a terminator `n` is implicit.
fn binary_search_bottleneck(
    prefix: &RequestPrefixSums,
    cost: &CostModel,
    k_cap: usize,
    mut lo: f64,
    mut hi: f64,
) -> Option<(f64, Vec<usize>)> {
    const TOL: f64 = 1e-6;
    const MAX_ITERS: usize = 64;

    let mut best_cuts: Option<Vec<usize>> = None;

    pack_within_bottleneck(prefix, cost, hi, k_cap)?;

    for _ in 0..MAX_ITERS {
        if (hi - lo).abs() <= TOL {
            break;
        }
        let mid = 0.5 * (lo + hi);
        match pack_within_bottleneck(prefix, cost, mid, k_cap) {
            Some(cuts) => {
                hi = mid;
                best_cuts = Some(cuts);
            }
            None => {
                lo = mid;
            }
        }
    }

    let cuts = best_cuts.or_else(|| pack_within_bottleneck(prefix, cost, hi, k_cap))?;
    Some((hi, cuts))
}

/// Greedy packer: left-to-right, extend the current group as long as the
/// resulting `w(len)` stays ≤ `b`. Start a new group when we can't
/// extend. Returns the cut points if the
/// resulting group count ≤ `k_cap`, else `None`.
fn pack_within_bottleneck(
    prefix: &RequestPrefixSums,
    cost: &CostModel,
    b: f64,
    k_cap: usize,
) -> Option<Vec<usize>> {
    let n = prefix.len();
    let mut cuts = Vec::new();
    let mut lo = 0usize;
    while lo < n {
        cuts.push(lo);
        // Binary search the largest `hi` such that the group `[lo, hi)`
        // stays within the constraints.
        let mut left = lo + 1;
        let mut right = n;
        while left < right {
            let mid = left + (right - left).div_ceil(2);
            if group_feasible(prefix, cost, lo, mid, b) {
                left = mid;
            } else {
                right = mid - 1;
            }
            if left == right {
                break;
            }
        }
        let hi = if group_feasible(prefix, cost, lo, left, b) {
            left
        } else {
            lo + 1
        };
        // Safety net: every group must contain at least one cluster even
        // if a single cluster already violates `b`. Callers guard against
        // that case via `bottleneck_lower_bound`.
        lo = hi.max(lo + 1);
        if cuts.len() > k_cap {
            return None;
        }
    }
    Some(cuts)
}

fn refine_cuts_to_exact(cuts: &[usize], n: usize, target_groups: usize) -> Option<Vec<usize>> {
    if cuts.is_empty() || cuts.first() != Some(&0) {
        return None;
    }
    if target_groups < cuts.len() || target_groups > n {
        return None;
    }

    let mut refined = cuts.to_vec();
    while refined.len() < target_groups {
        let mut split_any_group = false;
        let mut group = 0usize;
        while group < refined.len() && refined.len() < target_groups {
            let lo = refined[group];
            let hi = refined.get(group + 1).copied().unwrap_or(n);
            if hi > lo + 1 {
                refined.insert(group + 1, lo + 1);
                split_any_group = true;
                group += 2;
            } else {
                group += 1;
            }
        }
        if !split_any_group {
            return None;
        }
    }
    Some(refined)
}

#[inline]
fn group_feasible(
    prefix: &RequestPrefixSums,
    cost: &CostModel,
    lo: usize,
    hi: usize,
    b: f64,
) -> bool {
    if hi <= lo {
        return false;
    }
    let span = prefix.span(lo, hi);
    cost.request_latency_ms(span) <= b + 1e-9
}

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

    fn req(id: i32, offset: i64, length: i64) -> IoRequest {
        IoRequest {
            cluster_id: id,
            offset,
            length,
        }
    }

    fn cost() -> CostModel {
        CostModel {
            latency_per_request: 10.0,
            latency_per_byte: 0.001,
            ..CostModel::default()
        }
    }

    #[test]
    fn empty_input_gives_empty_plan() {
        let constraints = PlanConstraints {
            max_requests: 2,
            exact_requests: false,
        };
        let plan = Bottleneck.generate_plan(&[], &cost(), &constraints);
        assert!(plan.merged_requests.is_empty());
    }

    #[test]
    fn no_cap_degenerates_to_separate() {
        let rs = vec![req(0, 0, 100), req(1, 200, 100)];
        let cost = cost();
        let constraints = PlanConstraints::default();
        let concurrent = Bottleneck.generate_plan(&rs, &cost, &constraints);
        let separate = Separate.generate_plan(&rs, &cost, &constraints);
        assert_eq!(concurrent, separate);
    }

    #[test]
    fn capped_plan_respects_request_count() {
        let rs = (0..10)
            .map(|i| req(i as i32, i * 1_000, 100))
            .collect::<Vec<_>>();
        let constraints = PlanConstraints {
            max_requests: 3,
            exact_requests: false,
        };
        let plan = Bottleneck.generate_plan(&rs, &cost(), &constraints);
        assert!(plan.merged_requests.len() <= 3);
        plan.check_invariants(&rs).unwrap();
    }

    #[test]
    fn bottleneck_is_finite_under_cap() {
        // Ten clusters evenly spaced; under a cap of 3 we must group,
        // but the bottleneck latency should still be less than the sum
        // of all separate latencies.
        let rs = (0..10)
            .map(|i| req(i as i32, i * 1_000, 100))
            .collect::<Vec<_>>();
        let cost = cost();
        let constraints = PlanConstraints {
            max_requests: 3,
            exact_requests: false,
        };
        let concurrent_plan = Bottleneck.generate_plan(&rs, &cost, &constraints);
        let concurrent_m = evaluate(&rs, &concurrent_plan, &cost);

        // Bottleneck latency should be positive and finite.
        assert!(concurrent_m.predicted_latency_ms_concurrent > 0.0);
        assert!(concurrent_m.predicted_latency_ms_concurrent.is_finite());
    }

    #[test]
    fn bottleneck_grows_monotonically_with_tighter_cap() {
        let rs = (0..10)
            .map(|i| req(i as i32, i * 1_000, 100))
            .collect::<Vec<_>>();
        let cost = cost();
        let loose_constraints = PlanConstraints {
            max_requests: 5,
            exact_requests: false,
        };
        let tight_constraints = PlanConstraints {
            max_requests: 2,
            exact_requests: false,
        };
        let m_loose = evaluate(
            &rs,
            &Bottleneck.generate_plan(&rs, &cost, &loose_constraints),
            &cost,
        );
        let m_tight = evaluate(
            &rs,
            &Bottleneck.generate_plan(&rs, &cost, &tight_constraints),
            &cost,
        );
        assert!(m_tight.predicted_latency_ms_concurrent >= m_loose.predicted_latency_ms_concurrent);
    }

    #[test]
    fn exact_requests_refines_plan_instead_of_falling_back_to_separate() {
        let rs = vec![
            req(0, 0, 100),
            req(1, 100, 100),
            req(2, 200, 100),
            req(3, 300, 100),
        ];
        let cost = CostModel {
            latency_per_request: 10.0,
            latency_per_byte: 0.001,
            ..CostModel::default()
        };
        let constraints = PlanConstraints {
            max_requests: 3,
            exact_requests: true,
        };
        let plan = Bottleneck.generate_plan(&rs, &cost, &constraints);
        assert_eq!(plan.merged_requests.len(), 3);
        plan.check_invariants(&rs).unwrap();
    }
}
