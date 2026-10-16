//! Greedy — gap-threshold merge heuristic (paper's "Greedy Oracle").
//!
//! Walks sorted requests left to right; merges consecutive clusters
//! iff both:
//!
//! 1. the byte gap between them is `≤ min_gap_bytes`, and
//! 2. the resulting merged span is `≤ max_span_bytes`
//!    (or `max_span_bytes == 0`, meaning "no cap").
//!
//! This is the baseline the paper's evaluation compares against. See
//! `sections/evaluation.tex`:
//!
//! > *"It first sorts all raw ranges by file offset, and then merges
//! > adjacent requests when both of the following conditions are
//! > satisfied: (i) the gap between them is no larger than a tunable
//! > threshold `min_gap`, and (ii) the span of the merged request is
//! > no larger than a tunable cap `max_span`. We sweep `min_gap` over
//! > {1, 2, 4, 8, 16, 32} MB."*
//!
//! Why gap-based and not waste-ratio-based: the right decision at a
//! gap is `cost(merge) < cost(two separate requests)`, which for a
//! linear cost expands to `rate · gap_bytes < overhead` — i.e. a
//! pure **absolute byte threshold**, independent of cluster size.
//! Waste-ratio heuristics fail on heterogeneous clusters (typical of
//! IVF layouts) because ratio depends on cluster size rather than on
//! the actual cost of bridging the gap.
//!
//! Complexity: `O(n)` time, `O(n)` plan-construction space.

use crate::cost_model::CostModel;
use crate::io_plan::{IoPlan, IoRequest, MergedRequest};
use crate::plan_constraints::PlanConstraints;
use crate::strategy::MergeStrategy;

/// Gap-threshold greedy planner.
///
/// Parameters are per-planner (on the struct), not on the cost
/// model. The cost model describes the storage endpoint; the planner
/// parameters describe *how* you want to coalesce for it.
#[derive(Debug, Clone, Copy)]
pub struct Greedy {
    /// Maximum byte gap to coalesce across. The paper sweeps this
    /// over `{1, 2, 4, 8, 16, 32}` MiB to build the "Greedy Oracle"
    /// frontier. `0` means "merge only fully adjacent requests"
    /// (`gap == 0`).
    pub min_gap_bytes: i64,
    /// Optional cap on merged-request span (bytes). `0` disables
    /// the cap. Mirrors the paper Greedy Oracle's `max_span` knob.
    pub max_span_bytes: i64,
}

impl Default for Greedy {
    /// Paper-sweep mid-point + a generous span cap. Reasonable for
    /// same-region S3 where the `overhead / rate` break-even is
    /// around 3-10 MiB depending on the cost segment.
    fn default() -> Self {
        Self {
            min_gap_bytes: 4 * 1024 * 1024,   // 4 MiB
            max_span_bytes: 10 * 1024 * 1024, // 10 MiB
        }
    }
}

impl Greedy {
    /// Construct with an explicit `(min_gap_bytes, max_span_bytes)`.
    /// `max_span_bytes = 0` disables the span cap.
    pub fn new(min_gap_bytes: i64, max_span_bytes: i64) -> Self {
        Self {
            min_gap_bytes,
            max_span_bytes,
        }
    }
}

impl MergeStrategy for Greedy {
    fn name(&self) -> &'static str {
        "greedy"
    }

    fn generate_plan(
        &self,
        requests: &[IoRequest],
        _cost: &CostModel,
        _constraints: &PlanConstraints,
    ) -> IoPlan {
        if requests.is_empty() {
            return IoPlan::empty();
        }

        let mut run = RunState::start(requests[0]);
        let mut merged_requests: Vec<MergedRequest> = Vec::new();
        let mut assignments: Vec<usize> = Vec::with_capacity(requests.len());
        assignments.push(0);

        for next in &requests[1..] {
            let gap = next.offset - run.end;
            let after_join = run.extend_trial(*next);
            let join = gap <= self.min_gap_bytes
                && (self.max_span_bytes <= 0 || after_join.span() <= self.max_span_bytes);
            if join {
                run = after_join;
                assignments.push(merged_requests.len());
            } else {
                merged_requests.push(run.freeze());
                run = RunState::start(*next);
                assignments.push(merged_requests.len());
            }
        }
        merged_requests.push(run.freeze());

        IoPlan {
            merged_requests,
            assignments,
        }
    }
}

/// In-flight state for the merged run currently being assembled.
/// Kept as a tiny POD so extending a run is a copy, not an allocation.
#[derive(Debug, Clone, Copy)]
struct RunState {
    start: i64,
    end: i64,
}

impl RunState {
    #[inline]
    fn start(r: IoRequest) -> Self {
        Self {
            start: r.offset,
            end: r.end(),
        }
    }

    #[inline]
    fn extend_trial(&self, next: IoRequest) -> Self {
        Self {
            start: self.start,
            end: next.end().max(self.end),
        }
    }

    #[inline]
    fn span(&self) -> i64 {
        self.end - self.start
    }

    #[inline]
    fn freeze(self) -> MergedRequest {
        MergedRequest {
            offset: self.start,
            length: self.span(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::PlanConstraints;

    fn req(id: i32, offset: i64, length: i64) -> IoRequest {
        IoRequest {
            cluster_id: id,
            offset,
            length,
        }
    }

    fn default_cost() -> CostModel {
        CostModel::default()
    }

    #[test]
    fn empty_input_gives_empty_plan() {
        let plan =
            Greedy::default().generate_plan(&[], &default_cost(), &PlanConstraints::default());
        assert!(plan.merged_requests.is_empty());
        assert!(plan.assignments.is_empty());
    }

    #[test]
    fn adjacent_clusters_with_zero_gap_merge_into_one() {
        let rs = vec![req(0, 0, 100), req(1, 100, 100), req(2, 200, 100)];
        let plan = Greedy::new(1024, 10_000).generate_plan(
            &rs,
            &default_cost(),
            &PlanConstraints::default(),
        );
        assert_eq!(plan.merged_requests.len(), 1);
        assert_eq!(plan.merged_requests[0].offset, 0);
        assert_eq!(plan.merged_requests[0].length, 300);
        plan.check_invariants(&rs).unwrap();
    }

    #[test]
    fn gap_equal_to_threshold_joins() {
        // offset 100 + length 100 = end at 200; next at 300 → gap = 100
        let rs = vec![req(0, 100, 100), req(1, 300, 100)];
        let plan =
            Greedy::new(100, 0).generate_plan(&rs, &default_cost(), &PlanConstraints::default());
        assert_eq!(plan.merged_requests.len(), 1);
        plan.check_invariants(&rs).unwrap();
    }

    #[test]
    fn gap_over_threshold_cuts() {
        let rs = vec![req(0, 0, 100), req(1, 200, 100)]; // gap = 100
        let plan =
            Greedy::new(99, 0).generate_plan(&rs, &default_cost(), &PlanConstraints::default());
        assert_eq!(plan.merged_requests.len(), 2);
        plan.check_invariants(&rs).unwrap();
    }

    #[test]
    fn span_cap_forces_a_cut_even_with_small_gap() {
        // Three tiny clusters end-to-end; total span 300.
        let rs = vec![req(0, 0, 100), req(1, 100, 100), req(2, 200, 100)];
        // gap = 0 allows merging, but span cap 150 limits it.
        let plan =
            Greedy::new(1024, 150).generate_plan(&rs, &default_cost(), &PlanConstraints::default());
        assert!(plan.merged_requests.len() >= 2);
        // Each merged request must be ≤ span cap.
        for m in &plan.merged_requests {
            assert!(m.length <= 150, "merged length {} exceeded cap", m.length);
        }
        plan.check_invariants(&rs).unwrap();
    }

    #[test]
    fn max_span_zero_disables_span_cap() {
        // 10 clusters × 100 bytes with zero gap → span 1000.
        let rs: Vec<_> = (0..10).map(|i| req(i, i as i64 * 100, 100)).collect();
        let plan =
            Greedy::new(0, 0).generate_plan(&rs, &default_cost(), &PlanConstraints::default());
        assert_eq!(plan.merged_requests.len(), 1);
        assert_eq!(plan.merged_requests[0].length, 1000);
    }

    #[test]
    fn min_gap_zero_disables_merging() {
        // `min_gap = 0` only permits fully adjacent joins (`gap == 0`).
        // Any positive gap must cut.
        let rs = vec![req(0, 0, 100), req(1, 101, 100)]; // gap = 1
        let plan =
            Greedy::new(0, 0).generate_plan(&rs, &default_cost(), &PlanConstraints::default());
        assert_eq!(plan.merged_requests.len(), 2);
    }

    #[test]
    fn heterogeneous_clusters_with_small_gap_merge() {
        // Paper-motivating case: tiny + huge cluster with tiny gap.
        // Waste-ratio heuristics would cut here because ratio is
        // dominated by the small cluster being "padded" by the large
        // one; gap-threshold correctly merges because the gap itself
        // is cheap to bridge.
        let rs = vec![
            req(0, 0, 10),          // tiny
            req(1, 100, 1_000_000), // huge, 90-byte gap
        ];
        let plan =
            Greedy::new(1000, 0).generate_plan(&rs, &default_cost(), &PlanConstraints::default());
        assert_eq!(plan.merged_requests.len(), 1);
        plan.check_invariants(&rs).unwrap();
    }

    #[test]
    fn heterogeneous_clusters_with_large_gap_cut() {
        // Same clusters, but gap is now 10 MiB → well above any
        // reasonable min_gap. Must cut.
        let rs = vec![req(0, 0, 10), req(1, 10 * 1024 * 1024 + 10, 1_000_000)];
        let plan =
            Greedy::default().generate_plan(&rs, &default_cost(), &PlanConstraints::default());
        assert_eq!(plan.merged_requests.len(), 2);
        plan.check_invariants(&rs).unwrap();
    }

    /// Sweep `min_gap_bytes` ∈ {1, 2, 4, 8, 16, 32} MiB like the
    /// paper's Greedy Oracle and check the trend: request count
    /// monotonically non-increasing as the threshold grows.
    #[test]
    fn paper_oracle_sweep_is_monotone_in_gap() {
        // 32 clusters with mixed gaps — half < 1 MB, half in 2-30 MB.
        let mut rs = Vec::new();
        let mut offset = 0i64;
        let mut rng_state: u64 = 0xC0FFEE;
        for i in 0..32 {
            rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
            let pick = (rng_state >> 33) as u32;
            let length = 8_192 + (pick % 4_096) as i64;
            rs.push(IoRequest {
                cluster_id: i,
                offset,
                length,
            });
            let gap = if pick % 2 == 0 {
                (pick % (1 << 20)) as i64 // < 1 MiB
            } else {
                2 * 1024 * 1024 + ((pick as i64) & ((28 << 20) - 1)) // 2-30 MiB
            };
            offset += length + gap;
        }

        let sweeps_mib = [1, 2, 4, 8, 16, 32];
        let mut last_req_count = usize::MAX;
        for mib in sweeps_mib {
            let min_gap = mib * 1024 * 1024;
            let plan = Greedy::new(min_gap, 0).generate_plan(
                &rs,
                &default_cost(),
                &PlanConstraints::default(),
            );
            plan.check_invariants(&rs).unwrap();
            assert!(
                plan.merged_requests.len() <= last_req_count,
                "sweep {mib} MiB: {} > previous {last_req_count}",
                plan.merged_requests.len(),
            );
            last_req_count = plan.merged_requests.len();
        }
    }

    #[test]
    fn cost_model_is_not_read() {
        // Greedy's decisions depend only on gap/span thresholds; the
        // cost model is unused. Two very different cost models must
        // produce the same plan on the same input.
        let rs = vec![req(0, 0, 100), req(1, 200, 100), req(2, 500, 100)];
        let cheap = CostModel {
            latency_per_request: 0.1,
            latency_per_byte: 1.0,
            ..CostModel::default()
        };
        let expensive = CostModel {
            latency_per_request: 1000.0,
            latency_per_byte: 1e-9,
            ..CostModel::default()
        };
        let g = Greedy::new(1024, 0);
        let a = g.generate_plan(&rs, &cheap, &PlanConstraints::default());
        let b = g.generate_plan(&rs, &expensive, &PlanConstraints::default());
        assert_eq!(a.merged_requests, b.merged_requests);
        assert_eq!(a.assignments, b.assignments);
    }
}
