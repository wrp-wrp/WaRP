//! Separate — no merging.
//!
//! The baseline every other strategy is compared against. Each input
//! request becomes its own merged request, and the plan has zero waste.
//! Exists as a correctness anchor (we expect to beat it under any
//! positive per-request overhead) and as a sanity check for the other
//! strategies' invariant code.
//!
//! Complexity: `O(n)` time, `O(n)` space (the returned plan).

use crate::cost_model::CostModel;
use crate::io_plan::{IoPlan, IoRequest};
use crate::plan_constraints::PlanConstraints;
use crate::strategy::MergeStrategy;

/// The no-merge baseline strategy. Has no tunables.
#[derive(Debug, Clone, Copy, Default)]
pub struct Separate;

impl MergeStrategy for Separate {
    fn name(&self) -> &'static str {
        "separate"
    }

    fn generate_plan(
        &self,
        requests: &[IoRequest],
        _cost: &CostModel,
        _constraints: &PlanConstraints,
    ) -> IoPlan {
        IoPlan::one_per_request(requests)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io_plan::MergedRequest;
    use crate::PlanConstraints;

    fn req(id: i32, offset: i64, length: i64) -> IoRequest {
        IoRequest {
            cluster_id: id,
            offset,
            length,
        }
    }

    #[test]
    fn empty_input_gives_empty_plan() {
        let plan = Separate.generate_plan(&[], &CostModel::default(), &PlanConstraints::default());
        assert!(plan.merged_requests.is_empty());
        assert!(plan.assignments.is_empty());
    }

    #[test]
    fn one_per_request_matches_input() {
        let rs = vec![req(0, 0, 10), req(1, 100, 20), req(2, 500, 1)];
        let plan = Separate.generate_plan(&rs, &CostModel::default(), &PlanConstraints::default());
        assert_eq!(plan.merged_requests.len(), 3);
        assert_eq!(
            plan.merged_requests[1],
            MergedRequest {
                offset: 100,
                length: 20
            }
        );
        assert_eq!(plan.assignments, vec![0, 1, 2]);
        plan.check_invariants(&rs).unwrap();
    }
}
