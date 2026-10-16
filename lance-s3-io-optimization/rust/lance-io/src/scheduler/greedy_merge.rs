// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Greedy I/O Merge Strategy
//!
//! Greedy algorithm that merges adjacent clusters if the waste ratio
//! is acceptable. This is a fast heuristic that provides good results
//! but may not be globally optimal.
//!
//! Algorithm:
//! 1. Sort clusters by file offset
//! 2. Start with the first cluster
//! 3. For each subsequent cluster:
//!    - Try to merge with current request
//!    - If waste ratio exceeds threshold, start a new request
//! 4. Return the list of merged requests
//!
//! Time Complexity: O(n log n) for sorting + O(n) for merging = O(n log n)
//! Space Complexity: O(n)

use super::cost_model::CostModel;
use super::io_types::{IOMetrics, IOPlan, IORequest};
use super::merge_strategy::{calculate_plan_metrics, sort_cluster_ids_by_offset, MergeStrategy};
use std::collections::HashMap;

/// Greedy merge strategy
pub struct GreedyMerge;

impl GreedyMerge {
    pub fn new() -> Self {
        Self
    }

    /// Internal greedy merge implementation
    fn greedy_plan(
        cluster_ids: &[i32],
        cluster_offsets: &HashMap<i32, i64>,
        cluster_sizes: &HashMap<i32, i64>,
        cost_model: &CostModel,
        enforce_span: bool,
    ) -> IOPlan {
        if cluster_ids.is_empty() {
            return IOPlan::new(vec![], IOMetrics::default());
        }

        let sorted_ids = sort_cluster_ids_by_offset(cluster_ids, cluster_offsets);
        let mut requests = Vec::new();

        // Start with the first cluster
        let first_cid = sorted_ids[0];
        let mut current_start = *cluster_offsets.get(&first_cid).unwrap();
        let first_size = *cluster_sizes.get(&first_cid).unwrap();
        let mut current_end = current_start + first_size;
        let mut current_useful = first_size;

        // Try to merge subsequent clusters
        for &cid in &sorted_ids[1..] {
            let next_start = *cluster_offsets.get(&cid).unwrap();
            let next_size = *cluster_sizes.get(&cid).unwrap();
            let next_end = next_start + next_size;

            let merged_span = next_end - current_start;
            let merged_useful = current_useful + next_size;
            let waste_ratio = if merged_span > 0 {
                (merged_span - merged_useful) as f64 / merged_span as f64
            } else {
                0.0
            };

            let mut can_merge = waste_ratio <= cost_model.max_waste_ratio;

            // Check span constraint if enforced
            if enforce_span
                && cost_model.max_merge_span > 0
                && merged_span > cost_model.max_merge_span
            {
                can_merge = false;
            }

            if can_merge {
                // Merge: extend current request
                current_end = next_end;
                current_useful += next_size;
            } else {
                // Cannot merge: finalize current request and start new one
                requests.push(IORequest::new(current_start, current_end - current_start));
                current_start = next_start;
                current_end = next_end;
                current_useful = next_size;
            }
        }

        // Add the last request
        requests.push(IORequest::new(current_start, current_end - current_start));

        // Compute metrics
        let metrics = calculate_plan_metrics(&requests, cluster_ids, cluster_sizes, cost_model);

        IOPlan::new(requests, metrics)
    }
}

impl MergeStrategy for GreedyMerge {
    fn generate_plan(
        &self,
        cluster_ids: &[i32],
        cluster_offsets: &HashMap<i32, i64>,
        cluster_sizes: &HashMap<i32, i64>,
        cost_model: &CostModel,
    ) -> IOPlan {
        Self::greedy_plan(
            cluster_ids,
            cluster_offsets,
            cluster_sizes,
            cost_model,
            false,
        )
    }

    fn name(&self) -> &str {
        "Greedy"
    }
}

/// Strict greedy frontier merge used by the Figure 9 baseline.
///
/// This follows the paper rule exactly:
/// - sort raw intervals by byte offset
/// - scan from left to right
/// - merge only if both:
///   1. wasted_bytes / useful_bytes <= rho
///   2. merged_span <= Smax
pub struct GreedyFrontierMerge;

/// Public alias used by the paper docs.
pub type FrontierGreedyMerge = GreedyFrontierMerge;

impl GreedyFrontierMerge {
    pub fn new() -> Self {
        Self
    }
}

impl MergeStrategy for GreedyFrontierMerge {
    fn generate_plan(
        &self,
        cluster_ids: &[i32],
        cluster_offsets: &HashMap<i32, i64>,
        cluster_sizes: &HashMap<i32, i64>,
        cost_model: &CostModel,
    ) -> IOPlan {
        if cluster_ids.is_empty() {
            return IOPlan::new(vec![], IOMetrics::default());
        }

        let sorted_ids = sort_cluster_ids_by_offset(cluster_ids, cluster_offsets);
        let mut requests = Vec::new();

        let first_cid = sorted_ids[0];
        let mut current_start = *cluster_offsets.get(&first_cid).unwrap();
        let first_size = *cluster_sizes.get(&first_cid).unwrap();
        let mut current_end = current_start + first_size;
        let mut current_useful = first_size;

        for &cid in &sorted_ids[1..] {
            let next_start = *cluster_offsets.get(&cid).unwrap();
            let next_size = *cluster_sizes.get(&cid).unwrap();
            let next_end = next_start + next_size;

            let merged_span = next_end - current_start;
            let merged_useful = current_useful + next_size;
            let wasted_bytes = merged_span - merged_useful;
            let waste_ratio = if merged_useful > 0 {
                wasted_bytes as f64 / merged_useful as f64
            } else {
                0.0
            };

            let span_ok =
                cost_model.max_merge_span <= 0 || merged_span <= cost_model.max_merge_span;
            let can_merge = span_ok && waste_ratio <= cost_model.max_waste_ratio;

            if can_merge {
                current_end = next_end;
                current_useful = merged_useful;
            } else {
                requests.push(IORequest::new(current_start, current_end - current_start));
                current_start = next_start;
                current_end = next_end;
                current_useful = next_size;
            }
        }

        requests.push(IORequest::new(current_start, current_end - current_start));
        let metrics = calculate_plan_metrics(&requests, cluster_ids, cluster_sizes, cost_model);
        IOPlan::new(requests, metrics)
    }

    fn name(&self) -> &str {
        "GreedyFrontier"
    }
}

/// Greedy merge with an additional max-merge-span constraint.
///
/// This keeps the same waste-ratio rule as GreedyMerge, but also refuses to
/// merge if the merged request span would exceed cost_model.max_merge_span.
/// This mirrors common "range_size_limit" style baselines used in object-store
/// range coalescing.
pub struct GreedyMergeCapped;

/// Public alias with a clearer behavioral name than the historical one.
pub type SpanCappedGreedyMerge = GreedyMergeCapped;

impl GreedyMergeCapped {
    pub fn new() -> Self {
        Self
    }
}

impl MergeStrategy for GreedyMergeCapped {
    fn generate_plan(
        &self,
        cluster_ids: &[i32],
        cluster_offsets: &HashMap<i32, i64>,
        cluster_sizes: &HashMap<i32, i64>,
        cost_model: &CostModel,
    ) -> IOPlan {
        GreedyMerge::greedy_plan(
            cluster_ids,
            cluster_offsets,
            cluster_sizes,
            cost_model,
            true,
        )
    }

    fn name(&self) -> &str {
        "GreedyCapped"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_greedy_merge_basic() {
        let cluster_ids = vec![1, 2, 3];
        let mut cluster_offsets = HashMap::new();
        cluster_offsets.insert(1, 0);
        cluster_offsets.insert(2, 1000);
        cluster_offsets.insert(3, 2000);

        let mut cluster_sizes = HashMap::new();
        cluster_sizes.insert(1, 500);
        cluster_sizes.insert(2, 500);
        cluster_sizes.insert(3, 500);

        let mut cost_model = CostModel::default();
        cost_model.max_waste_ratio = 0.5; // Allow 50% waste

        let strategy = GreedyMerge::new();
        let plan =
            strategy.generate_plan(&cluster_ids, &cluster_offsets, &cluster_sizes, &cost_model);

        // Should merge all three clusters (waste ratio is acceptable)
        assert_eq!(plan.requests.len(), 1);
        assert_eq!(plan.requests[0].offset, 0);
        assert_eq!(plan.requests[0].length, 2500);
    }

    #[test]
    fn test_greedy_merge_high_waste() {
        let cluster_ids = vec![1, 2];
        let mut cluster_offsets = HashMap::new();
        cluster_offsets.insert(1, 0);
        cluster_offsets.insert(2, 10000); // Large gap

        let mut cluster_sizes = HashMap::new();
        cluster_sizes.insert(1, 500);
        cluster_sizes.insert(2, 500);

        let mut cost_model = CostModel::default();
        cost_model.max_waste_ratio = 0.1; // Only allow 10% waste

        let strategy = GreedyMerge::new();
        let plan =
            strategy.generate_plan(&cluster_ids, &cluster_offsets, &cluster_sizes, &cost_model);

        // Should NOT merge (waste ratio too high)
        assert_eq!(plan.requests.len(), 2);
    }

    #[test]
    fn test_greedy_capped_span_limit() {
        let cluster_ids = vec![1, 2];
        let mut cluster_offsets = HashMap::new();
        cluster_offsets.insert(1, 0);
        cluster_offsets.insert(2, 1000);

        let mut cluster_sizes = HashMap::new();
        cluster_sizes.insert(1, 500);
        cluster_sizes.insert(2, 500);

        let mut cost_model = CostModel::default();
        cost_model.max_waste_ratio = 1.0; // Allow any waste
        cost_model.max_merge_span = 1000; // But limit span to 1000 bytes

        let strategy = GreedyMergeCapped::new();
        let plan =
            strategy.generate_plan(&cluster_ids, &cluster_offsets, &cluster_sizes, &cost_model);

        // Should NOT merge (span exceeds limit)
        assert_eq!(plan.requests.len(), 2);
    }

    #[test]
    fn test_greedy_frontier_uses_wasted_over_useful_ratio() {
        let cluster_ids = vec![1, 2];
        let mut cluster_offsets = HashMap::new();
        cluster_offsets.insert(1, 0);
        cluster_offsets.insert(2, 80);

        let mut cluster_sizes = HashMap::new();
        cluster_sizes.insert(1, 30);
        cluster_sizes.insert(2, 30);

        let mut cost_model = CostModel::default();
        cost_model.max_waste_ratio = 0.5;
        cost_model.max_merge_span = 200;

        let legacy_plan = GreedyMerge::new().generate_plan(
            &cluster_ids,
            &cluster_offsets,
            &cluster_sizes,
            &cost_model,
        );
        assert_eq!(legacy_plan.requests.len(), 1);

        let strict_plan = GreedyFrontierMerge::new().generate_plan(
            &cluster_ids,
            &cluster_offsets,
            &cluster_sizes,
            &cost_model,
        );

        // merged_span = 110, useful = 60, wasted = 50
        // legacy ratio = 50/110 <= 0.5
        // strict ratio = 50/60 > 0.5
        assert_eq!(strict_plan.requests.len(), 2);
    }

    #[test]
    fn test_greedy_frontier_enforces_span_cap() {
        let cluster_ids = vec![1, 2];
        let mut cluster_offsets = HashMap::new();
        cluster_offsets.insert(1, 0);
        cluster_offsets.insert(2, 700);

        let mut cluster_sizes = HashMap::new();
        cluster_sizes.insert(1, 400);
        cluster_sizes.insert(2, 400);

        let mut cost_model = CostModel::default();
        cost_model.max_waste_ratio = 10.0;
        cost_model.max_merge_span = 1000;

        let plan = GreedyFrontierMerge::new().generate_plan(
            &cluster_ids,
            &cluster_offsets,
            &cluster_sizes,
            &cost_model,
        );

        assert_eq!(plan.requests.len(), 2);
    }

    #[test]
    fn test_greedy_frontier_merges_transitively_when_both_constraints_hold() {
        let cluster_ids = vec![1, 2, 3];
        let mut cluster_offsets = HashMap::new();
        cluster_offsets.insert(1, 0);
        cluster_offsets.insert(2, 100);
        cluster_offsets.insert(3, 200);

        let mut cluster_sizes = HashMap::new();
        cluster_sizes.insert(1, 80);
        cluster_sizes.insert(2, 80);
        cluster_sizes.insert(3, 80);

        let mut cost_model = CostModel::default();
        cost_model.max_waste_ratio = 0.5;
        cost_model.max_merge_span = 320;

        let plan = GreedyFrontierMerge::new().generate_plan(
            &cluster_ids,
            &cluster_offsets,
            &cluster_sizes,
            &cost_model,
        );

        assert_eq!(plan.requests.len(), 1);
        assert_eq!(plan.requests[0].offset, 0);
        assert_eq!(plan.requests[0].length, 280);
    }
}
