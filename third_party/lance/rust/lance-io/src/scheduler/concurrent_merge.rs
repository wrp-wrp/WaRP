// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Concurrent (Min-Max) I/O Merge Strategy
//!
//! For concurrent execution, the bottleneck is the LARGEST request.
//! Optimization goal: given exactly k requests, minimize max(span_j).
//!
//! This is the classic "partition ordered items into k contiguous groups
//! minimizing the maximum group weight" problem.
//!
//! Algorithm: binary search on the answer + greedy feasibility check.
//! Time Complexity: O(n log S) per k, O(n K log S) for full sweep.
//!   where n = number of clusters, K = max requests, S = total span.
//! Space Complexity: O(n)

use super::cost_model::CostModel;
use super::io_types::{IOMetrics, IOPlan, IORequest};
use super::merge_strategy::{
    calculate_plan_metrics, collect_ordered_cluster_layout, requests_from_ordered_spans,
    MergeStrategy,
};
use std::collections::HashMap;

/// Concurrent merge strategy
pub struct ConcurrentMerge;

impl ConcurrentMerge {
    pub fn new() -> Self {
        Self
    }

    /// Check if we can cover all clusters with <= k requests,
    /// each having span <= threshold
    fn is_feasible(offsets: &[i64], sizes: &[i64], k: usize, threshold: i64) -> bool {
        let n = offsets.len();
        if n == 0 {
            return true;
        }

        let mut groups = 1;
        let mut group_start = offsets[0];

        for i in 0..n {
            let group_end = offsets[i] + sizes[i];
            let span = group_end - group_start;

            if span > threshold {
                // Start a new group at cluster i
                groups += 1;
                group_start = offsets[i];

                // Check that a single cluster doesn't exceed threshold
                if sizes[i] > threshold {
                    return false;
                }

                if groups > k {
                    return false;
                }
            }
        }

        groups <= k
    }

    /// Greedy partition: given threshold, produce the actual requests.
    /// Each group extends as far right as possible without exceeding
    /// the threshold span.
    fn greedy_partition(offsets: &[i64], sizes: &[i64], threshold: i64) -> Vec<IORequest> {
        let n = offsets.len();
        if n == 0 {
            return vec![];
        }

        let mut requests = Vec::new();
        let mut group_start = offsets[0];
        let mut group_end = offsets[0] + sizes[0];

        for i in 1..n {
            let tentative_end = offsets[i] + sizes[i];
            let tentative_span = tentative_end - group_start;

            if tentative_span > threshold {
                // Emit current group
                requests.push(IORequest::new(group_start, group_end - group_start));
                // Start new group
                group_start = offsets[i];
                group_end = offsets[i] + sizes[i];
            } else {
                group_end = tentative_end;
            }
        }

        // Emit last group
        requests.push(IORequest::new(group_start, group_end - group_start));
        requests
    }

    /// Solve for a specific k: binary search for minimum feasible threshold
    fn solve_for_k(offsets: &[i64], sizes: &[i64], k: usize) -> (Vec<IORequest>, i64) {
        let n = offsets.len();
        if n == 0 {
            return (vec![], 0);
        }

        // Lower bound: max single-cluster size (can't go below this)
        let lo_init = *sizes.iter().max().unwrap();

        // Upper bound: total span from first to last cluster
        let hi_init = (offsets[n - 1] + sizes[n - 1]) - offsets[0];

        // Special case: k >= n → each cluster separate
        if k >= n {
            let requests = requests_from_ordered_spans(offsets, sizes);
            return (requests, lo_init);
        }

        // Special case: k == 1 → everything merged
        if k == 1 {
            return (vec![IORequest::new(offsets[0], hi_init)], hi_init);
        }

        // Binary search
        let mut lo = lo_init;
        let mut hi = hi_init;

        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            if Self::is_feasible(offsets, sizes, k, mid) {
                hi = mid;
            } else {
                lo = mid + 1;
            }
        }

        let max_span = lo;
        let requests = Self::greedy_partition(offsets, sizes, lo);
        (requests, max_span)
    }

    /// Sweep: for every k = 1..n, compute the min-max grouping
    pub fn sweep_all_solutions(
        cluster_ids: &[i32],
        cluster_offsets: &HashMap<i32, i64>,
        cluster_sizes: &HashMap<i32, i64>,
        cost_model: &CostModel,
    ) -> Vec<ConcurrentSweepResult> {
        if cluster_ids.is_empty() {
            return vec![];
        }

        let layout = collect_ordered_cluster_layout(cluster_ids, cluster_offsets, cluster_sizes);
        let n = layout.ordered_cluster_ids.len();
        let offsets = layout.offsets;
        let sizes = layout.sizes;

        let mut results = Vec::new();
        for k in 1..=n {
            let (requests, max_span) = Self::solve_for_k(&offsets, &sizes, k);
            let metrics = calculate_plan_metrics(&requests, cluster_ids, cluster_sizes, cost_model);
            results.push(ConcurrentSweepResult {
                num_requests: k,
                max_span: max_span as f64,
                requests,
                metrics,
            });
        }

        results
    }
}

impl MergeStrategy for ConcurrentMerge {
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

        let layout = collect_ordered_cluster_layout(cluster_ids, cluster_offsets, cluster_sizes);
        let n = layout.ordered_cluster_ids.len();
        let offsets = layout.offsets;
        let sizes = layout.sizes;

        // Try all k (up to K cap) and pick the one with minimum concurrent latency
        let mut best_latency = f64::INFINITY;
        let mut best_requests = Vec::new();

        let k_cap = cost_model.request_cap(n);

        for k in 1..=k_cap {
            let (requests, max_span) = Self::solve_for_k(&offsets, &sizes, k);

            // Concurrent latency model
            let latency = if cost_model.concurrent_parallelism > 0.0 {
                let parallelism = cost_model.concurrent_parallelism.max(1.0);
                let waves = (k as f64 / parallelism).max(1.0);
                let wave_k = (k as f64).min(parallelism.ceil()) as usize;

                let mut jitter = 0.0;
                if wave_k >= 2 && cost_model.latency_std > 0.0 {
                    jitter = cost_model.latency_std * (2.0 * (wave_k as f64).ln()).sqrt();
                }

                waves * cost_model.request_latency_ms(max_span) + jitter
            } else {
                // Legacy model: assume full parallelism (single wave)
                let mut effective_overhead = 0.0;
                if k >= 2 && cost_model.latency_std > 0.0 {
                    effective_overhead = cost_model.latency_std * (2.0 * (k as f64).ln()).sqrt();
                }
                cost_model.request_latency_ms(max_span) + effective_overhead
            };

            if latency < best_latency {
                best_latency = latency;
                best_requests = requests;
            }
        }

        let metrics =
            calculate_plan_metrics(&best_requests, cluster_ids, cluster_sizes, cost_model);
        IOPlan::new(best_requests, metrics)
    }

    fn name(&self) -> &str {
        "Concurrent"
    }
}

/// Result from concurrent sweep (one per k)
#[derive(Debug, Clone)]
pub struct ConcurrentSweepResult {
    pub num_requests: usize,
    pub max_span: f64,
    pub requests: Vec<IORequest>,
    pub metrics: IOMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_concurrent_merge_basic() {
        let cluster_ids = vec![1, 2, 3];
        let mut cluster_offsets = HashMap::new();
        cluster_offsets.insert(1, 0);
        cluster_offsets.insert(2, 1000);
        cluster_offsets.insert(3, 2000);

        let mut cluster_sizes = HashMap::new();
        cluster_sizes.insert(1, 500);
        cluster_sizes.insert(2, 500);
        cluster_sizes.insert(3, 500);

        let cost_model = CostModel::default();

        let strategy = ConcurrentMerge::new();
        let plan =
            strategy.generate_plan(&cluster_ids, &cluster_offsets, &cluster_sizes, &cost_model);

        assert!(!plan.requests.is_empty());
    }

    #[test]
    fn test_is_feasible() {
        let offsets = vec![0, 1000, 2000];
        let sizes = vec![500, 500, 500];

        // Should be feasible with k=1 and large threshold
        assert!(ConcurrentMerge::is_feasible(&offsets, &sizes, 1, 3000));

        // Should NOT be feasible with k=1 and small threshold
        assert!(!ConcurrentMerge::is_feasible(&offsets, &sizes, 1, 1000));

        // Should be feasible with k=3 (each cluster separate)
        assert!(ConcurrentMerge::is_feasible(&offsets, &sizes, 3, 500));
    }

    #[test]
    fn test_solve_for_k() {
        let offsets = vec![0, 1000, 2000, 3000];
        let sizes = vec![500, 500, 500, 500];

        // k=2: should split into 2 groups
        let (requests, max_span) = ConcurrentMerge::solve_for_k(&offsets, &sizes, 2);
        assert_eq!(requests.len(), 2);
        assert!(max_span > 0);

        // k=4: each cluster separate
        let (requests, _) = ConcurrentMerge::solve_for_k(&offsets, &sizes, 4);
        assert_eq!(requests.len(), 4);
    }
}
