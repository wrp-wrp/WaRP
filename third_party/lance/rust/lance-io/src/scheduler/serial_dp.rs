// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Serial Dynamic Programming Merge Strategy
//!
//! Optimal merge strategy using dynamic programming to minimize total work W.
//!
//! Objective:
//! - Minimize total work W = sum_i w_i under optional request-count cap K.
//!
//! State:
//! - dp[i][j] = minimum W to cover first i clusters using exactly j requests.
//!
//! Algorithm:
//! 1. Sort clusters by file offset
//! 2. Build edge latency matrix for all possible merged requests
//! 3. Run DP: dp[i][j] = min over k { dp[k][j-1] + w(k+1..i) }
//! 4. Backtrack to reconstruct optimal request sequence
//!
//! Time Complexity: O(n^2 * k) where n = number of clusters, k = max requests
//! Space Complexity: O(n * k)

use super::cost_model::CostModel;
use super::io_types::{IOMetrics, IOPlan, IORequest};
use super::merge_strategy::{
    calculate_plan_metrics, collect_ordered_cluster_layout, requests_from_ordered_spans,
    MergeStrategy,
};
use std::collections::HashMap;

const INF: f64 = 1e18;

/// 2D table stored as flat vector for cache efficiency
struct FlatTable<T> {
    data: Vec<T>,
    cols: usize,
}

impl<T: Clone> FlatTable<T> {
    fn new(rows: usize, cols: usize, init: T) -> Self {
        Self {
            data: vec![init; rows * cols],
            cols,
        }
    }

    fn at(&self, i: usize, j: usize) -> &T {
        &self.data[i * self.cols + j]
    }

    fn at_mut(&mut self, i: usize, j: usize) -> &mut T {
        &mut self.data[i * self.cols + j]
    }
}

/// Triangular matrix index for edge latency
#[inline]
fn triangular_edge_index(end_idx: usize, start_idx: usize) -> usize {
    (end_idx * (end_idx + 1)) / 2 + start_idx
}

/// Build edge latency matrix for all possible merged requests
fn build_edge_latency_matrix(offsets: &[i64], sizes: &[i64], cost_model: &CostModel) -> Vec<f64> {
    let n = offsets.len();
    let total_edges = (n * (n + 1)) / 2;
    let mut edge_latency = vec![0.0; total_edges];

    for end in 0..n {
        let range_end = offsets[end] + sizes[end];
        for start in 0..=end {
            let range_bytes = range_end - offsets[start];
            edge_latency[triangular_edge_index(end, start)] =
                cost_model.request_latency_ms(range_bytes);
        }
    }

    edge_latency
}

/// Run serial DP to find optimal merge plan
fn run_serial_dp(
    edge_latency: &[f64],
    n: usize,
    k_max: usize,
    dp: &mut FlatTable<f64>,
    parent: &mut FlatTable<i32>,
) {
    // Initialize: dp[0][0] = 0, all others = INF
    *dp.at_mut(0, 0) = 0.0;

    for i in 1..=n {
        let j_max = i.min(k_max);
        for j in 1..=j_max {
            let mut best_val = INF;
            let mut best_cut = -1;

            // Try all possible previous positions
            for k in (j - 1)..i {
                let prev = *dp.at(k, j - 1);
                if prev >= INF / 10.0 {
                    continue;
                }

                let w = edge_latency[triangular_edge_index(i - 1, k)];
                let cand = prev + w;

                if cand < best_val {
                    best_val = cand;
                    best_cut = k as i32;
                }
            }

            *dp.at_mut(i, j) = best_val;
            *parent.at_mut(i, j) = best_cut;
        }
    }
}

/// Reconstruct request sequence from parent pointers
fn reconstruct_requests_from_parent(
    n: usize,
    request_count: usize,
    parent: &FlatTable<i32>,
    offsets: &[i64],
    sizes: &[i64],
) -> Vec<IORequest> {
    let mut requests = Vec::new();
    let mut cur_i = n;
    let mut cur_j = request_count;

    while cur_j > 0 && cur_i > 0 {
        let prev = *parent.at(cur_i, cur_j);
        if prev < 0 || prev as usize >= cur_i {
            break;
        }

        let prev_usize = prev as usize;
        let range_start = offsets[prev_usize];
        let range_end = offsets[cur_i - 1] + sizes[cur_i - 1];
        requests.push(IORequest::new(range_start, range_end - range_start));

        cur_i = prev_usize;
        cur_j -= 1;
    }

    requests.reverse();
    requests
}

/// Serial DP merge strategy
pub struct SerialDP;

impl SerialDP {
    pub fn new() -> Self {
        Self
    }

    /// Sweep DP: Compute all solutions for different number of requests
    /// Returns all valid solutions (one for each j = 1, 2, ..., n)
    pub fn sweep_all_solutions(
        cluster_ids: &[i32],
        cluster_offsets: &HashMap<i32, i64>,
        cluster_sizes: &HashMap<i32, i64>,
        cost_model: &CostModel,
    ) -> Vec<DPSweepResult> {
        if cluster_ids.is_empty() {
            return vec![];
        }

        let layout = collect_ordered_cluster_layout(cluster_ids, cluster_offsets, cluster_sizes);
        let n = layout.ordered_cluster_ids.len();
        let offsets = layout.offsets;
        let sizes = layout.sizes;

        let edge_latency = build_edge_latency_matrix(&offsets, &sizes, cost_model);

        let mut dp = FlatTable::new(n + 1, n + 1, INF);
        let mut parent = FlatTable::new(n + 1, n + 1, -1);
        run_serial_dp(&edge_latency, n, n, &mut dp, &mut parent);

        let mut results = Vec::new();
        for j in 1..=n {
            if *dp.at(n, j) >= INF / 10.0 {
                continue;
            }

            let requests = reconstruct_requests_from_parent(n, j, &parent, &offsets, &sizes);
            let metrics = calculate_plan_metrics(&requests, cluster_ids, cluster_sizes, cost_model);
            results.push(DPSweepResult {
                num_requests: j,
                requests,
                metrics,
            });
        }

        results
    }
}

impl MergeStrategy for SerialDP {
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
        let k_cap = cost_model.request_cap(n);

        let edge_latency = build_edge_latency_matrix(&offsets, &sizes, cost_model);

        let mut dp = FlatTable::new(n + 1, k_cap + 1, INF);
        let mut parent = FlatTable::new(n + 1, k_cap + 1, -1);
        run_serial_dp(&edge_latency, n, k_cap, &mut dp, &mut parent);

        // Find best solution
        let mut best_j = 0;
        let mut best_value = INF;
        for j in 1..=k_cap {
            let v = *dp.at(n, j);
            if v < best_value {
                best_value = v;
                best_j = j;
            }
        }

        let requests = if best_j > 0 {
            reconstruct_requests_from_parent(n, best_j, &parent, &offsets, &sizes)
        } else {
            requests_from_ordered_spans(&offsets, &sizes)
        };

        let metrics = calculate_plan_metrics(&requests, cluster_ids, cluster_sizes, cost_model);
        IOPlan::new(requests, metrics)
    }

    fn name(&self) -> &str {
        "SerialDP"
    }
}

/// Result from DP sweep (Pareto frontier)
#[derive(Debug, Clone)]
pub struct DPSweepResult {
    pub num_requests: usize,
    pub requests: Vec<IORequest>,
    pub metrics: IOMetrics,
}

/// Public alias used in paper-facing docs.
pub type OptimalSerialMerge = SerialDP;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_serial_dp_basic() {
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

        let strategy = SerialDP::new();
        let plan =
            strategy.generate_plan(&cluster_ids, &cluster_offsets, &cluster_sizes, &cost_model);

        // Should find optimal solution
        assert!(!plan.requests.is_empty());
        assert!(plan.metrics.total_latency > 0.0);
    }

    #[test]
    fn test_serial_dp_with_cap() {
        let cluster_ids = vec![1, 2, 3, 4];
        let mut cluster_offsets = HashMap::new();
        cluster_offsets.insert(1, 0);
        cluster_offsets.insert(2, 1000);
        cluster_offsets.insert(3, 2000);
        cluster_offsets.insert(4, 3000);

        let mut cluster_sizes = HashMap::new();
        cluster_sizes.insert(1, 500);
        cluster_sizes.insert(2, 500);
        cluster_sizes.insert(3, 500);
        cluster_sizes.insert(4, 500);

        let mut cost_model = CostModel::default();
        cost_model.max_requests = 2; // Force at most 2 requests

        let strategy = SerialDP::new();
        let plan =
            strategy.generate_plan(&cluster_ids, &cluster_offsets, &cluster_sizes, &cost_model);

        // Should respect the cap
        assert!(plan.requests.len() <= 2);
    }

    #[test]
    fn test_sweep_all_solutions() {
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

        let results = SerialDP::sweep_all_solutions(
            &cluster_ids,
            &cluster_offsets,
            &cluster_sizes,
            &cost_model,
        );

        // Should have solutions for different request counts
        assert!(!results.is_empty());
        assert!(results.len() <= 3);

        // Verify solutions are valid
        for result in &results {
            assert_eq!(result.requests.len(), result.num_requests);
        }
    }
}
