// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Merge strategy interface and implementations

use super::cost_model::CostModel;
use super::io_types::{IOMetrics, IOPlan, IORequest};
use std::collections::HashMap;

/// Merge strategy type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MergeStrategyType {
    /// No merging (baseline)
    Separate,
    /// Greedy heuristic
    Greedy,
    /// Strict greedy frontier heuristic with wasted/useful ratio and span cap
    GreedyFrontier,
    /// Dynamic programming (serial optimal)
    DP,
    /// Min-max (bottleneck) grouping
    Concurrent,
    /// Finite-parallel planner: T_UB = W/P + (1-1/P) * B
    BalancedUB,
    /// Lance-IO coalesce-split algorithm
    Lance,
}

/// Trait for merge strategies
pub trait MergeStrategy {
    /// Generate an optimized I/O plan
    fn generate_plan(
        &self,
        cluster_ids: &[i32],
        cluster_offsets: &HashMap<i32, i64>,
        cluster_sizes: &HashMap<i32, i64>,
        cost_model: &CostModel,
    ) -> IOPlan;

    /// Strategy name
    fn name(&self) -> &str;
}

/// Cluster spans sorted by object-store byte offset.
pub(crate) struct OrderedClusterLayout {
    pub ordered_cluster_ids: Vec<i32>,
    pub offsets: Vec<i64>,
    pub sizes: Vec<i64>,
}

/// Return the input cluster ids sorted by byte offset.
pub(crate) fn sort_cluster_ids_by_offset(
    cluster_ids: &[i32],
    cluster_offsets: &HashMap<i32, i64>,
) -> Vec<i32> {
    let mut sorted = cluster_ids.to_vec();
    sorted.sort_unstable_by_key(|cid| cluster_offsets.get(cid).copied().unwrap_or(i64::MAX));
    sorted
}

/// Collect the sorted `(offset, size)` layout shared by most planners.
pub(crate) fn collect_ordered_cluster_layout(
    cluster_ids: &[i32],
    cluster_offsets: &HashMap<i32, i64>,
    cluster_sizes: &HashMap<i32, i64>,
) -> OrderedClusterLayout {
    let ordered_cluster_ids = sort_cluster_ids_by_offset(cluster_ids, cluster_offsets);
    let mut offsets = Vec::with_capacity(ordered_cluster_ids.len());
    let mut sizes = Vec::with_capacity(ordered_cluster_ids.len());
    for &cid in &ordered_cluster_ids {
        offsets.push(*cluster_offsets.get(&cid).unwrap());
        sizes.push(*cluster_sizes.get(&cid).unwrap());
    }
    OrderedClusterLayout {
        ordered_cluster_ids,
        offsets,
        sizes,
    }
}

/// Convert an already-sorted `(offset, size)` layout into one request per span.
pub(crate) fn requests_from_ordered_spans(offsets: &[i64], sizes: &[i64]) -> Vec<IORequest> {
    offsets
        .iter()
        .zip(sizes.iter())
        .map(|(&offset, &size)| IORequest::new(offset, size))
        .collect()
}

/// Helper function to create separate requests (no merging)
pub fn create_separate_requests(
    cluster_ids: &[i32],
    cluster_offsets: &HashMap<i32, i64>,
    cluster_sizes: &HashMap<i32, i64>,
) -> Vec<IORequest> {
    let layout = collect_ordered_cluster_layout(cluster_ids, cluster_offsets, cluster_sizes);
    requests_from_ordered_spans(&layout.offsets, &layout.sizes)
}

/// Calculate metrics for a given I/O plan
pub fn calculate_plan_metrics(
    requests: &[IORequest],
    cluster_ids: &[i32],
    cluster_sizes: &HashMap<i32, i64>,
    cost_model: &CostModel,
) -> IOMetrics {
    let num_requests = requests.len();
    let total_bytes: i64 = requests.iter().map(|r| r.length).sum();
    let useful_bytes: i64 = cluster_ids
        .iter()
        .filter_map(|cid| cluster_sizes.get(cid))
        .sum();
    let wasted_bytes = total_bytes - useful_bytes;
    let waste_ratio = if total_bytes > 0 {
        wasted_bytes as f64 / total_bytes as f64
    } else {
        0.0
    };

    // Cost calculation
    let request_cost = (num_requests as f64) * cost_model.cost_per_request;
    let bandwidth_cost = (total_bytes as f64) * cost_model.cost_per_byte;
    let waste_cost = (wasted_bytes as f64) * cost_model.cost_per_wasted_byte;
    let total_cost = request_cost + bandwidth_cost + waste_cost;

    // Latency calculation
    let mut request_latency = 0.0;
    let mut bandwidth_latency = 0.0;
    for req in requests {
        let latency = cost_model.request_latency_ms(req.length);
        request_latency += cost_model.latency_per_request;
        bandwidth_latency += latency - cost_model.latency_per_request;
    }
    let total_latency = request_latency + bandwidth_latency;

    IOMetrics {
        num_requests,
        total_bytes,
        useful_bytes,
        wasted_bytes,
        waste_ratio,
        total_cost,
        request_cost,
        bandwidth_cost,
        waste_cost,
        total_latency,
        request_latency,
        bandwidth_latency,
    }
}

/// Separate strategy (no merging)
pub struct SeparateStrategy;

/// Public alias with a less implementation-shaped name.
pub type SeparateMerge = SeparateStrategy;

impl MergeStrategy for SeparateStrategy {
    fn generate_plan(
        &self,
        cluster_ids: &[i32],
        cluster_offsets: &HashMap<i32, i64>,
        cluster_sizes: &HashMap<i32, i64>,
        cost_model: &CostModel,
    ) -> IOPlan {
        let requests = create_separate_requests(cluster_ids, cluster_offsets, cluster_sizes);
        let metrics = calculate_plan_metrics(&requests, cluster_ids, cluster_sizes, cost_model);
        IOPlan::new(requests, metrics)
    }

    fn name(&self) -> &str {
        "Separate"
    }
}
