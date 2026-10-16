// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Lance-IO merge strategy implementation
//!
//! This implements the current Lance-IO coalesce-split algorithm:
//! - Coalesce: Merge requests within 64KB of each other
//! - Split: Break requests larger than 16MB into smaller chunks

use super::cost_model::CostModel;
use super::io_types::{IOMetrics, IOPlan, IORequest};
use super::merge_strategy::{calculate_plan_metrics, create_separate_requests, MergeStrategy};
use std::collections::HashMap;

const COALESCE_GAP_BYTES: i64 = 64 * 1024; // 64KB
const MAX_COALESCED_REQUEST_BYTES: i64 = 16 * 1024 * 1024; // 16MB

/// Lance-IO coalesce-split merge strategy
pub struct LanceMerge;

impl LanceMerge {
    pub fn new() -> Self {
        Self
    }

    /// Coalesce adjacent requests within block_size distance
    fn coalesce_requests(requests: &[IORequest], block_size: i64) -> Vec<IORequest> {
        if requests.is_empty() {
            return vec![];
        }

        let mut merged = Vec::with_capacity(requests.len());
        let mut current = requests[0];

        for &next in &requests[1..] {
            // Check if close enough to merge
            if Self::is_close_together(current, next, block_size) {
                // Merge: extend current request to cover next
                let new_end = current.end().max(next.end());
                current.length = new_end - current.offset;
            } else {
                // Cannot merge: save current and start new
                merged.push(current);
                current = next;
            }
        }

        // Add the last request
        merged.push(current);
        merged
    }

    /// Split large requests into smaller chunks
    fn split_large_requests(requests: &[IORequest], max_iop_size: i64) -> Vec<IORequest> {
        let mut result = Vec::with_capacity(requests.len());

        for &req in requests {
            if req.length == 0 {
                result.push(req);
                continue;
            }

            if req.length <= max_iop_size {
                // No split needed
                result.push(req);
            } else {
                // Split into multiple requests
                let num_chunks = (req.length + max_iop_size - 1) / max_iop_size;
                let bytes_per_chunk = req.length / num_chunks;

                for i in 0..num_chunks {
                    let chunk_start = req.offset + i * bytes_per_chunk;
                    let chunk_length = if i == num_chunks - 1 {
                        // Last chunk gets the remainder
                        req.length - i * bytes_per_chunk
                    } else {
                        bytes_per_chunk
                    };

                    result.push(IORequest::new(chunk_start, chunk_length));
                }
            }
        }

        result
    }

    /// Check if two requests are close enough to merge
    fn is_close_together(r1: IORequest, r2: IORequest, block_size: i64) -> bool {
        // Two requests are close if the gap between them is <= block_size
        r2.offset <= r1.end() + block_size
    }
}

impl MergeStrategy for LanceMerge {
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

        let initial_requests =
            create_separate_requests(cluster_ids, cluster_offsets, cluster_sizes);

        // Phase 1: Coalesce adjacent requests
        let coalesced = Self::coalesce_requests(&initial_requests, COALESCE_GAP_BYTES);

        // Phase 2: Split large requests
        let final_requests = Self::split_large_requests(&coalesced, MAX_COALESCED_REQUEST_BYTES);

        // Compute metrics
        let metrics =
            calculate_plan_metrics(&final_requests, cluster_ids, cluster_sizes, cost_model);

        IOPlan::new(final_requests, metrics)
    }

    fn name(&self) -> &str {
        "Lance"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coalesce_close_requests() {
        let requests = vec![
            IORequest::new(0, 1000),
            IORequest::new(2000, 1000), // Gap of 1000 bytes
        ];

        let coalesced = LanceMerge::coalesce_requests(&requests, 64 * 1024);
        assert_eq!(coalesced.len(), 1);
        assert_eq!(coalesced[0].offset, 0);
        assert_eq!(coalesced[0].length, 3000);
    }

    #[test]
    fn test_no_coalesce_far_requests() {
        let requests = vec![
            IORequest::new(0, 1000),
            IORequest::new(100_000, 1000), // Gap > 64KB
        ];

        let coalesced = LanceMerge::coalesce_requests(&requests, 64 * 1024);
        assert_eq!(coalesced.len(), 2);
    }

    #[test]
    fn test_split_large_request() {
        let requests = vec![IORequest::new(0, 32 * 1024 * 1024)]; // 32MB

        let split = LanceMerge::split_large_requests(&requests, 16 * 1024 * 1024);
        assert_eq!(split.len(), 2);
        assert_eq!(split[0].length, 16 * 1024 * 1024);
        assert_eq!(split[1].length, 16 * 1024 * 1024);
    }
}
