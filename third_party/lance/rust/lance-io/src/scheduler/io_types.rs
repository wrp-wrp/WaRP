// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Shared data types for I/O merge planning

use serde::{Deserialize, Serialize};

/// Represents one range read request
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct IORequest {
    /// Starting byte offset
    pub offset: i64,
    /// Number of bytes to read
    pub length: i64,
}

impl IORequest {
    pub fn new(offset: i64, length: i64) -> Self {
        Self { offset, length }
    }

    /// End offset (exclusive)
    pub fn end(&self) -> i64 {
        self.offset + self.length
    }
}

/// Metrics for an I/O plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOMetrics {
    /// Number of I/O requests
    pub num_requests: usize,
    /// Total bytes to read (including waste)
    pub total_bytes: i64,
    /// Useful bytes (actual data needed)
    pub useful_bytes: i64,
    /// Wasted bytes (total - useful)
    pub wasted_bytes: i64,
    /// Ratio of wasted bytes (wasted / total)
    pub waste_ratio: f64,

    // === Cost Dimension ===
    /// Total monetary cost
    pub total_cost: f64,
    /// Cost from requests
    pub request_cost: f64,
    /// Cost from bandwidth
    pub bandwidth_cost: f64,
    /// Cost from wasted bandwidth
    pub waste_cost: f64,

    // === Latency Dimension ===
    /// Total latency (ms)
    pub total_latency: f64,
    /// Latency from requests
    pub request_latency: f64,
    /// Latency from data transfer
    pub bandwidth_latency: f64,
}

impl Default for IOMetrics {
    fn default() -> Self {
        Self {
            num_requests: 0,
            total_bytes: 0,
            useful_bytes: 0,
            wasted_bytes: 0,
            waste_ratio: 0.0,
            total_cost: 0.0,
            request_cost: 0.0,
            bandwidth_cost: 0.0,
            waste_cost: 0.0,
            total_latency: 0.0,
            request_latency: 0.0,
            bandwidth_latency: 0.0,
        }
    }
}

/// A complete I/O plan and its evaluated metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IOPlan {
    /// List of merged I/O requests
    pub requests: Vec<IORequest>,
    /// Evaluated metrics for this plan
    pub metrics: IOMetrics,
}

impl IOPlan {
    pub fn new(requests: Vec<IORequest>, metrics: IOMetrics) -> Self {
        Self { requests, metrics }
    }
}
