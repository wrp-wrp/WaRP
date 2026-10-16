// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Cost model for I/O merge optimization
//!
//! Models the cost of S3 requests as: `w(bytes) = overhead + rate * bytes`
//! Supports both simple linear and piecewise-linear models.

use serde::{Deserialize, Serialize};

/// Cost model parameters for S3-like object storage
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CostModel {
    // === Cost Components ===
    /// Cost per I/O request (e.g., $0.0004 per 1000 S3 requests)
    pub cost_per_request: f64,
    /// Cost per byte transferred (e.g., $0.01 per GB for cross-region)
    pub cost_per_byte: f64,
    /// Additional cost penalty for reading unnecessary data
    pub cost_per_wasted_byte: f64,

    // === Latency Components ===
    /// Latency per I/O request (e.g., 169.3 ms for S3 GET)
    pub latency_per_request: f64,
    /// Latency per byte transferred (e.g., based on bandwidth)
    pub latency_per_byte: f64,
    /// Std dev of per-request latency (ms), for concurrent jitter correction
    pub latency_std: f64,
    /// Effective parallelism for concurrent requests (0 = assume full parallelism)
    pub concurrent_parallelism: f64,

    // === Piecewise Linear Model (Optional) ===
    /// Latency model type: "linear" or "piecewise_linear"
    pub latency_model: String,
    /// Breakpoints for piecewise model (in bytes)
    pub latency_pw_breaks_bytes: Vec<i64>,
    /// Overhead for each segment (in ms)
    pub latency_pw_overheads_ms: Vec<f64>,
    /// Rate for each segment (ms per byte)
    pub latency_pw_rates_ms_per_byte: Vec<f64>,

    // === Physical Constraints ===
    /// Maximum wasted/planned byte fraction for every merged request.
    /// Values greater than or equal to 1.0 disable this hard constraint.
    pub max_waste_ratio: f64,
    /// Maximum span for a single merged request (bytes)
    pub max_merge_span: i64,
    /// Optional upper bound on request count (<=0 means no explicit cap)
    pub max_requests: i32,
    /// If true and max_requests>0, enforce exactly K requests (not ≤K)
    pub exact_requests: bool,
}

impl Default for CostModel {
    fn default() -> Self {
        Self {
            // S3-like defaults
            cost_per_request: 0.0000004, // $0.0004 per 1000 requests
            cost_per_byte: 0.0,          // Free for same-region
            cost_per_wasted_byte: 0.0,
            latency_per_request: 169.3, // 169.3 ms per request
            latency_per_byte: 0.0,
            latency_std: 0.0,
            concurrent_parallelism: 0.0,
            latency_model: "linear".to_string(),
            latency_pw_breaks_bytes: vec![],
            latency_pw_overheads_ms: vec![],
            latency_pw_rates_ms_per_byte: vec![],
            max_waste_ratio: 0.5,             // Allow up to 50% waste
            max_merge_span: 10 * 1024 * 1024, // 10 MB max span
            max_requests: 0,
            exact_requests: false,
        }
    }
}

impl CostModel {
    /// Resolve the effective request-count cap for a plan over `cluster_count`
    /// sorted cluster spans.
    pub(crate) fn request_cap(&self, cluster_count: usize) -> usize {
        if self.max_requests > 0 {
            (self.max_requests as usize).min(cluster_count).max(1)
        } else {
            cluster_count
        }
    }

    /// Calculate request latency for a given byte size
    pub fn request_latency_ms(&self, bytes: i64) -> f64 {
        if self.use_piecewise_linear() {
            let parts = self.request_latency_parts_ms(bytes);
            parts.overhead_ms + parts.byte_ms
        } else {
            self.latency_per_request + (bytes as f64) * self.latency_per_byte
        }
    }

    /// Check if piecewise linear model is valid and should be used
    pub fn use_piecewise_linear(&self) -> bool {
        if self.latency_model != "piecewise_linear" {
            return false;
        }
        if self.latency_pw_overheads_ms.is_empty() || self.latency_pw_rates_ms_per_byte.is_empty() {
            return false;
        }
        if self.latency_pw_overheads_ms.len() != self.latency_pw_rates_ms_per_byte.len() {
            return false;
        }
        if self.latency_pw_overheads_ms.len() != self.latency_pw_breaks_bytes.len() + 1 {
            return false;
        }
        for i in 0..self.latency_pw_breaks_bytes.len() {
            if self.latency_pw_breaks_bytes[i] <= 0 {
                return false;
            }
            if i > 0 && self.latency_pw_breaks_bytes[i] <= self.latency_pw_breaks_bytes[i - 1] {
                return false;
            }
        }
        true
    }

    /// Get the segment index for piecewise linear model
    fn latency_segment_index(&self, bytes: i64) -> usize {
        if bytes <= 0 {
            return 0;
        }
        self.latency_pw_breaks_bytes
            .iter()
            .position(|&b| bytes <= b)
            .unwrap_or(self.latency_pw_breaks_bytes.len())
    }

    /// Get latency parts (overhead and byte cost) for piecewise model
    fn request_latency_parts_ms(&self, bytes: i64) -> LatencyParts {
        if !self.use_piecewise_linear() {
            let byte_ms = if bytes > 0 {
                (bytes as f64) * self.latency_per_byte
            } else {
                0.0
            };
            return LatencyParts {
                overhead_ms: self.latency_per_request,
                byte_ms,
            };
        }

        let idx = self
            .latency_segment_index(bytes)
            .min(self.latency_pw_overheads_ms.len() - 1);

        // Sanitize piecewise params on-the-fly
        let mut o = self.latency_pw_overheads_ms[0];
        let mut r = self.latency_pw_rates_ms_per_byte[0].max(0.0);

        if idx > 0 && !self.latency_pw_breaks_bytes.is_empty() {
            let mut w_at_break = o + r * (self.latency_pw_breaks_bytes[0] as f64);
            for i in 1..=idx {
                let r_i = self.latency_pw_rates_ms_per_byte[i].max(0.0);
                let prev_break = self.latency_pw_breaks_bytes[i - 1] as f64;
                let o_i = w_at_break - r_i * prev_break;
                o = o_i;
                r = r_i;
                if i < self.latency_pw_breaks_bytes.len() {
                    w_at_break = o + r * (self.latency_pw_breaks_bytes[i] as f64);
                }
            }
        }

        let byte_ms = if bytes > 0 { (bytes as f64) * r } else { 0.0 };

        LatencyParts {
            overhead_ms: o,
            byte_ms,
        }
    }
}

#[derive(Debug, Clone)]
struct LatencyParts {
    overhead_ms: f64,
    byte_ms: f64,
}

#[cfg(test)]
mod tests {
    use super::CostModel;

    #[test]
    fn piecewise_break_boundary_is_left_closed() {
        let cm = CostModel {
            latency_model: "piecewise_linear".to_string(),
            latency_pw_breaks_bytes: vec![1024, 2048],
            latency_pw_overheads_ms: vec![10.0, 20.0, 30.0],
            latency_pw_rates_ms_per_byte: vec![0.0, 0.0, 0.0],
            ..Default::default()
        };

        // C++ implementation uses lower_bound (<= break stays in left segment).
        assert_eq!(cm.latency_segment_index(1023), 0);
        assert_eq!(cm.latency_segment_index(1024), 0);
        assert_eq!(cm.latency_segment_index(1025), 1);
    }
}
