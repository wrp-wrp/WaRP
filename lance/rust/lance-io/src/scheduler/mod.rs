// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! I/O Merge Optimization for Cloud Object Storage
//!
//! This module provides advanced I/O request merging strategies optimized for
//! cloud object storage (S3, GCS, Azure) where request overhead dominates latency.
//!
//! # Background
//!
//! Traditional vector databases rely on memory speed, but cloud-native architectures
//! separate compute and storage. When vector indices (e.g., IVF) are stored on S3,
//! a single query can trigger hundreds of small Range GET requests. The high Time-To-First-Byte
//! (TTFB) of S3 means most time is spent on request establishment rather than data transfer.
//!
//! # Key Concepts
//!
//! - **Cost Model**: Captures S3 request cost: `w(bytes) = O + f(bytes)` where O is fixed overhead
//! - **Merge Strategies**: Different algorithms for combining nearby requests
//! - **Cluster Reordering**: Spatial locality optimization to improve merge effectiveness
//!
//! # Strategies
//!
//! - `Separate`: No merging (baseline)
//! - `Greedy`: Simple heuristic-based merging
//! - `DP`: Dynamic programming for serial optimization (minimize total cost W)
//! - `Concurrent`: Min-max grouping for full parallelism (minimize bottleneck B)
//! - `BalancedUB`: Finite parallelism optimization: `T = W/P + (1-1/P)*B`
//! - `Lance`: Current Lance-IO coalesce-split algorithm (64KB threshold, 16MB max)

// Re-export the original scheduler module
mod scheduler_impl;
pub use scheduler_impl::*;

// I/O merge optimization submodules
pub mod balanced_ub;
pub mod concurrent_merge;
pub mod cost_model;
pub mod greedy_merge;
pub mod io_types;
pub mod lance_merge;
pub mod merge_strategy;
pub mod reorder;
pub mod serial_dp;

pub use balanced_ub::{BalancedUBMerge, FiniteParallelMerge, VecioMergePlanner};
pub use concurrent_merge::{ConcurrentMerge, ConcurrentSweepResult};
pub use cost_model::CostModel;
pub use greedy_merge::{
    FrontierGreedyMerge, GreedyFrontierMerge, GreedyMerge, GreedyMergeCapped, SpanCappedGreedyMerge,
};
pub use io_types::{IOMetrics, IOPlan, IORequest};
pub use lance_merge::LanceMerge;
pub use merge_strategy::{MergeStrategy, MergeStrategyType, SeparateMerge};
pub use reorder::{reorder_clusters, ReorderMethod};
pub use serial_dp::{DPSweepResult, OptimalSerialMerge, SerialDP};
