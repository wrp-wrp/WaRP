//! Engineering helpers shared by strategies.
//!
//! Everything here is implementation detail — it speeds up strategy
//! implementations without changing their mathematical output. If you
//! are reading the paper and trying to find an algorithm, you want the
//! [`crate::strategy`] module; this one holds caches, prefix-sum tables,
//! and pre-filters.

pub mod piecewise;
pub mod prefix_sum;

pub use prefix_sum::RequestPrefixSums;
