//! Merge strategies.
//!
//! One submodule per paper strategy. Each submodule exports exactly one
//! struct (with any necessary parameters as public fields) that
//! implements [`MergeStrategy`]. No strategy imports any other strategy;
//! shared helpers live in a crate-private `internal/` module.
//!
//! Scenario-named modules (one scenario ↔ one file):
//!
//! ```text
//! strategy/
//!   separate.rs       no merging — baseline, one request per cluster
//!   greedy.rs         paper's Greedy Oracle — gap-threshold + span-cap heuristic
//!   serial.rs         serial execution — minimises total work W
//!   bottleneck.rs     unlimited parallelism — minimises the slowest merged request
//!   main_solution.rs  finite parallelism — paper's main T_UB planner
//! ```
//!
//! Strategies are trait objects, not generics — the trait's methods are
//! called once per query, so v-table overhead is negligible and picking
//! a strategy at runtime (by name, from a config) stays easy.

use crate::cost_model::CostModel;
use crate::io_plan::{IoPlan, IoRequest};
use crate::plan_constraints::PlanConstraints;

pub mod bottleneck;
pub mod greedy;
pub mod main_solution;
pub mod pre_merge;
pub mod separate;
pub mod serial;

pub use bottleneck::Bottleneck;
pub use greedy::Greedy;
pub use main_solution::MainSolution;
pub use pre_merge::PreMerge;
pub use separate::Separate;
pub use serial::Serial;

/// The common unchecked hot-path shape of every merge planner.
///
/// Implementations:
///
/// - take a sorted `&[IoRequest]` (sorted by `offset`),
/// - read from a `&CostModel`,
/// - read from shared [`PlanConstraints`],
/// - return a brand-new [`IoPlan`] that satisfies
///   [`IoPlan::check_invariants`].
///
/// Strategies must be pure — no I/O, no logging, no globals. The
/// reference implementation of each satisfies `Send + Sync` and makes no
/// allocations beyond the returned plan and scratch space bounded by
/// the input length. Application code should normally call [`crate::plan`]
/// so all boundary checks run before and after this interface.
pub trait MergeStrategy {
    /// Short identifier used for bench labels and tie-breaking.
    fn name(&self) -> &'static str;

    /// Whether this strategy honours [`PlanConstraints::max_requests`]
    /// and [`PlanConstraints::exact_requests`].
    ///
    /// The checked [`crate::plan`] entry point uses this capability to
    /// reject constraints that would otherwise be silently ignored.
    fn supports_request_constraints(&self) -> bool {
        false
    }

    /// Produce a plan that covers every request in `requests`.
    ///
    /// Preconditions (caller is responsible; strategies do not re-sort):
    ///
    /// - `requests` is sorted by `offset` in ascending order,
    /// - `requests[i].length > 0` for every `i`.
    ///
    /// Postconditions:
    ///
    /// - the returned plan covers every request,
    /// - `merged_requests` are sorted and disjoint,
    /// - any strategy-specific constraints it claims to honour are
    ///   reflected in the returned plan.
    ///
    /// [`IoPlan::check_invariants`] validates only the structural
    /// coverage/ordering guarantees above; it does not re-run
    /// planner-configuration checks because those are strategy- and
    /// configuration-dependent.
    fn generate_plan(
        &self,
        requests: &[IoRequest],
        cost: &CostModel,
        constraints: &PlanConstraints,
    ) -> IoPlan;
}
