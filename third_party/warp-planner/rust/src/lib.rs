//! # warp-planner
//!
//! I/O merge planners for cluster-based vector search over cloud object storage.
//!
//! A *planner* takes a batch of per-cluster range-read requests into a
//! single cloud object and returns a plan — a list of merged range reads
//! that covers every input. The output is evaluated against a calibrated
//! cost model to minimise expected wall-clock latency.
//!
//! ## Quick start
//!
//! ```rust
//! use warp_planner::{
//!     plan, CostModel, IoRequest, PlanConstraints,
//!     strategy::{Separate, Greedy},
//! };
//!
//! let requests = vec![
//!     IoRequest { cluster_id: 0, offset: 0, length: 1024 },
//!     IoRequest { cluster_id: 1, offset: 1024, length: 1024 },
//!     IoRequest { cluster_id: 2, offset: 10 * 1024 * 1024, length: 1024 },
//! ];
//! let cost = CostModel::default();
//! let constraints = PlanConstraints::default();
//!
//! let greedy_plan = plan(&Greedy::default(), &requests, &cost, &constraints).unwrap();
//! let baseline   = plan(&Separate, &requests, &cost, &constraints).unwrap();
//!
//! assert!(greedy_plan.merged_requests.len() <= baseline.merged_requests.len());
//! ```
//!
//! ## Module map
//!
//! - [`planner`] — checked [`plan`] entry point and [`PlanningError`].
//! - [`cost_model`] — piecewise-linear S3 cost model and its calibration.
//! - [`io_plan`] — [`IoRequest`], [`IoPlan`], [`PlanMetrics`] and the
//!   [`evaluate`] free function.
//! - [`strategy`] — one file per merge strategy, scenario-named:
//!   [`Separate`] (baseline), [`Greedy`] (heuristic),
//!   [`PreMerge`] (gap+cap two-stage pre-coalesce, off by default),
//!   [`Serial`] (minimise `W`), [`Bottleneck`] (minimise `B`),
//!   [`MainSolution`] (finite-parallel `T_UB`). Each implements
//!   [`MergeStrategy`].
//!
//! A crate-private `internal/` module holds engineering helpers (prefix
//! sums, span caps) shared across strategies. The separation between
//! `strategy/` and `internal/` is deliberate: `strategy/` files hold the
//! paper's algorithms and their mathematical comments; `internal/` holds
//! implementation-detail optimisations that preserve strategy output
//! bit-for-bit. `internal` is not part of the public API.

pub mod calibrate;
pub mod cost_model;
pub(crate) mod internal;
pub mod io_plan;
pub mod plan_constraints;
pub mod planner;
#[cfg(feature = "probe")]
pub mod probe;
pub mod strategy;

pub use calibrate::{Calibrator, PiecewiseFit, Sample};
pub use cost_model::CostModel;
pub use io_plan::{
    evaluate, validate_requests, InputError, IoPlan, IoRequest, MergedRequest, PlanInvariantError,
    PlanMetrics,
};
pub use plan_constraints::{ConstraintError, PlanConstraints};
pub use planner::{plan, PlanningError};
pub use strategy::{Bottleneck, Greedy, MainSolution, MergeStrategy, PreMerge, Separate, Serial};
