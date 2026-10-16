//! Shared planning constraints.
//!
//! These are part of the planning problem, not the physical storage
//! cost model. They bound the shape of the returned plan without
//! changing the latency function `w(bytes)`.

use serde::{Deserialize, Serialize};

/// Shared planner constraints.
///
/// Strategy-private knobs stay on the strategy structs themselves
/// (for example [`crate::strategy::Greedy`]'s gap/span thresholds).
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct PlanConstraints {
    /// Optional upper bound on the number of merged requests. `0` means
    /// "no cap". The checked [`crate::plan`] entry point rejects negative
    /// values; the low-level strategy path retains a debug assertion.
    pub max_requests: i32,
    /// If `true` (and `max_requests > 0`), planners enforce *exactly*
    /// `max_requests` merged requests. If `false`, `<= max_requests` is
    /// enough.
    pub exact_requests: bool,
}

impl PlanConstraints {
    /// Validate constraints at an API boundary.
    ///
    /// The low-level strategy implementations assume this has already
    /// run. Prefer [`crate::plan`] unless the caller owns that validation
    /// contract itself.
    pub fn validate(&self) -> Result<(), ConstraintError> {
        if self.max_requests < 0 {
            return Err(ConstraintError::NegativeMaxRequests {
                max_requests: self.max_requests,
            });
        }
        if self.exact_requests && self.max_requests == 0 {
            return Err(ConstraintError::ExactRequestsWithoutCap);
        }
        Ok(())
    }

    /// Whether this value asks a strategy to constrain request count.
    #[inline]
    pub(crate) fn is_active(&self) -> bool {
        self.max_requests != 0 || self.exact_requests
    }

    /// Resolve the effective cap on request count given `cluster_count`
    /// candidate merged requests. Returns `0` iff `cluster_count == 0`
    /// (the cap cannot exceed the number of available inputs).
    pub fn request_cap(&self, cluster_count: usize) -> usize {
        debug_assert!(
            self.max_requests >= 0,
            "max_requests must be >= 0, got {}",
            self.max_requests
        );
        if cluster_count == 0 {
            return 0;
        }
        if self.max_requests > 0 {
            (self.max_requests as usize).clamp(1, cluster_count)
        } else {
            cluster_count
        }
    }
}

/// Why a [`PlanConstraints`] value cannot be applied.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum ConstraintError {
    /// `max_requests` must use `0` for unlimited or a positive cap.
    NegativeMaxRequests { max_requests: i32 },
    /// Exact request count has no meaning without a positive cap.
    ExactRequestsWithoutCap,
}

impl std::fmt::Display for ConstraintError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NegativeMaxRequests { max_requests } => {
                write!(f, "max_requests must be >= 0, got {max_requests}")
            }
            Self::ExactRequestsWithoutCap => {
                write!(f, "exact_requests=true requires max_requests > 0")
            }
        }
    }
}

impl std::error::Error for ConstraintError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn request_cap_zero_inputs_is_zero() {
        let c = PlanConstraints {
            max_requests: 16,
            exact_requests: false,
        };
        assert_eq!(c.request_cap(0), 0);
    }

    #[test]
    fn request_cap_uncapped_returns_cluster_count() {
        assert_eq!(PlanConstraints::default().request_cap(0), 0);
        assert_eq!(PlanConstraints::default().request_cap(7), 7);
    }

    #[test]
    fn request_cap_clamps_to_input_count() {
        let c = PlanConstraints {
            max_requests: 100,
            exact_requests: false,
        };
        assert_eq!(c.request_cap(3), 3);
        assert_eq!(c.request_cap(100), 100);
    }

    #[test]
    fn request_cap_honours_small_cap() {
        let c = PlanConstraints {
            max_requests: 4,
            exact_requests: false,
        };
        assert_eq!(c.request_cap(100), 4);
    }

    #[test]
    fn validate_rejects_negative_cap() {
        let c = PlanConstraints {
            max_requests: -1,
            exact_requests: false,
        };
        assert_eq!(
            c.validate(),
            Err(ConstraintError::NegativeMaxRequests { max_requests: -1 })
        );
    }

    #[test]
    fn validate_rejects_exact_without_cap() {
        let c = PlanConstraints {
            max_requests: 0,
            exact_requests: true,
        };
        assert_eq!(c.validate(), Err(ConstraintError::ExactRequestsWithoutCap));
    }
}
