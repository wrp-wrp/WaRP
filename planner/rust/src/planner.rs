//! Checked public planning entry point.
//!
//! Strategy implementations expose [`MergeStrategy::generate_plan`] as
//! an unchecked hot path for benchmarks and integrations that validate
//! once upstream. Most callers should use [`plan`]: it owns the complete
//! boundary contract while keeping strategy internals behind one small
//! interface.

use crate::{
    validate_requests, ConstraintError, CostModel, InputError, IoPlan, IoRequest, MergeStrategy,
    PlanConstraints, PlanInvariantError,
};

/// Validate a planning problem, run `strategy`, and validate its output.
///
/// This is the default application-facing seam. It checks request shape,
/// cost-model numerics, shared constraints, strategy support for those
/// constraints, and the returned plan's structural invariants. Call
/// [`MergeStrategy::generate_plan`] directly only after enforcing the
/// same preconditions elsewhere or when measuring the solver hot path.
pub fn plan<S: MergeStrategy + ?Sized>(
    strategy: &S,
    requests: &[IoRequest],
    cost: &CostModel,
    constraints: &PlanConstraints,
) -> Result<IoPlan, PlanningError> {
    validate_requests(requests).map_err(PlanningError::InvalidRequests)?;
    cost.validate().map_err(PlanningError::InvalidCostModel)?;
    constraints
        .validate()
        .map_err(PlanningError::InvalidConstraints)?;
    if constraints.is_active() && !strategy.supports_request_constraints() {
        return Err(PlanningError::UnsupportedRequestConstraints {
            strategy: strategy.name(),
        });
    }

    let output = strategy.generate_plan(requests, cost, constraints);
    output
        .check_invariants(requests)
        .map_err(PlanningError::InvalidPlan)?;
    if constraints.max_requests > 0 {
        let expected = constraints.request_cap(requests.len());
        let got = output.merged_requests.len();
        let violated = if constraints.exact_requests {
            got != expected
        } else {
            got > expected
        };
        if violated {
            return Err(PlanningError::RequestCountConstraintViolated {
                strategy: strategy.name(),
                got,
                expected,
                exact: constraints.exact_requests,
            });
        }
    }
    Ok(output)
}

/// Error returned by [`plan`].
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanningError {
    InvalidRequests(InputError),
    InvalidCostModel(String),
    InvalidConstraints(ConstraintError),
    UnsupportedRequestConstraints {
        strategy: &'static str,
    },
    InvalidPlan(PlanInvariantError),
    RequestCountConstraintViolated {
        strategy: &'static str,
        got: usize,
        expected: usize,
        exact: bool,
    },
}

impl std::fmt::Display for PlanningError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidRequests(error) => write!(f, "requests: {error}"),
            Self::InvalidCostModel(error) => write!(f, "cost model: {error}"),
            Self::InvalidConstraints(error) => write!(f, "constraints: {error}"),
            Self::UnsupportedRequestConstraints { strategy } => write!(
                f,
                "strategy '{strategy}' does not support max_requests/exact_requests"
            ),
            Self::InvalidPlan(error) => write!(f, "strategy returned an invalid plan: {error}"),
            Self::RequestCountConstraintViolated {
                strategy,
                got,
                expected,
                exact,
            } => {
                let relation = if *exact { "exactly" } else { "at most" };
                write!(
                    f,
                    "strategy '{strategy}' returned {got} requests, expected {relation} {expected}"
                )
            }
        }
    }
}

impl std::error::Error for PlanningError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidRequests(error) => Some(error),
            Self::InvalidConstraints(error) => Some(error),
            Self::InvalidPlan(error) => Some(error),
            Self::InvalidCostModel(_)
            | Self::UnsupportedRequestConstraints { .. }
            | Self::RequestCountConstraintViolated { .. } => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MainSolution, MergedRequest, Separate};

    fn requests() -> Vec<IoRequest> {
        vec![
            IoRequest {
                cluster_id: 0,
                offset: 0,
                length: 10,
            },
            IoRequest {
                cluster_id: 1,
                offset: 20,
                length: 10,
            },
        ]
    }

    #[test]
    fn checked_entry_point_accepts_valid_problem() {
        let output = plan(
            &MainSolution,
            &requests(),
            &CostModel::default(),
            &PlanConstraints::default(),
        )
        .unwrap();
        assert_eq!(output.assignments.len(), 2);
    }

    #[test]
    fn checked_entry_point_rejects_bad_input() {
        let mut invalid = requests();
        invalid[0].offset = -1;
        assert!(matches!(
            plan(
                &MainSolution,
                &invalid,
                &CostModel::default(),
                &PlanConstraints::default(),
            ),
            Err(PlanningError::InvalidRequests(
                InputError::NegativeOffset { .. }
            ))
        ));
    }

    #[test]
    fn checked_entry_point_rejects_ignored_constraints() {
        let constraints = PlanConstraints {
            max_requests: 1,
            exact_requests: false,
        };
        assert_eq!(
            plan(&Separate, &requests(), &CostModel::default(), &constraints,),
            Err(PlanningError::UnsupportedRequestConstraints {
                strategy: "separate"
            })
        );
    }

    #[test]
    fn checked_entry_point_rejects_invalid_cost_and_constraints() {
        let invalid_cost = CostModel {
            concurrent_parallelism: f64::NAN,
            ..CostModel::default()
        };
        assert!(matches!(
            plan(
                &MainSolution,
                &requests(),
                &invalid_cost,
                &PlanConstraints::default(),
            ),
            Err(PlanningError::InvalidCostModel(_))
        ));

        let invalid_constraints = PlanConstraints {
            max_requests: 0,
            exact_requests: true,
        };
        assert_eq!(
            plan(
                &MainSolution,
                &requests(),
                &CostModel::default(),
                &invalid_constraints,
            ),
            Err(PlanningError::InvalidConstraints(
                ConstraintError::ExactRequestsWithoutCap
            ))
        );
    }

    struct BrokenStrategy;

    impl MergeStrategy for BrokenStrategy {
        fn name(&self) -> &'static str {
            "broken"
        }

        fn generate_plan(
            &self,
            requests: &[IoRequest],
            _cost: &CostModel,
            _constraints: &PlanConstraints,
        ) -> IoPlan {
            IoPlan {
                merged_requests: vec![MergedRequest {
                    offset: 0,
                    length: 1,
                }],
                assignments: vec![0; requests.len()],
            }
        }
    }

    #[test]
    fn checked_entry_point_rejects_invalid_strategy_output() {
        assert!(matches!(
            plan(
                &BrokenStrategy,
                &requests(),
                &CostModel::default(),
                &PlanConstraints::default(),
            ),
            Err(PlanningError::InvalidPlan(
                PlanInvariantError::RequestNotCovered { .. }
            ))
        ));
    }

    struct LyingStrategy;

    impl MergeStrategy for LyingStrategy {
        fn name(&self) -> &'static str {
            "lying"
        }

        fn supports_request_constraints(&self) -> bool {
            true
        }

        fn generate_plan(
            &self,
            requests: &[IoRequest],
            _cost: &CostModel,
            _constraints: &PlanConstraints,
        ) -> IoPlan {
            IoPlan::one_per_request(requests)
        }
    }

    #[test]
    fn checked_entry_point_verifies_claimed_request_cap() {
        let constraints = PlanConstraints {
            max_requests: 1,
            exact_requests: false,
        };
        assert_eq!(
            plan(
                &LyingStrategy,
                &requests(),
                &CostModel::default(),
                &constraints,
            ),
            Err(PlanningError::RequestCountConstraintViolated {
                strategy: "lying",
                got: 2,
                expected: 1,
                exact: false,
            })
        );
    }
}
