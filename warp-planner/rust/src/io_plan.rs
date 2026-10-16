//! Input and output types for merge planners.
//!
//! A planner consumes a slice of [`IoRequest`]s (one per cluster) and
//! returns an [`IoPlan`] — a list of [`MergedRequest`]s together with an
//! assignment that maps each input cluster to the merged request that
//! covers it. The plan is then passed to [`evaluate`] with the
//! [`CostModel`] to get a [`PlanMetrics`] summary.
//!
//! Types are plain data, serde-friendly, and cheap to copy. No references,
//! no lifetimes, no iterators buried inside — a planner that returns an
//! `IoPlan` owns it.

use serde::{Deserialize, Serialize};

use crate::cost_model::CostModel;

/// A single range-read request for a cluster in the source object.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IoRequest {
    /// Caller-defined identifier; the library never inspects it beyond
    /// round-tripping through `assignments`.
    pub cluster_id: i32,
    /// Byte offset of the cluster's first useful byte in the source
    /// object.
    pub offset: i64,
    /// Number of useful bytes the caller wants.
    pub length: i64,
}

impl IoRequest {
    /// Byte after the last useful byte. Saturates at `i64::MAX` if
    /// the caller hands in an adversarial `offset + length` that would
    /// overflow; well-formed inputs (validated via [`validate_requests`])
    /// never trigger the saturation.
    #[inline]
    pub fn end(&self) -> i64 {
        self.offset.saturating_add(self.length)
    }
}

/// A merged range-read request the planner will issue to object storage.
///
/// Merged requests are disjoint and sorted by `offset` in every plan
/// returned by this library.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct MergedRequest {
    /// Start offset of the merged range, in bytes.
    pub offset: i64,
    /// Size of the merged range, in bytes. Always ≥ the sum of covered
    /// `IoRequest.length`s (equality iff no waste).
    pub length: i64,
}

impl MergedRequest {
    #[inline]
    pub fn end(&self) -> i64 {
        self.offset.saturating_add(self.length)
    }
}

/// An ordered, disjoint partition of the input requests into merged
/// range reads.
///
/// `assignments[i]` is the index into `merged_requests` that covers
/// input request `i`. Coverage is a library invariant:
///
/// - `merged_requests[assignments[i]].offset <= requests[i].offset`
/// - `merged_requests[assignments[i]].end()   >= requests[i].end()`
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IoPlan {
    /// Merged requests in ascending `offset` order, disjoint.
    pub merged_requests: Vec<MergedRequest>,
    /// Per-input assignment. `len()` == input count.
    pub assignments: Vec<usize>,
}

impl IoPlan {
    /// Empty plan — no requests, no assignments. Useful as a zero value.
    pub fn empty() -> Self {
        Self {
            merged_requests: Vec::new(),
            assignments: Vec::new(),
        }
    }

    /// The identity plan: one merged request per input, zero waste.
    /// Used by the Separate baseline and many tests.
    pub fn one_per_request(requests: &[IoRequest]) -> Self {
        let merged_requests = requests
            .iter()
            .map(|r| MergedRequest {
                offset: r.offset,
                length: r.length,
            })
            .collect();
        let assignments = (0..requests.len()).collect();
        Self {
            merged_requests,
            assignments,
        }
    }

    /// Assert structural plan invariants and return them as structured
    /// errors.
    ///
    /// This validates only shape/coverage properties of the materialised
    /// plan: assignment lengths, merged-range ordering/disjointness, and
    /// per-request coverage. Strategy-specific policy constraints such as
    /// request-count caps or planner-private knobs are not revalidated
    /// here because `IoPlan` does not retain the originating planner
    /// configuration.
    pub fn check_invariants(&self, requests: &[IoRequest]) -> Result<(), PlanInvariantError> {
        if self.assignments.len() != requests.len() {
            return Err(PlanInvariantError::AssignmentsLengthMismatch {
                got: self.assignments.len(),
                expected: requests.len(),
            });
        }
        for (i, m) in self.merged_requests.iter().enumerate() {
            if m.offset < 0 {
                return Err(PlanInvariantError::NegativeMergedOffset {
                    merged_index: i,
                    offset: m.offset,
                });
            }
            if m.length <= 0 {
                return Err(PlanInvariantError::NonPositiveMergedLength {
                    merged_index: i,
                    length: m.length,
                });
            }
            if m.offset.checked_add(m.length).is_none() {
                return Err(PlanInvariantError::MergedOffsetOverflow {
                    merged_index: i,
                    offset: m.offset,
                    length: m.length,
                });
            }
            if i > 0 {
                let prev = &self.merged_requests[i - 1];
                if m.offset < prev.end() {
                    return Err(PlanInvariantError::MergedRequestsOverlap {
                        left_index: i - 1,
                        right_index: i,
                    });
                }
            }
        }
        let mut assigned = vec![false; self.merged_requests.len()];
        for (i, &a) in self.assignments.iter().enumerate() {
            if a >= self.merged_requests.len() {
                return Err(PlanInvariantError::AssignmentOutOfRange {
                    request_index: i,
                    assignment: a,
                });
            }
            assigned[a] = true;
            let m = &self.merged_requests[a];
            let r = &requests[i];
            if m.offset > r.offset || m.end() < r.end() {
                return Err(PlanInvariantError::RequestNotCovered {
                    request_index: i,
                    assignment: a,
                });
            }
        }
        if let Some(merged_index) = assigned.iter().position(|used| !used) {
            return Err(PlanInvariantError::UnassignedMergedRequest { merged_index });
        }
        Ok(())
    }
}

/// Structured error returned by [`IoPlan::check_invariants`].
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum PlanInvariantError {
    AssignmentsLengthMismatch {
        got: usize,
        expected: usize,
    },
    NegativeMergedOffset {
        merged_index: usize,
        offset: i64,
    },
    NonPositiveMergedLength {
        merged_index: usize,
        length: i64,
    },
    MergedOffsetOverflow {
        merged_index: usize,
        offset: i64,
        length: i64,
    },
    MergedRequestsOverlap {
        left_index: usize,
        right_index: usize,
    },
    AssignmentOutOfRange {
        request_index: usize,
        assignment: usize,
    },
    RequestNotCovered {
        request_index: usize,
        assignment: usize,
    },
    UnassignedMergedRequest {
        merged_index: usize,
    },
}

impl std::fmt::Display for PlanInvariantError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::AssignmentsLengthMismatch { got, expected } => {
                write!(f, "assignments.len()={got}, expected {expected}")
            }
            Self::NegativeMergedOffset {
                merged_index,
                offset,
            } => write!(
                f,
                "merged_requests[{merged_index}].offset = {offset} (must be >= 0)"
            ),
            Self::NonPositiveMergedLength {
                merged_index,
                length,
            } => {
                write!(
                    f,
                    "merged_requests[{merged_index}].length = {length} (must be > 0)"
                )
            }
            Self::MergedOffsetOverflow {
                merged_index,
                offset,
                length,
            } => write!(
                f,
                "merged_requests[{merged_index}].offset ({offset}) + length ({length}) overflows i64"
            ),
            Self::MergedRequestsOverlap {
                left_index,
                right_index,
            } => {
                write!(
                    f,
                    "merged_requests[{left_index}] and [{right_index}] overlap"
                )
            }
            Self::AssignmentOutOfRange {
                request_index,
                assignment,
            } => {
                write!(
                    f,
                    "assignments[{request_index}] = {assignment} out of range"
                )
            }
            Self::RequestNotCovered {
                request_index,
                assignment,
            } => {
                write!(
                    f,
                    "request {request_index} not covered by merged_requests[{assignment}]"
                )
            }
            Self::UnassignedMergedRequest { merged_index } => {
                write!(f, "merged_requests[{merged_index}] is not assigned to any request")
            }
        }
    }
}

impl std::error::Error for PlanInvariantError {}

/// A summary of a plan's predicted cost and waste.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct PlanMetrics {
    /// Total bytes the plan will fetch (sum of `merged_requests[i].length`).
    pub total_bytes: i64,
    /// Bytes the caller actually asked for (sum of `requests[i].length`).
    pub useful_bytes: i64,
    /// `total_bytes - useful_bytes` — bytes fetched that nobody needed.
    pub wasted_bytes: i64,
    /// Number of merged requests in the plan.
    pub request_count: usize,
    /// Sum of `cost_model.request_latency_ms(merged.length)` across
    /// merged requests — the "serial" predicted latency.
    pub predicted_latency_ms_serial: f64,
    /// Max of `cost_model.request_latency_ms(merged.length)` — the
    /// "fully concurrent" predicted latency. Meaningful when effective
    /// parallelism ≥ `request_count`.
    pub predicted_latency_ms_concurrent: f64,
}

impl PlanMetrics {
    /// Finite-parallelism latency upper bound
    /// `T_UB(P) = W/P + (1 − 1/P) · B`, where `W` is
    /// [`Self::predicted_latency_ms_serial`] and `B` is
    /// [`Self::predicted_latency_ms_concurrent`].
    ///
    /// This is the objective [`crate::strategy::MainSolution`]
    /// minimises; both share one coefficient implementation, so the
    /// degenerate cases agree by construction: `P ≤ 0` means full
    /// parallelism (`T = B`) and fractional `P` in `(0, 1)` clamps to
    /// serial (`T = W`). Prefer this over hand-writing the formula —
    /// the `P ≤ 0` convention is easy to get wrong.
    pub fn t_ub_ms(&self, parallelism: f64) -> f64 {
        let (coeff_w, coeff_b) = crate::internal::piecewise::objective_coefficients(parallelism);
        coeff_w * self.predicted_latency_ms_serial + coeff_b * self.predicted_latency_ms_concurrent
    }
}

/// Evaluate a plan against the cost model. Pure function; every
/// planner-evaluation path goes through this so strategies can be
/// compared apples-to-apples.
pub fn evaluate(requests: &[IoRequest], plan: &IoPlan, cost: &CostModel) -> PlanMetrics {
    let mut total_bytes = 0i64;
    let mut serial = 0f64;
    let mut bottleneck = 0f64;
    for m in &plan.merged_requests {
        total_bytes += m.length;
        let w = cost.request_latency_ms(m.length);
        serial += w;
        if w > bottleneck {
            bottleneck = w;
        }
    }
    let useful: i64 = requests.iter().map(|r| r.length).sum();
    PlanMetrics {
        total_bytes,
        useful_bytes: useful,
        wasted_bytes: total_bytes - useful,
        request_count: plan.merged_requests.len(),
        predicted_latency_ms_serial: serial,
        predicted_latency_ms_concurrent: bottleneck,
    }
}

/// Why a batch of [`IoRequest`]s cannot be planned.
///
/// Strategies assume sorted, disjoint, non-negative-offset,
/// positive-length requests whose `offset + length` fits in `i64`.
/// Callers that cannot guarantee this should run [`validate_requests`]
/// first and surface the error to the user instead of feeding garbage
/// into `generate_plan`.
#[derive(Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum InputError {
    /// `requests[index].offset` was < 0.
    NegativeOffset { index: usize, offset: i64 },
    /// `requests[index].length` was ≤ 0.
    NonPositiveLength { index: usize, length: i64 },
    /// `requests[index].offset + length` would overflow `i64`.
    OffsetOverflow {
        index: usize,
        offset: i64,
        length: i64,
    },
    /// `requests[index].offset < requests[index-1].offset` — input is
    /// not sorted by offset.
    NotSorted { index: usize },
    /// `requests[left].end() > requests[right].offset` — two requests
    /// overlap, which the library's strategies do not accept.
    Overlap { left: usize, right: usize },
}

impl std::fmt::Display for InputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NegativeOffset { index, offset } => {
                write!(f, "requests[{index}].offset = {offset} (must be >= 0)")
            }
            Self::NonPositiveLength { index, length } => {
                write!(f, "requests[{index}].length = {length} (must be > 0)")
            }
            Self::OffsetOverflow {
                index,
                offset,
                length,
            } => write!(
                f,
                "requests[{index}].offset ({offset}) + length ({length}) overflows i64"
            ),
            Self::NotSorted { index } => write!(
                f,
                "requests[{index}].offset < requests[{}].offset (not sorted)",
                index - 1
            ),
            Self::Overlap { left, right } => {
                write!(f, "requests[{left}] and requests[{right}] overlap")
            }
        }
    }
}

impl std::error::Error for InputError {}

/// Verify that a request batch satisfies the preconditions every
/// strategy relies on: non-negative offsets, positive lengths, no
/// `offset + length` overflow, ascending `offset`, and no overlap
/// between consecutive requests.
///
/// Call this at your API boundary when you are not certain the caller
/// produced well-formed input; the Rust strategies leave the check to
/// the caller for the hot path, and the PyO3 binding runs it
/// unconditionally.
pub fn validate_requests(requests: &[IoRequest]) -> Result<(), InputError> {
    let mut prev_end: i64 = i64::MIN;
    for (i, r) in requests.iter().enumerate() {
        if r.offset < 0 {
            return Err(InputError::NegativeOffset {
                index: i,
                offset: r.offset,
            });
        }
        if r.length <= 0 {
            return Err(InputError::NonPositiveLength {
                index: i,
                length: r.length,
            });
        }
        if r.offset.checked_add(r.length).is_none() {
            return Err(InputError::OffsetOverflow {
                index: i,
                offset: r.offset,
                length: r.length,
            });
        }
        if i > 0 {
            if r.offset < requests[i - 1].offset {
                return Err(InputError::NotSorted { index: i });
            }
            if r.offset < prev_end {
                return Err(InputError::Overlap {
                    left: i - 1,
                    right: i,
                });
            }
        }
        prev_end = r.end();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(id: i32, offset: i64, length: i64) -> IoRequest {
        IoRequest {
            cluster_id: id,
            offset,
            length,
        }
    }

    #[test]
    fn one_per_request_is_always_valid() {
        let rs = vec![req(0, 0, 1), req(1, 10, 5), req(2, 100, 1)];
        let plan = IoPlan::one_per_request(&rs);
        assert!(plan.check_invariants(&rs).is_ok());
    }

    #[test]
    fn check_invariants_catches_overlap() {
        let rs = vec![req(0, 0, 10), req(1, 20, 10)];
        let bad = IoPlan {
            merged_requests: vec![
                MergedRequest {
                    offset: 0,
                    length: 15,
                },
                MergedRequest {
                    offset: 10,
                    length: 20,
                },
            ],
            assignments: vec![0, 1],
        };
        assert!(matches!(
            bad.check_invariants(&rs),
            Err(PlanInvariantError::MergedRequestsOverlap { .. })
        ));
    }

    #[test]
    fn check_invariants_catches_missing_coverage() {
        let rs = vec![req(0, 0, 10), req(1, 100, 10)];
        let bad = IoPlan {
            merged_requests: vec![MergedRequest {
                offset: 0,
                length: 10,
            }],
            assignments: vec![0, 0], // request 1 at offset 100 isn't covered
        };
        assert!(matches!(
            bad.check_invariants(&rs),
            Err(PlanInvariantError::RequestNotCovered { .. })
        ));
    }

    #[test]
    fn check_invariants_catches_negative_merged_offset() {
        let rs = vec![req(0, 0, 10)];
        let bad = IoPlan {
            merged_requests: vec![MergedRequest {
                offset: -1,
                length: 11,
            }],
            assignments: vec![0],
        };
        assert!(matches!(
            bad.check_invariants(&rs),
            Err(PlanInvariantError::NegativeMergedOffset { .. })
        ));
    }

    #[test]
    fn check_invariants_catches_merged_offset_overflow() {
        let rs = vec![req(0, i64::MAX - 10, 10)];
        let bad = IoPlan {
            merged_requests: vec![MergedRequest {
                offset: i64::MAX - 10,
                length: 100,
            }],
            assignments: vec![0],
        };
        assert!(matches!(
            bad.check_invariants(&rs),
            Err(PlanInvariantError::MergedOffsetOverflow { .. })
        ));
    }

    #[test]
    fn check_invariants_catches_unassigned_merged_request() {
        let rs = vec![req(0, 0, 10)];
        let bad = IoPlan {
            merged_requests: vec![
                MergedRequest {
                    offset: 0,
                    length: 10,
                },
                MergedRequest {
                    offset: 20,
                    length: 10,
                },
            ],
            assignments: vec![0],
        };
        assert_eq!(
            bad.check_invariants(&rs),
            Err(PlanInvariantError::UnassignedMergedRequest { merged_index: 1 })
        );
    }

    #[test]
    fn evaluate_matches_by_hand() {
        let rs = vec![req(0, 0, 100), req(1, 200, 100)];
        // One big merged request spanning both.
        let plan = IoPlan {
            merged_requests: vec![MergedRequest {
                offset: 0,
                length: 300,
            }],
            assignments: vec![0, 0],
        };
        let cost = CostModel {
            latency_per_request: 27.0,
            latency_per_byte: 0.0,
            ..CostModel::default()
        };
        let m = evaluate(&rs, &plan, &cost);
        assert_eq!(m.request_count, 1);
        assert_eq!(m.useful_bytes, 200);
        assert_eq!(m.total_bytes, 300);
        assert_eq!(m.wasted_bytes, 100);
        assert_eq!(m.predicted_latency_ms_serial, 27.0);
        assert_eq!(m.predicted_latency_ms_concurrent, 27.0);
    }

    #[test]
    fn t_ub_covers_degenerate_parallelism() {
        let m = PlanMetrics {
            total_bytes: 0,
            useful_bytes: 0,
            wasted_bytes: 0,
            request_count: 4,
            predicted_latency_ms_serial: 100.0,
            predicted_latency_ms_concurrent: 40.0,
        };
        assert_eq!(m.t_ub_ms(0.0), 40.0); // P ≤ 0 → full parallel, T = B
        assert_eq!(m.t_ub_ms(1.0), 100.0); // serial, T = W
        assert_eq!(m.t_ub_ms(0.5), 100.0); // fractional clamps to serial
        assert!((m.t_ub_ms(4.0) - (100.0 / 4.0 + 0.75 * 40.0)).abs() < 1e-12);
    }

    #[test]
    fn validate_accepts_well_formed_batch() {
        let rs = vec![req(0, 0, 10), req(1, 10, 10), req(2, 100, 10)];
        assert!(validate_requests(&rs).is_ok());
        assert!(validate_requests(&[]).is_ok());
    }

    #[test]
    fn validate_rejects_non_positive_length() {
        let rs = vec![req(0, 0, 10), req(1, 10, 0)];
        assert!(matches!(
            validate_requests(&rs),
            Err(InputError::NonPositiveLength {
                index: 1,
                length: 0
            })
        ));
    }

    #[test]
    fn validate_rejects_negative_offset() {
        let rs = vec![req(0, -1, 10)];
        assert!(matches!(
            validate_requests(&rs),
            Err(InputError::NegativeOffset {
                index: 0,
                offset: -1
            })
        ));
    }

    #[test]
    fn validate_rejects_offset_overflow() {
        let rs = vec![req(0, i64::MAX - 10, 100)];
        assert!(matches!(
            validate_requests(&rs),
            Err(InputError::OffsetOverflow { index: 0, .. })
        ));
    }

    #[test]
    fn validate_rejects_unsorted_input() {
        let rs = vec![req(0, 100, 10), req(1, 50, 10)];
        assert!(matches!(
            validate_requests(&rs),
            Err(InputError::NotSorted { index: 1 })
        ));
    }

    #[test]
    fn validate_rejects_overlap() {
        let rs = vec![req(0, 0, 20), req(1, 10, 10)];
        assert!(matches!(
            validate_requests(&rs),
            Err(InputError::Overlap { left: 0, right: 1 })
        ));
    }

    #[test]
    fn request_end_saturates_on_overflow() {
        let r = IoRequest {
            cluster_id: 0,
            offset: i64::MAX - 10,
            length: 100,
        };
        assert_eq!(r.end(), i64::MAX);
    }
}
