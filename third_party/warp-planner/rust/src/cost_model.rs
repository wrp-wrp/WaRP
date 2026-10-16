//! Piecewise-linear cost model for cloud-object range reads.
//!
//! The model assumes the wall-clock cost of a single range-GET for
//! `bytes` bytes can be approximated as
//!
//! ```text
//!     w(bytes) = O + r * bytes
//! ```
//!
//! where both `O` (per-request overhead, milliseconds) and `r` (per-byte
//! rate, milliseconds/byte) may themselves depend on `bytes` in a
//! piecewise-linear way. The VecIO paper fits a two-segment piecewise
//! model with a break point around 3 MiB for same-region S3.
//!
//! The model is pure data — construct one and pass it around; every
//! strategy consumes it read-only.

use serde::{Deserialize, Serialize};

/// Calibrated cost model for a cloud-object storage endpoint.
///
/// Fields fall into three groups: latency components (used by every
/// strategy), objective context, and cost components (currently unused
/// by the library's strategies but preserved so the type round-trips
/// the on-disk schema the paper's calibration tools emit).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct CostModel {
    // ---- Latency model ---------------------------------------------------
    /// Latency model selector: `"linear"` or `"piecewise_linear"`.
    /// Invalid or inconsistent piecewise parameters silently fall back to
    /// linear; see [`CostModel::use_piecewise_linear`].
    pub latency_model: String,
    /// Per-request overhead used in the linear (non-piecewise) model, in
    /// milliseconds. Example: S3 same-region warm TCP connection ≈ 27 ms.
    pub latency_per_request: f64,
    /// Per-byte rate used in the linear (non-piecewise) model, in
    /// milliseconds per byte.
    pub latency_per_byte: f64,
    /// Piecewise break points in ascending order (bytes). A model with N
    /// breaks has N + 1 segments.
    pub latency_pw_breaks_bytes: Vec<i64>,
    /// Overhead per piecewise segment, millisecond. Length must equal
    /// `breaks.len() + 1`.
    pub latency_pw_overheads_ms: Vec<f64>,
    /// Rate per piecewise segment, millisecond per byte. Length must
    /// equal `breaks.len() + 1`.
    pub latency_pw_rates_ms_per_byte: Vec<f64>,
    /// Std-dev of per-request latency in milliseconds. Unused by the
    /// current strategies; kept for on-disk schema compatibility.
    pub latency_std: f64,

    // ---- Objective context -----------------------------------------------
    /// Effective parallelism when merged requests are executed
    /// concurrently. `0.0` means "assume unlimited parallelism".
    ///
    /// This is not part of the physical storage endpoint, but it does
    /// change the planner objective (`T_UB`). It stays here because the
    /// planner family currently uses `CostModel` as the read-only bundle
    /// for all objective coefficients.
    pub concurrent_parallelism: f64,

    // ---- Monetary cost components (round-trip only, not used in plans) ---
    /// Dollar cost per request.
    pub cost_per_request: f64,
    /// Dollar cost per byte transferred.
    pub cost_per_byte: f64,
    /// Dollar cost per wasted byte (e.g. read-amplification budget).
    pub cost_per_wasted_byte: f64,
}

impl Default for CostModel {
    /// S3 same-region defaults calibrated against `us-east-1` / Aug 2025:
    /// ~27 ms per request, no significant byte cost.
    fn default() -> Self {
        Self {
            latency_model: "linear".to_string(),
            latency_per_request: 27.0,
            latency_per_byte: 0.0,
            latency_pw_breaks_bytes: vec![],
            latency_pw_overheads_ms: vec![],
            latency_pw_rates_ms_per_byte: vec![],
            latency_std: 0.0,
            concurrent_parallelism: 0.0,

            cost_per_request: 0.0000004,
            cost_per_byte: 0.0,
            cost_per_wasted_byte: 0.0,
        }
    }
}

impl CostModel {
    /// Expected latency of a single range-GET of `bytes` bytes, in ms.
    pub fn request_latency_ms(&self, bytes: i64) -> f64 {
        let (overhead, byte_cost) = self.latency_parts_ms(bytes);
        overhead + byte_cost
    }

    /// Return `true` iff the piecewise parameters are well-formed and the
    /// model should be used. Inconsistencies silently route callers to
    /// the linear fallback on the unchecked path; [`CostModel::validate`]
    /// and the checked [`crate::plan`] entry point return the specific
    /// error instead.
    pub fn use_piecewise_linear(&self) -> bool {
        self.validate_piecewise().is_ok()
    }

    /// Strictly validate the model for API-boundary use.
    ///
    /// The selector must be `"linear"` or `"piecewise_linear"`.
    /// Linear scalars and non-latency numeric fields must be finite and
    /// non-negative, except `concurrent_parallelism`, where any finite
    /// value is accepted (`P <= 0` means full parallelism). If
    /// `latency_model == "piecewise_linear"`, the piecewise shape must
    /// be well-formed, the first segment's overhead must be finite and
    /// non-negative, later segment overheads must be finite, and every
    /// segment rate must be finite and non-negative. Later segment
    /// overheads may be negative because evaluation rewrites them to a
    /// continuity-preserving intercept at the previous break point.
    /// This is intentionally stricter than
    /// [`CostModel::use_piecewise_linear`], which only answers whether
    /// evaluation should take the piecewise path or silently use the
    /// linear fallback.
    pub fn validate(&self) -> Result<(), String> {
        if !matches!(self.latency_model.as_str(), "linear" | "piecewise_linear") {
            return Err(format!(
                "latency_model must be \"linear\" or \"piecewise_linear\", got {:?}",
                self.latency_model
            ));
        }
        validate_non_negative_finite("latency_per_request", self.latency_per_request)?;
        validate_non_negative_finite("latency_per_byte", self.latency_per_byte)?;
        validate_non_negative_finite("latency_std", self.latency_std)?;
        validate_finite("concurrent_parallelism", self.concurrent_parallelism)?;
        validate_non_negative_finite("cost_per_request", self.cost_per_request)?;
        validate_non_negative_finite("cost_per_byte", self.cost_per_byte)?;
        validate_non_negative_finite("cost_per_wasted_byte", self.cost_per_wasted_byte)?;
        if self.latency_model == "piecewise_linear" {
            self.validate_piecewise()?;
            for (i, &v) in self.latency_pw_overheads_ms.iter().enumerate() {
                if i == 0 {
                    validate_non_negative_finite(&format!("latency_pw_overheads_ms[{i}]"), v)?;
                } else {
                    validate_finite(&format!("latency_pw_overheads_ms[{i}]"), v)?;
                }
            }
            for (i, &v) in self.latency_pw_rates_ms_per_byte.iter().enumerate() {
                validate_non_negative_finite(&format!("latency_pw_rates_ms_per_byte[{i}]"), v)?;
            }
        }
        Ok(())
    }

    fn validate_piecewise(&self) -> Result<(), String> {
        if self.latency_model != "piecewise_linear" {
            return Err(format!(
                "latency_model is {:?}, not piecewise_linear",
                self.latency_model
            ));
        }
        let n = self.latency_pw_overheads_ms.len();
        if n == 0 {
            return Err("overheads_ms vector is empty".into());
        }
        if self.latency_pw_rates_ms_per_byte.len() != n {
            return Err(format!(
                "rates_ms_per_byte has len {}, expected {}",
                self.latency_pw_rates_ms_per_byte.len(),
                n,
            ));
        }
        if self.latency_pw_breaks_bytes.len() + 1 != n {
            return Err(format!(
                "breaks_bytes has len {}, expected {} (n_segments - 1)",
                self.latency_pw_breaks_bytes.len(),
                n - 1,
            ));
        }
        for (i, &b) in self.latency_pw_breaks_bytes.iter().enumerate() {
            if b <= 0 {
                return Err(format!("break {i} is non-positive"));
            }
            if i > 0 && b <= self.latency_pw_breaks_bytes[i - 1] {
                return Err(format!("break {i} is not strictly greater than previous"));
            }
        }
        Ok(())
    }

    /// Return `(overhead_ms, byte_cost_ms)`. Piecewise segments are
    /// stitched together so the function stays continuous at break points
    /// even if callers supply independent `(O_i, r_i)` per segment; the
    /// stitching lives in [`crate::internal::piecewise`] and is shared
    /// bit-for-bit with the planner's edge-latency cache.
    fn latency_parts_ms(&self, bytes: i64) -> (f64, f64) {
        if !self.use_piecewise_linear() {
            let byte_cost = if bytes > 0 {
                (bytes as f64) * self.latency_per_byte.max(0.0)
            } else {
                0.0
            };
            return (self.latency_per_request, byte_cost);
        }

        let segment = self.segment_index(bytes);
        let (overhead, rate) = crate::internal::piecewise::stitched_segments(
            &self.latency_pw_breaks_bytes,
            &self.latency_pw_overheads_ms,
            &self.latency_pw_rates_ms_per_byte,
        )
        .nth(segment)
        .expect("segment_index is in bounds for a validated piecewise model");

        let byte_cost = if bytes > 0 {
            (bytes as f64) * rate
        } else {
            0.0
        };
        (overhead, byte_cost)
    }

    /// Index of the piecewise segment `bytes` falls in. Break points are
    /// left-closed: exactly `break_k` stays in segment `k`, and the first
    /// byte past `break_k` moves into segment `k + 1`.
    fn segment_index(&self, bytes: i64) -> usize {
        crate::internal::piecewise::segment_index(&self.latency_pw_breaks_bytes, bytes)
    }
}

fn validate_non_negative_finite(name: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() || value < 0.0 {
        return Err(format!("{name} must be finite and >= 0, got {value}"));
    }
    Ok(())
}

fn validate_finite(name: &str, value: f64) -> Result<(), String> {
    if !value.is_finite() {
        return Err(format!("{name} must be finite, got {value}"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_model_is_pure_linear() {
        let cm = CostModel {
            latency_per_request: 10.0,
            latency_per_byte: 2.0,
            ..CostModel::default()
        };
        assert_eq!(cm.request_latency_ms(0), 10.0);
        assert_eq!(cm.request_latency_ms(5), 10.0 + 5.0 * 2.0);
    }

    #[test]
    fn linear_model_clamps_negative_byte_rate_to_overhead_only() {
        let cm = CostModel {
            latency_per_request: 10.0,
            latency_per_byte: -2.0,
            ..CostModel::default()
        };
        assert_eq!(cm.request_latency_ms(5), 10.0);
    }

    #[test]
    fn piecewise_break_boundary_is_left_closed() {
        let cm = CostModel {
            latency_model: "piecewise_linear".into(),
            latency_pw_breaks_bytes: vec![1024, 2048],
            latency_pw_overheads_ms: vec![10.0, 20.0, 30.0],
            latency_pw_rates_ms_per_byte: vec![0.0, 0.0, 0.0],
            ..CostModel::default()
        };
        assert_eq!(cm.segment_index(1023), 0);
        assert_eq!(cm.segment_index(1024), 0);
        assert_eq!(cm.segment_index(1025), 1);
        assert_eq!(cm.segment_index(2048), 1);
        assert_eq!(cm.segment_index(2049), 2);
    }

    #[test]
    fn piecewise_is_continuous_at_breakpoints() {
        // Two segments with different rates but continuous values.
        let cm = CostModel {
            latency_model: "piecewise_linear".into(),
            latency_pw_breaks_bytes: vec![1000],
            latency_pw_overheads_ms: vec![10.0, 999.0], // second overhead is rewritten
            latency_pw_rates_ms_per_byte: vec![0.01, 0.001],
            ..CostModel::default()
        };
        let at_break_from_left = cm.request_latency_ms(1000);
        let at_break_from_right = cm.request_latency_ms(1001);
        // Difference over a single byte must be tiny (on the order of r_2).
        assert!((at_break_from_right - at_break_from_left).abs() < 0.01);
    }

    #[test]
    fn invalid_piecewise_falls_back_to_linear() {
        let cm = CostModel {
            latency_model: "piecewise_linear".into(),
            latency_per_request: 100.0,
            latency_per_byte: 1.0,
            latency_pw_breaks_bytes: vec![1, 1], // non-monotonic: invalid
            latency_pw_overheads_ms: vec![0.0, 0.0, 0.0],
            latency_pw_rates_ms_per_byte: vec![0.0, 0.0, 0.0],
            ..CostModel::default()
        };
        assert!(!cm.use_piecewise_linear());
        assert_eq!(cm.request_latency_ms(10), 100.0 + 10.0 * 1.0);
    }

    #[test]
    fn validate_rejects_non_finite_and_negative_linear_scalars() {
        for (field, cm) in [
            (
                "latency_per_request",
                CostModel {
                    latency_per_request: f64::NAN,
                    ..CostModel::default()
                },
            ),
            (
                "latency_per_byte",
                CostModel {
                    latency_per_byte: -0.001,
                    ..CostModel::default()
                },
            ),
        ] {
            let err = cm.validate().unwrap_err();
            assert!(err.contains(field), "{err}");
        }
    }

    #[test]
    fn validate_rejects_unknown_model_and_non_finite_objective() {
        let unknown = CostModel {
            latency_model: "quadratic".to_string(),
            ..CostModel::default()
        };
        assert!(unknown.validate().unwrap_err().contains("latency_model"));

        let non_finite = CostModel {
            concurrent_parallelism: f64::NAN,
            ..CostModel::default()
        };
        assert!(non_finite
            .validate()
            .unwrap_err()
            .contains("concurrent_parallelism"));
    }

    #[test]
    fn validate_rejects_non_finite_and_negative_piecewise_values() {
        let cm = CostModel {
            latency_model: "piecewise_linear".into(),
            latency_pw_breaks_bytes: vec![1024],
            latency_pw_overheads_ms: vec![10.0, f64::INFINITY],
            latency_pw_rates_ms_per_byte: vec![0.0, 0.001],
            ..CostModel::default()
        };
        let err = cm.validate().unwrap_err();
        assert!(err.contains("latency_pw_overheads_ms[1]"), "{err}");

        let cm = CostModel {
            latency_model: "piecewise_linear".into(),
            latency_pw_breaks_bytes: vec![1024],
            latency_pw_overheads_ms: vec![10.0, 20.0],
            latency_pw_rates_ms_per_byte: vec![0.0, -0.001],
            ..CostModel::default()
        };
        let err = cm.validate().unwrap_err();
        assert!(err.contains("latency_pw_rates_ms_per_byte[1]"), "{err}");
    }

    #[test]
    fn validate_accepts_negative_later_piecewise_overhead() {
        // Continuity-adjusted piecewise fits can have a negative
        // second-segment intercept while remaining non-negative over
        // the segment's actual byte domain.
        let cm = CostModel {
            latency_model: "piecewise_linear".into(),
            latency_pw_breaks_bytes: vec![4_194_304],
            latency_pw_overheads_ms: vec![10.0, -31.5236096],
            latency_pw_rates_ms_per_byte: vec![1e-7, 1e-5],
            ..CostModel::default()
        };
        cm.validate().unwrap();
        assert!(cm.request_latency_ms(8 * 1024 * 1024) > 0.0);
    }

    #[test]
    fn validate_reports_specific_errors() {
        let cm = CostModel {
            latency_model: "piecewise_linear".into(),
            latency_pw_breaks_bytes: vec![],
            latency_pw_overheads_ms: vec![],
            latency_pw_rates_ms_per_byte: vec![],
            ..CostModel::default()
        };
        let err = cm.validate().unwrap_err();
        assert!(err.contains("overheads_ms"));
    }
}
