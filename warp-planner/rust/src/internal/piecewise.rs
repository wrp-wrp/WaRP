//! Single source of truth for piecewise-linear latency stitching and
//! the `T_UB` objective coefficients.
//!
//! The evaluation path ([`crate::cost_model::CostModel`]) and the
//! planning path (`MainSolution`'s edge-latency cache) must agree
//! **bit-for-bit** on the stitched piecewise function — any divergence
//! makes the planner optimise a different objective than the one
//! [`crate::io_plan::evaluate`] reports. Both now call the helpers
//! below; `tests/algo_fuzz.rs` additionally guards the agreement
//! end-to-end.

/// Index of the piecewise segment `bytes` falls in. Break points are
/// left-closed: exactly `break_k` stays in segment `k`, and the first
/// byte past `break_k` moves into segment `k + 1`. Non-positive
/// `bytes` map to segment 0.
#[inline]
pub(crate) fn segment_index(breaks: &[i64], bytes: i64) -> usize {
    let bytes = bytes.max(0);
    breaks.partition_point(|&b| b < bytes)
}

/// Iterate over continuity-stitched `(overhead_ms, rate_ms_per_byte)`
/// pairs, one per segment, in segment order.
///
/// Callers supply independent per-segment `(O_i, r_i)`; each `O_i`
/// (for `i ≥ 1`) is rewritten so that `O_i + r_i · break_{i-1}` equals
/// the previous segment's value at the break, keeping the function
/// continuous. Rates are clamped to `≥ 0`.
///
/// Preconditions (guaranteed by
/// [`crate::cost_model::CostModel::use_piecewise_linear`]):
/// `overheads.len() == rates.len() == breaks.len() + 1`.
pub(crate) fn stitched_segments<'a>(
    breaks: &'a [i64],
    overheads: &'a [f64],
    rates: &'a [f64],
) -> impl Iterator<Item = (f64, f64)> + 'a {
    let mut value_at_break = f64::NAN;
    (0..overheads.len()).map(move |i| {
        if i == 0 {
            let o = overheads[0];
            let r = rates[0].max(0.0);
            if !breaks.is_empty() {
                value_at_break = o + r * (breaks[0] as f64);
            }
            (o, r)
        } else {
            let r = rates[i].max(0.0);
            let o = value_at_break - r * (breaks[i - 1] as f64);
            if i < breaks.len() {
                value_at_break = o + r * (breaks[i] as f64);
            }
            (o, r)
        }
    })
}

/// Split the `T_UB = W/P + (1 − 1/P) · B` objective into its
/// `(coeff_w, coeff_b)` linear combination.
///
/// `P ≤ 0` (within tolerance) is treated as full parallelism
/// (`T = B`); fractional `P` in `(0, 1)` is clamped to `1.0`
/// (serial).
#[inline]
pub(crate) fn objective_coefficients(parallelism: f64) -> (f64, f64) {
    if parallelism <= 1e-9 {
        return (0.0, 1.0);
    }
    let p = parallelism.max(1.0);
    let coeff_w = 1.0 / p;
    (coeff_w, 1.0 - coeff_w)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn segment_index_is_left_closed() {
        let breaks = [1024i64, 2048];
        assert_eq!(segment_index(&breaks, -5), 0);
        assert_eq!(segment_index(&breaks, 0), 0);
        assert_eq!(segment_index(&breaks, 1024), 0);
        assert_eq!(segment_index(&breaks, 1025), 1);
        assert_eq!(segment_index(&breaks, 2048), 1);
        assert_eq!(segment_index(&breaks, 2049), 2);
    }

    #[test]
    fn stitching_preserves_continuity_at_breaks() {
        let breaks = [1000i64, 5000];
        let overheads = [10.0, 999.0, -7.0]; // later overheads rewritten
        let rates = [0.01, 0.001, 0.05];
        let seg: Vec<(f64, f64)> = stitched_segments(&breaks, &overheads, &rates).collect();
        assert_eq!(seg.len(), 3);
        for (k, &b) in breaks.iter().enumerate() {
            let left = seg[k].0 + seg[k].1 * b as f64;
            let right = seg[k + 1].0 + seg[k + 1].1 * b as f64;
            assert!(
                (left - right).abs() < 1e-9,
                "discontinuity at break {k}: {left} vs {right}"
            );
        }
    }

    #[test]
    fn stitching_clamps_negative_rates() {
        let seg: Vec<(f64, f64)> = stitched_segments(&[100], &[5.0, 5.0], &[-0.5, 0.1]).collect();
        assert_eq!(seg[0].1, 0.0);
    }

    #[test]
    fn objective_coefficients_cover_degenerate_parallelism() {
        assert_eq!(objective_coefficients(0.0), (0.0, 1.0));
        assert_eq!(objective_coefficients(-3.0), (0.0, 1.0));
        assert_eq!(objective_coefficients(1.0), (1.0, 0.0));
        assert_eq!(objective_coefficients(0.5), (1.0, 0.0)); // clamped to serial
        let (cw, cb) = objective_coefficients(4.0);
        assert!((cw - 0.25).abs() < 1e-12 && (cb - 0.75).abs() < 1e-12);
    }
}
