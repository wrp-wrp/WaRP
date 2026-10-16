//! Cost-model calibration from latency samples.
//!
//! The planners consume a [`CostModel`] with `(overhead, rate)` per
//! request; those values come either from the caller (measured
//! externally, configured from a site-specific calibration snapshot,
//! ...) or from this module.
//!
//! # Usage
//!
//! Record `(bytes_fetched, wall_time_ms)` pairs as you probe, then ask
//! the calibrator to fit a model and materialise a [`CostModel`]:
//!
//! ```rust
//! use warp_planner::calibrate::Calibrator;
//!
//! let mut cal = Calibrator::new();
//! cal.record(    1_024, 27.05);
//! cal.record(   16_384, 27.24);
//! cal.record(  262_144, 29.80);
//! cal.record(3_145_728, 58.70);
//! cal.record(8_388_608, 120.5);
//!
//! let (overhead_ms, ms_per_byte) = cal.fit_linear();
//! let cost_model = cal.to_cost_model();
//! ```
//!
//! [`Calibrator::to_cost_model`] picks the better of the linear and
//! piecewise fits (two segments with an automatically chosen break) on
//! residual RMSE, and returns a [`CostModel`] ready to hand to a
//! [`crate::strategy::MergeStrategy`].
//!
//! The fit itself is pure math on samples — no I/O. For a ready-made
//! `object_store`-backed probe loop that *produces* the samples, see
//! the feature-gated `probe` module (enable with `--features probe`).

use crate::cost_model::CostModel;

/// A single latency observation: `(bytes_fetched, observed_wall_ms)`.
#[derive(Clone, Copy, Debug)]
pub struct Sample {
    pub bytes: i64,
    pub latency_ms: f64,
}

/// Accumulates samples and fits a cost model to them.
///
/// Robust to small sample counts: a linear fit needs ≥ 2 unique byte
/// values, a piecewise fit needs ≥ 5. Below those thresholds the
/// calibrator returns a conservative fallback and
/// [`Calibrator::fit_piecewise`] returns `None`.
#[derive(Clone, Debug, Default)]
pub struct Calibrator {
    samples: Vec<Sample>,
}

/// A two-segment piecewise-linear fit. `break_bytes` separates the
/// "small-request" regime (segment 0) from the "bandwidth-bound"
/// regime (segment 1).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PiecewiseFit {
    pub break_bytes: i64,
    pub overheads_ms: [f64; 2],
    pub rates_ms_per_byte: [f64; 2],
}

impl Calibrator {
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a single `(bytes, latency_ms)` sample.
    pub fn record(&mut self, bytes: i64, latency_ms: f64) {
        if bytes > 0 && latency_ms.is_finite() && latency_ms >= 0.0 {
            self.samples.push(Sample { bytes, latency_ms });
        }
    }

    /// Record every sample in `it`. Convenience for bulk-loading from
    /// a probe loop.
    pub fn record_all<I>(&mut self, it: I)
    where
        I: IntoIterator<Item = Sample>,
    {
        for s in it {
            self.record(s.bytes, s.latency_ms);
        }
    }

    pub fn samples(&self) -> &[Sample] {
        &self.samples
    }

    pub fn sample_count(&self) -> usize {
        self.samples.len()
    }

    /// Ordinary least squares fit of `latency_ms = overhead + rate · bytes`.
    ///
    /// Returns `(overhead_ms, ms_per_byte)`. Rate is clamped to `≥ 0`;
    /// overhead is clamped to `≥ 0` so callers can trust
    /// `w(0) = overhead`.
    pub fn fit_linear(&self) -> (f64, f64) {
        linear_fit(&self.samples)
    }

    /// Two-segment piecewise-linear fit. Chooses the break point that
    /// minimises residual RMSE among the observed byte values (each
    /// sample with at least one sample on each side of it is a
    /// candidate break).
    ///
    /// Returns `None` if there aren't enough samples to meaningfully
    /// split into two regimes (need ≥ 5 and ≥ 2 unique sizes on each
    /// side of the break).
    pub fn fit_piecewise(&self) -> Option<PiecewiseFit> {
        piecewise_fit(&self.samples)
    }

    /// Residual RMSE of a linear fit across the samples. Lower is
    /// better. Useful for sanity-checking the fit against expectations.
    pub fn linear_rmse(&self) -> f64 {
        let (o, r) = self.fit_linear();
        rmse_linear(&self.samples, o, r)
    }

    /// Residual RMSE of the piecewise fit. Returns the linear RMSE
    /// when no piecewise fit is possible.
    pub fn piecewise_rmse(&self) -> f64 {
        match self.fit_piecewise() {
            Some(fit) => rmse_piecewise(&self.samples, &fit),
            None => self.linear_rmse(),
        }
    }

    /// Pick the better of the linear and piecewise fits (lower RMSE
    /// with a 5 % piecewise-preference tiebreaker) and materialise a
    /// [`CostModel`]. Objective context such as parallelism stays at
    /// its default; the caller typically overrides it before planning.
    pub fn to_cost_model(&self) -> CostModel {
        let (o, r) = self.fit_linear();
        let mut cm = CostModel {
            latency_model: "linear".into(),
            latency_per_request: o,
            latency_per_byte: r,
            ..CostModel::default()
        };

        if let Some(pw) = self.fit_piecewise() {
            let lin_rmse = rmse_linear(&self.samples, o, r);
            let pw_rmse = rmse_piecewise(&self.samples, &pw);
            // Prefer piecewise only when it's meaningfully better AND
            // the two segments have genuinely distinct slopes — avoids
            // overfitting noise on a fundamentally linear dataset.
            let (r0, r1) = (pw.rates_ms_per_byte[0], pw.rates_ms_per_byte[1]);
            let slope_ratio = (r0.max(r1) + 1e-18) / (r0.min(r1) + 1e-18);
            let rmse_beat = pw_rmse < 0.80 * lin_rmse.max(1e-12);
            let slopes_differ = slope_ratio >= 1.5;
            if rmse_beat && slopes_differ {
                cm.latency_model = "piecewise_linear".into();
                cm.latency_pw_breaks_bytes = vec![pw.break_bytes];
                cm.latency_pw_overheads_ms = pw.overheads_ms.to_vec();
                cm.latency_pw_rates_ms_per_byte = pw.rates_ms_per_byte.to_vec();
            }
        }
        cm
    }
}

// ========================================================================
// OLS primitives.
// ========================================================================

fn linear_fit(samples: &[Sample]) -> (f64, f64) {
    if samples.len() < 2 {
        return (samples.first().map(|s| s.latency_ms).unwrap_or(0.0), 0.0);
    }

    let n = samples.len() as f64;
    let mut sum_x = 0.0f64;
    let mut sum_y = 0.0f64;
    let mut sum_xx = 0.0f64;
    let mut sum_xy = 0.0f64;
    for s in samples {
        let x = s.bytes as f64;
        let y = s.latency_ms;
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }
    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-18 {
        // All samples have the same byte size — fall back to mean lat.
        return (sum_y / n, 0.0);
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    (intercept.max(0.0), slope.max(0.0))
}

fn rmse_linear(samples: &[Sample], overhead: f64, rate: f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for s in samples {
        let pred = overhead + rate * s.bytes as f64;
        let e = pred - s.latency_ms;
        acc += e * e;
    }
    (acc / samples.len() as f64).sqrt()
}

fn piecewise_fit(samples: &[Sample]) -> Option<PiecewiseFit> {
    if samples.len() < 5 {
        return None;
    }
    let mut sorted = samples.to_vec();
    sorted.sort_by_key(|s| s.bytes);

    // Unique byte values sorted ascending — candidate break points.
    let mut uniques: Vec<i64> = sorted.iter().map(|s| s.bytes).collect();
    uniques.dedup();
    if uniques.len() < 3 {
        return None;
    }

    let mut best: Option<(f64, PiecewiseFit)> = None;
    for &break_bytes in &uniques[1..uniques.len().saturating_sub(1)] {
        let (low, high): (Vec<_>, Vec<_>) =
            sorted.iter().copied().partition(|s| s.bytes <= break_bytes);
        if low.len() < 2 || high.len() < 2 {
            continue;
        }
        // Fit each side independently.
        let (o0, r0) = linear_fit(&low);
        let (_, r1) = linear_fit(&high);
        // Enforce continuity: second segment's effective overhead is
        // derived so the two halves agree at `break_bytes`.
        let value_at_break = o0 + r0 * break_bytes as f64;
        let o1 = value_at_break - r1 * break_bytes as f64;

        let fit = PiecewiseFit {
            break_bytes,
            overheads_ms: [o0, o1],
            rates_ms_per_byte: [r0.max(0.0), r1.max(0.0)],
        };
        let rmse = rmse_piecewise(&sorted, &fit);
        if best.as_ref().map(|(r, _)| rmse < *r).unwrap_or(true) {
            best = Some((rmse, fit));
        }
    }
    best.map(|(_, f)| f)
}

fn rmse_piecewise(samples: &[Sample], fit: &PiecewiseFit) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let mut acc = 0.0f64;
    for s in samples {
        let pred = if s.bytes <= fit.break_bytes {
            fit.overheads_ms[0] + fit.rates_ms_per_byte[0] * s.bytes as f64
        } else {
            fit.overheads_ms[1] + fit.rates_ms_per_byte[1] * s.bytes as f64
        };
        let e = pred - s.latency_ms;
        acc += e * e;
    }
    (acc / samples.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear_fit_recovers_planted_line() {
        let mut cal = Calibrator::new();
        // latency = 27 + 1e-5 · bytes
        for &b in &[1024i64, 16_384, 262_144, 1_048_576, 4_194_304] {
            cal.record(b, 27.0 + 1e-5 * b as f64);
        }
        let (o, r) = cal.fit_linear();
        assert!((o - 27.0).abs() < 1e-3, "overhead: {o}");
        assert!((r - 1e-5).abs() < 1e-8, "rate: {r}");
    }

    #[test]
    fn piecewise_fit_recovers_planted_breakpoint() {
        let mut cal = Calibrator::new();
        // small regime: 27 + 1e-6 · bytes  (cheap per byte)
        for &b in &[1024i64, 16_384, 262_144, 1_048_576, 2_097_152] {
            cal.record(b, 27.0 + 1e-6 * b as f64);
        }
        // big regime: faster per request but much more expensive per byte.
        for &b in &[4_194_304i64, 6_291_456, 8_388_608, 12_582_912, 16_777_216] {
            // continuity at 3_145_728 isn't perfect in this planted
            // data — just check the fit breaks somewhere sensible.
            cal.record(b, 27.0 + 1e-5 * b as f64);
        }
        let fit = cal.fit_piecewise().expect("piecewise should fit");
        assert!(
            (100_000..=10_000_000).contains(&fit.break_bytes),
            "breakpoint {} out of plausible range",
            fit.break_bytes
        );
    }

    #[test]
    fn to_cost_model_picks_linear_when_no_break() {
        let mut cal = Calibrator::new();
        for &b in &[1024i64, 16_384, 262_144, 1_048_576, 4_194_304, 8_388_608] {
            cal.record(b, 27.0 + 1e-5 * b as f64);
        }
        let cm = cal.to_cost_model();
        // One-segment planted data: piecewise shouldn't beat linear.
        assert_eq!(cm.latency_model, "linear");
        assert!((cm.latency_per_request - 27.0).abs() < 1.0);
        assert!((cm.latency_per_byte - 1e-5).abs() < 1e-7);
    }

    #[test]
    fn to_cost_model_uses_piecewise_when_better() {
        let mut cal = Calibrator::new();
        // Clear two-segment planted shape — piecewise should win.
        let break_b: i64 = 3_145_728;
        for &b in &[1024i64, 16_384, 262_144, 1_048_576, 2_097_152] {
            cal.record(b, 27.0 + 5e-7 * b as f64);
        }
        let v_at_break = 27.0 + 5e-7 * break_b as f64;
        for &b in &[4_194_304i64, 6_291_456, 10_485_760, 16_777_216, 33_554_432] {
            // Extend continuously with a steeper slope.
            let pred = v_at_break + 1e-5 * (b - break_b) as f64;
            cal.record(b, pred);
        }
        let cm = cal.to_cost_model();
        assert_eq!(cm.latency_model, "piecewise_linear");
        cm.validate().unwrap();
        assert_eq!(cm.latency_pw_breaks_bytes.len(), 1);
        assert_eq!(cm.latency_pw_overheads_ms.len(), 2);
        assert_eq!(cm.latency_pw_rates_ms_per_byte.len(), 2);
    }

    #[test]
    fn empty_calibrator_returns_defaults() {
        let cm = Calibrator::new().to_cost_model();
        assert_eq!(cm.latency_model, "linear");
        assert_eq!(cm.latency_per_byte, 0.0);
    }

    #[test]
    fn invalid_samples_are_dropped() {
        let mut cal = Calibrator::new();
        cal.record(-1, 10.0); // bytes <= 0
        cal.record(100, f64::NAN); // nan
        cal.record(100, f64::INFINITY); // inf
        cal.record(100, -5.0); // negative latency
        cal.record(100, 10.0); // valid
        assert_eq!(cal.sample_count(), 1);
    }
}
