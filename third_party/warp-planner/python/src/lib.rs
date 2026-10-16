//! PyO3 binding for the warp-planner Rust crate.
//!
//! Minimal wrapper: one Python class per Rust struct, one function per
//! strategy. Data is passed in and out as plain tuples / dicts so the
//! binding has no dependencies beyond PyO3 itself.
//
// The `useless_conversion` allow is a known false-positive interaction
// between clippy 1.91 and the pyo3 0.22 macros — the macro-expanded
// return types look like redundant conversions to clippy, but they
// aren't.
#![allow(clippy::useless_conversion)]

use ::warp_planner::{
    evaluate, plan as checked_plan,
    strategy::{Bottleneck, Greedy, MainSolution, MergeStrategy, PreMerge, Separate, Serial},
    validate_requests, CostModel, IoPlan, IoRequest, MergedRequest, PlanConstraints, PlanMetrics,
    PlanningError,
};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};

fn build_cost(kwargs: Option<&Bound<'_, PyDict>>) -> PyResult<CostModel> {
    let mut c = CostModel::default();
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            let key: String = k.extract()?;
            match key.as_str() {
                "latency_model" => c.latency_model = v.extract()?,
                "latency_per_request" => c.latency_per_request = v.extract()?,
                "latency_per_byte" => c.latency_per_byte = v.extract()?,
                "latency_pw_breaks_bytes" => c.latency_pw_breaks_bytes = v.extract()?,
                "latency_pw_overheads_ms" => c.latency_pw_overheads_ms = v.extract()?,
                "latency_pw_rates_ms_per_byte" => c.latency_pw_rates_ms_per_byte = v.extract()?,
                "latency_std" => c.latency_std = v.extract()?,
                "concurrent_parallelism" => c.concurrent_parallelism = v.extract()?,
                "cost_per_request" => c.cost_per_request = v.extract()?,
                "cost_per_byte" => c.cost_per_byte = v.extract()?,
                "cost_per_wasted_byte" => c.cost_per_wasted_byte = v.extract()?,
                // Plan-only kwargs are accepted here so callers can
                // reuse the same kwargs dict for `plan` and
                // `evaluate_plan`; they do not affect evaluation.
                "max_requests" | "exact_requests"
                | "greedy_min_gap_bytes" | "greedy_max_span_bytes"
                | "pre_merge_max_gap_bytes" | "pre_merge_max_merged_size_bytes" => {}
                "max_merge_span" => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "max_merge_span was removed from CostModel; use greedy_max_span_bytes for strategy='greedy'",
                    ))
                }
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown CostModel field: {other}"
                    )))
                }
            }
        }
    }
    Ok(c)
}

#[derive(Default)]
struct GreedyKwargs {
    min_gap_bytes: Option<i64>,
    max_span_bytes: Option<i64>,
}

#[derive(Default)]
struct PreMergeKwargs {
    max_gap_bytes: Option<i64>,
    max_merged_size_bytes: Option<i64>,
}

#[derive(Default)]
struct SharedConstraintKwargs {
    has_max_requests: bool,
    has_exact_requests: bool,
}

impl SharedConstraintKwargs {
    fn any(&self) -> bool {
        self.has_max_requests || self.has_exact_requests
    }
}

fn parse_plan_kwargs(
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<(
    CostModel,
    PlanConstraints,
    SharedConstraintKwargs,
    GreedyKwargs,
    PreMergeKwargs,
)> {
    let mut cost = CostModel::default();
    let mut constraints = PlanConstraints::default();
    let mut shared = SharedConstraintKwargs::default();
    let mut greedy = GreedyKwargs::default();
    let mut pre_merge = PreMergeKwargs::default();
    if let Some(kw) = kwargs {
        for (k, v) in kw.iter() {
            let key: String = k.extract()?;
            match key.as_str() {
                "latency_model" => cost.latency_model = v.extract()?,
                "latency_per_request" => cost.latency_per_request = v.extract()?,
                "latency_per_byte" => cost.latency_per_byte = v.extract()?,
                "latency_pw_breaks_bytes" => cost.latency_pw_breaks_bytes = v.extract()?,
                "latency_pw_overheads_ms" => cost.latency_pw_overheads_ms = v.extract()?,
                "latency_pw_rates_ms_per_byte" => cost.latency_pw_rates_ms_per_byte = v.extract()?,
                "latency_std" => cost.latency_std = v.extract()?,
                "concurrent_parallelism" => cost.concurrent_parallelism = v.extract()?,
                "cost_per_request" => cost.cost_per_request = v.extract()?,
                "cost_per_byte" => cost.cost_per_byte = v.extract()?,
                "cost_per_wasted_byte" => cost.cost_per_wasted_byte = v.extract()?,
                "max_requests" => {
                    let val: i32 = v.extract()?;
                    if val < 0 {
                        return Err(pyo3::exceptions::PyValueError::new_err(format!(
                            "max_requests must be >= 0, got {val}"
                        )));
                    }
                    constraints.max_requests = val;
                    // Only flag as "constraint was requested" when the
                    // value is non-default; `max_requests=0` is the
                    // neutral marker and must round-trip cleanly
                    // through any strategy.
                    if val != 0 {
                        shared.has_max_requests = true;
                    }
                }
                "exact_requests" => {
                    let val: bool = v.extract()?;
                    constraints.exact_requests = val;
                    if val {
                        shared.has_exact_requests = true;
                    }
                }
                "greedy_min_gap_bytes" => greedy.min_gap_bytes = Some(v.extract()?),
                "greedy_max_span_bytes" => greedy.max_span_bytes = Some(v.extract()?),
                "pre_merge_max_gap_bytes" => pre_merge.max_gap_bytes = Some(v.extract()?),
                "pre_merge_max_merged_size_bytes" => pre_merge.max_merged_size_bytes = Some(v.extract()?),
                "max_merge_span" => {
                    return Err(pyo3::exceptions::PyValueError::new_err(
                        "max_merge_span was removed from CostModel; use greedy_max_span_bytes for strategy='greedy'",
                    ))
                }
                other => {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "unknown planning kwarg: {other}"
                    )))
                }
            }
        }
    }
    Ok((cost, constraints, shared, greedy, pre_merge))
}

fn parse_requests(py_requests: &Bound<'_, PyList>) -> PyResult<Vec<IoRequest>> {
    py_requests
        .iter()
        .map(|item| -> PyResult<IoRequest> {
            let t: (i32, i64, i64) = item.extract()?;
            Ok(IoRequest {
                cluster_id: t.0,
                offset: t.1,
                length: t.2,
            })
        })
        .collect()
}

fn plan_to_py<'py>(py: Python<'py>, plan: &IoPlan) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    let merged: Vec<(i64, i64)> = plan
        .merged_requests
        .iter()
        .map(|m: &MergedRequest| (m.offset, m.length))
        .collect();
    out.set_item("merged_requests", merged)?;
    out.set_item("assignments", plan.assignments.clone())?;
    Ok(out)
}

fn metrics_to_py<'py>(
    py: Python<'py>,
    m: &PlanMetrics,
    cost: &CostModel,
) -> PyResult<Bound<'py, PyDict>> {
    let out = PyDict::new_bound(py);
    out.set_item("total_bytes", m.total_bytes)?;
    out.set_item("useful_bytes", m.useful_bytes)?;
    out.set_item("wasted_bytes", m.wasted_bytes)?;
    out.set_item("request_count", m.request_count)?;
    out.set_item("predicted_latency_ms_serial", m.predicted_latency_ms_serial)?;
    out.set_item(
        "predicted_latency_ms_concurrent",
        m.predicted_latency_ms_concurrent,
    )?;
    // T_UB(P) at the cost model's concurrent_parallelism — the exact
    // objective MainSolution optimises, so callers don't hand-write
    // the formula (the P <= 0 convention is easy to get wrong).
    out.set_item(
        "predicted_latency_ms_t_ub",
        m.t_ub_ms(cost.concurrent_parallelism),
    )?;
    Ok(out)
}

fn dispatch(
    name: &str,
    requests: &[IoRequest],
    cost: &CostModel,
    constraints: &PlanConstraints,
    greedy_kwargs: &GreedyKwargs,
    pre_merge_kwargs: &PreMergeKwargs,
) -> Result<IoPlan, PlanningError> {
    let strat: Box<dyn MergeStrategy> = match name {
        "separate" => Box::new(Separate),
        "greedy" => {
            let defaults = Greedy::default();
            Box::new(Greedy::new(
                greedy_kwargs
                    .min_gap_bytes
                    .unwrap_or(defaults.min_gap_bytes),
                greedy_kwargs
                    .max_span_bytes
                    .unwrap_or(defaults.max_span_bytes),
            ))
        }
        "pre_merge" => {
            // Library default is the no-op (0, 0); callers must pass
            // explicit thresholds to actually pre-merge.
            let defaults = PreMerge::default();
            Box::new(PreMerge::new(
                pre_merge_kwargs
                    .max_gap_bytes
                    .unwrap_or(defaults.max_gap_bytes),
                pre_merge_kwargs
                    .max_merged_size_bytes
                    .unwrap_or(defaults.max_merged_size_bytes),
            ))
        }
        "serial" => Box::new(Serial),
        "bottleneck" => Box::new(Bottleneck),
        "main_solution" => Box::new(MainSolution),
        other => unreachable!("strategy name validated by plan(): {other}"),
    };
    checked_plan(strat.as_ref(), requests, cost, constraints)
}

/// Produce a plan from raw (cluster_id, offset, length) triples and a
/// strategy name. Cost-model fields, shared plan constraints, and
/// strategy-specific knobs are passed as kwargs.
#[pyfunction]
#[pyo3(signature = (requests, strategy, **plan_kwargs))]
fn plan<'py>(
    py: Python<'py>,
    requests: &Bound<'_, PyList>,
    strategy: &str,
    plan_kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    const KNOWN: &[&str] = &[
        "separate",
        "greedy",
        "pre_merge",
        "serial",
        "bottleneck",
        "main_solution",
    ];
    if !KNOWN.contains(&strategy) {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "unknown strategy: {strategy}"
        )));
    }
    let reqs = parse_requests(requests)?;
    let (cost, constraints, shared_constraints, greedy_kwargs, pre_merge_kwargs) =
        parse_plan_kwargs(plan_kwargs)?;
    if !matches!(strategy, "serial" | "bottleneck" | "main_solution") && shared_constraints.any() {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "max_requests / exact_requests are only honoured by strategy='serial', 'bottleneck', or 'main_solution', got strategy='{strategy}'"
        )));
    }
    if strategy != "greedy"
        && (greedy_kwargs.min_gap_bytes.is_some() || greedy_kwargs.max_span_bytes.is_some())
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "greedy_min_gap_bytes / greedy_max_span_bytes are only honoured by strategy='greedy', got strategy='{strategy}'"
        )));
    }
    if strategy != "pre_merge"
        && (pre_merge_kwargs.max_gap_bytes.is_some()
            || pre_merge_kwargs.max_merged_size_bytes.is_some())
    {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "pre_merge_max_gap_bytes / pre_merge_max_merged_size_bytes are only honoured by strategy='pre_merge', got strategy='{strategy}'"
        )));
    }
    // Run the pure-Rust compute with the GIL released so other Python
    // threads are not blocked during long plans (MainSolution DP can
    // run tens of milliseconds at n ≥ 1000).
    let p = py
        .allow_threads(|| {
            dispatch(
                strategy,
                &reqs,
                &cost,
                &constraints,
                &greedy_kwargs,
                &pre_merge_kwargs,
            )
        })
        .map_err(|error| match error {
            PlanningError::InvalidPlan(_)
            | PlanningError::RequestCountConstraintViolated { .. } => {
                pyo3::exceptions::PyRuntimeError::new_err(error.to_string())
            }
            _ => pyo3::exceptions::PyValueError::new_err(error.to_string()),
        })?;
    plan_to_py(py, &p)
}

/// Evaluate a plan (`merged_requests` + `assignments`) against the cost
/// model; returns a dict of metrics.
#[pyfunction]
#[pyo3(signature = (requests, plan_dict, **cost_kwargs))]
fn evaluate_plan<'py>(
    py: Python<'py>,
    requests: &Bound<'_, PyList>,
    plan_dict: &Bound<'_, PyDict>,
    cost_kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Bound<'py, PyDict>> {
    let reqs = parse_requests(requests)?;
    validate_requests(&reqs)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("requests: {e}")))?;
    let cost = build_cost(cost_kwargs)?;
    cost.validate()
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("cost model: {e}")))?;

    let merged_raw: Vec<(i64, i64)> = plan_dict
        .get_item("merged_requests")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("merged_requests"))?
        .extract()?;
    let assignments: Vec<usize> = plan_dict
        .get_item("assignments")?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err("assignments"))?
        .extract()?;
    let plan = IoPlan {
        merged_requests: merged_raw
            .into_iter()
            .map(|(offset, length)| MergedRequest { offset, length })
            .collect(),
        assignments,
    };
    plan.check_invariants(&reqs)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("plan: {e}")))?;
    let m = py.allow_threads(|| evaluate(&reqs, &plan, &cost));
    metrics_to_py(py, &m, &cost)
}

#[pymodule]
fn vecio_planner(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(plan, m)?)?;
    m.add_function(wrap_pyfunction!(evaluate_plan, m)?)?;
    m.add_class::<calibrate_py::Calibrator>()?;
    Ok(())
}

// Calibrator wrapper exposed below; see bottom of file for impl.
mod calibrate_py {
    use ::warp_planner::calibrate::{Calibrator as RustCalibrator, PiecewiseFit};
    use ::warp_planner::CostModel;
    use pyo3::prelude::*;
    use pyo3::types::PyDict;

    /// A Calibrator accumulates `(bytes, latency_ms)` samples and fits
    /// a linear or piecewise-linear cost model.
    #[pyclass]
    pub struct Calibrator {
        inner: RustCalibrator,
    }

    #[pymethods]
    impl Calibrator {
        #[new]
        fn new() -> Self {
            Self {
                inner: RustCalibrator::new(),
            }
        }

        /// Record a single latency sample.
        fn record(&mut self, bytes: i64, latency_ms: f64) {
            self.inner.record(bytes, latency_ms);
        }

        /// Number of samples currently held.
        fn sample_count(&self) -> usize {
            self.inner.sample_count()
        }

        /// Return (overhead_ms, ms_per_byte) under a linear fit.
        fn fit_linear(&self) -> (f64, f64) {
            self.inner.fit_linear()
        }

        /// Return the piecewise fit as a dict, or `None` when there are
        /// not enough samples to meaningfully fit two segments.
        fn fit_piecewise<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
            let Some(fit) = self.inner.fit_piecewise() else {
                return Ok(None);
            };
            let out = PyDict::new_bound(py);
            out.set_item("break_bytes", fit.break_bytes)?;
            out.set_item("overheads_ms", fit.overheads_ms.to_vec())?;
            out.set_item("rates_ms_per_byte", fit.rates_ms_per_byte.to_vec())?;
            Ok(Some(out))
        }

        /// Build a CostModel dict from the best-available fit. Uses the
        /// piecewise fit when it reduces residual error, else falls back
        /// to the linear fit.
        #[pyo3(signature = (*, concurrent_parallelism = 0.0))]
        fn to_cost_model<'py>(
            &self,
            py: Python<'py>,
            concurrent_parallelism: f64,
        ) -> PyResult<Bound<'py, PyDict>> {
            let mut cost = self.inner.to_cost_model();
            cost.concurrent_parallelism = concurrent_parallelism;
            cost_to_py(py, &cost)
        }
    }

    fn cost_to_py<'py>(py: Python<'py>, c: &CostModel) -> PyResult<Bound<'py, PyDict>> {
        let out = PyDict::new_bound(py);
        out.set_item("latency_model", c.latency_model.clone())?;
        out.set_item("latency_per_request", c.latency_per_request)?;
        out.set_item("latency_per_byte", c.latency_per_byte)?;
        out.set_item("latency_pw_breaks_bytes", c.latency_pw_breaks_bytes.clone())?;
        out.set_item("latency_pw_overheads_ms", c.latency_pw_overheads_ms.clone())?;
        out.set_item(
            "latency_pw_rates_ms_per_byte",
            c.latency_pw_rates_ms_per_byte.clone(),
        )?;
        out.set_item("concurrent_parallelism", c.concurrent_parallelism)?;
        out.set_item("cost_per_request", c.cost_per_request)?;
        out.set_item("cost_per_byte", c.cost_per_byte)?;
        out.set_item("cost_per_wasted_byte", c.cost_per_wasted_byte)?;
        Ok(out)
    }

    // Re-export for the outer module.
    #[allow(dead_code)]
    pub(crate) type _PiecewiseFitReExport = PiecewiseFit;
}
