// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Balanced-UB planner-overhead benchmark on real probe-list JSON exported by an external pipeline.

use lance_io::scheduler::{CostModel, IOPlan, MergeStrategy, VecioMergePlanner};
use rayon::prelude::*;
use rayon::ThreadPoolBuilder;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

const BENCHMARK_NAME: &str = "planner_overhead_real_piecewise";
const PER_QUERY_TIMEOUT_MS: u64 = 5_000;
const TIMEOUT_NPROBE_THRESHOLD: usize = 512;
type DynError = Box<dyn Error + Send + Sync>;

#[derive(Debug)]
struct Args {
    input_json: PathBuf,
    out_json: PathBuf,
    out_csv: PathBuf,
    jobs: usize,
    disable_timeouts: bool,
}

#[derive(Debug, Deserialize)]
struct PlannerInput {
    version: String,
    benchmark_name: String,
    latency_model: LatencyModelInput,
    benchmark_grid: BenchmarkGrid,
    datasets: Vec<DatasetInput>,
}

#[derive(Debug, Deserialize)]
struct LatencyModelInput {
    kind: String,
    breaks_bytes: Vec<i64>,
    overheads_ms: Vec<f64>,
    rates_ms_per_byte: Vec<f64>,
    concurrent_parallelism: f64,
    max_waste_ratio: f64,
    max_merge_span: i64,
}

#[derive(Debug, Deserialize)]
struct BenchmarkGrid {
    nprobes: Vec<usize>,
    ks: Vec<usize>,
    num_queries: usize,
    warmup_queries: usize,
}

#[derive(Debug, Deserialize)]
struct DatasetInput {
    dataset: String,
    label: String,
    nlist: usize,
    max_nprobe: usize,
    cluster_offsets: Vec<i64>,
    cluster_sizes: Vec<i64>,
    queries: Vec<QueryInput>,
}

#[derive(Debug, Deserialize)]
struct QueryInput {
    query_idx: usize,
    cluster_ids: Vec<i32>,
}

#[derive(Debug, Serialize)]
struct PlannerOutput {
    version: String,
    benchmark_name: String,
    rows: Vec<OutputRow>,
}

#[derive(Debug, Clone, Serialize)]
struct OutputRow {
    dataset: String,
    nprobe: usize,
    #[serde(rename = "K")]
    k: usize,
    queries: usize,
    plain_timed_out: bool,
    plain_med_ms: f64,
    plain_p95_ms: f64,
    plain_max_ms: f64,
    dc_timed_out: bool,
    dc_med_ms: f64,
    dc_p95_ms: f64,
    dc_max_ms: f64,
    adaptive_dc_timed_out: bool,
    adaptive_dc_med_ms: f64,
    adaptive_dc_p95_ms: f64,
    adaptive_dc_max_ms: f64,
    wqs_timed_out: bool,
    wqs_med_ms: f64,
    wqs_p95_ms: f64,
    wqs_max_ms: f64,
    adaptive_over_dc_p95: f64,
    wqs_over_adaptive_dc_p95: f64,
    adaptive_dc_err_avg_pct: f64,
    adaptive_dc_err_max_pct: f64,
    wqs_err_avg_pct: f64,
    wqs_err_max_pct: f64,
}

#[derive(Debug, Clone, Copy)]
struct BenchmarkTask {
    dataset_idx: usize,
    nprobe: usize,
    k: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BenchMethod {
    Plain,
    Dc,
    AdaptiveDc,
    Wqs,
}

impl BenchMethod {
    fn as_str(self) -> &'static str {
        match self {
            Self::Plain => "plain",
            Self::Dc => "dc",
            Self::AdaptiveDc => "adaptive_dc",
            Self::Wqs => "wqs",
        }
    }

    fn from_str(value: &str) -> Result<Self, DynError> {
        match value {
            "plain" => Ok(Self::Plain),
            "dc" => Ok(Self::Dc),
            "adaptive_dc" => Ok(Self::AdaptiveDc),
            "wqs" => Ok(Self::Wqs),
            _ => Err(format!("unknown method: {value}").into()),
        }
    }

    fn uses_timeout(self, nprobe: usize) -> bool {
        nprobe >= TIMEOUT_NPROBE_THRESHOLD && matches!(self, Self::Plain | Self::Dc)
    }
}

#[derive(Debug)]
struct SingleQueryConfig {
    dataset_label: String,
    query_pos: usize,
    nprobe: usize,
    k: usize,
    method: BenchMethod,
}

#[derive(Debug, Deserialize, Serialize)]
struct SingleQueryOutput {
    elapsed_ms: f64,
    objective: f64,
}

#[derive(Debug)]
struct EvalOutcome {
    elapsed_ms: f64,
    objective: f64,
    timed_out: bool,
}

fn usage() -> &'static str {
    "usage: planner_overhead_real --input-json <path> [--out-json <path>] [--out-csv <path>] [--jobs <N>] [--disable-timeouts]"
}

fn parse_args() -> Result<Args, DynError> {
    let mut input_json: Option<PathBuf> = None;
    let mut out_json: Option<PathBuf> = None;
    let mut out_csv: Option<PathBuf> = None;
    let mut jobs = 1usize;
    let mut disable_timeouts = false;

    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--input-json" => {
                input_json = Some(PathBuf::from(
                    args.next().ok_or("missing value for --input-json")?,
                ))
            }
            "--out-json" => {
                out_json = Some(PathBuf::from(
                    args.next().ok_or("missing value for --out-json")?,
                ))
            }
            "--out-csv" => {
                out_csv = Some(PathBuf::from(
                    args.next().ok_or("missing value for --out-csv")?,
                ))
            }
            "--jobs" => {
                jobs = args
                    .next()
                    .ok_or("missing value for --jobs")?
                    .parse::<usize>()?
            }
            "--disable-timeouts" => disable_timeouts = true,
            "-h" | "--help" => {
                println!("{}", usage());
                std::process::exit(0);
            }
            other => return Err(format!("unknown argument: {other}\n{}", usage()).into()),
        }
    }

    let input_json = input_json.ok_or_else(|| format!("missing --input-json\n{}", usage()))?;
    let base_dir = input_json
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    let out_json = out_json.unwrap_or_else(|| base_dir.join("planner_benchmark_output.json"));
    let out_csv = out_csv.unwrap_or_else(|| base_dir.join("planner_benchmark_output.csv"));
    Ok(Args {
        input_json,
        out_json,
        out_csv,
        jobs: jobs.max(1),
        disable_timeouts,
    })
}

fn single_query_config_from_env() -> Result<Option<SingleQueryConfig>, DynError> {
    let Some(dataset_label) = env::var_os("PLANNER_SINGLE_DATASET_LABEL") else {
        return Ok(None);
    };
    let dataset_label = dataset_label
        .into_string()
        .map_err(|_| "PLANNER_SINGLE_DATASET_LABEL must be valid UTF-8")?;
    let query_pos = env::var("PLANNER_SINGLE_QUERY_POS")?.parse::<usize>()?;
    let nprobe = env::var("PLANNER_SINGLE_NPROBE")?.parse::<usize>()?;
    let k = env::var("PLANNER_SINGLE_K")?.parse::<usize>()?;
    let method = BenchMethod::from_str(&env::var("PLANNER_SINGLE_METHOD")?)?;
    Ok(Some(SingleQueryConfig {
        dataset_label,
        query_pos,
        nprobe,
        k,
        method,
    }))
}

fn planner_for_method(method: BenchMethod) -> VecioMergePlanner {
    match method {
        BenchMethod::Plain => VecioMergePlanner::with_plain_dp_baseline(),
        BenchMethod::Dc => VecioMergePlanner::with_fast_path_optimizations(false),
        BenchMethod::AdaptiveDc => VecioMergePlanner::with_adaptive_dc_only(),
        BenchMethod::Wqs => VecioMergePlanner::with_fast_path_optimizations(true),
    }
}

fn summarize_times(values: &[f64], timed_out: bool) -> (f64, f64, f64) {
    if timed_out {
        let capped = PER_QUERY_TIMEOUT_MS as f64;
        return (capped, capped, capped);
    }
    (
        percentile(values, 50.0),
        percentile(values, 95.0),
        values.iter().copied().fold(0.0_f64, f64::max),
    )
}

fn cost_model_from_input(spec: &LatencyModelInput) -> Result<CostModel, DynError> {
    if spec.kind != "piecewise_linear" {
        return Err(format!("unsupported latency_model.kind: {}", spec.kind).into());
    }
    let mut cm = CostModel::default();
    cm.latency_model = spec.kind.clone();
    cm.latency_pw_breaks_bytes = spec.breaks_bytes.clone();
    cm.latency_pw_overheads_ms = spec.overheads_ms.clone();
    cm.latency_pw_rates_ms_per_byte = spec.rates_ms_per_byte.clone();
    cm.concurrent_parallelism = spec.concurrent_parallelism;
    cm.max_waste_ratio = spec.max_waste_ratio;
    cm.max_merge_span = spec.max_merge_span;
    if !cm.use_piecewise_linear() {
        return Err("invalid piecewise latency model".into());
    }
    Ok(cm)
}

fn cluster_maps(
    dataset: &DatasetInput,
) -> Result<(HashMap<i32, i64>, HashMap<i32, i64>), DynError> {
    if dataset.cluster_offsets.len() != dataset.cluster_sizes.len() {
        return Err(format!(
            "dataset {} has mismatched cluster layout lengths: offsets={} sizes={}",
            dataset.dataset,
            dataset.cluster_offsets.len(),
            dataset.cluster_sizes.len()
        )
        .into());
    }

    let mut offsets = HashMap::with_capacity(dataset.cluster_offsets.len());
    let mut sizes = HashMap::with_capacity(dataset.cluster_sizes.len());
    for (cluster_id, (&offset, &size)) in dataset
        .cluster_offsets
        .iter()
        .zip(dataset.cluster_sizes.iter())
        .enumerate()
    {
        offsets.insert(cluster_id as i32, offset);
        sizes.insert(cluster_id as i32, size);
    }
    Ok((offsets, sizes))
}

fn truncated_cluster_ids(query: &QueryInput, nprobe: usize) -> Result<Vec<i32>, DynError> {
    if query.cluster_ids.len() < nprobe {
        return Err(format!(
            "query {} has only {} probe ids, need nprobe={}",
            query.query_idx,
            query.cluster_ids.len(),
            nprobe
        )
        .into());
    }
    Ok(query.cluster_ids.iter().take(nprobe).copied().collect())
}

fn ub_objective(plan: &IOPlan, cost_model: &CostModel) -> f64 {
    let p = cost_model.concurrent_parallelism;
    let coeff_sum = if p > 0.0 { 1.0 / p.max(1.0) } else { 0.0 };
    let coeff_bottleneck = 1.0 - coeff_sum;

    let mut total = 0.0_f64;
    let mut bottleneck = 0.0_f64;
    for req in &plan.requests {
        let latency = cost_model.request_latency_ms(req.length);
        total += latency;
        bottleneck = bottleneck.max(latency);
    }
    coeff_sum * total + coeff_bottleneck * bottleneck
}

fn run_single_query_mode(
    input: &PlannerInput,
    base_cost_model: &CostModel,
    config: &SingleQueryConfig,
) -> Result<(), DynError> {
    let dataset = input
        .datasets
        .iter()
        .find(|dataset| dataset.label == config.dataset_label)
        .ok_or_else(|| format!("unknown dataset label: {}", config.dataset_label))?;
    let (cluster_offsets, cluster_sizes) = cluster_maps(dataset)?;
    let query = dataset
        .queries
        .get(config.query_pos)
        .ok_or_else(|| format!("query position out of range: {}", config.query_pos))?;
    let cluster_ids = truncated_cluster_ids(query, config.nprobe)?;

    let mut cost_model = base_cost_model.clone();
    cost_model.max_requests = config.k as i32;
    cost_model.exact_requests = false;

    let planner = planner_for_method(config.method);
    let t0 = Instant::now();
    let plan = planner.generate_plan(&cluster_ids, &cluster_offsets, &cluster_sizes, &cost_model);
    let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
    let output = SingleQueryOutput {
        elapsed_ms,
        objective: ub_objective(&plan, &cost_model),
    };
    println!("{}", serde_json::to_string(&output)?);
    Ok(())
}

fn run_timed_child_method_once(
    current_exe: &Path,
    input_json: &Path,
    dataset_label: &str,
    query_pos: usize,
    nprobe: usize,
    k: usize,
    method: BenchMethod,
) -> Result<EvalOutcome, DynError> {
    let mut child = Command::new(current_exe)
        .arg("--input-json")
        .arg(input_json)
        .env("PLANNER_SINGLE_DATASET_LABEL", dataset_label)
        .env("PLANNER_SINGLE_QUERY_POS", query_pos.to_string())
        .env("PLANNER_SINGLE_NPROBE", nprobe.to_string())
        .env("PLANNER_SINGLE_K", k.to_string())
        .env("PLANNER_SINGLE_METHOD", method.as_str())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()?;

    let timeout = Duration::from_millis(PER_QUERY_TIMEOUT_MS);
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            let mut stdout = String::new();
            if let Some(mut pipe) = child.stdout.take() {
                pipe.read_to_string(&mut stdout)?;
            }
            let mut stderr = String::new();
            if let Some(mut pipe) = child.stderr.take() {
                pipe.read_to_string(&mut stderr)?;
            }
            if !status.success() {
                return Err(format!(
                    "single-query child failed: method={} dataset={} query_pos={} stdout={} stderr={}",
                    method.as_str(),
                    dataset_label,
                    query_pos,
                    stdout.trim(),
                    stderr.trim()
                )
                .into());
            }
            let output: SingleQueryOutput = serde_json::from_str(stdout.trim())?;
            return Ok(EvalOutcome {
                elapsed_ms: output.elapsed_ms,
                objective: output.objective,
                timed_out: false,
            });
        }

        if start.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Ok(EvalOutcome {
                elapsed_ms: PER_QUERY_TIMEOUT_MS as f64,
                objective: f64::NAN,
                timed_out: true,
            });
        }

        thread::sleep(Duration::from_millis(10));
    }
}

fn percentile(values: &[f64], pct: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    if sorted.len() == 1 {
        return sorted[0];
    }
    let rank = (pct / 100.0) * ((sorted.len() - 1) as f64);
    let lo = rank.floor() as usize;
    let hi = rank.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let frac = rank - (lo as f64);
        sorted[lo] + (sorted[hi] - sorted[lo]) * frac
    }
}

fn benchmark_combo(
    input_json: &Path,
    current_exe: &Path,
    dataset: &DatasetInput,
    nprobe: usize,
    k: usize,
    disable_timeouts: bool,
    base_cost_model: &CostModel,
    num_queries: usize,
    warmup_queries: usize,
    cluster_offsets: &HashMap<i32, i64>,
    cluster_sizes: &HashMap<i32, i64>,
) -> Result<OutputRow, DynError> {
    let planner_wqs = VecioMergePlanner::with_fast_path_optimizations(true);
    let planner_adaptive_dc = VecioMergePlanner::with_adaptive_dc_only();
    let planner_dc = VecioMergePlanner::with_fast_path_optimizations(false);
    let planner_plain = VecioMergePlanner::with_plain_dp_baseline();

    let query_cluster_ids: Vec<Vec<i32>> = dataset
        .queries
        .iter()
        .take(num_queries)
        .map(|query| truncated_cluster_ids(query, nprobe))
        .collect::<Result<_, _>>()?;

    let mut adaptive_cm = base_cost_model.clone();
    adaptive_cm.max_requests = k as i32;
    adaptive_cm.exact_requests = false;

    let mut dc_cm = base_cost_model.clone();
    dc_cm.max_requests = k as i32;
    dc_cm.exact_requests = false;

    for cluster_ids in query_cluster_ids
        .iter()
        .take(warmup_queries.min(query_cluster_ids.len()))
    {
        let _ =
            planner_wqs.generate_plan(cluster_ids, cluster_offsets, cluster_sizes, &adaptive_cm);
        let _ = planner_adaptive_dc.generate_plan(
            cluster_ids,
            cluster_offsets,
            cluster_sizes,
            &adaptive_cm,
        );
        if disable_timeouts || !BenchMethod::Dc.uses_timeout(nprobe) {
            let _ = planner_dc.generate_plan(cluster_ids, cluster_offsets, cluster_sizes, &dc_cm);
        }
        if disable_timeouts || !BenchMethod::Plain.uses_timeout(nprobe) {
            let _ =
                planner_plain.generate_plan(cluster_ids, cluster_offsets, cluster_sizes, &dc_cm);
        }
    }

    let mut wqs_ms = Vec::with_capacity(query_cluster_ids.len());
    let mut adaptive_dc_ms = Vec::with_capacity(query_cluster_ids.len());
    let mut dc_ms = Vec::with_capacity(query_cluster_ids.len());
    let mut plain_ms = Vec::with_capacity(query_cluster_ids.len());
    let mut adaptive_dc_err_abs_pct = Vec::with_capacity(query_cluster_ids.len());
    let mut wqs_err_abs_pct = Vec::with_capacity(query_cluster_ids.len());
    let mut plain_timed_out = false;
    let mut dc_timed_out = false;
    let adaptive_dc_timed_out = false;
    let wqs_timed_out = false;

    for (query_pos, cluster_ids) in query_cluster_ids.iter().enumerate() {
        let plain_eval = if plain_timed_out {
            None
        } else if !disable_timeouts && BenchMethod::Plain.uses_timeout(nprobe) {
            let outcome = run_timed_child_method_once(
                current_exe,
                input_json,
                &dataset.label,
                query_pos,
                nprobe,
                k,
                BenchMethod::Plain,
            )?;
            plain_ms.push(outcome.elapsed_ms);
            if outcome.timed_out {
                plain_timed_out = true;
                None
            } else {
                Some(outcome)
            }
        } else {
            let t0 = Instant::now();
            let plan =
                planner_plain.generate_plan(cluster_ids, cluster_offsets, cluster_sizes, &dc_cm);
            let outcome = EvalOutcome {
                elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
                objective: ub_objective(&plan, &dc_cm),
                timed_out: false,
            };
            plain_ms.push(outcome.elapsed_ms);
            Some(outcome)
        };

        let t_wqs = Instant::now();
        let wqs_plan =
            planner_wqs.generate_plan(cluster_ids, cluster_offsets, cluster_sizes, &adaptive_cm);
        let wqs_outcome = EvalOutcome {
            elapsed_ms: t_wqs.elapsed().as_secs_f64() * 1000.0,
            objective: ub_objective(&wqs_plan, &adaptive_cm),
            timed_out: false,
        };
        wqs_ms.push(wqs_outcome.elapsed_ms);

        let t_adaptive_dc = Instant::now();
        let adaptive_dc_plan = planner_adaptive_dc.generate_plan(
            cluster_ids,
            cluster_offsets,
            cluster_sizes,
            &adaptive_cm,
        );
        let adaptive_dc_outcome = EvalOutcome {
            elapsed_ms: t_adaptive_dc.elapsed().as_secs_f64() * 1000.0,
            objective: ub_objective(&adaptive_dc_plan, &adaptive_cm),
            timed_out: false,
        };
        adaptive_dc_ms.push(adaptive_dc_outcome.elapsed_ms);

        let dc_eval = if dc_timed_out {
            None
        } else if !disable_timeouts && BenchMethod::Dc.uses_timeout(nprobe) {
            let outcome = run_timed_child_method_once(
                current_exe,
                input_json,
                &dataset.label,
                query_pos,
                nprobe,
                k,
                BenchMethod::Dc,
            )?;
            dc_ms.push(outcome.elapsed_ms);
            if outcome.timed_out {
                dc_timed_out = true;
                None
            } else {
                Some(outcome)
            }
        } else {
            let t0 = Instant::now();
            let plan =
                planner_dc.generate_plan(cluster_ids, cluster_offsets, cluster_sizes, &dc_cm);
            let outcome = EvalOutcome {
                elapsed_ms: t0.elapsed().as_secs_f64() * 1000.0,
                objective: ub_objective(&plan, &dc_cm),
                timed_out: false,
            };
            dc_ms.push(outcome.elapsed_ms);
            Some(outcome)
        };

        if let (Some(dc_outcome), Some(plain_outcome)) = (dc_eval.as_ref(), plain_eval.as_ref()) {
            let dc_plain_gap = (dc_outcome.objective - plain_outcome.objective).abs();
            let dc_plain_tol = 1e-6
                * dc_outcome
                    .objective
                    .abs()
                    .max(plain_outcome.objective.abs())
                    .max(1.0);
            if dc_plain_gap > dc_plain_tol {
                return Err(format!(
                    "dc baseline diverged from plain DP at dataset={} nprobe={} K={}: dc={} plain={} diff={}",
                    dataset.label,
                    nprobe,
                    k,
                    dc_outcome.objective,
                    plain_outcome.objective,
                    dc_plain_gap
                )
                .into());
            }
        }

        if let Some(dc_outcome) = dc_eval.as_ref() {
            let adaptive_err_pct = if dc_outcome.objective <= 1e-12 {
                0.0
            } else {
                ((adaptive_dc_outcome.objective - dc_outcome.objective).abs()
                    / dc_outcome.objective)
                    * 100.0
            };
            let wqs_err_pct = if dc_outcome.objective <= 1e-12 {
                0.0
            } else {
                ((wqs_outcome.objective - dc_outcome.objective).abs() / dc_outcome.objective)
                    * 100.0
            };
            adaptive_dc_err_abs_pct.push(adaptive_err_pct);
            wqs_err_abs_pct.push(wqs_err_pct);
        }

        let done = query_pos + 1;
        if done == 1 || done % 5 == 0 || done == query_cluster_ids.len() {
            println!(
                "[progress] dataset={} nprobe={} K={} queries={}/{} plain_done={} dc_done={}",
                dataset.label,
                nprobe,
                k,
                done,
                query_cluster_ids.len(),
                plain_eval.is_some() || plain_timed_out,
                dc_eval.is_some() || dc_timed_out,
            );
        }
    }

    let (plain_med_ms, plain_p95_ms, plain_max_ms) = summarize_times(&plain_ms, plain_timed_out);
    let (dc_med_ms, dc_p95_ms, dc_max_ms) = summarize_times(&dc_ms, dc_timed_out);
    let (adaptive_dc_med_ms, adaptive_dc_p95_ms, adaptive_dc_max_ms) =
        summarize_times(&adaptive_dc_ms, adaptive_dc_timed_out);
    let (wqs_med_ms, wqs_p95_ms, wqs_max_ms) = summarize_times(&wqs_ms, wqs_timed_out);

    let adaptive_over_dc_p95 = if dc_timed_out || adaptive_dc_timed_out {
        f64::NAN
    } else {
        dc_p95_ms / adaptive_dc_p95_ms.max(1e-12)
    };
    let wqs_over_adaptive_dc_p95 = if adaptive_dc_timed_out || wqs_timed_out {
        f64::NAN
    } else {
        adaptive_dc_p95_ms / wqs_p95_ms.max(1e-12)
    };
    let adaptive_dc_err_avg_pct = if dc_timed_out || adaptive_dc_err_abs_pct.is_empty() {
        f64::NAN
    } else {
        adaptive_dc_err_abs_pct.iter().sum::<f64>() / (adaptive_dc_err_abs_pct.len() as f64)
    };
    let adaptive_dc_err_max_pct = if dc_timed_out || adaptive_dc_err_abs_pct.is_empty() {
        f64::NAN
    } else {
        adaptive_dc_err_abs_pct
            .iter()
            .copied()
            .fold(0.0_f64, f64::max)
    };
    let wqs_err_avg_pct = if dc_timed_out || wqs_err_abs_pct.is_empty() {
        f64::NAN
    } else {
        wqs_err_abs_pct.iter().sum::<f64>() / (wqs_err_abs_pct.len() as f64)
    };
    let wqs_err_max_pct = if dc_timed_out || wqs_err_abs_pct.is_empty() {
        f64::NAN
    } else {
        wqs_err_abs_pct.iter().copied().fold(0.0_f64, f64::max)
    };

    Ok(OutputRow {
        dataset: dataset.label.clone(),
        nprobe,
        k,
        queries: query_cluster_ids.len(),
        plain_timed_out,
        plain_med_ms,
        plain_p95_ms,
        plain_max_ms,
        dc_timed_out,
        dc_med_ms,
        dc_p95_ms,
        dc_max_ms,
        adaptive_dc_timed_out,
        adaptive_dc_med_ms,
        adaptive_dc_p95_ms,
        adaptive_dc_max_ms,
        wqs_timed_out,
        wqs_med_ms,
        wqs_p95_ms,
        wqs_max_ms,
        adaptive_over_dc_p95,
        wqs_over_adaptive_dc_p95,
        adaptive_dc_err_avg_pct,
        adaptive_dc_err_max_pct,
        wqs_err_avg_pct,
        wqs_err_max_pct,
    })
}

fn write_csv(path: &Path, rows: &[OutputRow]) -> Result<(), DynError> {
    let mut csv = String::from(
        "dataset,nprobe,K,queries,plain_timed_out,plain_med_ms,plain_p95_ms,plain_max_ms,dc_timed_out,dc_med_ms,dc_p95_ms,dc_max_ms,adaptive_dc_timed_out,adaptive_dc_med_ms,adaptive_dc_p95_ms,adaptive_dc_max_ms,wqs_timed_out,wqs_med_ms,wqs_p95_ms,wqs_max_ms,adaptive_over_dc_p95,wqs_over_adaptive_dc_p95,adaptive_dc_err_avg_pct,adaptive_dc_err_max_pct,wqs_err_avg_pct,wqs_err_max_pct\n",
    );
    for row in rows {
        csv.push_str(&format!(
            "{},{},{},{},{},{:.12},{:.12},{:.12},{},{:.12},{:.12},{:.12},{},{:.12},{:.12},{:.12},{},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12},{:.12}\n",
            row.dataset,
            row.nprobe,
            row.k,
            row.queries,
            row.plain_timed_out,
            row.plain_med_ms,
            row.plain_p95_ms,
            row.plain_max_ms,
            row.dc_timed_out,
            row.dc_med_ms,
            row.dc_p95_ms,
            row.dc_max_ms,
            row.adaptive_dc_timed_out,
            row.adaptive_dc_med_ms,
            row.adaptive_dc_p95_ms,
            row.adaptive_dc_max_ms,
            row.wqs_timed_out,
            row.wqs_med_ms,
            row.wqs_p95_ms,
            row.wqs_max_ms,
            row.adaptive_over_dc_p95,
            row.wqs_over_adaptive_dc_p95,
            row.adaptive_dc_err_avg_pct,
            row.adaptive_dc_err_max_pct,
            row.wqs_err_avg_pct,
            row.wqs_err_max_pct,
        ));
    }
    fs::write(path, csv)?;
    Ok(())
}

fn main() -> Result<(), DynError> {
    let args = parse_args()?;
    let input: PlannerInput = serde_json::from_str(&fs::read_to_string(&args.input_json)?)?;

    if input.benchmark_name != BENCHMARK_NAME {
        return Err(format!(
            "unexpected benchmark_name: {}, expected {}",
            input.benchmark_name, BENCHMARK_NAME
        )
        .into());
    }

    let base_cost_model = cost_model_from_input(&input.latency_model)?;
    if let Some(config) = single_query_config_from_env()? {
        return run_single_query_mode(&input, &base_cost_model, &config);
    }

    let current_exe = env::current_exe()?;
    let mut task_layouts = Vec::with_capacity(input.datasets.len());
    let mut tasks = Vec::new();

    for (dataset_idx, dataset) in input.datasets.iter().enumerate() {
        println!(
            "[dataset] {} nlist={} max_nprobe={} queries={}",
            dataset.label,
            dataset.nlist,
            dataset.max_nprobe,
            dataset.queries.len()
        );
        if dataset.queries.len() < input.benchmark_grid.num_queries {
            return Err(format!(
                "dataset {} has only {} queries, need {}",
                dataset.label,
                dataset.queries.len(),
                input.benchmark_grid.num_queries
            )
            .into());
        }
        let layout = cluster_maps(dataset)?;
        task_layouts.push(layout);

        for &nprobe in &input.benchmark_grid.nprobes {
            if nprobe > dataset.max_nprobe {
                continue;
            }
            for &k in &input.benchmark_grid.ks {
                if k > nprobe {
                    continue;
                }
                tasks.push(BenchmarkTask {
                    dataset_idx,
                    nprobe,
                    k,
                });
            }
        }
    }

    let pool = ThreadPoolBuilder::new().num_threads(args.jobs).build()?;
    let rows: Vec<OutputRow> = pool.install(|| {
        tasks
            .into_par_iter()
            .map(|task| {
                let dataset = &input.datasets[task.dataset_idx];
                let (cluster_offsets, cluster_sizes) = &task_layouts[task.dataset_idx];
                let row = benchmark_combo(
                    &args.input_json,
                    &current_exe,
                    dataset,
                    task.nprobe,
                    task.k,
                    args.disable_timeouts,
                    &base_cost_model,
                    input.benchmark_grid.num_queries,
                    input.benchmark_grid.warmup_queries,
                    cluster_offsets,
                    cluster_sizes,
                )?;
                println!(
                    "[{}] nprobe={} K={} plain_p95={:.3}ms plain_to={} dc_p95={:.3}ms dc_to={} adaptive_dc_p95={:.3}ms wqs_p95={:.3}ms adaptive_err_avg={:.3}% wqs_err_avg={:.3}%",
                    row.dataset,
                    row.nprobe,
                    row.k,
                    row.plain_p95_ms,
                    row.plain_timed_out,
                    row.dc_p95_ms,
                    row.dc_timed_out,
                    row.adaptive_dc_p95_ms,
                    row.wqs_p95_ms,
                    row.adaptive_dc_err_avg_pct,
                    row.wqs_err_avg_pct,
                );
                Ok::<OutputRow, DynError>(row)
            })
            .collect::<Result<Vec<_>, _>>()
    })?;

    let mut rows = rows;
    rows.sort_by(|a, b| {
        a.dataset
            .cmp(&b.dataset)
            .then_with(|| a.nprobe.cmp(&b.nprobe))
            .then_with(|| a.k.cmp(&b.k))
    });

    if let Some(parent) = args.out_json.parent() {
        fs::create_dir_all(parent)?;
    }
    if let Some(parent) = args.out_csv.parent() {
        fs::create_dir_all(parent)?;
    }

    let output = PlannerOutput {
        version: input.version,
        benchmark_name: input.benchmark_name,
        rows,
    };
    fs::write(&args.out_json, serde_json::to_string_pretty(&output)?)?;
    write_csv(&args.out_csv, &output.rows)?;
    println!("[ok] json={}", args.out_json.display());
    println!("[ok] csv={}", args.out_csv.display());
    Ok(())
}
