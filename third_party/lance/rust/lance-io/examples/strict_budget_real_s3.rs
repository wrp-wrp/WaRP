// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Real-S3 strict-budget benchmark used by the paper Figure 9 pipeline.

use std::fs;
use std::ops::Range;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{
    CostModel, LanceMerge, MergeStrategy, ScanScheduler, SchedulerConfig, VecioMergePlanner,
};
use lance_io::utils::CachedFileSize;
use serde::{Deserialize, Serialize};
use snafu::location;
use tokio::sync::Semaphore;

#[derive(Parser, Debug)]
struct Args {
    #[arg(long)]
    dataset: String,
    #[arg(long)]
    original_query_json: PathBuf,
    #[arg(long)]
    reordered_query_json: PathBuf,
    #[arg(long)]
    latency_model_json: PathBuf,
    #[arg(long)]
    out_json: PathBuf,
    #[arg(long)]
    out_csv: Option<PathBuf>,
    #[arg(long, default_value = "8,16,32,64,96,128")]
    ks: String,
    #[arg(long, default_value = "64,128,256,512,1024,2048,4096,8192,16384")]
    greedy_gap_kb: String,
    #[arg(long, default_value = "1,2,4,8,16,32")]
    greedy_span_caps_mb: String,
    #[arg(long, default_value_t = 120)]
    num_queries: usize,
    #[arg(long, default_value_t = 20)]
    warmup_queries: usize,
    #[arg(long, default_value_t = 1)]
    query_concurrency: usize,
    #[arg(long, default_value = "30")]
    balanced_ps: String,
    #[arg(long, default_value_t = false)]
    balanced_exact_requests: bool,
    #[arg(long, default_value_t = 128)]
    process_io_threads_limit: usize,
    #[arg(long, default_value_t = 128)]
    cloud_io_parallelism: usize,
    #[arg(long, default_value_t = 16 * 1024 * 1024)]
    max_iop_size_bytes: u64,
    #[arg(long)]
    dump_plan_stats_json: Option<PathBuf>,
    /// Run only the paper's reordered-layout greedy or WaRP arm.
    #[arg(long, value_enum)]
    only_method: Option<OnlyMethod>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
enum OnlyMethod {
    #[value(name = "greedy_reorder")]
    GreedyReorder,
    #[value(name = "warp_reorder")]
    WarpReorder,
}

impl OnlyMethod {
    fn as_str(self) -> &'static str {
        match self {
            Self::GreedyReorder => "greedy_reorder",
            Self::WarpReorder => "warp_reorder",
        }
    }

    fn runs_warp(self) -> bool {
        self == Self::WarpReorder
    }

    fn runs_greedy(self) -> bool {
        self == Self::GreedyReorder
    }
}

#[derive(Debug, Deserialize)]
struct QueryRangesFile {
    uri: Option<String>,
    nlist: Option<usize>,
    nprobe: Option<usize>,
    queries: Vec<Vec<[u64; 2]>>,
    warmup_queries: Option<Vec<Vec<[u64; 2]>>>,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum QueryRangesInput {
    Bare(Vec<Vec<[u64; 2]>>),
    WithMeta(QueryRangesFile),
}

#[derive(Debug)]
struct LoadedQueryRanges {
    uri: String,
    nlist: usize,
    nprobe: usize,
    queries: Vec<Vec<Range<u64>>>,
    warmup_queries: Vec<Vec<Range<u64>>>,
}

#[derive(Debug, Deserialize)]
struct LatencyModelSpec {
    latency_model: String,
    latency_pw_breaks_bytes: Option<Vec<i64>>,
    latency_pw_overheads_ms: Option<Vec<f64>>,
    latency_pw_rates_ms_per_byte: Option<Vec<f64>>,
    lat_req_ms: Option<f64>,
    lat_byte_ms: Option<f64>,
}

#[derive(Debug, Clone)]
struct PerQuerySample {
    latency_ms: f64,
    planned_ranges: usize,
    planned_bytes: u64,
    model_work_ms: f64,
    model_bottleneck_ms: f64,
    model_objective_ms: f64,
}

#[derive(Debug, Clone, Serialize)]
struct BenchmarkRow {
    dataset: String,
    layout: String,
    method: String,
    budget_k: Option<usize>,
    status: String,
    coverage: f64,
    wall_ms: Option<f64>,
    qps: Option<f64>,
    avg_latency_ms: Option<f64>,
    p95_latency_ms: Option<f64>,
    avg_num_requests: Option<f64>,
    avg_planned_bytes: Option<f64>,
    avg_model_work_ms: Option<f64>,
    avg_model_bottleneck_ms: Option<f64>,
    avg_model_objective_ms: Option<f64>,
    best_gap_kb: Option<u64>,
    best_span_cap_bytes: Option<u64>,
}

#[derive(Debug, Clone, Serialize)]
struct GreedyCandidateRow {
    dataset: String,
    layout: String,
    budget_k: usize,
    gap_kb: u64,
    span_cap_bytes: u64,
    coverage: f64,
    wall_ms: Option<f64>,
    qps: Option<f64>,
    avg_latency_ms: Option<f64>,
    p95_latency_ms: Option<f64>,
    avg_num_requests: Option<f64>,
    avg_planned_bytes: Option<f64>,
    avg_model_work_ms: Option<f64>,
    avg_model_bottleneck_ms: Option<f64>,
    avg_model_objective_ms: Option<f64>,
    selected_for_curve: bool,
}

#[derive(Debug, Clone, Serialize)]
struct GreedyUnboundedCandidateRow {
    dataset: String,
    layout: String,
    gap_kb: u64,
    span_cap_bytes: u64,
    wall_ms: Option<f64>,
    qps: Option<f64>,
    avg_latency_ms: Option<f64>,
    p95_latency_ms: Option<f64>,
    avg_num_requests: Option<f64>,
    avg_planned_bytes: Option<f64>,
    avg_model_work_ms: Option<f64>,
    avg_model_bottleneck_ms: Option<f64>,
    avg_model_objective_ms: Option<f64>,
}

#[derive(Debug, Clone, Serialize)]
struct PlanStatsRow {
    dataset: String,
    layout: String,
    method: String,
    budget_k: Option<usize>,
    query_idx: usize,
    planned_ranges: usize,
    planned_bytes: u64,
    min_request_bytes: u64,
    p50_request_bytes: u64,
    p95_request_bytes: u64,
    max_request_bytes: u64,
    request_bytes_stddev: f64,
    coeff_var: f64,
}

#[derive(Debug, Serialize)]
struct OutputDoc {
    dataset: String,
    nlist: usize,
    nprobe: usize,
    num_queries: usize,
    warmup_queries: usize,
    query_concurrency: usize,
    balanced_ps: Vec<f64>,
    process_io_threads_limit: usize,
    cloud_io_parallelism: usize,
    max_iop_size_bytes: u64,
    only_method: Option<String>,
    rows: Vec<BenchmarkRow>,
    greedy_candidates: Vec<GreedyCandidateRow>,
    greedy_unbounded_candidates: Vec<GreedyUnboundedCandidateRow>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    plan_stats: Vec<PlanStatsRow>,
}

#[derive(Debug)]
struct RunSummary {
    per_query: Vec<PerQuerySample>,
    wall_ms: f64,
    qps: f64,
}

fn parse_int_list(raw: &str) -> lance_core::Result<Vec<usize>> {
    let mut out = Vec::new();
    for tok in raw.split(',') {
        let s = tok.trim();
        if s.is_empty() {
            continue;
        }
        let v = s.parse::<usize>().map_err(|e| {
            lance_core::Error::invalid_input(
                format!("failed to parse integer list element {}: {}", s, e),
                location!(),
            )
        })?;
        out.push(v);
    }
    out.sort_unstable();
    out.dedup();
    if out.is_empty() {
        return Err(lance_core::Error::invalid_input(
            "parsed empty integer list",
            location!(),
        ));
    }
    Ok(out)
}

fn parse_float_list(raw: &str) -> lance_core::Result<Vec<f64>> {
    let mut out = Vec::new();
    for tok in raw.split(',') {
        let s = tok.trim();
        if s.is_empty() {
            continue;
        }
        let v = s.parse::<f64>().map_err(|e| {
            lance_core::Error::invalid_input(
                format!("failed to parse float list element {}: {}", s, e),
                location!(),
            )
        })?;
        out.push(v);
    }
    out.sort_by(|a, b| a.total_cmp(b));
    out.dedup_by(|a, b| (*a - *b).abs() < 1e-12);
    if out.is_empty() {
        return Err(lance_core::Error::invalid_input(
            "parsed empty float list",
            location!(),
        ));
    }
    Ok(out)
}

fn format_p_tag(p: f64) -> String {
    let s = if (p - p.round()).abs() < 1e-9 {
        format!("{:.0}", p)
    } else {
        let mut s = format!("{:.3}", p);
        while s.contains('.') && s.ends_with('0') {
            s.pop();
        }
        if s.ends_with('.') {
            s.pop();
        }
        s
    };
    s.replace('.', "p")
}

fn percentile(vals: &[f64], p: f64) -> f64 {
    if vals.is_empty() {
        return f64::NAN;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.total_cmp(b));
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn percentile_u64(vals: &[u64], p: f64) -> u64 {
    if vals.is_empty() {
        return 0;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() - 1) as f64 * p).round() as usize;
    sorted[idx]
}

fn convert_raw_queries(raw: Vec<Vec<[u64; 2]>>) -> lance_core::Result<Vec<Vec<Range<u64>>>> {
    let mut out = Vec::with_capacity(raw.len());
    for (qi, ranges) in raw.into_iter().enumerate() {
        let mut q = Vec::with_capacity(ranges.len());
        for (ri, pair) in ranges.into_iter().enumerate() {
            let start = pair[0];
            let end = pair[1];
            if end <= start {
                return Err(lance_core::Error::invalid_input(
                    format!(
                        "invalid query range at query {} range {}: [{}, {})",
                        qi, ri, start, end
                    ),
                    location!(),
                ));
            }
            q.push(start..end);
        }
        out.push(q);
    }
    Ok(out)
}

fn load_query_ranges_file(
    path: &PathBuf,
    num_queries: usize,
    warmup_queries: usize,
) -> lance_core::Result<LoadedQueryRanges> {
    let text = fs::read_to_string(path).map_err(|e| {
        lance_core::Error::io(
            format!("failed to read query ranges file {}: {}", path.display(), e),
            location!(),
        )
    })?;
    let parsed: QueryRangesInput = serde_json::from_str(&text).map_err(|e| {
        lance_core::Error::invalid_input(
            format!(
                "failed to parse query ranges JSON {}: {}",
                path.display(),
                e
            ),
            location!(),
        )
    })?;

    let (uri, nlist, nprobe, queries, warmup_pool) = match parsed {
        QueryRangesInput::Bare(queries) => (None, None, None, convert_raw_queries(queries)?, None),
        QueryRangesInput::WithMeta(v) => (
            v.uri,
            v.nlist,
            v.nprobe,
            convert_raw_queries(v.queries)?,
            match v.warmup_queries {
                Some(w) => Some(convert_raw_queries(w)?),
                None => None,
            },
        ),
    };

    let query_pool = queries.into_iter().take(num_queries).collect::<Vec<_>>();
    let warmup = match warmup_pool {
        Some(w) if !w.is_empty() => w.into_iter().take(warmup_queries).collect::<Vec<_>>(),
        _ => query_pool
            .iter()
            .take(warmup_queries)
            .cloned()
            .collect::<Vec<_>>(),
    };
    if query_pool.is_empty() {
        return Err(lance_core::Error::invalid_input(
            format!("query file {} contains no usable queries", path.display()),
            location!(),
        ));
    }
    Ok(LoadedQueryRanges {
        uri: uri.ok_or_else(|| {
            lance_core::Error::invalid_input(
                format!("query file {} is missing uri", path.display()),
                location!(),
            )
        })?,
        nlist: nlist.unwrap_or(0),
        nprobe: nprobe.unwrap_or(0),
        queries: query_pool,
        warmup_queries: warmup,
    })
}

fn load_latency_model(path: &PathBuf) -> lance_core::Result<LatencyModelSpec> {
    let text = fs::read_to_string(path).map_err(|e| {
        lance_core::Error::io(
            format!("failed to read latency model {}: {}", path.display(), e),
            location!(),
        )
    })?;
    serde_json::from_str(&text).map_err(|e| {
        lance_core::Error::invalid_input(
            format!("failed to parse latency model {}: {}", path.display(), e),
            location!(),
        )
    })
}

fn apply_latency_model(
    cost_model: &mut CostModel,
    spec: &LatencyModelSpec,
) -> lance_core::Result<()> {
    match spec.latency_model.trim().to_ascii_lowercase().as_str() {
        "linear" => {
            cost_model.latency_model = "linear".to_string();
            cost_model.latency_per_request = spec.lat_req_ms.unwrap_or(0.0);
            cost_model.latency_per_byte = spec.lat_byte_ms.unwrap_or(0.0);
        }
        "piecewise_linear" => {
            cost_model.latency_model = "piecewise_linear".to_string();
            cost_model.latency_pw_breaks_bytes =
                spec.latency_pw_breaks_bytes.clone().unwrap_or_default();
            cost_model.latency_pw_overheads_ms =
                spec.latency_pw_overheads_ms.clone().unwrap_or_default();
            cost_model.latency_pw_rates_ms_per_byte = spec
                .latency_pw_rates_ms_per_byte
                .clone()
                .unwrap_or_default();
            if cost_model.latency_pw_breaks_bytes.is_empty()
                || cost_model.latency_pw_overheads_ms.is_empty()
                || cost_model.latency_pw_rates_ms_per_byte.is_empty()
            {
                return Err(lance_core::Error::invalid_input(
                    "piecewise latency model is missing breaks/overheads/rates",
                    location!(),
                ));
            }
        }
        other => {
            return Err(lance_core::Error::invalid_input(
                format!("unsupported latency_model: {}", other),
                location!(),
            ))
        }
    }
    Ok(())
}

fn split_large_range(req: Range<u64>, max_iop_size: u64, out: &mut Vec<Range<u64>>) {
    if req.is_empty() {
        out.push(req);
        return;
    }
    if req.end - req.start <= max_iop_size {
        out.push(req);
        return;
    }
    let num_requests = (req.end - req.start).div_ceil(max_iop_size);
    let bytes_per_request = (req.end - req.start) / num_requests;
    for i in 0..num_requests {
        let start = req.start + i * bytes_per_request;
        let end = if i == num_requests - 1 {
            req.end
        } else {
            start + bytes_per_request
        };
        out.push(start..end);
    }
}

fn split_planned_ranges(planned: Vec<Range<u64>>, max_iop_size: u64) -> Vec<Range<u64>> {
    let mut out = Vec::new();
    for req in planned {
        split_large_range(req, max_iop_size, &mut out);
    }
    out
}

fn query_to_cluster_maps(
    query: &[Range<u64>],
) -> (
    Vec<i32>,
    std::collections::HashMap<i32, i64>,
    std::collections::HashMap<i32, i64>,
) {
    let mut cluster_ids = Vec::with_capacity(query.len());
    let mut offsets = std::collections::HashMap::with_capacity(query.len());
    let mut sizes = std::collections::HashMap::with_capacity(query.len());
    for (idx, req) in query.iter().enumerate() {
        let cid = idx as i32;
        cluster_ids.push(cid);
        offsets.insert(cid, req.start as i64);
        sizes.insert(cid, (req.end - req.start) as i64);
    }
    (cluster_ids, offsets, sizes)
}

fn io_plan_to_ranges(plan: lance_io::scheduler::IOPlan, max_iop_size: u64) -> Vec<Range<u64>> {
    let mut planned = Vec::with_capacity(plan.requests.len());
    for req in plan.requests {
        let start = req.offset.max(0) as u64;
        let end = start.saturating_add(req.length.max(0) as u64);
        planned.push(start..end);
    }
    planned.sort_by_key(|r| r.start);
    split_planned_ranges(planned, max_iop_size)
}

fn plan_separate(query: &[Range<u64>], max_iop_size: u64) -> Vec<Range<u64>> {
    split_planned_ranges(query.to_vec(), max_iop_size)
}

fn plan_lance_native(query: &[Range<u64>], max_iop_size: u64) -> Vec<Range<u64>> {
    let (cluster_ids, offsets, sizes) = query_to_cluster_maps(query);
    let cost_model = CostModel::default();
    let plan = LanceMerge::new().generate_plan(&cluster_ids, &offsets, &sizes, &cost_model);
    io_plan_to_ranges(plan, max_iop_size)
}

fn plan_balanced(
    query: &[Range<u64>],
    latency_model: &LatencyModelSpec,
    budget_k: usize,
    balanced_p: f64,
    exact_requests: bool,
    max_iop_size: u64,
) -> lance_core::Result<Vec<Range<u64>>> {
    let (cluster_ids, offsets, sizes) = query_to_cluster_maps(query);
    let mut cost_model = CostModel::default();
    apply_latency_model(&mut cost_model, latency_model)?;
    cost_model.max_requests = budget_k as i32;
    cost_model.exact_requests = exact_requests;
    cost_model.concurrent_parallelism = balanced_p;
    // Keep the planner's feasible region aligned with the executor:
    // requests larger than max_iop_size would be split before issue,
    // so treating them as single requests during optimization is too optimistic.
    cost_model.max_merge_span = max_iop_size.min(i64::MAX as u64) as i64;
    let plan = VecioMergePlanner::with_fast_path_optimizations(true).generate_plan(
        &cluster_ids,
        &offsets,
        &sizes,
        &cost_model,
    );
    Ok(io_plan_to_ranges(plan, max_iop_size))
}

fn plan_greedy_gap_frontier(
    query: &[Range<u64>],
    gap_threshold_bytes: u64,
    span_cap_bytes: u64,
    max_iop_size: u64,
) -> Vec<Range<u64>> {
    if query.is_empty() {
        return Vec::new();
    }
    let mut sorted = query.to_vec();
    sorted.sort_by_key(|r| r.start);
    let mut merged: Vec<Range<u64>> = Vec::with_capacity(sorted.len());
    let mut current = sorted[0].clone();
    for next in sorted.into_iter().skip(1) {
        let gap = next.start.saturating_sub(current.end);
        let merged_end = current.end.max(next.end);
        let merged_span = merged_end.saturating_sub(current.start);
        if gap <= gap_threshold_bytes && merged_span <= span_cap_bytes {
            current.end = merged_end;
        } else {
            merged.push(current);
            current = next;
        }
    }
    merged.push(current);
    split_planned_ranges(merged, max_iop_size)
}

fn compute_objective_coefficients(parallelism: f64) -> (f64, f64) {
    if parallelism <= 1e-9 {
        return (0.0, 1.0);
    }
    let p = parallelism.max(1.0);
    let coeff_sum = 1.0 / p;
    (coeff_sum, 1.0 - coeff_sum)
}

fn work_and_bottleneck_for_ranges_ms(planned: &[Range<u64>], cost_model: &CostModel) -> (f64, f64) {
    if planned.is_empty() {
        return (f64::INFINITY, f64::INFINITY);
    }
    let mut total_work = 0.0_f64;
    let mut max_latency = 0.0_f64;
    for req in planned {
        let latency = cost_model.request_latency_ms((req.end.saturating_sub(req.start)) as i64);
        total_work += latency;
        max_latency = max_latency.max(latency);
    }
    (total_work, max_latency)
}

fn objective_for_ranges_ms(
    planned: &[Range<u64>],
    cost_model: &CostModel,
    parallelism: f64,
) -> f64 {
    let (total_work, max_latency) = work_and_bottleneck_for_ranges_ms(planned, cost_model);
    if !total_work.is_finite() || !max_latency.is_finite() {
        return f64::INFINITY;
    }
    let (coeff_sum, coeff_bottleneck) = compute_objective_coefficients(parallelism);
    coeff_sum * total_work + coeff_bottleneck * max_latency
}

fn build_plan_stats_row(
    dataset: &str,
    layout: &str,
    method: &str,
    budget_k: Option<usize>,
    query_idx: usize,
    planned: &[Range<u64>],
) -> PlanStatsRow {
    let mut sizes = planned
        .iter()
        .map(|r| r.end.saturating_sub(r.start))
        .collect::<Vec<_>>();
    sizes.sort_unstable();
    let planned_bytes = sizes.iter().copied().sum::<u64>();
    let min_request_bytes = sizes.first().copied().unwrap_or(0);
    let max_request_bytes = sizes.last().copied().unwrap_or(0);
    let p50_request_bytes = percentile_u64(&sizes, 0.50);
    let p95_request_bytes = percentile_u64(&sizes, 0.95);
    let mean = if sizes.is_empty() {
        0.0
    } else {
        planned_bytes as f64 / sizes.len() as f64
    };
    let variance = if sizes.len() <= 1 {
        0.0
    } else {
        sizes
            .iter()
            .map(|v| {
                let d = *v as f64 - mean;
                d * d
            })
            .sum::<f64>()
            / sizes.len() as f64
    };
    let stddev = variance.sqrt();
    let coeff_var = if mean > 0.0 { stddev / mean } else { 0.0 };
    PlanStatsRow {
        dataset: dataset.to_string(),
        layout: layout.to_string(),
        method: method.to_string(),
        budget_k,
        query_idx,
        planned_ranges: planned.len(),
        planned_bytes,
        min_request_bytes,
        p50_request_bytes,
        p95_request_bytes,
        max_request_bytes,
        request_bytes_stddev: stddev,
        coeff_var,
    }
}

async fn run_planned_queries(
    file_scheduler: Arc<lance_io::scheduler::FileScheduler>,
    warmup_queries: &[Vec<Range<u64>>],
    queries: &[Vec<Range<u64>>],
    planner: impl Fn(&[Range<u64>]) -> lance_core::Result<Vec<Range<u64>>>,
    objective_cost_model: &CostModel,
    objective_parallelism: f64,
    query_concurrency: usize,
) -> lance_core::Result<RunSummary> {
    for query in warmup_queries {
        let planned = planner(query)?;
        let _ = file_scheduler
            .submit_request_with_plan(query.clone(), planned, 0)
            .await?;
    }

    let mut planned_queries = Vec::with_capacity(queries.len());
    for query in queries {
        let planned = planner(query)?;
        let planned_ranges = planned.len();
        let planned_bytes = planned
            .iter()
            .map(|r| r.end.saturating_sub(r.start))
            .sum::<u64>();
        let (model_work_ms, model_bottleneck_ms) =
            work_and_bottleneck_for_ranges_ms(&planned, objective_cost_model);
        let model_objective_ms =
            objective_for_ranges_ms(&planned, objective_cost_model, objective_parallelism);
        planned_queries.push((
            query.clone(),
            planned,
            planned_ranges,
            planned_bytes,
            model_work_ms,
            model_bottleneck_ms,
            model_objective_ms,
        ));
    }

    let bench_start = Instant::now();
    let mut out = Vec::with_capacity(planned_queries.len());
    if query_concurrency <= 1 {
        for (
            query,
            planned,
            planned_ranges,
            planned_bytes,
            model_work_ms,
            model_bottleneck_ms,
            model_objective_ms,
        ) in planned_queries
        {
            let t0 = Instant::now();
            let _ = file_scheduler
                .submit_request_with_plan(query, planned, 0)
                .await?;
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            out.push(PerQuerySample {
                latency_ms: elapsed_ms,
                planned_ranges,
                planned_bytes,
                model_work_ms,
                model_bottleneck_ms,
                model_objective_ms,
            });
        }
    } else {
        let semaphore = Arc::new(Semaphore::new(query_concurrency));
        let fs = file_scheduler.clone();
        let mut handles = Vec::with_capacity(planned_queries.len());
        for (
            idx,
            (
                query,
                planned,
                planned_ranges,
                planned_bytes,
                model_work_ms,
                model_bottleneck_ms,
                model_objective_ms,
            ),
        ) in planned_queries.into_iter().enumerate()
        {
            let fs = fs.clone();
            let sem = semaphore.clone();
            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.map_err(|e| {
                    lance_core::Error::io(
                        format!("failed to acquire query semaphore: {e}"),
                        location!(),
                    )
                })?;
                let t0 = Instant::now();
                let _ = fs.submit_request_with_plan(query, planned, 0).await?;
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                Ok::<(usize, PerQuerySample), lance_core::Error>((
                    idx,
                    PerQuerySample {
                        latency_ms: elapsed_ms,
                        planned_ranges,
                        planned_bytes,
                        model_work_ms,
                        model_bottleneck_ms,
                        model_objective_ms,
                    },
                ))
            }));
        }
        let mut ordered: Vec<Option<PerQuerySample>> = (0..handles.len()).map(|_| None).collect();
        for handle in handles {
            let (idx, sample) = handle.await.map_err(|e| {
                lance_core::Error::io(
                    format!("task join failed in strict_budget_real_s3: {e}"),
                    location!(),
                )
            })??;
            ordered[idx] = Some(sample);
        }
        out = ordered.into_iter().flatten().collect::<Vec<_>>();
    }
    let wall_ms = bench_start.elapsed().as_secs_f64() * 1000.0;
    let qps = if wall_ms > 0.0 {
        out.len() as f64 / (wall_ms / 1000.0)
    } else {
        0.0
    };
    Ok(RunSummary {
        per_query: out,
        wall_ms,
        qps,
    })
}

fn summarize_unbounded(
    dataset: &str,
    layout: &str,
    method: &str,
    run: &RunSummary,
) -> BenchmarkRow {
    let per_query = &run.per_query;
    let latencies = per_query.iter().map(|r| r.latency_ms).collect::<Vec<_>>();
    let avg_latency = if latencies.is_empty() {
        None
    } else {
        Some(latencies.iter().sum::<f64>() / latencies.len() as f64)
    };
    let avg_requests = if per_query.is_empty() {
        None
    } else {
        Some(
            per_query
                .iter()
                .map(|r| r.planned_ranges as f64)
                .sum::<f64>()
                / per_query.len() as f64,
        )
    };
    let avg_planned_bytes = if per_query.is_empty() {
        None
    } else {
        Some(
            per_query
                .iter()
                .map(|r| r.planned_bytes as f64)
                .sum::<f64>()
                / per_query.len() as f64,
        )
    };
    let avg_model_work_ms = if per_query.is_empty() {
        None
    } else {
        Some(per_query.iter().map(|r| r.model_work_ms).sum::<f64>() / per_query.len() as f64)
    };
    let avg_model_bottleneck_ms = if per_query.is_empty() {
        None
    } else {
        Some(per_query.iter().map(|r| r.model_bottleneck_ms).sum::<f64>() / per_query.len() as f64)
    };
    let avg_model_objective_ms = if per_query.is_empty() {
        None
    } else {
        Some(per_query.iter().map(|r| r.model_objective_ms).sum::<f64>() / per_query.len() as f64)
    };
    BenchmarkRow {
        dataset: dataset.to_string(),
        layout: layout.to_string(),
        method: method.to_string(),
        budget_k: None,
        status: "ok".to_string(),
        coverage: 1.0,
        wall_ms: Some(run.wall_ms),
        qps: Some(run.qps),
        avg_latency_ms: avg_latency,
        p95_latency_ms: if latencies.is_empty() {
            None
        } else {
            Some(percentile(&latencies, 0.95))
        },
        avg_num_requests: avg_requests,
        avg_planned_bytes,
        avg_model_work_ms,
        avg_model_bottleneck_ms,
        avg_model_objective_ms,
        best_gap_kb: None,
        best_span_cap_bytes: None,
    }
}

fn summarize_under_k(
    dataset: &str,
    layout: &str,
    method: &str,
    budget_k: usize,
    run: &RunSummary,
) -> BenchmarkRow {
    let per_query = &run.per_query;
    let feasible = per_query
        .iter()
        // An empty plan on a non-empty query is not a successful under-K plan.
        // Balanced-UB can currently surface infeasible cases as zero planned
        // ranges; counting those as "0 requests" inflates QPS and hides the
        // boundary regime we want to report.
        .filter(|r| r.planned_ranges > 0 && r.planned_ranges <= budget_k)
        .collect::<Vec<_>>();
    let coverage = if per_query.is_empty() {
        0.0
    } else {
        feasible.len() as f64 / per_query.len() as f64
    };
    if feasible.is_empty() {
        return BenchmarkRow {
            dataset: dataset.to_string(),
            layout: layout.to_string(),
            method: method.to_string(),
            budget_k: Some(budget_k),
            status: "no_data".to_string(),
            coverage,
            wall_ms: Some(run.wall_ms),
            qps: Some(run.qps * coverage),
            avg_latency_ms: None,
            p95_latency_ms: None,
            avg_num_requests: None,
            avg_planned_bytes: None,
            avg_model_work_ms: None,
            avg_model_bottleneck_ms: None,
            avg_model_objective_ms: None,
            best_gap_kb: None,
            best_span_cap_bytes: None,
        };
    }
    let latencies = feasible.iter().map(|r| r.latency_ms).collect::<Vec<_>>();
    let avg_latency = latencies.iter().sum::<f64>() / latencies.len() as f64;
    let avg_requests = feasible
        .iter()
        .map(|r| r.planned_ranges as f64)
        .sum::<f64>()
        / feasible.len() as f64;
    let avg_planned_bytes =
        feasible.iter().map(|r| r.planned_bytes as f64).sum::<f64>() / feasible.len() as f64;
    let avg_model_work_ms =
        feasible.iter().map(|r| r.model_work_ms).sum::<f64>() / feasible.len() as f64;
    let avg_model_bottleneck_ms =
        feasible.iter().map(|r| r.model_bottleneck_ms).sum::<f64>() / feasible.len() as f64;
    let avg_model_objective_ms =
        feasible.iter().map(|r| r.model_objective_ms).sum::<f64>() / feasible.len() as f64;
    BenchmarkRow {
        dataset: dataset.to_string(),
        layout: layout.to_string(),
        method: method.to_string(),
        budget_k: Some(budget_k),
        status: if coverage >= 1.0 - 1e-12 {
            "ok".to_string()
        } else {
            "partial".to_string()
        },
        coverage,
        wall_ms: Some(run.wall_ms),
        qps: Some(run.qps * coverage),
        avg_latency_ms: Some(avg_latency),
        p95_latency_ms: Some(percentile(&latencies, 0.95)),
        avg_num_requests: Some(avg_requests),
        avg_planned_bytes: Some(avg_planned_bytes),
        avg_model_work_ms: Some(avg_model_work_ms),
        avg_model_bottleneck_ms: Some(avg_model_bottleneck_ms),
        avg_model_objective_ms: Some(avg_model_objective_ms),
        best_gap_kb: None,
        best_span_cap_bytes: None,
    }
}

fn choose_best_greedy_row(rows: &[BenchmarkRow]) -> Option<BenchmarkRow> {
    let feasible: Vec<&BenchmarkRow> = rows.iter().filter(|r| r.coverage >= 1.0 - 1e-12).collect();
    if feasible.is_empty() {
        return None;
    }
    if feasible.iter().any(|r| r.qps.is_some()) {
        feasible
            .into_iter()
            .filter(|r| r.qps.is_some())
            .max_by(|a, b| {
                let ka = (
                    a.qps.unwrap_or(f64::NEG_INFINITY),
                    -a.avg_latency_ms.unwrap_or(f64::INFINITY),
                    -a.avg_num_requests.unwrap_or(f64::INFINITY),
                );
                let kb = (
                    b.qps.unwrap_or(f64::NEG_INFINITY),
                    -b.avg_latency_ms.unwrap_or(f64::INFINITY),
                    -b.avg_num_requests.unwrap_or(f64::INFINITY),
                );
                ka.0.total_cmp(&kb.0)
                    .then_with(|| ka.1.total_cmp(&kb.1))
                    .then_with(|| ka.2.total_cmp(&kb.2))
            })
            .cloned()
    } else {
        feasible
            .into_iter()
            .filter(|r| r.avg_latency_ms.is_some())
            .min_by(|a, b| {
                let ka = (
                    a.avg_latency_ms.unwrap_or(f64::INFINITY),
                    a.p95_latency_ms.unwrap_or(f64::INFINITY),
                    a.avg_num_requests.unwrap_or(f64::INFINITY),
                );
                let kb = (
                    b.avg_latency_ms.unwrap_or(f64::INFINITY),
                    b.p95_latency_ms.unwrap_or(f64::INFINITY),
                    b.avg_num_requests.unwrap_or(f64::INFINITY),
                );
                ka.0.total_cmp(&kb.0)
                    .then_with(|| ka.1.total_cmp(&kb.1))
                    .then_with(|| ka.2.total_cmp(&kb.2))
            })
            .cloned()
    }
}

fn write_csv(
    path: &PathBuf,
    rows: &[BenchmarkRow],
    greedy_candidates: &[GreedyCandidateRow],
    greedy_unbounded_candidates: &[GreedyUnboundedCandidateRow],
) -> lance_core::Result<()> {
    let mut text = String::from(
        "section,dataset,layout,method,budget_k,status,coverage,qps,avg_latency_ms,p95_latency_ms,avg_num_requests,avg_planned_bytes,avg_model_work_ms,avg_model_bottleneck_ms,avg_model_objective_ms,best_gap_kb,best_span_cap_bytes\n",
    );
    for row in rows {
        text.push_str(&format!(
            "summary,{},{},{},{},{},{:.6},{},{},{},{},{},{},{},{},{},{}\n",
            row.dataset,
            row.layout,
            row.method,
            row.budget_k.map(|v| v.to_string()).unwrap_or_default(),
            row.status,
            row.coverage,
            row.qps.map(|v| format!("{:.6}", v)).unwrap_or_default(),
            row.avg_latency_ms
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.p95_latency_ms
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.avg_num_requests
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.avg_planned_bytes
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.avg_model_work_ms
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.avg_model_bottleneck_ms
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.avg_model_objective_ms
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.best_gap_kb
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.best_span_cap_bytes
                .map(|v| v.to_string())
                .unwrap_or_default(),
        ));
    }
    for row in greedy_candidates {
        text.push_str(&format!(
            "greedy_candidate,{},{},greedy_gap_frontier,{},{},{:.6},{},{},{},{},{},{},{},{},{},{}\n",
            row.dataset,
            row.layout,
            row.budget_k,
            if row.coverage >= 1.0 - 1e-12 {
                "ok"
            } else {
                "partial"
            },
            row.coverage,
            row.qps.map(|v| format!("{:.6}", v)).unwrap_or_default(),
            row.avg_latency_ms
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.p95_latency_ms
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.avg_num_requests
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.avg_planned_bytes
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.avg_model_work_ms
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.avg_model_bottleneck_ms
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.avg_model_objective_ms
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.gap_kb,
            row.span_cap_bytes,
        ));
    }
    for row in greedy_unbounded_candidates {
        text.push_str(&format!(
            "greedy_unbounded,{},{},greedy_gap_frontier_unbounded,,ok,1.000000,{},{},{},{},{},{},{},{},{},{}\n",
            row.dataset,
            row.layout,
            row.qps.map(|v| format!("{:.6}", v)).unwrap_or_default(),
            row.avg_latency_ms
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.p95_latency_ms
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.avg_num_requests
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.avg_planned_bytes
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.avg_model_work_ms
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.avg_model_bottleneck_ms
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.avg_model_objective_ms
                .map(|v| format!("{:.6}", v))
                .unwrap_or_default(),
            row.gap_kb,
            row.span_cap_bytes,
        ));
    }
    fs::write(path, text).map_err(|e| {
        lance_core::Error::io(
            format!("failed to write csv {}: {}", path.display(), e),
            location!(),
        )
    })
}

#[tokio::main]
async fn main() -> lance_core::Result<()> {
    let args = Args::parse();
    std::env::set_var(
        "LANCE_PROCESS_IO_THREADS_LIMIT",
        args.process_io_threads_limit.to_string(),
    );
    std::env::set_var("LANCE_IO_THREADS", args.cloud_io_parallelism.to_string());
    if std::env::var("LANCE_IO_MERGE_STRATEGY").is_ok() {
        return Err(lance_core::Error::invalid_input(
            "unset LANCE_IO_MERGE_STRATEGY when running strict_budget_real_s3; the example selects methods internally",
            location!(),
        ));
    }

    let ks = parse_int_list(&args.ks)?;
    let gap_kb_values = parse_int_list(&args.greedy_gap_kb)?;
    let span_caps_mb = parse_int_list(&args.greedy_span_caps_mb)?;
    let balanced_ps = parse_float_list(&args.balanced_ps)?;
    let span_caps_bytes = span_caps_mb
        .iter()
        .map(|mb| (*mb as u64) * 1024 * 1024)
        .collect::<Vec<_>>();

    let original = load_query_ranges_file(
        &args.original_query_json,
        args.num_queries,
        args.warmup_queries,
    )?;
    let reordered = load_query_ranges_file(
        &args.reordered_query_json,
        args.num_queries,
        args.warmup_queries,
    )?;
    let latency_model = load_latency_model(&args.latency_model_json)?;
    let greedy_objective_parallelism = balanced_ps.last().copied().unwrap_or(1.0);
    let mut objective_cost_model = CostModel::default();
    apply_latency_model(&mut objective_cost_model, &latency_model)?;

    let (orig_store, orig_path) = ObjectStore::from_uri(&original.uri).await?;
    let orig_scheduler = ScanScheduler::new(
        orig_store.clone(),
        SchedulerConfig::max_bandwidth(&orig_store),
    );
    let orig_file = Arc::new(
        orig_scheduler
            .open_file(&orig_path, &CachedFileSize::unknown())
            .await?,
    );

    let (re_store, re_path) = ObjectStore::from_uri(&reordered.uri).await?;
    let re_scheduler =
        ScanScheduler::new(re_store.clone(), SchedulerConfig::max_bandwidth(&re_store));
    let re_file = Arc::new(
        re_scheduler
            .open_file(&re_path, &CachedFileSize::unknown())
            .await?,
    );

    let mut rows = Vec::new();
    if args.only_method.is_none() {
        let native_per_query = run_planned_queries(
            orig_file.clone(),
            &original.warmup_queries,
            &original.queries,
            |query| Ok(plan_lance_native(query, args.max_iop_size_bytes)),
            &objective_cost_model,
            greedy_objective_parallelism,
            args.query_concurrency,
        )
        .await?;
        let native_reorder_per_query = run_planned_queries(
            re_file.clone(),
            &reordered.warmup_queries,
            &reordered.queries,
            |query| Ok(plan_lance_native(query, args.max_iop_size_bytes)),
            &objective_cost_model,
            greedy_objective_parallelism,
            args.query_concurrency,
        )
        .await?;
        let separate_per_query = run_planned_queries(
            orig_file.clone(),
            &original.warmup_queries,
            &original.queries,
            |query| Ok(plan_separate(query, args.max_iop_size_bytes)),
            &objective_cost_model,
            greedy_objective_parallelism,
            args.query_concurrency,
        )
        .await?;
        rows.extend([
            summarize_unbounded(&args.dataset, "original", "lance_native", &native_per_query),
            summarize_unbounded(
                &args.dataset,
                "reordered",
                "lance_native_reorder",
                &native_reorder_per_query,
            ),
            summarize_unbounded(
                &args.dataset,
                "original",
                "separate_unbounded",
                &separate_per_query,
            ),
        ]);
    }
    let mut plan_stats = Vec::new();

    for &balanced_p in &balanced_ps {
        let p_tag = format_p_tag(balanced_p);
        let original_method = if balanced_ps.len() == 1 {
            if args.balanced_exact_requests {
                "warp_original_exactk".to_string()
            } else {
                "warp_original".to_string()
            }
        } else {
            if args.balanced_exact_requests {
                format!("warp_original_exactk_p{}", p_tag)
            } else {
                format!("warp_original_p{}", p_tag)
            }
        };
        let reorder_method = if balanced_ps.len() == 1 {
            if args.balanced_exact_requests {
                "warp_reorder_exactk".to_string()
            } else {
                "warp_reorder".to_string()
            }
        } else {
            if args.balanced_exact_requests {
                format!("warp_reorder_exactk_p{}", p_tag)
            } else {
                format!("warp_reorder_p{}", p_tag)
            }
        };

        if args.only_method.is_none() {
            for &k in &ks {
                let per_query = run_planned_queries(
                    orig_file.clone(),
                    &original.warmup_queries,
                    &original.queries,
                    |query| {
                        plan_balanced(
                            query,
                            &latency_model,
                            k,
                            balanced_p,
                            args.balanced_exact_requests,
                            args.max_iop_size_bytes,
                        )
                    },
                    &objective_cost_model,
                    balanced_p,
                    args.query_concurrency,
                )
                .await?;
                rows.push(summarize_under_k(
                    &args.dataset,
                    "original",
                    &original_method,
                    k,
                    &per_query,
                ));
            }
        }

        if args.only_method.is_none() || args.only_method.is_some_and(OnlyMethod::runs_warp) {
            for &k in &ks {
                let per_query = run_planned_queries(
                    re_file.clone(),
                    &reordered.warmup_queries,
                    &reordered.queries,
                    |query| {
                        plan_balanced(
                            query,
                            &latency_model,
                            k,
                            balanced_p,
                            args.balanced_exact_requests,
                            args.max_iop_size_bytes,
                        )
                    },
                    &objective_cost_model,
                    balanced_p,
                    args.query_concurrency,
                )
                .await?;
                rows.push(summarize_under_k(
                    &args.dataset,
                    "reordered",
                    &reorder_method,
                    k,
                    &per_query,
                ));
                if args.dump_plan_stats_json.is_some() {
                    for (idx, query) in reordered.queries.iter().enumerate() {
                        let planned = plan_balanced(
                            query,
                            &latency_model,
                            k,
                            balanced_p,
                            args.balanced_exact_requests,
                            args.max_iop_size_bytes,
                        )?;
                        plan_stats.push(build_plan_stats_row(
                            &args.dataset,
                            "reordered",
                            &reorder_method,
                            Some(k),
                            idx,
                            &planned,
                        ));
                    }
                }
            }
        }
    }

    let mut greedy_candidates = Vec::new();
    let mut greedy_unbounded_candidates = Vec::new();
    let mut greedy_layouts = Vec::new();
    if args.only_method.is_none() {
        greedy_layouts.push((
            "original",
            orig_file.clone(),
            &original.warmup_queries,
            &original.queries,
        ));
    }
    if args.only_method.is_none() || args.only_method.is_some_and(OnlyMethod::runs_greedy) {
        greedy_layouts.push((
            "reordered",
            re_file.clone(),
            &reordered.warmup_queries,
            &reordered.queries,
        ));
    }
    for (layout_name, file, warmups, queries) in greedy_layouts {
        for &gap_kb in &gap_kb_values {
            for &span_cap_bytes in &span_caps_bytes {
                let gap_threshold_bytes = (gap_kb as u64) * 1024;
                let per_query = run_planned_queries(
                    file.clone(),
                    warmups,
                    queries,
                    |query| {
                        Ok(plan_greedy_gap_frontier(
                            query,
                            gap_threshold_bytes,
                            span_cap_bytes,
                            args.max_iop_size_bytes,
                        ))
                    },
                    &objective_cost_model,
                    greedy_objective_parallelism,
                    args.query_concurrency,
                )
                .await?;
                let summary_unbounded = summarize_unbounded(
                    &args.dataset,
                    layout_name,
                    "greedy_gap_frontier_unbounded",
                    &per_query,
                );
                greedy_unbounded_candidates.push(GreedyUnboundedCandidateRow {
                    dataset: args.dataset.clone(),
                    layout: layout_name.to_string(),
                    gap_kb: gap_kb as u64,
                    span_cap_bytes,
                    wall_ms: summary_unbounded.wall_ms,
                    qps: summary_unbounded.qps,
                    avg_latency_ms: summary_unbounded.avg_latency_ms,
                    p95_latency_ms: summary_unbounded.p95_latency_ms,
                    avg_num_requests: summary_unbounded.avg_num_requests,
                    avg_planned_bytes: summary_unbounded.avg_planned_bytes,
                    avg_model_work_ms: summary_unbounded.avg_model_work_ms,
                    avg_model_bottleneck_ms: summary_unbounded.avg_model_bottleneck_ms,
                    avg_model_objective_ms: summary_unbounded.avg_model_objective_ms,
                });
                for &k in &ks {
                    let summary = summarize_under_k(
                        &args.dataset,
                        layout_name,
                        "greedy_gap_frontier",
                        k,
                        &per_query,
                    );
                    greedy_candidates.push(GreedyCandidateRow {
                        dataset: args.dataset.clone(),
                        layout: layout_name.to_string(),
                        budget_k: k,
                        gap_kb: gap_kb as u64,
                        span_cap_bytes,
                        coverage: summary.coverage,
                        wall_ms: summary.wall_ms,
                        qps: summary.qps,
                        avg_latency_ms: summary.avg_latency_ms,
                        p95_latency_ms: summary.p95_latency_ms,
                        avg_num_requests: summary.avg_num_requests,
                        avg_planned_bytes: summary.avg_planned_bytes,
                        avg_model_work_ms: summary.avg_model_work_ms,
                        avg_model_bottleneck_ms: summary.avg_model_bottleneck_ms,
                        avg_model_objective_ms: summary.avg_model_objective_ms,
                        selected_for_curve: false,
                    });
                }
            }
        }
    }

    let greedy_output_layouts: &[&str] = match args.only_method {
        None => &["original", "reordered"],
        Some(OnlyMethod::GreedyReorder) => &["reordered"],
        Some(OnlyMethod::WarpReorder) => &[],
    };
    for &layout_name in greedy_output_layouts {
        for &k in &ks {
            let mut candidate_rows = Vec::new();
            for cand in greedy_candidates
                .iter()
                .filter(|r| r.layout == layout_name && r.budget_k == k)
            {
                candidate_rows.push(BenchmarkRow {
                    dataset: cand.dataset.clone(),
                    layout: cand.layout.clone(),
                    method: "greedy_gap_frontier".to_string(),
                    budget_k: Some(k),
                    status: if cand.coverage >= 1.0 - 1e-12 {
                        "ok".to_string()
                    } else {
                        "partial".to_string()
                    },
                    coverage: cand.coverage,
                    wall_ms: cand.wall_ms,
                    qps: cand.qps,
                    avg_latency_ms: cand.avg_latency_ms,
                    p95_latency_ms: cand.p95_latency_ms,
                    avg_num_requests: cand.avg_num_requests,
                    avg_planned_bytes: cand.avg_planned_bytes,
                    avg_model_work_ms: cand.avg_model_work_ms,
                    avg_model_bottleneck_ms: cand.avg_model_bottleneck_ms,
                    avg_model_objective_ms: cand.avg_model_objective_ms,
                    best_gap_kb: Some(cand.gap_kb),
                    best_span_cap_bytes: Some(cand.span_cap_bytes),
                });
            }
            let selected = choose_best_greedy_row(&candidate_rows).unwrap_or(BenchmarkRow {
                dataset: args.dataset.clone(),
                layout: layout_name.to_string(),
                method: "greedy_gap_frontier".to_string(),
                budget_k: Some(k),
                status: "no_data".to_string(),
                coverage: 0.0,
                wall_ms: None,
                qps: None,
                avg_latency_ms: None,
                p95_latency_ms: None,
                avg_num_requests: None,
                avg_planned_bytes: None,
                avg_model_work_ms: None,
                avg_model_bottleneck_ms: None,
                avg_model_objective_ms: None,
                best_gap_kb: None,
                best_span_cap_bytes: None,
            });
            for cand in greedy_candidates
                .iter_mut()
                .filter(|r| r.layout == layout_name && r.budget_k == k)
            {
                if selected.best_gap_kb == Some(cand.gap_kb)
                    && selected.best_span_cap_bytes == Some(cand.span_cap_bytes)
                {
                    cand.selected_for_curve = true;
                }
            }
            if args.dump_plan_stats_json.is_some() {
                if let (Some(gap_kb), Some(span_cap_bytes)) =
                    (selected.best_gap_kb, selected.best_span_cap_bytes)
                {
                    let queries = if layout_name == "original" {
                        &original.queries
                    } else {
                        &reordered.queries
                    };
                    for (idx, query) in queries.iter().enumerate() {
                        let planned = plan_greedy_gap_frontier(
                            query,
                            gap_kb.saturating_mul(1024),
                            span_cap_bytes,
                            args.max_iop_size_bytes,
                        );
                        plan_stats.push(build_plan_stats_row(
                            &args.dataset,
                            layout_name,
                            "greedy_gap_frontier",
                            Some(k),
                            idx,
                            &planned,
                        ));
                    }
                }
            }
            rows.push(selected);
        }
    }

    let doc = OutputDoc {
        dataset: args.dataset.clone(),
        nlist: original.nlist,
        nprobe: original.nprobe,
        num_queries: original.queries.len(),
        warmup_queries: original.warmup_queries.len(),
        query_concurrency: args.query_concurrency,
        balanced_ps,
        process_io_threads_limit: args.process_io_threads_limit,
        cloud_io_parallelism: args.cloud_io_parallelism,
        max_iop_size_bytes: args.max_iop_size_bytes,
        only_method: args.only_method.map(|value| value.as_str().to_string()),
        rows,
        greedy_candidates,
        greedy_unbounded_candidates,
        plan_stats,
    };
    let json_text = serde_json::to_string_pretty(&doc).map_err(|e| {
        lance_core::Error::io(
            format!("failed to serialize output json: {}", e),
            location!(),
        )
    })?;
    fs::write(&args.out_json, json_text).map_err(|e| {
        lance_core::Error::io(
            format!("failed to write json {}: {}", args.out_json.display(), e),
            location!(),
        )
    })?;
    if let Some(out_csv) = &args.out_csv {
        write_csv(
            out_csv,
            &doc.rows,
            &doc.greedy_candidates,
            &doc.greedy_unbounded_candidates,
        )?;
    }
    if let Some(path) = &args.dump_plan_stats_json {
        let json_text = serde_json::to_string_pretty(&doc.plan_stats).map_err(|e| {
            lance_core::Error::io(
                format!("failed to serialize plan stats json: {}", e),
                location!(),
            )
        })?;
        fs::write(path, json_text).map_err(|e| {
            lance_core::Error::io(
                format!("failed to write plan stats json {}: {}", path.display(), e),
                location!(),
            )
        })?;
    }
    println!("[ok] json={}", args.out_json.display());
    if let Some(out_csv) = &args.out_csv {
        println!("[ok] csv={}", out_csv.display());
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_only_method(value: &str) -> OnlyMethod {
        Args::try_parse_from([
            "strict_budget_real_s3",
            "--dataset",
            "msong",
            "--original-query-json",
            "original.json",
            "--reordered-query-json",
            "reordered.json",
            "--latency-model-json",
            "latency.json",
            "--out-json",
            "out.json",
            "--only-method",
            value,
        ])
        .unwrap()
        .only_method
        .unwrap()
    }

    #[test]
    fn test_only_method_accepts_public_figure11_names() {
        assert_eq!(
            parse_only_method("greedy_reorder"),
            OnlyMethod::GreedyReorder
        );
        assert_eq!(parse_only_method("warp_reorder"), OnlyMethod::WarpReorder);
    }

    fn sample_rows() -> Vec<BenchmarkRow> {
        vec![
            BenchmarkRow {
                dataset: "sift1m".to_string(),
                layout: "original".to_string(),
                method: "greedy_gap_frontier".to_string(),
                budget_k: Some(16),
                status: "ok".to_string(),
                coverage: 1.0,
                wall_ms: None,
                qps: None,
                avg_latency_ms: Some(90.0),
                p95_latency_ms: Some(130.0),
                avg_num_requests: Some(10.0),
                avg_planned_bytes: None,
                avg_model_work_ms: None,
                avg_model_bottleneck_ms: None,
                avg_model_objective_ms: Some(65.0),
                best_gap_kb: Some(64),
                best_span_cap_bytes: Some(8 * 1024 * 1024),
            },
            BenchmarkRow {
                dataset: "sift1m".to_string(),
                layout: "original".to_string(),
                method: "greedy_gap_frontier".to_string(),
                budget_k: Some(16),
                status: "ok".to_string(),
                coverage: 1.0,
                wall_ms: None,
                qps: None,
                avg_latency_ms: Some(85.0),
                p95_latency_ms: Some(140.0),
                avg_num_requests: Some(11.0),
                avg_planned_bytes: None,
                avg_model_work_ms: None,
                avg_model_bottleneck_ms: None,
                avg_model_objective_ms: Some(60.0),
                best_gap_kb: Some(128),
                best_span_cap_bytes: Some(16 * 1024 * 1024),
            },
            BenchmarkRow {
                dataset: "sift1m".to_string(),
                layout: "original".to_string(),
                method: "greedy_gap_frontier".to_string(),
                budget_k: Some(16),
                status: "partial".to_string(),
                coverage: 0.8,
                wall_ms: None,
                qps: None,
                avg_latency_ms: Some(60.0),
                p95_latency_ms: Some(100.0),
                avg_num_requests: Some(9.0),
                avg_planned_bytes: None,
                avg_model_work_ms: None,
                avg_model_bottleneck_ms: None,
                avg_model_objective_ms: Some(40.0),
                best_gap_kb: Some(256),
                best_span_cap_bytes: Some(16 * 1024 * 1024),
            },
        ]
    }

    #[test]
    fn test_choose_best_greedy_row_prefers_full_coverage_then_min_latency_without_qps() {
        let best = choose_best_greedy_row(&sample_rows()).unwrap();
        assert_eq!(best.best_gap_kb, Some(128));
        assert_eq!(best.best_span_cap_bytes, Some(16 * 1024 * 1024));
        assert_eq!(best.avg_latency_ms, Some(85.0));
    }

    #[test]
    fn test_choose_best_greedy_row_breaks_latency_ties_by_p95_then_requests() {
        let rows = vec![
            BenchmarkRow {
                dataset: "glove".to_string(),
                layout: "original".to_string(),
                method: "greedy_gap_frontier".to_string(),
                budget_k: Some(32),
                status: "ok".to_string(),
                coverage: 1.0,
                wall_ms: None,
                qps: None,
                avg_latency_ms: Some(70.0),
                p95_latency_ms: Some(100.0),
                avg_num_requests: Some(20.0),
                avg_planned_bytes: None,
                avg_model_work_ms: None,
                avg_model_bottleneck_ms: None,
                avg_model_objective_ms: Some(55.0),
                best_gap_kb: Some(64),
                best_span_cap_bytes: Some(2 * 1024 * 1024),
            },
            BenchmarkRow {
                dataset: "glove".to_string(),
                layout: "original".to_string(),
                method: "greedy_gap_frontier".to_string(),
                budget_k: Some(32),
                status: "ok".to_string(),
                coverage: 1.0,
                wall_ms: None,
                qps: None,
                avg_latency_ms: Some(70.0),
                p95_latency_ms: Some(95.0),
                avg_num_requests: Some(22.0),
                avg_planned_bytes: None,
                avg_model_work_ms: None,
                avg_model_bottleneck_ms: None,
                avg_model_objective_ms: Some(55.0),
                best_gap_kb: Some(128),
                best_span_cap_bytes: Some(4 * 1024 * 1024),
            },
            BenchmarkRow {
                dataset: "glove".to_string(),
                layout: "original".to_string(),
                method: "greedy_gap_frontier".to_string(),
                budget_k: Some(32),
                status: "ok".to_string(),
                coverage: 1.0,
                wall_ms: None,
                qps: None,
                avg_latency_ms: Some(70.0),
                p95_latency_ms: Some(95.0),
                avg_num_requests: Some(24.0),
                avg_planned_bytes: None,
                avg_model_work_ms: None,
                avg_model_bottleneck_ms: None,
                avg_model_objective_ms: Some(56.0),
                best_gap_kb: Some(256),
                best_span_cap_bytes: Some(8 * 1024 * 1024),
            },
        ];
        let best = choose_best_greedy_row(&rows).unwrap();
        assert_eq!(best.best_gap_kb, Some(128));
        assert_eq!(best.best_span_cap_bytes, Some(4 * 1024 * 1024));
    }

    #[test]
    fn test_choose_best_greedy_row_prefers_max_qps_when_present() {
        let rows = vec![
            BenchmarkRow {
                dataset: "sift1m".to_string(),
                layout: "reordered".to_string(),
                method: "greedy_gap_frontier".to_string(),
                budget_k: Some(64),
                status: "ok".to_string(),
                coverage: 1.0,
                wall_ms: Some(1000.0),
                qps: Some(80.0),
                avg_latency_ms: Some(200.0),
                p95_latency_ms: Some(320.0),
                avg_num_requests: Some(40.0),
                avg_planned_bytes: None,
                avg_model_work_ms: None,
                avg_model_bottleneck_ms: None,
                avg_model_objective_ms: Some(50.0),
                best_gap_kb: Some(128),
                best_span_cap_bytes: Some(4 * 1024 * 1024),
            },
            BenchmarkRow {
                dataset: "sift1m".to_string(),
                layout: "reordered".to_string(),
                method: "greedy_gap_frontier".to_string(),
                budget_k: Some(64),
                status: "ok".to_string(),
                coverage: 1.0,
                wall_ms: Some(1000.0),
                qps: Some(82.0),
                avg_latency_ms: Some(210.0),
                p95_latency_ms: Some(340.0),
                avg_num_requests: Some(42.0),
                avg_planned_bytes: None,
                avg_model_work_ms: None,
                avg_model_bottleneck_ms: None,
                avg_model_objective_ms: Some(45.0),
                best_gap_kb: Some(256),
                best_span_cap_bytes: Some(8 * 1024 * 1024),
            },
        ];
        let best = choose_best_greedy_row(&rows).unwrap();
        assert_eq!(best.best_gap_kb, Some(256));
        assert_eq!(best.qps, Some(82.0));
    }

    #[test]
    fn test_summarize_under_k_marks_partial_when_coverage_is_incomplete() {
        let per_query = vec![
            PerQuerySample {
                latency_ms: 10.0,
                planned_ranges: 8,
                planned_bytes: 1000,
                model_work_ms: 8.0,
                model_bottleneck_ms: 3.0,
                model_objective_ms: 8.0,
            },
            PerQuerySample {
                latency_ms: 20.0,
                planned_ranges: 20,
                planned_bytes: 2000,
                model_work_ms: 18.0,
                model_bottleneck_ms: 7.0,
                model_objective_ms: 18.0,
            },
        ];
        let run = RunSummary {
            per_query,
            wall_ms: 40.0,
            qps: 50.0,
        };
        let row = summarize_under_k("sift1m", "original", "greedy_gap_frontier", 16, &run);
        assert_eq!(row.status, "partial");
        assert!((row.coverage - 0.5).abs() < 1e-12);
        assert_eq!(row.qps, Some(25.0));
        assert_eq!(row.avg_latency_ms, Some(10.0));
        assert_eq!(row.avg_model_work_ms, Some(8.0));
        assert_eq!(row.avg_model_bottleneck_ms, Some(3.0));
        assert_eq!(row.avg_model_objective_ms, Some(8.0));
    }

    #[test]
    fn test_summarize_under_k_treats_zero_request_plan_as_infeasible() {
        let per_query = vec![
            PerQuerySample {
                latency_ms: 0.01,
                planned_ranges: 0,
                planned_bytes: 0,
                model_work_ms: 0.0,
                model_bottleneck_ms: 0.0,
                model_objective_ms: 0.0,
            },
            PerQuerySample {
                latency_ms: 12.0,
                planned_ranges: 12,
                planned_bytes: 4096,
                model_work_ms: 9.0,
                model_bottleneck_ms: 4.0,
                model_objective_ms: 9.0,
            },
        ];
        let run = RunSummary {
            per_query,
            wall_ms: 40.0,
            qps: 50.0,
        };
        let row = summarize_under_k("sift10m", "original", "warp_original", 16, &run);
        assert_eq!(row.status, "partial");
        assert!((row.coverage - 0.5).abs() < 1e-12);
        assert_eq!(row.qps, Some(25.0));
        assert_eq!(row.avg_num_requests, Some(12.0));
        assert_eq!(row.avg_planned_bytes, Some(4096.0));
    }

    #[test]
    fn test_choose_best_greedy_row_keeps_reordered_layout() {
        let rows = vec![
            BenchmarkRow {
                dataset: "sift1m".to_string(),
                layout: "reordered".to_string(),
                method: "greedy_gap_frontier".to_string(),
                budget_k: Some(32),
                status: "ok".to_string(),
                coverage: 1.0,
                wall_ms: None,
                qps: None,
                avg_latency_ms: Some(88.0),
                p95_latency_ms: Some(120.0),
                avg_num_requests: Some(28.0),
                avg_planned_bytes: None,
                avg_model_work_ms: None,
                avg_model_bottleneck_ms: None,
                avg_model_objective_ms: Some(58.0),
                best_gap_kb: Some(128),
                best_span_cap_bytes: Some(8 * 1024 * 1024),
            },
            BenchmarkRow {
                dataset: "sift1m".to_string(),
                layout: "reordered".to_string(),
                method: "greedy_gap_frontier".to_string(),
                budget_k: Some(32),
                status: "ok".to_string(),
                coverage: 1.0,
                wall_ms: None,
                qps: None,
                avg_latency_ms: Some(90.0),
                p95_latency_ms: Some(118.0),
                avg_num_requests: Some(25.0),
                avg_planned_bytes: None,
                avg_model_work_ms: None,
                avg_model_bottleneck_ms: None,
                avg_model_objective_ms: Some(62.0),
                best_gap_kb: Some(64),
                best_span_cap_bytes: Some(4 * 1024 * 1024),
            },
        ];
        let best = choose_best_greedy_row(&rows).unwrap();
        assert_eq!(best.layout, "reordered");
        assert_eq!(best.best_gap_kb, Some(128));
    }

    #[test]
    fn test_plan_greedy_gap_frontier_merges_when_gap_and_span_allow() {
        let query = vec![0_u64..1024_u64, 1536_u64..2048_u64];
        let planned = plan_greedy_gap_frontier(&query, 1024, 4096, 16 * 1024 * 1024);
        assert_eq!(planned, vec![0_u64..2048_u64]);
    }

    #[test]
    fn test_plan_greedy_gap_frontier_respects_gap_threshold() {
        let query = vec![0_u64..1024_u64, 3072_u64..4096_u64];
        let planned = plan_greedy_gap_frontier(&query, 1024, 8192, 16 * 1024 * 1024);
        assert_eq!(planned, vec![0_u64..1024_u64, 3072_u64..4096_u64]);
    }

    #[test]
    fn test_plan_greedy_gap_frontier_respects_span_cap() {
        let query = vec![0_u64..4096_u64, 4608_u64..12288_u64];
        let planned = plan_greedy_gap_frontier(&query, 4096, 8192, 16 * 1024 * 1024);
        assert_eq!(planned, vec![0_u64..4096_u64, 4608_u64..12288_u64]);
    }

    #[test]
    fn test_plan_lance_native_coalesces_within_64kb_gap() {
        let query = vec![0_u64..1000_u64, 2000_u64..3000_u64];
        let planned = plan_lance_native(&query, 16 * 1024 * 1024);
        assert_eq!(planned.len(), 1);
        assert_eq!(planned[0], 0_u64..3000_u64);
    }

    #[test]
    fn test_plan_balanced_respects_execution_max_iop_during_planning() {
        let query = vec![
            0_u64..(10 * 1024 * 1024_u64),
            (10 * 1024 * 1024_u64)..(20 * 1024 * 1024_u64),
            (20 * 1024 * 1024_u64)..(30 * 1024 * 1024_u64),
        ];
        let latency_model = LatencyModelSpec {
            latency_model: "piecewise_linear".to_string(),
            latency_pw_breaks_bytes: Some(vec![4 * 1024 * 1024]),
            latency_pw_overheads_ms: Some(vec![25.0, 5.0]),
            latency_pw_rates_ms_per_byte: Some(vec![4e-6, 1e-5]),
            lat_req_ms: Some(25.0),
            lat_byte_ms: Some(4e-6),
        };

        let planned =
            plan_balanced(&query, &latency_model, 2, 60.0, false, 16 * 1024 * 1024).unwrap();
        assert_eq!(planned.len(), 3);
        assert!(planned
            .iter()
            .all(|r| r.end - r.start <= 16 * 1024 * 1024_u64));
    }

    #[test]
    fn test_plan_balanced_feasible_case_respects_budget_and_max_iop() {
        let query = vec![
            0_u64..(4 * 1024 * 1024_u64),
            (4 * 1024 * 1024_u64)..(8 * 1024 * 1024_u64),
            (8 * 1024 * 1024_u64)..(12 * 1024 * 1024_u64),
            (12 * 1024 * 1024_u64)..(16 * 1024 * 1024_u64),
            (16 * 1024 * 1024_u64)..(20 * 1024 * 1024_u64),
            (20 * 1024 * 1024_u64)..(24 * 1024 * 1024_u64),
        ];
        let latency_model = LatencyModelSpec {
            latency_model: "piecewise_linear".to_string(),
            latency_pw_breaks_bytes: Some(vec![4 * 1024 * 1024]),
            latency_pw_overheads_ms: Some(vec![25.0, 5.0]),
            latency_pw_rates_ms_per_byte: Some(vec![4e-6, 1e-5]),
            lat_req_ms: Some(25.0),
            lat_byte_ms: Some(4e-6),
        };

        let planned =
            plan_balanced(&query, &latency_model, 3, 60.0, false, 16 * 1024 * 1024).unwrap();
        assert!(planned.len() <= 3);
        assert!(planned
            .iter()
            .all(|r| r.end - r.start <= 16 * 1024 * 1024_u64));
    }

    fn brute_force_best_objective(
        query: &[Range<u64>],
        latency_model: &LatencyModelSpec,
        budget_k: usize,
        parallelism: f64,
        exact_requests: bool,
        max_iop_size: u64,
    ) -> f64 {
        let mut cost_model = CostModel::default();
        apply_latency_model(&mut cost_model, latency_model).unwrap();
        let n = query.len();
        let mut best = f64::INFINITY;
        for mask in 0_u64..(1_u64 << (n - 1)) {
            let groups = mask.count_ones() as usize + 1;
            if exact_requests {
                if groups != budget_k {
                    continue;
                }
            } else if groups > budget_k {
                continue;
            }
            let mut start = 0usize;
            let mut planned = Vec::new();
            let mut feasible = true;
            for i in 0..(n - 1) {
                if ((mask >> i) & 1) == 1 {
                    let seg = query[start].start..query[i].end;
                    if seg.end - seg.start > max_iop_size {
                        feasible = false;
                        break;
                    }
                    planned.push(seg);
                    start = i + 1;
                }
            }
            if !feasible {
                continue;
            }
            let last = query[start].start..query[n - 1].end;
            if last.end - last.start > max_iop_size {
                continue;
            }
            planned.push(last);
            best = best.min(objective_for_ranges_ms(&planned, &cost_model, parallelism));
        }
        best
    }

    #[test]
    fn test_plan_balanced_final_plan_matches_bruteforce_when_feasible() {
        let query = vec![
            0_u64..(4 * 1024 * 1024_u64),
            (4 * 1024 * 1024_u64)..(8 * 1024 * 1024_u64),
            (8 * 1024 * 1024_u64)..(12 * 1024 * 1024_u64),
            (12 * 1024 * 1024_u64)..(16 * 1024 * 1024_u64),
        ];
        let latency_model = LatencyModelSpec {
            latency_model: "piecewise_linear".to_string(),
            latency_pw_breaks_bytes: Some(vec![4 * 1024 * 1024]),
            latency_pw_overheads_ms: Some(vec![25.0, 5.0]),
            latency_pw_rates_ms_per_byte: Some(vec![4e-6, 1e-5]),
            lat_req_ms: Some(25.0),
            lat_byte_ms: Some(4e-6),
        };
        let budget_k = 2usize;
        let parallelism = 60.0;
        let max_iop = 16 * 1024 * 1024_u64;

        let planned = plan_balanced(
            &query,
            &latency_model,
            budget_k,
            parallelism,
            false,
            max_iop,
        )
        .unwrap();
        assert!(planned.len() <= budget_k);
        assert!(planned.iter().all(|r| r.end - r.start <= max_iop));

        let mut cost_model = CostModel::default();
        apply_latency_model(&mut cost_model, &latency_model).unwrap();
        let got = objective_for_ranges_ms(&planned, &cost_model, parallelism);
        let best = brute_force_best_objective(
            &query,
            &latency_model,
            budget_k,
            parallelism,
            false,
            max_iop,
        );
        assert!((got - best).abs() <= 1e-6, "got={got} best={best}");
    }

    #[test]
    fn test_plan_balanced_exactk_final_plan_matches_bruteforce_when_feasible() {
        let query = vec![
            0_u64..(4 * 1024 * 1024_u64),
            (4 * 1024 * 1024_u64)..(8 * 1024 * 1024_u64),
            (8 * 1024 * 1024_u64)..(12 * 1024 * 1024_u64),
            (12 * 1024 * 1024_u64)..(16 * 1024 * 1024_u64),
        ];
        let latency_model = LatencyModelSpec {
            latency_model: "piecewise_linear".to_string(),
            latency_pw_breaks_bytes: Some(vec![4 * 1024 * 1024]),
            latency_pw_overheads_ms: Some(vec![25.0, 5.0]),
            latency_pw_rates_ms_per_byte: Some(vec![4e-6, 1e-5]),
            lat_req_ms: Some(25.0),
            lat_byte_ms: Some(4e-6),
        };
        let budget_k = 3usize;
        let parallelism = 60.0;
        let max_iop = 16 * 1024 * 1024_u64;

        let planned =
            plan_balanced(&query, &latency_model, budget_k, parallelism, true, max_iop).unwrap();
        assert_eq!(planned.len(), budget_k);
        assert!(planned.iter().all(|r| r.end - r.start <= max_iop));

        let mut cost_model = CostModel::default();
        apply_latency_model(&mut cost_model, &latency_model).unwrap();
        let got = objective_for_ranges_ms(&planned, &cost_model, parallelism);
        let best = brute_force_best_objective(
            &query,
            &latency_model,
            budget_k,
            parallelism,
            true,
            max_iop,
        );
        assert!((got - best).abs() <= 1e-6, "got={got} best={best}");
    }
}
