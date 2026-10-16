//! Detailed IVF-derived benchmark + quality/time report.
//!
//! Loads fixture files produced by `tests/data/generate_fixtures.py`
//! (per-query `IoRequest` sequences from a FAISS IVF coarse-quantizer
//! search) and prints, for every `(distribution × strategy × P)` cell:
//!
//! - **plan time**   — median and p99 microseconds spent in `generate_plan`
//! - **predicted latency** — mean latency at the configured concurrency
//! - **latency saved** — percentage reduction against `Separate`
//! - **waste ratio** — `(total_bytes − useful_bytes) / total_bytes`
//! - **requests**    — average merged request count per query
//!
//! All rows are also written to the workspace's
//! `target/bench-artifacts/ivf_workload.csv` for downstream plotting.
//!
//! The correctness gate asserts per fixture:
//! - Serial ≤ Greedy on W (Serial is exact for the W objective)
//! - Bottleneck output is finite and covers every request
//! - MainSolution at P=1 ≤ Serial on W

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::Instant;
use warp_planner::{
    evaluate,
    strategy::{Bottleneck, Greedy, MainSolution, MergeStrategy, Separate, Serial},
    CostModel, IoRequest, PlanConstraints, PlanMetrics,
};

// ---------------------------------------------------------------------
// Fixture loading.
// ---------------------------------------------------------------------

fn all_fixture_dirs() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    vec![
        // FAISS cluster_id order (= no reorder; paper baseline).
        root.join("tests/data/sift_paper_4096x128"),
        // After PC1 centroid reorder (paper's Hilbert-reorder analogue).
        root.join("tests/data/sift_paper_pca_4096x128"),
        // Adversarial heavy-tailed gap distribution.
        root.join("tests/data/heavy_tail_4096x128"),
    ]
}

fn load_fixtures(dir: &Path) -> Vec<(String, Vec<IoRequest>)> {
    let mut out = Vec::new();
    if !dir.exists() {
        return out;
    }
    let mut entries: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| {
            let name = e.file_name().into_string().unwrap_or_default();
            name.starts_with("query_") && name.ends_with(".json")
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());
    for entry in entries {
        let name = entry.file_name().into_string().unwrap();
        let body = fs::read_to_string(entry.path()).unwrap();
        let reqs: Vec<IoRequest> = serde_json::from_str(&body).unwrap();
        out.push((name, reqs));
    }
    out
}

// ---------------------------------------------------------------------
// Cost model — paper's most recent S3 calibration.
// ---------------------------------------------------------------------

/// Cost model from the paper artifact's most recent S3 calibration
/// snapshot, `experiments/s3-piecewise-calibration-curves/data/paper/
/// fit_summary.json::piecewise_continuous` — the same values the
/// closed-loop runner loads on every invocation.
fn paper_cost(parallelism: f64) -> CostModel {
    CostModel {
        latency_model: "piecewise_linear".to_string(),
        latency_pw_breaks_bytes: vec![3_145_728],
        latency_pw_overheads_ms: vec![28.849557293569053, 6.266035367002434],
        latency_pw_rates_ms_per_byte: vec![2.824569398517584e-06, 1.0003677676972243e-05],
        latency_per_request: 28.849557293569053,
        latency_per_byte: 2.824569398517584e-06,
        concurrent_parallelism: parallelism,
        ..CostModel::default()
    }
}

fn predicted_latency(m: &PlanMetrics, p: f64) -> f64 {
    m.t_ub_ms(p)
}

// ---------------------------------------------------------------------
// Per-(dist, strategy, P) aggregated stats.
// ---------------------------------------------------------------------

#[derive(Clone)]
struct StrategyStats {
    dist: String,
    strategy: String,
    parallelism: f64,
    avg_w: f64,
    avg_b: f64,
    avg_predicted_latency: f64,
    latency_reduction_pct_vs_separate: f64,
    avg_waste_pct: f64,
    avg_requests: f64,
    /// Mean `$ / query` under `CostModel::cost_per_*` — same-region S3
    /// GET dominates this, so for default pricing it tracks request count.
    avg_cost_usd_per_query: f64,
    /// Percentage cost reduction against `separate` (positive is cheaper).
    cost_reduction_pct_vs_separate: f64,
    /// Per-million-query $ savings vs. `separate`. Negative = savings.
    /// Useful when comparing strategies at realistic query volumes.
    usd_per_million_queries_vs_separate: f64,
    plan_us_median: f64,
    plan_us_p99: f64,
    n_queries: usize,
}

/// Compute monetary cost per query under a CostModel's `cost_per_*` fields.
fn plan_cost_usd(plan_metrics: &PlanMetrics, cost: &CostModel) -> f64 {
    let req_cost = cost.cost_per_request * plan_metrics.request_count as f64;
    let byte_cost = cost.cost_per_byte * plan_metrics.total_bytes as f64;
    let waste_cost = cost.cost_per_wasted_byte * plan_metrics.wasted_bytes as f64;
    req_cost + byte_cost + waste_cost
}

fn percentile(sorted: &[f64], q: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = ((sorted.len() - 1) as f64 * q).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

/// Evaluate one strategy across all fixtures at a fixed P. Returns
/// aggregated stats and the per-query predicted-latency vector.
fn measure_strategy(
    strat: &dyn MergeStrategy,
    fixtures: &[(String, Vec<IoRequest>)],
    cost: &CostModel,
    constraints: &PlanConstraints,
    reps: usize,
) -> (StrategyStats, Vec<f64>) {
    let p = cost.concurrent_parallelism.max(1.0);
    let mut tot_w = 0.0f64;
    let mut tot_b = 0.0f64;
    let mut tot_predicted_latency = 0.0f64;
    let mut tot_waste = 0.0f64;
    let mut tot_cost = 0.0f64;
    let mut tot_requests = 0usize;
    let mut per_query_predicted_latency = Vec::with_capacity(fixtures.len());
    let mut plan_us_samples = Vec::with_capacity(fixtures.len() * reps);

    for (_, reqs) in fixtures {
        // Time the plan repeatedly to reduce noise.
        let mut last_plan = None;
        for _ in 0..reps {
            let start = Instant::now();
            let plan = strat.generate_plan(reqs, cost, constraints);
            let dt_us = start.elapsed().as_nanos() as f64 / 1_000.0;
            plan_us_samples.push(dt_us);
            last_plan = Some(plan);
        }
        let plan = last_plan.unwrap();
        plan.check_invariants(reqs)
            .unwrap_or_else(|e| panic!("{}: invalid plan: {e}", strat.name()));
        let m = evaluate(reqs, &plan, cost);
        tot_w += m.predicted_latency_ms_serial;
        tot_b += m.predicted_latency_ms_concurrent;
        let query_latency = predicted_latency(&m, p);
        tot_predicted_latency += query_latency;
        per_query_predicted_latency.push(query_latency);
        if m.total_bytes > 0 {
            tot_waste += m.wasted_bytes as f64 / m.total_bytes as f64;
        }
        tot_cost += plan_cost_usd(&m, cost);
        tot_requests += m.request_count;
    }

    plan_us_samples.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = fixtures.len().max(1) as f64;

    let stats = StrategyStats {
        dist: String::new(),
        strategy: strat.name().to_string(),
        parallelism: p,
        avg_w: tot_w / n,
        avg_b: tot_b / n,
        avg_predicted_latency: tot_predicted_latency / n,
        latency_reduction_pct_vs_separate: 0.0,
        avg_waste_pct: 100.0 * tot_waste / n,
        avg_requests: tot_requests as f64 / n,
        avg_cost_usd_per_query: tot_cost / n,
        cost_reduction_pct_vs_separate: 0.0,
        usd_per_million_queries_vs_separate: 0.0,
        plan_us_median: percentile(&plan_us_samples, 0.50),
        plan_us_p99: percentile(&plan_us_samples, 0.99),
        n_queries: fixtures.len(),
    };
    (stats, per_query_predicted_latency)
}

fn format_header() -> String {
    format!(
        "  {:14} {:>7} {:>12} {:>13} {:>9} {:>12} {:>12} {:>11} {:>11}",
        "strategy",
        "reads",
        "latency(ms)",
        "latency saved",
        "requests",
        "$/M queries",
        "$/M saved",
        "plan p50 us",
        "plan p99 us",
    )
}

fn format_row(s: &StrategyStats) -> String {
    format!(
        "  {:14} {:>7.1} {:>12.2} {:>12.1}% {:>9.1} {:>12.2} {:>12.2} {:>11.2} {:>11.2}",
        s.strategy,
        s.parallelism,
        s.avg_predicted_latency,
        s.latency_reduction_pct_vs_separate,
        s.avg_requests,
        s.avg_cost_usd_per_query * 1_000_000.0,
        // Positive = strategy saves vs. separate; negative = costs more.
        -s.usd_per_million_queries_vs_separate,
        s.plan_us_median,
        s.plan_us_p99,
    )
}

// ---------------------------------------------------------------------
// Correctness gate (runs once per distribution at P=16).
// ---------------------------------------------------------------------

fn correctness_gate(fixtures: &[(String, Vec<IoRequest>)], cost: &CostModel) {
    let constraints = PlanConstraints::default();
    for (tag, reqs) in fixtures {
        let greedy = evaluate(
            reqs,
            &Greedy::default().generate_plan(reqs, cost, &constraints),
            cost,
        );
        let serial = evaluate(reqs, &Serial.generate_plan(reqs, cost, &constraints), cost);
        assert!(
            serial.predicted_latency_ms_serial <= greedy.predicted_latency_ms_serial + 1e-6,
            "{tag}: Serial W {} > Greedy W {} (Serial must be W-optimal)",
            serial.predicted_latency_ms_serial,
            greedy.predicted_latency_ms_serial
        );

        let cap_constraints = PlanConstraints {
            max_requests: (reqs.len() as i32 / 4).max(1),
            exact_requests: false,
        };
        let bottleneck_m = evaluate(
            reqs,
            &Bottleneck.generate_plan(reqs, cost, &cap_constraints),
            cost,
        );
        assert!(bottleneck_m.predicted_latency_ms_concurrent.is_finite());

        let mut p1 = cost.clone();
        p1.concurrent_parallelism = 1.0;
        let main_m = evaluate(
            reqs,
            &MainSolution.generate_plan(reqs, &p1, &constraints),
            &p1,
        );
        let serial_m = evaluate(reqs, &Serial.generate_plan(reqs, &p1, &constraints), &p1);
        assert!(
            main_m.predicted_latency_ms_serial <= serial_m.predicted_latency_ms_serial + 1e-6,
            "{tag}: MainSolution at P=1 W {} > Serial W {}",
            main_m.predicted_latency_ms_serial,
            serial_m.predicted_latency_ms_serial
        );
    }
    eprintln!(
        "[correctness] all invariants hold on {} fixtures",
        fixtures.len()
    );
}

// ---------------------------------------------------------------------
// CSV artifact.
// ---------------------------------------------------------------------

fn write_csv(rows: &[StrategyStats]) -> std::io::Result<PathBuf> {
    let out_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("the crate must live under the workspace root")
        .join("target")
        .join("bench-artifacts");
    fs::create_dir_all(&out_dir)?;
    let path = out_dir.join("ivf_workload.csv");
    let mut f = fs::File::create(&path)?;
    writeln!(
        f,
        "dist,strategy,P,n_queries,avg_serial_work_ms,avg_largest_request_ms,avg_predicted_latency_ms,latency_reduction_pct_vs_separate,avg_waste_pct,avg_requests,avg_cost_usd_per_query,cost_reduction_pct_vs_separate,usd_per_million_queries_saved_vs_separate,plan_us_p50,plan_us_p99"
    )?;
    for s in rows {
        writeln!(
            f,
            "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.3},{:.3},{:.9},{:.6},{:.6},{:.3},{:.3}",
            s.dist,
            s.strategy,
            s.parallelism,
            s.n_queries,
            s.avg_w,
            s.avg_b,
            s.avg_predicted_latency,
            s.latency_reduction_pct_vs_separate,
            s.avg_waste_pct,
            s.avg_requests,
            s.avg_cost_usd_per_query,
            s.cost_reduction_pct_vs_separate,
            -s.usd_per_million_queries_vs_separate,
            s.plan_us_median,
            s.plan_us_p99,
        )?;
    }
    Ok(path)
}

// ---------------------------------------------------------------------
// Main bench driver.
// ---------------------------------------------------------------------

fn bench_ivf(c: &mut Criterion) {
    // Parallelism sweep — the paper's headline dimension.
    let parallelism_sweep = [1.0f64, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0];
    // Strategies in reporting order. Separate first so we can compute savings.
    let strategies: Vec<Box<dyn MergeStrategy>> = vec![
        Box::new(Separate),
        Box::new(Greedy::default()),
        Box::new(Serial),
        Box::new(Bottleneck),
        Box::new(MainSolution),
    ];
    let reps = 20usize;
    let mut all_rows: Vec<StrategyStats> = Vec::new();

    for dir in all_fixture_dirs() {
        let fixtures = load_fixtures(&dir);
        if fixtures.is_empty() {
            eprintln!("[skip] no fixtures in {dir:?}");
            continue;
        }
        let tag = dir.file_name().unwrap().to_string_lossy().to_string();
        eprintln!("\n=== {} === ({} queries)", tag, fixtures.len());

        // One correctness pass per distribution at P=16.
        correctness_gate(&fixtures, &paper_cost(16.0));

        for &p in &parallelism_sweep {
            let cost = paper_cost(p);
            let no_constraints = PlanConstraints::default();
            eprintln!("\n  -- P = {:.1} --", p);
            eprintln!("{}", format_header());

            let (baseline_stats, _) =
                measure_strategy(&*strategies[0], &fixtures, &cost, &no_constraints, reps);
            let baseline_latency = baseline_stats.avg_predicted_latency.max(1e-12);
            let baseline_cost = baseline_stats.avg_cost_usd_per_query.max(1e-18);

            for strat in &strategies {
                let constraints = if strat.name() == "bottleneck" {
                    PlanConstraints {
                        max_requests: 16,
                        exact_requests: false,
                    }
                } else {
                    no_constraints
                };
                let (mut stats, _) =
                    measure_strategy(&**strat, &fixtures, &cost, &constraints, reps);
                stats.dist = tag.clone();
                stats.latency_reduction_pct_vs_separate =
                    100.0 * (1.0 - stats.avg_predicted_latency / baseline_latency);
                stats.cost_reduction_pct_vs_separate =
                    100.0 * (1.0 - stats.avg_cost_usd_per_query / baseline_cost);
                stats.usd_per_million_queries_vs_separate = (stats.avg_cost_usd_per_query
                    - baseline_stats.avg_cost_usd_per_query)
                    * 1_000_000.0;
                eprintln!("{}", format_row(&stats));
                all_rows.push(stats);
            }
        }
    }

    match write_csv(&all_rows) {
        Ok(p) => eprintln!(
            "\n[artifact] wrote {} rows to {}",
            all_rows.len(),
            p.display()
        ),
        Err(e) => eprintln!("\n[artifact] failed to write CSV: {e}"),
    }

    // Criterion throughput timing on the primary (paper) fixture so this
    // file still produces Criterion-style output for `cargo bench`.
    let primary = all_fixture_dirs().into_iter().next().unwrap();
    let fixtures = load_fixtures(&primary);
    if fixtures.is_empty() {
        return;
    }
    let inputs = &fixtures[0].1;
    let mut group = c.benchmark_group("ivf_workload/primary_fixture");
    let cost = paper_cost(16.0);
    let constraints = PlanConstraints::default();
    let named: [(&str, Box<dyn MergeStrategy>); 5] = [
        ("separate", Box::new(Separate)),
        ("greedy", Box::new(Greedy::default())),
        ("serial", Box::new(Serial)),
        ("bottleneck", Box::new(Bottleneck)),
        ("main_solution", Box::new(MainSolution)),
    ];
    for (name, strat) in named {
        group.bench_with_input(BenchmarkId::from_parameter(name), inputs, |b, inputs| {
            b.iter(|| {
                strat.generate_plan(black_box(inputs), black_box(&cost), black_box(&constraints))
            })
        });
    }
    group.finish();
}

criterion_group!(benches, bench_ivf);
criterion_main!(benches);
