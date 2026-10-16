//! Small self-contained savings demo.
//!
//! Builds a realistic IVF-style query (128 clusters scattered across a
//! 1 GB logical object), plans it with every strategy, and prints a
//! side-by-side cost comparison — latency, request count, waste, and
//! $ per million queries — under the paper's calibrated S3 cost model.
//!
//! No external data, no network, runs in well under a second:
//!
//! ```bash
//! cargo run --release --example savings_demo
//! ```

use warp_planner::{
    evaluate, plan,
    strategy::{Bottleneck, Greedy, MainSolution, MergeStrategy, Separate, Serial},
    CostModel, IoRequest, PlanConstraints, PlanMetrics,
};

/// Paper's piecewise-linear S3 fit + realistic dollar pricing.
///
/// `cost_per_request = $4e-7` matches AWS S3 `GetObject` (us-east-1,
/// standard tier, Aug 2025): $0.0004 per 1 000 GETs.
fn paper_cost(parallelism: f64) -> CostModel {
    CostModel {
        latency_model: "piecewise_linear".to_string(),
        latency_pw_breaks_bytes: vec![3_145_728],
        latency_pw_overheads_ms: vec![28.849557293569053, 6.266035367002434],
        latency_pw_rates_ms_per_byte: vec![2.824569398517584e-06, 1.0003677676972243e-05],
        latency_per_request: 28.849557293569053,
        latency_per_byte: 2.824569398517584e-06,
        concurrent_parallelism: parallelism,
        cost_per_request: 0.0000004, // $0.0004 per 1k GETs
        cost_per_byte: 0.0,          // same-region → ignore
        cost_per_wasted_byte: 0.0,
        ..CostModel::default()
    }
}

/// Build a deterministic, IVF-reorder-friendly query: 128 clusters of
/// ~8 KiB each, scattered across a 1 GiB object with the kind of gap
/// distribution a PCA-reordered IVF index produces (mostly tight, a
/// few bigger jumps).
fn ivf_query() -> Vec<IoRequest> {
    let mut rs = Vec::with_capacity(128);
    let mut offset = 0i64;
    // Simple reproducible LCG so this demo doesn't pull in `rand`.
    let mut state: u64 = 0x1234_5678_abcd_ef01;
    let mut next_u32 = || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (state >> 33) as u32
    };
    for i in 0i32..128 {
        let length = 6_000 + (next_u32() % 5_000) as i64;
        rs.push(IoRequest {
            cluster_id: i,
            offset,
            length,
        });
        let big_jump = (next_u32() % 10) == 0;
        let gap = if big_jump {
            400_000 + (next_u32() % 1_600_000) as i64
        } else {
            (next_u32() % 8_000) as i64
        };
        offset += length + gap;
    }
    rs
}

fn cost_usd(m: &PlanMetrics, c: &CostModel) -> f64 {
    c.cost_per_request * m.request_count as f64
        + c.cost_per_byte * m.total_bytes as f64
        + c.cost_per_wasted_byte * m.wasted_bytes as f64
}

fn main() {
    let requests = ivf_query();
    let cost = paper_cost(16.0);
    let constraints = PlanConstraints::default();
    let p = cost.concurrent_parallelism;

    println!(
        "Workload: {} clusters across {} MiB of object, P = {}",
        requests.len(),
        (requests.last().unwrap().end()) / (1024 * 1024),
        p,
    );
    println!(
        "Cost model: {} overhead / ${:.2e} per GET (same-region S3)\n",
        cost.latency_model, cost.cost_per_request,
    );

    let strategies: [(&str, Box<dyn MergeStrategy>); 5] = [
        ("Separate (baseline)", Box::new(Separate)),
        ("Greedy", Box::new(Greedy::default())),
        ("Serial DP", Box::new(Serial)),
        ("Bottleneck", Box::new(Bottleneck)),
        ("MainSolution", Box::new(MainSolution)),
    ];

    println!(
        "  {:<22} {:>8} {:>10} {:>10} {:>12} {:>14}",
        "strategy", "reqs", "waste %", "latency ms", "$/query", "$/M queries",
    );
    println!("  {}", "─".repeat(22 + 8 + 10 + 10 + 12 + 14 + 5));

    let mut baseline_t_ub = 0.0f64;
    for (i, (name, strat)) in strategies.iter().enumerate() {
        let output = plan(strat.as_ref(), &requests, &cost, &constraints)
            .expect("the built-in strategies must produce a valid plan");
        let m = evaluate(&requests, &output, &cost);
        let usd = cost_usd(&m, &cost);
        let lat = m.t_ub_ms(p);
        let waste_pct = if m.total_bytes > 0 {
            100.0 * m.wasted_bytes as f64 / m.total_bytes as f64
        } else {
            0.0
        };
        if i == 0 {
            baseline_t_ub = lat;
        }
        println!(
            "  {:<22} {:>8} {:>9.1}% {:>10.2} {:>12.3e} {:>14.2}",
            name,
            m.request_count,
            waste_pct,
            lat,
            usd,
            usd * 1_000_000.0,
        );
    }

    println!();
    // Headline: savings of MainSolution vs Separate and vs Greedy.
    let baseline_plan = plan(&Separate, &requests, &cost, &constraints).unwrap();
    let baseline_m = evaluate(&requests, &baseline_plan, &cost);
    let greedy_plan = plan(&Greedy::default(), &requests, &cost, &constraints).unwrap();
    let greedy_m = evaluate(&requests, &greedy_plan, &cost);
    let main_plan = plan(&MainSolution, &requests, &cost, &constraints).unwrap();
    let main_m = evaluate(&requests, &main_plan, &cost);

    let save_vs_sep = (cost_usd(&baseline_m, &cost) - cost_usd(&main_m, &cost)) * 1_000_000.0;
    let cost_delta_vs_greedy =
        (cost_usd(&main_m, &cost) - cost_usd(&greedy_m, &cost)) * 1_000_000.0;
    let main_latency = main_m.t_ub_ms(p);
    let latency_saved_vs_sep = 100.0 * (1.0 - main_latency / baseline_t_ub.max(1e-12));

    println!("MainSolution vs baseline (Separate):");
    println!(
        "  requests:  {:>4} → {:<4}   ({:.1}% fewer)",
        baseline_m.request_count,
        main_m.request_count,
        100.0 * (1.0 - main_m.request_count as f64 / baseline_m.request_count as f64),
    );
    println!(
        "  latency:   {:.2} → {:.2} ms ({:.1}% lower)",
        baseline_t_ub, main_latency, latency_saved_vs_sep,
    );
    println!("  S3 cost:   ${:.2} saved per million queries", save_vs_sep,);
    println!();
    println!("MainSolution vs Greedy heuristic:");
    println!(
        "  requests:  {:>4} → {:<4}   ({:+.1}% requests)",
        greedy_m.request_count,
        main_m.request_count,
        100.0 * (main_m.request_count as f64 / greedy_m.request_count.max(1) as f64 - 1.0),
    );
    println!(
        "  latency:   {:.2} → {:.2} ms ({:.1}% lower)",
        greedy_m.t_ub_ms(p),
        main_latency,
        100.0 * (1.0 - main_latency / greedy_m.t_ub_ms(p).max(1e-12)),
    );
    println!(
        "  S3 cost:   ${:+.2} per million queries (latency/cost tradeoff)",
        cost_delta_vs_greedy,
    );
}
