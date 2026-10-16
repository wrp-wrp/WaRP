//! Golden regression gate.
//!
//! For every committed fixture under `tests/data/<dist>/query_*.json`,
//! load the matching frozen plan from `expected_plans.json` and compare.
//!
//! - Deterministic strategies (`separate` / `greedy` / `serial` /
//!   `bottleneck`) must reproduce the frozen plan exactly.
//! - `main_solution` only freezes its `T_UB(P)`; the current run's
//!   plan is valid if (a) it passes `check_invariants`, and (b) its
//!   `T_UB(P)` is within 5 % of the frozen value. The adaptive-hull
//!   search's priority-queue tie-breaking can legitimately produce
//!   byte-different but near-optimal plans.
//!
//! Re-bless after any intentional algorithm change:
//! `cargo run --release --example bless_golden`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::Deserialize;
use warp_planner::{
    evaluate,
    strategy::{Bottleneck, Greedy, MainSolution, MergeStrategy, Separate, Serial},
    CostModel, IoPlan, IoRequest, MergedRequest, PlanConstraints, PlanMetrics,
};

// Matches `examples/bless_golden.rs::paper_cost`.
fn paper_cost() -> CostModel {
    CostModel {
        latency_model: "piecewise_linear".to_string(),
        latency_pw_breaks_bytes: vec![3_145_728],
        latency_pw_overheads_ms: vec![28.849557293569053, 6.266035367002434],
        latency_pw_rates_ms_per_byte: vec![2.824569398517584e-06, 1.0003677676972243e-05],
        latency_per_request: 28.849557293569053,
        latency_per_byte: 2.824569398517584e-06,
        concurrent_parallelism: 16.0,
        ..CostModel::default()
    }
}

const T_UB_TOLERANCE: f64 = 0.05;

#[derive(Deserialize)]
struct DeterministicGolden {
    merged_requests: Vec<MergedRequestGolden>,
    assignments: Vec<usize>,
}

#[derive(Deserialize)]
struct MergedRequestGolden {
    offset: i64,
    length: i64,
}

#[derive(Deserialize)]
struct AdaptiveGolden {
    t_ub_ms: f64,
}

#[derive(Deserialize)]
struct QueryGolden {
    separate: DeterministicGolden,
    greedy: DeterministicGolden,
    serial: DeterministicGolden,
    bottleneck: DeterministicGolden,
    main_solution: AdaptiveGolden,
}

#[derive(Deserialize)]
struct ExpectedPlans {
    #[allow(dead_code)]
    cost_model_tag: String,
    #[allow(dead_code)]
    parallelism: f64,
    queries: BTreeMap<String, QueryGolden>,
}

fn t_ub(m: &PlanMetrics, p: f64) -> f64 {
    m.t_ub_ms(p)
}

fn fixture_dirs() -> Vec<PathBuf> {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data");
    vec![
        root.join("gaussian_1024x64"),
        root.join("sift_paper_4096x128"),
        root.join("sift_paper_pca_4096x128"),
        root.join("heavy_tail_4096x128"),
    ]
}

fn load_inputs(dir: &Path) -> Vec<(String, Vec<IoRequest>)> {
    let mut entries: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .filter_map(Result::ok)
        .filter(|e| {
            let n = e.file_name().into_string().unwrap_or_default();
            n.starts_with("query_") && n.ends_with(".json")
        })
        .collect();
    entries.sort_by_key(|e| e.file_name());
    entries
        .into_iter()
        .map(|e| {
            let name = e.file_name().into_string().unwrap();
            let stem = name.strip_suffix(".json").unwrap().to_string();
            let body = fs::read_to_string(e.path()).unwrap();
            let reqs: Vec<IoRequest> = serde_json::from_str(&body).unwrap();
            (stem, reqs)
        })
        .collect()
}

fn assert_plan_matches(tag: &str, strategy: &str, plan: &IoPlan, golden: &DeterministicGolden) {
    let got: Vec<(i64, i64)> = plan
        .merged_requests
        .iter()
        .map(|m: &MergedRequest| (m.offset, m.length))
        .collect();
    let want: Vec<(i64, i64)> = golden
        .merged_requests
        .iter()
        .map(|m| (m.offset, m.length))
        .collect();
    assert_eq!(
        got, want,
        "{tag}/{strategy}: merged_requests diverged from golden",
    );
    assert_eq!(
        plan.assignments, golden.assignments,
        "{tag}/{strategy}: assignments diverged from golden",
    );
}

#[test]
fn fixtures_match_golden_plans() {
    let cost = paper_cost();
    let constraints = PlanConstraints::default();
    let p = cost.concurrent_parallelism;
    let mut checked = 0usize;
    let mut dirs_seen = 0usize;

    for dir in fixture_dirs() {
        let expected_path = dir.join("expected_plans.json");
        if !expected_path.exists() {
            panic!(
                "missing golden {}; run `cargo run --release --example bless_golden`",
                expected_path.display()
            );
        }
        dirs_seen += 1;
        let expected: ExpectedPlans =
            serde_json::from_str(&fs::read_to_string(&expected_path).unwrap()).unwrap();
        let inputs = load_inputs(&dir);
        let dir_tag = dir.file_name().unwrap().to_string_lossy().into_owned();

        for (stem, reqs) in &inputs {
            let g = expected
                .queries
                .get(stem)
                .unwrap_or_else(|| panic!("{dir_tag}/{stem}: no golden entry; re-bless?"));
            let tag = format!("{dir_tag}/{stem}");

            let sep = Separate.generate_plan(reqs, &cost, &constraints);
            sep.check_invariants(reqs).unwrap();
            assert_plan_matches(&tag, "separate", &sep, &g.separate);

            let gre = Greedy::default().generate_plan(reqs, &cost, &constraints);
            gre.check_invariants(reqs).unwrap();
            assert_plan_matches(&tag, "greedy", &gre, &g.greedy);

            let ser = Serial.generate_plan(reqs, &cost, &constraints);
            ser.check_invariants(reqs).unwrap();
            assert_plan_matches(&tag, "serial", &ser, &g.serial);

            let bot = Bottleneck.generate_plan(reqs, &cost, &constraints);
            bot.check_invariants(reqs).unwrap();
            assert_plan_matches(&tag, "bottleneck", &bot, &g.bottleneck);

            // main_solution: validate + T_UB tolerance check.
            let main = MainSolution.generate_plan(reqs, &cost, &constraints);
            main.check_invariants(reqs).unwrap();
            let m = evaluate(reqs, &main, &cost);
            let got_t_ub = t_ub(&m, p);
            let want_t_ub = g.main_solution.t_ub_ms;
            let denom = got_t_ub.abs().max(want_t_ub.abs()).max(1e-9);
            let rel = (got_t_ub - want_t_ub).abs() / denom;
            assert!(
                rel <= T_UB_TOLERANCE,
                "{tag}/main_solution: T_UB diverged — got {got_t_ub:.3} want {want_t_ub:.3} (rel {rel:.3})",
            );

            checked += 1;
        }
    }

    assert!(dirs_seen > 0, "no fixture directories found");
    assert!(checked > 0, "no fixtures checked");
    eprintln!("[golden] checked {checked} fixtures across {dirs_seen} directories");
}
