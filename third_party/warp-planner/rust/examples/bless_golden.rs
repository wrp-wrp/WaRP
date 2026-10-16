//! Re-generate frozen golden plan fixtures.
//!
//! For every `tests/data/<dist>/query_*.json` file, runs all five
//! strategies under the paper's calibrated cost model and writes
//! `tests/data/<dist>/expected_plans.json`.
//!
//! The contract matches what the cross-language parity harness used to
//! check:
//!
//! - `separate` / `greedy` / `serial` / `bottleneck` — full plan
//!   frozen; future runs must match bit-for-bit.
//! - `main_solution` — only `T_UB(P)` frozen; future runs must stay
//!   within 5 % relative error (the adaptive-hull search's priority
//!   queue can legitimately break ties differently across builds).
//!
//! Run with `cargo run --release --example bless_golden` after any
//! intentional algorithm change, review the diff, commit.
//!
//! The companion test lives in `tests/golden.rs`.
use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use warp_planner::{
    evaluate,
    strategy::{Bottleneck, Greedy, MainSolution, MergeStrategy, Separate, Serial},
    CostModel, IoPlan, IoRequest, PlanConstraints, PlanMetrics,
};

// Paper's most recent S3 calibration (same as the ivf_workload bench).
// Kept in-sync in `tests/golden.rs::paper_cost`.
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

#[derive(Serialize, Deserialize)]
struct DeterministicGolden {
    merged_requests: Vec<MergedRequestGolden>,
    assignments: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
struct MergedRequestGolden {
    offset: i64,
    length: i64,
}

impl From<&IoPlan> for DeterministicGolden {
    fn from(plan: &IoPlan) -> Self {
        Self {
            merged_requests: plan
                .merged_requests
                .iter()
                .map(|m| MergedRequestGolden {
                    offset: m.offset,
                    length: m.length,
                })
                .collect(),
            assignments: plan.assignments.clone(),
        }
    }
}

#[derive(Serialize, Deserialize)]
struct AdaptiveGolden {
    /// Plan's `T_UB(P)` under the paper cost model, in milliseconds.
    t_ub_ms: f64,
}

#[derive(Serialize, Deserialize)]
struct QueryGolden {
    separate: DeterministicGolden,
    greedy: DeterministicGolden,
    serial: DeterministicGolden,
    bottleneck: DeterministicGolden,
    main_solution: AdaptiveGolden,
}

#[derive(Serialize, Deserialize)]
struct ExpectedPlans {
    cost_model_tag: String,
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

fn load_fixtures(dir: &Path) -> Vec<(String, Vec<IoRequest>)> {
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

fn main() {
    let cost = paper_cost();
    let constraints = PlanConstraints::default();
    let p = cost.concurrent_parallelism;

    for dir in fixture_dirs() {
        if !dir.exists() {
            eprintln!("[skip] {}", dir.display());
            continue;
        }
        let fixtures = load_fixtures(&dir);
        let mut queries = BTreeMap::new();

        for (stem, reqs) in &fixtures {
            let sep = Separate.generate_plan(reqs, &cost, &constraints);
            let gre = Greedy::default().generate_plan(reqs, &cost, &constraints);
            let ser = Serial.generate_plan(reqs, &cost, &constraints);
            let bot = Bottleneck.generate_plan(reqs, &cost, &constraints);
            let main = MainSolution.generate_plan(reqs, &cost, &constraints);
            for (name, plan) in [
                ("separate", &sep),
                ("greedy", &gre),
                ("serial", &ser),
                ("bottleneck", &bot),
                ("main_solution", &main),
            ] {
                plan.check_invariants(reqs)
                    .unwrap_or_else(|e| panic!("{stem}/{name}: invariant: {e}"));
            }
            let main_metrics = evaluate(reqs, &main, &cost);
            queries.insert(
                stem.clone(),
                QueryGolden {
                    separate: (&sep).into(),
                    greedy: (&gre).into(),
                    serial: (&ser).into(),
                    bottleneck: (&bot).into(),
                    main_solution: AdaptiveGolden {
                        t_ub_ms: t_ub(&main_metrics, p),
                    },
                },
            );
        }

        let expected = ExpectedPlans {
            cost_model_tag: "paper_piecewise_s3".to_string(),
            parallelism: p,
            queries,
        };
        let out_path = dir.join("expected_plans.json");
        let serialized = serde_json::to_string_pretty(&expected).unwrap();
        fs::write(&out_path, serialized).unwrap();
        println!(
            "[wrote] {}  ({} queries)",
            out_path.display(),
            fixtures.len()
        );
    }
}
