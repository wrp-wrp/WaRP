//! Fit a `CostModel` from latency samples.
//!
//! Records `(bytes_fetched, observed_ms)` pairs you measured
//! externally — say from a probe loop or production tracing — and
//! turns them into a populated `CostModel` ready to feed into a
//! strategy.
//!
//! ```bash
//! cargo run --release --example calibrate
//! ```

use warp_planner::{evaluate, plan, Calibrator, IoRequest, MainSolution, PlanConstraints};

fn main() {
    // Imagine you measured these `(bytes, latency_ms)` pairs by
    // probing your bucket. They follow the paper's two-segment shape:
    // a small-request floor (~27 ms) then a bandwidth-bound regime
    // for larger reads.
    let samples = [
        (4 * 1024, 27.4),
        (64 * 1024, 28.1),
        (256 * 1024, 29.6),
        (1024 * 1024, 33.1),
        (3 * 1024 * 1024, 58.7),
        (6 * 1024 * 1024, 89.2),
        (16 * 1024 * 1024, 195.0),
    ];

    let mut cal = Calibrator::new();
    for (bytes, ms) in samples {
        cal.record(bytes, ms);
    }
    println!("Recorded {} samples.\n", cal.sample_count());

    let (overhead_ms, ms_per_byte) = cal.fit_linear();
    println!(
        "Linear fit:    overhead = {:.2} ms, per-byte = {:.3e} ms/byte",
        overhead_ms, ms_per_byte,
    );
    if let Some(pw) = cal.fit_piecewise() {
        println!(
            "Piecewise fit: break = {} bytes, overheads = [{:.2}, {:.2}] ms, \
             rates = [{:.3e}, {:.3e}] ms/byte",
            pw.break_bytes,
            pw.overheads_ms[0],
            pw.overheads_ms[1],
            pw.rates_ms_per_byte[0],
            pw.rates_ms_per_byte[1],
        );
    }

    // Materialise a CostModel from the better of the two fits.
    let mut cost = cal.to_cost_model();
    println!("\nFitted model: {}", cost.latency_model);

    // The Calibrator only fills the latency parts. Objective context
    // such as parallelism is still your call.
    cost.concurrent_parallelism = 16.0;

    // Use it.
    let requests = vec![
        IoRequest {
            cluster_id: 0,
            offset: 0,
            length: 4096,
        },
        IoRequest {
            cluster_id: 1,
            offset: 4096,
            length: 4096,
        },
        IoRequest {
            cluster_id: 2,
            offset: 9_000,
            length: 4096,
        },
    ];
    let plan = plan(&MainSolution, &requests, &cost, &PlanConstraints::default())
        .expect("the calibrated model and example inputs are valid");
    let m = evaluate(&requests, &plan, &cost);
    println!(
        "Plan: {} request(s), {:.2} ms predicted (serial)",
        plan.merged_requests.len(),
        m.predicted_latency_ms_serial,
    );
}
