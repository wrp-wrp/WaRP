//! End-to-end demo: probe an object store, fit a cost model, plan.
//!
//! Uses the in-memory `object_store` backend so the example runs
//! offline. Swap in `AmazonS3Builder::from_env().build()?` (or any
//! other `object_store` implementation) to point at a real bucket.
//!
//! Build + run:
//!
//! ```bash
//! cargo run --release --example probe_and_plan --features probe
//! ```

use std::sync::Arc;

use object_store::{memory::InMemory, path::Path, ObjectStore, PutPayload};
use warp_planner::{
    evaluate, plan,
    probe::{probe_object_store, ProbePlan},
    strategy::MainSolution,
    IoRequest, PlanConstraints,
};

#[tokio::main(flavor = "current_thread")]
async fn main() -> anyhow::Result<()> {
    // 1. Prepare an object to probe against. In production you'd point
    //    at an existing object in your bucket.
    let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
    let path = Path::from("warp-planner/probe-fixture.bin");
    let size = 32 * 1024 * 1024;
    let buf: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
    store.put(&path, PutPayload::from(buf)).await?;

    // 2. Probe: times range-GETs of varying sizes, fits a cost model.
    println!("[1/3] probing object store…");
    let probe_plan = ProbePlan::default();
    let calibrator = probe_object_store(store, &path, &probe_plan).await?;
    let cost = calibrator.to_cost_model();
    println!(
        "  {} samples fitted → {} model ({:.3} ms overhead, {:.3e} ms/byte)",
        calibrator.sample_count(),
        cost.latency_model,
        cost.latency_per_request,
        cost.latency_per_byte,
    );

    // 3. Use the fitted model to plan a synthetic IVF query.
    println!("[2/3] planning a 64-cluster query…");
    let requests: Vec<IoRequest> = (0i32..64)
        .map(|i| IoRequest {
            cluster_id: i,
            offset: i as i64 * 50_000,
            length: 8_192,
        })
        .collect();
    let final_plan = plan(&MainSolution, &requests, &cost, &PlanConstraints::default())?;
    let metrics = evaluate(&requests, &final_plan, &cost);

    println!("[3/3] plan ready:");
    println!("  merged_requests: {}", final_plan.merged_requests.len());
    println!(
        "  predicted_latency_ms_serial: {:.3}",
        metrics.predicted_latency_ms_serial,
    );
    println!(
        "  predicted_latency_ms_concurrent: {:.3}",
        metrics.predicted_latency_ms_concurrent,
    );
    println!(
        "  waste: {} bytes ({} useful + {} waste)",
        metrics.wasted_bytes, metrics.useful_bytes, metrics.wasted_bytes,
    );
    Ok(())
}
