// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Real S3 benchmark with actual AWS requests
//!
//! This benchmark tests the I/O merge optimization on real S3 storage.
//!
//! Required environment variables:
//! - `LANCE_REAL_S3_BUCKET`
//!
//! Optional environment variables:
//! - `LANCE_REAL_S3_REGION` (default: `us-east-1`)
//! - `LANCE_REAL_S3_INDEX_KEY` (default: `sift1m_ivf4096_pq32.index`)
//! - `LANCE_REAL_S3_METADATA_KEY` (default: `cluster_metadata.json`)
//! - `LANCE_REAL_S3_NPROBES` (default: `16,64,128`)

use lance_io::scheduler::{CostModel, MergeStrategy, SerialDP};
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjectPath;
use object_store::ObjectStore;
use serde::Deserialize;
use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::Semaphore;

const DEFAULT_REGION: &str = "us-east-1";
const DEFAULT_INDEX_KEY: &str = "sift1m_ivf4096_pq32.index";
const DEFAULT_METADATA_KEY: &str = "cluster_metadata.json";

#[derive(Debug, Clone)]
struct Config {
    bucket: String,
    region: String,
    index_key: String,
    metadata_key: String,
    nprobe_values: Vec<usize>,
}

#[derive(Debug, Deserialize)]
struct ClusterMetadata {
    cluster_id: i32,
    offset: i64,
    size: i64,
    #[allow(dead_code)]
    num_vectors: usize,
}

fn env_or(name: &str, default: &str) -> String {
    env::var(name).unwrap_or_else(|_| default.to_string())
}

fn parse_nprobe_values(raw: &str) -> Result<Vec<usize>, String> {
    let mut out = Vec::new();
    for token in raw.split(',') {
        let s = token.trim();
        if s.is_empty() {
            continue;
        }
        let parsed = s
            .parse::<usize>()
            .map_err(|e| format!("failed to parse LANCE_REAL_S3_NPROBES element `{s}`: {e}"))?;
        out.push(parsed);
    }
    out.sort_unstable();
    out.dedup();
    if out.is_empty() {
        return Err("LANCE_REAL_S3_NPROBES produced an empty list".to_string());
    }
    Ok(out)
}

fn load_config() -> Result<Config, String> {
    let bucket = env::var("LANCE_REAL_S3_BUCKET").map_err(|_| {
        "missing LANCE_REAL_S3_BUCKET; set it to the S3 bucket containing the benchmark objects"
            .to_string()
    })?;
    let region = env_or("LANCE_REAL_S3_REGION", DEFAULT_REGION);
    let index_key = env_or("LANCE_REAL_S3_INDEX_KEY", DEFAULT_INDEX_KEY);
    let metadata_key = env_or("LANCE_REAL_S3_METADATA_KEY", DEFAULT_METADATA_KEY);
    let nprobe_values = parse_nprobe_values(&env_or("LANCE_REAL_S3_NPROBES", "16,64,128"))?;
    Ok(Config {
        bucket,
        region,
        index_key,
        metadata_key,
        nprobe_values,
    })
}

/// Load cluster metadata from S3
async fn load_metadata(store: &dyn ObjectStore, metadata_key: &str) -> Vec<ClusterMetadata> {
    println!("Loading cluster metadata from S3: {}", metadata_key);

    let path = ObjectPath::from(metadata_key);
    let result = store.get(&path).await.expect("Failed to load metadata");
    let bytes = result.bytes().await.expect("Failed to read bytes");

    let metadata: Vec<ClusterMetadata> =
        serde_json::from_slice(&bytes).expect("Failed to parse metadata");

    println!("Loaded metadata for {} clusters", metadata.len());
    metadata
}

/// Simulate IVF query: select nprobe random clusters
fn select_clusters(metadata: &[ClusterMetadata], nprobe: usize) -> Vec<usize> {
    use rand::rng;
    use rand::seq::SliceRandom;

    let mut indices: Vec<usize> = (0..metadata.len()).collect();
    indices.shuffle(&mut rng());
    indices.truncate(nprobe);
    indices.sort();
    indices
}

/// Test baseline: sequential requests (no merge)
async fn test_baseline(
    store: &dyn ObjectStore,
    index_key: &str,
    metadata: &[ClusterMetadata],
    selected: &[usize],
) -> (f64, usize) {
    println!("\n=== Testing Baseline (Sequential) ===");

    let start = Instant::now();
    let mut total_bytes = 0;

    let path = ObjectPath::from(index_key);

    for &idx in selected {
        let cluster = &metadata[idx];
        let range = (cluster.offset as u64)..(cluster.offset + cluster.size) as u64;

        let result = store
            .get_range(&path, range)
            .await
            .expect("Failed to get range");
        total_bytes += result.len();
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    println!("Requests: {}", selected.len());
    println!(
        "Total bytes: {:.2} MB",
        total_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("Latency: {:.2} ms", elapsed);
    println!("Avg per request: {:.2} ms", elapsed / selected.len() as f64);

    (elapsed, selected.len())
}

/// Test with merge strategy
async fn test_with_strategy(
    store: &dyn ObjectStore,
    index_key: &str,
    metadata: &[ClusterMetadata],
    selected: &[usize],
    strategy: &dyn MergeStrategy,
    strategy_name: &str,
) -> (f64, usize) {
    println!("\n=== Testing {} ===", strategy_name);

    // Prepare data for strategy
    let cluster_ids: Vec<i32> = selected
        .iter()
        .map(|&idx| metadata[idx].cluster_id)
        .collect();
    let cluster_offsets: HashMap<i32, i64> = selected
        .iter()
        .map(|&idx| (metadata[idx].cluster_id, metadata[idx].offset))
        .collect();
    let cluster_sizes: HashMap<i32, i64> = selected
        .iter()
        .map(|&idx| (metadata[idx].cluster_id, metadata[idx].size))
        .collect();

    let mut cost_model = CostModel::default();
    cost_model.latency_per_request = 169.3; // S3 typical
    cost_model.max_waste_ratio = 0.5;

    // Generate merge plan
    let plan = strategy.generate_plan(&cluster_ids, &cluster_offsets, &cluster_sizes, &cost_model);

    println!("Generated plan: {} requests", plan.requests.len());
    println!("Predicted latency: {:.2} ms", plan.metrics.total_latency);

    // Execute requests
    let start = Instant::now();
    let mut total_bytes = 0;

    let path = ObjectPath::from(index_key);

    for request in &plan.requests {
        let range = (request.offset as u64)..(request.offset + request.length) as u64;

        let result = store
            .get_range(&path, range)
            .await
            .expect("Failed to get range");
        total_bytes += result.len();
    }

    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    println!("Actual requests: {}", plan.requests.len());
    println!(
        "Total bytes: {:.2} MB",
        total_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("Actual latency: {:.2} ms", elapsed);
    println!(
        "Avg per request: {:.2} ms",
        elapsed / plan.requests.len() as f64
    );

    (elapsed, plan.requests.len())
}

/// Test with concurrent execution
async fn test_concurrent(
    store: Arc<dyn ObjectStore>,
    index_key: &str,
    metadata: &[ClusterMetadata],
    selected: &[usize],
    strategy: &dyn MergeStrategy,
    strategy_name: &str,
    parallelism: usize,
) -> (f64, usize) {
    println!(
        "\n=== Testing {} (Concurrent P={}) ===",
        strategy_name, parallelism
    );

    // Generate merge plan
    let cluster_ids: Vec<i32> = selected
        .iter()
        .map(|&idx| metadata[idx].cluster_id)
        .collect();
    let cluster_offsets: HashMap<i32, i64> = selected
        .iter()
        .map(|&idx| (metadata[idx].cluster_id, metadata[idx].offset))
        .collect();
    let cluster_sizes: HashMap<i32, i64> = selected
        .iter()
        .map(|&idx| (metadata[idx].cluster_id, metadata[idx].size))
        .collect();

    let mut cost_model = CostModel::default();
    cost_model.latency_per_request = 169.3;
    cost_model.max_waste_ratio = 0.5;

    let plan = strategy.generate_plan(&cluster_ids, &cluster_offsets, &cluster_sizes, &cost_model);

    println!("Generated plan: {} requests", plan.requests.len());

    // Execute concurrent requests
    let start = Instant::now();
    let semaphore = Arc::new(Semaphore::new(parallelism));
    let path = ObjectPath::from(index_key);

    let futures: Vec<_> = plan
        .requests
        .iter()
        .map(|request| {
            let sem = semaphore.clone();
            let store = store.clone();
            let path = path.clone();
            let range = (request.offset as u64)..(request.offset + request.length) as u64;

            async move {
                let _permit = sem.acquire().await.unwrap();

                let result = store
                    .get_range(&path, range)
                    .await
                    .expect("Failed to get range");
                result.len()
            }
        })
        .collect();

    let results = futures::future::join_all(futures).await;
    let elapsed = start.elapsed().as_secs_f64() * 1000.0;

    let total_bytes: usize = results.iter().sum();

    println!("Actual requests: {}", plan.requests.len());
    println!(
        "Total bytes: {:.2} MB",
        total_bytes as f64 / (1024.0 * 1024.0)
    );
    println!("Actual latency: {:.2} ms", elapsed);
    println!(
        "Effective throughput: {:.2} MB/s",
        (total_bytes as f64 / (1024.0 * 1024.0)) / (elapsed / 1000.0)
    );

    (elapsed, plan.requests.len())
}

#[tokio::main]
async fn main() {
    let config = match load_config() {
        Ok(config) => config,
        Err(msg) => {
            eprintln!("{msg}");
            std::process::exit(2);
        }
    };

    println!("=== Real S3 I/O Merge Benchmark ===\n");
    println!("bucket: {}", config.bucket);
    println!("region: {}", config.region);
    println!("index_key: {}", config.index_key);
    println!("metadata_key: {}", config.metadata_key);

    // Setup S3 client using object_store
    let store = AmazonS3Builder::new()
        .with_bucket_name(&config.bucket)
        .with_region(&config.region)
        .build()
        .expect("Failed to create S3 store");

    let store = Arc::new(store);

    // Load metadata
    let metadata = load_metadata(store.as_ref(), &config.metadata_key).await;

    // Test parameters
    for &nprobe in &config.nprobe_values {
        println!("\n{}", "=".repeat(70));
        println!("Testing with nprobe = {}", nprobe);
        println!("{}", "=".repeat(70));

        // Select clusters
        let selected = select_clusters(&metadata, nprobe);

        // Test 1: Baseline (sequential)
        let (baseline_latency, baseline_requests) =
            test_baseline(store.as_ref(), &config.index_key, &metadata, &selected).await;

        // Test 2: SerialDP (sequential)
        let (serial_latency, serial_requests) = test_with_strategy(
            store.as_ref(),
            &config.index_key,
            &metadata,
            &selected,
            &SerialDP::new(),
            "SerialDP",
        )
        .await;

        // Test 3: SerialDP (concurrent P=16)
        let (concurrent_latency, concurrent_requests) = test_concurrent(
            store.clone(),
            &config.index_key,
            &metadata,
            &selected,
            &SerialDP::new(),
            "SerialDP",
            16,
        )
        .await;

        // Summary
        println!("\n=== Summary for nprobe={} ===", nprobe);
        println!(
            "{:<20} {:>12} {:>15} {:>12}",
            "Strategy", "Requests", "Latency (ms)", "Speedup"
        );
        println!("{}", "-".repeat(65));
        println!(
            "{:<20} {:>12} {:>15.1} {:>12}",
            "Baseline", baseline_requests, baseline_latency, 1.0
        );
        println!(
            "{:<20} {:>12} {:>15.1} {:>12.2}x",
            "SerialDP",
            serial_requests,
            serial_latency,
            baseline_latency / serial_latency
        );
        println!(
            "{:<20} {:>12} {:>15.1} {:>12.2}x",
            "SerialDP (P=16)",
            concurrent_requests,
            concurrent_latency,
            baseline_latency / concurrent_latency
        );
    }
}
