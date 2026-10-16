// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! QPS benchmark using Lance scheduler execution path.
//!
//! This benchmark exercises:
//! `ScanScheduler -> FileScheduler::submit_request -> scheduler queue -> object store`
//! and therefore includes real Lance I/O throttling behavior.
//!
//! Environment variables:
//! - `LANCE_QPS_URI` (required): object path, e.g. `s3://bucket/path/to/file.bin`
//! - `LANCE_QPS_NLIST` (default: 4096)
//! - `LANCE_QPS_NPROBE` (default: 128)
//! - `LANCE_QPS_NUM_QUERIES` (default: 200)
//! - `LANCE_QPS_WARMUP_QUERIES` (default: 20)
//! - `LANCE_QPS_QUERY_CONCURRENCY` (default: 8)
//! - `LANCE_QPS_SYNTH_CLUSTER_BYTES` (default: 262144)
//! - `LANCE_QPS_SEED` (default: 42)
//! - `LANCE_QPS_OUT_JSON` (optional): write metrics JSON to this path
//!
//! Merge planner is controlled by scheduler env vars, e.g.:
//! - `LANCE_IO_MERGE_STRATEGY=balanced_ub`
//! - `LANCE_IO_MERGE_MAX_REQUESTS=128`
//! - `LANCE_IO_MERGE_EXACT_REQUESTS=false`
//! - `LANCE_IO_MERGE_CONCURRENT_PARALLELISM=48`

use std::env;
use std::fs;
use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use lance_io::object_store::ObjectStore;
use lance_io::scheduler::{MergePlannerPolicy, ScanScheduler, SchedulerConfig};
use lance_io::utils::CachedFileSize;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use serde::Deserialize;
use snafu::location;
use tokio::sync::Semaphore;

#[derive(Clone, Copy, Debug)]
struct ClusterRange {
    offset: u64,
    size: u64,
}

#[derive(Debug, Clone, Copy)]
struct BenchConfig {
    nlist: usize,
    nprobe: usize,
    num_queries: usize,
    warmup_queries: usize,
    query_concurrency: usize,
    synth_cluster_bytes: u64,
    seed: u64,
}

#[derive(Debug)]
struct BenchResult {
    merge_strategy: String,
    uri: String,
    nlist: usize,
    nprobe: usize,
    num_queries: usize,
    warmup_queries: usize,
    query_concurrency: usize,
    wall_ms: f64,
    qps: f64,
    latency_avg_ms: f64,
    latency_p50_ms: f64,
    latency_p95_ms: f64,
    measured_requests: u64,
    measured_iops: u64,
    measured_bytes: u64,
}

#[derive(Debug)]
struct PerQueryRow {
    query_idx: usize,
    input_ranges: usize,
    latency_ms: f64,
    payload_bytes: u64,
    iops_delta: i64,
    bytes_read_delta: i64,
    planned_ranges: usize,
    planned_bytes: u64,
    planned_max_req_bytes: u64,
    model_w_ms: f64,
    model_b_ms: f64,
    model_tub_ms: f64,
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
    uri: Option<String>,
    nlist: Option<usize>,
    nprobe: Option<usize>,
    queries: Vec<Vec<Range<u64>>>,
    warmup_queries: Option<Vec<Vec<Range<u64>>>>,
}

fn parse_env_or<T: std::str::FromStr>(name: &str, default: T) -> T {
    match env::var(name) {
        Ok(v) => v.parse::<T>().unwrap_or(default),
        Err(_) => default,
    }
}

fn parse_env_bool(name: &str, default: bool) -> bool {
    match env::var(name) {
        Ok(v) => match v.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "y" | "on" => true,
            "0" | "false" | "no" | "n" | "off" => false,
            _ => default,
        },
        Err(_) => default,
    }
}

fn percentile(sorted_vals: &[f64], p: f64) -> f64 {
    if sorted_vals.is_empty() {
        return 0.0;
    }
    let idx = ((sorted_vals.len() - 1) as f64 * p).round() as usize;
    sorted_vals[idx]
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

fn load_query_ranges_file(path: &str) -> lance_core::Result<LoadedQueryRanges> {
    let text = fs::read_to_string(path).map_err(|e| {
        lance_core::Error::io(
            format!("failed to read query ranges file {}: {}", path, e),
            location!(),
        )
    })?;

    let parsed: QueryRangesInput = serde_json::from_str(&text).map_err(|e| {
        lance_core::Error::invalid_input(
            format!("failed to parse query ranges JSON {}: {}", path, e),
            location!(),
        )
    })?;

    match parsed {
        QueryRangesInput::Bare(queries) => Ok(LoadedQueryRanges {
            uri: None,
            nlist: None,
            nprobe: None,
            queries: convert_raw_queries(queries)?,
            warmup_queries: None,
        }),
        QueryRangesInput::WithMeta(v) => Ok(LoadedQueryRanges {
            uri: v.uri,
            nlist: v.nlist,
            nprobe: v.nprobe,
            queries: convert_raw_queries(v.queries)?,
            warmup_queries: match v.warmup_queries {
                Some(w) => Some(convert_raw_queries(w)?),
                None => None,
            },
        }),
    }
}

fn sample_queries_from_pool(
    pool: &[Vec<Range<u64>>],
    num_queries: usize,
    seed: u64,
) -> Vec<Vec<Range<u64>>> {
    if num_queries == 0 || pool.is_empty() {
        return Vec::new();
    }
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    if num_queries <= pool.len() {
        let mut ids: Vec<usize> = (0..pool.len()).collect();
        ids.shuffle(&mut rng);
        return ids
            .into_iter()
            .take(num_queries)
            .map(|i| pool[i].clone())
            .collect();
    }
    let mut out = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        let idx = rand::Rng::random_range(&mut rng, 0..pool.len());
        out.push(pool[idx].clone());
    }
    out
}

fn generate_synth_clusters(file_size: u64, nlist: usize, cluster_bytes: u64) -> Vec<ClusterRange> {
    if nlist == 0 {
        return Vec::new();
    }
    let clamped_cluster = cluster_bytes.min(file_size.max(1));
    if nlist == 1 || file_size <= clamped_cluster {
        return vec![ClusterRange {
            offset: 0,
            size: clamped_cluster,
        }];
    }

    let max_start = file_size.saturating_sub(clamped_cluster);
    let step = (max_start / (nlist as u64 - 1)).max(1);
    let mut clusters = Vec::with_capacity(nlist);
    for i in 0..nlist {
        let mut offset = i as u64 * step;
        if offset > max_start {
            offset = max_start;
        }
        clusters.push(ClusterRange {
            offset,
            size: clamped_cluster,
        });
    }
    clusters
}

fn generate_queries(
    clusters: &[ClusterRange],
    nprobe: usize,
    num_queries: usize,
    seed: u64,
) -> Vec<Vec<Range<u64>>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let nprobe = nprobe.min(clusters.len());

    let mut all_queries = Vec::with_capacity(num_queries);
    for _ in 0..num_queries {
        let mut ids: Vec<usize> = (0..clusters.len()).collect();
        ids.shuffle(&mut rng);
        ids.truncate(nprobe);
        ids.sort_unstable_by_key(|idx| clusters[*idx].offset);

        let ranges = ids
            .into_iter()
            .map(|idx| {
                let c = clusters[idx];
                c.offset..(c.offset + c.size)
            })
            .collect();
        all_queries.push(ranges);
    }
    all_queries
}

#[tokio::main]
async fn main() -> lance_core::Result<()> {
    let mut uri = env::var("LANCE_QPS_URI").ok();
    let query_ranges_path = env::var("LANCE_QPS_QUERY_RANGES_FILE").ok();
    let record_per_query = parse_env_bool("LANCE_QPS_RECORD_PER_QUERY", false);
    let out_per_query = env::var("LANCE_QPS_OUT_PER_QUERY").ok();
    let preplan = parse_env_bool("LANCE_QPS_PREPLAN", false);
    if uri.is_none() && query_ranges_path.is_none() {
        return Err(lance_core::Error::invalid_input(
            "LANCE_QPS_URI is required unless LANCE_QPS_QUERY_RANGES_FILE contains uri",
            location!(),
        ));
    }

    let mut cfg = BenchConfig {
        nlist: parse_env_or("LANCE_QPS_NLIST", 4096usize),
        nprobe: parse_env_or("LANCE_QPS_NPROBE", 128usize),
        num_queries: parse_env_or("LANCE_QPS_NUM_QUERIES", 200usize),
        warmup_queries: parse_env_or("LANCE_QPS_WARMUP_QUERIES", 20usize),
        query_concurrency: parse_env_or("LANCE_QPS_QUERY_CONCURRENCY", 8usize),
        synth_cluster_bytes: parse_env_or("LANCE_QPS_SYNTH_CLUSTER_BYTES", 256 * 1024u64),
        seed: parse_env_or("LANCE_QPS_SEED", 42u64),
    };
    let model_latency_req_ms = parse_env_or("LANCE_IO_MERGE_LATENCY_PER_REQUEST_MS", 169.3_f64);
    let model_latency_byte_ms = parse_env_or("LANCE_IO_MERGE_LATENCY_PER_BYTE_MS", 0.0_f64);
    let model_parallelism = parse_env_or("LANCE_IO_MERGE_CONCURRENT_PARALLELISM", 0.0_f64);

    let mut loaded_query_pool: Option<Vec<Vec<Range<u64>>>> = None;
    let mut loaded_warmup_pool: Option<Vec<Vec<Range<u64>>>> = None;

    if let Some(path) = query_ranges_path.as_deref() {
        let loaded = load_query_ranges_file(path)?;
        if uri.is_none() && loaded.uri.is_some() {
            uri = loaded.uri.clone();
        }
        if let Some(file_nlist) = loaded.nlist {
            cfg.nlist = file_nlist;
        }
        if let Some(file_nprobe) = loaded.nprobe {
            cfg.nprobe = file_nprobe;
        }
        loaded_query_pool = Some(loaded.queries);
        loaded_warmup_pool = loaded.warmup_queries;
    }

    let uri = uri.ok_or_else(|| {
        lance_core::Error::invalid_input(
            "No URI resolved. Set LANCE_QPS_URI or provide uri in LANCE_QPS_QUERY_RANGES_FILE",
            location!(),
        )
    })?;

    if cfg.query_concurrency == 0 {
        return Err(lance_core::Error::invalid_input(
            "LANCE_QPS_QUERY_CONCURRENCY must be > 0",
            location!(),
        ));
    }
    if cfg.nprobe == 0 || cfg.nlist == 0 {
        return Err(lance_core::Error::invalid_input(
            "LANCE_QPS_NLIST and LANCE_QPS_NPROBE must be > 0",
            location!(),
        ));
    }

    let (store, path) = ObjectStore::from_uri(&uri).await?;
    let scheduler = ScanScheduler::new(store.clone(), SchedulerConfig::max_bandwidth(&store));
    let file_scheduler = scheduler
        .open_file(&path, &CachedFileSize::unknown())
        .await?
        .with_merge_planner_policy(MergePlannerPolicy::Configured);

    let file_size = store.size(&path).await?;
    let (warmup_queries, queries, mut query_source) = if let Some(pool) = loaded_query_pool {
        if pool.is_empty() {
            return Err(lance_core::Error::invalid_input(
                "LANCE_QPS_QUERY_RANGES_FILE contains no queries",
                location!(),
            ));
        }
        let warm = match loaded_warmup_pool {
            Some(wpool) if !wpool.is_empty() => {
                sample_queries_from_pool(&wpool, cfg.warmup_queries, cfg.seed + 7)
            }
            _ => sample_queries_from_pool(&pool, cfg.warmup_queries, cfg.seed + 7),
        };
        let qs = sample_queries_from_pool(&pool, cfg.num_queries, cfg.seed + 17);
        (warm, qs, "real_query_ranges".to_string())
    } else {
        let clusters = generate_synth_clusters(file_size, cfg.nlist, cfg.synth_cluster_bytes);
        let warm = generate_queries(&clusters, cfg.nprobe, cfg.warmup_queries, cfg.seed + 7);
        let qs = generate_queries(&clusters, cfg.nprobe, cfg.num_queries, cfg.seed + 17);
        (warm, qs, "synthetic_clusters".to_string())
    };

    let mut preplanned_warmup: Option<Vec<Vec<Range<u64>>>> = None;
    let mut preplanned_queries: Option<Vec<Vec<Range<u64>>>> = None;
    let mut preplan_ms = 0.0_f64;
    if preplan {
        let t0 = Instant::now();
        preplanned_warmup = Some(
            warmup_queries
                .iter()
                .map(|req| file_scheduler.preview_request_plan(req))
                .collect(),
        );
        preplanned_queries = Some(
            queries
                .iter()
                .map(|req| file_scheduler.preview_request_plan(req))
                .collect(),
        );
        preplan_ms = t0.elapsed().as_secs_f64() * 1000.0;
        query_source = format!("{query_source}+preplanned");
    }

    println!("=== Lance Scheduler QPS Benchmark ===");
    println!("uri: {}", uri);
    println!(
        "merge_strategy: {}",
        env::var("LANCE_IO_MERGE_STRATEGY").unwrap_or_else(|_| "native".to_string())
    );
    println!(
        "nlist={}, nprobe={}, queries={}, warmup={}, concurrency={}, source={}",
        cfg.nlist,
        cfg.nprobe,
        cfg.num_queries,
        cfg.warmup_queries,
        cfg.query_concurrency,
        query_source
    );
    println!("preplan_enabled={}, preplan_ms={:.2}", preplan, preplan_ms);
    println!("file_size={:.2} MB", file_size as f64 / 1024.0 / 1024.0);

    // Warmup
    if let Some(plans) = preplanned_warmup.take() {
        for (req, planned) in warmup_queries.into_iter().zip(plans.into_iter()) {
            let _ = file_scheduler
                .submit_request_with_plan(req, planned, 0)
                .await?;
        }
    } else {
        for req in warmup_queries {
            let _ = file_scheduler.submit_request(req, 0).await?;
        }
    }

    let observed_nprobe = if queries.is_empty() {
        cfg.nprobe
    } else {
        let total = queries.iter().map(|q| q.len()).sum::<usize>();
        (total / queries.len()).max(1)
    };

    let before = scheduler.stats();
    let bench_start = Instant::now();
    let mut latencies_ms = Vec::with_capacity(cfg.num_queries);
    let mut returned_payload_bytes = 0_u64;
    let mut per_query_rows: Vec<PerQueryRow> = Vec::new();

    if record_per_query && cfg.query_concurrency == 1 {
        let query_plans = if let Some(plans) = preplanned_queries.take() {
            plans
        } else {
            queries
                .iter()
                .map(|req| file_scheduler.preview_request_plan(req))
                .collect()
        };
        for (qi, (req, planned_req)) in queries.into_iter().zip(query_plans.into_iter()).enumerate()
        {
            let input_ranges = req.len();
            let mut planned_bytes = 0_u64;
            let mut planned_max_req_bytes = 0_u64;
            for r in &planned_req {
                let len = r.end.saturating_sub(r.start);
                planned_bytes = planned_bytes.saturating_add(len);
                planned_max_req_bytes = planned_max_req_bytes.max(len);
            }
            let planned_ranges = planned_req.len();
            let model_w_ms = model_latency_req_ms * planned_ranges as f64
                + model_latency_byte_ms * planned_bytes as f64;
            let model_b_ms = if planned_ranges == 0 {
                0.0
            } else {
                model_latency_req_ms + model_latency_byte_ms * planned_max_req_bytes as f64
            };
            let model_tub_ms = if model_parallelism <= 0.0 {
                model_b_ms
            } else {
                let p = model_parallelism.max(1.0);
                model_w_ms / p + (1.0 - 1.0 / p) * model_b_ms
            };
            let stats_before = scheduler.stats();
            let t0 = Instant::now();
            let bytes = file_scheduler
                .submit_request_with_plan(req, planned_req, 0)
                .await?;
            let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let payload_bytes = bytes.iter().map(|b| b.len() as u64).sum::<u64>();
            let stats_after = scheduler.stats();

            latencies_ms.push(elapsed_ms);
            returned_payload_bytes += payload_bytes;
            per_query_rows.push(PerQueryRow {
                query_idx: qi,
                input_ranges,
                latency_ms: elapsed_ms,
                payload_bytes,
                iops_delta: (stats_after.iops as i64) - (stats_before.iops as i64),
                bytes_read_delta: (stats_after.bytes_read as i64)
                    - (stats_before.bytes_read as i64),
                planned_ranges,
                planned_bytes,
                planned_max_req_bytes,
                model_w_ms,
                model_b_ms,
                model_tub_ms,
            });
        }
    } else {
        let semaphore = Arc::new(Semaphore::new(cfg.query_concurrency));
        let fs = Arc::new(file_scheduler);
        let query_plans = if let Some(plans) = preplanned_queries.take() {
            plans
        } else {
            queries
                .iter()
                .map(|req| fs.preview_request_plan(req))
                .collect()
        };
        let mut handles = Vec::with_capacity(queries.len());

        for (req, planned_req) in queries.into_iter().zip(query_plans.into_iter()) {
            let fs = fs.clone();
            let sem = semaphore.clone();
            handles.push(tokio::spawn(async move {
                let _permit = sem.acquire_owned().await.unwrap();
                let t0 = Instant::now();
                let bytes = fs.submit_request_with_plan(req, planned_req, 0).await?;
                let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                let payload_bytes = bytes.iter().map(|b| b.len() as u64).sum::<u64>();
                Ok::<(f64, u64), lance_core::Error>((elapsed_ms, payload_bytes))
            }));
        }

        for handle in handles {
            let (lat_ms, payload) = handle.await.map_err(|e| {
                lance_core::Error::io(
                    format!("task join failed in QPS benchmark: {e}"),
                    location!(),
                )
            })??;
            latencies_ms.push(lat_ms);
            returned_payload_bytes += payload;
        }
    }

    let wall_ms = bench_start.elapsed().as_secs_f64() * 1000.0;
    let after = scheduler.stats();

    latencies_ms.sort_by(|a, b| a.total_cmp(b));
    let avg = if latencies_ms.is_empty() {
        0.0
    } else {
        latencies_ms.iter().sum::<f64>() / latencies_ms.len() as f64
    };

    let measured_requests = after.requests.saturating_sub(before.requests);
    let measured_iops = after.iops.saturating_sub(before.iops);
    let measured_bytes = after.bytes_read.saturating_sub(before.bytes_read);
    let qps = if wall_ms > 0.0 {
        cfg.num_queries as f64 / (wall_ms / 1000.0)
    } else {
        0.0
    };

    let result = BenchResult {
        merge_strategy: env::var("LANCE_IO_MERGE_STRATEGY")
            .unwrap_or_else(|_| "native".to_string()),
        uri,
        nlist: cfg.nlist,
        nprobe: observed_nprobe,
        num_queries: cfg.num_queries,
        warmup_queries: cfg.warmup_queries,
        query_concurrency: cfg.query_concurrency,
        wall_ms,
        qps,
        latency_avg_ms: avg,
        latency_p50_ms: percentile(&latencies_ms, 0.50),
        latency_p95_ms: percentile(&latencies_ms, 0.95),
        measured_requests,
        measured_iops,
        measured_bytes,
    };

    println!("--- Results ---");
    println!("wall_ms={:.2}", result.wall_ms);
    println!("qps={:.2}", result.qps);
    println!(
        "latency_ms avg={:.2} p50={:.2} p95={:.2}",
        result.latency_avg_ms, result.latency_p50_ms, result.latency_p95_ms
    );
    println!(
        "scheduler_stats requests={} iops={} bytes_read={:.2} MB",
        result.measured_requests,
        result.measured_iops,
        result.measured_bytes as f64 / 1024.0 / 1024.0
    );
    println!(
        "returned_payload_bytes={:.2} MB",
        returned_payload_bytes as f64 / 1024.0 / 1024.0
    );

    if let Some(path) = out_per_query {
        let mut text = String::from(
            "query_idx,input_ranges,latency_ms,payload_bytes,iops_delta,bytes_read_delta,planned_ranges,planned_bytes,planned_max_req_bytes,model_w_ms,model_b_ms,model_tub_ms\n",
        );
        for r in &per_query_rows {
            text.push_str(&format!(
                "{},{},{:.6},{},{},{},{},{},{},{:.6},{:.6},{:.6}\n",
                r.query_idx,
                r.input_ranges,
                r.latency_ms,
                r.payload_bytes,
                r.iops_delta,
                r.bytes_read_delta,
                r.planned_ranges,
                r.planned_bytes,
                r.planned_max_req_bytes,
                r.model_w_ms,
                r.model_b_ms,
                r.model_tub_ms
            ));
        }
        std::fs::write(&path, text).map_err(|e| {
            lance_core::Error::io(
                format!("failed to write per-query csv to {}: {}", path, e),
                location!(),
            )
        })?;
        println!("per_query_csv={}", path);
    }

    if let Ok(path) = env::var("LANCE_QPS_OUT_JSON") {
        let text = format!(
            "merge_strategy={}\nuri={}\nnlist={}\nnprobe={}\nnum_queries={}\nwarmup_queries={}\nquery_concurrency={}\nwall_ms={:.6}\nqps={:.6}\nlatency_avg_ms={:.6}\nlatency_p50_ms={:.6}\nlatency_p95_ms={:.6}\nmeasured_requests={}\nmeasured_iops={}\nmeasured_bytes={}\n",
            result.merge_strategy,
            result.uri,
            result.nlist,
            result.nprobe,
            result.num_queries,
            result.warmup_queries,
            result.query_concurrency,
            result.wall_ms,
            result.qps,
            result.latency_avg_ms,
            result.latency_p50_ms,
            result.latency_p95_ms,
            result.measured_requests,
            result.measured_iops,
            result.measured_bytes
        );
        std::fs::write(&path, text).map_err(|e| {
            lance_core::Error::io(
                format!("failed to write result json to {}: {}", path, e),
                location!(),
            )
        })?;
        println!("result_json={}", path);
    }

    Ok(())
}
