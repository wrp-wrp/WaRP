//! Full-pipeline Q=N/R=M microbench.
//!
//! Runs the complete Lance IVF search (`ds.scan().nearest().nprobes(...)`)
//! with strict Q concurrent queries via `Semaphore::new(Q)` — same
//! enforcement as `strict_budget_real_s3.rs` but exercising the full
//! read path (open_vector_index → find_partitions → batch_prefetch →
//! search_in_partition → top-K fetch) instead of bypassing it.
//!
//! Control knobs via env vars (for consistency with existing tooling):
//!   LANCE_PROCESS_IO_THREADS_LIMIT         = R (global in-flight cap)
//!   LANCE_IO_MERGE_STRATEGY                = separate|lance|balanced_ub
//!   LANCE_IO_MERGE_MAX_REQUESTS            = K (planner output cap)
//!   LANCE_IO_MERGE_LATENCY_MODEL           = piecewise_linear
//!   LANCE_IO_MERGE_LATENCY_PW_BREAKS_BYTES = ...
//!   LANCE_IO_MERGE_LATENCY_PW_OVERHEADS_MS = ...
//!   LANCE_IO_MERGE_LATENCY_PW_RATES_MS_PER_BYTE = ...
//!   LANCE_IVF_BATCH_PREFETCH               = 1 (batched methods)
//!   LANCE_IVF_STOCK_EXECUTION              = 1 (Stock Lance only)
//!   LANCE_DISABLE_PARTITION_CACHE          = 1 (optional; match paper
//!                                              microbench no-cache semantics)
//!
//! CLI flags (positional, parsed from argv):
//!   --uri s3://bucket/path/sift1m.lance
//!   --queries-npy /path/to/queries.npy
//!   --num-queries 120
//!   --warmup-queries 20
//!   --reuse-warmup-queries (optional; warm up on the first rows and then
//!       measure from query 0, for fixed query sets with exactly N rows)
//!   --query-concurrency 4
//!   --nprobe 128
//!   --k 10
//!   --out-json /tmp/result.json   (optional)
//!   --out-latencies-jsonl /tmp/latencies.jsonl (optional; writes one
//!       record per measured query after timing finishes, so capturing
//!       raw tail-latency samples does not perturb the measured interval.)
//!   --out-query-ranges-json /tmp/ranges.json (optional; exports the exact
//!       raw auxiliary-file ranges selected for each query. This mode is
//!       sequential and disables the partition cache.)
//!   --per-query-cache-jsonl /tmp/cache.jsonl (optional; records, for each measured
//!       query, the index-cache miss count attributable to that query
//!       (full-hit queries bypass remote partition fetches entirely, so
//!       miss_count==0 identifies them). Attribution is only exact under
//!       sequential execution -- refuses to run with
//!       --query-concurrency != 1, since concurrent queries would share
//!       overlapping before/after snapshots of the same global counter.)
//!   --measured-passes N (default 1; re-runs the exact same measured query set N times
//!       against the SAME open Dataset/Session, so the index cache
//!       carries over between passes -- pass 1 is the cold/warming-up
//!       pass (C1), pass 2+ are warm-cache passes (C2). Prints one JSON
//!       result line per pass, each tagged with "pass". Mutually
//!       exclusive with --per-query-cache-jsonl.)

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::Float32Array;
use futures::TryStreamExt;
use lance::dataset::builder::DatasetBuilder;
use lance::Dataset;
use lance_io::scheduler::{
    bytes_read_counter, iops_counter, iops_over_five_seconds, iops_over_one_second,
    max_iop_latency_us, max_queue_wait_us, max_quota_wait_us, peak_iops_in_flight,
    queue_waits_over_five_seconds, queue_waits_over_one_second, quota_waits_over_five_seconds,
    quota_waits_over_one_second, reset_iop_latency_stats, reset_payload_range_trace,
    reset_peak_iops_in_flight, take_payload_range_trace,
};
use serde::Serialize;
use tokio::sync::Semaphore;

const HELP: &str = "Usage: full_pipeline_real_s3 --uri <URI> --queries-npy <PATH> [OPTIONS]\n\
\n\
Options:\n\
  --num-queries <N>\n\
  --warmup-queries <N>\n\
  --reuse-warmup-queries\n\
  --query-concurrency <N>\n\
  --nprobe <N>\n\
  --k <N>\n\
  --refine-factor <N>\n\
  --out-json <PATH>\n\
  --out-latencies-jsonl <PATH>\n\
  --out-query-ranges-json <PATH>\n\
  --index-cache-mb <MIB>\n\
  --per-query-cache-jsonl <PATH>\n\
  --measured-passes <N>\n\
  -h, --help\n";

#[derive(Debug, Default)]
struct Args {
    uri: String,
    queries_npy: PathBuf,
    num_queries: usize,
    warmup_queries: usize,
    reuse_warmup_queries: bool,
    query_concurrency: usize,
    nprobe: usize,
    k: usize,
    refine_factor: u32,
    out_json: Option<PathBuf>,
    out_latencies_jsonl: Option<PathBuf>,
    out_query_ranges_json: Option<PathBuf>,
    index_cache_mb: Option<usize>,
    per_query_cache_jsonl: Option<PathBuf>,
    measured_passes: usize,
}

fn parse_args() -> Args {
    let mut args = Args {
        num_queries: 120,
        warmup_queries: 20,
        query_concurrency: 4,
        nprobe: 128,
        k: 10,
        refine_factor: 1,
        measured_passes: 1,
        ..Default::default()
    };
    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--uri" => {
                args.uri = argv[i + 1].clone();
                i += 2;
            }
            "--queries-npy" => {
                args.queries_npy = PathBuf::from(&argv[i + 1]);
                i += 2;
            }
            "--num-queries" => {
                args.num_queries = argv[i + 1].parse().unwrap();
                i += 2;
            }
            "--warmup-queries" => {
                args.warmup_queries = argv[i + 1].parse().unwrap();
                i += 2;
            }
            "--reuse-warmup-queries" => {
                args.reuse_warmup_queries = true;
                i += 1;
            }
            "--query-concurrency" => {
                args.query_concurrency = argv[i + 1].parse().unwrap();
                i += 2;
            }
            "--nprobe" => {
                args.nprobe = argv[i + 1].parse().unwrap();
                i += 2;
            }
            "--k" => {
                args.k = argv[i + 1].parse().unwrap();
                i += 2;
            }
            "--refine-factor" => {
                args.refine_factor = argv[i + 1].parse().unwrap();
                i += 2;
            }
            "--out-json" => {
                args.out_json = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--out-latencies-jsonl" => {
                args.out_latencies_jsonl = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--out-query-ranges-json" => {
                args.out_query_ranges_json = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--index-cache-mb" => {
                args.index_cache_mb = Some(argv[i + 1].parse().unwrap());
                i += 2;
            }
            "--per-query-cache-jsonl" => {
                args.per_query_cache_jsonl = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--measured-passes" => {
                args.measured_passes = argv[i + 1].parse().unwrap();
                i += 2;
            }
            "-h" | "--help" => {
                print!("{HELP}");
                std::process::exit(0);
            }
            flag => panic!("unknown flag {}", flag),
        }
    }
    if args.uri.is_empty() || args.queries_npy.as_os_str().is_empty() {
        eprintln!("required: --uri s3://... --queries-npy /path/queries.npy");
        std::process::exit(2);
    }
    if args.per_query_cache_jsonl.is_some() && args.query_concurrency != 1 {
        eprintln!(
            "--per-query-cache-jsonl requires --query-concurrency 1 \
                   (concurrent queries share overlapping cache-stat snapshots \
                   and cannot be attributed individually)"
        );
        std::process::exit(2);
    }
    if args.per_query_cache_jsonl.is_some() && args.measured_passes != 1 {
        eprintln!(
            "--per-query-cache-jsonl and --measured-passes are mutually \
                   exclusive; run separate invocations for warm-pass dynamics \
                   vs. per-query full-hit classification"
        );
        std::process::exit(2);
    }
    if args.measured_passes == 0 {
        eprintln!("--measured-passes must be >= 1");
        std::process::exit(2);
    }
    if args.out_query_ranges_json.is_some() && args.query_concurrency != 1 {
        eprintln!("--out-query-ranges-json requires --query-concurrency 1");
        std::process::exit(2);
    }
    args
}

#[derive(Serialize)]
struct QueryRangeTrace {
    schema_version: u32,
    index_uuids: Vec<String>,
    queries: Vec<Vec<[u64; 2]>>,
}

/// Minimal .npy reader for 2D float32 arrays (C-order).
/// Assumes header format produced by `np.save` of a float32 ndarray.
fn load_queries_f32(path: &std::path::Path) -> (Vec<Vec<f32>>, usize) {
    let mut f = File::open(path).expect("open queries.npy");
    let mut magic = [0u8; 6];
    f.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"\x93NUMPY");
    let mut ver = [0u8; 2];
    f.read_exact(&mut ver).unwrap();
    let header_len = if ver[0] == 1 {
        let mut hl = [0u8; 2];
        f.read_exact(&mut hl).unwrap();
        u16::from_le_bytes(hl) as usize
    } else {
        let mut hl = [0u8; 4];
        f.read_exact(&mut hl).unwrap();
        u32::from_le_bytes(hl) as usize
    };
    let mut header_bytes = vec![0u8; header_len];
    f.read_exact(&mut header_bytes).unwrap();
    let header = std::str::from_utf8(&header_bytes).unwrap();
    // e.g. {'descr': '<f4', 'fortran_order': False, 'shape': (10000, 128), }
    let shape_start = header.find("'shape':").unwrap() + "'shape':".len();
    let shape_str = &header[shape_start..];
    let open = shape_str.find('(').unwrap();
    let close = shape_str.find(')').unwrap();
    let inner = &shape_str[open + 1..close];
    let dims: Vec<usize> = inner
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| s.parse::<usize>().unwrap())
        .collect();
    assert_eq!(dims.len(), 2, "expected 2D queries array, got {:?}", dims);
    let (n, d) = (dims[0], dims[1]);
    assert!(
        header.contains("<f4") || header.contains("f4"),
        "only float32 supported: header = {}",
        header
    );
    let mut buf = vec![0u8; n * d * 4];
    f.read_exact(&mut buf).unwrap();
    let mut rows: Vec<Vec<f32>> = Vec::with_capacity(n);
    for i in 0..n {
        let mut row = Vec::with_capacity(d);
        for j in 0..d {
            let off = (i * d + j) * 4;
            let v = f32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
            row.push(v);
        }
        rows.push(row);
    }
    (rows, d)
}

async fn do_one_query(ds: &Dataset, q: &[f32], k: usize, nprobe: usize, refine_factor: u32) -> f64 {
    let q_arr = Float32Array::from(q.to_vec());
    let mut scanner = ds.scan();
    scanner.nearest("vector", &q_arr, k).expect("nearest");
    scanner.nprobes(nprobe);
    if refine_factor > 1 {
        scanner.refine(refine_factor);
    }
    scanner.project(&["id"]).expect("project");
    let t0 = Instant::now();
    let stream = scanner.try_into_stream().await.expect("into_stream");
    let _batches: Vec<_> = stream.try_collect().await.expect("collect");
    t0.elapsed().as_secs_f64() * 1000.0
}

async fn export_query_ranges(ds: &Dataset, queries: &[Vec<f32>], args: &Args, path: &PathBuf) {
    let mut all_index_uuids = Vec::new();
    let mut query_ranges = Vec::with_capacity(args.num_queries);
    for (query_index, query) in queries.iter().take(args.num_queries).enumerate() {
        reset_payload_range_trace();
        do_one_query(ds, query, args.k, args.nprobe, args.refine_factor).await;
        let (index_uuids, ranges) = take_payload_range_trace();
        assert!(
            !index_uuids.is_empty(),
            "query {query_index} did not select a vector index"
        );
        assert!(
            !ranges.is_empty(),
            "query {query_index} did not submit auxiliary payload ranges"
        );
        for index_uuid in index_uuids {
            if !all_index_uuids.contains(&index_uuid) {
                all_index_uuids.push(index_uuid);
            }
        }
        query_ranges.push(
            ranges
                .into_iter()
                .map(|range| [range.start, range.end])
                .collect(),
        );
    }
    all_index_uuids.sort();
    let trace = QueryRangeTrace {
        schema_version: 1,
        index_uuids: all_index_uuids,
        queries: query_ranges,
    };
    let payload = serde_json::to_vec_pretty(&trace).expect("serialize query-range trace");
    std::fs::write(path, payload).expect("write out-query-ranges-json");
}

#[tokio::main]
async fn main() {
    if std::env::var_os("WARP_TRACE_OBJECT_STORE").is_some() {
        let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
            tracing_subscriber::EnvFilter::new("object_store::client::retry=info")
        });
        let _ = tracing_subscriber::fmt()
            .with_env_filter(filter)
            .with_target(true)
            .try_init();
    }
    let args = parse_args();
    let trace_export = args.out_query_ranges_json.is_some();
    if trace_export {
        std::env::set_var("LANCE_PAYLOAD_PLAN_ONLY", "1");
        std::env::set_var("LANCE_IVF_BATCH_PREFETCH", "1");
        std::env::remove_var("LANCE_IVF_STOCK_EXECUTION");
        std::env::set_var("LANCE_DISABLE_PARTITION_CACHE", "1");
    }
    let batch_execution = std::env::var_os("LANCE_IVF_BATCH_PREFETCH").is_some();
    let stock_execution = std::env::var_os("LANCE_IVF_STOCK_EXECUTION").is_some();
    assert!(
        batch_execution ^ stock_execution,
        "full_pipeline_real_s3 requires exactly one execution path: \
         LANCE_IVF_BATCH_PREFETCH=1 or LANCE_IVF_STOCK_EXECUTION=1"
    );
    eprintln!(
        "[full_pipeline] uri={}  NQ={}  warmup={}  Q={}  nprobe={}  k={}  rf={}",
        args.uri,
        args.num_queries,
        args.warmup_queries,
        args.query_concurrency,
        args.nprobe,
        args.k,
        args.refine_factor
    );
    let ds = {
        let mut b = DatasetBuilder::from_uri(&args.uri);
        if let Some(mb) = args.index_cache_mb {
            let bytes = mb.saturating_mul(1024 * 1024);
            eprintln!(
                "[full_pipeline] index_cache_size_bytes={} ({} MiB)",
                bytes, mb
            );
            b = b.with_index_cache_size_bytes(bytes);
        } else {
            eprintln!("[full_pipeline] index_cache_size_bytes=default (6 GiB)");
        }
        Arc::new(b.load().await.expect("open dataset"))
    };
    let (queries, _dim) = load_queries_f32(&args.queries_npy);
    let measured_query_offset = if args.reuse_warmup_queries {
        0
    } else {
        args.warmup_queries
    };
    let total_needed = if args.reuse_warmup_queries {
        args.warmup_queries.max(args.num_queries)
    } else {
        args.warmup_queries + args.num_queries
    };
    let queries: Vec<Vec<f32>> = queries.into_iter().take(total_needed).collect();
    assert!(queries.len() >= total_needed, "queries file too small");

    if let Some(path) = args.out_query_ranges_json.as_ref() {
        export_query_ranges(&ds, &queries, &args, path).await;
        if let Some(result_path) = args.out_json.as_ref() {
            let result = format!(
                r#"{{"mode":"query_range_export","num_queries":{},"nprobe":{},"out_query_ranges_json":"{}"}}"#,
                args.num_queries,
                args.nprobe,
                path.display(),
            );
            std::fs::write(result_path, result).expect("write trace-export result");
        }
        return;
    }

    // Warm up with the same query concurrency as the measured workload.
    // A sequential warmup only initializes one-request-at-a-time connection
    // state. The first measured concurrent burst then pays connection-pool
    // expansion and can create a multi-second tail that follows whichever
    // method happens to run first in a cell, rather than the method itself.
    eprintln!("[full_pipeline] warmup {}...", args.warmup_queries);
    let warmup_sem = Arc::new(Semaphore::new(args.query_concurrency));
    let mut warmup_handles = Vec::with_capacity(args.warmup_queries);
    for query in queries.iter().take(args.warmup_queries) {
        let query = query.clone();
        let ds = ds.clone();
        let sem = warmup_sem.clone();
        let k = args.k;
        let nprobe = args.nprobe;
        let refine_factor = args.refine_factor;
        warmup_handles.push(tokio::spawn(async move {
            let _permit = sem.acquire_owned().await.unwrap();
            do_one_query(&ds, &query, k, nprobe, refine_factor).await
        }));
    }
    for handle in warmup_handles {
        let _ = handle.await.unwrap();
    }

    // Measured: Q concurrent queries via Semaphore (or strictly sequential
    // when --per-query-cache-jsonl needs clean per-query cache attribution),
    // repeated --measured-passes times against the SAME ds/Session so the
    // index cache carries over between passes.
    let mut pass_results: Vec<String> = Vec::with_capacity(args.measured_passes);
    let mut per_query_cache_records: Vec<String> = Vec::new();
    let mut per_query_latency_records: Vec<String> =
        Vec::with_capacity(args.measured_passes * args.num_queries);
    for pass in 0..args.measured_passes {
        reset_peak_iops_in_flight();
        reset_iop_latency_stats();
        let iops_before = iops_counter();
        let bytes_before = bytes_read_counter();
        let t0 = Instant::now();
        let mut lats: Vec<f64> = if let Some(_jsonl_path) = args.per_query_cache_jsonl.as_ref() {
            let mut lats = Vec::with_capacity(args.num_queries);
            for i in 0..args.num_queries {
                let before = ds.session().index_cache_stats().await;
                let lat = do_one_query(
                    &ds,
                    &queries[measured_query_offset + i],
                    args.k,
                    args.nprobe,
                    args.refine_factor,
                )
                .await;
                let after = ds.session().index_cache_stats().await;
                let miss_delta = after.misses.saturating_sub(before.misses);
                let hit_delta = after.hits.saturating_sub(before.hits);
                per_query_cache_records.push(format!(
                    r#"{{"query_index":{},"latency_ms":{:.3},"cache_hits":{},"cache_misses":{},"full_hit":{}}}"#,
                    i, lat, hit_delta, miss_delta, miss_delta == 0,
                ));
                lats.push(lat);
            }
            lats
        } else {
            let sem = Arc::new(Semaphore::new(args.query_concurrency));
            let mut handles = Vec::with_capacity(args.num_queries);
            for i in 0..args.num_queries {
                let qv = queries[measured_query_offset + i].clone();
                let ds = ds.clone();
                let sem = sem.clone();
                let k = args.k;
                let nprobe = args.nprobe;
                let rf = args.refine_factor;
                handles.push(tokio::spawn(async move {
                    let _permit = sem.acquire_owned().await.unwrap();
                    do_one_query(&ds, &qv, k, nprobe, rf).await
                }));
            }
            let mut lats = Vec::with_capacity(handles.len());
            for h in handles {
                lats.push(h.await.unwrap());
            }
            lats
        };
        let wall_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let iops_after = iops_counter();
        let bytes_after = bytes_read_counter();
        let peak_in_flight_iops = peak_iops_in_flight();
        let slow_iops_over_1s = iops_over_one_second();
        let slow_iops_over_5s = iops_over_five_seconds();
        let max_iop_latency_ms = max_iop_latency_us() as f64 / 1000.0;
        let quota_waits_over_1s = quota_waits_over_one_second();
        let quota_waits_over_5s = quota_waits_over_five_seconds();
        let max_quota_wait_ms = max_quota_wait_us() as f64 / 1000.0;
        let queue_waits_over_1s = queue_waits_over_one_second();
        let queue_waits_over_5s = queue_waits_over_five_seconds();
        let max_queue_wait_ms = max_queue_wait_us() as f64 / 1000.0;
        let iops_per_q = (iops_after - iops_before) as f64 / args.num_queries as f64;
        let bytes_per_q = (bytes_after - bytes_before) as f64 / args.num_queries as f64;
        let final_cache_stats = ds.session().index_cache_stats().await;

        if args.out_latencies_jsonl.is_some() {
            per_query_latency_records.extend(lats.iter().enumerate().map(|(query_index, lat)| {
                format!(
                    r#"{{"pass":{},"query_index":{},"latency_ms":{:.6}}}"#,
                    pass, query_index, lat,
                )
            }));
        }
        lats.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = lats.len();
        let p50 = lats[n / 2];
        let p95 = lats[((n as f64 * 0.95) as usize).min(n - 1)];
        let p99 = lats[((n as f64 * 0.99) as usize).min(n - 1)];
        let qps = args.num_queries as f64 * 1000.0 / wall_ms;

        let result = format!(
            r#"{{"pass":{},"qps":{:.3},"p50_ms":{:.1},"p95_ms":{:.1},"p99_ms":{:.1},"iops_per_q":{:.1},"bytes_per_q":{:.1},"peak_in_flight_iops":{},"slow_iops_over_1s":{},"slow_iops_over_5s":{},"max_iop_latency_ms":{:.3},"quota_waits_over_1s":{},"quota_waits_over_5s":{},"max_quota_wait_ms":{:.3},"queue_waits_over_1s":{},"queue_waits_over_5s":{},"max_queue_wait_ms":{:.3},"num_q":{},"elapsed_s":{:.2},"Q":{},"R":{},"cache_hits":{},"cache_misses":{},"cache_hit_ratio":{:.4},"cache_entries":{},"cache_size_bytes":{}}}"#,
            pass,
            qps,
            p50,
            p95,
            p99,
            iops_per_q,
            bytes_per_q,
            peak_in_flight_iops,
            slow_iops_over_1s,
            slow_iops_over_5s,
            max_iop_latency_ms,
            quota_waits_over_1s,
            quota_waits_over_5s,
            max_quota_wait_ms,
            queue_waits_over_1s,
            queue_waits_over_5s,
            max_queue_wait_ms,
            args.num_queries,
            wall_ms / 1000.0,
            args.query_concurrency,
            std::env::var("LANCE_PROCESS_IO_THREADS_LIMIT").unwrap_or_else(|_| "default".into()),
            final_cache_stats.hits,
            final_cache_stats.misses,
            final_cache_stats.hit_ratio(),
            final_cache_stats.num_entries,
            final_cache_stats.size_bytes,
        );
        println!("{}", result);
        pass_results.push(result);
    }
    if let Some(path) = args.out_json {
        if pass_results.len() == 1 {
            std::fs::write(&path, &pass_results[0]).expect("write out-json");
        } else {
            let joined = format!("[{}]", pass_results.join(","));
            std::fs::write(&path, joined).expect("write out-json");
        }
    }
    if let Some(path) = args.per_query_cache_jsonl {
        std::fs::write(&path, per_query_cache_records.join("\n") + "\n")
            .expect("write per-query-cache-jsonl");
    }
    if let Some(path) = args.out_latencies_jsonl {
        std::fs::write(&path, per_query_latency_records.join("\n") + "\n")
            .expect("write out-latencies-jsonl");
    }
}
