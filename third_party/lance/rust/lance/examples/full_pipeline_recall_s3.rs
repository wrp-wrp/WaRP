// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Export per-query result IDs for full-set Recall@k evaluation on Lance/S3.
//!
//! This runner uses the same `Dataset::scan().nearest()` path as
//! `full_pipeline_real_s3`, but preserves the result IDs for every query.

use std::fs::File;
use std::io::Read;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use arrow_array::{Float32Array, Int64Array};
use futures::TryStreamExt;
use lance::dataset::builder::DatasetBuilder;
use lance::Dataset;
use tokio::sync::Semaphore;

#[derive(Debug)]
struct Args {
    uri: String,
    queries_npy: PathBuf,
    out_jsonl: PathBuf,
    num_queries: usize,
    warmup_queries: usize,
    query_concurrency: usize,
    nprobe: usize,
    k: usize,
    refine_factor: u32,
}

fn parse_args() -> Args {
    let mut uri = None;
    let mut queries_npy = None;
    let mut out_jsonl = None;
    let mut num_queries = 10_000;
    let mut warmup_queries = 20;
    let mut query_concurrency = 4;
    let mut nprobe = 128;
    let mut k = 10;
    let mut refine_factor = 1;

    let argv: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--uri" => {
                uri = Some(argv[i + 1].clone());
                i += 2;
            }
            "--queries-npy" => {
                queries_npy = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--out-jsonl" => {
                out_jsonl = Some(PathBuf::from(&argv[i + 1]));
                i += 2;
            }
            "--num-queries" => {
                num_queries = argv[i + 1].parse().expect("valid --num-queries");
                i += 2;
            }
            "--warmup-queries" => {
                warmup_queries = argv[i + 1].parse().expect("valid --warmup-queries");
                i += 2;
            }
            "--query-concurrency" => {
                query_concurrency = argv[i + 1].parse().expect("valid --query-concurrency");
                i += 2;
            }
            "--nprobe" => {
                nprobe = argv[i + 1].parse().expect("valid --nprobe");
                i += 2;
            }
            "--k" => {
                k = argv[i + 1].parse().expect("valid --k");
                i += 2;
            }
            "--refine-factor" => {
                refine_factor = argv[i + 1].parse().expect("valid --refine-factor");
                i += 2;
            }
            flag => panic!("unknown flag {flag}"),
        }
    }

    Args {
        uri: uri.expect("--uri is required"),
        queries_npy: queries_npy.expect("--queries-npy is required"),
        out_jsonl: out_jsonl.expect("--out-jsonl is required"),
        num_queries,
        warmup_queries,
        query_concurrency,
        nprobe,
        k,
        refine_factor,
    }
}

fn load_queries_f32(path: &std::path::Path) -> Vec<Vec<f32>> {
    let mut file = File::open(path).expect("open queries.npy");
    let mut magic = [0_u8; 6];
    file.read_exact(&mut magic).expect("read npy magic");
    assert_eq!(&magic, b"\x93NUMPY");

    let mut version = [0_u8; 2];
    file.read_exact(&mut version).expect("read npy version");
    let header_len = if version[0] == 1 {
        let mut bytes = [0_u8; 2];
        file.read_exact(&mut bytes)
            .expect("read npy v1 header length");
        u16::from_le_bytes(bytes) as usize
    } else {
        let mut bytes = [0_u8; 4];
        file.read_exact(&mut bytes).expect("read npy header length");
        u32::from_le_bytes(bytes) as usize
    };

    let mut header_bytes = vec![0_u8; header_len];
    file.read_exact(&mut header_bytes).expect("read npy header");
    let header = std::str::from_utf8(&header_bytes).expect("utf8 npy header");
    assert!(
        header.contains("<f4") || header.contains("f4"),
        "only float32 queries are supported: {header}"
    );
    assert!(
        header.contains("'fortran_order': False") || header.contains("\"fortran_order\": False"),
        "only C-order queries are supported: {header}"
    );

    let shape = &header[header.find("shape").expect("npy shape")..];
    let open = shape.find('(').expect("npy shape open");
    let close = shape.find(')').expect("npy shape close");
    let dimensions: Vec<usize> = shape[open + 1..close]
        .split(',')
        .map(str::trim)
        .filter(|part| !part.is_empty())
        .map(|part| part.parse().expect("numeric npy dimension"))
        .collect();
    assert_eq!(dimensions.len(), 2, "expected a 2D query array");
    let (rows, columns) = (dimensions[0], dimensions[1]);

    let mut bytes = vec![0_u8; rows * columns * 4];
    file.read_exact(&mut bytes).expect("read npy data");
    bytes
        .chunks_exact(columns * 4)
        .map(|row| {
            row.chunks_exact(4)
                .map(|value| f32::from_le_bytes(value.try_into().unwrap()))
                .collect()
        })
        .collect()
}

async fn query_ids(
    dataset: &Dataset,
    query: &[f32],
    k: usize,
    nprobe: usize,
    refine_factor: u32,
) -> (f64, Vec<i64>) {
    let query = Float32Array::from(query.to_vec());
    let mut scanner = dataset.scan();
    scanner.nearest("vector", &query, k).expect("nearest");
    scanner.nprobes(nprobe);
    if refine_factor > 1 {
        scanner.refine(refine_factor);
    }
    scanner.project(&["id"]).expect("project id");

    let started = Instant::now();
    let stream = scanner.try_into_stream().await.expect("query stream");
    let batches: Vec<_> = stream.try_collect().await.expect("query batches");
    let latency_ms = started.elapsed().as_secs_f64() * 1000.0;

    let mut ids = Vec::with_capacity(k);
    for batch in batches {
        let array = batch
            .column_by_name("id")
            .expect("result id column")
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("int64 result id");
        ids.extend(array.values().iter().copied());
    }
    (latency_ms, ids)
}

#[tokio::main]
async fn main() {
    let args = parse_args();
    let batch_execution = std::env::var_os("LANCE_IVF_BATCH_PREFETCH").is_some();
    let stock_execution = std::env::var_os("LANCE_IVF_STOCK_EXECUTION").is_some();
    assert!(
        batch_execution ^ stock_execution,
        "full_pipeline_recall_s3 requires exactly one execution path: \
         LANCE_IVF_BATCH_PREFETCH=1 or LANCE_IVF_STOCK_EXECUTION=1"
    );
    let dataset = Arc::new(
        DatasetBuilder::from_uri(&args.uri)
            .load()
            .await
            .expect("open dataset"),
    );
    let queries = load_queries_f32(&args.queries_npy);
    assert!(
        queries.len() >= args.num_queries,
        "query file has {} rows, need {}",
        queries.len(),
        args.num_queries
    );

    for query in queries.iter().take(args.warmup_queries) {
        let _ = query_ids(&dataset, query, args.k, args.nprobe, args.refine_factor).await;
    }

    let started = Instant::now();
    let semaphore = Arc::new(Semaphore::new(args.query_concurrency));
    let mut handles = Vec::with_capacity(args.num_queries);
    for (query_index, query) in queries.into_iter().take(args.num_queries).enumerate() {
        let dataset = dataset.clone();
        let semaphore = semaphore.clone();
        let k = args.k;
        let nprobe = args.nprobe;
        let refine_factor = args.refine_factor;
        handles.push(tokio::spawn(async move {
            let _permit = semaphore.acquire_owned().await.expect("query permit");
            let (latency_ms, ids) = query_ids(&dataset, &query, k, nprobe, refine_factor).await;
            (query_index, latency_ms, ids)
        }));
    }

    let mut records = Vec::with_capacity(args.num_queries);
    for handle in handles {
        records.push(handle.await.expect("query task"));
    }
    records.sort_unstable_by_key(|record| record.0);

    let mut output = String::new();
    for (query_index, latency_ms, ids) in &records {
        let ids = ids
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(",");
        output.push_str(&format!(
            "{{\"query_index\":{query_index},\"latency_ms\":{latency_ms:.3},\"ids\":[{ids}]}}\n"
        ));
    }
    std::fs::write(&args.out_jsonl, output).expect("write result IDs");

    let elapsed = started.elapsed().as_secs_f64();
    println!(
        "{{\"num_queries\":{},\"nprobe\":{},\"k\":{},\"refine_factor\":{},\"query_concurrency\":{},\"elapsed_s\":{:.3},\"qps\":{:.3},\"out_jsonl\":\"{}\"}}",
        args.num_queries,
        args.nprobe,
        args.k,
        args.refine_factor,
        args.query_concurrency,
        elapsed,
        args.num_queries as f64 / elapsed,
        args.out_jsonl.display(),
    );
}
