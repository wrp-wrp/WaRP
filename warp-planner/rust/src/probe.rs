//! Object-store-backed cost-model calibration.
//!
//! Times range-GETs of various sizes against a real object store and
//! feeds the samples into a [`Calibrator`]. Works against any backend
//! supported by the `object_store` crate — S3, GCS, Azure, local file,
//! in-memory — so unit tests can exercise the probe loop without
//! touching the network.
//!
//! Gated behind the `probe` cargo feature so the core crate stays
//! free of async / HTTP dependencies.
//!
//! # Example
//!
//! ```no_run
//! use std::sync::Arc;
//! use object_store::{memory::InMemory, path::Path, ObjectStore};
//! use warp_planner::probe::{probe_object_store, ProbePlan};
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! // Swap `InMemory::new()` for `AmazonS3Builder::from_env().build()?`
//! // (enable object_store's `aws` feature) to point at real S3.
//! let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
//! let plan = ProbePlan::default();
//! let probe_path = Path::from("calibration/probe-object.bin");
//!
//! let cal = probe_object_store(store, &probe_path, &plan).await?;
//! let cost_model = cal.to_cost_model();
//! println!("fitted: {:?}", cost_model);
//! # Ok(()) }
//! ```

use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use bytes::Bytes;
use futures::stream::{FuturesUnordered, StreamExt};
use object_store::{path::Path, ObjectStore};

use crate::calibrate::Calibrator;

/// How the probe loop drives the object store.
#[derive(Clone, Debug)]
pub struct ProbePlan {
    /// Range sizes to probe, in bytes. One sample is taken per
    /// `(size, repeat)` pair.
    pub sizes_bytes: Vec<usize>,
    /// Number of repeats per size. Median of the repeats is kept as
    /// the sample — protects against one-off network jitter. Set to
    /// `1` to keep every sample.
    pub repeats: usize,
    /// Maximum concurrent in-flight range-GETs. `1` is sequential.
    /// Values above 1 are useful when probing large sizes that would
    /// otherwise dominate wall-clock time.
    pub concurrency: usize,
    /// Drop the first `warmup` repeats per size from the median
    /// computation — lets the connection pool warm up.
    pub warmup: usize,
}

impl Default for ProbePlan {
    /// A reasonable default sweep covering the small-request regime
    /// and crossing the ~3 MiB break point.
    fn default() -> Self {
        Self {
            sizes_bytes: vec![
                4 * 1024,
                16 * 1024,
                64 * 1024,
                256 * 1024,
                1024 * 1024,
                3 * 1024 * 1024,
                6 * 1024 * 1024,
                12 * 1024 * 1024,
                24 * 1024 * 1024,
            ],
            repeats: 5,
            concurrency: 1,
            warmup: 1,
        }
    }
}

/// Probe `object` on `store`, issuing ranged GETs per `plan`, and
/// return a [`Calibrator`] holding the observed samples.
///
/// Returns an error if any GET fails. Probe sizes larger than the
/// object are skipped, so callers can reuse one sweep across objects
/// of different sizes.
pub async fn probe_object_store(
    store: Arc<dyn ObjectStore>,
    object: &Path,
    plan: &ProbePlan,
) -> object_store::Result<Calibrator> {
    let head = store.head(object).await?;
    let object_size: usize = head.size;

    let mut samples_per_size: Vec<(usize, Vec<f64>)> =
        plan.sizes_bytes.iter().map(|&b| (b, Vec::new())).collect();

    for (size, latencies) in samples_per_size.iter_mut() {
        if *size == 0 || *size > object_size {
            continue;
        }
        if plan.concurrency <= 1 {
            for rep in 0..plan.repeats {
                let range = range_for_repeat(object_size, *size, rep);
                let ms = time_get_range(&*store, object, range).await?;
                latencies.push(ms);
            }
        } else {
            let mut futs = FuturesUnordered::new();
            for rep in 0..plan.repeats {
                let range = range_for_repeat(object_size, *size, rep);
                let store = store.clone();
                let object = object.clone();
                futs.push(async move { time_get_range(&*store, &object, range).await });
                if futs.len() >= plan.concurrency {
                    if let Some(r) = futs.next().await {
                        latencies.push(r?);
                    }
                }
            }
            while let Some(r) = futs.next().await {
                latencies.push(r?);
            }
        }
    }

    let mut cal = Calibrator::new();
    for (size, mut latencies) in samples_per_size {
        if latencies.is_empty() {
            continue;
        }
        // Drop warmup from the left.
        let drop = plan.warmup.min(latencies.len().saturating_sub(1));
        let kept: Vec<f64> = latencies.split_off(drop);
        let med = median(kept);
        cal.record(size as i64, med);
    }
    Ok(cal)
}

/// Return a single sample from one timed GET. Useful when callers want
/// full control over the probing schedule.
pub async fn time_get_range(
    store: &dyn ObjectStore,
    object: &Path,
    range: Range<usize>,
) -> object_store::Result<f64> {
    let start = Instant::now();
    let _bytes: Bytes = store.get_range(object, range).await?;
    Ok(start.elapsed().as_secs_f64() * 1_000.0)
}

fn range_for_repeat(object_size: usize, window: usize, rep: usize) -> Range<usize> {
    // Stride through the object so repeats hit different bytes (mitigates
    // single-byte caching effects); wrap around if the object is smaller
    // than `rep * window`.
    if window == 0 {
        return 0..0;
    }
    let max_start = object_size.saturating_sub(window);
    if max_start == 0 {
        return 0..window.min(object_size);
    }
    let stride = (window / 2).max(1);
    let offset = (rep.saturating_mul(stride)) % (max_start + 1);
    offset..(offset + window)
}

fn median(mut xs: Vec<f64>) -> f64 {
    xs.sort_by(|a, b| a.total_cmp(b));
    let n = xs.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        xs[n / 2]
    } else {
        0.5 * (xs[n / 2 - 1] + xs[n / 2])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use object_store::memory::InMemory;
    use object_store::PutPayload;

    async fn write_fixture(store: &dyn ObjectStore, path: &Path, size: usize) {
        let buf: Vec<u8> = (0..size).map(|i| (i % 251) as u8).collect();
        store.put(path, PutPayload::from(buf)).await.unwrap();
    }

    #[tokio::test]
    async fn probe_runs_against_in_memory_store() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("probe-fixture.bin");
        write_fixture(&*store, &path, 8 * 1024 * 1024).await;

        let plan = ProbePlan {
            sizes_bytes: vec![4096, 65536, 524288, 2 * 1024 * 1024],
            repeats: 3,
            concurrency: 1,
            warmup: 1,
        };
        let cal = probe_object_store(store, &path, &plan).await.unwrap();
        assert_eq!(cal.sample_count(), plan.sizes_bytes.len());
        let cm = cal.to_cost_model();
        // In-memory store has ~zero latency for both overhead and rate,
        // so both coefficients should be non-negative and finite.
        assert!(cm.latency_per_request.is_finite());
        assert!(cm.latency_per_byte.is_finite());
        assert!(cm.latency_per_request >= 0.0);
        assert!(cm.latency_per_byte >= 0.0);
    }

    #[tokio::test]
    async fn probe_skips_sizes_larger_than_object() {
        let store: Arc<dyn ObjectStore> = Arc::new(InMemory::new());
        let path = Path::from("tiny.bin");
        write_fixture(&*store, &path, 16 * 1024).await;

        let plan = ProbePlan {
            sizes_bytes: vec![4096, 32 * 1024, 1024 * 1024],
            repeats: 2,
            concurrency: 1,
            warmup: 0,
        };
        let cal = probe_object_store(store, &path, &plan).await.unwrap();
        // Only the 4 KiB size fits — others should be silently skipped.
        assert_eq!(cal.sample_count(), 1);
    }
}
