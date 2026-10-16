// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

use bytes::Bytes;
use futures::channel::oneshot;
use futures::{FutureExt, TryFutureExt};
use object_store::path::Path;
use snafu::location;
use std::collections::{BinaryHeap, HashMap};
use std::fmt::Debug;
use std::future::Future;
use std::num::NonZero;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::{Notify, Semaphore, SemaphorePermit};

use lance_core::{Error, Result};

use crate::object_store::ObjectStore;
use crate::scheduler::merge_strategy::SeparateStrategy;
use crate::scheduler::{
    BalancedUBMerge, ConcurrentMerge, CostModel, GreedyFrontierMerge, GreedyMerge,
    GreedyMergeCapped, LanceMerge, MergeStrategy, SerialDP,
};
use crate::traits::Reader;
use crate::utils::CachedFileSize;

// Don't log backpressure warnings until at least this many seconds have passed
const BACKPRESSURE_MIN: u64 = 5;
// Don't log backpressure warnings more than once / minute
const BACKPRESSURE_DEBOUNCE: u64 = 60;

// Global counter of how many IOPS we have issued
static IOPS_COUNTER: AtomicU64 = AtomicU64::new(0);
// Global counter of how many bytes were read by the scheduler
static BYTES_READ_COUNTER: AtomicU64 = AtomicU64::new(0);
// Process-wide diagnostic counters for the current and peak number of
// scheduler-issued I/O requests. The peak can be reset at a quiescent
// measurement boundary without changing scheduler admission behavior.
static IOPS_IN_FLIGHT: AtomicU64 = AtomicU64::new(0);
static PEAK_IOPS_IN_FLIGHT: AtomicU64 = AtomicU64::new(0);
static IOPS_OVER_ONE_SECOND: AtomicU64 = AtomicU64::new(0);
static IOPS_OVER_FIVE_SECONDS: AtomicU64 = AtomicU64::new(0);
static MAX_IOP_LATENCY_US: AtomicU64 = AtomicU64::new(0);
static QUOTA_WAITS_OVER_ONE_SECOND: AtomicU64 = AtomicU64::new(0);
static QUOTA_WAITS_OVER_FIVE_SECONDS: AtomicU64 = AtomicU64::new(0);
static MAX_QUOTA_WAIT_US: AtomicU64 = AtomicU64::new(0);
static QUEUE_WAITS_OVER_ONE_SECOND: AtomicU64 = AtomicU64::new(0);
static QUEUE_WAITS_OVER_FIVE_SECONDS: AtomicU64 = AtomicU64::new(0);
static MAX_QUEUE_WAIT_US: AtomicU64 = AtomicU64::new(0);

#[derive(Default)]
struct PayloadRangeTrace {
    index_uuids: Vec<String>,
    ranges: Vec<Range<u64>>,
}

// The paper trace exporter runs queries sequentially.  Keeping the recorder
// process-local avoids adding instrumentation to every scheduler API while
// still capturing the exact raw ranges submitted by the IVF payload reader.
static PAYLOAD_RANGE_TRACE: std::sync::LazyLock<Mutex<PayloadRangeTrace>> =
    std::sync::LazyLock::new(|| Mutex::new(PayloadRangeTrace::default()));

/// Clear the payload-range recorder before one trace-export query.
#[doc(hidden)]
pub fn reset_payload_range_trace() {
    *PAYLOAD_RANGE_TRACE.lock().unwrap() = PayloadRangeTrace::default();
}

/// Record the physical vector-index UUID selected for a trace-export query.
#[doc(hidden)]
pub fn record_payload_index_uuid(index_uuid: &str) {
    if std::env::var_os("LANCE_PAYLOAD_PLAN_ONLY").is_none() {
        return;
    }
    let mut trace = PAYLOAD_RANGE_TRACE.lock().unwrap();
    if !trace.index_uuids.iter().any(|value| value == index_uuid) {
        trace.index_uuids.push(index_uuid.to_owned());
    }
}

/// Return and clear the UUIDs and raw auxiliary-file ranges for one query.
#[doc(hidden)]
pub fn take_payload_range_trace() -> (Vec<String>, Vec<Range<u64>>) {
    let mut trace = PAYLOAD_RANGE_TRACE.lock().unwrap();
    trace.index_uuids.sort();
    trace.ranges.sort_by_key(|range| (range.start, range.end));
    trace.ranges.dedup();
    let completed = std::mem::take(&mut *trace);
    (completed.index_uuids, completed.ranges)
}
// By default, we limit the number of IOPS across the entire process to 128
//
// In theory this is enough for ~10GBps on S3 following the guidelines to issue
// 1 IOP per 80MBps.  In practice, I have noticed slightly better performance going
// up to 256.
//
// However, non-S3 stores (e.g. GCS, Azure) can suffer significantly from too many
// concurrent IOPS.  For safety, we set the default to 128 and let the user override
// this if needed.
//
// Note: this only limits things that run through the scheduler.  It does not limit
// IOPS from other sources like writing or commits.
static DEFAULT_PROCESS_IOPS_LIMIT: i32 = 128;

pub fn iops_counter() -> u64 {
    IOPS_COUNTER.load(Ordering::Acquire)
}

pub fn bytes_read_counter() -> u64 {
    BYTES_READ_COUNTER.load(Ordering::Acquire)
}

pub fn peak_iops_in_flight() -> u64 {
    PEAK_IOPS_IN_FLIGHT.load(Ordering::Acquire)
}

pub fn reset_peak_iops_in_flight() {
    let current = IOPS_IN_FLIGHT.load(Ordering::Acquire);
    PEAK_IOPS_IN_FLIGHT.store(current, Ordering::Release);
}

pub fn reset_iop_latency_stats() {
    IOPS_OVER_ONE_SECOND.store(0, Ordering::Release);
    IOPS_OVER_FIVE_SECONDS.store(0, Ordering::Release);
    MAX_IOP_LATENCY_US.store(0, Ordering::Release);
    QUOTA_WAITS_OVER_ONE_SECOND.store(0, Ordering::Release);
    QUOTA_WAITS_OVER_FIVE_SECONDS.store(0, Ordering::Release);
    MAX_QUOTA_WAIT_US.store(0, Ordering::Release);
    QUEUE_WAITS_OVER_ONE_SECOND.store(0, Ordering::Release);
    QUEUE_WAITS_OVER_FIVE_SECONDS.store(0, Ordering::Release);
    MAX_QUEUE_WAIT_US.store(0, Ordering::Release);
}

pub fn iops_over_one_second() -> u64 {
    IOPS_OVER_ONE_SECOND.load(Ordering::Acquire)
}

pub fn iops_over_five_seconds() -> u64 {
    IOPS_OVER_FIVE_SECONDS.load(Ordering::Acquire)
}

pub fn max_iop_latency_us() -> u64 {
    MAX_IOP_LATENCY_US.load(Ordering::Acquire)
}

pub fn quota_waits_over_one_second() -> u64 {
    QUOTA_WAITS_OVER_ONE_SECOND.load(Ordering::Acquire)
}

pub fn quota_waits_over_five_seconds() -> u64 {
    QUOTA_WAITS_OVER_FIVE_SECONDS.load(Ordering::Acquire)
}

pub fn max_quota_wait_us() -> u64 {
    MAX_QUOTA_WAIT_US.load(Ordering::Acquire)
}

pub fn queue_waits_over_one_second() -> u64 {
    QUEUE_WAITS_OVER_ONE_SECOND.load(Ordering::Acquire)
}

pub fn queue_waits_over_five_seconds() -> u64 {
    QUEUE_WAITS_OVER_FIVE_SECONDS.load(Ordering::Acquire)
}

pub fn max_queue_wait_us() -> u64 {
    MAX_QUEUE_WAIT_US.load(Ordering::Acquire)
}

// There are two structures that control the I/O scheduler concurrency.  First,
// we have a hard limit on the number of IOPS that can be issued concurrently.
// This limit is process-wide.
//
// Second, we try and limit how many I/O requests can be buffered in memory without
// being consumed by a decoder of some kind.  This limit is per-scheduler.  We cannot
// make this limit process wide without introducing deadlock (because the decoder for
// file 0 might be waiting on IOPS blocked by a queue filled with requests for file 1)
// and vice-versa.
//
// There is also a per-scan limit on the number of IOPS that can be issued concurrently.
//
// The process-wide limit exists when users need a hard limit on the number of parallel
// IOPS, e.g. due to port availability limits or to prevent multiple scans from saturating
// the network.  (Note: a process-wide limit of X will not necessarily limit the number of
// open TCP connections to exactly X.  The underlying object store may open more connections
// anyways)
//
// However, it can be too tough in some cases, e.g. when some scans are reading from
// cloud storage and other scans are reading from local disk.  In these cases users don't
// need to set a process-limit and can rely on the per-scan limits.

// The IopsQuota enforces the first of the above limits, it is the per-process hard cap
// on the number of IOPS that can be issued concurrently.
//
// The per-scan limits are enforced by IoQueue
struct IopsQuota {
    // An Option is used here to avoid mutex overhead if no limit is set
    iops_avail: Option<Semaphore>,
}

/// A reservation on the global IOPS quota
///
/// When the reservation is dropped, the IOPS quota is released unless
/// [`Self::forget`] is called.
struct IopsReservation<'a> {
    value: Option<SemaphorePermit<'a>>,
}

impl IopsReservation<'_> {
    // Forget the reservation, so it won't be released on drop
    fn forget(&mut self) {
        if let Some(value) = self.value.take() {
            value.forget();
        }
    }
}

impl IopsQuota {
    // By default, we throttle the number of scan IOPS across the entire process
    //
    // However, the user can disable this by setting the environment variable
    // LANCE_PROCESS_IO_THREADS_LIMIT to zero (or a negative integer).
    fn new() -> Self {
        let initial_capacity = std::env::var("LANCE_PROCESS_IO_THREADS_LIMIT")
            .map(|s| {
                s.parse::<i32>().unwrap_or_else(|_| {
                    log::warn!("Ignoring invalid LANCE_PROCESS_IO_THREADS_LIMIT: {}", s);
                    DEFAULT_PROCESS_IOPS_LIMIT
                })
            })
            .unwrap_or(DEFAULT_PROCESS_IOPS_LIMIT);
        let iops_avail = if initial_capacity <= 0 {
            None
        } else {
            Some(Semaphore::new(initial_capacity as usize))
        };
        Self { iops_avail }
    }

    // Return a reservation on the global IOPS quota
    fn release(&self) {
        if let Some(iops_avail) = self.iops_avail.as_ref() {
            iops_avail.add_permits(1);
        }
    }

    // Acquire a reservation on the global IOPS quota
    async fn acquire(&self) -> IopsReservation<'_> {
        if let Some(iops_avail) = self.iops_avail.as_ref() {
            IopsReservation {
                value: Some(iops_avail.acquire().await.unwrap()),
            }
        } else {
            IopsReservation { value: None }
        }
    }
}

static IOPS_QUOTA: std::sync::LazyLock<IopsQuota> = std::sync::LazyLock::new(IopsQuota::new);

// We want to allow requests that have a lower priority than any
// currently in-flight request.  This helps avoid potential deadlocks
// related to backpressure.  Unfortunately, it is quite expensive to
// keep track of which priorities are in-flight.
//
// TODO: At some point it would be nice if we can optimize this away but
// in_flight should remain relatively small (generally less than 256 items)
// and has not shown itself to be a bottleneck yet.
struct PrioritiesInFlight {
    in_flight: Vec<u128>,
}

impl PrioritiesInFlight {
    fn new(capacity: u32) -> Self {
        Self {
            in_flight: Vec::with_capacity(capacity as usize * 2),
        }
    }

    fn min_in_flight(&self) -> u128 {
        self.in_flight.first().copied().unwrap_or(u128::MAX)
    }

    fn push(&mut self, prio: u128) {
        let pos = match self.in_flight.binary_search(&prio) {
            Ok(pos) => pos,
            Err(pos) => pos,
        };
        self.in_flight.insert(pos, prio);
    }

    fn remove(&mut self, prio: u128) {
        if let Ok(pos) = self.in_flight.binary_search(&prio) {
            self.in_flight.remove(pos);
        }
    }
}

struct IoQueueState {
    // Number of IOPS we can issue concurrently before pausing I/O
    iops_avail: u32,
    // Number of bytes we are allowed to buffer in memory before pausing I/O
    //
    // This can dip below 0 due to I/O prioritization
    bytes_avail: i64,
    // Pending I/O requests
    pending_requests: BinaryHeap<IoTask>,
    // Priorities of in-flight requests
    priorities_in_flight: PrioritiesInFlight,
    // Set when the scheduler is finished to notify the I/O loop to shut down
    // once all outstanding requests have been completed.
    // Monotonic tie-breaker that preserves FIFO order within one priority.
    next_enqueue_order: u64,
    done_scheduling: bool,
    // Time when the scheduler started
    start: Instant,
    // Last time we warned about backpressure
    last_warn: AtomicU64,
}

impl IoQueueState {
    fn new(io_capacity: u32, io_buffer_size: u64) -> Self {
        Self {
            iops_avail: io_capacity,
            bytes_avail: io_buffer_size as i64,
            pending_requests: BinaryHeap::new(),
            priorities_in_flight: PrioritiesInFlight::new(io_capacity),
            next_enqueue_order: 0,
            done_scheduling: false,
            start: Instant::now(),
            last_warn: AtomicU64::from(0),
        }
    }

    fn warn_if_needed(&self) {
        let seconds_elapsed = self.start.elapsed().as_secs();
        let last_warn = self.last_warn.load(Ordering::Acquire);
        let since_last_warn = seconds_elapsed - last_warn;
        if (last_warn == 0
            && seconds_elapsed > BACKPRESSURE_MIN
            && seconds_elapsed < BACKPRESSURE_DEBOUNCE)
            || since_last_warn > BACKPRESSURE_DEBOUNCE
        {
            tracing::event!(tracing::Level::DEBUG, "Backpressure throttle exceeded");
            log::debug!("Backpressure throttle is full, I/O will pause until buffer is drained.  Max I/O bandwidth will not be achieved because CPU is falling behind");
            self.last_warn
                .store(seconds_elapsed.max(1), Ordering::Release);
        }
    }

    fn can_deliver(&self, task: &IoTask) -> bool {
        if self.iops_avail == 0 {
            false
        } else if task.priority <= self.priorities_in_flight.min_in_flight() {
            true
        } else if task.num_bytes() as i64 > self.bytes_avail {
            self.warn_if_needed();
            false
        } else {
            true
        }
    }

    fn next_task(&mut self) -> Option<IoTask> {
        let task = self.pending_requests.peek()?;
        if self.can_deliver(task) {
            self.priorities_in_flight.push(task.priority);
            self.iops_avail -= 1;
            self.bytes_avail -= task.num_bytes() as i64;
            if self.bytes_avail < 0 {
                // This can happen when we admit special priority requests
                log::debug!(
                    "Backpressure throttle temporarily exceeded by {} bytes due to priority I/O",
                    -self.bytes_avail
                );
            }
            Some(self.pending_requests.pop().unwrap())
        } else {
            None
        }
    }
}

// This is modeled after the MPSC queue described here: https://docs.rs/tokio/latest/tokio/sync/struct.Notify.html
//
// However, it only needs to be SPSC since there is only one "scheduler thread"
// and one I/O loop.
struct IoQueue {
    // Queue state
    state: Mutex<IoQueueState>,
    // Used to signal new I/O requests have arrived that might potentially be runnable
    notify: Notify,
}

impl IoQueue {
    fn new(io_capacity: u32, io_buffer_size: u64) -> Self {
        Self {
            state: Mutex::new(IoQueueState::new(io_capacity, io_buffer_size)),
            notify: Notify::new(),
        }
    }

    fn push(&self, mut task: IoTask) {
        log::trace!(
            "Inserting I/O request for {} bytes with priority ({},{}) into I/O queue",
            task.num_bytes(),
            task.priority >> 64,
            task.priority & 0xFFFFFFFFFFFFFFFF
        );
        let mut state = self.state.lock().unwrap();
        task.enqueue_order = state.next_enqueue_order;
        state.next_enqueue_order = state.next_enqueue_order.wrapping_add(1);
        state.pending_requests.push(task);
        drop(state);

        self.notify.notify_one();
    }

    async fn pop(&self) -> Option<IoTask> {
        loop {
            {
                // First, grab a reservation on the global IOPS quota
                // If we then get a task to run, transfer the reservation
                // to the task.  Otherwise, the reservation will be released
                // when iop_res is dropped.
                let quota_wait_started = Instant::now();
                let mut iop_res = IOPS_QUOTA.acquire().await;
                let quota_wait_us = quota_wait_started
                    .elapsed()
                    .as_micros()
                    .min(u64::MAX as u128) as u64;
                MAX_QUOTA_WAIT_US.fetch_max(quota_wait_us, Ordering::AcqRel);
                if quota_wait_us >= 1_000_000 {
                    QUOTA_WAITS_OVER_ONE_SECOND.fetch_add(1, Ordering::AcqRel);
                }
                if quota_wait_us >= 5_000_000 {
                    QUOTA_WAITS_OVER_FIVE_SECONDS.fetch_add(1, Ordering::AcqRel);
                }
                // Next, try and grab a reservation from the queue
                let mut state = self.state.lock().unwrap();
                if let Some(task) = state.next_task() {
                    // Reservation successfully acquired, we will release the global
                    // global reservation after task has run.
                    iop_res.forget();
                    return Some(task);
                }

                if state.done_scheduling {
                    return None;
                }
            }

            self.notify.notified().await;
        }
    }

    fn on_iop_complete(&self) {
        let mut state = self.state.lock().unwrap();
        state.iops_avail += 1;
        drop(state);

        self.notify.notify_one();
    }

    fn on_bytes_consumed(&self, bytes: u64, priority: u128, num_reqs: usize) {
        let mut state = self.state.lock().unwrap();
        state.bytes_avail += bytes as i64;
        for _ in 0..num_reqs {
            state.priorities_in_flight.remove(priority);
        }
        drop(state);

        self.notify.notify_one();
    }

    fn close(&self) {
        let mut state = self.state.lock().unwrap();
        state.done_scheduling = true;
        let pending_requests = std::mem::take(&mut state.pending_requests);
        drop(state);
        for request in pending_requests {
            request.cancel();
        }

        self.notify.notify_one();
    }
}

// There is one instance of MutableBatch shared by all the I/O operations
// that make up a single request.  When all the I/O operations complete
// then the MutableBatch goes out of scope and the batch request is considered
// complete
struct MutableBatch<F: FnOnce(Response) + Send> {
    when_done: Option<F>,
    data_buffers: Vec<Bytes>,
    num_bytes: u64,
    priority: u128,
    num_reqs: usize,
    err: Option<Box<dyn std::error::Error + Send + Sync + 'static>>,
}

impl<F: FnOnce(Response) + Send> MutableBatch<F> {
    fn new(when_done: F, num_data_buffers: u32, priority: u128, num_reqs: usize) -> Self {
        Self {
            when_done: Some(when_done),
            data_buffers: vec![Bytes::default(); num_data_buffers as usize],
            num_bytes: 0,
            priority,
            num_reqs,
            err: None,
        }
    }
}

// Rather than keep track of when all the I/O requests are finished so that we
// can deliver the batch of data we let Rust do that for us.  When all I/O's are
// done then the MutableBatch will go out of scope and we know we have all the
// data.
impl<F: FnOnce(Response) + Send> Drop for MutableBatch<F> {
    fn drop(&mut self) {
        // If we have an error, return that.  Otherwise return the data
        let result = if self.err.is_some() {
            Err(Error::Wrapped {
                error: self.err.take().unwrap(),
                location: location!(),
            })
        } else {
            let mut data = Vec::new();
            std::mem::swap(&mut data, &mut self.data_buffers);
            Ok(data)
        };
        // We don't really care if no one is around to receive it, just let
        // the result go out of scope and get cleaned up
        let response = Response {
            data: result,
            num_bytes: self.num_bytes,
            priority: self.priority,
            num_reqs: self.num_reqs,
        };
        (self.when_done.take().unwrap())(response);
    }
}

struct DataChunk {
    task_idx: usize,
    num_bytes: u64,
    data: Result<Bytes>,
}

trait DataSink: Send {
    fn deliver_data(&mut self, data: DataChunk);
}

impl<F: FnOnce(Response) + Send> DataSink for MutableBatch<F> {
    // Called by worker tasks to add data to the MutableBatch
    fn deliver_data(&mut self, data: DataChunk) {
        self.num_bytes += data.num_bytes;
        match data.data {
            Ok(data_bytes) => {
                self.data_buffers[data.task_idx] = data_bytes;
            }
            Err(err) => {
                // This keeps the original error, if present
                self.err.get_or_insert(Box::new(err));
            }
        }
    }
}

struct IoTask {
    reader: Arc<dyn Reader>,
    to_read: Range<u64>,
    when_done: Box<dyn FnOnce(Result<Bytes>) + Send>,
    priority: u128,
    enqueue_order: u64,
    enqueued_at: Instant,
}

impl Eq for IoTask {}

impl PartialEq for IoTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority && self.enqueue_order == other.enqueue_order
    }
}

impl PartialOrd for IoTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IoTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // This is intentionally inverted.  We want a min-heap
        other
            .priority
            .cmp(&self.priority)
            .then_with(|| other.enqueue_order.cmp(&self.enqueue_order))
    }
}

impl IoTask {
    fn num_bytes(&self) -> u64 {
        self.to_read.end - self.to_read.start
    }
    fn cancel(self) {
        (self.when_done)(Err(Error::Internal {
            message: "Scheduler closed before I/O was completed".to_string(),
            location: location!(),
        }));
    }

    async fn run(self) {
        let queue_wait_us = self.enqueued_at.elapsed().as_micros().min(u64::MAX as u128) as u64;
        MAX_QUEUE_WAIT_US.fetch_max(queue_wait_us, Ordering::AcqRel);
        if queue_wait_us >= 1_000_000 {
            QUEUE_WAITS_OVER_ONE_SECOND.fetch_add(1, Ordering::AcqRel);
        }
        if queue_wait_us >= 5_000_000 {
            QUEUE_WAITS_OVER_FIVE_SECONDS.fetch_add(1, Ordering::AcqRel);
        }
        let file_path = self.reader.path().as_ref();
        let num_bytes = self.num_bytes();
        let bytes = if self.to_read.start == self.to_read.end {
            Ok(Bytes::new())
        } else {
            let bytes_fut = self
                .reader
                .get_range(self.to_read.start as usize..self.to_read.end as usize);
            IOPS_COUNTER.fetch_add(1, Ordering::Release);
            let in_flight = IOPS_IN_FLIGHT.fetch_add(1, Ordering::AcqRel) + 1;
            PEAK_IOPS_IN_FLIGHT.fetch_max(in_flight, Ordering::AcqRel);
            let num_bytes = self.num_bytes();
            let iop_started = Instant::now();
            let result = bytes_fut
                .inspect(move |_| {
                    BYTES_READ_COUNTER.fetch_add(num_bytes, Ordering::Release);
                })
                .await
                .map_err(Error::from);
            let latency_us = iop_started.elapsed().as_micros().min(u64::MAX as u128) as u64;
            MAX_IOP_LATENCY_US.fetch_max(latency_us, Ordering::AcqRel);
            if latency_us >= 1_000_000 {
                IOPS_OVER_ONE_SECOND.fetch_add(1, Ordering::AcqRel);
            }
            if latency_us >= 5_000_000 {
                IOPS_OVER_FIVE_SECONDS.fetch_add(1, Ordering::AcqRel);
            }
            IOPS_IN_FLIGHT.fetch_sub(1, Ordering::AcqRel);
            result
        };
        // Emit per-file I/O trace event only when tracing is enabled
        tracing::trace!(
            file = file_path,
            bytes_read = num_bytes,
            requests = 1,
            range_start = self.to_read.start,
            range_end = self.to_read.end,
            "File I/O completed"
        );
        IOPS_QUOTA.release();
        (self.when_done)(bytes);
    }
}

// Every time a scheduler starts up it launches a task to run the I/O loop.  This loop
// repeats endlessly until the scheduler is destroyed.
async fn run_io_loop(tasks: Arc<IoQueue>) {
    // Pop the first finished task off the queue and submit another until
    // we are done
    loop {
        let next_task = tasks.pop().await;
        match next_task {
            Some(task) => {
                tokio::spawn(task.run());
            }
            None => {
                // The sender has been dropped, we are done
                return;
            }
        }
    }
}

#[derive(Debug)]
struct StatsCollector {
    iops: AtomicU64,
    requests: AtomicU64,
    bytes_read: AtomicU64,
}

impl StatsCollector {
    fn new() -> Self {
        Self {
            iops: AtomicU64::new(0),
            requests: AtomicU64::new(0),
            bytes_read: AtomicU64::new(0),
        }
    }

    fn iops(&self) -> u64 {
        self.iops.load(Ordering::Relaxed)
    }

    fn bytes_read(&self) -> u64 {
        self.bytes_read.load(Ordering::Relaxed)
    }

    fn requests(&self) -> u64 {
        self.requests.load(Ordering::Relaxed)
    }

    fn record_request(&self, request: &[Range<u64>]) {
        self.requests.fetch_add(1, Ordering::Relaxed);
        self.iops.fetch_add(request.len() as u64, Ordering::Relaxed);
        self.bytes_read.fetch_add(
            request.iter().map(|r| r.end - r.start).sum::<u64>(),
            Ordering::Relaxed,
        );
    }
}

pub struct ScanStats {
    pub iops: u64,
    pub requests: u64,
    pub bytes_read: u64,
}

impl ScanStats {
    fn new(stats: &StatsCollector) -> Self {
        Self {
            iops: stats.iops(),
            requests: stats.requests(),
            bytes_read: stats.bytes_read(),
        }
    }
}

/// An I/O scheduler which wraps an ObjectStore and throttles the amount of
/// parallel I/O that can be run.
///
/// The ScanScheduler will cancel any outstanding I/O requests when it is dropped.
/// For this reason it should be kept alive until all I/O has finished.
///
/// Note: The 2.X file readers already do this so this is only a concern if you are
/// using the ScanScheduler directly.
pub struct ScanScheduler {
    object_store: Arc<ObjectStore>,
    io_queue: Arc<IoQueue>,
    stats: Arc<StatsCollector>,
}

impl Debug for ScanScheduler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanScheduler")
            .field("object_store", &self.object_store)
            .finish()
    }
}

struct Response {
    data: Result<Vec<Bytes>>,
    priority: u128,
    num_reqs: usize,
    num_bytes: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct SchedulerConfig {
    /// the # of bytes that can be buffered but not yet requested.
    /// This controls back pressure.  If data is not processed quickly enough then this
    /// buffer will fill up and the I/O loop will pause until the buffer is drained.
    pub io_buffer_size_bytes: u64,
}

impl SchedulerConfig {
    /// Big enough for unit testing
    pub fn default_for_testing() -> Self {
        Self {
            io_buffer_size_bytes: 256 * 1024 * 1024,
        }
    }

    /// Configuration that should generally maximize bandwidth (not trying to save RAM
    /// at all).  We assume a max page size of 32MiB and then allow 32MiB per I/O thread
    pub fn max_bandwidth(store: &ObjectStore) -> Self {
        Self {
            io_buffer_size_bytes: 32 * 1024 * 1024 * store.io_parallelism() as u64,
        }
    }
}

impl ScanScheduler {
    /// Create a new scheduler with the given I/O capacity
    ///
    /// # Arguments
    ///
    /// * object_store - the store to wrap
    /// * config - configuration settings for the scheduler
    pub fn new(object_store: Arc<ObjectStore>, config: SchedulerConfig) -> Arc<Self> {
        let io_capacity = object_store.io_parallelism();
        let io_queue = Arc::new(IoQueue::new(
            io_capacity as u32,
            config.io_buffer_size_bytes,
        ));
        let slf = Arc::new(Self {
            object_store,
            io_queue: io_queue.clone(),
            stats: Arc::new(StatsCollector::new()),
        });
        // Best we can do here is fire and forget.  If the I/O loop is still running when the scheduler is
        // dropped we can't wait for it to finish or we'd block a tokio thread.  We could spawn a blocking task
        // to wait for it to finish but that doesn't seem helpful.
        tokio::task::spawn(async move { run_io_loop(io_queue).await });
        slf
    }

    /// Open a file for reading
    ///
    /// # Arguments
    ///
    /// * path - the path to the file to open
    /// * base_priority - the base priority for I/O requests submitted to this file scheduler
    ///   this will determine the upper 64 bits of priority (the lower 64 bits
    ///   come from `submit_request` and `submit_single`)
    pub async fn open_file_with_priority(
        self: &Arc<Self>,
        path: &Path,
        base_priority: u64,
        file_size_bytes: &CachedFileSize,
    ) -> Result<FileScheduler> {
        let file_size_bytes = if let Some(size) = file_size_bytes.get() {
            u64::from(size)
        } else {
            let size = self.object_store.size(path).await?;
            if let Some(size) = NonZero::new(size) {
                file_size_bytes.set(size);
            }
            size
        };
        let reader = self
            .object_store
            .open_with_size(path, file_size_bytes as usize)
            .await?;
        let block_size = self.object_store.block_size() as u64;
        let max_iop_size = self.object_store.max_iop_size();
        Ok(FileScheduler {
            reader: reader.into(),
            block_size,
            root: self.clone(),
            base_priority,
            max_iop_size,
            merge_planner_policy: MergePlannerPolicy::LanceNative,
        })
    }

    /// Open a file with a default priority of 0
    ///
    /// See [`Self::open_file_with_priority`] for more information on the priority
    pub async fn open_file(
        self: &Arc<Self>,
        path: &Path,
        file_size_bytes: &CachedFileSize,
    ) -> Result<FileScheduler> {
        self.open_file_with_priority(path, 0, file_size_bytes).await
    }

    fn do_submit_request(
        &self,
        reader: Arc<dyn Reader>,
        request: Vec<Range<u64>>,
        tx: oneshot::Sender<Response>,
        priority: u128,
    ) {
        let num_iops = request.len() as u32;

        let when_all_io_done = move |bytes_and_permits| {
            // We don't care if the receiver has given up so discard the result
            let _ = tx.send(bytes_and_permits);
        };

        let dest = Arc::new(Mutex::new(Box::new(MutableBatch::new(
            when_all_io_done,
            num_iops,
            priority,
            request.len(),
        ))));

        for (task_idx, iop) in request.into_iter().enumerate() {
            let dest = dest.clone();
            let io_queue = self.io_queue.clone();
            let num_bytes = iop.end - iop.start;
            let task = IoTask {
                reader: reader.clone(),
                to_read: iop,
                priority,
                enqueue_order: 0,
                enqueued_at: Instant::now(),
                when_done: Box::new(move |data| {
                    io_queue.on_iop_complete();
                    let mut dest = dest.lock().unwrap();
                    let chunk = DataChunk {
                        data,
                        task_idx,
                        num_bytes,
                    };
                    dest.deliver_data(chunk);
                }),
            };
            self.io_queue.push(task);
        }
    }

    fn submit_request(
        &self,
        reader: Arc<dyn Reader>,
        request: Vec<Range<u64>>,
        priority: u128,
    ) -> impl Future<Output = Result<Vec<Bytes>>> + Send {
        let (tx, rx) = oneshot::channel::<Response>();

        self.do_submit_request(reader, request, tx, priority);

        let io_queue = self.io_queue.clone();

        rx.map(move |wrapped_rsp| {
            // Right now, it isn't possible for I/O to be cancelled so a cancel error should
            // not occur
            let rsp = wrapped_rsp.unwrap();
            io_queue.on_bytes_consumed(rsp.num_bytes, rsp.priority, rsp.num_reqs);
            rsp.data
        })
    }

    pub fn stats(&self) -> ScanStats {
        ScanStats::new(self.stats.as_ref())
    }
}

impl Drop for ScanScheduler {
    fn drop(&mut self) {
        // If the user is dropping the ScanScheduler then they _should_ be done with I/O.  This can happen
        // even when I/O is in progress if, for example, the user is dropping a scan mid-read because they found
        // the data they wanted (limit after filter or some other example).
        //
        // Closing the I/O queue will cancel any requests that have not yet been sent to the I/O loop.  However,
        // it will not terminate the I/O loop itself.  This is to help prevent deadlock and ensure that all I/O
        // requests that are submitted will terminate.
        //
        // In theory, this isn't strictly necessary, as callers should drop any task expecting I/O before they
        // drop the scheduler.  In practice, this can be difficult to do, and it is better to spend a little bit
        // of time letting the I/O loop drain so that we can avoid any potential deadlocks.
        self.io_queue.close();
    }
}

/// A throttled file reader
#[derive(Clone, Debug)]
pub struct FileScheduler {
    reader: Arc<dyn Reader>,
    root: Arc<ScanScheduler>,
    block_size: u64,
    base_priority: u64,
    max_iop_size: u64,
    merge_planner_policy: MergePlannerPolicy,
}

fn is_close_together(range1: &Range<u64>, range2: &Range<u64>, block_size: u64) -> bool {
    // Note that range1.end <= range2.start is possible (e.g. when decoding string arrays)
    range2.start <= (range1.end + block_size)
}

fn is_overlapping(range1: &Range<u64>, range2: &Range<u64>) -> bool {
    range1.start < range2.end && range2.start < range1.end
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MergePlannerKind {
    Native,
    Separate,
    Lance,
    Greedy,
    GreedyCapped,
    GreedyFrontier,
    DP,
    Concurrent,
    BalancedUB,
}

/// Selects whether a request uses Lance's built-in coalesce-and-split policy
/// or the merge planner configured for an experiment.
///
/// File schedulers default to [`LanceNative`](Self::LanceNative). IVF cluster
/// payload readers opt into [`Configured`](Self::Configured) explicitly.
/// This keeps supporting I/O (sub-index, metadata, and rerank reads) on the
/// stock Lance policy while exposing the same batched payload ranges to every
/// experimental planner.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MergePlannerPolicy {
    #[default]
    LanceNative,
    Configured,
}

#[derive(Clone, Debug)]
struct MergePlannerConfig {
    kind: MergePlannerKind,
    max_waste_ratio: f64,
    max_merge_span: i64,
    max_requests: i32,
    exact_requests: bool,
    concurrent_parallelism: f64,
    latency_per_request_ms: Option<f64>,
    latency_per_byte_ms: Option<f64>,
    latency_std_ms: Option<f64>,
}

impl MergePlannerConfig {
    fn parse_i64_list(value: &str) -> Option<Vec<i64>> {
        let vals: std::result::Result<Vec<_>, _> = value
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(str::parse::<i64>)
            .collect();
        match vals {
            Ok(v) if !v.is_empty() => Some(v),
            _ => None,
        }
    }

    fn parse_f64_list(value: &str) -> Option<Vec<f64>> {
        let vals: std::result::Result<Vec<_>, _> = value
            .split(',')
            .map(|s| s.trim())
            .filter(|s| !s.is_empty())
            .map(str::parse::<f64>)
            .collect();
        match vals {
            Ok(v) if !v.is_empty() => Some(v),
            _ => None,
        }
    }

    fn parse_bool(value: &str) -> Option<bool> {
        match value.trim().to_ascii_lowercase().as_str() {
            "1" | "true" | "yes" | "y" | "on" => Some(true),
            "0" | "false" | "no" | "n" | "off" => Some(false),
            _ => None,
        }
    }

    fn parse_kind(value: &str) -> Option<MergePlannerKind> {
        match value.trim().to_ascii_lowercase().as_str() {
            "native" | "default" => Some(MergePlannerKind::Native),
            "separate" => Some(MergePlannerKind::Separate),
            "lance" => Some(MergePlannerKind::Lance),
            "greedy" => Some(MergePlannerKind::Greedy),
            "greedy_capped" | "greedy-capped" | "greedycapped" => {
                Some(MergePlannerKind::GreedyCapped)
            }
            "greedy_frontier" | "greedy-frontier" | "greedyfrontier" => {
                Some(MergePlannerKind::GreedyFrontier)
            }
            "dp" | "serial_dp" | "serial-dp" => Some(MergePlannerKind::DP),
            "concurrent" => Some(MergePlannerKind::Concurrent),
            "balanced_ub" | "balanced-ub" | "balancedub" => Some(MergePlannerKind::BalancedUB),
            _ => None,
        }
    }

    fn from_env() -> Self {
        let defaults = CostModel::default();
        let mut cfg = Self {
            kind: MergePlannerKind::Native,
            max_waste_ratio: defaults.max_waste_ratio,
            max_merge_span: defaults.max_merge_span,
            max_requests: defaults.max_requests,
            exact_requests: defaults.exact_requests,
            concurrent_parallelism: defaults.concurrent_parallelism,
            latency_per_request_ms: None,
            latency_per_byte_ms: None,
            latency_std_ms: None,
        };

        if let Ok(raw) = std::env::var("LANCE_IO_MERGE_STRATEGY") {
            if let Some(kind) = Self::parse_kind(&raw) {
                cfg.kind = kind;
            } else {
                log::warn!(
                    "Ignoring invalid LANCE_IO_MERGE_STRATEGY value: {} (using native)",
                    raw
                );
            }
        }

        if let Ok(raw) = std::env::var("LANCE_IO_MERGE_MAX_WASTE_RATIO") {
            match raw.parse::<f64>() {
                Ok(v) if v >= 0.0 => cfg.max_waste_ratio = v,
                Ok(_) | Err(_) => {
                    log::warn!("Ignoring invalid LANCE_IO_MERGE_MAX_WASTE_RATIO: {}", raw)
                }
            }
        }
        if let Ok(raw) = std::env::var("LANCE_IO_MERGE_MAX_SPAN_BYTES") {
            match raw.parse::<i64>() {
                Ok(v) if v >= 0 => cfg.max_merge_span = v,
                Ok(_) | Err(_) => {
                    log::warn!("Ignoring invalid LANCE_IO_MERGE_MAX_SPAN_BYTES: {}", raw)
                }
            }
        }
        if let Ok(raw) = std::env::var("LANCE_IO_MERGE_MAX_REQUESTS") {
            match raw.parse::<i32>() {
                Ok(v) => cfg.max_requests = v,
                Err(_) => log::warn!("Ignoring invalid LANCE_IO_MERGE_MAX_REQUESTS: {}", raw),
            }
        }
        if let Ok(raw) = std::env::var("LANCE_IO_MERGE_EXACT_REQUESTS") {
            match Self::parse_bool(&raw) {
                Some(v) => cfg.exact_requests = v,
                None => log::warn!("Ignoring invalid LANCE_IO_MERGE_EXACT_REQUESTS: {}", raw),
            }
        }
        if let Ok(raw) = std::env::var("LANCE_IO_MERGE_CONCURRENT_PARALLELISM") {
            match raw.parse::<f64>() {
                Ok(v) if v >= 0.0 => cfg.concurrent_parallelism = v,
                Ok(_) | Err(_) => log::warn!(
                    "Ignoring invalid LANCE_IO_MERGE_CONCURRENT_PARALLELISM: {}",
                    raw
                ),
            }
        }
        if let Ok(raw) = std::env::var("LANCE_IO_MERGE_LATENCY_PER_REQUEST_MS") {
            match raw.parse::<f64>() {
                Ok(v) if v >= 0.0 => cfg.latency_per_request_ms = Some(v),
                Ok(_) | Err(_) => log::warn!(
                    "Ignoring invalid LANCE_IO_MERGE_LATENCY_PER_REQUEST_MS: {}",
                    raw
                ),
            }
        }
        if let Ok(raw) = std::env::var("LANCE_IO_MERGE_LATENCY_PER_BYTE_MS") {
            match raw.parse::<f64>() {
                Ok(v) if v >= 0.0 => cfg.latency_per_byte_ms = Some(v),
                Ok(_) | Err(_) => log::warn!(
                    "Ignoring invalid LANCE_IO_MERGE_LATENCY_PER_BYTE_MS: {}",
                    raw
                ),
            }
        }
        if let Ok(raw) = std::env::var("LANCE_IO_MERGE_LATENCY_STD_MS") {
            match raw.parse::<f64>() {
                Ok(v) if v >= 0.0 => cfg.latency_std_ms = Some(v),
                Ok(_) | Err(_) => {
                    log::warn!("Ignoring invalid LANCE_IO_MERGE_LATENCY_STD_MS: {}", raw)
                }
            }
        }
        if let Ok(raw) = std::env::var("LANCE_IO_MERGE_LATENCY_MODEL") {
            match raw.trim().to_ascii_lowercase().as_str() {
                "linear" | "piecewise_linear" => {
                    // stored later in CostModel; no validation needed here
                }
                _ => log::warn!("Ignoring invalid LANCE_IO_MERGE_LATENCY_MODEL: {}", raw),
            }
        }

        if cfg.kind != MergePlannerKind::Native {
            log::info!(
                "I/O merge planner enabled: strategy={:?}, max_requests={}, exact={}, P={}",
                cfg.kind,
                cfg.max_requests,
                cfg.exact_requests,
                cfg.concurrent_parallelism
            );
        }

        cfg
    }
}

static MERGE_PLANNER_CONFIG: std::sync::LazyLock<MergePlannerConfig> =
    std::sync::LazyLock::new(MergePlannerConfig::from_env);

fn split_large_request(req: Range<u64>, max_iop_size: u64, out: &mut Vec<Range<u64>>) {
    if req.is_empty() {
        out.push(req);
        return;
    }

    let num_requests = (req.end - req.start).div_ceil(max_iop_size);
    let bytes_per_request = (req.end - req.start) / num_requests;
    for i in 0..num_requests {
        let start = req.start + i * bytes_per_request;
        let end = if i == num_requests - 1 {
            req.end
        } else {
            start + bytes_per_request
        };
        out.push(start..end);
    }
}

fn legacy_coalesce_and_split(
    request: &[Range<u64>],
    block_size: u64,
    max_iop_size: u64,
) -> Vec<Range<u64>> {
    let mut merged_requests = Vec::with_capacity(request.len());

    if !request.is_empty() {
        let mut curr_interval = request[0].clone();
        for req in request.iter().skip(1) {
            if is_close_together(&curr_interval, req, block_size) {
                curr_interval.end = curr_interval.end.max(req.end);
            } else {
                merged_requests.push(curr_interval);
                curr_interval = req.clone();
            }
        }
        merged_requests.push(curr_interval);
    }

    let mut updated_requests = Vec::with_capacity(merged_requests.len());
    for req in merged_requests {
        split_large_request(req, max_iop_size, &mut updated_requests);
    }
    updated_requests
}

fn strategy_plan_to_ranges_with_config(
    request: &[Range<u64>],
    max_iop_size: u64,
    cfg: &MergePlannerConfig,
) -> Option<Vec<Range<u64>>> {
    if cfg.kind == MergePlannerKind::Native || request.is_empty() {
        return None;
    }

    // FAST PATH: a single range has nothing to coalesce. Skipping the
    // strategy / cost-model construction (and the per-call env var
    // reads below) and returning None lets the caller fall through to
    // `legacy_coalesce_and_split`, whose output for n=1 is identical
    // (just `split_large_request` on the lone range). On the
    // SIFT100M IVF-PQ load-partition path Lance issues ~108 of these
    // single-range calls per query through the bridge; this fast
    // path eliminates ~120 ms / query of CPU overhead.
    if request.len() == 1 {
        return None;
    }

    let mut cluster_ids = Vec::with_capacity(request.len());
    let mut cluster_offsets = HashMap::with_capacity(request.len());
    let mut cluster_sizes = HashMap::with_capacity(request.len());

    for (idx, req) in request.iter().enumerate() {
        if req.end < req.start {
            return None;
        }
        let cid = idx as i32;
        cluster_ids.push(cid);
        cluster_offsets.insert(cid, req.start as i64);
        cluster_sizes.insert(cid, (req.end - req.start) as i64);
    }

    let mut cost_model = CostModel::default();
    cost_model.max_waste_ratio = cfg.max_waste_ratio;
    cost_model.max_merge_span = cfg.max_merge_span;
    cost_model.max_requests = cfg.max_requests;
    cost_model.exact_requests = cfg.exact_requests;
    cost_model.concurrent_parallelism = cfg.concurrent_parallelism;
    if let Some(v) = cfg.latency_per_request_ms {
        cost_model.latency_per_request = v;
    }
    if let Some(v) = cfg.latency_per_byte_ms {
        cost_model.latency_per_byte = v;
    }
    if let Some(v) = cfg.latency_std_ms {
        cost_model.latency_std = v;
    }
    if let Ok(raw) = std::env::var("LANCE_IO_MERGE_LATENCY_MODEL") {
        let lowered = raw.trim().to_ascii_lowercase();
        if lowered == "linear" || lowered == "piecewise_linear" {
            cost_model.latency_model = lowered;
        }
    }
    if let Ok(raw) = std::env::var("LANCE_IO_MERGE_LATENCY_PW_BREAKS_BYTES") {
        if let Some(v) = MergePlannerConfig::parse_i64_list(&raw) {
            cost_model.latency_pw_breaks_bytes = v;
        } else {
            log::warn!(
                "Ignoring invalid LANCE_IO_MERGE_LATENCY_PW_BREAKS_BYTES: {}",
                raw
            );
        }
    }
    if let Ok(raw) = std::env::var("LANCE_IO_MERGE_LATENCY_PW_OVERHEADS_MS") {
        if let Some(v) = MergePlannerConfig::parse_f64_list(&raw) {
            cost_model.latency_pw_overheads_ms = v;
        } else {
            log::warn!(
                "Ignoring invalid LANCE_IO_MERGE_LATENCY_PW_OVERHEADS_MS: {}",
                raw
            );
        }
    }
    if let Ok(raw) = std::env::var("LANCE_IO_MERGE_LATENCY_PW_RATES_MS_PER_BYTE") {
        if let Some(v) = MergePlannerConfig::parse_f64_list(&raw) {
            cost_model.latency_pw_rates_ms_per_byte = v;
        } else {
            log::warn!(
                "Ignoring invalid LANCE_IO_MERGE_LATENCY_PW_RATES_MS_PER_BYTE: {}",
                raw
            );
        }
    }

    let strategy: Box<dyn MergeStrategy> = match cfg.kind {
        MergePlannerKind::Native => return None,
        MergePlannerKind::Separate => Box::new(SeparateStrategy),
        MergePlannerKind::Lance => Box::new(LanceMerge::new()),
        MergePlannerKind::Greedy => Box::new(GreedyMerge::new()),
        MergePlannerKind::GreedyCapped => Box::new(GreedyMergeCapped::new()),
        MergePlannerKind::GreedyFrontier => Box::new(GreedyFrontierMerge::new()),
        MergePlannerKind::DP => Box::new(SerialDP::new()),
        MergePlannerKind::Concurrent => Box::new(ConcurrentMerge::new()),
        MergePlannerKind::BalancedUB => Box::new(BalancedUBMerge::new()),
    };

    let probe_t = std::env::var("WARP_PROBE").is_ok();
    let t_plan_start = if probe_t {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let plan = strategy.generate_plan(&cluster_ids, &cluster_offsets, &cluster_sizes, &cost_model);
    if let Some(t) = t_plan_start {
        let us = t.elapsed().as_micros();
        let pw = cost_model.use_piecewise_linear();
        let waste = cost_model.max_waste_ratio;
        let exact = cost_model.exact_requests;
        let kcap_hint = if cost_model.max_requests > 0 {
            cost_model.max_requests as i64
        } else {
            -1
        };
        eprintln!(
            "[planner_us] strategy={:?} us={} raw={} planned={} piecewise={} exact={} max_waste={:.2} kcap_env={}",
            cfg.kind,
            us,
            cluster_ids.len(),
            plan.requests.len(),
            pw,
            exact,
            waste,
            kcap_hint,
        );
    }
    if plan.requests.is_empty() {
        return None;
    }

    let mut planned: Vec<Range<u64>> = Vec::with_capacity(plan.requests.len());
    for req in plan.requests {
        if req.offset < 0 || req.length < 0 {
            return None;
        }
        let start = req.offset as u64;
        let end = start.saturating_add(req.length as u64);
        planned.push(start..end);
    }
    planned.sort_by_key(|r| r.start);

    let mut updated_requests = Vec::with_capacity(planned.len());
    for req in planned {
        split_large_request(req, max_iop_size, &mut updated_requests);
    }
    Some(updated_requests)
}

fn strategy_plan_to_ranges(request: &[Range<u64>], max_iop_size: u64) -> Option<Vec<Range<u64>>> {
    strategy_plan_to_ranges_with_config(request, max_iop_size, &MERGE_PLANNER_CONFIG)
}

/// Plan and split a request list using the currently configured merge strategy.
///
/// This helper is useful for benchmarking when we want to pre-plan ranges outside
/// the measured critical section while preserving identical planning semantics.
pub fn plan_ranges_for_env_strategy(
    request: &[Range<u64>],
    max_iop_size: u64,
) -> Option<Vec<Range<u64>>> {
    strategy_plan_to_ranges(request, max_iop_size)
}

pub fn resolve_ranges_for_env_strategy(
    request: &[Range<u64>],
    block_size: u64,
    max_iop_size: u64,
) -> Vec<Range<u64>> {
    resolve_ranges_with_config_for_policy(
        request,
        block_size,
        max_iop_size,
        &MERGE_PLANNER_CONFIG,
        MergePlannerPolicy::Configured,
    )
}

fn resolve_ranges_with_config_for_policy(
    request: &[Range<u64>],
    block_size: u64,
    max_iop_size: u64,
    cfg: &MergePlannerConfig,
    policy: MergePlannerPolicy,
) -> Vec<Range<u64>> {
    match policy {
        MergePlannerPolicy::LanceNative => {
            legacy_coalesce_and_split(request, block_size, max_iop_size)
        }
        MergePlannerPolicy::Configured => {
            strategy_plan_to_ranges_with_config(request, max_iop_size, cfg)
                .unwrap_or_else(|| legacy_coalesce_and_split(request, block_size, max_iop_size))
        }
    }
}

impl FileScheduler {
    /// Selects the merge policy for all requests submitted through this file
    /// scheduler, including requests created by decoder-spawned tasks.
    #[doc(hidden)]
    pub fn with_merge_planner_policy(mut self, policy: MergePlannerPolicy) -> Self {
        self.merge_planner_policy = policy;
        self
    }

    fn submit_request_impl(
        &self,
        request: Vec<Range<u64>>,
        updated_requests: Vec<Range<u64>>,
        priority: u64,
    ) -> impl Future<Output = Result<Vec<Bytes>>> + Send {
        let priority = ((self.base_priority as u128) << 64) + priority as u128;
        self.root.stats.record_request(&updated_requests);

        let bytes_vec_fut =
            self.root
                .submit_request(self.reader.clone(), updated_requests.clone(), priority);

        let mut updated_index = 0;
        let mut final_bytes = Vec::with_capacity(request.len());

        async move {
            let bytes_vec = bytes_vec_fut.await?;

            let mut orig_index = 0;
            while (updated_index < updated_requests.len()) && (orig_index < request.len()) {
                let updated_range = &updated_requests[updated_index];
                let orig_range = &request[orig_index];
                let byte_offset = updated_range.start as usize;

                if is_overlapping(updated_range, orig_range) {
                    let start = orig_range.start as usize - byte_offset;
                    if orig_range.end <= updated_range.end {
                        let end = orig_range.end as usize - byte_offset;
                        final_bytes.push(bytes_vec[updated_index].slice(start..end));
                    } else {
                        let orig_size = orig_range.end - orig_range.start;
                        let mut merged_bytes = Vec::with_capacity(orig_size as usize);
                        merged_bytes.extend_from_slice(&bytes_vec[updated_index].slice(start..));
                        let mut copy_offset = merged_bytes.len() as u64;
                        while copy_offset < orig_size {
                            updated_index += 1;
                            let next_range = &updated_requests[updated_index];
                            let bytes_to_take =
                                (orig_size - copy_offset).min(next_range.end - next_range.start);
                            merged_bytes.extend_from_slice(
                                &bytes_vec[updated_index].slice(0..bytes_to_take as usize),
                            );
                            copy_offset += bytes_to_take;
                        }
                        final_bytes.push(Bytes::from(merged_bytes));
                    }
                    orig_index += 1;
                } else {
                    updated_index += 1;
                }
            }

            Ok(final_bytes)
        }
    }

    pub fn preview_request_plan(&self, request: &[Range<u64>]) -> Vec<Range<u64>> {
        resolve_ranges_with_config_for_policy(
            request,
            self.block_size,
            self.max_iop_size,
            &MERGE_PLANNER_CONFIG,
            self.merge_planner_policy,
        )
    }

    pub fn submit_request_with_plan(
        &self,
        request: Vec<Range<u64>>,
        updated_requests: Vec<Range<u64>>,
        priority: u64,
    ) -> impl Future<Output = Result<Vec<Bytes>>> + Send {
        self.submit_request_impl(request, updated_requests, priority)
    }

    /// Submit a batch of I/O requests to the reader
    ///
    /// The requests will be queued in a FIFO manner and, when all requests
    /// have been fulfilled, the returned future will be completed.
    ///
    /// Each request has a given priority.  If the I/O loop is full then requests
    /// will be buffered and requests with the *lowest* priority will be released
    /// from the buffer first.
    ///
    /// Each request has a backpressure ID which controls which backpressure throttle
    /// is applied to the request.  Requests made to the same backpressure throttle
    /// will be throttled together.
    pub fn submit_request(
        &self,
        request: Vec<Range<u64>>,
        priority: u64,
    ) -> impl Future<Output = Result<Vec<Bytes>>> + Send {
        if self.merge_planner_policy == MergePlannerPolicy::Configured
            && std::env::var_os("LANCE_PAYLOAD_PLAN_ONLY").is_some()
        {
            PAYLOAD_RANGE_TRACE
                .lock()
                .unwrap()
                .ranges
                .extend(request.iter().cloned());
        }
        let updated_requests = self.preview_request_plan(&request);
        if std::env::var("WARP_PROBE").is_ok() {
            let path = self.reader.path().as_ref();
            let kind = if path.contains("/auxiliary.idx") {
                "aux"
            } else if path.contains("/index.idx") {
                "idx"
            } else if path.contains("/data/") {
                "data"
            } else {
                "other"
            };
            let first = request.first().map(|r| r.start).unwrap_or(0);
            let last_end = request.last().map(|r| r.end).unwrap_or(0);
            let total_bytes: u64 = request.iter().map(|r| r.end - r.start).sum();
            let planned_bytes: u64 = updated_requests.iter().map(|r| r.end - r.start).sum();
            eprintln!(
                "[warp_probe] file={} policy={:?} raw={} planned={} first_offset={} last_end={} total_bytes={} planned_bytes={}",
                kind,
                self.merge_planner_policy,
                request.len(),
                updated_requests.len(),
                first,
                last_end,
                total_bytes,
                planned_bytes,
            );
            // For small requests that bypass batch merging, capture an origin
            // backtrace so we can identify the call site in Lance.  Enabled via
            // WARP_PROBE=trace to avoid cost on hot paths.
            if std::env::var("WARP_PROBE").ok().as_deref() == Some("trace") && request.len() <= 8 {
                let bt = std::backtrace::Backtrace::force_capture();
                let s = format!("{}", bt);
                // Keep only frames from lance_encoding / lance_file / lance_index / lance
                let mut kept: Vec<&str> = Vec::new();
                for line in s.lines() {
                    let t = line.trim();
                    if t.contains("lance_encoding::")
                        || t.contains("lance_file::")
                        || t.contains("lance_index::")
                        || t.contains("lance::index::")
                    {
                        kept.push(line);
                        if kept.len() >= 10 {
                            break;
                        }
                    }
                }
                eprintln!("[warp_probe_trace] small_submit origin:");
                for line in kept {
                    eprintln!("{}", line);
                }
            }
        }
        self.submit_request_impl(request, updated_requests, priority)
    }

    pub fn with_priority(&self, priority: u64) -> Self {
        Self {
            reader: self.reader.clone(),
            root: self.root.clone(),
            block_size: self.block_size,
            max_iop_size: self.max_iop_size,
            base_priority: priority,
            merge_planner_policy: self.merge_planner_policy,
        }
    }

    /// Submit a single IOP to the reader
    ///
    /// If you have multiple IOPS to perform then [`Self::submit_request`] is going
    /// to be more efficient.
    ///
    /// See [`Self::submit_request`] for more information on the priority and backpressure.
    pub fn submit_single(
        &self,
        range: Range<u64>,
        priority: u64,
    ) -> impl Future<Output = Result<Bytes>> + Send {
        self.submit_request(vec![range], priority)
            .map_ok(|vec_bytes| vec_bytes.into_iter().next().unwrap())
    }

    /// Provides access to the underlying reader
    ///
    /// Do not use this for reading data as it will bypass any I/O scheduling!
    /// This is mainly exposed to allow metadata operations (e.g size, block_size,)
    /// which either aren't IOPS or we don't throttle
    pub fn reader(&self) -> &Arc<dyn Reader> {
        &self.reader
    }
}

#[cfg(test)]
mod tests {
    use std::{
        collections::{HashMap, VecDeque},
        sync::Mutex,
        time::Duration,
    };

    use futures::poll;
    use lance_core::utils::tempfile::TempObjFile;
    use rand::RngCore;

    use object_store::{memory::InMemory, GetRange, ObjectStore as OSObjectStore};
    use tokio::{runtime::Handle, time::timeout};
    use url::Url;

    use crate::{
        object_store::{DEFAULT_DOWNLOAD_RETRY_COUNT, DEFAULT_MAX_IOP_SIZE},
        testing::MockObjectStore,
    };

    use super::*;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    struct EnvVarGuard {
        key: &'static str,
        prev: Option<String>,
    }

    impl EnvVarGuard {
        fn set(key: &'static str, value: &str) -> Self {
            let prev = std::env::var(key).ok();
            unsafe { std::env::set_var(key, value) };
            Self { key, prev }
        }
    }

    impl Drop for EnvVarGuard {
        fn drop(&mut self) {
            if let Some(v) = &self.prev {
                unsafe { std::env::set_var(self.key, v) };
            } else {
                unsafe { std::env::remove_var(self.key) };
            }
        }
    }

    #[test]
    fn payload_range_trace_is_sorted_deduplicated_and_cleared() {
        reset_payload_range_trace();
        {
            let mut trace = PAYLOAD_RANGE_TRACE.lock().unwrap();
            trace.index_uuids = vec!["b".to_string(), "a".to_string()];
            trace.ranges = vec![20..30, 0..10, 20..30];
        }
        let (index_uuids, ranges) = take_payload_range_trace();
        assert_eq!(index_uuids, vec!["a", "b"]);
        assert_eq!(ranges, vec![0..10, 20..30]);
        assert_eq!(take_payload_range_trace(), (Vec::new(), Vec::new()));
    }

    #[test_log::test(tokio::test(flavor = "current_thread"))]
    async fn configured_policy_records_raw_payload_ranges() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _trace_mode = EnvVarGuard::set("LANCE_PAYLOAD_PLAN_ONLY", "1");
        reset_payload_range_trace();

        let path = Path::parse("payload.idx").unwrap();
        let object_store = Arc::new(ObjectStore::memory());
        object_store.put(&path, &[0_u8; 64]).await.unwrap();
        let scheduler = ScanScheduler::new(
            object_store,
            SchedulerConfig {
                io_buffer_size_bytes: 1024,
            },
        );
        let file = scheduler
            .open_file(&path, &CachedFileSize::new(64))
            .await
            .unwrap()
            .with_merge_planner_policy(MergePlannerPolicy::Configured);

        file.submit_request(vec![0..8, 16..24], 0).await.unwrap();
        let (_, ranges) = take_payload_range_trace();
        assert_eq!(ranges, vec![0..8, 16..24]);
    }

    #[test]
    fn test_merge_config_accepts_large_waste_ratio() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _ratio = EnvVarGuard::set("LANCE_IO_MERGE_MAX_WASTE_RATIO", "3.0");
        let cfg = MergePlannerConfig::from_env();
        assert!((cfg.max_waste_ratio - 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_merge_config_rejects_negative_waste_ratio() {
        let _guard = ENV_LOCK.lock().unwrap();
        let _ratio = EnvVarGuard::set("LANCE_IO_MERGE_MAX_WASTE_RATIO", "-0.1");
        let cfg = MergePlannerConfig::from_env();
        assert!((cfg.max_waste_ratio - CostModel::default().max_waste_ratio).abs() < 1e-12);
    }

    #[test]
    fn test_lance_native_policy_ignores_experimental_planner() {
        let requests = vec![0_u64..1024, 2048..3072];
        let block_size = 64 * 1024;
        let max_iop_size = 16 * 1024 * 1024;
        let cfg = MergePlannerConfig {
            kind: MergePlannerKind::Separate,
            max_waste_ratio: 0.0,
            max_merge_span: max_iop_size as i64,
            max_requests: -1,
            exact_requests: false,
            concurrent_parallelism: 1.0,
            latency_per_request_ms: None,
            latency_per_byte_ms: None,
            latency_std_ms: None,
        };

        let configured = resolve_ranges_with_config_for_policy(
            &requests,
            block_size,
            max_iop_size,
            &cfg,
            MergePlannerPolicy::Configured,
        );
        let native = resolve_ranges_with_config_for_policy(
            &requests,
            block_size,
            max_iop_size,
            &cfg,
            MergePlannerPolicy::LanceNative,
        );

        assert_eq!(configured, requests);
        assert_eq!(native, vec![0_u64..3072]);
    }

    #[tokio::test]
    async fn test_file_scheduler_policy_survives_spawn() {
        let tmp_file = TempObjFile::default();
        let object_store = Arc::new(ObjectStore::local());
        object_store.put(&tmp_file, &[0; 16]).await.unwrap();
        let scheduler = ScanScheduler::new(object_store, SchedulerConfig::default_for_testing());
        let file_scheduler = scheduler
            .open_file(&tmp_file, &CachedFileSize::unknown())
            .await
            .unwrap();
        assert_eq!(
            file_scheduler.merge_planner_policy,
            MergePlannerPolicy::LanceNative
        );

        let configured = file_scheduler.with_merge_planner_policy(MergePlannerPolicy::Configured);
        tokio::spawn(async move {
            assert_eq!(
                configured.merge_planner_policy,
                MergePlannerPolicy::Configured
            );
        })
        .await
        .unwrap();
    }

    #[test]
    fn test_single_range_fast_path_returns_none() {
        // n=1 must short-circuit to None so the caller falls through
        // to legacy_coalesce_and_split. Avoids ~1ms of CPU per call
        // (CostModel + env var reads + Box<dyn> dispatch + DP setup).
        let request = vec![1024_u64..3072_u64];
        let cfg = MergePlannerConfig {
            kind: MergePlannerKind::BalancedUB,
            max_waste_ratio: 0.5,
            max_merge_span: 16 * 1024 * 1024,
            max_requests: 3,
            exact_requests: false,
            concurrent_parallelism: 8.0,
            latency_per_request_ms: Some(28.0),
            latency_per_byte_ms: Some(1e-6),
            latency_std_ms: Some(0.0),
        };
        let result = strategy_plan_to_ranges_with_config(&request, 16 * 1024 * 1024, &cfg);
        assert_eq!(
            result, None,
            "single-range input must short-circuit to None"
        );
    }

    #[test]
    fn test_single_range_resolve_equivalent_to_legacy() {
        // For n=1 the *full* resolution path must produce the exact
        // same output as the legacy coalesce-split path. Asserts the
        // fast path is observationally equivalent at the public API.
        let request = vec![4096_u64..(4096 + 8 * 1024)];
        let max_iop_size = 16 * 1024 * 1024;
        let block_size = 4 * 1024;
        let legacy_out = legacy_coalesce_and_split(&request, block_size, max_iop_size);
        let resolve_out = resolve_ranges_for_env_strategy(&request, block_size, max_iop_size);
        assert_eq!(
            resolve_out, legacy_out,
            "resolve(n=1) must match legacy coalesce-split"
        );
        assert_eq!(resolve_out, vec![4096_u64..(4096 + 8 * 1024)]);
    }

    #[test]
    fn test_single_oversized_range_split_correctly() {
        // n=1 oversized range must be split via legacy path.
        let req_start = 0_u64;
        let req_end = 32_u64 * 1024 * 1024;
        let request = vec![req_start..req_end];
        let max_iop_size = 16 * 1024 * 1024;
        let block_size = 4 * 1024;
        let out = resolve_ranges_for_env_strategy(&request, block_size, max_iop_size);
        // Must cover [0, 32MB) without gap, in <=2 sub-requests.
        assert!(!out.is_empty());
        assert_eq!(out.first().unwrap().start, req_start);
        assert_eq!(out.last().unwrap().end, req_end);
        for w in out.windows(2) {
            assert_eq!(w[0].end, w[1].start, "no gap between split parts");
        }
        for r in &out {
            assert!(
                r.end - r.start <= max_iop_size,
                "each split <= max_iop_size"
            );
        }
    }

    #[test]
    fn test_balanced_strategy_bridge_matches_direct_planner() {
        let request = vec![0_u64..512, 700..1_200, 2_000..2_400, 2_600..3_000];
        let max_iop_size = 512_u64;
        let cfg = MergePlannerConfig {
            kind: MergePlannerKind::BalancedUB,
            max_waste_ratio: 0.5,
            max_merge_span: 16 * 1024 * 1024,
            max_requests: 3,
            exact_requests: false,
            concurrent_parallelism: 8.0,
            latency_per_request_ms: Some(120.0),
            latency_per_byte_ms: Some(0.001),
            latency_std_ms: Some(0.0),
        };

        let bridged = strategy_plan_to_ranges_with_config(&request, max_iop_size, &cfg)
            .expect("bridge should produce ranges");

        let mut cluster_ids = Vec::with_capacity(request.len());
        let mut cluster_offsets = HashMap::with_capacity(request.len());
        let mut cluster_sizes = HashMap::with_capacity(request.len());
        for (idx, req) in request.iter().enumerate() {
            let cid = idx as i32;
            cluster_ids.push(cid);
            cluster_offsets.insert(cid, req.start as i64);
            cluster_sizes.insert(cid, (req.end - req.start) as i64);
        }

        let mut cost_model = CostModel::default();
        cost_model.max_waste_ratio = cfg.max_waste_ratio;
        cost_model.max_merge_span = cfg.max_merge_span;
        cost_model.max_requests = cfg.max_requests;
        cost_model.exact_requests = cfg.exact_requests;
        cost_model.concurrent_parallelism = cfg.concurrent_parallelism;
        cost_model.latency_per_request = cfg.latency_per_request_ms.unwrap();
        cost_model.latency_per_byte = cfg.latency_per_byte_ms.unwrap();
        cost_model.latency_std = cfg.latency_std_ms.unwrap();

        let direct = BalancedUBMerge::new().generate_plan(
            &cluster_ids,
            &cluster_offsets,
            &cluster_sizes,
            &cost_model,
        );
        assert!(!direct.requests.is_empty());

        let mut planned = Vec::with_capacity(direct.requests.len());
        for req in &direct.requests {
            let start = req.offset as u64;
            let end = start + req.length as u64;
            planned.push(start..end);
        }
        planned.sort_by_key(|r| r.start);
        let mut expected = Vec::new();
        for req in planned {
            split_large_request(req, max_iop_size, &mut expected);
        }

        assert_eq!(bridged, expected);
    }

    #[tokio::test]
    async fn test_full_seq_read() {
        let tmp_file = TempObjFile::default();

        let obj_store = Arc::new(ObjectStore::local());

        // Write 1MiB of data
        const DATA_SIZE: u64 = 1024 * 1024;
        let mut some_data = vec![0; DATA_SIZE as usize];
        rand::rng().fill_bytes(&mut some_data);
        obj_store.put(&tmp_file, &some_data).await.unwrap();

        let config = SchedulerConfig::default_for_testing();

        let scheduler = ScanScheduler::new(obj_store, config);

        let file_scheduler = scheduler
            .open_file(&tmp_file, &CachedFileSize::unknown())
            .await
            .unwrap();

        // Read it back 4KiB at a time
        const READ_SIZE: u64 = 4 * 1024;
        let mut reqs = VecDeque::new();
        let mut offset = 0;
        while offset < DATA_SIZE {
            reqs.push_back(
                #[allow(clippy::single_range_in_vec_init)]
                file_scheduler
                    .submit_request(vec![offset..offset + READ_SIZE], 0)
                    .await
                    .unwrap(),
            );
            offset += READ_SIZE;
        }

        offset = 0;
        // Note: we should get parallel I/O even though we are consuming serially
        while offset < DATA_SIZE {
            let data = reqs.pop_front().unwrap();
            let actual = &data[0];
            let expected = &some_data[offset as usize..(offset + READ_SIZE) as usize];
            assert_eq!(expected, actual);
            offset += READ_SIZE;
        }
    }

    #[tokio::test]
    async fn test_split_coalesce() {
        let tmp_file = TempObjFile::default();

        let obj_store = Arc::new(ObjectStore::local());

        // Write 75MiB of data
        const DATA_SIZE: u64 = 75 * 1024 * 1024;
        let mut some_data = vec![0; DATA_SIZE as usize];
        rand::rng().fill_bytes(&mut some_data);
        obj_store.put(&tmp_file, &some_data).await.unwrap();

        let config = SchedulerConfig::default_for_testing();

        let scheduler = ScanScheduler::new(obj_store, config);

        let file_scheduler = scheduler
            .open_file(&tmp_file, &CachedFileSize::unknown())
            .await
            .unwrap();

        // These 3 requests should be coalesced into a single I/O because they are within 4KiB
        // of each other
        let req =
            file_scheduler.submit_request(vec![50_000..51_000, 52_000..53_000, 54_000..55_000], 0);

        let bytes = req.await.unwrap();

        assert_eq!(bytes[0], &some_data[50_000..51_000]);
        assert_eq!(bytes[1], &some_data[52_000..53_000]);
        assert_eq!(bytes[2], &some_data[54_000..55_000]);

        assert_eq!(1, scheduler.stats().iops);

        // This should be split into 5 requests because it is so large
        let req = file_scheduler.submit_request(vec![0..DATA_SIZE], 0);
        let bytes = req.await.unwrap();
        assert!(bytes[0] == some_data, "data is not the same");

        assert_eq!(6, scheduler.stats().iops);

        // None of these requests are bigger than the max IOP size but they will be coalesced into
        // one IOP that is bigger and then split back into 2 requests that don't quite align with the original
        // ranges.
        let chunk_size = *DEFAULT_MAX_IOP_SIZE;
        let req = file_scheduler.submit_request(
            vec![
                10..chunk_size,
                chunk_size + 10..(chunk_size * 2) - 20,
                chunk_size * 2..(chunk_size * 2) + 10,
            ],
            0,
        );

        let bytes = req.await.unwrap();
        let chunk_size = chunk_size as usize;
        assert!(
            bytes[0] == some_data[10..chunk_size],
            "data is not the same"
        );
        assert!(
            bytes[1] == some_data[chunk_size + 10..(chunk_size * 2) - 20],
            "data is not the same"
        );
        assert!(
            bytes[2] == some_data[chunk_size * 2..(chunk_size * 2) + 10],
            "data is not the same"
        );
        assert_eq!(8, scheduler.stats().iops);

        let reads = (0..44)
            .map(|i| i * 1_000_000..(i + 1) * 1_000_000)
            .collect::<Vec<_>>();
        let req = file_scheduler.submit_request(reads, 0);
        let bytes = req.await.unwrap();
        for (i, bytes) in bytes.iter().enumerate() {
            assert!(
                bytes == &some_data[i * 1_000_000..(i + 1) * 1_000_000],
                "data is not the same"
            );
        }
        assert_eq!(11, scheduler.stats().iops);
    }

    #[tokio::test]
    async fn test_priority() {
        let some_path = Path::parse("foo").unwrap();
        let base_store = Arc::new(InMemory::new());
        base_store
            .put(&some_path, vec![0; 1000].into())
            .await
            .unwrap();

        let semaphore = Arc::new(tokio::sync::Semaphore::new(0));
        let mut obj_store = MockObjectStore::default();
        let semaphore_copy = semaphore.clone();
        obj_store
            .expect_get_opts()
            .returning(move |location, options| {
                let semaphore = semaphore.clone();
                let base_store = base_store.clone();
                let location = location.clone();
                async move {
                    semaphore.acquire().await.unwrap().forget();
                    base_store.get_opts(&location, options).await
                }
                .boxed()
            });
        let obj_store = Arc::new(ObjectStore::new(
            Arc::new(obj_store),
            Url::parse("mem://").unwrap(),
            Some(500),
            None,
            false,
            false,
            1,
            DEFAULT_DOWNLOAD_RETRY_COUNT,
            None,
        ));

        let config = SchedulerConfig {
            io_buffer_size_bytes: 1024 * 1024,
        };

        let scan_scheduler = ScanScheduler::new(obj_store, config);

        let file_scheduler = scan_scheduler
            .open_file(&Path::parse("foo").unwrap(), &CachedFileSize::new(1000))
            .await
            .unwrap();

        // Issue a request, priority doesn't matter, it will be submitted
        // immediately (it will go pending)
        // Note: the timeout is to prevent a deadlock if the test fails.
        let first_fut = timeout(
            Duration::from_secs(10),
            file_scheduler.submit_single(0..10, 0),
        )
        .boxed();

        // Issue another low priority request (it will go in queue)
        let mut second_fut = timeout(
            Duration::from_secs(10),
            file_scheduler.submit_single(0..20, 100),
        )
        .boxed();

        // Issue a high priority request (it will go in queue and should bump
        // the other queued request down)
        let mut third_fut = timeout(
            Duration::from_secs(10),
            file_scheduler.submit_single(0..30, 0),
        )
        .boxed();

        // Finish one file, should be the in-flight first request
        semaphore_copy.add_permits(1);
        assert!(first_fut.await.unwrap().unwrap().len() == 10);
        // Other requests should not be finished
        assert!(poll!(&mut second_fut).is_pending());
        assert!(poll!(&mut third_fut).is_pending());

        // Next should be high priority request
        semaphore_copy.add_permits(1);
        assert!(third_fut.await.unwrap().unwrap().len() == 30);
        assert!(poll!(&mut second_fut).is_pending());

        // Finally, the low priority request
        semaphore_copy.add_permits(1);
        assert!(second_fut.await.unwrap().unwrap().len() == 20);
    }

    #[tokio::test]
    async fn test_equal_priority_queue_is_fifo() {
        let tmp_file = TempObjFile::default();
        let object_store = Arc::new(ObjectStore::local());
        object_store.put(&tmp_file, &[0; 16]).await.unwrap();

        let scheduler = ScanScheduler::new(object_store, SchedulerConfig::default_for_testing());
        let file_scheduler = scheduler
            .open_file(&tmp_file, &CachedFileSize::unknown())
            .await
            .unwrap();

        let queue = IoQueue::new(0, 1024);
        for offset in 0_u64..8 {
            queue.push(IoTask {
                reader: file_scheduler.reader.clone(),
                to_read: offset..offset + 1,
                when_done: Box::new(|_| {}),
                priority: 7,
                enqueue_order: 0,
                enqueued_at: Instant::now(),
            });
        }

        let observed = {
            let mut state = queue.state.lock().unwrap();
            std::iter::from_fn(|| state.pending_requests.pop())
                .map(|task| task.to_read.start)
                .collect::<Vec<_>>()
        };
        assert_eq!(observed, (0_u64..8).collect::<Vec<_>>());
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn test_backpressure() {
        let some_path = Path::parse("foo").unwrap();
        let base_store = Arc::new(InMemory::new());
        base_store
            .put(&some_path, vec![0; 100000].into())
            .await
            .unwrap();

        let bytes_read = Arc::new(AtomicU64::from(0));
        let mut obj_store = MockObjectStore::default();
        let bytes_read_copy = bytes_read.clone();
        // Wraps the obj_store to keep track of how many bytes have been read
        obj_store
            .expect_get_opts()
            .returning(move |location, options| {
                let range = options.range.as_ref().unwrap();
                let num_bytes = match range {
                    GetRange::Bounded(bounded) => bounded.end - bounded.start,
                    _ => panic!(),
                };
                bytes_read_copy.fetch_add(num_bytes, Ordering::Release);
                let location = location.clone();
                let base_store = base_store.clone();
                async move { base_store.get_opts(&location, options).await }.boxed()
            });
        let obj_store = Arc::new(ObjectStore::new(
            Arc::new(obj_store),
            Url::parse("mem://").unwrap(),
            Some(500),
            None,
            false,
            false,
            1,
            DEFAULT_DOWNLOAD_RETRY_COUNT,
            None,
        ));

        let config = SchedulerConfig {
            io_buffer_size_bytes: 10,
        };

        let scan_scheduler = ScanScheduler::new(obj_store.clone(), config);

        let file_scheduler = scan_scheduler
            .open_file(&Path::parse("foo").unwrap(), &CachedFileSize::new(100000))
            .await
            .unwrap();

        let wait_for_idle = || async move {
            let handle = Handle::current();
            while handle.metrics().num_alive_tasks() != 1 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        };
        let wait_for_bytes_read_and_idle = |target_bytes: u64| {
            // We need to move `target` but don't want to move `bytes_read`
            let bytes_read = &bytes_read;
            async move {
                let bytes_read_copy = bytes_read.clone();
                while bytes_read_copy.load(Ordering::Acquire) < target_bytes {
                    tokio::time::sleep(Duration::from_millis(10)).await;
                }
                wait_for_idle().await;
            }
        };

        // This read will begin immediately
        let first_fut = file_scheduler.submit_single(0..5, 0);
        // This read should also begin immediately
        let second_fut = file_scheduler.submit_single(0..5, 0);
        // This read will be throttled
        let third_fut = file_scheduler.submit_single(0..3, 0);
        // Two tasks (third_fut and unit test)
        wait_for_bytes_read_and_idle(10).await;

        assert_eq!(first_fut.await.unwrap().len(), 5);
        // One task (unit test)
        wait_for_bytes_read_and_idle(13).await;

        // 2 bytes are ready but 5 bytes requested, read will be blocked
        let fourth_fut = file_scheduler.submit_single(0..5, 0);
        wait_for_bytes_read_and_idle(13).await;

        // Out of order completion is ok, will unblock backpressure
        assert_eq!(third_fut.await.unwrap().len(), 3);
        wait_for_bytes_read_and_idle(18).await;

        assert_eq!(second_fut.await.unwrap().len(), 5);
        // At this point there are 5 bytes available in backpressure queue
        // Now we issue multi-read that can be partially fulfilled, it will read some bytes but
        // not all of them. (using large range gap to ensure request not coalesced)
        //
        // I'm actually not sure this behavior is great.  It's possible that we should just
        // block until we can fulfill the entire request.
        let fifth_fut = file_scheduler.submit_request(vec![0..3, 90000..90007], 0);
        wait_for_bytes_read_and_idle(21).await;

        // Fifth future should eventually finish due to deadlock prevention
        let fifth_bytes = tokio::time::timeout(Duration::from_secs(10), fifth_fut)
            .await
            .unwrap();
        assert_eq!(
            fifth_bytes.unwrap().iter().map(|b| b.len()).sum::<usize>(),
            10
        );

        // And now let's just make sure that we can read the rest of the data
        assert_eq!(fourth_fut.await.unwrap().len(), 5);
        wait_for_bytes_read_and_idle(28).await;

        // Ensure deadlock prevention timeout can be disabled
        let config = SchedulerConfig {
            io_buffer_size_bytes: 10,
        };

        let scan_scheduler = ScanScheduler::new(obj_store, config);
        let file_scheduler = scan_scheduler
            .open_file(&Path::parse("foo").unwrap(), &CachedFileSize::new(100000))
            .await
            .unwrap();

        let first_fut = file_scheduler.submit_single(0..10, 0);
        let second_fut = file_scheduler.submit_single(0..10, 0);

        std::thread::sleep(Duration::from_millis(100));
        assert_eq!(first_fut.await.unwrap().len(), 10);
        assert_eq!(second_fut.await.unwrap().len(), 10);
    }

    #[test_log::test(tokio::test(flavor = "multi_thread"))]
    async fn stress_backpressure() {
        // This test ensures that the backpressure mechanism works correctly with
        // regards to priority.  In other words, as long as all requests are consumed
        // in priority order then the backpressure mechanism should not deadlock
        let some_path = Path::parse("foo").unwrap();
        let obj_store = Arc::new(ObjectStore::memory());
        obj_store
            .put(&some_path, vec![0; 100000].as_slice())
            .await
            .unwrap();

        // Only one request will be allowed in
        let config = SchedulerConfig {
            io_buffer_size_bytes: 1,
        };
        let scan_scheduler = ScanScheduler::new(obj_store.clone(), config);
        let file_scheduler = scan_scheduler
            .open_file(&some_path, &CachedFileSize::unknown())
            .await
            .unwrap();

        let mut futs = Vec::with_capacity(10000);
        for idx in 0..10000 {
            futs.push(file_scheduler.submit_single(idx..idx + 1, idx));
        }

        for fut in futs {
            fut.await.unwrap();
        }
    }
}
