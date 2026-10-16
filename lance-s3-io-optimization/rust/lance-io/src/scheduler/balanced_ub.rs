// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright The Lance Authors

//! Balanced-UB merge strategy
//!
//! Objective:
//! T = W / P + (1 - 1/P) * B
//! where W = sum of request latencies and B = max request latency.
//! If P<=0 we treat it as full-parallel objective (T=B).

use std::cmp::Ordering;
use std::collections::{BTreeMap, BinaryHeap, HashMap, VecDeque};
use std::ops::Bound::{Excluded, Unbounded};

use super::cost_model::CostModel;
use super::io_types::{IOMetrics, IOPlan, IORequest};
use super::merge_strategy::{
    calculate_plan_metrics, collect_ordered_cluster_layout, requests_from_ordered_spans,
    MergeStrategy,
};

const INF: f64 = 1e18;
const EPS: f64 = 1e-9;
// Keep the exact DP path for the paper's nprobe <= 256 operating points.
// The lossy pre-merge and adaptive approximations are only justified for
// larger inputs where the exact O(n^2) edge construction is prohibitive.
const APPROX_NPROBE_THRESHOLD: usize = 256;
const APPROX_MIN_EVAL_BUDGET: usize = 16;
const APPROX_MAX_EVAL_BUDGET: usize = 64;
const APPROX_TARGET_RELATIVE_ERROR: f64 = 0.05;
const WQS_LARGE_K_NUM: usize = 1;
const WQS_LARGE_K_DEN: usize = 4;
const DC_SMALL_I_THRESHOLD: usize = 8;
const DC_SMALL_K_THRESHOLD: usize = 8;
const DC_SMALL_AREA_THRESHOLD: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SubproblemOptimizationMode {
    Auto,
    DivideAndConquerOnly,
    PlainDP,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SubproblemSolverKind {
    LinearFastPath,
    PiecewiseWqs,
    DivideAndConquer,
    PlainDP,
}

struct FlatTable<T> {
    data: Vec<T>,
    cols: usize,
}

impl<T: Clone> FlatTable<T> {
    fn new(rows: usize, cols: usize, init: T) -> Self {
        Self {
            data: vec![init; rows * cols],
            cols,
        }
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> &T {
        &self.data[i * self.cols + j]
    }

    #[inline]
    fn set(&mut self, i: usize, j: usize, val: T) {
        self.data[i * self.cols + j] = val;
    }

    fn fill(&mut self, val: T) {
        self.data.fill(val);
    }
}

#[inline]
fn triangular_edge_index(end_idx: usize, start_idx: usize) -> usize {
    (end_idx * (end_idx + 1)) / 2 + start_idx
}

#[inline]
fn segment_span_bytes(offsets: &[i64], sizes: &[i64], start_idx: usize, end_idx: usize) -> i64 {
    (offsets[end_idx] + sizes[end_idx]) - offsets[start_idx]
}

#[inline]
fn segment_span_ok(
    offsets: &[i64],
    sizes: &[i64],
    start_idx: usize,
    end_idx: usize,
    max_merge_span: i64,
) -> bool {
    max_merge_span <= 0 || segment_span_bytes(offsets, sizes, start_idx, end_idx) <= max_merge_span
}

#[inline]
fn dc_row_base(end_idx: usize) -> usize {
    triangular_edge_index(end_idx, 0)
}

#[inline]
fn dc_min_feasible_k_for_end(
    end_idx: usize,
    k_lo: usize,
    bottleneck_cap: f64,
    offsets: &[i64],
    sizes: &[i64],
    max_merge_span: i64,
    edge_latency: &[f64],
) -> usize {
    let mut scan_lo = k_lo;

    if max_merge_span > 0 {
        let min_start_offset = offsets[end_idx] + sizes[end_idx] - max_merge_span;
        let span_lo = offsets[..=end_idx].partition_point(|&offset| offset < min_start_offset);
        scan_lo = scan_lo.max(span_lo);
    }

    // For a fixed end_idx, edge_latency[end_idx, start] is monotone non-increasing
    // in start because later starts imply shorter spans and non-negative latency slopes.
    let row_base = dc_row_base(end_idx);
    let latency_lo = edge_latency[row_base..(row_base + end_idx + 1)]
        .partition_point(|&latency| latency > bottleneck_cap + EPS);
    scan_lo.max(latency_lo)
}

#[inline]
fn should_use_small_dc_solver(i_lo: usize, i_hi: usize, k_lo: usize, k_hi: usize) -> bool {
    let i_len = i_hi - i_lo + 1;
    let k_len = k_hi - k_lo + 1;
    i_len <= DC_SMALL_I_THRESHOLD
        || k_len <= DC_SMALL_K_THRESHOLD
        || i_len.saturating_mul(k_len) <= DC_SMALL_AREA_THRESHOLD
}

#[inline]
fn is_large_k_for_wqs(n: usize, k_cap: usize) -> bool {
    k_cap.saturating_mul(WQS_LARGE_K_DEN) >= n.saturating_mul(WQS_LARGE_K_NUM)
}

struct PreparedPiecewiseLatency {
    breaks_bytes: Vec<i64>,
    effective_overheads_ms: Vec<f64>,
    effective_rates_ms_per_byte: Vec<f64>,
}

fn prepare_piecewise_latency(cost_model: &CostModel) -> Option<PreparedPiecewiseLatency> {
    if !cost_model.use_piecewise_linear() {
        return None;
    }

    let segment_count = cost_model.latency_pw_overheads_ms.len();
    let mut effective_overheads_ms = vec![0.0; segment_count];
    let mut effective_rates_ms_per_byte = vec![0.0; segment_count];

    effective_overheads_ms[0] = cost_model.latency_pw_overheads_ms[0];
    effective_rates_ms_per_byte[0] = cost_model.latency_pw_rates_ms_per_byte[0].max(0.0);

    if !cost_model.latency_pw_breaks_bytes.is_empty() {
        let mut w_at_break = effective_overheads_ms[0]
            + effective_rates_ms_per_byte[0] * cost_model.latency_pw_breaks_bytes[0] as f64;
        for i in 1..segment_count {
            let rate = cost_model.latency_pw_rates_ms_per_byte[i].max(0.0);
            let prev_break = cost_model.latency_pw_breaks_bytes[i - 1] as f64;
            let overhead = w_at_break - rate * prev_break;
            effective_overheads_ms[i] = overhead;
            effective_rates_ms_per_byte[i] = rate;
            if i < cost_model.latency_pw_breaks_bytes.len() {
                w_at_break = overhead + rate * cost_model.latency_pw_breaks_bytes[i] as f64;
            }
        }
    }

    Some(PreparedPiecewiseLatency {
        breaks_bytes: cost_model.latency_pw_breaks_bytes.clone(),
        effective_overheads_ms,
        effective_rates_ms_per_byte,
    })
}

#[inline]
fn prepared_piecewise_latency_ms(prepared: &PreparedPiecewiseLatency, bytes: i64) -> f64 {
    let bytes = bytes.max(0);
    let segment_idx = prepared
        .breaks_bytes
        .partition_point(|&break_bytes| break_bytes < bytes);
    prepared.effective_overheads_ms[segment_idx]
        + prepared.effective_rates_ms_per_byte[segment_idx] * bytes as f64
}

/// Compute the break-even gap_max for `cheap_pre_merge` from the
/// cost model: gap = alpha / (beta * (P - 1)), clamped to [4 KiB, 256 KiB].
///
/// alpha (per-request overhead) and beta (per-byte rate) come from the
/// linear or piecewise model. For piecewise the larger (post-breakpoint)
/// rate is used so the threshold stays conservative.
fn compute_adaptive_premerge_gap(cost_model: &CostModel) -> i64 {
    const GAP_FLOOR: i64 = 4 * 1024;
    const GAP_CEIL: i64 = 256 * 1024;

    let p = cost_model.concurrent_parallelism.max(1.0);
    let denom = (p - 1.0).max(1.0); // avoid divide-by-zero at P<=1
    let alpha = if cost_model.latency_model == "piecewise_linear"
        && !cost_model.latency_pw_overheads_ms.is_empty()
    {
        cost_model.latency_pw_overheads_ms[0]
    } else {
        cost_model.latency_per_request
    };
    let beta = if cost_model.latency_model == "piecewise_linear"
        && !cost_model.latency_pw_rates_ms_per_byte.is_empty()
    {
        // Use the largest (post-breakpoint) rate so the threshold
        // stays conservative on the bandwidth-dominated regime.
        cost_model
            .latency_pw_rates_ms_per_byte
            .iter()
            .cloned()
            .fold(0.0_f64, f64::max)
    } else {
        cost_model.latency_per_byte
    };

    if !alpha.is_finite() || !beta.is_finite() || alpha <= 0.0 || beta <= 0.0 {
        // Degenerate cost model — fall back to the historical fixed value.
        return 64 * 1024;
    }
    let gap = alpha / (beta * denom);
    if !gap.is_finite() {
        return 64 * 1024;
    }
    let gap_b = gap.round() as i64;
    gap_b.clamp(GAP_FLOOR, GAP_CEIL)
}

/// Greedy closest-pair pre-merge, threshold-gated.
///
/// Merges adjacent ranges while their byte gap and merged span remain within
/// the supplied limits. If the result is still too large for the DP, the
/// caller splits it into independent chunks with `split_into_chunks`.
fn cheap_pre_merge(
    offsets: Vec<i64>,
    sizes: Vec<i64>,
    max_gap_bytes: i64,
    max_merged_size_bytes: i64,
) -> (Vec<i64>, Vec<i64>) {
    debug_assert_eq!(offsets.len(), sizes.len());
    if offsets.is_empty() {
        return (Vec::new(), Vec::new());
    }

    // Stage 1: single-pass forward sweep, O(n). Greedy-merge each
    // incoming range into the tail of `groups` whenever the gap and
    // size cap allow. Under hard gap/size thresholds this pass alone
    // is optimal for "minimize group count". It shrinks raw_n → m by
    // ~10× on real MiniBlock inputs, so Stage 2's heap only operates
    // on a small tail.
    let mut groups: Vec<(i64, i64)> = Vec::with_capacity(offsets.len());
    for (o, s) in offsets.into_iter().zip(sizes.into_iter()) {
        if let Some(&mut (prev_o, ref mut prev_s)) = groups.last_mut() {
            let gap = o - (prev_o + *prev_s);
            // `max` guards against a fully-contained (o, s) inside the
            // previous group — release builds strip the debug_assert
            // below, so this is the only correctness backstop.
            let merged_end = (prev_o + *prev_s).max(o + s);
            let merged_size = merged_end - prev_o;
            if gap <= max_gap_bytes && merged_size <= max_merged_size_bytes {
                *prev_s = merged_end - prev_o;
                continue;
            }
            debug_assert!(o >= prev_o + *prev_s, "overlap or unsorted at pre_merge");
        }
        groups.push((o, s));
    }

    // Stage 2: O(m log m) min-gap priority refinement. After Stage 1,
    // any remaining merge opportunity must cross a boundary where the
    // forward sweep had to give up (either gap > threshold *at that
    // time*, or the earlier group had already grown too large). Stage
    // 2 finds these cases via a min-heap on adjacent-pair gaps with
    // lazy deletion: each merge is O(log m); total O(m log m).
    //
    // In practice Stage 2 does very little extra work under hard
    // thresholds (Stage 1 is already optimal for minimize-count). It
    // exists as a safety net for edge cases where forward-sweep and
    // min-gap priority disagree — e.g. when overlapping/duplicate
    // ranges let a merge happen out-of-order. The total cost stays
    // bounded by the (already-small) m rather than the original n.
    let m = groups.len();
    if m < 2 {
        let (out_offsets, out_sizes): (Vec<_>, Vec<_>) = groups.into_iter().unzip();
        return (out_offsets, out_sizes);
    }

    #[derive(Clone)]
    struct Node {
        offset: i64,
        size: i64,
        prev: usize,
        next: usize,
        alive: bool,
        // Version bumped whenever this node's size or next changes.
        // Heap entries keyed on a node carry the version at push time;
        // a mismatch at pop time means the entry is stale.
        version: u64,
    }
    const NIL: usize = usize::MAX;

    let mut nodes: Vec<Node> = groups
        .into_iter()
        .enumerate()
        .map(|(i, (o, s))| Node {
            offset: o,
            size: s,
            prev: if i == 0 { NIL } else { i - 1 },
            next: if i + 1 == m { NIL } else { i + 1 },
            alive: true,
            version: 0,
        })
        .collect();

    use std::cmp::Reverse;
    use std::collections::BinaryHeap;
    let mut heap: BinaryHeap<Reverse<(i64, usize, u64)>> = BinaryHeap::with_capacity(m);
    for i in 0..m - 1 {
        let node_i = &nodes[i];
        let node_j = &nodes[i + 1];
        let gap = node_j.offset - (node_i.offset + node_i.size);
        if gap <= max_gap_bytes {
            heap.push(Reverse((gap, i, node_i.version)));
        }
    }

    while let Some(Reverse((_heap_gap, i, ver))) = heap.pop() {
        // Lazy validation.
        if !nodes[i].alive || nodes[i].version != ver {
            continue;
        }
        let j = nodes[i].next;
        if j == NIL || !nodes[j].alive {
            continue;
        }
        // Recompute both gap and feasibility from current state.  The
        // pushed gap can still be correct even when a neighbor-change
        // has silently invalidated the merge (e.g. i grew into the
        // size cap via a previous merge).
        let cur_gap = nodes[j].offset - (nodes[i].offset + nodes[i].size);
        if cur_gap > max_gap_bytes {
            continue;
        }
        let merged_end = (nodes[i].offset + nodes[i].size).max(nodes[j].offset + nodes[j].size);
        let merged_size = merged_end - nodes[i].offset;
        if merged_size > max_merged_size_bytes {
            continue;
        }

        // Perform the merge: i absorbs j, j dies.
        let new_next = nodes[j].next;
        nodes[i].size = merged_size;
        nodes[i].next = new_next;
        nodes[i].version += 1;
        nodes[j].alive = false;
        if new_next != NIL {
            nodes[new_next].prev = i;

            // Re-push the (i, new_next) candidate with the fresh gap.
            let g = nodes[new_next].offset - (nodes[i].offset + nodes[i].size);
            if g <= max_gap_bytes {
                heap.push(Reverse((g, i, nodes[i].version)));
            }
        }
        // The (prev, i) entry, if any, is still in the heap with the
        // pre-merge gap. The gap itself is unchanged (i's offset did
        // not move), but merged_size with the now-larger i might fail
        // the cap. The pop-time recheck above drops it when that
        // happens, so no action needed here.
    }

    let mut out_offsets = Vec::with_capacity(nodes.len());
    let mut out_sizes = Vec::with_capacity(nodes.len());
    for node in &nodes {
        if node.alive {
            out_offsets.push(node.offset);
            out_sizes.push(node.size);
        }
    }
    (out_offsets, out_sizes)
}

/// Split (offsets, sizes) into roughly-equal chunks of at most
/// `max_chunk_size` ranges each, so each chunk can be fed to the DP
/// independently. Chunks are contiguous index ranges (preserving the
/// input order). Input must already be sorted by offset.
///
/// The split points fall on index boundaries, *not* on gap boundaries,
/// because the caller already ran `cheap_pre_merge` first — any
/// remaining inter-range gaps are > threshold, i.e., "not worth merging
/// across" from the cost-model's perspective, so even splits incur no
/// meaningful optimality loss.
fn split_into_chunks(
    offsets: Vec<i64>,
    sizes: Vec<i64>,
    max_chunk_size: usize,
) -> Vec<(Vec<i64>, Vec<i64>)> {
    debug_assert_eq!(offsets.len(), sizes.len());
    let n = offsets.len();
    if n <= max_chunk_size {
        return vec![(offsets, sizes)];
    }
    let num_chunks = n.div_ceil(max_chunk_size);
    // Evenly sized chunks: floor(n / num_chunks) with the remainder
    // spread across the first few chunks.
    let base = n / num_chunks;
    let rem = n % num_chunks;
    let mut out = Vec::with_capacity(num_chunks);
    let mut start = 0usize;
    for i in 0..num_chunks {
        let take = base + if i < rem { 1 } else { 0 };
        let end = start + take;
        out.push((offsets[start..end].to_vec(), sizes[start..end].to_vec()));
        start = end;
    }
    out
}

/// Allocate one global request cap across independent DP chunks.
///
/// Every non-empty chunk needs at least one request.  Remaining requests are
/// distributed approximately in proportion to each chunk's input range count,
/// with the final allocation summing to `global_cap` (or the total number of
/// ranges when that is smaller).
fn allocate_chunk_request_caps(chunk_sizes: &[usize], global_cap: usize) -> Vec<usize> {
    if chunk_sizes.is_empty() {
        return Vec::new();
    }
    debug_assert!(chunk_sizes.iter().all(|&size| size > 0));
    let total_ranges: usize = chunk_sizes.iter().sum();
    let target = global_cap.min(total_ranges);
    assert!(
        target >= chunk_sizes.len(),
        "each non-empty chunk requires at least one request"
    );

    let mut caps = Vec::with_capacity(chunk_sizes.len());
    let mut remaining_cap = target;
    let mut remaining_ranges = total_ranges;
    for (index, &chunk_size) in chunk_sizes.iter().enumerate() {
        let remaining_chunks = chunk_sizes.len() - index - 1;
        let max_for_chunk = chunk_size.min(remaining_cap - remaining_chunks);
        let proportional = remaining_cap.saturating_mul(chunk_size) / remaining_ranges;
        let cap = proportional.clamp(1, max_for_chunk);
        caps.push(cap);
        remaining_cap -= cap;
        remaining_ranges -= chunk_size;
    }
    debug_assert_eq!(caps.iter().sum::<usize>(), target);
    caps
}

fn build_edge_latency_matrix(offsets: &[i64], sizes: &[i64], cost_model: &CostModel) -> Vec<f64> {
    let n = offsets.len();
    let total_edges = n * (n + 1) / 2;
    let mut edge_latency = vec![0.0; total_edges];
    let mut useful_prefix = Vec::with_capacity(n + 1);
    useful_prefix.push(0_i64);
    for &size in sizes {
        useful_prefix.push(useful_prefix.last().copied().unwrap() + size);
    }

    let edge_is_allowed = |start: usize, end: usize, range_bytes: i64| {
        if cost_model.max_waste_ratio >= 1.0 {
            return true;
        }
        let useful_bytes = useful_prefix[end + 1] - useful_prefix[start];
        let wasted_bytes = (range_bytes - useful_bytes).max(0);
        range_bytes <= 0
            || (wasted_bytes as f64 / range_bytes as f64) <= cost_model.max_waste_ratio + EPS
    };

    if let Some(prepared) = prepare_piecewise_latency(cost_model) {
        for end in 0..n {
            let range_end = offsets[end] + sizes[end];
            for start in 0..=end {
                let range_bytes = range_end - offsets[start];
                edge_latency[triangular_edge_index(end, start)] =
                    if edge_is_allowed(start, end, range_bytes) {
                        prepared_piecewise_latency_ms(&prepared, range_bytes)
                    } else {
                        INF
                    };
            }
        }
    } else {
        let request_overhead = cost_model.latency_per_request;
        let request_slope = cost_model.latency_per_byte;
        for end in 0..n {
            let range_end = offsets[end] + sizes[end];
            for start in 0..=end {
                let range_bytes = range_end - offsets[start];
                edge_latency[triangular_edge_index(end, start)] =
                    if edge_is_allowed(start, end, range_bytes) {
                        request_overhead + request_slope * range_bytes as f64
                    } else {
                        INF
                    };
            }
        }
    }

    edge_latency
}

fn reconstruct_requests(
    n: usize,
    request_count: usize,
    parent: &FlatTable<i32>,
    offsets: &[i64],
    sizes: &[i64],
) -> Vec<IORequest> {
    let mut requests = Vec::new();
    let mut cur_i = n;
    let mut cur_j = request_count;

    while cur_j > 0 && cur_i > 0 {
        let prev = *parent.get(cur_i, cur_j);
        if prev < 0 || prev as usize >= cur_i {
            break;
        }
        let prev = prev as usize;
        let range_start = offsets[prev];
        let range_end = offsets[cur_i - 1] + sizes[cur_i - 1];
        requests.push(IORequest::new(range_start, range_end - range_start));
        cur_i = prev;
        cur_j -= 1;
    }

    requests.reverse();
    requests
}

#[cfg(test)]
fn objective_from_requests(
    requests: &[IORequest],
    cost_model: &CostModel,
    coeff_sum: f64,
    coeff_bottleneck: f64,
) -> f64 {
    if requests.is_empty() {
        return INF;
    }

    let mut total_work = 0.0;
    let mut max_latency = 0.0_f64;
    if let Some(prepared) = prepare_piecewise_latency(cost_model) {
        for req in requests {
            let latency = prepared_piecewise_latency_ms(&prepared, req.length);
            total_work += latency;
            max_latency = max_latency.max(latency);
        }
    } else {
        let request_overhead = cost_model.latency_per_request;
        let request_slope = cost_model.latency_per_byte;
        for req in requests {
            let latency = request_overhead + request_slope * req.length as f64;
            total_work += latency;
            max_latency = max_latency.max(latency);
        }
    }

    coeff_sum * total_work + coeff_bottleneck * max_latency
}

fn objective_from_parent(
    n: usize,
    request_count: usize,
    parent: &FlatTable<i32>,
    edge_latency: &[f64],
    coeff_sum: f64,
    coeff_bottleneck: f64,
) -> Option<f64> {
    let mut total_work = 0.0_f64;
    let mut max_latency = 0.0_f64;
    let mut cur_i = n;
    let mut cur_j = request_count;

    while cur_j > 0 && cur_i > 0 {
        let prev = *parent.get(cur_i, cur_j);
        if prev < 0 || prev as usize >= cur_i {
            return None;
        }
        let prev = prev as usize;
        let latency = edge_latency[triangular_edge_index(cur_i - 1, prev)];
        total_work += latency;
        max_latency = max_latency.max(latency);
        cur_i = prev;
        cur_j -= 1;
    }

    if cur_i == 0 && cur_j == 0 {
        Some(coeff_sum * total_work + coeff_bottleneck * max_latency)
    } else {
        None
    }
}

fn compute_objective_coefficients(parallelism: f64) -> (f64, f64) {
    if parallelism <= EPS {
        return (0.0, 1.0);
    }
    let p = parallelism.max(1.0);
    let coeff_sum = 1.0 / p;
    (coeff_sum, 1.0 - coeff_sum)
}

fn is_convex_piecewise_linear(cost_model: &CostModel) -> bool {
    if !cost_model.use_piecewise_linear() {
        return false;
    }
    let slopes = &cost_model.latency_pw_rates_ms_per_byte;
    if slopes.is_empty() {
        return false;
    }
    for i in 0..slopes.len().saturating_sub(1) {
        if slopes[i] > slopes[i + 1] + 1e-12 {
            return false;
        }
    }
    true
}

struct PlannerContext<'a> {
    n: usize,
    k_cap: usize,
    offsets: &'a [i64],
    sizes: &'a [i64],
    cost_model: &'a CostModel,
    edge_latency: &'a [f64],
    coeff_sum: f64,
    coeff_bottleneck: f64,
    exact_requests: bool,
    max_merge_span: i64,
    is_linear_model: bool,
    is_convex_piecewise: bool,
    use_wqs_linear_kcap: bool,
    use_wqs_piecewise_kcap: bool,
    subproblem_optimization_mode: SubproblemOptimizationMode,
}

fn select_subproblem_solver(ctx: &PlannerContext) -> SubproblemSolverKind {
    if ctx.cost_model.max_waste_ratio < 1.0 {
        return SubproblemSolverKind::PlainDP;
    }
    match ctx.subproblem_optimization_mode {
        SubproblemOptimizationMode::PlainDP => SubproblemSolverKind::PlainDP,
        SubproblemOptimizationMode::DivideAndConquerOnly => SubproblemSolverKind::DivideAndConquer,
        SubproblemOptimizationMode::Auto => {
            if ctx.use_wqs_piecewise_kcap {
                SubproblemSolverKind::PiecewiseWqs
            } else if ctx.is_linear_model || ctx.use_wqs_linear_kcap {
                SubproblemSolverKind::LinearFastPath
            } else if ctx.is_convex_piecewise {
                SubproblemSolverKind::DivideAndConquer
            } else {
                SubproblemSolverKind::PlainDP
            }
        }
    }
}

struct WarmStartResult {
    best_obj: f64,
    min_w: f64,
    best_plan: Vec<IORequest>,
}

fn compute_warm_start(ctx: &PlannerContext) -> WarmStartResult {
    let probe_t = std::env::var("WARP_PROBE").is_ok();
    let t_alloc = if probe_t {
        Some(std::time::Instant::now())
    } else {
        None
    };
    let mut dp = FlatTable::new(ctx.n + 1, ctx.k_cap + 1, INF);
    let mut parent = FlatTable::new(ctx.n + 1, ctx.k_cap + 1, -1);
    dp.set(0, 0, 0.0);
    let alloc_us = t_alloc.map(|t| t.elapsed().as_micros()).unwrap_or(0);

    let solver = select_subproblem_solver(ctx);
    let t_inner = if probe_t {
        Some(std::time::Instant::now())
    } else {
        None
    };
    match solver {
        SubproblemSolverKind::LinearFastPath => {
            run_inner_dp_linear_with_b_parent(
                ctx.n,
                ctx.k_cap,
                ctx.offsets,
                ctx.sizes,
                ctx.cost_model,
                ctx.max_merge_span,
                INF,
                &mut dp,
                &mut parent,
            );
        }
        SubproblemSolverKind::PiecewiseWqs | SubproblemSolverKind::DivideAndConquer => {
            solve_all_layers_dc(
                ctx.n,
                ctx.k_cap,
                INF,
                &mut dp,
                Some(&mut parent),
                ctx.offsets,
                ctx.sizes,
                ctx.max_merge_span,
                ctx.edge_latency,
            );
        }
        SubproblemSolverKind::PlainDP => {
            run_inner_dp_general_with_b_parent(
                ctx.n,
                ctx.k_cap,
                ctx.offsets,
                ctx.sizes,
                ctx.max_merge_span,
                ctx.edge_latency,
                INF,
                &mut dp,
                &mut parent,
            );
        }
    }
    let inner_us = t_inner.map(|t| t.elapsed().as_micros()).unwrap_or(0);
    if probe_t {
        eprintln!(
            "[warm_start] solver={:?} n={} k_cap={} alloc_us={} inner_us={}",
            solver, ctx.n, ctx.k_cap, alloc_us, inner_us
        );
    }

    let mut warm = WarmStartResult {
        best_obj: INF,
        min_w: INF,
        best_plan: Vec::new(),
    };
    let mut best_j = None;

    let j_lo = if ctx.exact_requests { ctx.k_cap } else { 1 };
    for j in j_lo..=ctx.k_cap {
        let w = *dp.get(ctx.n, j);
        if w >= INF / 10.0 {
            continue;
        }

        warm.min_w = warm.min_w.min(w);
        let Some(obj) = objective_from_parent(
            ctx.n,
            j,
            &parent,
            ctx.edge_latency,
            ctx.coeff_sum,
            ctx.coeff_bottleneck,
        ) else {
            continue;
        };

        if obj < warm.best_obj - EPS {
            warm.best_obj = obj;
            best_j = Some(j);
        }
    }

    if let Some(best_j) = best_j {
        warm.best_plan = reconstruct_requests(ctx.n, best_j, &parent, ctx.offsets, ctx.sizes);
    }

    warm
}

fn solve_layer_dc(
    i_lo: usize,
    i_hi: usize,
    k_lo: usize,
    k_hi: usize,
    j: usize,
    bottleneck_cap: f64,
    dp: &mut FlatTable<f64>,
    mut parent: Option<&mut FlatTable<i32>>,
    offsets: &[i64],
    sizes: &[i64],
    max_merge_span: i64,
    edge_latency: &[f64],
) {
    if i_lo > i_hi {
        return;
    }

    if should_use_small_dc_solver(i_lo, i_hi, k_lo, k_hi) {
        for i in i_lo..=i_hi {
            let end_idx = i - 1;
            let k_upper = k_hi.min(end_idx);
            if k_lo > k_upper {
                continue;
            }

            let scan_lo = dc_min_feasible_k_for_end(
                end_idx,
                k_lo,
                bottleneck_cap,
                offsets,
                sizes,
                max_merge_span,
                edge_latency,
            );
            if scan_lo > k_upper {
                continue;
            }

            let row_base = dc_row_base(end_idx);
            let mut best_k: Option<usize> = None;
            let mut best_val = INF;
            for k in scan_lo..=k_upper {
                let prev = *dp.get(k, j - 1);
                if prev >= INF / 10.0 {
                    continue;
                }
                let cand = prev + edge_latency[row_base + k];
                if cand < best_val {
                    best_val = cand;
                    best_k = Some(k);
                }
            }

            if let Some(best_k_val) = best_k {
                if best_val < *dp.get(i, j) {
                    dp.set(i, j, best_val);
                    if let Some(parent_table) = parent.as_deref_mut() {
                        parent_table.set(i, j, best_k_val as i32);
                    }
                }
            }
        }
        return;
    }

    let i_mid = i_lo + (i_hi - i_lo) / 2;
    let end_idx = i_mid - 1;
    let k_upper = k_hi.min(end_idx);

    if k_lo > k_upper {
        if i_mid < i_hi {
            solve_layer_dc(
                i_mid + 1,
                i_hi,
                k_lo,
                k_hi,
                j,
                bottleneck_cap,
                dp,
                parent.as_deref_mut(),
                offsets,
                sizes,
                max_merge_span,
                edge_latency,
            );
        }
        return;
    }

    let scan_lo = dc_min_feasible_k_for_end(
        end_idx,
        k_lo,
        bottleneck_cap,
        offsets,
        sizes,
        max_merge_span,
        edge_latency,
    );
    if scan_lo > k_upper {
        if i_mid > i_lo {
            solve_layer_dc(
                i_lo,
                i_mid - 1,
                k_lo,
                k_hi,
                j,
                bottleneck_cap,
                dp,
                parent.as_deref_mut(),
                offsets,
                sizes,
                max_merge_span,
                edge_latency,
            );
        }
        if i_mid < i_hi && scan_lo <= k_hi {
            solve_layer_dc(
                i_mid + 1,
                i_hi,
                scan_lo,
                k_hi,
                j,
                bottleneck_cap,
                dp,
                parent.as_deref_mut(),
                offsets,
                sizes,
                max_merge_span,
                edge_latency,
            );
        }
        return;
    }

    let mut best_k: Option<usize> = None;
    let mut best_val = INF;
    let row_base = dc_row_base(end_idx);
    for k in scan_lo..=k_upper {
        let prev = *dp.get(k, j - 1);
        if prev >= INF / 10.0 {
            continue;
        }
        let w = edge_latency[row_base + k];
        let cand = prev + w;
        if cand < best_val {
            best_val = cand;
            best_k = Some(k);
        }
    }

    if let Some(best_k_val) = best_k {
        if best_val < *dp.get(i_mid, j) {
            dp.set(i_mid, j, best_val);
            if let Some(parent_table) = parent.as_deref_mut() {
                parent_table.set(i_mid, j, best_k_val as i32);
            }
        }
    }

    let left_hi = best_k.unwrap_or(k_upper);
    let right_lo = best_k.unwrap_or(scan_lo);
    if i_mid > i_lo {
        solve_layer_dc(
            i_lo,
            i_mid - 1,
            k_lo,
            left_hi,
            j,
            bottleneck_cap,
            dp,
            parent.as_deref_mut(),
            offsets,
            sizes,
            max_merge_span,
            edge_latency,
        );
    }
    if i_mid < i_hi {
        solve_layer_dc(
            i_mid + 1,
            i_hi,
            right_lo,
            k_hi,
            j,
            bottleneck_cap,
            dp,
            parent.as_deref_mut(),
            offsets,
            sizes,
            max_merge_span,
            edge_latency,
        );
    }
}

fn solve_all_layers_dc(
    n: usize,
    k_cap: usize,
    bottleneck_cap: f64,
    dp: &mut FlatTable<f64>,
    mut parent: Option<&mut FlatTable<i32>>,
    offsets: &[i64],
    sizes: &[i64],
    max_merge_span: i64,
    edge_latency: &[f64],
) {
    for j in 1..=k_cap {
        if j <= n {
            solve_layer_dc(
                j,
                n,
                j - 1,
                n - 1,
                j,
                bottleneck_cap,
                dp,
                parent.as_deref_mut(),
                offsets,
                sizes,
                max_merge_span,
                edge_latency,
            );
        }
    }
}

fn greedy_feasible(
    n: usize,
    request_cap: usize,
    b_cap: f64,
    edge_latency: &[f64],
    offsets: &[i64],
    sizes: &[i64],
    max_merge_span: i64,
) -> bool {
    let mut groups = 0usize;
    let mut start = 0usize;

    while start < n {
        groups += 1;
        if groups > request_cap {
            return false;
        }

        let mut end = start;
        while end + 1 < n {
            if !segment_span_ok(offsets, sizes, start, end + 1, max_merge_span) {
                break;
            }
            let w = edge_latency[triangular_edge_index(end + 1, start)];
            if w <= b_cap + EPS {
                end += 1;
            } else {
                break;
            }
        }

        start = end + 1;
    }

    groups <= request_cap
}

fn collect_candidate_b_values(
    ctx: &PlannerContext,
    min_bottleneck: f64,
    max_bottleneck: f64,
) -> Vec<f64> {
    let mut candidates = Vec::with_capacity(ctx.n * (ctx.n + 1) / 2);

    for end_idx in 0..ctx.n {
        let mut start_idx = end_idx as i64;
        while start_idx >= 0 {
            let start = start_idx as usize;
            if !segment_span_ok(ctx.offsets, ctx.sizes, start, end_idx, ctx.max_merge_span) {
                break;
            }
            let w = ctx.edge_latency[triangular_edge_index(end_idx, start)];
            if w < min_bottleneck - EPS {
                start_idx -= 1;
                continue;
            }
            if w > max_bottleneck + EPS {
                break;
            }
            candidates.push(w);
            start_idx -= 1;
        }
    }

    candidates.sort_by(|a, b| a.total_cmp(b));
    candidates.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    if ctx.k_cap < ctx.n && !candidates.is_empty() {
        let mut lo = 0usize;
        let mut hi = candidates.len() - 1;
        let mut first_feasible = hi;

        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            if greedy_feasible(
                ctx.n,
                ctx.k_cap,
                candidates[mid],
                ctx.edge_latency,
                ctx.offsets,
                ctx.sizes,
                ctx.max_merge_span,
            ) {
                first_feasible = mid;
                if mid == 0 {
                    break;
                }
                hi = mid - 1;
            } else {
                lo = mid + 1;
            }
        }

        if first_feasible > 0 {
            candidates.drain(0..first_feasible);
        }
    }

    candidates
}

fn run_inner_dp_linear_with_b(
    n: usize,
    k_cap: usize,
    offsets: &[i64],
    sizes: &[i64],
    cost_model: &CostModel,
    max_merge_span: i64,
    b_cap: f64,
    dp: &mut FlatTable<f64>,
) {
    let request_overhead = cost_model.latency_per_request;
    let request_slope = cost_model.latency_per_byte.max(0.0);

    if request_slope < EPS {
        if request_overhead > b_cap + EPS {
            return;
        }

        for j in 1..=k_cap {
            let mut dq: VecDeque<(usize, f64)> = VecDeque::new();

            for i in 1..=n {
                let k = i - 1;
                let prev = *dp.get(k, j - 1);
                if k >= j - 1 && prev < INF / 10.0 {
                    while let Some((_, back_val)) = dq.back() {
                        if *back_val >= prev - EPS {
                            dq.pop_back();
                        } else {
                            break;
                        }
                    }
                    dq.push_back((k, prev));
                }

                let end = (offsets[i - 1] + sizes[i - 1]) as f64;
                let min_start = if max_merge_span > 0 {
                    end - max_merge_span as f64
                } else {
                    f64::NEG_INFINITY
                };
                while let Some((front_k, _)) = dq.front() {
                    if (offsets[*front_k] as f64) < min_start - EPS {
                        dq.pop_front();
                    } else {
                        break;
                    }
                }

                if let Some((_, best_prefix_cost)) = dq.front() {
                    let cand = *best_prefix_cost + request_overhead;
                    if cand < *dp.get(i, j) {
                        dp.set(i, j, cand);
                    }
                }
            }
        }
        return;
    }

    let mut max_span = (b_cap - request_overhead) / request_slope;
    if max_merge_span > 0 {
        max_span = max_span.min(max_merge_span as f64);
    }
    if max_span < -EPS {
        return;
    }

    for j in 1..=k_cap {
        let mut dq: VecDeque<(usize, f64)> = VecDeque::new();

        for i in 1..=n {
            let k = i - 1;
            let prev = *dp.get(k, j - 1);
            if k >= j - 1 && prev < INF / 10.0 {
                let shifted = prev - request_slope * offsets[k] as f64;
                while let Some((_, back_val)) = dq.back() {
                    if *back_val >= shifted - EPS {
                        dq.pop_back();
                    } else {
                        break;
                    }
                }
                dq.push_back((k, shifted));
            }

            let end = (offsets[i - 1] + sizes[i - 1]) as f64;
            let min_start = end - max_span;

            while let Some((front_k, _)) = dq.front() {
                if (offsets[*front_k] as f64) < min_start - EPS {
                    dq.pop_front();
                } else {
                    break;
                }
            }

            if let Some((_, best_shifted)) = dq.front() {
                let cand = request_overhead + request_slope * end + *best_shifted;
                if cand < *dp.get(i, j) {
                    dp.set(i, j, cand);
                }
            }
        }
    }
}

fn run_inner_dp_linear_with_b_parent(
    n: usize,
    k_cap: usize,
    offsets: &[i64],
    sizes: &[i64],
    cost_model: &CostModel,
    max_merge_span: i64,
    b_cap: f64,
    dp: &mut FlatTable<f64>,
    parent: &mut FlatTable<i32>,
) {
    let request_overhead = cost_model.latency_per_request;
    let request_slope = cost_model.latency_per_byte.max(0.0);

    if request_slope < EPS {
        if request_overhead > b_cap + EPS {
            return;
        }

        for j in 1..=k_cap {
            let mut dq: VecDeque<(usize, f64)> = VecDeque::new();

            for i in 1..=n {
                let k = i - 1;
                let prev = *dp.get(k, j - 1);
                if k >= j - 1 && prev < INF / 10.0 {
                    while let Some((_, back_val)) = dq.back() {
                        if *back_val >= prev - EPS {
                            dq.pop_back();
                        } else {
                            break;
                        }
                    }
                    dq.push_back((k, prev));
                }

                let end = (offsets[i - 1] + sizes[i - 1]) as f64;
                let min_start = if max_merge_span > 0 {
                    end - max_merge_span as f64
                } else {
                    f64::NEG_INFINITY
                };
                while let Some((front_k, _)) = dq.front() {
                    if (offsets[*front_k] as f64) < min_start - EPS {
                        dq.pop_front();
                    } else {
                        break;
                    }
                }

                if let Some((best_k, best_prefix_cost)) = dq.front() {
                    let cand = *best_prefix_cost + request_overhead;
                    if cand < *dp.get(i, j) {
                        dp.set(i, j, cand);
                        parent.set(i, j, *best_k as i32);
                    }
                }
            }
        }
        return;
    }

    let mut max_span = (b_cap - request_overhead) / request_slope;
    if max_merge_span > 0 {
        max_span = max_span.min(max_merge_span as f64);
    }
    if max_span < -EPS {
        return;
    }

    for j in 1..=k_cap {
        let mut dq: VecDeque<(usize, f64)> = VecDeque::new();

        for i in 1..=n {
            let k = i - 1;
            let prev = *dp.get(k, j - 1);
            if k >= j - 1 && prev < INF / 10.0 {
                let shifted = prev - request_slope * offsets[k] as f64;
                while let Some((_, back_val)) = dq.back() {
                    if *back_val >= shifted - EPS {
                        dq.pop_back();
                    } else {
                        break;
                    }
                }
                dq.push_back((k, shifted));
            }

            let end = (offsets[i - 1] + sizes[i - 1]) as f64;
            let min_start = end - max_span;

            while let Some((front_k, _)) = dq.front() {
                if (offsets[*front_k] as f64) < min_start - EPS {
                    dq.pop_front();
                } else {
                    break;
                }
            }

            if let Some((best_k, best_shifted)) = dq.front() {
                let cand = request_overhead + request_slope * end + *best_shifted;
                if cand < *dp.get(i, j) {
                    dp.set(i, j, cand);
                    parent.set(i, j, *best_k as i32);
                }
            }
        }
    }
}

fn run_inner_dp_general_with_b(
    n: usize,
    k_cap: usize,
    offsets: &[i64],
    sizes: &[i64],
    max_merge_span: i64,
    edge_latency: &[f64],
    b_cap: f64,
    dp: &mut FlatTable<f64>,
) {
    for i in 1..=n {
        let j_max = i.min(k_cap);
        for j in 1..=j_max {
            let mut best = *dp.get(i, j);
            for k in (j - 1)..i {
                let prev = *dp.get(k, j - 1);
                if prev >= INF / 10.0 {
                    continue;
                }
                if !segment_span_ok(offsets, sizes, k, i - 1, max_merge_span) {
                    continue;
                }
                let w = edge_latency[triangular_edge_index(i - 1, k)];
                if w > b_cap + EPS {
                    continue;
                }
                let cand = prev + w;
                if cand < best {
                    best = cand;
                }
            }
            if best < *dp.get(i, j) {
                dp.set(i, j, best);
            }
        }
    }
}

fn run_inner_dp_general_with_b_parent(
    n: usize,
    k_cap: usize,
    offsets: &[i64],
    sizes: &[i64],
    max_merge_span: i64,
    edge_latency: &[f64],
    b_cap: f64,
    dp: &mut FlatTable<f64>,
    parent: &mut FlatTable<i32>,
) {
    for i in 1..=n {
        let j_max = i.min(k_cap);
        for j in 1..=j_max {
            let mut best = *dp.get(i, j);
            let mut best_k = -1;
            for k in (j - 1)..i {
                let prev = *dp.get(k, j - 1);
                if prev >= INF / 10.0 {
                    continue;
                }
                if !segment_span_ok(offsets, sizes, k, i - 1, max_merge_span) {
                    continue;
                }
                let w = edge_latency[triangular_edge_index(i - 1, k)];
                if w > b_cap + EPS {
                    continue;
                }
                let cand = prev + w;
                if cand < best {
                    best = cand;
                    best_k = k as i32;
                }
            }
            if best_k >= 0 {
                dp.set(i, j, best);
                parent.set(i, j, best_k);
            }
        }
    }
}

#[derive(Default)]
struct WQSSolution {
    feasible: bool,
    penalized: f64,
    total_work: f64,
    used_requests: usize,
    parent: Vec<i32>,
}

fn solve_piecewise_penalized_with_b(
    n: usize,
    edge_latency: &[f64],
    b_cap: f64,
    lambda_penalty: f64,
    with_parent: bool,
) -> WQSSolution {
    let mut out = WQSSolution {
        feasible: false,
        penalized: INF,
        total_work: INF,
        used_requests: usize::MAX / 4,
        parent: Vec::new(),
    };
    if n == 0 {
        return out;
    }

    let mut dp = vec![INF; n + 1];
    let mut cnt = vec![usize::MAX / 4; n + 1];
    let mut parent = if with_parent {
        vec![-1; n + 1]
    } else {
        Vec::new()
    };
    dp[0] = 0.0;
    cnt[0] = 0;

    for i in 1..=n {
        let mut best = INF;
        let mut best_cnt = usize::MAX / 4;
        let mut best_k = -1_i32;

        for k in (0..i).rev() {
            let w = edge_latency[triangular_edge_index(i - 1, k)];
            if w > b_cap + EPS {
                break;
            }
            if dp[k] >= INF / 10.0 {
                continue;
            }

            let cand = dp[k] + w + lambda_penalty;
            let cand_cnt = cnt[k] + 1;
            if cand < best - EPS || ((cand - best).abs() <= EPS && cand_cnt < best_cnt) {
                best = cand;
                best_cnt = cand_cnt;
                best_k = k as i32;
            }
        }

        if best < INF / 10.0 {
            dp[i] = best;
            cnt[i] = best_cnt;
            if with_parent {
                parent[i] = best_k;
            }
        }
    }

    if dp[n] >= INF / 10.0 {
        return out;
    }

    out.feasible = true;
    out.penalized = dp[n];
    out.used_requests = cnt[n];
    out.total_work = dp[n] - lambda_penalty * cnt[n] as f64;
    if with_parent {
        out.parent = parent;
    }
    out
}

fn solve_piecewise_wqs_minw_with_b(
    n: usize,
    request_cap: usize,
    edge_latency: &[f64],
    b_cap: f64,
    need_parent: bool,
) -> WQSSolution {
    let base = solve_piecewise_penalized_with_b(n, edge_latency, b_cap, 0.0, false);
    if !base.feasible {
        return base;
    }
    if base.used_requests <= request_cap {
        if !need_parent {
            return base;
        }
        return solve_piecewise_penalized_with_b(n, edge_latency, b_cap, 0.0, true);
    }

    let mut lo = 0.0_f64;
    let mut hi = 1.0_f64;
    let mut hi_sol = base;
    const MAX_LAMBDA: f64 = 1e8;
    while hi <= MAX_LAMBDA {
        hi_sol = solve_piecewise_penalized_with_b(n, edge_latency, b_cap, hi, false);
        if !hi_sol.feasible || hi_sol.used_requests <= request_cap {
            break;
        }
        lo = hi;
        hi *= 2.0;
    }
    if !hi_sol.feasible || hi_sol.used_requests > request_cap {
        return WQSSolution::default();
    }

    let mut best_feasible = hi_sol;
    let mut best_lambda = hi;
    for _ in 0..45 {
        let mid = 0.5 * (lo + hi);
        let mid_sol = solve_piecewise_penalized_with_b(n, edge_latency, b_cap, mid, false);
        if !mid_sol.feasible || mid_sol.used_requests > request_cap {
            lo = mid;
            continue;
        }
        if mid_sol.total_work < best_feasible.total_work - EPS
            || ((mid_sol.total_work - best_feasible.total_work).abs() <= EPS
                && mid_sol.used_requests < best_feasible.used_requests)
        {
            best_feasible = mid_sol;
            best_lambda = mid;
        }
        hi = mid;
    }

    if !need_parent {
        return best_feasible;
    }

    let mut with_parent =
        solve_piecewise_penalized_with_b(n, edge_latency, b_cap, best_lambda, true);
    if with_parent.feasible
        && with_parent.used_requests <= request_cap
        && !with_parent.parent.is_empty()
    {
        return with_parent;
    }

    let mut lambda = best_lambda;
    for _ in 0..8 {
        lambda = f64::from_bits(lambda.to_bits() + 1);
        with_parent = solve_piecewise_penalized_with_b(n, edge_latency, b_cap, lambda, true);
        if with_parent.feasible
            && with_parent.used_requests <= request_cap
            && !with_parent.parent.is_empty()
        {
            return with_parent;
        }
    }

    WQSSolution::default()
}

struct SolveState {
    best_obj: f64,
    best_plan: Vec<IORequest>,
}

fn evaluate_objective_at_b(
    ctx: &PlannerContext,
    b_cap: f64,
    dp: &mut FlatTable<f64>,
    solve: &mut SolveState,
) -> f64 {
    if select_subproblem_solver(ctx) == SubproblemSolverKind::PiecewiseWqs {
        let wqs = solve_piecewise_wqs_minw_with_b(ctx.n, ctx.k_cap, ctx.edge_latency, b_cap, false);
        if !wqs.feasible || wqs.total_work >= INF / 10.0 {
            return INF;
        }
        let obj = ctx.coeff_sum * wqs.total_work + ctx.coeff_bottleneck * b_cap;
        if obj < solve.best_obj - EPS {
            solve.best_obj = obj;
        }
        return obj;
    }

    dp.fill(INF);
    dp.set(0, 0, 0.0);

    match select_subproblem_solver(ctx) {
        SubproblemSolverKind::LinearFastPath => {
            run_inner_dp_linear_with_b(
                ctx.n,
                ctx.k_cap,
                ctx.offsets,
                ctx.sizes,
                ctx.cost_model,
                ctx.max_merge_span,
                b_cap,
                dp,
            );
        }
        SubproblemSolverKind::DivideAndConquer => {
            solve_all_layers_dc(
                ctx.n,
                ctx.k_cap,
                b_cap,
                dp,
                None,
                ctx.offsets,
                ctx.sizes,
                ctx.max_merge_span,
                ctx.edge_latency,
            );
        }
        SubproblemSolverKind::PlainDP => {
            run_inner_dp_general_with_b(
                ctx.n,
                ctx.k_cap,
                ctx.offsets,
                ctx.sizes,
                ctx.max_merge_span,
                ctx.edge_latency,
                b_cap,
                dp,
            );
        }
        SubproblemSolverKind::PiecewiseWqs => unreachable!(),
    }

    let mut best_obj_this_b = INF;

    if ctx.exact_requests {
        let w = *dp.get(ctx.n, ctx.k_cap);
        if w < INF / 10.0 {
            best_obj_this_b = ctx.coeff_sum * w + ctx.coeff_bottleneck * b_cap;
        }
    } else {
        for j in 1..=ctx.k_cap {
            let w = *dp.get(ctx.n, j);
            if w >= INF / 10.0 {
                continue;
            }
            best_obj_this_b = best_obj_this_b.min(ctx.coeff_sum * w + ctx.coeff_bottleneck * b_cap);
        }
    }

    if best_obj_this_b < solve.best_obj - EPS {
        solve.best_obj = best_obj_this_b;
    }

    best_obj_this_b
}

fn materialize_plan_at_b(
    ctx: &PlannerContext,
    b_cap: f64,
    dp: &mut FlatTable<f64>,
    parent: &mut FlatTable<i32>,
    solve: &mut SolveState,
) -> bool {
    if select_subproblem_solver(ctx) == SubproblemSolverKind::PiecewiseWqs {
        let wqs = solve_piecewise_wqs_minw_with_b(ctx.n, ctx.k_cap, ctx.edge_latency, b_cap, true);
        if !wqs.feasible || wqs.parent.is_empty() || wqs.used_requests == 0 {
            return false;
        }

        let mut requests = Vec::with_capacity(wqs.used_requests);
        let mut i = ctx.n;
        while i > 0 {
            let prev = wqs.parent[i];
            if prev < 0 || prev as usize >= i {
                return false;
            }
            let prev = prev as usize;
            let request_start = ctx.offsets[prev];
            let request_end = ctx.offsets[i - 1] + ctx.sizes[i - 1];
            requests.push(IORequest::new(request_start, request_end - request_start));
            i = prev;
        }
        requests.reverse();
        if requests.is_empty() || requests.len() > ctx.k_cap {
            return false;
        }

        solve.best_obj = ctx.coeff_sum * wqs.total_work + ctx.coeff_bottleneck * b_cap;
        solve.best_plan = requests;
        return true;
    }

    dp.fill(INF);
    parent.fill(-1);
    dp.set(0, 0, 0.0);

    match select_subproblem_solver(ctx) {
        SubproblemSolverKind::LinearFastPath => {
            run_inner_dp_linear_with_b_parent(
                ctx.n,
                ctx.k_cap,
                ctx.offsets,
                ctx.sizes,
                ctx.cost_model,
                ctx.max_merge_span,
                b_cap,
                dp,
                parent,
            );
        }
        SubproblemSolverKind::DivideAndConquer => {
            solve_all_layers_dc(
                ctx.n,
                ctx.k_cap,
                b_cap,
                dp,
                Some(parent),
                ctx.offsets,
                ctx.sizes,
                ctx.max_merge_span,
                ctx.edge_latency,
            );
        }
        SubproblemSolverKind::PlainDP => {
            run_inner_dp_general_with_b_parent(
                ctx.n,
                ctx.k_cap,
                ctx.offsets,
                ctx.sizes,
                ctx.max_merge_span,
                ctx.edge_latency,
                b_cap,
                dp,
                parent,
            );
        }
        SubproblemSolverKind::PiecewiseWqs => unreachable!(),
    }

    let mut best_obj_this_b = INF;
    let mut best_j = None;

    if ctx.exact_requests {
        let w = *dp.get(ctx.n, ctx.k_cap);
        if w < INF / 10.0 {
            best_obj_this_b = ctx.coeff_sum * w + ctx.coeff_bottleneck * b_cap;
            best_j = Some(ctx.k_cap);
        }
    } else {
        for j in 1..=ctx.k_cap {
            let w = *dp.get(ctx.n, j);
            if w >= INF / 10.0 {
                continue;
            }
            let obj = ctx.coeff_sum * w + ctx.coeff_bottleneck * b_cap;
            if obj < best_obj_this_b {
                best_obj_this_b = obj;
                best_j = Some(j);
            }
        }
    }

    let Some(best_j) = best_j else {
        return false;
    };

    solve.best_obj = best_obj_this_b;
    solve.best_plan = reconstruct_requests(ctx.n, best_j, parent, ctx.offsets, ctx.sizes);
    !solve.best_plan.is_empty()
}

fn solve_by_exact_b_enumeration(
    ctx: &PlannerContext,
    candidate_b: &[f64],
    init_best_obj: f64,
    init_plan: Vec<IORequest>,
    min_w: f64,
) -> SolveState {
    let mut solve = SolveState {
        best_obj: init_best_obj,
        best_plan: init_plan,
    };

    if candidate_b.is_empty() {
        return solve;
    }

    let mut dp = FlatTable::new(ctx.n + 1, ctx.k_cap + 1, INF);
    let mut best_b_for_plan: Option<f64> = None;
    let mut best_obj_seen = solve.best_obj;

    for &b_cap in candidate_b {
        if ctx.coeff_bottleneck > EPS
            && ctx.coeff_sum * min_w + ctx.coeff_bottleneck * b_cap > solve.best_obj + EPS
        {
            break;
        }

        let obj = evaluate_objective_at_b(ctx, b_cap, &mut dp, &mut solve);
        if obj < best_obj_seen - EPS {
            best_obj_seen = obj;
            best_b_for_plan = Some(b_cap);
        }
    }

    if let Some(b_cap) = best_b_for_plan {
        let mut parent = FlatTable::new(ctx.n + 1, ctx.k_cap + 1, -1);
        if !materialize_plan_at_b(ctx, b_cap, &mut dp, &mut parent, &mut solve) {
            return SolveState {
                best_obj: init_best_obj,
                best_plan: solve.best_plan,
            };
        }
    }

    solve
}

#[derive(Clone, Copy, Debug)]
struct EvalPoint {
    b: f64,
    w: f64,
}

#[derive(Clone, Copy, Debug)]
struct IntervalNode {
    l: usize,
    r: usize,
    lb: f64,
}

impl PartialEq for IntervalNode {
    fn eq(&self, other: &Self) -> bool {
        self.l == other.l && self.r == other.r && self.lb.to_bits() == other.lb.to_bits()
    }
}

impl Eq for IntervalNode {}

impl PartialOrd for IntervalNode {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for IntervalNode {
    fn cmp(&self, other: &Self) -> Ordering {
        // Min-heap by LB via reversed ordering.
        let lb_cmp = other.lb.total_cmp(&self.lb);
        if lb_cmp != Ordering::Equal {
            return lb_cmp;
        }
        // Wider interval first on tie.
        let width_cmp = (self.r - self.l).cmp(&(other.r - other.l));
        if width_cmp != Ordering::Equal {
            return width_cmp;
        }
        self.l.cmp(&other.l).then_with(|| self.r.cmp(&other.r))
    }
}

fn current_hi_idx(
    ctx: &PlannerContext,
    candidate_b: &[f64],
    active_left: usize,
    initial_right: usize,
    min_w: f64,
    best_obj: f64,
) -> Option<usize> {
    if active_left > initial_right {
        return None;
    }
    if ctx.coeff_bottleneck <= EPS || min_w >= INF / 10.0 || best_obj >= INF / 10.0 {
        return Some(initial_right);
    }

    let bottleneck_limit = (best_obj - ctx.coeff_sum * min_w) / ctx.coeff_bottleneck;
    let upper = candidate_b.partition_point(|&x| x <= bottleneck_limit + EPS);
    let idx = upper.saturating_sub(1).min(initial_right);
    if idx < active_left {
        None
    } else {
        Some(idx)
    }
}

fn interval_lb(
    points: &BTreeMap<usize, EvalPoint>,
    l: usize,
    r: usize,
    ctx: &PlannerContext,
) -> f64 {
    let Some(pl) = points.get(&l) else {
        return INF;
    };
    let Some(pr) = points.get(&r) else {
        return INF;
    };
    let w_right = pr.w.min(pl.w);
    ctx.coeff_sum * w_right + ctx.coeff_bottleneck * pl.b
}

fn is_open_adjacent_interval(
    points: &BTreeMap<usize, EvalPoint>,
    l: usize,
    r: usize,
    active_left: usize,
    active_right: usize,
) -> bool {
    if l < active_left || r > active_right || l >= r || r - l <= 1 {
        return false;
    }
    let mut iter = points.range(l..=r).map(|(k, _)| *k);
    matches!(
        (iter.next(), iter.next(), iter.next()),
        (Some(k1), Some(k2), None) if k1 == l && k2 == r
    )
}

fn push_interval_if_open(
    intervals: &mut BinaryHeap<IntervalNode>,
    points: &BTreeMap<usize, EvalPoint>,
    l: usize,
    r: usize,
    active_left: usize,
    active_right: usize,
    ctx: &PlannerContext,
) {
    if !is_open_adjacent_interval(points, l, r, active_left, active_right) {
        return;
    }
    intervals.push(IntervalNode {
        l,
        r,
        lb: interval_lb(points, l, r, ctx),
    });
}

fn add_point(
    points: &mut BTreeMap<usize, EvalPoint>,
    intervals: &mut BinaryHeap<IntervalNode>,
    candidate_b: &[f64],
    idx: usize,
    w: f64,
    active_left: usize,
    active_right: usize,
    ctx: &PlannerContext,
) {
    if let Some(existing) = points.get_mut(&idx) {
        existing.w = existing.w.min(w);
        return;
    }

    points.insert(
        idx,
        EvalPoint {
            b: candidate_b[idx],
            w,
        },
    );

    if let Some(prev) = points.range(..idx).next_back().map(|(k, _)| *k) {
        push_interval_if_open(intervals, points, prev, idx, active_left, active_right, ctx);
    }
    if let Some(next) = points
        .range((Excluded(idx), Unbounded))
        .next()
        .map(|(k, _)| *k)
    {
        push_interval_if_open(intervals, points, idx, next, active_left, active_right, ctx);
    }
}

fn refresh_best_interval(
    intervals: &mut BinaryHeap<IntervalNode>,
    points: &BTreeMap<usize, EvalPoint>,
    active_left: usize,
    active_right: usize,
    ctx: &PlannerContext,
) -> Option<IntervalNode> {
    while let Some(mut node) = intervals.peek().copied() {
        if !is_open_adjacent_interval(points, node.l, node.r, active_left, active_right) {
            intervals.pop();
            continue;
        }
        let lb_now = interval_lb(points, node.l, node.r, ctx);
        if (lb_now - node.lb).abs() > 1e-9 * lb_now.abs().max(1.0) {
            intervals.pop();
            node.lb = lb_now;
            intervals.push(node);
            continue;
        }
        return Some(node);
    }
    None
}

fn has_unvisited(visited: &[bool], active_left: usize, active_right: usize) -> bool {
    if active_right < active_left || active_right >= visited.len() {
        return false;
    }
    visited[active_left..=active_right].iter().any(|v| !*v)
}

fn certified_rel_err(
    intervals: &mut BinaryHeap<IntervalNode>,
    points: &BTreeMap<usize, EvalPoint>,
    visited: &[bool],
    active_left: usize,
    active_right: usize,
    ctx: &PlannerContext,
    best_obj: f64,
) -> f64 {
    if active_right < active_left || active_right >= visited.len() {
        return 0.0;
    }
    if !visited[active_left] || !visited[active_right] {
        return INF;
    }

    let Some(best_interval) =
        refresh_best_interval(intervals, points, active_left, active_right, ctx)
    else {
        return if has_unvisited(visited, active_left, active_right) {
            INF
        } else {
            0.0
        };
    };

    if best_interval.lb >= best_obj - EPS {
        return 0.0;
    }
    (best_obj - best_interval.lb) / best_interval.lb.max(EPS)
}

fn find_unvisited_idx(visited: &[bool], active_left: usize, active_right: usize) -> Option<usize> {
    if active_right < active_left || active_right >= visited.len() {
        return None;
    }
    let mid = active_left + (active_right - active_left) / 2;
    let mut l = mid as isize;
    let mut r = mid + 1;

    while l >= active_left as isize || r <= active_right {
        if l >= active_left as isize {
            let li = l as usize;
            if !visited[li] {
                return Some(li);
            }
        }
        if r <= active_right && !visited[r] {
            return Some(r);
        }
        l -= 1;
        r += 1;
    }
    None
}

struct AdaptiveState {
    visited: Vec<bool>,
    points: BTreeMap<usize, EvalPoint>,
    intervals: BinaryHeap<IntervalNode>,
    total_evals: usize,
    active_left: usize,
    best_b_for_plan: Option<f64>,
    best_obj_seen: f64,
}

fn eval_and_record(
    ctx: &PlannerContext,
    candidate_b: &[f64],
    initial_right: usize,
    idx: usize,
    active_right_now: usize,
    dp: &mut FlatTable<f64>,
    solve: &mut SolveState,
    state: &mut AdaptiveState,
) -> bool {
    if idx < state.active_left || idx > initial_right || state.visited[idx] {
        return false;
    }
    state.visited[idx] = true;
    state.total_evals += 1;

    let objective_value = evaluate_objective_at_b(ctx, candidate_b[idx], dp, solve);
    if objective_value < state.best_obj_seen - EPS {
        state.best_obj_seen = objective_value;
        state.best_b_for_plan = Some(candidate_b[idx]);
    }

    if ctx.coeff_sum > EPS && objective_value < INF / 10.0 {
        let work_star = (objective_value - ctx.coeff_bottleneck * candidate_b[idx]) / ctx.coeff_sum;
        add_point(
            &mut state.points,
            &mut state.intervals,
            candidate_b,
            idx,
            work_star,
            state.active_left,
            active_right_now,
            ctx,
        );
        true
    } else {
        if objective_value >= INF / 10.0 && idx >= state.active_left {
            state.active_left = idx.saturating_add(1);
        }
        false
    }
}

fn solve_by_adaptive_hull(
    ctx: &PlannerContext,
    candidate_b: &[f64],
    init_best_obj: f64,
    init_plan: Vec<IORequest>,
    min_w: f64,
) -> SolveState {
    let mut solve = SolveState {
        best_obj: init_best_obj,
        best_plan: init_plan,
    };
    if candidate_b.is_empty() {
        return solve;
    }

    let initial_right = candidate_b.len() - 1;
    let mut dp = FlatTable::new(ctx.n + 1, ctx.k_cap + 1, INF);
    let mut state = AdaptiveState {
        visited: vec![false; candidate_b.len()],
        points: BTreeMap::new(),
        intervals: BinaryHeap::new(),
        total_evals: 0,
        active_left: 0,
        best_b_for_plan: None,
        best_obj_seen: solve.best_obj,
    };

    let scaled_budget = ctx.k_cap + 8;
    let eval_budget = scaled_budget.clamp(APPROX_MIN_EVAL_BUDGET, APPROX_MAX_EVAL_BUDGET);
    let max_total_evals = candidate_b.len().min(eval_budget);

    let Some(mut active_right) = current_hi_idx(
        ctx,
        candidate_b,
        state.active_left,
        initial_right,
        min_w,
        solve.best_obj,
    ) else {
        return solve;
    };
    let _ = eval_and_record(
        ctx,
        candidate_b,
        initial_right,
        state.active_left,
        active_right,
        &mut dp,
        &mut solve,
        &mut state,
    );

    let Some(ar) = current_hi_idx(
        ctx,
        candidate_b,
        state.active_left,
        initial_right,
        min_w,
        solve.best_obj,
    ) else {
        return solve;
    };
    active_right = ar;
    if !state.visited[state.active_left] && state.total_evals < max_total_evals {
        let _ = eval_and_record(
            ctx,
            candidate_b,
            initial_right,
            state.active_left,
            active_right,
            &mut dp,
            &mut solve,
            &mut state,
        );
    }

    let Some(ar) = current_hi_idx(
        ctx,
        candidate_b,
        state.active_left,
        initial_right,
        min_w,
        solve.best_obj,
    ) else {
        return solve;
    };
    active_right = ar;
    if !state.visited[active_right] && state.total_evals < max_total_evals {
        let _ = eval_and_record(
            ctx,
            candidate_b,
            initial_right,
            active_right,
            active_right,
            &mut dp,
            &mut solve,
            &mut state,
        );
    }

    let mut converged = false;
    while state.total_evals < max_total_evals {
        let Some(ar) = current_hi_idx(
            ctx,
            candidate_b,
            state.active_left,
            initial_right,
            min_w,
            solve.best_obj,
        ) else {
            converged = true;
            break;
        };
        active_right = ar;

        if !state.visited[state.active_left] && state.total_evals < max_total_evals {
            let _ = eval_and_record(
                ctx,
                candidate_b,
                initial_right,
                state.active_left,
                active_right,
                &mut dp,
                &mut solve,
                &mut state,
            );
        }
        let Some(ar) = current_hi_idx(
            ctx,
            candidate_b,
            state.active_left,
            initial_right,
            min_w,
            solve.best_obj,
        ) else {
            converged = true;
            break;
        };
        active_right = ar;
        if !state.visited[active_right] && state.total_evals < max_total_evals {
            let _ = eval_and_record(
                ctx,
                candidate_b,
                initial_right,
                active_right,
                active_right,
                &mut dp,
                &mut solve,
                &mut state,
            );
        }
        let Some(ar) = current_hi_idx(
            ctx,
            candidate_b,
            state.active_left,
            initial_right,
            min_w,
            solve.best_obj,
        ) else {
            converged = true;
            break;
        };
        active_right = ar;

        let rel_err_now = certified_rel_err(
            &mut state.intervals,
            &state.points,
            &state.visited,
            state.active_left,
            active_right,
            ctx,
            solve.best_obj,
        );
        if rel_err_now <= APPROX_TARGET_RELATIVE_ERROR {
            converged = true;
            break;
        }

        if let Some(best_interval) = refresh_best_interval(
            &mut state.intervals,
            &state.points,
            state.active_left,
            active_right,
            ctx,
        ) {
            state.intervals.pop();
            if best_interval.r - best_interval.l > 1 {
                let mid = best_interval.l + (best_interval.r - best_interval.l) / 2;
                if !eval_and_record(
                    ctx,
                    candidate_b,
                    initial_right,
                    mid,
                    active_right,
                    &mut dp,
                    &mut solve,
                    &mut state,
                ) {
                    let mut left = mid.saturating_sub(1);
                    let mut right = mid + 1;
                    while left > best_interval.l || right < best_interval.r {
                        if left > best_interval.l
                            && eval_and_record(
                                ctx,
                                candidate_b,
                                initial_right,
                                left,
                                active_right,
                                &mut dp,
                                &mut solve,
                                &mut state,
                            )
                        {
                            break;
                        }
                        if right < best_interval.r
                            && eval_and_record(
                                ctx,
                                candidate_b,
                                initial_right,
                                right,
                                active_right,
                                &mut dp,
                                &mut solve,
                                &mut state,
                            )
                        {
                            break;
                        }
                        left = left.saturating_sub(1);
                        right += 1;
                    }
                }
            }
        } else if let Some(rescue) =
            find_unvisited_idx(&state.visited, state.active_left, active_right)
        {
            if state.total_evals < max_total_evals {
                let _ = eval_and_record(
                    ctx,
                    candidate_b,
                    initial_right,
                    rescue,
                    active_right,
                    &mut dp,
                    &mut solve,
                    &mut state,
                );
                continue;
            }
            converged = true;
            break;
        } else {
            converged = true;
            break;
        }
    }

    if !converged {
        if let Some(ar) = current_hi_idx(
            ctx,
            candidate_b,
            state.active_left,
            initial_right,
            min_w,
            solve.best_obj,
        ) {
            active_right = ar;
            if state.total_evals < candidate_b.len() {
                let _ = certified_rel_err(
                    &mut state.intervals,
                    &state.points,
                    &state.visited,
                    state.active_left,
                    active_right,
                    ctx,
                    solve.best_obj,
                ) <= APPROX_TARGET_RELATIVE_ERROR;
            }
        }
    }

    if let Some(b_cap) = state.best_b_for_plan {
        let mut parent = FlatTable::new(ctx.n + 1, ctx.k_cap + 1, -1);
        if !materialize_plan_at_b(ctx, b_cap, &mut dp, &mut parent, &mut solve) {
            return solve;
        }
    }

    solve
}

/// Finite-parallel planner for the objective
/// `T = W / P + (1 - 1/P) * B`.
///
/// The implementation keeps the exact planner but can optionally enable
/// search accelerations for larger workloads.
pub struct BalancedUBMerge {
    enable_fast_path_optimizations: bool,
    subproblem_optimization_mode: SubproblemOptimizationMode,
}

impl BalancedUBMerge {
    pub fn new() -> Self {
        Self {
            enable_fast_path_optimizations: true,
            subproblem_optimization_mode: SubproblemOptimizationMode::Auto,
        }
    }

    /// Enable or disable the internal fast-path optimizations used for larger
    /// candidate sets. Disabling them keeps exhaustive-$B$ search and the
    /// optimized divide-and-conquer DP backend, without adaptive-hull or WQS
    /// search accelerations.
    pub fn with_fast_path_optimizations(enable_fast_path_optimizations: bool) -> Self {
        Self {
            enable_fast_path_optimizations,
            subproblem_optimization_mode: if enable_fast_path_optimizations {
                SubproblemOptimizationMode::Auto
            } else {
                SubproblemOptimizationMode::DivideAndConquerOnly
            },
        }
    }

    /// Exhaustive-$B$ baseline with the plain O(n^2 K) capped-DP backend.
    pub fn with_plain_dp_baseline() -> Self {
        Self {
            enable_fast_path_optimizations: false,
            subproblem_optimization_mode: SubproblemOptimizationMode::PlainDP,
        }
    }

    /// Adaptive-search planner with the divide-and-conquer capped-DP backend,
    /// explicitly disabling WQS-based subproblem solves.
    pub fn with_adaptive_dc_only() -> Self {
        Self {
            enable_fast_path_optimizations: true,
            subproblem_optimization_mode: SubproblemOptimizationMode::DivideAndConquerOnly,
        }
    }

    /// Backward-compatible constructor retaining the older internal term.
    pub fn with_monotone_opt(enable_monotone_opt: bool) -> Self {
        Self::with_fast_path_optimizations(enable_monotone_opt)
    }

    fn use_adaptive_search(&self, coeff_sum: f64, n: usize) -> bool {
        coeff_sum > EPS && self.enable_fast_path_optimizations && n > APPROX_NPROBE_THRESHOLD
    }

    /// Run the full BalancedUB DP on a single chunk of ranges. Returns
    /// the list of merged Range-GETs for that chunk. This is the body
    /// of the original `generate_plan`, factored out so the caller can
    /// run it repeatedly on chunks of a larger (pre-merged) input.
    fn solve_chunk_dp(
        &self,
        offsets: Vec<i64>,
        sizes: Vec<i64>,
        cost_model: &CostModel,
    ) -> Vec<IORequest> {
        let n = offsets.len();
        if n == 0 {
            return Vec::new();
        }
        let k_cap = cost_model.request_cap(n);

        let edge_latency = build_edge_latency_matrix(&offsets, &sizes, cost_model);
        let (coeff_sum, coeff_bottleneck) =
            compute_objective_coefficients(cost_model.concurrent_parallelism);
        let has_waste_constraint = cost_model.max_waste_ratio < 1.0;

        let ctx = PlannerContext {
            n,
            k_cap,
            offsets: &offsets,
            sizes: &sizes,
            cost_model,
            edge_latency: &edge_latency,
            coeff_sum,
            coeff_bottleneck,
            exact_requests: cost_model.exact_requests && k_cap > 0,
            max_merge_span: if cost_model.max_merge_span > 0 {
                cost_model.max_merge_span
            } else {
                0
            },
            is_linear_model: !cost_model.use_piecewise_linear() && !has_waste_constraint,
            is_convex_piecewise: is_convex_piecewise_linear(cost_model) && !has_waste_constraint,
            use_wqs_linear_kcap: !cost_model.exact_requests
                && is_large_k_for_wqs(n, k_cap)
                && !cost_model.use_piecewise_linear()
                && coeff_sum > EPS,
            use_wqs_piecewise_kcap: !cost_model.exact_requests
                && is_large_k_for_wqs(n, k_cap)
                && cost_model.use_piecewise_linear()
                && coeff_sum > EPS
                && self.enable_fast_path_optimizations,
            subproblem_optimization_mode: self.subproblem_optimization_mode,
        };

        let warm = compute_warm_start(&ctx);
        if coeff_bottleneck < EPS {
            return warm.best_plan;
        }

        let mut max_single = 0.0_f64;
        for i in 0..n {
            max_single = max_single.max(edge_latency[triangular_edge_index(i, i)]);
        }

        let mut max_bottleneck = INF;
        if coeff_bottleneck > EPS && warm.min_w < INF / 10.0 {
            max_bottleneck = (warm.best_obj - coeff_sum * warm.min_w) / coeff_bottleneck;
        }

        let candidate_b = collect_candidate_b_values(&ctx, max_single, max_bottleneck);
        if candidate_b.is_empty() {
            return warm.best_plan;
        }

        let probe_t = std::env::var("WARP_PROBE").is_ok();
        let t_outer = if probe_t {
            Some(std::time::Instant::now())
        } else {
            None
        };
        let use_adaptive = self.use_adaptive_search(coeff_sum, n);
        let solve = if use_adaptive {
            solve_by_adaptive_hull(
                &ctx,
                &candidate_b,
                warm.best_obj,
                warm.best_plan,
                warm.min_w,
            )
        } else {
            solve_by_exact_b_enumeration(
                &ctx,
                &candidate_b,
                warm.best_obj,
                warm.best_plan,
                warm.min_w,
            )
        };
        if let Some(t) = t_outer {
            let us = t.elapsed().as_micros();
            eprintln!(
                "[outer_us] kind={} n={} k_cap={} num_candidate_b={} us={}",
                if use_adaptive {
                    "adaptive_hull"
                } else {
                    "exact_b"
                },
                n,
                ctx.k_cap,
                candidate_b.len(),
                us
            );
        }

        if solve.best_plan.is_empty() {
            return requests_from_ordered_spans(&offsets, &sizes);
        }
        solve.best_plan
    }
}

impl Default for BalancedUBMerge {
    fn default() -> Self {
        Self::new()
    }
}

/// Public alias used as the paper-facing main method name.
pub type VecioMergePlanner = BalancedUBMerge;

/// Public alias with a more descriptive name for the finite-parallel planner.
pub type FiniteParallelMerge = BalancedUBMerge;

impl MergeStrategy for BalancedUBMerge {
    fn generate_plan(
        &self,
        cluster_ids: &[i32],
        cluster_offsets: &HashMap<i32, i64>,
        cluster_sizes: &HashMap<i32, i64>,
        cost_model: &CostModel,
    ) -> IOPlan {
        if cluster_ids.is_empty() {
            return IOPlan::new(vec![], IOMetrics::default());
        }

        let layout = collect_ordered_cluster_layout(cluster_ids, cluster_offsets, cluster_sizes);
        let raw_n = layout.offsets.len();

        // Stage 1: greedy closest-pair pre-merge, threshold-gated.
        // Merges any adjacent pair whose gap is <= 64 KB; that's a
        // "free" merge (the merged range only costs the gap as extra
        // S3 bytes, bounded). The 1 MB merged-size cap prevents a
        // chain of micro-clusters from coalescing into a multi-MB
        // super-request that would dominate the DP's bandwidth-cost
        // term and inflate per-query waste. Leaves us with ranges
        // separated by gaps > 64 KB which are not worth merging
        // under our cost model.
        // gap_max derived from the cost model: a single merge of two
        // adjacent ranges saves alpha/P (one per-request overhead
        // amortised over P-way concurrency) and pays (1-1/P)*beta*gap.
        // Setting gap = alpha / (beta * (P-1)) puts cheap_pre_merge
        // exactly at the break-even point. At P=128, alpha=28ms,
        // beta=1e-5 ms/B that's ~22 KiB; at P=16 it's ~187 KiB. The
        // previous fixed 64 KiB compromised both regimes.
        // PREMERGE_MAX_MERGED_SIZE stays a hard cap to prevent a
        // chain of micro-clusters from coalescing into a multi-MB
        // super-request that would dominate the DP's bandwidth-cost
        // term and inflate per-query waste.
        const PREMERGE_MAX_MERGED_SIZE: i64 = 1024 * 1024;
        let premerge_max_gap: i64 = compute_adaptive_premerge_gap(cost_model);
        let premerge_max_size = if cost_model.max_merge_span > 0 {
            PREMERGE_MAX_MERGED_SIZE.min(cost_model.max_merge_span)
        } else {
            PREMERGE_MAX_MERGED_SIZE
        };
        let use_premerge = self.enable_fast_path_optimizations
            && !cost_model.exact_requests
            && cost_model.max_waste_ratio >= 1.0
            && raw_n > APPROX_NPROBE_THRESHOLD;
        let (merged_offsets, merged_sizes) = if use_premerge {
            cheap_pre_merge(
                layout.offsets,
                layout.sizes,
                premerge_max_gap,
                premerge_max_size,
            )
        } else {
            (layout.offsets, layout.sizes)
        };
        let premerged_n = merged_offsets.len();
        if std::env::var("WARP_PROBE").is_ok() && raw_n != premerged_n {
            eprintln!(
                "[pre_merge] raw_n={} -> merged_n={}  (max_gap={} B, max_size={} B, P={:.1})",
                raw_n,
                premerged_n,
                premerge_max_gap,
                premerge_max_size,
                cost_model.concurrent_parallelism
            );
        }

        // Stage 2: if the pre-merged input is still too large for the
        // DP's O(n^2) edge-latency precompute to be cheap, split into
        // chunks and DP each chunk independently. The chunk boundaries
        // fall on index boundaries (not gap boundaries) — cheap_pre_merge
        // already guarantees remaining inter-range gaps exceed the
        // break-even threshold, so cross-chunk merges are not promising
        // anyway. Per-chunk DP is O(m^2 K) with m <= DP_CHUNK_SIZE; total
        // planner wall time is O((n/m) * m^2 K) = O(n * m * K).
        //
        // 256 is the sweet spot for the SIFT/DEEP100M scale we care about:
        // m=256 gives ~8M ops/chunk at K=128, well under 5 ms in release;
        // 512 was 4x more work per chunk and pushed planner_us into the
        // 20-30 ms range when the post-cheap_pre_merge n stayed above ~1k.
        // 128 is too small: cross-chunk merges are missed and the planner
        // emits ~20% more GETs without commensurate latency wins.
        const DP_CHUNK_SIZE: usize = 256;
        let global_request_cap = cost_model.request_cap(premerged_n);
        // A non-empty chunk needs at least one output request.  When K is
        // smaller than the nominal chunk count, enlarge chunks just enough
        // that a global at-most-K allocation remains feasible.
        let max_chunk_size = if cost_model.max_requests > 0 {
            DP_CHUNK_SIZE.max(premerged_n.div_ceil(global_request_cap))
        } else {
            DP_CHUNK_SIZE
        };
        let chunks = split_into_chunks(merged_offsets, merged_sizes, max_chunk_size);
        let chunk_sizes = chunks
            .iter()
            .map(|(offsets, _)| offsets.len())
            .collect::<Vec<_>>();
        let chunk_request_caps = allocate_chunk_request_caps(&chunk_sizes, global_request_cap);
        if std::env::var("WARP_PROBE").is_ok() && chunks.len() > 1 {
            eprintln!(
                "[pre_merge] split into {} chunks of <= {} ranges each; \
                 global_k={} chunk_k={:?}",
                chunks.len(),
                max_chunk_size,
                global_request_cap,
                chunk_request_caps,
            );
        }

        let mut all_plan: Vec<IORequest> = Vec::new();
        for ((chunk_offsets, chunk_sizes), chunk_request_cap) in
            chunks.into_iter().zip(chunk_request_caps)
        {
            let mut chunk_cost_model = cost_model.clone();
            chunk_cost_model.max_requests = chunk_request_cap as i32;
            let chunk_plan = self.solve_chunk_dp(chunk_offsets, chunk_sizes, &chunk_cost_model);
            all_plan.extend(chunk_plan);
        }
        // A cap can be infeasible when request-span or waste constraints
        // prevent enough adjacent ranges from being merged.  In that case
        // solve_chunk_dp returns the minimum feasible plan and the caller
        // reports partial budget coverage instead of emitting an invalid I/O.

        let metrics = calculate_plan_metrics(&all_plan, cluster_ids, cluster_sizes, cost_model);
        IOPlan::new(all_plan, metrics)
    }

    fn name(&self) -> &str {
        "BalancedUB"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheduler::merge_strategy::sort_cluster_ids_by_offset;
    use crate::scheduler::serial_dp::SerialDP;
    use rand::seq::SliceRandom;
    use rand::{rngs::StdRng, Rng, SeedableRng};
    use std::time::Instant;

    fn basic_clusters() -> (Vec<i32>, HashMap<i32, i64>, HashMap<i32, i64>) {
        let cluster_ids = vec![1, 2, 3, 4];
        let mut offsets = HashMap::new();
        offsets.insert(1, 0);
        offsets.insert(2, 1_000);
        offsets.insert(3, 4_000);
        offsets.insert(4, 5_000);

        let mut sizes = HashMap::new();
        sizes.insert(1, 500);
        sizes.insert(2, 500);
        sizes.insert(3, 500);
        sizes.insert(4, 500);

        (cluster_ids, offsets, sizes)
    }

    #[test]
    fn test_balanced_ub_basic() {
        let (cluster_ids, offsets, sizes) = basic_clusters();
        let mut cost_model = CostModel::default();
        cost_model.latency_per_request = 100.0;
        cost_model.latency_per_byte = 0.001;
        cost_model.concurrent_parallelism = 16.0;
        cost_model.max_requests = 4;

        let plan =
            BalancedUBMerge::new().generate_plan(&cluster_ids, &offsets, &sizes, &cost_model);

        assert!(!plan.requests.is_empty());
        assert!(plan.requests.len() <= 4);
    }

    #[test]
    fn test_balanced_ub_respects_per_request_waste_ratio() {
        let cluster_ids = vec![1, 2, 3];
        let offsets = HashMap::from([(1, 0), (2, 1_000), (3, 2_000)]);
        let sizes = HashMap::from([(1, 100), (2, 100), (3, 100)]);

        let mut cost_model = CostModel::default();
        cost_model.latency_per_request = 100.0;
        cost_model.latency_per_byte = 0.0;
        cost_model.concurrent_parallelism = 64.0;
        cost_model.max_merge_span = 0;
        cost_model.max_requests = 3;
        cost_model.max_waste_ratio = 0.25;

        let bounded =
            BalancedUBMerge::new().generate_plan(&cluster_ids, &offsets, &sizes, &cost_model);
        assert_eq!(bounded.requests.len(), 3);
        assert!(bounded.metrics.waste_ratio <= 0.25 + EPS);

        cost_model.max_waste_ratio = 1.0;
        let unbounded =
            BalancedUBMerge::new().generate_plan(&cluster_ids, &offsets, &sizes, &cost_model);
        assert_eq!(unbounded.requests.len(), 1);
        assert!(unbounded.metrics.waste_ratio > 0.25);
    }

    #[test]
    fn test_balanced_ub_respects_exact_k() {
        let (cluster_ids, offsets, sizes) = basic_clusters();
        let mut cost_model = CostModel::default();
        cost_model.latency_per_request = 100.0;
        cost_model.latency_per_byte = 0.001;
        cost_model.concurrent_parallelism = 12.0;
        cost_model.max_requests = 2;
        cost_model.exact_requests = true;

        let plan =
            BalancedUBMerge::new().generate_plan(&cluster_ids, &offsets, &sizes, &cost_model);

        assert_eq!(plan.requests.len(), 2);
    }

    #[test]
    fn test_allocate_chunk_request_caps_shares_one_global_budget() {
        assert!(allocate_chunk_request_caps(&[], 128).is_empty());
        assert_eq!(allocate_chunk_request_caps(&[256], 128), vec![128]);
        assert_eq!(allocate_chunk_request_caps(&[129, 128], 128), vec![64, 64]);
        assert_eq!(allocate_chunk_request_caps(&[256, 256], 128), vec![64, 64]);

        let caps = allocate_chunk_request_caps(&[100, 50, 25], 20);
        assert_eq!(caps.iter().sum::<usize>(), 20);
        assert!(caps.iter().all(|&cap| cap >= 1));
        assert!(caps
            .iter()
            .zip([100, 50, 25])
            .all(|(&cap, chunk_size)| cap <= chunk_size));
    }

    #[test]
    fn test_balanced_ub_global_request_cap_across_chunks() {
        const K: usize = 128;
        for n in [255_usize, 256, 257, 512] {
            let cluster_ids = (0..n as i32).collect::<Vec<_>>();
            let offsets = cluster_ids
                .iter()
                .map(|&id| (id, i64::from(id) * 4 * 1024 * 1024))
                .collect::<HashMap<_, _>>();
            let sizes = cluster_ids
                .iter()
                .map(|&id| (id, 4 * 1024))
                .collect::<HashMap<_, _>>();
            let mut cost_model = CostModel::default();
            cost_model.latency_per_request = 28.0;
            cost_model.latency_per_byte = 1.0e-5;
            cost_model.concurrent_parallelism = 16.0;
            cost_model.max_waste_ratio = 99.0;
            cost_model.max_merge_span = 0;
            cost_model.max_requests = K as i32;
            for exact_requests in [false, true] {
                cost_model.exact_requests = exact_requests;
                let plan = BalancedUBMerge::new().generate_plan(
                    &cluster_ids,
                    &offsets,
                    &sizes,
                    &cost_model,
                );

                if exact_requests {
                    assert_eq!(
                        plan.requests.len(),
                        K,
                        "n={n} did not honor exact global K={K}"
                    );
                } else {
                    assert!(
                        plan.requests.len() <= K,
                        "n={n} emitted {} requests under K={K}",
                        plan.requests.len()
                    );
                }
            }
        }
    }

    #[test]
    fn test_balanced_ub_p1_matches_serial_objective() {
        let (cluster_ids, offsets, sizes) = basic_clusters();

        let mut cost_model = CostModel::default();
        cost_model.latency_per_request = 100.0;
        cost_model.latency_per_byte = 0.001;
        cost_model.concurrent_parallelism = 1.0;
        cost_model.max_requests = 3;
        cost_model.max_waste_ratio = 1.0;

        let bal = BalancedUBMerge::new().generate_plan(&cluster_ids, &offsets, &sizes, &cost_model);
        let dp = SerialDP::new().generate_plan(&cluster_ids, &offsets, &sizes, &cost_model);

        // P=1 => objective is pure W, should match serial DP optimum.
        assert!((bal.metrics.total_latency - dp.metrics.total_latency).abs() < 1e-6);
    }

    fn plan_objective(plan: &IOPlan, cm: &CostModel) -> f64 {
        let (coeff_sum, coeff_bottleneck) =
            compute_objective_coefficients(cm.concurrent_parallelism);
        let mut w = 0.0_f64;
        let mut b = 0.0_f64;
        for req in &plan.requests {
            let x = cm.request_latency_ms(req.length);
            w += x;
            b = b.max(x);
        }
        coeff_sum * w + coeff_bottleneck * b
    }

    fn brute_force_best_objective(
        offsets: &[i64],
        sizes: &[i64],
        cm: &CostModel,
        k_cap: usize,
        exact: bool,
    ) -> Option<f64> {
        let n = offsets.len();
        if n == 0 {
            return Some(0.0);
        }
        let (coeff_sum, coeff_bottleneck) =
            compute_objective_coefficients(cm.concurrent_parallelism);
        let mut best = f64::INFINITY;

        // Use iterative stack to keep code straightforward with CostModel access.
        #[derive(Clone, Copy)]
        struct State {
            start: usize,
            groups_used: usize,
            w: f64,
            b: f64,
        }

        let mut stack = vec![State {
            start: 0,
            groups_used: 0,
            w: 0.0,
            b: 0.0,
        }];

        while let Some(st) = stack.pop() {
            if st.start == n {
                let k = st.groups_used;
                let feasible = if exact {
                    k == k_cap
                } else {
                    k >= 1 && k <= k_cap
                };
                if feasible {
                    best = best.min(coeff_sum * st.w + coeff_bottleneck * st.b);
                }
                continue;
            }
            if st.groups_used >= k_cap {
                continue;
            }
            for end in (st.start..n).rev() {
                let bytes = (offsets[end] + sizes[end]) - offsets[st.start];
                if cm.max_merge_span > 0 && bytes > cm.max_merge_span {
                    continue;
                }
                let lat = cm.request_latency_ms(bytes);
                stack.push(State {
                    start: end + 1,
                    groups_used: st.groups_used + 1,
                    w: st.w + lat,
                    b: st.b.max(lat),
                });
            }
        }

        if best.is_finite() {
            Some(best)
        } else {
            None
        }
    }

    #[test]
    fn test_balanced_ub_matches_bruteforce_small_linear() {
        let mut rng = StdRng::seed_from_u64(0xC11F_F00D_u64);
        for n in 2..=8 {
            for _case_id in 0..25 {
                let cluster_ids: Vec<i32> = (0..n as i32).collect();
                let mut offsets = HashMap::new();
                let mut sizes = HashMap::new();
                let mut cur = 0_i64;
                for cid in &cluster_ids {
                    let sz = rng.random_range(4_096_i64..65_536_i64);
                    if *cid > 0 {
                        cur += rng.random_range(0_i64..131_072_i64);
                    }
                    offsets.insert(*cid, cur);
                    sizes.insert(*cid, sz);
                    cur += sz;
                }

                for &p in &[0.0_f64, 1.0, 3.0, 12.0] {
                    for k in 1..=n {
                        for &exact in &[false, true] {
                            let mut cm = CostModel::default();
                            cm.latency_model = "linear".to_string();
                            cm.latency_per_request = 120.0;
                            cm.latency_per_byte = 0.00002;
                            cm.concurrent_parallelism = p;
                            cm.max_waste_ratio = 1.0;
                            cm.max_requests = k as i32;
                            cm.exact_requests = exact;

                            let sorted = sort_cluster_ids_by_offset(&cluster_ids, &offsets);
                            let offs: Vec<i64> = sorted.iter().map(|c| offsets[c]).collect();
                            let szs: Vec<i64> = sorted.iter().map(|c| sizes[c]).collect();
                            let brute = brute_force_best_objective(&offs, &szs, &cm, k, exact);

                            let plan = BalancedUBMerge::new().generate_plan(
                                &cluster_ids,
                                &offsets,
                                &sizes,
                                &cm,
                            );
                            assert!(
                                !plan.requests.is_empty(),
                                "balanced returned empty plan (n={n}, k={k}, exact={exact}, p={p})"
                            );
                            let got = plan_objective(&plan, &cm);
                            let Some(opt) = brute else {
                                panic!("bruteforce infeasible unexpectedly (n={n}, k={k}, exact={exact})");
                            };
                            let diff = (got - opt).abs();
                            assert!(
                                diff <= 1e-6 * opt.max(1.0),
                                "objective mismatch n={n} k={k} exact={exact} p={p}: got={got}, opt={opt}, diff={diff}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn test_balanced_ub_matches_bruteforce_small_piecewise() {
        let mut rng = StdRng::seed_from_u64(0xBADA551_u64);
        for n in 2..=7 {
            for _case_id in 0..15 {
                let cluster_ids: Vec<i32> = (0..n as i32).collect();
                let mut offsets = HashMap::new();
                let mut sizes = HashMap::new();
                let mut cur = 0_i64;
                for cid in &cluster_ids {
                    let sz = rng.random_range(8_192_i64..96_000_i64);
                    if *cid > 0 {
                        cur += rng.random_range(0_i64..200_000_i64);
                    }
                    offsets.insert(*cid, cur);
                    sizes.insert(*cid, sz);
                    cur += sz;
                }

                let mut cm = CostModel::default();
                cm.latency_model = "piecewise_linear".to_string();
                cm.latency_pw_breaks_bytes = vec![256 * 1024, 2 * 1024 * 1024];
                cm.latency_pw_overheads_ms = vec![95.0, 120.0, 180.0];
                cm.latency_pw_rates_ms_per_byte = vec![0.00002, 0.00002, 0.00004];
                cm.concurrent_parallelism = 10.0;
                cm.max_waste_ratio = 1.0;

                for k in 1..=n {
                    cm.max_requests = k as i32;
                    cm.exact_requests = false;
                    let sorted = sort_cluster_ids_by_offset(&cluster_ids, &offsets);
                    let offs: Vec<i64> = sorted.iter().map(|c| offsets[c]).collect();
                    let szs: Vec<i64> = sorted.iter().map(|c| sizes[c]).collect();
                    let opt = brute_force_best_objective(&offs, &szs, &cm, k, false).unwrap();
                    let got_plan =
                        BalancedUBMerge::new().generate_plan(&cluster_ids, &offsets, &sizes, &cm);
                    let got = plan_objective(&got_plan, &cm);
                    let diff = (got - opt).abs();
                    assert!(
                        diff <= 1e-6 * opt.max(1.0),
                        "piecewise objective mismatch n={n} k={k}: got={got}, opt={opt}, diff={diff}"
                    );
                }
            }
        }
    }

    #[test]
    fn test_balanced_ub_respects_max_merge_span() {
        let (cluster_ids, offsets, sizes) = basic_clusters();
        let mut cost_model = CostModel::default();
        cost_model.latency_per_request = 80.0;
        cost_model.latency_per_byte = 0.001;
        cost_model.concurrent_parallelism = 10.0;
        cost_model.max_requests = 4;
        cost_model.max_merge_span = 900; // any merge of adjacent clusters exceeds this

        let plan =
            BalancedUBMerge::new().generate_plan(&cluster_ids, &offsets, &sizes, &cost_model);
        assert!(!plan.requests.is_empty());
        assert!(plan.requests.iter().all(|r| r.length <= 900));

        let exact_plan = BalancedUBMerge::with_fast_path_optimizations(false).generate_plan(
            &cluster_ids,
            &offsets,
            &sizes,
            &cost_model,
        );
        assert!(!exact_plan.requests.is_empty());
        assert!(exact_plan.requests.iter().all(|r| r.length <= 900));
    }

    #[test]
    fn test_balanced_ub_approx_within_5pct_large_n() {
        let n = 64usize;
        let mut rng = StdRng::seed_from_u64(0xA11CE_u64);
        let cluster_ids: Vec<i32> = (0..n as i32).collect();
        let mut offsets = HashMap::new();
        let mut sizes = HashMap::new();

        let mut cur = 0_i64;
        for cid in &cluster_ids {
            let sz = rng.random_range(8_192_i64..96_000_i64);
            if *cid > 0 {
                cur += rng.random_range(0_i64..180_000_i64);
            }
            offsets.insert(*cid, cur);
            sizes.insert(*cid, sz);
            cur += sz;
        }

        let mut cm = CostModel::default();
        cm.latency_model = "linear".to_string();
        cm.latency_per_request = 120.0;
        cm.latency_per_byte = 0.00002;
        cm.concurrent_parallelism = 12.0;
        cm.max_waste_ratio = 1.0;
        cm.max_requests = 32;
        cm.exact_requests = false;

        let sorted = sort_cluster_ids_by_offset(&cluster_ids, &offsets);
        let offs: Vec<i64> = sorted.iter().map(|c| offsets[c]).collect();
        let szs: Vec<i64> = sorted.iter().map(|c| sizes[c]).collect();
        let edge = build_edge_latency_matrix(&offs, &szs, &cm);
        let (coeff_sum, coeff_bottleneck) =
            compute_objective_coefficients(cm.concurrent_parallelism);
        let ctx = PlannerContext {
            n,
            k_cap: 32,
            offsets: &offs,
            sizes: &szs,
            cost_model: &cm,
            edge_latency: &edge,
            coeff_sum,
            coeff_bottleneck,
            exact_requests: false,
            max_merge_span: if cm.max_merge_span > 0 {
                cm.max_merge_span
            } else {
                0
            },
            is_linear_model: !cm.use_piecewise_linear(),
            is_convex_piecewise: is_convex_piecewise_linear(&cm),
            use_wqs_linear_kcap: !cm.exact_requests
                && is_large_k_for_wqs(n, 32)
                && !cm.use_piecewise_linear()
                && coeff_sum > EPS,
            use_wqs_piecewise_kcap: !cm.exact_requests
                && is_large_k_for_wqs(n, 32)
                && cm.use_piecewise_linear()
                && coeff_sum > EPS,
            subproblem_optimization_mode: SubproblemOptimizationMode::Auto,
        };

        let warm = compute_warm_start(&ctx);
        let mut max_single = 0.0_f64;
        for i in 0..n {
            max_single = max_single.max(edge[triangular_edge_index(i, i)]);
        }
        let mut max_bottleneck = INF;
        if coeff_bottleneck > EPS && warm.min_w < INF / 10.0 {
            max_bottleneck = (warm.best_obj - coeff_sum * warm.min_w) / coeff_bottleneck;
        }
        let cands = collect_candidate_b_values(&ctx, max_single, max_bottleneck);
        assert!(!cands.is_empty());

        let approx = solve_by_adaptive_hull(
            &ctx,
            &cands,
            warm.best_obj,
            warm.best_plan.clone(),
            warm.min_w,
        );
        let exact =
            solve_by_exact_b_enumeration(&ctx, &cands, warm.best_obj, warm.best_plan, warm.min_w);
        assert!(!approx.best_plan.is_empty());
        assert!(!exact.best_plan.is_empty());

        let approx_obj =
            objective_from_requests(&approx.best_plan, &cm, coeff_sum, coeff_bottleneck);
        let exact_obj = objective_from_requests(&exact.best_plan, &cm, coeff_sum, coeff_bottleneck);
        assert!(
            approx_obj <= exact_obj * 1.05 + 1e-6,
            "approx obj too large: approx={approx_obj}, exact={exact_obj}"
        );
    }

    #[test]
    #[ignore]
    fn perf_balanced_ub_exact_vs_approx_large() {
        let n = 256usize;
        let mut rng = StdRng::seed_from_u64(0xFACE_B00C_u64);
        let cluster_ids: Vec<i32> = (0..n as i32).collect();
        let mut offsets = HashMap::new();
        let mut sizes = HashMap::new();

        let mut cur = 0_i64;
        for cid in &cluster_ids {
            let sz = rng.random_range(8_192_i64..96_000_i64);
            if *cid > 0 {
                cur += rng.random_range(0_i64..220_000_i64);
            }
            offsets.insert(*cid, cur);
            sizes.insert(*cid, sz);
            cur += sz;
        }

        let mut cm = CostModel::default();
        cm.latency_model = "linear".to_string();
        cm.latency_per_request = 120.0;
        cm.latency_per_byte = 0.00002;
        cm.concurrent_parallelism = 20.0;
        cm.max_waste_ratio = 1.0;
        cm.max_requests = 256;
        cm.exact_requests = false;

        let sorted = sort_cluster_ids_by_offset(&cluster_ids, &offsets);
        let offs: Vec<i64> = sorted.iter().map(|c| offsets[c]).collect();
        let szs: Vec<i64> = sorted.iter().map(|c| sizes[c]).collect();
        let edge = build_edge_latency_matrix(&offs, &szs, &cm);
        let (coeff_sum, coeff_bottleneck) =
            compute_objective_coefficients(cm.concurrent_parallelism);
        let ctx = PlannerContext {
            n,
            k_cap: 256,
            offsets: &offs,
            sizes: &szs,
            cost_model: &cm,
            edge_latency: &edge,
            coeff_sum,
            coeff_bottleneck,
            exact_requests: false,
            max_merge_span: if cm.max_merge_span > 0 {
                cm.max_merge_span
            } else {
                0
            },
            is_linear_model: !cm.use_piecewise_linear(),
            is_convex_piecewise: is_convex_piecewise_linear(&cm),
            use_wqs_linear_kcap: !cm.exact_requests
                && is_large_k_for_wqs(n, 256)
                && !cm.use_piecewise_linear()
                && coeff_sum > EPS,
            use_wqs_piecewise_kcap: !cm.exact_requests
                && is_large_k_for_wqs(n, 256)
                && cm.use_piecewise_linear()
                && coeff_sum > EPS,
            subproblem_optimization_mode: SubproblemOptimizationMode::Auto,
        };

        let warm = compute_warm_start(&ctx);
        let mut max_single = 0.0_f64;
        for i in 0..n {
            max_single = max_single.max(edge[triangular_edge_index(i, i)]);
        }
        let mut max_bottleneck = INF;
        if coeff_bottleneck > EPS && warm.min_w < INF / 10.0 {
            max_bottleneck = (warm.best_obj - coeff_sum * warm.min_w) / coeff_bottleneck;
        }
        let cands = collect_candidate_b_values(&ctx, max_single, max_bottleneck);
        assert!(!cands.is_empty());

        let t0 = Instant::now();
        let approx = solve_by_adaptive_hull(
            &ctx,
            &cands,
            warm.best_obj,
            warm.best_plan.clone(),
            warm.min_w,
        );
        let approx_ms = t0.elapsed().as_secs_f64() * 1000.0;

        let t1 = Instant::now();
        let exact =
            solve_by_exact_b_enumeration(&ctx, &cands, warm.best_obj, warm.best_plan, warm.min_w);
        let exact_ms = t1.elapsed().as_secs_f64() * 1000.0;

        let approx_obj =
            objective_from_requests(&approx.best_plan, &cm, coeff_sum, coeff_bottleneck);
        let exact_obj = objective_from_requests(&exact.best_plan, &cm, coeff_sum, coeff_bottleneck);
        let rel_err = if exact_obj > 0.0 {
            (approx_obj - exact_obj) / exact_obj
        } else {
            0.0
        };
        println!(
            "perf_balanced_ub_exact_vs_approx_large: approx_ms={:.3}, exact_ms={:.3}, speedup={:.2}x, approx_obj={:.6}, exact_obj={:.6}, rel_err={:.4}%",
            approx_ms,
            exact_ms,
            if approx_ms > 0.0 { exact_ms / approx_ms } else { f64::INFINITY },
            approx_obj,
            exact_obj,
            rel_err * 100.0
        );
        assert!(approx_obj <= exact_obj * 1.05 + 1e-6);
    }

    #[test]
    fn test_exact_baseline_uses_dc_backend_for_piecewise_large_k() {
        let planner = BalancedUBMerge::with_fast_path_optimizations(false);
        assert!(!planner.enable_fast_path_optimizations);
        assert_eq!(
            planner.subproblem_optimization_mode,
            SubproblemOptimizationMode::DivideAndConquerOnly
        );

        let mut cm = CostModel::default();
        cm.latency_model = "piecewise_linear".to_string();
        cm.latency_pw_breaks_bytes = vec![256 * 1024];
        cm.latency_pw_overheads_ms = vec![95.0, 130.0];
        cm.latency_pw_rates_ms_per_byte = vec![0.00002, 0.00004];
        cm.concurrent_parallelism = 8.0;
        cm.max_waste_ratio = 1.0;
        cm.max_requests = 32;

        let offsets: Vec<i64> = (0..64).map(|i| i as i64 * 32_768).collect();
        let sizes: Vec<i64> = vec![16_384; 64];
        let edge = build_edge_latency_matrix(&offsets, &sizes, &cm);
        let (coeff_sum, coeff_bottleneck) =
            compute_objective_coefficients(cm.concurrent_parallelism);
        let ctx = PlannerContext {
            n: offsets.len(),
            k_cap: 32,
            offsets: &offsets,
            sizes: &sizes,
            cost_model: &cm,
            edge_latency: &edge,
            coeff_sum,
            coeff_bottleneck,
            exact_requests: false,
            max_merge_span: 0,
            is_linear_model: !cm.use_piecewise_linear(),
            is_convex_piecewise: is_convex_piecewise_linear(&cm),
            use_wqs_linear_kcap: false,
            use_wqs_piecewise_kcap: false,
            subproblem_optimization_mode: planner.subproblem_optimization_mode,
        };
        assert_eq!(
            select_subproblem_solver(&ctx),
            SubproblemSolverKind::DivideAndConquer
        );
    }

    #[test]
    fn test_default_fast_path_uses_wqs_backend_for_piecewise_large_k() {
        let planner = BalancedUBMerge::new();
        assert!(planner.enable_fast_path_optimizations);
        assert_eq!(
            planner.subproblem_optimization_mode,
            SubproblemOptimizationMode::Auto
        );

        let mut cm = CostModel::default();
        cm.latency_model = "piecewise_linear".to_string();
        cm.latency_pw_breaks_bytes = vec![256 * 1024];
        cm.latency_pw_overheads_ms = vec![95.0, 130.0];
        cm.latency_pw_rates_ms_per_byte = vec![0.00002, 0.00004];
        cm.concurrent_parallelism = 8.0;
        cm.max_waste_ratio = 1.0;
        cm.max_requests = 32;

        let offsets: Vec<i64> = (0..64).map(|i| i as i64 * 32_768).collect();
        let sizes: Vec<i64> = vec![16_384; 64];
        let edge = build_edge_latency_matrix(&offsets, &sizes, &cm);
        let (coeff_sum, coeff_bottleneck) =
            compute_objective_coefficients(cm.concurrent_parallelism);
        let ctx = PlannerContext {
            n: offsets.len(),
            k_cap: 32,
            offsets: &offsets,
            sizes: &sizes,
            cost_model: &cm,
            edge_latency: &edge,
            coeff_sum,
            coeff_bottleneck,
            exact_requests: false,
            max_merge_span: 0,
            is_linear_model: !cm.use_piecewise_linear(),
            is_convex_piecewise: is_convex_piecewise_linear(&cm),
            use_wqs_linear_kcap: false,
            use_wqs_piecewise_kcap: true,
            subproblem_optimization_mode: planner.subproblem_optimization_mode,
        };
        assert_eq!(
            select_subproblem_solver(&ctx),
            SubproblemSolverKind::PiecewiseWqs
        );
    }

    #[test]
    fn test_default_fast_path_uses_dc_backend_for_piecewise_small_k() {
        let planner = BalancedUBMerge::new();
        assert!(planner.enable_fast_path_optimizations);
        assert_eq!(
            planner.subproblem_optimization_mode,
            SubproblemOptimizationMode::Auto
        );

        let mut cm = CostModel::default();
        cm.latency_model = "piecewise_linear".to_string();
        cm.latency_pw_breaks_bytes = vec![256 * 1024];
        cm.latency_pw_overheads_ms = vec![95.0, 130.0];
        cm.latency_pw_rates_ms_per_byte = vec![0.00002, 0.00004];
        cm.concurrent_parallelism = 8.0;
        cm.max_waste_ratio = 1.0;
        cm.max_requests = 8;

        let offsets: Vec<i64> = (0..64).map(|i| i as i64 * 32_768).collect();
        let sizes: Vec<i64> = vec![16_384; 64];
        let edge = build_edge_latency_matrix(&offsets, &sizes, &cm);
        let (coeff_sum, coeff_bottleneck) =
            compute_objective_coefficients(cm.concurrent_parallelism);
        let ctx = PlannerContext {
            n: offsets.len(),
            k_cap: 8,
            offsets: &offsets,
            sizes: &sizes,
            cost_model: &cm,
            edge_latency: &edge,
            coeff_sum,
            coeff_bottleneck,
            exact_requests: false,
            max_merge_span: 0,
            is_linear_model: !cm.use_piecewise_linear(),
            is_convex_piecewise: is_convex_piecewise_linear(&cm),
            use_wqs_linear_kcap: false,
            use_wqs_piecewise_kcap: is_large_k_for_wqs(offsets.len(), 8),
            subproblem_optimization_mode: planner.subproblem_optimization_mode,
        };
        assert!(!ctx.use_wqs_piecewise_kcap);
        assert_eq!(
            select_subproblem_solver(&ctx),
            SubproblemSolverKind::DivideAndConquer
        );
    }

    #[test]
    fn test_adaptive_dc_only_uses_dc_backend_for_piecewise_large_k() {
        let planner = BalancedUBMerge::with_adaptive_dc_only();
        assert!(planner.enable_fast_path_optimizations);
        assert_eq!(
            planner.subproblem_optimization_mode,
            SubproblemOptimizationMode::DivideAndConquerOnly
        );

        let mut cm = CostModel::default();
        cm.latency_model = "piecewise_linear".to_string();
        cm.latency_pw_breaks_bytes = vec![256 * 1024];
        cm.latency_pw_overheads_ms = vec![95.0, 130.0];
        cm.latency_pw_rates_ms_per_byte = vec![0.00002, 0.00004];
        cm.concurrent_parallelism = 8.0;
        cm.max_waste_ratio = 1.0;
        cm.max_requests = 32;

        let offsets: Vec<i64> = (0..64).map(|i| i as i64 * 32_768).collect();
        let sizes: Vec<i64> = vec![16_384; 64];
        let edge = build_edge_latency_matrix(&offsets, &sizes, &cm);
        let (coeff_sum, coeff_bottleneck) =
            compute_objective_coefficients(cm.concurrent_parallelism);
        let ctx = PlannerContext {
            n: offsets.len(),
            k_cap: 32,
            offsets: &offsets,
            sizes: &sizes,
            cost_model: &cm,
            edge_latency: &edge,
            coeff_sum,
            coeff_bottleneck,
            exact_requests: false,
            max_merge_span: 0,
            is_linear_model: !cm.use_piecewise_linear(),
            is_convex_piecewise: is_convex_piecewise_linear(&cm),
            use_wqs_linear_kcap: false,
            use_wqs_piecewise_kcap: true,
            subproblem_optimization_mode: planner.subproblem_optimization_mode,
        };
        assert_eq!(
            select_subproblem_solver(&ctx),
            SubproblemSolverKind::DivideAndConquer
        );
    }

    #[test]
    #[ignore]
    fn perf_balanced_ub_adaptive_wqs_vs_dc_piecewise() {
        let nlist = 1000usize;
        let nprobe = 256usize;
        let k = 128usize;
        let num_queries = 12usize;
        let warmup_queries = 3usize;

        let mut rng = StdRng::seed_from_u64(0x51A7_DC01_u64);
        let all_cluster_ids: Vec<i32> = (0..nlist as i32).collect();
        let mut offsets = HashMap::new();
        let mut sizes = HashMap::new();

        let mut cur = 0_i64;
        for cid in &all_cluster_ids {
            let sz = rng.random_range(8_192_i64..96_000_i64);
            if *cid > 0 {
                cur += rng.random_range(0_i64..220_000_i64);
            }
            offsets.insert(*cid, cur);
            sizes.insert(*cid, sz);
            cur += sz;
        }

        let mut queries = Vec::with_capacity(num_queries);
        for _ in 0..num_queries {
            let mut ids = all_cluster_ids.clone();
            ids.shuffle(&mut rng);
            ids.truncate(nprobe);
            queries.push(ids);
        }

        let mut cm = CostModel::default();
        cm.latency_model = "piecewise_linear".to_string();
        cm.latency_pw_breaks_bytes = vec![1 * 1024 * 1024];
        cm.latency_pw_overheads_ms = vec![26.05, 15.06];
        cm.latency_pw_rates_ms_per_byte = vec![0.0, 1.0486458329132897e-05];
        cm.concurrent_parallelism = 8.215944927495842;
        cm.max_requests = k as i32;
        cm.exact_requests = false;

        let planner_wqs = BalancedUBMerge::new();
        let planner_dc = BalancedUBMerge::with_adaptive_dc_only();

        for cluster_ids in queries.iter().take(warmup_queries) {
            let _ = planner_wqs.generate_plan(cluster_ids, &offsets, &sizes, &cm);
            let _ = planner_dc.generate_plan(cluster_ids, &offsets, &sizes, &cm);
        }

        let mut wqs_ms = Vec::with_capacity(num_queries);
        let mut dc_ms = Vec::with_capacity(num_queries);
        let mut max_obj_gap = 0.0_f64;

        for cluster_ids in &queries {
            let t0 = Instant::now();
            let wqs_plan = planner_wqs.generate_plan(cluster_ids, &offsets, &sizes, &cm);
            let wqs_elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;

            let t1 = Instant::now();
            let dc_plan = planner_dc.generate_plan(cluster_ids, &offsets, &sizes, &cm);
            let dc_elapsed_ms = t1.elapsed().as_secs_f64() * 1000.0;

            let wqs_obj = plan_objective(&wqs_plan, &cm);
            let dc_obj = plan_objective(&dc_plan, &cm);
            max_obj_gap = max_obj_gap.max((wqs_obj - dc_obj).abs());

            wqs_ms.push(wqs_elapsed_ms);
            dc_ms.push(dc_elapsed_ms);
        }

        wqs_ms.sort_by(|a, b| a.total_cmp(b));
        dc_ms.sort_by(|a, b| a.total_cmp(b));
        let pct = |sorted: &[f64], p: f64| -> f64 {
            let rank = (p / 100.0) * ((sorted.len() - 1) as f64);
            let lo = rank.floor() as usize;
            let hi = rank.ceil() as usize;
            if lo == hi {
                sorted[lo]
            } else {
                let frac = rank - lo as f64;
                sorted[lo] + (sorted[hi] - sorted[lo]) * frac
            }
        };

        let wqs_p50 = pct(&wqs_ms, 50.0);
        let wqs_p95 = pct(&wqs_ms, 95.0);
        let dc_p50 = pct(&dc_ms, 50.0);
        let dc_p95 = pct(&dc_ms, 95.0);
        println!(
            "perf_balanced_ub_adaptive_wqs_vs_dc_piecewise: nprobe={nprobe} K={k} queries={num_queries} wqs_p50={wqs_p50:.3}ms wqs_p95={wqs_p95:.3}ms dc_p50={dc_p50:.3}ms dc_p95={dc_p95:.3}ms wqs_over_dc_p95={:.2}x max_obj_gap={max_obj_gap:.6}",
            if dc_p95 > 0.0 { wqs_p95 / dc_p95 } else { f64::INFINITY },
        );
    }

    #[test]
    fn test_plain_dp_baseline_matches_exact_piecewise_large_k() {
        let mut rng = StdRng::seed_from_u64(0xDEC0DE_u64);
        let n = 24usize;
        let cluster_ids: Vec<i32> = (0..n as i32).collect();
        let mut offsets = HashMap::new();
        let mut sizes = HashMap::new();

        let mut cur = 0_i64;
        for cid in &cluster_ids {
            let sz = rng.random_range(8_192_i64..96_000_i64);
            if *cid > 0 {
                cur += rng.random_range(0_i64..160_000_i64);
            }
            offsets.insert(*cid, cur);
            sizes.insert(*cid, sz);
            cur += sz;
        }

        let mut cm = CostModel::default();
        cm.latency_model = "piecewise_linear".to_string();
        cm.latency_pw_breaks_bytes = vec![256 * 1024, 2 * 1024 * 1024];
        cm.latency_pw_overheads_ms = vec![95.0, 120.0, 180.0];
        cm.latency_pw_rates_ms_per_byte = vec![0.00002, 0.00002, 0.00004];
        cm.concurrent_parallelism = 10.0;
        cm.max_requests = 16;
        cm.exact_requests = false;

        let exact_plan = BalancedUBMerge::with_fast_path_optimizations(false).generate_plan(
            &cluster_ids,
            &offsets,
            &sizes,
            &cm,
        );
        let plain_plan = BalancedUBMerge::with_plain_dp_baseline().generate_plan(
            &cluster_ids,
            &offsets,
            &sizes,
            &cm,
        );

        let exact_obj = plan_objective(&exact_plan, &cm);
        let plain_obj = plan_objective(&plain_plan, &cm);
        let diff = (exact_obj - plain_obj).abs();
        assert!(
            diff <= 1e-6 * exact_obj.max(plain_obj).max(1.0),
            "plain DP and exact DC baseline diverged: exact={exact_obj}, plain={plain_obj}, diff={diff}"
        );
    }
}
