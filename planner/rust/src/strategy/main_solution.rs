//! MainSolution — finite-parallel `T_UB = W/P + (1 − 1/P) · B` planner.
//!
//! The paper's main planner. It upper-bounds wall-clock latency under
//! finite parallelism `P`:
//!
//! ```text
//!     T_UB(P) = W/P + (1 − 1/P) · B
//! ```
//!
//! where `W = Σ w(merged.length)` is the serial work and
//! `B = max w(merged.length)` is the slowest merged request. Minimising
//! `T_UB` interpolates between the serial-total-work objective
//! ([`crate::strategy::Serial`]) at `P = 1` and the bottleneck objective
//! ([`crate::strategy::Bottleneck`]) at `P → ∞`.
//!
//! # Algorithm (main line)
//!
//! 1. Derive `(coeff_w, coeff_b)` from the parallelism coefficient `P`.
//! 2. Precompute the lower-triangular edge-latency matrix (skipped on
//!    the linear fast path, where `w(i, j)` is `O(1)` from `(O, r)`).
//! 3. **Warm start**: run the inner DP at `B = ∞` to get an initial
//!    upper bound, the global lower bound `W*`, and a fallback plan.
//!    If the objective has no `B` term (`P = 1`, the serial case), this
//!    warm-start plan is already optimal.
//! 4. Enumerate **candidate B values** — the distinct per-segment
//!    latencies in the feasible envelope — and prune by the `k_cap`
//!    greedy-feasibility bound plus the `W*`-derived upper cap.
//! 5. For each candidate, the inner DP ("given `B_cap`, minimise `W`")
//!    yields a `(obj, plan)` pair. Pick the best across candidates —
//!    either by exact enumeration or by adaptive convex-hull search.
//! 6. Convert the winning cut list into an `IoPlan`.
//!
//! Everything after the `impl MergeStrategy` block is a helper used by
//! one of these six steps, laid out in the same order.
//!
//! # Optimality vs. approximation
//!
//! The inner DPs in step 5 are **exact** for the `B_cap` they're
//! given — linear-fast-path / DC / general are all optimal solvers
//! for the "minimise W subject to all merged latencies ≤ B_cap"
//! subproblem. Two places where approximation sneaks in:
//!
//! - **Sparse candidate enumeration** (step 4) kicks in when
//!   `n·(n+1)/2 > 2048` (so n ≳ 64). Above that threshold,
//!   candidate B values are sampled at log-spaced strides instead
//!   of enumerated exhaustively — `O(n log n)` candidates instead
//!   of `O(n²)`. The true optimal `B*` may not be in the sampled
//!   set, so the reported plan can be slightly suboptimal.
//!
//! - **Adaptive convex-hull search** (step 5) kicks in when the
//!   candidate list has at least `ADAPTIVE_MIN_CANDIDATES` entries.
//!   It evaluates `O(log n)` candidates instead of every candidate,
//!   **stopping early once the gap between the best evaluated
//!   objective and the best live convex-hull lower bound falls
//!   below `ADAPTIVE_TARGET_RELATIVE_ERROR = 5 %`**. Plans are
//!   therefore guaranteed within 5 % of the optimal `T_UB(P)` that
//!   the hull-search regime *could* certify, but not strictly
//!   minimal.
//!
//! Below those thresholds (small `n`, short candidate list) the
//! planner is bit-exact — the brute-force optimality test in
//! `tests/main_solution_optimality.rs` exercises that regime. The
//! golden regression in `tests/golden.rs` correspondingly checks
//! `T_UB` within 5 % for `MainSolution` rather than byte-equal plans.

use std::collections::{BTreeMap, BinaryHeap, VecDeque};

use crate::cost_model::CostModel;
use crate::io_plan::{IoPlan, IoRequest, MergedRequest};
use crate::plan_constraints::PlanConstraints;
use crate::strategy::{MergeStrategy, Separate};

// ========================================================================
// 1. Public entry — the main line.
// ========================================================================

/// Finite-parallel `T_UB` planner — the paper's main solution.
///
/// `P` is read from [`CostModel::concurrent_parallelism`]. `P = 1`
/// degenerates to the [`crate::strategy::Serial`] objective, `P ≤ 0`
/// is treated as full parallelism (`T = B`), and values `≥ K`
/// degenerate toward the [`crate::strategy::Bottleneck`] solution.
#[derive(Debug, Clone, Copy, Default)]
pub struct MainSolution;

impl MergeStrategy for MainSolution {
    fn name(&self) -> &'static str {
        "main_solution"
    }

    fn supports_request_constraints(&self) -> bool {
        true
    }

    fn generate_plan(
        &self,
        requests: &[IoRequest],
        cost: &CostModel,
        constraints: &PlanConstraints,
    ) -> IoPlan {
        if requests.is_empty() {
            return IoPlan::empty();
        }
        let ctx = PlannerContext::build(requests, cost, constraints);

        // One scratch allocation reused for warm-start + every candidate
        // eval — pooling drops plan time 2–4× at large n.
        let mut scratch = Scratch::new(ctx.n, ctx.k_cap);

        // Step 3 — warm-start at B = ∞.
        let warm = compute_warm_start(&ctx, &mut scratch);
        if ctx.coeff_b < EPS {
            // No bottleneck term; warm-start (which minimises W) is optimal.
            return finalize_plan(requests, &ctx, warm.best_plan);
        }

        // Step 4 — candidate B values within the feasible envelope.
        let (min_b, max_b) = candidate_b_bounds(&ctx, &warm);
        let candidates = collect_candidate_b_values(&ctx, min_b, max_b);

        // Step 5 — search for the best candidate.
        let (_best_obj, best_plan) = if candidates.len() >= ADAPTIVE_MIN_CANDIDATES {
            solve_by_adaptive_hull(
                &ctx,
                &mut scratch,
                &candidates,
                warm.best_obj,
                warm.best_plan,
                warm.min_w,
            )
        } else {
            solve_by_exact_enumeration(
                &ctx,
                &mut scratch,
                &candidates,
                warm.best_obj,
                warm.best_plan,
            )
        };

        // Step 6 — materialise the plan.
        if best_plan.is_empty() {
            return Separate.generate_plan(requests, cost, constraints);
        }
        finalize_plan(requests, &ctx, best_plan)
    }
}

// ========================================================================
// 2. Constants, types, context.
// ========================================================================

/// Numeric guard representing "infeasible / uninitialised" in the DP
/// tables. Values above `INF / 10` are treated as unreachable.
const INF: f64 = 1e18;

/// Tolerance used for all f64 comparisons.
const EPS: f64 = 1e-9;

// Adaptive hull search knobs. When the candidate B list exceeds
// `ADAPTIVE_MIN_CANDIDATES`, the planner skips exhaustive enumeration
// and performs a convex-hull interval search, evaluating at most
// `ADAPTIVE_MAX_EVAL_BUDGET` points and stopping early once the
// certified relative error falls below `ADAPTIVE_TARGET_RELATIVE_ERROR`.
const ADAPTIVE_MIN_CANDIDATES: usize = 24;
const ADAPTIVE_MIN_EVAL_BUDGET: usize = 16;
const ADAPTIVE_MAX_EVAL_BUDGET: usize = 64;
const ADAPTIVE_TARGET_RELATIVE_ERROR: f64 = 0.05;

/// Candidate enumeration switches to log-spaced sampling above this
/// cell count, holding the total candidates near `O(n log n)` instead
/// of the dense `O(n²)`.
const DENSE_ENUMERATION_LIMIT: usize = 2048;

/// Ordered list of cut points defining merged groups.
///
/// Invariant: `cuts[0] = 0`, `cuts.last() = n`, strictly increasing;
/// merged group `k` covers requests `[cuts[k]..cuts[k+1])`.
type Cuts = Vec<usize>;

/// Column-major 2D table backing the DP state. Cheaper than
/// `Vec<Vec<T>>` because the storage is a single contiguous allocation;
/// column-major (rather than the more common row-major) so that the
/// inner-loop access pattern `dp[k][j-1]` for fixed `j-1` and varying
/// `k` is sequential — critical for the O(n²)-class DPs at n ≥ 512,
/// where row-major would miss a cache line per inner step.
///
/// The `(i, j)` semantics of `get` / `set` match their row-major
/// equivalent; only the physical layout changes.
struct FlatTable<T: Clone> {
    data: Vec<T>,
    rows: usize,
}

impl<T: Clone> FlatTable<T> {
    fn new(rows: usize, cols: usize, init: T) -> Self {
        Self {
            data: vec![init; rows * cols],
            rows,
        }
    }

    #[inline]
    fn get(&self, i: usize, j: usize) -> &T {
        &self.data[j * self.rows + i]
    }

    #[inline]
    fn set(&mut self, i: usize, j: usize, v: T) {
        self.data[j * self.rows + i] = v;
    }

    /// Pointer to the start of column `j` — a contiguous slice of
    /// `rows` entries. Hot-loop friendly: the inner DPs can iterate
    /// over `dp[k][j-1]` for k ∈ [lo, hi] by reading a contiguous
    /// slice, which the compiler auto-vectorises and the CPU
    /// sequentially prefetches.
    #[inline]
    fn col(&self, j: usize) -> &[T] {
        let base = j * self.rows;
        &self.data[base..base + self.rows]
    }

    #[inline]
    fn fill(&mut self, v: T)
    where
        T: Copy,
    {
        self.data.fill(v);
    }

    /// Disjoint `(&col[prev_j], &mut col[cur_j])` views — lets a DP
    /// inner loop read the previous column as a slice and write the
    /// current column via index without upsetting the borrow checker.
    /// Requires `prev_j != cur_j`.
    #[inline]
    fn two_cols_mut(&mut self, prev_j: usize, cur_j: usize) -> (&[T], &mut [T]) {
        debug_assert_ne!(prev_j, cur_j);
        let rows = self.rows;
        if prev_j < cur_j {
            let (before, rest) = self.data.split_at_mut(cur_j * rows);
            (
                &before[prev_j * rows..(prev_j + 1) * rows],
                &mut rest[..rows],
            )
        } else {
            let (before, rest) = self.data.split_at_mut(prev_j * rows);
            (&rest[..rows], &mut before[cur_j * rows..(cur_j + 1) * rows])
        }
    }
}

/// Scratch buffers reused across every candidate-B evaluation.
///
/// Allocating `dp` + `parent` fresh per `evaluate_at_b` was the single
/// biggest bookkeeping cost at n ≥ 512 (each call touched
/// `~O(n · k_cap)` = 2–4 MB of heap). Pooling drops plan time on the
/// piecewise-cost path where the adaptive-hull loop makes 30–60
/// evaluations per plan.
///
/// The parent table is **not** reset between evaluations: its entries
/// are only read via `cuts_from_parent` / `objective_from_parent`,
/// both of which walk backward from `dp[n][j]` and stop as soon as a
/// cell's `dp` is infeasible (`INF / 10`). Every write to `parent`
/// is paired with a write to `dp`, so stale parent entries whose
/// matching `dp` got reset to `INF` are unreachable.
struct Scratch {
    dp: FlatTable<f64>,
    parent: FlatTable<i32>,
}

impl Scratch {
    fn new(n: usize, k_cap: usize) -> Self {
        Self {
            dp: FlatTable::new(n + 1, k_cap + 1, INF),
            parent: FlatTable::new(n + 1, k_cap + 1, -1),
        }
    }

    /// Restore the DP table to its initial state. Only `dp` is reset;
    /// `parent` follows `dp` by construction (see struct docstring).
    #[inline]
    fn reset(&mut self) {
        self.dp.fill(INF);
        self.dp.set(0, 0, 0.0);
    }
}

/// Inner-DP dispatch tag. Picked once per `PlannerContext::build` from
/// the cost-model shape.
#[derive(Clone, Copy, PartialEq, Eq)]
enum SolverKind {
    /// Linear cost — monotonic-deque O(n·k) fast path.
    LinearFastPath,
    /// Convex piecewise-linear cost — DC O(n·k·log n) per layer.
    DivideAndConquer,
    /// Convex piecewise-linear cost with `k_cap ≥ n/4` — Lagrangian
    /// (WQS / Aliens trick) with a single 1D DP and ≤ ~50 binary-search
    /// iterations. Drops the explicit `j` dimension; ~1.2-1.3× faster
    /// than DC on the large-`k` regime tested by the paper.
    PiecewiseWqs,
    /// General fallback — dense O(n²·k) DP; always correct.
    PlainDP,
}

/// WQS dispatch threshold — `WQS_LARGE_K_NUM/WQS_LARGE_K_DEN` (= 1/4).
/// When `k_cap ≥ n/4` and the cost is piecewise-convex, Lagrangian WQS
/// beats DC's per-layer scaling. Threshold chosen empirically from the
/// `main_solution_speed` benchmark.
const WQS_LARGE_K_NUM: usize = 1;
const WQS_LARGE_K_DEN: usize = 4;

#[inline]
fn is_large_k_for_wqs(n: usize, k_cap: usize) -> bool {
    k_cap.saturating_mul(WQS_LARGE_K_DEN) >= n.saturating_mul(WQS_LARGE_K_NUM)
}

/// Read-only bundle threaded through every DP helper.
///
/// Owns the derived `offsets` / `sizes` / `edge_latency` buffers so
/// the main line doesn't juggle them. `cost` is borrowed — the caller
/// keeps ownership.
struct PlannerContext<'a> {
    n: usize,
    k_cap: usize,
    offsets: Vec<i64>,
    sizes: Vec<i64>,
    cost: &'a CostModel,
    constraints: &'a PlanConstraints,
    /// Lower-triangular `edge_latency[triangular_edge_index(end, start)]` =
    /// `w(merge requests[start..=end])`. Empty on the linear fast path
    /// (values computed on the fly via [`edge_at`]).
    edge_latency: Vec<f64>,
    coeff_w: f64,
    coeff_b: f64,
    exact_requests: bool,
    is_convex_piecewise: bool,
    /// True when the WQS Lagrangian path applies: piecewise-convex
    /// cost + `k_cap ≥ n/4` + `coeff_w > EPS` + `!exact_requests`.
    /// See [`SolverKind::PiecewiseWqs`].
    use_wqs_piecewise: bool,
    /// Effective `(overhead, slope)` for the linear fast path, when
    /// applicable. See [`effective_linear_params`].
    linear_fast_params: Option<(f64, f64)>,
}

impl<'a> PlannerContext<'a> {
    /// Build the context from raw planner inputs (steps 1–2 of the
    /// main line): derive the objective coefficients, pick a solver
    /// shape, and materialise the edge-latency cache (unless the
    /// linear fast path makes it unnecessary).
    fn build(
        requests: &[IoRequest],
        cost: &'a CostModel,
        constraints: &'a PlanConstraints,
    ) -> Self {
        let n = requests.len();
        let offsets: Vec<i64> = requests.iter().map(|r| r.offset).collect();
        let sizes: Vec<i64> = requests.iter().map(|r| r.length).collect();
        let k_cap = constraints.request_cap(n);
        let (coeff_w, coeff_b) = compute_objective_coefficients(cost.concurrent_parallelism);
        let linear_fast_params = effective_linear_params(cost);

        // Skip the O(n²) matrix build on the linear fast path: every
        // lookup becomes a cheap `O + r·bytes` via `edge_at`.
        let edge_latency = if linear_fast_params.is_some() {
            Vec::new()
        } else {
            build_edge_latency_matrix(&offsets, &sizes, cost, linear_fast_params)
        };

        let exact_requests = constraints.exact_requests && constraints.max_requests > 0;
        let is_convex_piecewise = is_convex_piecewise_linear(cost);
        let use_wqs_piecewise =
            !exact_requests && is_convex_piecewise && is_large_k_for_wqs(n, k_cap) && coeff_w > EPS;
        Self {
            n,
            k_cap,
            offsets,
            sizes,
            cost,
            constraints,
            edge_latency,
            coeff_w,
            coeff_b,
            exact_requests,
            is_convex_piecewise,
            use_wqs_piecewise,
            linear_fast_params,
        }
    }

    fn solver_kind(&self) -> SolverKind {
        if self.linear_fast_params.is_some() {
            SolverKind::LinearFastPath
        } else if self.use_wqs_piecewise {
            SolverKind::PiecewiseWqs
        } else if self.is_convex_piecewise {
            SolverKind::DivideAndConquer
        } else {
            SolverKind::PlainDP
        }
    }
}

// ========================================================================
// 3. Objective shape + edge latency cache (step 1–2 helpers).
// ========================================================================

/// Split the `T_UB = W/P + (1 − 1/P)·B` objective into its
/// `(coeff_w, coeff_b)` linear combination.
///
/// `P ≤ 0` is treated as full-parallel (`T = B`); `P ≥ 1` uses the
/// formula verbatim, with fractional `P` clamped to `1.0`. Shared with
/// [`crate::io_plan::PlanMetrics::t_ub_ms`] so the planner objective
/// and the reported metric can never drift apart.
fn compute_objective_coefficients(parallelism: f64) -> (f64, f64) {
    crate::internal::piecewise::objective_coefficients(parallelism)
}

/// A piecewise-linear cost is convex iff its per-segment slopes are
/// non-decreasing. The DC solver needs this for the Monge condition;
/// the plain DP is used otherwise.
fn is_convex_piecewise_linear(cost: &CostModel) -> bool {
    if !cost.use_piecewise_linear() {
        return false;
    }
    let slopes = &cost.latency_pw_rates_ms_per_byte;
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

/// Derive the effective `(overhead, slope)` that drives the linear
/// fast path, or `None` if the cost shape forces DC / plain DP.
///
/// - Linear cost → its `(latency_per_request, latency_per_byte)`.
/// - Otherwise → `None`.
fn effective_linear_params(cost: &CostModel) -> Option<(f64, f64)> {
    if !cost.use_piecewise_linear() {
        return Some((cost.latency_per_request, cost.latency_per_byte.max(0.0)));
    }
    let breaks = &cost.latency_pw_breaks_bytes;
    if breaks.is_empty() {
        return cost
            .latency_pw_overheads_ms
            .first()
            .zip(cost.latency_pw_rates_ms_per_byte.first())
            .map(|(&o, &r)| (o, r.max(0.0)));
    }
    None
}

/// Lower-triangular flat index. `end >= start`; valid indices are
/// `0 <= start <= end < n`.
#[inline]
fn triangular_edge_index(end_idx: usize, start_idx: usize) -> usize {
    (end_idx * (end_idx + 1)) / 2 + start_idx
}

/// Continuity-adjusted per-segment piecewise parameters. Callers
/// supply independent `(O_i, r_i)`; we rewrite each `O_i` so that
/// `O_i + r_i · break_{i-1}` equals the previous segment's value at
/// the break, enabling O(1) evaluation per query.
struct PiecewiseParams {
    breaks_bytes: Vec<i64>,
    overheads_ms: Vec<f64>,
    rates_ms_per_byte: Vec<f64>,
}

fn prepare_piecewise_latency(cost: &CostModel) -> Option<PiecewiseParams> {
    if !cost.use_piecewise_linear() {
        return None;
    }
    let (overheads, rates) = crate::internal::piecewise::stitched_segments(
        &cost.latency_pw_breaks_bytes,
        &cost.latency_pw_overheads_ms,
        &cost.latency_pw_rates_ms_per_byte,
    )
    .unzip();
    Some(PiecewiseParams {
        breaks_bytes: cost.latency_pw_breaks_bytes.clone(),
        overheads_ms: overheads,
        rates_ms_per_byte: rates,
    })
}

#[inline]
fn piecewise_latency_ms(p: &PiecewiseParams, bytes: i64) -> f64 {
    let seg = crate::internal::piecewise::segment_index(&p.breaks_bytes, bytes);
    p.overheads_ms[seg] + p.rates_ms_per_byte[seg] * bytes.max(0) as f64
}

/// Fill the lower-triangular edge-latency matrix for the given request
/// layout. When `linear_params = Some((o, r))` the caller has already
/// established that every merged span stays in a linear regime — we
/// then skip the piecewise `partition_point` per cell and use the
/// cheap `o + r · bytes` formula.
fn build_edge_latency_matrix(
    offsets: &[i64],
    sizes: &[i64],
    cost: &CostModel,
    linear_params: Option<(f64, f64)>,
) -> Vec<f64> {
    let n = offsets.len();
    let mut edge = vec![0.0; n * (n + 1) / 2];
    // `triangular_edge_index(end, start) = end*(end+1)/2 + start`; hoisting
    // the base keeps the inner loop to one add per cell.
    if let Some((o, r)) = linear_params {
        for end in 0..n {
            let range_end = offsets[end] + sizes[end];
            let base = end * (end + 1) / 2;
            let row = &mut edge[base..base + end + 1];
            for (start, cell) in row.iter_mut().enumerate() {
                let bytes = range_end - offsets[start];
                *cell = o + r * bytes as f64;
            }
        }
        return edge;
    }
    if let Some(p) = prepare_piecewise_latency(cost) {
        for end in 0..n {
            let range_end = offsets[end] + sizes[end];
            let base = end * (end + 1) / 2;
            let row = &mut edge[base..base + end + 1];
            for (start, cell) in row.iter_mut().enumerate() {
                let bytes = range_end - offsets[start];
                *cell = piecewise_latency_ms(&p, bytes);
            }
        }
    } else {
        let o = cost.latency_per_request;
        let r = cost.latency_per_byte;
        for end in 0..n {
            let range_end = offsets[end] + sizes[end];
            let base = end * (end + 1) / 2;
            let row = &mut edge[base..base + end + 1];
            for (start, cell) in row.iter_mut().enumerate() {
                let bytes = range_end - offsets[start];
                *cell = o + r * bytes as f64;
            }
        }
    }
    edge
}

/// Look up `w(merge requests[start..=end])`. Uses the precomputed
/// matrix when available, else falls back to the linear formula.
/// When the linear fast path applies we skip materialising the full
/// `O(n²)` matrix — this saves the matrix-build cost that otherwise
/// dominates for large `n`.
#[inline]
fn edge_at(ctx: &PlannerContext, end: usize, start: usize) -> f64 {
    if !ctx.edge_latency.is_empty() {
        return ctx.edge_latency[triangular_edge_index(end, start)];
    }
    if let Some((o, r)) = ctx.linear_fast_params {
        let bytes = ctx.offsets[end] + ctx.sizes[end] - ctx.offsets[start];
        return o + r * (bytes as f64);
    }
    // Unreachable: either the matrix is built or linear_fast_params is set.
    f64::INFINITY
}

// ========================================================================
// 4. Inner DP — "given B_cap, minimise W."
// ========================================================================
//
// Every candidate-B evaluation runs the inner DP once. `evaluate_at_b`
// is the uniform entry point; the three solver variants
// (`run_linear_dp_with_b_parent`, `solve_all_layers_dc`,
// `run_general_dp_with_b_parent`) share the DP-table shape and the
// post-processing helpers (`objective_from_parent`, `cuts_from_parent`).

/// Build the DP, pick the best group count, reconstruct the cut list.
/// Reuses the caller-provided [`Scratch`] buffers rather than
/// allocating fresh DP tables.
fn evaluate_at_b(ctx: &PlannerContext, scratch: &mut Scratch, b_cap: f64) -> (f64, Cuts) {
    // WQS path uses its own 1D DP — no Scratch reset needed.
    if ctx.solver_kind() == SolverKind::PiecewiseWqs {
        return evaluate_at_b_wqs(ctx, b_cap);
    }

    scratch.reset();
    run_inner_dp(ctx, b_cap, &mut scratch.dp, &mut scratch.parent);

    let mut best_obj = INF;
    let mut best_j: Option<usize> = None;
    let j_lo = if ctx.exact_requests { ctx.k_cap } else { 1 };
    for j in j_lo..=ctx.k_cap {
        let w = *scratch.dp.get(ctx.n, j);
        if w >= INF / 10.0 {
            continue;
        }
        if let Some(obj) = objective_from_parent(ctx, j, &scratch.parent) {
            if obj < best_obj {
                best_obj = obj;
                best_j = Some(j);
            }
        }
    }

    let cuts = match best_j {
        Some(j) => cuts_from_parent(ctx.n, j, &scratch.parent),
        None => Cuts::new(),
    };
    (best_obj, cuts)
}

/// WQS evaluation: Lagrangian DP returns one `(W, k)` answer (not a
/// full DP table), which we convert into the same `(obj, Cuts)` shape
/// the rest of the planner expects.
fn evaluate_at_b_wqs(ctx: &PlannerContext, b_cap: f64) -> (f64, Cuts) {
    let wqs = solve_piecewise_wqs_minw_with_b(ctx.n, ctx.k_cap, &ctx.edge_latency, b_cap, true);
    if !wqs.feasible || wqs.parent.is_empty() || wqs.used_requests == 0 {
        return (INF, Cuts::new());
    }
    // Walk WQS's 1D parent chain to reconstruct cut points.
    let mut cuts = vec![ctx.n];
    let mut i = ctx.n;
    while i > 0 {
        let prev = wqs.parent[i];
        if prev < 0 || (prev as usize) >= i {
            return (INF, Cuts::new());
        }
        let prev = prev as usize;
        cuts.push(prev);
        i = prev;
    }
    cuts.sort_unstable();
    if cuts.first() != Some(&0) || cuts.last() != Some(&ctx.n) {
        return (INF, Cuts::new());
    }
    let obj = ctx.coeff_w * wqs.total_work + ctx.coeff_b * b_cap;
    (obj, cuts)
}

/// Dispatch to the right inner DP based on the cost-model shape.
fn run_inner_dp(
    ctx: &PlannerContext,
    b_cap: f64,
    dp: &mut FlatTable<f64>,
    parent: &mut FlatTable<i32>,
) {
    match ctx.solver_kind() {
        SolverKind::LinearFastPath => run_linear_dp_with_b_parent(ctx, b_cap, dp, parent),
        SolverKind::DivideAndConquer => solve_all_layers_dc(ctx, b_cap, dp, Some(parent)),
        SolverKind::PlainDP => run_general_dp_with_b_parent(ctx, b_cap, dp, parent),
        // WQS doesn't use the (n+1)×(k_cap+1) Scratch table — its
        // caller short-circuits to evaluate_at_b_wqs above. Warm
        // start (which calls run_inner_dp directly) falls back to DC
        // for piecewise-convex cost; that's only one extra eval per
        // plan, dwarfed by the per-candidate WQS savings.
        SolverKind::PiecewiseWqs => solve_all_layers_dc(ctx, b_cap, dp, Some(parent)),
    }
}

// ------- 4a. Linear fast path (monotonic deque) ------------------------

/// Linear-cost inner DP using a sliding-window-minimum deque.
///
/// For linear cost `w(merge j..i) = O + r · (end(i-1) − offset(j))`,
/// the transition rewrites as
///
/// ```text
///     dp[i][j] = O + r·end(i-1) + min_{valid k} (dp[k][j-1] − r·offset(k))
/// ```
///
/// The feasibility window (`offset(k) ≥ end(i-1) − max_span`) shifts
/// monotonically right as `i` grows, so a deque keyed by the shifted
/// value answers each query in O(1) amortised.
fn run_linear_dp_with_b_parent(
    ctx: &PlannerContext,
    b_cap: f64,
    dp: &mut FlatTable<f64>,
    parent: &mut FlatTable<i32>,
) {
    let (overhead, slope) = ctx.linear_fast_params.unwrap_or((
        ctx.cost.latency_per_request,
        ctx.cost.latency_per_byte.max(0.0),
    ));

    if slope < EPS {
        run_zero_slope_fast_path(ctx, b_cap, dp, parent, overhead);
        return;
    }

    let max_span = (b_cap - overhead) / slope;
    if max_span < -EPS {
        return;
    }

    // Reuse one deque across all `j` layers — saves 512 VecDeque
    // allocations per eval × 64 evals per plan at n=512 k_cap=512.
    let mut dq: VecDeque<(usize, f64)> = VecDeque::with_capacity(ctx.n);
    for j in 1..=ctx.k_cap {
        dq.clear();
        // Hold prev_col as a shared slice + cur_col as a mutable one,
        // disjoint — the inner loop can index both without re-computing
        // `dp.get(k, j-1)` per iteration.
        let (prev_col, cur_col) = dp.two_cols_mut(j - 1, j);
        #[allow(clippy::needless_range_loop)]
        for i in 1..=ctx.n {
            let k = i - 1;
            let prev = prev_col[k];
            if k >= j - 1 && prev < INF / 10.0 {
                let shifted = prev - slope * ctx.offsets[k] as f64;
                while let Some((_, back_val)) = dq.back() {
                    if *back_val >= shifted - EPS {
                        dq.pop_back();
                    } else {
                        break;
                    }
                }
                dq.push_back((k, shifted));
            }
            let end = (ctx.offsets[i - 1] + ctx.sizes[i - 1]) as f64;
            let min_start = end - max_span;
            while let Some((front_k, _)) = dq.front() {
                if (ctx.offsets[*front_k] as f64) < min_start - EPS {
                    dq.pop_front();
                } else {
                    break;
                }
            }
            if let Some((front_k, best_shifted)) = dq.front() {
                let cand = overhead + slope * end + *best_shifted;
                if cand < cur_col[i] {
                    cur_col[i] = cand;
                    parent.set(i, j, *front_k as i32);
                }
            }
        }
    }
}

/// Zero-slope subcase: cost is a constant-per-request `O`. The deque
/// still applies but the shifted value reduces to `dp[k]` — no byte
/// term.
fn run_zero_slope_fast_path(
    ctx: &PlannerContext,
    b_cap: f64,
    dp: &mut FlatTable<f64>,
    parent: &mut FlatTable<i32>,
    overhead: f64,
) {
    if overhead > b_cap + EPS {
        return;
    }
    let mut dq: VecDeque<(usize, f64)> = VecDeque::with_capacity(ctx.n);
    for j in 1..=ctx.k_cap {
        dq.clear();
        let (prev_col, cur_col) = dp.two_cols_mut(j - 1, j);
        #[allow(clippy::needless_range_loop)]
        for i in 1..=ctx.n {
            let k = i - 1;
            let prev = prev_col[k];
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
            if let Some((front_k, best_prefix)) = dq.front() {
                let cand = *best_prefix + overhead;
                if cand < cur_col[i] {
                    cur_col[i] = cand;
                    parent.set(i, j, *front_k as i32);
                }
            }
        }
    }
}

// ------- 4b. General DP (fallback) -------------------------------------

/// The plain O(n²·k) inner DP — always correct; used when neither the
/// linear fast path nor the DC solver applies (non-convex piecewise
/// cost).
///
/// `j` is the outer loop so we can hold `dp[..][j-1]` as a slice
/// during the whole layer — the inner `k` loop then reads contiguously.
fn run_general_dp_with_b_parent(
    ctx: &PlannerContext,
    b_cap: f64,
    dp: &mut FlatTable<f64>,
    parent: &mut FlatTable<i32>,
) {
    let bcap_plus = b_cap + EPS;
    for j in 1..=ctx.k_cap {
        let (prev_col, cur_col) = dp.two_cols_mut(j - 1, j);
        #[allow(clippy::needless_range_loop)]
        for i in j..=ctx.n {
            let end_idx = i - 1;
            let row_base = triangular_edge_index(end_idx, 0);
            let edges_row = &ctx.edge_latency[row_base..row_base + i];
            let mut best = INF;
            let mut best_k = -1i32;
            for k in (j - 1..i).rev() {
                let w_edge = edges_row[k];
                if w_edge > bcap_plus {
                    continue;
                }
                let prev = prev_col[k];
                if prev >= INF / 10.0 {
                    continue;
                }
                let cand = prev + w_edge;
                if cand < best {
                    best = cand;
                    best_k = k as i32;
                }
            }
            if best_k >= 0 {
                cur_col[i] = best;
                parent.set(i, j, best_k);
            }
        }
    }
}

// ------- 4c. Divide-and-conquer DP (convex piecewise cost) -------------
//
// When the cost model is convex (non-decreasing piecewise slopes),
// the optimal `k*(i)` is monotone in `i` (Monge condition). That lets
// each layer `j` be solved in `O(n log n)` amortised instead of
// `O(n²)`:
//
// 1. Split the `i` range in half at `i_mid`.
// 2. Scan the feasible `k` window at `i_mid`, record `k*`.
// 3. Recurse on `[i_lo, i_mid-1]` with `k_hi = k*` and
//    `[i_mid+1, i_hi]` with `k_lo = k*` — monotonicity restricts both
//    halves' optimal `k` to the respective windows.

// DC recursion terminates into a brute-force base case below these
// thresholds. The brute-force path doesn't re-run the Monge
// feasibility prune per `i`, so raising these thresholds *slows*
// the hot path at n ≥ 512 — DC's per-node prune beats a tight loop
// when the feasible window is narrow.
const DC_SMALL_I_THRESHOLD: usize = 8;
const DC_SMALL_K_THRESHOLD: usize = 8;
const DC_SMALL_AREA_THRESHOLD: usize = 256;

fn solve_all_layers_dc(
    ctx: &PlannerContext,
    b_cap: f64,
    dp: &mut FlatTable<f64>,
    mut parent: Option<&mut FlatTable<i32>>,
) {
    // Precompute per-end_idx span + latency prunes once per b_cap
    // instead of re-running two binary searches at every DC node.
    // With `max_requests = 16` these were the hottest path — caching
    // drops the small-`k_cap` DC path by ~30% on n=512.
    let prune = PrunePerEnd::build(ctx, b_cap);
    for j in 1..=ctx.k_cap {
        if j > ctx.n {
            break;
        }
        dc_solve_layer(
            j,
            ctx.n,
            j - 1,
            ctx.n - 1,
            j,
            &prune,
            dp,
            parent.as_deref_mut(),
            ctx,
        );
        // If no cell in column j is feasible, no later layer can be
        // either (layer j+1 can only populate cells from `dp[*][j]`).
        // Short-circuits tight-b_cap evaluations where most layers
        // beyond some cutoff are all-INF anyway.
        let col = dp.col(j);
        if !col[j..=ctx.n].iter().any(|&v| v < INF / 10.0) {
            break;
        }
    }
}

/// Per-`end_idx` lower bounds on the feasible `start` (= `k`) index.
/// Built once per `evaluate_at_b`; consumed O(1) inside the hot DC
/// inner loop.
struct PrunePerEnd {
    lat_lo: Vec<usize>,
}

impl PrunePerEnd {
    fn build(ctx: &PlannerContext, b_cap: f64) -> Self {
        let n = ctx.n;
        let mut lat_lo = vec![0usize; n];
        let bcap_plus = b_cap + EPS;
        for (end_idx, lat_lo_slot) in lat_lo.iter_mut().enumerate() {
            let row_base = triangular_edge_index(end_idx, 0);
            *lat_lo_slot = ctx.edge_latency[row_base..(row_base + end_idx + 1)]
                .partition_point(|&l| l > bcap_plus);
        }
        Self { lat_lo }
    }
}

#[allow(clippy::too_many_arguments)]
fn dc_solve_layer(
    i_lo: usize,
    i_hi: usize,
    k_lo: usize,
    k_hi: usize,
    j: usize,
    prune: &PrunePerEnd,
    dp: &mut FlatTable<f64>,
    mut parent: Option<&mut FlatTable<i32>>,
    ctx: &PlannerContext,
) {
    if i_lo > i_hi {
        return;
    }
    if dc_use_small_solver(i_lo, i_hi, k_lo, k_hi) {
        dc_solve_layer_small(i_lo, i_hi, k_lo, k_hi, j, prune, dp, parent, ctx);
        return;
    }

    let i_mid = i_lo + (i_hi - i_lo) / 2;
    let end_idx = i_mid - 1;
    let k_upper = k_hi.min(end_idx);

    if k_lo > k_upper {
        if i_mid < i_hi {
            dc_solve_layer(
                i_mid + 1,
                i_hi,
                k_lo,
                k_hi,
                j,
                prune,
                dp,
                parent.as_deref_mut(),
                ctx,
            );
        }
        return;
    }

    let scan_lo = k_lo.max(prune.lat_lo[end_idx]);
    if scan_lo > k_upper {
        if i_mid > i_lo {
            dc_solve_layer(
                i_lo,
                i_mid - 1,
                k_lo,
                k_hi,
                j,
                prune,
                dp,
                parent.as_deref_mut(),
                ctx,
            );
        }
        if i_mid < i_hi && scan_lo <= k_hi {
            dc_solve_layer(
                i_mid + 1,
                i_hi,
                scan_lo,
                k_hi,
                j,
                prune,
                dp,
                parent.as_deref_mut(),
                ctx,
            );
        }
        return;
    }

    let (best, best_k) = dc_scan_best_k(ctx, end_idx, scan_lo, k_upper, j, dp);
    if let Some(bk) = best_k {
        if best < *dp.get(i_mid, j) {
            dp.set(i_mid, j, best);
            if let Some(p) = parent.as_deref_mut() {
                p.set(i_mid, j, bk as i32);
            }
        }
    }

    let left_hi = best_k.unwrap_or(k_upper);
    let right_lo = best_k.unwrap_or(scan_lo);
    if i_mid > i_lo {
        dc_solve_layer(
            i_lo,
            i_mid - 1,
            k_lo,
            left_hi,
            j,
            prune,
            dp,
            parent.as_deref_mut(),
            ctx,
        );
    }
    if i_mid < i_hi {
        dc_solve_layer(i_mid + 1, i_hi, right_lo, k_hi, j, prune, dp, parent, ctx);
    }
}

#[inline]
fn dc_use_small_solver(i_lo: usize, i_hi: usize, k_lo: usize, k_hi: usize) -> bool {
    let i_len = i_hi - i_lo + 1;
    let k_len = k_hi - k_lo + 1;
    i_len <= DC_SMALL_I_THRESHOLD
        || k_len <= DC_SMALL_K_THRESHOLD
        || i_len.saturating_mul(k_len) <= DC_SMALL_AREA_THRESHOLD
}

#[allow(clippy::too_many_arguments)]
fn dc_solve_layer_small(
    i_lo: usize,
    i_hi: usize,
    k_lo: usize,
    k_hi: usize,
    j: usize,
    prune: &PrunePerEnd,
    dp: &mut FlatTable<f64>,
    mut parent: Option<&mut FlatTable<i32>>,
    ctx: &PlannerContext,
) {
    for i in i_lo..=i_hi {
        let end_idx = i - 1;
        let k_upper = k_hi.min(end_idx);
        if k_lo > k_upper {
            continue;
        }
        let scan_lo = k_lo.max(prune.lat_lo[end_idx]);
        if scan_lo > k_upper {
            continue;
        }
        let (best, best_k) = dc_scan_best_k(ctx, end_idx, scan_lo, k_upper, j, dp);
        if let Some(bk) = best_k {
            if best < *dp.get(i, j) {
                dp.set(i, j, best);
                if let Some(p) = parent.as_deref_mut() {
                    p.set(i, j, bk as i32);
                }
            }
        }
    }
}

/// Pick the best `k ∈ [scan_lo, k_upper]` for a fixed `(end_idx, j)`
/// under the current DP state.
///
/// Hot loop. Reads `dp[k][j-1]` for every `k` in range — with the
/// column-major layout that's a contiguous slice, so we pass a
/// reference and let LLVM auto-vectorise the scan.
#[inline(always)]
fn dc_scan_best_k(
    ctx: &PlannerContext,
    end_idx: usize,
    scan_lo: usize,
    k_upper: usize,
    j: usize,
    dp: &FlatTable<f64>,
) -> (f64, Option<usize>) {
    let row_base = triangular_edge_index(end_idx, 0);
    let prev_col = dp.col(j - 1);
    let edges = &ctx.edge_latency[row_base..(row_base + end_idx + 1)];
    let mut best = INF;
    let mut best_k: Option<usize> = None;
    // No explicit infeasibility check: when `prev_col[k]` is the INF
    // sentinel (1e18), `prev + edges[k]` stays ≥ 1e18 and the
    // `cand < best` branch is false — so infeasible cells can't win.
    // Removing the branch lets LLVM turn this loop into straight-line
    // SIMD (vminpd + index tracking) on AVX2 hardware.
    for k in scan_lo..=k_upper {
        let cand = prev_col[k] + edges[k];
        if cand < best {
            best = cand;
            best_k = Some(k);
        }
    }
    (best, best_k)
}

// ------- 4cw. WQS (Lagrangian) inner DP --------------------------------
//
// `solve_piecewise_wqs_minw_with_b` is the alternative inner solver
// the paper's "Full Opt" benchmark uses when `k_cap ≥ n/4`. It drops
// the explicit `j` (group-count) dimension by adding a Lagrangian
// penalty `λ` per used request to the objective:
//
//     min  Σ w(group) + λ · |groups|     subject to  every w(group) ≤ B_cap
//
// then binary-searches `λ` so the minimiser uses ≤ k_cap requests.
//
// Per-`λ` DP is a tight 1D pass — `O(n²)` worst case but with an
// effective break-on-`b_cap` that slashes the constant. Total cost is
// ~50 inner solves (1 + 45 binary-search rounds + a few parent
// retries), beating DC's `O(n²·log n) × k_cap` accounting at large
// `k_cap`.

#[derive(Default)]
struct WqsSolution {
    feasible: bool,
    /// `Σ edge_latency` over the chosen groups (no penalty term).
    total_work: f64,
    /// Number of groups in the chosen plan.
    used_requests: usize,
    /// `parent[i]` = previous cut point for `dp[i]`, or `-1` if none.
    /// Empty when the caller didn't request reconstruction.
    parent: Vec<i32>,
}

/// Inner DP minimising `Σ w(group) + λ · |groups|` subject to
/// `every w(group) ≤ b_cap`. Returns `(total_work, used_requests,
/// parent?)` for the optimal plan.
fn solve_piecewise_penalized_with_b(
    n: usize,
    edge_latency: &[f64],
    b_cap: f64,
    lambda_penalty: f64,
    with_parent: bool,
) -> WqsSolution {
    let mut out = WqsSolution {
        feasible: false,
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
        vec![-1i32; n + 1]
    } else {
        Vec::new()
    };
    dp[0] = 0.0;
    cnt[0] = 0;

    for i in 1..=n {
        let mut best = INF;
        let mut best_cnt = usize::MAX / 4;
        let mut best_k = -1i32;
        // Scan k from i-1 down to 0; latency grows as start shrinks
        // (larger byte span), so a `break` on `b_cap` exits early.
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
    out.used_requests = cnt[n];
    out.total_work = dp[n] - lambda_penalty * cnt[n] as f64;
    if with_parent {
        out.parent = parent;
    }
    out
}

/// Lagrangian outer loop: find the smallest `λ` such that the optimal
/// penalised plan uses `≤ request_cap` groups, then return that plan.
///
/// 1. Try `λ = 0`. If used_requests ≤ cap, done — that's the
///    unpenalised optimum and it already fits.
/// 2. Otherwise grow `λ` exponentially (1, 2, 4, …, 1e8) until the
///    penalised optimum drops to ≤ cap groups.
/// 3. Binary-search `λ` over `[lo, hi]` for 45 rounds, keeping the
///    feasible plan with smallest `total_work`.
/// 4. Re-run with `with_parent = true` at the chosen `λ` to recover
///    the cut chain. A few `f64::next_up` retries handle the case
///    where the parent reconstruction at exactly `best_lambda`
///    selects a slightly different partition than the value-only run.
fn solve_piecewise_wqs_minw_with_b(
    n: usize,
    request_cap: usize,
    edge_latency: &[f64],
    b_cap: f64,
    need_parent: bool,
) -> WqsSolution {
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

    let mut lo = 0.0f64;
    let mut hi = 1.0f64;
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
        return WqsSolution::default();
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
    // Tie-breaking nudge: at exactly `best_lambda` two plans may have
    // equal penalised cost; the no-parent run picked one, the
    // with-parent run another. Step λ up by one ULP a few times to
    // disambiguate.
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
    WqsSolution::default()
}

// ------- 4d. Plan reconstruction ---------------------------------------

/// Recover the objective value `T_UB = coeff_w · W + coeff_b · B` for
/// the plan ending at `dp[n][j]` by walking the parent chain.
fn objective_from_parent(ctx: &PlannerContext, j: usize, parent: &FlatTable<i32>) -> Option<f64> {
    let mut w_sum = 0.0f64;
    let mut b_max = 0.0f64;
    let mut i = ctx.n;
    let mut k = j;
    while k > 0 && i > 0 {
        let prev = *parent.get(i, k);
        if prev < 0 {
            return None;
        }
        let prev = prev as usize;
        let w = edge_at(ctx, i - 1, prev);
        w_sum += w;
        b_max = b_max.max(w);
        i = prev;
        k -= 1;
    }
    if i != 0 {
        return None;
    }
    Some(ctx.coeff_w * w_sum + ctx.coeff_b * b_max)
}

/// Recover the cut list from `dp[n][j]` by walking parent pointers.
fn cuts_from_parent(n: usize, j: usize, parent: &FlatTable<i32>) -> Cuts {
    let mut cuts = Vec::new();
    let mut i = n;
    let mut k = j;
    while k > 0 && i > 0 {
        let prev = *parent.get(i, k);
        if prev < 0 {
            return Cuts::new();
        }
        let prev = prev as usize;
        cuts.push(prev);
        i = prev;
        k -= 1;
    }
    cuts.push(n);
    cuts.sort_unstable();
    cuts
}

// ========================================================================
// 5. Warm start — solve once at B = ∞.
// ========================================================================

struct WarmStart {
    best_obj: f64,
    min_w: f64,
    best_plan: Cuts,
}

fn compute_warm_start(ctx: &PlannerContext, scratch: &mut Scratch) -> WarmStart {
    scratch.reset();
    run_inner_dp(ctx, INF, &mut scratch.dp, &mut scratch.parent);

    let mut best_obj = INF;
    let mut best_j: Option<usize> = None;
    let mut min_w = INF;
    let j_lo = if ctx.exact_requests { ctx.k_cap } else { 1 };
    for j in j_lo..=ctx.k_cap {
        let w = *scratch.dp.get(ctx.n, j);
        if w >= INF / 10.0 {
            continue;
        }
        min_w = min_w.min(w);
        if let Some(obj) = objective_from_parent(ctx, j, &scratch.parent) {
            if obj < best_obj - EPS {
                best_obj = obj;
                best_j = Some(j);
            }
        }
    }

    let best_plan = match best_j {
        Some(j) => cuts_from_parent(ctx.n, j, &scratch.parent),
        None => Cuts::new(),
    };

    WarmStart {
        best_obj,
        min_w,
        best_plan,
    }
}

/// Derive the `[min_b, max_b]` envelope for candidate enumeration from
/// the warm-start state. `min_b` is the largest single-request latency
/// (no plan can have a bottleneck below this); `max_b` is the
/// bottleneck at which `coeff_w · W* + coeff_b · B` already matches
/// the warm-start objective (candidates beyond this can't improve).
fn candidate_b_bounds(ctx: &PlannerContext, warm: &WarmStart) -> (f64, f64) {
    let mut max_single = 0.0f64;
    for i in 0..ctx.n {
        max_single = max_single.max(edge_at(ctx, i, i));
    }
    let max_b = if ctx.coeff_b > EPS && warm.min_w < INF / 10.0 {
        (warm.best_obj - ctx.coeff_w * warm.min_w) / ctx.coeff_b
    } else {
        INF
    };
    (max_single, max_b)
}

// ========================================================================
// 6. Candidate-B enumeration + pruning.
// ========================================================================

/// Enumerate the sorted list of distinct segment latencies in
/// `[min_b, max_b]`, then drop values whose greedy packing exceeds
/// `k_cap` (the tail below `first_feasible` can never yield a plan
/// respecting the request budget).
fn collect_candidate_b_values(ctx: &PlannerContext, min_b: f64, max_b: f64) -> Vec<f64> {
    let use_sparse = ctx.n * (ctx.n + 1) / 2 > DENSE_ENUMERATION_LIMIT;
    let mut c: Vec<f64> = if use_sparse {
        collect_sparse(ctx, min_b, max_b)
    } else {
        collect_dense(ctx, min_b, max_b)
    };
    c.sort_by(|a, b| a.total_cmp(b));
    c.dedup_by(|a, b| (*a - *b).abs() < 1e-12);

    if ctx.k_cap < ctx.n && !c.is_empty() {
        let mut lo = 0usize;
        let mut hi = c.len() - 1;
        let mut first_feasible = hi;
        while lo <= hi {
            let mid = lo + (hi - lo) / 2;
            if greedy_feasible(ctx, c[mid]) {
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
            c.drain(0..first_feasible);
        }
    }
    c
}

/// Full lower-triangular enumeration — every `(end, start)` latency
/// that falls in `[min_b, max_b]`.
fn collect_dense(ctx: &PlannerContext, min_b: f64, max_b: f64) -> Vec<f64> {
    let mut c = Vec::with_capacity(ctx.n * (ctx.n + 1) / 2);
    for end in 0..ctx.n {
        let mut start = end as i64;
        while start >= 0 {
            let s = start as usize;
            let w = edge_at(ctx, end, s);
            if w < min_b - EPS {
                start -= 1;
                continue;
            }
            if w > max_b + EPS {
                break;
            }
            c.push(w);
            start -= 1;
        }
    }
    c
}

/// Log-spaced sampling: for each `end`, collect latencies at
/// `start ∈ {end, end-1, end-2, end-4, end-8, …, 0}`. Produces
/// `O(n log n)` candidates — enough for the adaptive-hull search
/// while avoiding the `O(n²)` cost on large inputs.
fn collect_sparse(ctx: &PlannerContext, min_b: f64, max_b: f64) -> Vec<f64> {
    let mut c = Vec::with_capacity(ctx.n * (ctx.n + 1) / 2 / 16);
    for end in 0..ctx.n {
        let mut stride = 1i64;
        let mut start = end as i64;
        while start >= 0 {
            let s = start as usize;
            let w = edge_at(ctx, end, s);
            if w > max_b + EPS {
                break;
            }
            if w >= min_b - EPS {
                c.push(w);
            }
            start -= stride;
            stride = (stride * 2).min(1 << 20);
        }
        if end > 0 {
            let w = edge_at(ctx, end, 0);
            if w >= min_b - EPS && w <= max_b + EPS {
                c.push(w);
            }
        }
    }
    c
}

/// Cheap feasibility probe: greedily pack segments under `b_cap` and
/// check the group count stays within `k_cap`.
fn greedy_feasible(ctx: &PlannerContext, b_cap: f64) -> bool {
    let mut groups = 0usize;
    let mut start = 0usize;
    while start < ctx.n {
        groups += 1;
        if groups > ctx.k_cap {
            return false;
        }
        let mut end = start;
        while end + 1 < ctx.n {
            let w = edge_at(ctx, end + 1, start);
            if w <= b_cap + EPS {
                end += 1;
            } else {
                break;
            }
        }
        start = end + 1;
    }
    groups <= ctx.k_cap
}

// ========================================================================
// 7. Candidate-B search.
// ========================================================================

/// Evaluate every candidate; keep the best. Used when the candidate
/// list is short enough that the hull-search bookkeeping isn't worth it.
fn solve_by_exact_enumeration(
    ctx: &PlannerContext,
    scratch: &mut Scratch,
    candidates: &[f64],
    init_best_obj: f64,
    init_plan: Cuts,
) -> (f64, Cuts) {
    let mut best_obj = init_best_obj;
    let mut best_plan = init_plan;
    for &b_cap in candidates {
        let (obj, plan) = evaluate_at_b(ctx, scratch, b_cap);
        if obj < best_obj {
            best_obj = obj;
            best_plan = plan;
        }
    }
    (best_obj, best_plan)
}

// ------- 7a. Adaptive hull search --------------------------------------
//
// Maintain a priority queue of still-unsplit `(l, r)` gaps keyed by a
// convex-hull lower bound on the objective over `candidates[l..=r]`.
// Repeatedly split the most promising gap until the gap between
// `best_obj` and the best live lower bound falls below
// `ADAPTIVE_TARGET_RELATIVE_ERROR`, or the budget is exhausted.
// Evaluates O(log n) candidates in practice.

#[derive(Clone, Copy)]
struct EvalPoint {
    w_star: f64,
}

/// A still-unsplit gap `[l, r]` with its hull lower bound.
#[derive(Clone, Copy)]
struct IntervalNode {
    l: usize,
    r: usize,
    lb: f64,
}

impl PartialEq for IntervalNode {
    fn eq(&self, o: &Self) -> bool {
        self.l == o.l && self.r == o.r && self.lb.to_bits() == o.lb.to_bits()
    }
}
impl Eq for IntervalNode {}
impl Ord for IntervalNode {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        // Min-heap on lb, breaking ties toward wider intervals first.
        let by_lb = o.lb.total_cmp(&self.lb);
        if by_lb != std::cmp::Ordering::Equal {
            return by_lb;
        }
        (self.r - self.l)
            .cmp(&(o.r - o.l))
            .then_with(|| self.l.cmp(&o.l))
            .then_with(|| self.r.cmp(&o.r))
    }
}
impl PartialOrd for IntervalNode {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

struct HullState {
    visited: Vec<bool>,
    points: BTreeMap<usize, EvalPoint>,
    intervals: BinaryHeap<IntervalNode>,
    best_obj: f64,
    best_plan: Option<Cuts>,
    active_left: usize,
    total_evals: usize,
}

fn solve_by_adaptive_hull(
    ctx: &PlannerContext,
    scratch: &mut Scratch,
    candidates: &[f64],
    init_best_obj: f64,
    init_plan: Cuts,
    min_w: f64,
) -> (f64, Cuts) {
    if candidates.is_empty() {
        return (init_best_obj, init_plan);
    }
    let initial_right = candidates.len() - 1;

    let mut state = HullState {
        visited: vec![false; candidates.len()],
        points: BTreeMap::new(),
        intervals: BinaryHeap::new(),
        best_obj: init_best_obj,
        best_plan: None,
        active_left: 0,
        total_evals: 0,
    };

    let eval_budget = (ctx.k_cap + 8)
        .clamp(ADAPTIVE_MIN_EVAL_BUDGET, ADAPTIVE_MAX_EVAL_BUDGET)
        .min(candidates.len());

    // Seed with the two endpoints, lazily clipped by the current best.
    let Some(mut active_right) = hull_right_bound(
        ctx,
        candidates,
        initial_right,
        state.active_left,
        min_w,
        state.best_obj,
    ) else {
        return (state.best_obj, init_plan);
    };
    hull_eval_at(
        ctx,
        scratch,
        candidates,
        initial_right,
        state.active_left,
        active_right,
        &mut state,
    );

    if let Some(ar) = hull_right_bound(
        ctx,
        candidates,
        initial_right,
        state.active_left,
        min_w,
        state.best_obj,
    ) {
        active_right = ar;
        if !state.visited[active_right] && state.total_evals < eval_budget {
            hull_eval_at(
                ctx,
                scratch,
                candidates,
                initial_right,
                active_right,
                active_right,
                &mut state,
            );
        }
    }

    while state.total_evals < eval_budget {
        let Some(ar) = hull_right_bound(
            ctx,
            candidates,
            initial_right,
            state.active_left,
            min_w,
            state.best_obj,
        ) else {
            break;
        };
        active_right = ar;

        // Pop the top gap, re-computing stale lower bounds lazily.
        let top = loop {
            let Some(node) = state.intervals.peek().copied() else {
                break None;
            };
            if !is_open_gap(
                &state.points,
                node.l,
                node.r,
                state.active_left,
                active_right,
            ) {
                state.intervals.pop();
                continue;
            }
            let lb_now = hull_lb(ctx, candidates, &state.points, node.l, node.r);
            if (lb_now - node.lb).abs() > 1e-9 * lb_now.abs().max(1.0) {
                state.intervals.pop();
                state.intervals.push(IntervalNode {
                    l: node.l,
                    r: node.r,
                    lb: lb_now,
                });
                continue;
            }
            break Some(node);
        };

        let Some(best_interval) = top else { break };
        if best_interval.lb >= state.best_obj - EPS {
            break;
        }
        let rel_err = (state.best_obj - best_interval.lb) / best_interval.lb.abs().max(EPS);
        if rel_err <= ADAPTIVE_TARGET_RELATIVE_ERROR {
            break;
        }

        state.intervals.pop();
        if best_interval.r - best_interval.l > 1 {
            let mid = best_interval.l + (best_interval.r - best_interval.l) / 2;
            let ok = hull_eval_at(
                ctx,
                scratch,
                candidates,
                initial_right,
                mid,
                active_right,
                &mut state,
            );
            if !ok {
                // `mid` was infeasible; walk outward for a usable neighbour.
                let mut left = mid.saturating_sub(1);
                let mut right = mid + 1;
                while left > best_interval.l || right < best_interval.r {
                    if left > best_interval.l
                        && hull_eval_at(
                            ctx,
                            scratch,
                            candidates,
                            initial_right,
                            left,
                            active_right,
                            &mut state,
                        )
                    {
                        break;
                    }
                    if right < best_interval.r
                        && hull_eval_at(
                            ctx,
                            scratch,
                            candidates,
                            initial_right,
                            right,
                            active_right,
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
    }

    match state.best_plan {
        Some(plan) => (state.best_obj, plan),
        None => (init_best_obj, init_plan),
    }
}

/// Hull lower bound for the open interval `(l, r)`: a valid lower
/// bound on the objective for any B strictly inside, derived from the
/// two endpoint `(B, W*)` samples.
fn hull_lb(
    ctx: &PlannerContext,
    candidates: &[f64],
    pts: &BTreeMap<usize, EvalPoint>,
    l: usize,
    r: usize,
) -> f64 {
    let Some(pl) = pts.get(&l) else { return INF };
    let Some(pr) = pts.get(&r) else { return INF };
    let w_right = pr.w_star.min(pl.w_star);
    ctx.coeff_w * w_right + ctx.coeff_b * candidates[l]
}

/// Is `(l, r)` currently "open adjacent" — no sampled points lie
/// strictly between them, and both endpoints are inside the active
/// search window?
fn is_open_gap(
    pts: &BTreeMap<usize, EvalPoint>,
    l: usize,
    r: usize,
    a_lo: usize,
    a_hi: usize,
) -> bool {
    if l < a_lo || r > a_hi || l >= r || r - l <= 1 {
        return false;
    }
    let mut iter = pts.range(l..=r).map(|(k, _)| *k);
    matches!(
        (iter.next(), iter.next(), iter.next()),
        (Some(k1), Some(k2), None) if k1 == l && k2 == r
    )
}

/// Rightmost candidate index still able to beat `best_obj_now`. Uses
/// the warm-start `min_w` to derive a tight bottleneck cap.
fn hull_right_bound(
    ctx: &PlannerContext,
    candidates: &[f64],
    initial_right: usize,
    a_lo: usize,
    min_w: f64,
    best_obj_now: f64,
) -> Option<usize> {
    if a_lo > initial_right {
        return None;
    }
    if ctx.coeff_b <= EPS || min_w >= INF / 10.0 || best_obj_now >= INF / 10.0 {
        return Some(initial_right);
    }
    let bottleneck_limit = (best_obj_now - ctx.coeff_w * min_w) / ctx.coeff_b;
    let upper = candidates.partition_point(|&x| x <= bottleneck_limit + EPS);
    let idx = upper.saturating_sub(1).min(initial_right);
    if idx < a_lo {
        None
    } else {
        Some(idx)
    }
}

/// Evaluate a single candidate index, update caches, and return
/// `true` iff the evaluation produced a usable hull point.
fn hull_eval_at(
    ctx: &PlannerContext,
    scratch: &mut Scratch,
    candidates: &[f64],
    initial_right: usize,
    idx: usize,
    a_right_now: usize,
    state: &mut HullState,
) -> bool {
    if idx > initial_right || state.visited[idx] {
        return false;
    }
    state.visited[idx] = true;
    state.total_evals += 1;

    let (obj, plan) = evaluate_at_b(ctx, scratch, candidates[idx]);
    if obj < state.best_obj - EPS {
        state.best_obj = obj;
        state.best_plan = Some(plan);
    }

    if ctx.coeff_w > EPS && obj < INF / 10.0 {
        // Back out `W*` from `objective = coeff_w · W + coeff_b · B`.
        let w_star = (obj - ctx.coeff_b * candidates[idx]) / ctx.coeff_w;
        if let std::collections::btree_map::Entry::Vacant(e) = state.points.entry(idx) {
            e.insert(EvalPoint { w_star });
            if let Some(prev) = state.points.range(..idx).next_back().map(|(k, _)| *k) {
                if is_open_gap(&state.points, prev, idx, state.active_left, a_right_now) {
                    let lb = hull_lb(ctx, candidates, &state.points, prev, idx);
                    state.intervals.push(IntervalNode {
                        l: prev,
                        r: idx,
                        lb,
                    });
                }
            }
            if let Some(next) = state
                .points
                .range((std::ops::Bound::Excluded(idx), std::ops::Bound::Unbounded))
                .next()
                .map(|(k, _)| *k)
            {
                if is_open_gap(&state.points, idx, next, state.active_left, a_right_now) {
                    let lb = hull_lb(ctx, candidates, &state.points, idx, next);
                    state.intervals.push(IntervalNode {
                        l: idx,
                        r: next,
                        lb,
                    });
                }
            }
        } else {
            let entry = state.points.get_mut(&idx).unwrap();
            entry.w_star = entry.w_star.min(w_star);
        }
        true
    } else {
        if obj >= INF / 10.0 && idx >= state.active_left {
            state.active_left = idx.saturating_add(1);
        }
        false
    }
}

// ========================================================================
// 8. Plan finalization.
// ========================================================================

/// Turn a validated cut list into the public `IoPlan` shape. Falls
/// back to `Separate` if the cut list is empty (no feasible plan was
/// found).
fn finalize_plan(requests: &[IoRequest], ctx: &PlannerContext, cuts: Cuts) -> IoPlan {
    if cuts.len() < 2 {
        return Separate.generate_plan(requests, ctx.cost, ctx.constraints);
    }
    let n = requests.len();
    let mut merged_requests = Vec::with_capacity(cuts.len() - 1);
    let mut assignments = vec![0usize; n];
    for (k, win) in cuts.windows(2).enumerate() {
        let lo = win[0];
        let hi = win[1];
        let span = ctx.offsets[hi - 1] + ctx.sizes[hi - 1] - ctx.offsets[lo];
        merged_requests.push(MergedRequest {
            offset: ctx.offsets[lo],
            length: span,
        });
        assignments[lo..hi].fill(k);
    }
    IoPlan {
        merged_requests,
        assignments,
    }
}

// ========================================================================
// 9. Tests — smoke-level; heavier brute-force checks live in
//    `tests/main_solution_optimality.rs`.
// ========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io_plan::evaluate;
    use crate::plan_constraints::PlanConstraints;
    use crate::strategy::{Bottleneck, Separate, Serial};

    fn req(id: i32, offset: i64, length: i64) -> IoRequest {
        IoRequest {
            cluster_id: id,
            offset,
            length,
        }
    }

    fn cost_with(p: f64) -> CostModel {
        CostModel {
            latency_per_request: 10.0,
            latency_per_byte: 0.001,
            concurrent_parallelism: p,
            ..CostModel::default()
        }
    }

    fn no_constraints() -> PlanConstraints {
        PlanConstraints::default()
    }

    fn predicted_t_ub(m: &crate::io_plan::PlanMetrics, p: f64) -> f64 {
        m.t_ub_ms(p)
    }

    #[test]
    fn empty_input_gives_empty_plan() {
        let plan =
            MainSolution.generate_plan(&[], &CostModel::default(), &PlanConstraints::default());
        assert!(plan.merged_requests.is_empty());
    }

    #[test]
    fn parallelism_one_recovers_serial_optimum() {
        let rs = vec![req(0, 0, 100), req(1, 100, 100), req(2, 200, 100)];
        let cost = cost_with(1.0);
        let constraints = no_constraints();
        let b = MainSolution.generate_plan(&rs, &cost, &constraints);
        let d = Serial.generate_plan(&rs, &cost, &constraints);
        let b_m = evaluate(&rs, &b, &cost);
        let d_m = evaluate(&rs, &d, &cost);
        assert!(b_m.predicted_latency_ms_serial <= d_m.predicted_latency_ms_serial + 1e-6);
    }

    #[test]
    fn non_positive_parallelism_degenerates_to_bottleneck_objective() {
        let rs = vec![
            req(0, 0, 100),
            req(1, 100, 100),
            req(2, 5_000, 100),
            req(3, 5_100, 100),
        ];
        let main = CostModel {
            concurrent_parallelism: 0.0,
            ..cost_with(4.0)
        };
        let bottleneck = CostModel { ..cost_with(4.0) };
        let constraints = PlanConstraints {
            max_requests: 2,
            exact_requests: false,
        };

        let main_plan = MainSolution.generate_plan(&rs, &main, &constraints);
        let bottleneck_plan = Bottleneck.generate_plan(&rs, &bottleneck, &constraints);
        let main_m = evaluate(&rs, &main_plan, &main);
        let bottleneck_m = evaluate(&rs, &bottleneck_plan, &bottleneck);

        assert!(
            (main_m.predicted_latency_ms_concurrent - bottleneck_m.predicted_latency_ms_concurrent)
                .abs()
                <= 1e-6
        );
    }

    #[test]
    fn plan_is_always_valid_across_parallelism_settings() {
        let rs = vec![
            req(0, 0, 100),
            req(1, 1_000, 100),
            req(2, 2_000, 100),
            req(3, 3_000, 100),
            req(4, 4_000, 100),
        ];
        for &p in &[1.0, 2.0, 4.0, 8.0, 64.0] {
            let cost = cost_with(p);
            let plan = MainSolution.generate_plan(&rs, &cost, &no_constraints());
            plan.check_invariants(&rs)
                .unwrap_or_else(|e| panic!("bad plan at P={p}: {e}"));
        }
    }

    #[test]
    fn beats_or_ties_separate_under_any_parallelism() {
        let rs: Vec<_> = (0i32..12).map(|i| req(i, i as i64 * 500, 200)).collect();
        for &p in &[1.0, 2.0, 4.0, 8.0] {
            let cost = cost_with(p);
            let constraints = no_constraints();
            let balanced_m = evaluate(
                &rs,
                &MainSolution.generate_plan(&rs, &cost, &constraints),
                &cost,
            );
            let sep_m = evaluate(
                &rs,
                &Separate.generate_plan(&rs, &cost, &constraints),
                &cost,
            );
            let t_balanced = predicted_t_ub(&balanced_m, p);
            let t_separate = predicted_t_ub(&sep_m, p);
            assert!(
                t_balanced <= t_separate + 1e-6,
                "P={p}: balanced {t_balanced} > separate {t_separate}"
            );
        }
    }

    #[test]
    fn honours_max_requests() {
        let rs = (0..10)
            .map(|i| req(i as i32, i * 1_000, 100))
            .collect::<Vec<_>>();
        let cost = CostModel {
            latency_per_request: 10.0,
            latency_per_byte: 0.001,
            concurrent_parallelism: 2.0,
            ..CostModel::default()
        };
        let constraints = PlanConstraints {
            max_requests: 3,
            exact_requests: false,
        };
        let plan = MainSolution.generate_plan(&rs, &cost, &constraints);
        assert!(plan.merged_requests.len() <= 3);
        plan.check_invariants(&rs).unwrap();
    }

    #[test]
    fn beats_bottleneck_when_p_is_small() {
        let rs = (0..8)
            .map(|i| req(i as i32, i * 1_000, 100))
            .collect::<Vec<_>>();
        let cost = CostModel {
            latency_per_request: 10.0,
            latency_per_byte: 0.001,
            concurrent_parallelism: 1.0,
            ..CostModel::default()
        };
        let constraints = PlanConstraints {
            max_requests: 3,
            exact_requests: false,
        };
        let balanced_m = evaluate(
            &rs,
            &MainSolution.generate_plan(&rs, &cost, &constraints),
            &cost,
        );
        let concurrent_m = evaluate(
            &rs,
            &Bottleneck.generate_plan(&rs, &cost, &constraints),
            &cost,
        );
        let t_b = predicted_t_ub(&balanced_m, 1.0);
        let t_c = predicted_t_ub(&concurrent_m, 1.0);
        assert!(t_b <= t_c + 1e-6);
    }
}
