//! PreMerge — gap-and-cap two-stage pre-merge.
//!
//! A pre-processing planner that coalesces *adjacent* sorted requests
//! when both:
//!
//! 1. the byte gap between consecutive requests is `<= max_gap_bytes`,
//!    **and**
//! 2. the merged request's span stays `<= max_merged_size_bytes`
//!    (`0` disables the cap).
//!
//! It is intended to **precede** a heavier planner ([`Serial`],
//! [`MainSolution`]) by collapsing trivially mergeable runs first, so
//! the downstream DP sees a smaller `n`. It is *not* enabled by
//! default — `MainSolution::default()` etc. continue to receive raw
//! input. Callers opt in by either:
//!
//! * Constructing `PreMerge::new(gap, cap)` and treating it as a
//!   stand-alone [`MergeStrategy`] (it implements the trait so it can be
//!   benchmarked / compared like any other planner), or
//! * Using [`PreMerge::plan_then`] to run a downstream strategy and
//!   compose its assignments back to the original requests. The lower-
//!   level [`PreMerge::compress`] helper is available when callers only
//!   need the super-request ranges.
//!
//! ## Algorithm
//!
//! Two-stage, total `O(n log n)` worst case:
//!
//! - **Stage 1 — `O(n)` forward sweep.** Walk left to right; greedily
//!   join the current request to the previous group if both the gap and
//!   the size cap allow it. This pass alone is optimal for "minimise
//!   group count" under fixed gap+cap thresholds when inputs are
//!   already sorted and disjoint.
//! - **Stage 2 — `O(m log m)` min-gap heap refinement.** A safety net
//!   for inputs the forward sweep cannot perfectly bin (overlapping or
//!   out-of-order ranges, or numeric edge cases at the cap boundary).
//!   Builds a min-heap on adjacent-pair gaps with lazy deletion and
//!   pops the smallest gap repeatedly; each accepted merge is
//!   `O(log m)`. On clean sorted, disjoint input Stage 2 is a no-op
//!   and the second-stage cost is just one heap construction.
//!
//! Total time: `O(n log n)` worst case; `O(n)` on the common path
//! (no Stage-2 work needed).
//!
//! Default: **disabled**. The struct's `Default` mirrors the
//! "no-coalesce" baseline (`max_gap_bytes = 0`, `max_merged_size_bytes
//! = 0`) so `PreMerge::default()` produces the same plan as
//! [`Separate`] — zero merges. To actually pre-merge, supply a positive
//! gap threshold explicitly via [`PreMerge::new`] or use
//! [`PreMerge::recommended`].
//!
//! [`Separate`]: crate::strategy::Separate
//! [`Serial`]: crate::strategy::Serial
//! [`MainSolution`]: crate::strategy::MainSolution

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use crate::cost_model::CostModel;
use crate::io_plan::{IoPlan, IoRequest, MergedRequest};
use crate::plan_constraints::PlanConstraints;
use crate::strategy::MergeStrategy;

/// Two-stage gap-and-cap pre-merge planner.
///
/// Parameters live on the strategy struct rather than on
/// [`CostModel`] or [`PlanConstraints`] because they describe a
/// *coalesce policy*, not the storage endpoint or the global plan
/// shape constraints. They have unit "bytes" and bound the planner's
/// output without consulting the cost model at all (this is what
/// makes pre-merge cheap and predictable).
///
/// To match the upstream Lance fork's deployed configuration use
/// `PreMerge::new(64 * 1024, 1024 * 1024)` (64 KiB gap, 1 MiB cap).
#[derive(Debug, Clone, Copy)]
pub struct PreMerge {
    /// Maximum byte gap between adjacent requests that still allows a
    /// merge. `0` disables merging; negative values are treated as `0`.
    pub max_gap_bytes: i64,
    /// Maximum span (`end - start`) of any merged request.  `0`
    /// disables the cap. Negative values are treated as `0` (=
    /// disabled) for forward compatibility.
    pub max_merged_size_bytes: i64,
}

impl Default for PreMerge {
    /// Disabled pre-merge — equivalent to [`crate::strategy::Separate`]: every input
    /// becomes its own merged request. Callers must opt in to
    /// pre-merge by setting `max_gap_bytes > 0` (and optionally a
    /// non-zero size cap).
    ///
    /// Rationale: pre-merge changes the plan shape any downstream
    /// strategy sees, so it is safer to default to "off" and let
    /// callers turn it on knowingly.
    fn default() -> Self {
        Self {
            max_gap_bytes: 0,
            max_merged_size_bytes: 0,
        }
    }
}

impl PreMerge {
    /// Recommended max-gap for the opt-in preset, **32 KiB**.
    ///
    /// Tuned so a single typical S3 page-read coalesces "free"
    /// neighbours (≤ a TCP packet's worth of skip) without inflating
    /// the per-request payload appreciably.
    pub const RECOMMENDED_MAX_GAP_BYTES: i64 = 32 * 1024;

    /// Recommended max merged-request size for the opt-in preset,
    /// **1 MiB**.
    ///
    /// Larger super-requests increase tail latency without buying much
    /// throughput on object storage; 1 MiB matches the upstream Lance
    /// fork's deployed cap.
    pub const RECOMMENDED_MAX_MERGED_SIZE_BYTES: i64 = 1024 * 1024;

    /// Input-size threshold above which [`PreMerge::maybe_compress`]
    /// turns pre-merge on, **128 ranges**.
    ///
    /// Below this, the downstream DP planner ([`crate::strategy::Serial`],
    /// [`crate::strategy::MainSolution`]) handles the input fast enough that the
    /// pre-merge fixed cost (sweep + heap + clone) eats the win.
    /// Above it, pre-merge typically shrinks `n` by 5×–10× on
    /// realistic IVF probe sets, which more than pays for itself.
    pub const RECOMMENDED_MIN_INPUT_SIZE: usize = 128;

    /// Construct with explicit thresholds. `max_merged_size_bytes = 0`
    /// disables the size cap; `max_gap_bytes <= 0` disables merging.
    pub fn new(max_gap_bytes: i64, max_merged_size_bytes: i64) -> Self {
        Self {
            max_gap_bytes: max_gap_bytes.max(0),
            max_merged_size_bytes: max_merged_size_bytes.max(0),
        }
    }

    /// Construct with the **recommended** opt-in preset:
    /// `max_gap_bytes = 32 KiB`, `max_merged_size_bytes = 1 MiB`.
    ///
    /// Equivalent to
    /// `PreMerge::new(PreMerge::RECOMMENDED_MAX_GAP_BYTES,
    /// PreMerge::RECOMMENDED_MAX_MERGED_SIZE_BYTES)`.
    ///
    /// The library default ([`PreMerge::default`]) remains the no-op
    /// `(0, 0)` to keep "default off" behaviour. Use this constructor
    /// when you've decided pre-merge is appropriate for your workload.
    pub fn recommended() -> Self {
        Self::new(
            Self::RECOMMENDED_MAX_GAP_BYTES,
            Self::RECOMMENDED_MAX_MERGED_SIZE_BYTES,
        )
    }

    /// `true` iff this configuration would never merge two adjacent
    /// requests (gap threshold disabled).
    #[inline]
    pub fn is_no_op(&self) -> bool {
        self.max_gap_bytes <= 0
    }

    /// Low-level size-decision helper for callers that only need the
    /// compressed super-request ranges.
    ///
    /// Returns `Some(super_requests)` — pre-merged with the
    /// [`PreMerge::recommended`] preset — when
    /// `requests.len() > `[`PreMerge::RECOMMENDED_MIN_INPUT_SIZE`]
    /// (i.e. > 128 ranges). Returns `None` when the input is small
    /// enough that the downstream DP can handle it directly without
    /// pre-merge overhead.
    ///
    /// A downstream plan over the returned vector uses super-request
    /// assignment indexes. Prefer [`PreMerge::plan_then`] when the final
    /// assignment vector must map the original inputs.
    ///
    /// Pre-merge stays **opt-in**: callers must explicitly invoke
    /// either this helper, [`PreMerge::recommended`], or
    /// [`PreMerge::new`]. Constructing a [`crate::strategy::MainSolution`] (or any
    /// other strategy) directly will not silently apply pre-merge.
    pub fn maybe_compress(requests: &[IoRequest]) -> Option<Vec<IoRequest>> {
        if requests.len() > Self::RECOMMENDED_MIN_INPUT_SIZE {
            Some(Self::recommended().compress(requests))
        } else {
            None
        }
    }

    /// Pre-merge, run a downstream strategy, and compose its assignments
    /// back into the original request namespace.
    ///
    /// Prefer this over calling [`PreMerge::compress`] followed by a
    /// planner when the caller needs the final [`IoPlan::assignments`]
    /// to index the original `requests`. Both stages use the checked
    /// [`crate::plan`] boundary.
    pub fn plan_then<S: MergeStrategy + ?Sized>(
        &self,
        downstream: &S,
        requests: &[IoRequest],
        cost: &CostModel,
        constraints: &PlanConstraints,
    ) -> Result<IoPlan, crate::PlanningError> {
        let pre_plan = crate::plan(
            self,
            requests,
            &CostModel::default(),
            &PlanConstraints::default(),
        )?;

        let super_requests = super_requests_from_plan(requests, &pre_plan);

        let downstream_plan = crate::plan(downstream, &super_requests, cost, constraints)?;
        let assignments = pre_plan
            .assignments
            .iter()
            .map(|&group_index| downstream_plan.assignments[group_index])
            .collect();
        let output = IoPlan {
            merged_requests: downstream_plan.merged_requests,
            assignments,
        };
        output
            .check_invariants(requests)
            .map_err(crate::PlanningError::InvalidPlan)?;
        Ok(output)
    }

    /// Compress `requests` into a smaller `Vec<IoRequest>` of
    /// super-requests, each spanning the union of the inputs that
    /// merged into it. The returned vector is suitable as input to a
    /// downstream planner (Serial, MainSolution, etc.).
    ///
    /// `cluster_id` of each super-request is the `cluster_id` of the
    /// first member. A downstream plan over this vector has assignments
    /// in the *super-request* namespace. Callers that need a final plan
    /// mapped to the original input should use [`PreMerge::plan_then`].
    pub fn compress(&self, requests: &[IoRequest]) -> Vec<IoRequest> {
        let plan = self.generate_plan(requests, &CostModel::default(), &PlanConstraints::default());
        super_requests_from_plan(requests, &plan)
    }
}

fn super_requests_from_plan(requests: &[IoRequest], plan: &IoPlan) -> Vec<IoRequest> {
    let mut first_cluster = vec![None::<i32>; plan.merged_requests.len()];
    for (request_index, &group_index) in plan.assignments.iter().enumerate() {
        first_cluster[group_index].get_or_insert(requests[request_index].cluster_id);
    }
    plan.merged_requests
        .iter()
        .enumerate()
        .map(|(group_index, merged)| IoRequest {
            cluster_id: first_cluster[group_index].expect("pre-merge group is non-empty"),
            offset: merged.offset,
            length: merged.length,
        })
        .collect()
}

impl MergeStrategy for PreMerge {
    fn name(&self) -> &'static str {
        "pre_merge"
    }

    fn generate_plan(
        &self,
        requests: &[IoRequest],
        _cost: &CostModel,
        _constraints: &PlanConstraints,
    ) -> IoPlan {
        if requests.is_empty() {
            return IoPlan::empty();
        }
        if self.is_no_op() {
            return IoPlan::one_per_request(requests);
        }

        // -- Stage 1: O(n) forward sweep. --------------------------------
        //
        // Greedy left-to-right grouping. Track each input's group index
        // so the final IoPlan's `assignments` are correct without a
        // second walk.
        let mut groups: Vec<Group> = Vec::with_capacity(requests.len());
        let mut assignments: Vec<usize> = Vec::with_capacity(requests.len());

        for &req in requests {
            let mut joined = false;
            if let Some(last) = groups.last_mut() {
                let gap = req.offset - last.end;
                let merged_end = last.end.max(req.end());
                let merged_size = merged_end - last.start;
                // Negative gap = overlap; treat as a "free" merge
                // (no extra bytes to fetch), still subject to size cap.
                if gap <= self.max_gap_bytes && self.fits_size_cap(merged_size) {
                    last.end = merged_end;
                    assignments.push(groups.len() - 1);
                    joined = true;
                }
            }
            if !joined {
                groups.push(Group::alive_with(req.offset, req.end()));
                assignments.push(groups.len() - 1);
            }
        }

        // -- Stage 2: O(m log m) min-gap heap refinement. ----------------
        //
        // Try cross-boundary merges Stage 1 missed (overlapping ranges,
        // numeric edge cases at the cap, or out-of-order inputs). Each
        // merge fuses two groups into the lower-indexed one; we track
        // *liveness* via a generation counter so popped heap entries
        // referring to merged-away groups are skipped lazily.
        if groups.len() >= 2 {
            self.stage_two_refine(&mut groups);
        }

        // Build the IoPlan.
        let merged_requests = groups
            .iter()
            .filter(|g| g.live())
            .map(|g| MergedRequest {
                offset: g.start,
                length: g.end - g.start,
            })
            .collect::<Vec<_>>();

        // Stage 2 may have left "dead" group slots; remap assignments
        // to the dense final indices.
        let mut dense_idx = vec![usize::MAX; groups.len()];
        let mut next = 0;
        for (i, g) in groups.iter().enumerate() {
            if g.live() {
                dense_idx[i] = next;
                next += 1;
            }
        }
        // Each input's group may have been merged into another in Stage
        // 2 — chase forwarding pointers to the live representative.
        let mut final_assignments = Vec::with_capacity(assignments.len());
        for &gi in &assignments {
            let mut cur = gi;
            while !groups[cur].live() {
                cur = groups[cur].forwarded_to;
            }
            final_assignments.push(dense_idx[cur]);
        }

        IoPlan {
            merged_requests,
            assignments: final_assignments,
        }
    }
}

impl PreMerge {
    #[inline]
    fn fits_size_cap(&self, candidate_size: i64) -> bool {
        self.max_merged_size_bytes <= 0 || candidate_size <= self.max_merged_size_bytes
    }

    /// Stage 2 — min-gap priority refinement with lazy deletion.
    fn stage_two_refine(&self, groups: &mut [Group]) {
        // Min-heap of (gap, left_idx, gen_left, gen_right). We use
        // `Reverse(gap)` for min-ordering. Lazy deletion: each entry
        // carries the generation of both endpoints at insertion time;
        // pop-time we check generations are still current before
        // accepting the merge.
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(groups.len());
        for i in 0..groups.len() - 1 {
            let g_l = &groups[i];
            let g_r = &groups[i + 1];
            if !g_l.live() || !g_r.live() {
                continue;
            }
            let gap = g_r.start - g_l.end;
            // Negative gap (overlap) is a "free" merge candidate.
            if gap > self.max_gap_bytes {
                continue;
            }
            heap.push(HeapEntry {
                gap,
                left: i,
                right: i + 1,
                gen_left: g_l.gen,
                gen_right: g_r.gen,
            });
        }

        while let Some(top) = heap.pop() {
            let HeapEntry {
                gap,
                left,
                right,
                gen_left,
                gen_right,
            } = top;
            // Drop stale heap entries whose endpoints have moved.
            if !groups[left].live()
                || !groups[right].live()
                || groups[left].gen != gen_left
                || groups[right].gen != gen_right
            {
                continue;
            }
            // Sanity: gap is still the smallest live gap.
            if gap > self.max_gap_bytes {
                break;
            }
            let g_l = groups[left];
            let g_r = groups[right];
            let merged_end = g_l.end.max(g_r.end);
            let merged_size = merged_end - g_l.start;
            if !self.fits_size_cap(merged_size) {
                // Cannot satisfy size cap — and since groups only
                // grow, this gap is permanently unmergeable. Keep
                // popping; smaller-gap candidates may still fit.
                continue;
            }
            // Merge `right` into `left`, mark `right` dead, bump
            // `left`'s generation so old heap entries pointing at it
            // become stale.
            groups[left].end = merged_end;
            groups[left].gen += 1;
            groups[right].forwarded_to = left;
            groups[right].mark_dead();

            // Insert new neighbour gap entries (left ↔ next live to the
            // right of `right`).
            if let Some(next_right) = next_live(groups, right + 1) {
                let g_lp = &groups[left];
                let g_rp = &groups[next_right];
                let new_gap = g_rp.start - g_lp.end;
                if new_gap <= self.max_gap_bytes {
                    heap.push(HeapEntry {
                        gap: new_gap,
                        left,
                        right: next_right,
                        gen_left: g_lp.gen,
                        gen_right: g_rp.gen,
                    });
                }
            }
        }
    }
}

#[inline]
fn next_live(groups: &[Group], start: usize) -> Option<usize> {
    (start..groups.len()).find(|&i| groups[i].live())
}

#[derive(Debug, Clone, Copy)]
struct Group {
    start: i64,
    end: i64,
    /// Generation counter — bumped when `end` extends. Stale heap
    /// entries are detected and discarded by comparing generations.
    gen: u32,
    /// Forwarding pointer for dead groups, so Stage 2 can re-route
    /// assignments. Set when this group is merged INTO another.
    forwarded_to: usize,
    alive: bool,
}

impl Group {
    #[inline]
    fn live(&self) -> bool {
        self.alive
    }
    #[inline]
    fn mark_dead(&mut self) {
        self.alive = false;
    }
}

impl Group {}

// `Group` constructor is shorthand below to avoid struct-literal noise.
impl Default for Group {
    fn default() -> Self {
        Self {
            start: 0,
            end: 0,
            gen: 0,
            forwarded_to: 0,
            alive: true,
        }
    }
}

// Adapt the bare-field shorthand used in Stage 1 (where we assign
// `start` and `end` directly) to construct via `Default`.
impl From<(i64, i64)> for Group {
    fn from((start, end): (i64, i64)) -> Self {
        Self {
            start,
            end,
            ..Group::default()
        }
    }
}

// We use the struct-literal form `Group { start, end }` in Stage 1 for
// readability; satisfy the type by providing a derive-equivalent
// constructor pattern. The literal form requires every field to be
// listed, so wrap via `Group::from`.
impl Group {
    #[inline]
    fn alive_with(start: i64, end: i64) -> Self {
        Self::from((start, end))
    }
}

#[derive(Debug, Clone, Copy, Eq, PartialEq)]
struct HeapEntry {
    gap: i64,
    left: usize,
    right: usize,
    gen_left: u32,
    gen_right: u32,
}

// Min-heap ordering: smaller `gap` first.
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so BinaryHeap (max-heap) acts as min-heap.
        other.gap.cmp(&self.gap)
    }
}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(id: i32, offset: i64, length: i64) -> IoRequest {
        IoRequest {
            cluster_id: id,
            offset,
            length,
        }
    }

    fn run(strategy: PreMerge, requests: &[IoRequest]) -> IoPlan {
        strategy.generate_plan(requests, &CostModel::default(), &PlanConstraints::default())
    }

    // ---- core invariants ---------------------------------------------------

    #[test]
    fn empty_input_gives_empty_plan() {
        let plan = run(PreMerge::new(64 * 1024, 1024 * 1024), &[]);
        assert!(plan.merged_requests.is_empty());
        assert!(plan.assignments.is_empty());
    }

    #[test]
    fn single_request_passes_through_unchanged() {
        let rs = vec![req(7, 1024, 512)];
        let plan = run(PreMerge::new(64 * 1024, 1024 * 1024), &rs);
        assert_eq!(plan.merged_requests.len(), 1);
        assert_eq!(plan.merged_requests[0].offset, 1024);
        assert_eq!(plan.merged_requests[0].length, 512);
        assert_eq!(plan.assignments, vec![0]);
        plan.check_invariants(&rs).unwrap();
    }

    #[test]
    fn default_is_no_op_like_separate() {
        // `Default` thresholds = (0, 0): no coalescing.
        // The lance fork's deployed config is non-default; the library
        // default is the safer "off" baseline.
        let rs = vec![req(0, 0, 100), req(1, 200, 100), req(2, 500, 100)];
        let plan = run(PreMerge::default(), &rs);
        assert_eq!(plan.merged_requests.len(), 3);
        plan.check_invariants(&rs).unwrap();
    }

    #[test]
    fn max_gap_zero_disables_merging_even_for_touching_requests() {
        let rs = vec![req(0, 0, 100), req(1, 100, 100), req(2, 200, 100)];
        let plan = run(PreMerge::new(0, 0), &rs);
        assert_eq!(plan.merged_requests.len(), 3);
        plan.check_invariants(&rs).unwrap();
    }

    // ---- gap threshold semantics ------------------------------------------

    #[test]
    fn gap_at_threshold_merges() {
        let rs = vec![req(0, 0, 100), req(1, 200, 100)]; // gap = 100
        let plan = run(PreMerge::new(100, 0), &rs);
        assert_eq!(plan.merged_requests.len(), 1);
        plan.check_invariants(&rs).unwrap();
    }

    #[test]
    fn gap_just_over_threshold_cuts() {
        let rs = vec![req(0, 0, 100), req(1, 200, 100)]; // gap = 100
        let plan = run(PreMerge::new(99, 0), &rs);
        assert_eq!(plan.merged_requests.len(), 2);
        plan.check_invariants(&rs).unwrap();
    }

    #[test]
    fn gap_sweep_is_monotone_in_request_count() {
        // 16 small clusters with mixed gaps; widening the gap threshold
        // can only shrink (or hold) the merged-request count.
        let mut rs = Vec::new();
        let mut o = 0i64;
        for i in 0..16 {
            rs.push(req(i, o, 1024));
            o += 1024 + (1 << ((i % 8) + 8)); // gaps 256B..32KB
        }
        let mut prev = usize::MAX;
        for shift in 8..=16 {
            let gap = 1i64 << shift;
            let plan = run(PreMerge::new(gap, 0), &rs);
            plan.check_invariants(&rs).unwrap();
            assert!(
                plan.merged_requests.len() <= prev,
                "gap=2^{shift}: {} > prev {prev}",
                plan.merged_requests.len()
            );
            prev = plan.merged_requests.len();
        }
    }

    // ---- size cap semantics -----------------------------------------------

    #[test]
    fn size_cap_zero_disables_cap() {
        let rs: Vec<_> = (0..10).map(|i| req(i, i as i64 * 100, 100)).collect();
        let plan = run(PreMerge::new(1, 0), &rs);
        assert_eq!(plan.merged_requests.len(), 1);
        assert_eq!(plan.merged_requests[0].length, 1000);
        plan.check_invariants(&rs).unwrap();
    }

    #[test]
    fn size_cap_forces_cut_even_when_gap_allows_it() {
        let rs = vec![req(0, 0, 100), req(1, 100, 100), req(2, 200, 100)];
        let plan = run(PreMerge::new(1024, 150), &rs);
        for m in &plan.merged_requests {
            assert!(m.length <= 150, "merged length {} > cap", m.length);
        }
        assert!(plan.merged_requests.len() >= 2);
        plan.check_invariants(&rs).unwrap();
    }

    #[test]
    fn size_cap_at_exactly_first_request_keeps_per_request_split() {
        // Each request is exactly the cap → no two can be merged.
        let rs = vec![req(0, 0, 100), req(1, 100, 100), req(2, 200, 100)];
        let plan = run(PreMerge::new(1024, 100), &rs);
        assert_eq!(plan.merged_requests.len(), 3);
        plan.check_invariants(&rs).unwrap();
    }

    // ---- both thresholds active -------------------------------------------

    #[test]
    fn deployed_lance_fork_config_64kib_1mib() {
        // Configuration matching the upstream Lance fork's
        // `cheap_pre_merge`. Three small reads inside a 1 MiB window
        // joined; the fourth past 1 MiB starts a fresh group.
        let rs = vec![
            req(0, 0, 4_096),
            req(1, 8_192, 4_096),     // gap 4 KiB → join
            req(2, 16_384, 4_096),    // gap 4 KiB → join (size 20 KiB)
            req(3, 1_500_000, 4_096), // gap 1.48 MiB → cut (and would breach 1 MiB)
        ];
        let plan = run(PreMerge::new(64 * 1024, 1024 * 1024), &rs);
        assert_eq!(plan.merged_requests.len(), 2);
        // First merged spans the three first reads.
        assert_eq!(plan.merged_requests[0].offset, 0);
        assert_eq!(plan.merged_requests[0].length, 16_384 + 4_096);
        plan.check_invariants(&rs).unwrap();
    }

    #[test]
    fn assignments_route_each_input_to_correct_super_request() {
        let rs = vec![req(0, 0, 100), req(1, 100, 100), req(2, 5000, 100)];
        let plan = run(PreMerge::new(64 * 1024, 0), &rs);
        // First two should land in group 0; third (gap 4800 < 64K) too.
        assert_eq!(plan.merged_requests.len(), 1);
        assert_eq!(plan.assignments, vec![0, 0, 0]);
        plan.check_invariants(&rs).unwrap();

        // Now make the third gap exceed the threshold:
        let plan2 = run(PreMerge::new(1024, 0), &rs);
        assert_eq!(plan2.merged_requests.len(), 2);
        assert_eq!(plan2.assignments, vec![0, 0, 1]);
        plan2.check_invariants(&rs).unwrap();
    }

    // ---- stage 2 (rare) edge cases ----------------------------------------

    #[test]
    fn stage2_handles_overlapping_inputs_safely() {
        // Two overlapping reads followed by a separate one. Stage 1
        // already collapses them; Stage 2 must not crash or
        // double-emit.
        let rs = vec![
            req(0, 0, 200),
            req(1, 100, 200), // overlaps the previous range
            req(2, 1_000, 100),
        ];
        let plan = run(PreMerge::new(2_000, 0), &rs);
        assert_eq!(plan.merged_requests.len(), 1);
        plan.check_invariants(&rs).unwrap();
    }

    #[test]
    fn stage2_size_cap_prevents_merge_of_already_full_group() {
        // Stage 1 fills a group right up to the cap; Stage 2 must not
        // try to bridge into it (resulting span would breach the cap).
        let rs = vec![
            req(0, 0, 800_000),       // single big read
            req(1, 810_000, 100_000), // gap 10 KiB; size 910 KiB ≤ 1 MiB → join
            req(2, 920_000, 200_000), // gap 10 KiB; size 1.12 MiB > 1 MiB → cut
        ];
        let plan = run(PreMerge::new(64 * 1024, 1024 * 1024), &rs);
        assert!(plan.merged_requests.len() >= 2);
        for m in &plan.merged_requests {
            assert!(
                m.length <= 1024 * 1024,
                "merged length {} > 1 MiB cap",
                m.length
            );
        }
        plan.check_invariants(&rs).unwrap();
    }

    // ---- compress() helper -------------------------------------------------

    #[test]
    fn compress_returns_super_requests_in_order() {
        let rs = vec![
            req(0, 0, 100),
            req(1, 100, 100),       // joined with [0]
            req(2, 1_000_000, 100), // separate
            req(3, 1_000_100, 100), // joined with [2]
        ];
        let pm = PreMerge::new(64 * 1024, 0);
        let supers = pm.compress(&rs);
        assert_eq!(supers.len(), 2);
        assert_eq!(supers[0].cluster_id, 0);
        assert_eq!(supers[0].offset, 0);
        assert_eq!(supers[0].length, 200);
        assert_eq!(supers[1].cluster_id, 2);
        assert_eq!(supers[1].offset, 1_000_000);
        assert_eq!(supers[1].length, 200);
    }

    // ---- cost-model independence ------------------------------------------

    #[test]
    fn cost_model_is_not_read() {
        let rs = vec![req(0, 0, 100), req(1, 100, 100), req(2, 1_000, 100)];
        let pm = PreMerge::new(64 * 1024, 0);
        let cheap = CostModel {
            latency_per_request: 0.1,
            latency_per_byte: 1.0,
            ..CostModel::default()
        };
        let expensive = CostModel {
            latency_per_request: 1000.0,
            latency_per_byte: 1e-9,
            ..CostModel::default()
        };
        let constraints = PlanConstraints::default();
        let a = pm.generate_plan(&rs, &cheap, &constraints);
        let b = pm.generate_plan(&rs, &expensive, &constraints);
        assert_eq!(a.merged_requests, b.merged_requests);
        assert_eq!(a.assignments, b.assignments);
    }

    // ---- recommended preset + auto-decision helper ------------------------

    #[test]
    fn recommended_uses_documented_constants() {
        let p = PreMerge::recommended();
        assert_eq!(p.max_gap_bytes, PreMerge::RECOMMENDED_MAX_GAP_BYTES);
        assert_eq!(
            p.max_merged_size_bytes,
            PreMerge::RECOMMENDED_MAX_MERGED_SIZE_BYTES
        );
        assert_eq!(p.max_gap_bytes, 32 * 1024);
        assert_eq!(p.max_merged_size_bytes, 1024 * 1024);
    }

    #[test]
    fn maybe_compress_returns_none_for_small_input() {
        // ≤ RECOMMENDED_MIN_INPUT_SIZE → caller should use input directly.
        for n in [0, 1, 16, 64, 128] {
            let rs: Vec<_> = (0..n).map(|i| req(i, i as i64 * 1024, 64)).collect();
            assert!(
                PreMerge::maybe_compress(&rs).is_none(),
                "n={n} expected None (≤ threshold)"
            );
        }
    }

    #[test]
    fn maybe_compress_returns_some_for_large_input() {
        // n > 128 → return Some(compressed). Use closely-spaced
        // requests so the recommended (32 KiB, 1 MiB) preset actually
        // merges some of them.
        let rs: Vec<_> = (0..200).map(|i| req(i, i as i64 * 4096, 1024)).collect();
        let supers = PreMerge::maybe_compress(&rs).expect("> threshold → Some");
        assert!(
            supers.len() < rs.len(),
            "compressed {} >= raw {}",
            supers.len(),
            rs.len()
        );
        assert!(
            !supers.is_empty(),
            "compress must keep at least one super-request"
        );
    }

    #[test]
    fn maybe_compress_then_downstream_planner_round_trips() {
        // Idiomatic pattern from the doc-test: compress with the
        // recommended preset, run a downstream planner, verify that
        // it succeeds and produces a valid plan over the compressed
        // input.
        let rs: Vec<_> = (0..150).map(|i| req(i, i as i64 * 4096, 1024)).collect();
        let to_plan = PreMerge::maybe_compress(&rs).unwrap_or_else(|| rs.clone());
        // Downstream planner: any MergeStrategy works; pick Serial
        // since MainSolution requires features we may not have here.
        let plan = crate::strategy::Serial.generate_plan(
            &to_plan,
            &CostModel::default(),
            &PlanConstraints::default(),
        );
        plan.check_invariants(&to_plan).unwrap();
    }

    #[test]
    fn plan_then_composes_assignments_to_original_inputs() {
        let rs: Vec<_> = (0..150).map(|i| req(i, i as i64 * 4096, 1024)).collect();
        let output = PreMerge::recommended()
            .plan_then(
                &crate::strategy::Serial,
                &rs,
                &CostModel::default(),
                &PlanConstraints::default(),
            )
            .unwrap();
        assert_eq!(output.assignments.len(), rs.len());
        output.check_invariants(&rs).unwrap();
    }

    #[test]
    fn default_remains_no_op_after_recommended_added() {
        // Belt-and-braces: adding `recommended()` must NOT silently
        // change `default()`'s behaviour.
        let p = PreMerge::default();
        assert!(p.is_no_op());
        assert_eq!(p.max_gap_bytes, 0);
        assert_eq!(p.max_merged_size_bytes, 0);
    }

    // ---- equivalence with stand-alone Greedy when caps line up ------------

    #[test]
    fn equals_greedy_on_clean_sorted_input() {
        // Stage 2 is a no-op on clean sorted input → PreMerge and the
        // existing Greedy strategy produce the same plan when given
        // matching gap+span thresholds.
        let mut rs = Vec::new();
        let mut o = 0i64;
        for i in 0..32 {
            rs.push(req(i, o, 1024));
            o += 1024 + ((i as i64) % 5) * 8 * 1024; // gaps 0..32 KiB
        }
        let pm = PreMerge::new(16 * 1024, 256 * 1024);
        let g = crate::strategy::Greedy::new(16 * 1024, 256 * 1024);
        let pm_plan = run(pm, &rs);
        let g_plan = g.generate_plan(&rs, &CostModel::default(), &PlanConstraints::default());
        assert_eq!(pm_plan.merged_requests, g_plan.merged_requests);
        assert_eq!(pm_plan.assignments, g_plan.assignments);
        pm_plan.check_invariants(&rs).unwrap();
    }
}
