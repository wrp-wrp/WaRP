# Benchmark methodology

The Rust benches live under `rust/benches/` (Criterion) and are driven
by `scripts/bench.sh`. Together they cover two distinct axes:

1. **Plan time** — wall-clock microseconds per planner call. Full suites
   isolate `generate_plan`; `planner_smoke` includes the checked public
   boundary. Criterion reports confidence intervals and stores HTML output.
2. **Plan quality** — predicted latency, request count, wasted bytes,
   and GET cost under a fixed cost model. Results include absolute
   values and percentage savings against `Separate`.

`ivf_workload` additionally emits a machine-readable CSV with median
and p99 planner timing per fixture. The other suites emit Criterion
reports and stderr logs; they do not pretend to share one CSV schema.

## Bench files

| File | Input | Axes |
| --- | --- | --- |
| `ivf_workload.rs`       | FAISS-IVF fixtures (`rust/tests/data/*`) | distribution × strategy × concurrency, latency/cost table + CSV |
| `plan_time.rs`          | synthetic distributions | strategy × cluster-count, Criterion timing |
| `plan_quality.rs`       | synthetic distributions | strategy × distribution, quality table + Criterion timing |
| `main_solution_speed.rs`| realistic exp-gap inputs | n-scaling across linear-fast, DC/small-cap, and WQS/large-cap regimes + P-sweep |
| `planner_smoke.rs`      | deterministic synthetic inputs | short checked-public-API timing for linear, piecewise small-cap, and piecewise large-cap paths |

`MainSolution` dispatches between four inner solvers based on the
cost shape:

| Cost shape | k_cap | Solver |
|---|---|---|
| linear (or piecewise with span ≤ first break) | any | monotonic-deque `O(n·k)` linear fast path |
| piecewise convex | `≥ n/4` | **WQS** (Lagrangian, ~50 1D `O(n²)` solves) |
| piecewise convex | `< n/4` | divide-and-conquer `O(n·k·log n)` per layer |
| non-convex piecewise | any | dense `O(n²·k)` plain DP (fallback) |

The WQS dispatch (`is_large_k_for_wqs(n, k) = k·4 ≥ n`) covers the
large-cap shape represented by the paper's K=n cells. Current-host
synthetic timing is not treated as a reproduction of those published
measurements.

## Distributions

- **uniform_contiguous** — every cluster is the same size, zero gaps.
  Stresses the overhead-vs-bytes tradeoff; merging is unambiguously
  beneficial.
- **exp_gaps** — cluster sizes and gaps from independent exponential
  distributions. Approximates the `sift1m` IVF layout after PC1
  (Hilbert-like) reorder.
- **heavy_tail** — 90 % tiny gaps, 10 % huge gaps. Approximates the
  unreordered IVF layout.
- **sift_paper_4096x128** — SIFT1M with FAISS `IndexIVFFlat`,
  `nlist=4096`, `nprobe=128`, and cluster-id order.
- **sift_paper_pca_4096x128** — the same SIFT1M index with lists ordered
  by centroid PC1, a simple spatial-reordering proxy.

Each SIFT1M fixture manifest records the input-array hashes, FAISS and
NumPy versions, training size, index geometry, query count, and layout.

Synthetic sizes sweep 64 → 4096 clusters per query.

## Cost model

All benches use a piecewise-linear S3 calibration:

- `latency_pw_breaks_bytes = [3_145_728]`
- `latency_pw_overheads_ms = [28.85, 6.27]`
- `latency_pw_rates_ms_per_byte = [2.82e-6, 1.00e-5]`
- no shared span cap
- `concurrent_parallelism` swept over `{1, 2, 4, 8, 16, 32, 64}` in
  the headline `ivf_workload` bench.

To use your own calibration instead, either edit the `paper_cost`
helper in each bench file or drive the planner directly from
application code with a `CostModel` populated by `Calibrator`.

## Quick demo (sub-second, no external data)

```bash
cargo run --release --example savings_demo     # latency / cost comparison
cargo run --release --example paper_table4     # synthetic current-host overhead spot check
```

`savings_demo` builds an IVF-style 128-cluster query, runs every
strategy, and prints requests, wasted bytes, predicted latency, and
cost per million queries.

`paper_table4` uses deterministic synthetic requests with the paper's
`(nprobe, K)` shapes. It is a local spot check, not a reproduction of
the paper's workload or hardware.

## Running

```bash
# Short public-API smoke — seconds after the benchmark binary is built.
make bench-smoke

# Full suite — typically 15-25 minutes, archives CSV + Criterion reports.
bash scripts/bench.sh

# One bench at a time:
(cd rust && cargo bench --bench ivf_workload)         # P × dist × strategy table + CSV
(cd rust && cargo bench --bench plan_quality)         # quality tables across distributions
(cd rust && cargo bench --bench plan_time)            # Criterion plan-time across strategies
(cd rust && cargo bench --bench main_solution_speed)  # n-scaling + P-scaling for MainSolution
(cd rust && cargo bench --bench planner_smoke)         # short checked-API smoke

# Single cell, short timing (skip analysis):
(cd rust && cargo bench --bench ivf_workload -- --profile-time 1)
```

Criterion writes HTML reports into `target/criterion/`.

## Output artifacts

`scripts/bench.sh` archives:

```
target/bench-artifacts/
  ivf_workload.csv              -- P × dist × strategy table (~100 rows)
  ivf_workload.stderr.log       -- human-readable quality table
  plan_quality.stderr.log
  plan_time.stderr.log
  main_solution_speed.stderr.log
  planner_smoke.stderr.log
```

Criterion's HTML reports are written separately under `target/criterion/`.

CSV schema:

```
dist, strategy, P, n_queries,
avg_serial_work_ms, avg_largest_request_ms, avg_predicted_latency_ms,
latency_reduction_pct_vs_separate,
avg_waste_pct, avg_requests,
avg_cost_usd_per_query, cost_reduction_pct_vs_separate,
usd_per_million_queries_saved_vs_separate,
plan_us_p50, plan_us_p99
```

### Comparison columns

`avg_cost_usd_per_query` uses the `CostModel::cost_per_*` fields. For
same-region S3 the byte term is ignored and the total collapses to
`cost_per_request × request_count` — so the headline number tracks how
well each planner reduces GET fan-out.

Default pricing matches AWS S3 us-east-1 (Aug 2025): `$0.0004 per 1 000
GetObject` → `$4e-7` per request. Swap in your own numbers by editing
`paper_cost` in the bench file or by passing a different `CostModel`.

- **`latency_reduction_pct_vs_separate`**: positive means lower
  predicted latency; negative means slower than `Separate`.
- **`cost_reduction_pct_vs_separate`**: positive means cheaper than
  `Separate`.
- **`usd_per_million_queries_saved_vs_separate`**: dollars saved per
  million queries at the listed parallelism; easier to translate to a
  deployment budget than raw per-query numbers.

## Correctness invariants

Every per-distribution run passes a gate that asserts:

- `Serial.W ≤ Greedy.W` — Serial is the W-optimal DP.
- `MainSolution(P=1).W ≤ Serial.W` — at `P = 1` the objective reduces
  to minimising W over a superset of Serial's feasible region.
- Every plan satisfies `check_invariants` (coverage, range validity,
  ordering, and assignment completeness). Constraint counts are checked
  separately where a benchmark supplies a request cap.

The quality table also makes these expected patterns easy to inspect:

- `Bottleneck` degenerates to `Separate` when there is no request cap.
- `Separate` has zero latency and cost savings by definition.
- `MainSolution` behaves more like `Serial` at low concurrency and
  more like `Bottleneck` at high concurrency.

A strategy violating any asserted invariant is either buggy or hit an
input distribution outside its design envelope; the bench aborts with
the offending fixture tagged.

## Regenerating the FAISS-IVF fixtures

The committed fixtures under `rust/tests/data/` are sufficient for CI
and for reproducing the headline numbers. Normal users do not need to
download SIFT1M. Run `make bench` against the committed ranges.

Fixture regeneration is a maintainer task. With converted SIFT1M
`base.npy` and `queries.npy` arrays available locally, run:

```bash
SIFT1M_DIR=/path/to/sift1m bash scripts/prepare-fixtures.sh
make bless
```

Dataset download and conversion belong in the paper experiment repository,
not this planner package. The generator validates the canonical array hashes
before replacing fixtures. Omit `SIFT1M_DIR` to refresh only synthetic data.
