# Examples

Read these in order if you're new — each builds on the previous.

| File | What it shows | Runtime |
|---|---|---|
| [`quickstart.rs`](quickstart.rs) | The smallest possible call: build `IoRequest`s, define a `CostModel`, plan, evaluate. ~50 lines. | < 0.1 s |
| [`calibrate.rs`](calibrate.rs) | Fit a `CostModel` from your own `(bytes, latency_ms)` samples, then plan with it. ~80 lines. | < 0.1 s |
| [`probe_and_plan.rs`](probe_and_plan.rs) | End-to-end: probe an `object_store` backend → fit → plan. Uses the in-memory backend so it runs offline. **Requires the `probe` feature.** | ~1 s |
| [`savings_demo.rs`](savings_demo.rs) | Side-by-side comparison of every strategy on a realistic IVF-style query: requests, waste %, latency, $/query, $/M queries. Useful for the README screenshot or sanity-checking after a change. | < 1 s |
| [`paper_table4.rs`](paper_table4.rs) | Current-host synthetic overhead spot check for the paper Table 4 `(nprobe, K)` shapes. Published values are context, not a reproduction. | seconds |

Run any of them:

```bash
cargo run --release --example quickstart
cargo run --release --example calibrate
cargo run --release --example probe_and_plan --features probe
cargo run --release --example savings_demo
cargo run --release --example paper_table4
```

For repeatable Criterion measurements, run `make bench-smoke` (short,
checked public API) or `make bench` (full scaling and quality suites).

## Maintainer tool (not a usage example)

| File | What it does |
|---|---|
| [`bless_golden.rs`](bless_golden.rs) | Re-runs every strategy on every committed FAISS-IVF fixture and freezes the outputs into `tests/data/<dist>/expected_plans.json`. Run after an intentional algorithm change, then review the diff before committing. The companion regression test lives in `tests/golden.rs`. |

```bash
make bless                     # convenience wrapper
# or directly:
cargo run --release --example bless_golden
```
