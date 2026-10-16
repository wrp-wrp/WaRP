# Real-Lance concurrency scaling

**Paper figure:** 8

**Paper label:** `fig:eval-real-lance-concurrency-scaling`

This Figure Experiment owns the three-panel SIFT100M concurrency result:
Lance latency versus the I/O concurrency limit, followed by throughput and
p95 latency versus query concurrency for Lance, WaRP, and WaRP + Reorder.
Panel (a) uses 1,000 measured queries per cell; panels (b,c) use 10,000.
All panels use `nprobe=256`, `refine=8`, and three counterbalanced repeats.

## Reproduce

```bash
./run.sh                 # query-concurrency + I/O-limit measurements -> data/latest
./render.sh              # fresh data/latest -> _latest
./render.sh light        # frozen data/paper -> _regen
```

The render validates every normalized frozen summary before writing
`figure_real_lance_concurrency_scaling.pdf` plus ignored PNG/SVG previews and a
provenance JSON file. The committed PDF is covered by the repository artifact
and frozen-data integrity checks.

## Frozen evidence

- `data/paper/io_summaries/` contains the three normalized I/O-limit summaries,
  including query, binary, Lance, index-inventory, and runtime contracts
  recovered from the original measurement records.
- `data/paper/query_summary.json` is the 54-cell query-concurrency
  campaign summary.
- `experiments/DATA_MANIFEST.json` anchors all inputs.

The frozen I/O summaries record the historical Lance commit, its tracked-diff
hash, and the measured binary hash. Fresh full runs instead build the pinned,
tracked-clean public Lance release and record its build receipt. The public
runner reproduces the measurement protocol; it does not claim that a fresh
compiler output is byte-identical to the historical binary.

Figure 7's cross-dataset and cache comparison is independently owned by
`../figure-07-real-lance-metrics/`; neither renderer validates or reads the other
figure's artifact.

## Environment

Rendering is local and requires Python, NumPy, Matplotlib, and Poppler's
`pdftoppm`. `run.sh` executes the heavyweight Figure 8 measurement protocols
documented under `runners/figure-08-sift100m-concurrency-scaling/`.
