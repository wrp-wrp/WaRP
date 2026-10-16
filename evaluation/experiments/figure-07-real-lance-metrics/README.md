# Real-Lance end-to-end validation

**Paper artifacts:** Figure 7 and the payload-accounting table

**Paper labels:** `fig:eval-real-lance-metrics`, `tab:eval-payload-accounting`

Reproduces Figure 7 and the payload-accounting table from the performance
and cache campaigns. The concurrency sweep is in
`../figure-08-sift100m-concurrency-scaling/`.

## Reproduce

```bash
./run.sh                 # performance, recall, and cache campaigns
./render.sh              # fresh data/latest -> _latest
./render.sh light        # frozen data/paper -> _regen
```

The renderer validates the frozen source table and both campaign summaries by
SHA-256 before writing:

- `figure_real_lance_metrics.pdf` — the paper figure;
- `figure_real_lance_metrics.normalized.csv` — the displayed normalization;
- `figure_real_lance_metrics.provenance.json` — the rendering contract;
- `table_payload_accounting.tex` — the paper's bytes/bandwidth/cost table;
- `table_payload_accounting.csv` — the table's machine-readable values;
- PNG/SVG previews, which are ignored by Git.

The checked-in PDF is the committed paper artifact and is covered by the
repository artifact and frozen-data integrity checks.

## Frozen inputs

`data/paper/` contains the frozen inputs:

- `figure7_source.csv` — completed Figure 7 display table;
- `figure7_performance_summary.json` — 63-cell end-to-end campaign summary;
- `figure7_cache_summary.json` — 27-cell cache campaign summary;
- `aws_pricing.json` — frozen price inputs used by the payload-accounting table;
  `experiments/DATA_MANIFEST.json` covers every frozen input.

## Environment

Rendering is local and requires Python, NumPy, Matplotlib, and Poppler's
`pdftoppm`. `run.sh` executes the heavyweight Figure 7 runner; its seven query,
ground-truth, matched-S3-dataset, and NIC requirements are documented under
`runners/figure-07-real-lance-metrics/`.
