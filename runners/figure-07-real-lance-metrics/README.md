# Figure 7 real-Lance validation runner

This runner executes the complete Figure 7 paper protocol: 63 cache-off
performance cells, 21 exact Recall@10 exports, two cache-residency census
cells, and 18 cache-on cells. It records the original whole-process NIC
RX/TX boundary and produces fresh `figure7_performance_summary.json`,
`figure7_cache_summary.json`, `figure7_recall_summary.json`, and
`figure7_source.csv` under the experiment's `data/latest/` directory.

The declared preparation phase downloads/converts missing public arrays and
creates or validates the matched Random/Reordered IVF-PQ Lance datasets below
`VECIO_REAL_LANCE_S3_ROOT`. It trains one IVF model and one PQ codebook for
each pair, then changes only centroid order. The runner never falls back to
bundled `data/paper/` evidence. `--data-root` must contain
`<dataset>/queries.npy` and `<dataset>/gt_I.npy` for all seven workloads, plus
`gist1m/query_origin_10k.npy`.

```bash
export VECIO_LANCE_ROOT=/path/to/lance-s3-io-optimization
export VECIO_REAL_LANCE_S3_ROOT=s3://your-bucket/real-lance-format-sanity
bash scripts/setup_lance_python.sh
python3 -m artifact prepare figure-07-real-lance-metrics
python3 -m artifact run figure-07-real-lance-metrics -- --check-only
python3 -m artifact run figure-07-real-lance-metrics
```

The seven exact production object-name contracts are printed by
`python3 -m artifact prepare figure-07-real-lance-metrics --dry-run`. Existing
complete pairs previously built with their local manifest are validated and
reused; partial or untracked pairs are rejected. Measurement preflight hashes
the local manifest and reads the current remote dataset version, physical
index UUID, and PQ geometry. Existing cells are reusable only while those
identities still match.

`run_all.sh --no-prepare` is therefore only for reusing a prior successful
managed preparation with its `data/latest/dataset-builds/` manifests intact.
For externally managed objects, create equivalent complete identity manifests
and invoke the runner directly with `--index-manifest-root`; an S3 URI alone
is deliberately rejected.
