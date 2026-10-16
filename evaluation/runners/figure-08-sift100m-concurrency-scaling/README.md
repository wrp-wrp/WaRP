# Figure 8 real-Lance concurrency runner

This runner executes both paper protocols. Panels (b,c) use a 54-cell
query-concurrency sweep with 10,000 measured queries per cell. Panel (a) uses
a 30-cell I/O-limit sweep with 1,000 measured queries per cell. Both use
`nprobe=256`, `refine=8`, 200 reused warm-up queries, and three
counterbalanced repeats. The base I/O ladder varies the file limit while
holding the process limit at 2048; the two extensions vary both limits jointly.

Fresh summaries are written under
`experiments/figure-08-sift100m-concurrency-scaling/data/latest/`.

It uses the same SIFT100M matched IVF-PQ pair as Figure 7. The preparation
phase creates or validates that pair and its public arrays:

```bash
export VECIO_LANCE_ROOT=/path/to/WaRP/lance
export VECIO_REAL_LANCE_S3_ROOT=s3://your-bucket/real-lance-format-sanity
python3 runners/shared/verify_lance_base.py --lance-root "$VECIO_LANCE_ROOT"
cargo build --manifest-path "$VECIO_LANCE_ROOT/Cargo.toml" --release \
  -p lance --example full_pipeline_real_s3 \
  --target-dir "$VECIO_LANCE_ROOT/target"
python3 runners/shared/lance_build_receipt.py \
  --lance-root "$VECIO_LANCE_ROOT" \
  --binary "$VECIO_LANCE_ROOT/target/release/examples/full_pipeline_real_s3" \
  --package lance --example full_pipeline_real_s3
bash scripts/setup_lance_python.sh
python3 -m artifact prepare figure-08-sift100m-concurrency-scaling
python3 -m artifact run figure-08-sift100m-concurrency-scaling -- --check-only
python3 -m artifact run figure-08-sift100m-concurrency-scaling
```

The runner uses the SIFT100M build manifest produced by Figure 7 preparation.
It also requires the build receipt above, which binds the executable hash to
the pinned, tracked-clean companion commit and `Cargo.lock`.
Before measuring or reusing a cell, it verifies the live dataset version,
physical index UUID, and PQ geometry against that manifest. A custom managed
object therefore needs both `--s3-root` and its matching
`--index-manifest-root`; the URI alone is not a reproducible input identity.
