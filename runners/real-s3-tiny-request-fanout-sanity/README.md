## Figure 4: tiny-request fanout stage

This stage measures serial 1 KiB S3 reads as the requested fanout grows from 1
to 128. Figure 4 uses its summary together with the separate latency-calibration
stage.

From the repository root, inspect the fixed paper parameters with:

```bash
python3 -m artifact show real-s3-tiny-request-fanout-sanity --parameters
```

Run the complete stage with:

```bash
bash scripts/run_all.sh --only real-s3-tiny-request-fanout-sanity
```

The command builds the two pinned Lance companion binaries, creates a 512 MB
object in the reader-owned S3 bucket when no URI is supplied, runs the 128-point
fanout sweep and the 8,192-query per-I/O sample, and writes fresh results under:

```text
experiments/figure-04-object-store-io/
└── stages/real-s3-tiny-request-fanout-sanity/data/latest/
    ├── summary.csv
    ├── fanout_001_per_iop.csv
    └── data_provenance.json
```

Set `VECIO_S3_BUCKET`, `VECIO_S3_REGION`, and `VECIO_LANCE_ROOT` in the private,
gitignored environment file described by `scripts/run_all.env.example`. Use
`VECIO_FIG4_TINY_REQUEST_URI` only when reusing an existing compatible object.
