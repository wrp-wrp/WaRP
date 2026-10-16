# Figure 11 greedy-policy transfer runner

This is a two-stage experiment, not a plot-only script:

1. SIFT builds the SIFT1M IVF-PQ payloads, selects payload-matched tight/loose
   query populations without observing performance, tunes Greedy on only the
   tight/nprobe=32 calibration cell, freezes the policy, and measures the 2x2
   shift matrix against fixed WaRP P=32/K=128.
2. MSong selects disjoint calibration/evaluation queries, extracts exact
   Lance payload ranges, tunes only the MSong Greedy control on calibration,
   then compares SIFT-frozen Greedy, MSong-tuned Greedy, and WaRP on the held-out
   evaluation queries.

The corresponding Lance version must be built from the paper companion
branch and provide `strict_budget_real_s3 --only-method` plus
`full_pipeline_real_s3 --out-query-ranges-json`. These capabilities belong in
the Lance companion branch, not as unapplied patches in this experiment
repository. `--check-only` reports both
compiled capabilities, and a real run fails before touching S3 if either is
absent. The pinned public companion release includes both flags.

```bash
export VECIO_LANCE_ROOT=/path/to/lance-s3-io-optimization
export VECIO_S3_BUCKET=your-bucket
export VECIO_REAL_LANCE_S3_ROOT=s3://your-bucket/real-lance-format-sanity
bash scripts/setup_lance_python.sh
python3 -m artifact prepare figure-11-greedy-locality-shift
python3 -m artifact run figure-11-greedy-locality-shift -- --check-only
python3 -m artifact run figure-11-greedy-locality-shift
```

Preparation downloads/converts the public SIFT1M and MSong arrays, builds and
checks the two companion examples, and creates or validates the historical
MSong `nlist=4096, M=60` pair. The transfer stage uses
`msong_pq_reorder.lance`; Figure 7's later `M=420` MSong object is a different
experiment and is never substituted. Set `VECIO_FIG11_MSONG_URI` only to reuse
an explicitly managed compatible object, and pair it with
`VECIO_FIG11_MSONG_INDEX_IDENTITY`. The identity must be either the managed
build manifest or a frozen inventory that pins the physical index UUID; seeing
an arbitrary single IVF-PQ UUID is not accepted. A frozen inventory also
declares the dataset URI, payload-object ETag, and byte size. Preflight performs
a live S3 HEAD and requires all three to remain unchanged.

The newly built object follows the production geometry and layout algorithm,
but its UUID/ETag and fresh measurements are explicitly labeled
`reconstructed-replication`, not the exact MSong paper object. The committed
`data/paper/` summaries remain the byte-identified paper evidence.

Fresh paper inputs are `sift_summary.json` and `msong_summary.json` under the
owning experiment's `data/latest/` directory.
