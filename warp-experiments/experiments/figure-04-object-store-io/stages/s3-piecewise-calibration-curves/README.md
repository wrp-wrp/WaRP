# Background: S3 piecewise calibration curves

**Spec id:** `s3-piecewise-calibration-curves`
**Role:** stage of `figure-04-object-store-io`
**Paper labels:** (owned by the parent Figure Experiment)
**Depends on:** (none)

## Directory layout

This directory is a measurement stage owned by
`experiments/figure-04-object-store-io/`. It preserves frozen input
data and a fresh runner-output slot; the parent Figure Experiment owns the
paper renderer and PDFs.

| Path | What | State |
|---|---|---|
| `data/paper/` | authoritative frozen stage evidence | committed, SHA-256 hashed in `experiments/DATA_MANIFEST.json` |
| `data/latest/` | runner's fresh output data; written by `./run.sh` or `bash scripts/run_all.sh` (full mode) | gitignored |

To promote a fresh `data/latest/` run into the frozen stage slot, review the
result, copy `latest/` to `paper/`, then run `python3 -m artifact freeze`.

## What this experiment produces

Paper-facing outputs (`\includegraphics` targets):
- (none; owned by the parent Figure Experiment)

Frozen stage evidence:
- `data/paper/dense_probe.csv`
- `data/paper/source_latency_model.json`

## Backend requirements

- Requires live S3 access via AWS credentials + bucket.

## Running

```bash
# Refresh this measurement stage:
./run.sh

# Render the paper figure from the parent owner:
../../render.sh light
```

Both scripts delegate to `python3 -m artifact`; commands come from
`artifact/registry.py`.

## Summary

Piecewise calibration of S3 GET latency used by the background microbenchmark figure.
