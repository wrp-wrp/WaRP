# Background: tiny-request fanout sanity

**Spec id:** `real-s3-tiny-request-fanout-sanity`
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
- `data/paper/summary.csv`

## Backend requirements

- Requires a built Lance example binary (see `scripts/setup_ec2.sh`).
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

Tiny-request fanout stress measurement used by the background microbenchmark figure.
