# Evaluation: SIFT1M real-S3 sensitivity

Measures sensitivity to probe count and cluster granularity on SIFT1M with real S3 reads.

Spec: `figure-14-sift1m-sensitivity`. Paper labels: fig:eval-sensitivity-sift1m.

## Directory layout

| Path | Contents |
| --- | --- |
| `data/paper/` | Frozen measurements |
| `data/latest/` | New measurements |
| `_regen/` | Outputs rendered from frozen measurements |
| `_latest/` | Outputs rendered from new measurements |

Committed figures and tables are the paper outputs. Fresh renders
go to the directories above.

## What this experiment produces

- `experiments/figure-14-sift1m-sensitivity/figure_sift1m_real_s3_sensitivity_combined.pdf`

## Backend requirements

- Requires a built Lance example binary (see `scripts/setup_ec2.sh`).
- Requires live S3 access via AWS credentials + bucket.

## Running

Run from this directory:

```bash
# Collect measurements and render the result
./run.sh
./render.sh

# Render frozen measurements
./render.sh light
```

The wrappers call the commands in `artifact/registry.py`.
