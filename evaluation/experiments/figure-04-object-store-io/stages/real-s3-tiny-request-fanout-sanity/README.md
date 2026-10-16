# Background: tiny-request fanout sanity

Feeds the parent experiment.

Tiny-request fanout stress measurement used by the background microbenchmark figure.

Spec: `real-s3-tiny-request-fanout-sanity`. The parent experiment provides the paper figure.

## Directory layout

| Path | Contents |
| --- | --- |
| `data/paper/` | Frozen inputs |
| `data/latest/` | New stage results |

The parent experiment renders the paper figure from these stage results.

## What this experiment produces

- `data/paper/summary.csv`

## Backend requirements

- Requires a built Lance example binary (see `scripts/setup_ec2.sh`).
- Requires live S3 access via AWS credentials + bucket.

## Running

Run from this directory:

```bash
# Refresh this measurement stage:
./run.sh

# Render the parent figure from frozen measurements:
../../render.sh light
```

The wrappers call the commands in `artifact/registry.py`.
