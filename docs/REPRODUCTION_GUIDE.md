# Reproduction guide

This consolidated release includes the exact paper Lance source at
`third_party/lance`; it is discovered automatically. The standalone library
is included at `third_party/warp-planner`. Separate companion checkouts and
anonymous mirrors are unnecessary for these workflows. The original source
fingerprint gate remains active for the bundled Lance source.

This document covers the commands and evidence boundaries that are not already
obvious from the root `README.md`. Paper-artifact ownership is generated separately in
[`PAPER_ARTIFACTS_MAP.md`](PAPER_ARTIFACTS_MAP.md).

## What can be reproduced

The repository has three distinct workflows:

| Workflow | Input | Output | External services |
|---|---|---|---|
| Light validation | committed `data/paper/` | regenerated artifacts under `_regen/` | none |
| Single-spec rerun | runner-specific datasets and binaries | fresh `data/latest/` | depends on the spec |
| Full orchestrator | supported local and live-S3 runners | fresh `data/latest/` plus regenerated artifacts | Lance, datasets, AWS/S3 |

Light validation covers every current Paper Experiment. Full mode reaches every
measurement-backed Paper Experiment through `artifact/registry.py`, including
Figures 7, 8, and 11. The figure-05-planning-frontier figure is intentionally
render-only because it is a deterministic explanatory construction rather than
a measured system result.

## Light validation

Install the Python render dependencies. A virtual environment is recommended:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements-light.txt
```

Then run:

```bash
python3 -m artifact check
bash scripts/run_all.sh --profile light
python3 -m artifact compare
```

The smoke test performs offline checks including frozen-data hashes,
schema fingerprints, runner CLI compatibility, artifact ownership, shell syntax,
and runner/render path consistency. Light rendering writes only to each Paper
Experiment's `_regen/` directory; authoritative paper artifacts are restored
byte-for-byte after rendering.

`python3 -m artifact compare` requires `pdftoppm` from Poppler. Its default 10%
pixel threshold tolerates host font/rasterizer drift for PDFs; generated TeX
tables must match exactly. It is not a replacement for the data-integrity and
schema checks.

## Find and run one spec

The execution registry is `artifact/registry.py`. Query it through the module interface instead
of a duplicated static document:

```bash
python3 -m artifact list
python3 -m artifact show <spec-id>
python3 -m artifact show <spec-id> --parameters
python3 -m artifact prepare <spec-id> --dry-run
python3 -m artifact run <spec-id> --dry-run
python3 -m artifact render <spec-id> --dry-run
```

Paper Experiment directories also provide thin
`run.sh` / `render.sh` wrappers. Supporting stages stay under their owning
Paper Experiment. The repository contains only code needed by current paper
artifacts.

The ownership seam is strict: `runners/` contains measurement implementations
but no generated figures or tables. Runtime output belongs in
`experiments/**/data/latest/`; renderers and paper artifacts belong in the
owning `experiments/` directory.

`show --parameters` is the machine-readable reproduction contract. Its
`paper_parameters` are resolved by this repository and should not be supplied
or tuned by the reader. `reader_inputs` are the environment-specific resources
the reader must provide, such as a writable bucket, region, topology hosts, or
the companion checkout. Runner flags after `--` are useful for exploratory
experiments, but changing a paper parameter changes the experiment and is not
reported as paper reproduction.

For a real Lance/S3 sanity run on EC2:

```bash
bash scripts/setup_ec2.sh
cp scripts/run_all.env.example scripts/run_all.env
# Edit VECIO_S3_BUCKET and VECIO_S3_REGION.

bash scripts/run_all.sh --only real-s3-tiny-request-fanout-sanity
```

Use `--only` with comma-separated spec ids to narrow execution. The script
prepares and builds only the requirements declared by that dependency closure.
It uses the explicitly supplied companion checkout when needed, but never
clones, checks out, or changes that repository.

## Runtime configuration

There is no repository-global `configs/` directory. Runtime endpoints belong
to the run environment; experiment parameters belong to the experiment.

| Source | Purpose |
|---|---|
| `scripts/run_all.env` or `.vecio-run.env` | private local values loaded by `run_all.sh` |
| `VECIO_S3_BUCKET`, `VECIO_S3_REGION` | writable live-S3 endpoint inherited by runners and preparation tools |
| `VECIO_REAL_LANCE_S3_ROOT` | optional Figures 7/8/11 dataset prefix; defaults below the selected bucket |
| `VECIO_LANCE_ROOT` | existing checkout of the paper Lance companion; never mutated by this repository |
| `VECIO_FIG11_MSONG_URI` | optional override for an already prepared Figure 11 MSong reordered index |
| `VECIO_FIG11_MSONG_INDEX_IDENTITY` | build manifest or frozen inventory pinning that index's physical UUID |
| `--s3-bucket`, `--s3-region` | explicit endpoint for the standalone S3 calibration path |
| `--config <yaml>` | compatibility with archived S3 protocols carrying their own endpoint YAML |
| `experiments/table-01-cluster-coarsening/config/base.yaml` | pipeline template owned by that table experiment |

The standalone S3 probe resolves command-line flags first, then an explicitly
supplied YAML, then environment variables. No credentials or real bucket name
should be committed.

The paper reports AWS EC2 `c7gn.8xlarge` hosts with 32 vCPUs, 64 GiB memory,
100 Gbps networking, and same-region S3 in `ap-northeast-1`. This environment
is recorded in each relevant `show --parameters` contract. Readers may use
other infrastructure, but performance values should then be treated as a new
measurement rather than an exact hardware reproduction.

The heavyweight paper campaigns require the companion source at public paper
release `c768d004506231f4f5d81f61b9cbc55ccfc904f3`. A normal Git checkout must
have that exact clean `HEAD`. An Anonymous GitHub ZIP has no `.git`, so the
same gate recomputes a SHA-256 fingerprint over the complete Cargo/Rust/proto/Python
build-source closure and compares it with the value pinned in
`artifact/campaigns.py`. Neither path changes the supplied source directory.
Figure 11 also verifies its two required compiled CLI capabilities before any
measurement begins.

## Full orchestrator

On an EC2 host with AWS credentials:

```bash
bash scripts/setup_ec2.sh
cp scripts/run_all.env.example scripts/run_all.env
# Edit the bucket and region in scripts/run_all.env.
bash scripts/run_all.sh
```

The script resolves the paper Lance companion from `VECIO_LANCE_ROOT`, a
sibling checkout, or `third_party/lance`, builds the required examples,
prepares the selected public arrays and S3 objects, preflights every selected
runner, runs the dependency-ordered registry, then renders from
`data/latest/`. Preparation never overwrites an existing partial, untracked,
or mismatched pair. An interrupted pair must be inspected and removed only at
its two printed target URIs, or retried under a new empty S3 root. Use
`--no-prepare` only to reuse an earlier successful preparation whose local
`dataset-builds/*/manifest.json` files are still present. Externally managed
Figure 7/8 objects require equivalent identity manifests supplied directly to
the runner with `--index-manifest-root`; a URI by itself is not sufficient.
The default workload is expensive, downloads large public corpora,
writes large S3 indexes, incurs cloud charges, and can take several days. Use
`--only <spec-id> --plan` and then a single-spec run first.

Local execution is the global default. A runner that genuinely needs multiple
network locations, such as Figure 4, owns its own explicit SSH/topology flags;
there is no misleading repository-global SSH switch.

The heavyweight campaign contracts are:

| Paper experiment | Measurement protocol | Fresh output |
|---|---|---|
| `figure-07-real-lance-metrics` | 63 performance + 21 recall cells; 2 census + 18 cache cells | Figure 7 + payload table inputs |
| `figure-08-sift100m-concurrency-scaling` | 54 query-concurrency cells; 30 I/O-limit cells | Figure 8 query/I/O summaries |
| `figure-11-greedy-locality-shift` | SIFT shift calibration/evaluation; MSong held-out transfer | Figure 11 transfer summaries |

The default public Figure 11 preparation reconstructs the historical MSong
geometry from the public arrays and labels the result
`reconstructed-replication`. It does not claim the new UUID/ETag is the exact
private paper object. Reusing an external object requires both
`VECIO_FIG11_MSONG_URI` and an identity document via
`VECIO_FIG11_MSONG_INDEX_IDENTITY`; the trace must resolve to that pinned UUID.

Figure 11 requires a companion Lance build exposing
`strict_budget_real_s3 --only-method` and
`full_pipeline_real_s3 --out-query-ranges-json`. Its preflight reports those
compiled capabilities and a real run fails before S3 access if they are absent.
The experiment repository deliberately does not carry or auto-apply source
patches that belong to the corresponding Lance branch.

## Preparation contracts for Figures 7, 8, and 11

The heavyweight figures have an explicit three-stage interface:

```bash
python3 -m artifact show figure-07-real-lance-metrics
bash scripts/setup_lance_python.sh
python3 -m artifact prepare figure-07-real-lance-metrics
python3 -m artifact run figure-07-real-lance-metrics
python3 -m artifact render figure-07-real-lance-metrics
```

`run_all.sh` performs the Lance Python setup automatically for selected specs;
the explicit setup line is only needed when invoking `artifact prepare`
directly. The setup script verifies the companion Git ancestry before it
installs or compiles any Python package.

Figures 7/8 prepare seven public array sets and matched Random/Reordered
IVF-PQ pairs. Each pair shares one trained centroid set and one PQ codebook;
only centroid order differs. Figure 11 separately prepares the historical
MSong `nlist=4096, M=60` pair named `msong_pq_random.lance` and
`msong_pq_reorder.lance`. It must not be substituted with Figure 7's
`M=420` MSong object.

A freshly built index follows the frozen production geometry and layout
algorithm but is not claimed to have the paper object's byte identity, UUID,
ETag, or performance measurements. The committed `data/paper/` evidence
remains the exact paper record. Fresh manifests under `data/latest/` record
the newly built array hashes, layout permutation, PQ codebook, S3 URIs,
dataset versions, and physical index UUIDs. Figure 7/8 preflight also reads
the live Lance index metadata and requires it to match those manifests before
any cached cell can be reused.

Figure 8's normalized I/O summaries retain the frozen source-harness hashes,
Lance commit and tracked-diff identity, index inventory, query and binary
hashes, and file/process concurrency pairs. Fresh runs build the pinned,
tracked-clean public companion release and preserve its build receipt; they
reproduce the protocol without claiming a byte-identical historical binary.
The internal campaign archives are therefore not part of the public
reproduction interface.

Public frozen evidence uses `s3://paper-artifact/...` in place of the original
private bucket. That label is deliberately non-routable. The redaction changes
only the endpoint string: committed numerical results, object-relative names,
physical-index evidence, and source hashes remain available for validation.
A fresh run writes the reader-owned endpoint only to the gitignored
`data/latest/` slot.

## Evidence slots

Each artifact owner separates frozen paper evidence from fresh output:

```text
experiments/figure-NN-name/   # or table-NN-name
  data/paper/    committed, checksummed input used by light rendering
  data/latest/   gitignored output from a fresh runner
  _regen/        gitignored output from light rendering
  _latest/       gitignored output rendered from fresh data/latest
```

A table-only experiment uses the same slots under `experiments/`; there is no
separate top-level `tables/` ownership model. A successful fresh run does not
automatically replace paper evidence.

Promoting new evidence is deliberately explicit:

1. Inspect `data/latest/` and the generated artifacts.
2. Copy only the validated run into the matching `data/paper/` slot.
3. Refresh integrity and schema metadata:

   ```bash
   python3 -m artifact freeze
   ```

4. Re-render with `python3 -m artifact rebuild --target paper` only when replacing the authoritative paper
   artifacts is intended.

Do not bulk-promote every `data/latest/` directory. Promotion is a paper update,
not a side effect of successful reproduction, and must remain reviewable per
experiment.

## Verification commands

```bash
# Full offline verification: ownership, hashes, schemas, CLI, syntax, and render.
python3 -m artifact check

# Compare regenerated PDFs and TeX tables with authoritative artifacts.
python3 -m artifact compare
```

## Source-of-truth map

| Question | Authority |
|---|---|
| Which directory owns a paper figure or experimental table? | `experiments/PAPER_ARTIFACT_MANIFEST.json` |
| Which command runs or renders a spec? | `artifact/registry.py` |
| Which parameters define the paper run? | `artifact/contracts.py` via `artifact show <id> --parameters` |
| Which frozen files are integrity-protected? | `experiments/DATA_MANIFEST.json` |
| Which CSV/JSON shapes are expected? | `docs/SCHEMA_FINGERPRINTS.json` |

`PAPER_ARTIFACTS_MAP.md` is generated from the artifact manifest and should not be
edited by hand.
