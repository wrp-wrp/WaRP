# WaRP Experiments

Reproduction artifact for the paper *WaRP: Workload-Aware Request Planning for
Vector Retrieval on Object Storage*. The repo holds the runner
code, render scripts, and a frozen copy of the data that underlies
every experimental figure and table. A fresh source checkout or downloaded archive can regenerate every current
Paper Artifact locally from checksummed Frozen Evidence. Heavy measurement
reruns remain experiment-specific internally, but every measurement-backed
Paper Experiment is reachable from the same registry-driven command.

Artifact ownership is defined by `experiments/PAPER_ARTIFACT_MANIFEST.json`; execution
commands are defined by `artifact/registry.py`. The catalog check verifies that their
figure- and table-role entries agree exactly.

### Companion repositories

- [`vecio-planner`](https://anonymous.4open.science/r/vecio-planner-anon-2523) — the
  paper's algorithm distilled into a standalone Rust crate + Python
  bindings. The Rust crate is `warp-planner`; the Python distribution is
  `vecio-planner`. Index-agnostic; use it when
  you want the planner without Lance as a dependency.
- [`lance-s3-io-optimization`](https://anonymous.4open.science/r/lance-s3-io-optimization-anon-95B5) —
  Lance fork with the planner integrated into its I/O scheduler;
  this repo's full-mode pipeline depends on its example binary.

## Project structure

```text
WaRP paper
|
+-- warp-experiments                  this repository
|   |
|   +-- artifact/                     registry, CLI, integrity checks
|   +-- runners/                      measurement implementations
|   +-- src/                          shared experiment computations
|   +-- experiments/figure-NN-short-name/
|   +-- experiments/table-NN-short-name/
|   |   +-- data/paper/               frozen, checksummed paper evidence
|   |   +-- data/latest/              fresh runner output (gitignored)
|   |   +-- render*.py                paper-artifact renderer
|   |   +-- *.pdf / *.tex             authoritative paper artifact
|   +-- scripts/                      setup and one-command orchestration
|   +-- docs/                         reproduction and artifact maps
|
+-- warp-planner                      standalone public planner library
|
+-- lance-s3-io-optimization          Lance integration and benchmark binaries

Execution flow inside `warp-experiments`

scripts/run_all.sh
|
+-- reads artifact/registry.py to select experiments
    |
    +-- light: re-render committed results (no measurements)
    |   |
    |   +-- data/paper/         committed measurements
    |       |
    |       +-- render*.py      generate figures and tables
    |           |
    |           +-- _regen/     regenerated artifacts
    |               |
    |               +-- compare with committed PDF and TeX files
    |
    +-- full: run measurements again
        |
        +-- prepare datasets and binaries
            |
            +-- run measurement
                |
                +-- data/latest/   new measurements
                    |
                    +-- render*.py
                        |
                        +-- _latest/  figures and tables from the new run
```

Use `light` to verify that the committed paper data still produces the paper
artifacts. Use `full` only when you want to collect new measurements and render
new artifacts from them.

The three repositories have separate ownership boundaries: this repository
owns paper measurements and artifacts, `warp-planner` owns the reusable
planner interface, and the Lance fork owns engine integration. This repository
does not vendor or patch either companion repository.

## Reproducing the paper

Three ways in, depending on how much you want to exercise. Light mode
re-renders every experimental figure and table from the committed data and only needs
Python. A single-experiment run exercises the real Lance + S3 pipeline
end-to-end on one spec of your choice — useful as a cheap sanity
check before a heavyweight run. The full orchestrator runs the complete
dependency closure and can take **several days**. This includes the complete
Figure 7, Figure 8, and Figure 11 measurement protocols; their committed
snapshots remain available for fast validation.

### Light (~1 minute)

Open the [vecio-experiments anonymous repository](https://anonymous.4open.science/r/vecio-experiments-anon-E7F8),
click **Download Repository**, extract the ZIP, and run the commands below from
the extracted repository root.

Then run:

```
python3 -m pip install --break-system-packages \
    -r requirements-light.txt

python3 -m artifact check
bash scripts/run_all.sh --profile light
python3 -m artifact compare
```

(`--break-system-packages` is needed on Ubuntu 24.04+ / PEP 668
externally-managed Python; drop it if you are in a venv or on an
older distro. If you prefer a venv, `python3 -m venv .venv && source
.venv/bin/activate` before the pip command.)

The first command runs the offline checks (hashes, schemas, artifact ownership,
CLI flag consistency, check-only runner plans, etc.) and finishes with an
`[ok] completed ... smoke checks` summary. The second re-renders every Paper Experiment into
`experiments/<id>/_regen/`, leaving committed artifacts untouched.
The third compares each regenerated PDF and TeX table against its committed
counterpart. Figures should come within 10% pixel diff, while generated tables
must match exactly. Anything larger is usually matplotlib/font drift rather
than data drift. Separately, `python3 -m artifact catalog check`
verifies ownership and enforces that `experiments/` contains only current
Paper Experiments.

### One experiment end-to-end (minutes, EC2)

Before kicking off the multi-day full run, it's worth running a
single experiment through the real Lance + S3 pipeline to confirm
the toolchain works. Set up the environment (same as full mode), then:

```
bash scripts/setup_ec2.sh
cp scripts/run_all.env.example scripts/run_all.env   # set VECIO_S3_BUCKET + REGION

# cheapest real-Lance-and-S3 spec (~4 minutes):
bash scripts/run_all.sh --only real-s3-tiny-request-fanout-sanity

# pure CPU, no S3 (seconds):
bash scripts/run_all.sh --only reorder-cost
```

`--only` takes a comma-separated list of spec ids from
`python3 -m artifact list`. Setup is selection-aware: a CPU-only spec does not
build Lance or prepare unrelated datasets; a Figure 7/8/11 selection prepares
only its declared public inputs and companion tools.

The paper parameters are already fixed by the registry and campaign modules;
readers provide infrastructure, not scientific tuning choices. Inspect the
resolved contract before a run with:

```bash
python3 -m artifact show figure-07-real-lance-metrics --parameters
```

The JSON output separates `paper_parameters` from `reader_inputs` such as a
writable S3 bucket and companion checkout. Passing different runner flags is
supported for exploratory work, but such a run is not a paper reproduction.

### Full (~5 days, EC2)

```
bash scripts/setup_ec2.sh

cp scripts/run_all.env.example scripts/run_all.env
# edit: set VECIO_S3_BUCKET and VECIO_S3_REGION to a bucket you can write to.
# make AWS credentials available through the standard AWS provider chain.
# check out the paper Lance companion separately and set VECIO_LANCE_ROOT.

bash scripts/run_all.sh --plan
bash scripts/run_all.sh
```

Anonymous-review downloads do not contain `.git`. That path is supported: set
`VECIO_LANCE_ROOT` to the extracted Lance companion, and the orchestrator
verifies the complete build-source fingerprint against the pinned paper commit.
A normal Git checkout is still required to be exactly at that commit and clean.

`setup_ec2.sh` installs the system, Python, and Rust toolchain prerequisites.
`run_all.sh` builds the companion examples, downloads/converts missing public
benchmark arrays, creates or validates the declared S3 Lance objects,
preflights the runner contracts, executes the dependency-ordered registry, and
renders fresh results. It never clones, checks out, or modifies the companion
repository. A completed managed preparation is validated and reused on rerun;
an interrupted partial pair fails closed and must be inspected and removed at
its two declared object paths, or retried under a new empty S3 root. Preparation
can itself take hours; use `--no-prepare` only after a successful preparation pass
has left the validated local build manifests in `data/latest/dataset-builds/`.
The heavyweight campaigns require the companion source at public paper release
`c768d004506231f4f5d81f61b9cbc55ccfc904f3`. Figure 11 additionally verifies
its two required CLI capabilities before Cargo runs, so a mismatched checkout
or stale binary fails during preflight rather than running a different protocol.
Runner outputs land in the owning `data/latest/` slot and fresh artifacts in
`_latest/`; committed `data/paper/` snapshots and authoritative artifacts are
not touched.

Most S3-generating runners use `VECIO_S3_BUCKET` and `VECIO_S3_REGION`.
Figures 7/8/11 create their matched Lance datasets under
`VECIO_REAL_LANCE_S3_ROOT` (default:
`s3://$VECIO_S3_BUCKET/real-lance-format-sanity`). Figure 4 requires four topology definitions; see
`runners/figure-04-object-store-io/README.md`.

Committed paper evidence replaces the original private S3 bucket with the
non-routable `s3://paper-artifact/...` label. This is a privacy redaction, not
an endpoint readers should access. Numerical measurements, object names,
physical-index evidence, and checksums remain unchanged; fresh runs record the
reader-owned endpoint under `data/latest/`.

## Repository layout

```
experiments/figure-NN-name/ figure-producing Paper Experiments
experiments/table-NN-name/  table-producing Paper Experiments
  stages/<stage-id>/ supporting measurement/analysis owned by that experiment
  render*.py          experiment-owned paper artifact renderer
  *.pdf *.tex         committed authoritative paper artifacts
  data/paper/         committed frozen input data (SHA-256 hashed, see DATA_MANIFEST.json)
  data/latest/        gitignored; runner output slot
  _regen/             gitignored; light-mode render output
  _latest/            gitignored; fresh-data render output
runners/<id>/       measurement code; one-to-one owners use the numbered id
src/                reusable computation (datasets, IVF/reordering, S3 requests, plotting)
artifact/           one registry and CLI for prepare/run/render/rebuild/compare/check
scripts/            setup_ec2.sh, run_all.sh
third_party/lance/  optional paper Lance companion checkout
docs/               reproduction guide, generated artifact map, schema metadata
```

Every top-level directory under `experiments/` is named after the paper artifact
it owns, for example `figure-07-real-lance-metrics` or
`table-01-cluster-coarsening`. Figures 2, 3, and 6 are explanatory diagrams, so
they have no empty experiment directory. A single owner may produce a figure,
a table, or both from the same frozen evidence.
`experiments/PAPER_ARTIFACT_MANIFEST.json` also classifies the paper's
non-experimental diagrams and documentation tables.

`runners/` never owns figures or tables. It produces candidate measurements
under the corresponding Paper Experiment's `data/latest/`; `experiments/`
owns rendering, frozen evidence, and committed paper artifacts.

## Further reading

If reproduction breaks or you want more detail than this README
covers, start with [`docs/REPRODUCTION_GUIDE.md`](docs/REPRODUCTION_GUIDE.md).
The generated [`docs/PAPER_ARTIFACTS_MAP.md`](docs/PAPER_ARTIFACTS_MAP.md)
maps each paper label to its Paper Experiment and Paper Artifact, when one exists.

To inspect and execute a single experiment, use
`python3 -m artifact show <id>` and `python3 -m artifact show <id> --parameters`,
then `prepare`, `run`, and `render` for that
same id. The thin `experiments/<id>/run.sh` and `render.sh light` wrappers are
also available. To list every spec, `python3 -m artifact list`.

## Citation

The paper is under revision. Until its final PVLDB metadata is available, use:

```bibtex
@misc{wan2026warp,
  author = {Ruipeng Wan and Xinyu Zeng and Yiming Qiao and Huanchen Zhang},
  title  = {{WaRP}: Workload-Aware Request Planning for Vector Retrieval on Object Storage},
  year   = {2026},
  note   = {Manuscript under revision}
}
```

This entry will be replaced by the official PVLDB citation after publication.
