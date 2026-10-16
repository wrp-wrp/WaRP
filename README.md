# WaRP: Workload-Aware Request Planning for Vector Retrieval on Object Storage

This repository contains the complete source release for the WaRP paper:
the paper experiments and frozen measurements, the Lance implementation used
by those experiments, and an independently callable Rust/Python planner.
One checkout or source archive contains all three components. No companion
repository checkout or Git submodule is required for the paper workflows.

Authors: Ruipeng Wan, Xinyu Zeng, Yiming Qiao, and Huanchen Zhang.
Publication metadata will be added after the final PVLDB decision.

Permanent source repository: [wrp-wrp/warp-artifact](https://github.com/wrp-wrp/warp-artifact).

## Choose a workflow

| Goal | Start here | Requirements |
| --- | --- | --- |
| Regenerate the paper figures and tables | [Offline reproduction](#offline-reproduction) | Python 3.10+, Poppler; no AWS account |
| Call the planner from another application | [Standalone planner](docs/STANDALONE_PLANNER.md) | Rust 1.82+; Python 3.9+ for the binding |
| Collect new measurements | [Live measurements](#live-measurements) | Linux, the bundled Lance toolchain, public datasets, reader-owned AWS resources |
| Locate a paper figure or table | [Paper artifact map](docs/PAPER_ARTIFACTS_MAP.md) | None |

## Source layout

```text
artifact/                     experiment registry, CLI, and integrity checks
experiments/                  frozen measurements, renderers, paper figures/tables
runners/                      measurement programs
src/                          shared experiment computations
scripts/                      environment setup and experiment entrypoints
docs/                         reproduction and standalone-library guides
third_party/lance/            fixed Lance source used by the paper experiments
third_party/warp-planner/      standalone Rust core and Python binding
SOURCE_MANIFEST.json          source revisions and original file SHA-256 hashes
PUBLIC_RELEASE_FILES.txt      complete source-release file list
```

The paper's engine experiments use `VecioMergePlanner` inside the bundled
Lance fork. The standalone library provides a separate checked API; its Rust
and Python interfaces share one core. Lance does not depend on that library
in this paper release. The implementations have different search paths, so
standalone examples are not a substitute for reproducing the paper's engine
measurements. See [implementation boundaries](docs/STANDALONE_PLANNER.md#implementation-boundaries).

## Offline reproduction

Run these commands from this repository's root. On Ubuntu, install Poppler
with `sudo apt-get install poppler-utils`; on macOS, use `brew install poppler`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-light.txt

python -m artifact.checks.source_integrity
python -m artifact check
python -m artifact compare
```

The check includes frozen-data hashes, schema and CLI checks, the experiment
catalog, existing unit tests, runner plans, and full offline rendering.
Regenerated figures/tables go under each experiment's `_regen/`
directory; committed paper artifacts are preserved. The comparison uses the
documented PDF pixel threshold and requires generated TeX tables to match
exactly. PDF aspect ratios must also agree within 2%; the comparison rejects
layout distortions before resizing. An offline data-to-figure check does not
recollect measurements.

The source-integrity check verifies both bundled source trees byte for byte
against the recorded snapshots and checks unmodified experiment files.
Packaging, test-portability, and Figure 10 layout adaptations are listed
explicitly in `SOURCE_MANIFEST.json`.

## Live measurements

The bundled Lance source is discovered automatically at `third_party/lance`.
The paper version is `c768d004506231f4f5d81f61b9cbc55ccfc904f3`; the
experiment toolchain checks its source fingerprint before building. Keep
that source unchanged for a paper reproduction.

The reported end-to-end environment is EC2 `c7gn.8xlarge` (32 vCPUs,
64 GiB RAM, 100 Gbps networking) and same-region S3 in `ap-northeast-1`.
Other hardware can validate functionality but produces new performance
measurements. Public input datasets are fetched/converted by the preparation
tools; frozen paper measurements are included under `data/paper/`.
Dataset licenses and upstream access conditions continue to apply.

On the target Linux/EC2 host:

```bash
bash scripts/setup_ec2.sh
cp scripts/run_all.env.example scripts/run_all.env
# Set your writable VECIO_S3_BUCKET and VECIO_S3_REGION in that local file.
# Supply AWS credentials through the standard provider chain.

# Inspect a selection without building, downloading data, or accessing S3:
bash scripts/run_all.sh --only real-s3-tiny-request-fanout-sanity --plan

# Small real-Lance/S3 measurement (creates objects and incurs cloud charges):
bash scripts/run_all.sh --only real-s3-tiny-request-fanout-sanity
```

For each experiment, inspect fixed paper parameters and infrastructure inputs:

```bash
python -m artifact list
python -m artifact show figure-07-real-lance-metrics --parameters
```

The full default run covers all registered measurements, prepares large public
datasets, writes S3 indexes, and can take several days. Inspect its plan first:

```bash
bash scripts/run_all.sh --plan
# Run only when the required hosts, buckets, and budget are ready:
bash scripts/run_all.sh
```

Full runner outputs go to `data/latest/`, and fresh figures/tables go to
`_latest/`. Frozen `data/paper/` inputs are preserved. Figure 4 requires
explicit definitions for its multiple network topologies. See the
[reproduction guide](docs/REPRODUCTION_GUIDE.md) for preparation, resource
requirements, and measurement boundaries.

## Data and experiment mapping

- `experiments/DATA_MANIFEST.json` pins 191 frozen measurement files by size
  and SHA-256. These are the inputs used to regenerate the paper artifacts.
- Original vector datasets are distinct from frozen measurements. Preparation
  code and sources are described in the reproduction guide and the
  per-experiment README files.
- `experiments/PAPER_ARTIFACT_MANIFEST.json` and the generated
  [artifact map](docs/PAPER_ARTIFACTS_MAP.md) connect paper numbers to their
  owning experiments, frozen inputs, renderers, and outputs.
- Figures 2, 3, and 6 are explanatory diagrams. Tables 2 and 3 document methods
  and datasets; they do not require a measurement runner.

## License and citation

The [local validation record](docs/VALIDATION.md) lists completed checks,
packaging adaptations, exact tool versions, and remaining live measurements.

Experiment code is Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE).
Both bundled components retain their own LICENSE and NOTICE files and all
upstream attribution. External datasets and downloaded dependencies retain
their original licenses.

Until final publication metadata is assigned, cite the manuscript:

```bibtex
@misc{wan2026warp,
  author = {Ruipeng Wan and Xinyu Zeng and Yiming Qiao and Huanchen Zhang},
  title = {{WaRP}: Workload-Aware Request Planning for Vector Retrieval on Object Storage},
  year = {2026},
  note = {Manuscript; final PVLDB metadata pending}
}
```
