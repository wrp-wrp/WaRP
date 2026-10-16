# [VLDB2027] WaRP: Workload-Aware Request Planning for Vector Retrieval on Object Storage

Authors: Ruipeng Wan, Xinyu Zeng, Yiming Qiao, and Huanchen Zhang.

This repository provides one delivery location for the three submitted WaRP
projects. Each component retains its submitted directory structure, source,
documentation, tests, and frozen measurements. Only this outer README,
source manifest, citation, license, and Git configuration files are added.

Repository: [wrp-wrp/WaRP](https://github.com/wrp-wrp/WaRP).
Final publication metadata will be added when assigned.

## Layout and fixed versions

```text
WaRP/
├── README.md
├── SOURCE_MANIFEST.json
├── CITATION.cff
├── LICENSE
├── warp-experiments/             original experiment/reproduction project
├── warp-planner/                 original standalone Rust/Python library
└── lance-s3-io-optimization/      original paper Lance fork
```

| Component | Submitted source commit | Files |
| --- | --- | --- |
| `warp-experiments` | `23b54eea35c60605e690ae3d2f63142545d60400` | 386 |
| `warp-planner` | `6e8c3fe04ff8e9b4ce4eb477e965d6ed315e41c3` | 196 |
| `lance-s3-io-optimization` | `c768d004506231f4f5d81f61b9cbc55ccfc904f3` | 1,419 |

`SOURCE_MANIFEST.json` records every original file's SHA-256, size, Git mode,
and source revision. The three directories contain all 2,001 original files
with no component-level adaptations. Their contents are included directly
in this Git repository; no submodules or companion downloads are required.

## Reproduce figures and tables offline

From the repository root:

```bash
cd warp-experiments
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-light.txt

# pdftoppm from Poppler must be on PATH.
python -m artifact check
python -m artifact compare
```

Install Poppler with `sudo apt-get install poppler-utils` on Ubuntu or
`brew install poppler` on macOS. For the original tests on macOS, set
`export TMPDIR=/private/tmp` before running the checks so temporary paths
are canonical.

The original pipeline verifies the 191 frozen measurement files and
regenerates paper artifacts under `_regen/`. It preserves committed
measurements and paper figures. Original vector datasets are separate from
these frozen measurements; their preparation code and download sources are
included in the experiment project.

Start with the [reproduction guide](warp-experiments/docs/REPRODUCTION_GUIDE.md)
and [paper figure/table map](warp-experiments/docs/PAPER_ARTIFACTS_MAP.md).
The component documentation retains its historical source links; use the
local component directories in this repository for the complete source.

## Use the standalone planner

The Rust crate is `warp-planner` (import `warp_planner`); the Python package
is `vecio-planner` (import `vecio_planner`). The standalone planner accepts
byte ranges, a cost model, and constraints and returns a merge plan. It can
be called without Lance, AWS credentials, or an S3 connection.

From the repository root, run the Rust example:

```bash
cargo run --locked --release --manifest-path warp-planner/Cargo.toml --example quickstart
```

To install and call the Python binding, use Python 3.9+ and Rust 1.82+:

```bash
python3 -m venv .venv
source .venv/bin/activate
CARGO_PROFILE_RELEASE_STRIP=none python -m pip install ./warp-planner/python
python warp-planner/python/examples/quickstart.py
```

Keeping native symbols avoids a Rust/macOS linker-stripping import issue on
recent macOS versions. No library source change is required.
The [original planner README](warp-planner/README.md) contains the Rust API,
Python API, examples, and the standalone implementation's guarantee scope.

The standalone Rust and Python interfaces share one core. The paper's engine
measurements use `VecioMergePlanner` inside the Lance fork, which is a
separate implementation. Both are preserved at their submitted versions.

## Run live measurements

The experiment project already discovers its sibling
`lance-s3-io-optimization` directory. Its pinned-source fingerprint gate
continues to check the original Lance source.

On the target Linux/EC2 environment, from the repository root:

```bash
cd warp-experiments
bash scripts/setup_ec2.sh
cp scripts/run_all.env.example scripts/run_all.env
# Set a reader-owned VECIO_S3_BUCKET and VECIO_S3_REGION.
# Supply credentials through the standard AWS provider chain.

# Inspect the smallest real-Lance/S3 selection without executing it:
bash scripts/run_all.sh --only real-s3-tiny-request-fanout-sanity --plan

# Collect new measurements; this writes S3 objects and incurs cloud charges:
bash scripts/run_all.sh --only real-s3-tiny-request-fanout-sanity
```

The complete paper campaign is available through the original registry and
`scripts/run_all.sh`. Inspect `bash scripts/run_all.sh --plan` before running
the multi-day full workflow. Figure 4 requires explicit network-topology
definitions. The paper's reported environment is EC2 `c7gn.8xlarge` and
same-region S3 in `ap-northeast-1`; other environments produce new performance
measurements. Live S3 measurements and the complete Lance build have not
been rerun for this reorganization.

See the [Lance artifact guide](lance-s3-io-optimization/ARTIFACT.md) for the
engine binaries and implementation map.

## Validation of this layout

Checked locally on 2026-10-16:

- All 2,001 component files and their Git modes match the recorded source commits.
- The original 17 offline smoke checks and 88 experiment-framework tests passed.
- The original 12 PDF and 3 TeX comparisons passed their submitted thresholds;
  the Figure 10 layout limitation is described below.
- All 121 standalone Rust tests/doctests and 25 Python tests passed; both
  minimal calling examples ran successfully from the new locations.
- The sibling Lance discovery and pinned build-source fingerprint passed.

The macOS checks used the canonical temporary-directory and native-symbol
settings documented above. This is local packaging validation, not a new
measurement of the paper's S3 performance.

## Known reproduction issues and preserved fixes

The submitted Figure 10 renderer generates a different aspect ratio from
the committed paper PDF. The original comparison resizes images and can
accept that layout drift within its pixel threshold. Frozen measurement
data are unchanged. A matching figure layout and an aspect-ratio check are
preserved in the earlier
[packaged-layout branch](https://github.com/wrp-wrp/WaRP/tree/archive/packaged-layout-2026-10-16).
Those changes are separate from the submitted snapshots on `main`.

That branch also retains the earlier macOS path-test fixes and additional
packaging checks. The canonical temporary-directory and native-symbol
settings above allow the original source to be used locally without
silently editing its internal files.

Each component retains its own LICENSE and NOTICE files. Experiment and
planner source are Apache-2.0, and the Lance fork retains its upstream
attribution. External datasets and dependencies retain their original
licenses. See `CITATION.cff` for the paper citation.
