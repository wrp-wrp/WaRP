"""Single source of truth for every paper artifact in this repository.

Each `ExperimentSpec` entry describes:
  * the paper label(s) and authoritative PDF/TEX artifact(s) it produces,
  * the artifact directory that owns its frozen `data/paper/` snapshot,
  * the `prepare` phase (datasets and companion binaries),
  * the `run` phase (heavy work; skipped in light mode), and
  * the `render` phase (produces the paper PDF / TEX for figure/table owners;
    measurement stages may have no render command).

The `python3 -m artifact` interface, render/compare implementation, and checks
read from this registry and never duplicate the spec data.

Profiles
--------

`full`  Executes every declared `run` then renders from fresh `data/latest/`
        evidence. Typically hours or days; needs the prepared datasets, EC2 +
        S3, and the paper Lance companion branch. The only intentional
        render-only paper figure is Figure 5, a deterministic planning-frontier
        explanation.

`light` Skips `run` entirely and executes `render` against the frozen data
        under the spec's artifact directory. ~5 minutes, no network, no Lance,
        no S3. Intended as a same-morning "does my plot pipeline still
        reproduce the paper" check.

The `$PY` token in command argvs is replaced with `sys.executable` at call
time by `Cmd.resolve()`.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final


ROOT: Final[Path] = Path(__file__).resolve().parents[1]
PY_TOKEN: Final[str] = "$PY"


@dataclass(frozen=True)
class Cmd:
    """One subprocess invocation. `$PY` is substituted with sys.executable."""

    argv: tuple[str, ...]

    def resolve(self) -> list[str]:
        return [sys.executable if part == PY_TOKEN else part for part in self.argv]


@dataclass(frozen=True)
class Phase:
    """Commands to run for one phase. An empty `commands` means no-op."""

    commands: tuple[Cmd, ...] = ()


@dataclass(frozen=True)
class ExperimentSpec:
    id: str                                    # kebab-case stable key
    title: str
    role: str                                  # figure | table | stage
    paper_labels: tuple[str, ...]              # TeX \label refs; empty for intermediates
    paper_artifacts: tuple[str, ...]            # authoritative PDF/TEX outputs owned by the spec
    out_dir: str                               # where render writes paper-facing outputs
    summary: str
    depends_on: tuple[str, ...] = ()           # other spec ids that must run before this one
    needs_lance: bool = False                  # run phase needs a built Lance example binary
    needs_lance_python: bool = False           # prepare phase builds Lance datasets
    needs_s3: bool = False                     # run phase needs live S3 access
    prepare: Phase = field(default_factory=Phase)
    run: Phase = field(default_factory=Phase)
    render: Phase = field(default_factory=Phase)
    expected_outputs: tuple[str, ...] = ()     # files relative to ROOT after render
    # Files that the run phase writes under <out_dir>/data/latest/
    # (relative paths, not absolute). Checked by the dry-run-full smoke
    # against what the render phase subsequently reads. Keep this list
    # tight: only entries that downstream specs or renders actually
    # consume.
    expected_runner_outputs: tuple[str, ...] = ()


def cmd(*argv: str) -> Cmd:
    return Cmd(argv=tuple(argv))


def phase(*commands: Cmd) -> Phase:
    return Phase(commands=tuple(commands))


def py(*argv: str) -> Cmd:
    return cmd(PY_TOKEN, *argv)


def public_datasets(*dataset_keys: str) -> Phase:
    """Prepare public arrays required by a fresh-clone full run."""
    return phase(
        *(
            py(
                "-m",
                "artifact",
                "prepare-dataset",
                "--dataset",
                dataset_key,
                "--download",
            )
            for dataset_key in dataset_keys
        )
    )


# ----------------------------------------------------------------------
# Experiment specs
#
# Order follows paper flow: intro -> background -> method -> evaluation
# -> sensitivity -> reorder analysis.
# ----------------------------------------------------------------------

EXPERIMENTS: Final[tuple[ExperimentSpec, ...]] = (
    ExperimentSpec(
        id="figure-01-s3-range-get-latency",
        title="Intro: S3 range-GET latency decomposition",
        role="figure",
        paper_labels=("fig:intro-latency-composition",),
        paper_artifacts=("experiments/figure-01-s3-range-get-latency/latency_composition.pdf",),
        out_dir="experiments/figure-01-s3-range-get-latency",
        summary="Latency decomposition microbenchmark illustrating per-phase contributions of an S3 range-GET.",
        run=phase(py("runners/figure-01-s3-range-get-latency/run_data.py")),
        render=phase(
            py("experiments/figure-01-s3-range-get-latency/render_intro_latency_composition.py")
        ),
        expected_outputs=(
            "experiments/figure-01-s3-range-get-latency/latency_composition.pdf",
        ),
        expected_runner_outputs=(
            "experiments/figure-01-s3-range-get-latency/data/latest/data.csv",
        ),
    ),
    ExperimentSpec(
        id="table-01-cluster-coarsening",
        title="Intro: fine vs coarse clustering recall-matched comparison",
        role="table",
        paper_labels=("tab:intro-coarse-cluster",),
        paper_artifacts=(
            "experiments/table-01-cluster-coarsening/intro_cluster_coarsening_sanity_check.tex",
        ),
        out_dir="experiments/table-01-cluster-coarsening",
        summary="Recall-matched fine-vs-coarse clustering numbers backing tab:intro-coarse-cluster.",
        prepare=public_datasets("sift1m"),
        run=phase(
            py(
                "runners/table-01-cluster-coarsening/run_data.py",
                "tune",
                "--dataset",
                "sift1m",
                "--fine-nlist",
                "4096",
                "--coarse-nlist",
                "1024",
                "--fine-nprobes",
                "128,160,192,224,256",
                "--coarse-nprobes",
                "32,48,64,80,96,112,128",
                "--queries",
                "300",
                "--query-seed",
                "12345",
                "--train-size",
                "100000",
                "--recall-k",
                "10",
            ),
            py(
                "runners/table-01-cluster-coarsening/run_data.py",
                "plan",
                "--dataset",
                "sift1m",
                "--fine-nlist",
                "4096",
                "--coarse-nlist",
                "1024",
            ),
            py(
                "runners/table-01-cluster-coarsening/run_data.py",
                "summarize",
                "--manifest",
                "experiments/table-01-cluster-coarsening/data/latest/manifest.json",
            ),
        ),
        render=phase(
            py(
                "runners/table-01-cluster-coarsening/generate.py",
                "--input-json",
                "experiments/table-01-cluster-coarsening/data/paper/summary.json",
                "--out-csv",
                "experiments/table-01-cluster-coarsening/intro_cluster_coarsening_sanity_check.csv",
                "--out-tex",
                "experiments/table-01-cluster-coarsening/intro_cluster_coarsening_sanity_check.tex",
            ),
        ),
        expected_outputs=(
            "experiments/table-01-cluster-coarsening/intro_cluster_coarsening_sanity_check.csv",
            "experiments/table-01-cluster-coarsening/intro_cluster_coarsening_sanity_check.tex",
        ),
        expected_runner_outputs=(
            "experiments/table-01-cluster-coarsening/data/latest/summary.json",
        ),
    ),
    ExperimentSpec(
        id="s3-piecewise-calibration-curves",
        title="Stage: S3 piecewise calibration curves",
        role="stage",
        paper_labels=(),
        paper_artifacts=(),
        out_dir="experiments/figure-04-object-store-io/stages/s3-piecewise-calibration-curves",
        summary="Piecewise calibration of S3 GET latency used by the background microbenchmark figure.",
        needs_s3=True,
        run=phase(py("runners/s3-piecewise-calibration-curves/run_data.py", "--rerun-probe")),
        expected_outputs=(
            "experiments/figure-04-object-store-io/stages/s3-piecewise-calibration-curves/data/paper/dense_probe.csv",
            "experiments/figure-04-object-store-io/stages/s3-piecewise-calibration-curves/data/paper/source_latency_model.json",
        ),
        expected_runner_outputs=(
            "experiments/figure-04-object-store-io/stages/s3-piecewise-calibration-curves/data/latest/dense_probe.csv",
            "experiments/figure-04-object-store-io/stages/s3-piecewise-calibration-curves/data/latest/source_latency_model.json",
        ),
    ),
    ExperimentSpec(
        id="real-s3-tiny-request-fanout-sanity",
        title="Stage: tiny-request fanout sanity",
        role="stage",
        paper_labels=(),
        paper_artifacts=(),
        out_dir="experiments/figure-04-object-store-io/stages/real-s3-tiny-request-fanout-sanity",
        summary="Tiny-request fanout stress measurement used by the background microbenchmark figure.",
        needs_lance=True,
        needs_s3=True,
        run=phase(py("runners/real-s3-tiny-request-fanout-sanity/run_data.py")),
        expected_outputs=(
            "experiments/figure-04-object-store-io/stages/real-s3-tiny-request-fanout-sanity/data/paper/summary.csv",
        ),
        expected_runner_outputs=(
            "experiments/figure-04-object-store-io/stages/real-s3-tiny-request-fanout-sanity/data/latest/summary.csv",
        ),
    ),
    ExperimentSpec(
        id="figure-04-object-store-io",
        title="Background: object-store I/O microbenchmarks",
        role="figure",
        paper_labels=("fig:bg-microbenchmarks",),
        paper_artifacts=(
            "experiments/figure-04-object-store-io/request_fanout.pdf",
            "experiments/figure-04-object-store-io/surrogate_runtime_semantics.pdf",
        ),
        out_dir="experiments/figure-04-object-store-io",
        summary=(
            "One top-level paper figure composed from the request-fanout and "
            "four-topology object-store calibration panels."
        ),
        depends_on=(
            "s3-piecewise-calibration-curves",
            "real-s3-tiny-request-fanout-sanity",
        ),
        needs_s3=True,
        run=phase(py("runners/figure-04-object-store-io/run_data.py")),
        render=phase(
            py("experiments/figure-04-object-store-io/render_request_fanout.py"),
            py("experiments/figure-04-object-store-io/render_surrogate_runtime_semantics.py"),
        ),
        expected_outputs=(
            "experiments/figure-04-object-store-io/request_fanout.pdf",
            "experiments/figure-04-object-store-io/surrogate_runtime_semantics.pdf",
        ),
        expected_runner_outputs=(
            "experiments/figure-04-object-store-io/data/latest/request_fanout.csv",
            "experiments/figure-04-object-store-io/data/latest/runtime/ap_same_region_serial_stats.json",
            "experiments/figure-04-object-store-io/data/latest/runtime/us_same_region_serial_stats.json",
            "experiments/figure-04-object-store-io/data/latest/runtime/cross_region_serial_stats.json",
            "experiments/figure-04-object-store-io/data/latest/runtime/public_internet_serial_stats.json",
        ),
    ),
    ExperimentSpec(
        id="figure-05-planning-frontier",
        title="Request planning: feasible plans and Pareto frontier",
        role="figure",
        paper_labels=("fig:planning-frontier",),
        paper_artifacts=(
            "experiments/figure-05-planning-frontier/figure_planning_bw_frontier.pdf",
        ),
        out_dir="experiments/figure-05-planning-frontier",
        summary=(
            "Deterministic explanatory figure showing the feasible (B, W) plan "
            "cloud and its lower Pareto frontier."
        ),
        render=phase(
            py("experiments/figure-05-planning-frontier/render_planning_frontier.py"),
        ),
        expected_outputs=(
            "experiments/figure-05-planning-frontier/figure_planning_bw_frontier.pdf",
        ),
    ),
    ExperimentSpec(
        id="figure-09-qps-cost",
        title="Evaluation: all-6-dataset QPS-vs-cost grid",
        role="figure",
        paper_labels=("fig:eval-qps-cost",),
        paper_artifacts=(
            "experiments/figure-09-qps-cost/figure_qps_vs_savings_q4_io64_all6_grid_qps_p95_3x4.pdf",
        ),
        out_dir="experiments/figure-09-qps-cost",
        summary="Main all-six-dataset QPS vs savings grid at q=4, io=64.",
        needs_lance=True,
        needs_s3=True,
        prepare=public_datasets(
            "sift1m",
            "glove",
            "deep1m",
            "openimage",
            "msong",
            "random1m128",
        ),
        run=phase(
            py("runners/figure-09-qps-cost/run_data.py"),
        ),
        render=phase(
            py(
                "experiments/figure-09-qps-cost/render_qps_vs_savings_all6_grid.py"
            )
        ),
        expected_outputs=(
            "experiments/figure-09-qps-cost/figure_qps_vs_savings_q4_io64_all6_grid_qps_p95_3x4.pdf",
        ),
        expected_runner_outputs=(
            "experiments/figure-09-qps-cost/data/latest/aggregated_results.json",
        ),
    ),
    ExperimentSpec(
        id="figure-10-p-sweep-closedloop",
        title="Evaluation: closed-loop p-sweep QPS/latency tradeoff",
        role="figure",
        paper_labels=("fig:eval-p-sweep-closedloop",),
        paper_artifacts=(
            "experiments/figure-10-p-sweep-closedloop/figure_closed_loop_env3_reorder_only_combined.pdf",
        ),
        out_dir="experiments/figure-10-p-sweep-closedloop",
        summary="Closed-loop env3 reorder-only QPS/latency tradeoff across the p-sweep.",
        needs_lance=True,
        needs_s3=True,
        prepare=public_datasets(
            "sift1m",
            "glove",
            "deep1m",
            "openimage",
            "msong",
            "random1m128",
        ),
        run=phase(
            py("runners/figure-10-p-sweep-closedloop/run_data.py", "--preset", "env3"),
        ),
        render=phase(
            py(
                "experiments/figure-10-p-sweep-closedloop/render_closed_loop_p_sweep.py"
            )
        ),
        expected_outputs=(
            "experiments/figure-10-p-sweep-closedloop/figure_closed_loop_env3_reorder_only_combined.pdf",
        ),
        expected_runner_outputs=(
            "experiments/figure-10-p-sweep-closedloop/data/latest/data_gap_env3_v3/figure_closed_loop_env3_reorder_only_data.json",
        ),
    ),
    ExperimentSpec(
        id="figure-11-greedy-locality-shift",
        title="Evaluation: greedy policy transfer under workload shift",
        role="figure",
        paper_labels=("fig:eval-greedy-locality-shift",),
        paper_artifacts=(
            "experiments/figure-11-greedy-locality-shift/figure_greedy_policy_transfer.pdf",
        ),
        out_dir="experiments/figure-11-greedy-locality-shift",
        summary=(
            "SIFT-tuned Greedy versus WaRP under query-load and cross-dataset "
            "shift using the two-stage SIFT/MSong protocol."
        ),
        needs_lance=True,
        needs_lance_python=True,
        needs_s3=True,
        prepare=phase(
            py("runners/figure-11-greedy-locality-shift/prepare_lance_toolchain.py"),
            py(
                "runners/figure-11-greedy-locality-shift/prepare_public_data.py",
                "--download-public-data",
            ),
            py(
                "runners/figure-11-greedy-locality-shift/prepare_msong_index.py",
                "--download-public-data",
            ),
        ),
        run=phase(py("runners/figure-11-greedy-locality-shift/run_data.py")),
        render=phase(
            py("experiments/figure-11-greedy-locality-shift/render_greedy_policy_transfer.py"),
        ),
        expected_outputs=(
            "experiments/figure-11-greedy-locality-shift/figure_greedy_policy_transfer.pdf",
        ),
        expected_runner_outputs=(
            "experiments/figure-11-greedy-locality-shift/data/latest/sift_summary.json",
            "experiments/figure-11-greedy-locality-shift/data/latest/msong_summary.json",
        ),
    ),
    ExperimentSpec(
        id="figure-12-planner-overhead",
        title="Evaluation: real-data planner overhead (figure + table)",
        role="figure",
        paper_labels=("fig:eval-planner-overhead-breakdown", "tab:eval-planner-overhead"),
        paper_artifacts=(
            "experiments/figure-12-planner-overhead/figure_03_real_data_planner_overhead_piecewise.pdf",
            "experiments/figure-12-planner-overhead/table_03_real_data_planner_overhead_piecewise.tex",
        ),
        out_dir="experiments/figure-12-planner-overhead",
        summary="Planner-overhead benchmark; produces both the four-way figure and the LaTeX table.",
        needs_lance=True,
        prepare=public_datasets("sift1m", "glove"),
        run=phase(py("runners/figure-12-planner-overhead/run_data.py")),
        render=phase(
            py(
                "experiments/figure-12-planner-overhead/render_planner_overhead_fourway.py",
                "--input-csv",
                "experiments/figure-12-planner-overhead/data/paper/merged_summary.csv",
                "--out-dir",
                "experiments/figure-12-planner-overhead",
                "--palette-family",
                "blue",
            ),
        ),
        expected_outputs=(
            "experiments/figure-12-planner-overhead/figure_03_real_data_planner_overhead_piecewise.pdf",
            "experiments/figure-12-planner-overhead/table_03_real_data_planner_overhead_piecewise.tex",
        ),
        expected_runner_outputs=(
            "experiments/figure-12-planner-overhead/data/latest/merged_summary.csv",
        ),
    ),
    ExperimentSpec(
        id="locality-aware-coaccess-gap",
        title="Stage: locality-aware co-access gap",
        role="stage",
        paper_labels=(),
        paper_artifacts=(),
        out_dir="experiments/figure-13-reordering-methods/stages/locality-aware-coaccess-gap",
        summary="Locality-quality signal computed per reordering method; feeds the figure-13-reordering-methods right panel.",
        prepare=public_datasets(
            "sift1m",
            "glove",
            "deep1m",
            "openimage",
            "msong",
            "random1m128",
        ),
        run=phase(
            py("runners/locality-aware-coaccess-gap/run_data.py"),
        ),
        expected_outputs=(
            "experiments/figure-13-reordering-methods/stages/locality-aware-coaccess-gap/data/paper/figure_locality_aware_coaccess_gap.csv",
        ),
        expected_runner_outputs=(
            "experiments/figure-13-reordering-methods/stages/locality-aware-coaccess-gap/data/latest/figure_locality_aware_coaccess_gap.csv",
        ),
    ),
    ExperimentSpec(
        id="reorder-cost",
        title="Stage: offline reordering cost sweep",
        role="stage",
        paper_labels=(),
        paper_artifacts=(),
        out_dir="experiments/figure-13-reordering-methods/stages/reorder-cost",
        summary="Offline reordering runtime sweep across methods and cluster counts; feeds the figure-13-reordering-methods left panel.",
        prepare=public_datasets("sift1m"),
        run=phase(py("runners/reorder-cost/run_data.py")),
        expected_outputs=(
            "experiments/figure-13-reordering-methods/stages/reorder-cost/data/paper/figure_reorder_cost_cluster_sweep.csv",
        ),
        expected_runner_outputs=(
            "experiments/figure-13-reordering-methods/stages/reorder-cost/data/latest/figure_reorder_cost_cluster_sweep.csv",
        ),
    ),
    ExperimentSpec(
        id="figure-13-reordering-methods",
        title="Evaluation: composite reorder panels",
        role="figure",
        paper_labels=("fig:reorder-panels",),
        paper_artifacts=("experiments/figure-13-reordering-methods/figure_reorder_panels.pdf",),
        out_dir="experiments/figure-13-reordering-methods",
        summary="Composite reorder figure; left panel from reorder-cost, right panel from locality-aware-coaccess-gap.",
        depends_on=("reorder-cost", "locality-aware-coaccess-gap"),
        run=phase(py("runners/figure-13-reordering-methods/run_data.py")),
        render=phase(
            py(
                "experiments/figure-13-reordering-methods/render_reorder_panels.py",
                "--palette-family",
                "red",
            )
        ),
        expected_outputs=("experiments/figure-13-reordering-methods/figure_reorder_panels.pdf",),
        expected_runner_outputs=(
            "experiments/figure-13-reordering-methods/data/latest/stages/reorder_cost.csv",
            "experiments/figure-13-reordering-methods/data/latest/stages/locality_gap.csv",
        ),
    ),
    ExperimentSpec(
        id="figure-14-sift1m-sensitivity",
        title="Evaluation: SIFT1M real-S3 sensitivity",
        role="figure",
        paper_labels=("fig:eval-sensitivity-sift1m",),
        paper_artifacts=(
            "experiments/figure-14-sift1m-sensitivity/figure_sift1m_real_s3_sensitivity_combined.pdf",
        ),
        out_dir="experiments/figure-14-sift1m-sensitivity",
        summary="SIFT1M nprobe/granularity sweep under real-S3 latency.",
        needs_lance=True,
        needs_s3=True,
        prepare=public_datasets("sift1m"),
        run=phase(
            py("runners/figure-14-sift1m-sensitivity/run_data.py"),
        ),
        render=phase(
            py(
                "experiments/figure-14-sift1m-sensitivity/render_sift1m_sensitivity_split.py",
                "--palette-family",
                "blue",
            )
        ),
        expected_outputs=(
            "experiments/figure-14-sift1m-sensitivity/figure_sift1m_real_s3_sensitivity_combined.pdf",
        ),
        expected_runner_outputs=(
            "experiments/figure-14-sift1m-sensitivity/data/latest/table_sift1m_real_s3_sensitivity.csv",
        ),
    ),
    ExperimentSpec(
        id="figure-07-real-lance-metrics",
        title="Evaluation: real-Lance normalized end-to-end validation",
        role="figure",
        paper_labels=("fig:eval-real-lance-metrics", "tab:eval-payload-accounting"),
        paper_artifacts=(
            "experiments/figure-07-real-lance-metrics/figure_real_lance_metrics.pdf",
            "experiments/figure-07-real-lance-metrics/table_payload_accounting.tex",
        ),
        out_dir="experiments/figure-07-real-lance-metrics",
        summary=(
            "Four-panel normalized comparison and payload-accounting table "
            "from the complete Figure 7 real-Lance campaigns."
        ),
        needs_lance=True,
        needs_lance_python=True,
        needs_s3=True,
        prepare=phase(
            py("runners/shared/verify_lance_base.py"),
            py(
                "runners/figure-07-real-lance-metrics/prepare_matched_datasets.py",
                "--download-public-data",
            )
        ),
        run=phase(py("runners/figure-07-real-lance-metrics/run_data.py")),
        render=phase(
            py("experiments/figure-07-real-lance-metrics/render_figure7.py"),
            py("experiments/figure-07-real-lance-metrics/generate_payload_accounting_table.py"),
        ),
        expected_outputs=(
            "experiments/figure-07-real-lance-metrics/figure_real_lance_metrics.pdf",
            "experiments/figure-07-real-lance-metrics/table_payload_accounting.tex",
            "experiments/figure-07-real-lance-metrics/table_payload_accounting.csv",
        ),
        expected_runner_outputs=(
            "experiments/figure-07-real-lance-metrics/data/latest/figure7_source.csv",
            "experiments/figure-07-real-lance-metrics/data/latest/figure7_performance_summary.json",
            "experiments/figure-07-real-lance-metrics/data/latest/figure7_cache_summary.json",
        ),
    ),
    ExperimentSpec(
        id="figure-08-sift100m-concurrency-scaling",
        title="Evaluation: real-Lance concurrency scaling",
        role="figure",
        paper_labels=("fig:eval-real-lance-concurrency-scaling",),
        paper_artifacts=(
            "experiments/figure-08-sift100m-concurrency-scaling/figure_real_lance_concurrency_scaling.pdf",
        ),
        out_dir="experiments/figure-08-sift100m-concurrency-scaling",
        summary=(
            "SIFT100M I/O-limit latency plus query-concurrency throughput and "
            "p95 latency from the Figure 8 I/O-limit and query-concurrency campaigns."
        ),
        needs_lance=True,
        needs_lance_python=True,
        needs_s3=True,
        prepare=phase(
            py("runners/shared/verify_lance_base.py"),
            py(
                "runners/figure-07-real-lance-metrics/prepare_matched_datasets.py",
                "--datasets",
                "sift100m",
                "--download-public-data",
            )
        ),
        run=phase(py("runners/figure-08-sift100m-concurrency-scaling/run_data.py")),
        render=phase(
            py("experiments/figure-08-sift100m-concurrency-scaling/render_concurrency_scaling.py"),
        ),
        expected_outputs=(
            "experiments/figure-08-sift100m-concurrency-scaling/figure_real_lance_concurrency_scaling.pdf",
        ),
        expected_runner_outputs=(
            "experiments/figure-08-sift100m-concurrency-scaling/data/latest/query_summary.json",
            "experiments/figure-08-sift100m-concurrency-scaling/data/latest/io_summaries/ladder.json",
            "experiments/figure-08-sift100m-concurrency-scaling/data/latest/io_summaries/2048-4096.json",
            "experiments/figure-08-sift100m-concurrency-scaling/data/latest/io_summaries/4096-8192.json",
        ),
    ),
)


# ----------------------------------------------------------------------
# Accessors
# ----------------------------------------------------------------------

def experiment_ids() -> list[str]:
    return [spec.id for spec in EXPERIMENTS]


def get(spec_id: str) -> ExperimentSpec:
    for spec in EXPERIMENTS:
        if spec.id == spec_id:
            return spec
    raise KeyError(f"unknown experiment id: {spec_id} (known: {', '.join(experiment_ids())})")


def paper_artifact_specs() -> list[ExperimentSpec]:
    return [spec for spec in EXPERIMENTS if spec.role in {"figure", "table"}]


def intermediates() -> list[ExperimentSpec]:
    return [spec for spec in EXPERIMENTS if spec.role == "stage"]


def run_phase(spec: ExperimentSpec, profile: str) -> Phase:
    """Return the commands to execute for the run phase at `profile`.

    Light mode always returns an empty Phase so that reviewers can validate
    the plot pipeline without Lance, S3, or EC2 access.
    """
    if profile == "light":
        return Phase()
    if profile == "full":
        return spec.run
    raise ValueError(f"unknown profile: {profile!r} (expected 'full' or 'light')")


def prepare_phase(spec: ExperimentSpec, profile: str) -> Phase:
    """Return public asset/toolchain preparation commands for a profile."""
    if profile == "light":
        return Phase()
    if profile == "full":
        return spec.prepare
    raise ValueError(f"unknown profile: {profile!r} (expected 'full' or 'light')")


def render_phase(spec: ExperimentSpec, profile: str) -> Phase:
    """Return the commands to execute for the render phase at `profile`.

    Render behaviour is identical across profiles; the profile only affects
    whether the run phase has re-populated the data first.
    """
    if profile not in {"full", "light"}:
        raise ValueError(f"unknown profile: {profile!r} (expected 'full' or 'light')")
    return spec.render


def expected_outputs(spec: ExperimentSpec, profile: str) -> tuple[str, ...]:
    del profile  # same outputs regardless of profile
    return spec.expected_outputs


def ordered_for_run(selected: list[str] | None = None) -> list[ExperimentSpec]:
    """Return specs ordered so that each spec's `depends_on` ids precede it."""
    picked: dict[str, ExperimentSpec] = {}
    order: list[ExperimentSpec] = []

    def visit(spec_id: str, path: tuple[str, ...]) -> None:
        if spec_id in picked:
            return
        if spec_id in path:
            raise ValueError(f"dependency cycle: {' -> '.join(path + (spec_id,))}")
        spec = get(spec_id)
        for dep in spec.depends_on:
            visit(dep, path + (spec_id,))
        picked[spec_id] = spec
        order.append(spec)

    targets = (
        selected
        if selected is not None
        else [spec.id for spec in EXPERIMENTS]
    )
    for spec_id in targets:
        visit(spec_id, ())
    return order
