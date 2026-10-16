#!/usr/bin/env python3
"""Render the SIFT100M I/O and Query Concurrency sweeps as one triptych."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.plotting.paper_plot_style import style_legend_frame
from artifact.data import data_slot

Q_RENDERER = HERE / "render_query_series.py"
IO_RENDERER = HERE / "render_io_series.py"
OUT = HERE / "figure_real_lance_concurrency_scaling"
FIGSIZE = (7.0, 1.20)

IO_LIMITS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)
IO_TICK_LABELS = ("64", "128", "256", "512", "1k", "2k", "4k", "8k")
IO_METRICS = (
    ("p50_ms", "p50", "#2b8cbe"),
    ("p95_ms", "p95", "#d95f0e"),
)

EDGE = "#1a1a1a"
GRID = "#adb5bd"
PLATEAU_FILL = "#d8e5ef"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path: Path, expected: str, label: str) -> None:
    if not path.is_file():
        raise ValueError(f"{label} is missing: {path}")
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{label} checksum mismatch: expected {expected}, got {actual}")


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def configure_style(q_renderer: ModuleType) -> None:
    q_renderer.apply_paper_preset(
        plt,
        "double_col",
        font_size=7.1,
        axes_label_size=7.1,
        axes_title_size=7.1,
        legend_size=5.3,
        xtick_size=5.9,
        ytick_size=5.9,
        axes_linewidth=0.75,
        extra_rc={
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.major.width": 0.62,
            "ytick.major.width": 0.62,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "savefig.bbox": None,
            "svg.hashsalt": "warp-sift100m-concurrency-triptych",
        },
    )


def load_io_data(io_renderer: ModuleType) -> tuple[ModuleType, dict[str, dict]]:
    base = io_renderer.load_module(
        io_renderer.BASE_RENDERER,
        "concurrency_triptych_io_base",
    )
    summaries = {
        name: base.load_bundle(spec)
        for name, spec in base.BUNDLES.items()
    }
    for name, summary in summaries.items():
        for limit in base.BUNDLES[name].expected_limits:
            for metric, _, _ in IO_METRICS:
                values = summary["groups"][str(limit)][metric]["values"]
                if len(values) != 3:
                    raise ValueError(
                        f"expected three {metric} repeats for {name}:{limit}"
                    )
    return base, summaries


def plot_io_panel(
    axis,
    io_renderer: ModuleType,
    summaries: dict[str, dict],
) -> dict[str, list[dict[str, float]]]:
    positions = np.arange(len(IO_LIMITS), dtype=float)
    axis.axvspan(4.65, 7.45, color=PLATEAU_FILL, alpha=0.48, linewidth=0, zorder=0)
    axis.axvline(5.0, color="#607889", linewidth=0.55, linestyle="--", zorder=1)

    paired_evidence: dict[str, list[dict[str, float]]] = {}
    for metric, label, color in IO_METRICS:
        means, errors, paired_changes = io_renderer.paired_chained_series(
            summaries,
            metric,
        )
        paired_evidence[label] = paired_changes
        axis.errorbar(
            positions,
            means,
            yerr=errors,
            color=color,
            marker="o",
            markersize=2.8,
            markerfacecolor=color,
            markeredgecolor=EDGE,
            markeredgewidth=0.32,
            linewidth=0.72,
            linestyle="-",
            capsize=0.9,
            elinewidth=0.40,
            capthick=0.40,
            alpha=0.96,
            zorder=3,
        )

    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=0.72,
            marker="o",
            markersize=2.8,
            markerfacecolor=color,
            markeredgecolor=EDGE,
            markeredgewidth=0.32,
            label=label,
        )
        for _, label, color in IO_METRICS
    ]
    legend = axis.legend(
        handles=handles,
        loc="upper right",
        frameon=True,
        ncol=1,
        borderpad=0.22,
        labelspacing=0.16,
        handlelength=1.25,
        handletextpad=0.28,
    )
    style_legend_frame(legend)

    axis.text(
        6.25,
        980,
        "plateau from 2k",
        ha="center",
        va="center",
        fontsize=5.0,
        color="#4b6577",
    )
    axis.set_title("(a) Lance latency", pad=2.8, fontweight="regular")
    axis.set_xlim(-0.45, len(IO_LIMITS) - 0.55)
    axis.set_yscale("log", base=2)
    axis.set_ylim(350, 9000)
    axis.yaxis.set_major_locator(FixedLocator([400, 800, 1600, 3200, 6400]))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
    axis.set_xticks(positions)
    axis.set_xticklabels(IO_TICK_LABELS)
    for tick_label, limit in zip(axis.get_xticklabels(), IO_LIMITS):
        if limit == 2048:
            tick_label.set_fontweight("bold")
            tick_label.set_color("#3f596b")
    axis.set_xlabel("I/O Concurrency Limit", labelpad=1.3)
    axis.set_ylabel("Latency (ms)", labelpad=1.8)
    axis.yaxis.grid(
        True,
        linestyle="--",
        dashes=(2, 2),
        alpha=0.62,
        linewidth=0.48,
        color=GRID,
    )
    axis.xaxis.grid(
        True,
        linestyle=":",
        alpha=0.22,
        linewidth=0.36,
        color=GRID,
    )
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", length=2.4, width=0.62, direction="out")
    return paired_evidence


def plot_query_panels(axes, q_renderer: ModuleType, matrix: dict) -> None:
    panels = (
        ("(b) Throughput", "QPS", "qps", (0, 165), (0, 40, 80, 120, 160)),
        (
            "(c) p95 end-to-end latency",
            "Latency (ms)",
            "p95_ms",
            (0, 1650),
            (0, 400, 800, 1200, 1600),
        ),
    )
    x = np.asarray(q_renderer.Q_LEVELS, dtype=float)
    for axis, (title, ylabel, metric, ylim, yticks) in zip(axes, panels):
        for label, method, color, marker, linestyle in q_renderer.METHODS:
            groups = [
                matrix[(method, q)]["metrics"][metric]
                for q in q_renderer.Q_LEVELS
            ]
            means = np.asarray([group["mean"] for group in groups], dtype=float)
            axis.plot(
                x,
                means,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=0.88,
                marker=marker,
                markersize=3.3,
                markerfacecolor=color,
                markeredgecolor=EDGE,
                markeredgewidth=0.38,
                zorder=3,
            )

        axis.set_title(title, pad=2.8, fontweight="regular")
        axis.set_ylabel(ylabel, labelpad=1.8)
        axis.set_xlabel("Query Concurrency ($Q$)", labelpad=1.3)
        axis.set_xticks(q_renderer.Q_LEVELS)
        axis.set_xlim(11.0, 133.0)
        axis.set_ylim(*ylim)
        axis.set_yticks(yticks)
        axis.yaxis.grid(
            True,
            linestyle="--",
            dashes=(2, 2),
            alpha=0.62,
            linewidth=0.48,
            color=GRID,
        )
        axis.xaxis.grid(
            True,
            linestyle=":",
            alpha=0.24,
            linewidth=0.38,
            color=GRID,
        )
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", length=2.4, width=0.62, direction="out")
        handles, labels = axis.get_legend_handles_labels()
        legend = axis.legend(
            handles,
            labels,
            loc="lower right",
            ncol=1,
            frameon=True,
            fontsize=5.0,
            handlelength=1.18,
            handletextpad=0.24,
            borderpad=0.20,
            labelspacing=0.12,
        )
        style_legend_frame(legend)


def main() -> None:
    q_renderer = load_module(Q_RENDERER, "concurrency_triptych_query_renderer")
    io_renderer = load_module(IO_RENDERER, "concurrency_triptych_io_renderer")

    q_data = q_renderer.load_summary()
    q_matrix = q_renderer.aggregate_matrix(q_data)
    io_base, io_summaries = load_io_data(io_renderer)
    configure_style(q_renderer)

    fig, axes = plt.subplots(1, 3, figsize=FIGSIZE, dpi=320)
    paired_evidence = plot_io_panel(axes[0], io_renderer, io_summaries)
    plot_query_panels(axes[1:], q_renderer, q_matrix)
    fig.subplots_adjust(left=0.067, right=0.996, bottom=0.285, top=0.875, wspace=0.31)

    pdf_path = OUT.with_suffix(".pdf")
    svg_path = OUT.with_suffix(".svg")
    png_path = OUT.with_suffix(".png")
    fig.savefig(
        pdf_path,
        bbox_inches=None,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        svg_path,
        bbox_inches=None,
        metadata={"Date": None},
    )
    plt.close(fig)
    subprocess.run(
        (
            "pdftoppm",
            "-png",
            "-r",
            "320",
            "-singlefile",
            str(pdf_path),
            str(OUT),
        ),
        check=True,
    )

    slot = data_slot()
    if slot == "paper":
        io_sources = {
            key: {
                "filename": spec.filename,
                "sha256": spec.sha256,
                "campaign": spec.expected_campaign,
                "limits": list(spec.expected_limits),
            }
            for key, spec in io_base.BUNDLES.items()
        }
    else:
        io_sources = {}
        for key, spec in io_base.BUNDLES.items():
            path = io_base.SUMMARY_ROOT / spec.filename
            io_sources[key] = {
                "filename": str(path.relative_to(ROOT)),
                "sha256": sha256(path),
                "campaign": spec.expected_campaign,
                "limits": list(spec.expected_limits),
            }
    provenance = {
        "status": (
            "paper-frozen"
            if slot == "paper"
            else "fresh-run-render"
        ),
        "evidence_slot": slot,
        "matches_committed_paper_evidence": True if slot == "paper" else None,
        "render_preserves_source_values": False,
        "deterministic_export": True,
        "physical_size_inches": list(FIGSIZE),
        "panels": [
            {
                "panel": "a",
                "dataset": "SIFT100M",
                "method": "Lance",
                "query_concurrency": 32,
                "queries_per_cell": 1_000,
                "warmup_queries": 200,
                "refine_factor": 8,
                "repeats": 3,
                "io_concurrency_limits": list(IO_LIMITS),
                "metrics": [metric for metric, _, _ in IO_METRICS],
                "source_summaries": io_sources,
                "paired_normalization": paired_evidence,
            },
            {
                "panels": ["b", "c"],
                "dataset": "SIFT100M",
                "methods": [label for label, _, _, _, _ in q_renderer.METHODS],
                "query_concurrency_levels": list(q_renderer.Q_LEVELS),
                "io_concurrency_limit": 2048,
                "metrics": ["qps", "p95_ms"],
                "queries_per_cell": 10_000,
                "warmup_queries": 200,
                "refine_factor": 8,
                "repeats": 3,
                "source_summary": str(q_renderer.DATA.relative_to(ROOT)),
                "source_summary_sha256": sha256(q_renderer.DATA),
            },
        ],
        "source_renderers": {
            str(Q_RENDERER.relative_to(ROOT)): sha256(Q_RENDERER),
            str(IO_RENDERER.relative_to(ROOT)): sha256(IO_RENDERER),
        },
        "output_sha256": {
            pdf_path.name: sha256(pdf_path),
            png_path.name: sha256(png_path),
            svg_path.name: sha256(svg_path),
        },
    }
    OUT.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
