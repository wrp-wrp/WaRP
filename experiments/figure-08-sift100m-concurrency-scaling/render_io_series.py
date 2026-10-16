#!/usr/bin/env python3
"""Render the Figure 8(a) I/O-limit panel with p50 and p95 latency."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, FixedLocator


BASE_RENDERER = Path(__file__).resolve().parent / "render_io_source.py"
OUT = Path(__file__).resolve().parent / "03_figure8a_q32_io_limit_p50_p95_preview"

EDGE = "#1a1a1a"
GRID = "#adb5bd"
FILL = "#d8e5ef"
LINE_WIDTH = 0.58
MARKER_SIZE = 2.15
MARKER_EDGE_WIDTH = 0.28
METRICS = (
    ("p50_ms", "p50", "#2b8cbe"),
    ("p95_ms", "p95", "#d95f0e"),
)
ALL_LIMITS = (64, 128, 256, 512, 1024, 2048, 4096, 8192)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_module(path: Path, name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def paired_chained_series(
    summaries: dict[str, dict],
    metric: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float]]]:
    ladder = summaries["ladder"]["groups"]
    means = [float(ladder[str(limit)][metric]["mean"]) for limit in ALL_LIMITS[:6]]
    errors = [
        float(ladder[str(limit)][metric]["sample_sd"]) for limit in ALL_LIMITS[:6]
    ]
    paired_changes: list[dict[str, float]] = []
    current_mean = means[-1]
    current_sd = errors[-1]
    for campaign, low_limit, high_limit in (
        ("2048-4096", 2048, 4096),
        ("4096-8192", 4096, 8192),
    ):
        groups = summaries[campaign]["groups"]
        low = groups[str(low_limit)][metric]
        high = groups[str(high_limit)][metric]
        low_values = [float(value) for value in low["values"]]
        high_values = [float(value) for value in high["values"]]
        ratios = np.asarray(
            [high_value / low_value for low_value, high_value in zip(low_values, high_values)]
        )
        ratio_mean = float(high["mean"]) / float(low["mean"])
        ratio_sd = float(np.std(ratios, ddof=1))
        next_mean = current_mean * ratio_mean
        relative_variance = (current_sd / current_mean) ** 2 + (ratio_sd / ratio_mean) ** 2
        next_sd = next_mean * float(np.sqrt(relative_variance))
        means.append(next_mean)
        errors.append(next_sd)
        paired_changes.append(
            {
                "from_limit": float(low_limit),
                "to_limit": float(high_limit),
                "raw_from_mean_ms": float(low["mean"]),
                "raw_to_mean_ms": float(high["mean"]),
                "paired_mean_change_pct": 100.0 * (ratio_mean - 1.0),
                "display_mean_ms": next_mean,
                "display_sample_sd_ms": next_sd,
            }
        )
        current_mean = next_mean
        current_sd = next_sd
    return np.asarray(means), np.asarray(errors), paired_changes


def main() -> None:
    base = load_module(BASE_RENDERER, "figure8a_q32_all_quantiles_base")
    summaries = {name: base.load_bundle(spec) for name, spec in base.BUNDLES.items()}

    for name, summary in summaries.items():
        for limit in base.BUNDLES[name].expected_limits:
            for metric, _, _ in METRICS:
                values = summary["groups"][str(limit)][metric]["values"]
                if len(values) != 3:
                    raise ValueError(f"expected three {metric} repeats for {name}:{limit}")

    base.apply_style()
    fig, axis = plt.subplots(figsize=(1.675, 1.58), dpi=320)
    all_limits = np.asarray(ALL_LIMITS)
    positions = np.arange(len(all_limits), dtype=float)
    axis.axvspan(5.35, 7.35, color=FILL, alpha=0.48, linewidth=0, zorder=0)
    axis.axvline(5.0, color="#607889", linewidth=0.5, linestyle="--", zorder=1)

    paired_evidence: dict[str, list[dict[str, float]]] = {}
    for metric, label, color in METRICS:
        means, errors, paired_changes = paired_chained_series(summaries, metric)
        paired_evidence[label] = paired_changes
        axis.errorbar(
            positions,
            means,
            yerr=errors,
            color=color,
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor=EDGE,
            markeredgewidth=MARKER_EDGE_WIDTH,
            linewidth=LINE_WIDTH,
            linestyle="-",
            capsize=0.8,
            elinewidth=0.34,
            capthick=0.34,
            alpha=0.94,
            zorder=3,
        )

    metric_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=LINE_WIDTH,
            marker="o",
            markersize=MARKER_SIZE,
            markerfacecolor=color,
            markeredgecolor=EDGE,
            markeredgewidth=MARKER_EDGE_WIDTH,
            label=label,
        )
        for _, label, color in METRICS
    ]
    legend = axis.legend(
        handles=metric_handles,
        loc="upper right",
        frameon=True,
        ncol=1,
        borderpad=0.25,
        labelspacing=0.2,
        handlelength=1.35,
        handletextpad=0.32,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor("#333333")
    legend.get_frame().set_linewidth(0.42)

    axis.text(
        6.25,
        900,
        "p95 rises\nabove 2048",
        ha="center",
        va="center",
        fontsize=4.6,
        color="#4b6577",
        linespacing=0.95,
    )
    axis.set_title("(a) Lance latency", pad=2.8, fontweight="regular")
    axis.set_xlim(-0.45, len(all_limits) - 0.55)
    axis.set_yscale("log", base=2)
    axis.set_ylim(350, 9000)
    axis.yaxis.set_major_locator(FixedLocator([400, 800, 1600, 3200, 6400]))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
    axis.set_xticks(positions)
    axis.set_xticklabels([str(limit) for limit in all_limits], rotation=42, ha="right")
    for tick_label, limit in zip(axis.get_xticklabels(), ALL_LIMITS):
        if limit == 2048:
            tick_label.set_fontweight("bold")
            tick_label.set_color("#3f596b")
    axis.set_xlabel("I/O Concurrency Limit", labelpad=1.0)
    axis.set_ylabel("Latency (ms)", labelpad=1.5)
    axis.yaxis.grid(
        True,
        linestyle="--",
        dashes=(2, 2),
        alpha=0.62,
        linewidth=0.45,
        color=GRID,
    )
    axis.set_axisbelow(True)
    axis.tick_params(axis="both", length=2.0, width=0.55, direction="out")

    fig.subplots_adjust(left=0.235, right=0.985, bottom=0.27, top=0.86)
    fig.savefig(
        OUT.with_suffix(".pdf"),
        bbox_inches=None,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches=None)
    plt.close(fig)

    provenance = {
        "status": "preview",
        "query_concurrency": 32,
        "metrics": [metric for metric, _, _ in METRICS],
        "y_scale": "log2",
        "base_renderer": BASE_RENDERER.name,
        "base_renderer_sha256": sha256_file(BASE_RENDERER),
        "source_summaries": {
            key: {
                "filename": spec.filename,
                "sha256": spec.sha256,
                "campaign": spec.expected_campaign,
                "limits": list(spec.expected_limits),
            }
            for key, spec in base.BUNDLES.items()
        },
        "visual_encoding": {
            "color": "latency quantile",
            "line": "one continuous thin solid line per quantile",
            "error_bars": (
                "sample standard deviation for the base ladder; propagated repeat-ratio "
                "dispersion for paired-normalized extension points"
            ),
            "legend": (
                "same line width, marker, color, font preset, frame, and spacing "
                "as the plotted series and the frozen Figure 8(a) legend"
            ),
        },
        "paired_normalization": {
            "absolute_anchor": "base-ladder means for 64 through 2048",
            "extension_rule": (
                "apply the within-campaign 4096/2048 and 8192/4096 mean "
                "ratios successively to the base-ladder 2048 anchor"
            ),
            "paired_changes": paired_evidence,
        },
        "note": (
            "The extension is paired-normalized rather than pooled, so the "
            "continuous line preserves within-campaign direction despite campaign drift."
        ),
    }
    OUT.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
