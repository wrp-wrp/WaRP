#!/usr/bin/env python3
"""Render normalized Figure 7 from the frozen performance and cache evidence."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.plotting.paper_plot_style import apply_paper_preset
from artifact.data import frozen_evidence, paper_data_path


SOURCE_CSV = paper_data_path(HERE / "data" / "paper" / "figure7_source.csv")
SOURCE_CSV_SHA256 = "ee3efbd596619ea86c875cf405bae1ab2dde6278bd7bc3d57ba577f6e907de07"
PERFORMANCE_SUMMARY = paper_data_path(HERE / "data" / "paper" / "figure7_performance_summary.json")
PERFORMANCE_SUMMARY_SHA256 = "73bdc1ba41410907b3b816fa3f202af2c0c84ef12ebc331686a2b8e35c0cbbca"
CACHE_SUMMARY = paper_data_path(HERE / "data" / "paper" / "figure7_cache_summary.json")
CACHE_SUMMARY_SHA256 = "839f0e821c9a9443df28215eb98ff8188cea5eb2240910a95db9815168b2bfd8"
OUT = Path(__file__).resolve().parent / "01_figure7_normalized_red_cache_right"
FIGSIZE = (7.2, 2.52)
BOTTOM_MARGIN = 0.225
SUBPLOT_LEFT = 0.064
SUBPLOT_RIGHT = 0.996
SUBPLOT_WSPACE = 0.19

FONT_SIZE = 7.15
AXES_LABEL_SIZE = 6.8
AXES_TITLE_SIZE = 7.2
LEGEND_SIZE = 6.2
XTICK_SIZE = 5.25
YTICK_SIZE = 5.9
REGION_LABEL_SIZE = 4.25
CACHE_LABEL_SIZE = 4.45
VALUE_LABEL_SIZE = 4.15

WORKLOADS = (
    ("sift1m", "SIFT1M", "dataset"),
    ("glove", "GloVe", "dataset"),
    ("openimage", "Open\nImage", "dataset"),
    ("msong", "MSong", "dataset"),
    ("gist1m", "GIST1M", "dataset"),
    ("sift100m", "SIFT\n100M", "dataset"),
    ("deep100m", "DEEP\n100M", "dataset"),
    ("sift100m_cache10", "10%", "cache"),
    ("sift100m_cache50", "50%", "cache"),
)
POSITIONS = np.asarray((0, 0.88, 1.76, 2.64, 3.52, 4.40, 5.28, 7.32, 8.20), dtype=float)
BAR_WIDTH = 0.22
CACHE_BAND = (6.85, 8.62)
CACHE_SEPARATOR = 6.55
CACHE_CENTER = 7.76
X_LIMITS = (-0.45, 8.65)
METHODS = (
    ("stock_lance", "Lance", "#fdcc8a"),
    ("warp", "WaRP", "#fc8d59"),
    ("warp_reorder", "WaRP + Reorder", "#d7301f"),
)
PANELS = (
    ("(a) Throughput speedup", "qps_mean", "qps_sample_sd", False, (0.0, 6.25), (0, 1, 2, 3, 4, 5, 6)),
    ("(b) p95 latency reduction", "p95_ms_mean", "p95_ms_sample_sd", True, (0.0, 6.25), (0, 1, 2, 3, 4, 5, 6)),
    ("(c) p99 latency reduction", "p99_ms_mean", "p99_ms_sample_sd", True, (0.0, 8.25), (0, 1, 2, 3, 4, 5, 6, 7, 8)),
    ("(d) Number of GETs reduction", "get_per_q_mean", "get_per_q_sample_sd", True, (0.0, 7.25), (0, 1, 2, 3, 4, 5, 6, 7)),
)

EDGE = "#2a1b17"
LABEL = "#292421"
GRID = "#c9bdb7"
REFERENCE = "#626262"
CACHE_FILL = "#fff2e8"
CACHE_EDGE = "#c56549"
CACHE_LABEL = "#8f3b28"
LEGEND_EDGE = "#4a3832"
SVG_HASHSALT = "warp-figure7-normalized-red"
PALETTE_FAMILY = "paper red"
PAPER_INTEGRATION = False
STATUS = "preview"
CONFIG_BANDS = (
    (-0.40, 3.92, "#fff8f3", "#8b6c61", 1.76, "IVF-PQ 4×  ·  nlist=4,096"),
    (3.96, 5.72, "#fbece5", "#8b4d3b", 4.84, "IVF-PQ 32×  ·  nlist=65,536"),
)


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


def load_rows() -> dict[tuple[str, str], dict[str, float]]:
    if frozen_evidence():
        require_hash(SOURCE_CSV, SOURCE_CSV_SHA256, "combined Figure 7 table")
        require_hash(PERFORMANCE_SUMMARY, PERFORMANCE_SUMMARY_SHA256, "Figure 7 performance summary")
        require_hash(CACHE_SUMMARY, CACHE_SUMMARY_SHA256, "Figure 7 cache summary")

    rows: dict[tuple[str, str], dict[str, float]] = {}
    with SOURCE_CSV.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            key = (row["dataset"], row["method"])
            if key in rows:
                raise ValueError(f"duplicate Figure 7 row: {key}")
            rows[key] = {
                field: float(row[field])
                for field in (
                    "qps_mean",
                    "qps_sample_sd",
                    "p95_ms_mean",
                    "p95_ms_sample_sd",
                    "p99_ms_mean",
                    "p99_ms_sample_sd",
                    "get_per_q_mean",
                    "get_per_q_sample_sd",
                )
            }

    expected = {
        (workload, method)
        for workload, _, _ in WORKLOADS
        for method, _, _ in METHODS
    }
    if set(rows) != expected:
        raise ValueError(f"Figure 7 display matrix mismatch: {set(rows) ^ expected}")

    performance = json.loads(PERFORMANCE_SUMMARY.read_text(encoding="utf-8"))
    cache = json.loads(CACHE_SUMMARY.read_text(encoding="utf-8"))
    if performance.get("status") != "complete" or len(performance.get("cells", [])) != 63:
        raise ValueError("Figure 7 performance evidence is not the complete 63-cell campaign")
    if cache.get("status") != "complete" or cache.get("cells") != 27:
        raise ValueError("Figure 7 cache evidence is not the complete 27-cell campaign")
    # The compact performance summary predates the failure_violations field;
    # the cache summary includes it explicitly.
    if cache.get("failure_violations") != []:
        raise ValueError("Figure 7 cache evidence contains a validation failure")
    return rows


def propagated_ratio_error(
    numerator: float,
    numerator_sd: float,
    denominator: float,
    denominator_sd: float,
) -> float:
    ratio = numerator / denominator
    if numerator_sd == 0.0 and denominator_sd == 0.0:
        return 0.0
    return ratio * math.sqrt(
        (numerator_sd / numerator) ** 2 + (denominator_sd / denominator) ** 2
    )


def factor(
    rows: dict[tuple[str, str], dict[str, float]],
    workload: str,
    method: str,
    mean_key: str,
    sd_key: str,
    inverse: bool,
) -> tuple[float, float, str]:
    # Cache-on workloads use one fixed reference: Lance with cache off.
    reference_workload = "sift100m" if workload.startswith("sift100m_cache") else workload
    reference = rows[(reference_workload, "stock_lance")]
    candidate = rows[(workload, method)]

    # A row divided by itself is identically one, not an independent ratio.
    if workload == reference_workload and method == "stock_lance":
        return 1.0, 0.0, f"{reference_workload}:stock_lance"

    if inverse:
        value = reference[mean_key] / candidate[mean_key]
        error = propagated_ratio_error(
            reference[mean_key],
            reference[sd_key],
            candidate[mean_key],
            candidate[sd_key],
        )
    else:
        value = candidate[mean_key] / reference[mean_key]
        error = propagated_ratio_error(
            candidate[mean_key],
            candidate[sd_key],
            reference[mean_key],
            reference[sd_key],
        )
    return value, error, f"{reference_workload}:stock_lance"


def normalized_matrix(
    rows: dict[tuple[str, str], dict[str, float]],
) -> dict[tuple[str, str, str], tuple[float, float, str]]:
    matrix = {}
    for title, mean_key, sd_key, inverse, _, _ in PANELS:
        for workload, _, _ in WORKLOADS:
            for method, _, _ in METHODS:
                matrix[(title, workload, method)] = factor(
                    rows, workload, method, mean_key, sd_key, inverse
                )
    return matrix


def write_normalized_csv(
    path: Path,
    matrix: dict[tuple[str, str, str], tuple[float, float, str]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("panel", "workload", "method", "factor", "propagated_sd", "reference"))
        for title, *_ in PANELS:
            for workload, _, _ in WORKLOADS:
                for method, _, _ in METHODS:
                    value, error, reference = matrix[(title, workload, method)]
                    writer.writerow((title, workload, method, f"{value:.12g}", f"{error:.12g}", reference))


def render(
    matrix: dict[tuple[str, str, str], tuple[float, float, str]],
) -> None:
    apply_paper_preset(
        plt,
        "double_col",
        font_size=FONT_SIZE,
        axes_label_size=AXES_LABEL_SIZE,
        axes_title_size=AXES_TITLE_SIZE,
        legend_size=LEGEND_SIZE,
        xtick_size=XTICK_SIZE,
        ytick_size=YTICK_SIZE,
        axes_linewidth=0.72,
        extra_rc={
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.major.width": 0.58,
            "ytick.major.width": 0.58,
            "xtick.major.size": 2.3,
            "ytick.major.size": 2.3,
            "savefig.bbox": None,
            "svg.hashsalt": SVG_HASHSALT,
        },
    )

    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE, dpi=320, sharex=True)
    bar_width = BAR_WIDTH
    offsets = (np.arange(len(METHODS), dtype=float) - 1.0) * bar_width
    value_labels = []
    region_labels = []

    for axis, (title, _mean_key, _sd_key, _inverse, ylim, yticks) in zip(
        axes.flat, PANELS
    ):
        # Separate index configurations with thin rules instead of colored
        # background regions. The cache block already has extra horizontal
        # whitespace, so the two rules are sufficient visual structure.
        config_separator = (CONFIG_BANDS[0][1] + CONFIG_BANDS[1][0]) / 2.0
        axis.axvline(
            config_separator,
            color=CACHE_EDGE,
            linestyle=(0, (2, 2)),
            linewidth=0.62,
            alpha=0.9,
            zorder=1,
        )
        axis.axvline(
            CACHE_SEPARATOR,
            color=CACHE_EDGE,
            linestyle=(0, (2, 2)),
            linewidth=0.62,
            alpha=0.9,
            zorder=1,
        )
        for band_index, (
            band_left,
            band_right,
            band_fill,
            band_label_color,
            band_center,
            band_text,
        ) in enumerate(CONFIG_BANDS):
            scale_label = "1M" if band_index == 0 else "100M"
            index_label, nlist_label = band_text.split("  ·  ", maxsplit=1)
            region_labels.append(
                axis.text(
                    band_center,
                    0.965,
                    f"{scale_label} datasets\n{index_label}\n{nlist_label}",
                    ha="center",
                    va="top",
                    fontsize=REGION_LABEL_SIZE,
                    color=band_label_color,
                    transform=axis.get_xaxis_transform(),
                    zorder=4,
                )
            )
        region_labels.append(
            axis.text(
                CACHE_CENTER,
                0.965,
                "SIFT100M cache",
                ha="center",
                va="top",
                fontsize=CACHE_LABEL_SIZE,
                color=CACHE_LABEL,
                transform=axis.get_xaxis_transform(),
                zorder=4,
            )
        )
        for method_index, (method, method_label, color) in enumerate(METHODS):
            values = np.asarray(
                [matrix[(title, workload, method)][0] for workload, _, _ in WORKLOADS]
            )
            errors = np.asarray(
                [matrix[(title, workload, method)][1] for workload, _, _ in WORKLOADS]
            )
            bar_positions = POSITIONS + offsets[method_index]
            axis.bar(
                bar_positions,
                values,
                width=bar_width,
                yerr=errors,
                color=color,
                edgecolor=EDGE,
                linewidth=0.40,
                capsize=1.15,
                error_kw={"elinewidth": 0.52, "capthick": 0.52},
                label=method_label,
                zorder=3,
            )
            if method in {"stock_lance", "warp", "warp_reorder"}:
                label_offset = (ylim[1] - ylim[0]) * 0.026
                for (workload, _, workload_kind), xpos, value, error in zip(
                    WORKLOADS, bar_positions, values, errors
                ):
                    if method == "stock_lance" and workload_kind != "cache":
                        continue
                    annotation = axis.text(
                        xpos,
                        value + error + label_offset,
                        f"{value:.2f}×",
                        ha="center",
                        va="bottom",
                        rotation=90,
                        fontsize=VALUE_LABEL_SIZE,
                        fontweight="regular",
                        color=LABEL,
                        clip_on=False,
                        zorder=5,
                    )
                    value_labels.append(annotation)

        axis.axhline(
            1.0,
            color=REFERENCE,
            linestyle=(0, (4, 2)),
            linewidth=0.62,
            zorder=2,
        )
        axis.set_title(title, pad=2.1, fontweight="regular")
        axis.set_ylabel("Factor", labelpad=1.8)
        axis.set_ylim(*ylim)
        axis.set_yticks(yticks)
        axis.yaxis.set_major_formatter(
            FuncFormatter(lambda value, _position: f"{value:g}×")
        )
        axis.yaxis.grid(
            True,
            linestyle="--",
            dashes=(2, 2),
            alpha=0.52,
            linewidth=0.45,
            color=GRID,
        )
        axis.set_xlim(*X_LIMITS)
        axis.set_axisbelow(True)
        axis.tick_params(axis="both", length=2.3, width=0.58, direction="out")

    tick_labels = [label for _, label, _ in WORKLOADS]
    for axis in axes[1]:
        axis.set_xticks(POSITIONS)
        axis.set_xticklabels(tick_labels)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.992),
        ncol=3,
        frameon=True,
        handlelength=1.15,
        handletextpad=0.34,
        columnspacing=0.88,
        borderpad=0.24,
        labelspacing=0.18,
    )
    legend.get_frame().set_facecolor("white")
    legend.get_frame().set_edgecolor(LEGEND_EDGE)
    legend.get_frame().set_linewidth(0.45)

    fig.subplots_adjust(
        left=SUBPLOT_LEFT,
        right=SUBPLOT_RIGHT,
        bottom=BOTTOM_MARGIN,
        top=0.865,
        wspace=SUBPLOT_WSPACE,
        hspace=0.31,
    )
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    overlaps = []
    for axis in axes.flat:
        labels_in_axis = [label for label in value_labels if label.axes is axis]
        boxes = [label.get_window_extent(renderer).expanded(1.02, 1.04) for label in labels_in_axis]
        axis_box = axis.get_window_extent(renderer)
        for label, label_box in zip(labels_in_axis, boxes):
            if not axis_box.contains(label_box.x0, label_box.y0) or not axis_box.contains(
                label_box.x1, label_box.y1
            ):
                overlaps.append((axis.get_title(), "outside-axis", label.get_text()))
        for left_index, left_box in enumerate(boxes):
            for right_index in range(left_index + 1, len(boxes)):
                if left_box.overlaps(boxes[right_index]):
                    overlaps.append((axis.get_title(), left_index, right_index))
        region_labels_in_axis = [label for label in region_labels if label.axes is axis]
        region_boxes = [
            label.get_window_extent(renderer).expanded(1.01, 1.04)
            for label in region_labels_in_axis
        ]
        for label, label_box in zip(region_labels_in_axis, region_boxes):
            if not axis_box.contains(label_box.x0, label_box.y0) or not axis_box.contains(
                label_box.x1, label_box.y1
            ):
                overlaps.append((axis.get_title(), "region-outside-axis", label.get_text()))
            if any(label_box.overlaps(value_box) for value_box in boxes):
                overlaps.append((axis.get_title(), "region-vs-value", label.get_text()))
        for left_index, left_box in enumerate(region_boxes):
            for right_index in range(left_index + 1, len(region_boxes)):
                if left_box.overlaps(region_boxes[right_index]):
                    overlaps.append((axis.get_title(), "region-overlap", left_index, right_index))
    if overlaps:
        raise ValueError(f"value labels overlap: {overlaps}")

    pdf_path = OUT.with_suffix(".pdf")
    fig.savefig(
        pdf_path,
        bbox_inches=None,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(
        OUT.with_suffix(".svg"),
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


def main() -> None:
    rows = load_rows()
    matrix = normalized_matrix(rows)
    write_normalized_csv(OUT.with_suffix(".normalized.csv"), matrix)
    render(matrix)
    source_hashes = {
        "csv": sha256(SOURCE_CSV),
        "performance": sha256(PERFORMANCE_SUMMARY),
        "cache": sha256(CACHE_SUMMARY),
    }
    paper_hashes = {
        "csv": SOURCE_CSV_SHA256,
        "performance": PERFORMANCE_SUMMARY_SHA256,
        "cache": CACHE_SUMMARY_SHA256,
    }
    provenance = {
        "cache_display": "rightmost separated block containing 10% and 50% budgets",
        "cache_reference": "SIFT100M Lance with cache off",
        "campaigns": [
            "figure7-main",
            "figure7-cache",
        ],
        "evidence_slot": "paper" if frozen_evidence() else "latest",
        "matches_committed_paper_evidence": source_hashes == paper_hashes,
        "render_preserves_source_values": True,
        "deterministic_export": True,
        "error_bars": "first-order propagation of sample SD from three repeats",
        "get_label": "Number of GETs",
        "higher_is_better": True,
        "label_overlap_check": "passed",
        "method_colors": {label: color for _, label, color in METHODS},
        "bar_width_data_units": BAR_WIDTH,
        "subplot_bottom_fraction": BOTTOM_MARGIN,
        "subplot_left_fraction": SUBPLOT_LEFT,
        "subplot_right_fraction": SUBPLOT_RIGHT,
        "subplot_wspace_fraction": SUBPLOT_WSPACE,
        "typography_points": {
            "base": FONT_SIZE,
            "axes_label": AXES_LABEL_SIZE,
            "axes_title": AXES_TITLE_SIZE,
            "legend": LEGEND_SIZE,
            "x_tick": XTICK_SIZE,
            "y_tick": YTICK_SIZE,
            "region_label": REGION_LABEL_SIZE,
            "cache_label": CACHE_LABEL_SIZE,
            "value_label": VALUE_LABEL_SIZE,
        },
        "group_centers_data_units": POSITIONS.tolist(),
        "palette_family": PALETTE_FAMILY,
        "normalization": {
            "cache_groups": "all methods and cache budgets use cache-off Lance",
            "dataset_groups": "each dataset uses its own Lance",
            "latency_and_gets": "reference Lance divided by method",
            "qps": "method divided by reference Lance",
        },
        "index_configuration_bands": [
            {
                "label": band_text,
                "x_extent": [band_left, band_right],
            }
            for band_left, band_right, _, _, _, band_text in CONFIG_BANDS
        ],
        "index_configuration_annotation": "centered direct labels separated by thin vertical rules",
        "index_configuration_background": "none",
        "paper_integration": PAPER_INTEGRATION if frozen_evidence() else False,
        "panel_y_limits": {title: list(ylim) for title, _, _, _, ylim, _ in PANELS},
        "physical_size_inches": list(FIGSIZE),
        "png_export": "320 dpi deterministic rasterization from the final PDF",
        "repeats": 3,
        "source_csv": str(SOURCE_CSV.relative_to(ROOT)),
        "source_csv_sha256": source_hashes["csv"],
        "source_performance_summary": str(PERFORMANCE_SUMMARY.relative_to(ROOT)),
        "source_performance_summary_sha256": source_hashes["performance"],
        "source_cache_summary": str(CACHE_SUMMARY.relative_to(ROOT)),
        "source_cache_summary_sha256": source_hashes["cache"],
        "status": STATUS if frozen_evidence() else "fresh-run-render",
        "value_labels": "WaRP and WaRP + Reorder everywhere; Lance in cache groups; vertical",
        "workload_order": [workload for workload, _, _ in WORKLOADS],
    }
    OUT.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
