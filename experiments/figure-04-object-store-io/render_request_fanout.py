#!/usr/bin/env python3
"""Render Figure 3a at its final one-third-column dimensions."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plotting.paper_plot_style import apply_paper_preset  # noqa: E402
from src.plotting.bar_palette import get_bar_palette  # noqa: E402
from artifact.data import paper_data_path  # noqa: E402
from figure_layout import (  # noqa: E402
    AXES_BOTTOM,
    AXES_HSPACE,
    AXES_TOP,
    FIGURE_HEIGHT_IN,
    LEFT_WIDTH_IN,
    X_LABEL_Y,
)


DATA_PATH = paper_data_path(HERE / "data" / "paper" / "request_fanout.csv")
OUTPUT_STEM = HERE / "request_fanout"

BLUE_PALETTE = get_bar_palette("blue", 5)
RED_PALETTE = get_bar_palette("red", 7)
AVERAGE_COLOR = BLUE_PALETTE[-2]
P95_COLOR = RED_PALETTE[4]


def main() -> None:
    with DATA_PATH.open(encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    fanout = np.asarray([int(row["fanout"]) for row in rows], dtype=float)
    average = np.asarray(
        [float(row["avg_latency_ms"]) for row in rows], dtype=float
    )
    p95 = np.asarray([float(row["p95_latency_ms"]) for row in rows], dtype=float)

    apply_paper_preset(
        plt,
        "single_col",
        font_size=5.8,
        axes_label_size=6.2,
        axes_title_size=5.8,
        legend_size=5.3,
        xtick_size=5.2,
        ytick_size=5.2,
        axes_linewidth=0.58,
        grid_color="#cfd4da",
        grid_alpha=0.30,
        grid_linewidth=0.42,
        extra_rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.major.width": 0.50,
            "ytick.major.width": 0.50,
            "xtick.major.size": 1.7,
            "ytick.major.size": 1.7,
            "savefig.bbox": None,
        },
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(LEFT_WIDTH_IN, FIGURE_HEIGHT_IN),
        dpi=640,
        sharex=True,
    )
    series = (
        (axes[0], average, AVERAGE_COLOR, "-", "Average"),
        (axes[1], p95, P95_COLOR, "--", "p95"),
    )
    for axis, latency, color, line_style, label in series:
        axis.plot(
            fanout,
            latency,
            color=color,
            linewidth=0.78,
            linestyle=line_style,
            zorder=3,
        )
        axis.text(
            0.055,
            0.90,
            label,
            transform=axis.transAxes,
            color=color,
            ha="left",
            va="top",
            fontsize=5.8,
        )
        axis.set_xlim(0.0, 132.0)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
        axis.grid(
            axis="both",
            linestyle="--",
            dashes=(2.0, 2.0),
            alpha=0.55,
            zorder=0,
        )
        axis.set_axisbelow(True)

    axes[0].set_ylim(20.0, 53.0)
    axes[1].set_ylim(25.0, 91.0)
    axes[1].set_xticks([1.0, 64.0, 128.0])
    fig.supxlabel("Requests/query", x=0.62, y=X_LABEL_Y, fontsize=6.2)
    fig.supylabel("Latency (ms)", x=0.055, y=0.52, fontsize=6.2)
    fig.subplots_adjust(
        top=AXES_TOP,
        bottom=AXES_BOTTOM,
        left=0.29,
        right=0.96,
        hspace=AXES_HSPACE,
    )

    fig.savefig(
        OUTPUT_STEM.with_suffix(".pdf"),
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(OUTPUT_STEM.with_suffix(".png"))
    plt.close(fig)


if __name__ == "__main__":
    main()
