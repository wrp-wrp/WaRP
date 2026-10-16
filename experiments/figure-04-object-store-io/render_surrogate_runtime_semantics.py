#!/usr/bin/env python3
"""Refit and render the F-004 surrogate-generality figure.

The convex model matches the accelerated runtime semantics:

    L(s) = O + r1 * s + delta * max(s - B, 0)

where O, r1, and delta are non-negative. The second slope is therefore
r1 + delta >= r1, which makes the two-segment model continuous, monotone, and
convex. The general model is continuous and monotone, but its three segment
slopes need not increase. Alternating request sizes form the fitting and
held-out sets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import numpy as np
from scipy.optimize import lsq_linear


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.plotting.bar_palette import get_bar_palette  # noqa: E402
from figure_layout import (  # noqa: E402
    AXES_BOTTOM,
    AXES_HSPACE,
    AXES_TOP,
    FIGURE_HEIGHT_IN,
    RIGHT_WIDTH_IN,
    X_LABEL_Y,
)
from src.plotting.paper_plot_style import apply_paper_preset  # noqa: E402
from artifact.data import paper_data_path  # noqa: E402


MIB = 1024 * 1024
DATA_DIR = paper_data_path(HERE / "data" / "paper" / "runtime")
OUTPUT_STEM = HERE / "surrogate_runtime_semantics"
SUMMARY_PATH = HERE / "surrogate_runtime_semantics.provenance.json"

BLUE_PALETTE = get_bar_palette("blue", 5)
RED_PALETTE = get_bar_palette("red", 7)
OBSERVED_EDGE = "#59636e"
OBSERVED_FACE = "#ffffff"
MONOTONE_COLOR = BLUE_PALETTE[-2]
PIECEWISE_COLOR = RED_PALETTE[4]
PIECEWISE_EDGE = RED_PALETTE[5]

TOPOLOGIES = [
    ("ap_same_region_serial_stats.json", "AP Same-Region"),
    ("us_same_region_serial_stats.json", "US Same-Region"),
    ("cross_region_serial_stats.json", "US to AP Cross-Region"),
    ("public_internet_serial_stats.json", "Public Internet to AP"),
]


def load_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    x = np.asarray([float(row["size_bytes"]) / MIB for row in rows], dtype=float)
    y = np.asarray([float(row["p50_ms"]) for row in rows], dtype=float)
    order = np.argsort(x)
    return x[order], y[order]


def design_linear(x: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones_like(x), x])


def design_piecewise(x: np.ndarray, breakpoint_mib: float) -> np.ndarray:
    return np.column_stack(
        [np.ones_like(x), x, np.maximum(x - breakpoint_mib, 0.0)]
    )


def weighted_fit(design: np.ndarray, y: np.ndarray, x: np.ndarray) -> np.ndarray:
    weights = np.sqrt(x)
    result = lsq_linear(
        design * weights[:, None],
        y * weights,
        bounds=(np.zeros(design.shape[1]), np.full(design.shape[1], np.inf)),
        lsmr_tol="auto",
    )
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def fit_linear(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    coefficients = weighted_fit(design_linear(x), y, x)
    return {"overhead_ms": float(coefficients[0]), "slope_ms_per_mib": float(coefficients[1])}


def predict_linear(model: dict[str, float], x: np.ndarray) -> np.ndarray:
    return model["overhead_ms"] + model["slope_ms_per_mib"] * x


def fit_piecewise(x: np.ndarray, y: np.ndarray) -> dict[str, float]:
    weights = np.sqrt(x)
    best: dict[str, float] | None = None
    for breakpoint_mib in np.arange(0.5, 32.0 + 1e-12, 0.25):
        design = design_piecewise(x, float(breakpoint_mib))
        coefficients = weighted_fit(design, y, x)
        prediction = design @ coefficients
        weighted_sse = float(np.sum((weights * (y - prediction)) ** 2))
        candidate = {
            "breakpoint_mib": float(breakpoint_mib),
            "overhead_ms": float(coefficients[0]),
            "first_slope_ms_per_mib": float(coefficients[1]),
            "second_slope_ms_per_mib": float(coefficients[1] + coefficients[2]),
            "weighted_sse": weighted_sse,
        }
        if best is None or weighted_sse < best["weighted_sse"]:
            best = candidate
    if best is None:
        raise RuntimeError("no constrained two-segment fit found")
    return best


def predict_piecewise(model: dict[str, float], x: np.ndarray) -> np.ndarray:
    breakpoint = model["breakpoint_mib"]
    first_slope = model["first_slope_ms_per_mib"]
    second_slope = model["second_slope_ms_per_mib"]
    return (
        model["overhead_ms"]
        + first_slope * np.minimum(x, breakpoint)
        + second_slope * np.maximum(x - breakpoint, 0.0)
    )


def design_monotone_three_segment(
    x: np.ndarray, first_breakpoint: float, second_breakpoint: float
) -> np.ndarray:
    return np.column_stack(
        [
            np.ones_like(x),
            np.minimum(x, first_breakpoint),
            np.clip(
                x - first_breakpoint,
                0.0,
                second_breakpoint - first_breakpoint,
            ),
            np.maximum(x - second_breakpoint, 0.0),
        ]
    )


def fit_monotone_three_segment(x: np.ndarray, y: np.ndarray) -> dict[str, object]:
    weights = np.sqrt(x)
    breakpoint_candidates = x[1:-1]
    best: dict[str, object] | None = None

    for first_index, first_breakpoint in enumerate(breakpoint_candidates[:-1]):
        for second_breakpoint in breakpoint_candidates[first_index + 1 :]:
            design = design_monotone_three_segment(
                x, float(first_breakpoint), float(second_breakpoint)
            )
            coefficients = weighted_fit(design, y, x)
            weighted_sse = float(
                np.sum((weights * (y - design @ coefficients)) ** 2)
            )
            candidate: dict[str, object] = {
                "first_breakpoint_mib": float(first_breakpoint),
                "second_breakpoint_mib": float(second_breakpoint),
                "overhead_ms": float(coefficients[0]),
                "segment_slopes_ms_per_mib": [
                    float(value) for value in coefficients[1:]
                ],
                "weighted_sse": weighted_sse,
            }
            if best is None or weighted_sse < float(best["weighted_sse"]):
                best = candidate

    if best is None:
        raise RuntimeError("no monotone three-segment fit found")
    return best


def predict_monotone_three_segment(
    model: dict[str, object], x: np.ndarray
) -> np.ndarray:
    design = design_monotone_three_segment(
        x,
        float(model["first_breakpoint_mib"]),
        float(model["second_breakpoint_mib"]),
    )
    coefficients = np.asarray(
        [model["overhead_ms"], *model["segment_slopes_ms_per_mib"]],
        dtype=float,
    )
    return design @ coefficients


def metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residual = observed - predicted
    return {
        "rmse_ms": float(np.sqrt(np.mean(residual**2))),
        "mape_percent": float(
            np.mean(np.abs(residual) / np.maximum(observed, 1e-9)) * 100.0
        ),
    }


def fit_topology(path: Path) -> dict[str, object]:
    x, y = load_points(path)
    train_mask = np.arange(x.size) % 2 == 0
    heldout_mask = ~train_mask

    linear = fit_linear(x[train_mask], y[train_mask])
    piecewise = fit_piecewise(x[train_mask], y[train_mask])
    monotone = fit_monotone_three_segment(x[train_mask], y[train_mask])
    linear_prediction = predict_linear(linear, x)
    piecewise_prediction = predict_piecewise(piecewise, x)
    monotone_prediction = predict_monotone_three_segment(monotone, x)

    return {
        "source": str(path.relative_to(ROOT)),
        "request_sizes_mib": x.tolist(),
        "p50_ms": y.tolist(),
        "fit_indices": np.flatnonzero(train_mask).astype(int).tolist(),
        "heldout_indices": np.flatnonzero(heldout_mask).astype(int).tolist(),
        "linear": {
            "model": linear,
            "heldout_metrics": metrics(y[heldout_mask], linear_prediction[heldout_mask]),
        },
        "piecewise_continuous_convex": {
            "model": piecewise,
            "heldout_metrics": metrics(
                y[heldout_mask], piecewise_prediction[heldout_mask]
            ),
        },
        "piecewise_continuous_monotone": {
            "model": monotone,
            "heldout_metrics": metrics(
                y[heldout_mask], monotone_prediction[heldout_mask]
            ),
        },
    }


def main() -> None:
    apply_paper_preset(
        plt,
        "single_col",
        font_size=5.8,
        axes_label_size=6.2,
        axes_title_size=5.4,
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

    summaries: dict[str, object] = {
        "model_semantics": (
            "continuous monotone convex two-segment and continuous monotone "
            "three-segment; sqrt(size) WLS"
        ),
        "validation": "alternating request sizes; fits use even-indexed sizes",
        "topologies": {},
    }
    topology_results: list[tuple[str, dict[str, object]]] = []
    for filename, title in TOPOLOGIES:
        source = DATA_DIR / filename
        summary = fit_topology(source)
        summaries["topologies"][title] = summary
        topology_results.append((title, summary))

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(RIGHT_WIDTH_IN, FIGURE_HEIGHT_IN),
        dpi=640,
    )
    dense_x = np.geomspace(0.5, 32.0, 320)
    panel_labels = ["(a)", "(b)", "(c)", "(d)"]

    for axis, panel_label, (title, summary) in zip(
        axes.flat, panel_labels, topology_results
    ):
        x = np.asarray(summary["request_sizes_mib"], dtype=float)
        y = np.asarray(summary["p50_ms"], dtype=float)
        piecewise = summary["piecewise_continuous_convex"]["model"]
        monotone = summary["piecewise_continuous_monotone"]["model"]
        piecewise_mape = float(
            summary["piecewise_continuous_convex"]["heldout_metrics"][
                "mape_percent"
            ]
        )
        monotone_mape = float(
            summary["piecewise_continuous_monotone"]["heldout_metrics"][
                "mape_percent"
            ]
        )

        axis.plot(
            dense_x,
            predict_monotone_three_segment(monotone, dense_x),
            color=MONOTONE_COLOR,
            linestyle="--",
            linewidth=0.64,
            zorder=2,
        )
        axis.plot(
            dense_x,
            predict_piecewise(piecewise, dense_x),
            color=PIECEWISE_COLOR,
            linestyle="--",
            linewidth=0.74,
            zorder=3,
        )
        axis.scatter(
            x,
            y,
            s=4.0,
            facecolor=OBSERVED_FACE,
            edgecolor=OBSERVED_EDGE,
            linewidth=0.48,
            zorder=5,
        )
        axis.text(
            0.045,
            0.955,
            "MAPE",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=4.3,
            color="#4e555c",
        )
        axis.text(
            0.045,
            0.815,
            f"Conv. {piecewise_mape:.1f}%",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=4.3,
            color=PIECEWISE_EDGE,
        )
        axis.text(
            0.045,
            0.685,
            f"Mono. {monotone_mape:.1f}%",
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=4.3,
            color=MONOTONE_COLOR,
        )
        axis.set_title(title, pad=1.1)
        axis.set_xscale("log", base=2)
        axis.set_xlim(0.43, 37.0)
        axis.set_xticks([0.5, 4.0, 32.0])
        axis.set_xticklabels(["0.5", "4", "32"])
        axis.set_ylim(bottom=0.0, top=float(np.max(y)) * 1.13)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=3))
        axis.grid(
            axis="both",
            linestyle="--",
            dashes=(2.0, 2.0),
            alpha=0.55,
            zorder=0,
        )
        axis.set_axisbelow(True)

    handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="None",
            markersize=2.0,
            markerfacecolor=OBSERVED_FACE,
            markeredgecolor=OBSERVED_EDGE,
            markeredgewidth=0.55,
            label="Measured",
        ),
        Line2D([], [], color=PIECEWISE_COLOR, linestyle="--", linewidth=0.74, label="Convex"),
        Line2D([], [], color=MONOTONE_COLOR, linestyle="--", linewidth=0.64, label="Monotone"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=3,
        frameon=False,
        borderpad=0.0,
        handlelength=1.05,
        handletextpad=0.22,
        columnspacing=0.42,
    )
    fig.supxlabel("Range size (MiB)", x=0.59, y=X_LABEL_Y, fontsize=6.2)
    fig.supylabel("p50 latency (ms)", x=0.060, y=0.51, fontsize=6.2)
    fig.subplots_adjust(
        top=AXES_TOP,
        bottom=AXES_BOTTOM,
        left=0.22,
        right=0.99,
        hspace=AXES_HSPACE,
        wspace=0.55,
    )

    fig.savefig(
        OUTPUT_STEM.with_suffix(".pdf"),
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(OUTPUT_STEM.with_suffix(".png"))
    plt.close(fig)

    SUMMARY_PATH.write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
