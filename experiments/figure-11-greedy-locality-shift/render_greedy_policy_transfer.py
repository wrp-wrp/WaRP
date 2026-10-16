#!/usr/bin/env python3
"""Render the compact single-column three-metric Figure 11 artifact."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.plotting.paper_plot_style import apply_paper_preset, style_legend_frame
from artifact.data import frozen_evidence, paper_data_path
from load_evidence import load_sift


OUT = HERE / "figure_greedy_policy_transfer"
MSONG_EVIDENCE = paper_data_path(HERE / "data" / "paper" / "msong_summary.json")
MSONG_EVIDENCE_SHA256 = "039e52a7271bcf41bf3973dfbf86584c9f7715e76de019785afae43f80132b35"
GREEDY_COLOR = "#74a9cf"
GREEDY_EDGE = "#2f3f53"
WARP_COLOR = "#0570b0"
WARP_EDGE = "#23384f"
GRID = "#adb5bd"
ROWS = [
    ("Validation", "tight", 32),
    ("Locality", "loose", 32),
    ("Probe-count", "tight", 256),
    ("Joint", "loose", 256),
    ("Dataset\n(MSong)", None, None),
]
METRICS = [
    ("QPS $\\uparrow$", "median_qps", (45, 122), [60, 120]),
    ("p95 (ms) $\\downarrow$", "median_p95_latency_ms", (45, 132), [60, 120]),
    ("GETs per query $\\downarrow$", "median_avg_num_requests", (0, 48), [0, 40]),
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_msong() -> dict:
    if not MSONG_EVIDENCE.is_file() or (
        frozen_evidence() and sha256(MSONG_EVIDENCE) != MSONG_EVIDENCE_SHA256
    ):
        raise ValueError("Figure 11 paper evidence is missing or has the wrong checksum")
    data = json.loads(MSONG_EVIDENCE.read_text(encoding="utf-8"))
    if data.get("protocol_id") != "figure11-msong-transfer-v1":
        raise ValueError("unexpected Figure 11 protocol")
    valid_status = "complete-paper-evidence" if frozen_evidence() else "complete"
    if data.get("status") != valid_status or data.get("failure_violations") != []:
        raise ValueError("Figure 11 evidence did not pass its gates")
    return data


def values(sift: dict, msong: dict, metric: str) -> tuple[np.ndarray, np.ndarray]:
    greedy = []
    warp = []
    for _, population, nprobe in ROWS[:-1]:
        cell = sift[(population, nprobe)]
        greedy.append(float(cell["greedy_frozen_on_tight_nprobe32"][metric]))
        warp.append(float(cell["warp_p32_k128"][metric]))
    greedy.append(float(msong["arms"]["sift_frozen_greedy"][metric]))
    warp.append(float(msong["arms"]["warp_p32_k128"][metric]))
    return np.asarray(greedy), np.asarray(warp)


def main() -> None:
    sift = load_sift()
    msong = load_msong()
    apply_paper_preset(
        plt,
        "single_col",
        font_size=6.3,
        axes_title_size=6.4,
        legend_size=5.8,
        xtick_size=5.3,
        ytick_size=5.4,
        axes_linewidth=0.60,
        extra_rc={
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.02,
        },
    )

    fig, axes = plt.subplots(1, 3, figsize=(3.35, 1.24), dpi=320)
    y = np.arange(len(ROWS) - 1, -1, -1, dtype=float)
    for panel_index, (axis, (title, metric, xlim, xticks)) in enumerate(zip(axes, METRICS)):
        greedy, warp = values(sift, msong, metric)
        axis.axhspan(-0.38, 0.38, color="#fff1e8", linewidth=0, zorder=0)
        axis.axhspan(0.62, 1.38, color="#f1f5f8", linewidth=0, zorder=0)
        for row, greedy_value, warp_value in zip(y, greedy, warp):
            axis.plot(
                [greedy_value, warp_value],
                [row, row],
                color="#96a5b3",
                linewidth=0.72,
                solid_capstyle="round",
                zorder=2,
            )
        axis.scatter(
            greedy,
            y,
            s=13,
            color=GREEDY_COLOR,
            edgecolor=GREEDY_EDGE,
            linewidth=0.36,
            zorder=3,
            label="SIFT-tuned Greedy",
        )
        axis.scatter(
            warp,
            y,
            s=13,
            color=WARP_COLOR,
            edgecolor=WARP_EDGE,
            linewidth=0.36,
            zorder=3,
            label="WaRP",
        )
        axis.set_title(title, pad=2.2, fontsize=6.2, weight="normal")
        axis.set_xlim(*xlim)
        axis.set_xticks(xticks)
        axis.set_ylim(-0.48, len(ROWS) - 0.52)
        axis.set_yticks(y)
        if panel_index == 0:
            axis.set_yticklabels([label for label, _, _ in ROWS])
        else:
            axis.set_yticklabels([])
        axis.xaxis.grid(
            True,
            linestyle="--",
            dashes=(2, 2),
            alpha=0.48,
            linewidth=0.38,
            color=GRID,
        )
        axis.set_axisbelow(True)
        axis.tick_params(axis="x", length=1.6, width=0.45, direction="out", pad=1.4)
        axis.tick_params(axis="y", length=0, pad=1.8)
        axis.spines["left"].set_visible(False)
        axis.spines["bottom"].set_linewidth(0.55)

    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.61, 0.99),
        ncol=2,
        frameon=True,
        fancybox=True,
        framealpha=1.0,
        facecolor="white",
        handlelength=0.45,
        handletextpad=0.22,
        columnspacing=0.55,
        borderpad=0.24,
    )
    style_legend_frame(legend)
    fig.subplots_adjust(left=0.30, right=0.995, bottom=0.12, top=0.72, wspace=0.24)
    fig.savefig(OUT.with_suffix(".pdf"))
    fig.savefig(OUT.with_suffix(".png"), dpi=320)
    plt.close(fig)
    evidence_slot = "paper" if frozen_evidence() else "latest"
    sift_evidence = paper_data_path(HERE / "data" / "paper" / "sift_summary.json")
    provenance = {
        "status": (
            "paper-frozen"
            if frozen_evidence()
            else "fresh-run-render"
        ),
        "evidence_slot": evidence_slot,
        "figure": 11,
        "chart_type": "single-column aligned dumbbell panels",
        "physical_size_inches": [3.35, 1.24],
        "metrics": [metric for _, metric, _, _ in METRICS],
        "methods": ["SIFT-tuned Greedy", "WaRP"],
        "rows": [label.replace("\n", " ") for label, _, _ in ROWS],
        "sift_evidence": str(sift_evidence),
        "sift_evidence_sha256": sha256(sift_evidence),
        "msong_evidence": str(MSONG_EVIDENCE),
        "msong_evidence_sha256": sha256(MSONG_EVIDENCE),
        "msong_source_archive_sha256": msong["source_archive_sha256"],
        "msong_failure_violations": msong["failure_violations"],
    }
    OUT.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
