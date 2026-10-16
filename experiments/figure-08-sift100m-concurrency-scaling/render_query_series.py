#!/usr/bin/env python3
"""Render the completed Figure 8 query-concurrency ladder."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.plotting.paper_plot_style import apply_paper_preset
from artifact.data import frozen_evidence, paper_data_path


DATA = paper_data_path(HERE / "data" / "paper" / "query_summary.json")
DATA_SHA256 = "33aae9f976ce782c9d0168d0cb8e73bb54f9cfda0ad75fdb642404576830028b"
OUT = Path(__file__).resolve().parent / "01_query_concurrency_p64_io2048_paper_style"
FIGSIZE = (3.35, 1.33)

Q_LEVELS = (16, 32, 48, 64, 96, 128)
METHODS = (
    ("Lance", "stock_lance", "#d7301f", "s", (0, (4, 2))),
    ("WaRP", "warp", "#9ecae1", "^", (0, (2, 1))),
    ("WaRP + Reorder", "warp_reorder", "#3182bd", "o", "-"),
)
PANELS = (
    ("(a) Throughput", "QPS", "qps", (0, 165), (0, 40, 80, 120, 160)),
    (
        "(b) p95 end-to-end latency",
        "Latency (ms)",
        "p95_ms",
        (0, 1650),
        (0, 400, 800, 1200, 1600),
    ),
)

EDGE = "#1a1a1a"
GRID = "#adb5bd"


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


def load_summary() -> dict:
    if frozen_evidence():
        require_hash(DATA, DATA_SHA256, "Figure 8 query summary")
    data = json.loads(DATA.read_text(encoding="utf-8"))
    if data.get("schema_version") != 1 or data.get("status") != "complete":
        raise ValueError("Figure 8 query summary is not a complete schema-v1 result")
    if data.get("campaign") != "figure8-query-concurrency":
        raise ValueError("Figure 8 query summary has the wrong campaign identity")
    contract = data.get("measurement_contract")
    if not isinstance(contract, dict):
        raise ValueError("Figure 8 query summary is missing its measurement contract")
    expected_contract = {
        "dataset": "sift100m",
        "index": "IVF65536-PQ16",
        "nlist": 65536,
        "num_sub_vectors": 16,
        "bits_per_subcode": 8,
        "nprobe": 256,
        "refine_factor": 8,
        "k": 10,
        "methods": [method for _, method, _, _, _ in METHODS],
        "query_concurrency_levels": list(Q_LEVELS),
        "io_limit": 2048,
        "num_queries": 10_000,
        "warmup_queries": 200,
        "warmup_reused": True,
        "repeats": 3,
        "counterbalanced": True,
        "warp_planner_p": 64,
        "request_budget": "unbounded",
        "max_merge_span_bytes": 524_288,
    }
    changed = [
        key
        for key, expected in expected_contract.items()
        if contract.get(key) != expected
    ]
    if changed:
        raise ValueError(
            "Figure 8 query measurement contract mismatch: " + ", ".join(changed)
        )
    for key in ("query_sha256", "binary_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(contract.get(key, ""))):
            raise ValueError(f"Figure 8 query contract has an invalid {key}")
    source_hashes = contract.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError("Figure 8 query contract is missing source hashes")
    if any(
        not re.fullmatch(r"[0-9a-f]{64}", str(value))
        for value in source_hashes.values()
    ):
        raise ValueError("Figure 8 query contract has an invalid source hash")
    if frozen_evidence():
        if not re.fullmatch(
            r"[0-9a-f]{64}", str(contract.get("source_archive_sha256", ""))
        ):
            raise ValueError("Figure 8 frozen query contract is missing its source archive")
    else:
        fresh_identity = {
            "lance_commit": r"[0-9a-f]{40}",
            "binary_build_receipt_sha256": r"[0-9a-f]{64}",
            "cargo_lock_sha256": r"[0-9a-f]{64}",
        }
        for key, pattern in fresh_identity.items():
            if not re.fullmatch(pattern, str(contract.get(key, ""))):
                raise ValueError(f"Figure 8 fresh query contract has an invalid {key}")
        identities = contract.get("index_identities")
        if not isinstance(identities, dict) or set(identities) != {"random", "reorder"}:
            raise ValueError("Figure 8 fresh query contract has incomplete index identities")
    if len(data.get("cells", [])) != 54 or len(data.get("aggregate", [])) != 18:
        raise ValueError("Figure 8 query summary does not contain the expected 54 cells and 18 groups")

    expected_groups = {
        (method, q, 2048)
        for _, method, _, _, _ in METHODS
        for q in Q_LEVELS
    }
    actual_groups = {
        (row["method"], int(row["q"]), int(row["io_cap"]))
        for row in data["aggregate"]
    }
    if actual_groups != expected_groups:
        raise ValueError("Figure 8 method-Q matrix is incomplete or contains extra groups")
    for row in data["aggregate"]:
        if row["dataset"] != "sift100m" or row["repeats"] != 3:
            raise ValueError("unexpected Figure 8 aggregate contract")
        for metric in ("qps", "p95_ms"):
            values = row["metrics"][metric]
            if len(values["values"]) != 3:
                raise ValueError(f"{metric} does not contain three repeats")

    expected_cells = {
        (method, q, repeat)
        for _, method, _, _, _ in METHODS
        for q in Q_LEVELS
        for repeat in (1, 2, 3)
    }
    actual_cells = set()
    for cell in data["cells"]:
        meta = cell["meta"]
        method = meta["method"]
        q = int(meta["query_concurrency"])
        repeat = int(meta["repeat"])
        actual_cells.add((method, q, repeat))
        common = {
            "dataset": "sift100m",
            "experiment": "figure8-query-concurrency",
            "index": "IVF65536-PQ16",
            "compression": "32x",
            "io_limits": "2048/2048",
            "nprobe": "256",
            "refine_factor": "8",
            "k": "10",
            "num_queries": "10000",
            "warmup_queries": "200",
        }
        if any(meta.get(key) != value for key, value in common.items()):
            raise ValueError(f"unexpected contract in cell {cell['cell']}")
        for meta_key, contract_key in (
            ("query_sha256", "query_sha256"),
            ("binary_sha256", "binary_sha256"),
        ):
            if meta.get(meta_key) != contract[contract_key]:
                raise ValueError(
                    f"{meta_key} differs from the measurement contract in {cell['cell']}"
                )
        if method == "stock_lance":
            if meta["planner_p"] != "na" or meta["layout"] != "random":
                raise ValueError("unexpected Lance contract")
        else:
            expected_layout = "random" if method == "warp" else "pca_nd_zorder"
            if (
                meta["planner_p"] != "64"
                or meta["request_budget"] != "unbounded"
                or meta["layout"] != expected_layout
            ):
                raise ValueError(f"unexpected WaRP contract in cell {cell['cell']}")
    if actual_cells != expected_cells:
        raise ValueError("Figure 8 cell schedule is incomplete or duplicated")
    return data


def aggregate_matrix(data: dict) -> dict[tuple[str, int], dict]:
    return {
        (row["method"], int(row["q"])): row
        for row in data["aggregate"]
    }


def main() -> None:
    data = load_summary()
    matrix = aggregate_matrix(data)
    apply_paper_preset(
        plt,
        "single_col",
        font_size=7.1,
        axes_label_size=7.1,
        axes_title_size=7.1,
        legend_size=5.0,
        xtick_size=5.7,
        ytick_size=5.7,
        axes_linewidth=0.75,
        extra_rc={
            "axes.spines.top": True,
            "axes.spines.right": True,
            "xtick.major.width": 0.62,
            "ytick.major.width": 0.62,
            "xtick.major.size": 2.4,
            "ytick.major.size": 2.4,
            "savefig.bbox": None,
            "svg.hashsalt": "warp-figure8-query-concurrency-paper-style",
        },
    )

    fig, axes = plt.subplots(1, 2, figsize=FIGSIZE, dpi=320)
    x = np.asarray(Q_LEVELS, dtype=float)
    for axis, (title, ylabel, metric, ylim, yticks) in zip(axes, PANELS):
        for label, method, color, marker, linestyle in METHODS:
            groups = [matrix[(method, q)]["metrics"][metric] for q in Q_LEVELS]
            means = np.asarray([group["mean"] for group in groups], dtype=float)
            axis.plot(
                x,
                means,
                label=label,
                color=color,
                linestyle=linestyle,
                linewidth=0.82,
                marker=marker,
                markersize=3.25,
                markerfacecolor=color,
                markeredgecolor=EDGE,
                markeredgewidth=0.38,
                zorder=3,
            )

        axis.set_title(title, pad=2.8, fontweight="regular")
        axis.set_ylabel(ylabel, labelpad=2.0)
        axis.set_xticks(Q_LEVELS)
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
            fontsize=4.7,
            handlelength=1.18,
            handletextpad=0.24,
            borderpad=0.20,
            labelspacing=0.12,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#333333")
        legend.get_frame().set_linewidth(0.45)

    fig.supxlabel("Query Concurrency ($Q$)", y=0.012, fontsize=6.5)
    fig.subplots_adjust(left=0.12, right=0.995, bottom=0.215, top=0.86, wspace=0.34)
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

    provenance = {
        "campaign_id": "figure8-query-concurrency",
        "evidence_slot": "paper" if frozen_evidence() else "latest",
        "matches_committed_paper_evidence": sha256(DATA) == DATA_SHA256,
        "render_preserves_source_values": True,
        "deterministic_export": True,
        "error_bars": False,
        "method_colors": {label: color for label, _, color, _, _ in METHODS},
        "paper_integration": frozen_evidence(),
        "paper_caption_visible": True,
        "physical_size_inches": list(FIGSIZE),
        "query_concurrency_levels": list(Q_LEVELS),
        "queries_per_cell": 10_000,
        "warmup_queries": 200,
        "refine_factor": 8,
        "png_export": "320 dpi deterministic rasterization from the final PDF",
        "repeats": 3,
        "request_parallelism_estimate": 64,
        "runtime_io_concurrency_limit": 2048,
        "source_summary": str(DATA.relative_to(ROOT)),
        "source_summary_sha256": sha256(DATA),
        "status": (
            "paper-frozen"
            if frozen_evidence()
            else "fresh-run-render"
        ),
        "variability_retained_in_source_summary": True,
    }
    OUT.with_suffix(".provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
