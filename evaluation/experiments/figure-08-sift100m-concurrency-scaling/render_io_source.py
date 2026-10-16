#!/usr/bin/env python3
"""Render the Figure 8 Q=32 I/O-limit panel from normalized summaries."""

from __future__ import annotations

import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, FixedLocator


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.plotting.paper_plot_style import apply_paper_preset
from artifact.data import data_slot, paper_data_path


SUMMARY_ROOT = paper_data_path(HERE / "data" / "paper" / "io_summaries")
OUT = Path(__file__).resolve().parent / "01_figure8_q32_io_limit"

BLUE = "#2b8cbe"
ORANGE = "#d95f0e"
BROWN = "#8c510a"
EDGE = "#1a1a1a"
GRID = "#adb5bd"
FILL = "#d8e5ef"
LABEL = "#222222"


@dataclass(frozen=True)
class BundleSpec:
    filename: str
    sha256: str
    expected_campaign: str
    expected_limits: tuple[int, ...]
    expected_process_limits: tuple[int, ...]


BUNDLES = {
    "ladder": BundleSpec(
        filename="ladder.json",
        sha256="903642c3f12c01b440743ad0a69f262b2bb088432894377f44cdf52612682515",
        expected_campaign="figure8-io-limit-ladder",
        expected_limits=(64, 128, 256, 512, 1024, 2048),
        expected_process_limits=(2048, 2048, 2048, 2048, 2048, 2048),
    ),
    "2048-4096": BundleSpec(
        filename="2048-4096.json",
        sha256="c1173ed3ff40927e5da824a311203acbdd463956f6fbb74abbf868179dfd771c",
        expected_campaign="figure8-io-limit-2048-4096",
        expected_limits=(2048, 4096),
        expected_process_limits=(2048, 4096),
    ),
    "4096-8192": BundleSpec(
        filename="4096-8192.json",
        sha256="f39f626d9a6dbec42e1c64246855b84a181e062dbdc5ff9df5e492101d2c7c1b",
        expected_campaign="figure8-io-limit-4096-8192",
        expected_limits=(4096, 8192),
        expected_process_limits=(4096, 8192),
    ),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_bundle(spec: BundleSpec) -> dict:
    path = SUMMARY_ROOT / spec.filename
    if data_slot() == "paper" and sha256_file(path) != spec.sha256:
        raise ValueError(f"summary checksum mismatch: {path}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    if summary.get("schema_version") != 2:
        raise ValueError(f"invalid summary schema for {path}")
    if summary.get("status") != "complete":
        raise ValueError(f"invalid campaign status for {path}")
    if summary.get("campaign") != spec.expected_campaign:
        raise ValueError(f"campaign mismatch for {path}")
    contract = summary.get("measurement_contract")
    expected_contract = {
        "dataset": "sift100m",
        "layout": "random",
        "method": "stock_lance",
        "query_concurrency": 32,
        "num_queries": 1_000,
        "warmup_queries": 200,
        "warmup_reused": True,
        "nlist": 65_536,
        "num_sub_vectors": 16,
        "bits_per_subcode": 8,
        "nprobe": 256,
        "refine_factor": 8,
        "k": 10,
        "repeats": 3,
        "counterbalanced": True,
        "limit_pairs": [
            {"file": file_limit, "process": process_limit}
            for file_limit, process_limit in zip(
                spec.expected_limits,
                spec.expected_process_limits,
            )
        ],
    }
    if not isinstance(contract, dict):
        raise ValueError(f"measurement contract is missing for {path}")
    mismatches = [
        key for key, expected in expected_contract.items() if contract.get(key) != expected
    ]
    if mismatches:
        raise ValueError(f"measurement contract mismatch for {path}: {mismatches}")
    for key in ("query_sha256", "binary_sha256"):
        if not re.fullmatch(r"[0-9a-f]{64}", str(contract.get(key, ""))):
            raise ValueError(f"invalid {key} for {path}")
    if not re.fullmatch(r"[0-9a-f]{40}", str(contract.get("lance_commit", ""))):
        raise ValueError(f"invalid lance_commit for {path}")
    if data_slot() == "paper" and not re.fullmatch(
        r"[0-9a-f]{64}", str(contract.get("lance_tracked_diff_sha256", ""))
    ):
        raise ValueError(f"invalid lance_tracked_diff_sha256 for {path}")
    if data_slot() == "paper" and not re.fullmatch(
        r"[0-9a-f]{64}", str(contract.get("source_archive_sha256", ""))
    ):
        raise ValueError(f"invalid source_archive_sha256 for {path}")
    index_identity = contract.get("index_identity")
    if not isinstance(index_identity, dict) or not index_identity:
        raise ValueError(f"index identity is missing for {path}")
    if "inventory" in index_identity:
        inventory = index_identity.get("inventory")
        if (
            not str(index_identity.get("dataset_uri", "")).startswith("s3://")
            or not isinstance(inventory, dict)
            or int(inventory.get("object_count", 0)) <= 0
            or int(inventory.get("total_bytes", 0)) <= 0
            or not re.fullmatch(r"[0-9a-f]{64}", str(inventory.get("sha256", "")))
        ):
            raise ValueError(f"frozen index inventory is invalid for {path}")
    elif (
        not re.fullmatch(r"[0-9a-f]{64}", str(index_identity.get("manifest_sha256", "")))
        or not isinstance(index_identity.get("index"), dict)
    ):
        raise ValueError(f"managed index identity is invalid for {path}")
    source_hashes = contract.get("source_hashes")
    if not isinstance(source_hashes, dict) or not source_hashes or any(
        not re.fullmatch(r"[0-9a-f]{64}", str(value))
        for value in source_hashes.values()
    ):
        raise ValueError(f"source hashes are invalid for {path}")
    limits = tuple(sorted(int(limit) for limit in summary["groups"]))
    if limits != spec.expected_limits:
        raise ValueError(f"limit mismatch for {spec.filename}: {limits}")
    for limit in spec.expected_limits:
        group = summary["groups"][str(limit)]
        if len(group["p50_ms"]["values"]) != 3:
            raise ValueError(f"expected three repeats for {path}:{limit}")
    return summary


def series(summary: dict, limits: tuple[int, ...]) -> tuple[np.ndarray, np.ndarray]:
    means = np.asarray([summary["groups"][str(limit)]["p50_ms"]["mean"] for limit in limits])
    errors = np.asarray(
        [summary["groups"][str(limit)]["p50_ms"]["sample_sd"] for limit in limits]
    )
    return means, errors


def apply_style() -> None:
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
        },
    )



def draw_io_sweep(axis: plt.Axes, summaries: dict[str, dict], *, show_legend: bool = True) -> None:
    all_limits = np.asarray([64, 128, 256, 512, 1024, 2048, 4096, 8192])
    positions = np.arange(len(all_limits), dtype=float)
    position = {int(limit): float(index) for index, limit in enumerate(all_limits)}
    axis.axvspan(4.65, 7.35, color=FILL, alpha=0.48, linewidth=0, zorder=0)

    plot_specs = (
        ("ladder", "Q32 ladder", BLUE, "o", "-"),
        ("2048-4096", "2048/4096", ORANGE, "s", "--"),
        ("4096-8192", "4096/8192", BROWN, "^", ":"),
    )
    for key, label, color, marker, linestyle in plot_specs:
        limits = BUNDLES[key].expected_limits
        means, errors = series(summaries[key], limits)
        x = np.asarray([position[limit] for limit in limits])
        axis.errorbar(
            x,
            means,
            yerr=errors,
            label=label,
            color=color,
            marker=marker,
            markersize=2.9,
            markerfacecolor="white" if key != "ladder" else color,
            markeredgecolor=EDGE,
            markeredgewidth=0.42,
            linewidth=0.9,
            linestyle=linestyle,
            capsize=1.25,
            elinewidth=0.55,
            capthick=0.55,
            zorder=3,
        )

    ladder = summaries["ladder"]["groups"]
    axis.annotate(
        f'{ladder["64"]["p50_ms"]["mean"]:.0f}',
        (position[64], ladder["64"]["p50_ms"]["mean"]),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        fontsize=5.2,
        color=LABEL,
    )
    axis.text(
        6.0,
        920,
        "no material gain\nbeyond 2048",
        ha="center",
        va="center",
        fontsize=5.1,
        color="#4b6577",
        linespacing=0.95,
    )

    axis.set_title("(a) Lance p50 latency", pad=2.8, fontweight="regular")
    axis.set_xlim(-0.45, len(all_limits) - 0.55)
    axis.set_yscale("log", base=2)
    axis.set_ylim(350, 7600)
    axis.yaxis.set_major_locator(FixedLocator([400, 800, 1600, 3200, 6400]))
    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{int(value)}"))
    axis.set_xticks(positions)
    axis.set_xticklabels([str(limit) for limit in all_limits], rotation=42, ha="right")
    axis.set_xlabel("I/O Concurrency Limit", labelpad=1.0)
    axis.set_ylabel("p50 latency (ms)", labelpad=1.5)
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
    if show_legend:
        legend = axis.legend(
            loc="upper right",
            frameon=True,
            borderpad=0.25,
            labelspacing=0.2,
            handlelength=1.35,
            handletextpad=0.32,
        )
        legend.get_frame().set_facecolor("white")
        legend.get_frame().set_edgecolor("#333333")
        legend.get_frame().set_linewidth(0.42)


def main() -> None:
    summaries = {name: load_bundle(spec) for name, spec in BUNDLES.items()}
    apply_style()
    fig, axis = plt.subplots(figsize=(1.675, 1.45), dpi=320)
    draw_io_sweep(axis, summaries)
    fig.subplots_adjust(left=0.235, right=0.985, bottom=0.29, top=0.84)
    fig.savefig(
        OUT.with_suffix(".pdf"),
        bbox_inches=None,
        metadata={"CreationDate": None, "ModDate": None},
    )
    fig.savefig(OUT.with_suffix(".png"), dpi=320, bbox_inches=None)
    plt.close(fig)

    provenance = {
        "status": "paper-frozen" if data_slot() == "paper" else "fresh-run-render",
        "query_concurrency": 32,
        "metric": "p50_ms",
        "y_scale": "log2",
        "source_summaries": {
            key: {
                "filename": spec.filename,
                "sha256": spec.sha256,
                "campaign": spec.expected_campaign,
                "limits": list(spec.expected_limits),
            }
            for key, spec in BUNDLES.items()
        },
        "note": (
            "The solid six-point curve is the base ladder. The two hollow-marker "
            "segments are independently counterbalanced extensions; they are not "
            "silently pooled into the main ladder."
        ),
    }
    OUT.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
    main()
