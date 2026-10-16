#!/usr/bin/env python3
"""Generate the paper payload-accounting table from frozen Figure 7 evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from artifact.data import frozen_evidence, paper_data_path


HERE = Path(__file__).resolve().parent
DEFAULT_SUMMARY = paper_data_path(HERE / "data" / "paper" / "figure7_performance_summary.json")
DEFAULT_PRICING = HERE / "data" / "paper" / "aws_pricing.json"
DEFAULT_TEX = HERE / "table_payload_accounting.tex"
DEFAULT_CSV = HERE / "table_payload_accounting.csv"

SUMMARY_SHA256 = "73bdc1ba41410907b3b816fa3f202af2c0c84ef12ebc331686a2b8e35c0cbbca"
PRICING_SHA256 = "8c5e173fab429a29ac683eb5974dc717ee223fb6cc693cb6175a990a77ff2c47"

DATASETS = (
    ("sift1m", "SIFT1M"),
    ("glove", "GloVe"),
    ("openimage", "OpenImage"),
    ("msong", "MSong"),
    ("gist1m", "GIST1M"),
    ("deep100m", "DEEP100M"),
    ("sift100m", "SIFT100M"),
)
METHODS = ("stock_lance", "warp", "warp_reorder")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--pricing", default=str(DEFAULT_PRICING))
    parser.add_argument("--out-tex", default=str(DEFAULT_TEX))
    parser.add_argument("--out-csv", default=str(DEFAULT_CSV))
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_hash(path: Path, expected: str) -> None:
    actual = _sha256(path)
    if actual != expected:
        raise ValueError(f"checksum mismatch for {path}: expected {expected}, got {actual}")


def _display(value: Decimal) -> str:
    return format(value.quantize(Decimal("0.1"), rounding=ROUND_HALF_UP), ".1f")


def _load_inputs(summary_path: Path, pricing_path: Path) -> tuple[dict, dict]:
    if frozen_evidence():
        _require_hash(summary_path, SUMMARY_SHA256)
    _require_hash(pricing_path, PRICING_SHA256)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    pricing = json.loads(pricing_path.read_text(encoding="utf-8"))
    rows = {(row["dataset"], row["method"]): row for row in summary.get("aggregate", [])}
    expected = {(dataset, method) for dataset, _ in DATASETS for method in METHODS}
    if summary.get("status") != "complete" or set(rows) != expected:
        raise ValueError("Figure 7 evidence does not contain the complete 7-dataset x 3-method matrix")
    return rows, pricing


def _table_rows(rows: dict, pricing: dict) -> list[dict[str, str]]:
    queries = Decimal(str(pricing["calculation_scope"]["queries"]))
    get_price = Decimal(str(pricing["s3_standard_get"]["price_per_request"]))
    transfer_price = Decimal(str(pricing["cross_region_transfer"]["price_per_gib"]))
    billing_bytes = Decimal(str(pricing["cross_region_transfer"]["billing_unit_bytes"]))
    output: list[dict[str, str]] = []
    for dataset, label in DATASETS:
        lance_bytes = Decimal(
            str(rows[(dataset, "stock_lance")]["metrics"]["bytes_per_q"]["mean"])
        )
        columns: dict[str, list[str]] = {
            "fetched_mib": [],
            "overfetch": [],
            "payload_gbps": [],
            "same_region_cents": [],
            "cross_region_usd": [],
        }
        for method in METHODS:
            metrics = rows[(dataset, method)]["metrics"]
            fetched_bytes = Decimal(str(metrics["bytes_per_q"]["mean"]))
            gets = Decimal(str(metrics["iops_per_q"]["mean"]))
            payload_gbps = Decimal(str(metrics["measured_payload_gbps"]["mean"]))
            same_region = gets * queries * get_price
            cross_region = (
                same_region
                + fetched_bytes * queries / billing_bytes * transfer_price
            )
            columns["fetched_mib"].append(_display(fetched_bytes / Decimal(2**20)))
            columns["overfetch"].append(_display(fetched_bytes / lance_bytes - Decimal(1)))
            columns["payload_gbps"].append(_display(payload_gbps))
            columns["same_region_cents"].append(_display(same_region * Decimal(100)))
            columns["cross_region_usd"].append(_display(cross_region))
        output.append(
            {
                "dataset": label,
                **{name: "/".join(values) for name, values in columns.items()},
            }
        )
    return output


def _render_tex(rows: list[dict[str, str]]) -> str:
    lines = [
        r"\begin{table*}[!t]",
        r"\caption{\textbf{End-to-end bytes, bandwidth, overfetch, and cost.}",
        r"Each entry reports Lance / \system{} / \system{} + Reorder.}",
        r"\label{tab:eval-payload-accounting}",
        r"\centering",
        r"\small",
        r"\renewcommand{\arraystretch}{1.0}",
        r"\setlength{\tabcolsep}{1.5pt}",
        r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}lccccccc@{}}",
        r"\toprule",
        r"Metric & SIFT1M & GloVe & OpenImage & MSong & GIST1M & DEEP100M & SIFT100M \\",
        r"\midrule",
    ]
    metrics = (
        ("Fetched (MiB/q)", "fetched_mib"),
        (r"Overfetch ($\times$)", "overfetch"),
        ("Payload bandwidth (Gbps)", "payload_gbps"),
        ("Same-region (cents)", "same_region_cents"),
        (r"Cross-region (\$)", "cross_region_usd"),
    )
    for label, key in metrics:
        lines.append(label + " & " + " & ".join(row[key] for row in rows) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular*}", r"\end{table*}", ""])
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    summary_path = Path(args.summary).resolve()
    pricing_path = Path(args.pricing).resolve()
    out_tex = Path(args.out_tex).resolve()
    out_csv = Path(args.out_csv).resolve()
    evidence, pricing = _load_inputs(summary_path, pricing_path)
    rows = _table_rows(evidence, pricing)

    out_tex.parent.mkdir(parents=True, exist_ok=True)
    out_tex.write_text(_render_tex(rows), encoding="utf-8")
    with out_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] tex={out_tex}")
    print(f"[ok] csv={out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
