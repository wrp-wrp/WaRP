"""Load and validate the frozen Figure 11 query-shift evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from artifact.data import frozen_evidence, paper_data_path


HERE = Path(__file__).resolve().parent
SIFT_DATA = paper_data_path(HERE / "data" / "paper" / "sift_summary.json")
SIFT_SHA256 = "bf6cb2e0b9eec6542f76175d9129d89827670200f1ffc42234704227e9ad6e22"
EXPECTED_CELLS = {
    ("tight", 32),
    ("loose", 32),
    ("tight", 256),
    ("loose", 256),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_sift() -> dict[tuple[str, int], dict]:
    if frozen_evidence() and sha256(SIFT_DATA) != SIFT_SHA256:
        raise ValueError("Figure 11 SIFT summary checksum mismatch")
    data = json.loads(SIFT_DATA.read_text(encoding="utf-8"))
    if data.get("protocol_id") != "figure11-sift-transfer-v1":
        raise ValueError("unexpected SIFT protocol")
    cells = {
        (cell["population"], int(cell["nprobe"])): cell
        for cell in data["cells"]
    }
    if set(cells) != EXPECTED_CELLS:
        raise ValueError("incomplete SIFT workload matrix")
    return cells
