"""Resolve renderer inputs from frozen or freshly measured evidence."""

from __future__ import annotations

import os
from pathlib import Path


DATA_SLOT_ENV = "WARP_DATA_SLOT"
VALID_DATA_SLOTS = {"paper", "latest"}


def data_slot() -> str:
    """Return the active evidence slot, defaulting to frozen paper data."""

    slot = str(os.environ.get(DATA_SLOT_ENV, "paper")).strip().lower()
    if slot not in VALID_DATA_SLOTS:
        raise ValueError(
            f"{DATA_SLOT_ENV} must be one of {sorted(VALID_DATA_SLOTS)}, got {slot!r}"
        )
    return slot


def paper_data_path(path: Path) -> Path:
    """Map a canonical ``data/paper`` path to the active evidence slot."""

    path = Path(path)
    if data_slot() == "paper":
        return path
    parts = list(path.parts)
    for index in range(len(parts) - 1):
        if parts[index] == "data" and parts[index + 1] == "paper":
            parts[index + 1] = "latest"
            return Path(*parts)
    raise ValueError(f"path is not under a data/paper slot: {path}")


def frozen_evidence() -> bool:
    """Whether renderers should enforce committed paper-data hashes."""

    return data_slot() == "paper"
