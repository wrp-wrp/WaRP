#!/usr/bin/env python3
from __future__ import annotations

import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VENDOR_LANCE_ROOT = ROOT / "third_party" / "lance"


def discover_lance_root(explicit: str = "") -> Path | None:
    candidates: list[Path] = []
    if str(explicit).strip():
        candidates.append(Path(str(explicit)).expanduser())
    env_root = str(os.environ.get("VECIO_LANCE_ROOT", "")).strip()
    if env_root:
        candidates.append(Path(env_root).expanduser())
    candidates.append(DEFAULT_VENDOR_LANCE_ROOT)
    candidates.append(ROOT.parent / "lance-s3-io-optimization")
    candidates.append(ROOT.parent / "lance")

    seen: set[str] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)
        if (resolved / "Cargo.toml").exists() and (resolved / "rust" / "lance-io").exists():
            return resolved
    return None


def with_lance_env(base: dict[str, str] | None, lance_root: Path) -> dict[str, str]:
    env = dict(base or os.environ)
    env["VECIO_LANCE_ROOT"] = str(lance_root)
    return env
