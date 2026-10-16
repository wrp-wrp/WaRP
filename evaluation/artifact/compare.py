#!/usr/bin/env python3
"""Compare authoritative paper artifacts against freshly regenerated ones.

Workflow
--------

1. `python3 -m artifact rebuild`
      -> writes fresh PDF/TEX artifacts under `experiments/<id>/_regen/`.
2. `python3 -m artifact compare`
      -> pixel-compares PDFs and byte-compares generated TeX tables.

A small PDF diff (<1%) is normal because matplotlib and font-renderer
versions drift over time. Generated TeX must match byte-for-byte. A failed
comparison means the render scripts or frozen data no longer reproduce the
paper artifact and should be investigated before shipping.

Requires `pdftoppm` (poppler) on PATH.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .registry import ROOT, paper_artifact_specs


def _require_pdftoppm() -> None:
    if shutil.which("pdftoppm") is None:
        raise SystemExit(
            "pdftoppm not found on PATH. Install poppler "
            "(`brew install poppler` on macOS, `apt install poppler-utils` on Ubuntu)."
        )


_rasterise_counter = [0]


def _rasterise(pdf: Path, out_dir: Path, dpi: int) -> Path:
    # Use a unique prefix per call so that two PDFs with the same stem name
    # (e.g., the authoritative and _regen copies) do not overwrite each
    # other's rasterisation output.
    _rasterise_counter[0] += 1
    prefix = f"{_rasterise_counter[0]:04d}_{pdf.stem}"
    stem = out_dir / prefix
    subprocess.run(
        ["pdftoppm", "-r", str(dpi), "-png", "-f", "1", "-l", "1", str(pdf), str(stem)],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    candidates = sorted(out_dir.glob(f"{prefix}-*.png"))
    if not candidates:
        raise RuntimeError(f"pdftoppm produced no PNG for {pdf}")
    return candidates[0]


def _pixel_diff(left: Path, right: Path) -> tuple[float, str]:
    try:
        from PIL import Image, ImageChops
    except ImportError:
        raise SystemExit("Pillow required: `pip install pillow`")
    l_img = Image.open(left).convert("RGB")
    r_img = Image.open(right).convert("RGB")
    dimensions = f"{l_img.width}x{l_img.height} vs {r_img.width}x{r_img.height}"
    left_aspect = l_img.width / l_img.height
    right_aspect = r_img.width / r_img.height
    if abs(right_aspect / left_aspect - 1.0) > 0.02:
        return float("inf"), dimensions + " (aspect mismatch)"
    if r_img.size != l_img.size:
        r_img = r_img.resize(l_img.size, Image.LANCZOS)
    # Sum of absolute per-channel differences across all pixels,
    # normalised to percent of the maximum possible (3 * 255 per pixel).
    diff_img = ImageChops.difference(l_img, r_img)
    total_abs_diff = sum(
        (bucket % 256) * count
        for bucket, count in enumerate(diff_img.histogram())
    )
    num_pixels = l_img.size[0] * l_img.size[1]
    max_total = num_pixels * 3 * 255
    pct = 100.0 * total_abs_diff / max_total
    return pct, dimensions


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dpi", type=int, default=150, help="rasterisation DPI (default: 150)")
    ap.add_argument("--only", default="", help="comma-separated spec ids")
    ap.add_argument("--threshold", type=float, default=10.0,
                    help="flag figures whose pct diff exceeds this (default 10.0). "
                         "Text-heavy figures drift up to ~8%% across matplotlib + font "
                         "host setups; lower the threshold if you want strict byte parity.")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    _require_pdftoppm()

    specs = paper_artifact_specs()
    if args.only.strip():
        wanted = {tok.strip() for tok in args.only.split(",") if tok.strip()}
        specs = [s for s in specs if s.id in wanted]

    failed: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        print(f"{'spec':<45}{'paper vs regenerated':<40}{'diff':>10}")
        for spec in specs:
            for rel in spec.paper_artifacts:
                authoritative = ROOT / rel
                regenerated = authoritative.parent / "_regen" / authoritative.name
                if not authoritative.exists():
                    print(f"{spec.id:<45}[missing authoritative: {rel}]")
                    failed.append(f"{spec.id}:{rel}")
                    continue
                if not regenerated.exists():
                    print(f"{spec.id:<45}[missing regen; run rebuild_paper_artifacts first]")
                    failed.append(f"{spec.id}:{rel}")
                    continue
                if authoritative.suffix.lower() == ".pdf":
                    left = _rasterise(authoritative, tmp_dir, args.dpi)
                    right = _rasterise(regenerated, tmp_dir, args.dpi)
                    pct, dim = _pixel_diff(left, right)
                    status = "!" if pct > args.threshold else " "
                    print(f"{spec.id:<45}{dim:<40}{pct:>8.3f}% {status}")
                    if pct > args.threshold:
                        failed.append(f"{spec.id}:{rel}")
                else:
                    expected = authoritative.read_bytes()
                    actual = regenerated.read_bytes()
                    matches = expected == actual
                    result = "exact" if matches else "MISMATCH"
                    print(
                        f"{spec.id:<45}{len(expected):<18}{len(actual):<18}{result:>10}"
                    )
                    if not matches:
                        failed.append(f"{spec.id}:{rel}")
    if failed:
        print(f"\n[warn] {len(failed)} artifact comparison(s) failed:")
        for f in failed:
            print(f"  - {f}")
        return 1
    print("\n[ok] every paper artifact matches its regenerated counterpart.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
