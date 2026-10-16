#!/usr/bin/env python3
"""Re-render paper artifacts from frozen or freshly measured evidence.

Two output targets:

  regen   (default)   Writes into each Paper Experiment's `_regen/` so the
                      authoritative Paper Artifacts stay untouched. Use this
                      when validating that the current render scripts +
                      current frozen data still produce the paper figures and
                      tables. With ``--data-slot latest``, the destination is
                      `_latest/` instead.

  paper   (opt-in)    Writes directly into the Paper Experiment and
                      overwrites the authoritative artifacts. Pass
                      `--target paper` only when you have a deliberate
                      reason to refresh the paper snapshot (e.g. after
                      a full-mode re-run that updates the frozen data).

Light vs full profile affects which `render` phase runs; the scripts
themselves are the same across profiles.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from .registry import EXPERIMENTS, ROOT, paper_artifact_specs, render_phase


def _render_argv(command, data_slot: str) -> list[str]:
    argv = command.resolve()
    if data_slot == "latest":
        argv = [part.replace("/data/paper/", "/data/latest/") for part in argv]
    return argv


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--profile",
        default="light",
        choices=("full", "light"),
        help="Which render profile to invoke per spec. Default: light.",
    )
    ap.add_argument(
        "--target",
        default="regen",
        choices=("regen", "paper"),
        help="regen -> each Paper Experiment's _regen/ (default, safe). "
             "paper -> the Paper Experiment (overwrites the authoritative paper snapshot).",
    )
    ap.add_argument(
        "--data-slot",
        default="paper",
        choices=("paper", "latest"),
        help="paper -> committed frozen evidence; latest -> fresh runner outputs.",
    )
    ap.add_argument("--only", default="", help="Comma-separated spec ids to render.")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Alias for --profile light (kept for back-compat).",
    )
    return ap.parse_args()


def _selected_specs(args: argparse.Namespace):
    specs = paper_artifact_specs()
    if not args.only.strip():
        return specs
    requested = {tok.strip() for tok in args.only.split(",") if tok.strip()}
    known = {spec.id for spec in EXPERIMENTS}
    missing = sorted(requested - known)
    if missing:
        raise SystemExit(f"unknown experiment ids: {', '.join(missing)}")
    return [spec for spec in specs if spec.id in requested]


def _run_one(spec, profile: str, target: str, data_slot: str) -> None:
    phase = render_phase(spec, profile)
    if not phase.commands:
        return
    out_dir = ROOT / spec.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    render_env = dict(os.environ)
    render_env["WARP_DATA_SLOT"] = data_slot

    if target == "paper":
        # Render directly into the authoritative out_dir; overwrites paper PDFs.
        for c in phase.commands:
            argv = _render_argv(c, data_slot)
            print("$ " + " ".join(argv))
            subprocess.run(argv, cwd=ROOT, check=True, env=render_env)
        return

    # target == "regen": render through the legacy hard-coded output paths,
    # copy changed top-level artifacts into _regen/, then restore the exact
    # pre-render bytes. Never use Git as the backup: the authoritative file may
    # contain intentional uncommitted work.
    regen_dir = out_dir / ("_regen" if data_slot == "paper" else "_latest")
    regen_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"vecio-{spec.id}-backup-") as tmp:
        backup_dir = Path(tmp)
        before: dict[str, tuple[int, int]] = {}
        for path in out_dir.iterdir():
            if not path.is_file():
                continue
            stat = path.stat()
            before[path.name] = (stat.st_mtime_ns, stat.st_size)
            shutil.copy2(path, backup_dir / path.name)

        render_succeeded = False
        try:
            for c in phase.commands:
                argv = _render_argv(c, data_slot)
                print("$ " + " ".join(argv))
                subprocess.run(argv, cwd=ROOT, check=True, env=render_env)
            render_succeeded = True
        finally:
            after_files = {
                path.name: path
                for path in out_dir.iterdir()
                if path.is_file()
            }
            changed_names = []
            for name, path in after_files.items():
                stat = path.stat()
                if before.get(name) != (stat.st_mtime_ns, stat.st_size):
                    changed_names.append(name)

            if render_succeeded:
                for name in changed_names:
                    shutil.copy2(after_files[name], regen_dir / name)

            # Restore every pre-existing top-level file, including dirty JSON,
            # CSV, PDF, and preview artifacts. Remove only files created by
            # this render invocation.
            for name in before:
                shutil.copy2(backup_dir / name, out_dir / name)
            for name in set(after_files) - set(before):
                (out_dir / name).unlink()


def main() -> int:
    args = parse_args()
    profile = "light" if args.smoke else args.profile
    specs = _selected_specs(args)
    target = args.target
    rendered = 0
    for spec in specs:
        phase = render_phase(spec, profile)
        if not phase.commands:
            continue
        print(f"[artifact] {spec.id} ({spec.title}) -> target={target}")
        _run_one(spec, profile, target, args.data_slot)
        rendered += 1
    if target == "regen":
        destination = "_regen" if args.data_slot == "paper" else "_latest"
        print(
            f"[ok] re-rendered {rendered} Paper Experiments from "
            f"data/{args.data_slot}/ into {destination}/ directories "
            "(authoritative artifacts untouched)"
        )
        if args.data_slot == "paper":
            print("     use `python3 -m artifact compare` to verify regenerated outputs")
    else:
        print(
            f"[ok] re-rendered {rendered} Paper Experiments, overwriting authoritative artifacts"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
