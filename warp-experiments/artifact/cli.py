"""List, inspect, run, and render registered Paper Experiments."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from typing import Sequence

from .registry import (
    ROOT,
    experiment_ids,
    get,
    ordered_for_run,
    prepare_phase,
    render_phase,
    run_phase,
)


def _parse_args(argv: Sequence[str] | None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("list", "show", "prepare", "run", "render"))
    parser.add_argument("experiment_id", nargs="?")
    parser.add_argument("--profile", default="full", choices=("full", "light"))
    parser.add_argument(
        "--data-slot",
        default="",
        choices=("", "paper", "latest"),
        help="Render evidence slot. Defaults to latest for full and paper for light.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--parameters",
        action="store_true",
        help="For show, print the canonical reproduction contract as JSON.",
    )
    args, extra = parser.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]
    return args, extra


def _show(experiment_id: str) -> None:
    spec = get(experiment_id)
    print(f"[experiment] {spec.id}")
    print(f"[title] {spec.title}")
    print(f"[role] {spec.role}")
    print(f"[dir] {spec.out_dir}")
    print(f"[summary] {spec.summary}")
    requirements = []
    if spec.needs_lance:
        requirements.append("lance")
    if spec.needs_s3:
        requirements.append("s3")
    print(f"[requires] {','.join(requirements) if requirements else 'none'}")
    print(f"[depends-on] {','.join(spec.depends_on) if spec.depends_on else 'none'}")
    print(f"[paper-labels] {','.join(spec.paper_labels) if spec.paper_labels else 'none'}")
    print("[paper-artifacts]")
    for item in spec.paper_artifacts:
        print(f"  - {item}")
    print("[expected-runner-outputs]")
    for item in spec.expected_runner_outputs:
        print(f"  - {item}")
    print("[expected-outputs]")
    for item in spec.expected_outputs:
        print(f"  - {item}")
    for action, phase in (
        ("prepare", prepare_phase(spec, "full")),
        ("run", run_phase(spec, "full")),
        ("render", render_phase(spec, "full")),
    ):
        print(f"[{action}]")
        for command in phase.commands:
            print("  $ " + shlex.join(command.resolve()))


def _execute(spec_id: str, action: str, profile: str, extra: list[str], dry_run: bool) -> None:
    target = get(spec_id)
    specs = ordered_for_run([spec_id]) if action in {"prepare", "run"} else [target]

    def selected_phase(spec):
        if action == "prepare":
            return prepare_phase(spec, profile)
        if action == "run":
            return run_phase(spec, profile)
        return render_phase(spec, profile)

    target_phase = selected_phase(target)
    if extra and len(target_phase.commands) != 1:
        raise SystemExit(
            "extra arguments are supported only when the selected phase has one command"
        )
    executed = 0
    for spec in specs:
        phase = selected_phase(spec)
        commands = [command.resolve() for command in phase.commands]
        if not commands:
            print(f"[skip] {spec.id} has no {action} commands for profile={profile}")
            continue
        if extra and spec.id == target.id:
            if len(commands) != 1:
                # Kept defensive even though target_phase was checked above.
                raise SystemExit(
                    "extra arguments are supported only when the selected phase has one command"
                )
            commands[0].extend(extra)
        (ROOT / spec.out_dir).mkdir(parents=True, exist_ok=True)
        for command in commands:
            print("$ " + shlex.join(command))
            if not dry_run:
                subprocess.run(command, cwd=ROOT, check=True)
            executed += 1
    if executed == 0 and specs:
        if extra:
            raise SystemExit(
                f"{target.id} has no {action} command that can accept extra arguments"
            )


def _render_safely(
    spec_id: str,
    profile: str,
    data_slot: str,
    extra: list[str],
    dry_run: bool,
) -> None:
    """Route single-spec rendering through rebuild's non-destructive slots."""
    get(spec_id)
    if extra:
        raise SystemExit(
            "renderer-specific arguments are not supported by the safe render entrypoint; "
            "invoke the registered renderer directly for development-only overrides"
        )
    slot = data_slot or ("paper" if profile == "light" else "latest")
    command = [
        sys.executable,
        "-m",
        "artifact",
        "rebuild",
        "--only",
        spec_id,
        "--profile",
        profile,
        "--data-slot",
        slot,
    ]
    print("$ " + shlex.join(command))
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def main(argv: Sequence[str] | None = None) -> int:
    args, extra = _parse_args(argv)
    if args.action == "list":
        if args.parameters:
            raise SystemExit("--parameters is supported only with show")
        if extra:
            raise SystemExit(f"unknown list argument(s): {' '.join(extra)}")
        for spec_id in experiment_ids():
            spec = get(spec_id)
            print(f"{spec.id}\t{spec.title}")
        return 0
    if not args.experiment_id:
        raise SystemExit("experiment_id is required for show/prepare/run/render")
    if args.action == "show":
        if extra:
            raise SystemExit(f"unknown show argument(s): {' '.join(extra)}")
        if args.parameters:
            from .contracts import contract_payload

            print(json.dumps(contract_payload(args.experiment_id), indent=2, sort_keys=True))
            return 0
        _show(args.experiment_id)
        return 0
    if args.parameters:
        raise SystemExit("--parameters is supported only with show")
    if args.action == "render":
        _render_safely(
            args.experiment_id,
            args.profile,
            args.data_slot,
            extra,
            bool(args.dry_run),
        )
        return 0
    _execute(
        args.experiment_id,
        args.action,
        args.profile,
        extra,
        bool(args.dry_run),
    )
    return 0
