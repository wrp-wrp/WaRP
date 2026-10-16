#!/usr/bin/env python3
"""Validate and render the paper figure-and-table ownership catalog."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .registry import EXPERIMENTS


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "experiments" / "PAPER_ARTIFACT_MANIFEST.json"
DOC_PATH = ROOT / "docs" / "PAPER_ARTIFACTS_MAP.md"


def load_manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _validate_sources(where: str, entry: dict[str, Any], errors: list[str]) -> None:
    sources = entry.get("paper_sources")
    if not isinstance(sources, list) or not sources:
        errors.append(f"{where}.paper_sources must be a non-empty list")
    elif any(not isinstance(source, str) or not source for source in sources):
        errors.append(f"{where}.paper_sources contains an invalid path")


def _register_owner(
    where: str,
    entry: dict[str, Any],
    owner_entries: dict[str, list[dict[str, Any]]],
    seen_artifacts: set[str],
    errors: list[str],
) -> None:
    owner = entry.get("experiment_id")
    artifacts = entry.get("paper_artifacts")
    if not isinstance(owner, str) or not owner:
        errors.append(f"{where}.experiment_id is required")
        return
    if not isinstance(artifacts, list) or not artifacts:
        errors.append(f"{where}: experimental artifact must own at least one file")
        return

    owner_entries[owner].append(entry)
    prefix = f"experiments/{owner}/"
    for artifact in artifacts:
        if not isinstance(artifact, str) or not artifact.startswith(prefix):
            errors.append(f"{where}: artifact must live under {prefix}: {artifact!r}")
            continue
        if artifact in seen_artifacts:
            errors.append(f"Paper Artifact has multiple owners: {artifact}")
        seen_artifacts.add(artifact)
        if not (ROOT / artifact).is_file():
            errors.append(f"Paper Artifact is missing: {artifact}")


def validate(manifest: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if manifest.get("schema_version") != 4:
        errors.append("schema_version must be 4")

    figures = manifest.get("figures")
    tables = manifest.get("tables")
    if not isinstance(figures, list) or not figures:
        errors.append("figures must be a non-empty list")
        figures = []
    if not isinstance(tables, list) or not tables:
        errors.append("tables must be a non-empty list")
        tables = []

    seen_labels: set[str] = set()
    seen_numbers: set[str] = set()
    seen_artifacts: set[str] = set()
    owner_entries: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for index, figure in enumerate(figures):
        where = f"figures[{index}]"
        if not isinstance(figure, dict):
            errors.append(f"{where} must be an object")
            continue
        number = str(figure.get("number", ""))
        label = figure.get("label")
        kind = figure.get("kind")
        artifacts = figure.get("paper_artifacts")
        _validate_sources(where, figure, errors)

        if not number:
            errors.append(f"{where}.number is required")
        elif number in seen_numbers:
            errors.append(f"duplicate paper figure number: {number}")
        seen_numbers.add(number)
        if not isinstance(label, str) or not label.startswith("fig:"):
            errors.append(f"{where}.label must be a fig: label")
        elif label in seen_labels:
            errors.append(f"duplicate paper label: {label}")
        else:
            seen_labels.add(label)
        for sublabel in figure.get("subfigure_labels", []):
            if not isinstance(sublabel, str) or not sublabel.startswith("fig:"):
                errors.append(f"{where}.subfigure_labels contains an invalid label")
            elif sublabel in seen_labels:
                errors.append(f"duplicate paper label: {sublabel}")
            else:
                seen_labels.add(sublabel)

        if kind == "diagram":
            if figure.get("experiment_id") is not None:
                errors.append(f"{where}: conceptual diagram cannot own an experiment")
            if artifacts != []:
                errors.append(f"{where}: conceptual diagram cannot own repository artifacts")
        elif kind == "experiment":
            _register_owner(where, figure, owner_entries, seen_artifacts, errors)
        else:
            errors.append(f"{where}.kind must be experiment or diagram")

    for index, table in enumerate(tables):
        where = f"tables[{index}]"
        if not isinstance(table, dict):
            errors.append(f"{where} must be an object")
            continue
        label = table.get("label")
        kind = table.get("kind")
        artifacts = table.get("paper_artifacts")
        _validate_sources(where, table, errors)
        if not isinstance(label, str) or not label.startswith("tab:"):
            errors.append(f"{where}.label must be a tab: label")
        elif label in seen_labels:
            errors.append(f"duplicate paper label: {label}")
        else:
            seen_labels.add(label)

        if kind == "documentation":
            if table.get("experiment_id") is not None:
                errors.append(f"{where}: documentation table cannot own an experiment")
            if artifacts != []:
                errors.append(f"{where}: documentation table cannot own repository artifacts")
        elif kind == "experiment":
            _register_owner(where, table, owner_entries, seen_artifacts, errors)
        else:
            errors.append(f"{where}.kind must be experiment or documentation")

    artifact_specs = [spec for spec in EXPERIMENTS if spec.role in {"figure", "table"}]
    specs_by_owner: dict[str, Any] = {}
    for spec in artifact_specs:
        out_dir = Path(spec.out_dir)
        if out_dir.parent.as_posix() != "experiments":
            errors.append(f"{spec.id}: Paper Experiment must be a top-level experiments/ directory")
            continue
        owner = out_dir.name
        if owner in specs_by_owner:
            errors.append(f"multiple artifact specs target experiments/{owner}/")
        specs_by_owner[owner] = spec

    owners = set(owner_entries)
    if owners != set(specs_by_owner):
        missing_specs = sorted(owners - set(specs_by_owner))
        missing_catalog = sorted(set(specs_by_owner) - owners)
        if missing_specs:
            errors.append("Paper Experiments missing artifact specs: " + ", ".join(missing_specs))
        if missing_catalog:
            errors.append("artifact specs missing catalog owners: " + ", ".join(missing_catalog))

    for owner in sorted(owners & set(specs_by_owner)):
        spec = specs_by_owner[owner]
        entries = owner_entries[owner]
        figure_entries = [
            entry for entry in entries if str(entry["label"]).startswith("fig:")
        ]
        numbered_entry = figure_entries[0] if figure_entries else entries[0]
        owner_kind = "figure" if figure_entries else "table"
        try:
            expected_owner_prefix = (
                f"{owner_kind}-{int(numbered_entry['number']):02d}-"
            )
        except (KeyError, TypeError, ValueError):
            errors.append(f"{owner}: paper artifact number must be an integer")
        else:
            if not owner.startswith(expected_owner_prefix):
                errors.append(
                    f"{owner}: owner directory must start with "
                    f"{expected_owner_prefix}"
                )
        labels = {str(entry["label"]) for entry in entries}
        artifacts = {
            str(artifact)
            for entry in entries
            for artifact in entry["paper_artifacts"]
        }
        owns_figure = any(str(entry["label"]).startswith("fig:") for entry in entries)
        expected_role = "figure" if owns_figure else "table"
        if spec.role != expected_role:
            errors.append(f"{spec.id}: role must be {expected_role}, got {spec.role}")
        if labels != set(spec.paper_labels):
            errors.append(f"{spec.id}: spec paper_labels do not match catalog labels")
        if artifacts != set(spec.paper_artifacts):
            errors.append(f"{spec.id}: spec paper_artifacts do not match catalog artifacts")

        owner_dir = ROOT / "experiments" / owner
        if not owner_dir.is_dir():
            errors.append(f"Paper Experiment directory is missing: {owner_dir}")
            continue
        actual = {
            path.relative_to(ROOT).as_posix()
            for suffix in ("*.pdf", "*.tex")
            for path in owner_dir.glob(suffix)
            if path.is_file()
        }
        for extra in sorted(actual - artifacts):
            errors.append(f"{owner}: unowned top-level Paper Artifact remains: {extra}")

    actual_dirs = {
        path.name
        for path in (ROOT / "experiments").iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    for directory in sorted(actual_dirs - owners):
        errors.append(f"unclassified top-level experiment directory: {directory}")
    for directory in sorted(owners - actual_dirs):
        errors.append(f"catalog owner directory is missing: {directory}")

    registered_runner_dirs: set[str] = set()
    for spec in EXPERIMENTS:
        for phase in (spec.run, spec.render):
            for command in phase.commands:
                for argument in command.argv:
                    parts = Path(argument).parts
                    if len(parts) >= 3 and parts[0] == "runners":
                        registered_runner_dirs.add(parts[1])
    actual_runner_dirs = {
        path.name
        for path in (ROOT / "runners").iterdir()
        if path.is_dir() and not path.name.startswith(".")
    }
    allowed_runner_dirs = registered_runner_dirs | {"shared"}
    for directory in sorted(actual_runner_dirs - allowed_runner_dirs):
        errors.append(f"runner directory has no artifact owner: {directory}")
    for directory in sorted(registered_runner_dirs - actual_runner_dirs):
        errors.append(f"registered runner directory is missing: {directory}")
    forbidden_runner_suffixes = {".jpeg", ".jpg", ".pdf", ".png", ".svg", ".tex"}
    for path in sorted((ROOT / "runners").rglob("*")):
        if path.is_file() and path.suffix.lower() in forbidden_runner_suffixes:
            errors.append(
                "generated artifact must live under experiments/, not runners/: "
                + path.relative_to(ROOT).as_posix()
            )
    return errors


def _artifact_cell(paths: list[str]) -> str:
    return "<br>".join(f"`{path}`" for path in paths)


def render_markdown(manifest: dict[str, Any]) -> str:
    figures = manifest["figures"]
    tables = manifest["tables"]
    experimental_figures = [item for item in figures if item["kind"] == "experiment"]
    diagrams = [item for item in figures if item["kind"] == "diagram"]
    experimental_tables = [item for item in tables if item["kind"] == "experiment"]
    documentation_tables = [item for item in tables if item["kind"] == "documentation"]

    lines = [
        "# Paper artifact -> Paper Experiment map",
        "",
        "Generated from `experiments/PAPER_ARTIFACT_MANIFEST.json` by",
        "`python3 -m artifact catalog write-doc`. Do not hand-edit.",
        "",
        "## Ownership rule",
        "",
        "Every experiment-backed paper figure or table has exactly one Paper",
        "Experiment owner. Artifacts derived from the same frozen evidence stay",
        "together. Owner directories use `figure-NN-short-name` or",
        "`table-NN-short-name`. Conceptual diagrams and documentation tables have",
        "no experiment directory.",
        "",
        "## Experimental figures",
        "",
        "| Figure | Paper label | Paper Experiment | Repository artifact(s) |",
        "| --- | --- | --- | --- |",
    ]
    for item in experimental_figures:
        lines.append(
            f"| {item['number']} | `{item['label']}` | `experiments/{item['experiment_id']}/` | "
            f"{_artifact_cell(item['paper_artifacts'])} |"
        )

    lines.extend(
        [
            "",
            "## Conceptual diagrams",
            "",
            "| Figure | Paper label | Paper source |",
            "| --- | --- | --- |",
        ]
    )
    for item in diagrams:
        lines.append(
            f"| {item['number']} | `{item['label']}` | {_artifact_cell(item['paper_sources'])} |"
        )

    lines.extend(
        [
            "",
            "## Experimental tables",
            "",
            "| Table | Paper label | Paper Experiment | Repository artifact(s) |",
            "| --- | --- | --- | --- |",
        ]
    )
    for item in experimental_tables:
        lines.append(
            f"| {item['number']} | `{item['label']}` | `experiments/{item['experiment_id']}/` | "
            f"{_artifact_cell(item['paper_artifacts'])} |"
        )

    lines.extend(
        [
            "",
            "## Documentation tables",
            "",
            "These tables describe datasets or methods; they do not report a measured result.",
            "",
            "| Table | Paper label | Paper source |",
            "| --- | --- | --- |",
        ]
    )
    for item in documentation_tables:
        lines.append(
            f"| {item['number']} | `{item['label']}` | {_artifact_cell(item['paper_sources'])} |"
        )

    lines.extend(
        [
            "",
            "## Verification",
            "",
            "```bash",
            "python3 -m artifact catalog check",
            "python3 -m artifact catalog check-doc",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def print_summary(manifest: dict[str, Any]) -> None:
    figures = manifest.get("figures", [])
    tables = manifest.get("tables", [])
    print(
        "[catalog] "
        f"experimental_figures={sum(item.get('kind') == 'experiment' for item in figures)} "
        f"diagrams={sum(item.get('kind') == 'diagram' for item in figures)} "
        f"experimental_tables={sum(item.get('kind') == 'experiment' for item in tables)} "
        f"documentation_tables={sum(item.get('kind') == 'documentation' for item in tables)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        nargs="?",
        choices=("check", "write-doc", "check-doc"),
        default="check",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = load_manifest()
    errors = validate(manifest)
    print_summary(manifest)
    if errors:
        for error in errors:
            print(f"[error] {error}")
        return 1

    rendered = render_markdown(manifest)
    if args.action == "write-doc":
        DOC_PATH.write_text(rendered, encoding="utf-8")
        print(f"[ok] wrote {DOC_PATH.relative_to(ROOT)}")
    elif args.action == "check-doc":
        if not DOC_PATH.is_file():
            print(f"[error] generated document is missing: {DOC_PATH.relative_to(ROOT)}")
            return 1
        if DOC_PATH.read_text(encoding="utf-8") != rendered:
            print("[error] generated document is stale; run artifact_catalog.py write-doc")
            return 1
        print(f"[ok] {DOC_PATH.relative_to(ROOT)} is current")
    else:
        print("[ok] paper artifact ownership is complete and internally consistent")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
