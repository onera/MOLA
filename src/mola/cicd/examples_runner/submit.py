"""Example discovery and submission."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from mola import server as SV

from .discovery import (
    discover_remote_layout,
    scan_example_definitions,
)


def detect_execution_mode(example_path: Path) -> Optional[str]:
    lines = Path(example_path).read_text(encoding="utf-8").splitlines()
    for line in lines[-5:]:
        if not line.strip().startswith("# MOLA_CI_MODE:"):
            continue
        mode = line.split(":", 1)[1].strip().strip("\"'")
        if mode in {"mono", "multi"}:
            return mode
    return None


def execute_example(example_path: Path, mode: str) -> str:
    """Execute one marked example and return its raw command output."""

    example_path = Path(example_path).absolute()
    solver = SV.extract_param_from_script(example_path, "Solver")
    command = f"cd {example_path.parent} && python3 {example_path.name}"
    return SV.submit_command(
        command=command,
        machine="localhost",
        use_mola_env=True,
        remote_solver=solver,
    )


def submit_examples(
        force: bool = False,
        dry_run: bool = False,
        failed_only: bool = False,
        source_user: Optional[str] = None,
        examples_root: str = "examples"):
    print("=" * 60)
    print("MOLA CI/CD Examples - Submit Phase")
    print("=" * 60)
    if dry_run:
        print("\n[DRY-RUN MODE] No actual submission will occur.\n")

    definitions = scan_example_definitions(
        examples_root=examples_root,
        source_user=source_user,
    )
    layout = discover_remote_layout(definitions)
    failed_ids = set()
    failed_directories = {}
    if failed_only:
        from .status import resolve_live_statuses

        live_results = resolve_live_statuses(
            examples_root=examples_root,
            source_user=source_user,
        )
        failed_ids = {
            identifier
            for identifier, result in live_results.items()
            if str(result.get("overall", "")).upper() in {
                "FAILED", "ERROR", "ERROR_PREPARING_WORKFLOW"
            }
        }
        failed_directories = {
            identifier: [
                detail["run_directory"]
                for detail in result.get("details", [])
                if str(detail.get("status", "")).upper() in {
                    "FAILED", "ERROR", "ERROR_PREPARING_WORKFLOW"
                }
            ]
            for identifier, result in live_results.items()
        }

    submitted = skipped = overridden = 0

    for definition in definitions:
        example_path = definition.script_path
        mode = definition.mode

        print(f"\n{example_path} (Mode {mode})")
        existing = bool(layout.get("exists", {}).get(definition.identifier))
        action = "submit"
        message = "Ready for submission"
        directories_to_remove = []

        if failed_only:
            if definition.identifier not in failed_ids:
                print("  [SKIP] Example is not marked FAILED by live status")
                skipped += 1
                continue
            action = "override"
            directories_to_remove = failed_directories.get(definition.identifier, [])
            message = "Submitting only failed case(s)"
        elif existing:
            if force:
                action = "override"
                directories_to_remove = [definition.root_directory]
                message = "Force override (user requested)"
            else:
                action = "skip"
                message = (
                    f"Remote root already exists at {definition.root_directory}; "
                    "skipping submission"
                )

        if dry_run:
            print(f"  [{action.upper()}] {message}")
            if action == "skip":
                skipped += 1
            elif action == "override":
                overridden += 1
            else:
                submitted += 1
            continue

        if action == "skip":
            print(f"  Skipped: {message}")
            skipped += 1
            continue
        if action == "override" and directories_to_remove:
            errors = []
            for directory in dict.fromkeys(directories_to_remove):
                try:
                    SV.remove_path(
                        directory,
                        machine=definition.machine,
                        user=None,
                        file_only=False,
                    )
                except Exception as error:
                    errors.append((directory, error))
            if errors:
                raise RuntimeError(
                    "Could not remove all remote run directories before submission"
                )
            overridden += 1

        print(f"  Executing {mode} example: {example_path.name}")
        execute_example(example_path, mode)
        print("  Submitted; remote layout will be rediscovered for status checks")
        submitted += 1

    print("\n" + "=" * 60)
    print(f"Summary: {submitted} submitted, {overridden} overridden, {skipped} skipped")
    print("=" * 60)
