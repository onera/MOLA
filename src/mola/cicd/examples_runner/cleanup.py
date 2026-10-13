"""Cleanup operations for local artifacts and remote example cases."""

from __future__ import annotations

import shutil
from pathlib import Path

import mola.naming_conventions as names
from mola import server as SV

from .discovery import scan_example_definitions
from .status import resolve_live_statuses


LEGACY_GENERATED_FILES = (
    names.FILE_WORKFLOW_MANAGER,
    names.FILE_GATHER_RESULTS,
)


def _remove_legacy_generated_files(examples_directory: Path) -> list:
    removed = []
    if not examples_directory.exists():
        return removed
    for filename in LEGACY_GENERATED_FILES:
        for path in examples_directory.rglob(filename):
            if path.is_file():
                path.unlink()
                removed.append(path)
    return removed


def clear_examples(
        local_only=False, failed_only=False, all_cases=False,
        source_user=None, examples_root="examples"):
    if sum(bool(value) for value in (local_only, failed_only, all_cases)) != 1:
        raise ValueError("Choose exactly one of --local-only, --failed-only, or --all")

    release_directory = Path(".release")
    errors = []

    if failed_only or all_cases:
        definitions = scan_example_definitions(
            examples_root=examples_root,
            source_user=source_user,
        )
        if failed_only:
            live_results = resolve_live_statuses(
                examples_root=examples_root,
                source_user=source_user,
            )
            targets = []
            for definition in definitions:
                result = live_results.get(definition.identifier, {})
                for detail in result.get("details", []):
                    if str(detail.get("status", "")).upper() in names.FAILED_STATUSES:
                        targets.append((definition, detail.get("run_directory")))
        else:
            targets = [
                (definition, definition.root_directory)
                for definition in definitions
            ]

        seen = set()
        for definition, directory in targets:
            if not directory or directory in seen:
                continue
            seen.add(directory)
            try:
                print(f"  Removing remote run directory: {directory}")
                SV.remove_path(
                    directory,
                    machine=definition.machine,
                    user=None,
                    file_only=False,
                )
            except Exception as error:
                errors.append((definition.script_path, directory, error))
                print(f"  Error removing {directory}: {error}")

    if local_only or (all_cases and not errors):
        try:
            for path in _remove_legacy_generated_files(Path(examples_root)):
                print(f"  Removed legacy generated file: {path}")
            if release_directory.exists():
                shutil.rmtree(release_directory)
                print(f"  Removed local directory: {release_directory}")
        except Exception as error:
            errors.append((release_directory, None, error))

    if errors:
        print(f"Clear completed with {len(errors)} error(s).")
        return False
    print("Clear completed successfully.")
    return True
