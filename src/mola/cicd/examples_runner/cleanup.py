"""Cleanup operations for example state and remote cases."""

from __future__ import annotations

import shutil
from pathlib import Path

import mola.naming_conventions as names

from .state import ExampleCIState


LEGACY_GENERATED_FILES = (
    names.FILE_LOG_REMOTE_DIRECTORY,
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


def clear_examples(states_only=False, failed_only=False, all_cases=False):
    if sum(bool(value) for value in (states_only, failed_only, all_cases)) != 1:
        raise ValueError("Choose exactly one of --states-only, --failed-only, or --all")

    state_directory = Path(names.DIRECTORY_CI_STATE)
    release_directory = state_directory.parent
    state_files = sorted(state_directory.glob(f"**/*{names.FILE_CI_STATE_SUFFIX}")) \
        if state_directory.exists() else []
    errors = []

    if failed_only or all_cases:
        for state_path in state_files:
            try:
                state = ExampleCIState.read(state_path)
                directories = (
                    ExampleCIState.get_failed_run_directories(state)
                    if failed_only else ExampleCIState.get_all_run_directories(state)
                )
                errors.extend(
                    (state_path, directory, error)
                    for directory, error in ExampleCIState.remove_remote_run_directories(
                        state, directories)
                )
            except Exception as error:
                errors.append((state_path, None, error))

    if states_only or (all_cases and not errors):
        try:
            for path in _remove_legacy_generated_files(Path("examples")):
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
