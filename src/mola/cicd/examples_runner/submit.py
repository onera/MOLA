"""Example discovery and submission."""

from __future__ import annotations

import os
import posixpath
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from mola import server as SV
from mola.server.submission import Submission, parse_job_ids, parse_submission_events

from .status_helper import normalize_submissions, unique
from .state import ExampleCIState


def detect_execution_mode(example_path: Path) -> Optional[str]:
    lines = Path(example_path).read_text(encoding="utf-8").splitlines()
    for line in lines[-5:]:
        if not line.strip().startswith("# MOLA_CI_MODE:"):
            continue
        mode = line.split(":", 1)[1].strip().strip("\"'")
        if mode in {"mono", "multi"}:
            return mode
    return None


def parse_job_id_from_output(output: str) -> list[str]:
    """Compatibility wrapper around the shared scheduler output parser."""
    return [str(job_id) for job_id in parse_job_ids(output)]


def _machine_for_run_directory(run_directory: str) -> str:
    try:
        return SV.guess_machine(run_directory)
    except Exception:
        return "localhost"


def _submission_state(example_path: Path, mode: str, solver: str,
                      run_directory: str, records: list[dict],
                      example_name: Optional[str] = None) -> Dict:
    submissions = normalize_submissions(records)
    if not submissions:
        submissions = [{
            "job_id": None,
            "scheduler": "SLURM",
            "machine": _machine_for_run_directory(run_directory),
            "user": None,
            "solver": solver,
            "run_directories": [run_directory],
        }]
    run_directories = unique(
        directory
        for submission in submissions
        for directory in submission.get("run_directories", [])
    )
    if not run_directories:
        run_directories = [run_directory]
    first = submissions[0]
    try:
        state_example_path = str(example_path.relative_to(Path.cwd()))
    except ValueError:
        state_example_path = str(example_path)
    state = {
        "example_path": state_example_path,
        "example_name": example_name or example_path.stem,
        "execution_mode": mode,
        "solver": solver,
        "machine": first.get("machine", _machine_for_run_directory(run_directory)),
        "user": first.get("user"),
        "source_user": os.environ.get("USER") or os.environ.get("USERNAME"),
        "run_directory": run_directory,
        "run_directories": run_directories,
        "submissions": submissions,
        "status": "submitted",
    }
    if mode == "multi":
        sequence_roots = unique(
            submission.get("sequence_id")
            for submission in submissions
            if submission.get("sequence_id")
        )
        if len(sequence_roots) == 1:
            state["root_directory"] = sequence_roots[0]
        elif sequence_roots:
            try:
                state["root_directory"] = posixpath.commonpath(sequence_roots)
            except ValueError:
                pass
    return state


def execute_example(example_path: Path, mode: str) -> Dict:
    example_path = Path(example_path).absolute()
    solver = SV.extract_param_from_script(example_path, "Solver")
    run_directory = os.path.expandvars(
        SV.extract_param_from_script(example_path, "RunDirectory")
    )
    try:
        example_name = str(SV.extract_param_from_script(example_path, "JobName"))
    except (ValueError, TypeError):
        example_name = example_path.stem
    command = f"cd {example_path.parent} && python3 {example_path.name}"

    output = SV.submit_command(
        command=command,
        machine="localhost",
        use_mola_env=True,
        remote_solver=solver,
    )
    records = parse_submission_events(output)

    if not records:
        records = [
            Submission.create(
                job_id=job_id,
                scheduler="SLURM",
                machine=_machine_for_run_directory(run_directory),
                user=None,
                run_directories=[run_directory],
                solver=solver,
            ).to_dict()
            for job_id in parse_job_ids(output)
        ]

    if records:
        print(f"  Jobs submitted: {', '.join(str(item['job_id']) for item in records)}")
    else:
        print("  Warning: no scheduler job was recorded")
    return _submission_state(
        example_path, mode, solver, run_directory, records, example_name
    )


def _existing_state(state_manager: ExampleCIState, state_path: Path) -> Optional[Dict]:
    return state_manager.read(state_path) if state_path.exists() else None


def submit_examples(
        pipeline_id: Optional[str] = None,
        version: Optional[str] = None,
        force: bool = False,
        dry_run: bool = False,
        failed_only: bool = False):
    version = version or os.environ.get("CI_COMMIT_TAG")
    print("=" * 60)
    print("MOLA CI/CD Examples - Submit Phase")
    print("=" * 60)
    if dry_run:
        print("\n[DRY-RUN MODE] No actual submission will occur.\n")

    state_manager = ExampleCIState()
    examples = sorted(Path("examples").rglob("run*.py"))
    submitted = skipped = overridden = 0

    for example_path in examples:
        mode = detect_execution_mode(example_path)
        if mode is None:
            print(f"Skipping {example_path} (no MOLA_CI_MODE marker)")
            skipped += 1
            continue

        print(f"\n{example_path} (Mode {mode})")
        state_path = state_manager.get_state_path(example_path)
        old_state = _existing_state(state_manager, state_path)
        directories = ExampleCIState.get_all_run_directories(old_state)
        action = "submit"
        message = "Ready for submission"
        directories_to_remove = []

        if failed_only:
            if not ExampleCIState.is_failed_state(old_state):
                print("  [SKIP] State is not marked FAILED")
                skipped += 1
                continue
            action = "override"
            directories_to_remove = ExampleCIState.get_failed_run_directories(old_state)
            message = "Submitting only failed case(s)"
        elif old_state and directories:
            machine = old_state.get("machine", "localhost")
            user = old_state.get("user")
            exists = any(SV.is_existing_path(
                directory, machine=machine, user=user, file_only=False
            ) for directory in directories)
            if exists:
                if ExampleCIState.is_failed_state(old_state):
                    action = "override"
                    directories_to_remove = ExampleCIState.get_failed_run_directories(old_state)
                    message = f"Auto-overriding failed case (status: {old_state.get('status')})"
                elif force:
                    action = "override"
                    directories_to_remove = directories
                    message = "Force override (user requested)"
                else:
                    action = "skip"
                    message = f"Exists with status '{old_state.get('status', 'UNKNOWN')}' (use -f to override)"

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
        if action == "override" and old_state and directories_to_remove:
            errors = ExampleCIState.remove_remote_run_directories(old_state, directories_to_remove)
            if errors:
                raise RuntimeError("Could not remove all remote run directories before submission")
            overridden += 1

        print(f"  Executing {mode} example: {example_path.name}")
        state_data = execute_example(example_path, mode)
        state_path = state_manager.create_state(example_path, mode, state_data)
        state_manager.update_state(state_path, {
            "ci_metadata": {
                "pipeline_id": pipeline_id,
                "version": version,
                "submitted_at": datetime.now().isoformat(),
                "last_checked_at": None,
            }
        })
        print(f"  Submitted, state saved to {state_path}")
        submitted += 1

    print("\n" + "=" * 60)
    print(f"Summary: {submitted} submitted, {overridden} overridden, {skipped} skipped")
    print("=" * 60)
