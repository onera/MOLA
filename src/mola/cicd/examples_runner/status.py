"""Batched status resolution for submitted examples."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable

import mola.naming_conventions as names

from .status_helper import overall_status
from .state import ExampleCIState

from ...logging.formatters import colors

def query_multi_status_remote(sequence_jobs, machine=None, user=None):
    from mola.workflow.manager import WorkflowManager

    return WorkflowManager.get_simulation_status_remote(
        sequence_jobs=sequence_jobs,
        machine=machine,
        user=user,
    )


def _status_requests(state: Dict) -> list[dict]:
    submissions = ExampleCIState.get_submissions(state)
    if not submissions:
        return []
    requests = []
    for submission in submissions:
        request = {
            "job_id": submission.get("job_id"),
            "machine": submission.get("machine", state.get("machine", "localhost")),
            "user": submission.get("user", state.get("user")),
            "run_directories": submission.get("run_directories", []),
        }
        if submission.get("scheduler", "SLURM") != "SLURM":
            request["scheduler"] = submission["scheduler"]
        requests.append(request)
    return requests


def _details_for_requests(requests: list[dict], remote_details: list[dict]) -> list[dict]:
    by_directory = defaultdict(list)
    for detail in remote_details:
        directory = detail.get("run_directory")
        if directory is not None:
            by_directory[str(directory)].append(detail)

    details = []
    seen_directories = set()
    for request in requests:
        for run_directory in request["run_directories"]:
            if str(run_directory) in seen_directories:
                continue
            seen_directories.add(str(run_directory))
            candidates = by_directory.get(str(run_directory), [])
            detail = candidates[0] if candidates else None
            if detail is None:
                detail = {
                    "run_directory": run_directory,
                    "status": "ERROR",
                    "error": "Remote status query returned no result for this case",
                }
            details.append(detail)
    return details


def _aggregate_state(state: Dict, details: list[dict]):
    submitted = any(
        submission.get("job_id") is not None
        for submission in ExampleCIState.get_submissions(state)
    )
    aggregate = overall_status(details, submitted=submitted)
    if state.get("execution_mode") == "multi":
        return {"overall": aggregate, "details": details}
    return details[0].get("status", "ERROR") if len(details) == 1 else aggregate


def resolve_state_status(state: Dict):
    requests = _status_requests(state)
    if not requests:
        return {"overall": "not_started", "details": []} \
            if state.get("execution_mode") == "multi" else "not_started"

    machine = requests[0]["machine"]
    user = requests[0]["user"]
    remote_status = query_multi_status_remote(requests, machine=machine, user=user)
    details = _details_for_requests(requests, remote_status.get("details", []))
    return _aggregate_state(state, details)


def refresh_states(state_manager: ExampleCIState, state_paths: Iterable[Path]) -> Dict[Path, object]:
    """Resolve all states with one remote query per target environment."""
    loaded = {Path(path): state_manager.read(path) for path in state_paths}
    grouped = defaultdict(list)
    for state_path, state in loaded.items():
        requests = _status_requests(state)
        if not requests:
            continue
        for request in requests:
            key = (request["machine"], request["user"], request.get("solver"))
            grouped[key].append((state_path, request))

    details_by_state = defaultdict(list)
    for (machine, user, _solver), entries in grouped.items():
        requests = [request for _, request in entries]
        remote_status = query_multi_status_remote(requests, machine=machine, user=user)
        details = list(remote_status.get("details", []))
        by_directory = defaultdict(list)
        for detail in details:
            by_directory[str(detail.get("run_directory"))].append(detail)
        seen_directories_by_state = defaultdict(set)
        for state_path, request in entries:
            for run_directory in request["run_directories"]:
                if str(run_directory) in seen_directories_by_state[state_path]:
                    continue
                seen_directories_by_state[state_path].add(str(run_directory))
                candidates = by_directory.get(str(run_directory), [])
                detail = candidates[0] if candidates else {
                    "run_directory": run_directory,
                    "status": "ERROR",
                    "error": "Remote status query returned no result for this case",
                }
                details_by_state[state_path].append(detail)

    results = {}
    for state_path, state in loaded.items():
        details = details_by_state.get(state_path, [])
        result = _aggregate_state(state, details) if details else (
            {"overall": "not_started", "details": []}
            if state.get("execution_mode") == "multi" else "not_started"
        )
        results[state_path] = result
        updates = {
            "status": result.get("overall", "unknown") if isinstance(result, dict) else result,
            "ci_metadata": {
                **state.get("ci_metadata", {}),
                "last_checked_at": datetime.now().isoformat(),
            },
        }
        if isinstance(result, dict):
            updates["details"] = result["details"]
        state_manager.update_state(state_path, updates)
    return results


def _display_status(status):
    aggregate = status.get("overall", "unknown") if isinstance(status, dict) else status
    normalized = str(aggregate).upper()
    if normalized == names.FILE_JOB_COMPLETED:
        return "SUCCESS"
    if normalized in {names.FILE_JOB_FAILED, names.FILE_ERROR_PREPARING_WORKFLOW, "ERROR"}:
        return "FAILED"
    if normalized == "NOT_STARTED" or normalized in names.RUNNING_STATUSES:
        return "RUNNING"
    return normalized


def _display_case_status(status):
    """Display status for an individual case without collapsing intermediate
    states (READY, PREPROCESS, TO_PREPARE, etc.) into RUNNING.

    Only COMPLETED and terminal failure states are mapped to SUCCESS/FAILED;
    all other statuses are returned as-is (uppercased) to preserve accuracy.
    """
    normalized = str(status).upper()
    if normalized == names.FILE_JOB_COMPLETED:
        return "SUCCESS"
    if normalized in {names.FILE_JOB_FAILED, names.FILE_ERROR_PREPARING_WORKFLOW, "ERROR"}:
        return "FAILED"
    return normalized


def _summary_rows(state_manager: ExampleCIState, results):
    rows = []
    for state_path, status in results.items():
        state = state_manager.read(state_path)
        details = status.get("details", []) if isinstance(status, dict) else []
        if details:
            rows.extend(
                (
                    detail.get("run_directory", state.get("example_path", "unknown")),
                    _display_case_status(detail.get("status", "ERROR")),
                )
                for detail in details
            )
            continue

        run_directories = ExampleCIState.get_all_run_directories(state)
        if run_directories:
            rows.extend(
                (run_directory, _display_case_status(status))
                for run_directory in run_directories
            )
        else:
            rows.append((state.get("example_path", "unknown"), _display_case_status(status)))
    return rows


def _print_summary(state_manager: ExampleCIState, results):
    rows = _summary_rows(state_manager, results)
    print("\n" + "=" * 60)
    print("EXAMPLE STATUS SUMMARY")
    print("=" * 60)
    width = max((len(str(path)) for path, _ in rows), default=20)
    print(f"{'Run directory':<{width}}  Status")
    print("-" * (width + 8))
    for path, status in rows:
        if status == 'SUCCESS':
            status = colors["GREEN"]+status+colors["ENDC"]
        elif status == 'FAILED':
            status = colors["RED"]+status+colors["ENDC"]
        elif status in ('TO_PREPARE','PREPROCESS'):
            status = colors["YELLOW"]+status+colors["ENDC"]
        elif status == 'RUNNING':
            status = colors["CYAN"]+status+colors["ENDC"]

        print(f"{str(path):<{width}}  {status}")
    print("=" * 60)


def check_status(state_dir=None):
    state_manager = ExampleCIState(state_dir)
    state_paths = sorted(
        state_manager.state_dir.glob(f"**/*{names.FILE_CI_STATE_SUFFIX}")
    )
    if not state_paths:
        print(f"No CI state files found in {state_manager.state_dir}. Run submit first.")
        return "retry"

    results = refresh_states(state_manager, state_paths)
    _print_summary(state_manager, results)
    statuses = [_display_status(status) for status in results.values()]
    if "FAILED" in statuses:
        print("\n❌ Some examples failed.")
        return "fail" if all(status != "RUNNING" for status in statuses) else "retry"
    if "RUNNING" in statuses:
        print("\n⚠️  Some examples are still running or pending.")
        return "retry"
    print("\n✅ All examples completed successfully!")
    return "continue"


def validate_examples(state_dir=None):
    print('scanning examples for monitoring their state...')
    state_manager = ExampleCIState(state_dir)
    state_paths = sorted(
        state_manager.state_dir.glob(f"**/*{names.FILE_CI_STATE_SUFFIX}")
    )
    if not state_paths:
        print("\nNo CI state files found. Run submit first.")
        raise SystemExit(2)

    results = refresh_states(state_manager, state_paths)
    _print_summary(state_manager, results)
    statuses = [_display_status(status) for status in results.values()]
    if "RUNNING" in statuses:
        print("\n⚠️  Some examples are still running or pending.")
        raise SystemExit(2)
    if "FAILED" in statuses:
        print("\n❌ Some examples failed.")
        raise SystemExit(1)
    print("\n✅ All examples completed successfully!")
    raise SystemExit(0)
