"""Live status resolution for marked examples."""

from __future__ import annotations

from collections import defaultdict

import mola.naming_conventions as names

from ...logging.formatters import colors
from .discovery import (
    discover_remote_layout,
    make_case_requests,
    scan_example_definitions,
)
from .status_helper import overall_status


def query_multi_status_remote(sequence_jobs, machine=None, user=None, solver=None):
    from mola.workflow.manager import WorkflowManager

    kwargs = {
        "sequence_jobs": sequence_jobs,
        "machine": machine,
        "user": user,
    }
    if solver is not None:
        kwargs["solver"] = solver
    return WorkflowManager.get_simulation_status_remote(**kwargs)


def _live_overall_status(details):
    scheduler_active = any(
        detail.get("scheduler_status") in {names.STATUS_RUNNING, names.STATUS_PENDING}
        for detail in details
    )
    return overall_status(details, submitted=scheduler_active)


def resolve_live_statuses(
        examples_root="examples",
        source_user=None,
        connection_user=None,
):
    """Discover and resolve all marked examples without local state."""

    definitions = scan_example_definitions(
        examples_root=examples_root,
        source_user=source_user,
    )
    layout = discover_remote_layout(
        definitions,
        connection_user=connection_user,
    )
    for machine, error in layout.get("scheduler_errors", {}).items():
        print(f"Warning: squeue query failed on {machine}: {error}")

    requests = make_case_requests(definitions, layout)
    for request in requests:
        request["user"] = connection_user

    details_by_example = defaultdict(list)
    grouped = defaultdict(list)
    definitions_by_id = {definition.identifier: definition for definition in definitions}
    for request in requests:
        definition = definitions_by_id[request["example_id"]]
        grouped[(definition.machine, request.get("user"), definition.solver)].append(request)

    for (machine, user, solver), machine_requests in grouped.items():
        response = query_multi_status_remote(
            machine_requests,
            machine=machine,
            user=user,
            solver=solver,
        )
        for detail in response.get("details", []):
            details_by_example[detail.get("example_id")].append(detail)

    results = {}
    for definition in definitions:
        details = details_by_example.get(definition.identifier, [])
        if not details:
            details = [{
                "example_id": definition.identifier,
                "run_directory": definition.root_directory,
                "status": names.STATUS_TO_PREPARE,
                "scheduler_status": "",
            }]
        results[definition.identifier] = {
            "definition": definition,
            "overall": _live_overall_status(details),
            "details": details,
        }
    return results


def _display_status(status):
    normalized = str(status.get("overall", "unknown")).upper()
    if normalized == names.FILE_JOB_COMPLETED:
        return "SUCCESS"
    if normalized in {names.FILE_JOB_FAILED, names.FILE_ERROR_PREPARING_WORKFLOW, "ERROR"}:
        return "FAILED"
    if normalized == "NOT_STARTED" or normalized in names.RUNNING_STATUSES:
        return "RUNNING"
    return normalized


def _display_case_status(status):
    """Keep individual intermediate statuses visible in the summary."""

    normalized = str(status).upper()
    if normalized == names.FILE_JOB_COMPLETED:
        return "SUCCESS"
    if normalized in {names.FILE_JOB_FAILED, names.FILE_ERROR_PREPARING_WORKFLOW, "ERROR"}:
        return "FAILED"
    return normalized


def _print_live_summary(results):
    rows = []
    for result in results.values():
        definition = result["definition"]
        details = result.get("details", [])
        if details:
            rows.extend(
                (
                    detail.get("run_directory", definition.script_path),
                    _display_case_status(detail.get("status", "ERROR")),
                )
                for detail in details
            )
        else:
            rows.append((definition.script_path, _display_case_status(result["overall"])))

    print("\n" + "=" * 60)
    print("EXAMPLE STATUS SUMMARY")
    print("=" * 60)
    width = max((len(str(path)) for path, _ in rows), default=20)
    print(f"{'Run directory':<{width}}  Status")
    print("-" * (width + 8))
    for path, status in rows:
        if status == "SUCCESS":
            status = colors["GREEN"] + status + colors["ENDC"]
        elif status == "FAILED":
            status = colors["RED"] + status + colors["ENDC"]
        elif status in ("TO_PREPARE", "PREPROCESS"):
            status = colors["YELLOW"] + status + colors["ENDC"]
        elif status == "RUNNING":
            status = colors["CYAN"] + status + colors["ENDC"]
        print(f"{str(path):<{width}}  {status}")
    print("=" * 60)


def _live_display_status(result):
    return _display_status(result)


def check_status(source_user=None, examples_root="examples"):
    results = resolve_live_statuses(
        examples_root=examples_root,
        source_user=source_user,
    )
    if not results:
        print(f"No marked examples found in {examples_root}.")
        return "retry"

    _print_live_summary(results)
    statuses = [_live_display_status(result) for result in results.values()]
    if "FAILED" in statuses:
        print("\n❌ Some examples failed.")
        return "fail" if all(status != "RUNNING" for status in statuses) else "retry"
    if "RUNNING" in statuses:
        print("\n⚠️  Some examples are still running or pending.")
        return "retry"
    print("\n✅ All examples completed successfully!")
    return "continue"


def validate_examples(source_user=None, examples_root="examples"):
    print("scanning examples for monitoring their state...")
    results = resolve_live_statuses(
        examples_root=examples_root,
        source_user=source_user,
    )
    if not results:
        print(f"\nNo marked examples found in {examples_root}.")
        raise SystemExit(2)

    _print_live_summary(results)
    statuses = [_live_display_status(result) for result in results.values()]
    if "RUNNING" in statuses:
        print("\n⚠️  Some examples are still running or pending.")
        raise SystemExit(2)
    if "FAILED" in statuses:
        print("\n❌ Some examples failed.")
        raise SystemExit(1)
    print("\n✅ All examples completed successfully!")
    raise SystemExit(0)
