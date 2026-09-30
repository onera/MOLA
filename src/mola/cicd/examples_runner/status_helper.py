"""Small data helpers used by the examples runner."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List

import mola.naming_conventions as names


def unique(values: Iterable[Any]) -> list:
    result = []
    seen = set()
    for value in values:
        if value is None:
            continue
        value = str(value)
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def status_is_failed(status: Any) -> bool:
    return str(status or "").upper() in names.FAILED_STATUSES


def normalize_submission(submission: Dict[str, Any]) -> Dict[str, Any]:
    result = dict(submission)
    result["run_directories"] = unique(result.get("run_directories", []))
    result.setdefault("scheduler", "SLURM")
    result.setdefault("machine", "localhost")
    return result


def normalize_submissions(submissions: Iterable[Dict[str, Any]]) -> List[dict]:
    return [normalize_submission(item) for item in submissions if isinstance(item, dict)]


def overall_status(details: Iterable[Dict[str, Any]], submitted: bool = False) -> str:
    statuses = [str(detail.get("status", "ERROR")).upper() for detail in details]
    if not statuses:
        return "not_started"
    if any(status in names.FAILED_STATUSES for status in statuses):
        return "FAILED"
    if all(status == names.FILE_JOB_COMPLETED for status in statuses):
        return names.FILE_JOB_COMPLETED
    if any(status in {names.STATUS_PREPROCESS, names.STATUS_RUNNING} for status in statuses):
        return "RUNNING"
    if any(status == names.STATUS_PENDING for status in statuses):
        return names.STATUS_PENDING
    if submitted or any(status in names.RUNNING_STATUSES for status in statuses):
        return "RUNNING"
    if any(status == names.FILE_JOB_COMPLETED for status in statuses):
        return "RUNNING"
    return "not_started"
