"""Helpers for aggregating live example statuses."""

from __future__ import annotations

from typing import Dict, Iterable

import mola.naming_conventions as names


def overall_status(details: Iterable[Dict], submitted: bool = False) -> str:
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
