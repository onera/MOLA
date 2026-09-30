"""Submission records shared by workflow execution and CI tooling."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Iterable, Optional, Tuple, Union

import yaml

import mola.naming_conventions as names

_JOB_ID_PATTERNS = (
    re.compile(r"submitted\s+batch\s+job\s+(\d+(?:_[\w-]+)?)", re.IGNORECASE),
    re.compile(r"submit(?:ted)?\s+job\s+(\d+(?:_[\w-]+)?)", re.IGNORECASE),
)

JobId = Union[int, str]


def parse_job_ids(output: str) -> list[JobId]:
    matches = []
    for pattern in _JOB_ID_PATTERNS:
        matches.extend(match.group(1) for match in pattern.finditer(output or ""))
    unique = []
    seen = set()
    for value in matches:
        if value in seen:
            continue
        seen.add(value)
        unique.append(int(value) if value.isdigit() else value)
    return unique


def _normalize_job_id(job_id: Optional[JobId]) -> Optional[JobId]:
    return int(job_id) if isinstance(job_id, str) and job_id.isdigit() else job_id


@dataclass(frozen=True)
class Submission:
    job_id: Optional[JobId]
    scheduler: str
    machine: str
    user: Optional[str]
    run_directories: Tuple[str, ...]
    solver: Optional[str] = None
    sequence_id: Optional[str] = None
    submitted_at: str = ""

    @classmethod
    def create(
        cls,
        job_id: Optional[JobId],
        scheduler: str,
        machine: str,
        user: Optional[str],
        run_directories: Iterable[str],
        solver: Optional[str] = None,
        sequence_id: Optional[str] = None,
    ) -> "Submission":
        return cls(
            job_id=_normalize_job_id(job_id),
            scheduler=scheduler,
            machine=machine,
            user=user,
            run_directories=tuple(str(path) for path in run_directories),
            solver=solver,
            sequence_id=sequence_id,
            submitted_at=datetime.now().isoformat(),
        )

    def to_dict(self) -> dict:
        data = asdict(self)
        data["job_id"] = _normalize_job_id(data["job_id"])
        data["run_directories"] = list(self.run_directories)
        return data


def emit_submission(submission: Submission) -> None:
    payload = yaml.safe_dump(
        submission.to_dict(),
        default_flow_style=True,
        sort_keys=False,
        width=4096,
    ).strip()
    print(f"{names.SUBMISSION_EVENT_PREFIX}{payload}", flush=True)


def parse_submission_events(output: str) -> list[dict]:
    submissions = []
    for line in (output or "").splitlines():
        if not line.startswith(names.SUBMISSION_EVENT_PREFIX):
            continue
        payload = line[len(names.SUBMISSION_EVENT_PREFIX):].strip()
        try:
            submission = yaml.safe_load(payload)
        except yaml.YAMLError:
            continue
        if isinstance(submission, dict):
            submissions.append(submission)
    return submissions
