"""Tests for scheduler submission records."""

import pytest

from mola.server.submission import (
    Submission,
    emit_submission,
    parse_job_ids,
    parse_submission_events,
)


@pytest.mark.unit
def test_parse_multiple_scheduler_output_formats():
    output = """
Submitted batch job 101
Submit job 202
logger: Submitted batch job 101
"""

    assert parse_job_ids(output) == [101, 202]


@pytest.mark.unit
def test_yaml_submission_event_preserves_case_association(capsys):
    emit_submission(Submission.create(
        job_id=123,
        scheduler="SLURM",
        machine="juno",
        user="mola",
        run_directories=["/remote/case_1", "/remote/case_2"],
        solver="elsa",
    ))

    records = parse_submission_events(capsys.readouterr().out)
    assert records[0]["job_id"] == 123
    assert records[0]["run_directories"] == ["/remote/case_1", "/remote/case_2"]
