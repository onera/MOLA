"""Tests for state-free remote example discovery."""

import ast
from pathlib import Path
from unittest.mock import patch

import pytest

from mola.cicd.examples_runner.discovery import (
    ExampleDefinition,
    _remote_probe_code,
    discover_remote_layout,
    make_case_requests,
)


@pytest.mark.unit
def test_remote_probe_code_is_valid_python():
    ast.parse(_remote_probe_code([], "lbernard"))


@pytest.mark.unit
def test_discovery_matches_sequence_workdir_to_slurm_job():
    definition = ExampleDefinition(
        identifier="example",
        script_path=Path("examples/example/run.py"),
        mode="multi",
        solver="elsa",
        example_name="example",
        root_directory="/tmp_user/sator/lbernard/mola_examples/example",
        run_directory="/tmp_user/sator/lbernard/mola_examples/example",
        machine="sator",
        source_user="lbernard",
    )
    sequence_root = definition.root_directory + "/sequence"
    response = {
        "exists": {definition.identifier: True},
        "cases": [{
            "example_id": definition.identifier,
            "run_directory": sequence_root + "/case1",
            "sequence_root": sequence_root,
            "scheduler_workdir": sequence_root,
        }],
        "jobs": [],
        "scheduler_errors": {},
    }
    with patch(
            "mola.cicd.examples_runner.discovery._run_remote_probe",
            return_value={
                **response,
                "jobs": [{
                    "state": "R",
                    "job_id": "123",
                    "elapsed": "00:10",
                    "work_directory": sequence_root,
                }],
            }):
        layout = discover_remote_layout([definition], connection_user="mola")

    requests = make_case_requests([definition], layout)
    assert requests[0]["scheduler_status"] == "RUNNING"
    assert requests[0]["scheduler_job_id"] == "123"
    assert requests[0]["sequence_root"] == sequence_root
