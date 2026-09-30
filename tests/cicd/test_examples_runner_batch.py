"""Tests for cross-example status batching."""

from pathlib import Path
from unittest.mock import patch

import pytest

from mola.cicd.examples_runner import ExampleCIState
from mola.cicd.examples_runner.status import _print_summary, _display_case_status, refresh_states


@pytest.mark.unit
def test_refresh_states_uses_one_query_for_one_machine(tmp_path):
    state_manager = ExampleCIState(tmp_path / "states")
    paths = []
    for index in (1, 2):
        example = tmp_path / f"run_{index}.py"
        example.write_text("# MOLA_CI_MODE: mono\n", encoding="utf-8")
        paths.append(state_manager.create_state(example, "mono", {
            "machine": "juno",
            "user": "mola",
            "solver": "elsa",
            "submissions": [{
                "job_id": index,
                "machine": "juno",
                "user": "mola",
                "run_directories": [f"/remote/case_{index}"],
            }],
        }))

    response = {
        "details": [
            {"run_directory": "/remote/case_1", "status": "COMPLETED"},
            {"run_directory": "/remote/case_2", "status": "COMPLETED"},
        ]
    }
    with patch(
            "mola.cicd.examples_runner.status.query_multi_status_remote",
            return_value=response) as query:
        results = refresh_states(state_manager, paths)

    query.assert_called_once()
    assert len(query.call_args.args[0]) == 2
    assert all(result == "COMPLETED" for result in results.values())


@pytest.mark.unit
def test_status_summary_displays_run_directories_and_multi_details(tmp_path, capsys):
    state_manager = ExampleCIState(tmp_path / "states")
    mono_example = tmp_path / "mono.py"
    multi_example = tmp_path / "multi.py"
    mono_path = state_manager.create_state(mono_example, "mono", {
        "run_directory": "/remote/mono",
    })
    multi_path = state_manager.create_state(multi_example, "multi", {
        "run_directories": ["/remote/multi/one", "/remote/multi/two"],
    })

    _print_summary(state_manager, {
        mono_path: "COMPLETED",
        multi_path: {
            "overall": "RUNNING",
            "details": [
                {"run_directory": "/remote/multi/one", "status": "COMPLETED"},
                {"run_directory": "/remote/multi/two", "status": "PREPROCESS"},
            ],
        },
    })

    output = capsys.readouterr().out
    assert "/remote/mono" in output
    assert "/remote/multi/one" in output
    assert "/remote/multi/two" in output
    assert str(mono_path) not in output
    assert str(multi_path) not in output


@pytest.mark.unit
def test_display_case_status_preserves_intermediate_states():
    assert _display_case_status("COMPLETED") == "SUCCESS"
    assert _display_case_status("FAILED") == "FAILED"
    assert _display_case_status("ERROR") == "FAILED"
    assert _display_case_status("READY") == "READY"
    assert _display_case_status("PREPROCESS") == "PREPROCESS"
    assert _display_case_status("RUNNING") == "RUNNING"
    assert _display_case_status("TO_PREPARE") == "TO_PREPARE"
    assert _display_case_status("NOT_STARTED") == "NOT_STARTED"
    assert _display_case_status("PENDING") == "PENDING"
    assert _display_case_status("QUEUED") == "QUEUED"
    assert _display_case_status("TO_LAUNCH") == "TO_LAUNCH"
    assert _display_case_status("NEWJOB_REQUIRED") == "NEWJOB_REQUIRED"


@pytest.mark.unit
def test_print_summary_shows_individual_case_statuses(tmp_path, capsys):
    state_manager = ExampleCIState(tmp_path / "states")
    multi_example = tmp_path / "multi.py"
    multi_path = state_manager.create_state(multi_example, "multi", {
        "run_directories": ["/remote/multi/one", "/remote/multi/two"],
    })

    _print_summary(state_manager, {
        multi_path: {
            "overall": "RUNNING",
            "details": [
                {"run_directory": "/remote/multi/one", "status": "COMPLETED"},
                {"run_directory": "/remote/multi/two", "status": "READY"},
                {"run_directory": "/remote/multi/three", "status": "PREPROCESS"},
                {"run_directory": "/remote/multi/four", "status": "NOT_STARTED"},
            ],
        },
    })

    output = capsys.readouterr().out
    # COMPLETED maps to SUCCESS
    assert "SUCCESS" in output
    # Intermediate states should NOT be collapsed to RUNNING
    assert "READY" in output
    assert "PREPROCESS" in output
    assert "NOT_STARTED" in output
    # Old behaviour: RUNNING should not appear for intermediate states
    assert output.count("RUNNING") == 0
