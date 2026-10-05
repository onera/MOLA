"""Tests for live examples-runner discovery and status handling."""

from pathlib import Path
from unittest.mock import patch

import pytest

from mola.cicd.examples_runner import clear_examples, detect_execution_mode, main, submit_examples
from mola.cicd.examples_runner.cli import build_parser
from mola.cicd.examples_runner.discovery import ExampleDefinition
from mola.cicd.examples_runner.status import (
    _display_case_status,
    _print_live_summary,
    resolve_live_statuses,
)


def _definition(identifier, root, mode="mono"):
    return ExampleDefinition(
        identifier=identifier,
        script_path=Path(f"examples/{identifier}/run.py"),
        mode=mode,
        solver="elsa",
        example_name=identifier,
        root_directory=root,
        run_directory=root,
        machine="juno",
        source_user="lbernard",
    )


@pytest.mark.unit
def test_detect_execution_mode(tmp_path):
    example = tmp_path / "run_example.py"
    example.write_text("# MOLA_CI_MODE: \"multi\"\n", encoding="utf-8")
    assert detect_execution_mode(example) == "multi"


@pytest.mark.unit
def test_live_status_resolution_uses_discovered_cases_and_batches_queries():
    definitions = [_definition("mono", "/remote/mono")]
    layout = {
        "cases": [{
            "example_id": "mono",
            "run_directory": "/remote/mono",
            "sequence_root": None,
            "scheduler_workdir": "/remote/mono",
        }],
        "jobs_by_workdir": {},
        "scheduler_errors": {},
    }
    with patch(
            "mola.cicd.examples_runner.status.scan_example_definitions",
            return_value=definitions), \
            patch(
                "mola.cicd.examples_runner.status.discover_remote_layout",
                return_value=layout), \
            patch(
                "mola.cicd.examples_runner.status.query_multi_status_remote",
                return_value={"details": [{
                    "example_id": "mono",
                    "run_directory": "/remote/mono",
                    "status": "COMPLETED",
                    "scheduler_status": "",
                }]}) as query:
        results = resolve_live_statuses(connection_user="mola")

    assert results["mono"]["overall"] == "COMPLETED"
    query.assert_called_once()
    assert query.call_args.kwargs == {
        "machine": "juno",
        "user": "mola",
        "solver": "elsa",
    }


@pytest.mark.unit
def test_status_summary_preserves_individual_case_statuses(capsys):
    definition = _definition("multi", "/remote/multi", mode="multi")
    _print_live_summary({
        "multi": {
            "definition": definition,
            "overall": "RUNNING",
            "details": [
                {"run_directory": "/remote/multi/case1", "status": "COMPLETED"},
                {"run_directory": "/remote/multi/case2", "status": "READY"},
                {"run_directory": "/remote/multi/case3", "status": "PREPROCESS"},
            ],
        },
    })
    output = capsys.readouterr().out
    assert "SUCCESS" in output
    assert "READY" in output
    assert "PREPROCESS" in output
    assert "RUNNING" not in output


@pytest.mark.unit
def test_display_case_status_preserves_intermediate_states():
    assert _display_case_status("COMPLETED") == "SUCCESS"
    assert _display_case_status("FAILED") == "FAILED"
    assert _display_case_status("READY") == "READY"
    assert _display_case_status("PREPROCESS") == "PREPROCESS"
    assert _display_case_status("PENDING") == "PENDING"


@pytest.mark.unit
def test_clear_local_only_removes_release_and_generated_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    generated = tmp_path / "examples" / "nested"
    generated.mkdir(parents=True)
    (generated / "workflow_manager.yaml").write_text("generated")
    (generated / "gather_results.py").write_text("generated")
    release = tmp_path / ".release"
    release.mkdir()

    with patch("mola.cicd.examples_runner.cleanup.SV.remove_path") as remove_path:
        assert clear_examples(local_only=True)

    assert not release.exists()
    assert not (generated / "workflow_manager.yaml").exists()
    assert not (generated / "gather_results.py").exists()
    remove_path.assert_not_called()


@pytest.mark.unit
def test_clear_all_discovers_remote_roots(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    definition = _definition("mono", "/remote/mono")
    (tmp_path / ".release").mkdir()
    with patch(
            "mola.cicd.examples_runner.cleanup.scan_example_definitions",
            return_value=[definition]), \
            patch("mola.cicd.examples_runner.cleanup.SV.remove_path") as remove_path:
        assert clear_examples(all_cases=True)

    remove_path.assert_called_once_with(
        "/remote/mono", machine="juno", user=None, file_only=False
    )
    assert not (tmp_path / ".release").exists()


@pytest.mark.unit
def test_submit_failed_only_uses_live_statuses():
    failed = _definition("failed", "/remote/failed")
    successful = _definition("successful", "/remote/successful")
    layout = {
        "exists": {failed.identifier: True, successful.identifier: True},
        "cases": [],
        "jobs_by_workdir": {},
        "scheduler_errors": {},
    }
    live_results = {
        failed.identifier: {
            "overall": "FAILED",
            "details": [{"run_directory": "/remote/failed", "status": "FAILED"}],
        },
        successful.identifier: {
            "overall": "COMPLETED",
            "details": [{"run_directory": "/remote/successful", "status": "COMPLETED"}],
        },
    }
    with patch(
            "mola.cicd.examples_runner.submit.scan_example_definitions",
            return_value=[failed, successful]), \
            patch(
                "mola.cicd.examples_runner.submit.discover_remote_layout",
                return_value=layout), \
            patch(
                "mola.cicd.examples_runner.status.resolve_live_statuses",
                return_value=live_results), \
            patch("mola.cicd.examples_runner.submit.SV.remove_path") as remove_path, \
            patch("mola.cicd.examples_runner.submit.execute_example") as execute_example:
        submit_examples(failed_only=True)

    remove_path.assert_called_once_with(
        "/remote/failed", machine="juno", user=None, file_only=False
    )
    execute_example.assert_called_once_with(failed.script_path, "mono")


@pytest.mark.unit
def test_cli_uses_state_free_arguments(monkeypatch):
    with patch("mola.cicd.examples_runner.cli.submit_examples") as submit:
        main(["submit", "--failed-only"])
    submit.assert_called_once_with(
        force=False, dry_run=False, failed_only=True, source_user=None
    )

    parsed = build_parser().parse_args(["clear", "--local-only"])
    assert parsed.local_only is True
