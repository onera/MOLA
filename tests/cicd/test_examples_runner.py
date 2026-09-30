#!/usr/bin/env python3
"""
Unit tests for mola.cicd.examples_runner
"""

import pytest
import os
import sys
import yaml
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

from mola.cicd.examples_runner import (
    ExampleCIState,
    clear_examples,
    detect_execution_mode,
    main,
    submit_examples,
)
from mola.cicd.examples_runner.cli import build_parser


@pytest.fixture
def temp_state_dir():
    """Create a temporary directory for state files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def temp_example_file():
    """Create a temporary example file with MOLA_CI_MODE marker."""
    with tempfile.TemporaryDirectory() as tmpdir:
        example_path = Path(tmpdir) / "test_example.py"
        example_path.write_text("""
# Some example code
w = Workflow()
w.prepare()
w.submit()

# MOLA_CI_MODE: mono
""")
        yield example_path


@pytest.mark.unit
def test_example_ci_state_init(temp_state_dir):
    """Test ExampleCIState initialization."""
    state_manager = ExampleCIState(state_dir=str(temp_state_dir))
    assert state_manager.state_dir.exists()


@pytest.mark.unit
def test_get_state_path(temp_state_dir, temp_example_file):
    """Test get_state_path method."""
    state_manager = ExampleCIState(state_dir=str(temp_state_dir))
    state_path = state_manager.get_state_path(temp_example_file)
    
    assert state_path.suffix == ".yaml"
    assert ".ci_state" in state_path.name
    assert state_path.parent.parent == temp_state_dir / "_external"


@pytest.mark.unit
def test_create_state(temp_state_dir, temp_example_file):
    """Test create_state method."""
    state_manager = ExampleCIState(state_dir=str(temp_state_dir))
    state_data = {
        "run_directory": "/tmp/test_run",
        "machine": "juno",
        "jobs": {"cfd": []}
    }
    
    state_path = state_manager.create_state(temp_example_file, "mono", state_data)
    
    assert state_path.exists()
    
    with open(state_path) as f:
        state = yaml.safe_load(f)
    
    assert state["example_path"] == str(temp_example_file)
    assert state["execution_mode"] == "mono"
    assert state["run_directory"] == "/tmp/test_run"
    assert state["machine"] == "juno"
    assert "ci_metadata" in state


@pytest.mark.unit
def test_update_state(temp_state_dir, temp_example_file):
    """Test update_state method."""
    state_manager = ExampleCIState(state_dir=str(temp_state_dir))
    state_data = {
        "run_directory": "/tmp/test_run",
        "machine": "juno",
        "jobs": {"cfd": []}
    }
    
    state_path = state_manager.create_state(temp_example_file, "mono", state_data)
    
    # Update the state
    state_manager.update_state(state_path, {
        "ci_metadata": {
            "pipeline_id": "12345",
            "version": "v1.0"
        }
    })
    
    with open(state_path) as f:
        state = yaml.safe_load(f)
    
    assert state["ci_metadata"]["pipeline_id"] == "12345"
    assert state["ci_metadata"]["version"] == "v1.0"


@pytest.mark.unit
def test_detect_execution_mode_with_marker(temp_example_file):
    """Test detect_execution_mode with a file that has a marker."""
    mode = detect_execution_mode(temp_example_file)
    assert mode == "mono"


@pytest.mark.unit
def test_detect_execution_mode_without_marker(temp_state_dir):
    """Test detect_execution_mode with a file that has no marker."""
    example_path = temp_state_dir / "no_marker.py"
    example_path.write_text("""
# Some example code
w = Workflow()
w.prepare()
w.submit()
""")
    
    mode = detect_execution_mode(example_path)
    assert mode is None




@pytest.mark.unit
def test_detect_execution_mode_multi(temp_state_dir):
    """Test detect_execution_mode with a multi marker."""
    example_path = temp_state_dir / "multi_example.py"
    example_path.write_text("""
# Some example code
manager = WorkflowManager(w)
manager.submit()

# MOLA_CI_MODE: "multi"
""")
    
    mode = detect_execution_mode(example_path)
    assert mode == "multi"


@pytest.mark.unit
def test_create_state_with_all_modes(temp_state_dir, temp_example_file):
    """Test creating state files for all execution modes."""
    state_manager = ExampleCIState(state_dir=str(temp_state_dir))
    state_data = {
        "run_directory": "/tmp/test_run",
        "machine": "juno",
        "jobs": {"cfd": []}
    }
    
    # Test all current modes
    for mode in ["mono", "multi"]:
        state_path = state_manager.create_state(temp_example_file, mode, state_data)
        
        with open(state_path) as f:
            state = yaml.safe_load(f)
        
        assert state["execution_mode"] == mode


@pytest.mark.unit
def test_update_state_with_job_ids(temp_state_dir, temp_example_file):
    """Test update_state with job IDs."""
    state_manager = ExampleCIState(state_dir=str(temp_state_dir))
    state_data = {
        "run_directory": "/tmp/test_run",
        "machine": "juno",
        "jobs": {"cfd": [12345]}
    }
    
    state_path = state_manager.create_state(temp_example_file, "mono", state_data)
    
    # Update with new job IDs
    state_manager.update_state(state_path, {
        "jobs": {"cfd": [12346, 12347]}
    })
    
    with open(state_path) as f:
        state = yaml.safe_load(f)
    
    # Should have all job IDs
    assert 12345 in state["jobs"]["cfd"]
    assert 12346 in state["jobs"]["cfd"]
    assert 12347 in state["jobs"]["cfd"]


@pytest.mark.unit
def test_create_state_creates_directory(temp_state_dir):
    """Test that create_state creates the state directory if it doesn't exist."""
    new_state_dir = temp_state_dir / "new_state_dir"
    state_manager = ExampleCIState(state_dir=str(new_state_dir))
    
    assert new_state_dir.exists()


@pytest.mark.unit
def test_update_state_nonexistent_file(temp_state_dir):
    """Test that update_state raises FileNotFoundError for nonexistent file."""
    state_manager = ExampleCIState(state_dir=str(temp_state_dir))
    nonexistent_path = temp_state_dir / "nonexistent.ci_state.yaml"
    
    with pytest.raises(FileNotFoundError):
        state_manager.update_state(nonexistent_path, {"key": "value"})


@pytest.mark.unit
def test_multi_status_uses_state_run_directories_and_persists_result(temp_state_dir, temp_example_file):
    """Status must enumerate cases from local state and refresh that state."""
    state_manager = ExampleCIState(state_dir=str(temp_state_dir))
    state_path = state_manager.create_state(
        temp_example_file,
        "multi",
        {
            "root_directory": "/remote/root",
            "script_directory": "/local/example",
            "machine": "juno",
            "user": "mola",
            "run_directories": ["/remote/root/case_1", "/remote/root/case_2"],
            "jobs": {"sequential": [12345]},
            "status": "submitted",
        },
    )

    with patch(
            "mola.cicd.examples_runner.status.query_multi_status_remote",
            return_value={"details": [
                {"run_directory": "/remote/root/case_1", "status": "COMPLETED"},
                {"run_directory": "/remote/root/case_2", "status": "PREPROCESS"},
            ]}) as query_status:
        status = state_manager.check_status(state_path)

    assert status["overall"] == "RUNNING"
    assert status["details"] == [
        {"run_directory": "/remote/root/case_1", "status": "COMPLETED"},
        {"run_directory": "/remote/root/case_2", "status": "PREPROCESS"},
    ]
    query_status.assert_called_once_with(
        [{
            "job_id": 12345,
            "run_directories": [
                "/remote/root/case_1", "/remote/root/case_2"],
            "machine": "juno",
            "user": "mola",
        }],
        machine="juno",
        user="mola",
    )
    with open(state_path) as file:
        refreshed_state = yaml.safe_load(file)
    assert refreshed_state["status"] == "RUNNING"
    assert refreshed_state["details"] == status["details"]


@pytest.mark.unit
def test_multi_status_manager_fallback_accepts_workflow_manager_yaml_keys(temp_state_dir):
    """Legacy states can recover cases from WorkflowManager's capitalized key."""
    state_manager = ExampleCIState(state_dir=str(temp_state_dir))

    def copy_manager_file(destination_path, **kwargs):
        Path(destination_path).write_text(yaml.safe_dump({
            "RunDirectories": ["/remote/root/case_1", "/remote/root/case_2"]
        }))

    with patch("mola.cicd.examples_runner.SV.is_existing_path", return_value=True), \
            patch("mola.cicd.examples_runner.SV.copy_remote", side_effect=copy_manager_file):
        run_directories = state_manager._get_run_directories_from_manager_file(
            "/remote/root", "/local/example", "juno", "mola")

    assert run_directories == ["/remote/root/case_1", "/remote/root/case_2"]


@pytest.mark.unit
def test_multi_status_preserves_multiple_sequence_job_groups(temp_state_dir, temp_example_file):
    """Each sequence job must be queried with its own case group."""
    state_manager = ExampleCIState(state_dir=str(temp_state_dir))
    state_path = state_manager.create_state(
        temp_example_file,
        "multi",
        {
            "root_directory": "/remote/root",
            "machine": "juno",
            "user": "mola",
            "run_directories": [
                "/remote/root/sequence_1/case_1",
                "/remote/root/sequence_1/case_2",
                "/remote/root/sequence_2/case_1",
            ],
            "jobs": {
                "sequential": [101, 202],
                "sequences": [
                    {
                        "job_id": 101,
                        "machine": "juno",
                        "user": "mola",
                        "run_directories": [
                            "/remote/root/sequence_1/case_1",
                            "/remote/root/sequence_1/case_2",
                        ],
                    },
                    {
                        "job_id": 202,
                        "machine": "juno",
                        "user": "mola",
                        "run_directories": ["/remote/root/sequence_2/case_1"],
                    },
                ],
            },
        },
    )

    details = [
        {"run_directory": "/remote/root/sequence_1/case_1", "status": "COMPLETED"},
        {"run_directory": "/remote/root/sequence_1/case_2", "status": "READY"},
        {"run_directory": "/remote/root/sequence_2/case_1", "status": "PREPROCESS"},
    ]
    with patch(
            "mola.cicd.examples_runner.status.query_multi_status_remote",
            return_value={"details": details}) as query_status:
        status = state_manager.check_status(state_path)

    assert status["overall"] == "RUNNING"
    query_status.assert_called_once_with(
        [
            {
                "job_id": 101,
                "machine": "juno",
                "user": "mola",
                "run_directories": [
                    "/remote/root/sequence_1/case_1",
                    "/remote/root/sequence_1/case_2",
                ],
            },
            {
                "job_id": 202,
                "machine": "juno",
                "user": "mola",
                "run_directories": ["/remote/root/sequence_2/case_1"],
            },
        ],
        machine="juno",
        user="mola",
    )


@pytest.mark.unit
def test_clear_failed_only_removes_failed_mono_and_multi_cases(tmp_path, monkeypatch):
    """Failed cleanup targets only failed remote case directories."""
    monkeypatch.chdir(tmp_path)
    state_directory = tmp_path / ".release" / "example_states"
    state_directory.mkdir(parents=True)

    (state_directory / "mono.ci_state.yaml").write_text(yaml.safe_dump({
        "execution_mode": "mono",
        "status": "FAILED",
        "run_directory": "/remote/mono-failed",
        "machine": "juno",
        "user": "mola",
    }))
    (state_directory / "multi.ci_state.yaml").write_text(yaml.safe_dump({
        "execution_mode": "multi",
        "status": "FAILED",
        "run_directories": ["/remote/multi-failed", "/remote/multi-completed"],
        "details": [
            {"run_directory": "/remote/multi-failed", "status": "FAILED"},
            {"run_directory": "/remote/multi-completed", "status": "COMPLETED"},
        ],
        "machine": "juno",
        "user": "mola",
    }))

    with patch("mola.cicd.examples_runner.SV.remove_path") as remove_path:
        assert clear_examples(failed_only=True)

    assert remove_path.call_args_list == [
        (('/remote/mono-failed',), {
            "machine": "juno", "user": "mola", "file_only": False}),
        (('/remote/multi-failed',), {
            "machine": "juno", "user": "mola", "file_only": False}),
    ]
    assert state_directory.exists()


@pytest.mark.unit
def test_clear_states_only_removes_release_and_generated_files(tmp_path, monkeypatch):
    """State-only cleanup does not touch remote directories."""
    monkeypatch.chdir(tmp_path)
    examples_directory = tmp_path / "examples" / "nested"
    examples_directory.mkdir(parents=True)
    (examples_directory / "workflow_manager.yaml").write_text("generated")
    (examples_directory / "gather_results.py").write_text("generated")
    release_directory = tmp_path / ".release"
    release_directory.mkdir()
    (release_directory / "keep-until-clear").write_text("state")

    with patch("mola.cicd.examples_runner.SV.remove_path") as remove_path:
        assert clear_examples(states_only=True)

    assert not release_directory.exists()
    assert not (examples_directory / "workflow_manager.yaml").exists()
    assert not (examples_directory / "gather_results.py").exists()
    remove_path.assert_not_called()


@pytest.mark.unit
def test_clear_all_removes_all_case_directories_and_local_state(tmp_path, monkeypatch):
    """All cleanup removes every known remote case and local artifact."""
    monkeypatch.chdir(tmp_path)
    state_directory = tmp_path / ".release" / "example_states"
    state_directory.mkdir(parents=True)
    (state_directory / "multi.ci_state.yaml").write_text(yaml.safe_dump({
        "execution_mode": "multi",
        "run_directories": ["/remote/case-1", "/remote/case-2"],
        "machine": "sator",
        "user": "mola",
    }))
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "workflow_manager.yaml").write_text("generated")

    with patch("mola.cicd.examples_runner.SV.remove_path") as remove_path:
        assert clear_examples(all_cases=True)

    assert [call.args[0] for call in remove_path.call_args_list] == [
        "/remote/case-1", "/remote/case-2"]
    assert not (tmp_path / ".release").exists()


@pytest.mark.unit
def test_clear_all_preserves_state_when_remote_cleanup_fails(tmp_path, monkeypatch):
    """Failed remote cleanup must not destroy the retry information."""
    monkeypatch.chdir(tmp_path)
    state_directory = tmp_path / ".release" / "example_states"
    state_directory.mkdir(parents=True)
    (state_directory / "mono.ci_state.yaml").write_text(yaml.safe_dump({
        "execution_mode": "mono",
        "run_directory": "/remote/case",
        "machine": "juno",
    }))

    with patch(
            "mola.cicd.examples_runner.SV.remove_path",
            side_effect=RuntimeError("remote unavailable")):
        assert not clear_examples(all_cases=True)

    assert (tmp_path / ".release").exists()


@pytest.mark.unit
def test_submit_failed_only_skips_nonfailed_examples(tmp_path, monkeypatch):
    """Failed-only submission removes and re-submits failed examples only."""
    monkeypatch.chdir(tmp_path)
    examples_directory = tmp_path / "examples"
    examples_directory.mkdir()
    failed_example = examples_directory / "run_failed.py"
    successful_example = examples_directory / "run_successful.py"
    for example in (failed_example, successful_example):
        example.write_text("# MOLA_CI_MODE: mono\n")

    state_manager = ExampleCIState()
    state_manager.create_state(
        failed_example,
        "mono",
        {
            "status": "FAILED",
            "run_directory": "/remote/failed",
            "machine": "juno",
            "user": "mola",
            "jobs": {"cfd": [1]},
        },
    )
    state_manager.create_state(
        successful_example,
        "mono",
        {
            "status": "COMPLETED",
            "run_directory": "/remote/successful",
            "machine": "juno",
            "user": "mola",
            "jobs": {"cfd": [2]},
        },
    )

    replacement_state = {
        "status": "submitted",
        "run_directory": "/remote/failed",
        "machine": "juno",
        "jobs": {"cfd": [3]},
    }
    with patch("mola.cicd.examples_runner.SV.is_existing_path", return_value=True), \
            patch("mola.cicd.examples_runner.SV.remove_path") as remove_path, \
            patch("mola.cicd.examples_runner.submit.execute_example",
                  return_value=replacement_state) as execute_example:
        submit_examples(failed_only=True)

    execute_example.assert_called_once_with(Path("examples/run_failed.py"), "mono")
    remove_path.assert_called_once_with(
        "/remote/failed", machine="juno", user="mola", file_only=False)


@pytest.mark.unit
def test_cli_forwards_failed_only_options():
    """CLI options reach submit and clear handlers."""
    with patch.object(sys, "argv", ["examples_runner", "submit", "--failed-only"]), \
            patch("mola.cicd.examples_runner.cli.submit_examples") as submit:
        main()
    submit.assert_called_once_with(None, None, False, False, True)

    with patch.object(sys, "argv", ["examples_runner", "clear", "--all"]), \
            patch("mola.cicd.examples_runner.cli.clear_examples", return_value=True) as clear:
        main()
    clear.assert_called_once_with(states_only=False, failed_only=False, all_cases=True)


@pytest.mark.unit
def test_cli_repatriation_modes_default_to_full():
    assert build_parser().parse_args(["repatriate"]).repatriate_mode == "full"
    assert build_parser().parse_args(["repatriate", "--light"]).repatriate_mode == "light"
