"""Tests for server-side repatriation helpers."""

import ast
from unittest.mock import patch

import pytest

from mola.server import (
    RepatriationCase,
    get_repatriation_filters,
    get_script_repatriation_info,
    repatriate_cases,
    resolve_remote_cases,
)
from mola.server.repatriate import _remote_sequence_probe_code


@pytest.mark.unit
def test_repatriation_filters_are_shared_by_mode():
    included, excluded = get_repatriation_filters("full")
    assert included is None
    assert excluded is None

    included, excluded = get_repatriation_filters("light")
    assert "COMPLETED" in included
    assert "**.png" in included
    assert "LOGS/coprocess-*.log" in included
    assert "**/LOGS/coprocess-*.log" in included
    assert "coprocess.log" not in included
    assert excluded == {"*"}


@pytest.mark.unit
def test_sequence_probe_is_valid_python_and_uses_job_sequence_sentinel():
    code = _remote_sequence_probe_code("/tmp_user/sator/user/study")
    ast.parse(code)
    assert "job_sequence.sh" in code
    assert "SEQUENCE_OF_PATHS" in code


@pytest.mark.unit
def test_resolve_remote_cases_prefers_discovered_sequence_cases():
    probe = {
        "root_exists": True,
        "root_is_case": False,
        "cases": ["/remote/root/a", "/remote/root/b"],
    }
    with patch(
            "mola.server.repatriate.discover_job_sequence_cases",
            return_value=probe):
        cases = resolve_remote_cases(
            "/remote/root",
            "sator",
            user="lbernard",
            fallback_run_directories=["/remote/root/legacy"],
        )

    assert cases == [
        RepatriationCase("/remote/root/a", "/remote/root", "sator", "lbernard"),
        RepatriationCase("/remote/root/b", "/remote/root", "sator", "lbernard"),
    ]


@pytest.mark.unit
def test_repatriate_cases_preserves_root_relative_layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    case = RepatriationCase(
        "/remote/root/sequence/case", "/remote/root", "sator", "lbernard"
    )
    with patch("mola.server.repatriate.rsync") as rsync:
        repatriate_cases([case], included_files={"COMPLETED"}, excluded_files={"*"})

    assert (tmp_path / "sequence" / "case").is_dir()
    rsync.assert_called_once_with(
        source_path="/remote/root/sequence/case/",
        source_machine="sator",
        source_user="lbernard",
        destination_path="sequence/case",
        destination_machine="localhost",
        included_files={"COMPLETED"},
        excluded_files={"*"},
    )


@pytest.mark.unit
def test_script_repatriation_info_infers_root_and_expands_user(tmp_path):
    script = tmp_path / "run_example.py"
    script.write_text(
        """
from mola.workflow import Workflow, WorkflowManager
w = Workflow(RunDirectory='/tmp_user/sator/$USER/study/base')
manager = WorkflowManager(w, root_directory='/tmp_user/sator/$USER/study')
""",
        encoding="utf-8",
    )
    with patch("mola.server.repatriate.guess_machine", return_value="sator"):
        root, run, machine, user = get_script_repatriation_info(
            script, source_user="lbernard"
        )

    assert root == "/tmp_user/sator/lbernard/study"
    assert run == "/tmp_user/sator/lbernard/study/base"
    assert machine == "sator"
    assert user == "lbernard"
