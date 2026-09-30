"""Tests for state-based repatriation helpers."""

import os

import pytest
import yaml

from mola.server import get_cases_from_submission_state, get_repatriation_filters


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
def test_submission_state_preserves_submission_machine_and_flattens_single_case(
        tmp_path, monkeypatch):
    state_path = tmp_path / "state.yaml"
    state_path.write_text(yaml.safe_dump({
        "submissions": [{
            "machine": "juno",
            "user": "mola",
            "run_directories": ["/remote/case"],
        }],
    }))
    calls = []
    monkeypatch.setattr(
        "mola.server.repatriate.rsync",
        lambda **kwargs: calls.append(kwargs),
    )

    get_cases_from_submission_state(
        state_path,
        destination_path=str(tmp_path / "example"),
        flatten_single_case=True,
    )

    assert calls == [{
        "source_path": "/remote/case/",
        "source_machine": "juno",
        "source_user": "mola",
        "destination_path": f"{tmp_path / 'example'}{os.path.sep}",
        "destination_machine": "localhost",
        "included_files": None,
        "excluded_files": None,
    }]
