"""Unit tests for static parameter extraction from example scripts."""

import pytest

from mola.server.parameter_extractor import (
    ParamExtractionError,
    extract_param_from_script,
)


def write_script(tmp_path, source):
    path = tmp_path / "example.py"
    path.write_text(source, encoding="utf-8")
    return path


@pytest.mark.unit
def test_extract_literals_and_containers(tmp_path):
    path = write_script(tmp_path, """
SOLVER = 'elsa'
Flow = dict(Velocity=50.0, Density=1.225)
Numerics = {'CFL': [1.0, 2.0], 'Iterations': 10}
Workflow(Solver=SOLVER, Flow=Flow, Numerics=Numerics)
""")

    assert extract_param_from_script(path, "Solver") == "elsa"
    assert extract_param_from_script(path, "Flow") == {
        "Velocity": 50.0, "Density": 1.225
    }
    assert extract_param_from_script(path, "Numerics") == {
        "CFL": [1.0, 2.0], "Iterations": 10
    }


@pytest.mark.unit
def test_extract_aliases_unary_values_and_builtin_constructors(tmp_path):
    path = write_script(tmp_path, """
velocity = -50.0
values = tuple([1, 2])
Workflow(Velocity=velocity, Values=values)
""")

    assert extract_param_from_script(path, "Velocity") == -50.0
    assert extract_param_from_script(path, "Values") == (1, 2)


@pytest.mark.unit
def test_extract_call_name_accepts_qualified_and_short_names(tmp_path):
    path = write_script(tmp_path, """
mola.workflow.Workflow(Solver='elsa')
other.Workflow(Solver='elsa')
""")

    assert extract_param_from_script(path, "Solver", "mola.workflow.Workflow") == "elsa"
    assert extract_param_from_script(path, "Solver", "Workflow") == "elsa"


@pytest.mark.unit
def test_equal_repeated_values_are_not_ambiguous(tmp_path):
    path = write_script(tmp_path, """
Workflow(Solver='elsa')
Workflow(Solver='elsa')
""")

    assert extract_param_from_script(path, "Solver") == "elsa"


@pytest.mark.unit
def test_different_repeated_values_are_ambiguous(tmp_path):
    path = write_script(tmp_path, """
Workflow(Solver='elsa')
Workflow(Solver='fast')
""")

    with pytest.raises(ParamExtractionError, match="Ambiguous"):
        extract_param_from_script(path, "Solver")


@pytest.mark.unit
def test_missing_parameter_and_call_filter_are_reported(tmp_path):
    path = write_script(tmp_path, "Workflow(Solver='elsa')\n")

    with pytest.raises(ParamExtractionError, match="not found"):
        extract_param_from_script(path, "Flow")
    with pytest.raises(ParamExtractionError, match="not found"):
        extract_param_from_script(path, "Solver", "OtherWorkflow")


@pytest.mark.unit
def test_hidden_keyword_arguments_are_rejected(tmp_path):
    path = write_script(tmp_path, """
options = {'Solver': 'elsa'}
Workflow(**options)
""")

    with pytest.raises(ParamExtractionError, match="hidden"):
        extract_param_from_script(path, "Solver")


@pytest.mark.unit
def test_invalid_parameter_name_is_rejected(tmp_path):
    path = write_script(tmp_path, "Workflow(Solver='elsa')\n")

    with pytest.raises(ValueError, match="Invalid Python parameter"):
        extract_param_from_script(path, "not-a-name")


@pytest.mark.unit
def test_invalid_syntax_is_reported(tmp_path):
    path = write_script(tmp_path, "Workflow(Solver=\n")

    with pytest.raises(ParamExtractionError, match="Invalid Python syntax"):
        extract_param_from_script(path, "Solver")


@pytest.mark.unit
def test_dynamic_expressions_are_not_executed(tmp_path):
    path = write_script(tmp_path, """
def get_solver():
    raise RuntimeError('must not run')
Workflow(Solver=get_solver())
""")

    with pytest.raises(ParamExtractionError, match="Cannot statically evaluate"):
        extract_param_from_script(path, "Solver")


@pytest.mark.unit
def test_circular_constants_are_rejected(tmp_path):
    path = write_script(tmp_path, """
first = second
second = first
Workflow(Solver=first)
""")

    with pytest.raises(ParamExtractionError, match="Circular"):
        extract_param_from_script(path, "Solver")
