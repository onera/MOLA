from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import ast


class ParamExtractionError(ValueError):
    pass


def _dotted_name(node: ast.AST) -> Optional[str]:
    """Return the dotted name represented by an AST node."""

    if isinstance(node, ast.Name):
        return node.id

    if isinstance(node, ast.Attribute):
        parent = _dotted_name(node.value)

        if parent is not None:
            return f"{parent}.{node.attr}"

        return node.attr

    return None


def _collect_simple_constants(tree: ast.Module) -> Dict[str, ast.AST]:
    """Collect simple top-level assignments such as SOLVER = 'elsa'."""

    constants = {}

    for statement in tree.body:
        if (
            isinstance(statement, ast.Assign)
            and len(statement.targets) == 1
            and isinstance(statement.targets[0], ast.Name)
        ):
            constants[statement.targets[0].id] = statement.value

        elif (
            isinstance(statement, ast.AnnAssign)
            and isinstance(statement.target, ast.Name)
            and statement.value is not None
        ):
            constants[statement.target.id] = statement.value

    return constants


def _evaluate_static(
    node: ast.AST,
    constants: Dict[str, ast.AST],
    resolving: Optional[set] = None,
) -> Any:
    """
    Safely evaluate a restricted set of Python expressions.

    No arbitrary Python code is executed.
    """
    if resolving is None:
        resolving = set()

    # Resolve simple variables, for example:
    # solver_name = "elsa"
    # Solver = solver_name
    if isinstance(node, ast.Name):
        if node.id in {"True", "False", "None"}:
            return {"True": True, "False": False, "None": None}[node.id]

        if node.id not in constants:
            raise ParamExtractionError(
                f"Cannot statically resolve variable {node.id!r}"
            )

        if node.id in resolving:
            raise ParamExtractionError(
                f"Circular variable definition involving {node.id!r}"
            )

        return _evaluate_static(
            constants[node.id],
            constants,
            resolving | {node.id},
        )

    # Strings, integers, floats, booleans, None, etc.
    if isinstance(node, ast.Constant):
        return node.value

    # Support f-strings whose interpolated expressions can themselves be
    # statically evaluated, for example:
    #
    # name = "toto"
    # RunDirectory = f"/path/to/$USER/{name}"
    #
    # The literal ``$USER`` is deliberately left untouched: it is part of the
    # Python string and is not a Python expression to evaluate.
    if isinstance(node, ast.JoinedStr):
        parts = []

        for value_node in node.values:
            if isinstance(value_node, ast.Constant):
                if not isinstance(value_node.value, str):
                    raise ParamExtractionError(
                        "Only string literals are supported inside f-strings"
                    )

                parts.append(value_node.value)
                continue

            if not isinstance(value_node, ast.FormattedValue):
                raise ParamExtractionError(
                    "Cannot statically evaluate f-string component"
                )

            value = _evaluate_static(
                value_node.value,
                constants,
                resolving,
            )

            if value_node.conversion == -1:
                converted_value = value
            elif value_node.conversion == ord("s"):
                converted_value = str(value)
            elif value_node.conversion == ord("r"):
                converted_value = repr(value)
            elif value_node.conversion == ord("a"):
                converted_value = ascii(value)
            else:
                raise ParamExtractionError(
                    "Unsupported f-string conversion"
                )

            if value_node.format_spec is not None:
                format_spec = _evaluate_static(
                    value_node.format_spec,
                    constants,
                    resolving,
                )

                if not isinstance(format_spec, str):
                    raise ParamExtractionError(
                        "The f-string format specification must be a string"
                    )

                try:
                    converted_value = format(converted_value, format_spec)
                except (ValueError, TypeError) as exc:
                    raise ParamExtractionError(
                        "Invalid f-string format specification"
                    ) from exc

            parts.append(str(converted_value))

        return "".join(parts)

    if isinstance(node, ast.List):
        return [
            _evaluate_static(element, constants, resolving)
            for element in node.elts
        ]

    if isinstance(node, ast.Tuple):
        return tuple(
            _evaluate_static(element, constants, resolving)
            for element in node.elts
        )

    if isinstance(node, ast.Set):
        return {
            _evaluate_static(element, constants, resolving)
            for element in node.elts
        }

    if isinstance(node, ast.Dict):
        result = {}

        for key_node, value_node in zip(node.keys, node.values):
            # This corresponds to **some_dict.
            if key_node is None:
                raise ParamExtractionError(
                    "Dictionary expansion using ** cannot be resolved statically"
                )

            key = _evaluate_static(key_node, constants, resolving)
            value = _evaluate_static(value_node, constants, resolving)
            result[key] = value

        return result

    # Support -1, +1, -0.5, etc.
    if isinstance(node, ast.UnaryOp):
        operand = _evaluate_static(node.operand, constants, resolving)

        if isinstance(node.op, ast.USub):
            return -operand

        if isinstance(node.op, ast.UAdd):
            return +operand

    # Support static string concatenation, for example:
    #
    # solver = "toto"
    # example_name = "isol_wing_" + solver
    #
    # Restrict this to strings so that adding arithmetic operators does not
    # accidentally broaden the evaluator beyond its intended use.
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left = _evaluate_static(node.left, constants, resolving)
        right = _evaluate_static(node.right, constants, resolving)

        if not isinstance(left, str) or not isinstance(right, str):
            raise ParamExtractionError(
                "String concatenation requires two string operands"
            )

        return left + right

    # Support expressions such as:
    #
    # Flow = dict(
    #     Velocity=50.0,
    #     Density=1.225,
    # )
    if isinstance(node, ast.Call):
        function_name = _dotted_name(node.func)

        if function_name == "dict":
            result = {}

            if len(node.args) > 1:
                raise ParamExtractionError(
                    "Only one positional argument is supported for dict(...)"
                )

            if node.args:
                positional_value = _evaluate_static(
                    node.args[0],
                    constants,
                    resolving,
                )

                if not isinstance(positional_value, dict):
                    raise ParamExtractionError(
                        "The positional argument of dict(...) must be a dictionary"
                    )

                result.update(positional_value)

            for keyword in node.keywords:
                if keyword.arg is None:
                    raise ParamExtractionError(
                        "Dictionary expansion using ** cannot be resolved statically"
                    )

                result[keyword.arg] = _evaluate_static(
                    keyword.value,
                    constants,
                    resolving,
                )

            return result

        if function_name in {"list", "tuple", "set"} and len(node.args) == 1:
            value = _evaluate_static(
                node.args[0],
                constants,
                resolving,
            )

            return {
                "list": list,
                "tuple": tuple,
                "set": set,
            }[function_name](value)

    line = getattr(node, "lineno", "?")

    raise ParamExtractionError(
        f"Cannot statically evaluate expression on line {line}"
    )


def extract_param_from_script(
    script_path: Union[str, Path],
    param: str,
    call_name: Optional[str] = None,
) -> Any:
    """
    Extract a keyword argument from a Python script without executing it.

    Parameters
    ----------
    script_path:
        Path to the Python script.

    param:
        Keyword argument to retrieve, for example "Solver", "Flow",
        "Velocity", or "Numerics".

    call_name:
        Optional constructor/function filter.

        Examples:
            "WorkflowAirplane"
            "mola.workflow.WorkflowAirplane"

        If omitted, every function/class call in the script is inspected.

    Returns
    -------
    Any
        The statically evaluated value of the requested parameter.
    """
    if not param.isidentifier():
        raise ValueError(f"Invalid Python parameter name: {param!r}")

    script_path = Path(script_path)
    source = script_path.read_text(encoding="utf-8")

    try:
        tree = ast.parse(source, filename=str(script_path))
    except SyntaxError as exc:
        raise ParamExtractionError(
            f"Invalid Python syntax in {script_path}: {exc}"
        ) from exc

    constants = _collect_simple_constants(tree)

    matches: List[Tuple[int, Any]] = []
    possible_hidden_parameters = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue

        function_name = _dotted_name(node.func)

        if call_name is not None:
            matches_call = (
                function_name == call_name
                or (
                    function_name is not None
                    and function_name.rsplit(".", 1)[-1] == call_name
                )
            )

            if not matches_call:
                continue

        parameter_keywords = [
            keyword
            for keyword in node.keywords
            if keyword.arg == param
        ]

        if parameter_keywords:
            if len(parameter_keywords) > 1:
                raise ParamExtractionError(
                    f"Multiple {param!r} arguments found "
                    f"on line {node.lineno}"
                )

            value = _evaluate_static(
                parameter_keywords[0].value,
                constants,
            )

            matches.append((node.lineno, value))

        elif any(keyword.arg is None for keyword in node.keywords):
            # The parameter could potentially be hidden inside **kwargs.
            possible_hidden_parameters.append(node.lineno)

    if not matches:
        if possible_hidden_parameters:
            raise ParamExtractionError(
                f"No explicit {param!r} argument found. "
                f"It may be hidden inside **kwargs on lines "
                f"{possible_hidden_parameters}."
            )

        call_description = (
            f" in calls to {call_name!r}"
            if call_name is not None
            else ""
        )

        raise ParamExtractionError(
            f"Parameter {param!r} was not found{call_description}"
        )

    values = [value for _, value in matches]

    # If several calls contain the same parameter, accept them only if
    # they all have the same value.
    first_value = values[0]

    if any(value != first_value for value in values[1:]):
        locations = ", ".join(
            f"line {line}: {value!r}"
            for line, value in matches
        )

        raise ParamExtractionError(
            f"Ambiguous parameter {param!r}; different values found: "
            f"{locations}"
        )

    return first_value
