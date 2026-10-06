"""Discover CI examples and their remote case directories.

The examples runner deliberately keeps no local scheduler state.  The Python
example is the source of the high-level configuration, while the generated
``job_sequence.sh`` files are the source of the concrete multi-case layout.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

from mola import server as SV


_USER_VARIABLE = re.compile(r"\$(?:\{USER\}|USER)")


@dataclass(frozen=True)
class ExampleDefinition:
    """Static information extracted from one marked ``run*.py`` script."""

    identifier: str
    script_path: Path
    mode: str
    solver: str
    example_name: str
    root_directory: str
    run_directory: str
    machine: str
    source_user: Optional[str]


def resolve_source_user(source_user: Optional[str] = None) -> Optional[str]:
    """Return the account used when ``$USER`` is expanded in example paths."""

    return source_user or os.environ.get("USER") or os.environ.get("USERNAME")


def expand_example_path(path: str, source_user: Optional[str] = None) -> str:
    """Expand MOLA example paths without relying on the current process user."""

    source_user = resolve_source_user(source_user)
    value = str(path)
    if source_user:
        value = _USER_VARIABLE.sub(source_user, value)
    return os.path.expandvars(value)


def _extract_optional(script_path: Path, parameter: str, call_name: str = None):
    try:
        return SV.extract_param_from_script(
            script_path, parameter, call_name=call_name
        )
    except (ValueError, TypeError):
        return None


def _detect_execution_mode(script_path: Path) -> Optional[str]:
    for line in script_path.read_text(encoding="utf-8").splitlines()[-5:]:
        if not line.strip().startswith("# MOLA_CI_MODE:"):
            continue
        mode = line.split(":", 1)[1].strip().strip("\"'")
        if mode in {"mono", "multi"}:
            return mode
    return None


def scan_example_definitions(
        examples_root: str | Path = "examples",
        source_user: Optional[str] = None,
) -> list[ExampleDefinition]:
    """Scan marked example scripts without executing them."""

    source_user = resolve_source_user(source_user)
    definitions = []
    for script_path in sorted(Path(examples_root).rglob("run*.py")):
        mode = _detect_execution_mode(script_path)
        if mode is None:
            continue

        solver = str(SV.extract_param_from_script(script_path, "Solver"))
        run_directory = expand_example_path(
            str(SV.extract_param_from_script(script_path, "RunDirectory")),
            source_user,
        )

        if mode == "multi":
            root_directory = _extract_optional(
                script_path, "root_directory", call_name="WorkflowManager"
            )
            if root_directory is None:
                root_directory = run_directory
            root_directory = expand_example_path(str(root_directory), source_user)
        else:
            root_directory = run_directory

        example_name = _extract_optional(script_path, "JobName")
        if example_name is None:
            example_name = script_path.stem

        try:
            machine = SV.guess_machine(root_directory)
        except Exception:
            machine = "localhost"

        identifier = str(script_path.resolve())
        definitions.append(ExampleDefinition(
            identifier=identifier,
            script_path=script_path,
            mode=mode,
            solver=solver,
            example_name=str(example_name),
            root_directory=root_directory,
            run_directory=run_directory,
            machine=machine,
            source_user=source_user,
        ))
    return definitions


def _remote_probe_code(items: list[dict], scheduler_user: Optional[str]) -> str:
    payload = json.dumps({"items": items, "scheduler_user": scheduler_user})
    return f'''import json
import os
import re
import shlex
import subprocess

payload = json.loads({payload!r})
result = {{"exists": {{}}, "cases": [], "jobs": [], "scheduler_error": ""}}

def parse_sequence_cases(sequence_file):
    try:
        text = open(sequence_file, encoding="utf-8").read()
    except OSError:
        return []

    match = re.search(r"^\\s*SEQUENCE_OF_PATHS\\s*=\\s*(.+)$", text, re.MULTILINE)
    if not match:
        return []

    try:
        paths = shlex.split(match.group(1).strip())
    except ValueError:
        return []
    if len(paths) == 1 and " " in paths[0]:
        paths = paths[0].split()
    return paths

def add_case(item, run_directory, sequence_root=None):
    run_directory = os.path.normpath(os.path.abspath(run_directory))
    sequence_root = (
        os.path.normpath(os.path.abspath(sequence_root))
        if sequence_root else None
    )
    key = (item["identifier"], run_directory)
    if key in seen_cases:
        return
    seen_cases.add(key)
    result["cases"].append({{
        "example_id": item["identifier"],
        "run_directory": run_directory,
        "sequence_root": sequence_root,
        "scheduler_workdir": sequence_root or run_directory,
        "mode": item["mode"],
        "solver": item["solver"],
        "machine": item["machine"],
        "source_user": item.get("source_user"),
    }})

seen_cases = set()
for item in payload["items"]:
    root = os.path.normpath(os.path.abspath(os.path.expandvars(item["root_directory"])))
    result["exists"][item["identifier"]] = os.path.exists(root)

    if item["mode"] != "multi":
        add_case(item, root)
        continue

    if not os.path.isdir(root):
        continue

    for directory, _subdirectories, filenames in os.walk(root):
        if "job_sequence.sh" not in filenames:
            continue
        sequence_root = os.path.abspath(directory)
        sequence_file = os.path.join(sequence_root, "job_sequence.sh")
        paths = parse_sequence_cases(sequence_file)
        for path in paths:
            add_case(item, os.path.join(sequence_root, path), sequence_root)

try:
    command = ["squeue", "-h", "-o", "%t\\t%i\\t%M\\t%Z"]
    if payload.get("scheduler_user"):
        command[1:1] = ["-u", payload["scheduler_user"]]
    else:
        command.insert(1, "--me")
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        result["scheduler_error"] = completed.stderr.strip()
    else:
        for line in completed.stdout.splitlines():
            fields = line.split("\\t", 3)
            if len(fields) != 4:
                continue
            state, job_id, elapsed, work_directory = fields
            result["jobs"].append({{
                "state": state,
                "job_id": job_id,
                "elapsed": elapsed,
                "work_directory": os.path.normpath(os.path.abspath(work_directory)),
            }})
except OSError as error:
    result["scheduler_error"] = str(error)

print(json.dumps(result))
'''


def _run_remote_probe(
        items: list[dict],
        machine: str,
        connection_user: Optional[str],
        scheduler_user: Optional[str],
) -> dict:
    from mola.logging.exceptions import MolaException

    code = _remote_probe_code(items, scheduler_user)
    encoded = base64.b64encode(code.encode()).decode()

    remote_script = f"""
    source /etc/bashrc
    set -eo pipefail
    module load python >&2
    command -v python3 >&2
    python3 --version >&2
    printf '%s' {shlex.quote(encoded)} | base64 -d | python3
    """

    remote_command = f"bash -lc {shlex.quote(remote_script)}"

    try:
        result = subprocess.run(
            [
                "ssh", 
                "-T",
                "-o", "BatchMode=yes",
                f"{connection_user}@{machine}" if connection_user else machine,
                remote_command,
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f"return code: {exc.returncode}")
        print(f"stdout: {exc.stdout}")
        print(f"stderr: {exc.stderr}")
        raise

    output = result.stdout
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise MolaException(f"Failed to parse remote examples discovery result: {output}")


def _job_state(value: str) -> str:
    return {
        "R": "RUNNING",
        "CG": "RUNNING",
        "PD": "PENDING",
        "CF": "PENDING",
        "S": "RUNNING",
        "F": "FAILED",
        "TO": "TIMEOUT",
        "CD": "COMPLETED",
    }.get(value, value)


def discover_remote_layout(
        definitions: Iterable[ExampleDefinition],
        connection_user: Optional[str] = None,
) -> dict:
    """Discover cases and active scheduler jobs, batched by machine.

    ``connection_user`` is the SSH account.  The source user embedded in the
    paths is passed separately to SLURM, which is important when repatriation
    runs as the ``mola`` service account.
    """

    definitions = list(definitions)
    grouped = {}
    for definition in definitions:
        grouped.setdefault(definition.machine, []).append(definition)

    all_cases = []
    exists = {}
    jobs_by_workdir = {}
    scheduler_errors = {}

    for machine, machine_definitions in grouped.items():
        items = [
            {
                "identifier": definition.identifier,
                "mode": definition.mode,
                "solver": definition.solver,
                "root_directory": definition.root_directory,
                "machine": definition.machine,
                "source_user": definition.source_user,
            }
            for definition in machine_definitions
        ]
        scheduler_users = {
            definition.source_user
            for definition in machine_definitions
            if definition.source_user
        }
        if len(scheduler_users) > 1:
            raise ValueError(
                f"Examples on {machine} use several source users: "
                f"{sorted(scheduler_users)}. Pass one source user per invocation."
            )
        scheduler_user = next(iter(scheduler_users), None)
        response = _run_remote_probe(
            items, machine, connection_user, scheduler_user
        )
        exists.update(response.get("exists", {}))
        all_cases.extend(response.get("cases", []))
        if response.get("scheduler_error"):
            scheduler_errors[machine] = response["scheduler_error"]
        for job in response.get("jobs", []):
            state = _job_state(str(job.get("state", "")))
            workdir = os.path.normpath(str(job.get("work_directory", "")))
            if not workdir:
                continue
            previous = jobs_by_workdir.get((machine, workdir))
            if previous is None or _job_priority(state) > _job_priority(previous["state"]):
                jobs_by_workdir[(machine, workdir)] = {
                    **job,
                    "state": state,
                }

    return {
        "cases": all_cases,
        "exists": exists,
        "jobs_by_workdir": jobs_by_workdir,
        "scheduler_errors": scheduler_errors,
    }


def _job_priority(state: str) -> int:
    return {
        "RUNNING": 4,
        "PENDING": 3,
        "SUSPENDED": 2,
        "TIMEOUT": 1,
        "FAILED": 1,
        "COMPLETED": 0,
    }.get(state, 1)


def make_case_requests(
        definitions: Iterable[ExampleDefinition],
        layout: dict,
) -> list[dict]:
    """Attach scheduler state to discovered case records."""

    definitions_by_id = {definition.identifier: definition for definition in definitions}
    cases_by_example = {definition.identifier: 0 for definition in definitions}
    requests = []
    for case in layout.get("cases", []):
        definition = definitions_by_id[case["example_id"]]
        cases_by_example[definition.identifier] += 1
        workdir = os.path.normpath(case["scheduler_workdir"])
        job = layout.get("jobs_by_workdir", {}).get((definition.machine, workdir))
        requests.append({
            **case,
            "user": None,
            "scheduler_error": layout.get("scheduler_errors", {}).get(
                definition.machine
            ),
            "scheduler_status": job["state"] if job else "",
            "scheduler_job_id": job.get("job_id") if job else None,
            "example_name": definition.example_name,
        })

    # A missing multi root still needs a visible status row.  It is not a
    # real CFD case, but TO_PREPARE is more useful than silently omitting the
    # example from validation output.
    for definition in definitions:
        if cases_by_example[definition.identifier]:
            continue
        requests.append({
            "example_id": definition.identifier,
            "run_directory": definition.root_directory,
            "sequence_root": None,
            "scheduler_workdir": definition.root_directory,
            "mode": definition.mode,
            "solver": definition.solver,
            "machine": definition.machine,
            "source_user": definition.source_user,
            "user": None,
            "scheduler_error": layout.get("scheduler_errors", {}).get(
                definition.machine
            ),
            "scheduler_status": "",
            "scheduler_job_id": None,
            "example_name": definition.example_name,
            "missing_layout": True,
        })
    return requests
