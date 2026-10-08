#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.

"""Utilities for discovering and repatriating MOLA simulation cases."""



import base64
import json
import os
import posixpath
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import mola.naming_conventions as names
from mola.logging import MolaException

from .files_operations import rsync
from .parameter_extractor import extract_param_from_script
from .remote import guess_machine, submit_command


@dataclass(frozen=True)
class RepatriationCase:
    """A remote case and the local root used to preserve its relative path."""

    run_directory: str
    root_directory: str
    machine: str
    user: Optional[str] = None


def get_repatriation_filters(mode="full", extra_patterns=()):
    """Return rsync filters for a full, light, or custom repatriation."""
    mode = str(mode).lower()
    if mode == "full":
        return None, None
    if mode not in {"light", "custom"}:
        raise ValueError(f"Unknown repatriation mode: {mode}")

    included_files = set(names.STATUS_FILES)
    if mode == "light":
        included_files.update(names.REPATRIATION_LIGHT_PATTERNS)
    else:
        included_files.update(names.REPATRIATION_LOG_PATTERNS)
    included_files.update(extra_patterns)
    return included_files, {"*"}


def expand_remote_path(path, source_user=None):
    value = str(path)
    if source_user:
        value = value.replace("${USER}", source_user).replace("$USER", source_user)
    return posixpath.normpath(os.path.expandvars(value))


def _optional_script_parameter(script_path, parameter, call_name=None):
    try:
        return extract_param_from_script(
            script_path, parameter, call_name=call_name
        )
    except (ValueError, TypeError):
        return None


def get_script_repatriation_info(
        script_path, machine=None, user=None, source_user=None):
    """Infer a script's run/root directory, machine, and remote user.

    A mono script uses its ``RunDirectory`` as both root and case.  For a
    script constructing a ``WorkflowManager``, ``root_directory`` becomes
    the discovery root; its generated ``job_sequence.sh`` files provide the
    concrete case directories.
    """
    script_path = Path(script_path)
    run_directory = extract_param_from_script(script_path, "RunDirectory")
    root_directory = _optional_script_parameter(
        script_path, "root_directory", call_name="WorkflowManager"
    )
    if root_directory is None:
        root_directory = run_directory

    run_directory = expand_remote_path(run_directory, source_user)
    root_directory = expand_remote_path(root_directory, source_user)
    machine = machine or guess_machine(root_directory)
    user = user or _optional_script_parameter(script_path, "User") or source_user
    return root_directory, run_directory, machine, user


def get_workflow_manager_info(manager=None):
    """Return a manager's root, known cases, machine, and remote user."""
    if manager is None:
        from mola.workflow import WorkflowManager

        manager = WorkflowManager(names.FILE_WORKFLOW_MANAGER)

    root_directory = expand_remote_path(manager.root_directory)
    run_directories = [
        expand_remote_path(run_directory)
        for run_directory in getattr(manager, "run_directories", [])
    ]
    machine = manager.machine
    user = getattr(manager, "user", None)
    return root_directory, run_directories, machine, user


def _remote_sequence_probe_code(root_directory):
    payload = json.dumps({
        "root_directory": root_directory,
        "sequence_filename": names.FILE_JOB_SEQUENCE,
        "case_filename": names.FILE_JOB,
    })
    return f'''import json
import os
import re
import shlex

payload = json.loads({payload!r})
root = os.path.normpath(os.path.abspath(payload["root_directory"]))
result = {{"root_exists": os.path.isdir(root), "root_is_case": False, "cases": []}}
seen = set()

def sequence_paths(sequence_file):
    try:
        text = open(sequence_file, encoding="utf-8").read()
    except OSError:
        return []
    match = re.search(
        r"^\\s*SEQUENCE_OF_PATHS\\s*=\\s*(.+)$", text, re.MULTILINE
    )
    if not match:
        return []
    try:
        paths = shlex.split(match.group(1).strip())
    except ValueError:
        return []
    if len(paths) == 1 and " " in paths[0]:
        paths = paths[0].split()
    return paths

if os.path.isfile(os.path.join(root, payload["case_filename"])):
    result["root_is_case"] = True

if os.path.isdir(root):
    for directory, _subdirectories, filenames in os.walk(root):
        if payload["sequence_filename"] not in filenames:
            continue
        sequence_root = os.path.abspath(directory)
        sequence_file = os.path.join(sequence_root, payload["sequence_filename"])
        for relative_path in sequence_paths(sequence_file):
            case = os.path.normpath(os.path.abspath(
                os.path.join(sequence_root, relative_path)
            ))
            if case not in seen:
                seen.add(case)
                result["cases"].append(case)

print(json.dumps(result))
'''


def discover_job_sequence_cases(root_directory, machine, user=None):
    """Discover concrete case directories below a remote root.

    ``job_sequence.sh`` is treated as the structural sentinel.  The command
    reads each sequence's ``SEQUENCE_OF_PATHS`` assignment remotely, so the
    local process never needs to guess whether a directory is a case or only
    a parent of several cases.
    """
    root_directory = expand_remote_path(root_directory)
    code = _remote_sequence_probe_code(root_directory)
    encoded = base64.b64encode(code.encode()).decode()
    command = f'echo "{encoded}" | base64 -d | python3 2>/dev/null'
    output = submit_command(
        command=command,
        machine=machine,
        user=user,
        use_mola_env=False,
    )
    for line in reversed(output.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    raise MolaException(
        f"Failed to parse remote job-sequence discovery result: {output}"
    )


def resolve_remote_cases(
        root_directory, machine, user=None, fallback_run_directories=()):
    """Resolve a root into concrete cases without copying a parent directory."""
    root_directory = expand_remote_path(root_directory)
    result = discover_job_sequence_cases(root_directory, machine, user=user)
    discovered = result.get("cases", [])
    if discovered:
        return [
            RepatriationCase(case, root_directory, machine, user)
            for case in discovered
        ]

    if result.get("root_is_case"):
        return [RepatriationCase(root_directory, root_directory, machine, user)]

    fallback = [
        expand_remote_path(directory)
        for directory in fallback_run_directories
        if directory
    ]
    if fallback:
        return [
            RepatriationCase(directory, root_directory, machine, user)
            for directory in dict.fromkeys(fallback)
        ]

    if not result.get("root_exists"):
        raise MolaException(f"Remote root directory does not exist: {root_directory}")
    raise MolaException(
        f"No {names.FILE_JOB_SEQUENCE} files were found below {root_directory}"
    )


def _local_destination(case):
    relative = posixpath.relpath(case.run_directory, case.root_directory)
    if relative == ".":
        return "."
    if relative == ".." or relative.startswith("../"):
        raise MolaException(
            f"Case {case.run_directory} is outside root {case.root_directory}"
        )
    return relative


def repatriate_cases(cases: Iterable[RepatriationCase], included_files=None,
                     excluded_files=None):
    """Rsync discovered cases while preserving their root-relative layout."""
    for case in cases:
        destination = _local_destination(case)
        if destination != ".":
            os.makedirs(destination, exist_ok=True)
        source = case.run_directory.rstrip("/") + "/"
        rsync(
            source_path=source,
            source_machine=case.machine,
            source_user=case.user,
            destination_path=destination,
            destination_machine="localhost",
            included_files=included_files,
            excluded_files=excluded_files,
        )


def get_one_case(run_directory, machine=None, user=None,
                 included_files=None, excluded_files=None):
    """Repatriate one explicitly identified case."""
    run_directory = expand_remote_path(run_directory)
    machine = machine or guess_machine(run_directory)
    repatriate_cases(
        [RepatriationCase(run_directory, run_directory, machine, user)],
        included_files=included_files,
        excluded_files=excluded_files,
    )


def get_all_cases_from_workflow_manager(
        included_files=None, excluded_files=None, workflow_manager=None):
    """Repatriate manager cases discovered below its remote root."""
    root_directory, run_directories, machine, user = get_workflow_manager_info(
        workflow_manager
    )
    cases = resolve_remote_cases(
        root_directory,
        machine,
        user=user,
        fallback_run_directories=run_directories,
    )
    repatriate_cases(cases, included_files, excluded_files)
