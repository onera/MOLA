"""Run example comparison and visualization scripts on each HPC machine."""

from __future__ import annotations

import os
import posixpath
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import mola
from mola import server as SV

from .discovery import ExampleDefinition, scan_example_definitions
from .repatriate import _relocate_ssh_command, _target_user_from_path
from .status import resolve_live_statuses

POSTPROCESS_SCRIPTS = ["compare.py", "visualize.py"]


def _script_command(script_path):
    return f"python3 {shlex.quote(str(script_path))}"


def _definition_parent(definition: ExampleDefinition) -> Path:
    parts = definition.script_path.parts
    if "examples" in parts:
        index = parts.index("examples")
        return Path(*parts[index + 1:-1])
    return definition.script_path.parent


def _definition_release_root(definition, tag, root=None):
    if root:
        return f"/tmp_user/{definition.machine}/{root}/examples/{_definition_parent(definition)}"
    return f"/tmp_user/{definition.machine}/mola/{tag}/examples/{_definition_parent(definition)}"


def _live_result_directories(definition, result, tag, root=None):
    release_root = _definition_release_root(definition, tag, root=root)
    if definition.mode != "multi":
        return [f"{release_root}/example_{definition.example_name}"]

    result_directories = []
    for detail in result.get("details", []):
        run_directory = detail.get("run_directory")
        sequence_root = detail.get("sequence_root")
        if not run_directory or not sequence_root:
            continue
        relative = posixpath.relpath(
            os.path.expandvars(str(run_directory)),
            os.path.expandvars(str(definition.root_directory)),
        )
        if relative in {".", ".."} or relative.startswith("../"):
            continue
        result_directories.append(
            posixpath.join(
                release_root,
                f"example_{definition.example_name}",
                relative,
            )
        )
    return result_directories or [f"{release_root}/example_{definition.example_name}"]


def postprocess_live_examples(
        tag=None, root=None, source_user=None, examples_root="examples",
        dry_run=False):
    if tag is None:
        tag = os.environ.get("CI_COMMIT_TAG")
    if not tag:
        raise ValueError("A release tag is required for postprocessing")

    definitions = scan_example_definitions(
        examples_root=examples_root,
        source_user=source_user,
    )
    statuses = resolve_live_statuses(
        examples_root=examples_root,
        source_user=source_user,
    )
    grouped = defaultdict(lambda: {"commands": []})
    seen = set()
    for definition in definitions:
        result = statuses.get(definition.identifier)
        if not result or str(result.get("overall", "")).upper() != "COMPLETED":
            continue

        local_parent = Path(examples_root) / _definition_parent(definition)
        remote_root = _definition_release_root(definition, tag, root=root)
        target_user = _target_user_from_path(remote_root, definition.machine)
        group = grouped[(definition.machine, target_user)]

        for script_name in POSTPROCESS_SCRIPTS:
            for script_path in local_parent.rglob(script_name):
                relative_script = script_path.relative_to(Path(examples_root))
                remote_script = Path(
                    f"/tmp_user/{definition.machine}/mola/{tag}/examples"
                    if not root else f"/tmp_user/{definition.machine}/{root}/examples"
                ) / relative_script

                script_key = (definition.machine, str(remote_script), script_name)
                if script_key in seen:
                    continue
                work_dir = shlex.quote(posixpath.dirname(str(remote_script)))
                group["commands"].append(
                    f"echo going to {work_dir};"
                    f"cd {work_dir}; "
                    f"{_script_command(remote_script)}"
                )
                seen.add(script_key)

    for (machine, target_user), group in sorted(grouped.items()):
        if not group["commands"]:
            continue
        print(f"postprocessing {len(group['commands'])} command(s) on {machine}")
        if dry_run:
            continue
        ssh_cmd = _relocate_ssh_command(machine, target_user)
        mola_env_cmd = SV.add_mola_env(machine, "elsa", target_user=target_user)
        bash_cmds = f"{mola_env_cmd}export MOLA_HEADLESS=1; {'; '.join(group['commands'])}"
        embedded = shlex.quote(f"bash -lc {shlex.quote(bash_cmds)}")
        full_cmd = f"{ssh_cmd} {embedded}"
        result = subprocess.run(full_cmd, shell=True, capture_output=False, text=True)
        if result.returncode != 0:
            print(f"ERROR: postprocess failed on {machine}: {result.stderr}", file=sys.stderr)


def postprocess_examples(
        tag=None, root=None, examples_root="examples", dry_run=False,
        source_user=None):
    """Run postprocessors once per relevant machine and result directory."""

    return postprocess_live_examples(
        tag=tag,
        root=root,
        source_user=source_user,
        examples_root=examples_root,
        dry_run=dry_run,
    )
