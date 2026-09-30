"""Run example comparison and visualization scripts on each HPC machine."""

from __future__ import annotations

import os
import posixpath
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from mola import server as SV

from .repatriate import (
    _completed_entries,
    _example_parent,
    _resolve_tag,
    _state_paths,
    _target_user_from_path,
    _relocate_ssh_command,
    release_directory,
)
from .state import ExampleCIState

POSTPROCESS_SCRIPTS = ["compare.py", "visualize.py"]


def _remote_examples_root(machine, tag, root=None):
    if root:
        return f"/tmp_user/{machine}/{root}/examples"
    return f"/tmp_user/{machine}/mola/{tag}/examples"


def _script_command(script_path):
    return f"python3 {shlex.quote(str(script_path))}"


def _result_directories(entry, tag, root=None):
    state = entry["state"]
    machine = entry["machine"]
    release_root = state.get("release_directory") or release_directory(
        state, tag, machine, root=root
    )
    if state.get("execution_mode") != "multi":
        return [release_root]

    root_directory = state.get("root_directory")
    if not root_directory:
        return [release_root]
    root_directory = os.path.expandvars(str(root_directory))
    result_directories = []
    for run_directory in ExampleCIState.get_all_run_directories(state):
        relative_directory = posixpath.relpath(
            os.path.expandvars(str(run_directory)), root_directory
        )
        if relative_directory == ".":
            continue
        if relative_directory.startswith("../") or relative_directory == "..":
            continue
        result_directories.append(
            posixpath.normpath(posixpath.join(release_root, relative_directory))
        )
    return result_directories or [release_root]


def postprocess_examples(state_dir=None, tag=None, root=None, examples_root="examples",
                         dry_run=False):
    """Run postprocessors once per relevant machine and result directory."""
    manager, state_paths = _state_paths(state_dir)
    states = [manager.read(path) for path in state_paths]
    if tag is None:
        tag = _resolve_tag(states, tag)
    entries = _completed_entries(manager, state_paths)

    grouped = defaultdict(lambda: {"commands": [], "entries": []})
    seen = set()
    for entry in entries:
        state = entry["state"]
        machine = entry["machine"]
        local_parent = Path(examples_root) / _example_parent(state)
        remote_root = _remote_examples_root(machine, tag, root=root)
        target_user = _target_user_from_path(remote_root, machine)
        key = (machine, target_user)
        group = grouped[key]
        group["entries"].append(entry)

        for script_name in POSTPROCESS_SCRIPTS:
            for script_path in local_parent.rglob(script_name):
                relative_script = script_path.relative_to(Path(examples_root))
                remote_script = Path(remote_root) / relative_script

                if script_name == "visualize.py":
                    for result_directory in _result_directories(entry, tag, root=root):
                        script_key = (machine, str(result_directory), str(remote_script))
                        if script_key not in seen:
                            group["commands"].append(
                                f"cd {shlex.quote(result_directory)}; "
                                f"{_script_command(remote_script)}"
                            )
                            seen.add(script_key)
                else:
                    script_key = (machine, str(remote_script), script_name)
                    if script_key not in seen:
                        group["commands"].append(
                            f"cd {shlex.quote(posixpath.dirname(str(remote_script)))}; "
                            f"{_script_command(remote_script)}"
                        )
                        seen.add(script_key)

    for (machine, target_user), group in sorted(grouped.items()):
        if not group["commands"]:
            continue
        print(f"postprocessing {len(group['commands'])} command(s) on {machine}")
        if not dry_run:
            ssh_cmd = _relocate_ssh_command(machine, target_user)
            mola_env_cmd = SV.add_mola_env(machine, 'elsa', target_user=target_user)
            bash_cmds = f"{mola_env_cmd}export MOLA_HEADLESS=1; {'; '.join(group['commands'])}"
            embedded = shlex.quote(f"bash -lc {shlex.quote(bash_cmds)}")
            full_cmd = f"{ssh_cmd} {embedded}"
            result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"ERROR: postprocess failed on {machine}: {result.stderr}",
                      file=sys.stderr)
                continue
            for entry in group["entries"]:
                manager.update_state(entry["state_path"], {
                    "postprocessed_at": datetime.now().isoformat(),
                })
