"""Move completed cases and publish their lightweight results."""

from __future__ import annotations

import os
import posixpath
import shlex
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import mola.naming_conventions as names
from mola import server as SV

from .state import ExampleCIState


def _state_paths(state_dir=None):
    manager = ExampleCIState(state_dir)
    return manager, sorted(
        manager.state_dir.glob(f"**/*{names.FILE_CI_STATE_SUFFIX}")
    )


def _version_from_state(state):
    return (state.get("ci_metadata", {}) or {}).get("version")


def _resolve_tag(states, tag=None):
    if tag:
        return str(tag)

    candidates = {
        str(version)
        for version in (_version_from_state(state) for state in states)
        if version
    }
    if not candidates and os.environ.get("CI_COMMIT_TAG"):
        return os.environ["CI_COMMIT_TAG"]
    if len(candidates) == 1:
        return candidates.pop()
    if not candidates:
        raise ValueError("A release tag is required for example repatriation")
    raise ValueError(f"Several release tags are present in CI state: {sorted(candidates)}")


def _example_parent(state):
    example_path = Path(str(state.get("example_path", "")))
    parts = example_path.parts
    if "examples" in parts:
        index = parts.index("examples")
        return Path(*parts[index + 1:-1])
    return example_path.parent


def _example_name(state):
    name = state.get("example_name")
    if name:
        return str(name)
    return Path(str(state.get("example_path", "example"))).stem


def _case_source_directory(state):
    if state.get("execution_mode") == "multi" and state.get("root_directory"):
        return str(state["root_directory"])
    if state.get("run_directory"):
        return str(state["run_directory"])
    directories = ExampleCIState.get_all_run_directories(state)
    if directories:
        return str(directories[0])
    raise ValueError(f"No remote case directory recorded for {_example_name(state)}")


def _machine(state, source):
    return str(state.get("machine") or SV.guess_machine(source))


def _release_user(state):
    return str(
        state.get("repatriation_user")
        or os.environ.get("MOLA_RELEASE_USER")
        or state.get("source_user")
        or state.get("user")
        or os.environ.get("USER")
        or "mola"
    )


def _target_user_from_path(path):
    """Extract the target SSH user from a destination path.

    For paths like /tmp_user/{machine}/{user}/..., user is {user}.
    For paths like /stck/{user}/..., user is {user}.
    Returns None if the user cannot be determined.
    """
    parts = path.strip("/").split("/")
    if parts[0] == "tmp_user" and len(parts) >= 4:
        return parts[3]
    if parts[0] == "stck" and len(parts) >= 2:
        return parts[1]
    return None


def _relocate_ssh_command(machine, target_user):
    if not target_user:
        raise ValueError(
            f"Cannot determine target user from destination path for machine {machine}"
        )
    ssh_key = f"/stck/{target_user}/.ssh/{machine}_ci_deploy_ed25519"
    return (
        f"ssh -4 -o BatchMode=yes -o IdentitiesOnly=yes "
        f"-o StrictHostKeyChecking=no "
        f"-i {shlex.quote(ssh_key)} "
        f"{target_user}@{machine}"
    )


def release_examples_root(machine, tag, root=None):
    if root:
        return f"/tmp_user/{machine}/{root}/examples"
    return f"/tmp_user/{machine}/mola/{tag}/examples"


def public_examples_root(tag, root=None):
    if root:
        return f"/stck/{root}/examples"
    return f"/stck/mola/{tag}/examples"


def release_directory(state, tag, machine=None, root=None):
    machine = machine or _machine(state, _case_source_directory(state))
    relative_parent = _example_parent(state)
    if root:
        target = Path(f"/tmp_user/{machine}/{root}/examples") / relative_parent
    else:
        target = Path(release_examples_root(machine, tag)) / relative_parent
    return str(target / f"example_{_example_name(state)}")


def _completed_entries(manager, state_paths):
    entries = []
    for state_path in state_paths:
        state = manager.read(state_path)
        status = str(state.get("status", "")).upper()
        if status != names.FILE_JOB_COMPLETED:
            print(f"Skipping {state_path.stem} (status: {status or 'UNKNOWN'})")
            continue
        source = _case_source_directory(state)
        machine = _machine(state, source)
        entries.append({
            "state_path": state_path,
            "state": state,
            "source": source,
            "machine": machine,
            "remote_user": _release_user(state),
        })
    return entries


def _copy_command(source, destination):
    source = os.path.expandvars(source)
    return (
        f"mkdir -p {shlex.quote(posixpath.dirname(destination))}; "
        f"rm -rf {shlex.quote(destination)}; "
        f"cp -r {shlex.quote(source)} {shlex.quote(destination)}"
    )


def _target_user_from_path(destination, machine):
    """Extract target SSH user from destination path.

    /tmp_user/{machine}/{user}/... → {user}
    /stck/{user}/... → {user}
    Returns None if cannot determine.
    """
    parts = destination.split("/")
    if len(parts) >= 4 and parts[1] == "tmp_user" and parts[2] == machine:
        return parts[3]
    if len(parts) >= 3 and parts[1] == "stck":
        return parts[2]
    return None


def _relocate_ssh_command(machine, target_user):

    if not target_user:
        raise ValueError(
            f"Cannot determine target user for machine {machine}"
        )
    local_user = os.environ.get("USER")
    if not local_user:
        raise ValueError(
            "Cannot determine local user (USER environment variable not set)"
        )

    if target_user != local_user:
        ssh_key = f"/stck/{local_user}/.ssh/{machine}_ci_deploy_ed25519"
        return (
            f"ssh -4 -o BatchMode=yes -o IdentitiesOnly=yes "
            f"-o StrictHostKeyChecking=no "
            f"-i {shlex.quote(ssh_key)} "
            f"{target_user}@{machine}"
        )
    else:
        return (
            f"ssh -4 -o BatchMode=yes "
            f"{target_user}@{machine}"
                )

def relocate_examples(state_dir=None, tag=None, root=None, dry_run=False):
    """Move completed remote cases into their machine-specific release tree."""
    manager, state_paths = _state_paths(state_dir)
    states = [manager.read(path) for path in state_paths]
    tag = _resolve_tag(states, tag)
    entries = _completed_entries(manager, state_paths)
    grouped = defaultdict(list)
    for entry in entries:
        destination = release_directory(entry["state"], tag, machine=entry["machine"], root=root)
        target_user = _target_user_from_path(destination, entry["machine"])
        if not target_user:
            raise ValueError(
                f"Cannot determine target user for destination: {destination}"
            )
        entry["destination"] = destination
        entry["target_user"] = target_user
        grouped[(entry["machine"], target_user)].append(entry)

    for (machine, target_user), machine_entries in sorted(grouped.items()):
        commands = []
        destinations = {}
        for entry in machine_entries:
            destination = entry["destination"]
            destinations[entry["state_path"]] = destination
            commands.append(_copy_command(entry["source"], destination))

        print(f"relocating {len(commands)} example(s) on {machine}")
        if not dry_run:
            ssh_cmd = _relocate_ssh_command(machine, target_user)
            bash_cmd = "; ".join(commands)
            full_cmd = f"{ssh_cmd} {shlex.quote(bash_cmd)}"
            result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"ERROR: relocation failed on {machine}: {result.stderr}", file=sys.stderr)
                continue
            for state_path, destination in destinations.items():
                manager.update_state(state_path, {
                    "release_directory": destination,
                    "repatriated_at": datetime.now().isoformat(),
                    "repatriation_machine": machine,
                })


def _light_filters():
    included = set(names.REPATRIATION_LIGHT_PATTERNS)
    included.update(names.STATUS_FILES)
    return included, {"*"}


def publish_light_results(state_dir=None, tag=None, dry_run=False):
    """Publish signals, coprocess logs, and generated images to /stck."""
    manager, state_paths = _state_paths(state_dir)
    states = [manager.read(path) for path in state_paths]
    tag = _resolve_tag(states, tag)
    entries = _completed_entries(manager, state_paths)
    included, excluded = _light_filters()
    published_parents = set()

    for entry in entries:
        state = entry["state"]
        machine = entry["machine"]
        source = state.get("release_directory") or release_directory(state, tag, machine)
        destination = str(
            Path(public_examples_root(tag))
            / _example_parent(state)
            / f"example_{_example_name(state)}"
        )
        print(f"Publishing {_example_name(state)} from {machine}")
        if not dry_run:
            SV.rsync(
                source_path=f"{source}/",
                source_machine=machine,
                source_user=entry["remote_user"],
                destination_path=f"{destination}/",
                destination_machine="localhost",
                included_files=included,
                excluded_files=excluded,
            )

        parent_key = (machine, posixpath.dirname(source), str(_example_parent(state)))
        if parent_key in published_parents:
            continue
        published_parents.add(parent_key)
        source_parent = posixpath.dirname(source)
        destination_parent = str(Path(public_examples_root(tag)) / _example_parent(state))
        if not dry_run:
            SV.rsync(
                source_path=f"{source_parent}/",
                source_machine=machine,
                source_user=entry["remote_user"],
                destination_path=f"{destination_parent}/",
                destination_machine="localhost",
                included_files=set(names.REPATRIATION_IMAGE_PATTERNS),
                excluded_files={"*"},
            )


def repatriate_state(state_path, mode="full", destination=None, tag=None, root=None):
    """Compatibility wrapper for callers that operate on one state file."""
    state_path = Path(state_path)
    if destination is not None:
        included, excluded = SV.get_repatriation_filters(mode)
        SV.get_cases_from_submission_state(
            state_path,
            included_files=included,
            excluded_files=excluded,
            destination_path=str(destination),
            flatten_single_case=True,
        )
        return
    if mode == "full":
        relocate_examples(state_path.parent, tag=tag, root=root)
    elif mode == "light":
        publish_light_results(state_path.parent, tag=tag)
    else:
        raise ValueError(f"Unknown repatriation mode: {mode}")


def repatriate_examples(state_dir=None, mode="full", destination=None, tag=None, root=None):
    if destination is not None:
        for state_path in _state_paths(state_dir)[1]:
            repatriate_state(state_path, mode=mode, destination=destination, tag=tag)
        return
    if mode == "full":
        return relocate_examples(state_dir=state_dir, tag=tag, root=root)
    if mode == "light":
        return publish_light_results(state_dir=state_dir, tag=tag)
    raise ValueError(f"Unknown repatriation mode: {mode}")
