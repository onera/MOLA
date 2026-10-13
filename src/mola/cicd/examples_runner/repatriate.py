"""Move completed cases and publish their lightweight results."""

from __future__ import annotations

import os
import posixpath
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import mola.naming_conventions as names
from mola import server as SV

from .discovery import ExampleDefinition, scan_example_definitions
from .status import resolve_live_statuses


def release_examples_root(machine, tag, root=None):
    if root:
        return f"/tmp_user/{machine}/{root}/examples"
    return f"/tmp_user/{machine}/mola/{tag}/examples"


def public_examples_root(tag, root=None):
    if root:
        return f"/stck/{root}/examples"
    return f"/stck/mola/{tag}/examples"


def _definition_parent(definition: ExampleDefinition) -> Path:
    parts = definition.script_path.parts
    if "examples" in parts:
        index = parts.index("examples")
        return Path(*parts[index + 1:-1])
    return definition.script_path.parent


def _definition_release_directory(definition, tag, root=None):
    return str(
        Path(release_examples_root(definition.machine, tag, root=root))
        / _definition_parent(definition)
        / f"example_{definition.example_name}"
    )


def _definition_source_directory(definition):
    return (
        definition.root_directory
        if definition.mode == "multi"
        else definition.run_directory
    )


def _live_completed_entries(
        examples_root="examples", source_user=None, tag=None, root=None):
    if tag is None:
        tag = os.environ.get("CI_COMMIT_TAG")
    if not tag:
        raise ValueError("A release tag is required for example repatriation")

    statuses = resolve_live_statuses(
        examples_root=examples_root,
        source_user=source_user,
    )
    entries = []
    for result in statuses.values():
        definition = result["definition"]
        status = str(result.get("overall", "")).upper()
        if status != names.FILE_JOB_COMPLETED:
            print(f"\033[93mSkipping {definition.script_path} (status: {status or 'UNKNOWN'})\033[0m")
            continue
        entries.append({
            "definition": definition,
            "source": _definition_source_directory(definition),
            "machine": definition.machine,
            "destination": _definition_release_directory(definition, tag, root=root),
        })
    return tag, entries


def _copy_command(source, destination):
    source = os.path.expandvars(source)
    return (
        f"mkdir -p {shlex.quote(posixpath.dirname(destination))}; "
        f"rm -rf {shlex.quote(destination)}; "
        f"cp -r {shlex.quote(source)} {shlex.quote(destination)}"
    )


def _target_user_from_path(destination, machine):
    """Extract the target SSH user from a result path."""

    parts = destination.split("/")
    if len(parts) >= 4 and parts[1] == "tmp_user" and parts[2] == machine:
        return parts[3]
    if len(parts) >= 3 and parts[1] == "stck":
        return parts[2]
    return None


def _relocate_ssh_command(machine, target_user):
    if not target_user:
        raise ValueError(f"Cannot determine target user for machine {machine}")
    local_user = os.environ.get("USER")
    if not local_user:
        raise ValueError("Cannot determine local user (USER environment variable not set)")

    machine_key = f"/stck/{local_user}/.ssh/{machine}_ci_deploy_ed25519"
    default_key = f"/stck/{local_user}/.ssh/mola_ci_deploy_ed25519"
    if os.path.isfile(machine_key) and os.access(machine_key, os.R_OK):
        ssh_key = machine_key
    elif os.path.isfile(default_key) and os.access(default_key, os.R_OK):
        ssh_key = default_key
    else:
        return f"ssh -4 -o BatchMode=yes {target_user}@{machine}"

    return (
        f"ssh -4 -o BatchMode=yes -o IdentitiesOnly=yes "
        f"-o StrictHostKeyChecking=no -i {shlex.quote(ssh_key)} "
        f"{target_user}@{machine}"
    )


def relocate_live_examples(
        tag=None, root=None, source_user=None, examples_root="examples",
        dry_run=False):
    
    tag, entries = _live_completed_entries(
        examples_root=examples_root,
        source_user=source_user,
        tag=tag,
        root=root,
    )
    
    grouped = defaultdict(list)
    for entry in entries:
        target_user = _target_user_from_path(entry["destination"], entry["machine"])
        if not target_user:
            raise ValueError(
                f"Cannot determine target user for destination: {entry['destination']}"
            )
        entry["target_user"] = target_user
        grouped[(entry["machine"], target_user)].append(entry)

    for (machine, target_user), machine_entries in sorted(grouped.items()):
        commands = [
            _copy_command(entry["source"], entry["destination"])
            for entry in machine_entries
        ]
        print(f"relocating {len(commands)} example(s) on {machine}")
        if dry_run:
            continue
        ssh_cmd = _relocate_ssh_command(machine, target_user)
        full_cmd = f"{ssh_cmd} {shlex.quote('; '.join(commands))}"
        result = subprocess.run(full_cmd, shell=True, capture_output=True, text=True)
        if result.returncode != 0:
            msg = f"ERROR: relocation failed on {machine}: {result.stderr}"
            print(msg,file=sys.stderr)
            raise RuntimeError(msg)



def relocate_examples(
        tag=None, root=None, dry_run=False, source_user=None,
        examples_root="examples"):
    """Move completed remote cases into the release tree."""

    return relocate_live_examples(
        tag=tag,
        root=root,
        source_user=source_user,
        examples_root=examples_root,
        dry_run=dry_run,
    )


def _light_filters():
    included = set(names.REPATRIATION_LIGHT_PATTERNS)
    included.update(names.STATUS_FILES)
    return included, {"*"}


def publish_live_results(
        tag=None, root=None, source_user=None, examples_root="examples",
        dry_run=False):
    if tag is None:
        tag = os.environ.get("CI_COMMIT_TAG")
    if not tag:
        raise ValueError("A release tag is required for example repatriation")

    definitions = scan_example_definitions(
        examples_root=examples_root,
        source_user=source_user,
    )
    included, excluded = _light_filters()
    published_parents = set()
    for definition in definitions:
        machine = definition.machine
        source = _definition_release_directory(definition, tag, root=root)
        if not SV.is_existing_path(source, machine=machine, user=None, file_only=False):
            print(f"Skipping {definition.example_name}: release directory is absent")
            continue

        destination = str(
            Path(public_examples_root(tag, root=root))
            / _definition_parent(definition)
            / f"example_{definition.example_name}"
        )
        release_user = _target_user_from_path(source, machine)
        if not release_user:
            raise ValueError(f"Cannot determine release user for {source}")

        print(f"Publishing {definition.example_name} from {machine}")
        if not dry_run:
            SV.rsync(
                source_path=f"{source}/",
                source_machine=machine,
                source_user=release_user,
                destination_path=f"{destination}/",
                destination_machine="localhost",
                included_files=included,
                excluded_files=excluded,
            )

        parent_key = (machine, posixpath.dirname(source), str(_definition_parent(definition)))
        if parent_key in published_parents:
            continue
        published_parents.add(parent_key)
        source_parent = posixpath.dirname(source)
        destination_parent = str(
            Path(public_examples_root(tag, root=root)) / _definition_parent(definition)
        )
        if not dry_run:
            SV.rsync(
                source_path=f"{source_parent}/",
                source_machine=machine,
                source_user=release_user,
                destination_path=f"{destination_parent}/",
                destination_machine="localhost",
                included_files=set(names.REPATRIATION_IMAGE_PATTERNS),
                excluded_files={"*"},
            )


def publish_light_results(
        tag=None, dry_run=False, root=None, source_user=None,
        examples_root="examples"):
    """Publish signals, coprocess logs, and generated images to ``/stck``."""

    return publish_live_results(
        tag=tag,
        root=root,
        source_user=source_user,
        examples_root=examples_root,
        dry_run=dry_run,
    )


def repatriate_examples(
        mode="full", tag=None, root=None, source_user=None,
        examples_root="examples", dry_run=False):
    if mode == "full":
        return relocate_examples(
            tag=tag,
            root=root,
            source_user=source_user,
            examples_root=examples_root,
            dry_run=dry_run,
        )
    if mode == "light":
        return publish_light_results(
            tag=tag,
            root=root,
            source_user=source_user,
            examples_root=examples_root,
            dry_run=dry_run,
        )
    raise ValueError(f"Unknown repatriation mode: {mode}")
