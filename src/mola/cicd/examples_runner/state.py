"""Persistent state for submitted examples."""

from __future__ import annotations

import hashlib
import os
import posixpath
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Optional

import yaml

import mola.naming_conventions as names
from mola import server as SV

from .status_helper import normalize_submission, normalize_submissions, status_is_failed, unique


STATE_SCHEMA_VERSION = 3


class ExampleCIState:
    """Read and update example submission state files."""

    def __init__(self, state_dir=None):
        self.state_dir = Path(state_dir or names.DIRECTORY_CI_STATE)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def get_state_path(self, example_path: Path) -> Path:
        example_path = Path(example_path)
        try:
            relative_directory = example_path.parent.resolve().relative_to(Path.cwd().resolve())
        except ValueError:
            parent_id = hashlib.sha256(
                str(example_path.parent.resolve()).encode("utf-8")
            ).hexdigest()[:16]
            relative_directory = Path("_external") / parent_id
        return self.state_dir / relative_directory / (
            f"{example_path.stem}{names.FILE_CI_STATE_SUFFIX}"
        )

    def create_state(self, example_path: Path, execution_mode: str, state_data: Dict) -> Path:
        state = {
            "schema_version": STATE_SCHEMA_VERSION,
            "example_path": str(example_path),
            "execution_mode": execution_mode,
            **(state_data or {}),
        }
        state.setdefault("submissions", [])
        state["submissions"] = normalize_submissions(state["submissions"])
        state.setdefault("ci_metadata", {
            "pipeline_id": None,
            "version": None,
            "submitted_at": None,
            "last_checked_at": None,
        })
        state_path = self.get_state_path(example_path)
        self.write(state_path, state)
        return state_path

    @staticmethod
    def read(state_path: Path) -> Dict:
        with open(state_path, encoding="utf-8") as stream:
            return yaml.safe_load(stream) or {}

    @staticmethod
    def write(state_path: Path, state: Dict) -> None:
        state_path = Path(state_path)
        state_path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=state_path.parent,
                prefix=f".{state_path.name}.", delete=False) as stream:
            yaml.safe_dump(state, stream, default_flow_style=False, sort_keys=False)
            temporary_path = Path(stream.name)
        temporary_path.replace(state_path)

    def update_state(self, state_path: Path, updates: Dict) -> None:
        if not Path(state_path).exists():
            raise FileNotFoundError(f"CI state file not found: {state_path}")
        state = self.read(state_path)
        for key, value in updates.items():
            if key == "jobs" and isinstance(value, dict) and "cfd" in value:
                jobs = state.setdefault("jobs", {})
                existing = jobs.setdefault("cfd", [])
                for job_id in value["cfd"]:
                    if job_id not in existing:
                        existing.append(job_id)
            else:
                state[key] = value
        if "submissions" in state:
            state["submissions"] = normalize_submissions(state["submissions"])
        self.write(state_path, state)

    def check_status(self, state_path: Path):
        from .status import resolve_state_status

        state = self.read(state_path)
        status = resolve_state_status(state)
        updates = {"status": status}
        if isinstance(status, dict):
            updates = {
                "status": status.get("overall", "unknown"),
                "details": status.get("details", []),
            }
        self.update_state(state_path, updates)
        return status

    def check_status_mode_mono(self, state: Dict):
        from .status import resolve_state_status
        return resolve_state_status({**state, "execution_mode": "mono"})

    def check_status_mode_multi(self, state: Dict):
        from .status import resolve_state_status
        return resolve_state_status({**state, "execution_mode": "multi"})

    @staticmethod
    def _unique_run_directories(run_directories: Iterable) -> list:
        return unique(run_directories)

    @staticmethod
    def _is_failed_status(status) -> bool:
        return status_is_failed(status)

    @classmethod
    def is_failed_state(cls, state: Optional[Dict]) -> bool:
        if not state:
            return False
        if cls._is_failed_status(state.get("status")):
            return True
        return any(
            cls._is_failed_status(detail.get("status"))
            for detail in state.get("details", []) or []
            if isinstance(detail, dict)
        )

    @classmethod
    def get_submissions(cls, state: Optional[Dict]) -> list[dict]:
        if not state:
            return []
        if state.get("submissions"):
            return normalize_submissions(state["submissions"])

        jobs = state.get("jobs", {}) or {}
        machine = state.get("machine", "localhost")
        user = state.get("user")
        scheduler = state.get("scheduler", "SLURM")
        solver = state.get("solver")

        sequences = jobs.get("sequences") or []
        if sequences:
            return normalize_submissions([
                {
                    **sequence,
                    "scheduler": sequence.get("scheduler", scheduler),
                    "machine": sequence.get("machine", machine),
                    "user": sequence.get("user", user),
                    "solver": sequence.get("solver", solver),
                }
                for sequence in sequences
            ])

        run_directories = cls.get_all_run_directories(state)
        job_ids = jobs.get("sequential") or jobs.get("cfd") or []
        if not job_ids:
            return [{
                "job_id": None,
                "scheduler": scheduler,
                "machine": machine,
                "user": user,
                "solver": solver,
                "run_directories": run_directories,
            }] if run_directories else []

        return normalize_submissions([
            {
                "job_id": job_id,
                "scheduler": scheduler,
                "machine": machine,
                "user": user,
                "solver": solver,
                "run_directories": run_directories,
            }
            for job_id in job_ids
        ])

    @classmethod
    def get_all_run_directories(cls, state: Optional[Dict]) -> list:
        if not state:
            return []
        directories = []
        for submission in state.get("submissions", []) or []:
            directories.extend(submission.get("run_directories", []))
        directories.extend(state.get("run_directories", []) or [])
        directories.extend(
            detail.get("run_directory")
            for detail in state.get("details", []) or []
            if isinstance(detail, dict)
        )
        if state.get("execution_mode") == "mono":
            directories.append(state.get("run_directory"))
        return cls._unique_run_directories(directories)

    @classmethod
    def get_failed_run_directories(cls, state: Optional[Dict]) -> list:
        if not cls.is_failed_state(state):
            return []
        failed = [
            detail.get("run_directory")
            for detail in state.get("details", []) or []
            if isinstance(detail, dict) and cls._is_failed_status(detail.get("status"))
        ]
        return cls._unique_run_directories(failed or cls.get_all_run_directories(state))

    @staticmethod
    def remove_remote_run_directories(state: Dict, run_directories: list) -> list:
        machine = state.get("machine", "localhost")
        user = state.get("user")
        errors = []
        for run_directory in run_directories:
            print(f"  Removing remote run directory: {run_directory}")
            try:
                SV.remove_path(run_directory, machine=machine, user=user, file_only=False)
            except Exception as error:
                errors.append((run_directory, error))
                print(f"  Error removing {run_directory}: {error}")
        return errors

    @staticmethod
    def _join_case_path(run_directory: str, filename: str, machine: str) -> str:
        if machine == "localhost":
            return str(Path(run_directory) / filename)
        return posixpath.join(str(run_directory), filename)

    def _get_run_directories_from_manager_file(
            self, root_dir: str, script_dir: str, machine: str, user: Optional[str]) -> list:
        """Read legacy manager state while old artifacts are being migrated."""
        candidates = []
        if root_dir:
            candidates.append((
                self._join_case_path(root_dir, names.FILE_WORKFLOW_MANAGER, machine),
                machine,
                user,
            ))
        if script_dir:
            candidates.append((
                self._join_case_path(script_dir, names.FILE_WORKFLOW_MANAGER, "localhost"),
                "localhost",
                None,
            ))

        for manager_yaml, source_machine, source_user in candidates:
            try:
                if not SV.is_existing_path(manager_yaml, machine=source_machine,
                                           user=source_user, file_only=True):
                    continue
                with tempfile.TemporaryDirectory(prefix="mola-ci-") as directory:
                    local_yaml = Path(directory) / names.FILE_WORKFLOW_MANAGER
                    SV.copy_remote(
                        source_path=manager_yaml,
                        source_machine=source_machine,
                        source_user=source_user,
                        destination_path=str(local_yaml),
                        destination_machine="localhost",
                        force_copy=True,
                    )
                    data = yaml.safe_load(local_yaml.read_text(encoding="utf-8")) or {}
                return list(data.get("RunDirectories", data.get("run_directories", [])))
            except Exception:
                continue
        return []
