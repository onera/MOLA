#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

import os
import pathlib
import copy
import numpy as np
from pathlib import Path
import json


import psutil

from treelab import cgns

import mola.naming_conventions as names
from mola import server as SV 
from mola import solver, __version__, __MOLA_PATH__
from mola.logging import (mola_logger,
                       MolaException,
                       MolaUserError,
                       redirect_streams_to_logger, 
                       GREEN, CYAN, BOLD, ENDC)
from  mola.cfd.preprocess.mesh import (
    io,
    positioning,
    connect,
    split,
    families,
    duplicate,
    overset
    )
from mola.cfd.preprocess.mesh.io import unstructured
from  mola.cfd.preprocess import (
                               compatibilities_adjustor,
                               flow_generators,
                               boundary_conditions,
                               initialization,
                               motion,
                               cfd_parameters,
                               extractions,
                               run_manager,
                               finalization,
                               write_cfd_files,
                               check)
from mola.cfd.postprocess import remove_cfd_files
from mola.cfd.compute import compute
from mola.server.remote import guess_machine

from mola.workflow.interface import WorkflowInterface


class Workflow(object):

    def __init__(self, **kwargs):
        self._interface = WorkflowInterface(self, **kwargs)
        
    def prepare(self):
        start_msg = f'START PREPROCESS WITH MOLA {__version__}'
        mola_logger.info(f'{CYAN}{BOLD}{"="*len(start_msg)}\n{start_msg}\n{"="*len(start_msg)}{ENDC}', rank=0)
        self.adjust_compatibilities()
        self.prepare_job()
        self.process_mesh()
        self.process_overset()
        self.compute_flow_and_turbulence()
        self.set_motion()
        self.set_boundary_conditions()
        self.set_cfd_parameters()
        self.initialize_flow()
        self.set_extractions()
        self.check_preprocess()
        self.finalize_preprocess()
        mola_logger.info(f'✅ {GREEN}{BOLD}PREPROCESS FINALIZED{ENDC}', rank=0)

    @property
    def tree(self):
        return self._tree

    @tree.setter
    def tree(self, value):
        if value is None or isinstance(value, (str, Path)):
            self._tree = value
        else:
            # convert to treelab format 
            self._tree = cgns.castNode(value)

    def process_mesh(self):
        mola_logger.info("🔧 processing mesh", rank=0)
        self.assemble()
        self.positioning()
        self.connect()
        self.define_families()
        unstructured.apply(self)
        self.duplicate()
        self.split_and_distribute()

    def check_consistency_between_solver_and_environment(self):
        requested_solver = self.Solver
        env_solver = os.environ.get('MOLA_SOLVER')
        if requested_solver != env_solver:
            raise MolaException((f'the requested solver "{requested_solver}" does not'
                f'match the type of environment "{env_solver}"'))

    def prepare_job(self):
        run_manager.apply(self)

    def assemble(self):
        self.read_meshes()
        self.set_workflow_parameters_in_tree()

    def positioning(self):
        positioning.apply(self)
        self.set_workflow_parameters_in_tree()

    def connect(self):
        connect.apply(self)
        self.set_workflow_parameters_in_tree()

    def define_families(self):
        families.apply(self)
        self.set_workflow_parameters_in_tree()

    def duplicate(self):
        duplicate.apply(self)
        self.set_workflow_parameters_in_tree()

    def read_meshes(self):
        io.apply(self)

    def split_and_distribute(self):
        split.apply(self)
        self.set_workflow_parameters_in_tree()

    def process_overset(self):
        self.tree = overset.addOversetData(self.tree, self.RawMeshComponents, 
                        run_directory=self.RunManagement["RunDirectory"],
                        **self.Overset)
        self.set_workflow_parameters_in_tree()

    def compute_flow_and_turbulence(self):
        mola_logger.info("🌀 setting flow and turbulence parameters", rank=0)
        # mola-generic set of parameters

        # CAVEAT weird way of passing data, excessively relying on direct members
        # access(publicly, which is dangerous). Should redesign the flow of data,
        # for instance using constructor's arguments and dedicated accessors
        # (getters/setters)
        FlowGen = flow_generators.get_flow_generator(self.Flow['Generator'])(self)
        FlowGen.generate()
        self.Fluid = FlowGen.Fluid
        self.Flow = FlowGen.Flow
        self.Turbulence = FlowGen.Turbulence
        self.set_workflow_parameters_in_tree()

    def initialize_flow(self):
        mola_logger.info("🔥 initialize flow", rank=0)
        initialization.apply(self)
        self.set_workflow_parameters_in_tree()
    
    def set_boundary_conditions(self):
        mola_logger.info("🚧 setting boundary conditions", rank=0)
        boundary_conditions.apply(self)
        self.set_workflow_parameters_in_tree()

    def set_motion(self):
        mola_logger.info("🐎 setting motion parameters", rank=0)
        motion.apply(self)
        self.set_workflow_parameters_in_tree()

    def set_cfd_parameters(self):
        mola_logger.info("💻 setting cfd modeling and numerical parameters", rank=0)
        cfd_parameters.apply(self)
        self.set_workflow_parameters_in_tree()

    def set_extractions(self):
        mola_logger.info("🔎 setting extractions", rank=0)
        extractions.apply(self)
        self.set_workflow_parameters_in_tree()
    
    def check_preprocess(self):
        mola_logger.info("✔ checking preprocess consistency", rank=0)
        check.apply(self)

    def finalize_preprocess(self):
        mola_logger.info("🏆 finalizing preprocess", rank=0)
        finalization.apply(self)
        self.RunManagement['PreprocessDone'] = True

    def write_cfd_files(self, items : str = 'all'):
        '''
            items : str 
                See options in `:py:func:write_cfd_files.apply`
        '''
        mola_logger.info("💾 write cfd files", rank=0)
        write_cfd_files.apply(self, items=items) 
    
    def remove_cfd_files(self):
        remove_cfd_files.apply(self)

    def compute(self):
        compute.apply(self)

    def visu(self):
        pass

    def get_component(self, base_name):
        for component in self.RawMeshComponents:
            if component['Name']==base_name:
                return component

    def has_overset_component(self):
        for component in self.RawMeshComponents:
            if 'OversetOptions' not in component: 
                continue
            if component['OversetOptions']:
                return True
        return False

    def has_moving_overset_component(self):
        for component in self.RawMeshComponents:
            if 'OversetMotion' in component and component['OversetMotion']:
                return True
        return False


    def submit(self, command=None):
        """Submit the workflow and return the scheduler command output."""
        
        run_directory = self.RunManagement['RunDirectory']
        machine = self.RunManagement.get('Machine',guess_machine(run_directory))

        if not self.RunManagement.get('PreprocessDone',False):
            mola_logger.info("🔧 Will do preprocess remotely at:")
            mola_logger.info(f"   directory: {run_directory}")
            mola_logger.info(f"   machine: {machine}")
            self.prepare_job()
            self.set_workflow_parameters_in_tree()
            self.write_cfd_files(items='only_job')
            self.write_tree_remote()

        
        mola_logger.info(f"🚀 Submit job on machine {machine}")
        if command is None:
            command = self.RunManagement['LauncherCommand']
        user = self.RunManagement.get('User')
        out = SV.submit_command(command, machine, user=user)
        return out
    
    def write_tree_remote(self, data_directory=None, copy_options=None):
        from . import manager as WM
        sender = WM.WorkflowSender(self, data_directory=data_directory, copy_options=copy_options)
        sender.apply()

    def write_tree(self, filename=names.FILE_INPUT_SOLVER):
        if not self.tree: 
            self.tree = cgns.Tree()
        # with redirect_streams_to_logger(mola_logger):
        io.writer.write(self, self.tree, filename)

    def merge(self, other_workflow):
        # TODO Still in development, not validated
        mola_logger.user_warning(f'Merge workflows')

        self._merge_trees(other_workflow)

        self._update_interfaces_between_workflows(other_workflow)
        
        # merge attributes
        self.RawMeshComponents += other_workflow.RawMeshComponents
        # Handle ApplicationContext ?
        self.BoundaryConditions += other_workflow.BoundaryConditions
        self.BodyForceModeling += other_workflow.BodyForceModeling
        self.Extractions += other_workflow.Extractions
        self.ConvergenceCriteria += other_workflow.ConvergenceCriteria

    def _merge_trees(self, other_workflow):
        other_tree = other_workflow.tree
        other_tree.findAndRemoveNode(Name=self._workflow_parameters_container_, Depth=1)
        main_base = self.tree.get(Type='CGNSBase', Depth=1)
        secondary_basename = other_tree.get(Type='CGNSBase', Depth=1).name()

        self.tree.merge(other_tree)

        # Move children of secondary base to the main base if they don't already exists in main base
        secondary_base = self.tree.get(Name=secondary_basename, Type='CGNSBase', Depth=1)
        children_to_move = copy.copy(secondary_base.children())
        for child in children_to_move:
            if not main_base.get(Name=child.name(), Type=child.type(), Depth=1):
                child.moveTo(main_base)
        secondary_base.remove()

    def _update_interfaces_between_workflows(self, other_workflow):
        updated_boundary_conditions = []
        for bc in self.BoundaryConditions + other_workflow.BoundaryConditions:
            if bc['Type'] != 'InterfaceBetweenWorkflows':
                continue

            if not 'TypeOfInterface' in bc:
                raise MolaException(
                    f"The boundary condition on Family {bc['Family']} is of Type {bc['Type']},"
                    "and for this Type the key 'TypeOfInterface' must be defined."
                    )
            
            elif isinstance(bc['TypeOfInterface'], str):
                assert bc['TypeOfInterface'] in ['Match']
                raise NotImplementedError

            elif isinstance(bc['TypeOfInterface'], dict):
                bc.update(bc['TypeOfInterface'])
                bc.pop('TypeOfInterface')
                if bc['Type'] in boundary_conditions.turbomachinery_interfaces:
                    bc.pop('Family')
                updated_boundary_conditions.append(bc)

            else:
                raise MolaException(
                    f"For BC on Family {bc['Family']}, the value of 'TypeOfInterface' must be of type str or dict."
                    )
        
        # set again boundary conditions because it have changed
        boundary_conditions.apply(self, updated_boundary_conditions)


    def convert_to_dict(self):
        params= dict()
        for a in list(self.__dict__):
            if not a.startswith('_') and a not in self._interface._fake_attributes:
                att = getattr(self,a)
                if not callable(att):
                    params[a] = att
        return params
    
    def read_tree(self, io_tool=None):
        if not hasattr(self, 'tree'):
            raise MolaUserError('The attribute tree must be given to Workflow to use the method read_tree')
        
        if isinstance(self.tree, str):
            from mola.cfd.preprocess.mesh.io import reader
            self.tree = reader.read(self, self.tree, io_tool=io_tool)

    def set_workflow_parameters_in_tree(self):
        if not self.tree: self.tree = cgns.Tree()

        params = self.convert_to_dict()
        self.tree.setParameters(self._workflow_parameters_container_,**params)
    
    def set_workflow_parameters_in_file(self, filename='setup.py'):

        import mola
        import pprint
        Lines = '#!/usr/bin/env python3\n'
        Lines+= f"'''\nMOLA {mola.__version__} setup.py file automatically generated in PREPROCESS\n"
        Lines+= f"Path to MOLA: {mola.__MOLA_PATH__}\n"
        Lines+= f"Commit SHA: {mola.__SHA__}\n'''\n\n"

        params = self.convert_to_dict()
        for key, value in params.items():
            Lines += f"{key}={pprint.pformat(value)}\n\n"

        with open(filename,'w') as f: f.write(Lines)

        try: os.remove(filename+'c')
        except: pass


    def print(self):
        print(self.__str__())
    
    def __str__(self):
        params= self.convert_to_dict()
        import pprint
        return pprint.pformat(params)

    @classmethod
    def get_status_marker(cls, run_directory, machine='localhost', user=None):
        """Return the persisted status marker for a run directory.

        This is the shared marker interpretation used by both single-case and
        sequence status checks. ``PREPROCESSING`` is included even though it
        is not a terminal status file.
        """
        marker_statuses = (
            (names.FILE_ERROR_PREPARING_WORKFLOW, names.FILE_ERROR_PREPARING_WORKFLOW),
            (names.FILE_JOB_FAILED, names.FILE_JOB_FAILED),
            (names.FILE_JOB_COMPLETED, names.FILE_JOB_COMPLETED),
            (names.FILE_NEWJOB_REQUIRED, names.FILE_NEWJOB_REQUIRED),
            (names.FILE_RELAUNCH_JOB, names.FILE_RELAUNCH_JOB),
            (names.FILE_PREPROCESSING, names.STATUS_PREPROCESS),
        )
        for marker, status in marker_statuses:
            marker_path = os.path.join(str(run_directory), marker)
            if SV.is_existing_path(marker_path, machine=machine, user=user, file_only=True):
                return status
        return None

    @classmethod
    def simulation_status_from_run_directory(
            cls, run_directory, machine='localhost', user=None,
            job_id=None, scheduler=None, slurm_status=None,
            max_lines_of_catched_error=1000):
        """Determine a case status, including pre-workflow phases.

        This method is intended to run on the machine that owns
        ``run_directory``. It can inspect ``workflow.cgns`` before
        ``main.cgns`` exists and only constructs a Workflow when the solver
        input is available. Remote callers should execute it through the
        sequence-level remote status API.
        """
        run_directory = os.path.expandvars(str(run_directory))
        if not SV.is_existing_path(
                run_directory, machine=machine, user=user, file_only=False):
            return names.STATUS_TO_PREPARE

        marker_status = cls.get_status_marker(run_directory, machine, user)
        if marker_status is not None:
            return marker_status

        workflow_path = os.path.join(run_directory, names.FILE_INPUT_WORKFLOW)
        if SV.is_existing_path(workflow_path, machine=machine, user=user, file_only=True):
            # The input is still waiting in the sequence, or is about to be
            # consumed by prepare_from_workflow_file(). The latter writes
            # FILE_PREPROCESSING while it is active.
            if slurm_status == names.STATUS_PENDING:
                return names.STATUS_PENDING
            if slurm_status == names.STATUS_RUNNING:
                return names.STATUS_RUNNING
            return names.STATUS_READY

        main_path = os.path.join(run_directory, names.FILE_INPUT_SOLVER)
        if not SV.is_existing_path(main_path, machine=machine, user=user, file_only=True):
            return names.STATUS_TO_LAUNCH

        # Import lazily so importing CI/CD tooling does not require MPI.
        from mola.workflow import read_workflow
        workflow = read_workflow(main_path)
        workflow.RunManagement['RunDirectory'] = run_directory
        workflow.RunManagement['Machine'] = machine
        if user is not None:
            workflow.RunManagement['User'] = user

        status = workflow.simulation_status(
            max_lines_of_catched_error=max_lines_of_catched_error,
            job_id=job_id,
            scheduler=scheduler,
            slurm_status=slurm_status,
        )
        return status[0] if isinstance(status, tuple) else status

    def simulation_status(
            self, max_lines_of_catched_error=1000, job_id=None, scheduler=None,
            slurm_status=None):
        run_dir = self.RunManagement['RunDirectory']
        machine = self.RunManagement['Machine']
        user = self.RunManagement.get('User')

        marker_status = self.get_status_marker(run_dir, machine, user)
        is_complete = marker_status == names.FILE_JOB_COMPLETED
        is_fail = marker_status in [
            names.FILE_JOB_FAILED,
            names.FILE_ERROR_PREPARING_WORKFLOW,
        ]
        last_run_error_msg = self.get_last_run_error_message(max_lines_of_catched_error)
        last_run_had_errors = bool(last_run_error_msg)

        crash_msg = self.get_crash_message(max_lines_of_catched_error)
        is_crash = bool(crash_msg)

        expect_new_job = marker_status == names.FILE_NEWJOB_REQUIRED
        expect_relaunch = marker_status == names.FILE_RELAUNCH_JOB
        failed_at_preprocess = marker_status == names.FILE_ERROR_PREPARING_WORKFLOW
        is_preprocessing = marker_status == names.STATUS_PREPROCESS

        had_issues = any([failed_at_preprocess, last_run_had_errors, is_crash])

        if had_issues:
            is_complete = False
            is_fail = True

        if is_complete:
            return names.FILE_JOB_COMPLETED, ''
        
        elif is_fail:
            error_msg = last_run_error_msg+crash_msg

            if failed_at_preprocess:
                return names.FILE_ERROR_PREPARING_WORKFLOW, error_msg

            else:
                return names.FILE_JOB_FAILED, error_msg

        elif expect_new_job:
            return names.FILE_NEWJOB_REQUIRED, ''

        elif expect_relaunch:
            return names.FILE_RELAUNCH_JOB, ''

        elif is_preprocessing:
            return names.STATUS_PREPROCESS, ''

        elif slurm_status in {names.STATUS_RUNNING, names.STATUS_PENDING}:
            return slurm_status, ''

        elif slurm_status == '':
            coprocess_log = os.path.join(run_dir, names.FILE_COLOG)
            if SV.is_existing_path(
                    coprocess_log, machine=machine, user=user, file_only=True):
                return names.FILE_JOB_FAILED, (
                    'The scheduler job is no longer present after '
                    f'{names.FILE_COLOG} was created.'
                )
            return names.STATUS_READY, ''

        elif self.is_running(job_id=job_id, scheduler=scheduler):
            return names.STATUS_RUNNING, ''

        elif self.is_pending(job_id=job_id, scheduler=scheduler):
            return names.STATUS_PENDING, ''

        else:
            return names.STATUS_READY, ''

    def get_simulation_status_remote(
            self, max_lines_of_catched_error=1000, job_id=None, scheduler=None,
            slurm_status=None):
        """
        Get simulation status by executing simulation_status() remotely via SSH.
        
        This method is more efficient than calling simulation_status() locally when
        the run directory is on a remote machine, as it avoids:
        - Copying large main.cgns files
        - Multiple SSH calls for individual file checks
        
        Instead, it executes the entire simulation_status() logic remotely and
        returns only the result strings.
        
        Args:
            max_lines_of_catched_error: Maximum lines of error message to retrieve
            
        Returns:
            tuple: (status_string, error_message_string)
            
        Raises:
            MolaException: If SSH command fails or remote execution errors
        """
        run_dir = self.RunManagement['RunDirectory']
        machine = self.RunManagement['Machine']
        user = self.RunManagement.get('User')
        job_id = job_id if job_id is not None else self.RunManagement.get('JobID')
        scheduler = scheduler or self.RunManagement.get('Scheduler')
        import base64
        import json
        
        # Execute the shared run-directory resolver remotely. This supports
        # both workflow.cgns and main.cgns and keeps status precedence in one
        # implementation.
        python_code = f'''import json
try:
    from mola.workflow.workflow import Workflow
    status = Workflow.simulation_status_from_run_directory(
        run_directory=json.loads({json.dumps(str(run_dir))!r}),
        machine="localhost",
        job_id=json.loads({json.dumps(job_id)!r}),
        scheduler=json.loads({json.dumps(scheduler)!r}),
        slurm_status=json.loads({json.dumps(slurm_status)!r}),
        max_lines_of_catched_error={max_lines_of_catched_error},
    )
    result = {{"status": status, "error": ""}}
    print(json.dumps(result))
except Exception as e:
    result = {{"status": "ERROR", "error": str(e)}}
    print(json.dumps(result))
'''
        
        from mola import server as SV
        
        try:
            # Encode Python code as base64 to avoid shell escaping issues
            encoded_code = base64.b64encode(python_code.encode()).decode()
            
            # Command to decode and execute
            # Redirect stderr to /dev/null to avoid DSI system messages polluting stdout
            wrapped_command = f'echo "{encoded_code}" | base64 -d | python3 2>/dev/null'
            
            output = SV.submit_command(
                command=wrapped_command,
                machine=machine,
                user=user,
                use_mola_env=True
            )
            
            # Parse JSON result (extract last non-empty line which should be our JSON)
            try:
                # Find the last line that looks like JSON
                lines = output.strip().split('\n')
                json_line = None
                for line in reversed(lines):
                    line = line.strip()
                    if line.startswith('{') and line.endswith('}'):
                        json_line = line
                        break
                
                if json_line is None:
                    raise MolaException(f"Failed to find JSON in remote simulation_status result: {output}")
                
                result = json.loads(json_line)
                return result['status'], result.get('error', '')
            except json.JSONDecodeError as e:
                raise MolaException(f"Failed to parse remote simulation_status result: {output}") from e
                
        except MolaException as e:
            raise MolaException(f"SSH execution failed for simulation_status on {machine}: {str(e)}") from e

    def is_running(self, job_id=None, scheduler=None, machine=None, user=None) -> bool:

        """
        Check if the CFD job is currently running.
        
        For SLURM scheduler: checks job status via squeue
        For local scheduler: checks if solver process is running
        
        Returns:
            bool: True if job is running, False otherwise
        """
        # First try SLURM context
        try:
            status = self.get_current_job_status(
                job_id=job_id,
                scheduler=scheduler,
                machine=machine,
                user=user,
            )
            if status in ["RUNNING", "R", "ACTIVE"]:
                return True
        except (MolaException, KeyError):
            # Not in SLURM context or no JobID, continue to local check
            pass
        
        # Local context: check if solver is running
        return self._is_local_solver_running()

    def is_pending(self, job_id=None, scheduler=None, machine=None, user=None) -> bool:
        """
        Check if the CFD job is pending in the scheduler queue.
        
        For SLURM scheduler: checks job status via squeue
        For local scheduler: always returns False (no queue)
        
        Returns:
            bool: True if job is pending, False otherwise
        """
        try:
            status = self.get_current_job_status(
                job_id=job_id,
                scheduler=scheduler,
                machine=machine,
                user=user,
            )
            return status in ["PENDING", "PD", "QUEUED"]
        except (MolaException, KeyError):
            # Not in SLURM context or no JobID
            return False

    def get_current_job_status(
            self, job_id=None, scheduler=None, machine=None, user=None) -> str:
        """
        Get the current SLURM job status using an explicit job ID when given.
        
        Returns:
            str: Job status ('RUNNING', 'PENDING', 'COMPLETED', 'FAILED', etc.)
            
        Raises:
            MolaException: If not running under SLURM scheduler or no JobID found
        """
        scheduler = scheduler or self.RunManagement.get("Scheduler", None)
        
        if scheduler != "SLURM":
            raise MolaException(
                f"Cannot get job status: scheduler is '{scheduler}', expected 'SLURM'. "
                "This method only works for SLURM jobs."
            )
        
        job_id = job_id if job_id is not None else self.RunManagement.get("JobID", None)
        
        if job_id is None or job_id == "":
            raise MolaException(
                "Cannot get job status: no scheduler job ID was supplied. "
                "The workflow may not have been submitted yet."
            )
        
        # Get machine context
        machine = machine or self.RunManagement.get("Machine", "localhost")
        
        # Use squeue to check job status
        import subprocess
        try:
            # Build squeue command
            if machine != "localhost":
                # Run squeue on remote machine via SSH
                user = user if user is not None else self.RunManagement.get("User")
                ssh_host = f"{user}@{machine}" if user else machine
                ssh_cmd = f"ssh -T {ssh_host} 'squeue -j {job_id} -o %T -h'"
                result = subprocess.run(
                    ssh_cmd, 
                    shell=True, 
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
            else:
                # Run squeue locally
                result = subprocess.run(
                    ["squeue", "-j", str(job_id), "-o", "%T", "-h"],
                    capture_output=True, 
                    text=True, 
                    timeout=10
                )
            
            if result.returncode != 0:
                # Job may not exist anymore or squeue failed
                raise MolaException(f"squeue failed: {result.stderr}")
            
            status = result.stdout.strip()
            
            # Map SLURM status codes to standard names
            status_map = {
                "PD": "PENDING",
                "R": "RUNNING",
                "CG": "RUNNING",  # Completing
                "CD": "COMPLETED",
                "F": "FAILED",
                "CF": "PENDING",  # Configuring
                "S": "SUSPENDED",
                "TO": "TIMEOUT",
            }
            
            return status_map.get(status, status)
            
        except subprocess.TimeoutExpired:
            raise MolaException("Timeout while checking job status with squeue")
        except Exception as e:
            raise MolaException(f"Error checking job status: {str(e)}")

    def _is_local_solver_running(self) -> bool:
        """
        Check if solver process is running locally (for non-SLURM context).
        
        Checks for running processes that are executing the compute file
        (names.FILE_COMPUTE) in the RunDirectory.
        
        Returns:
            bool: True if solver process is found running, False otherwise
        """
        run_dir = self.RunManagement.get("RunDirectory", ".")
        compute_file = names.FILE_COMPUTE
        
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline is None:
                    continue
                
                cmdline_str = ' '.join(cmdline)
                
                # Check if process is running the compute file
                if compute_file in cmdline_str and run_dir in cmdline_str:
                    return True
                        
            except (psutil.NoSuchProcess, psutil.AccessDenied, ValueError):
                continue
            
        return False


    def _is_local_solver_running_fallback(self) -> bool:
        """
        Fallback method to check if solver is running by checking file modification times.
        
        Returns:
            bool: True if solver output files have been recently modified, False otherwise
        """
        import time
        from pathlib import Path
        
        run_dir = Path(self.RunManagement.get("RunDirectory", "."))
        
        # Check for common solver output files
        output_files = [
            run_dir / "output.log",
            run_dir / "solver.log",
            run_dir / names.FILE_STDOUT,
            run_dir / names.FILE_STDERR,
        ]
        
        current_time = time.time()
        recent_threshold = 300  # 5 minutes
        
        for output_file in output_files:
            if output_file.exists():
                mtime = output_file.stat().st_mtime
                if (current_time - mtime) < recent_threshold:
                    return True
        
        return False



    def get_last_run_error_message(self, max_lines_of_catched_error=1000):
        run_dir = self.RunManagement['RunDirectory']
        errmsg = SV.read_last_run_error_file_in_log_directory(run_dir, max_lines=max_lines_of_catched_error)
        return errmsg

        
    def get_crash_message(self, max_lines_of_catched_error=1000):
        run_dir = self.RunManagement['RunDirectory']
        machine = self.RunManagement['Machine']

        try:
            crashmsg = SV.read_text_file_from_errors(os.path.join(run_dir, names.FILE_STDERR),
                machine=machine, max_lines=max_lines_of_catched_error)            
            return crashmsg
        except FileNotFoundError:
            pass

        
    def assert_completed_without_errors(self, max_lines_of_catched_error=1000):
        status, error_msg = self.simulation_status(max_lines_of_catched_error)

        if status != names.FILE_JOB_COMPLETED:
            raise MolaException(f"status was not completed, but:{status}\n"+error_msg)

    def get_bc_dispatcher(self):
        if not hasattr(self,'_bc_dispatcher'):
            boundary_conditions._instantiate_bc_dispatcher(self)
        return self._bc_dispatcher

    def adjust_compatibilities(self):
        if not hasattr(self,"_compatibilities_adjustor"):
            adjustor = compatibilities_adjustor._instantiate_compatibilities_adjustor(self.Solver)
            self._compatibilities_adjustor = adjustor
        self._compatibilities_adjustor.apply(self)

    def print_interface(self, keep=None, maxlevel : int = 1000):
        if keep is not None:
            if not isinstance(keep, (list, tuple)):
                keep = [keep]
        print(self._interface.__str__(keep_args=keep, maxlevel=maxlevel))

    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return False
        else:
            from mola.misc import allclose_dict

            def get_filtered_parameters(w):
                params = w.convert_to_dict()
                for component in params['RawMeshComponents']:
                    if not isinstance(component['Source'], (str, pathlib.PosixPath)):
                        component.pop('Source')
                return params
            
            return allclose_dict(get_filtered_parameters(self), get_filtered_parameters(other))
