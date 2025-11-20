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

from treelab import cgns

import mola.naming_conventions as names
from mola import server as SV 
from mola import solver, __version__
from mola.logging import (mola_logger,
                       MolaException,
                       MolaUserError,
                       redirect_streams_to_logger, 
                       GREEN, CYAN, BOLD, ENDC)
from  mola.cfd.preprocess.mesh import (io,
                                    positioning,
                                    connect,
                                    split,
                                    families,
                                    duplicate,
                                    overset)
from  mola.cfd.preprocess import (flow_generators,
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

from mola.workflow.interface import WorkflowInterface
from mola.pytree.user import checker 

class Workflow(object):

    def __init__(self, **kwargs):
        self._interface = WorkflowInterface(self, **kwargs)
        
    def prepare(self):
        start_msg = f'START PREPROCESS WITH MOLA {__version__}'
        mola_logger.info(f'{CYAN}{BOLD}{"="*len(start_msg)}\n{start_msg}\n{"="*len(start_msg)}{ENDC}', rank=0)
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

    def process_mesh(self):
        mola_logger.info("🔧 processing mesh", rank=0)
        if self.Solver == 'sonics': # CAVEAT specifically verifying sonics, should reverse dependency properly
            from mola.cfd.preprocess.boundary_conditions.solver_sonics import adapt_workflow_for_sonics
            adapt_workflow_for_sonics(self)

        elif self.Solver == 'fast': # CAVEAT specifically verifying fast, should reverse dependency properly
            from mola.cfd.preprocess.initialization.solver_fast import adapt_workflow_for_fast
            adapt_workflow_for_fast(self)

        self.assemble() 
        self.positioning()
        self.connect()
        self.define_families()
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

    def write_cfd_files(self):
        mola_logger.info("💾 write cfd files", rank=0)
        write_cfd_files.apply(self) 
    
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
        from mpi4py import MPI
        job_nb = None
        if MPI.COMM_WORLD.Get_rank() == 0:
            mola_logger.info(f"🚀 Submit job on machine {self.RunManagement['Machine']}")
            if command is None:
                command = self.RunManagement['LauncherCommand']
            user = self.RunManagement.get('User')
            out = SV.submit_command(command, self.RunManagement['Machine'], user=user)
            for line in out.split('\n'):
                if line.startswith('Submit'):
                    print(line)
                    try: job_nb = int(line.split('')[-1])
                    except: pass
        MPI.COMM_WORLD.barrier()
        return job_nb
    
    def prepare_and_submit_remotely(self):
        self.prepare_job()
        assert self.RunManagement['Scheduler'] == 'SLURM'
        self.set_workflow_parameters_in_tree()
        self.write_tree_remote()
        write_cfd_files.write_info_for_data_retrieval(self.RunManagement)

        def write_job_prepare(RunManagement, scheduler_options, solver):
            from mola.cfd.preprocess.write_cfd_files.write_cfd_files import get_job_text

            if RunManagement['Scheduler'] == 'SLURM':
                scheduler_options['time'] = '00:30:00'
                # FIXME allow parallel preprocessing
                scheduler_options['ntasks'] = 1

            job_text = get_job_text(solver, RunManagement, scheduler_options)+'\n\n'
            job_text += f'mola_prepare {names.FILE_INPUT_WORKFLOW} 1>stdout-preprocess.log 2>stderr-preprocess.log\n'
            SV.save_file_maybe_remote(names.FILE_JOB_PREPARE, job_text, RunManagement['RunDirectory'], machine=RunManagement['Machine'], force_copy=True)

        write_job_prepare(self.RunManagement, self.RunManagement['SchedulerOptions'], self.Solver)

        run_directory = self.RunManagement['RunDirectory']
        machine = self.RunManagement['Machine']
        user = self.RunManagement.get('User')

        mola_logger.info(f"Preparing job on machine {self.RunManagement['Machine']}...")
        SV.submit_command(
            f"cd {run_directory}; sbatch --wait {names.FILE_JOB_PREPARE}", 
            machine=machine, 
            user=user
            )
        
        error_file_path = os.path.join(run_directory, names.FILE_ERROR_PREPARING_WORKFLOW)
        if SV.is_existing_path(error_file_path, machine=machine, user=user, file_only=True):
            try:
                SV.copy_remote(
                    source_path=error_file_path, source_machine=machine, source_user=user,
                    destination_path=names.FILE_ERROR_PREPARING_WORKFLOW, destination_machine='localhost'
                    )
            except:  
                pass
            raise MolaException(f'Error preparing workflow: see file {names.FILE_ERROR_PREPARING_WORKFLOW}')
        
        self.submit()

    def write_tree_remote(self, data_directory=None, copy_options=None):
        from . import manager as WM
        sender = WM.WorkflowSender(self, data_directory=data_directory, copy_options=copy_options)
        sender.apply()

    def write_tree(self, filename=names.FILE_INPUT_SOLVER):
        if not self.tree: 
            self.tree = cgns.Tree()
        with redirect_streams_to_logger(mola_logger):
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

    def simulation_status(self, max_lines_of_catched_error=1000):
        run_dir = self.RunManagement['RunDirectory']
        machine = self.RunManagement['Machine']
        user = self.RunManagement.get('User')


        if SV.is_existing_path(os.path.join(run_dir, names.FILE_JOB_COMPLETED),
                machine=machine, user=user, file_only=True):
            
            errmsg = self.get_last_run_error_message(max_lines_of_catched_error)
            if errmsg:
                raise MolaException(errmsg)

            status = names.FILE_JOB_COMPLETED
        
        elif SV.is_existing_path(os.path.join(run_dir, names.FILE_JOB_FAILED),
                machine=machine, user=user, file_only=True):
            status = names.FILE_JOB_FAILED

        else:
            status = 'RUNNING, NOT STARTED OR CRASHED'

        crashmsg = self.get_crash_message(max_lines_of_catched_error)            
        if crashmsg:
            raise MolaException(crashmsg)

        return status

    def get_last_run_error_message(self, max_lines_of_catched_error=1000):
        run_dir = self.RunManagement['RunDirectory']
        errmsg = SV.read_last_run_error_file_in_log_directory(run_dir, max_lines=max_lines_of_catched_error)
        return errmsg

        
    def get_crash_message(self, max_lines_of_catched_error=1000):
        run_dir = self.RunManagement['RunDirectory']
        machine = self.RunManagement['Machine']
        user = self.RunManagement.get('User')

        try:
            crashmsg = SV.read_text_file_from_errors(os.path.join(run_dir, names.FILE_STDERR),
                machine=machine, user=user, max_lines=max_lines_of_catched_error)            
            return crashmsg
        except FileNotFoundError:
            pass

        
    def assert_completed_without_errors(self, max_lines_of_catched_error=1000):
        crash_msg = self.get_crash_message(max_lines_of_catched_error)
        last_err_msg = self.get_last_run_error_message(max_lines_of_catched_error)
        
        has_crashed = bool(crash_msg)
        had_errors = bool(last_err_msg)
        status = self.simulation_status()
        was_completed = status == 'COMPLETED'
        maybe_running = status.startswith('RUNNING')
        
        msg_to_raise = None
        if has_crashed:
            msg_to_raise = 'CRASH MESSAGE:\n'+crash_msg
        
        if had_errors:
            msg_to_raise = 'HAD ERROR:\n'+last_err_msg

        if not has_crashed and not had_errors:
            
            if maybe_running:
                msg_to_raise = 'simulation was aborted before completion'
            
            elif was_completed:
                msg_to_raise = None
            
            else:
                msg_to_raise = 'FATAL ERROR: UNREACHABLE, BAD CONDITIONING'
    
        if msg_to_raise:
            raise MolaException(msg_to_raise)

    def get_bc_dispatcher(self):
        if not hasattr(self,'_bc_dispatcher'):
            boundary_conditions._instantiate_bc_dispatcher(self)
        return self._bc_dispatcher


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
