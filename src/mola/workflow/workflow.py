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
import copy
import copy

from treelab import cgns

import mola.naming_conventions as names
from mola import server as SV 
from mola.logging import (mola_logger,
                       MolaException,
                       MolaUserError,
                       redirect_streams_to_logger)
from  mola.cfd.preprocess.mesh import (io,
                                    positioning,
                                    connect,
                                    split,
                                    families)
from  mola.cfd.preprocess import (flow_generators,
                               boundary_conditions,
                               initialization,
                               motion,
                               cfd_parameters,
                               extractions,
                               write_cfd_files)
from mola.cfd.postprocess import remove_cfd_files
from mola.cfd.compute import compute

from .workflow_interface import WorkflowInterface


class Workflow(object):

    def __init__(self, tree=None, **kwargs):

        self._workflow_parameters_container_ = names.CONTAINER_WORKLFOW_PARAMETERS
        self.Name = self.__class__.__name__
        self.tree = tree
        self._interface = WorkflowInterface(self, **kwargs)
        if tree is not None: self.get_workflow_parameters_from_tree()
        
    def prepare(self):
        self.assemble() # if mpi, distributed from here ?
        self.positioning()
        self.connect()
        self.define_families()
        self.split_and_distribute() # if mpi, partitioned from here ?
        self.process_overset()
        self.compute_flow_and_turbulence()
        self.set_motion()
        self.set_boundary_conditions()
        self.set_cfd_parameters()  # model, numerics, others...
        self.initialize_flow()  # eventually + distance to wall
        self.set_extractions()
        # self.check_preprocess() # empty BCs... maybe solver-specific
        self.set_workflow_parameters_in_tree()
        # self.set_workflow_parameters_in_file()

    def check_consistency_between_solver_and_environment(self):
        requested_solver = self.Solver
        env_solver = os.environ.get('MOLA_SOLVER')
        if requested_solver != env_solver:
            raise MolaException((f'the requested solver "{requested_solver}" does not'
                f'match the type of environment "{env_solver}"'))


    def assemble(self):
        self.read_meshes()
        self.set_workflow_parameters_in_tree()

    def positioning(self):
        positioning.apply(self)

    def connect(self):
        connect.apply(self)

    def define_families(self):
        families.apply(self)

    def read_meshes(self):
        io.read(self)

    def split_and_distribute(self):
        split.apply(self)

    def process_overset(self):
        pass

    def get_flow_generator(self, fg):
        if isinstance(fg, str):
            return flow_generators.AvailableFlowGenerators[fg]
        else:
            return fg


    def compute_flow_and_turbulence(self):
        # mola-generic set of parameters
        FlowGen = self.get_flow_generator(self.Flow['Generator'])(self)
        FlowGen.generate()
        self.Fluid = FlowGen.Fluid
        self.Flow = FlowGen.Flow
        self.Turbulence = FlowGen.Turbulence

    def initialize_flow(self):
        initialization.apply(self)
    
    def set_boundary_conditions(self):
        boundary_conditions.apply(self)

    def set_motion(self):
        motion.apply(self)

    def set_cfd_parameters(self):
        cfd_parameters.apply(self)

    def set_extractions(self):
        extractions.apply(self)

    def write_cfd_files(self):
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
    
    def submit(self, command=None):
        from mpi4py import MPI
        job_nb = None
        if MPI.COMM_WORLD.Get_rank() == 0:
            mola_logger.info(f"Submit job on machine {self.RunManagement['Machine']}")
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

    def write_tree_remote(self, data_directory=None):
        from . import workflow_manager as WM
        sender = WM.WorkflowSender(self, data_directory=data_directory)
        sender.apply()

    def write_tree(self, filename=names.FILE_INPUT_SOLVER):
        if not self.tree: 
            self.tree = cgns.Tree()
        with redirect_streams_to_logger(mola_logger):
            io.writer.write(self, self.tree, filename)

    def merge(self, other_workflow):
        # TODO Still in development, not validated
        mola_logger.warning(f'Merge workflows')

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


    def convert_to_dict(self, skip_attributes=['self','tree','workflow']):
        params= dict()
        for a in list(self.__dict__):
            if not a.startswith('_') and a not in skip_attributes:
                att = getattr(self,a)
                if not callable(att):
                    params[a] = att
        return params

    def get_workflow_parameters_from_tree(self, skip_attributes=['self','tree','workflow']):
        
        if isinstance(self.tree, str):
            workflow_parameters = cgns.load_workflow_parameters(self.tree)
        elif isinstance(self.tree, cgns.Tree):
            workflow_parameters = self.tree.getParameters(self._workflow_parameters_container_, transform_numpy_scalars=True)
        else:
            raise MolaUserError(f'The given tree must be either a filename or a Tree read by treelab.')
        
        for parameter in workflow_parameters:
            setattr(self, parameter, workflow_parameters[parameter])

        # for attributes appearing in constructor signature
        expected_types = self._interface.get_argument_types(WorkflowInterface.__init__)
        for attribute_name, expected_type in expected_types.items():
            if attribute_name in skip_attributes: continue
            if getattr(self, attribute_name) is None:
                setattr(self, attribute_name, expected_type())

        if self.SolverParameters is None: self.SolverParameters = dict()

        if isinstance(self.tree, str):
            from mola.cfd.preprocess.mesh.io import reader
            self.tree = reader.read(self, self.tree)
        

    def set_workflow_parameters_in_tree(self):
        if not self.tree: self.tree = cgns.Tree()

        params= self.convert_to_dict()
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

    def simulation_status(self, raise_error_if_not_completed=True,
            max_lines_of_catched_error=1000):
        run_dir = self.RunManagement['RunDirectory']
        machine = self.RunManagement['Machine']
        user = self.RunManagement.get('User')

        if SV.is_existing_path(os.path.join(run_dir, names.FILE_JOB_COMPLETED),
                machine=machine, user=user, file_only=True):
            return names.FILE_JOB_COMPLETED
        
        elif SV.is_existing_path(os.path.join(run_dir, names.FILE_JOB_FAILED),
                machine=machine, user=user, file_only=True):
            status = names.FILE_JOB_FAILED
        else:
            status = 'RUNNING, NOT STARTED OR CRASHED'

        errmsg = SV.read_text_file_from_errors(os.path.join(run_dir, names.FILE_STDERR),
            machine=machine, user=user, max_lines=max_lines_of_catched_error)
        if raise_error_if_not_completed:
            raise MolaException(errmsg)
        else:
            mola_logger.warning(errmsg)
            status += '\n'+errmsg

        return status

    def print_interface(self, maxlevel : int = 1000):
        print(self._interface.__str__(maxlevel=maxlevel))
