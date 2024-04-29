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
import numpy as np
import copy
from treelab import cgns
import inspect
from typing import Union, get_type_hints
from mola.logging import (mola_logger,
                       MolaException,
                       MolaUserError,
                       MolaUserAttributeError,
                       redirect_streams_to_logger,
                       get_signature)
from mola.logging.formatters import BOLD, RED, CYAN, PINK, YELLOW, ENDC
from  mola.cfd.preprocess.mesh import (reader,
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
from mola import server as SV 
from mola.cfd.postprocess import remove_cfd_files
from mola.cfd.compute import compute
from . import WorkflowInterface

class Workflow(WorkflowInterface):

    def __init__(self, 
            tree=None,
            Solver : str = os.environ.get('MOLA_SOLVER'),
            RawMeshComponents : list = None,
            Fluid : dict = None,
            Flow : dict = None,
            Turbulence : dict = None,
            BoundaryConditions : list = None,
            SplittingAndDistribution : dict = None,
            Numerics : dict = None,
            BodyForceModeling : list = None,
            Motion : dict = None, 
            Initialization : dict = None,
            ExtractionsDefaults : list = None,
            Extractions : list = None,
            ConvergenceCriteria : list = None,
            RunManagement : dict = None,
            ApplicationContext : dict = None,
            ):

        
        super().__init__(**WorkflowInterface.repack_kwargs())
    
    def prepare(self):
        self.assemble()
        self.positioning()
        self.connect()
        self.define_families()
        self.split_and_distribute()
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
        meshes = []
        for component in self.RawMeshComponents:
            base = reader.apply(component)
            meshes += [base]
        
        self.tree = cgns.merge(meshes)

        dimOfBases = set(base.dim() for base in self.tree.bases())
        if len(dimOfBases) != 1:
            raise MolaUserError('All bases must have the same physical dimension')
        self.ProblemDimension = int(list(dimOfBases)[0])

    def split_and_distribute(self):
        split.apply(self)

    def process_overset(self):
        pass

    def compute_flow_and_turbulence(self):
        FlowGen = flow_generators.AvailableFlowGenerators[self.Flow['Generator']]
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
        mola_logger.info(f"Submit job on machine {self.RunManagement['Machine']}")
        if command is None:
            command = self.RunManagement['LauncherCommand']
        user = self.RunManagement.get('User')
        SV.submit_command(command, self.RunManagement['Machine'], user=user)

    def write_tree_remote(self, data_directory=None):
        from . import workflow_manager as WM
        sender = WM.WorkflowSender(self, data_directory=data_directory)
        sender.apply()

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
            if bc['type'] != 'InterfaceBetweenWorkflows':
                continue

            if not 'interface_type' in bc:
                raise MolaException(
                    f"The boundary condition on Family {bc['Family']} is of type {bc['type']},"
                    "and for this type the key 'interface_type' must be defined."
                    )
            
            elif isinstance(bc['interface_type'], str):
                assert bc['interface_type'] in ['Match']
                raise NotImplementedError

            elif isinstance(bc['interface_type'], dict):
                bc.update(bc['interface_type'])
                bc.pop('interface_type')
                if bc['type'] in boundary_conditions.turbomachinery_interfaces:
                    bc.pop('Family')
                updated_boundary_conditions.append(bc)

            else:
                raise MolaException(
                    f"For BC on Family {bc['Family']}, the value of 'interface_type' must be of type str or dict."
                    )
        
        # set again boundary conditions because it have changed
        boundary_conditions.apply(self, updated_boundary_conditions)
