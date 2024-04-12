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
from treelab import cgns
from mola.logging import mola_logger, MolaException, redirect_streams_to_logger
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

def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict):
            d[k] = deep_update(d.get(k, {}), v)
        elif isinstance(v, list):
            d[k].extent(v)
        else:
            d[k] = v
    return d

class Workflow(object):

    def __init__(self, 
            tree=None,
            RawMeshComponents=None,
            Fluid=None,
            Flow=None,
            Turbulence=None,
            BoundaryConditions=None,
            Solver=os.environ.get('MOLA_SOLVER'),
            SplittingAndDistribution=None,
            Numerics=None,
            BodyForceModeling=None,
            Motion=None,
            Initialization=None,
            Extractions=None,
            ConvergenceCriteria=None,
            Monitoring=None,
            RunManagement=None,
            FlowGenerator='External_rho_V_T',
            ApplicationContext=None
            ):
            # Extractions=[
            #     dict(type='signals', name='Integrals', fields=['CL', 'std-CL'],
            #          Period=10),
            #     dict(type='probe', name='probe1', fields=['std-Pressure'], Period=5),
            #     dict(type='probe', name='probe2', fields=['std-Density'], Period=5),
            #     dict(type='3D', fields=['Mach', 'q_criterion'], Family='ROW1'),
            #     dict(type='bc', BCType='BCWall*', storage='ByFamily',
            #          fields=['normalvector', 'frictionvector']),
            #     dict(type='bc', BCType='*', storage='ByFamily',
            #          fields=['Pressure']),
            #     dict(type='IsoSurface',
            #         name='MySurface',
            #         postprocess=[dict(operation='AzimuthalAverage',
            #                           selectedZones=dict()),
            #                      dict(operation='MassFlowLoss',
            #                           selectedZones=dict(Component=),
            #                           SecondMassFlowRegion=,
            #                           flowComputation='from_compressor',
            #                           workflowReference=),
            #                      dict(operation='CnM²'
            #                           selectedZones=dict(Component=),
            #                           flowComputation='from_helicopter')],
            #         field='CoordinateY',
            #         value=1.e-6,
            #         AllowedFields=['Mach','cellN']),
            # ]
            # Monitoring=dict(SaveSignalsPeriod=30,
            #                 SaveExtractionsPeriod=30,
            #                 SaveFieldsPeriod=30,
            #                 SaveBodyForcePeriod=2000,
            #                 TagExtractionsWithIteration='auto'),

        self._workflow_parameters_container_ = 'WorkflowParameters'

        self.Name = self.__class__.__name__
        self.tree = tree

        if self.tree is not None:
            self.get_workflow_parameters_from_tree()

        else:
            # if isinstance(_defaults, str):
            #     # Read file with defaults values
            #     _defaults = ...
            # else:
            #     ERR_MSG = '_defaults must be either a dictionary or a string (path to a file)'
            #     assert isinstance(_defaults, dict), ERR_MSG
            # self._defaults = _workflow_defaults
            # deep_update(self._defaults, _defaults)
            # deep_update(self.__dict__, self._defaults)

            self.RawMeshComponents = RawMeshComponents if RawMeshComponents is not None else []
            self.ApplicationContext = ApplicationContext if ApplicationContext is not None else dict()
            self.Fluid = Fluid if Fluid is not None else dict()
            self.Flow = Flow if Flow is not None else dict()
            self.Turbulence = Turbulence if Turbulence is not None else dict()
            self.FlowGenerator = FlowGenerator
            self._FlowGenerator = self.get_flow_generator(self.FlowGenerator)
            self.BoundaryConditions = BoundaryConditions if BoundaryConditions is not None else []
            self.Solver = Solver.lower()
            self.SplittingAndDistribution = SplittingAndDistribution if SplittingAndDistribution is not None else dict()
            self.Numerics = Numerics if Numerics is not None else dict()
            self.BodyForceModeling = BodyForceModeling if BodyForceModeling is not None else []
            self.Motion = Motion if Motion is not None else dict()
            self.Initialization = Initialization if Initialization is not None else dict(method='uniform')
            self.Extractions = Extractions if Extractions is not None else []
            self.ConvergenceCriteria = ConvergenceCriteria if ConvergenceCriteria is not None else []
            self.Monitoring = Monitoring if Monitoring is not None else dict()
            self.RunManagement = RunManagement if RunManagement is not None else dict()

    def write_tree(self, filename='main.cgns'):
        if not self.tree: 
            self.tree = cgns.Tree()
        with redirect_streams_to_logger(mola_logger):
            self.tree.save(filename)

    def convert_to_dict(self):
        params= dict()
        for a in list(self.__dict__):
            if not a.startswith('_') and a != 'tree':
                att = getattr(self,a)
                if not callable(att):
                    params[a] = att
        return params

    def print(self):
        print(self.__str__())
    
    def __str__(self):
        params= self.convert_to_dict()
        import pprint
        return pprint.pformat(params)

    def get_workflow_parameters_from_tree(self):
        
        self.tree = cgns.load(self.tree)
        
        workflow_parameters = self.tree.getParameters(
            self._workflow_parameters_container_, transform_numpy_scalars=True)
        
        for parameter in workflow_parameters:
            setattr(self, parameter, workflow_parameters[parameter])

        self._FlowGenerator = self.get_flow_generator(self.FlowGenerator)
        if self.RawMeshComponents is None: self.RawMeshComponents = []
        if self.ApplicationContext is None: self.ApplicationContext = dict()
        if self.Fluid is None: self.Fluid = dict()
        if self.Flow is None: self.Flow = dict()
        if self.Turbulence is None: self.Turbulence = dict()
        if self.BoundaryConditions is None: self.BoundaryConditions = []
        if self.SplittingAndDistribution is None: self.SplittingAndDistribution = dict()
        if self.Numerics is None: self.Numerics = dict()
        if self.BodyForceModeling is None: self.BodyForceModeling = []
        if self.Motion is None: self.Motion = dict()
        if self.Initialization is None: self.Initialization = dict(method='uniform')
        if self.Extractions is None: self.Extractions = []
        if self.ConvergenceCriteria is None: self.ConvergenceCriteria = []
        if self.Monitoring is None: self.Monitoring = dict()
        if self.RunManagement is None: self.RunManagement = dict()


    def set_workflow_parameters_in_tree(self):
        if not self.tree: self.tree = cgns.Tree()

        params= self.convert_to_dict()
        self.tree.setParameters(self._workflow_parameters_container_,
                                **params)
    
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
        self.set_extractions()
        self.initialize_flow()  # eventually + distance to wall
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
        FlowGen = self._FlowGenerator(self)
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
        # merge trees
        self.tree.merge(other_workflow.tree)

        # update BCs or GCs at the interface
        for bc in self.BoundaryConditions + other_workflow.BoundaryConditions:
            if bc['type'] == 'WorkflowInterface':
                bc['type'] = bc.pop['final_type']
        
        # merge attributes
        self.RawMeshComponents += other_workflow.RawMeshComponents
        # Handle ApplicationContext ?
        self.BoundaryConditions += other_workflow.BoundaryConditions
        self.BodyForceModeling += other_workflow.BodyForceModeling
        self.Extractions += other_workflow.Extractions
        self.ConvergenceCriteria += other_workflow.ConvergenceCriteria
        self.Monitoring += other_workflow.Monitoring

        # set again boundary conditions because it may have changed
        self.set_boundary_conditions()

