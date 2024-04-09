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

from  mola.cfd.compute import compute

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

    def __init__(self, tree=None,

            RawMeshComponents=[],
            # RawMeshComponent with:
            # -> component name
            # -> file or tree
            # -> mesher type
            # -> family_bc definition
            # -> Overset Options
            # -> Connection
            # -> Positioning (previously Tranform)

            Fluid=dict(Gamma=1.4,
                       IdealGasConstant=287.053,
                       Prandtl=0.72,
                       PrandtlTurbulent=0.9,
                       SutherlandConstant=110.4,
                       SutherlandViscosity=1.78938e-05,
                       SutherlandTemperature=288.15),

            Flow=dict(),

            Turbulence=dict(Model='Wilcox2006-klim',
                            Level=0.001,
                            Viscosity_EddyMolecularRatio=0.1,
                            TurbulenceCutOffRatio=1e-8,
                            TransitionMode=None),

            BoundaryConditions=[],
            
            Solver=os.environ.get('MOLA_SOLVER'),

            SplittingAndDistribution=dict(
                Strategy='AtPreprocess', # "AtPreprocess" or "AtComputation"
                Splitter='Cassiopee', # or 'maia', 'PyPart' etc..
                Distributor='Cassiopee', 
                ComponentsToSplit='all', # 'all', or None or ['first', 'second'...]
                NumberOfProcessors='auto', 
                MinimumAllowedNodes=1,
                MaximumAllowedNodes=20,
                MaximumNumberOfPointsPerNode=1e9,
                CoresPerNode=48,
                DistributeExclusivelyOnFullNodes=True,
                ),

            Numerics=dict(Scheme='Jameson',
                          TimeMarching='Steady',
                          NumberOfIterations=10000,
                          MinimumNumberOfIterations = 1000,
                          TimeStep=None,
                          CFL=None),

            BodyForceModeling=dict(),

            Motion=dict(),

            Initialization=dict(method='uniform'),

            ExtractionsDefaults=dict(
                                    #  signals=dict(Surface=, Length=, Period=,
                                    #  TorqueOrigin=, AveragingIterations=
                                     ),

            Extractions=[],

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

            ConvergenceCriteria=[],

            Monitoring=dict(SaveSignalsPeriod=30,
                            SaveExtractionsPeriod=30,
                            SaveFieldsPeriod=30,
                            SaveBodyForcePeriod=2000,
                            TagExtractionsWithIteration='auto'),

            RunManagement=dict(
                JobName='MOLAjob',
                RunDirectory='.',
                NumberOfProcessors=None,
                AER='',
                FilesAndDirectories=[f"{os.getenv('MOLA')}/templates/compute.py"],
                TimeOutInSeconds = 'auto',
                Machine = 'auto', # or 'spiro-dtis', 'topaze'...
                LauncherCommand = 'auto', # or 'sbatch job.sh', './job.sh'...
                SecondsMargin4QuitBeforeTimeOut = 180.0),

            FlowGenerator='External_rho_V_T',

            ApplicationContext=dict(),

            # _defaults = dict(),

            ):

        self._workflow_parameters_container_ = 'WorkflowParameters'

        self.Name = self.__class__.__name__
        self.tree = tree

        if self.tree is not None:
            self.get_workflow_parameters_from_tree()
            self._FlowGenerator=self.get_flow_generator(self.FlowGenerator)

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

            self.RawMeshComponents=RawMeshComponents
            self.ApplicationContext = ApplicationContext
            self.Fluid=Fluid
            self.Flow=Flow
            self.FlowGenerator=FlowGenerator
            self._FlowGenerator=self.get_flow_generator(self.FlowGenerator)
            self.Turbulence=Turbulence
            self.BoundaryConditions=BoundaryConditions
            self.Solver=Solver.lower()
            self.SplittingAndDistribution=SplittingAndDistribution
            self.Numerics=Numerics
            self.BodyForceModeling=BodyForceModeling
            self.Motion=Motion
            self.Initialization=Initialization
            self.ExtractionsDefaults=ExtractionsDefaults
            self.Extractions=Extractions
            self.ConvergenceCriteria=ConvergenceCriteria
            self.Monitoring=Monitoring
            self.RunManagement=RunManagement

    def write_tree(self, filename='main.cgns'):
        if not self.tree: self.tree = cgns.Tree()
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
        if command is None:
            command = self.RunManagement['LauncherCommand']
        user = self.RunManagement.get('User')
        SV.submit_command(command, self.RunManagement['Machine'], user=user)

    def merge(self, other_workflow):
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

