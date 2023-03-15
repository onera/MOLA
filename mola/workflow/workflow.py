#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

import os
from ..cfd import preprocess as PRE
from ..application import external_flow as ExtFlow
from .. import cgns as c



class Workflow(object):

    def __init__(self, tree=None,

            RawMeshComponents=[],
            # RawMeshComponent with:
            # -> file or tree
            # -> component name
            # -> mesher type
            # -> family_bc definition
            # -> Overset Options
            # -> Connection
            # -> splitBlocks 

            # operations :
            # -> read file
            # -> clean tree
            # -> adapt Motion attribute
            # -> create families of zones
            # -> merge tree (creating bases if Overset)

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
                            ReferenceVelocity='auto',
                            Viscosity_EddyMolecularRatio=0.1,
                            TurbulenceCutOffRatio=1e-8,
                            TransitionMode=None),

            BoundaryConditions=[],
            
            Solver='elsA',

            Splitter=None,

            Numerics=dict(Scheme='Jameson',
                          TimeMarching='Steady',
                          NumberOfIterations=1e4,
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
            #          Period=10)
            #     dict(type='probe', name='probe1', fields=['std-Pressure'], Period=5)
            #     dict(type='probe', name='probe2', fields=['std-Density'], Period=5)
            #     dict(type='3D', fields=['Mach', 'q_criterion']),
            #     dict(type='bc', BCType='BCWall*', storage='ByFamily',
            #          fields=['normalvector', 'frictionvector'])
            #     dict(type='bc', BCType='*', storage='ByFamily',
            #          fields=['Pressure'])
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
            #         AllowedFields=['Mach','cellN'])]

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
                SubmitJob=False,
                TimeOutInSeconds = 'auto',
                Machine = 'auto', # or 'spiro-dtis', 'topaze'...
                LauncherCommand = 'auto', # or 'sbatch job.sh', './job.sh'...
                SecondsMargin4QuitBeforeTimeOut = 180.0),

            ):

        self._workflow_parameters_container_ = 'WorkflowParameters'

        self.Name = 'Standard'
        self.tree = tree

        if self.tree is not None:
            self.read_workflow_parameters_from_tree()

        else:
            self.RawMeshComponents=RawMeshComponents
            self.Fluid=Fluid
            self.Flow=Flow
            self.Turbulence=Turbulence
            self.BoundaryConditions=BoundaryConditions
            self.Solver=Solver
            self.Splitter=Splitter
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
        if not self.tree: self.tree = c.Tree()
        self.tree.save(filename)


    def read_workflow_parameters_from_tree(self):
        
        if isinstance(self.tree,str): self.tree = c.load(self.tree)
        
        workflow_parameters = self.tree.getParameters(self._workflow_parameters_container_)
        
        for parameter in workflow_parameters:
            setattr(self, parameter, workflow_parameters[parameter])


    def set_workflow_parameters_in_tree(self):
        if not self.tree: self.tree = c.Tree()
        self.tree.setParameters(self._workflow_parameters_container_,
                        Name=self.Name,
                        RawMeshComponents=self.RawMeshComponents,
                        Fluid=self.Fluid,
                        Flow=self.Flow,
                        Turbulence=self.Turbulence,
                        BoundaryConditions=self.BoundaryConditions,
                        Solver=self.Solver,
                        Splitter=self.Splitter,
                        Numerics=self.Numerics,
                        BodyForceModeling=self.BodyForceModeling,
                        Motion=self.Motion,
                        Initialization=self.Initialization,
                        ExtractionsDefaults=self.ExtractionsDefaults,
                        Extractions=self.Extractions,
                        ConvergenceCriteria=self.ConvergenceCriteria,
                        Monitoring=self.Monitoring,
                        RunManagement=self.RunManagement)

            
    def prepare(self):
        self.assemble()
        self.transform()
        self.connect()
        self.define_bc_families() # will include "addFamilies"
        self.split_and_distribute()
        self.process_overset()
        self.compute_reference_values()
        self.initialize_flow() # eventually + distance to wall
        self.set_solver_boundary_conditions()
        self.set_solver_motion()
        self.set_solver_keys() # model, numerics, others...
        self.set_solver_extractions()
        self.adapt_tree_to_solver()
        self.check_preprocess() # empty BCs... maybe solver-specific

    def write_cfd_files(self):
        return
    
    def assemble(self):
        return

    def compute_reference_values(self):
        # mola-generic set of parameters
        self.compute_flow_properties()
        self.set_modeling_parameters()
        self.set_numerical_parameters()

    def write_cfd_files(self):
        self.write_setup()
        self.write_run_scripts() # including job bash file(s)
        self.write_data_files() # CGNS, FSDM...


    def save_main(self):
        PRE.writeSetup(AllSetupDicts)
        PRE.mesh.saveMainCGNSwithLinkToOutputFields(writeOutputFields=True)

    def addTiggerReferenceStateGoverningEquations(self):
        PRE.addTrigger()
        PRE.addReferenceState(self.tree, self.FluidProperties, self.ReferenceValues) 
        dim = int(self.solver_parameters['cfdpb']['config'][0])
        PRE.addGoverningEquations(self.tree, dim=dim)

    def write_compute(self):
        pass

    def visu(self):
        pass



