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
import mola.cfd.preprocess as PRE
import mola.application.external_flow as ExtFlow


class Workflow(object):

    def __init__(self, tree=None,

            RawMeshComponents=[],

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
                            Viscosity_EddyMolecularRatio=0.1),

            BoundaryConditions=[],
            
            Solver='elsA',

            Splitter=None,

            Numerics=dict(Scheme='Jameson',
                          TimeMarching='Steady',
                          NumberOfIterations=1e4,
                          TimeStep=None,
                          CFL=None),

            BodyForceModeling=dict(),

            Motion=dict(),

            Initialization=dict(method='uniform'),

            Extractions=[],

            ConvergenceCriteria=[],

            CoprocessOptions=dict(SaveSignalsPeriod=30,
                                  SaveExtractionsPeriod=30,
                                  SaveFieldsPeriod=30),

            PostprocessOptions=dict(),
            
            RunManagement=dict(
                JobName='MOLAjob',
                RunDirectory='.',
                NumberOfProcessors=None,
                AER='',
                FilesAndDirectories=[f"{os.getenv('MOLA')}/templates/compute.py"],
                SubmitJob=False),

            ):

        
        self.name = 'Standard'

        if tree is not None:
            self.read_workflow_parameters(tree)

        else:
            self.tree = None
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
            self.Extractions=Extractions
            # Extractions=[
            #     dict(type='default', AveragingIterations=500,
            #                          SaveSignalsPeriod=30,
            #                          SaveExtractionsPeriod=30,
            #                          SaveFieldsPeriod=30)
            #     dict(type='signals', name='Integrals', fields=['CL', 'std-CL'], AveragingIterations=1000, Period=10)
            #     dict(type='probe', name='probe1', fields=['std-Pressure'], Period=5)
            #     dict(type='probe', name='probe2', fields=['std-Density'], Period=5)
            #     dict(type='3D', fields=['Mach', 'q_criterion']),
            #     dict(type='AllBCWall'),
            #     dict(type='IsoSurface',
            #         field='CoordinateY',
            #         value=1.e-6,
            #         AllowedFields=['Mach','cellN'])]

            self.ConvergenceCriteria=ConvergenceCriteria
            self.CoprocessOptions=CoprocessOptions
            self.PostprocessOptions=PostprocessOptions
            self.RunManagement=RunManagement
            
    def prepare(self):
        self.assemble()
        self.transform()
        self.connect()
        self.define_bc_families() # will include "addFamilies"
        self.split_and_distribute()
        self.process_overset()

        self.compute_reference_values()
        self.set_solver_parameters() # BC, motion, extractions, solver keys
        self.initialize_flow() # eventually + distance to wall
        self.write_cfd_files()
        self.check_preprocess() # empty BCs... maybe solver-specific
        self.submit_jobs()

    def assemble(self):
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
        pass

    def read_tree(self, t):
        if isinstance(t, str):
            self.filename = t
            self.tree = C.convertFile2PyTree(t)
        elif I.isTopTree(t):
            self.tree = t
        else:
            raise ValueError('parameter mesh must be either a filename or a PyTree')
        
        self.unstructured = PRE.mesh.hasAnyUnstructuredZones(self.tree) # To be replaced with Tree attr.



    def compute_reference_values(self):
        # mola-generic set of paramaterers
        self.compute_flow_properties()
        self.set_modeling_parameters()
        self.set_numerical_parameters()

    def set_solver_parameters(self):
        # CGNS, solver-specific obj, or to be parsed UserDefinedData_t
        self.set_solver_boundary_conditions()
        self.set_solver_motion()
        self.set_solver_keys() # model, numerics, others...
        self.set_solver_extractions()
        self.adapt_tree_to_solver()

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

def future_test_Workflow():

    workflow = Workflow()

    workflow.prepare()
    workflow.save_main()

