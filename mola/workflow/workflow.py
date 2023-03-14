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

import mola.cfd.preprocess as PRE
import mola.application.external_flow as ExtFlow


class Workflow(object):

    def __init__(self, **UserParameters):

        self.filename = None
        self.tree = None
        self.unstructured = False
        self.read_tree(UserParameters['mesh']) # set attributes filename and tree

        self.name = 'Standard'
        self.solver = 'elsA'
        self.Splitter = None
        self.JobInformation  = UserParameters['JobInformation']
        self.ReferenceValues = UserParameters['ReferenceValues']
        self.FluidProperties = UserParameters['FluidProperties']
        self.OversetMotion = dict()
        self.solver_parameters = dict()
        self.Extractions = []

        self.BCExtractions = dict()
        

    def read_tree(self, t):
        if isinstance(t, str):
            self.filename = t
            self.tree = C.convertFile2PyTree(t)
        elif I.isTopTree(t):
            self.tree = t
        else:
            raise ValueError('parameter mesh must be either a filename or a PyTree')
        
        self.unstructured = PRE.mesh.hasAnyUnstructuredZones(self.tree) # To be replaced with Tree attr.

    def prepare_chimera(self):
        self.tree = PRE.mesh.getMeshesAssembled(InputMeshes)
        self.prepare()

    def prepare(self, solver='elsA', **Parameters):

        for component in OversetComponents:


            self.tree = PRE.mesh.getMeshesAssembled(['toto.cgns'])

            self.tree, self.InputMeshes = PRE.mesh.fun(t, mesher=None)  # Must add families 
            self.transform()
            self.connect_mesh()

            self.adapt2elsA() # to put at the end, with solver specific methods ? 
        
        self.split_and_distribute(**splitOptions)
        self.add_overset_data(**globalOversetOptions)

        J.checkEmptyBC(self.tree)

        PRE.compute_fluid_properties()
        ExtFlow.compute_reference_values()

        # JobInformation['NumberOfProcessors'] = int(max(getProc(t))+1)
        
        PRE.get_solver_parameters(self)  # Must return dict(cfdpb=dict(), models=dict(), numerics=())

        self.addTiggerReferenceStateGoverningEquations()
        PRE.addExtractions(self)
        self.save_main()

        # if COPY_TEMPLATES:
        #     JM.getTemplates('Standard', JobInformation=self.JobInformation)
        #     if 'DIRECTORY_WORK' in JobInformation:
        #         sendSimulationFiles(JobInformation['DIRECTORY_WORK'],
        #                                 overrideFields=writeOutputFields)

        #     for i in range(SubmitJob):
        #         singleton = False if i==0 else True
        #         JM.submitJob(JobInformation['DIRECTORY_WORK'], singleton=singleton)

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

def test_Workflow():

    InputMeshes = [

    dict(file='INPUT_MESHES/raw_mesh.cgns',
        baseName='Base',
        Connection=[dict(type='Match',
                        tolerance=1e-8),],
        BoundaryConditions= [
            dict(name='WingBCWall',
                type='FamilySpecified:WING',
                familySpecifiedType='BCWall',
                location='kmin'
                ),
            dict(name='WingSymmetry',
                type='FamilySpecified:symmetry',
                familySpecifiedType='BCSymmetryPlane',
                location='special',
                specialLocation='planeXZ'),
            dict(name='Farfield',
                type='FamilySpecified:farfield',
                familySpecifiedType='BCFarfield',
                location='special',
                specialLocation='fillEmpty'),
                            ],
        SplitBlocks=True,
        ),
    ]

    workflow = Workflow(
        mesh = 'mesh.cgns',
        ReferenceValuesParams=dict(
            Density=1.225,
            Temperature=288.,
            Velocity=50.,
            AngleOfAttackDeg=4.0,
            Surface=0.15,
            Length=0.15,
            TorqueOrigin=[0, 0, 0],
            TurbulenceModel='SA',
            YawAxis=[0.,0.,1.],
            PitchAxis=[0.,1.,0.],
            FieldsAdditionalExtractions=['VorticityX',
                                         'q_criterion'],
            CoprocessOptions=dict(
                RequestedStatistics=['std-CD','std-Cm'],

                ConvergenceCriteria = [dict(Family='WING',
                                            Variable='std-CL',
                                            Threshold=1e-3)],
                AveragingIterations = 1000,
                ItersMinEvenIfConverged = 1000,

                UpdateArraysFrequency     = 100,
                UpdateSurfacesFrequency   = 500,
                UpdateFieldsFrequency     = 2000,
                ),),

        NumericalParams=dict(niter=10000),


        Extractions=[
            dict(type='BCWall'),
            dict(type='IsoSurface',
                 field='CoordinateX',
                 value=0.75),
            dict(type='IsoSurface',
                 field='CoordinateX',
                 value=0.5),
            dict(type='IsoSurface',
                 field='CoordinateX',
                 value=0.25),
            dict(type='IsoSurface',
                 field='VorticityX',
                 value=10.),],

        writeOutputFields=True,

        JobInformation=dict(
            JobName = 'LIGHT_WING',
            AER = '32877010F',
            ),
    )
    
    workflow.prepare()
    workflow.save_main()

