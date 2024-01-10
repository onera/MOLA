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

import numpy as np

from mola.workflow.workflow import Workflow
import mola.cfd.preprocess as PRE
import mola.application.internal_flow as IntFlow
import mola.application.turbomachine as Turb


class WorkflowPropeller(Workflow):

    def __init__(self, RPM=0., AxialVelocity=0., ReferenceTurbulenceSetAtRelativeSpan=0.75, **UserParameters):
        super(WorkflowPropeller, self).__init__(**UserParameters)

        self.name = 'Propeller'
        self.Splitter = 'PyPart'
        self.TurboConfiguration = dict()
        self.BodyForceInputData = None

        self.BCExtractions = UserParameters.get('BCExtractions', 
            dict(
                BCWall    = ['normalvector', 'frictionvector','psta', 'bl_quantities_2d', 'yplusmeshsize'],
                BCInflow  = ['convflux_ro'],
                BCOutflow = ['convflux_ro']
            )
        )

        self.ReferenceValues.update(
            dict(
            FieldsAdditionalExtractions=['q_criterion'],
            CoprocessOptions=dict(
                RequestedStatistics=['std-Thrust','std-Power'],
                ConvergenceCriteria=[dict(Family='BLADE',
                                        Variable='std-Thrust',
                                        Threshold=1e-3)],
                AveragingIterations = 1000,
                ItersMinEvenIfConverged = 1000,
                UpdateArraysFrequency = 100,
                UpdateSurfacesFrequency = 500,
                UpdateFieldsFrequency = 2000)
                )
        )


    def prepare(self):
        
        PRE.mesh.read_mesh(mesher='Autogrid')

        self.prepare_mesh(**kwargs)

        self.prepare_configuration()

        PRE.compute_fluid_properties()
        IntFlow.compute_reference_values()
        
        PRE.get_solver_parameters(self)  # Must return dict(cfdpb=dict(), models=dict(), numerics=())

        self.initializeFlowSolution(self)

        self.addTiggerReferenceStateGoverningEquations()
        PRE.addExtractions(self)
        self.save_main()

        nb_blades, Dir = self.getPropellerKinematic()
        span = self.maximumSpan()
        omega = -Dir * RPM * np.pi / 30.
        TangentialVelocity = abs(omega)*span*ReferenceTurbulenceSetAtRelativeSpan
        VelocityUsedForScalingAndTurbulence = np.sqrt(TangentialVelocity**2 + AxialVelocity**2)
        self.ReferenceValues['Velocity'] = UserParameters['AxialVelocity']
        self.ReferenceValues['VelocityUsedForScalingAndTurbulence'] = VelocityUsedForScalingAndTurbulence
        
        RowTurboConfDict = {}
        for b in I.getBases(t):
            RowTurboConfDict[b[0]+'Zones'] = {'RotationSpeed':omega,
                                            'NumberOfBlades':nb_blades,
                                            'NumberOfBladesInInitialMesh':nb_blades}
        SpinnerRotationInterval=(-1e6,+1e6)
        TurboConfiguration = WC.getTurboConfiguration(t, ShaftRotationSpeed=omega,
                                    HubRotationSpeed=[SpinnerRotationInterval],
                                    Rows=RowTurboConfDict)
        FluidProperties = PRE.computeFluidProperties()
        if not 'Surface' in ReferenceValuesParams:
            ReferenceValuesParams['Surface'] = 1.0

        MainDirection = np.array([1,0,0]) # Strong assumption here
        YawAxis = np.array([0,0,1])
        PitchAxis = np.cross(YawAxis, MainDirection)
        self.ReferenceValues.update(dict(PitchAxis=PitchAxis, YawAxis=YawAxis))

        self.ReferenceValues = PRE.computeReferenceValues(FluidProperties, **self.ReferenceValues)
        self.ReferenceValues['RPM'] = RPM
        self.ReferenceValues['NumberOfBlades'] = nb_blades
        self.ReferenceValues['AxialVelocity'] = AxialVelocity
        self.ReferenceValues['MaximumSpan'] = span


 
        WC.setMotionForRowsFamilies(t, TurboConfiguration)
        WC.setBC_Walls(t, TurboConfiguration)

        WC.computeFluxCoefByRow(t, ReferenceValues, TurboConfiguration)

        allowed_override_objects = ['cfdpb','numerics','model']
        for v in OverrideSolverKeys:
            if v == 'cfdpb':
                elsAkeysCFD.update(OverrideSolverKeys[v])
            elif v == 'numerics':
                elsAkeysNumerics.update(OverrideSolverKeys[v])
            elif v == 'model':
                elsAkeysModel.update(OverrideSolverKeys[v])
            else:
                raise AttributeError('OverrideSolverKeys "%s" must be one of %s'%(v,
                                                    str(allowed_override_objects)))

        AllSetupDicts = dict(Workflow='Propeller',
                            Splitter=Splitter,
                            JobInformation=JobInformation,
                            TurboConfiguration=TurboConfiguration,
                            FluidProperties=FluidProperties,
                            ReferenceValues=ReferenceValues,
                            elsAkeysCFD=elsAkeysCFD,
                            elsAkeysModel=elsAkeysModel,
                            elsAkeysNumerics=elsAkeysNumerics,
                            Extractions=Extractions)

        PRE.addTrigger(t)
        PRE.addExtractions(t, AllSetupDicts['ReferenceValues'],
                            AllSetupDicts['elsAkeysModel'], extractCoords=False,
                            BCExtractions=ReferenceValues['BCExtractions'])

        if elsAkeysNumerics['time_algo'] != 'steady':
            PRE.addAverageFieldExtractions(t, AllSetupDicts['ReferenceValues'],
                AllSetupDicts['ReferenceValues']['CoprocessOptions']['FirstIterationForAverage'])

        PRE.addReferenceState(t, AllSetupDicts['FluidProperties'],
                            AllSetupDicts['ReferenceValues'])
        dim = int(AllSetupDicts['elsAkeysCFD']['config'][0])
        PRE.addGoverningEquations(t, dim=dim)
        AllSetupDicts['ReferenceValues']['NumberOfProcessors'] = int(max(PRE.getProc(t))+1)
        PRE.writeSetup(AllSetupDicts)

        if FULL_CGNS_MODE:
            PRE.addElsAKeys2CGNS(t, [AllSetupDicts['elsAkeysCFD'],
                                    AllSetupDicts['elsAkeysModel'],
                                    AllSetupDicts['elsAkeysNumerics']])

        PRE.saveMainCGNSwithLinkToOutputFields(t,writeOutputFields=writeOutputFields)


    def prepare_mesh(self, splitOptions={'maximum_allowed_nodes':3},
                     match_tolerance=1e-7, periodic_match_tolerance=1e-7):
        blade_number, _ = self.getPropellerKinematic(self.tree)
        InputMeshes = [dict(file='mesh.cgns',
                            baseName='Base',
                            SplitBlocks=True,
                            BoundaryConditions=[
                                dict(name='blade_wall',
                                    type='FamilySpecified:BLADE',
                                    familySpecifiedType='BCWall'),
                                dict(name='spinner_wall',
                                    type='FamilySpecified:SPINNER',
                                    familySpecifiedType='BCWall'),
                                dict(name='farfield',
                                    type='FamilySpecified:FARFIELD',
                                    familySpecifiedType='BCFarfield',
                                    location='special',
                                    specialLocation='fillEmpty')],
                            Connection=[
                                dict(type='Match',
                                    tolerance=match_tolerance),
                                dict(type='PeriodicMatch',
                                    tolerance=periodic_match_tolerance,
                                    rotationCenter=[0.,0.,0.],
                                    rotationAngle=[360./float(blade_number),0.,0.])])]

        return super().prepareMesh4ElsA(InputMeshes, splitOptions=splitOptions)
    
    def getPropellerKinematic(self):
        mesh_params = I.getNodeFromName(self.tree,'.MeshingParameters')
        if mesh_params is None:
            raise ValueError(J.FAIL+'node .MeshingParameters not found in tree'+J.ENDC)

        try:
            nb_blades = int(I.getValue(I.getNodeFromName(mesh_params,'blade_number')))
        except:
            ERRMSG = 'could not find .MeshingParameters/blade_number in tree'
            raise ValueError(J.FAIL+ERRMSG+J.ENDC)

        try:
            Dir = int(I.getValue(I.getNodeFromName(mesh_params,'RightHandRuleRotation')))
            Dir = +1 if Dir else -1
        except:
            ERRMSG = 'could not find .MeshingParameters/RightHandRuleRotation in tree'
            raise ValueError(J.FAIL+ERRMSG+J.ENDC)

        return nb_blades, Dir
    
    def getMaximumSpan(self):
        zones = C.extractBCOfName(self.tree,'FamilySpecified:BLADE')
        W.addDistanceRespectToLine(zones, [0,0,0],[-1,0,0], FieldNameToAdd='span')
        return C.getMaxValue(zones, 'span')
