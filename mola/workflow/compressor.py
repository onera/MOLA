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

from mola.workflow.workflow import Workflow
import mola.cfd.preprocess as PRE
import mola.application.internal_flow as IntFlow
import mola.application.turbomachine as Turb


class WorkflowCompressor(Workflow):

    def __init__(self, **UserParameters):
        super(WorkflowCompressor, self).__init__(**UserParameters)

        self.name = 'Compressor'
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

    def prepare(self):
        
        self.prepare_mesh()

        self.prepare_configuration()

        PRE.compute_fluid_properties()
        IntFlow.compute_reference_values()
        
        PRE.get_solver_parameters(self)  # Must return dict(cfdpb=dict(), models=dict(), numerics=())

        self.initializeFlowSolution(self)

        self.tree = Turb.duplicateFlowSolution(self.tree, self.TurboConfiguration)

        Turb.setMotionForRowsFamilies(self.tree, self.TurboConfiguration)
        self.setBoundaryConditions(t, BoundaryConditions, TurboConfiguration,
                                FluidProperties, ReferenceValues,
                                bladeFamilyNames=bladeFamilyNames)

        computeFluxCoefByRow(t, ReferenceValues, TurboConfiguration)

        addMonitoredRowsInExtractions(Extractions, TurboConfiguration)

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

        AllSetupDicts = dict(Workflow='Compressor',
                            Splitter=Splitter,
                            JobInformation=JobInformation,
                            TurboConfiguration=TurboConfiguration,
                            FluidProperties=FluidProperties,
                            ReferenceValues=ReferenceValues,
                            elsAkeysCFD=elsAkeysCFD,
                            elsAkeysModel=elsAkeysModel,
                            elsAkeysNumerics=elsAkeysNumerics,
                            Extractions=Extractions, 
                            PostprocessOptions=PostprocessOptions)
        if BodyForceInputData: 
            AllSetupDicts['BodyForceInputData'] = BodyForceInputData

        self.addTiggerReferenceStateGoverningEquations()
        PRE.addExtractions(self)
        self.save_main()
    
    def prepare_mesh(self):
        self.tree = PRE.mesh.read_mesh(mesher='Autogrid')

        self.transform()
        self.connect_mesh()
        self.split_and_distribute(**splitOptions)
        self.add_overset_data(**globalOversetOptions)

        self.adapt2elsA() # to put at the end, with solver specific methods ? 

        J.checkEmptyBC(self.tree)

    def prepare_configuration(self):

        self.TurboConfiguration = Turb.getTurboConfiguration(self.tree, BodyForceInputData=self.BodyForceInputData, **self.TurboConfiguration)

        
        if not 'Surface' in self.ReferenceValues:
            self.ReferenceValues['Surface'] = Turb.getReferenceSurface(self.tree, self.BoundaryConditions, self.TurboConfiguration)

        # if 'PeriodicTranslation' in self.TurboConfiguration:
        #     MainDirection = np.array([1,0,0]) # Strong assumption here
        #     YawAxis = np.array(self.TurboConfiguration['PeriodicTranslation'])
        #     YawAxis /= np.sqrt(np.sum(YawAxis**2))
        #     PitchAxis = np.cross(YawAxis, MainDirection)
        #     self.ReferenceValues.update(dict(PitchAxis=PitchAxis, YawAxis=YawAxis))

    def initializeFlow(self):
        if self.Initialization['method'] == 'turbo':
            self.tree = Turb.initializeFlowSolutionWithTurbo(self.tree, self.FluidProperties, self.ReferenceValues, self.TurboConfiguration)
        else:
            self.tree = PRE.initializeFlowSolution(self.tree, self.Initialization, self.ReferenceValues)
