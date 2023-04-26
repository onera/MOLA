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

from mola.workflow.workflow import Workflow
import .internal_flow.FlowGenerator as InternalFlowGenerator
import mola.application.turbomachine as Turb


class WorkflowCompressor(Workflow):

    def __init__(self, 
                
                 Splitter='PyPart',

                 **kwargs
                 ):
        
        super().__init__(Splitter=Splitter, FlowGenerator=InternalFlowGenerator, **kwargs)

        self.name = 'Compressor'

        if self.tree is not None:
            for meshInfo in self.RawMeshComponents:
                meshInfo.setdefault('mesher', 'Autogrid')

            self.TurboConfiguration = dict()

            self.Extractions.append(
                dict(type='bc', BCType='BCInflow*', fields=['convflux_ro']),
                dict(type='bc', BCType='BCOutflow*', fields=['convflux_ro']),
            )
        

    def prepare(self):
        
        self.prepare_mesh()

        self.prepare_configuration()

        self.initializeFlowSolution(self)

        self.tree = Turb.duplicate_flow_solution(self.tree, self.TurboConfiguration)

        Turb.setMotionForRowsFamilies(self.tree, self.TurboConfiguration)
        self.setBoundaryConditions(t, BoundaryConditions, TurboConfiguration,
                                FluidProperties, ReferenceValues,
                                bladeFamilyNames=bladeFamilyNames)

        computeFluxCoefByRow(t, ReferenceValues, TurboConfiguration)

        addMonitoredRowsInExtractions(Extractions, TurboConfiguration)

        if BodyForceInputData: 
            AllSetupDicts['BodyForceInputData'] = BodyForceInputData

    
    def prepare_mesh(self):
        self.tree = PRE.mesh.read_mesh(mesher='Autogrid')
        self.InputMeshes = generate_input_meshes_from_autogrid(t,
            scale=scale, rotation=rotation, tol=tol, PeriodicTranslation=PeriodicTranslation)

        t = clean_mesh_from_autogrid(t, basename=InputMeshes[0]['baseName'], zonesToRename=zonesToRename)


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

    def initialize_flow(self):
        if self.Initialization['method'] == 'turbo':
            self.tree = Turb.initialize_flow_with_turbo(self)
        else:
            self.initialize_flow()

    def set_turbo_configuration(self):
        for row, rowParams in self.TurboConfiguration['Rows'].items():
            for key, value in rowParams.items():
                if key == 'RotationSpeed' and value == 'auto':
                    rowParams[key] = self.TurboConfiguration['ShaftRotationSpeed']
            if hasattr(self, 'BodyForceInputData') and row in self.BodyForceInputData:
                # Replace the number of blades to be consistant with the body-force mesh
                deltaTheta = computeAzimuthalExtensionFromFamily(self.tree, row)
                rowParams['NumberOfBlades'] = int(2*np.pi / deltaTheta)
                rowParams['NumberOfBladesInInitialMesh'] = 1
                print(f'Number of blades for {row}: {rowParams["NumberOfBlades"]} (got from the body-force mesh)')
            if not 'NumberOfBladesSimulated' in rowParams:
                rowParams['NumberOfBladesSimulated'] = 1
            if not 'NumberOfBladesInInitialMesh' in rowParams:
                rowParams['NumberOfBladesInInitialMesh'] = getNumberOfBladesInMeshFromFamily(self.tree, row, rowParams['NumberOfBlades'])


    def set_solver_motion_elsa(self):
        # Add info on row movement (.Solver#Motion)
        for row, rowParams in self.TurboConfiguration['Rows'].items():
            famNode = I.getNodeFromNameAndType(self.tree, row, 'Family_t')
            try: 
                omega = rowParams['RotationSpeed']
            except KeyError:
                # No RotationSpeed --> zones attached to this family are not moving
                continue

            # Test if zones in that family are modelled with Body Force
            for zone in C.getFamilyZones(self.tree, row):
                if I.getNodeFromName1(zone, 'FlowSolution#DataSourceTerm'):
                    # If this node is present, body force is used
                    # Then the frame of this row must be the absolute frame
                    omega = 0.
                    break
            
            print(f'setting .Solver#Motion at family {row} (omega={omega}rad/s)')
            J.set(famNode, '.Solver#Motion',
                    motion='mobile',
                    omega=omega,
                    axis_pnt_x=0., axis_pnt_y=0., axis_pnt_z=0.,
                    axis_vct_x=1., axis_vct_y=0., axis_vct_z=0.)