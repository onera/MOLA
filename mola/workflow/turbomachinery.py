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


class WorkflowTurbomachinery(Workflow):

    def __init__(self, 
                
                 Splitter='PyPart',

                 **kwargs
                 ):
        
        super().__init__(Splitter=Splitter, FlowGenerator=InternalFlowGenerator, **kwargs)

        self.name = 'Turbomachinery'

        if self.tree is not None:
            for meshInfo in self.RawMeshComponents:
                meshInfo.setdefault('mesher', 'Autogrid')

            self.TurboConfiguration = dict()
            # self.TurboConfiguration = dict(
            #     Component1 = dict(
            #         ShaftRotationSpeed = 500.,
            #         HubRotationSpeed = [(xmin, xmax)],
            #         Rows = dict(
            #             rotor = dict(
            #                 NumberOfBlades = 16,
            #             )
            #         )
            #     ),
            # )

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
            

    def reader_autogrid(base, 
                        unit='m',
                        Tolerance=1e-8, 
                        InitialFrame=dict(Point=[0,0,0], Axis1=[0,0,1], Axis2=[1,0,0], Axis3=[0,1,0])
                        ):
        
        # CAREFUL : Update Positioning, Connection, etc. rather than write over them

        ScaleDict = dict(
            mm = 0.001,
            cm = 0.01,
            dm = 0.1,
            m  = 1.
        )

        Positioning=[
            dict(
                Type='TranslationAndRotation',
                InitialFrame=InitialFrame,
                RequestedFrame=dict(
                    Point=[0,0,0],
                    Axis1=[1,0,0],
                    Axis2=[0,1,0],
                    Axis3=[0,0,1]),
                ),
            dict(
                Type  = 'scale',
                Scale = ScaleDict[unit],
                ),
        ]

        # Only if grid connectivities are not already in the mesh
        # TODO: Test on the presence of GC
        Connection = [
            dict(Type='Match', Tolerance=Tolerance),
        ]

        # Set automatic periodic connections
        angles = set()
        for node in base.group(Name='BladeNumber'):
            angles.add(360./float(node.value()))
        for angle in angles:
            print('  angle = {:g} deg ({} blades)'.format(angle, int(360./angle)))
            Connection.append(
                dict(type='PeriodicMatch', Tolerance=Tolerance, rotationAngle=[angle,0.,0.])
                )

        return component
    
    def clean_mesh_from_autogrid(t, basename='Base#1', zonesToRename={}):
        '''
        Clean a CGNS mesh from Autogrid 5.
        The sequence of operations performed are the following:

        #. remove useless nodes specific to AG5
        #. rename base
        #. rename zones
        #. clean Joins & Periodic Joins
        #. clean Rotor/Stator interfaces
        #. join HUB and SHROUD families

        Parameters
        ----------

            t : PyTree
                CGNS mesh from Autogrid 5

            basename: str
                Name of the base. Will replace the default AG5 name.

            zonesToRename : dict
                Each key corresponds to the name of a zone to modify, and the associated
                value is the new name to give.

        Returns
        -------

            t : PyTree
                modified mesh tree

        '''

        t.findAndRemoveNodes(Name='Numeca*')
        t.findAndRemoveNodes(Name='blockName')
        t.findAndRemoveNodes(Name='meridional_base')
        t.findAndRemoveNodes(Name='tools_base')

        # Clean Names
        # - Recover BladeNumber and Clean Families
        for fam in t.group(Type='Family'): 
            t.findAndRemoveNodes(Name='RotatingCoordinates')
            t.findAndRemoveNodes(Name='Periodicity')
            t.findAndRemoveNodes(Name='DynamicData')
        t.findAndRemoveNodes(Name='FamilyProperty')

        # - Rename base
        base = t.get(Type='CGNSBase')
        base.setName(basename)

        # - Rename Zones
        for zone in t.zones():
            name = zone.name()
            if name in zonesToRename:
                newName = zonesToRename[name]
                print("Zone {} is renamed: {}".format(name, newName))
                I._renameNode(t, name, newName)
                continue
            # Delete some usual patterns in AG5
            new_name = name
            for pattern in ['_flux_1', '_flux_2', '_flux_3', '_Main_Blade']:
                new_name = new_name.replace(pattern, '')
            I._renameNode(t, name, new_name)

        # Clean Joins & Periodic Joins
        # TODO: The objective should be to keep GC if there are already in the tree
        t.findAndRemoveNodes(Type='ZoneGridConnectivity_t')

        periodicFamilies = t.group(Name='*PER*', Type='Family', Depth=2)
        for familyNode in periodicFamilies:
            for BC in t.group(Type='BC'):
                for FamilyName in BC.group(Type='*FamilyName'): # FamilyName or AdditionalFamilyName
                    if FamilyName.name() == familyNode.name():
                        BC.remove()
                        break
            familyNode.remove()

        # Clean RS interfaces
        t.findAndRemoveNodes(Type='InterfaceType')
        t.findAndRemoveNodes(Type='DonorFamily')

        # Join HUB and SHROUD families
        J.joinFamilies(t, 'HUB')
        J.joinFamilies(t, 'SHROUD')
        return t
