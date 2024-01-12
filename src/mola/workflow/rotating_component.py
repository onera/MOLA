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
import copy
from mola import cgns
from mola.workflow.workflow import Workflow


class WorkflowRotatingComponent(Workflow):

    '''
    Base workflow, that inheritates from Workflow and add methods for every applications with 
    rotating components. If possible, it should not impose different default values that Workflow.
    Other applicative workflows with rotating components inherites from WorkflowRotatingComponent, 
    for instance WorkflowTurbomachinery and WorkflowPropeller.
    '''

    def __init__(self, **kwargs):
        
        super().__init__(**kwargs)

        self.name = 'RotatingComponent'

        # New TurboConfiguration ? 
        # Very important : axis of the machine
        # extract reference surface and flux computation
        # duplication
        # motion

        if self.tree is None:
            self.get_yaw_and_pitch_axes()
            # self.TurboConfiguration = dict()
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

    def get_yaw_and_pitch_axes(self):

        self.ShaftRotationSpeed = 100. # FIXME Need to be change by user
        # Axis of the engine, with the same direection that the flow
        self.ComponentAxis = np.array([1.,0,0]) # Strong assumption here
        
        # Normalize the axis 
        self.ComponentAxis /= np.sqrt(self.ComponentAxis.dot(self.ComponentAxis))

        non_colinear_axis = np.array([0,1,0])
        if np.all(self.ComponentAxis == non_colinear_axis):
            non_colinear_axis = np.array([0,0,1])

        YawAxis   = np.cross(self.ComponentAxis, non_colinear_axis)
        PitchAxis = np.cross(YawAxis, self.ComponentAxis)

        self.Flow.update(dict(
            PitchAxis=PitchAxis, 
            YawAxis=YawAxis
        ))

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

            
    # def set_motion(self):
    #     for row, rowParams in self.TurboConfiguration['Rows'].items():
    #         try: 
    #             omega = rowParams['RotationSpeed']
    #         except KeyError:
    #             # No RotationSpeed --> zones attached to this family are not moving
    #             continue

    #         try: 
    #             # Test if zones in that family are modelled with Body Force
    #             for zone in self.zones():
    #                 if zone.get(Type='FamilyName', Depth=1).value() == row:
    #                     if zone.get(Name='FlowSolution#DataSourceTerm', Depth=1):
    #                         # If this node is present, body force is used
    #                         # Then the frame of this row must be the absolute frame
    #                         assert False
    #         except AssertionError:
    #             # zones attached to this family are not moving
    #             continue

    #         if not row in self.Motion:
    #             self.Motion[row] = dict(RotationSpeed = omega * self.ComponentAxis)

    #     super().set_motion(self)

    def set_boundary_conditions(self):

        def extendListOfFamilies(FamilyNames):
            '''
            For each <NAME> in the list **FamilyNames**, add Name, name and NAME.
            '''
            ExtendedFamilyNames = copy.deepcopy(FamilyNames)
            for fam in FamilyNames:
                newNames = [fam.lower(), fam.upper(), fam.capitalize()]
                for name in newNames:
                    if name not in ExtendedFamilyNames:
                        ExtendedFamilyNames.append(name)
            return ExtendedFamilyNames

        def is_boundary_already_defined(FamilyBoundary):
            return any([bc['Family'] == FamilyBoundary for bc in self.BoundaryConditions])
        
        def is_boundary_to_skeep(FamilyBoundary):
            # TODO Is it possible to remove this condition ?
            return FamilyBoundary.startswith('F_OV_') or FamilyBoundary.endswith('Zones')  
        
        for shroud_family in extendListOfFamilies(['shroud', 'carter']):
            for famNode in self.tree.group(Type='Family', Name=f'*{shroud_family}*'):
                FamilyBoundary = famNode.name()
                if is_boundary_already_defined(FamilyBoundary) or is_boundary_to_skeep(FamilyBoundary):
                    continue
                
                self.BoundaryConditions.append(
                    dict(Family=FamilyBoundary, type='Wall')
                    )
        
        for blade_family in extendListOfFamilies(['blade', 'aube']):
            for famNode in self.tree.group(Type='Family', Name=f'*{blade_family}*'):
                FamilyBoundary = famNode.name()
                if is_boundary_already_defined(FamilyBoundary) or is_boundary_to_skeep(FamilyBoundary):
                    continue
                
                # Get one bc attached to this family
                one_bc_FamilyName = self.tree.get(Type='FamilyName', Value=FamilyBoundary)
                zone = one_bc_FamilyName.getParent(Type='Zone_t')
                row_family = zone.get(Type='FamilyName', Depth=1).value()

                self.BoundaryConditions.append(
                    dict(Family=FamilyBoundary, type='Wall', Motion=self.Motion[row_family])
                    )
        
        def hub_rotation_function(CoordinateX):
            omega = np.zeros(CoordinateX.shape, dtype=float)
            for (x1, x2) in [(0., 0.1)]: #self.TurboConfiguration['HubRotationSpeed']:  # FIXME
                omega[(x1<=CoordinateX) & (CoordinateX<=x2)] = self.ShaftRotationSpeed
            return np.asfortranarray(omega).ravel(order='K')
                
        for hub_family in extendListOfFamilies(['hub', 'moyeu']):
            for famNode in self.tree.group(Type='Family', Name=f'*{hub_family}*'):
                FamilyBoundary = famNode.name()
                if is_boundary_already_defined(FamilyBoundary) or is_boundary_to_skeep(FamilyBoundary):
                    continue

                self.BoundaryConditions.append(
                    dict(Family=FamilyBoundary, type='Wall', Motion=dict(RotationSpeed=hub_rotation_function))
                    )

        super().set_boundary_conditions()