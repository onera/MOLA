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

import copy
import numpy as np

from . import Workflow
from .rotating_component_interface import WorkflowRotatingComponentInterface
from mola.logging import mola_logger, MolaException, MolaAssertionError
from mola.cfd.preprocess.boundary_conditions import permeable_boundaries, turbomachinery_interfaces 
from  mola.cfd.preprocess.mesh import duplicate


class WorkflowRotatingComponent(Workflow):

    '''
    Base workflow, that inheritates from Workflow and add methods for every applications with 
    rotating components. If possible, it should not impose different default values that Workflow.
    Other applicative workflows with rotating components inherites from WorkflowRotatingComponent, 
    for instance WorkflowTurbomachinery and WorkflowPropeller.

    Examples
    --------

    .. code:: python

        ApplicationContext = dict(
            ShaftAxis = [1, 0, 0],
            RPM = 5000., # rotation speed in rotation per minute
            # or you can also provide:
            # ShaftRotationSpeed = ..., # in rad/s
            HubRotationSpeed = [(xmin1, xmax1), (xmin2, xmax2)]  
            Rows = dict(
                Rotor = dict(
                    IsRotating = True,
                    NumberOfBlades = 36,
                )
            )
        )

    '''

    def __init__(self, tree=None, **kwargs):

        self.Name = self.__class__.__name__
        self.tree = tree
        self._interface = WorkflowRotatingComponentInterface(self, **kwargs)
        if tree is not None:
            self.get_workflow_parameters_from_tree()

    def define_families(self):
        super().define_families()
        self.set_default_parameters_for_rows()
        self.compute_fluxcoef_by_row() 
        # duplicate.duplicate_workflow_with_cassiopee(self)
        duplicate.duplicate_workflow_with_maia(self)

    def set_default_parameters_for_rows(self):

        for row, rowParams in self.ApplicationContext['Rows'].items():

            if not self.tree.get(Name=row, Type='Family', Depth=2):
                raise MolaException(f'The family {row} given in ApplicationContext is not found in the mesh.')

            if hasattr(self, 'BodyForceInputData') and row in self.BodyForceInputData:
                # Replace the number of blades to be consistant with the body-force mesh
                deltaTheta = self.compute_azimuthal_extension_from_family(self.tree, row, self.ApplicationContext['ShaftAxis'])
                rowParams['NumberOfBlades'] = int(2*np.pi / deltaTheta)
                rowParams['NumberOfBladesInInitialMesh'] = 1
                mola_logger.info(f'Number of blades for {row}: {rowParams["NumberOfBlades"]} (got from the body-force mesh)')

            rowParams.setdefault('NumberOfBladesInInitialMesh', self.get_number_of_blades_in_mesh_from_family(row, rowParams['NumberOfBlades']))     

    def set_motion(self):
        for row, rowParams in self.ApplicationContext['Rows'].items():

            IsModelledWithBodyForce = hasattr(self, 'BodyForceInputData') and row in self.BodyForceInputData

            if (
                not row in self.Motion 
                and rowParams['IsRotating']
                and not IsModelledWithBodyForce
                ):
                self.Motion[row] = dict(RotationSpeed = self.ApplicationContext['ShaftRotationSpeed'] * self.ApplicationContext['ShaftAxis'])

        super().set_motion()

    def set_boundary_conditions(self):

        self.set_shroud_boundary_conditions()
        self.set_blade_boundary_conditions()
        self.set_hub_boundary_conditions()

        super().set_boundary_conditions()  

    def set_shroud_boundary_conditions(self, families=['shroud', 'carter']):
        for shroud_family in self._extendListOfFamilies(families):
            for famNode in self.tree.group(Type='Family', Name=f'*{shroud_family}*'):
                FamilyBoundary = famNode.name()
                if self._is_boundary_already_defined(FamilyBoundary) or self._is_boundary_to_skip(FamilyBoundary):
                    continue
                
                self.BoundaryConditions.append(
                    dict(Family=FamilyBoundary, Type='Wall')
                    )
    
    def set_blade_boundary_conditions(self, families=['blade', 'aube']):
        for blade_family in self._extendListOfFamilies(families):
            for famNode in self.tree.group(Type='Family', Name=f'*{blade_family}*'):
                FamilyBoundary = famNode.name()
                if self._is_boundary_already_defined(FamilyBoundary) or self._is_boundary_to_skip(FamilyBoundary):
                    continue
                
                row_family = self._get_row_from_BC_Family(self.tree, FamilyBoundary)

                try:
                    self.BoundaryConditions.append(
                        dict(Family=FamilyBoundary, Type='Wall', Motion=self.Motion[row_family])
                        )
                except KeyError:
                    self.BoundaryConditions.append(dict(Family=FamilyBoundary, Type='Wall'))
    
    def set_hub_boundary_conditions(self, families=['hub', 'moyeu']):
        for hub_family in self._extendListOfFamilies(families):
            for famNode in self.tree.group(Type='Family', Name=f'*{hub_family}*'):
                FamilyBoundary = famNode.name()
                if self._is_boundary_already_defined(FamilyBoundary) or self._is_boundary_to_skip(FamilyBoundary):
                    continue

                if not 'HubRotationSpeed' in self.ApplicationContext:
                    # Assume that hub rotates at the same speed that the zone family
                    mola_logger.warning(f'Assume that motion is uniform on Family {FamilyBoundary}.')
                    row_family = self._get_row_from_BC_Family(self.tree, FamilyBoundary)
                    try:
                        self.BoundaryConditions.append(
                            dict(Family=FamilyBoundary, Type='Wall', Motion=self.Motion[row_family])
                            )
                    except KeyError:
                        self.BoundaryConditions.append(dict(Family=FamilyBoundary, Type='Wall'))
                else:
                    self.BoundaryConditions.append(
                        dict(Family=FamilyBoundary, Type='Wall', Motion=dict(RotationSpeed=self._get_hub_rotation_function()))
                        )

    @staticmethod
    def _extendListOfFamilies(FamilyNames):
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

    def _is_boundary_already_defined(self, FamilyBoundary):
        for bc in self.BoundaryConditions:
            for key in ['Family', 'LinkedFamily']:
                if key in bc and bc[key] == FamilyBoundary:
                    return True
        return False
    
    @staticmethod
    def _is_boundary_to_skip(FamilyBoundary):
        # TODO Is it possible to remove this condition ?
        return FamilyBoundary.startswith('F_OV_') or FamilyBoundary.endswith('Zones')  
    
    @staticmethod
    def _get_row_from_BC_Family(tree, FamilyBoundary):
        # Get one bc attached to this family
        one_bc_FamilyName = tree.get(Type='FamilyName', Value=FamilyBoundary)
        zone = one_bc_FamilyName.getParent(Type='Zone_t')
        row_family = zone.get(Type='FamilyName', Depth=1).value()
        return row_family
       
    def _get_hub_rotation_function(self):
        if isinstance(self.ApplicationContext['HubRotationSpeed'], (list, np.ndarray)):

            if list(self.ApplicationContext['ShaftAxis']) != [1., 0., 0.]:
                raise MolaAssertionError(f"Cannot handle hub rotation if the shaft axis is not the X axis.")

            def hub_rotation_function(CoordinateX):
                omega = np.zeros(CoordinateX.shape, dtype=float)
                for (x1, x2) in self.ApplicationContext['HubRotationSpeed']:  
                    omega[(x1<=CoordinateX) & (CoordinateX<=x2)] = self.ShaftRotationSpeed
                return np.asfortranarray(omega).ravel(order='K')

        else:
            assert callable(self.ApplicationContext['HubRotationSpeed'])
            hub_rotation_function = self.ApplicationContext['HubRotationSpeed']

        return hub_rotation_function     

    def get_number_of_blades_in_mesh_from_family(self, FamilyName, NumberOfBlades):
        '''
        Compute the number of blades for the row **FamilyName** in the mesh.

        Returns
        -------
        int
            Number of blades in the mesh for row **FamilyName**

        '''
        deltaTheta = self.compute_azimuthal_extension_from_family(self.tree, FamilyName, self.ApplicationContext['ShaftAxis'])
        # Compute number of blades in the mesh
        Nb = NumberOfBlades * deltaTheta / (2*np.pi)
        Nb = int(np.round(Nb))
        mola_logger.info(f'Number of blades in initial mesh for {FamilyName}: {Nb}')
        if Nb < 1:
            raise MolaAssertionError(
                f'The number of blades in initial mesh {FamilyName} cannot be computed correctly.'
                ' Please check the orientation and scale of the mesh. If the mesh is correct,'
                ' but the error is persistent, you may use the argument NumberOfBladesInInitialMesh'
                ' in ApplicationContext to fix manually fix this.'
                )
        return Nb

    @staticmethod
    def compute_azimuthal_extension_from_family(t, FamilyName, axis):
        '''
        Compute the azimuthal extension in radians of the mesh **t** for the row **FamilyName**.

        .. warning:: This function needs to calculate the surface of the slice in X
                    at Xmin + 5% (Xmax - Xmin). If this surface is crossed by a
                    solid (e.g. a blade) or by the inlet boundary, the function
                    will compute a wrong value of the number of blades inside the
                    mesh.

        Parameters
        ----------

            t : PyTree
                mesh tree

            FamilyName : str
                Name of the row, identified by a ``FamilyName``.
            
            axis : list
                Directing vector of the shaft axis.

        Returns
        -------

            deltaTheta : float
                Azimuthal extension in radians

        '''
        import Converter.PyTree as C
        import Post.PyTree as P

        if list(axis) != [1.0, 0.0, 0.0]:
            raise MolaAssertionError('For now, this function only handles axis=[1., 0., 0.]')

        # Extract zones in family
        zonesInFamily = C.getFamilyZones(t, FamilyName)
        # Slice in x direction at middle range
        xmin = C.getMinValue(zonesInFamily, 'CoordinateX')
        xmax = C.getMaxValue(zonesInFamily, 'CoordinateX')
        sliceX = P.isoSurfMC(zonesInFamily, 'CoordinateX', value=xmin+0.05*(xmax-xmin))
        # Compute Radius
        C._initVars(sliceX, '{Radius}=({CoordinateY}**2+{CoordinateZ}**2)**0.5')
        Rmin = C.getMinValue(sliceX, 'Radius')
        Rmax = C.getMaxValue(sliceX, 'Radius')
        # Compute surface
        SurfaceTree = C.convertArray2Tetra(sliceX)
        SurfaceTree = C.initVars(SurfaceTree, 'ones=1')
        Surface = P.integ(SurfaceTree, var='ones')[0]
        # Compute deltaTheta
        mola_logger.debug(f'Surface={Surface}, Rmax={Rmax}, Rmin={Rmin}')
        deltaTheta = 2* Surface / (Rmax**2 - Rmin**2)
        return deltaTheta

    def compute_fluxcoef_by_row(self):
        '''
        Compute the parameter **FluxCoef** for boundary conditions (except wall BC)
        and rotor/stator intefaces (``GridConnectivity_t`` nodes).
        **FluxCoef** will be used later to normalize the massflow.

        Modify **ReferenceValues** by adding:

        >>> ReferenceValues['NormalizationCoefficient'][<FamilyName>]['FluxCoef'] = FluxCoef

        for <FamilyName> in the list of BC families, except families of type 'BCWall*'.

        '''
        self.ApplicationContext.setdefault('NormalizationCoefficient', dict())

        for bc in self.BoundaryConditions:

            if bc['Type'] not in permeable_boundaries+turbomachinery_interfaces:
                continue
            
            Families = [value for key, value in bc.items() if key in ['Family', 'LinkedFamily']]
            for Family in Families:
                row = self._get_row_from_BC_Family(self.tree, Family)
        
                try:
                    rowParams = self.ApplicationContext['Rows'][row]
                    fluxcoeff = rowParams['NumberOfBlades'] / float(rowParams['NumberOfBladesSimulated'])
                except KeyError:
                    # since a FamilyNode does not necessarily belong to a row
                    fluxcoeff = 1.
                
                mola_logger.debug(f'fluxcoeff on Family {Family} is {fluxcoeff}')
                self.ApplicationContext['NormalizationCoefficient'][Family] = dict(FluxCoef=fluxcoeff)
            