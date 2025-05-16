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

from treelab import cgns

from mola.logging import mola_logger, MolaException, MolaAssertionError, redirect_streams_to_null, redirect_streams_to_logger
from mola.cfd.preprocess.mesh import duplicate
from mola.cfd.preprocess.mesh.families import get_family_nodes_from_patterns, get_family_names_from_patterns
from mola.cfd.preprocess.mesh.tools import parametrize_with_height

from .. import Workflow
from .interface import WorkflowRotatingComponentInterface


class WorkflowRotatingComponent(Workflow):

    '''
    Base workflow, that inherits from Workflow and add methods for every applications with 
    rotating components. If possible, it should not impose different default values that Workflow.
    Other applicative workflows with rotating components inherits from WorkflowRotatingComponent, 
    for instance WorkflowTurbomachinery and WorkflowPropeller.

    Examples
    --------

    .. code:: python

        ApplicationContext = dict(
            ShaftAxis = [1, 0, 0],
            RPM = 5000., # rotation speed in rotation per minute
            # or you can also provide:
            # ShaftRotationSpeed = ..., # in rad/s
            HubRotationIntervals = [(xmin1, xmax1), (xmin2, xmax2)]  
            Rows = dict(
                Rotor = dict(
                    IsRotating = True,
                    NumberOfBlades = 36,
                )
            )
        )

    '''

    def __init__(self, **kwargs):
        self._interface = WorkflowRotatingComponentInterface(self, **kwargs)

    def duplicate(self):
        # duplicate.duplicate_workflow_with_cassiopee(self)
        duplicate.duplicate_workflow_with_maia(self)

    def process_mesh(self):
        super().process_mesh()
        self.set_default_parameters_for_rows()
        self.compute_fluxcoef_by_row() 
        self.duplicate()

    def initialize_flow(self):
        self.Initialization.setdefault('ParametrizeWithHeight', None)
        if self.Initialization['ParametrizeWithHeight'] is None \
            and any([ext['Type'] == 'IsoSurface' and ext['IsoSurfaceField'] == 'ChannelHeight' for ext in self.Extractions]):
            self.Initialization['ParametrizeWithHeight'] = 'maia'

        if self.Initialization['ParametrizeWithHeight'] == 'maia':
            self.parametrize_with_height()
        elif self.Initialization['ParametrizeWithHeight'] == 'turbo':
            self.parametrize_with_height_with_turbo()
            
        super().initialize_flow()

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

            n = self.get_number_of_blades_in_mesh_from_family(row, rowParams['NumberOfBlades'])
            rowParams.setdefault('NumberOfBladesInInitialMesh', n)     

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
        self.set_hub_boundary_conditions()
        self.set_blade_boundary_conditions()

        super().set_boundary_conditions()  

    def set_shroud_boundary_conditions(self, families=['shroud', 'carter']):
        for famNode in get_family_nodes_from_patterns(self.tree, families):
            FamilyBoundary = famNode.name()
            if self._is_boundary_already_defined(FamilyBoundary) or self._is_boundary_to_skip(FamilyBoundary):
                continue
            
            self.BoundaryConditions.append(
                # Careful, it is mandatory to impose a null Motion on the shroud, 
                # otherwise the frame of reference of the BC will be inheritated 
                # from the zone with a FoR in rotation 
                dict(Family=FamilyBoundary, Type='Wall', Motion=dict(RotationSpeed=[0.,0.,0.]))  
                )
    
    def set_blade_boundary_conditions(self, families=['blade', 'aube']):
        for famNode in get_family_nodes_from_patterns(self.tree, families):
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
        for famNode in get_family_nodes_from_patterns(self.tree, families):
            FamilyBoundary = famNode.name()
            if self._is_boundary_already_defined(FamilyBoundary) or self._is_boundary_to_skip(FamilyBoundary):
                continue

            if not 'HubRotationIntervals' in self.ApplicationContext:
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
        if not one_bc_FamilyName:
            raise MolaException(
                f'No FamilyName found with value {FamilyBoundary}. '
                'Check Families in the MOLA attribute "BoundaryConditions".'
                )
        zone = one_bc_FamilyName.getParent(Type='Zone_t')
        row_family = zone.get(Type='FamilyName', Depth=1).value()
        return row_family
       
    def _get_hub_rotation_function(self):
        if isinstance(self.ApplicationContext['HubRotationIntervals'], list):
            # FIXME not working for now because treelab cannot write a list of tuples or lists

            if list(self.ApplicationContext['ShaftAxis']) != [1., 0., 0.]:
                raise MolaAssertionError(f"Cannot handle hub rotation if the shaft axis is not the X axis.")

            def hub_rotation_function(CoordinateX):
                omega = np.zeros(CoordinateX.shape, dtype=float)
                for interval in self.ApplicationContext['HubRotationIntervals']:
                    omega[(interval['xmin']<=CoordinateX) & (CoordinateX<=interval['xmax'])] = self.ApplicationContext['ShaftRotationSpeed']
                return omega

        else:
            assert callable(self.ApplicationContext['HubRotationIntervals'])
            hub_rotation_function = self.ApplicationContext['HubRotationIntervals']

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
        zonesInFamily = [z for z in t.zones() if z.get(Type='FamilyName', Value=FamilyName)]
        # Slice in x direction at middle range
        xmin = np.amin([np.amin(zone.x()) for zone in zonesInFamily])
        xmax = np.amax([np.amax(zone.x()) for zone in zonesInFamily])
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

    def parametrize_with_height(self, hub_families=['hub', 'moyeu'], 
                                shroud_families=['shroud', 'carter'], GridLocation='Vertex'):
        self.tree = parametrize_with_height(
            self.tree, 
            hub_families=get_family_names_from_patterns(self.tree, hub_families), 
            shroud_families=get_family_names_from_patterns(self.tree, shroud_families), 
            GridLocation=GridLocation
            )

    def parametrize_with_height_with_turbo(self, method=2):
        '''
        Compute the variable *ChannelHeight* from a mesh PyTree **t**. This function
        relies on the turbo module.

        .. important::

            Dependency to *turbo* module. See file:///stck/jmarty/TOOLS/turbo/doc/html/index.html

        Parameters
        ----------

            method : int
                Method used for ``turbo.height.generateHLinesAxial()``. Default value is 2.
        '''
        import os
        import Converter.Internal as I
        import turbo.height as TH

        def plot_hub_and_shroud_lines(t):
            # Get geometry
            hub     = I.getNodeFromName(t, 'Hub')
            xHub    = I.getValue(I.getNodeFromName(hub, 'CoordinateX'))
            yHub    = I.getValue(I.getNodeFromName(hub, 'CoordinateY'))
            shroud  = I.getNodeFromName(t, 'Shroud')
            xShroud = I.getValue(I.getNodeFromName(shroud, 'CoordinateX'))
            yShroud = I.getValue(I.getNodeFromName(shroud, 'CoordinateY'))
            # Import matplotlib
            import matplotlib.pyplot as plt
            # Plot
            plt.figure()
            plt.plot(xHub, yHub, '-', label='Hub')
            plt.plot(xShroud, yShroud, '-', label='Shroud')
            plt.axis('equal')
            plt.grid()
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            # Save
            merid_lines_image_filename = os.path.join(self.RunManagement['RunDirectory'], 'shroud_hub_lines.png')
            plt.savefig(merid_lines_image_filename, dpi=150, bbox_inches='tight')
            return 0

        mola_logger.info('Add ChannelHeight in the mesh...')
        OLD_FlowSolutionNodes = I.__FlowSolutionNodes__
        I.__FlowSolutionNodes__ = 'FlowSolution#Height'

        # HACK ETC needs that IndexRange_t nodes have a value of type int32, and not int64
        for node in self.tree.group(Type='IndexRange'):
            node.setValue(np.asarray(node.value(), dtype=np.int32))

        with redirect_streams_to_logger(mola_logger, stdout_level='DEBUG', stderr_level='ERROR'):
            
            merid_lines_filename = 'shroud_hub_lines.plt'  #os.path.join(self.RunManagement['RunDirectory'], 'shroud_hub_lines.plt')
            endlinesTree = TH.generateHLinesAxial(self.tree, filename=merid_lines_filename, method=method)
            try: 
                plot_hub_and_shroud_lines(endlinesTree)
            except: 
                pass

            # - Generation of the mask file
            m = TH.generateMaskWithChannelHeight(self.tree, merid_lines_filename)
            os.remove(merid_lines_filename)
            # mask_filename = os.path.join(self.RunManagement['RunDirectory'], 'mask.cgns')
            # os.remove(mask_filename) # remove this file for now, but it will be maybe necessary for other operations later

            # - Generation of the ChannelHeight field
            TH._computeHeightFromMask(self.tree, m)
        
        I.__FlowSolutionNodes__ = OLD_FlowSolutionNodes
        
        self.tree = cgns.castNode(self.tree)

    # def normalize_data_from_extraction(self, Family, data_tree):
    #     data_to_normalize = dict(
    #         MassFlow = dict(Name='MassFlowTotal', Coef='FluxCoef'),
    #         CL = dict(Name='CL', Coef='FluxCoef'),
    #         CD = dict(Name='CD', Coef='FluxCoef'),
    #         CY = dict(Name='CY', Coef='FluxCoef'),
    #         Cn = dict(Name='Cn', Coef='TorqueCoef'),
    #         Cl = dict(Name='Cl', Coef='TorqueCoef'),
    #         Cm = dict(Name='Cm', Coef='TorqueCoef'),
    #     )
    #     for name, params in data_to_normalize.items():
    #         new_name = params['Name']
    #         try:
    #             coef = self.ApplicationContext['NormalizationCoefficient'][Family][params['Coef']]
    #         except:
    #             continue
    #         for node in data_tree.group(Name=name, Type='DataArray'):
    #             # node.setName(new_name)
    #             # node.setValue(node.value()*coef)
    #             node.Parent.findAndRemoveNode(Name=new_name, Depth=1)
    #             cgns.Node(Type='DataArray', Name=new_name, Value=node.value()*coef, Parent=node.Parent)
            
    def plot_radial_profiles(self, *args, **kwargs):
        from mpi4py import MPI
        if MPI.COMM_WORLD.Get_rank() == 0:
            from mola.visu import plot_radial_profiles
            plot_radial_profiles(*args, **kwargs)

    @staticmethod
    def remove_row(tree, row, interface_family=None, new_interface_family=None):
        mola_logger.info(f'Remove row {row}')
        tree.findAndRemoveNodes(Type='Family', Name=f'{row}*')
        tree.findAndRemoveNodes(Type='Zone', Name=f'{row}*')
        tree.findAndRemoveNodes(Type='Family', Name='Rotor_stator_10_right')
        tree.findAndRemoveNodes(Type='GridConnectivity', Value=f'{row}*')

        # modifies interface Family
        if interface_family is not None and new_interface_family is not None: 
            mola_logger.info(f'Rename {interface_family} to {new_interface_family}')
            fam = tree.get(Type='Family', Name=interface_family)
            fam.setName(new_interface_family)
            bcs = [bc for bc in tree.group(Type='BC') if bc.get(Type='FamilyName', Value=interface_family)]
            for bc in bcs:
                bc.findAndRemoveNode(Name='InterfaceType')
                bc.findAndRemoveNode(Name='DonorFamily')
                bc.get(Type='FamilyName').setValue(new_interface_family)
