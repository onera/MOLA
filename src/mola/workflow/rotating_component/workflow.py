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
from mola.cfd.preprocess.mesh.families import get_bc_family_nodes_from_patterns, get_bc_family_names_from_patterns
from mola.cfd.preprocess.mesh.tools import parametrize_with_height, compute_azimuthal_extension
from mola.cfd.preprocess import initialization

from .. import Workflow
from .interface import WorkflowRotatingComponentInterface

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
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

    def __init__(self,
            
            # CAVEAT required for making accessible private attributes of
            # RotatingComponent to lower-level (inherited) workflows, without
            # suffering from the inconvenience of instantiating RotatingComponent
            # interface, and requiring to provide mandatory inputs to RotatingComponent
            # that will be overriden (or incompatible) to inherited workflow inputs.
            # This should be redesigned, for exemple, by never requiring mandatory
            # inputs on intermediary workflows.
            _skip_interface=False, 

            **kwargs):
        if not _skip_interface:
            self._interface = WorkflowRotatingComponentInterface(self, **kwargs)
        self._hub_patterns = ['hub', 'moyeu', 'spinner']
        self._blade_patterns = ['blade', 'aube', 'propeller', 'rotor', 'stator']
        self._shroud_patterns = ['shroud', 'carter']

    def define_families(self):
        super().define_families()
        self.set_default_parameters_for_rows() 

    def initialize_flow(self):
        if self.Initialization['Method'] in ['copy', 'interpolate']:
            mola_logger.info("🔥 prepare source tree for initialization", rank=0)
            # We need to check if the source mesh and the target mesh has the same 
            # azimuthal extension. Otherwise, we need to duplicate accordly the source mesh
            self.Initialization['Source'] = cgns.load(self.Initialization['Source'])

            # Check the azimuthal extension of the source mesh
            # CAVEAT: Only for one row for now -> either the source tree is from a steady simulation with one blade per row, 
            # or it is already duplicated for all rows as in the target tree
            for row, rowParams in self.ApplicationContext['Rows'].items():
                if not self.Initialization['Source'].get(Name=row, Type='Family', Depth=2):
                    row = None
                n = self.get_number_of_blades_in_mesh_from_family(self.Initialization['Source'], row, rowParams['NumberOfBlades'])

                if n != rowParams['NumberOfBladesSimulated']:
                    from mola.cfd.preprocess.mesh.duplicate import apply_duplication_on_tree
                    mola_logger.info('  source mesh is duplicated to initialize the flow')
                    self.Initialization['Source'] = apply_duplication_on_tree(self, self.Initialization['Source'])
                    break

        if self.Initialization['Method'] == 'turbo':
            self.parametrize_with_height()
            super().initialize_flow()

        else:
            super().initialize_flow()
            # Do not recompute ChannelHeight if it was in source data
            if not self.tree.get(Name='ChannelHeight', Type='DataArray'):
                self.parametrize_with_height()

    def set_default_parameters_for_rows(self):

        some_duplication_has_been_done = False

        for row, rowParams in self.ApplicationContext['Rows'].items():

            if not self.tree.get(Name=row, Type='Family', Depth=2):
                raise MolaException(f'The family {row} given in ApplicationContext is not found in the mesh.')

            if hasattr(self, 'BodyForceModeling') and row in self.BodyForceModeling:
                # Replace the number of blades to be consistant with the body-force mesh
                azimuthal_extension = compute_azimuthal_extension(self.tree, row, axis=self.ApplicationContext['ShaftAxis'])
                rowParams['NumberOfBlades'] = int(2*np.pi / azimuthal_extension)
                rowParams['NumberOfBladesInInitialMesh'] = 1
                mola_logger.info(f'Number of blades for {row}: {rowParams["NumberOfBlades"]} (got from the body-force mesh)')

            if "NumberOfBladesInInitialMesh" not in rowParams:
                azimuthal_extension = compute_azimuthal_extension(self.tree, row, axis=self.ApplicationContext['ShaftAxis'])
                n = self.get_number_of_blades_in_mesh_from_family(self.tree, row, rowParams['NumberOfBlades'], azimuthal_extension)
                rowParams.setdefault('NumberOfBladesInInitialMesh', n)    

            duplications_to_do = rowParams['NumberOfBladesSimulated'] - rowParams['NumberOfBladesInInitialMesh']
            if duplications_to_do > 0:
                operation = dict(
                    Type = 'DuplicateByRotation',
                    Family = row,
                    NumberOfDuplications = duplications_to_do,
                )
                
                # CAVEAT: works only for one component
                if len(self.RawMeshComponents) > 1:
                    raise MolaAssertionError('Multiple components are not supported yet in this case')
                component = self.RawMeshComponents[0]
                component.setdefault('Positioning', [])
                component['Positioning'].append(operation)

                some_duplication_has_been_done = True

        if some_duplication_has_been_done:
            # HACK if duplication is done with Cassiopee, connectivities are lost
            # In the following block, new connections are added to workflow, 
            # to be done after duplication with Cassiopee 

            # component.setdefault('Connection', [])
            component['Connection'] = []

            if not any([elem['Type']=='Match' for elem in component['Connection']]):
                component['Connection'].append(dict(Type='Match', Tolerance=component['DefaultToleranceForConnection']))

            for row, rowParams in self.ApplicationContext['Rows'].items():
                azimuthal_extension = compute_azimuthal_extension(self.tree, row, axis=self.ApplicationContext['ShaftAxis'])
                duplications_to_do = rowParams['NumberOfBladesSimulated'] - rowParams['NumberOfBladesInInitialMesh']
                angle = np.degrees(azimuthal_extension) * (duplications_to_do+1)

                if not np.isclose(angle, 360.):
                    RotationAngle = angle*self.ApplicationContext['ShaftAxis']
                    # check if the same connection PeriodicMatch does not already exist with the same angle
                    if not any([elem['Type']=='PeriodicMatch' and 
                                np.allclose(elem['RotationAngle'], RotationAngle) 
                                for elem in component['Connection']]):
                        component['Connection'].append(
                            dict(
                                Type='PeriodicMatch', 
                                Tolerance=component['DefaultToleranceForConnection'], 
                                RotationAngle=RotationAngle,
                                Families=(f'{row}_PER1', f'{row}_PER2'),
                                )
                            )
                        
        self.compute_fluxcoef_by_row()

    def set_motion(self):
        for row, rowParams in self.ApplicationContext['Rows'].items():

            IsModelledWithBodyForce = hasattr(self, 'BodyForceModeling') and row in self.BodyForceModeling

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

    def set_shroud_boundary_conditions(self):
        for famNode in get_bc_family_nodes_from_patterns(self.tree, self._shroud_patterns):
            bc_family_name = famNode.name()
            
            if self._is_boundary_already_defined(bc_family_name) or self._is_boundary_to_skip(bc_family_name):
                continue

            self.BoundaryConditions.append(
                # Careful, it is mandatory to impose a null Motion on the shroud, 
                # otherwise the frame of reference of the BC will be inheritated 
                # from the zone with a FoR in rotation 
                dict(Family=bc_family_name, Type='Wall', Motion=dict(RotationSpeed=[0.,0.,0.]))  
                )
    
    def set_blade_boundary_conditions(self):
        for famNode in get_bc_family_nodes_from_patterns(self.tree, self._blade_patterns):
            bc_family_name = famNode.name()
            
            if self._is_boundary_already_defined(bc_family_name) or self._is_boundary_to_skip(bc_family_name):
                continue
            
            row_family = self._get_row_from_BC_Family(self.tree, bc_family_name)

            try:
                self.BoundaryConditions.append(
                    dict(Family=bc_family_name, Type='Wall', Motion=self.Motion[row_family])
                    )
            except KeyError:
                self.BoundaryConditions.append(dict(Family=bc_family_name, Type='Wall'))
    
    def set_hub_boundary_conditions(self):
        for famNode in get_bc_family_nodes_from_patterns(self.tree, self._hub_patterns):
            bc_family_name = famNode.name()
            
            if self._is_boundary_already_defined(bc_family_name) or self._is_boundary_to_skip(bc_family_name):
                continue

            if not 'HubRotationIntervals' in self.ApplicationContext:
                # Assume that hub rotates at the same speed that the zone family
                mola_logger.user_warning(f'Assume that motion is uniform on bc family "{bc_family_name}".')
                row_family = self._get_row_from_BC_Family(self.tree, bc_family_name)
                try:
                    self.BoundaryConditions.append(
                        dict(Family=bc_family_name, Type='Wall', Motion=self.Motion[row_family])
                        )
                except KeyError:
                    self.BoundaryConditions.append(dict(Family=bc_family_name, Type='Wall'))
            else:
                self.BoundaryConditions.append(
                    dict(Family=bc_family_name, Type='Wall', Motion=dict(RotationSpeed=self._get_hub_rotation_function()))
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
        from mpi4py.MPI import COMM_WORLD as comm
        one_bc_FamilyName = tree.get(Type='FamilyName', Value=FamilyBoundary)
        nodes_on_all_ranks = comm.allgather(one_bc_FamilyName)
        one_bc_FamilyName = [node for node in nodes_on_all_ranks if node is not None][0]

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

    @staticmethod
    def get_number_of_blades_in_mesh_from_family(tree, FamilyName, NumberOfBlades, azimuthal_extension=None):
        '''
        Compute the number of blades for the row **FamilyName** in the mesh.

        Returns
        -------
        int
            Number of blades in the mesh for row **FamilyName**

        '''
        if azimuthal_extension is None:
            azimuthal_extension = compute_azimuthal_extension(tree, FamilyName)
        # Compute number of blades in the mesh
        Nb = NumberOfBlades * azimuthal_extension / (2*np.pi)
        Nb = int(np.round(Nb))
        mola_logger.info(f'Number of blades in initial mesh for {FamilyName}: {Nb}', rank=0)
        if Nb < 1:
            raise MolaAssertionError(
                f'The number of blades in initial mesh {FamilyName} cannot be computed correctly.'
                ' Please check the orientation and scale of the mesh. If the mesh is correct,'
                ' but the error is persistent, you may use the argument NumberOfBladesInInitialMesh'
                ' in ApplicationContext to fix manually fix this.'
                )
        return Nb

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

    def get_hub_family_names(self, must_be_unique=False, must_exist=False)  -> list:

        tree= self.__choose_skeleton_tree_if_existent()
        names = get_bc_family_names_from_patterns(tree, self._hub_patterns)
        
        if must_be_unique:
            must_exist = True

        if must_exist and len(names)==0:
            raise MolaException('did not find any family associated to hub')
        
        elif must_be_unique and len(names)!=1:
            raise MolaException(f"expected a unique family name for hub but got: {names}")
        
        return names

    def __choose_skeleton_tree_if_existent(self):
        # this is relevant in the context of parallel computation using PyPart,
        # since Skeleton is not merged into tree and information on relevant may
        # be missing in main tree
        if hasattr(self,"_Skeleton") and bool(self._Skeleton):
            return self._Skeleton
        return self.tree


    def get_blade_family_names(self, must_be_unique=False, must_exist=False) -> list:

        tree = self.__choose_skeleton_tree_if_existent()
        names = get_bc_family_names_from_patterns(tree, self._blade_patterns)
        # Filter blade tip families
        # names = [name for name in names if not name.lower().endswith("tip")]
        
        if must_be_unique:
            must_exist = True

        if must_exist and len(names)==0:
            raise MolaException('did not find any family associated to blade')
        
        elif must_be_unique and len(names)!=1:
            raise MolaException(f"expected a unique family name for blade but got: {names}")
        
        return names

    def get_shroud_family_names(self, must_be_unique=False, must_exist=False) -> list:

        tree= self.__choose_skeleton_tree_if_existent()
        names = get_bc_family_names_from_patterns(tree, self._shroud_patterns)
        
        if must_be_unique:
            must_exist = True

        if must_exist and len(names)==0:
            raise MolaException('did not find any family associated to shroud')
        
        elif must_be_unique and len(names)!=1:
            raise MolaException(f"expected a unique family name for shroud but got: {names}")
        
        return names



    def parametrize_with_height(self):
        self.Initialization.setdefault('ParametrizeWithHeight', None)
        if self.Initialization['ParametrizeWithHeight'] is None \
            and any([ext['Type'] == 'IsoSurface' and ext['IsoSurfaceField'] == 'ChannelHeight' for ext in self.Extractions]):
            self.Initialization['ParametrizeWithHeight'] = 'maia'

        if self.Initialization['ParametrizeWithHeight'] == 'maia':
            self.parametrize_with_height_with_maia()
        elif self.Initialization['ParametrizeWithHeight'] == 'turbo':
            self.parametrize_with_height_with_turbo()

    def parametrize_with_height_with_maia(self, GridLocation='Vertex'):
        self.tree = parametrize_with_height(
            self.tree, 
            hub_families = self.get_hub_family_names(), 
            shroud_families = self.get_shroud_family_names(), 
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
            
    def plot_radial_profiles(self, *args, **kwargs):
        if MPI.COMM_WORLD.Get_rank() == 0:
            from mola.visu import plot_radial_profiles
            plot_radial_profiles(*args, **kwargs)

    @staticmethod
    def _compute_maximum_distance_to_axis_from(tree : cgns.Tree, 
            axis = np.array([1.0,0.0,0.0]), center = np.array([0.0,0.0,0.0]) ):
        
        c = center
        a = axis
        max_squared_distance = 0.0
        zone : cgns.Zone
        zones = tree.zones()

        if not zones:
            tree.save(f'debug_tree_rank_{rank}.cgns')
            raise TypeError("no zones contained in tree, check debug file")

        for zone in zones:
            x, y, z = zone.xyz(ravel=True)
            for i in range(len(x)):
                p = np.array([x[i], y[i], z[i]])
                v = (c-p)- ((c-p).dot(a))*a
                squared_distance = v.dot(v)
                max_squared_distance = np.maximum(max_squared_distance, squared_distance)
        
        comm.barrier()
        each_rank_max_squared_distances = comm.gather(max_squared_distance, 0)
        if rank ==0:
            absolute_max_squared_distance = max(each_rank_max_squared_distances)
            radius = np.sqrt(absolute_max_squared_distance)
        comm.barrier()
        assert radius > 0, max_squared_distance
        radius = comm.bcast(radius,0)
        return radius


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
