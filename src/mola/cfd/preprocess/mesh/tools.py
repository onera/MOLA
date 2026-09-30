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
from fnmatch import fnmatch
from packaging.version import Version
from mola.logging import mola_logger, MolaException, MolaAssertionError, YELLOW, ENDC
from mola.pytree.user.checker import (is_partitioned_for_use_in_maia,
                                      is_distributed_for_use_in_maia)
from mola.cfd import parallel_execution_with_maia
from treelab import cgns

from mola.mpi import get_mpi_elements_safely
comm, rank, size, _ = get_mpi_elements_safely()

@parallel_execution_with_maia()
def parametrize_with_height(tree, hub_families, shroud_families, GridLocation='Vertex'):
    import maia
    import maia.pytree as PT
    from maia.algo.part.wall_distance import compute_projection_to
    from mpi4py.MPI import LAND

    mola_logger.info('Parametrize domain with channel height (add FlowSolution#Height)', rank=0)

    tree = to_partitioned(tree) 

    if Version(maia.__version__) > Version('1.7'):
        # Change of name of the module "predicate" in "pred"
        hub_bc_predicate    = PT.pred.any([PT.pred.belongs_to_family(family) for family in hub_families])
        shroud_bc_predicate = PT.pred.any([PT.pred.belongs_to_family(family) for family in shroud_families])
    else:
        hub_bc_predicate = lambda n : any([PT.predicate.belongs_to_family(n, wall_bc_family) for wall_bc_family in hub_families])
        shroud_bc_predicate = lambda n : any([PT.predicate.belongs_to_family(n, wall_bc_family) for wall_bc_family in shroud_families])

    hub_was_not_found = comm.reduce(len(PT.get_nodes_from_predicate(tree, hub_bc_predicate)) == 0, op=LAND)
    if hub_was_not_found:
        raise MolaException(f'Cannot find hub families in tree from names {hub_families}')
    shroud_was_not_found = comm.reduce(len(PT.get_nodes_from_predicate(tree, shroud_bc_predicate)) == 0, op=LAND)
    if shroud_was_not_found:
        raise MolaException(f'Cannot find shroud families in tree from names {shroud_families}')
    
    # TODO make this operation separately for each row family to prevent errors
    # and to reduce the duration of the process
    # Compute distances to hub and shroud
    compute_projection_to(tree, hub_bc_predicate, comm, out_fs_name='DistanceToHub', point_cloud=GridLocation)
    compute_projection_to(tree, shroud_bc_predicate, comm, out_fs_name='DistanceToShroud', point_cloud=GridLocation)

    # Compute ChannelHeight
    for zone in PT.get_all_Zone_t(tree):
        d1 = PT.get_value(PT.get_node_from_path(zone, 'DistanceToHub/Distance'))
        d2 = PT.get_value(PT.get_node_from_path(zone, 'DistanceToShroud/Distance'))
        d2[np.abs(d1) < 1e-16] = 1.  # if the hub radius tends to 0, regularize the expression 
        PT.new_FlowSolution(
            name='FlowSolution#Height', 
            loc=GridLocation, 
            fields=dict(ChannelHeight = d1 / (d1+d2) ), 
            parent=zone
            )
        # remove distances to hub and shroud
        PT.rm_node_from_path(zone, 'DistanceToHub')
        PT.rm_node_from_path(zone, 'DistanceToShroud')

    return cgns.castNode(tree)

def get_bc_from_bc_type(workflow, bctypes):
    if isinstance(bctypes, str):
        bctypes = [bctypes]

    # Get BCs
    BCs = []
    for bctype in bctypes:
        BCs += [bc for bc in workflow.BoundaryConditions if fnmatch(bc['Type'], bctype)]

    # Check unicity
    if len(BCs) == 0:
        raise MolaException(f'There is no Family in BoundaryConditions matching the type {bctypes}')
    elif len(BCs) > 1:
        raise MolaException(f'There is more than one Family in BoundaryConditions matching the type {bctypes}: {[BC for BC in BCs]}')
    else:
        BC = BCs[0]

    return BC

def get_surface_of_family(tree, Family):
    import Converter.PyTree as C
    import Converter.Internal as I
    import Post.PyTree as P

    from mpi4py.MPI import SUM

    tree = I.fixNGon(tree)  # it would be better without making a copy, but it would need adaptation latter for maia
    zones = C.extractBCOfName(tree, f'FamilySpecified:{Family}')
    if len(zones) == 0:
        raise MolaException(f'Cannot extract family {Family}')
    SurfaceTree = C.convertArray2Tetra(zones)
    SurfaceTree = C.initVars(SurfaceTree, 'ones=1')
    Surface = P.integ(SurfaceTree, var='ones')[0]        # Compute normalization coefficient
    Surface = comm.allreduce(Surface, op=SUM)
    mola_logger.debug(f'Surface of family {Family} = {Surface} m^2', rank=0)

    return Surface


def compute_azimuthal_extension(tree, Family, method='from_periodic', axis=None):
    
    # Extract zones in family
    if Family is None:
        sub_tree = tree
    else:
        zonesInFamily = [z for z in tree.zones() if z.get(Type='FamilyName', Value=Family)]
        sub_tree = cgns.Tree()
        base = cgns.Base(Parent=sub_tree)
        base.addChildren(zonesInFamily)

    if method == 'from_slice':
        dθ = _compute_azimuthal_extension_from_slice(sub_tree, axis=axis)
    elif method == 'from_periodic':
        try:
            dθ = _compute_azimuthal_extension_from_periodic(sub_tree)
        except MolaAssertionError as err:
            mola_logger.debug(f'{err} {YELLOW}--> Try to compute azimuthal extension with method "from_slice"{ENDC}')
            dθ = _compute_azimuthal_extension_from_slice(sub_tree, axis=axis)
    else:
        raise MolaAssertionError(f'unknown {method=} for compute_azimuthal_extension')
    
    return dθ

def _compute_azimuthal_extension_from_slice(t, axis=None):
    '''
    Compute the azimuthal extension in radians of the mesh **t**.

    .. warning:: This function needs to calculate the surface of the slice in X
                at Xmin + 5% (Xmax - Xmin). If this surface is crossed by a
                solid (e.g. a blade) or by the inlet boundary, the function
                will compute a wrong value of the number of blades inside the
                mesh.
    '''
    import Converter.PyTree as C
    import Post.PyTree as P

    if axis is None:
        axis = [1.0, 0.0, 0.0]

    if list(axis) != [1.0, 0.0, 0.0]:
        # CAVEAT
        raise MolaAssertionError('For now, this function only handles axis=[1., 0., 0.]')

    # Slice in x direction at middle range
    xmin = np.amin([np.amin(zone.x()) for zone in t.zones()])
    xmax = np.amax([np.amax(zone.x()) for zone in t.zones()])
    sliceX = P.isoSurfMC(t, 'CoordinateX', value=xmin+0.05*(xmax-xmin))
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

def _compute_azimuthal_extension_from_periodic(t):
    periodic_node = t.get(Type='Periodic')
    if periodic_node is None:
        raise MolaAssertionError(f'Cannot found a Periodic node in tree.')
    
    # RotationCenter = periodic_node.get(Name='RotationCenter').value()
    RotationAngle = periodic_node.get(Name='RotationAngle').value()
    # Translation = periodic_node.get(Name='Translation').value()
    if np.isclose(RotationAngle[1], 0.) and np.isclose(RotationAngle[2], 0.):
        dθ = abs(RotationAngle[0])
    elif np.isclose(RotationAngle[0], 0.) and np.isclose(RotationAngle[2], 0.):
        dθ = abs(RotationAngle[1])
    elif np.isclose(RotationAngle[0], 0.) and np.isclose(RotationAngle[1], 0.):
        dθ = abs(RotationAngle[2])
    else:
        raise MolaException('Cannot found the rotation axis')

    try:
        unit = RotationAngle.get(Type='DimensionalUnits').value()[4]
    except:
        unit = 'Radian'

    if unit =='Radian':
        pass
    elif unit == 'Degree':
        dθ = np.radians(dθ)
    else:
        raise MolaException(f'unknown unit for rotation angle: {unit}. Must be Radian or Degree')

    return dθ


def to_distributed(tree : cgns.Tree, data_transfer = 'ALL'):

    import maia

    if bool(tree.get(':CGNS#Distribution')): 
        t = tree
    
    else:
        mola_logger.debug('convert tree to maia dist_tree')

        if bool(tree.get(':CGNS#GlobalNumbering')):
            t = maia.factory.recover_dist_tree(tree, comm,
                                                       data_transfer = data_transfer)
        else:
            t = maia.factory.full_to_dist_tree(tree, comm)
        
        t = cgns.castNode(t)
     
    return t

def to_partitioned(tree : cgns.Tree):

    import maia
    
    is_maia_dist = is_distributed_for_use_in_maia(tree)
    distrib = extract_cassiopee_distribution(tree)
    is_cass_part = bool(distrib)
    
    is_maia_part = is_partitioned_for_use_in_maia(tree)

    if is_maia_part: 
        return tree
    
    else:
        _apply_all_maia_check(tree)
        if not is_maia_dist:
            tree = maia.factory.full_to_dist_tree(tree, comm)
        tree = cgns.castNode(tree)
        return to_partitioned_if_distributed(tree, cassiopee_distribution=distrib)

def extract_cassiopee_distribution(tree : cgns.Tree) -> dict:
    distribution = {}
    for base in tree.bases():
        for zone in base.zones():
            base_name_plus_zone_name = base.name() + "/" + zone.name()

            solver_param = zone.get(Name='.Solver#Param', Depth=1)
            if not solver_param: continue
            
            proc_node = solver_param.get(Name='proc', Depth=1)
            if not proc_node: continue
            
            proc = int(proc_node.value())
            
            distribution[base_name_plus_zone_name] = proc

    return distribution


def to_partitioned_if_distributed(tree : cgns.Tree, cassiopee_distribution={}):
    is_dist = is_distributed_for_use_in_maia(tree)
    if not is_dist: return tree


    import maia

    mola_logger.debug('convert tree to maia part_tree')

    t = maia.factory.partition_dist_tree(tree, comm, data_transfer='ALL', preserve_orientation=True)

    t = cgns.castNode(t)
    for base in t.bases():
        for zone in base.zones():
            base_name_plus_zone_name = base.name() + "/" + remove_maia_part_zone_suffix(zone.name())
            proc = cassiopee_distribution.get(base_name_plus_zone_name, rank)

            zone.setParameters('.Solver#Param', proc=proc)

            if zone.isStructured(): 
                reshape_DataArray(zone)
        
    return t


def remove_maia_part_zone_suffix(zone_name : str) -> str:
    import re
    return re.sub(r'\.P\d+\.N\d+$', '', zone_name)


def to_full_tree_at_rank_0(tree : cgns.Tree):

    import maia
    comm.barrier()
    
    try:
        import maia.pytree.maia.check_tree as check
        is_part = check.is_cgns_part_tree(tree)
        is_dist = check.is_cgns_dist_tree(tree)
        is_full = check.is_cgns_full_tree(tree)
    
    except ModuleNotFoundError:
        import mola.pytree.user.checker as check
        is_part = check.is_partitioned_for_use_in_maia(tree)
        is_dist = check.is_distributed_for_use_in_maia(tree)
        is_full = not is_part and not is_dist
                   
    if is_full or size == 1:
        return tree
    else:
        mola_logger.debug('convert tree to maia full_tree')
    
    if is_part:
        additional_nodes_to_transfer = _get_additional_nodes_to_transfer(tree)
        tree = maia.factory.recover_dist_tree(tree, comm, data_transfer='ALL')
        tree = cgns.castNode(tree)
        _transfer_additional_nodes(tree, additional_nodes_to_transfer)

    empty_node_manager = EmptyNodesManager()
    empty_node_manager.get_empty_FlowSolution_nodes(tree, remove=True)

    tree = maia.factory.dist_to_full_tree(tree, comm, target=0)
    tree = cgns.castNode(tree)
    empty_node_manager.restore_empty_FlowSolution_nodes(tree)
    
    if tree is not None:
        tree = cgns.castNode(tree)

        for zone in tree.zones():
            if zone.isStructured(): 
                reshape_DataArray(zone)
            
        tree = cgns.castNode(tree)
    comm.barrier()
    return tree

def is_partitioned(tree, backend='maia'):
    if backend == 'maia':
        from mola.cfd.preprocess.mesh import maia_wrapper
        return maia_wrapper.is_partitioned(tree)

def _get_additional_nodes_to_transfer(tree):
    # HACK see https://gitlab.onera.net/numerics/mesh/maia/-/issues/175
    types = ['ReferenceState_t', 'FlowEquationSet_t', 'Descriptor_t', 'ConvergenceHistory_t', 'IntegralData_t']
    nodes = []
    for node_type in types:
        nodes.extend(tree.group(Type=node_type, Depth=2))  # workaround only for bases children, not zones children, just for safety
    return nodes

def _transfer_additional_nodes(tree, nodes):
    # HACK see https://gitlab.onera.net/numerics/mesh/maia/-/issues/175
    for node in nodes:
        parent = tree.getAtPath(node.parent().path())
        parent.addChild(node)

def _apply_all_maia_check(tree):
    # HACK check operations normally done in maia.io._hdf_io_h5py.load_size_tree
    import maia
    maia.io.fix_tree.check_namings(tree)
    # maia.io.fix_tree.rm_legacy_nodes(tree)
    maia.io.fix_tree.corr_index_range_names(tree)
    # maia.io.fix_tree.check_datasize(tree)
    maia.io.fix_tree.fix_point_ranges(tree)
    maia.io.fix_tree.fix_structured_pr_shape(tree)
    pred_1to1 = 'CGNSBase_t/Zone_t/ZoneGridConnectivity_t/GridConnectivity1to1_t'
    if maia.pytree.get_node_from_predicates(tree, pred_1to1) is not None:
        maia.io.fix_tree.ensure_symmetric_gc1to1(tree)
    maia.io.fix_tree.add_missing_pr_in_bcdataset(tree)

def reshape_DataArray(zone):
    vertex_shape = zone.value()[:,0]
    nvertex = np.sum(vertex_shape)
    cell_shape = zone.value()[:,1]
    ncell = np.sum(cell_shape)
    for coord in zone.xyz():
        coord.shape = vertex_shape

    for field in zone.allFields(return_type='list'):
        nfield = np.size(field)
        if nfield == nvertex:
            field.shape = vertex_shape
        elif nfield == ncell:
            field.shape = cell_shape


def ravel_FlowSolution(t):
    for fs in t.group(Type='FlowSolution'):
        for da in fs.group(Type='DataArray', Depth=1):
            value = da.value()
            if value is not None:
                da.setValue(value.ravel(order='K'))

def remove_empty_BCDataSet(t):
    for node in t.group(Type='BCDataSet'):
        if not node.hasChildren():
            node.remove()

def force_FamilyBC_as_FamilySpecified(t):
    # https://elsa.onera.fr/issues/10928
    for base in t.bases():
        for zone in base.zones():
            for bc in zone.group(Type='BC', Depth=2):
                FamilyName_node = bc.get(Type='FamilyName', Depth=1)
                if FamilyName_node is not None:
                    bc.setValue('FamilySpecified')
                    family = FamilyName_node.value()
                    if not base.get(Name=family, Type='Family', Depth=1):
                        Family_node = cgns.Node(Name=family, Type='Family', Parent=base)
                        cgns.Node(Name='FamilyBC', Type='FamilyBC', Value='UserDefined', Parent=Family_node)
                    continue

class EmptyNodesManager:
    """
    Manager class for handling empty FlowSolution nodes in distributed CGNS trees.

    Empty FlowSolution nodes (nodes with no DataArray children or with DataArray
    nodes storing None values) are important for certain CFD codes like elsA to
    request 3D field extractions. However, mesh partitioning tools (PyPart/Cassiopée)
    do not preserve these empty nodes correctly during distributed operations.

    This class provides a mechanism to:
    1. Detect and store empty FlowSolution nodes before merge operation by the partitioning tool
    2. Distribute node path information across all MPI ranks for consistency
    3a. Restore nodes to their original zones in the merged tree
    3b. or Save empty nodes directly to the CGNS file with the saved merged tree 

    Attributes:
        empty_nodes_by_path (dict): Maps node paths to (owner_rank, node) tuples.
            - owner_rank: MPI rank that originally owned the node
            - node: The FlowSolution node (None if not on source rank)

    Example:
        >>> manager = EmptyNodesManager()
        >>> manager.get_empty_FlowSolution_nodes(tree, remove=True)
        >>> manager.restore_empty_FlowSolution_nodes(tree)
    """

    def __init__(self):
        """Initialize the EmptyNodesManager.

        Sets up the internal dictionary to track empty nodes by their CGNS path.
        """
        self.empty_nodes_by_path = dict()
    
    def _is_empty_zone(self, zone: cgns.Zone):
        return all([
            self._is_empty_flow_solution(node) 
            for node in zone.group(Type='FlowSolution', Depth=1)
            ])

    def _is_empty_flow_solution(self, fs_node: cgns.Node):
        no_DataArray_nodes = len(fs_node.group(Type='DataArray', Depth=1)) == 0
        empty_DataArray_nodes = any(
            n.value() is None for n in fs_node.group(Type='DataArray', Depth=1)
        )
        is_empty = no_DataArray_nodes or empty_DataArray_nodes
        return is_empty
    
    def _add_grid_location(self, fs_node: cgns.Node):
        # if not GridLocation in a FlowSolution, PyPart will raise a Warning/Error in stderr.log
        if fs_node.get(Type='GridLocation'):
            return
        
        GridLocation_Vertex = cgns.Node(Name='GridLocation', Type='GridLocation', Value='Vertex')
        GridLocation_CellCenter = cgns.Node(Name='GridLocation', Type='GridLocation', Value='CellCenter')
        try:
            loc = fs_node.get(Name='loc', Type='DataArray').value()
            if loc == 'cell':
                fs_node.addChild(GridLocation_CellCenter, position=0)
            elif loc == 'node':
                fs_node.addChild(GridLocation_Vertex, position=0)
            else:
                raise MolaException(f'Unknown loc node value ={loc}. Must be cell or node')
        except:
            fs_node.addChild(GridLocation_Vertex, position=0)

    def _remove_PyPart_suffix(self, path: str):
        import re
        # regular expression to find a pattern ".P*.N*", with * a number with 1 to 5 figures
        pattern = r'\.P(\d{1,5})\.N(\d{1,5})'
        try:
            path = re.sub(pattern, '', path)
        except:
            pass
        return path

    def get_empty_FlowSolution_nodes(self, tree: cgns.Tree, remove=False, remove_pypart_suffix=False):
        """Detect and store empty FlowSolution nodes from a CGNS tree.

        This method scans the provided tree for FlowSolution nodes that are either:
        - Empty (no DataArray children)
        - Have DataArray children with None values

        These nodes are stored internally for later restoration. The method also
        distributes the node path information across all MPI ranks so that all
        processes have consistent knowledge of which nodes exist and where they
        originated from.

        Args:
            tree (treelab.cgns.Tree): The CGNS tree to scan for empty nodes.
            remove (bool): If True, remove the empty nodes from the tree after
                storing them. Useful when the nodes need to be removed before
                merging operation.
            remove_pypart_suffix (bool): If True, strip the PyPart partitioning
                suffix (.P*.N*) from node paths before storing.

        Side Effects:
            - Populates self.empty_nodes_by_path with discovered empty nodes
            - If remove=True, modifies the input tree by removing empty nodes
            - Synchronizes all MPI ranks with allgather operation

        Note:
            The GridLocation node added by PyPart is removed from empty nodes
            before storage, as it's not part of the original node structure.
        """
        import copy
    

        current_rank = comm.Get_rank()

        # Scan all FlowSolution nodes in the tree
        for FS in tree.group(Type='FlowSolution'):
            
            if self._is_empty_flow_solution(FS):
                # Remove PyPart-added GridLocation if present
                # if no_DataArray_nodes:
                FS.findAndRemoveNode(Type='GridLocation')

                # Store a deep copy of the node
                node = copy.deepcopy(FS)
                self._add_grid_location(node)

                # Normalize the path by removing PyPart suffix if requested
                path = node.path()
                if remove_pypart_suffix:
                    path = self._remove_PyPart_suffix(path)

                self.empty_nodes_by_path[path] = (current_rank, node)

                # Optionally remove from the tree
                if remove:
                    FS.remove()

        # Distribute path information to all MPI ranks
        empty_nodes_paths = list(self.empty_nodes_by_path)
        empty_nodes_ranks_and_paths = comm.allgather((current_rank, empty_nodes_paths))

        # Populate entries for paths found on other ranks
        for item in empty_nodes_ranks_and_paths:
            rank, paths = item[0], item[1]
            for path in paths:
                if path not in self.empty_nodes_by_path:
                    # Store path with rank info but no node data (not on this rank)
                    self.empty_nodes_by_path[path] = (rank, None)

        # Sort to ensure consistent ordering across all ranks
        self.empty_nodes_by_path = dict(sorted(self.empty_nodes_by_path.items()))

        self._ensure_unique_rank_for_empty_nodes()

    def print_empty_nodes_by_path(self):
        """Print the current state of tracked empty nodes (debug utility).

        This method is intended for debugging purposes and prints the
        empty_nodes_by_path dictionary in a formatted way.
        """
        from pprint import pprint
        pprint(self.empty_nodes_by_path)

    def _ensure_unique_rank_for_empty_nodes(self):
        """Determine canonical owner rank for each empty node path.

        When a node path appears on multiple ranks, this method ensures that
        the lowest-ranked process is designated as the canonical owner. This
        is important for avoiding duplicate operations and ensuring consistent
        behavior across MPI processes.

        Side Effects:
            Updates self.empty_nodes_by_path to set node to None on non-owner ranks.
        """
    

        current_rank = comm.Get_rank()

        # Collect all ranks that have each path
        ranks_by_path = {}
        for path, value in self.empty_nodes_by_path.items():
            rank = value[0]
            if path in ranks_by_path:
                ranks_by_path[path].append(rank)
            else:
                ranks_by_path[path] = [rank]

        # Designate lowest rank as owner, set others to None
        for path in ranks_by_path:
            sorted_ranks = sorted(ranks_by_path[path])
            first_rank_with_path = sorted_ranks[0]
            if current_rank != first_rank_with_path:
                self.empty_nodes_by_path[path] = (first_rank_with_path, None)

    def restore_empty_FlowSolution_nodes(self, tree: cgns.Tree):
        """Restore empty FlowSolution nodes to their original zones in a tree.

        This method restores the stored empty nodes back into the CGNS tree.
        It handles MPI communication to transfer nodes between ranks when the
        source rank (where the node was stored) differs from the target rank
        (where the parent zone exists).

        Args:
            tree (treelab.cgns.Tree): The CGNS tree to restore nodes into.

        Note:
            This method performs MPI send/recv operations for nodes that need
            to be transferred between ranks. The MPI_TAG_EMPTY_NODE_RESTORE
            constant is used for communication tagging.
        """
    
        current_rank = comm.Get_rank()

        def _get_parent_from_path(tree, path):
            parent_zone_path = '/'.join(path.split('/')[:-1])
            return tree.getAtPath(parent_zone_path)

        # Step 1: Collect local target information (path -> (source_rank, parent, target_rank))
        local_target_info = []
        for path, value in self.empty_nodes_by_path.items():
            rank_with_source_zone, node = value[0], value[1] 
            parent_zone = _get_parent_from_path(tree, path)
            # rank_with_target_zone = current_rank if parent is not None else -1
            zone_proc = lambda zone: zone.get(Name='.Solver#Param', Depth=1).get(Name='proc').value()
            zone_loaded_on_current_rank = parent_zone is not None \
                and zone_proc(parent_zone) == current_rank
            rank_with_target_zone = current_rank if zone_loaded_on_current_rank else -1
            # print(f"[{current_rank}] STEP1 {path=} {rank_with_target_zone=}")
            local_target_info.append({
                'path': path,
                'rank_with_source_zone': rank_with_source_zone,
                'rank_with_target_zone': rank_with_target_zone,
            })

        # Step 2: Gather all target info from all ranks to determine where each zone exists
        # We use allgather to collect (rank, local_info_list) from all ranks
        comm.barrier()
        all_local_info = comm.allgather((current_rank, local_target_info))
        comm.barrier()

        # Step 3: Build a global map: path -> (source_rank, target_rank, parent_node)
        global_path_info = {}
        for rank, local_info in all_local_info:
            for info in local_info:
                path = info['path']
                # Update if we found the target zone on this rank
                if info['rank_with_target_zone'] != -1:
                    if path not in global_path_info:
                        global_path_info[path] = {
                            'rank_with_source_zone': info['rank_with_source_zone'],
                            'ranks_with_target_zone': [info['rank_with_target_zone']],
                        }
                    else:
                        global_path_info[path]['ranks_with_target_zone'].append(info['rank_with_target_zone'])

        # Step 4: keep only the lowest (canonical) rank for the target zone
        for path in global_path_info:
            ranks_with_target_zone = global_path_info[path].pop('ranks_with_target_zone')
            lowest_rank_with_target_zone = sorted(ranks_with_target_zone)[0]
            global_path_info[path]['rank_with_target_zone'] = lowest_rank_with_target_zone
            # print(f"[{current_rank}] STEP4 {path=} {ranks_with_target_zone=} {lowest_rank_with_target_zone=}")


        # Step 5: now that there is only a unique rank_with_source_zone (the lowest one)
        # and a unique rank_with_target_zone (also the lowest one), loop on the path and
        # exchange only once the empty nodes to put them back in the tree
        comm.barrier()
        for path in global_path_info:
            rank_with_source_zone = global_path_info[path]['rank_with_source_zone']
            rank_with_target_zone = global_path_info[path]['rank_with_target_zone']
            # print(f"[{current_rank}] STEP5 {path=} {rank_with_source_zone=} {rank_with_target_zone=}")
            # comm.barrier()
            
            if rank_with_source_zone == rank_with_target_zone:
                # no MPI exchange to do
                # Only the given rank must make the addChild operation
                if current_rank == rank_with_source_zone:
                    node = self.empty_nodes_by_path[path][1]
                    parent = _get_parent_from_path(tree, path)
                    parent.addChild(node)
                else:
                    # nothing to do for this path
                    pass
            else:
                if current_rank == rank_with_source_zone:
                    node = self.empty_nodes_by_path[path][1]
                    comm.send(node, dest=rank_with_target_zone, tag=46)
                if current_rank == rank_with_target_zone:    
                    node = comm.recv(source=rank_with_source_zone, tag=46) 
                    parent = _get_parent_from_path(tree, path)
                    parent.addChild(node)
            comm.barrier()

    def restore_empty_FlowSolution_nodes_in_file(self, dst):
        """Restore empty FlowSolution nodes directly to a CGNS file.

        This method writes the stored empty nodes directly to a CGNS file
        without requiring them to be in a tree structure first. It compares
        the stored nodes with existing nodes in the file and only writes
        nodes that have more DataArray children than the file version.

        Args:
            dst (str): Path to the destination CGNS file.

        """
    
        current_rank = comm.Get_rank()

        for path, value in self.empty_nodes_by_path.items():
            rank_with_source_node, fs_node = value[0], value[1]

            # Only process if we have the actual node data
            if fs_node is not None:
                # Read existing node from file
                print(f"[{current_rank}] {path=}")
                saved_FS = cgns.readNode(dst, path)

                # Write if stored node has more DataArrays than file version
                if len(saved_FS.group(Type='DataArray')) < len(fs_node.group(Type='DataArray')):
                    fs_node.saveThisNodeOnly(dst)
                    for child in fs_node.children():
                        child.saveThisNodeOnly(dst)


def is_partitioned(tree, backend='maia'):
    if backend == 'maia':
        from mola.cfd.preprocess.mesh import maia_wrapper
        return maia_wrapper.is_partitioned(tree)
    
def tree_has_ngon(tree : cgns.Tree):
    return any([elt_type in ['NGON_n', 'NFACE_n'] for elt_type in tree.getElementsTypes()])

def tree_has_not_any_ngon(tree : cgns.Tree):
    return any([elt_type not in ['NGON_n', 'NFACE_n'] for elt_type in tree.getElementsTypes()])

def tree_has_all_ngon(tree : cgns.Tree):
    return all([elt_type in ['NGON_n', 'NFACE_n'] for elt_type in tree.getElementsTypes()])