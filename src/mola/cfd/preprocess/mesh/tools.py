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
from mola.logging import mola_logger, MolaException
from treelab import cgns

def parametrize_with_height(tree, hub_families, shroud_families, GridLocation='Vertex'):
    from mpi4py import MPI
    import maia.pytree as PT
    from maia.algo.part.wall_distance import compute_projection_to

    mola_logger.info('Parametrize domain with channel height (add FlowSolution#Height)')

    tree = to_partitioned(tree) 

    hub_bc_predicate = lambda n : any([PT.predicate.belongs_to_family(n, wall_bc_family) for wall_bc_family in hub_families])
    shroud_bc_predicate = lambda n : any([PT.predicate.belongs_to_family(n, wall_bc_family) for wall_bc_family in shroud_families])

    if len(PT.get_nodes_from_predicate(tree, hub_bc_predicate)) == 0:
        raise MolaException(f'Cannot find hub families in tree from names {hub_families}')
    if len(PT.get_nodes_from_predicate(tree, shroud_bc_predicate)) == 0:
        raise MolaException(f'Cannot find shroud families in tree from names {shroud_families}')
    
    # Compute distances to hub and shroud
    compute_projection_to(tree, hub_bc_predicate, MPI.COMM_WORLD, out_fs_name='DistanceToHub', point_cloud=GridLocation)
    compute_projection_to(tree, shroud_bc_predicate, MPI.COMM_WORLD, out_fs_name='DistanceToShroud', point_cloud=GridLocation)

    # Compute ChannelHeight
    for zone in PT.get_all_Zone_t(tree):
        d1 = PT.get_value(PT.get_node_from_path(zone, 'DistanceToHub/Distance'))
        d2 = PT.get_value(PT.get_node_from_path(zone, 'DistanceToShroud/Distance'))
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
    import Post.PyTree as P

    zones = C.extractBCOfName(tree, f'FamilySpecified:{Family}')
    SurfaceTree = C.convertArray2Tetra(zones)
    SurfaceTree = C.initVars(SurfaceTree, 'ones=1')
    Surface = P.integ(SurfaceTree, var='ones')[0]        # Compute normalization coefficient
    mola_logger.debug(f'Surface of family {Family} = {Surface} m^2')

    return Surface

def to_distributed(tree : cgns.Tree):
    from mpi4py import MPI
    import maia

    if bool(tree.get(':CGNS#Distribution')): 
        t = tree
    
    elif bool(tree.get(':CGNS#GlobalNumbering')):
        t = maia.factory.recover_dist_tree(tree, MPI.COMM_WORLD)
        t = cgns.castNode(t)
        
    else:
        t = maia.factory.full_to_dist_tree(tree, MPI.COMM_WORLD)
        t = cgns.castNode(t)
     
    return t

def to_partitioned(tree : cgns.Tree):
    from mpi4py import MPI
    import maia
    
    is_dist = bool(tree.get(':CGNS#Distribution'))
    is_part = bool(tree.get(':CGNS#GlobalNumbering')) \
        or bool(tree.get(Type='Zone', Depth=2).getAtPath('.Solver#Param/proc'))

    if is_part:
        return tree
    else:
        _apply_all_maia_check(tree)
        if not is_dist:
            tree = maia.factory.full_to_dist_tree(tree, MPI.COMM_WORLD)
        tree = cgns.castNode(tree)
        return to_partitioned_if_distributed(tree)

def to_partitioned_if_distributed(tree : cgns.Tree):
    is_dist = bool(tree.get(':CGNS#Distribution'))
    if not is_dist: return tree

    from mpi4py import MPI
    import maia

    t = maia.factory.partition_dist_tree(tree, MPI.COMM_WORLD, data_transfer='ALL')

    # fix_FaceCenter_in_BCDataSet(t)
    t = cgns.castNode(t)
    for zone in t.zones():
        zone.setParameters('.Solver#Param', proc=int(MPI.COMM_WORLD.Get_rank()))
        if zone.isStructured(): 
            reshape_DataArray(zone)
        
    return t

def to_full_tree_at_rank_0(tree : cgns.Tree):
    from mpi4py import MPI
    import maia
    MPI.COMM_WORLD.barrier()
    
    is_dist = bool(tree.get(':CGNS#Distribution'))
    is_part = bool(tree.get(':CGNS#GlobalNumbering'))
                   
    if not is_dist and not is_part:
        return tree
    
    if is_part:
        additionnal_nodes_to_transfer = _get_additionnal_nodes_to_transfer(tree)
        tree = maia.factory.recover_dist_tree(tree, MPI.COMM_WORLD, data_transfer='ALL')
        tree = cgns.castNode(tree)
        _transfer_additionnal_nodes(tree, additionnal_nodes_to_transfer)

    empty_FlowSolution_nodes = get_empty_FlowSolution_nodes(tree, remove=True)
    tree = maia.factory.dist_to_full_tree(tree, MPI.COMM_WORLD, target=0)
    tree = cgns.castNode(tree)
    restore_empty_FlowSolution_nodes(tree, empty_FlowSolution_nodes)
    
    if tree is not None:
        tree = cgns.castNode(tree)

        for zone in tree.zones():
            if zone.isStructured(): 
                reshape_DataArray(zone)
            
        tree = cgns.castNode(tree)
    MPI.COMM_WORLD.barrier()
    return tree

def _get_additionnal_nodes_to_transfer(tree):
    # HACK see https://gitlab.onera.net/numerics/mesh/maia/-/issues/175
    types = ['ReferenceState_t', 'FlowEquationSet_t', 'Descriptor_t', 'ConvergenceHistory_t', 'IntegralData_t']
    nodes = []
    for node_type in types:
        nodes.extend(tree.group(Type=node_type, Depth=2))  # workaround only for bases children, not zones children, just for safety
    return nodes

def _transfer_additionnal_nodes(tree, nodes):
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

def ravel_BCDataSet(t):
    # HACK https://elsa.onera.fr/issues/11219
    # HACK https://elsa-e.onera.fr/issues/10750
    for bcd in t.group(Type='BCData'):
        for da in bcd.group(Type='DataArray'):
            value = da.value()
            if value is not None:
                da.setValue(value.ravel(order='K'))

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

def fix_FaceCenter_in_BCDataSet(t):
    import maia.pytree as PT

    for zone in PT.get_all_Zone_t(t):
        if PT.get_value(PT.get_node_from_label(zone, 'ZoneType_t')) == 'Structured':
            for node in PT.get_nodes_from_label(zone, 'BCDataSet_t'):
                if PT.Subset.GridLocation(node) == 'FaceCenter':
                    axis = PT.Subset.normal_axis(node)
                    PT.update_child(node, 'GridLocation', value='IJK'[axis] + 'FaceCenter')

def get_empty_FlowSolution_nodes(tree, remove=False):
    # NOTE Cmpi.convertPyTree2File does not write DataArray in FlowSolution
    # if its value is None on all ranks, but this is a way for elsA to 
    # ask extraction in a FlowSolution (for 3D fields)
    # -> keep these nodes in a list

    # NOTE With unstructured mesh, PyPart does not support empty FlowSolution 
    # (with DataArray nodes that store None). We need to remove these nodes, 
    # keep its path, and restore them later in the final file.

    import Converter.Mpi as Cmpi
    import copy

    empty_FlowSolution_nodes = []
    for FS in tree.group(Type='FlowSolution'):
        no_DataArray_nodes = len(FS.group(Type='DataArray', Depth=1)) == 0
        empty_DataArray_nodes = any([n.value() is None for n in FS.group(Type='DataArray', Depth=1)])
        if no_DataArray_nodes or empty_DataArray_nodes:
            if no_DataArray_nodes:
                # The node GridLocation has been added by PyPart during the splitting, we need to remove it.
                FS.findAndRemoveNode(Type='GridLocation')
            empty_FlowSolution_nodes.append(copy.deepcopy(FS))
            if remove:
                FS.remove()

    return empty_FlowSolution_nodes

def restore_empty_FlowSolution_nodes_in_file(dst, empty_FlowSolution_nodes):
    import Converter.Mpi as Cmpi

    empty_FlowSolution_nodes = Cmpi.gather(empty_FlowSolution_nodes, root=0)

    if Cmpi.rank == 0:
        empty_FlowSolution_nodes = [FS for listFS in empty_FlowSolution_nodes for FS in listFS]
        for FS in empty_FlowSolution_nodes:
            saved_FS = cgns.readNode(dst, FS.path()) 
            if len(saved_FS.group(Type='DataArray')) < len(FS.group(Type='DataArray')):
                FS.saveThisNodeOnly(dst) 
                for child in FS.children():
                    child.saveThisNodeOnly(dst) 

def restore_empty_FlowSolution_nodes(tree, empty_FlowSolution_nodes):
    for fs_node in empty_FlowSolution_nodes:
        zone_path = fs_node.parent().path()
        zone_path = _remove_PyPart_suffix(zone_path)
        parent = tree.getAtPath(zone_path) 
        parent.addChild(fs_node)

def _remove_PyPart_suffix(path):
    import re
    # regular expression to find a pattern ".P*.N*", with * a number with 1 to 5 figures
    pattern = r'\.P(\d{1,5})\.N(\d{1,5})'
    try:
        path = re.sub(pattern, '', path)
    except:
        pass
    return path
