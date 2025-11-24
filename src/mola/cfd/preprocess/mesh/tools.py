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

@parallel_execution_with_maia()
def parametrize_with_height(tree, hub_families, shroud_families, GridLocation='Vertex'):
    from mpi4py import MPI
    import maia
    import maia.pytree as PT
    from maia.algo.part.wall_distance import compute_projection_to

    mola_logger.info('Parametrize domain with channel height (add FlowSolution#Height)', rank=0)

    tree = to_partitioned(tree) 

    if Version(maia.__version__) > Version('1.7'):
        # Change of name of the module "predicate" in "pred"
        hub_bc_predicate    = PT.pred.any([PT.pred.belongs_to_family(family) for family in hub_families])
        shroud_bc_predicate = PT.pred.any([PT.pred.belongs_to_family(family) for family in shroud_families])
    else:
        hub_bc_predicate = lambda n : any([PT.predicate.belongs_to_family(n, wall_bc_family) for wall_bc_family in hub_families])
        shroud_bc_predicate = lambda n : any([PT.predicate.belongs_to_family(n, wall_bc_family) for wall_bc_family in shroud_families])

    hub_was_not_found = MPI.COMM_WORLD.reduce(len(PT.get_nodes_from_predicate(tree, hub_bc_predicate)) == 0, op=MPI.LAND)
    if hub_was_not_found:
        raise MolaException(f'Cannot find hub families in tree from names {hub_families}')
    shroud_was_not_found = MPI.COMM_WORLD.reduce(len(PT.get_nodes_from_predicate(tree, shroud_bc_predicate)) == 0, op=MPI.LAND)
    if shroud_was_not_found:
        raise MolaException(f'Cannot find shroud families in tree from names {shroud_families}')
    
    # TODO make this operation separately for each row family to prevent errors
    # and to reduce the duration of the process
    # Compute distances to hub and shroud
    compute_projection_to(tree, hub_bc_predicate, MPI.COMM_WORLD, out_fs_name='DistanceToHub', point_cloud=GridLocation)
    compute_projection_to(tree, shroud_bc_predicate, MPI.COMM_WORLD, out_fs_name='DistanceToShroud', point_cloud=GridLocation)

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
    from mpi4py import MPI

    tree = I.fixNGon(tree)  # it would be better without making a copy, but it would need adaptation latter for maia
    zones = C.extractBCOfName(tree, f'FamilySpecified:{Family}')
    SurfaceTree = C.convertArray2Tetra(zones)
    SurfaceTree = C.initVars(SurfaceTree, 'ones=1')
    Surface = P.integ(SurfaceTree, var='ones')[0]        # Compute normalization coefficient
    Surface = MPI.COMM_WORLD.allreduce(Surface, op=MPI.SUM)
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


def to_distributed(tree : cgns.Tree):
    from mpi4py import MPI
    import maia

    if bool(tree.get(':CGNS#Distribution')): 
        t = tree
    
    else:
        mola_logger.debug('convert tree to maia dist_tree')

        if bool(tree.get(':CGNS#GlobalNumbering')):
            t = maia.factory.recover_dist_tree(tree, MPI.COMM_WORLD)
            t = cgns.castNode(t)
            
        else:
            t = maia.factory.full_to_dist_tree(tree, MPI.COMM_WORLD)
            t = cgns.castNode(t)
     
    return t

def to_partitioned(tree : cgns.Tree):
    from mpi4py import MPI
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
            tree = maia.factory.full_to_dist_tree(tree, MPI.COMM_WORLD)
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

    from mpi4py import MPI
    import maia

    mola_logger.debug('convert tree to maia part_tree')

    t = maia.factory.partition_dist_tree(tree, MPI.COMM_WORLD, data_transfer='ALL')

    t = cgns.castNode(t)
    for base in t.bases():
        for zone in base.zones():
            rank = int(MPI.COMM_WORLD.Get_rank())
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
    from mpi4py import MPI
    import maia
    MPI.COMM_WORLD.barrier()
    
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
                   
    if is_full or MPI.COMM_WORLD.Get_size() == 1:
        return tree
    else:
        mola_logger.debug('convert tree to maia full_tree')
    
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

def get_empty_FlowSolution_nodes(tree, remove=False):
    # NOTE Cmpi.convertPyTree2File does not write DataArray in FlowSolution
    # if its value is None on all ranks, but this is a way for elsA to 
    # ask extraction in a FlowSolution (for 3D fields)
    # -> keep these nodes in a list

    # NOTE With unstructured mesh, PyPart does not support empty FlowSolution 
    # (with DataArray nodes that store None). We need to remove these nodes, 
    # keep its path, and restore them later in the final file.

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
    for FS in empty_FlowSolution_nodes:
        saved_FS = cgns.readNode(dst, FS.path()) 
        if len(saved_FS.group(Type='DataArray')) < len(FS.group(Type='DataArray')):
            FS.saveThisNodeOnly(dst) 
            for child in FS.children():
                child.saveThisNodeOnly(dst) 

def restore_empty_FlowSolution_nodes(tree, empty_FlowSolution_nodes):
    for fs_node in empty_FlowSolution_nodes:
        _add_grid_location(fs_node)
        zone_path = fs_node.parent().path()
        zone_path = _remove_PyPart_suffix(zone_path)
        parent = tree.getAtPath(zone_path) 
        parent.addChild(fs_node)

def _add_grid_location(fs_node: cgns.Node):
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


def _remove_PyPart_suffix(path):
    import re
    # regular expression to find a pattern ".P*.N*", with * a number with 1 to 5 figures
    pattern = r'\.P(\d{1,5})\.N(\d{1,5})'
    try:
        path = re.sub(pattern, '', path)
    except:
        pass
    return path

def is_partitioned(tree, backend='maia'):
    if backend == 'maia':
        from mola.cfd.preprocess.mesh import maia_wrapper
        return maia_wrapper.is_partitioned(tree)