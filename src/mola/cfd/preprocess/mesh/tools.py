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

def to_partitioned_if_distributed(tree : cgns.Tree):
    is_dist = bool(tree.get(':CGNS#Distribution'))
    if not is_dist: return tree

    from mpi4py import MPI
    import maia
    t = maia.factory.partition_dist_tree(tree, MPI.COMM_WORLD)
    copyRelevantUserDefinedDataNodes(tree, t, MPI.COMM_WORLD)
    t = cgns.castNode(t)


    for zone in t.zones():
        zone.setParameters('.Solver#Param', proc=int(MPI.COMM_WORLD.Get_rank()))
        if zone.isStructured(): 
            reshape_DataArray(zone)
        
    t = cgns.castNode(t)
    return t

def copyRelevantUserDefinedDataNodes(dist_tree, part_tree, comm):
    # TODO should be deprecated, it is possible to do instead : 
    #     maia.factory.partition_dist_tree(tree, MPI.COMM_WORLD, data_transfer='ALL')
    # on the last version of Maia. To test and check availability with all env 
    
    from packaging.version import Version
    import maia

    maia_version = maia.__version__
    if maia_version.startswith("dev-"):
        maia_version = maia_version.replace('dev-','')+'dev'
    if Version( maia_version ) < Version("1.5"): return

    import maia.pytree as PT
    match_name = lambda name: name == 'WorkflowParameters' or \
                              name.startswith('.Solver#') or \
                              name.startswith('.MOLA')

    to_copy = lambda n : PT.get_label(n) == 'UserDefinedData_t' and match_name(PT.get_name(n))

    maia.transfer.dist_tree_to_part_tree_copy(dist_tree, part_tree, [to_copy], comm)
    maia.transfer.dist_tree_to_part_tree_copy(dist_tree, part_tree, ['CGNSBase_t', to_copy], comm)
    maia.transfer.dist_tree_to_part_tree_copy(dist_tree, part_tree, ['CGNSBase_t', 'Zone_t', to_copy], comm)
    maia.transfer.dist_tree_to_part_tree_copy(dist_tree, part_tree, ['CGNSBase_t', 'Family_t', to_copy], comm)
    maia.transfer.dist_tree_to_part_tree_copy(dist_tree, part_tree, ['CGNSBase_t', 'Zone_t', 'ZoneBC_t', 'BC_t', to_copy], comm)





def to_full_tree_at_rank_0(tree : cgns.Tree):
    is_dist = bool(tree.get(':CGNS#Distribution'))
    if not is_dist: raise MolaException('expected distributed tree')

    from mpi4py import MPI
    import maia
    MPI.COMM_WORLD.barrier()
    t = maia.factory.dist_to_full_tree(tree, MPI.COMM_WORLD, target=0)
    if t is not None:
        t = cgns.castNode(t)

        for zone in t.zones():
            if zone.isStructured(): 
                reshape_DataArray(zone)
            
        t = cgns.castNode(t)
    MPI.COMM_WORLD.barrier()
    return t

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
