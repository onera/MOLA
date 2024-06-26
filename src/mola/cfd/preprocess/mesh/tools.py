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
    t = cgns.castNode(t)


    for zone in t.zones():
        zone.setParameters('.Solver#Param', proc=int(MPI.COMM_WORLD.Get_rank()))
        if zone.isStructured(): 
            reshape_DataArray(zone)
        
    t = cgns.castNode(t)
    return t

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

def to_distributed(tree : cgns.Tree):
    from mpi4py import MPI
    import maia
    t = maia.factory.recover_dist_tree(tree, MPI.COMM_WORLD)
    t = cgns.castNode(t)

    return t
