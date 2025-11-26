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

import os
import glob
import numpy as np
from .utils import get_io_tool
from ..tools import (to_full_tree_at_rank_0, get_empty_FlowSolution_nodes, 
                     restore_empty_FlowSolution_nodes_in_file, restore_empty_FlowSolution_nodes)
from treelab import cgns
import mola.naming_conventions as names

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
nb_digit = int(np.ceil(np.log10(size+1)))

def write(w, tree, dst, io_tool=None):

    if io_tool is None:
        io_tool = get_io_tool(w, dst)

    write_with_selected_tool = dict(
        treelab = write_with_treelab,
        cassiopee = write_with_cassiopee,
        cassiopee_mpi = write_with_cassiopee_mpi,
        maia = write_with_maia,
        pypart = write_with_pypart,
    )

    write_with_selected_tool[io_tool](w, tree, dst)

def write_with_treelab(w, tree, dst):
    t = tree.copy()
    t = to_full_tree_at_rank_0(t)
    cgns.save(t, dst)

def write_with_cassiopee(w, tree, dst):
    import Converter.PyTree as C

    file_path = dst.split('.')
    start_path = '.'.join(file_path[:-1])
    fmt = file_path[-1]
    
    if size > 1:        
        if _shall_write_one_file_per_proc(tree):        
            try: os.makedirs(start_path)
            except: pass
            dst = os.path.join(start_path, ('rank_{:0%d}.'%nb_digit).format(rank) + fmt)

    links = tree.getLinks()
    for l in links: l[0] = '.' # HACK treelab 0.1.1
    C.convertPyTree2File(tree, dst, links=links)


def _shall_write_one_file_per_proc(tree):
    comm.barrier()
    has_tree = comm.gather(bool(tree))
    at_least_one_rank_except_root_has_a_tree = None
    if rank == 0:
        at_least_one_rank_except_root_has_a_tree = any(has_tree[1:])
    comm.barrier()
    at_least_one_rank_except_root_has_a_tree = comm.bcast(at_least_one_rank_except_root_has_a_tree)

    return at_least_one_rank_except_root_has_a_tree


def write_with_cassiopee_mpi(w, tree, dst):
    import Converter.Mpi as Cmpi
    Cmpi.barrier()
    links = tree.getLinks()
    for l in links: l[0] = '.' # HACK treelab 0.1.1
    empty_FlowSolution_nodes = get_empty_FlowSolution_nodes(tree)

    Cmpi.barrier()    
    Cmpi.convertPyTree2File(tree,dst,links=links)
    Cmpi.barrier()
    restore_empty_FlowSolution_nodes_in_file(dst, empty_FlowSolution_nodes)        
    Cmpi.barrier()
    
def write_with_maia(w, tree, dst):
    from mpi4py import MPI
    import maia

    def get_links_for_maia(tree):
        links = tree.getLinks()
        for l in links:
            l[0] = '.' # HACK treelab 0.1.1
            del l[4]   # HACK maia only supports 4 elements
            # HACK maia requires no "/" root at CGNS links https://gitlab.onera.net/numerics/mesh/maia/-/issues/108#note_30623
            if l[3].startswith('/'): l[3] = l[3][1:]
        return links
    
    links = get_links_for_maia(tree)
    MPI.COMM_WORLD.barrier()
    if maia.pytree.get_node_from_name(tree, ':CGNS#GlobalNumbering') is not None:
        tree.removeEmptyZones()
        # maia.io.part_tree_to_file(tree, dst, MPI.COMM_WORLD, single_file=True, links=links)
        tree = maia.factory.recover_dist_tree(tree, MPI.COMM_WORLD, data_transfer='ALL')
        maia.io.dist_tree_to_file(tree, dst, MPI.COMM_WORLD, links=links)


    elif maia.pytree.get_node_from_name(tree, ':CGNS#Distribution') is not None:
        tree.removeEmptyZones()
        maia.io.dist_tree_to_file(tree, dst, MPI.COMM_WORLD, links=links)
    
    else:
        # The tree is nor partitioned neither distributed.
        # It is then considered as full on rank 0
        if MPI.COMM_WORLD.Get_rank() == 0:
            maia.io.write_tree(tree, dst, links=links)

    MPI.COMM_WORLD.barrier()

def write_with_pypart(w, tree, dst):
    import Converter.Mpi as Cmpi
    import Distributor2.PyTree as D2

    # HACK mergeAndSave bugs with empty FlowSolution nodes for unstructured mesh
    empty_FlowSolution_nodes = get_empty_FlowSolution_nodes(tree, remove=True)
    
    # NOTE Careful: For structured mesh, FlowSolution data must not be ravelized!!
    # Otherwise, data nodes will be full of zeros after mergeAndSave.
    from mola.cfd.postprocess.extractions_with_cassiopee.tools import reshapeFieldsForStructuredGrid
    reshapeFieldsForStructuredGrid(tree)

    # Write in parallel with PyPart
    Cmpi._convert2PartialTree(tree)
    Cmpi.barrier()
    w._PyPartBase.mergeAndSave(tree, os.path.join(names.DIRECTORY_OUTPUT, 'PyPart_fields'), cgns_standard=True)
    Cmpi.barrier()

    # Read PyPart files in parallel using Cassiopee
    # NOTE since mpi size may be > nb of zones, we have warnings (unnallocated zones)
    # but this is not an issue for the scope of this function
    t = Cmpi.convertFile2SkeletonTree(os.path.join(names.DIRECTORY_OUTPUT, 'PyPart_fields_all.hdf'))
    t, stats = D2.distribute(t, w.RunManagement['NumberOfProcessors'], useCom=0, algorithm='fast')
    t = Cmpi.readZones(t, os.path.join(names.DIRECTORY_OUTPUT, 'PyPart_fields_all.hdf'), rank=Cmpi.rank)
    t = cgns.castNode(t)
    Cmpi.barrier()

    if dst.endswith(names.FILE_INPUT_SOLVER):
        # Bug PyPart: mergeAndSave does not write WorkflowParameters, maybe because it is at the Base level
        # TODO open an issue
        t.findAndRemoveNode(Name=w._workflow_parameters_container_, Depth=1)
        params = w.convert_to_dict()
        t.setParameters(w._workflow_parameters_container_, **params)

    restore_empty_FlowSolution_nodes(t, empty_FlowSolution_nodes)
    t = _add_GridLocation_and_PointRange_in_BCDataSet(t)

    # Write a unique file
    Cmpi._convert2PartialTree(t)
    Cmpi.barrier()
    Cmpi.convertPyTree2File(t, dst)
    Cmpi.barrier()

    # Remove PyPart 
    if Cmpi.rank == 0:
        for fn in glob.glob(os.path.join(names.DIRECTORY_OUTPUT, 'PyPart_fields_*.hdf')):
            try: os.remove(fn)
            except: pass


def _add_GridLocation_and_PointRange_in_BCDataSet(tree):
    # HACK add GridLocation and PointRange or PointList nodes in each BCDataSet
    # It is needed for compatibility with maia, otherwise maia cannot read the mesh from file.
    # for BCDataSet in tree.group(Type='BCDataSet'):
    #     GridLocation = BCDataSet.get(Type='GridLocation')
    #     if GridLocation is None:
    #         cgns.Node(Name='GridLocation', Type='GridLocation', Value='FaceCenter', Parent=BCDataSet)
    from mola.cfd.preprocess.boundary_conditions.boundary_conditions import fix_FaceCenter_in_BCDataSet
    fix_FaceCenter_in_BCDataSet(tree)

    from maia.io.fix_tree import add_missing_pr_in_bcdataset
    add_missing_pr_in_bcdataset(tree)
    return cgns.castNode(tree)
