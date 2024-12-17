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
from .utils import get_io_tool
from treelab import cgns
import mola.naming_conventions as names

def write(w, tree, dst, io_tool=None):
    if tree.get(Name=':CGNS#Ppart', Depth=3):
        io_tool = 'pypart'

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
    t.findAndRemoveNodes(Name=':CGNS#Distribution')
    t.findAndRemoveNodes(Name=':CGNS#GlobalNumbering')
    cgns.save(t, dst)

def write_with_cassiopee(w, tree, dst):
    import Converter.PyTree as C
    links = tree.getLinks()
    for l in links: l[0] = '.' # HACK treelab 0.1.1
    C.convertPyTree2File(tree, dst, links=links)

def write_with_cassiopee_mpi(w, tree, dst):
    import Converter.Mpi as Cmpi
    Cmpi.barrier()
    links = tree.getLinks()
    for l in links: l[0] = '.' # HACK treelab 0.1.1
    empty_FlowSolution_nodes = get_empty_FlowSolution_nodes(tree)
    Cmpi.convertPyTree2File(tree,dst,links=links)
    Cmpi.barrier()
    restore_empty_FlowSolution_nodes(dst, empty_FlowSolution_nodes)        
    Cmpi.barrier()
    
def write_with_maia(w, tree, dst):
    from mpi4py import MPI
    import maia
    
    MPI.COMM_WORLD.barrier()
    if maia.pytree.get_node_from_name(tree, ':CGNS#Distribution') is not None:
        maia.io.dist_tree_to_file(tree, dst, MPI.COMM_WORLD)

    elif maia.pytree.get_node_from_name(tree, ':CGNS#GlobalNumbering') is not None:

        links = tree.getLinks()
        for l in links:
            l[0] = '.' # HACK treelab 0.1.1
            del l[4]   # HACK maia only supports 4 elements
            # HACK maia requires no "/" root at CGNS links https://gitlab.onera.net/numerics/mesh/maia/-/issues/108#note_30623
            if l[3].startswith('/'): l[3] = l[3][1:]
        for zone in tree.zones():
            if is_empty(zone):  # TODO transform this function into a Zone method in Treelab: zone.isEmpty()
                zone.remove()
        MPI.COMM_WORLD.barrier()
        # TODO this function does not save UserDefinedData_t nodes under bases
        # see https://gitlab.onera.net/numerics/mesh/maia/-/issues/112
        maia.io.part_tree_to_file(tree, dst, MPI.COMM_WORLD, links=links, single_file=True)

    else:
        dist_tree = maia.factory.full_to_dist_tree(tree, MPI.COMM_WORLD)
        maia.io.dist_tree_to_file(dist_tree, dst, MPI.COMM_WORLD)
    MPI.COMM_WORLD.barrier()

def write_with_pypart(w, tree, dst):
    import Converter.PyTree as C
    import Converter.Mpi as Cmpi
    # HACK mergeAndSave bugs with empty FlowSolution nodes for unstructured mesh
    empty_FlowSolution_nodes = get_empty_FlowSolution_nodes(tree, remove=True)
    Cmpi.barrier()
    w._PyPartBase.mergeAndSave(tree, os.path.join(names.DIRECTORY_OUTPUT, 'PyPart_fields'))
    Cmpi.barrier()
    if Cmpi.rank == 0:
        t_merged = C.convertFile2PyTree(os.path.join(names.DIRECTORY_OUTPUT, 'PyPart_fields_all.hdf'))
        t_merged = cgns.castNode(t_merged)

        if dst.endswith(names.FILE_INPUT_SOLVER):
            # Bug PyPart: mergeAndSave does not write WorkflowParameters, maybe because it is at the Base level
            # TODO open an issue
            t_merged.findAndRemoveNode(Name=w._workflow_parameters_container_, Depth=1)
            params = w.convert_to_dict()
            t_merged.setParameters(w._workflow_parameters_container_, **params)

        for FS in empty_FlowSolution_nodes:
            path = _remove_PyPart_suffix(FS.path())
            zone_path = '/'.join(path.split('/')[:-1])
            zone = t_merged.getAtPath(zone_path)
            zone.addChild(FS)

        t_merged.save(dst)
        for fn in glob.glob(os.path.join(names.DIRECTORY_OUTPUT, 'PyPart_fields_*.hdf')):
            try:
                os.remove(fn)
            except:
                pass
        
    Cmpi.barrier()

def is_empty(zone):
    GridCoordinates = zone.get(Type='GridCoordinates', Depth=1)
    if GridCoordinates is None:
        return True
    coord = GridCoordinates.get(Type='DataArray')
    if coord is None or coord.value() is None:
        return True
    
    return False

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
            if Cmpi.rank == 0:
                if no_DataArray_nodes:
                    FS.findAndRemoveNode(Type='GridLocation')
                empty_FlowSolution_nodes.append(copy.deepcopy(FS))
            if remove:
                FS.remove()
 
    return empty_FlowSolution_nodes

def restore_empty_FlowSolution_nodes(dst, empty_FlowSolution_nodes):
    import Converter.Mpi as Cmpi

    if Cmpi.rank == 0:
        for FS in empty_FlowSolution_nodes:
            saved_FS = cgns.readNode(dst, FS.path()) 
            if len(saved_FS.group(Type='DataArray')) < len(FS.group(Type='DataArray')):
                FS.saveThisNodeOnly(dst) 
                for child in FS.children():
                    child.saveThisNodeOnly(dst) 

def _remove_PyPart_suffix(path):
    path_split = path.split('/')
    zone_name = path_split[-2]
    if '.P0.N' in zone_name:
        new_zone_name = zone_name.split('.P0.N')[0]
        path = path.replace(zone_name, new_zone_name)
    return path
