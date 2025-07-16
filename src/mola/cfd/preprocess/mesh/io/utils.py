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

from treelab import cgns
from mola.logging import MolaUserError

def is_using_mpi():
    try:
        import mpi4py.MPI as MPI
        comm = MPI.COMM_WORLD 
        return comm.Get_size() > 1
    except:
        return False
    

def get_io_tool(w, src):
    using_mpi = is_using_mpi()
    is_cgns = src.endswith('.cgns') or src.endswith('.hdf') or src.endswith('.hdf5')

    if not using_mpi:
        io_tool = 'treelab' if is_cgns else 'cassiopee'

    else:
        if not is_cgns: 
            return MolaUserError('parallel file load/write requires mesh in cgns format')
        io_tool = 'maia' if w.Solver != 'fast' else 'cassiopee_mpi'

    return io_tool

def get_full_tree_skeleton_from_partitioned_tree(tree : cgns.Tree):
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    import maia.pytree as PT

    shallow = PT.shallow_copy(tree)
    WorkflowParametersNode = tree.get(Name='WorkflowParameters',Depth=1)

    class data_eraser:
        def pre(self, node):
            # This function will be applied to each node in the tree. You can customize
            # the settings here
            if PT.get_value_type(node) not in ["MT", "C1"]: # Skip empty nodes and strings
                if PT.get_value(node).size > 10:
                    PT.set_value(node, None)

    PT.graph.cgns.depth_first_search(shallow, data_eraser())

    # Second part -> merge into one tree
    from maia.factory.dist_from_part import discover_nodes_from_matching

    skeleton = PT.new_CGNSTree()

    discover_nodes_from_matching(skeleton, 
                                [shallow], 
                                'CGNSBase_t',         # Find nodes having this pattern in shallow
                                comm, 
                                child_list=[lambda n: PT.get_label(n) != 'Zone_t']) # When a node is find, keep all children

    discover_nodes_from_matching(skeleton, 
                                [shallow], 
                                'CGNSBase_t/Zone_t',         # Find nodes having this pattern in shallow
                                comm, 
                                child_list=[lambda n: True], # When a node is find, keep all children
                                get_value='all')             # Do not clear data for Base & Zone


    skeleton = cgns.castNode(skeleton)
    if WorkflowParametersNode: skeleton.addChild(WorkflowParametersNode)

    return skeleton