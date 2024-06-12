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

from mola.logging import mola_logger, MolaUserError
from .utils import get_io_tool, get_full_tree_skeleton_from_partitioned_tree
from treelab import cgns

def write(w, tree, dst):
    io_tool = get_io_tool(w, dst)

    if io_tool == 'treelab':
        cgns.save(tree, dst)

    elif io_tool == 'cassiopee':
        import Converter.PyTree as C
        links = tree.getLinks()
        for l in links: l[0] = '.' # HACK
        C.convertPyTree2File(tree, dst, links=links)

    elif io_tool == 'cassiopee_mpi':
        import Converter.Mpi as Cmpi
        MPI.COMM_WORLD.barrier()
        links = tree.getLinks()
        for l in links: l[0] = '.' # HACK
        Cmpi.convertPyTree2File(tree,dst,links=links)
        MPI.COMM_WORLD.barrier()

    elif io_tool == 'maia':
        from mpi4py import MPI
        import maia
        
        MPI.COMM_WORLD.barrier()
        if maia.pytree.get_node_from_name(tree, ':CGNS#Distribution') is not None:
            maia.io.dist_tree_to_file(tree, dst, MPI.COMM_WORLD)

        elif maia.pytree.get_node_from_name(tree, ':CGNS#GlobalNumbering') is not None:

            # TODO https://gitlab.onera.net/numerics/mesh/maia/-/issues/108
            # maia.io.part_tree_to_file(tree, dst, MPI.COMM_WORLD, single_file=True)

            # temporary HACK: write with Cassiopee MPI
            import Converter.PyTree as C
            import Converter.Mpi as Cmpi
            import Converter.Internal as I
            links = tree.getLinks()
            for l in links: l[0] = '.' # HACK
            Cmpi.barrier()
            if Cmpi.rank == 0:
                mola_logger.info(dst)
            links_gathered = Cmpi.allgather(links)
            links = []
            for g_links in links_gathered: links.extend(g_links)
            if links : links = [links[3]]
            if Cmpi.rank == 0:
                for l in links: mola_logger.info(l)
            Cmpi.barrier()
            skel = get_full_tree_skeleton_from_partitioned_tree(tree)
            tree_w_skel = I.merge([skel,tree])
            # if Cmpi.rank > 0: links = []
            Cmpi.barrier()
            Cmpi.convertPyTree2File(tree_w_skel,dst, links=links)

        else:
            dist_tree = maia.factory.full_to_dist_tree(tree, MPI.COMM_WORLD)
            maia.io.dist_tree_to_file(dist_tree, dst, MPI.COMM_WORLD)
        MPI.COMM_WORLD.barrier()
