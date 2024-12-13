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
from mola.logging import MolaException
from .utils import get_io_tool

def is_using_mpi():
    try:
        import mpi4py.MPI as MPI
        comm = MPI.COMM_WORLD 
        return comm.Get_size() > 1
    except:
        return False
    

def read(w, src, io_tool=None):

    if not isinstance(src,str): return src

    if io_tool is None:
        io_tool = get_io_tool(w, src)

    if io_tool == 'treelab':
        mesh = cgns.load(src)

    elif io_tool == 'cassiopee':
        import Converter.PyTree as C
        links = []
        mesh = C.convertFile2PyTree(src, links=links)
        mesh = cgns.castNode(mesh)
        for link in links:
            mesh.addLink(path=link[3], target_file=link[1], target_path=link[2])


    elif io_tool == 'cassiopee_mpi':
        import Converter.Mpi as Cmpi
        import Distributor2.PyTree as Distributor2
        Cmpi.barrier()
        links = []
        mesh = Cmpi.convertFile2SkeletonTree(src, links=links)
        mesh, _ = Distributor2.distribute(mesh, NProc=Cmpi.size, algorithm='fast')
        Cmpi._readZones(mesh, src, rank=Cmpi.rank)
        for link in links:
            mesh.addLink(path=link[3], target_file=link[1], target_path=link[2])
        mesh = cgns.castNode(mesh)
        Cmpi.barrier()

    elif io_tool == 'maia':
        # Nodes DataArray with value=None are not read, see https://gitlab.onera.net/numerics/mesh/maia/-/issues/164
        from mpi4py import MPI
        import maia
        MPI.COMM_WORLD.barrier()
        mesh = maia.io.file_to_dist_tree(src, MPI.COMM_WORLD)
        mesh = cgns.castNode(mesh)
        MPI.COMM_WORLD.barrier()

    else:
        raise MolaException(f'unknown value for io_tool: {io_tool}')

    return mesh

