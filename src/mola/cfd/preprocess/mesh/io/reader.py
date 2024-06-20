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
from .utils import get_io_tool

def is_using_mpi():
    try:
        import mpi4py.MPI as MPI
        comm = MPI.COMM_WORLD 
        return comm.Get_size() > 1
    except:
        return False
    

def read(w, src):

    if not isinstance(src,str): return src

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
        w._Skeleton = cgns.castNode(mesh).copy()
        mesh, _ = Distributor2.distribute(mesh, NProc=Cmpi.size, algorithm='fast')
        Cmpi._readZones(mesh, src, rank=Cmpi.rank)
        for link in links:
            mesh.addLink(path=link[3], target_file=link[1], target_path=link[2])
        mesh = cgns.castNode(mesh)
        add_coordinates_in_skeleton(w._Skeleton, mesh)
        w._Skeleton = cgns.castNode(w._Skeleton)
        Cmpi.barrier()

    elif io_tool == 'maia':
        from mpi4py import MPI
        import maia
        MPI.COMM_WORLD.barrier()
        mesh = maia.io.file_to_dist_tree(src, MPI.COMM_WORLD)
        mesh = cgns.castNode(mesh)
        MPI.COMM_WORLD.barrier()

    elif io_tool == 'pypart':
        from . import pypart
        part_tree, skeleton, PyPartBase = pypart.read_with_pypart(src)
        mesh = cgns.castNode(part_tree)
        add_coordinates_in_skeleton(skeleton, mesh)
        w._Skeleton = cgns.castNode(skeleton)
        w._PyPartBase = PyPartBase
        # mesh = pypart.pypart_to_maia(part_tree, skeleton) # only possible for unstructured mesh

    mesh = cgns.merge(mesh)
    return mesh

def add_coordinates_in_skeleton(Skeleton, PartTree):
    import Converter.Internal as I
    import Converter.Mpi as Cmpi

    I._rmNodesByName(Skeleton, 'FlowSolution*')

    # Needed nodes are read from PartTree
    def readNodesFromPaths(path):
        split_path = path.split('/')
        path_begining = '/'.join(split_path[:-1])
        name = split_path[-1]
        parent = I.getNodeFromPath(PartTree, path_begining)
        return I.getNodesFromName(parent, name)
        
    def replaceNodeByName(parent, parentPath, name):
        oldNode = I.getNodeFromName1(parent, name)
        path = '{}/{}'.format(parentPath, name)
        newNode = readNodesFromPaths(path)
        I._rmNode(parent, oldNode)
        I._addChild(parent, newNode)

    def replaceNodeValuesRecursively(node_skel, node_path):
        new_node = readNodesFromPaths(node_path)[0]
        node_skel[1] = new_node[1]
        for child in node_skel[2]:
            replaceNodeValuesRecursively(child, node_path+'/'+child[0])
        
    # containers2read = ['FlowSolution#Height',
    #                    ':CGNS#Ppart',
    #                    'FlowSolution#DataSourceTerm',
    #                    'FlowSolution#Average']

    containers2read = [':CGNS#Ppart']
    if not I.getNodeFromName1(PartTree, 'FlowSolution#EndOfRun#Coords'):
        containers2read.append('GridCoordinates')
    
    for base in I.getBases(Skeleton):
        basename = I.getName(base)
        for zone in I.getNodesFromType1(base, 'Zone_t'):
            # Only for local zones on proc
            proc = I.getValue(I.getNodeFromName(zone, 'proc'))
            if proc != Cmpi.rank: 
                continue

            zonePath = '{}/{}'.format(basename, I.getName(zone))
            zoneInPartialTree = readNodesFromPaths(zonePath)[0]

            for nodeName2read in containers2read:
                if I.getNodeFromName1(zoneInPartialTree, nodeName2read):
                    replaceNodeByName(zone, zonePath, nodeName2read)

