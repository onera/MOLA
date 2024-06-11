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

import Converter.PyTree as C
import Converter.Internal as I
import Converter.Mpi as Cmpi
import Converter.Filter as Filter

import mola.naming_conventions as names

from . import mola_logger, rank, comm
from .utils import getStructure, renameTooLongZones, removeNonLocalZones


def save_with_cassiopee(t, filename):
    t = I.copyRef(t) if I.isTopTree(t) else C.newPyTree(['Base', I.getZones(t)])
    removeNonLocalZones(t) # HACK https://elsa.onera.fr/issues/11397
    I._adaptZoneNamesForSlash(t)
    for z in I.getZones(t):
        SolverParam = I.getNodeFromName(z,'.Solver#Param')
        if not SolverParam or not I.getNodeFromName(SolverParam,'proc'):
            Cmpi._setProc(z, Cmpi.rank)
    I._rmNodesByName(t,'ID_*')
    I._rmNodesByType(t,'IntegralData_t')

    Skel = getStructure(t)

    UseMerge = False
    try:
        trees = comm.allgather( Skel )        
        trees.insert( 0, t )
        tWithSkel = I.merge( trees )
        renameTooLongZones(tWithSkel)
        for l in 2,3: I._correctPyTree(tWithSkel,l) # unique base and zone names
    except SystemError:
        UseMerge = True
        mola_logger.warning(f'Cmpi.KCOMM.gather FAILED. Using merge=True', rank=0)
        UseMerge = comm.bcast(UseMerge,root=Cmpi.rank)
        tWithSkel = t

    comm.barrier()
    if rank==0:
        try:
            if os.path.islink(filename):
                os.unlink(filename)
            else:
                os.remove(filename)
        except:
            pass
    comm.barrier()

    Cmpi.convertPyTree2File(tWithSkel, filename, merge=UseMerge) # BUG https://elsa.onera.fr/issues/11398
    comm.barrier()

def load_skeleton(Skeleton=None, PartTree=None):
    '''
    Load the skeleton tree (if not given) and add nodes that are required for
    coprocessing.

    Parameters
    ----------

        Skeleton : PyTree or :py:obj:`None`
            Skeleton tree, got from ``Cmpi.convertFile2SkeletonTree`` with
            Cassiopee or ``PyPartBase.getPyPartSkeletonTree`` with PyPart.
            If :py:obj:`None`, load the Skeleton tree with
            ``Cmpi.convertFile2SkeletonTree(FILE_CGNS)``.

        PartTree : PyTree or :py:obj:`None`
            Partial tree, got from ``Cmpi.convertFile2PyTree(..., proc=rank)``
            with Cassiopee or ``PyPartBase.runPyPart`` with PyPart.
            If :py:obj:`None`, only the needed nodes are read with
            ``Converter.Filter``.

    Returns
    -------

        Skeleton : PyTree
            Skeleton Tree with additional information, required for Coprocess
            functions.
    '''
    addCoordinates = True
    if not Skeleton: Skeleton = Cmpi.convertFile2SkeletonTree(names.FILE_INPUT_SOLVER)

    FScoords = I.getNodeFromName1(Skeleton, 'FlowSolution#EndOfRun#Coords')
    if FScoords: addCoordinates = False

    I._rmNodesByName(Skeleton, 'FlowSolution#EndOfRun*')
    I._rmNodesByName(Skeleton, 'ID_*')

    if PartTree:
        # Needed nodes are read from PartTree
        def readNodesFromPaths(path):
            split_path = path.split('/')
            path_begining = '/'.join(split_path[:-1])
            name = split_path[-1]
            parent = I.getNodeFromPath(PartTree, path_begining)
            return I.getNodesFromName(parent, name)

        FScoords = I.getNodeFromName1(PartTree, 'FlowSolution#EndOfRun#Coords')
        if FScoords: addCoordinates = False
    else:
        # Needed nodes are read from FILE_CGNS with Converter.Filter
        def readNodesFromPaths(path):
            return Filter.readNodesFromPaths(names.FILE_INPUT_SOLVER, [path])


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

    containers2read = ['FlowSolution#Height',
                       ':CGNS#Ppart',
                       'FlowSolution#DataSourceTerm',
                       'FlowSolution#Average']

    for base in I.getBases(Skeleton):
        basename = I.getName(base)
        for zone in I.getNodesFromType1(base, 'Zone_t'):
            # Only for local zones on proc
            proc = I.getValue(I.getNodeFromName(zone, 'proc'))
            if proc != rank: continue

            zonePath = '{}/{}'.format(basename, I.getName(zone))
            zoneInPartialTree = readNodesFromPaths(zonePath)[0]

            # Coordinates
            if addCoordinates: replaceNodeByName(zone, zonePath, 'GridCoordinates')

            for nodeName2read in containers2read:
                if I.getNodeFromName1(zoneInPartialTree, nodeName2read):
                    replaceNodeByName(zone, zonePath, nodeName2read)

            # For unstructured mesh
            if I.getZoneType(zone) == 2: # unstructured zone
                replaceNodeByName(zone, zonePath, ':elsA#Hybrid')
                # TODO: Add other types of Elements_t nodes if needed
                replaceNodeByName(zone, zonePath, 'NGonElements')
                replaceNodeByName(zone, zonePath, 'NFaceElements')
                # PointList in BCs and GridConnectivities
                for BC in I.getNodesFromType2(zone, 'BC_t'):
                    BCpath = '{}/ZoneBC/{}'.format(zonePath, I.getName(BC))
                    replaceNodeByName(BC, BCpath, 'PointList')
                for GC in I.getNodesFromType2(zone, 'GridConnectivity_t'):
                    GCpath = '{}/ZoneGridConnectivity/{}'.format(zonePath, I.getName(GC))
                    replaceNodeByName(GC, GCpath, 'PointList')

            # put BCDataSet#Average in Skeleton
            if not PartTree: # from file
                for zonebc in I.getNodesFromType1(zone,'ZoneBC_t'):
                    for bc in I.getNodesFromType1(zonebc,'BC_t'):
                        bcds_avg = I.getNodeFromName1(bc,'BCDataSet#Average')
                        if bcds_avg is None: continue
                        for bcdata in I.getNodesFromType1(bcds_avg,'BCData_t'):
                            bcdatapath = '/'.join([basename,
                                                   zone[0],
                                                   zonebc[0],
                                                   bc[0],
                                                   bcds_avg[0],
                                                   bcdata[0]])
                            for data in I.getNodesFromType1(bcdata,'DataArray_t'):
                                replaceNodeByName(bcdata, bcdatapath, data[0])
            else: # from PartTree
                for zonebc in I.getNodesFromType1(zone,'ZoneBC_t'):
                    for bc in I.getNodesFromType1(zonebc,'BC_t'):
                        bcds_avg = I.getNodeFromName1(bc,'BCDataSet#Average')
                        if bcds_avg is None: continue
                        bcpath = '/'.join([basename, zone[0], zonebc[0], bc[0]])
                        replaceNodeByName(bc, bcpath, 'BCDataSet#Average')

        # always require to fully read Mask nodes 
        masks = I.getNodeFromName1(base, '.MOLA#Masks')
        if masks:
            replaceNodeValuesRecursively(masks, '/'.join([basename, masks[0]]))

    return Skeleton

