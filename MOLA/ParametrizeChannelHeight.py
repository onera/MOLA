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

#!/usr/bin/python

import MOLA

if not MOLA.__ONLY_DOC__:
    # Python general packages
    import os
    import sys
    import numpy as np
    from mpi4py import MPI
    # Cassiopee packages
    import Converter.PyTree   as C
    import Converter.Internal as I
    import Post.PyTree        as P
    # ETC packages
    import etc.transform as trf
    import etc.post as epost

from . import InternalShortcuts as J

@J.mute_stdout
def generateHLinesAxial(t, filename, nbslice=21, comm=MPI.COMM_WORLD, tol=1e-10, offset=4, hubFirst=False):
    print("generateHLinesAxial: Generation of Hub&Shroud lines")
    t = trf.cartToCyl(t) # CoordinateY=R, CoordinateZ=T
    xmin = C.getMinValue(t,'CoordinateX')
    xmax = C.getMaxValue(t,'CoordinateX')
    xmin = comm.allreduce(xmin, MPI.MIN)
    xmax = comm.allreduce(xmax, MPI.MAX)
    xmin += tol
    xmax -= tol
    dx   = (xmax-xmin)/(nbslice-1)
    ymin = C.getMinValue(t,'CoordinateY')
    ymax = C.getMaxValue(t,'CoordinateY')
    ymin = comm.allreduce(ymin, MPI.MIN)
    ymax = comm.allreduce(ymax, MPI.MAX)
    xList = [xmin + dx*n for n in range(nbslice)]
    rminL = []
    rmaxL = []
    for x in xList:
        iso  = P.isoSurfMC(t,'CoordinateX',x)
        rmin = C.getMinValue(iso,'CoordinateY')
        rmax = C.getMaxValue(iso,'CoordinateY')
        rmin = comm.allreduce(rmin, MPI.MIN)
        rmax = comm.allreduce(rmax, MPI.MAX)
        print("generateHLinesAxial [{}/{}]: @x={}, r_hub={}, r_shroud={}".format(xList.index(x)+1,len(xList),x,rmin,rmax))
        rminL.append(rmin)
        rmaxL.append(rmax)
    xList.insert(0,xmin-offset*tol)
    rminL.insert(0,rminL[0])
    rmaxL.insert(0,rmaxL[0])
    xList.append(xmax+offset*tol)
    rminL.append(rminL[-1])
    rmaxL.append(rmaxL[-1])
    n = I.newCGNSTree()
    b = I.newCGNSBase('Base', 3, 2, parent=n)
    if hubFirst:
        z = I.newZone('Hub', [[len(xList)],[1],[1]], 'Structured', parent=b)
        g = I.newGridCoordinates(parent=z)
        I.newDataArray('CoordinateX', value=np.asarray(xList), parent=g)
        I.newDataArray('CoordinateY', value=np.asarray(rminL), parent=g)
    z = I.newZone('Shroud', [[len(xList)],[1],[1]], 'Structured', parent=b)
    g = I.newGridCoordinates(parent=z)
    I.newDataArray('CoordinateX', value=np.asarray(xList), parent=g)
    I.newDataArray('CoordinateY', value=np.asarray(rmaxL), parent=g)
    if not hubFirst:
        z = I.newZone('Hub', [[len(xList)],[1],[1]], 'Structured', parent=b)
        g = I.newGridCoordinates(parent=z)
        I.newDataArray('CoordinateX', value=np.asarray(xList), parent=g)
        I.newDataArray('CoordinateY', value=np.asarray(rminL), parent=g)
    if MPI.COMM_WORLD.Get_rank() == 0 : C.convertPyTree2File(n,filename)
    t = trf.cylToCart(t)
    print("generateHLinesAxial: done")
    return n

@J.mute_stdout
def computeHeight(t,hLines,hFormat='bin_tp',constraint=5.,mode='accurate',isInterp=False,writeMask=None,writeMaskCart=None,writePyTreeCyl=None, fsname=None):
    t = trf.cartToCyl(t)
    m = epost.createChannelMesh(t, hLines, format=hFormat) # This function is very verbose
    m = epost.computeChannelHeight(m, fsname=I.__FlowSolutionNodes__)
    if isInterp: m = C.initVars(m, '{isInterp}=1.0')
    t = P.extractMesh(m,t,constraint=constraint,mode=mode)
    for node in I.getNodesFromNameAndType(t, 'FlowSolution', 'FlowSolution_t'):
        I.newGridLocation(value='Vertex', parent=node)
    if fsname is not None:
        I._renameNode(t, 'FlowSolution', fsname)
    if writeMask:      C.convertPyTree2File(m,writeMask)
    if writeMaskCart:  m = trf.cylToCart(m) ; C.convertPyTree2File(m,writeMaskCart)
    if writePyTreeCyl: C.convertPyTree2File(t,writePyTreeCyl)
    t = trf.cylToCart(t)
    return t

@J.mute_stdout
def computeHeight_from_mask(t, mask, constraint=5., mode='accurate', fsname=None):
    t = trf.cartToCyl(t)
    t = P.extractMesh(mask,t,constraint=constraint,mode=mode)
    for node in I.getNodesFromNameAndType(t, 'FlowSolution', 'FlowSolution_t'):
        I.newGridLocation(value='Vertex', parent=node)
    if fsname is not None:
        I._renameNode(t, 'FlowSolution', fsname)
    t = trf.cylToCart(t)
    return t

def filter_internalFlow(t, xmin=-1e20, xmax=1e20, rmin=-1.0, rmax=1e20):
    filtered_tree = I.copyTree(t)
    filtered_tree = C.initVars(filtered_tree,'{Radius}=numpy.sqrt({CoordinateY}**2+{CoordinateZ}**2)')
    for zone in I.getNodesFromType(filtered_tree,'Zone_t'):
        name = I.getName(zone)
        Xmin = C.getMinValue(zone,'CoordinateX')
        Xmax = C.getMaxValue(zone,'CoordinateX')
        Rmin = C.getMinValue(zone,'Radius')
        Rmax = C.getMaxValue(zone,'Radius')
        if not(Xmin > xmin and Xmax < xmax and Rmin > rmin and Rmax < rmax):
            I.rmNode(filtered_tree, zone)
    return filtered_tree

def plot_hub_and_shroud_lines(filename):
    t   = C.convertFile2PyTree(filename)
    # Get geometry
    hub     = I.getNodeFromName(t, 'Hub')
    xHub    = I.getValue(I.getNodeFromName(hub, 'CoordinateX'))
    yHub    = I.getValue(I.getNodeFromName(hub, 'CoordinateY'))
    shroud  = I.getNodeFromName(t, 'Shroud')
    xShroud = I.getValue(I.getNodeFromName(shroud, 'CoordinateX'))
    yShroud = I.getValue(I.getNodeFromName(shroud, 'CoordinateY'))
    # Import matplotlib
    import matplotlib.pyplot as plt
    # Plot
    plt.figure()
    plt.plot(xHub, yHub, '-', label='Hub')
    plt.plot(xShroud, yShroud, '-', label='Shroud')
    plt.axis('equal')
    plt.grid()
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    # Save
    plt.savefig(filename.replace('.plt', '.png'), dpi=150, bbox_inches='tight')
    return 0

if __name__ == '__main__':

    links = []
    t = C.convertFile2PyTree('mesh.cgns', links=links)
    parametrizeChannelHeight(t, nbslice=101)
    C.convertPyTree2File(t, fileOut, links=links)
