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

'''
MOLA - Interpolate.py

MODULE IN PROGRESS - NOT FUNCTIONAL YET.

This module proposes a set of functions for applying interpolation and
extrapolation between vectors, fields, etc.

Some functionalities require MPI.

First creation:
13/04/2021 - L. Bernardos
'''

import MOLA

if not MOLA.__ONLY_DOC__:
    import sys
    import pprint
    import os
    from timeit import default_timer as tic
    import numpy as np

    import Converter.PyTree as C
    import Converter.Internal as I
    import Converter.Distributed as Distributed
    import Converter.Mpi as Cmpi
    import Connector.PyTree as X
    import Generator.PyTree as G
    import Distributor2.PyTree as D2


def prepareInterpolation(donor, receiver, container='FlowSolution#SourceTerm'):
    '''
    donor AND receiver MUST BE distributed PyTrees
    donor MUST have fields located in 'container', which MUST BE CellCentered
    '''
    Cmpi.barrier()
    donor = I.copyRef(donor)

    graph, ZonesIntersecting = getGraphAndIntersectingDomain(donor, receiver)

    setInterpolationDataOnReceiver(donor, receiver, ZonesIntersecting,
                                   container=container)

    Cmpi.barrier()

    return graph


def applyInterpolation(donor, receiver, graph, Fields2Interpolate=[],
                       container='FlowSolution#SourceTerm'):

    Cmpi.barrier()
    donor = I.copyRef(donor)

    PreviousContainer = I.__FlowSolutionCenters__
    I.__FlowSolutionCenters__ = container

    # try:
    #     Fields2Exclude = C.getVarNames(donor, excludeXYZ=True, loc='both')[0]
    #     for FieldName in Fields2Interpolate: Fields2Exclude.remove(FieldName)
    #     C._rmVars(donor, Fields2Exclude)
    # except:
    #     pass

    Cmpi.barrier()
    Cmpi._addXZones(donor, graph)
    Cmpi.barrier()


    for FieldName in Fields2Interpolate:
        C._initVars(receiver, 'centers:'+FieldName, 0.)
    Cmpi.barrier()

    X._setInterpTransfers(receiver,donor,storage=0,variables=Fields2Interpolate)

    I.__FlowSolutionCenters__ = PreviousContainer
    Cmpi.barrier()


def setInterpolationDataOnReceiver(donor, receiver, ZonesIntersecting,
                                   container='FlowSolution#SourceTerm'):

    PreviousContainer = I.__FlowSolutionCenters__
    I.__FlowSolutionCenters__ = container
    for ReceiverZone in I.getZones(receiver):
        ReceiverZoneName = I.getName(ReceiverZone)
        DonorZones = [I.getNodeFromName2(donor, DonorZoneName) \
                      for DonorZoneName in ZonesIntersecting[ReceiverZoneName]]

        if not DonorZones: continue

        # TODO: VERIFY THIS IS WORKING

        C._initVars(ReceiverZone, 'centers:cellN',2.)
        X._setInterpData(ReceiverZone, DonorZones, nature=0, penalty=1,
                         loc='centers', storage='direct', sameName=1,
                         interpDataType=1, itype='chimera')
        C._initVars(ReceiverZone, 'centers:cellN',1.)

    I.__FlowSolutionCenters__ = PreviousContainer


def getGraphAndIntersectingDomain(donor, receiver):

    Cmpi.barrier()

    donor = I.copyRef(donor)
    I._rmNodesByType(donor,'FlowSolution_t')
    receiver = I.copyRef(receiver)
    I._rmNodesByType(receiver,'FlowSolution_t')

    donorAABB = Cmpi.createBBoxTree(donor, method='AABB')
    D2._copyDistribution(donor, donorAABB)

    donorOBB = Cmpi.createBBoxTree(donor, method='OBB')
    D2._copyDistribution(donor, donorOBB)

    receiverAABB = Cmpi.createBBoxTree(receiver, method='AABB')
    D2._copyDistribution(receiver, receiverAABB)

    receiverOBB = Cmpi.createBBoxTree(receiver, method='OBB')
    D2._copyDistribution(receiver, receiverOBB)

    graph = computeDistributedGraphOfIntersections(donorAABB, donorOBB,
                                                   receiverAABB, receiverOBB)

    Cmpi._addXZones(donor, graph)

    ZonesIntersecting = X.getIntersectingDomains(receiver, t2=donor,
                                                 method='hybrid',
                                                 taabb=receiverAABB,
                                                 tobb=receiverOBB)

    return graph, ZonesIntersecting


def computeDistributedGraphOfIntersections(donorAABB, donorOBB,
                                           receiverAABB, receiverOBB):
    '''
    Input must be distributed PyTrees of bounding boxes
    '''
    Cmpi.barrier()
    graph={}
    for receiverAABBzone, receiverOBBzone in zip(I.getZones(receiverAABB),
                                                 I.getZones(receiverOBB)):
        for donorAABBzone, donorOBBzone in zip(I.getZones(donorAABB),
                                               I.getZones(donorOBB)):
            if boundingBoxesIntersect(receiverAABBzone, receiverOBBzone,
                                         donorAABBzone,    donorOBBzone):
                rcvRank = Cmpi.getProc(receiverAABBzone)
                dnrRank = Cmpi.getProc(donorAABBzone)
                DnrZoneName = I.getName( donorAABBzone )
                Distributed.updateGraph__(graph, dnrRank, rcvRank, DnrZoneName)

    # this is useless... right?
    # Cmpi.barrier()
    # allGraph = Cmpi.KCOMM.allgather(graph)
    # Cmpi.barrier()
    # graph = {}
    # for i in allGraph:
    #     for k in i:
    #         if not k in graph: graph[k] = {}
    #         for j in i[k]:
    #             if not j in graph[k]: graph[k][j] = []
    #             graph[k][j] += i[k][j]
    #             graph[k][j] = list(set(graph[k][j]))
    # Cmpi.barrier()

    return graph


def boundingBoxesIntersect(AABB1, OBB1, AABB2, OBB2):
    '''
    Inputs must be individual zones of bounding-boxes
    '''
    AABBintersect = G.bboxIntersection(AABB1, AABB2, isBB=True, method='AABB')
    if not AABBintersect: return False

    OBBintersect = G.bboxIntersection(OBB1, OBB2, isBB=True, method='OBB')
    if not OBBintersect: return False

    AO12intersect = G.bboxIntersection(AABB1, OBB2, isBB=True, method='AABBOBB')
    if not AO12intersect: return False

    AO21intersect = G.bboxIntersection(AABB2, OBB1, isBB=True, method='AABBOBB')
    if not AO21intersect: return False

    return True


def write4Debug(MSG):
    with open('LOGS/rank%d.log'%Cmpi.rank,'a') as f: f.write('%s\n'%MSG)
