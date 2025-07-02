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
from itertools import product

from mola.cfd.preprocess.mesh.tools import to_partitioned
import mola.pytree.InternalShortcuts as J

import maia
import Converter.Internal as I
import Converter.PyTree as C

def interpolate_closest(tree_source, tree_target, comm):

    containers_at_vertex = [
        fs.name() for fs in tree_source.group(Type='FlowSolution') 
        if not fs.get(Type='GridLocation') or fs.get(Type='GridLocation').value() == 'Vertex'
        ]
    containers_at_cellcenter = [
        fs.name() for fs in tree_source.group(Type='FlowSolution') 
        if fs.get(Type='GridLocation') and fs.get(Type='GridLocation').value() == 'CellCenter'
        ]
    
    tree_source = to_partitioned(tree_source) 
    tree_target = to_partitioned(tree_target) 

    maia.algo.part.interpolate(
        tree_source, 
        tree_target, 
        comm, 
        containers_name=containers_at_vertex, 
        location='Vertex',
        strategy='Closest',
        n_closest_pt=4,
        )
    maia.algo.part.interpolate(
        tree_source, 
        tree_target, 
        comm, 
        containers_name=containers_at_cellcenter, 
        location='CellCenter',
        strategy='Closest',
        n_closest_pt=4,
        )



def migrateFields(Donor, Receiver, keepMigrationDataForReuse=False,
                 forceAddMigrationData=False):
    '''
    Migrate all fields contained in ``FlowSolution_t`` type nodes of **Donor**
    towards **Receiver** using a zero-th order interpolation (nearest) strategy.

    The same structure of FlowSolution containers of **Donor** are kept in
    **Receiver**. Specifically, interpolations are done from Vertex containers
    towards vertex containers and CellCenter containers towards CellCenter
    containers.

    Parameters
    ----------

        Donor : Tree/base/zone, :py:class:`list` of zone/bases/Trees
            Donor elements.

        Receiver : Tree/base/zone, :py:class:`list` of zone/bases/Trees
            Receiver elements.

            .. important:: **Receiver** is modified.

        keepMigrationDataForReuse : bool
            if :py:obj:`True`, special nodes ``.MigrationData``
            are stored on **Receiver** zones so that migration can be further
            reused, for numerical efficiency. Otherwise, special nodes
            ``.MigrationData`` are destroyed.

            .. hint:: use ``keepMigrationDataForReuse=True`` if you plan
                doing additional migrations of fields **only if** **Donor** and
                **Receiver** fields do not move.

        forceAddMigrationData : bool
            if True, re-compute special nodes ``.MigrationData``, regardless of
            their previous existence.

    '''
    import Geom.PyTree as D


    def addMigrationDataIfForcedOrNotExisting(DonorZones, ReceiverZones):
        for ReceiverZone in ReceiverZones:

            MigrationNode = I.getNodeFromName1(ReceiverZone,
                                               MigrateDataNodeReservedName)


            if forceAddMigrationData or not MigrationNode:
                I._rmNode(ReceiverZone, MigrateDataNodeReservedName)
                MigrationNode = I.createNode(MigrateDataNodeReservedName,
                                            'UserDefinedData_t',
                                            parent=ReceiverZone)

                for DonorZone in DonorZones:
                    addMigrationDataAtReceiver(DonorZone,
                                               ReceiverZone,
                                               MigrationNode)

                updateMasks(ReceiverZone)


    def invokeFieldsAtReceiver(DonorZones, ReceiverZones):
        for DonorZone, ReceiverZone in product(DonorZones, ReceiverZones):

            ContainersNames = getFlowSolutionNamesBasedOnLocations(DonorZone)
            VertexNames, CentersNames = ContainersNames
            for VertexName in VertexNames:
                FlowSolution = I.getNodeFromNameAndType(DonorZone,
                                                        VertexName,
                                                       'FlowSolution_t')
                FieldNames = [I.getName(c) for c in I.getChildren(FlowSolution)]

                invokeReceiverZoneFieldsByContainer(ReceiverZone, VertexName,
                                                    FieldNames, 'Vertex')

            for CentersName in CentersNames:
                FlowSolution = I.getNodeFromNameAndType(DonorZone,
                                                        CentersName,
                                                       'FlowSolution_t')
                FieldNames = [I.getName(c) for c in I.getChildren(FlowSolution)]

                invokeReceiverZoneFieldsByContainer(ReceiverZone, CentersName,
                                                    FieldNames, 'CellCenter')


    def invokeReceiverZoneFieldsByContainer(ReceiverZone, ContainerName,
                                            FieldNames, GridLocation):
        PreviousNodesInternalName = I.__FlowSolutionNodes__[:]
        PreviousCentersInternalName = I.__FlowSolutionCenters__[:]


        DimensionOfReceiver = I.getZoneDim(ReceiverZone)[4]
        isCellCenter = GridLocation == 'CellCenter' and DimensionOfReceiver > 1
        FieldSuffix = 'centers:' if isCellCenter else 'nodes:'


        if isCellCenter:
            I.__FlowSolutionCenters__ = ContainerName
        else:
            I.__FlowSolutionNodes__ = ContainerName

        for FieldName in FieldNames:
            FullVarName = FieldSuffix+FieldName
            FieldNotPresent = C.isNamePresent(ReceiverZone, FullVarName) == -1
            if FieldName != 'GridLocation' and FieldNotPresent:
                C._initVars(ReceiverZone, FullVarName, 0.)

        if isCellCenter:
            I.__FlowSolutionCenters__ = PreviousCentersInternalName

        else:
            I.__FlowSolutionNodes__ = PreviousNodesInternalName


    def migrateDonorFields2ReceiverZone(DonorZones, ReceiverZone):
        MigrationDataNode = I.getNodeFromName1(ReceiverZone,
                                               MigrateDataNodeReservedName)
        DonorMigrationNodes = I.getChildren(MigrationDataNode)

        for DonorMigrationNode in DonorMigrationNodes:
            DonorName = I.getName(DonorMigrationNode)
            DonorZone = J.getZoneFromListByName(DonorZones, DonorName)
            if not DonorZone:
                C.convertPyTree2File(ReceiverZone,'debug.cgns')
                raise ValueError('could not find DonorZone %s. Check debug.cgns.'%DonorName)
            FlowSolutions = I.getNodesFromType1(DonorZone, 'FlowSolution_t')

            for FlowSolution in FlowSolutions:
                GridLocation = getGridLocationOfFlowSolutionNode(FlowSolution)

                Keyname = 'Point' if GridLocation == 'Vertex' else 'Cell'
                MaskNode = I.getNodeFromName(DonorMigrationNode,
                                             Keyname+'Mask')
                Mask = I.getValue(MaskNode)
                Mask = np.asarray(Mask, dtype=bool, order='F')
                PointListDonorNode = I.getNodeFromName(DonorMigrationNode,
                                                 Keyname+'ListDonor')
                PointListDonor = PointListDonorNode[1]

                FlowSolutionName = I.getName(FlowSolution)
                FieldsNodes = I.getChildren(FlowSolution)
                for FieldNode in FieldsNodes:
                    assignReceiverFieldFromDonorFieldNode(FieldNode,
                                                          ReceiverZone,
                                                          Mask,
                                                          PointListDonor,
                                                          FlowSolutionName)


    def assignReceiverFieldFromDonorFieldNode(FieldNode, ReceiverZone, Mask,
                                              PointListDonor, FlowSolutionName):
        FieldName = I.getName(FieldNode)
        if FieldName == 'GridLocation': return

        DonorFieldArray = FieldNode[1]
        isNumpyArray = type(DonorFieldArray) == np.ndarray

        if not isNumpyArray: return
        DonorFieldArray = DonorFieldArray.ravel(order='F')

        ReceiverFlowSolution = I.getNodeFromName1(ReceiverZone,FlowSolutionName)
        ReceiverFieldNode = I.getNodeFromName1(ReceiverFlowSolution, FieldName)

        if ReceiverFieldNode:
            ReceiverFieldArray = ReceiverFieldNode[1].ravel(order='F')
            try:
                ReceiverFieldArray[Mask] = DonorFieldArray[PointListDonor][Mask]
            except IndexError:
                print(len(Mask))
                print(len(PointListDonor))
                ERRMSG = ('Wrong dimensions for '
                          '{FieldName} on {RcvName}/{FlowSolName} container.'
                          ).format(
                          FieldName=FieldName,
                          RcvName=I.getName(ReceiverZone),
                          FlowSolName=FlowSolutionName,
                          )
                raise ValueError(ERRMSG)

        else:
            ERRMSG = ('Did not find field '
                      '{FieldName} on {RcvName}/{FlowSolName} container.\n'
                      'Try migrateFields() function again using option '
                      'forceAddMigrationData=True').format(
                      FieldName=FieldName,
                      RcvName=I.getName(ReceiverZone),
                      FlowSolName=FlowSolutionName)
            raise ValueError(ERRMSG)


    def addMigrationDataAtReceiver(DonorZone, ReceiverZone, MigrationNode):
        hasVertexField = hasFlowSolutionAtVertex(DonorZone)
        if hasVertexField:
            addMigrationDataFromZoneAndKeyName(ReceiverZone,
                                               DonorZone,
                                               MigrationNode,
                                               Keyname='Point')

        hasCenterField = hasFlowSolutionAtCenters(DonorZone)
        if hasCenterField:
            DonorZoneName = I.getName(DonorZone)
            ReceiverZoneRef = I.copyRef(ReceiverZone)
            I._rmNodesByType1(ReceiverZoneRef, 'FlowSolution_t')
            ReceiverZoneCenters = C.node2Center(ReceiverZoneRef)
            DonorZoneRef = I.copyRef(DonorZone)
            I._rmNodesByType1(DonorZoneRef, 'FlowSolution_t')
            DonorZoneCenters = C.node2Center(DonorZoneRef)
            I.setName(DonorZoneCenters, DonorZoneName)
            addMigrationDataFromZoneAndKeyName(ReceiverZoneCenters,
                                               DonorZoneCenters,
                                               MigrationNode,
                                               Keyname='Cell')


    def addMigrationDataFromZoneAndKeyName(zone, DonorZone, MigrationNode,
                                           Keyname='Point'):
        RcvX = I.getNodeFromName2(zone, 'CoordinateX')[1].ravel(order='F')
        RcvY = I.getNodeFromName2(zone, 'CoordinateY')[1].ravel(order='F')
        RcvZ = I.getNodeFromName2(zone, 'CoordinateZ')[1].ravel(order='F')
        NPts = len(RcvX)
        ReceiverAsPoints = [(RcvX[i], RcvY[i], RcvZ[i]) for i in range(NPts)]

        # The following function call is the most costly part
        PointIndex = D.getNearestPointIndex(DonorZone, ReceiverAsPoints)

        PointListDonor = []
        SquaredDistances = []
        for Index, SquaredDistance in PointIndex:
            PointListDonor.append(Index)
            SquaredDistances.append(SquaredDistance)
        PointListDonor = np.array(PointListDonor, dtype=np.int32, order='F')
        SquaredDistances = np.array(SquaredDistances, order='F')

        PointListDonorNode = I.createNode(Keyname+'ListDonor',
                                          'DataArray_t',
                                          value=PointListDonor)

        SquaredDistancesNode = I.createNode(Keyname+'SquaredDistances',
                                            'DataArray_t',
                                            value=SquaredDistances)

        MaskNode = I.createNode(Keyname+'Mask',
                                'DataArray_t',
                                value=np.zeros(NPts,dtype=np.int32, order='F'))

        DonorMigrationDataChildren = [PointListDonorNode,
                                      SquaredDistancesNode,
                                      MaskNode]
        DonorMigrationDataName = DonorZone[0]

        DonorZoneMigrationNode = I.getNodeFromName1(MigrationNode,
                                                    DonorMigrationDataName)
        if not DonorZoneMigrationNode:
            DonorZoneMigrationNode = I.createNode(DonorMigrationDataName,
                                                 'UserDefinedData_t',
                                                  parent=MigrationNode)
        DonorZoneMigrationNode[2].extend(DonorMigrationDataChildren)


    def hasFlowSolutionAtCenters(DonorZone):
        return hasFlowSolutionAtRequestedLocation(DonorZone, 'CellCenter')


    def hasFlowSolutionAtVertex(DonorZone):
        return hasFlowSolutionAtRequestedLocation(DonorZone, 'Vertex')


    def hasFlowSolutionAtRequestedLocation(DonorZone, RequestedLocation):
        FlowSolutionNodes = I.getNodesFromType1(DonorZone, 'FlowSolution_t')
        for FlowSolutionNode in FlowSolutionNodes:
            GridLocation = getGridLocationOfFlowSolutionNode(FlowSolutionNode)
            if GridLocation == RequestedLocation:
                return True
        return False


    def getGridLocationOfFlowSolutionNode(FlowSolutionNode):
        GridLocationNode = I.getNodeFromType1(FlowSolutionNode,
                                              'GridLocation_t')
        return I.getValue(GridLocationNode)



    def getFlowSolutionNamesBasedOnLocations(DonorZone):
        VertexNames = []
        CenterNames = []
        FlowSolutionNodes = I.getNodesFromType1(DonorZone, 'FlowSolution_t')
        for FlowSolutionNode in FlowSolutionNodes:
            GridLocationNode = I.getNodeFromName1(FlowSolutionNode,
                                                  'GridLocation')
            GridLocation = I.getValue(GridLocationNode)
            FlowSolutionName = I.getName(FlowSolutionNode)
            if GridLocation == 'Vertex':
                VertexNames.append(FlowSolutionName)
            elif GridLocation == 'CellCenter':
                CenterNames.append(FlowSolutionName)

        return VertexNames, CenterNames


    def addGridLocationNodeIfAbsent(DonorZones):
        for DonorZone in DonorZones:
            GridDimension = I.getZoneDim(DonorZone)[1]
            FlowSolutionNodes = I.getNodesFromType1(DonorZone, 'FlowSolution_t')
            for FlowSolutionNode in FlowSolutionNodes:
                GridLocationNode = I.getNodeFromName1(FlowSolutionNode,
                                                      'GridLocation')
                if not GridLocationNode:
                    FieldDimension = getFlowSolutionDimension(FlowSolutionNode)
                    if GridDimension == FieldDimension:
                        Location = 'Vertex'
                    else:
                        Location = 'CellCenter'
                    GridLocationNode = I.createNode('GridLocation',
                                                    'GridLocation_t',
                                                    Location)
                    I.addChild(FlowSolutionNode, GridLocationNode, pos=0)


    def getFlowSolutionDimension(FlowSolutionNode):
        for Field in I.getChildren(FlowSolutionNode):
            if I.getName(Field) == 'GridLocation':
                continue
            else:
                return I.getValue(Field).shape[0]


    def raiseErrorIfNotValidDonorZoneNames(DonorZones):
        for DonorZone in DonorZones:
            DonorZoneName = I.getName(DonorZone)
            CenterReservedSuffix = '.c'
            if DonorZoneName.endswith(CenterReservedSuffix):
                ERRMSG = ('Invalid donor zone name {}. '
                          'It cannot end with reserved suffix "{}"').format(
                          DonorZoneName, CenterReservedSuffix)
                raise ValueError(ERRMSG)

    def updateMasks(ReceiverZone):
        MigrationDataNode = I.getNodeFromName1(ReceiverZone,
                                               MigrateDataNodeReservedName)

        for LocationKey in ('Point', 'Cell'):
            SquaredDistances = []
            Masks            = []
            for MigrationDonorZone in I.getChildren(MigrationDataNode):
                SquaredDistanceNode = I.getNodeFromName(MigrationDonorZone,
                                                LocationKey+'SquaredDistances')
                if not SquaredDistanceNode: continue
                SquaredDistance = SquaredDistanceNode[1]
                MaskNode = I.getNodeFromName(MigrationDonorZone,
                                       LocationKey+'Mask')
                Mask = MaskNode[1]
                SquaredDistances.append(SquaredDistance)
                Masks.append(Mask)

            if len(Masks) == 0: continue

            ReceiverZoneNPts = len(Masks[0])
            for i in range(ReceiverZoneNPts):
                LocalPointDistances = np.array([s[i] for s in SquaredDistances])
                ClosestPointOfDonorZoneNumber = np.argmin(LocalPointDistances)
                for maskNumber, mask in enumerate(Masks):
                    if maskNumber == ClosestPointOfDonorZoneNumber:
                        mask[i] = 1
                        break

    MigrateDataNodeReservedName = '.MigrateData'

    Donor = I.copyRef( Donor )
    DonorZones = I.getZones( Donor )
    ReceiverZones = I.getZones( Receiver )

    # https://gitlab.onera.net/numerics/mola/-/issues/191#note_24456
    I._correctPyTree(DonorZones,level=3)
    I._correctPyTree(ReceiverZones,level=3)

    raiseErrorIfNotValidDonorZoneNames( DonorZones )

    addGridLocationNodeIfAbsent( DonorZones )

    addMigrationDataIfForcedOrNotExisting(DonorZones, ReceiverZones)

    invokeFieldsAtReceiver(DonorZones, ReceiverZones)

    for ReceiverZone in ReceiverZones:
        migrateDonorFields2ReceiverZone(DonorZones,
                                        ReceiverZone)

    if not keepMigrationDataForReuse:
        I._rmNodesByName(Receiver, MigrateDataNodeReservedName)

