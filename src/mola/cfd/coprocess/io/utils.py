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

import Converter.Internal as I

from mola.logging import MolaException

def getStructure(t):
    '''
    Get a PyTree's base structure (children of base nodes are empty)

    Parameters
    ----------

        t : PyTree
            tree from which structure is to be extracted

    Returns
    -------
        Structure : PyTree
            reference copy of **t**, with empty bases
    '''
    tR = I.copyRef(t)
    for n in I.getZones(tR):
        n[2] = []
    return tR

def removeNonLocalZones(t):
    # HACK https://elsa.onera.fr/issues/11397
    for base in I.getBases(t):
        children_to_keep = []
        for child in base[2]:
            if child[3] != 'Zone_t':
                children_to_keep += [ child ]
            elif zoneHasData(child):
                children_to_keep += [ child ]
        base[2] = children_to_keep

def renameTooLongZones(to, n=25):
    '''
    .. warning:: this is a private function, employed by :py:func:`saveSurfaces`

    This function rename zones in a PyTree **to** if their names are too long
    to be save in a CGNS file (maximum length = 32 characters).

    The new name of a zone follows this format:
    ``<NewName>`` = ``<First <n> characters of old name>_<ID>``
    with ``<n>`` an integer and ``<ID>`` the lowest integer (starting form 0) such as
    ``<NewName>`` does not already exist in the PyTree.

    Parameters
    ----------

        to : PyTree
            PyTree to check. Zones with a too long name will be renamed.

            .. note:: tree **to** is modified

        n : int
            Number of characters to keep in the old zone name.
    '''
    for zone in I.getZones(to):
        zoneName = I.getName(zone)
        if len(zoneName) > 32:
            CurrentZoneNames = [I.getName(z) for z in I.getZones(to)]
            c = 0
            newName = '{}_{}'.format(zoneName[:n+1], c)
            while newName in CurrentZoneNames and c < 1000:
                c += 1
                newName = '{}_{}'.format(zoneName[:n+1], c)
            if c == 1000:
                ERRMSG = 'Zone {} has not been renamed by renameTooLongZones() but its length ({}) is greater than maximum authorized length (32)'.format(zoneName, len(zoneName))
                raise MolaException(ERRMSG)
            I.setName(zone, newName)

def zoneHasData(zone):
    if zone[3] != 'Zone_t': raise AttributeError('argument must be a zone')
    gcs = I.getNodesFromType1(zone, 'GridCoordinates_t')
    fss = I.getNodesFromType1(zone, 'FlowSolution_t')
    containers = gcs + fss
    if not containers: return False
    for container in containers:
        for data in I.getNodesFromType1(container,'DataArray_t'):
            if data[1] is not None: 
                return True
            
def ravelBCDataSet(t):
    # HACK https://elsa.onera.fr/issues/11219
    # HACK https://elsa-e.onera.fr/issues/10750
    for zone in I.getZones(t):
        for zbc in I.getNodesFromType1(zone,'ZoneBC_t'):
            for bc in I.getNodesFromType1(zbc,'BC_t'):
                for bcds in I.getNodesFromType1(bc,'BCDataSet_t'):
                    for bcd in I.getNodesFromType1(bcds,'BCData_t'):
                        for da in I.getNodesFromType1(bcd,'DataArray_t'):
                            if da[1] is not None:
                                da[1] = da[1].ravel(order='K')

def forceFamilyBCasFamilySpecified(t):
    # https://elsa.onera.fr/issues/10928
    for base in I.getBases(t):
        for zone in I.getZones(base):
            for ZoneBC in I.getNodesFromType1(zone,'ZoneBC_t'):
                for BC in I.getNodesFromType1(ZoneBC,'BC_t'):
                    FamilyNameNode = I.getNodeFromType1(BC,'FamilyName_t')
                    if FamilyNameNode is not None:
                        I.setValue(BC,'FamilySpecified')
                        FamilyName = I.getValue(FamilyNameNode)
                        if not I.getNodeFromName1(base,FamilyName):
                            FamilyAtBase = I.createNode(FamilyName,'Family_t',parent=base)
                            I.createNode('FamilyBC','FamilyBC_t',value='UserDefined',parent=FamilyAtBase)
                        continue

