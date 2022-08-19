'''
ExtractSurfacesProcessor.py module
15/07/2021 - L. Bernardos
'''

import numpy as np
from timeit import default_timer as tic

import Converter.PyTree as C
import Converter.Internal as I
import Transform.PyTree as T
import Intersector.PyTree as XOR
import Post.PyTree as P
import Geom.PyTree as D

ijk2ind = {'i':0, 'j':1, 'k':2}


def extractSurfacesByOffsetCellsFromBCFamilyName(t, BCFamilyName='MyBC',
                                                                NCellsOffset=2):
    OffsetSurfaces = []
    FamilyBCs = C.getFamilyBCs(t, BCFamilyName)
    BCNames = [fbc[0] for fbc in FamilyBCs]
    ZoneName2ZoneAndWindows = getZonesAndWindowsOfBCNames(t, BCNames)
    AllZonesForSubzoning, AllWindowsForSubzoning = [], []
    for ZoneName in ZoneName2ZoneAndWindows:
        zone = ZoneName2ZoneAndWindows[ZoneName]['zone']
        zoneDims = I.getZoneDim(zone)[1:4]
        for window in ZoneName2ZoneAndWindows[ZoneName]['windows']:
            ZoneNamesAlreadyPropagated = []
            InwardsIndex = findInwardsIndexFromExteriorWindow(window)
            OpposedWindow, MaximumOffset = getOpposedWindowAndCumulatedOffsetFromInwardsIndex(zoneDims, window, InwardsIndex)

            ZonesForSubzoning, WindowsForSubzoning = [], []

            addZonesAndWindowsForSubzoning(WindowsForSubzoning,
                                           ZonesForSubzoning,
                                           t, zone, window, NCellsOffset,
                                           ZoneNamesAlreadyPropagated)

            AllZonesForSubzoning.extend(ZonesForSubzoning)
            AllWindowsForSubzoning.extend(WindowsForSubzoning)

    for z, wnd in zip(AllZonesForSubzoning, AllWindowsForSubzoning):
        OffsetSurface = T.subzone(z, tuple(wnd[:,0]),
                                     tuple(wnd[:,1]))
        OffsetSurface[0] += '.offset'
        OffsetSurfaces += [OffsetSurface]
    Nodes2Remove = ('ZoneBC','.Solver#ownData','ZoneGridConnectivity')
    [I._rmNodesByName(OffsetSurfaces, n) for n in Nodes2Remove]

    if NCellsOffset > 0:
        BCsurfaces = extractSurfacesByOffsetCellsFromBCFamilyName(t,
                                                                BCFamilyName, 0)
        if not BCsurfaces:
            print('warning: no BCsurfaces for '+BCFamilyName)
            return []
        hook, indir = C.createGlobalHook(BCsurfaces,function='nodes',indir=1)
        OffsetSurfaces = trimExteriorFaces(OffsetSurfaces, NCellsOffset, hook)
    OffsetSurfaces = T.merge(OffsetSurfaces)
    for z in OffsetSurfaces: z[0] += '.offset'

    return OffsetSurfaces

def getZonesAndWindowsOfBCNames(t, BCNames):
    ZoneName2ZoneAndWindows = {}
    for zone in I.getZones(t):
        ZoneName = zone[0]
        ZoneBC = I.getNodeFromName1(zone,'ZoneBC')
        if not ZoneBC: continue
        for BC in I.getChildren(ZoneBC):
            if BC[0] in BCNames:
                print('found %s'%BC[0])
                if ZoneName not in ZoneName2ZoneAndWindows:
                    ZoneName2ZoneAndWindows[ZoneName] = {'zone':zone,
                                                         'windows':[]}
                PointRange = I.getNodeFromName(BC,'PointRange')[1]
                ZoneName2ZoneAndWindows[ZoneName]['windows'] += [PointRange]

    if not ZoneName2ZoneAndWindows:
        raise ValueError('could not find requested BC windows')

    return ZoneName2ZoneAndWindows

def findInwardsIndexFromExteriorWindow(window):
    if window[0,0] == window[0,1]:
        if window[0,0] == 1: return '+i'
        else: return '-i'
    elif window[1,0] == window[1,1]:
        if window[1,0] == 1: return '+j'
        else: return '-j'
    elif window[2,0] == window[2,1]:
        if window[2,0] == 1: return '+k'
        else: return '-k'

def getOpposedWindowAndCumulatedOffsetFromInwardsIndex(zoneDims, window,
                                                                  InwardsIndex):
    OpposedWindow = window + 0
    ijk = InwardsIndex[1]
    CummulatedOffset = zoneDims[ijk2ind[ijk]] - 1
    if '+' in InwardsIndex:
        OpposedWindow[ijk2ind[ijk],:] += CummulatedOffset
    else:
        OpposedWindow[ijk2ind[ijk],:] -= CummulatedOffset

    return OpposedWindow, CummulatedOffset

def addZonesAndWindowsForSubzoning(WindowsForSubzoning, ZonesForSubzoning,
                 t, zone, window, NCellsOffset, ZoneNamesAlreadyPropagated):

    zoneDims = I.getZoneDim(zone)[1:4]
    InwardsIndex = findInwardsIndexFromExteriorWindow(window)
    OpposedWindow, MaximumOffset = getOpposedWindowAndCumulatedOffsetFromInwardsIndex(zoneDims, window, InwardsIndex)

    if NCellsOffset > MaximumOffset:
        zones, windows = getConnectedZonesWindows(t, zone, OpposedWindow,
                                                    ZoneNamesAlreadyPropagated)
        NewOffset = NCellsOffset - MaximumOffset
        for z, wnd in zip(zones, windows):
            addZonesAndWindowsForSubzoning(WindowsForSubzoning,
                                           ZonesForSubzoning,
                                           t, z, wnd, NewOffset,
                                           ZoneNamesAlreadyPropagated)
    elif zone[0] not in ZoneNamesAlreadyPropagated:
        SubzoneWindow = getNewWindowFromIndwardsIndexAndOffset(window,
                                            InwardsIndex, NCellsOffset)
        WindowsForSubzoning.append( SubzoneWindow )
        ZonesForSubzoning.append( zone )

def getNewWindowFromIndwardsIndexAndOffset(window, InwardsIndex, NCellsOffset):
    SubzoneWindow = window + 0
    ijk = InwardsIndex[1]
    if '+' in InwardsIndex: SubzoneWindow[ijk2ind[ijk],:] += NCellsOffset
    else: SubzoneWindow[ijk2ind[ijk],:] -= NCellsOffset

    return SubzoneWindow

def trimExteriorFaces(OffsetSurfaces, NCellsOffset, hookBCsurfaces):
    OffsetSurfaces = T.merge(OffsetSurfaces)
    SplitOffsetSurfaces = splitSurfacesAtOffset(OffsetSurfaces, NCellsOffset)
    TrimmedOffsetSurfaces = [s for s in SplitOffsetSurfaces if not surfaceTouchesBCsurface(s, hookBCsurfaces)]
    return TrimmedOffsetSurfaces

def getConnectedZonesWindows(t, zone, window, ZoneNamesAlreadyPropagated):
    AllZonesInTree = I.getZones(t)
    zones, windows = [], []
    for BCMatch in I.getNodesFromType2(zone, 'GridConnectivity1to1_t'):
        PointRange = I.getNodeFromName(BCMatch, 'PointRange')[1]
        if windowsOverlap(window, PointRange):
            ConnectedZoneName = I.getValue(BCMatch)
            if ConnectedZoneName in ZoneNamesAlreadyPropagated: continue

            connzone, = [z for z in AllZonesInTree if z[0]==ConnectedZoneName]
            zones.append(connzone)
            ZoneNamesAlreadyPropagated.append(zone[0])

            PointRangeDonor = I.getNodeFromName(BCMatch, 'PointRangeDonor')[1]
            windows.append(PointRangeDonor)

    return zones, windows

def windowsOverlap(window1, window2):
    if (window1[0,0] == window1[0,1]) and (window2[0,0] == window2[0,1]):
        overlap = any([window1[1,1] > window2[1,0],
                       window1[2,1] > window2[2,1]])

    elif (window1[1,0] == window1[1,1]) and (window2[1,0] == window2[1,1]):
        overlap = any([window1[0,1] > window2[0,0],
                       window1[2,1] > window2[2,1]])

    elif (window1[2,0] == window1[2,1]) and (window2[2,0] == window2[2,1]):
        overlap = any([window1[0,1] > window2[0,0],
                       window1[1,1] > window2[1,1]])

    else:
        return False

    return overlap

def splitSurfacesAtOffset(OffsetSurfaces, NCellsOffset):
    JcstSplitOffsetSurfaces = []
    for zone in OffsetSurfaces:
        Ni, Nj = I.getZoneDim(zone)[1:3]

        if Nj-1 > 2*NCellsOffset:
            JcstSplitOffsetSurfaces.append(T.subzone(zone,
                                            (1, 1, 1),
                                            (Ni, 1+NCellsOffset, 1)))
            JcstSplitOffsetSurfaces.append(T.subzone(zone,
                                                (1,1+NCellsOffset,1),
                                                (Ni,Nj-(NCellsOffset),1)))
            JcstSplitOffsetSurfaces.append(T.subzone(zone,
                                                (1, Nj-(NCellsOffset), 1),
                                                (Ni ,Nj,1)))

        elif Nj == 2*NCellsOffset:
            JcstSplitOffsetSurfaces.append(T.subzone(zone,
                    (1, 1, 1),
                    (Ni, NCellsOffset+1, 1)))
            JcstSplitOffsetSurfaces.append(T.subzone(zone,
                    (1, NCellsOffset+1, 1),
                    (Ni, Nj, 1)))


        elif Nj > NCellsOffset + 1:
            JcstSplitOffsetSurfaces.append(T.subzone(zone,
                    (1, 1, 1),
                    (Ni, Nj-(NCellsOffset), 1)))
            JcstSplitOffsetSurfaces.append(T.subzone(zone,
                    (1, 1+NCellsOffset, 1),
                    (Ni, Nj, 1)))

        else:
            JcstSplitOffsetSurfaces.append(zone)

    SplitOffsetSurfaces = []
    for zone in JcstSplitOffsetSurfaces:
        Ni, Nj = I.getZoneDim(zone)[1:3]

        if Ni-1 > 2*NCellsOffset:
            SplitOffsetSurfaces.append(T.subzone(zone,
                                        (1, 1, 1),
                                        (1+NCellsOffset, Nj, 1)))
            SplitOffsetSurfaces.append(T.subzone(zone,
                                        (1+NCellsOffset, 1, 1),
                                        (Ni-(NCellsOffset), Nj, 1)))

            SplitOffsetSurfaces.append(T.subzone(zone,
                                        (Ni-(NCellsOffset), 1, 1),
                                        (Ni, Nj, 1)))

        elif Ni == 2*NCellsOffset:
            SplitOffsetSurfaces.append(T.subzone(zone,
                    (1, 1, 1),
                    (NCellsOffset+1, Nj, 1)))
            SplitOffsetSurfaces.append(T.subzone(zone,
                    (NCellsOffset+1, 1, 1),
                    (Ni, Nj, 1)))


        elif Ni > NCellsOffset + 1:
            SplitOffsetSurfaces.append(T.subzone(zone,
                    (1, 1, 1),
                    (Ni-(NCellsOffset), Nj, 1)))
            SplitOffsetSurfaces.append(T.subzone(zone,
                    (NCellsOffset+1, 1, 1),
                    (Ni, Nj, 1)))

        else:
            SplitOffsetSurfaces.append(zone)

    return SplitOffsetSurfaces



def surfaceTouchesBCsurface(surface, hookBCsurfaces, tol=1e-10):
    nodes, distances = C.nearestNodes(hookBCsurfaces, surface)
    MinDistance = np.min(distances)
    if MinDistance < tol:
        return True
    return False
