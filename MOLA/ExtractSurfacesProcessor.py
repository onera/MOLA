'''
ExtractSurfacesProcessor.py module
15/07/2021 - L. Bernardos
'''

import MOLA

if not MOLA.__ONLY_DOC__:
    import numpy as np
    from timeit import default_timer as tic

    import Converter.PyTree as C
    import Converter.Internal as I
    import Transform.PyTree as T
    import Connector.PyTree as X
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
        for window in ZoneName2ZoneAndWindows[ZoneName]['windows']:
            ZoneNamesAlreadyPropagated = []
            ZonesForSubzoning, WindowsForSubzoning = [], []

            addZonesAndWindowsForSubzoning(WindowsForSubzoning,
                                           ZonesForSubzoning,
                                           t, zone, window, NCellsOffset,
                                           ZoneNamesAlreadyPropagated)

            AllZonesForSubzoning.extend(ZonesForSubzoning)
            AllWindowsForSubzoning.extend(WindowsForSubzoning)

    i = -1
    for z, wnd in zip(AllZonesForSubzoning, AllWindowsForSubzoning):
        i += 1
        OffsetSurface = T.subzone(z, tuple(wnd[:,0]),
                                     tuple(wnd[:,1]))
        info = I.createUniqueChild(OffsetSurface, '.MOLA#Offset', 'UserDefinedData_t')
        I.createUniqueChild(info, 'Zone', 'DataArray_t', value=z[0])
        I.createUniqueChild(info, 'BCFamily', 'DataArray_t', value=BCFamilyName)
        I.createUniqueChild(info, 'Window', 'DataArray_t', value=np.array(wnd,order='F'))
        OffsetSurface[0] = 'offset.%d'%i
        OffsetSurfaces += [OffsetSurface]
    Nodes2Remove = ('ZoneBC','.Solver#ownData','ZoneGridConnectivity')
    [I._rmNodesByName(OffsetSurfaces, n) for n in Nodes2Remove]

    if NCellsOffset > 0:
        BCsurfaces = extractSurfacesByOffsetCellsFromBCFamilyName(t,
                                                                BCFamilyName, 0)
        for z in I.getZones(t): I._rmNodesByName1(z, '.MOLA#Offset')
        if not BCsurfaces:
            print('warning: no BCsurfaces for '+BCFamilyName)
            return []
        OffsetSurfaces = trimExteriorFaces(OffsetSurfaces, BCsurfaces,
                                           NCellsOffset)


    _includeMedianCellHeight(t, OffsetSurfaces)
    migrateOffsetData(t, OffsetSurfaces)

    return OffsetSurfaces

def getZonesAndWindowsOfBCNames(t, BCNames):
    ZoneName2ZoneAndWindows = {}
    for zone in I.getZones(t):
        ZoneName = zone[0]
        ZoneBC = I.getNodeFromName1(zone,'ZoneBC')
        if not ZoneBC: continue
        for BC in I.getChildren(ZoneBC):
            if BC[0] in BCNames:
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

def trimExteriorFaces(OffsetSurfaces, BCSurfaces, NCellsOffset):
    surfs = OffsetSurfaces
    X.connectMatch(surfs, tol=1e-8, dim=2)
    for s in surfs:
        windows = windowsOfSurfaceTouchingGrid(s, BCSurfaces)
        if len(windows) == 0: continue

        for window in windows:
            propagateOffsetAndTagSurface(s, window, NCellsOffset, surfs)


    mergeSplitWindows(surfs)
    addSurfacesByWindowComposition(surfs)
    surfs = filterSurfacesUsingTag(surfs)
    I._rmNodesByType(surfs, 'GridConnectivity1to1_t')
    X.connectMatch(surfs, tol=1e-8, dim=2)
    for s in surfs: tagSurface(s,'keep', True)

    for s in surfs:
        windows = windowsOfSurfaceTouchingGrid(s, BCSurfaces)
        if len(windows) == 0: continue

        for window in windows:
            propagateOffsetAndTagSurface(s, window, NCellsOffset, surfs)

    trimmed_surfaces = filterSurfacesUsingTag(surfs)
    return trimmed_surfaces

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

def windowsOfSurfaceTouchingGrid(surface, grid):
    hook, _ = C.createGlobalHook(surface, function='nodes', indir=1)
    windows = []

    for block in I.getZones(grid):
        nodes = np.array(C.identifyNodes(hook, block))
        nodes = np.sort(nodes[nodes>0])
        Ni, Nj = I.getZoneDim(surface)[1:3]
        unr = np.vstack(np.unravel_index(nodes-1, (Ni,Nj), order='F'))+1
        NbOfTouchingPts = len(unr[0])
        if NbOfTouchingPts == 0 or NbOfTouchingPts == 1: continue
        windows += [ np.array([[unr[0,0],unr[0,-1]],
                               [unr[1,0],unr[1,-1]]], order='F') ]

    return windows

def getOffsetExceess(surface, NCellOffset, InwardsIndex):
    dims = I.getZoneDim(surface)[1:3]
    return NCellOffset - dims[ijk2ind[InwardsIndex[1]]]

def tagSurface(surface, tag='remove', forced=False):
    info = I.getNodeFromName1(surface,'.MOLA#Trim')
    if not info:
        info = I.createUniqueChild(surface, '.MOLA#Trim', 'UserDefinedData_t')
    else:
        previous_tag = I.getValue( I.getNodeFromName1(info, 'tag') )
        if not forced and previous_tag == 'to_split': return
    I.createUniqueChild(info, 'tag', 'DataArray_t', value=tag)

def getOpposedConnectedSurfaces(surface, InwardIndex, AllSurfaces):
    zones, windows = [], []
    for BCMatch in I.getNodesFromType2(surface, 'GridConnectivity1to1_t'):
        PointRange = I.getNodeFromName(BCMatch, 'PointRange')[1]
        OpposedIndex = findInwardsIndexFromExteriorWindow(PointRange)
        if OpposedIndex[1] != InwardIndex[1]: continue
        if OpposedIndex[0] == InwardIndex[0]: continue

        ConnectedZoneName = I.getValue(BCMatch)
        
        connzone, = [z for z in AllSurfaces if z[0]==ConnectedZoneName]
        zones.append(connzone)

        PointRangeDonor = I.getNodeFromName(BCMatch, 'PointRangeDonor')[1]
        windows.append(PointRangeDonor)

    return zones, windows    

def addSplitWindow(surface, window, InwardIndex, NCellsOffset):
    splitWindow = getNewWindowFromIndwardsIndexAndOffset(window, InwardIndex,
                                                                   NCellsOffset)
    
    if splitWindow[0,0]==0 or splitWindow[1,0]==0:
        print("\033[93mWARNING for surface %s\033[0m"%surface[0])
        return
    info = I.getNodeFromName1(surface, '.MOLA#Trim')
    tag = I.getValue(I.getNodeFromName1(info, 'tag')) 
    if tag == 'remove':
        C.convertPyTree2File(surface,'debug.cgns')
        raise ValueError('UNEXPECTED BEHAVIOR FOR SURFACE %s'%surface[0])
    sw = I.createChild(info, 'SplitWindow', 'DataArray_t')
    I.createUniqueChild(sw, 'Window', 'DataArray_t', value=splitWindow)
    I.createUniqueChild(sw, 'InwardIndex', 'DataArray_t', value=InwardIndex)
    for i, sw in enumerate(I.getNodesFromName(info,'SplitWindow*')):
        sw[0] = 'SplitWindow.%d'%i

def propagateOffsetAndTagSurface(surface, window, NCellsOffset, AllSurfaces):
    inwardInd = findInwardsIndexFromExteriorWindow(window)
    excess = getOffsetExceess(surface, NCellsOffset, inwardInd)

    if excess >= 0:
        tagSurface(surface, 'remove')
        for surf, wndw in zip(*getOpposedConnectedSurfaces(surface, inwardInd,
                                                          AllSurfaces)):
            propagateOffsetAndTagSurface(surf, wndw, excess+1,
                                         AllSurfaces)
            
    else:
        tagSurface(surface, 'to_split')
        addSplitWindow(surface, window, inwardInd, NCellsOffset)

def addSurfacesByWindowComposition(surfaces):
    newSurfaces = []
    for s in I.getZones(surfaces):
        
        trimInfo = I.getNodeFromName1(s,'.MOLA#Trim')
        if not trimInfo: continue

        tag_node = I.getNodeFromName1(trimInfo,'tag')
        if not tag_node:
            print('node tag absent of surface "%s"'%(s[0]))
            continue

        tag = I.getValue(tag_node)
        if tag == 'remove': continue

        SplitWindows = I.getNodesFromName(trimInfo,'SplitWindow*')
        NbOfSplit = len(SplitWindows)
        if NbOfSplit ==0:
            print('no split windows at surface "%s"'%s[0])
            continue

        Ni, Nj = I.getZoneDim(s)[1:3]

        if NbOfSplit == 1:
            w = I.getValue(I.getNodeFromName1(SplitWindows[0],'Window'))
            inwInd = I.getValue(I.getNodeFromName1(SplitWindows[0],'InwardIndex'))

            if   inwInd == '+i':
                slice = (w[0,0],w[1,0],1),(w[0,1],w[1,1],1)
            elif inwInd == '-i':
                slice = (1,w[1,0],1),(w[0,0],w[1,1],1)
            elif inwInd == '+j':
                slice = (w[0,0],w[1,0],1),(w[0,1],w[1,1],1)
            elif inwInd == '-j':
                slice = (w[0,0],1,1),(w[0,1],w[1,1],1)
            else:
                raise ValueError('InwardIndex "%s" not implemented'%inwInd)


        elif NbOfSplit == 2:
            w0 = I.getValue(I.getNodeFromName1(SplitWindows[0],'Window'))
            inwInd0 = I.getValue(I.getNodeFromName1(SplitWindows[0],'InwardIndex'))
            w1 = I.getValue(I.getNodeFromName1(SplitWindows[1],'Window'))
            inwInd1 = I.getValue(I.getNodeFromName1(SplitWindows[1],'InwardIndex'))

            if inwInd0[1] != 'i':
                w0, w1 = w1, w0
                inwInd0, inwInd1 = inwInd1, inwInd0

            if inwInd0[1] == inwInd1[0]:
                C.convertPyTree2File(s,'debug.cgns')
                raise ValueError('impossible split topology. Check debug.cgns')

            if   inwInd0 == '+i' and inwInd1 == '+j':
                slice = (w0[0,0],w1[1,0],1),(Ni,Nj,1)
            elif inwInd0 == '+i' and inwInd1 == '-j':
                slice = (w0[0,0],1,1),(Ni,w1[1,1],1)
            elif inwInd0 == '-i' and inwInd1 == '+j':
                slice = (1,w1[1,0],1),(w0[0,1],Nj,1)
            elif inwInd0 == '-i' and inwInd1 == '-j':
                slice = (1,1,1),(w0[0,1],w1[1,1],1)

        else:
            raise ValueError('unexpected number of SplitWindows. '
                             'Make sure to merge SplitWindows before composition.')

        try:
            newSurf = T.subzone(s, *slice) 
        except TypeError:
            C.convertPyTree2File(s,'debug.cgns')
            msg = ('failed slice %s for surface %s with dims %dx%d. '
                   'Check debug.cgns')%(str(slice),s[0],Ni,Nj)
            raise ValueError(msg)

        newSurf[0] = s[0]+'.split'
        tagSurface(newSurf, tag='keep')
        tagSurface(s, tag='remove', forced=True)
        updateOffsetWindowInfoAfterSplit(newSurf,slice)
        offsetInfo = I.getNodeFromName1(s,'.MOLA#Offset')
        if offsetInfo: I.addChild(s, offsetInfo)

        newSurfaces += [ newSurf ]

    surfaces.extend(newSurfaces)

def updateOffsetWindowInfoAfterSplit(surface, slice):
    info = I.getNodeFromName1(surface,'.MOLA#Offset')
    window = I.getValue(I.getNodeFromName1(info,'Window'))
    s = 0
    for i in range(3):
        if window[i,0] == window[i,1]: continue
        window[i,0] = slice[0][s]
        window[i,1] = slice[1][s]
        s += 1


def mergeSplitWindows(surfaces):

    for s in I.getZones(surfaces):
    
        trimInfo = I.getNodeFromName1(s,'.MOLA#Trim')
        if not trimInfo: continue
        
        tag_node = I.getNodeFromName1(trimInfo,'tag')
        if not tag_node:
            print('node tag absent of surface "%s"'%(s[0]))
            continue

        tag = I.getValue(tag_node)
        if tag == 'remove': continue

        SplitWindows = I.getNodesFromName(trimInfo,'SplitWindow*')
        NbOfSplit = len(SplitWindows)
        if NbOfSplit < 2: continue

        SplitWindows_I, wnds_I, inds_I = [], [], []
        SplitWindows_J, wnds_J, inds_J = [], [], []

        for sw in SplitWindows:
            wnd = I.getValue(I.getNodeFromName1(sw,'Window'))
            ind = I.getValue(I.getNodeFromName1(sw,'InwardIndex'))
            if ind[1] == 'i':
                SplitWindows_I += [ sw ]
                wnds_I += [ wnd ]
                inds_I += [ ind ]
            elif ind[1] == 'j':
                SplitWindows_J += [ sw ]
                wnds_J += [ wnd ]
                inds_J += [ ind ]
            else:
                raise ValueError('IndwardIndex "%s" not supported'%ind[1])

        # NOTE -> beware, it is not necessary !
        # if not _allEqual(inds_I) or not _allEqual(inds_J):
        #     C.convertPyTree2File(s,'debug.cgns')
        #     raise ValueError('incoherent InwardIndex for surface %s'%s[0])

        if len(inds_I) > 1:
            # imin = imax = wnds_I[0][0,0]
            imin = min( [w[0,0] for w in wnds_I] )
            imax = max( [w[0,1] for w in wnds_I] )
            jmin = min( [w[1,0] for w in wnds_I] )
            jmax = max( [w[1,1] for w in wnds_I] )
            newWnd_I = np.array([[imin, imax],[jmin,jmax]], order='F')
            for sw in SplitWindows_I: I.rmNode(trimInfo, sw)
            sw = I.createChild(trimInfo, 'SplitWindow.I', 'DataArray_t')
            I.createUniqueChild(sw, 'Window', 'DataArray_t', value=newWnd_I)
            I.createUniqueChild(sw, 'InwardIndex', 'DataArray_t', value=inds_I[0])

        if len(inds_J) > 1:
            # jmin = jmax = wnds_J[0][1,0]
            imin = min( [w[0,0] for w in wnds_J] )
            imax = max( [w[0,1] for w in wnds_J] )
            jmin = min( [w[1,0] for w in wnds_J] )
            jmax = max( [w[1,1] for w in wnds_J] )
            newWnd_J = np.array([[imin, imax],[jmin,jmax]], order='F')
            for sw in SplitWindows_J: I.rmNode(trimInfo, sw)
            sw = I.createChild(trimInfo, 'SplitWindow.J', 'DataArray_t')
            I.createUniqueChild(sw, 'Window', 'DataArray_t', value=newWnd_J)
            I.createUniqueChild(sw, 'InwardIndex', 'DataArray_t', value=inds_J[0])
        

def filterSurfacesUsingTag(surfaces):
    filteredSurfaces = []
    for s in I.getZones(surfaces):
        dim = I.getZoneDim(s)[4]
        try:
            trimInfo = I.getNodeFromName1(s,'.MOLA#Trim')
            tag_node = I.getNodeFromName1(trimInfo,'tag')
            tag = I.getValue(tag_node)
        except:
            if dim == 2: filteredSurfaces += [ s ]
            continue
        if tag != 'remove' and dim == 2: filteredSurfaces += [ s ]
    I._correctPyTree(filteredSurfaces, level=3)
    return filteredSurfaces


def migrateOffsetData(t, OffsetSurfaces):
    zones = I.getZones(t)
    names = ['Window', 'MinimumHeight', 'MaximumHeight', 'MeanHeight', 'MedianHeight']
    for surface in OffsetSurfaces:
        info = I.copyTree(I.getNodeFromName1(surface, '.MOLA#Offset'))
        zone_name = I.getValue(I.getNodeFromName1(info, 'Zone'))
        zone = _pickZoneFromName(zones, zone_name)
        zone_info = I.getNodeFromName1(zone, '.MOLA#Offset')
        if not zone_info:
            I.addChild(zone, info)
        else:
            for nodeName in names:
                node = I.getNodeFromName1(info, nodeName)
                I.addChild(zone_info, node)
    for z in zones:
        info = I.getNodeFromName1(z, '.MOLA#Offset')
        if not info: continue
        for name in names:
            nodes = I.getNodesByName(info,name+'*')
            for i, n in enumerate(nodes): n[0] = name+'.%d'%i
        

def extractWindows(t):
    surfaces = []
    for zone in I.getZones(t):
        info = I.getNodeFromName1(zone, '.MOLA#Offset')
        if not info: continue
        windows = I.getNodesByName(info, 'Window*')
        for window in windows:
            w = I.getValue( window )
            try:
                surface = T.subzone(zone,(w[0,0],w[1,0],w[2,0]),
                                         (w[0,1],w[1,1],w[2,1]))
            except TypeError:
                C.convertPyTree2File(zone,'debug.cgns')
                msg = 'failed extraction for zone %s'%(zone[0])
                raise ValueError(msg)
            surface[0] = 'window'
            I.addChild(surface,info)
            surfaces += [ surface ]
    I._correctPyTree(surfaces, level=3)
    return surfaces


def _extractSplitWindows(t):
    surfs = I.getZones(t)
    treeBuildList = []
    for s in surfs:
        SplitWindows = I.getNodesFromName(s,'SplitWindow*')
        if not SplitWindows: continue
        curves = []
        for sw in SplitWindows:
            wnd = I.getValue(I.getNodeFromName1(sw,'Window'))
            try:
                curve = T.subzone(s,(wnd[0,0],wnd[1,0],1),(wnd[0,1],wnd[1,1],1))
            except TypeError:
                Ni, Nj = I.getZoneDim(s)[1:3]
                slice=str([(wnd[0,0],wnd[1,0],1),(wnd[0,1],wnd[1,1],1)])
                msg = 'failed slice %s for surface %s with dims %dx%d'%(slice,s[0],Ni,Nj)
                raise ValueError(msg+' with wnd %s'%str(wnd))
            curves += [ curve ]
        treeBuildList.extend([s[0]+'.SW', curves])
    tOut = C.newPyTree(treeBuildList)
    I._correctPyTree(tOut,level=3)
    return tOut

def _allEqual(x):
    '''
    x is a python list
    '''
    if len(x) == 0: return True
    return x.count(x[0]) == len(x)

def _pickZoneFromName(zones, zone_name):
    for z in zones:
        if z[0] == zone_name:
            return z

def _includeMedianCellHeight(t, OffsetSurfaces):
    for s in OffsetSurfaces:
        info = I.getNodeFromName1(s,'.MOLA#Offset')
        if not info: continue
        zoneNode = I.getNodeFromName1(info,'Zone')
        zone = I.getNodeFromName3(t, I.getValue(zoneNode) )
        Nijk = I.getZoneDim(zone)[1:4]
        window = I.getNodeFromName(info, 'Window')
        
        w = np.copy(I.getValue(window), order='F')
        actual_surface = T.subzone(zone, (w[0,0],w[1,0],w[2,0]),
                                            (w[0,1],w[1,1],w[2,1]))
        ijk = findInwardsIndexFromExteriorWindow(w)[1]
        offset = w[ijk2ind[ijk], 0]
        if offset < Nijk[ijk2ind[ijk]]:
            w[ijk2ind[ijk], :] += 1
        else:
            w[ijk2ind[ijk], :] -= 1
        slice = (w[0,0],w[1,0],w[2,0]), (w[0,1],w[1,1],w[2,1])
    
        try:
            next_surface = T.subzone(zone, *slice) 
        except TypeError:
            C.convertPyTree2File(zone,'debug.cgns')
            msg = ('failed slice %s for surface %s with dims %dx%dx%d. '
                'Check debug.cgns')%(str(slice),zone[0],*Nijk)
            raise ValueError(msg)

        min,max,mean,median =_getStatisticPointwiseDistances(actual_surface,
                                                                next_surface)
        I.createUniqueChild(info,'MinimumHeight','DataArray_t',value=min)
        I.createUniqueChild(info,'MaximumHeight','DataArray_t',value=max)
        I.createUniqueChild(info,'MeanHeight','DataArray_t',value=mean)
        I.createUniqueChild(info,'MedianHeight','DataArray_t',value=median)
        

def _getStatisticPointwiseDistances(zone1, zone2):
    x1 = I.getNodeFromName3(zone1, 'CoordinateX')[1].ravel(order='F')
    y1 = I.getNodeFromName3(zone1, 'CoordinateY')[1].ravel(order='F')
    z1 = I.getNodeFromName3(zone1, 'CoordinateZ')[1].ravel(order='F')

    x2 = I.getNodeFromName3(zone2, 'CoordinateX')[1].ravel(order='F')
    y2 = I.getNodeFromName3(zone2, 'CoordinateY')[1].ravel(order='F')
    z2 = I.getNodeFromName3(zone2, 'CoordinateZ')[1].ravel(order='F')

    dist = np.sqrt((x2-x1)**2 + (y2-y1)**2 + (z2-z1)**2)

    min = np.min(dist)
    max = np.max(dist)
    mean = np.mean(dist)
    median = np.mean(dist)

    return min, max, mean, median
