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
UnsteadyOverset

13/09/2022 - L. Bernardos - creation
'''

import MOLA

if not MOLA.__ONLY_DOC__:
    from ctypes import alignment
    from multiprocessing.sharedctypes import Value
    import os
    import numpy as np
    from itertools import product
    from timeit import default_timer as ToK

    import Converter.PyTree as C
    import Converter.Internal as I
    import Connector.PyTree as X
    import Transform.PyTree as T
    import Generator.PyTree as G
    import Post.PyTree as P
    import Geom.PyTree as D
    cost_est = np.array([0.0])

from . import InternalShortcuts as J
from . import GenerativeShapeDesign as GSD
from . import GenerativeVolumeDesign as GVD
from . import ExtractSurfacesProcessor as ESP
from . import JobManager as JM
from .Preprocess import getBodyName

MOLA_MASK = 'mask'


def setMaskParameters(t, InputMeshes):
    mask_params = dict(type='cart_elts', blanking_criteria='center_in',
                       dim1=100, dim2=100)
    
    default_mask_axis = np.array([0.0, 0.0, 1.0])
    for base in I.getBases(t):
        MasksInfo = I.getNodeFromName1(base,'.MOLA#Masks')
        if not MasksInfo: continue
        try:
            default_mask_axis = np.array(meshInfo['Motion']['RequestedFrame']['RotationAxis'],dtype=float)
            break
        except:
            pass



    for base in I.getBases(t):
        MasksInfo = I.getNodeFromName1(base,'.MOLA#Masks')
        if not MasksInfo: continue
        for mask in MasksInfo[2]:
            if not mask[0].startswith(MOLA_MASK): continue
            # we suppose all patches belong to the same base
            mask_zone = I.getNodeFromName2(mask,'Zone')
            MaskBase = J._getBaseWithZoneName(t, I.getValue(mask_zone))
            meshInfo = [m for m in InputMeshes if m['baseName']==MaskBase[0]][0]
            if 'UnsteadyMaskOptions' in meshInfo['OversetOptions']:
                mask_params.update(meshInfo['OversetOptions']['UnsteadyMaskOptions'])
            if mask_params['type'] == 'cart_elts' and 'proj_direction' not in mask_params:
                if 'Motion' in meshInfo:
                    a = np.array(meshInfo['Motion']['RequestedFrame']['RotationAxis'],dtype=float)
                else: 
                    a = default_mask_axis
                alignment = [np.abs(a.dot(np.array([1.0,0.0,0.0]))),
                             np.abs(a.dot(np.array([0.0,1.0,0.0]))),
                             np.abs(a.dot(np.array([0.0,0.0,1.0])))]
                mask_params['proj_direction'] = 'xyz'[np.argmax(alignment)]

            J.set(mask,'Parameters',**mask_params)

def setMaskedZonesOfMasks(t, InputMeshes, BlankingMatrix, BodyNames):
    '''
    Set the ``MaskedZones`` node of each base by considering the rotation of 
    the component. This information is required for unsteady masking.
    

    Parameters
    ----------

        t : PyTree
            assembled tree including BC overlaps

            .. note::
                **t** is modified

        InputMeshes : :py:class:`list` of :py:class:`dict`
            user-provided preprocessing instructions as described in
            :py:func:`prepareMesh4ElsA` .

            Component-specific instructions for unsteady overlap settings are
            provided through **InputMeshes** component by means of keyword
            ``Motion``, which accepts a Python dictionary with several allowed
            pairs of keywords and their associated values:

            * ``'RotationCenter'`` : :py:class:`list` of 3 :py:class:`float`
                the three :math:`(x,y,z)` coordinates of the rotation center 
                of the component

            * ``'RotationAxis'`` : :py:class:`list` of 3 :py:class:`float`
                the vector direction :math:`(x,y,z)` of the rotation of the
                component

            * ``'NeighbourSearchScaleFactor'`` : :py:class:`float`
                A scaling factor used for the neigbour search. Use big enough
                value so that maximum expected mesh deformation is correctly 
                accounted for. (default value is **1.2**)

            * ``'NeighbourSearchAzimutResolution'`` : :py:class:`float`
                The azimuthal spatial step for the neighbour search, in degrees.
                (default value is **5**)

                .. tip:: 
                    use a value of ``0`` if you want to make unsteady treatment 
                    of a fixed component
    '''

    
    tR = I.copyRef(t)

    for meshInfo in InputMeshes:
        if 'Motion' not in meshInfo: continue
        BaseName = meshInfo['baseName']
        base = I.getNodeFromName2(tR,BaseName)

        rot_ctr, rot_axis, scale, Dpsi = _getRotorMotionParameters(meshInfo)
        _addAzimuthalGhostComponent(base, rot_ctr, rot_axis, scale, Dpsi)

    print('building trees of boxes for intersection estimation...')
    aabb, obb = _buildBoxesTrees(tR)

    print('estimating intersections...')
    IntersectingZones = _getIntersectingZones(aabb, obb)

    NeighbourDict = {}
    for meshInfo in InputMeshes:
        BaseName = meshInfo['baseName']
        
        try: is_duplicated = bool(meshInfo['DuplicatedFrom'] != base[0])
        except KeyError: is_duplicated = False

        print('computing intersection between boxes from base %s...'%BaseName)
        MaskedZones = _findMaskedZonesOfBase(BaseName, t, IntersectingZones)
        NeighbourDict[BaseName] = MaskedZones
    
    _updateMaskedZonesOfMasks(t, NeighbourDict, BlankingMatrix, BodyNames)

def removeOversetHolesOfUnsteadyMaskedGrids(t):
    zoneNames_to_process = []
    for base in I.getBases(t):
        MasksContainer = I.getNodeFromName1(base, '.MOLA#Masks')
        if not MasksContainer: continue
        MaskedZones = I.getNodesFromName2(MasksContainer, 'MaskedZones')
        for nl in MaskedZones:
            Neighbours = I.getValue(nl).split(' ')
            for n in Neighbours:
                zone_name = n.split('/')[1]
                if zone_name not in zoneNames_to_process:
                    zoneNames_to_process += [ zone_name ]
    
    zones = [ z for z in I.getZones(t) if z[0] in zoneNames_to_process ]
    if zones: I._rmNodesByName(zones,'OversetHoles')
    for z in I.getZones(t):
        gc = I.getNodeFromType1(z, 'ZoneGridConnectivity_t')
        if not gc: continue
        if gc and not gc[2]: I.rmNode(t, gc)


def _getRotorMotionParameters(meshInfo):
    try: rot_ctr = tuple(meshInfo['Motion']['RequestedFrame']['RotationCenter'])
    except KeyError: rot_ctr = (0,0,0)

    try: rot_axis = tuple(meshInfo['Motion']['RequestedFrame']['RotationAxis'])
    except KeyError: rot_axis = (0,0,1)

    try: scale_factor = meshInfo['Motion']['RequestedFrame']['NeighbourSearchScaleFactor']
    except KeyError: scale_factor = 1.2

    try: Dpsi = meshInfo['Motion']['RequestedFrame']['NeighbourSearchAzimutResolution']
    except KeyError: Dpsi = 15.0 

    return rot_ctr, rot_axis, scale_factor, Dpsi


def _addAzimuthalGhostComponent(base, rot_ctr, rot_axis, scale, Dpsi):
    exteriorFaces = []
    for z in I.getZones(base):
        dims = I.getZoneDim(z)[4]
        if dims == 3:
            newExtFaces = I.getZones(P.exteriorFaces(z))
            for nz in newExtFaces: nz[0] = z[0]
            exteriorFaces.extend(newExtFaces)
        elif dims == 2:
            exteriorFaces.append( z )
        else:
            raise ValueError('cannot make azimuthal ghost component of 1D grid')
    T._scale(exteriorFaces,scale)

    newAzimuts = []
    Azimuts = []
    if Dpsi > 0:
        newAzimuts = []
        for ef in exteriorFaces:
            for i in range(int(360/Dpsi)):
                newAzimuts += [ T.rotate(ef,rot_ctr,rot_axis,i*Dpsi) ]
                newAzimuts[-1][0] = ef[0]
            
            xmin, ymin, zmin, xmax, ymax, zmax = G.bbox(newAzimuts)
            az = G.cart((xmin,ymin,zmin),(xmax-xmin,ymax-ymin,zmax-zmin),(2,2,2))
            az[0] = ef[0]
            Azimuts += [az]
    else:
        # fixed component but with unsteady treatment
        Azimuts = I.getZones(base)
        
    if base[3] == 'CGNSBase_t' and Azimuts: base[2] = Azimuts
    
    return newAzimuts



def _findMaskedZonesOfBase(BaseName, t, IntersectingZones):

    base = I.getNodeFromName2(t, BaseName)
    NewMaskedZones = []
    for zone in I.getZones(base):
        try: LocalIntersectingZones = IntersectingZones[ zone[0] ]
        except KeyError: continue
        for IntersectingZone in LocalIntersectingZones:
            BaseNameOfIntersectingZone = J._getBaseWithZoneName(t,IntersectingZone)[0]
            if BaseNameOfIntersectingZone == BaseName: continue
            BaseNameAndZoneName = BaseNameOfIntersectingZone + '/' + IntersectingZone
            if BaseNameAndZoneName not in NewMaskedZones:
                NewMaskedZones += [ BaseNameAndZoneName ]
    NewMaskedZones.sort()
    return NewMaskedZones

def _getIntersectingZones(aabb, obb):
    TiK = ToK()
    zones_aabb = I.getNodesFromType2(aabb,'Zone_t')
    zones_obb = I.getNodesFromType2(obb,'Zone_t')
    nb_zones = len(zones_aabb)
    Xmatrix = computeRoughIntersectionMatrix(aabb)

    
    for i in range(nb_zones):
        z_aabb0 = zones_aabb[i]
        z_obb0 = zones_obb[i]
        baseName0 = _getBaseNameFromUserData(z_aabb0)
        for j in range(nb_zones):
            if i==j:
                Xmatrix[i,j] = Xmatrix[j,i] = 0
                continue

            z_aabb = zones_aabb[j]
            z_obb = zones_obb[j]
            if Xmatrix[j,i] == 0:
                Xmatrix[i,j] = 0 
                continue

            baseName1 = _getBaseNameFromUserData(z_aabb)

            if baseName0 == baseName1:
                Xmatrix[i,j] = Xmatrix[j,i] = 0
                continue

            if not G.bboxIntersection(z_aabb0, z_aabb, tol=1e-8, isBB=True, method='AABB'):
                Xmatrix[i,j] = Xmatrix[j,i] = 0
                continue

            if not G.bboxIntersection(z_aabb, z_obb0, tol=1e-8, isBB=True, method='AABBOBB'):
                Xmatrix[i,j] = Xmatrix[j,i] = 0
                continue

            if not G.bboxIntersection(z_aabb0, z_obb, tol=1e-8, isBB=True, method='AABBOBB'):
                Xmatrix[i,j] = Xmatrix[j,i] = 0
                continue
                    
            if not G.bboxIntersection(z_obb0, z_obb, tol=1e-8, isBB=True, method='OBB'):
                Xmatrix[i,j] = Xmatrix[j,i] = 0
                continue

    IntersectingZones = dict()
    for i in range(nb_zones):
        nameI = zones_aabb[i][0]
        IntersectingZones[nameI] = []
        for j in range(nb_zones):
            if Xmatrix[i,j] == 1:
                nameJ = zones_aabb[j][0]
                IntersectingZones[nameI] += [ nameJ ]

    cost_est[0] += ToK()-TiK
    print('intersections estimation cost: %g s'%cost_est)

    return IntersectingZones



def buildSpheres(t):
    x, y, z, r = [], [], [], []
    for zone in I.getNodesFromType2(t, 'Zone_t'):
        barycenter = G.barycenter(zone)
        xmin, ymin, zmin, xmax, ymax, zmax = G.bbox(zone)
        x += [ barycenter[0] ]
        y += [ barycenter[1] ]
        z += [ barycenter[2] ]
        r += [ (xmax-xmin)**2 + (ymax-ymin)**2 + (zmax-zmin)**2 ]
    x = np.array(x,order='F')
    y = np.array(y,order='F')
    z = np.array(z,order='F')
    r = np.sqrt(0.5*np.array(r,order='F'))
    N = len(x)
    spheres = I.newZone(name='spheres', zsize=[[N,N-1,0]], ztype='Structured')
    gc = I.newGridCoordinates(parent=spheres)
    I.newDataArray('CoordinateX', value=x, parent=gc)
    I.newDataArray('CoordinateY', value=y, parent=gc)
    I.newDataArray('CoordinateZ', value=z, parent=gc)
    fs = I.newFlowSolution(parent=spheres)
    I.newDataArray('radius', value=r, parent=fs)

    return spheres

def computeRoughIntersectionMatrix(t):
    from scipy.spatial import distance
    spheres = buildSpheres(t)
    x,y,z = J.getxyz(spheres)
    r = J.getVars(spheres,['radius'])[0]
    ri_plus_rj = r[:,None] + r[None,:]
    xyz = np.vstack((x,y,z)).T
    distanceMatrix = distance.cdist(xyz,xyz,'euclidean')
    X = np.array(distanceMatrix < ri_plus_rj, dtype=int)

    return X
    

def _getBaseNameFromUserData(zone):
    data = I.getNodeFromName1(zone, '.MOLA#BoxData')
    return I.getValue(I.getNodeFromName1(data, 'ParentBaseName'))

def _buildBoxesTrees(tR):

    print('---> building exterior faces')
    AllExteriorFaces = dict()
    for b in I.getBases(tR):
        AllExteriorFaces[b[0]] = dict()
        for i, z in enumerate(b[2]):
            if z[3] != 'Zone_t': continue
            dims = I.getZoneDim(z)[4]
            exterior = P.exteriorFaces(z) if dims == 3 else z
            AllExteriorFaces[b[0]][i] = exterior

    print('---> building axis-aligned bounding-boxes (AABB)')
    tR_aabb = I.copyRef(tR)
    for b in I.getBases(tR_aabb):
        for i, z in enumerate(b[2]):
            if z[3] != 'Zone_t': continue
            exterior = AllExteriorFaces[b[0]][i]
            AllExteriorFaces[b[0]][i] = exterior
            aabb = G.BB(exterior,method='AABB')
            J.set(aabb,'.MOLA#BoxData',ParentBaseName=b[0])
            aabb[0] = z[0]
            b[2][i] = aabb

    print('---> building oriented bounding-boxes (OBB)')
    tR_obb = I.copyRef(tR)
    for b in I.getBases(tR_obb):
        for i, z in enumerate(b[2]):
            if z[3] != 'Zone_t': continue
            exterior = AllExteriorFaces[b[0]][i]
            obb = G.BB(exterior,method='OBB')
            J.set(obb,'.MOLA#BoxData',ParentBaseName=b[0])
            obb[0] = z[0]
            b[2][i] = obb

    return tR_aabb, tR_obb


def _updateMaskedZonesOfMasks(t, NeighbourDict, BlankingMatrix, BodyNames):
    if not NeighbourDict: return
    for base in I.getBases(t):
        print(f'updating MaskedZones of MOLA#Masks at base {base[0]}')
        MasksInfo = I.getNodeFromName1(base,'.MOLA#Masks')
        if not MasksInfo: continue
        nodes_2_remove = []
        for mask in MasksInfo[2]:
            mask_name = I.getValue(mask)
            if not mask[0].startswith(MOLA_MASK): continue
            BodyName = mask_name.split('_by_')[-1]
            # we suppose all patches belong to the same base
            mask_zone = I.getNodeFromName2(mask,'Zone')
            if not mask_zone: continue
            MaskBaseName = J._getBaseWithZoneName(t, I.getValue(mask_zone))[0]
            j = _getBodyNumber(BodyName, BodyNames)
            Neighbours = []
            for baseZonePath in NeighbourDict[MaskBaseName]:
                baseNameOfNeighbour = baseZonePath.split('/')[0]
                i = _getBaseNumber(baseNameOfNeighbour, t)
                if BlankingMatrix[i,j] and baseNameOfNeighbour == base[0]:
                    Neighbours += [ baseZonePath ]
            if not Neighbours:
                msg = f'skip mask since {base[0]} does not intersect {MaskBaseName} at {BodyName}'
                print(J.WARN+msg+J.ENDC)
                nodes_2_remove += [mask]
            else:
                I.createUniqueChild(mask,'MaskedZones','DataArray_t',
                                    value=' '.join( Neighbours ))
        for n in nodes_2_remove: I._rmNode(base, n)

def _getBodyNumber(BodyName, BodyNames):
    for j, bn in enumerate(BodyNames):
        if BodyName == bn: return j
    raise ValueError('body name %s not contained in %s'%(BodyName,str(BodyNames)))
    
def _getBaseNumber(BaseName, t):
    for i, b in enumerate(I.getBases(t)):
        if b[0] == BaseName: return i
    raise ValueError('BaseName %s not found'%(BaseName))

def addMaskData(t, InputMeshes, bodies, BlankingMatrix):
    for i, meshInfo in enumerate(InputMeshes):
        baseName = meshInfo['baseName']
        base = _getBaseFromName(t, baseName)
        MaskBodies = [ b for j, b in enumerate(bodies) if BlankingMatrix[i,j] ]
        if not MaskBodies: continue
        AllMasksContainer = I.createUniqueChild(base, '.MOLA#Masks',
                                                'UserDefinedData_t')
        for body in MaskBodies:
            maskName = base[0] + '_by_' + getBodyName(body)
            mask = I.createChild(AllMasksContainer, MOLA_MASK,
                                'UserDefinedData_t', value=maskName)
            if isinstance(body[0], str): body = [body]
            for k, patch in enumerate(body):
                offset = I.getNodeFromName1(patch,'.MOLA#Offset')
                if not offset:
                    msg = '.MOLA#Offset not found at %s'%maskName
                    print(J.WARN+msg+J.ENDC)
                    continue
                offset = I.copyTree(offset)
                offset = I.addChild(mask, offset)
                offset[0] = 'patch.%d'%k
        
        for l, mask in enumerate(AllMasksContainer[2]):
            mask[0]=MOLA_MASK+'.%d'%l
            
def _getBaseFromName(t, base_name):
    bases = I.getBases(t)
    for b in bases:
        if b[0] == base_name: 
            return b
