'''
UnsteadyOverset

13/09/2022 - L. Bernardos - creation
'''

import os
import numpy as np

import Converter.PyTree as C
import Converter.Internal as I
import Connector.PyTree as X
import Transform.PyTree as T
import Generator.PyTree as G
import Post.PyTree as P

from . import InternalShortcuts as J
from . import GenerativeShapeDesign as GSD
from . import GenerativeVolumeDesign as GVD
from . import ExtractSurfacesProcessor as ESP
from . import JobManager as JM
from .Preprocess import getBodyName

MOLA_MASK = 'mask'

def setMaskParameters(t, InputMeshes):
    mask_params = dict(type='cart_elts', proj_direction='z', dim1=100, dim2=100,
                       blanking_criteria='center_in')
    for base in I.getBases(t):
        MasksInfo = I.getNodeFromName1(base,'.MOLA#Masks')
        if not MasksInfo: continue
        for mask in MasksInfo[2]:
            if not mask[0].startswith(MOLA_MASK): continue
            # we suppose all patches belong to the same base
            mask_zone = I.getNodeFromName2(mask,'Zone')
            if not mask_zone:
                C.convertPyTree2File(base,'debug.cgns')
                raise ValueError('inexistent Zone node at mask %s'%os.path.join(
                        base[0],MasksInfo[0],mask[0]))
            MaskBase = J._getBaseWithZoneName(t, I.getValue(mask_zone))
            meshInfo = [m for m in InputMeshes if m['baseName']==MaskBase[0]][0]
            if 'UnsteadyMaskOptions' in meshInfo['OversetOptions']:
                mask_params.update(meshInfo['OversetOptions']['UnsteadyMaskOptions'])
            J.set(mask,'Parameters',**mask_params)

def setNeighbourListOfMasks(t, InputMeshes, BlankingMatrix, BodyNames):
    '''
    Set the ``NeighbourList`` node of each base by considering the rotation of 
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
                The azimutal spatial step for the neighbour search, in degrees.
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
        _addAzimutalGhostComponent(base, rot_ctr, rot_axis, scale, Dpsi)

    print('building trees of boxes for intersection estimation...')
    aabb, obb = _buildBoxesTrees(tR)

    NeighbourDict = {}
    for meshInfo in InputMeshes:
        BaseName = meshInfo['baseName']
        
        print('computing intersection between boxes from base %s...'%BaseName)
        NeighbourList = _findNeighbourListOfBase(BaseName, aabb, obb)
        NeighbourDict[BaseName] = NeighbourList
    
    print('updating NeighbourList of MOLA#Masks at base %s...'%BaseName)
    _updateNeighbourListOfMasks(t, NeighbourDict, BlankingMatrix, BodyNames)



def removeOversetHolesOfUnsteadyMaskedGrids(t):
    zoneNames_to_process = []
    for base in I.getBases(t):
        MasksContainer = I.getNodeFromName1(base, '.MOLA#Masks')
        if not MasksContainer: continue
        NeighbourLists = I.getNodesFromName2(MasksContainer, 'NeighbourList')
        for nl in NeighbourLists:
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
    try: rot_ctr = tuple(meshInfo['Motion']['RotationCenter'])
    except KeyError: rot_ctr = (0,0,0)

    try: rot_axis = tuple(meshInfo['Motion']['RotationAxis'])
    except KeyError: rot_axis = (0,0,1)

    try: scale_factor = meshInfo['Motion']['NeighbourSearchScaleFactor']
    except KeyError: scale_factor = 1.2

    try: Dpsi = meshInfo['Motion']['NeighbourSearchAzimutResolution']
    except KeyError: Dpsi = 5.0    

    return rot_ctr, rot_axis, scale_factor, Dpsi


def _addAzimutalGhostComponent(base, rot_ctr, rot_axis, scale, Dpsi):
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
            raise ValueError('cannot make azimutal ghost component of 1D grid')
    T._scale(exteriorFaces,scale)

    Azimuts = []
    if Dpsi > 0:
        for i in range(int(360/Dpsi)):
            Azimuts.extend(T.rotate(exteriorFaces,rot_ctr,rot_axis,i*Dpsi))
    else:
        # fixed component but with unsteady treatment
        Azimuts = I.getZones(base)
    
    if base[3] == 'CGNSBase_t': base[2] = Azimuts
    
    return Azimuts


def _findNeighbourListOfBase(BaseName, t_aabb, t_obb):

    base_aabb = I.getNodeFromName2(t_aabb, BaseName)
    base_obb = I.getNodeFromName2(t_obb, BaseName)


    NewNeighbourList = []
    for z_aabb0, z_obb0 in zip(I.getZones(base_aabb),I.getZones(base_obb)):
        for b, bo in zip(I.getBases(t_aabb), I.getBases(t_obb)):
            if b[0] == BaseName: continue
            for z_aabb, z_obb in zip(I.getZones(b), I.getZones(bo)):
                if not G.bboxIntersection(z_aabb0, z_aabb,
                                          tol=1e-8, isBB=True, method='AABB'):
                    continue
                if not G.bboxIntersection(z_aabb, z_obb0,
                                          tol=1e-8, isBB=True, method='AABBOBB'):
                    continue
                if not G.bboxIntersection(z_aabb0, z_obb,
                                          tol=1e-8, isBB=True, method='AABBOBB'):
                    continue
                if not G.bboxIntersection(z_obb0, z_obb,
                                          tol=1e-8, isBB=True, method='OBB'):
                    continue
                
                NewNeighbour = b[0] + '/' + z_aabb[0]

                if NewNeighbour not in NewNeighbourList:
                    NewNeighbourList += [ NewNeighbour ]
    NewNeighbourList.sort()
    return NewNeighbourList


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
            aabb[0] = z[0]
            b[2][i] = aabb

    print('---> building oriented bounding-boxes (OBB)')
    tR_obb = I.copyRef(tR)
    for b in I.getBases(tR_obb):
        for i, z in enumerate(b[2]):
            if z[3] != 'Zone_t': continue
            exterior = AllExteriorFaces[b[0]][i]
            obb = G.BB(exterior,method='OBB')
            obb[0] = z[0]
            b[2][i] = obb

    return tR_aabb, tR_obb


def _updateNeighbourListOfMasks(t, NeighbourDict, BlankingMatrix, BodyNames):
    if not NeighbourDict: return
    for base in I.getBases(t):
        MasksInfo = I.getNodeFromName1(base,'.MOLA#Masks')
        if not MasksInfo: continue
        for mask in MasksInfo[2]:
            if not mask[0].startswith(MOLA_MASK): continue
            BodyName = I.getValue(mask).split('_by_')[-1]
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
            I.createUniqueChild(mask,'NeighbourList','DataArray_t',
                                    value=' '.join( Neighbours ))

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
