'''
UnsteadyOverset

13/09/2022 - L. Bernardos - creation
'''

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


def setNeighbourListOfRotors(t, InputMeshes):
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

    for meshInfo in InputMeshes:
        if 'Motion' not in meshInfo: continue
        BaseName = meshInfo['baseName']
        
        print('computing intersection between boxes from base %s...'%BaseName)
        NewNeighbourList= _findNeighbourListOfBase(BaseName, aabb, obb)
        print('updating NeighbourList of base %s...'%BaseName)
        _updateNeighbourListOfBase(BaseName, t, NewNeighbourList)


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
        newExtFaces = I.getZones(P.exteriorFaces(z))
        for nz in newExtFaces: nz[0] = z[0]
        exteriorFaces.extend(newExtFaces)
    T._scale(exteriorFaces,scale)

    Azimuts = []
    if Dpsi > 0:
        for i in range(int(360/Dpsi)):
            Azimuts.extend(T.rotate(exteriorFaces,rot_ctr,rot_axis,i*Dpsi))
    else:
        # fixed component but with unsteady treatment
        Azimuts = I.getZones(base)
    base[2] = Azimuts


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


def _updateNeighbourListOfBase(BaseName, t, NewNeighbourList):
    
    if not NewNeighbourList: return
    base = I.getNodeFromName2(t, BaseName)
    NeighbourListNodes = I.getNodesFromName(base,'NeighbourList')

    AmountOfNeighbourListNodes = len(NeighbourListNodes)
    if AmountOfNeighbourListNodes != 1:
        msg = ('found %d NeighbourList nodes in base "%s", '
              'but exactly 1 was expected')%(AmountOfNeighbourListNodes,BaseName)
        raise ValueError(msg)

    NeighbourListNode = NeighbourListNodes[0]
    NeighbourList = I.getValue(NeighbourListNode)
    
    if NeighbourList is None:
        NeighbourList = []
    else:
        NeighbourList = NeighbourList.split(' ')

    for NewNeighbour in NewNeighbourList:
        if NewNeighbour not in NeighbourList:
            NeighbourList.append( NewNeighbour )
    I.setValue(NeighbourListNode, ' '.join(NeighbourList))


# def addMaskWindowLocation(t, mask, parent_node):
#     '''
#     Attach ``.MOLA#Mask`` node to *parent_node*, containing windows location 
#     of the mask by means of point ranges (structured) or point lists
#     (unstructured).

#     Parameters
#     ----------

#         t : PyTree
#             full grid, where mask is applied

#         mask : PyTree, base, zone or list of zones
#             geometrical surfaces of the mask, as obtained for example 
#             using :py:func:`MOLA.ExtractSurfacesProcessor.extractSurfacesByOffsetCellsFromBCFamilyName`

#             .. important::
#                 the mask must exactly match grid points of **t**.

#         parent_node : node
#             CGNS node where ``.MOLA#Mask`` will be attached. It is recommended 
#             to be a relevant *Family_t*.
#     '''
