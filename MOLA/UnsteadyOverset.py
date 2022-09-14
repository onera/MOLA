'''
UnsteadyOverset

13/09/2022 - L. Bernardos - creation
'''

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
    
    .. warning::
        Currently available for structured meshes only.

    .. warning::
        this function does not currently consider cross intersections between 
        dynamic components

    Parameters
    ----------

        t : PyTree
            assembled tree including BC overlaps

            .. note::
                **t** is modified

        InputMeshes : :py:class:`list` of :py:class:`dict`
            user-provided preprocessing instructions as described in
            :py:func:`prepareMesh4ElsA` .

            Component-specific instructions for unsteady overlap settings are provided
            through **InputMeshes** component by means of keyword ``Motion``, which
            accepts a Python dictionary with several allowed pairs of keywords and their
            associated values:

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
                (default value is **1**)

                .. tip:: 
                    use a value of ``0`` if you want to make unsteady treatment 
                    of a fixed component
    '''

    
    tR = I.copyRef(t)

    for meshInfo in InputMeshes:
        if 'Motion' not in meshInfo: continue
        BaseName = meshInfo['baseName']
        base = I.getNodeFromName2(tR,BaseName)
        I.rmNode(tR,base)


        try: rot_ctr = tuple(meshInfo['Motion']['RotationCenter'])
        except KeyError: rot_ctr = (0,0,0)

        try: rot_axis = tuple(meshInfo['Motion']['RotationAxis'])
        except KeyError: rot_axis = (0,0,1)

        try: scale_factor = meshInfo['Motion']['NeighbourSearchScaleFactor']
        except KeyError: scale_factor = 1.2

        try: Dpsi = meshInfo['Motion']['NeighbourSearchAzimutResolution']
        except KeyError: Dpsi = 1.0

        exteriorFaces = I.getZones(P.exteriorFacesStructured(base))
        T._scale(exteriorFaces,scale_factor)

        Azimuts = []
        if Dpsi > 0:
            for i in range(1,int(360/Dpsi)):
                Azimuts.extend( T.rotate(exteriorFaces,rot_ctr,rot_axis,i*Dpsi) )
        else:
            Azimuts = I.getZones(exteriorFaces)
        tAzms = C.newPyTree([BaseName, Azimuts])
        I._correctPyTree(tAzms,level=3)
        baseAzms = I.getBases(tAzms)[0]
        bases_tR = I.getBases(tR)
        
        domains = X.getCEBBIntersectingDomains(baseAzms, bases_tR, 0)
        IntersectingZones = []
        for domain in domains:
            for zoneName in domain:
                if zoneName not in IntersectingZones:
                    IntersectingZones.append(zoneName)
        IntersectingZones.sort()
        IntersectingZonesBaseNames = []
        for zoneName in IntersectingZones:
            for base in bases_tR:
                hasZone = I.getNodeFromName1(base,zoneName)
                if hasZone and base[0] not in IntersectingZonesBaseNames:
                    IntersectingZonesBaseNames.append(base[0])
                    

        XFamily = 'X_'+BaseName
        XFamilyWithBases = [ xb+'/'+XFamily for xb in IntersectingZonesBaseNames ]
        for zoneName in IntersectingZones:
            zone = I.getNodeFromName3(t, zoneName)
            C._tagWithFamily(zone,XFamily, add=True)
        
        for xbasename in IntersectingZonesBaseNames:
            xbase = I.getNodeFromName1(t,xbasename)
            C._addFamily2Base(xbase, XFamily)

        base = I.getNodeFromName2(t, BaseName)
        NeighbourListNode = I.getNodeFromName(base,'NeighbourList')
        NeighbourList = I.getValue(NeighbourListNode)
        XFamilyWithBasesStr = ' '.join(XFamilyWithBases)
        if NeighbourList is None:
            I.setValue(NeighbourListNode,XFamilyWithBasesStr)
        else:
            I.setValue(NeighbourListNode,
                       ' '.join(NeighbourList+' '+XFamilyWithBasesStr))
    C._tagWithFamily(t,'ALLZONES', add=True)
    [C._addFamily2Base(b, XFamily) for b in I.getBases(t)]