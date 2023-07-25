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
MOLA - InternalShortcuts.py

This module defines some handy shortcuts of the Cassiopee's
Converter.Internal module.

To complete sphinx doc

First creation:
27/02/2019 - L. Bernardos
'''

import sys
usingPython2 = sys.version_info[0] == 2
import os

import MOLA
from . import __version__, __MOLA_PATH__
if not MOLA.__ONLY_DOC__:
    import threading
    import time
    import glob
    import numpy as np
    import pprint
    from itertools import product
    from timeit import default_timer as tic
    from fnmatch import fnmatch

    import Converter.PyTree as C
    import Converter.Internal as I

FAIL  = '\033[91m'
GREEN = '\033[92m'
WARN  = '\033[93m'
MAGE  = '\033[95m'
CYAN  = '\033[96m'
ENDC  = '\033[0m'
BOLD = '\033[1m'
UNDERLINE = '\033[4m'

# ref: https://github.com/CGNS/CGNS/blob/develop/src/cgnslib.h#L510
element_types = [
    'Null', 'UserDefined',
    'NODE', 'BAR_2', 'BAR_3', 'TRI_3', 'TRI_6', 'QUAD_4', 'QUAD_8', 'QUAD_9',
    'TETRA_4', 'TETRA_10', 'PYRA_5', 'PYRA_14', 'PENTA_6', 'PENTA_15',
    'PENTA_18', 'HEXA_8', 'HEXA_20', 'HEXA_27', 'MIXED', 'PYRA_13',
    'NGON_n', 'NFACE_n', 'BAR_4', 'TRI_9', 'TRI_10', 'QUAD_12', 'QUAD_16',
    'TETRA_16', 'TETRA_20', 'PYRA_21', 'PYRA_29', 'PYRA_30', 'PENTA_24',
    'PENTA_38', 'PENTA_40', 'HEXA_32', 'HEXA_56', 'HEXA_64', 'BAR_5', 'TRI_12',
    'TRI_15', 'QUAD_P4_16', 'QUAD_25', 'TETRA_22', 'TETRA_34', 'TETRA_35',
    'PYRA_P4_29', 'PYRA_50', 'PYRA_55', 'PENTA_33', 'PENTA_66', 'PENTA_75',
    'HEXA_44', 'HEXA_98', 'HEXA_125']


def set(parent, childname, childType='UserDefinedData_t', **kwargs):
    '''
    Set (or add, if inexistent) a child node containing an arbitrary number
    of nodes.

    Return the pointers of the new CGNS nodes as a python dictionary get()

    Parameters
    ----------

        parent : node
            root node where children will be added

        childname : str
            name of the new child node.

        **kwargs
            Each pair *name* = **value** will be a node of
            type `DataArray_t`_ added as child to the node named *childname*.
            If **value** is a python dictionary, then their contents are added
            recursively following the same logic

    Returns
    -------

        pointers : dict
            literally, result of :py:func:`get` once all nodes have been
            added
    '''
    children = []
    SubChildren = []
    SubSet = []
    for v in kwargs:
        if isinstance(kwargs[v], dict):
            SubChildren += [[v,kwargs[v]]]
        elif isinstance(kwargs[v], list):
            if len(kwargs[v])>0:
                if isinstance(kwargs[v][0], str):
                    value = ' '.join(kwargs[v])
                elif isinstance(kwargs[v][0], dict):
                    p = I.createNode(v, childType)
                    for i in range(len(kwargs[v])):
                        set(p, 'set.%d'%i, **kwargs[v][i])
                    SubSet += [p]
                    continue
                else:
                    try:
                        value = np.atleast_1d(kwargs[v])
                        if value.dtype == 'O': 
                            # 'O' for 'Object'. It generally means that kwargs[v] contains mixed values
                            value = None
                    except ValueError:
                        value = None

            else:
                value = None
            children += [[v,value]]
        else:
            children += [[v,kwargs[v]]]
    _addSetOfNodes(parent,childname,children, type1=childType)
    NewNode = I.getNodeFromName1(parent,childname)
    NewNode[2].extend(SubSet)
    for sc in SubChildren: set(NewNode, sc[0], **sc[1])

    return get(parent, childname)


def get(parent, childname):
    '''
    Recover the name and values of children of a node named *childname* inside a
    *parent* node. Such pair of name and values are
    recovered as a python dictionary:
    Dict[*nodename*] = **nodevalue**

    Parameters
    ----------
        parent : node
            the CGNS node where the child named *childname* is found
        childname : str
            a child node name contained in node *parent*, from which children
            are extracted. This operation is recursive.

    Returns
    -------
        pointers - (dict):
            A dictionary Dict[*nodename*] = **nodevalue**

    See Also
    --------
    set : set a CGNS node containing children
    '''

    child_n = I.getNodeFromName1(parent,childname)
    Dict = {}
    if child_n is not None:
        for n in child_n[2]:
            if n[1] is not None:
                if isinstance(n[1], float) or isinstance(n[1], int):
                    Dict[n[0]] = np.atleast_1d(n[1])
                elif n[1].dtype == '|S1':
                    Dict[n[0]] = I.getValue(n) # Cannot further modify
                else:
                    Dict[n[0]] = np.atleast_1d(n[1]) # Can further modify
            elif n[2]:
                Dict[n[0]] = get(child_n, n[0])
            else:
                Dict[n[0]] = None
    return Dict



def getVars(zone, VariablesName, Container='FlowSolution'):
    """
    Get the list of numpy arrays from a *zone* of the variables
    specified in *VariablesName*.

    Parameters
    ----------

        zone : zone
            The CGNS zone from which numpy arrays are being retreived

        VariablesName : :py:class:`list` of :py:class:`str`
            List of the field names to be retreived

        Container : str
            The name of the node to look for the requested variable
            (e.g. ``'FlowSolution'``). Container should be at 1 depth level
            inside zone.

    Returns
    -------
        numpies : list of numpy.ndarray
            If a variable is not found, :py:obj:`None` is returned by the function.

    Examples
    --------
    ::

        import Converter.PyTree as C
        import Generator.PyTree as G
        import MOLA.InternalShortcuts as J

        zone = G.cart((0,0,0),(1,1,1),(3,3,3))

        C._initVars(zone,'ViscosityMolecular',1.78938e-5)
        C._initVars(zone,'Density',1.225)

        mu, rho = J.getVars(zone,['ViscosityMolecular', 'Density'])

        print(mu)
        print(mu.shape)

    will produce the following output ::

        [[[1.78938e-05 1.78938e-05 1.78938e-05]
          [1.78938e-05 1.78938e-05 1.78938e-05]
          [1.78938e-05 1.78938e-05 1.78938e-05]]

         [[1.78938e-05 1.78938e-05 1.78938e-05]
          [1.78938e-05 1.78938e-05 1.78938e-05]
          [1.78938e-05 1.78938e-05 1.78938e-05]]

         [[1.78938e-05 1.78938e-05 1.78938e-05]
          [1.78938e-05 1.78938e-05 1.78938e-05]
          [1.78938e-05 1.78938e-05 1.78938e-05]]]
        (3, 3, 3)

    See also
    --------
    getVars2Dict
    """
    Pointers = []
    FlowSolution = I.getNodeFromName1(zone, Container)
    for v in VariablesName:
        node = I.getNodeFromName1(FlowSolution,v) if FlowSolution else None

        if node:
            Pointers += [node[1]]
        else:
            Pointers += [None]
            print ("Field %s not found in container %s of zone %s. Check spelling or data."%(v,Container,zone[0]))

    return Pointers


def getVars2Dict(zone, VariablesName=None, Container='FlowSolution'):
    """
    Get a dict containing the numpy arrays from a *zone* of the variables
    specified in *VariablesName*.

    Parameters
    ----------
        zone : zone
            The CGNS zone from which numpy arrays are being retreived
        VariablesName : :py:class:`list` of :py:class:`str`
            List of the field names to be retreived.
            If : py:obj:`None`, all variables in the **Container** are retreived.
        Container : str
            The name of the node to look for the requested variable
            (e.g. ``'FlowSolution'``). Container should be at 1 depth level
            inside zone.

    Returns
    -------
        VarsDict : dict
            Contains the numpy arrays as ``VarsDict[<FieldName>]``

            .. note:: if a variable is not found, :py:obj:`None` is returned for such
                occurrence.

    Examples
    --------
    ::

        import Converter.PyTree as C
        import Generator.PyTree as G
        import MOLA.InternalShortcuts as J

        zone = G.cart((0,0,0),(1,1,1),(3,3,3))

        C._initVars(zone,'ViscosityMolecular',1.78938e-5)
        C._initVars(zone,'Density',1.225)

        v = J.getVars2Dict(zone,['ViscosityMolecular', 'Density'])

        print(v['ViscosityMolecular'])
        print(v['ViscosityMolecular'].shape)

    will produce the following output ::

        [[[1.78938e-05 1.78938e-05 1.78938e-05]
          [1.78938e-05 1.78938e-05 1.78938e-05]
          [1.78938e-05 1.78938e-05 1.78938e-05]]

         [[1.78938e-05 1.78938e-05 1.78938e-05]
          [1.78938e-05 1.78938e-05 1.78938e-05]
          [1.78938e-05 1.78938e-05 1.78938e-05]]

         [[1.78938e-05 1.78938e-05 1.78938e-05]
          [1.78938e-05 1.78938e-05 1.78938e-05]
          [1.78938e-05 1.78938e-05 1.78938e-05]]]
        (3, 3, 3)
    """
    Pointers = {}
    FlowSolution = I.getNodeFromName1(zone,Container)
    if VariablesName is None:
        VariablesName = [I.getName(n) for n in I.getNodesFromType1(FlowSolution, 'DataArray_t')]
    for v in VariablesName:
        node = I.getNodeFromName1(FlowSolution,v)
        if node is not None:
            Pointers[v] = node[1]
        else:
            Pointers[v] = [None]
            print ("Field %s not found in container %s of zone %s. Check spelling or data."%(v,Container,zone[0]))
    return Pointers

def getVars2DictPerZone(t, **getVars2DictOpts):
    '''
    higher-level version of :py:func:`getVars2Dict`, where a new level is provided
    to the returned dict, which includes the zone names (which must be unique 
    in the tree).
    '''
    zoneDict = dict()
    for zone in I.getZones(t):
        zoneDict[zone[0]] = getVars2Dict(zone, **getVars2DictOpts)
    return zoneDict

def invokeFields(zone, VariableNames, locationTag='nodes:'):
    """
    Initializes the variables by the names provided as argument
    for the input zone. Returns the list of numpy arrays of such
    new created variables.
    Exists also inplace :py:func:`_invokeFields` and returns :py:obj:`None`.

    Parameters
    ----------
        zone : zone
            CGNS zone where fields are initialized
        VariablesName : :py:class:`list` of :py:class:`str`
            List of the variables names.
        locationTag : str
            Can be either ``nodes:`` or ``centers:``

    Returns
    -------
        numpies : list of numpy.ndarray
            List of numpy.array of the newly created fields.
    """
    _invokeFields(zone,VariableNames,locationTag=locationTag)
    # TODO: replace locationTag by general Container
    Container = I.__FlowSolutionNodes__ if locationTag == 'nodes:' else I.__FlowSolutionCenters__

    return getVars(zone,VariableNames,Container)


def _invokeFields(zone,VariableNames,locationTag='nodes:'):
    '''
    See documentation of :py:func:`invokeFields`.
    '''
    # TODO: Make more efficient variables initialization (using numpy and
    # adding children)
    # TODO: replace locationTag by general Container
    for v in VariableNames: C._initVars(zone,locationTag+v,0.)


def invokeFieldsDict(zone,VariableNames,locationTag='nodes:'):
    """
    Initializes the variables by the names provided as argument
    for the input zone. Returns a dictionary of numpy arrays of
    such newly created variables.

    Parameters
    ----------
        zone : zone
            The CGNS zone from which numpy arrays are being retreived
        VariablesName : :py:class:`list` of :py:class:`str`
            List of the field names to be retreived
        Container : str
            The name of the node to look for the requested variable
            (e.g. ``'FlowSolution'``). Container should be at 1 depth level
            inside zone.

    Returns
    -------
        VarsDict : dict
            Contains the numpy arrays as ``VarsDict[<FieldName>]``

            .. note:: if a variable is not found, :py:obj:`None` is returned for such
                occurrence.
    """
    # TODO: replace locationTag by general Container
    ListOfVars = invokeFields(zone,VariableNames,locationTag=locationTag)
    VarsDict = {}
    for i, VariableName in enumerate(VariableNames):
        VarsDict[VariableName] = ListOfVars[i]

    return VarsDict


def _setField(zone, FieldName, FieldNumpy, locationTag='nodes:'):
    '''
    Set field named <FieldName> contained in <zone> at FlowSolution of tag
    <locationTag> using a numpy array <FieldNumpy>.
    '''
    # TODO: replace locationTag by general Container
    Field, = invokeFields(zone, [FieldName], locationTag=locationTag)
    Field[:] = FieldNumpy


def _add2Field(zone,FieldName,FieldNumpy):
    '''
    Add to field named <FieldName> contained in <zone> at FlowSolution of tag
    <locationTag> a numpy array <FieldNumpy>.
    '''
    # TODO: replace locationTag by general Container
    Field, = invokeFields(zone, [FieldName], locationTag=locationTag)
    Field[:] += FieldNumpy


def getx(zone):
    '''
    Get the pointer of the numpy array of *CoordinateX*.

    Parameters
    ----------
        zone : zone
            Zone PyTree node from where *CoordinateX* is being extracted

    Returns
    -------

        x : numpy.ndarray
            the x-coordinate

    See also
    --------
    gety, getz, getxy, getxyz
    '''
    return I.getNodeFromName2(zone,'CoordinateX')[1]


def gety(zone):
    '''
    Get the pointer of the numpy array of *CoordinateY*.

    Parameters
    ----------

        zone : zone
            Zone PyTree node from where *CoordinateY* is being extracted

    Returns
    -------

        y : numpy.ndarray
            the y-coordinate

    See also
    --------
    getx, getz, getxy, getxyz
    '''
    return I.getNodeFromName2(zone,'CoordinateY')[1]

def getz(zone):
    '''
    Get the pointer of the numpy array of *CoordinateZ*.

    Parameters
    ----------

        zone : zone
            Zone PyTree node from where *CoordinateZ* is being extracted

    Returns
    -------

        z : numpy.ndarray
            the z-coordinate

    See also
    --------
    getx, gety, getxy, getxyz
    '''
    return I.getNodeFromName2(zone,'CoordinateZ')[1]

def getxy(zone):
    '''
    Get the pointers of the numpy array of *CoordinateX* and *CoordinateY*.

    Parameters
    ----------

        zone : zone
            Zone PyTree node from where *CoordinateX* and *CoordinateY* are
            being extracted

    Returns
    -------

        x : numpy.ndarray
            the x-coordinate

        y : numpy.ndarray
            the y-coordinate

    See also
    --------
    getx, gety, getz, getxyz
    '''
    return getx(zone), gety(zone)


def getxyz(zone):
    '''
    Get the pointers of the numpy array of *CoordinateX*,
    *CoordinateY* and *CoordinateZ*.

    Parameters
    ----------

        zone : zone
            Zone PyTree node from where coordinates are being extracted

    Returns
    -------

        x : numpy.ndarray
            the x-coordinate

        y : numpy.ndarray
            the y-coordinate

        z : numpy.ndarray
            the z-coordinate

    See also
    --------
    getx, gety, getz, getxy
    '''
    return getx(zone), gety(zone), getz(zone)

def getRadiusTheta(zone, axis=(1,0,0)):
    '''
    Get the radius and the angle theta.

    Parameters
    ----------

        zone : zone
            Zone PyTree node from where coordinates are being extracted

        axis : tuple
            Axis of the cylindrical frame of reference

    Returns
    -------

        r : numpy.ndarray
            radius

        theta : numpy.ndarray
            angle in radians

    See also
    --------
    getx, gety, getz, getxy, getxyz
    '''
    x, y, z = getxyz(zone)
    if axis != (1,0,0):
        raise Exception('getRadiusTheta is available only for axis=(1,0,0)')

    r = (y**2 + z**2)**0.5
    theta = np.arctan2(z, y)
    return r, theta

def getNearestPointIndex(a,P):
    '''

    .. danger:: AVOID USAGE - this function will be replaced in future

    '''
    NPts = len(P) if isinstance(P, list) else 1

    if I.isTopTree(a): # a is a tree
        zones = I.getNodesFromType2(a,'Zone_t')
        res = []
        Points = [P] if NPts == 1 else P
        for pt in Points:
            IndxDist4AllZones = map(lambda zone: getNearestPointIndexOfZone__(zone,pt), zones)
            Distances = np.array(map(lambda i:IndxDist4AllZones[i][1],range(NPts)))
            NearestZoneNb = np.argmin(Distances)
            res += [IndxDist4AllZones[NearestZoneNb]]
        return res

    else:
        stdNode = I.isStdNode(a)
        if stdNode == -1: # a is a zone
            if NPts == 1:
                return getNearestPointIndexOfZone__(a,P)
            else:
                return map(lambda pt: getNearestPointIndexOfZone__(a,pt), P)
        elif stdNode == 0: # a is a :py:class:`list` of zone
            zones = a
            res = []
            Points = [P] if NPts == 1 else P
            for pt in Points:
                IndxDist4AllZones = map(lambda zone: getNearestPointIndexOfZone__(zone,pt), zones)
                Distances = np.array(map(lambda i:IndxDist4AllZones[i][1],range(NPts)))
                NearestZoneNb = np.argmin(Distances)
                res += [IndxDist4AllZones[NearestZoneNb]]
            return res
        else:
            raise AttributeError('Could not recognize the first argument. Please provide Tree, Zone or :py:class:`list` of zone')


def getNearestPointIndexOfZone__(zone1, Point):
    '''

    .. danger:: AVOID USAGE - this function will be replaced in future

    '''
    x = I.getNodeFromName2(zone1,'CoordinateX')[1].ravel(order='F')
    y = I.getNodeFromName2(zone1,'CoordinateY')[1].ravel(order='F')
    z = I.getNodeFromName2(zone1,'CoordinateZ')[1].ravel(order='F')

    x2, y2, z2 = Point

    Distances = ((x2-x)**2+(y2-y)**2+(z2-z)**2)**0.5
    NearPtIndx = np.argmin(Distances)
    NearPtDist = Distances[NearPtIndx]

    return NearPtIndx, NearPtDist


def getNearestZone(ZonesOrPyTree, Point):
    '''
    Retrieve the nearest zone with respect to a provided Point
    from a provided :py:class:`list` of zone.

    .. warning:: this function is being deprecated

    Parameters
    ----------
        ZonesOrPyTree : PyTree or :py:class:`list` of zone
        Point : Tuple of 3-float (x,y,z) or PyTree Point
            Location from which the distance between zones and this Point is
            measured

    Returns
    -------
        NearestZone : zone
            It is the closest zone from the :py:class:`list` of zone with respect to the
            provided *Point*.
        NearestZoneNo : int
            The nearest zone index.
    '''
    if I.isStdNode(Point) == -1: Point = getxyz(Point)

    if I.isStdNode(ZonesOrPyTree) == 0:
        zones = ZonesOrPyTree
    else:
        zones = I.getNodesFromType(ZonesOrPyTree,'Zone_t')

    # TODO replace getNearestPointIndex !!
    Distances = [getNearestPointIndex(z,Point)[1] for z in zones]
    DistancesNumpy = np.array(Distances)
    NearestZoneNo  = np.argmin(DistancesNumpy)
    NearestZone    = zones[NearestZoneNo]

    return NearestZone, NearestZoneNo


def createZone(Name, Arrays, Vars):
    """
    Convenient function for creating a PyTree zone for I/O
    and writing data.

    Parameters
    ----------
        Name : str
            Name of the new zone
        Arrays : list of numpy.ndarray
            List of the numpy arrays defining the list (coordinates and
            fields).

            .. note:: all numpy arrays contained in argument *Arrays* must
                have the **same** dimensions

        Vars : :py:class:`list` of :py:class:`str`
            The field name (or coordinate name) corresponding to the provided
            array, in the same order.

            .. note:: since all fields must have a name, one must verify
                ``len(Arrays) == len(Vars)``

    Returns
    -------
        zone : zone
            newly created zone

    Examples
    --------
    ::

        import numpy as np
        import Converter.PyTree as C
        import MOLA.InternalShortcuts as J

        x = np.linspace(0,1,10)
        y = x*1.5
        z = y*1.5
        Ro = np.zeros((10)) + 1.225
        MyZone = J.createZone('MyTitle',
                              [            x,            y,            z,       rho],
                              ['CoordinateX','CoordinateY','CoordinateZ','Density'])
        C.convertPyTree2File(MyZone,'MyZone.cgns')


    """
    if not Vars: return
    ni,nj,nk=(list(Arrays[0].shape)+[1,1,1])[:3]
    if ni==0 or nj==0 or nk==0: return
    try:
        ar=np.concatenate([aa.reshape((1,ni*nj*nk),order='F') for aa in Arrays],axis=0)
    except ValueError:
        ERRMSG = FAIL+'ERROR - COULD NOT CONCATENATE ARRAYS FOR %s:\n'%Name
        for i,v in enumerate(Vars):
            ERRMSG += v+' with shape: '+str(Arrays[i].shape)+'\n'
        ERRMSG += ENDC
        raise ValueError(ERRMSG)

    zone = I.createZoneNode(Name,array=[','.join(Vars),ar,ni,nj,nk])

    return zone


def _addSetOfNodes(parent, name, ListOfNodes, type1='UserDefinedData_t', type2='DataArray_t'):
    '''
    parent : Parent node
    name : name of the node
    ListOfNodes : First element is the node name,
    and the second element is its value...
    ... -> [[nodename1, value1],[nodename2, value2],etc...]
    '''

    children = []
    for e in ListOfNodes:
        typeOfNode = type2 if len(e) == 2 else e[2]
        children += [I.createNode(e[0],typeOfNode,value=e[1])]

    node = I.createUniqueChild(parent,name,type1, children=children)
    I._rmNodesByName1(parent, node[0])
    I.addChild(parent, node)


def convertNode2Tetra(zone):
    '''
    Makes use of scipy's Delaunay function in order to produce
    a TRI (2D) or TETRA (3D) mesh from a point cloud defined by
    the input zone of type NODE. If FlowSolutions exist in the
    input zone, those are preserved in the final output.

    Parameters
    ----------

        zone : zone
            Points cloud zone as a form of `NODE`_ zone (as got from
            ``C.convertArray2Node()``).

            .. note:: if all values of CoordinateZ are the same, then **zone**
                is supposed to be 2D.

    Returns
    -------

        zoneUns : zone
            Unstructured meshed of type TRI (2D) or TETRA (3D).

        Delaunay : scipy's Delaunay object
            Returns also the scipy's Delaunay object
    '''

    from scipy.spatial import Delaunay

    x,y,z = getxyz(zone)

    zoneIs2D = np.unique(z).size == 1

    # Stack coordinates
    points = np.vstack((x.flatten(),y.flatten())).T if zoneIs2D else np.vstack((x.flatten(),y.flatten(),z.flatten())).T

    # Apply Delaunay's function
    tri = Delaunay(points, qhull_options='Qj')

    # Create the unstructured zone
    NPts = len(tri.points[:,0])
    NElts= len(tri.vertices[:,0])

    zoneUns = I.createNode(zone[0],ntype='Zone_t',value=np.array([[NPts, NElts,0]],dtype=np.int32,order='F'))
    zt_n = I.createNode('ZoneType', ntype='ZoneType_t',parent=zoneUns)
    I.setValue(zt_n,'Unstructured')
    # Add grid coordinates
    gc_n = I.newGridCoordinates(parent=zoneUns)
    I.createNode('CoordinateX',ntype='DataArray_t',value=tri.points[:,0].reshape((NPts),order='F'), parent=gc_n)
    I.createNode('CoordinateY',ntype='DataArray_t',value=tri.points[:,1].reshape((NPts),order='F'), parent=gc_n)
    Zcoords = z.reshape((NPts),order='F') if zoneIs2D else tri.points[:,2].reshape((NPts),order='F')
    I.createNode('CoordinateZ',ntype='DataArray_t',value=Zcoords, parent=gc_n)
    # Add grid elements
    GEval = 6 if zoneIs2D else 10
    ge_n = I.createNode('GridElements', ntype='Elements_t', value=np.array([GEval,0],dtype=np.int32, order='F'), parent=zoneUns)
    I.createNode('ElementRange', ntype='Inderange_t', value=np.array([1,NElts],dtype=np.int32, order='F'), parent=ge_n)
    I.createNode('ElementConnectivity', ntype='DataArray_t', value=tri.vertices.flatten()+1, parent=ge_n)

    FS_n = I.getNodeFromName(zone,'FlowSolution')
    if FS_n is not None:
        I.addChild(zoneUns,FS_n)
        for child in FS_n[2]:
            child[1] = child[1].ravel(order='C') # 'C' important

    return zoneUns, tri

def interpolate__(AbscissaRequest, AbscissaData, ValuesData,
                  Law='interp1d_linear', axis= -1, **kwargs):
    """
    This is a general-purpose interpolation macro for
    N-dimensional data. This function conveniently wraps
    a set of `scipy.interpolate <https://docs.scipy.org/doc/scipy/reference/interpolate.html>`_
    functions. Future implementations may include other 3rd party libraries.

    Interpolations (and extrapolations) are applied on the last
    axis of **ValuesData**, based on the reference vector
    **AbscissaData**, for the requested points contained in the
    vector **AbscissaRequest**.

    Parameters
    ----------

        AbscissaRequest : 1D numpy array
            The user-requested points where inter/extrapolation will be
            performed.

        AbscissaData : 1D numpy array
            Reference abscissa where **ValuesData** are coherent.

            .. warning:: **AbscissaData** must be monotonically increasing.

        ValuesData : N-d numpy array
            Set of data to interpolate.

            .. warning:: the last dimension of **ValuesData** must be equal to
                the length of **AbscissaData**.

        Law : str
            Controls the algorithm of interpolation
            to be employed. Current implementation includes:

            ``'linear'`` : linear interpolation :math:`\mathcal{O}(1)` mode
                Makes use of the function `numpy.interp() <https://numpy.org/doc/stable/reference/generated/numpy.interp.html>`_

            ``'interp1d_<kind>'`` : one-dimensional interpolation (multiple orders)
                Makes use of the function
                `scipy.interpolate.interp1d <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html>`_,
                where ``<kind>`` may be one of *(for scipy v1.4.1)*:
                (``linear``, ``nearest``, ``zero``, ``slinear``, ``quadratic``,
                ``cubic``, ``previous`` or ``next``).

            ``'pchip'`` : Piecewise Cubic Hermite Interpolating Polynomial :math:`\mathcal{O}(3)`
                Makes use of the function `scipy.interpolate.PchipInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.PchipInterpolator.html>`_

            ``'akima'`` : Akima interpolator :math:`\mathcal{O}(3)`
                Makes use of the function `scipy.interpolate.Akima1DInterpolator <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Akima1DInterpolator.html>`_

            ``'cubic'`` : Cubic spline :math:`\mathcal{O}(3)`
                Makes use of the function `scipy.interpolate.CubicSpline <https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.CubicSpline.html>`_

                In this case, an additional keyword is available for
                specification of spline's *boundary conditions* in **kwargs**,
                for example:

                ::

                    CubicSplineBoundaryConditions = ('clamped', 'not_a_knot')

                The first element of the tupple indicates the boundary
                condition of the spline at the start and the second one
                indicates the boundary condition at the end.

    Returns
    -------

        Result : N-dimension numpy array
            Result of interpolation
    """
    import scipy.interpolate
    LawLower = Law.lower()

    if 'linear' == LawLower:
        return np.interp(AbscissaRequest, AbscissaData, ValuesData)

    elif 'interp1d' in LawLower:
        ScipyLaw =  Law.split('_')[1]
        interp = scipy.interpolate.interp1d( AbscissaData, ValuesData, axis=axis, kind=ScipyLaw, bounds_error=False, fill_value='extrapolate', assume_sorted=True, copy=False)
        return interp(AbscissaRequest)

    elif 'pchip' == LawLower:
        interp = scipy.interpolate.PchipInterpolator(AbscissaData, ValuesData, axis=axis, extrapolate=True)
        return interp(AbscissaRequest)

    elif 'akima' == LawLower:
        interp = scipy.interpolate.Akima1DInterpolator(AbscissaData, ValuesData, axis=axis)
        return interp(AbscissaRequest, extrapolate=True)

    elif 'cubic' == LawLower:
        try: bc_type = kwargs['CubicSplineBoundaryConditions']
        except KeyError: bc_type = 'not-a-knot'

        interp = scipy.interpolate.CubicSpline(AbscissaData, ValuesData, axis=axis, extrapolate=True, **kwargs)
        return interp(AbscissaRequest)

    else:
        raise AttributeError('interpolate__(): Law %s not recognized.'%Law)


def getDistributionFromHeterogeneousInput__(InputDistrib):
    """
    This function accepts a polymorphic object **InputDistrib** and
    conveniently translates it into 1D numpy distributions
    and ``D.getDistribution()``-compliant distribution zone.

    Parameters
    ----------

        InputDistrib : polymorphic
            One of the following objects are accepted:

            * numpy 1D vector
                for example,
                ::

                    np.array([15., 20., 25., 30.])

            * Python list of float
                for example,
                ::

                    [15., 20., 25., 30.]

            * Python dictionary
                A ``W.linelaw()``-compliant dictionary which must
                include, at least, the following keys:

                ``'P1'``, ``'P2'`` and ``'N'``.

                Other possible keys are the
                ``distrib`` possible keys and values of ``W.linelaw()``.

                For example,
                ::

                    dict(P1=(15,0,0), P2=(20,0,0),
                         N=100, kind='tanhOneSide',
                         FirstCellHeight=0.01)

    Returns
    -------

        Span : 1D numpy
            vector monotonically increasing. **Absolute length** dimensions.

        Abscissa : 1D numpy
            corresponding curvilinear abscissa (from 0 to 1) **dimensionless**.

        Distribution : zone
            ``G.map()``-compliant 1D PyTree curve as got from
            ``D.getDistribution()``
    """
    import Geom.PyTree as D

    def buildResultFromNode__(n):
        x,y,z = getxyz(n)
        xIsNone = x is None
        yIsNone = y is None
        zIsNone = z is None
        if (xIsNone and yIsNone and zIsNone):
            ErrMsg = "Input argument was a PyTree node (named %s), but no coordinates were found.\nPerhaps you forgot GridCoordinates nodes?"%n[0]
            raise AttributeError(ErrMsg)
        else:
            if xIsNone:
                if not yIsNone: x = y*0
                else:           x = z*0
            if yIsNone: y = x*0
            if zIsNone: z = x*0
        zone = createZone('distribution',[x,y,z],['CoordinateX', 'CoordinateY', 'CoordinateZ'])
        D._getCurvilinearAbscissa(zone)
        Abscissa, = getVars(zone,['s'])
        Distribution = D.getDistribution(zone)
        x,y,z = getxyz(zone)
        Span = np.sqrt(x*x+y*y+z*z)

        return Span, Abscissa, Distribution


    typeInput=type(InputDistrib)
    NodeKind = I.isStdNode(InputDistrib)
    if NodeKind == -1: # It is a node
        return buildResultFromNode__(InputDistrib)
    elif NodeKind == 0: # List of Nodes
        return buildResultFromNode__(InputDistrib[0])
    elif typeInput is np.ndarray: # It is a numpy array
        s  = InputDistrib
        if len(s.shape)>1:
            ErrMsg = "Input argument was detected as a numpy array of dimension %g!\nInput distribution MUST be a monotonically increasing VECTOR (1D numpy array)."%len(s.shape)
            raise AttributeError(ErrMsg)
        if any( np.diff(s)<0):
            ErrMsg = "Input argument was detected as a numpy array.\nHowever, it was NOT monotonically increasing. Input distribution MUST be monotonically increasing. Check that, please."
            raise AttributeError(ErrMsg)

        zone = createZone('distribution',[s,s*0,s*0],['CoordinateX', 'CoordinateY', 'CoordinateZ'])
        return buildResultFromNode__(zone)

    elif isinstance(InputDistrib, list): # It is a list
        try:
            s = np.array(InputDistrib,dtype=np.float64)
        except:
            raise AttributeError('Could not transform InputDistrib argument into a numpy array.\nCheck your InputDistrib argument.')
        if len(s.shape)>1:
            ErrMsg = "InputDistrib argument was converted from list to a numpy array of shape %s!\nSpan MUST be a monotonically increasing VECTOR (1D numpy array)."%(str(s.shape))
            raise AttributeError(ErrMsg)
        if any( np.diff(s)<0):
            ErrMsg = "Input argument was detected as a numpy array.\nHowever, it was NOT monotonically increasing. Input distribution MUST be monotonically increasing. Check that, please."
            raise AttributeError(ErrMsg)

        zone = createZone('distribution',[s,s*0,s*0],['CoordinateX', 'CoordinateY', 'CoordinateZ'])
        return buildResultFromNode__(zone)

    elif isinstance(InputDistrib,dict):
        from . import Wireframe as W
        try: P1 = InputDistrib['P1']
        except KeyError: P1 = (0,0,0)
        try: P2 = InputDistrib['P2']
        except KeyError: P2 = (1,0,0)
        try: N = InputDistrib['N']
        except KeyError: raise AttributeError('distribution requires number of pts "N"')
        zone = W.linelaw(P1=P1, P2=P2, N=InputDistrib['N'],Distribution=InputDistrib)
        return buildResultFromNode__(zone)

    else:
        raise AttributeError('Type of Span argument not recognized. Check your input.')



def get2DQhullZone__(x,y,rescale=True):
    '''
    Construct the convex-hull *(Qhull)* of 2D data defined by a set of
    scattered **(x, y)** points.

    Parameters
    ----------

        x : 1D-numpy array
            A vector containing all X-values defining the scattered data

        y : 1D-numpy array
            A vector containing all Y-values defining the scattered data

        rescale : bool
            if :py:obj:`True`, then rescales the data for computation of Qhull.

    Returns
    -------

        QhullZone : zone
            a CGNS Structured zone containing the curve of the convex-hull
            around the provided scattered data

        xScale : float
            Employed value for rescaling X-data

        yScale : float
            Employed value for rescaling Y-data

    See also
    --------
    sampleIn2DQhull__
    '''
    from scipy.spatial import ConvexHull
    import Transform.PyTree as T

    if rescale:
        xScale = x.max()-x.min()
        yScale = y.max()-y.min()
    else:
        xScale = 1.0
        yScale = 1.0

    x /= xScale
    y /= yScale

    points = np.vstack((x.flatten(),y.flatten())).T

    hull = ConvexHull(points)

    curves = [createZone('Curve%d'%i,[points[hull.simplices[i],0], points[hull.simplices[i],1], 0*points[hull.simplices[i],1]],['CoordinateX', 'CoordinateY', 'CoordinateZ'] ) for i in range(len(hull.simplices))]

    QhullZone = T.merge(curves)[0]
    xq, yq = getxy(QhullZone)
    xq *= xScale
    yq *= yScale
    x *= xScale
    y *= yScale

    return QhullZone, xScale, yScale


def sampleIn2DQhull__(x,y,QhullNPts=20,QhullScale=1.2, grading=0.1, rescale=True):
    '''
    Produce a new set of scattered data **(x,y)**.

    The technique performs a 2D discretization of initially provided scattered
    data, by first constructing  the convex-hull (*Qhull*), and then making
    a sampling of points *inside* the *Qhull*.

    Parameters
    ----------

        x : 1D-numpy array
            A vector containing all X-values defining the scattered data

        y : 1D-numpy array
            A vector containing all Y-values defining the scattered data

        QhullNPts : int
            Number of points used to uniformly discretize the *Qhull*

        QhullScale : float
            Scaling factor used for deforming resulting *Qhull* from its
            barycenter.

            .. hint:: use **QhullScale** slightly greater than 1 in order to
                obtain a margin for sampling the interior of a scattered region.

        grading : float
            Refinement criterion used for sampling the interior points of the
            *Qhull*, as employed by **grading** attribute of function
            ``G.T3mesher2D``

        rescale : bool
            if :py:obj:`True`, then rescales the scatter data from which *Qhull*
            is computed.

    Returns
    -------

        xnew : 1D numpy vector
            New set of X-values defining scattered data at the interior of
            the *Qhull*

        ynew : 1D numpy vector
            New set of Y-values defining scattered data at the interior of
            the *Qhull*

        QhullZone : zone
            a CGNS Structured zone containing the curve of the convex-hull
            around the provided scattered data

    See also
    --------
    get2DQhullZone__
    '''
    import Generator.PyTree as G
    import Transform.PyTree as T
    from . import Wireframe as W

    QhullZone, xScale, yScale = get2DQhullZone__(x,y,rescale)
    xq, yq = getxy(QhullZone)
    xq /= xScale
    yq /= yScale

    QhullZone = W.discretize(QhullZone,QhullNPts)
    T._scale(QhullZone,QhullScale)
    QhullZoneBAR = C.convertArray2Tetra(QhullZone)
    QhullZoneBAR = G.close(QhullZoneBAR)
    mesh = G.T3mesher2D(QhullZoneBAR, triangulateOnly=0, grading=grading, metricInterpType=0)

    # Rescale back to original
    x *= xScale
    y *= yScale
    xhull, yhull = getxy(QhullZone)
    xhull *= xScale
    yhull *= yScale
    xnew, ynew = getxy(mesh)
    xnew *= xScale
    ynew *= yScale

    return xnew, ynew, QhullZone


def secant(fun, x0=None, x1=None, ftol=1e-6, bounds=None, maxiter=20, args=()):
    '''
    Optimization function with similar interface as scipy's `root_scalar <https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.root_scalar.html>`_
    routine, but this version yields enhanced capabilities of error and bounds
    managment.

    Parameters
    ----------

        fun : callable function
            the scalar callable function where root has to be found.

            .. attention:: for convenience, ``fun()`` can
                return more than two objects, but **only the first one** is
                intended to be the float value where root has to be found.

        ftol : float
            absolute tolerance of function for termination.

        x0 : float
            first guess of the secant method

        x1 : float
            second guess of the secant method

        bounds : 2-float tuple
            minimum and maximum bounds of **x** for accepted search of the root.

        maxiter : int
            maximum number of search iterations. If
            algorithm reaches this number and **ftol** is not satisfied,
            then it returns the closest candidate to the root

        args : tuple
            Additional set of arguments to be passed to the function

    Returns
    -------

        sol : dict
            Contains the optimization problem solution and information
    '''

    if bounds is None: bounds = (-np.inf, +np.inf)

    # Allocate variables
    xguess=np.zeros(maxiter,dtype=np.float64)
    fval  =np.zeros(maxiter,dtype=np.float64)
    root  =np.array([0.0])
    froot =np.array([0.0])
    iters =np.array([0])

    sol = dict(
        xguess = xguess,
        fval   = fval,
        root   = root,
        froot  = froot,
        iters  = iters,
        converged = False,
        message = '',
        )


    def linearRootGuess(x,y,samples=2):
        xs = x[-samples:]
        if xs.max() - xs.min() < 1.e-6: return xs[-1], [0,0,0]
        p = np.polyfit(xs,y[-samples:],1)
        Roots = np.roots(p)
        if len(Roots) > 0: Xroot = Roots[0]
        else: Xroot = np.mean(x)

        return Xroot, p

    def parabolicRootGuess(x,y,samples=3):
        xs = x[-samples:]
        # Check if exist at least three different values in xs
        v0, v1, v2 = xs[-1], xs[-2], xs[-3]
        tol = 1.e-6
        if abs(v0-v1)<tol or abs(v0-v2)<tol or abs(v1-v2)<tol:
            return np.nan, [0,0,0]

        p = np.polyfit(xs,y[-samples:],2)
        roots = np.roots(p)
        dist = np.array([np.min(np.abs(xs-roots[0])),np.min(np.abs(xs-roots[1]))])
        closestRoot = np.argmin(dist)
        Xroot = roots[closestRoot]
        return Xroot, p


    # -------------- ROOT SEARCH ALGORITHM -------------- #
    GoodProgressSamplesCriterion = 5
    CheckIts = np.arange(GoodProgressSamplesCriterion)

    # Initialization
    xguess[0] = x0
    xguess[1] = x1
    fval[0]   = fun(x0,*args)
    fval[1]   = fun(x1,*args)
    bestInitialGuess = np.argmin(np.abs(fval[:2]))
    root[0] = bestInitialGuess
    iters[0] = 2

    for it in range(2,maxiter):

        iters[0] = it

        # Make new guess based on linear and parabolic fit
        rootL, pL = linearRootGuess(xguess[:it],fval[:it])
        rootP = rootL if it==2 else parabolicRootGuess(xguess[:it],fval[:it])[0]
        if np.iscomplex(rootP) or np.isnan(rootP): rootP=rootL
        newguess = 0.5*(rootL+rootP)

        # Handle bounds
        OutOfMaxBound =   newguess > bounds[1]
        if OutOfMaxBound: newguess = bounds[0]
        OutOfMinBound =   newguess < bounds[0]
        if OutOfMinBound: newguess = bounds[1]

        if OutOfMinBound or OutOfMaxBound:
            xguess[it] = newguess
            fval[it] = fun(newguess,*args)
            # Attempt largest set linear fit including new guess
            rootL, pL = linearRootGuess(xguess[:it+1],fval[:it+1],2)
            newguess = rootL

            inBounds = newguess >= bounds[0] and newguess <= bounds[1]

            if not inBounds:
                # Still not in bounds. Attempt to find a new
                # local gradient close to minimum bound
                xguessNew = np.array([bounds[0],bounds[0]+0.01*(bounds[1]-bounds[0])])
                fvalNew = xguessNew*0
                fvalNew[0] = fun(xguessNew[0],*args)
                fvalNew[1] = fun(xguessNew[1],*args)

                rootL, pL = linearRootGuess(xguessNew,fvalNew,2)
                newguess = rootL

                inBounds = newguess >= bounds[0] and newguess <= bounds[1]
                if not inBounds:
                    # Still not in bounds. Last attempt: try
                    # find root estimate in bounds by making
                    # a large linear fit on all iterations
                    rootL, pL = linearRootGuess(np.hstack((xguess[:it+1],xguessNew)),np.hstack((fval[:it+1],fvalNew)),it)
                    newguess = rootL
                    inBounds = newguess >= bounds[0] and newguess <= bounds[1]
                    if not inBounds:
                        # Ok, I give up now
                        # store current best guess
                        indBestGuess = np.argmin(np.abs(fval[:it+1]))
                        root[0]  = xguess[indBestGuess]
                        froot[0] = fval[indBestGuess]

                        sol['message'] = 'Out of bounds guess (%g). If your problem has a solution, try increasing the bounds and/or xtol.'%newguess
                        sol['converged'] = False
                        return sol

        # new guess may be acceptable
        if newguess == xguess[it-1]:
            newguess = np.mean(xguess[:it])
        if newguess == xguess[it-1]:
            newguess = 0.5*(bounds[0]+bounds[1])
        xguess[it] = newguess
        fval[it]   = fun(newguess,*args)

        # stores current best guess
        indBestGuess = np.argmin(np.abs(fval[:it+1]))
        root[0]  = xguess[indBestGuess]
        froot[0] = fval[indBestGuess]

        # Check if solution falls within tolerance
        converged = np.abs(fval[it]) < ftol
        sol['converged'] = converged
        if converged:
            sol['message'] = 'Solution converged within tolerance (ftol=%g)'%ftol
            sol['converged'] = converged
            break

        # Check if algorithm is making good progress
        GoodProgress = True
        if it >= GoodProgressSamplesCriterion:
            FinalIt, progress = linearRootGuess(it+CheckIts,fval[:it],GoodProgressSamplesCriterion)

            # if progress[1] <= 0:
            #     GoodProgress = False
            #     sol['message'] = 'Algorithm is making bad progress in the last %d iterations. Convergence would be obtained after %d iters, which is greater than user-provided maxiters (%d). Aborting.'%(GoodProgressSamplesCriterion, FinalIt,maxiter)
            #     return sol

            if FinalIt > maxiter:
                GoodProgress = False
                sol['message'] = 'Algorithm is not making good enough progress. Convergence would be obtained after %d iters, which is greater than user-provided maxiters (%d).'%(FinalIt,maxiter)

    if not converged:
        sol['message'] += '\nMaximum number of iterations reached.'

    return sol


def writePythonFile(filename,DictOfVariables,writemode='w'):
    '''
    This function writes a Python-compatible file using a dictionary
    where each key corresponds to the variable name and the value is
    associated to the assignment.


    Parameters
    ----------

        filename : str
            New file name (e.g. ``'toto.py'``)

        DictOfVariables : dict
            Pairs of key:value to be written as ``key = value``

        writemode : str
            ``'w'``-for write or ``'a'``-for append

    Returns
    -------

        writes file : None

    Examples
    --------

    >>> writePythonFile('toto.py', {'MyValue':50.0,'MyList':[1.,2.,3]})

    will create a file ``toto.py`` containing:

    ::

        MyValue=50.
        MyList=[1.,2.,3.]


    '''
    import pprint

    with open(filename,writemode) as f:
            if writemode == "w":
                Header = "'''\n%s file automatically generated\n'''\n\n"%filename
                f.write(Header)

            for k in DictOfVariables:
                Variable = str(k)
                PrettyVariable = pprint.pformat(DictOfVariables[k])
                if Variable == "#":
                    f.write(Variable+' '+PrettyVariable+'\n\n')
                else:
                    f.write(Variable+'='+PrettyVariable+'\n\n\n')


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
            DonorZone = getZoneFromListByName(DonorZones, DonorName)
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
            if DonorZoneName[-2:] == '.c':
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

    def getZoneFromListByName(ZoneList, ZoneName):
        for zone in ZoneList:
            if I.getName(zone) == ZoneName:
                return zone

    MigrateDataNodeReservedName = '.MigrateData'

    Donor = I.copyRef( Donor )
    DonorZones = I.getZones( Donor )
    ReceiverZones = I.getZones( Receiver )

    raiseErrorIfNotValidDonorZoneNames( DonorZones )

    addGridLocationNodeIfAbsent( DonorZones )

    addMigrationDataIfForcedOrNotExisting(DonorZones, ReceiverZones)

    invokeFieldsAtReceiver(DonorZones, ReceiverZones)

    for ReceiverZone in ReceiverZones:
        migrateDonorFields2ReceiverZone(DonorZones,
                                        ReceiverZone)

    if not keepMigrationDataForReuse:
        I._rmNodesByName(Receiver, MigrateDataNodeReservedName)


def checkEmptyBC(t):
    '''
    Check if input PyTree has undefined zones in it and prints a message.

    Parameters
    ----------

        t : PyTree
            the tree to be checked

    Returns
    -------

        hasEmpty : bool
            :py:obj:`True` if **t** has at least one empty BC
    '''
    def isEmpty(emptyBC):
        if isinstance(emptyBC, list) or isinstance(emptyBC, np.ndarray):
            for i in emptyBC:
                return isEmpty(i)
            return False
        elif np.isfinite(emptyBC):
            return True
        else:
            raise ValueError('unexpected type %s'%type(emptyBC))

    emptyBC = C.getEmptyBC(t, dim=3)
    hasEmpty = isEmpty(emptyBC)
    if hasEmpty:
        print(FAIL+'UNDEFINED BC IN PYTREE'+ENDC)
    else:
        print(GREEN+'No undefined BC found on PyTree'+ENDC)

    return hasEmpty


def sortListsUsingSortOrderOfFirstList(*arraysOrLists):
    '''
    This function accepts an arbitrary number of lists (or arrays) as input.
    It sorts all input lists (or arrays) following the ordering of the first
    list after sorting.

    Returns all lists with new ordering.

    Parameters
    ----------

        arraysOrLists : comma-separated arrays or lists
            Arbitrary number of arrays or lists

    Returns
    -------

        NewArrays : list
            list containing the new sorted arrays or lists following the order
            of first the list or array (after sorting).

    Examples
    --------

    ::

        import numpy as np
        import MOLA.InternalShortcuts as J

        First = [5,1,6,4]
        Second = ['a','c','f','h']
        Third = np.array([10,20,30,40])

        NewFirst, NewSecond, NewThird = J.sortListsUsingSortOrderOfFirstList(First,Second,Third)
        print(NewFirst)
        print(NewSecond)
        print(NewThird)

    will produce

    ::

        [1, 4, 5, 6]
        ['c', 'h', 'a', 'f']
        [20, 40, 10, 30]

    '''
    SortInd = np.argsort(arraysOrLists[0])
    NewArrays = []
    for a in arraysOrLists:
        if type(a) == 'ndarray':
            NewArray = np.copy(a,order='K')
            for i in SortInd:
                NewArray[i] = a[i]

        else:
            NewArray = [a[i] for i in SortInd]

        NewArrays.append( NewArray )

    return NewArrays

def sortNodesByName(nodes):
    names = [n[0] for n in nodes]
    sorted_nodes = sortListsUsingSortOrderOfFirstList(names, nodes)[1]
    nodes[:] = sorted_nodes



def getSkeleton(t, keepNumpyOfSizeLessThan=20):
    '''
    .. danger:: workaround. See ticket `8815 <https://elsa.onera.fr/issues/8815>`_
    '''
    tR = I.copyRef(t)
    nodes = I.getNodesFromType(tR, 'DataArray_t')
    for n in nodes:
        try:
            if n[1].size > keepNumpyOfSizeLessThan-1: n[1] = None
        except:
            pass
    return tR

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


def forceZoneDimensionsCoherency(t):
    for zone in I.getZones(t):
        ZoneType = I.getValue(I.getNodeFromName(zone,'ZoneType'))
        if ZoneType == 'Structured':
            x = I.getNodeFromName(zone,'CoordinateX')[1]
            if x is None: continue
            dim = len(x.shape)
            if dim == 1:
                zone[1] = np.array([x.shape[0],x.shape[0]-1,0],
                                    dtype=np.int32,order='F')
            elif dim == 2:
                zone[1] = np.array([[x.shape[0],x.shape[0]-1,0],
                                    [x.shape[1],x.shape[1]-1,0]],
                                    dtype=np.int32,order='F')
            elif dim == 3:
                zone[1] = np.array([[x.shape[0],x.shape[0]-1,0],
                                    [x.shape[1],x.shape[1]-1,0],
                                    [x.shape[2],x.shape[2]-1,0],],
                                    dtype=np.int32,order='F')


def getZones(t):
    '''
    .. danger:: workaround. See ticket `8816 <https://elsa.onera.fr/issues/8816>`_
    '''
    if t is None: return []
    else: return I.getZones(t)


def deprecated(v1, v2=None, comment=None):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    import functools
    import warnings
    def decorator(f):
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            WMSG = '{} deprecated since version {}'.format(f.__name__, v1)
            if v2:
                WMSG += ', will be removed in version {}'.format(v2)
            if comment: WMSG += '\n{}'.format(comment)
            warnings.simplefilter('always', DeprecationWarning)  # turn off filter
            warnings.warn(WARN+WMSG+ENDC, category=DeprecationWarning, stacklevel=2)
            warnings.simplefilter('default', DeprecationWarning)  # reset filter
            return f(*args, **kwargs)
        return decorated
    return decorator

def mute_stdout(func):
    '''
    This is a decorator to redirect standard output to /dev/null.
    '''
    def wrap(*args, **kwargs):
        with open(os.devnull, 'w') as devnull:
            old_stdout = sys.stdout
            sys.stdout = devnull
            res = func(*args, **kwargs)
            sys.stdout = old_stdout
        return res
    return wrap

def mute_stderr(func):
    '''
    This is a decorator to redirect standard error to /dev/null.
    '''
    def wrap(*args, **kwargs):
        with open(os.devnull, 'w') as devnull:
            old_stderr = sys.stderr
            sys.stderr = devnull
            res = func(*args, **kwargs)
            sys.stderr = old_stderr
        return res
    return wrap

class OutputGrabber(object):
    """
    Class used to grab standard output or another stream.
    """
    escape_char = "\b"

    def __init__(self, stream=None, threaded=False):
        self.origstream = stream
        self.threaded = threaded
        if self.origstream is None:
            self.origstream = sys.stdout
        self.origstreamfd = self.origstream.fileno()
        self.capturedtext = ""
        # Create a pipe so the stream can be captured:
        self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        """
        Start capturing the stream data.
        """
        self.capturedtext = ""
        # Save a copy of the stream:
        self.streamfd = os.dup(self.origstreamfd)
        # Replace the original stream with our write pipe:
        os.dup2(self.pipe_in, self.origstreamfd)
        if self.threaded:
            # Start thread that will read the stream:
            self.workerThread = threading.Thread(target=self.readOutput)
            self.workerThread.start()
            # Make sure that the thread is running and os.read() has executed:
            time.sleep(0.01)

    def stop(self):
        """
        Stop capturing the stream data and save the text in `capturedtext`.
        """
        # Print the escape character to make the readOutput method stop:
        self.origstream.write(self.escape_char)
        # Flush the stream to make sure all our data goes in before
        # the escape character:
        self.origstream.flush()
        if self.threaded:
            # wait until the thread finishes so we are sure that
            # we have until the last character:
            self.workerThread.join()
        else:
            self.readOutput()
        # Close the pipe:
        os.close(self.pipe_in)
        os.close(self.pipe_out)
        # Restore the original stream:
        os.dup2(self.streamfd, self.origstreamfd)
        # Close the duplicate stream:
        os.close(self.streamfd)

    def readOutput(self):
        """
        Read the stream data (one byte at a time)
        and save the text in `capturedtext`.
        """
        while True:
            if sys.version_info.major == 3:
                char = os.read(self.pipe_out, 1).decode(self.origstream.encoding)
            else:
                char = os.read(self.pipe_out, 1)
            if not char or self.escape_char in char:
                break
            self.capturedtext += char

def selectZonesExceptThatWithHighestNumberOfPoints(ListOfZones):
    '''
    return a list of zones excluding the zone yielding the highest number
    of points

    Parameters
    ----------

        ListOfZones : PyTree, base, :py:class:`list` of zone
            Container of zones from which the selection will be applied

    Returns
    -------

        Zones : :py:class:`list` of zone
            as the input, but excluding the zone with highest number of points
    '''
    zones = I.getZones(ListOfZones)
    ListOfNPts = [C.getNPts(z) for z in zones]
    IndexOfZoneWithMaximumNPts = np.argmax(ListOfNPts)
    return [z for i, z in enumerate(zones) if i != IndexOfZoneWithMaximumNPts]

def selectZoneWithHighestNumberOfPoints(ListOfZones):
    '''
    return the zone with highest number of points

    Parameters
    ----------

        ListOfZones : PyTree, base, :py:class:`list` of zone
            Container of zones from which the selection will be applied

    Returns
    -------

        zone : zone
            the zone with highest number of points
    '''
    zones = I.getZones(ListOfZones)
    ListOfNPts = [C.getNPts(z) for z in zones]
    IndexOfZoneWithMaximumNPts = np.argmax(ListOfNPts)
    return zones[IndexOfZoneWithMaximumNPts]

def load_source(ModuleName, filename, safe=True):
    '''
    Load a python file as a module guaranteeing intercompatibility between
    different Python versions

    Parameters
    ----------

        ModuleName : str
            name to be provided to the new module

        filename : str
            full or relative path of the file containing the source (moudule)
            to be loaded

        safe : bool
            if :py:obj:`True`, then cached files of previously loaded versions
            are explicitely removed

    Returns
    -------

        module : module
            the loaded module
    '''
    if safe:
        current_path_file = filename.split(os.path.sep)[-1]
        for fn in [filename, current_path_file]:
            try: os.remove(fn+'c')
            except: pass
        try: shutil.rmtree('__pycache__')
        except: pass

    if sys.version_info[0] == 3 and sys.version_info[1] >= 5:
        import importlib.util
        spec = importlib.util.spec_from_file_location(ModuleName, filename)
        LoadedModule = importlib.util.module_from_spec(spec)
        sys.modules[ModuleName] = LoadedModule
        spec.loader.exec_module(LoadedModule)
    elif sys.version_info[0] == 3 and sys.version_info[1] < 5:
        from importlib.machinery import SourceFileLoader
        LoadedModule = SourceFileLoader(ModuleName, filename).load_module()
    elif sys.version_info[0] == 2:
        import imp
        LoadedModule = imp.load_source(ModuleName, filename)
    else:
        raise ValueError("Not supporting Python version "+sys.version)
    return LoadedModule

def reload_source(module):
    '''
    Reload a python module guaranteeing intercompatibility between
    different Python versions

    Parameters
    ----------

        module : module
            pointer towards the previously loaded module
    '''
    if sys.version_info[0] == 3:
        import importlib
        importlib.reload(module)
    elif sys.version_info[0] == 2:
        import imp
        imp.reload(module)
    else:
        raise ValueError("Not supporting Python version "+sys.version)


def printEnvironment():

    def getMajorMinorMicro(version_string):
        version_string = version_string.replace('v','')
        MajorMinorMicro = version_string.split('.')
        try:
            Major, Minor, Micro = MajorMinorMicro
        except ValueError:
            Major, Minor = MajorMinorMicro
            Micro = '0'
        return int(Major), int(Minor), int(Micro)

    def gatherMOLAversions():
        ALL_MOLAS_DIR = os.path.sep+os.path.join(*__MOLA_PATH__.split(os.path.sep)[:-1])+os.path.sep
        ALL_MOLAS_VER = [v.replace(ALL_MOLAS_DIR,'') for v in glob.glob(os.path.join(ALL_MOLAS_DIR,'*'))]
        v = {}
        for ver in ALL_MOLAS_VER:
            if not ver.startswith('v'): continue
            M, m, n = getMajorMinorMicro(ver)
            if M not in v:
                v.update({M:{m:[n]}})
            elif m not in v[M]:
                v[M][m] = [n]
            else:
                v[M][m].append(n)

        return v

    def mostUpToDateVersion(AllVersions):
        Major = max(list(AllVersions))
        Minor = max(list(AllVersions[Major]))
        Micro = max(list(AllVersions[Major][Minor]))
        MajorMinorMicro = [str(Major),str(Minor)]
        if Micro > 0: MajorMinorMicro.append( str(Micro) )
        return 'v'+'.'.join(MajorMinorMicro)

    def mostUpToDateMicroVersion(AllVersions, Major, Minor):
        return max(AllVersions[Major][Minor])

    def fullStringOfMostUpToDateMicroVersion(AllVersions, Major, Minor):
        Micro = mostUpToDateMicroVersion(AllVersions, Major, Minor)
        fullString = 'v'+'.'.join([str(Major),str(Minor)])
        if Micro != '0': fullString += '.'+str(Micro)
        return fullString

    def microVersionIsUpToDate(AllVersions, Major, Minor, Micro):
        mostUpToDateMicro = mostUpToDateMicroVersion(AllVersions, Major, Minor)
        return mostUpToDateMicro == Micro

    def usingMostUpToDateVersion(AllVersions, Major, Minor, Micro):
        latestVersion = mostUpToDateVersion(AllVersions)
        MajorMinorMicro = [str(Major),str(Minor)]
        if Micro > 0: MajorMinorMicro.append( str(Micro) )
        usedVersion =  'v'+'.'.join(MajorMinorMicro)
        return usedVersion == latestVersion

    machine = os.getenv('MAC', 'UNKNOWN')
    vELSA = os.getenv('ELSAVERSION', 'UNAVAILABLE')
    totoV = __version__
    if totoV == 'Dev':
        vMOLA = WARN + totoV + ENDC
    else:
        vMOLA = totoV
    print('MOLA version '+vMOLA+' at '+machine+' (%s)'%os.getenv('ARCH', '-'))
    print(' --> Python '+sys.version.split(' ')[0])

    # elsA
    if vELSA == 'UNAVAILABLE': vELSA = FAIL + vELSA + ENDC
    print(' --> elsA '+vELSA)

    # elsA tools chain
    try:
        import etc
    except:
        vETC = FAIL + 'UNAVAILABLE' + ENDC
    else:
        vETC = WARN + 'UNKNOWN' + ENDC
        for vatt in ('__version__','version'):
            if hasattr(etc, vatt):
                vETC = getattr(etc,vatt)
                break
    
    print(' --> ETC '+vETC)

    # Cassiopee
    try:
        import Converter.PyTree as C
        vCASSIOPEE = C.__version__
    except:
        vCASSIOPEE = FAIL + 'UNAVAILABLE' + ENDC
    print(' --> Cassiopee '+vCASSIOPEE)

    # OCC # TODO CAUTION https://elsa.onera.fr/issues/10950
    try:
        import OCC
        vOCC = OCC.__version__
    except:
        vOCC = FAIL + 'UNAVAILABLE' + ENDC
    print(' --> OCC '+vOCC)

    try:
        import Apps
        vApps = Apps.__version__
    except:
        vApps = FAIL + 'UNAVAILABLE' + ENDC
    print(' --> Apps '+vApps)



    # Vortex Particle Method
    try:
        import VortexParticleMethod.vortexparticlemethod
        import MOLA.VPM as VPM
        vVPM = VPM.__version__
    except:
        vVPM = FAIL + 'UNAVAILABLE' + ENDC
    print(' --> VPM '+vVPM)

    # PUMA
    vPUMA = os.getenv('PUMAVERSION', 'UNAVAILABLE')
    if vPUMA == 'UNAVAILABLE':
        vPUMA = FAIL + vPUMA + ENDC
    else:
        try:
            silence = OutputGrabber()
            with silence:
                import PUMA
        except:
            vPUMA = FAIL + 'UNAVAILABLE' + ENDC
    print(' --> PUMA '+vPUMA)

    # turbo
    try:
        import turbo
        vTURBO = turbo.__version__
    except:
        vTURBO = FAIL + 'UNAVAILABLE' + ENDC 
    print(' --> turbo '+vTURBO)

    # ErsatZ
    try:
        import Ersatz
        vERSATZ = Ersatz.__version__
    except:
        vERSATZ = FAIL + 'UNAVAILABLE' + ENDC 
    print(' --> Ersatz '+vERSATZ)


    # maia
    try:
        import maia
        try:
            vMAIA = maia.__version__
        except:
            vMAIA = 'dev'
    except:
        vMAIA = FAIL+'UNAVAILABLE'+ENDC
    print(' --> maia '+vMAIA)

    if totoV == 'Dev':
        print(WARN+'WARNING: you are using an UNSTABLE version of MOLA.\nConsider using a stable version.'+ENDC)
    else:
        Major, Minor, Micro = getMajorMinorMicro(totoV)
        AllVersions = gatherMOLAversions()
        if not microVersionIsUpToDate(AllVersions, Major, Minor, Micro):
            print(WARN+'WARNING: a most updated micro version exist: '+fullStringOfMostUpToDateMicroVersion(AllVersions, Major, Minor)+ENDC)
        if not usingMostUpToDateVersion(AllVersions,Major, Minor, Micro):
            print('INFO: a most updated version exist: '+mostUpToDateVersion(AllVersions)+ENDC)
        else:
            print(GREEN+'You are using the latest version of MOLA'+ENDC)


def getBCFamilies(t):
    '''
    Get all the Families of BC in the tree **t**.

    Parameters
    ----------

        t : PyTree
            May be a top tree, a base or a zone.

    Returns
    -------

        familyNames : :py:class:`list` of :py:class:`str`
            list of all found families of BCs
    '''
    familyNames = []
    # Automatically get BC Families
    if I.getType(t) == 'Zone_t':
        for BC in I.getNodesFromType2(t, 'BC_t'):
            FamilyNameNode = I.getNodeFromType1(BC, 'FamilyName_t')
            if not FamilyNameNode:
                continue
            FamilyName = I.getValue(FamilyNameNode)
            if FamilyName not in familyNames:
                familyNames.append(FamilyName)
    else:
        # assume that t is a top tree or a base
        for fam in I.getNodesFromType(t, 'Family_t'):
            if I.getNodeFromType(fam, 'FamilyBC_t'):
                familyNames.append(I.getName(fam))
    return familyNames

def createSymbolicLink(src, dst):
    try:
        if os.path.islink(dst):
            os.unlink(dst)
        else:
            os.remove(dst)
    except:
        pass
    os.symlink(src, dst)

def getSignal(filename):
    '''
    Get a signal using an temporary auxiliar file technique.

    If the intermediary file exists (signal received) then it is removed, and
    the function returns :py:obj:`True`. Otherwise, it returns
    :py:obj:`False`.

    This function is employed for controling a simulation in a simple manner,
    for example using UNIX command ``touch``:

    .. code-block:: bash

        touch filename

    at the same directory where :py:func:`getSignal` is called.

    Parameters
    ----------

        filename : str
            the name of the file (the signal keyword)

    Returns
    -------

        isOrder : bool
            :py:obj:`True` if the signal is received, otherwise :py:obj:`False`
    '''
    isOrder = False
    try:
        os.remove(filename)
        isOrder = True
        print(CYAN+"Received signal %s"%filename+ENDC)
    except:
        pass
    return isOrder

def rampFunction(iteri, iterf, vali, valf):
    '''
    Create a ramp function, going from **vali** to **valf** between **iteri** to **iterf**.
    If **iteri**=**iterf**, then the return ramp function is a constant equal to **valf**.

    Parameters
    ----------

        iteri, iterf, vali, valf : float

    Returns
    -------

        f : function
            Ramp function. Could be called in ``x`` with:

            >> f(x)
    '''
    if iteri == iterf: 
        return lambda x:  valf
    slope = (valf-vali) / (iterf-iteri)
    if vali == valf:
        f = lambda x: vali*np.ones(np.shape(x))
    elif vali < valf:
        f = lambda x: np.maximum(vali, np.minimum(valf, slope*(x-iteri)+vali))
    else:
        f = lambda x: np.minimum(vali, np.maximum(valf, slope*(x-iteri)+vali))
    return f


def joinFamilies(t, pattern):
    '''
    In the CGNS tree t, gather all the Families <ROW_I>_<PATTERN>_<SUFFIXE> into
    Families <ROW_I>_<PATTERN>, so as many as rows.
    Useful to join all the row_i_HUB* or (row_i_SHROUD*) together

    Parameters
    ----------

        t : PyTree
            A PyTree read by Cassiopee

        pattern : str
            The pattern used to gather CGNS families. Should be for example 'HUB' or 'SHROUD'
    '''
    fam2remove = []
    fam2keep = []
    # Loop on the BCs in the tree
    for bc in I.getNodesFromType(t, 'BC_t'):
        # Get BC family name
        famBC_node = I.getNodeFromType(bc, 'FamilyName_t')
        if not famBC_node: 
            continue
        famBC = I.getValue(famBC_node)
        # Check if the pattern is present in FamilyBC name
        if pattern not in famBC:
            continue
        # Split to get the short name based on pattern
        split_fanBC = famBC.split(pattern)
        assert len(split_fanBC) == 2, 'The pattern {} is present more than once in the FamilyBC {}. It must be more selective.'.format(
            pattern, famBC)
        preffix, suffix = split_fanBC
        # Add the short name to the set fam2keep
        short_name = '{}{}'.format(preffix, pattern)
        if short_name not in fam2keep: 
            fam2keep.append(short_name)
        if suffix != '':
            # Change the family name
            I.setValue(famBC_node, '{}'.format(short_name))
            if famBC not in fam2remove: 
                fam2remove.append(famBC)

    # Remove families
    for fam in fam2remove:
        print('Remove family {}'.format(fam))
        I._rmNodesByNameAndType(t, fam, 'Family_t')

    # Check that families to keep still exist
    base = I.getNodeFromType(t, 'CGNSBase_t')
    for fam in fam2keep:
        fam_node = I.getNodeFromNameAndType(t, fam, 'Family_t')
        if fam_node is None:
            print('Add family {}'.format(fam))
            I.newFamily(fam, parent=base)

def _getBaseWithZoneName(t,zone_name):
    for base in I.getBases(t):
        lbase = len(base)
        for child in base[2]:
            if child[0] == zone_name and child[3] == 'Zone_t':
                return base

def zoneOfFamily(zone, family_name, wildcard_used=False):
    families = I.getNodesFromType1(zone,'FamilyName_t')
    families += I.getNodesFromType1(zone,'AdditionalFamilyName_t')
    for f in families:
        if wildcard_used:
            if fnmatch(I.getValue(f), family_name): return True
        elif f[0] == family_name: return True
    return False

def selectZones(t, baseName=None, familyName=None, zoneName=None):
    '''
    Gather zones contained in **t** such that they verify a given set of 
    conditions defined by pair of keyword-argument. All conditions are taken 
    as ``AND`` condition (all must be verified simultaneously).

    .. note::
        if a condition is given the value :py:obj:`None`, then it is interpreted
        as *any* (criterion is verified).

    Parameters
    ----------

        t : PyTree
            input tree where zones will be selected

        baseName : str
            select only zones that are contained in a base name **baseName**.

            .. hint:: wildcards are accepted (e.g. ``Iso*``)
            
        familyName : str
            select only zones that belongs to a family named **familyName**.

            .. hint:: wildcards are accepted (e.g. ``BLADE*``)

        zoneName : str
            select only zones that has name **zoneName**.

            .. hint:: wildcards are accepted (e.g. ``cart*``)

    Returns
    -------

        zones : :py:class:`list` of zone
            zones verifying the selection conditions
    '''
    zones = []
    for base in I.getBases(t):
        currentBaseName = I.getName(base)
        BaseMatch = fnmatch(currentBaseName, baseName) if baseName is not None else True
        if not BaseMatch: continue
        for zone in I.getZones(base):
            FamilyMatch = zoneOfFamily(zone, familyName, wildcard_used=True) if familyName is not None else True
            ZoneMatch = fnmatch(zone[0], zoneName) if zoneName is not None else True
            if BaseMatch == ZoneMatch == FamilyMatch == True: zones += [ zone ]
    return zones

def _reorderBases(t):
    '''Reorder bases of the PyTree **t** in the alphabetical order.'''
    tmp = {}
    for base in I.getBases(t):
        tmp[I.getName(base)] = base
        I._rmNode(t, base)

    for tmpKey in sorted(tmp):
        I.addChild(t, tmp[tmpKey])

def moveFields(t, origin='FlowSolution#EndOfRun#Relative',
        destination='FlowSolution#Init'):
    '''
    For each zone contained in PyTree **t**, move all the fields of container
    **origin** into the container **destination**, possibly overriding nodes of 
    same name already present in **destination**.

    Parameters
    ----------

        t : PyTree
            input tree (will *not* be modified)

        origin : str 
            name of the container where fields to be moved are present

        destination : str
            name of the container where fields of **origin** are being moved

    Returns
    ----------

        tR : PyTree
            reference copy of **t**, with the modification
    '''
    tR = I.copyRef(t)
    for zone in I.getZones(tR):
    
        container_origin = I.getNodeFromName1(zone, origin)
        if not container_origin: continue
        
        container_destination = I.getNodeFromName1(zone, destination)
        if not container_destination:
            container_origin[0] = destination
            continue

        for field_src in container_origin[2]:
            
            if field_src[3] != 'DataArray_t': continue
            
            replaced = False
            for i, field_dst in enumerate(container_destination[2]):
                if field_dst[0] == field_src[0]:
                    del container_destination[2][i]
                    container_destination[2].insert(i, field_src)
                    replaced = True
                    break
            if not replaced: container_destionation += [ field_src ]

        I._rmNode(zone, container_origin)
    
    return tR

def load(*args, **kwargs):
    '''
    load a file using either Cassiopee convertFile2PyTree or maia
    file_to_dist_tree depending on the chosen backend (``'maia'`` or
    ``'cassiopee'`` ). By default, keyword ``backend='auto'`` will use maia
    for  ``*.cgns`` and ``*.hdf`` formats; and Cassiopee for the rest.
    '''
    try:
        backend = kwargs['backend'].lower()
        del kwargs['backend']
    except KeyError:
        backend = 'auto'

    if backend == 'auto':
        filename = args[0]
        if filename.endswith('.cgns') or filename.endswith('.hdf'):
            import maia
            from mpi4py import MPI
            try:
                t = maia.io.file_to_dist_tree(filename, MPI.COMM_WORLD, **kwargs)
            except BaseException as e:
                print(WARN+f'could not open {filename} with maia, received error:')
                print(e)
                print('switching to Cassiopee...'+ENDC)
                t = C.convertFile2PyTree(*args, **kwargs)
        else:
            t = C.convertFile2PyTree(*args, **kwargs)
    elif backend.lower() == 'cassiopee':
        t = C.convertFile2PyTree(*args, **kwargs)
    elif backend.lower() == 'maia':
        import maia
        from mpi4py import MPI
        filename = args[0]
        t = maia.io.file_to_dist_tree(filename, MPI.COMM_WORLD,**kwargs)
    else:
        raise NotImplementedError(f'backend {backend} unknown')

    return t

def save(*args, **kwargs):
    '''
    literally, a shortcut of C.convertPyTree2File
    '''
    C.convertPyTree2File(*args, **kwargs)

def extractBCFromFamily(t, Family):        
    BCList = []
    for zone in I.getZones(t):
        zoneType = I.getValue(I.getNodeFromName1(zone, 'ZoneType'))
        
        if zoneType == 'Unstructured':
            BCList += C.extractBCOfName(zone, f'FamilySpecified:{Family}', reorder=False)
        
        else:
            x, y, z = getxyz(zone)
            for BC in I.getNodesFromType2(zone, 'BC_t'):
                FamilyNode = I.getNodeFromType1(BC, 'FamilyName_t') # Adapt for several families
                if not FamilyNode: continue
                FamilyName = I.getValue(FamilyNode)
                if Family == FamilyName:
                    PointRange = I.getValue(I.getNodeFromName1(BC, 'PointRange'))
                    bc_shape = PointRange[:, 1] - PointRange[:, 0]
                    if bc_shape[0] == 0:
                        squeezedAxis = 0
                        if PointRange[0, 0]-1 == 0: 
                            indexBC = 0 # BC on imin
                        else:
                            indexBC = -1 # BC on imax
                        SliceOnVertex = np.s_[[indexBC], 
                                                PointRange[1, 0]-1:PointRange[1, 1], 
                                                PointRange[2, 0]-1:PointRange[2, 1]]
                        SliceOnCell = np.s_[[indexBC],
                                                PointRange[1, 0]-1:PointRange[1, 1]-1, 
                                                PointRange[2, 0]-1:PointRange[2, 1]-1]

                    elif bc_shape[1] == 0:
                        squeezedAxis = 1
                        if PointRange[1, 0]-1 == 0: 
                            indexBC = 0 # BC on jmin
                        else:
                            indexBC = -1 # BC on jmax
                        SliceOnVertex = np.s_[PointRange[0, 0]-1:PointRange[0, 1],
                                            [indexBC], 
                                            PointRange[2, 0]-1:PointRange[2, 1]]
                        SliceOnCell = np.s_[PointRange[0, 0]-1:PointRange[0, 1]-1,
                                            [indexBC], 
                                            PointRange[2, 0]-1:PointRange[2, 1]-1]
                        

                    elif bc_shape[2] == 0:
                        squeezedAxis = 2
                        if PointRange[2, 0]-1 == 0: 
                            indexBC = 0 # BC on kmin
                        else:
                            indexBC = -1 # BC on kmax
                        SliceOnVertex = np.s_[PointRange[0, 0]-1:PointRange[0, 1],
                                            PointRange[1, 0]-1:PointRange[1, 1],
                                            [indexBC]]
                        SliceOnCell = np.s_[PointRange[0, 0]-1:PointRange[0, 1]-1,
                                            PointRange[1, 0]-1:PointRange[1, 1]-1,
                                            [indexBC]]

                    ni, nj, nk = x[SliceOnVertex].shape
                    zsize = np.zeros((3,3),dtype=int)
                    zsize[0,0] = ni
                    zsize[1,0] = nj
                    zsize[2,0] = nk
                    zsize[0,1] = np.maximum(ni-1,1)
                    zsize[1,1] = np.maximum(nj-1,1)
                    zsize[2,1] = np.maximum(nk-1,1)
                
                    newZoneForBC = I.newZone(f'{I.getName(zone)}\{I.getName(BC)}', zsize=zsize, ztype='Structured', family=Family)
                    set(newZoneForBC, 'GridCoordinates', childType='GridCoordinates_t', 
                        CoordinateX=x[SliceOnVertex], CoordinateY=y[SliceOnVertex], CoordinateZ=z[SliceOnVertex])

                    # FlowSolution nodes
                    for FS in I.getNodesFromType1(zone, 'FlowSolution_t'):
                        FSName = I.getName(FS)
                        try:
                            GridLocation = I.getValue(I.getNodeFromType1(FS, 'GridLocation_t'))
                        except:
                            GridLocation = 'Vertex'
                        if GridLocation == 'Vertex':
                            localSlice = SliceOnVertex
                        else:
                            localSlice = SliceOnCell

                        FSdata = dict()
                        for node in I.getNodesFromType(FS, 'DataArray_t'):
                            FSdata[I.getName(node)] = I.getValue(node)[localSlice]

                        set(newZoneForBC, FSName, childType='FlowSolution_t', **FSdata)
                        newFS = I.getNodeFromNameAndType(newZoneForBC, FSName, 'FlowSolution_t')
                        I.createNode('GridLocation', 'GridLocation_t', value=GridLocation, parent=newFS)

                    # BCDataSet
                    for BCDataSet in I.getNodesFromType(BC, 'BCDataSet_t'):
                        BCDataSetName = I.getName(BCDataSet)

                        # Assumption: Only one BCData per BCDataSet and GridLocation = FaceCenter
                        GridLocationNode = I.getNodeFromType1(BCDataSet, 'GridLocation_t')
                        if GridLocationNode and I.getValue(GridLocationNode) != 'FaceCenter':
                            raise Exception('GridLocation is must be "FaceCenter" for BCDataSet.')
                        
                        BCData = dict()
                        for node in I.getNodesFromType(BCDataSet, 'DataArray_t'):
                            value = np.ravel(I.getValue(node), order='F')
                            BCData[I.getName(node)] = value.reshape((zsize[0,1], zsize[1,1], zsize[2,1]))

                        set(newZoneForBC, BCDataSetName, childType='FlowSolution_t', **BCData)
                        newFS = I.getNodeFromNameAndType(newZoneForBC, BCDataSetName, 'FlowSolution_t')
                        I.createNode('GridLocation', 'GridLocation_t', value='CellCenter', parent=newFS)                        
                
                    BCList.append(newZoneForBC)

    return BCList

def elementTypes(t):
    f'''
    returns a list of all CGNS unstructured element types contained in **t**

    Possible values are: {element_types}
    '''
    types = []
    for zone in I.getZones(t):
        elts_nodes = I.getNodesFromType1(zone, 'Elements_t')

        if not elts_nodes:
            zone_type = I.getValue(I.getNodeFromName1(zone,'ZoneType'))
            if zone_type == 'Structured':
                types += ['STRUCTURED']

        for elts in elts_nodes:
            enum = int(elts[1][0])
            types += [element_types[enum]]
            
    return types

def hasAllNGon(t):
    return all(elt_type in ['NGON_n', 'NFACE_n'] for elt_type in elementTypes(t))

def anyNotNGon(t):
    return any(elt_type not in ['NGON_n', 'NFACE_n'] for elt_type in elementTypes(t))