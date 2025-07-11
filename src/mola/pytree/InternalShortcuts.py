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
Creation by recycling InternalShortcuts.py of v1.18.1
'''

import sys
import os

import threading
import time
import glob
import numpy as np
import datetime
from itertools import product
from timeit import default_timer as tic
from fnmatch import fnmatch
from contextlib import contextmanager


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

def getAllVars(zone, Container='FlowSolution'):
    '''
    get all fields of a zone stored in a dict
    '''
    fs = I.getNodeFromName1(zone,Container)
    field_names = [n[0] for n in fs[2] if n[3]=='DataArray_t']
    return getVars2Dict(zone, field_names, Container=Container)

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

def getyz(zone):
    '''
    Get the pointers of the numpy array of *CoordinateY* and *CoordinateZ*.

    Parameters
    ----------

        zone : zone
            Zone PyTree node from where *CoordinateY* and *CoordinateZ* are
            being extracted

    Returns
    -------

        y : numpy.ndarray
            the y-coordinate

        z : numpy.ndarray
            the z-coordinate

    See also
    --------
    getx, gety, getz, getxy, getxyz
    '''
    return gety(zone), getz(zone)

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

def getZoneFromListByName(ZoneList, ZoneName):
    '''
    Extract a zone from list of nodes (children are not parsed)

    Parameters
    ----------

        ZoneList : list
            list of nodes

        ZoneName : str
            name of the zone to extract from the list

    Returns
    -------

        zone : zone
            zone named **ZoneName**
    '''
    for zone in ZoneList:
        if I.getType(zone) != 'Zone_t': continue
        if I.getName(zone) == ZoneName:
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

@contextmanager
def stdout_redirected(to=os.devnull):
    '''
    import os

    with stdout_redirected(to=filename):
        print("from Python")
        os.system("echo non-Python applications are also supported")
    '''
    fd = sys.stdout.fileno()

    ##### assert that Python and C stdio write using the same file descriptor
    ####assert libc.fileno(ctypes.c_void_p.in_dll(libc, "stdout")) == fd == 1

    def _redirect_stdout(to):
        sys.stdout.close() # + implicit flush()
        os.dup2(to.fileno(), fd) # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, 'w') # Python writes to fd

    with os.fdopen(os.dup(fd), 'w') as old_stdout:
        with open(to, 'w') as file:
            _redirect_stdout(to=file)
        try:
            yield # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout) # restore stdout.
                                            # buffering and flags such as
                                            # CLOEXEC may be different

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
    ``'cassiopee'`` ). 
    
    Special keyword ``backend='auto'`` will use maia
    for  ``*.cgns`` and ``*.hdf`` formats; and Cassiopee for the rest.
    Default backend is Cassiopee

    Special keyword ``return_type='zones'`` will return only a list of zones
    contained in the file (if any). By default, ``return_type='tree'``. If
    ``return_type='zone'`` (singular) only the first zone in tree is returned.
    '''
    try:
        backend = kwargs['backend'].lower()
        del kwargs['backend']
    except KeyError:
        backend = 'Cassiopee' # this is the default value 

    try:
        return_type = kwargs['return_type'].lower()
        del kwargs['return_type']
    except KeyError:
        return_type = 'tree' # this is the default value 


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

    if return_type == 'tree':
        return t
    elif return_type == 'zones':
        return I.getZones(t)
    elif return_type == 'zone':
        return I.getZones(t)[0]
    else:
        raise NotImplementedError(f'return_type must be "tree" or "zones", got "{return_type}"')

def loadZone(*args, **kwargs):
    '''
    shortcut for :py:func:`load` using option ``return_type='zone'``
    '''
    kwargs['return_type']='zone'
    return load(*args,**kwargs)

def loadZones(*args, **kwargs):
    '''
    shortcut for :py:func:`load` using option ``return_type='zones'``
    '''
    kwargs['return_type']='zones'
    return load(*args,**kwargs)

def save(*args, **kwargs):
    '''
    shortcut of C.convertPyTree2File.

    If special keyword ``force_unique_zones_names=True`` then uses I.correctPyTree(level=3)
    before saving file, in order to avoid loosing data
    '''
    try:
        force_unique_zones_names = kwargs['force_unique_zones_names']
        del kwargs['force_unique_zones_names']
    except KeyError:
        force_unique_zones_names = False # this is the default value 

    if force_unique_zones_names: I._correctPyTree(args[0],level=3)

    C.convertPyTree2File(*args, **kwargs)

def extractBCFromFamily(t, Family, squeeze=False):  
    '''
    Like C.extractBCOfName or C.extractBCOfType, except two points:

    #. **Family** is directly the name of the BC family (without 'FamilySpecified:')

    #. the orientation of the BC is preserved through this operation (unlike C.extractBCOfName)
    '''      
    if I.isType(t, 'Zone_t'):
        zones = [t]
    else:
        zones = I.getZones(t)

    BCList = []
    for zone in zones:
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
                        if not squeeze:
                            indexBC = [indexBC]
                        SliceOnVertex = np.s_[indexBC, 
                                                PointRange[1, 0]-1:PointRange[1, 1], 
                                                PointRange[2, 0]-1:PointRange[2, 1]]
                        SliceOnCell = np.s_[indexBC,
                                                PointRange[1, 0]-1:PointRange[1, 1]-1, 
                                                PointRange[2, 0]-1:PointRange[2, 1]-1]

                    elif bc_shape[1] == 0:
                        squeezedAxis = 1
                        if PointRange[1, 0]-1 == 0: 
                            indexBC = 0 # BC on jmin
                        else:
                            indexBC = -1 # BC on jmax
                        if not squeeze:
                            indexBC = [indexBC]
                        SliceOnVertex = np.s_[PointRange[0, 0]-1:PointRange[0, 1],
                                            indexBC, 
                                            PointRange[2, 0]-1:PointRange[2, 1]]
                        SliceOnCell = np.s_[PointRange[0, 0]-1:PointRange[0, 1]-1,
                                            indexBC, 
                                            PointRange[2, 0]-1:PointRange[2, 1]-1]
                        

                    elif bc_shape[2] == 0:
                        squeezedAxis = 2
                        if PointRange[2, 0]-1 == 0: 
                            indexBC = 0 # BC on kmin
                        else:
                            indexBC = -1 # BC on kmax
                        if not squeeze:
                            indexBC = [indexBC]
                        SliceOnVertex = np.s_[PointRange[0, 0]-1:PointRange[0, 1],
                                            PointRange[1, 0]-1:PointRange[1, 1],
                                            indexBC]
                        SliceOnCell = np.s_[PointRange[0, 0]-1:PointRange[0, 1]-1,
                                            PointRange[1, 0]-1:PointRange[1, 1]-1,
                                            indexBC]

                    if squeeze:
                        ni, nj = x[SliceOnVertex].shape
                        zsize = np.zeros((2,2),dtype=int)
                        zsize[0,0] = ni
                        zsize[1,0] = nj
                        zsize[0,1] = np.maximum(ni-1,1)
                        zsize[1,1] = np.maximum(nj-1,1)
                    else:
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
                            if squeeze:
                                BCData[I.getName(node)] = value.reshape((zsize[0,1], zsize[1,1]))
                            else:
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

def checkUniqueChildren(t, recursive=False):
    nodes_names_and_types = [(I.getName(child), I.getType(child)) for child in I.getChildren(t)]
    # Check unicity of each child
    # assert len(nodes_names_and_types) == len(set(nodes_names_and_types)) # Cannot do that because of the function set defined in the current module...
    tmp_list = []
    for name_type in nodes_names_and_types:
        if name_type not in tmp_list:
            tmp_list.append(name_type)
        else:
            save(t, 'debug.cgns')
            raise Exception(FAIL+f'The node {name_type[0]} of type {name_type[1]} is defined twice'+ENDC)
    if recursive:
        for node in I.getChildren(t):
            checkUniqueChildren(node, recursive=True)

def printElapsedTime(message='', previous_timer=0.0):
    ElapsedTime = str(datetime.timedelta(seconds=tic()-previous_timer))
    hours, minutes, seconds = ElapsedTime.split(':')
    int_hours = int(hours)
    int_minutes = int(minutes)
    if int_hours < 1:
        if int_minutes < 1:
            ElapsedTimeHuman = f'{seconds} seconds'
        else:
            s = 's' if int_minutes!=1 else ''
            ElapsedTimeHuman = f'{int_minutes} minute{s} and {seconds} seconds'
    else:
        sh = 's' if int_hours!=1 else ''
        sm = 's' if int_hours!=1 else ''
        ElapsedTimeHuman = f'{int_hours} hour{sh} {int_minutes} minute{sm} and {seconds} seconds'
    msg = message + ' ' + ElapsedTimeHuman
    print(BOLD+msg+ENDC)

def checkUniqueChildren(t, recursive=False):
    nodes_names_and_types = [(I.getName(child), I.getType(child)) for child in I.getChildren(t)]
    # Check unicity of each child
    # assert len(nodes_names_and_types) == len(set(nodes_names_and_types)) # Cannot do that because of the function set defined in the current module...
    tmp_list = []
    for name_type in nodes_names_and_types:
        if name_type not in tmp_list:
            tmp_list.append(name_type)
        else:
            save(t, 'debug.cgns')
            raise Exception(FAIL+f'The node {name_type[0]} of type {name_type[1]} is defined twice'+ENDC)
    if recursive:
        for node in I.getChildren(t):
            checkUniqueChildren(node, recursive=True)

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

def getFieldOrCoordinate(zone, field_or_coordinate : str):
    foc = field_or_coordinate
    if foc.startswith('Coordinate') or foc.lower() in 'xyz':
        coord = foc[-1].lower()
        
        if coord == 'x':
            return getx(zone)

        elif coord == 'y':
            return gety(zone)

        elif coord == 'z':
            return gety(zone)

        else:
            raise AttributeError('unsupported coordinate "%s"'%foc)
    else:
        return getVars(zone, [foc])[0]

def save_and_raise(t, filename='debug.cgns'):
    I._correctPyTree(t,level=3)
    save(t, filename)
    raise Exception(filename)

def getVector(zone, vector_name='s', vector_coordinates=['x','y','z']):
    return getVars(zone,[vector_name+c for c in vector_coordinates])

def tree(**kwargs):
    base_zones_list = []
    for basename in kwargs:
        base_zones_list += [basename, I.getZones(kwargs[basename])]
    t = C.newPyTree([*base_zones_list])
    I._correctPyTree(t,level=3)
    return t
    
def getZonesByCopy( tree_base_zone_or_list ):
    zones = I.getZones(tree_base_zone_or_list)
    if not zones: raise AttributeError('did not find zones in provided argument')
    t = C.newPyTree(['BASE',zones])
    copied_zones = I.getZones(I.copyTree(t))
    return copied_zones
