'''
Main module for tree-based operations (CGNS) in an object-oriented manner

Please use the following convention when importing:

.. code::

    import MOLA.Data as M

21/12/2021 - L. Bernardos - first creation
'''

from . import Core 
np = Core.np
RED = Core.RED 
GREEN = Core.GREEN
WARN = Core.WARN
PINK = Core.PINK
CYAN = Core.CYAN
ENDC = Core.ENDC

from .Node import Node, readNode, castNode
from .Zone import Zone
from .Base import Base
from .Tree import Tree
from . import Mesh

CoordinatesShortcuts = dict(CoordinateX='CoordinateX',
                            CoordinateY='CoordinateY',
                            CoordinateZ='CoordinateZ',
                            x='CoordinateX',
                            y='CoordinateY',
                            z='CoordinateZ',
                            X='CoordinateX',
                            Y='CoordinateY',
                            Z='CoordinateZ',)


def load(filename, only_skeleton=False):

    if Core.settings.backend == 'h5py2cgns':
        t, f, links = Core.h.load(filename, only_skeleton=only_skeleton)
        t = Tree(t)
        for link in links:
            t.addLink(path=link[3], target_file=link[1], target_path=link[2])

    elif Core.settings.backend == 'pycgns':
        t, links, paths = Core.CGM.load(filename)
        for p in paths:
            raise IOError('file %s : could not read node %s'%(filename,str(p)))
        t = Tree(t)
        for link in links:
            t.addLink(path=link[3], target_file=link[1], target_path=link[2])

    elif Core.settings.backend == 'cassiopee':
        links = []
        t = Core.C.convertFile2PyTree(filename, links=links)
        t = Tree(t)
        for link in links:
            t.addLink(path=link[3], target_file=link[1], target_path=link[2])


    else:
        raise ModuleNotFoundError('%s backend not supported'%Core.settings.backend)

    return t


def save(data, *args, **kwargs):
    if type(data) in ( Tree, Base, Zone ):
        t = data
    elif isinstance(data, list) or isinstance(data, dict):
        t = merge( data )
    else:
        raise ValueError('saving data of type %s not supported'%type(data))

    t.save(*args, **kwargs)

def merge(*data1, **data2 ):
    if isinstance(data1, dict):
        t1 = Tree( **data1 )
    else:
        t1 = Tree( )
        t1.merge( data1 )

    if isinstance(data2, dict):
        t2 = Tree( **data2 )
    else:
        t2 = Tree( )
        t2.merge( data2 )

    t1.merge(t2)

    return t1

def getZones( data ):
    t = merge( data ) # TODO avoid merge
    return t.zones()

def getBases( data ):
    t = merge( data ) # TODO avoid merge
    return t.bases()

def getStructure(t):
    '''
    Get a :py:class:`Tree` base structure (children of base nodes are empty)

    Parameters
    ----------

        t : Tree
            tree from which structure is to be extracted

    Returns
    -------
        Structure : Tree
            reference copy of **t**, with empty zones
    '''
    Structure = t.copy()
    for zone in Structure.zones():
        zone[2] = []
    return Structure


def useEquation(data, *args, **kwargs):
    t = merge( data ) # TODO avoid merge
    for zone in t.zones(): zone.useEquation(*args, **kwargs)

    return t


def newZoneFromArrays(Name, ArraysNames, Arrays):

    CoordinatesNames = list(CoordinatesShortcuts)
    numpyarray = np.array(Arrays[0])
    dimensions = []
    for d in range(len(numpyarray.shape)):
        dimensions += [[numpyarray.shape[d], numpyarray.shape[d]-1, 0]]
    zone = Zone(Name=Name, Value=np.array(dimensions,dtype=np.int32,order='F'))

    Coordinates = Node(Name='GridCoordinates', Type='GridCoordinates_t')
    Fields = Node(Name='FlowSolution', Type='FlowSolution_t')

    for name, array in zip(ArraysNames, Arrays):
        numpyarray = np.array(array,order='F')
        if name in CoordinatesNames:
            Node(Name=CoordinatesShortcuts[name], Value=numpyarray,
                 Type='DataArray_t', Parent=Coordinates)
        else:
            Node(Name=name, Value=numpyarray, Type='DataArray_t', Parent=Fields)

    if Coordinates.hasChildren(): zone.addChild( Coordinates )
    if Fields.hasChildren: zone.addChild( Fields )

    return zone

def newZoneFromDict(Name, DictWithArrays):
    ArraysNames, Arrays = [], []
    for k in DictWithArrays:
        ArraysNames += [k]
        Arrays += [DictWithArrays[k]]
    return newZoneFromDict(Name, DictWithArrays)
