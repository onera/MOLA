'''
Main module for tree-based operations (CGNS) in an object-oriented manner

Please use the following convention when importing:

.. code::

    import MOLA.Data as M

21/12/2021 - L. Bernardos - first creation
'''

from .Core import np,RED,GREEN,WARN,PINK,CYAN,ENDC,CGM, toc

from .Node import Node
from .Zone import Zone
from .Base import Base
from .Tree import Tree
from . import Mesh

def load(filename):
    t, links, paths = CGM.load(filename)
    for p in paths:
        raise IOError('file %s : could not read node %s'%(filename,str(p)))
    t = Tree(t)
    for link in links:
        t.addLink(path=link[3], target_file=link[1], target_path=link[2])

    return t


def readNode(filename, path):
    if path.startswith('CGNSTree/'):
        path_map = path.replace('CGNSTree','')
    else:
        path_map = path
    t, l, p = CGM.load( filename, subtree=path_map )
    t = Node(t)
    node = t.getAtPath( path )
    if node is None:
        raise ValueError("node %s not found in %s"%( path, filename ))

    return node

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
