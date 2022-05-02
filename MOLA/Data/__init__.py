'''
Main module for tree-based operations (CGNS) in an object-oriented manner

Please use the following convention when importing:

.. code::

    import MOLA.Data as D

21/12/2021 - L. Bernardos - first creation
'''

RED  = '\033[91m'
GREEN = '\033[92m'
WARN  = '\033[93m'
PINK  = '\033[95m'
CYAN  = '\033[96m'
ENDC  = '\033[0m'

try:
    import CGNS.MAP
    import CGNS.PAT.cgnsutils as CGU
except:
    raise ImportError(RED+'could not import CGNS.MAP, try installing it:\npip3 install --user h5py pycgns'+ENDC)

from .Node import Node
from .Zone import Zone
from .Base import Base
from .Tree import Tree
from . import Mesh

def load(filename):
    t, links, paths = CGNS.MAP.load(filename)
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
    t, l, p = CGNS.MAP.load( filename, subtree=path_map )
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

def merge( data ):
    if isinstance(data, dict):
        t = Tree( **data )
    else:
        t = Tree( )
        t.merge( *data )

    return t

def getZones( data ):
    t = merge( *data )
    return t.zones()

def getBases( data ):
    t = merge( *data )
    return t.bases()
