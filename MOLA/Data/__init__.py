'''
.. role:: python(code)
  :language: python
  :class: highlight

.. _CGNS: http://cgns.github.io/
.. _Cassiopee: http://elsa.onera.fr/Cassiopee/
.. _pycgns: https://pycgns.github.io/

*21/12/2021 - L. Bernardos - first creation*

Main subpackage for `CGNS`_ tree-based operations in an object-oriented manner.

Please use the following convention when importing:

.. code::

    import MOLA.Data as M

This is an experimental subpackage that explores an object-oriented implementation
of tree-based (`CGNS`_) operations. The objective of this approach is to make use
of object-oriented programming paradigm in order to attempt a readably,
comprehensible and efficient manipulation of `CGNS`_ components, with minimal
external dependencies, hence allowing for high-portability. Nevertheless, by 
construction, this approach allows for interoperability with other `CGNS`_ 
manipulation functional-paradigm libraries such as `Cassiopee`_ or `pycgns`_.

The main hierarchical structure of the node-based classes is as follows:

* :py:class:`list` :python:`['name', value, [children], 'type_t']`

    * :py:class:`~MOLA.Data.Node.Node`

        * :py:class:`~MOLA.Data.Tree.Tree`

        * :py:class:`~MOLA.Data.Base.Base`

        * :py:class:`~MOLA.Data.Zone.Zone`

            * :py:class:`~MOLA.Data.Mesh.Curves.Point.Point`

            * :py:class:`~MOLA.Data.Mesh.Curves.Line.Line`

            * :py:class:`~MOLA.Data.Mesh.Curves.Curve`

                * :py:class:`~MOLA.Data.LiftingLine.LiftingLine`


In the following, some relevant user-level functions and shortcuts are provided:
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
    '''
    Open a file and return a :py:class:`~MOLA.Data.Tree.Tree`.

    Parameters
    ----------

        filename : str
            relative or absolute path of the file name in ``*.cgns`` or ``*.hdf5``
            format containing the `CGNS`_ tree.

        only_skeleton : bool
            if :py:obj:`True`, then data associated to *DataArray_t* nodes is 
            not loaded.

            .. note::
                this is currently available with *h5py2cgns* backend

    Returns
    -------

        t : :py:class:`~MOLA.Data.Tree.Tree`
            the tree node contained in file

    Examples
    --------

        Let's suppose that you want to open data contained in an existing
        file named ``myfile.cgns``:

        ::

          import MOLA.Data as M
          t = M.load('myfile.cgns')

        will open the file, including numpy contained in *DataArray_t* nodes.
        If you want to open a file without loading *DataArray_t* data, then you
        can do it like this:

        ::

          import MOLA.Data as M
          t = M.load('myfile.cgns', only_skeleton=True)


    '''


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
    '''
    Make a merge of information contained in **data** and then save the resulting
    :py:class:`~MOLA.Data.Tree.Tree`.

    Parameters
    ----------

        data : list
            heterogeneous container of nodes compatible with :py:func:`merge`

        args
            mandatory comma-separated arguments of
            :py:class:`~MOLA.Data.Node.Node`'s :py:meth:`~MOLA.Data.Node.Node.save` method

        kwargs
            optional pair of ``keyword=value`` arguments of
            :py:class:`~MOLA.Data.Node.Node`'s :py:meth:`~MOLA.Data.Node.Node.save` method

    Returns
    -------

        t : :py:class:`~MOLA.Data.Tree.Tree`
            a tree including all merged nodes contained in **data**

    Examples
    --------
    
    For the following examples, we only need to import main :py:mod:`MOLA.Data` subpackage:

    ::
    
        import MOLA.Data as M

    Let us start easy. We create a node, and write it down into a file: 

    ::

        n = M.Node() # create a Node
        M.save( n, 'out.cgns' )   # save it into a file named out.cgns

    Please note that in this case :python:`M.save(n, 'out.cgns')` is 
    equivalent to :python:`n.save('out.cgns')`.

    Let us make things just a little bit more complicated. We create several
    nodes and we save them into a file using an auxiliary list:

    ::

        n1 = M.Node()  # create first node
        n2 = M.Node()  # create second node  
        M.save( [n1, n2], 'out.cgns' ) # save all into a file using a list


    and both nodes are written, producing the following tree structure:

    .. code-block:: text

        CGNSTree
        CGNSTree/CGNSLibraryVersion
        CGNSTree/Node
        CGNSTree/Node.0    

    .. note::
        nodes at same hierarchical level with identical names are automatically
        renamed. In this example, :python:`n2` was initally named :python:`'Node'`,
        but since :python:`n1` is located at same hierarchical level and is also
        named :python:`'Node'`, then :python:`n2` has been renamed :python:`'Node.0'`

    Conveniently, this also works in combination with standard `pycgns`_ 
    lists. For example, let us slightly modify previous example by including a
    CGNS node through its pure python list definition:

    ::

        n1 = M.Node()  # create first node using MOLA
        n2 = ['MyNode',None,[],'DataArray_t']  # create second node using a standard pycgns list
        M.save( [n1, n2], 'out.cgns' ) # save all into a file using a list

    Which produces the following tree structure:

    .. code-block:: text

        CGNSTree
        CGNSTree/CGNSLibraryVersion
        CGNSTree/Node
        CGNSTree/MyNode

    Now let us make things more complicated. Suppose we have two lists of nodes
    which may possibly include children nodes. We can effectively create another 
    list whose items are the two aforementioned lists, and pass it to the save
    function:

    ::

        # create a node named 'NodeA'
        nA = M.Node(Name='NodeA') 

        # create 3 children nodes using unique names and attach them to nA
        for _ in range(3): M.Node(Parent=nA, override_brother_by_name=False) 

        # let us make a copy of nA and rename it
        nB = nA.copy()
        nB.setName('NodeB')

        # now we have our first list of nodes (including children):
        first_list = [nA, nB]

        # We can repeat the previous operations in order to declare a second list
        nC = M.Node(Name='NodeC') 
        for _ in range(3): M.Node(Parent=nC, override_brother_by_name=False) 

        nD = nC.copy()
        nD.setName('NodeD')
        second_list = [nC, nD]

        # Now we have two lists of nodes (which includes children), and we want
        # to save all this information into a file. Then, we can simply do:

        M.save( [ first_list, second_list ], 'out.cgns')

    which will produce a tree with following structure:

    .. code-block:: text

        CGNSTree
        CGNSTree/CGNSLibraryVersion
        CGNSTree/NodeA
        CGNSTree/NodeA/Node
        CGNSTree/NodeA/Node.0
        CGNSTree/NodeA/Node.1
        CGNSTree/NodeB
        CGNSTree/NodeB/Node
        CGNSTree/NodeB/Node.0
        CGNSTree/NodeC
        CGNSTree/NodeC/Node
        CGNSTree/NodeC/Node.0
        CGNSTree/NodeC/Node.1
        CGNSTree/NodeD
        CGNSTree/NodeD/Node
        CGNSTree/NodeD/Node.0
        CGNSTree/NodeD/Node.1

    One may note that nodes ``NodeA``, ``NodeB``, ``NodeC`` and ``NodeD`` are 
    all put into the same hierarchical level, but their children remains in second
    level. Actually, the hierarchical level does **not**  depend on the items'
    arrangement of the argument list given to save function, but rather on their
    own `CGNS`_ path structure.

    Indeed, the save call of previous example would be equivalent to any of these
    calls:

    ::
        
        # equivalent calls:
        M.save( [nA, nB, nC, nD], 'out.cgns') 
        M.save( [ [nA, nB], [nC, nD]], 'out.cgns') 
        M.save( [nA, [nB], nC, nD], 'out.cgns') 
        M.save( [nA, nB, nC, [[[nD]]]], 'out.cgns') 
        M.save( [nA, [nB], [nC, [[nD]]]], 'out.cgns') 
        ...


    '''
    if type(data) in ( Tree, Base, Zone ):
        t = data
    elif isinstance(data, list):
        t = merge( data )
    elif isinstance(data, dict):
        t = merge( **data )
    else:
        raise TypeError('saving data of type %s not supported'%type(data))

    t.save(*args, **kwargs)

    return t

def merge(*data1, **data2 ):
    '''
    Merge nodes into a single :py:class:`~MOLA.Data.Tree.Tree` structure.

    Parameters
    ----------
        
        data1
            comma-separated arguments of type :py:class:`list` containing the 
            nodes to be merged

        data2
            pair of ``keyword=value`` where each value is a :py:class:`list`
            containing the nodes to be merged and ``keyword`` is the name of the
            new :py:class:`~MOLA.Data.Base.Base` container automatically created
            where nodes are being placed.

    Returns
    -------

        t : :py:class:`~MOLA.Data.Tree.Tree`
            a tree including all merged nodes contained in **data1** and/or **data2**

    Examples
    --------

    Declare a :py:class:`list` of :py:class:`~MOLA.Data.Tree.Tree` and get the
    :py:class:`~MOLA.Data.Tree.Tree` which contains them all:

    ::

        import MOLA.Data as M
        a = M.Zone(Name='A')
        b = M.Zone(Name='B')
        c = M.Zone(Name='C')

        t = M.merge(a,b,c)

    will produce the following tree structure:

    .. code-block:: text

        CGNSTree/CGNSLibraryVersion
        CGNSTree/Base
        CGNSTree/Base/A
        CGNSTree/Base/A/ZoneType
        CGNSTree/Base/B
        CGNSTree/Base/B/ZoneType
        CGNSTree/Base/C
        CGNSTree/Base/C/ZoneType

    .. hint:: 
        you may easily print the structure of a :py:class:`~MOLA.Data.Tree.Tree`
        (or any :py:class:`~MOLA.Data.Node.Node`) using :py:meth:`~MOLA.Data.Node.Node.printPaths()`

    Note that the function has automatically recognized that the input nodes are
    of type :py:class:`~MOLA.Data.Zone.Zone` and they have been put into a 
    container of type :py:class:`~MOLA.Data.Base.Base`, according to `CGNS`_ standard.
    
    You may want to put the zones in different bases. You can achieve this 
    easily using a pairs of ``keyword=value`` arguments:

    ::

        import MOLA.Data as M
        a = M.Zone(Name='A')
        b = M.Zone(Name='B')
        c = M.Zone(Name='C')

        t = M.merge( FirstBase=a, SecondBase=[b, c] )

    which will produce a :py:class:`~MOLA.Data.Tree.Tree` with following structure:

    .. code-block:: text

        CGNSTree
        CGNSTree/FirstBase
        CGNSTree/FirstBase/A
        CGNSTree/FirstBase/A/ZoneType
        CGNSTree/SecondBase
        CGNSTree/SecondBase/B
        CGNSTree/SecondBase/B/ZoneType
        CGNSTree/SecondBase/C
        CGNSTree/SecondBase/C/ZoneType

    .. hint:: 
        In Python you can replace a call of pairs of ``keyword=value`` with 
        an unpacking of a :py:class:`dict`. In other terms, this:

        >>> t = M.merge( FirstBase=a, SecondBase=[b, c] )

        is equivalent to

        >>> myDict = dict(FirstBase=a, SecondBase=[b, c])
        >>> t = M.merge( **myDict )

        note the use of the :py:class:`dict` unpacking operator :python:`**`

    .. hint:: 
        In Python you can replace a call of comma-separated arguments with 
        an unpacking of a :py:class:`list`. In other terms, this:

        >>> t = M.merge( a, b, c )

        is equivalent to

        >>> myList = [ a, b, c ]
        >>> t = M.merge( *myList )

        note the use of the :py:class:`list` unpacking operator :python:`*`

    .. seealso::
        Since this function is used internally in :py:func:`save`, refer to its doc
        for more relevant examples.


    '''

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
    '''
    Get a list of zones contained in **data**.
    
    .. note::
        this function makes literally:

        >>> zones = M.merge(data).getZones()

    Parameters
    ----------

        data : list
            heterogeneous container of nodes compatible with :py:func:`merge`

    Returns
    -------

        zones: list
            list of :py:class:`~MOLA.Data.Zone.Zone` contained in **data**

    Examples
    --------

    Gather all zones of a combination of list and an existing tree both 
    containing zones:

    ::

        import MOLA.Data as M

        # arbitrarily create an heterogeneous list containing zones
        a = M.Zone(Name='A')
        b = M.Zone(Name='B')
        c = M.Zone(Name='C')
        some_zones = [a, [b,c] ]

        # arbitrarily create a tree containing zones (possibly in several Bases)
        t = M.Tree( FirstBase=[M.Zone(Name='D'),M.Zone(Name='E')],
                   SecondBase=[M.Zone(Name='A'),M.Zone(Name='F')])

        # create a container including all data
        container = some_zones + [t]

        zones = M.getZones( container )
        for z in zones: print(z.name())

    This will produce:

    .. code-block:: text

        A.0
        B
        C
        D
        E
        A
        F

    .. important::
        since this function makes use of :py:func:`merge`, zones with identical
        names are renamed
    

    '''
    t = merge( data ) # TODO avoid merge
    return t.zones()

def getBases( data ):
    '''
    Get a list of bases contained in **data**.
    
    .. note::
        this function makes literally:

        >>> bases = M.merge(data).getBases()

    Parameters
    ----------

        data : list
            heterogeneous container of nodes compatible with :py:func:`merge`

    Returns
    -------

        bases: list
            list of :py:class:`~MOLA.Data.Base.Base` contained in **data**

    Examples
    --------

    Gather all bases of a combination of list and an existing tree both 
    containing bases:

    ::

        import MOLA.Data as M

        # arbitrarily create an heterogeneous list containing bases
        a = M.Base(Name='BaseA')
        b = M.Base(Name='BaseB')
        c = M.Base(Name='BaseC')
        some_bases = [a, [b,c] ]

        # arbitrarily create a tree containing bases
        t = M.Tree( FirstBase=[M.Zone(Name='D'),M.Zone(Name='E')],
                    SecondBase=[M.Zone(Name='A'),M.Zone(Name='F')])

        # create a container including all data
        container = some_bases + [t]

        bases = M.getBases( container )
        for base in bases: print(base.name())

    This will produce:

    .. code-block:: text

        BaseA
        BaseB
        BaseC
        FirstBase
        SecondBase

    .. important::
        since this function makes use of :py:func:`merge`, bases with identical
        names are renamed
    
    '''
    t = merge( data ) # TODO avoid merge
    return t.bases()


def useEquation(data, *args, **kwargs):
    '''
    Call the :py:meth:`~MOLA.Data.Zone.Zone.useEquation` method 
    for each :py:class:`~MOLA.Data.Zone.Zone` contained in 
    argument **data**

    Parameters
    ----------

        data : list
            heterogeneous container of nodes compatible with :py:func:`merge`

            .. note::
                :py:class:`~MOLA.Data.Zone.Zone`'s are modified    

        args
            mandatory comma-separated arguments of
            :py:class:`~MOLA.Data.Zone.Zone`'s :py:meth:`~MOLA.Data.Zone.Zone.useEquation` method

        kwargs
            optional pair of ``keyword=value`` arguments of
            :py:class:`~MOLA.Data.Zone.Zone`'s :py:meth:`~MOLA.Data.Zone.Zone.useEquation` method

    Returns
    -------

        t : :py:class:`~MOLA.Data.Tree.Tree`
            same result as :py:func:`merge`

    Examples
    --------

    Create two :py:class:`~MOLA.Data.Tree.Tree`, each containing a different 
    number of :py:class:`~MOLA.Data.Zone.Zone` and create a new field named 
    ``field`` attributing a specific value:

    ::

        import MOLA.Data as M


        zoneA = M.Mesh.Line( Name='zoneA', N=2 )
        zoneB = M.Mesh.Line( Name='zoneB', N=4 )
        zoneC = M.Mesh.Line( Name='zoneC', N=6 )

        tree1 = M.Tree(Base1=[zoneA, zoneB])
        tree2 = M.Tree(Base2=[zoneC])

        M.useEquation( [tree1, tree2], '{field} = 12.0' )

        for zone in zoneA, zoneB, zoneC:
            print(zone.name()+' has field '+str(zone.field('field')))


    ok

    '''

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

    return newZoneFromArrays(Name, ArraysNames, Arrays)
