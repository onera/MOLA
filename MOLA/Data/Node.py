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
.. role:: python(code)
  :language: python
  :class: highlight

.. _CGNS: http://cgns.github.io/

Node class
==========

*21/12/2021 - L. Bernardos - first creation*
'''

from fnmatch import fnmatch
from . import Core
np = Core.np
RED = Core.RED 
GREEN = Core.GREEN
WARN = Core.WARN
PINK = Core.PINK
CYAN = Core.CYAN
ENDC = Core.ENDC


class Node(list):
    '''
    Implements class :py:class:`~MOLA.Data.Node.Node`, which inherits from standard
    Python :py:class:`list` with `CGNS`_ structure: 

    ::

        # name       value      children (list)   type (str)
        ['name', numpy.ndarray,             [],     'type_t']

    .. note::
        Some classes inheriting from :py:class:`~MOLA.Data.Node.Node` are:

        * :py:class:`~MOLA.Data.Tree.Tree`
        * :py:class:`~MOLA.Data.Base.Base`
        * :py:class:`~MOLA.Data.Zone.Zone`


    Creation of nodes
    -----------------

    Please note the following construction parameters for the creation of
    :py:class:`~MOLA.Data.Node.Node` objects:

    Parameters
    ----------

        args : :py:class:`list`
            a python `CGNS`_ list 

        Parent : :py:class:`~MOLA.Data.Node.Node` or :py:obj:`None`
            an existing :py:class:`~MOLA.Data.Node.Node` where the new :py:class:`~MOLA.Data.Node.Node`
            will be attached

        Children : :py:class:`list`
            a :py:class:`list` of :py:class:`~MOLA.Data.Node.Node` or a
            :py:class:`list` of `CGNS`_ lists. These will be the nodes attached 
            to the new :py:class:`~MOLA.Data.Node.Node`

        Name : :py:class:`str`
            the name of the new :py:class:`~MOLA.Data.Node.Node`

        Value : multiple
            the value to be attributed to the new :py:class:`~MOLA.Data.Node.Node`.
            It can be one of :

                * :py:obj:`None`
                
                * :py:class:`str` (or :py:class:`list` of :py:class:`str`)
            
                * :py:class:`int` (or :py:class:`list` of :py:class:`int`)
                
                * :py:class:`float` (or :py:class:`list` of :py:class:`float`)
                
                * :py:class:`numpy.ndarray`

        Type : :py:class:`str`
            The type of the `CGNS`_ node. 

            .. hint:: 
                you can omit the suffix ``_t`` if you like *(it will be 
                automatically added if omitted)*

        override_brother_by_name : :py:class:`bool`
            if :py:obj:`True`, all children at same hierarchical level with 
            same **Name** are overriden (only last is kept). If :py:obj:`False`,
            an incremental numeric tag is appended to the **Name** such that 
            no other brother has same name.

            .. important::
                brother nodes must have different names

        position : :py:class:`str` or :py:class:`int`
            the position among the children of **Parent** where the new 
            :py:class:`~MOLA.Data.Node.Node` will be inserted (defaults to :python:`'last'`)

            .. note::
                only relevant if **Parent** is provided


    Examples
    ********

    You can easily invoke (create) a :py:class:`~MOLA.Data.Node.Node` like this:

    ::

        import MOLA.Data as M
        node = M.Node()
        print(node)
        # >>> ['Node', None, [], 'DataArray_t']

    Of course, when invoking a :py:class:`~MOLA.Data.Node.Node`, you can specify 
    some of their attributes, like its **name**, its **value**, its **type** or
    its **children**:

    ::

        import MOLA.Data as M
        node = M.Node(Name='MyName', Value=[1,2,3,4], Type='DataArray')
        print(node)
        # >>> ['MyName', array([1, 2, 3, 4], dtype=int32), [], 'DataArray_t']

    
    Alternatively, you can use a python :py:class:`list` corresponding to your
    `CGNS`_ node:

    ::

        import numpy as np
        import MOLA.Data as M
        node = M.Node( ['name', np.array([0,1,2]), [], 'DataArray_t'] )
        print(node)
        # >>> ['name', array([0, 1, 2]), [], 'DataArray_t']

    However, since the new node may also belong to a different inherited class 
    (like a :py:class:`~MOLA.Data.Tree.Tree` or a :py:class:`~MOLA.Data.Base.Base`, etc...)
    it is more recommended to rather use the function :py:func:`~MOLA.Data.Node.castNode`
    as follows:

    ::

        import numpy as np
        import MOLA.Data as M
        node = M.castNode( ['name', np.array([0,1,2]), [], 'DataArray_t'] )
        print(node)
        # >>> ['name', array([0, 1, 2]), [], 'DataArray_t']


    '''

    def __init__(self, args=['Node',None,[],'DataArray_t'], Parent=None, Children=[],
                             Name=None, Value=None, Type=None,
                             override_brother_by_name=True,
                             position='last'):
        list.__init__(self, args)
        if Name is not None:
            self[0] = Name
        else:
            Name = self[0]
        if Value is not None: self.setValue(Value)
        if isinstance(Type,str):
            if not Type.endswith('_t'): Type += '_t'
            self[3] = Type
        self.Parent = Parent
        if isinstance(Parent, Node):
            self.Path = Parent.Path + '/' + self[0]
        else:
            self.Path = self[0]

        if Parent:
            brothers = Parent[2]
            brothersNames = [b[0] for b in brothers]


            willWrite = True
            if Name in brothersNames:
                willWrite = False
                if not override_brother_by_name:
                    i = 0
                    newName = Name+'.%d'%i
                    while newName in brothersNames:
                        i += 1
                        newName = Name+'.%d'%i
                    self[0] = Name = newName
                    willWrite = True

            if willWrite:
                if position == 'last':
                    Parent[2].append(self)
                elif isinstance(position,int):
                    Parent[2].insert(position, self)

        self[2] = []
        self[2].extend(args[2])
        self.addChildren(Children, override_brother_by_name)
        self._updateSelfAndChildrenPaths()

    def parent(self):
        '''
        Get the parent of a given :py:class:`~MOLA.Data.Node.Node`

        Returns
        -------

            node : :py:class:`~MOLA.Data.Node.Node` or :py:obj:`None`
                the parent :py:class:`~MOLA.Data.Node.Node` or if current node is
                not attached to any node, then returns :py:obj:`None`

        Example
        *******

        Create two nodes, with parent/child hierarchy, and get back the parent 
        from the child node:

        ::

            import MOLA.Data as M

            # create a node and attach it to another node 
            a = M.Node( Name='TheParent')
            b = M.Node( Name='TheChild', Parent=a )

            # show the actual hierarchy
            a.printPaths()
            # >>> TheParent
            # >>> TheParent/TheChild

            # get the parent of node b
            p = b.parent()

            print(p is a)
            # >>> True

            # get the parent of node a
            p = a.parent()
            print(p)
            # >>> None

        '''
        
        return self.Parent

    def path(self):
        '''
        Get the `CGNS`_ Path of a given node

        Returns
        -------

            Path : str
                :py:class:`str` of its path from the most top parent 
                :py:class:`~MOLA.Data.Node.Node`

        Example
        *******

        Create two nodes, with parent/child hierarchy, and show path of child:

        ::

            import MOLA.Data as M

            # create a node and attach it to another node 
            a = M.Node( Name='TheParent')
            b = M.Node( Name='TheChild', Parent=a )

            path = b.path() # get the path
            print( path ) 
            # >>> TheParent/TheChild


        '''

        return self.Path

    def save(self, filename, verbose=True):
        '''
        Save into a CGNS file the node and their children keeping **exactly**
        the same hierarchy as defined by their :py:meth:`~MOLA.Data.Node.Node.path`

        Parameters
        ----------

            filename : str
                the absolute or relative filename path where the :py:class:`~MOLA.Data.Node.Node`
                is being saved 

            verbose : bool
                if :py:obj:`True`, prints saving status into standard output

        Example
        *******

        Create a node and save it into a file

        ::

            import MOLA.Data as M
            n = M.Node( Name='jamon' )
            n.save('out.cgns', verbose=True)
            # >>> saving out.cgns ... ok


        '''
        from .Tree import Tree
        if isinstance(self, Tree):
            node_to_save = self
        elif self.type() == 'CGNSTree':
            node_to_save = Tree( self )
        else:
            node_to_save = Tree()
            self.attachTo(node_to_save)

        links = node_to_save.getLinks()

        if Core.settings.backend == 'h5py2cgns':
            from . import h5py2cgns as h
            if verbose: print('saving %s ... '%filename, end='')
            h.save(node_to_save, filename, links=links)
            if verbose: print('ok')

        elif Core.settings.backend == 'pycgns':
            import CGNS.MAP as CGM
            if verbose: print('saving %s ... '%filename, end='')
            CGM.save(filename, node_to_save, links=links)
            if verbose: print('ok')
        
        elif Core.settings.backend == 'cassiopee':
            import Converter.PyTree as C
            C.convertPyTree2File(node_to_save, filename, links=links)
        

        else:
            raise ModuleNotFoundError('%s backend not supported'%Core.settings.backend)

    def name(self):
        '''
        get the name of the current :py:class:`~MOLA.Data.Node.Node`

        Returns
        -------

            Name : str 
                the name of the node

        Example
        *******

        ::

            import MOLA.Data as M
            n = M.Node( Name='tortilla' )
            print( n.name() )
            # >>> tortilla


        '''
        
        return self[0]


    def printName(self):
        '''
        prints into standard output the name of the current :py:class:`~MOLA.Data.Node.Node`

        Example
        *******

        ::

            import MOLA.Data as M
            n = M.Node( Name='croquetas' )
            n.printName()
            # >>> croquetas


        '''
        
        print(self.name())

    def value(self, ravel=False):
        '''
        get the value associated to the current :py:class:`~MOLA.Data.Node.Node`

        Parameters
        ----------

            ravel : bool
                if :py:obj:`True` and when returned value is an object of type 
                :py:class:`numpy.ndarray`, then such returned :py:class:`numpy.ndarray`
                is a flattened view of the original, without copy, as in
                :py:func:`~numpy.ravel` function (with :python:`order='K'`)

        Returns
        -------

            value : multiple

                * the value is a chain of characters
                    a :py:class:`str` is returned except if the original chain of
                    characters contains spaces; in such case, a :py:class:`list`
                    of :py:class:`str` is returned. A copy is made. 

                * the value is numeric
                    a :py:class:`numpy.ndarray` is returned. No copy is made.
                    If :python:`ravel=True`, then a flattened view is returned
                
                * no value exist
                    the object :py:obj:`None` is returned

                * the value is not loaded into memory
                    the private :py:class:`str` :python:`'_skeleton'` is 
                    returned (see :py:meth:`~MOLA.Data.Node.Node.reloadNodeData`)

        Examples
        --------

        It is very important to note that in many cases (specially when :py:class:`str`
        are involved) the value returned by :py:meth:`~MOLA.Data.Node.Node.value`
        does **not** correspond to the actual object stored in :python:`node[1]`.

        For this reason, the following example will show the 
        differences between :python:`node[1]` and :python:`node.value()`

        In this example, we will be getting a :py:class:`str`:
        
        ::

            import MOLA.Data as M

            node = M.Node( Value='jamon' )

            value_str = node.value() # will return a readable str
            print(value_str)
            # >>> jamon

            value_np = node[1] # will return an unreadable numpy.ndarray
            print(value_np)
            # >>> [b'j' b'a' b'm' b'o' b'n']

        as shown, when the value has :py:class:`str`, it is preferred using 
        :python:`node.value()` instead of :python:`node[1]`.

        This is still true when the value is a chain of characters including 
        spaces, which is interpreted as different *words* and stored into a
        :py:class:`list`:

        ::

            import MOLA.Data as M

            node = M.Node( Value='jamon tortilla croquetas' )

            value_str = node.value() # will return a readable list of str
            print(value_str)
            # >>> ['jamon', 'tortilla', 'croquetas']


            value_np = node[1] # will return an unreadable numpy.ndarray
            print(value_np)
            # >>> [b'j' b'a' b'm' b'o' b'n' b' ' b't' b'o' b'r' b't' b'i' b'l' b'l' b'a'
            #      b' ' b'c' b'r' b'o' b'q' b'u' b'e' b't' b'a' b's']

        Once again, you can see that working with 

        >>> ['jamon', 'tortilla', 'croquetas']

        is much more practical than working with 

        >>> [b'j' b'a' b'm' b'o' b'n' b' ' b't' b'o' b'r' b't' b'i' b'l' b'l' b'a' b' ' b'c' b'r' b'o' b'q' b'u' b'e' b't' b'a' b's']

        so, again, you will prefer :python:`node.value()` over :python:`node[1]`

        If the value contained in the :py:class:`~MOLA.Data.Node.Node` is a 
        numeric :py:class:`numpy.ndarray`, then its value is always directly 
        returned, without copy. Hence, you can make modifications of the :py:class:`~MOLA.Data.Node.Node`,
        or share its value with other nodes without concern. If the :py:class:`numpy.ndarray`
        is multi-dimensional, it is sometimes preferred to work with a flattened 
        view. This is why the option **ravel** is included in :py:meth:`~MOLA.Data.Node.Node.value`.
        The next example illustrates how to create a :py:class:`numpy.ndarray`,
        share it between two different nodes, get it in flattened view,
        modify it, and still check that no copy is made:

        ::

            import MOLA.Data as M
            import numpy as np 

            # create a multi-dimensional array to be attributed to several nodes
            # it is very important to set order='F' !
            array = np.array( [[0,1,2],
                               [3,4,5],
                               [6,7,8]], order='F' )

            # create our two nodes, and attribute their values to our array
            jamon    = M.Node( Name='jamon', Value=array )
            tortilla = M.Node( Name='tortilla', Value=array )

            # get the value of jamon
            jamon_value = jamon.value() # in this case, the same as jamon[1]
            print(jamon_value)
            # >>> [[0 1 2]
            #      [3 4 5]
            #      [6 7 8]]


            # get a flattened view of the array of node tortilla
            tortilla_value = tortilla.value(ravel=True)
            print(tortilla_value)
            # >>> [0 3 6 1 4 7 2 5 8]


            # our arrays share memory
            print(np.shares_memory(tortilla_value, jamon_value))
            # >>> True
            print(np.shares_memory(tortilla_value, array))
            # >>> True

            # hence we can modify it in different fashions, all changes will be propagated
            tortilla_value[0] = 12
            array[1,:] = -2

            print(array)
            print(jamon_value)
            print(tortilla_value)
            # >>> [[12  1  2]
            #      [-2 -2 -2]
            #      [ 6  7  8]]
            #     [[12  1  2]
            #      [-2 -2 -2]
            #      [ 6  7  8]]
            #     [12 -2  6  1 -2  7  2 -2  8]

        See also
        --------

        :py:func:`numpy.shares_memory`
        
        :py:obj:`numpy.ndarray.flags`

        '''

        v = self._value(ravel)
        if isinstance(v,str):
            words = v.split(' ')
            if len(words) > 1:
                return words
        return v

    def _value(self, ravel):
        n = self[1]
        if isinstance(n, np.ndarray):
            out = []
            if n.dtype.char == 'S':
                if len(n.shape) == 1:
                    return n.tostring().decode()
                elif len(n.shape) == 0:
                    return n.tostring().decode()
                for i in range(n.shape[1]):
                    v = n[:,i].tostring().decode()
                    out.append(v.strip())
                return out
            elif n.dtype.char == 'c':
                if len(n.shape) == 1:
                    return n.tostring().decode()
                elif len(n.shape) == 0:
                    return n.tostring().decode()
                for i in range(n.shape[1]):
                    v = n[:,i].tostring().decode()
                    out.append(v.strip())
                return out
            elif ravel:
                return n.ravel(order='K')
        return n

    def printValue(self): print(self.value())

    def children(self): return self[2]

    def hasChildren(self): return bool(self.children())

    def brothers(self, include_myself=True):
        if include_myself: return self.Parent.children()
        return [c for c in self.Parent.children() if c is not self]

    def hasBrothers(self): return bool(self.brothers())

    def type(self): return self[3]

    def printType(self): print(self.type())

    def setName(self, name):
        self[0] = name
        if hasattr(self,'Parent'): self._updateSelfAndChildrenPaths()

    def setValue(self, value):
        if isinstance(value,np.ndarray):
            if not value.flags['F_CONTIGUOUS']:
                print('WARNING: numpy array being set to node %s is not order="F"'%self.name())
            self[1] = np.atleast_1d(value)
        elif isinstance(value,list) or isinstance(value,tuple):
            if isinstance(value[0],str):
                value = np.array(' '.join(value),dtype='c',order='F').ravel()
            elif isinstance(value[0],float):
                value = np.array(value,dtype=float,order='F')
            elif isinstance(value[0],int):
                value = np.array(value,dtype=np.int32,order='F')
            else:
                MSG = ('could not make a numpy array from an object of type {} '
                       'with first element of type {}').format(type(value),
                                                               type(value[0]))
                raise TypeError(RED+MSG+ENDC)
            self[1] = np.atleast_1d(value)
        elif isinstance(value, int) or isinstance(value, bool):
            value = np.array([value],dtype=np.int32,order='F')
            self[1] = np.atleast_1d(value)

        elif isinstance(value, float):
            value = np.array([value],dtype=np.float64,order='F')
            self[1] = np.atleast_1d(value)

        elif isinstance(value, str):
            value = np.array([value],dtype='c',order='F').ravel()
            self[1] = np.atleast_1d(value)

        elif value is None:
            self[1] = None

        else:
            MSG = 'type of value %s not recognized'%type(value)
            raise TypeError(RED+MSG+ENDC)

        

    def setType(self, newType):
        try:
            if not newType.endswith('_t'):
                newType += '_t'
        except:
            pass

        self[3] = newType

    def getChildrenNames(self):
        return [c[0] for c in self[2]]

    def addChild(self, child, override_brother_by_name=True,
                 position='last'):
        brothersNames = [ c[0] for c in self[2] ]
        childname = child[0]

        if childname in brothersNames:
            if override_brother_by_name:
                self.get(childname).remove()
            else:
                i = 0
                newchildname = childname+'.%d'%i
                while newchildname in brothersNames:
                    i += 1
                    newchildname = childname+'.%d'%i
                child[0] = newchildname

        child = castNode( child )
        child.Parent = self
        child.Path = self.Path + '/' + child[0]
        if position == 'last':
            self[2].append(child)
        elif isinstance(position,int):
            self[2].insert(position, child)
        child._updateSelfAndChildrenPaths()


    def addChildren(self, children, override_brother_by_name=True):
        if isinstance(children, Node):
            children = [children]
        for child in children:
            self.addChild(child, override_brother_by_name)

    def attachTo(self, Parent, override_brother_by_name=True, position='last'):
        if isinstance(Parent, Node):
            Parent.addChild( self , override_brother_by_name, position)
        elif Parent is not None:
            Parent = Node(Parent, Children=[self])

    def dettach(self):
        try:
            brothers = self.Parent.children()
            for i, n in enumerate(brothers):
                if n is self:
                    brothers.pop(i)
        except AttributeError as error:
            if self.Parent is None:
                raise AttributeError('cannot dettach from no Parent -> Node (%s)'%self.name())
            else:
                raise AttributeError(error)
        self.Parent = None
        self.Path = self.name()
        self._updateSelfAndChildrenPaths()

    def swap(self, node):

        if self.Parent is not None:
            selfBrothers = self.Parent.children()
            for i, n in enumerate(selfBrothers):
                if n is self:
                    break
        else:
            selfBrothers = [ self ]
            i = 0

        if node.Parent is not None:
            nodeBrothers = node.Parent.children()
            for j, n in enumerate(nodeBrothers):
                if n is node:
                    break
        else:
            nodeBrothers = [ node ]
            j = 0
        selfBrothers[i], nodeBrothers[j] = nodeBrothers[j], selfBrothers[i]
        self.Parent, node.Parent = node.Parent, self.Parent
        self._updateSelfAndChildrenPaths()
        node._updateSelfAndChildrenPaths()

    def moveTo(self, Parent, position='last'):
        if self.Parent: self.dettach()
        self.attachTo(Parent, position=position)

    def getTopParent(self):
        TopParent = self.Parent
        if TopParent is not None:
            return TopParent.getTopParent()
        else:
            return self

    def get(self, Name=None, Value=None, Type=None, Depth=100):
        if Depth < 1: return
        if Type is not None and not Type.endswith('_t'): Type += '_t'
        for child in self.children():
            NameMatch = fnmatch(child.name(), Name) if Name is not None else True
            TypeMatch = fnmatch(child.type(), Type) if Type is not None else True
            ValueMatch = _compareValue(child, Value) if Value is not None else True
            if NameMatch == ValueMatch == TypeMatch == True:
                found_node = child
            else:
                found_node = child.get(Name, Value, Type, Depth-1)

            if found_node is not None: return found_node

    def _group(self, Name, Value, Type, Depth, Found):
        if Depth < 1: return
        for child in self.children():
            NameMatch = fnmatch(child.name(), Name) if Name is not None else True
            TypeMatch = fnmatch(child.type(), Type) if Type is not None else True
            ValueMatch = _compareValue(child, Value) if Value is not None else True

            if all([NameMatch, ValueMatch, TypeMatch]):
                Found.append( child )
            child._group(Name,Value,Type,Depth-1,Found)

    def group(self, Name=None, Value=None, Type=None, Depth=100):
        if Type is not None and not Type.endswith('_t'): Type += '_t'
        Found = []
        self._group(Name, Value, Type, Depth, Found)
        return Found

    def _adaptChildren(self):
        children = self[2]
        for i in range(len(children)):
            child = children[i]
            if not isinstance(child, Node):
                children[i] = Node(child, Parent=self)
            else:
                child._adaptChildren()

    def getPaths(self, starting_at_top_parent=False):

        if starting_at_top_parent: node = self.getTopParent()
        else: node = self
        Paths = [node.Path]
        for child in node.children():
            Paths.extend(child.getPaths(starting_at_top_parent=False))
        return Paths

    def _updateSelfAndChildrenPaths(self):
        if isinstance(self.Parent, Node):
            self.Path = self.Parent.Path+'/'+self[0]
        elif self.Parent is None:
            self.Path = self[0]
        children = self[2]
        for i in range(len(children)):
            child = castNode( children[i] )
            child.Parent = self
            child.Path = self.Path + '/' + child[0]
            child._updateSelfAndChildrenPaths()

    def _updateAllPaths(self):
        t = self.getTopParent()
        t._updateSelfAndChildrenPaths()

    def getParent(self, Name=None, Value=None, Type=None, Depth=100):
        if Depth < 1: return
        Parent = self.Parent
        if not Parent: return
        NameMatch = fnmatch(Parent.name(), Name) if Name is not None else True
        TypeMatch = fnmatch(Parent.type(), Type) if Type is not None else True
        ValueMatch = _compareValue(Parent, Value) if Value is not None else True

        if all([NameMatch, ValueMatch, TypeMatch]):
            found_node = Parent
        else:
            found_node = Parent.getParent(Name, Value, Type, Depth-1)

        if found_node is not None: return found_node

    def printPaths(self, starting_at_top_parent=False):
        for path in self.getPaths(starting_at_top_parent): print(path)

    def remove(self):
        try:
            brothers = self.Parent.children()
            for i, n in enumerate(brothers):
                if n is self:
                    del brothers[i]
                    return
        except AttributeError as error:
            if self.Parent is None:
                raise AttributeError('cannot remove a Parent Node (%s)'%self.name())
            else:
                raise AttributeError(error)

    def findAndRemoveNode(self, **kwargs):
        n = self.get(**kwargs)
        if n is not None: n.remove()

    def findAndRemoveNodes(self, **kwargs):
        nodes = self.group(**kwargs)
        for n in nodes: n.remove()

    def copy(self, deep=False):
        ValueIsNumpy = isinstance(self[1], np.ndarray)
        ValueCopy = self[1].copy(order='K') if deep and ValueIsNumpy else self[1]
        CopiedNode = self.__class__()
        CopiedNode.setName( self[0] )
        CopiedNode.setValue( ValueCopy )
        CopiedNode.setType( self[3] )
        for child in self[2]: CopiedNode.addChild( child.copy(deep) )

        return CopiedNode

    def getAtPath(self, Path, path_is_relative=False):
        t = self if path_is_relative else self.getTopParent()
        PathElements = Path.split('/')
        n = t
        for p in PathElements[1:]:
            n = n.get(p, Depth=1)
        
        if n is self: return None

        return n

    def addLink(self, path='', target_file='', target_path=''):
        if not path.startswith('CGNSTree'):
            if not path.startswith('/'):
                path = 'CGNSTree/'+path
            else:
                path = 'CGNSTree'+path

        if not target_path.startswith('CGNSTree'):
            if not target_path.startswith('/'):
                target_path = 'CGNSTree/'+target_path
            else:
                target_path = 'CGNSTree'+target_path

        path_levels = path.split('/')
        new_node_name = path_levels[-1]
        parent_path = '/'.join(path_levels[:-1])
        parent_node = self.getAtPath( parent_path )
        if not parent_node:
            raise ValueError('could not add link %s because node %s was not found'%(path, parent_path))

        new_node_with_link = Node(Name=new_node_name,
                                  Value=['target_file:'+target_file,
                                         'target_path:'+target_path],
                                  Type='Link_t',
                                  Parent=parent_node)

    def getLinks(self):
        links_nodes = self.group( Type='Link_t' )
        links = []
        for n in links_nodes:
            value = n.value()

            if value is None:
                raise ValueError('node of type Link_t %s has None value'%n.path())

            for v in value:
                if v.startswith('target_file:'):
                    target_file = v.replace('target_file:','')
                elif v.startswith('target_path:'):
                    target_path = v.replace('target_path:','')
            path = n.path().replace('CGNSTree','')
            target_path = target_path.replace('CGNSTree','')
            link_CGNS_MAP = ['',target_file, path, target_path, 5]
            links.append( link_CGNS_MAP )

        return links

    def replaceLink(self):
        if self.type() != 'Link_t': return

        value = self.value()
        for v in value:
            if v.startswith('target_file:'):
                target_file = v.replace('target_file:','')
            elif v.startswith('target_path:'):
                target_path = v.replace('target_path:','')
        full_node = readNode(target_file, target_path)
        self.setValue(full_node.value())
        self.setType(full_node.type())

    def replaceLinks(self, starting_at_top_parent=False):
        n = self.getTopParent() if starting_at_top_parent else self
        for link in n.group( Type='Link_t' ): link.replaceLink()

    def reloadNodeData(self, filename):
        updated_node = readNode( filename, self.path() )
        self.setValue( updated_node.value() )

    def saveThisNodeOnly( self, filename ):

        if Core.settings.backend == 'h5py2cgns':
            from . import h5py2cgns as h
            value = self.value()
            if isinstance(value,str) and value == '_skeleton':
                raise IOError('%s cannot write a skeleton node'%self.path())
            path = self.path().replace('CGNSTree/','')
            f = h.load_h5(filename,'r+')
            node_exists = True if path in f else False
            if not node_exists:
                group = h.nodelist_to_group(f, self, path)
            else:
                if isinstance(value, list) and isinstance(value[0],str):
                    newvalue = ' '.join(value)
                elif isinstance(value, str):
                    newvalue = value
                else:
                    newvalue = self[1]
                h._setData(f[path], newvalue)

        elif Core.settings.backend == 'pycgns':
            import CGNS.MAP as CGM
            t = self.getTopParent()
            flags = CGM.S2P_UPDATE
            path = self.path().replace('CGNSTree','')
            CGM.save(filename, t, update={path:self}, flags=flags)

        elif Core.settings.backend == 'cassiopee':
            import Converter.Filter as Filter
            # NOTE beware of BUG https://elsa.onera.fr/issues/10833
            Filter.writeNodesFromPaths(filename, [self.path()], [self], mode=1)

        else:
            raise ModuleNotFoundError('%s backend not supported'%Core.settings.backend)


    def setParameters(self, ContainerName, ContainerType='UserDefinedData_t',
                      ParameterType='DataArray_t', **parameters):

        Container = self.get( Name=ContainerName, Depth=1 )
        if not Container:
            Container = Node(Parent=self, Name=ContainerName, Type=ContainerType)
        for parameterName in parameters:
            parameterValue = parameters[parameterName]
            if isinstance(parameterValue, dict):
                Container.setParameters(parameterName,
                    ContainerType=ContainerType,
                    ParameterType=ParameterType,**parameterValue)
            else:
                paramNode = Container.get( Name=parameterName )
                if paramNode:
                    paramNode.setValue( parameterValue )
                    paramNode.setType( ParameterType )
                else:
                    Node(Parent=Container,Name=parameterName,
                         Value=parameterValue,Type=ParameterType)

        return self.getParameters( ContainerName )

    def getParameters(self, ContainerName):
        Container = self.get( Name=ContainerName, Depth=1 )
        Params = dict()
        if Container is None:
            raise ValueError('node %s not found in %s'%(ContainerName,self.path()))

        for param in Container.children():
            if param.children():
                Params[param.name()] = Container.getParameters(param.name())
            else:
                Params[param.name()] = param.value()

        return Params

    def childNamed(self, Name):
        for n in self.children():
            if n[0] == Name:
                return n

    def replaceSkeletonWithDataRecursively(self, filename):
        nodes = self.group(Value='_skeleton')
        if self.value() == '_skeleton': nodes += [self]
        for n in nodes: n.reloadNodeData(filename)

def _compareValue(node, Value):
    NodeValue = node.value()
    if NodeValue is None: return False
    if isinstance(Value, str) and isinstance(NodeValue, str):
        return fnmatch(NodeValue, Value)
    if isinstance(NodeValue, np.ndarray) and not isinstance(NodeValue, str):
        try:
            areclose = np.allclose(NodeValue, Value)
        except:
            areclose = False
        return areclose
    return False


def castNode( NodeOrNodelikeList ):
    '''
    toto
    '''

    if not isinstance(NodeOrNodelikeList, Node):
        node = Node(NodeOrNodelikeList)
    else:
        node = NodeOrNodelikeList

    for i, n in enumerate(node[2]):
        node[2][i] = castNode(n)

    if node[3] == 'Zone_t':
        from .Zone import Zone
        if not isinstance(node, Zone):
            node = Zone(node)
        try: Kind = node.childNamed('.Component#Info').childNamed('kind').value()
        except: Kind = None
        if Kind is None:
            if node.isStructured() and node.dim() == 1:
                from .Mesh.Curves import Curve
                if not isinstance(node, Curve):
                    node = Curve(node)
        elif Kind == 'LiftingLine':
            from .LiftingLine import LiftingLine
            if not isinstance(node, LiftingLine):
                node = LiftingLine(node)
        else:
            raise IOError('kind of zone "%s" not implemented'%Kind)

    elif node[3] == 'CGNSBase_t':
        from .Base import Base
        if not isinstance(node, Base):
            node = Base(node)

    elif node[3] == 'CGNSTree_t':
        from .Tree import Tree
        t = Tree(node)
        node = t

    return node

def readNode(filename, path):

    if path.startswith('CGNSTree/'):
        path_map = path.replace('CGNSTree','')
    else:
        path_map = path

    if Core.settings.backend == 'h5py2cgns':
        from . import h5py2cgns as h
        if path_map.startswith('/'): path_map = path_map[1:]
        f =h.load_h5(filename)
        group = f[ path_map ]
        node = h.build_cgns_nodelist( group )
        node = castNode( node )

    elif Core.settings.backend == 'pycgns':
        import CGNS.MAP as CGM
        t, _, _ = CGM.load( filename, subtree=path_map )
        t = castNode(t)
        node = t.getAtPath( path )
    
    elif Core.settings.backend == 'cassiopee':
        import Converter.Filter as Filter
        node = Filter.readNodesFromPaths(filename, [path_map])[0]
        node = castNode( node )

    
    else:
        raise ModuleNotFoundError('%s backend not supported'%Core.settings.backend)

    if node is None:
        raise ValueError("node %s not found in %s"%( path, filename ))


    return node
