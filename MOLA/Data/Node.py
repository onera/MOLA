'''
Implements class **Node**, which inherits from standard Python :py:class:`list`

Other classes inheriting from **Node** are: :py:class:`Tree`,
:py:class:`Base` and :py:class:`Zone`

21/12/2021 - L. Bernardos - first creation
'''

import sys
MIN_PYTHON = (3,6,2)
if sys.version_info < MIN_PYTHON:
    raise SystemError("Python %s or later is required.\n"%'.'.join([str(i) for i in MIN_PYTHON]))
import numpy as np
from fnmatch import fnmatch

import CGNS.MAP

class Node(list):
    """docstring for Node."""

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

    def parent(self): return self.Parent

    def path(self): return self.Path

    def save(self, filename, verbose=True):
        from .Tree import Tree
        node_to_save = self if isinstance(self, Tree) else Tree( self )
        links = node_to_save.getLinks()
        if verbose: print('saving %s ...'%filename)
        CGNS.MAP.save(filename, node_to_save, links=links)
        if verbose: print('saving %s ... ok'%filename)


    def name(self): return self[0]

    def printName(self): print(self.name())

    def value(self, ravel=False):
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

    def brothers(self, include_myself=True):
        if include_myself: return self.Parent.children()
        return [c for c in self.Parent.children() if c is not self]

    def type(self): return self[3]

    def printType(self): print(self.type())

    def setName(self, name):
        self[0] = name
        if hasattr(self,'Parent'): self._updateSelfAndChildrenPaths()

    def setValue(self, value):
        if isinstance(value,np.ndarray):
            if not value.flags['F_CONTIGUOUS']:
                print('WARNING: numpy array being set to node %s is not order="F"'%self.name())
        elif isinstance(value,list) and isinstance(value[0],str):
            value = np.array(' '.join(value),dtype='c',order='F').ravel()
        elif isinstance(value, int):
            value = np.array([value],dtype=np.int,order='F')
        elif isinstance(value, float):
            value = np.array([value],dtype=np.float,order='F')
        elif isinstance(value, str):
            value = np.array([value],dtype='c',order='F').ravel()
        elif value is None:
            value = None
        else:
            raise TypeError('type of value %s not recognized'%type(value))

        self[1] = value

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
        brothersNames = [c[0] for c in self[2]]
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

        if child[3] == 'Zone_t':
            from .Zone import Zone
            child = Zone(child, Parent = self, position=position)

        elif child[3] == 'CGNSBase_t':
            from .Base import Base
            child = Base(child, Parent = self, position=position)

        else:
            child = Node(child, Parent = self, position=position)

        child.Parent = self
        child.Path = self.Path+'/'+child.name()
        self._updateSelfAndChildrenPaths()

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
        selfBrothers = self.Parent.children()
        for i, n in enumerate(selfBrothers):
            if n is self:
                break
        nodeBrothers = node.Parent.children()
        for j, n in enumerate(nodeBrothers):
            if n is node:
                break
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
        try:
            if not Type.endswith('_t'):
                Type += '_t'
        except:
            pass
        for child in self.children():
            NameMatch = fnmatch(child.name(), Name) if Name is not None else True
            TypeMatch = fnmatch(child.type(), Type) if Type is not None else True
            ValueMatch = _compareValue(child, Value) if Value is not None else True

            if all([NameMatch, ValueMatch, TypeMatch]):
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
        try:
            if not Type.endswith('_t'):
                Type += '_t'
        except:
            pass
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
                # child.Parent = self
                # child.Path = self[0] + '/' + child[0]
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
            child = children[i]
            if isinstance(child, Node):
                child.Parent = self
                child.Path = self.Path + '/' + child[0]
                child._updateSelfAndChildrenPaths()
            else:
                if child[3] == 'Zone_t':
                    from .Zone import Zone
                    children[i] = Zone(child, Parent = self)

                elif child[3] == 'CGNSBase_t':
                    from .Base import Base
                    children[i] = Base(child, Parent = self)

                else:
                    children[i] = Node(child, Parent = self)

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
        node = self._copy(deep, None)
        return Node(node)

    def _copy(self, deep, parent):
        ValueIsNumpy = isinstance(self[1], np.ndarray)
        ValueCopy = self[1].copy(order='K') if deep and ValueIsNumpy else self[1]
        CopiedNode = [self[0], ValueCopy, [], self[3]]
        if parent is not None: parent[2].append(CopiedNode)
        for child in self[2]: child._copy(deep, CopiedNode)

        return CopiedNode

    def getAtPath(self, Path, path_is_relative=False):
        t = self if path_is_relative else self.getTopParent()
        PathElements = Path.split('/')
        n = t
        for p in PathElements[1:]:
            n = n.get(p, Depth=1)

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
        self.swap(full_node)

    def replaceLinks(self, starting_at_top_parent=False):
        n = self.getTopParent() if starting_at_top_parent else self
        for link in n.group( Type='Link_t' ): link.replaceLink()

    def reloadNodeData(self, filename):
        updated_node = readNode( filename, self.path() )
        self.setValue( updated_node.value() )

    def saveThisNodeOnly(self, filename ):
        flags = CGNS.MAP.S2P_UPDATE
        t = self.getTopParent()
        path = self.path().replace('CGNSTree','')
        CGNS.MAP.save( filename, t, update={path:self}, flags=flags)


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
