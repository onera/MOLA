'''
Tree.py module for CGNS tree operations in an object-oriented manner, while
preserving interpoperability with Cassiopee functional paradigm

21/12/2021 - L. Bernardos - first creation
'''

import sys
usingPython2 = sys.version_info[0] == 2
import numpy as np
from fnmatch import fnmatch

RED  = '\033[91m'
GREEN = '\033[92m'
WARN  = '\033[93m'
PINK  = '\033[95m'
CYAN  = '\033[96m'
ENDC  = '\033[0m'


class Node(list):
    """docstring for Node."""

    def __init__(self, args=['Node',None,[],'DataArray_t'], Parent=None, Children=[],
                             Name=None, Value=None, Type=None):
        list.__init__(self, args)
        if Name is not None: self[0] = Name
        if Value is not None: self.setValue(Value)
        if isinstance(Type,str):
            if not Type.endswith('_t'): Type += '_t'
            self[3] = Type
        self.Parent = Parent
        if isinstance(Parent, Node):
            self.Path = Parent.Path + '/' + self[0]
        else:
            self.Path = self[0]
        self.addChildren(Children)
        self._updateChildrenPaths()

    def save(self, filename):
        import Converter.PyTree as C
        C.convertPyTree2File(self, filename)

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
                    if usingPython2: return n.tostring()
                    else: return n.tostring().decode()
                for i in range(n.shape[1]):
                    if usingPython2: v = n[:,i].tostring()
                    else: v = n[:,i].tostring().decode()
                    out.append(v.strip())
                return out
            elif n.dtype.char == 'c':
                if len(n.shape) == 1:
                    if usingPython2: return n.tostring()
                    else: return n.tostring().decode()
                for i in range(n.shape[1]):
                    if usingPython2: v = n[:,i].tostring()
                    else: v = n[:,i].tostring().decode()
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
        if hasattr(self,'Parent'): self._updateChildrenPaths()

    def setValue(self, value):
        if isinstance(value,np.ndarray) and not value.flags['F_CONTIGUOUS']:
            print(WARN+'WARNING: numpy array being set to node %s is not order="F"'%self.name()+ENDC)
        elif isinstance(value,list) and isinstance(value[0],str):
            value = ' '.join(value)
        self[1] = value

    def setType(self, newType):
        try:
            if not newType.endswith('_t'):
                newType += '_t'
        except:
            pass

        self[3] = newType

    def addChild(self, child):
        if not isinstance(child, Node):
            child = Node(child, Parent = self)
        child.Parent = self
        child.Path = self.Path+'/'+child.name()
        self[2].append( child )
        child._updateChildrenPaths()

    def addChildren(self, children):
        for child in children: self.addChild(child)

    def setParent(self, Parent):
        if isinstance(Parent, Node):
            Parent.addChild(self)
        elif Parent is not None:
            Parent = Node(Parent, Children=[self])
        self.Parent = Parent

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
            else:
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

    def _updateChildrenPaths(self):
        if isinstance(self.Parent, Node):
            self.Path = self.Parent.Path+'/'+self.name()
        elif self.Parent is None:
            self.Path = self.name()
        children = self.children()
        for i in range(len(children)):
            child = children[i]
            if isinstance(child, Node):
                child.Parent = self
                child.Path = self[0] + '/' + child[0]
                child._updateChildrenPaths()
            else:
                children[i] = Node(child, Parent=self)

    def _updateAllPaths(self):
        t = self.getTopParent()
        t._updateChildrenPaths()

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


def load(*args, **kwargs):
    import Converter.PyTree as C
    t = C.convertFile2PyTree(*args, **kwargs)
    return Node(t)

def _compareValue(node, Value):
    NodeValue = node.value()
    if NodeValue is None: return False
    if isinstance(NodeValue, str): return fnmatch(NodeValue, Value)
    return False # TODO implement np.array comparisons
