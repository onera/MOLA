'''
Implements class **Tree**, which inherits from :py:class:`Node`

21/12/2021 - L. Bernardos - first creation
'''

import numpy as np
from .Node import Node
from .Base import Base
from .Zone import Zone

class Tree(Node):
    """docstring for Tree"""
    def __init__(self, input=[], **kwargs):
        if isinstance(input, Node):
            self.setName('CGNSTree')
            self.setType('CGNSTree_t')
            self.addChildren( input.children() )
        else:
            super().__init__(Name='CGNSTree',Type='CGNSTree_t')
            self.merge( input[2] )

        if not self.get('CGNSLibraryVersion', Depth=1):
            ver=Node(Name='CGNSLibraryVersion',
                     Value=3.1,
                     Type='CGNSLibraryVersion_t')
            self.addChild(ver,position=0)


        for k in kwargs:
            children = [kwargs[k]] if isinstance(kwargs[k], Node) else kwargs[k]
            Base(Name=k, Children=children, Parent=self,
                 override_brother_by_name=False)

        self.setUniqueBaseNames()
        self.setUniqueZoneNames()

    def bases(self):
        return [c for c in self.children() if isinstance(c, Base)]

    def baseNames(self):
        return [c.name() for c in self.children() if isinstance(c, Base)]

    def setUniqueBaseNames(self):
        BaseNames = self.baseNames()
        for base in self.bases():
            Name = base.name()
            RestOfBaseNames = [c.name() for c in self.children() if isinstance(c, Base) and c is not base]
            if base.name() in RestOfBaseNames:
                i = 0
                newName = Name+'.%d'%i
                while newName in RestOfBaseNames:
                    i += 1
                    newName = Name+'.%d'%i
                base.setName(newName)

    def zones(self):
        allzones = []
        for b in self.bases():
            allzones.extend(b.zones())
        return allzones

    def zoneNames(self):
        allzones = []
        for b in self.bases():
            allzones.extend([z.name() for z in b.children() if isinstance(z,Zone)])

    def setUniqueZoneNames(self):
        ZoneNames = self.zoneNames()
        for zone in self.zones():
            Name = zone.name()
            RestOfZoneNames = [z.name() for z in self.zones() if z is not zone]
            if zone.name() in RestOfZoneNames:
                i = 0
                newName = Name+'.%d'%i
                while newName in RestOfZoneNames:
                    i += 1
                    newName = Name+'.%d'%i
                zone.setName(newName)


    def merge(self, *elements):
        if isinstance(elements[0],str):
            if isinstance(elements[3], str) and len(elements)==4:
                elements = [Node(elements)]

        for t in elements:
            if isinstance(t, Tree):
                self.addChildren(t.children(), override_brother_by_name=False)
            elif isinstance(t, Base):
                self.addChild(t, override_brother_by_name=False)
            elif isinstance(t, Zone):
                try:
                    base = self.bases()[-1]
                except:
                    base = Base()
                base.addChild(t, override_brother_by_name=False)
                self.addChild(base, override_brother_by_name=True)
            elif isinstance(t, Node):
                self.addChild(t, override_brother_by_name=False)
            elif isinstance(t, list):
                self.merge( *t )

        self.findAndRemoveNodes(Name='CGNSLibraryVersion.*', Depth=1)
        self.setUniqueBaseNames()
        self.setUniqueZoneNames()
