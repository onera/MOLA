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
Implements class **Tree**, which inherits from :py:class:`Node`

21/12/2021 - L. Bernardos - first creation
'''

import numpy as np
from .. import misc as m
from .node import Node
from .base import Base
from .zone import Zone

class Tree(Node):
    """docstring for Tree"""
    def __init__(self, input=[], **kwargs):
        if isinstance(input, Node):
            super().__init__(Name='CGNSTree',Type='CGNSTree_t')
            self.addChildren( input.children() )
        else:
            super().__init__(Name='CGNSTree',Type='CGNSTree_t')
            if input:
                self.merge( input[2] )

        if not self.get('CGNSLibraryVersion', Depth=1):
            ver=Node(Name='CGNSLibraryVersion',
                     Value=np.array([3.1],dtype=np.float32),
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

    def base(self,**kwargs):
        kwargs['Type'] = 'CGNSBase_t'
        kwargs['Depth'] = 1
        return self.get(**kwargs)

    def setUniqueBaseNames(self):
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


    def merge(self, elements):

        if not isinstance(elements, list):
            AttributeError(m.RED+'elements must be a list or a Node'+m.ENDC)

        if isinstance(elements, Node):
            elements = [ elements ]
        else:
            try:
                if isinstance(elements[3], str):
                    elements = [ Node(elements) ]
            except IndexError:
                pass

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
                self.merge( t )

        self.findAndRemoveNodes(Name='CGNSLibraryVersion.*', Depth=1)
        self.setUniqueBaseNames()
        self.setUniqueZoneNames()

    def isStructured(self):
        return all([base.isStructured() for base in self.bases()])
    
    def isUnstructured(self):
        return all([base.isUnstructured() for base in self.bases()])

    def isHybrid(self):
        return not self.isStructured() and not self.isUnstructured()
    
    def newFields(self, FieldNames, Container='FlowSolution',
                  GridLocation='auto', dtype=np.float64, return_type='dict',
                  ravel=False):

        arrays = []
        zoneNames = []
        for zone in self.zones():
            zoneNames.append( zone.name() )
            arrays.append( zone.newFields(FieldNames, Container=Container, 
                GridLocation=GridLocation,dtype=dtype, return_type=return_type,
                ravel=False))

        if return_type == 'list':
            return arrays 
        else:
            v = dict()
            for name, array in zip(zoneNames, arrays):
                v[ name ] = array
            return v

    def getFields(self, FieldNames, Container='FlowSolution',
                  BehaviorIfNotFound='create', dtype=np.float64, return_type='dict',
                  ravel=False):

        arrays = []
        zoneNames = []
        for zone in self.zones():
            zoneNames.append( zone.name() )
            arrays.append( zone.newFields(FieldNames, Container=Container, 
                BehaviorIfNotFound=BehaviorIfNotFound,dtype=dtype,
                return_type=return_type,ravel=False))

        if return_type == 'list':
            return arrays 
        else:
            v = dict()
            for name, array in zip(zoneNames, arrays):
                v[ name ] = array
            return v

    def getAllFields(self,include_coordinates=True, return_type='dict',ravel=False):

        arrays = []
        zoneNames = []
        for zone in self.zones():
            zoneNames.append( zone.name() )
            arrays.append( zone.getAllFields(
                include_coordinates=include_coordinates,
                return_type=return_type,ravel=ravel) )

        if return_type == 'list':
            return arrays 
        else:
            v = dict()
            for name, array in zip(zoneNames, arrays):
                v[ name ] = array
            return v

    def useEquation(self, *args, **kwargs):
        for zone in self.zones(): zone.useEquation(*args, **kwargs)

    def numberOfPoints(self):
        return int( np.sum( [ z.numberOfPoints() for z in self.zones() ]) )

    def numberOfCells(self):
        return int( np.sum( [ z.numberOfPoints() for z in self.zones() ]) )

    def numberOfZones(self):
        return len( self.zones() ) 

    def numberOfBases(self):
        return len( self.bases() ) 