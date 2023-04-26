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
Implements class **Base**, which inherits from :py:class:`Node`

21/12/2021 - L. Bernardos - first creation
'''
import numpy as np
from .. import misc as m
from .node import Node
from .zone import Zone

class Base(Node):
    """docstring for Base"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setType('CGNSBase_t')

        if self.value() is None:
            BaseValue = np.array([3,3],dtype=np.int32,order='F')
            for child in self.children():
                if isinstance(child, Zone):
                    BaseValue = np.array([child.dim(),3],dtype=np.int32,order='F')
                    break
            self.setValue(BaseValue)

        if self.name() == 'Node': self.setName( 'Base' )

    def save(self,*args,**kwargs):
        from .Tree import Tree
        t = Tree()
        t.addChild( self )
        t.save(*args,**kwargs)

    def dim(self):
        return self.value()[0]

    def zones(self):
        return [c for c in self.children() if isinstance(c, Zone)]

    def setPhysicalDimension(self, PhysicalDimension):
        self.value()[0] = PhysicalDimension

    def isStructured(self):
        return all([zone.isStructured() for zone in self.zones()])
    
    def isUnstructured(self):
        return all([zone.isUnstructured() for zone in self.zones()])

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
