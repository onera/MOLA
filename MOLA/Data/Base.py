'''
Implements class **Base**, which inherits from :py:class:`Node`

21/12/2021 - L. Bernardos - first creation
'''

from .Core import np, RED,GREEN,WARN,PINK,CYAN,ENDC,CGM
from .Node import Node
from .Zone import Zone

class Base(Node):
    """docstring for Base"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setType('CGNSBase_t')

        if self.value() is None:
            BaseValue = np.array([3,3],dtype=np.int,order='F')
            for child in self.children():
                if isinstance(child, Zone):
                    BaseValue = np.array([child.dim(),3],dtype=np.int,order='F')
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

    def newFields(self, FieldNames, Container='FlowSolution',
                  GridLocation='auto', dtype=np.float, return_type='dict',
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
                  BehaviorIfNotFound='create', dtype=np.float, return_type='dict',
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