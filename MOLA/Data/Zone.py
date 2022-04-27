'''
Implements class **Zone**, which inherits from :py:class:`Node`

21/12/2021 - L. Bernardos - first creation
'''

import numpy as np
from .Node import Node

AutoGridLocation = {'FlowSolution':'Vertex',
                    'FlowSolution#Center':'CellCenter',
                    'FlowSolution#EndOfRun':'CellCenter',
                    'FlowSolution#Init':'CellCenter',
                    'FlowSolution#SourceTerm':'CellCenter',
                    'FlowSolution#EndOfRun#Coords':'Vertex'}

class Zone(Node):
    """docstring for Zone"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setType('Zone_t')

        if self.value() is None:
            self.setValue(np.array([[2,1,0]],dtype=np.int,order='F'))


        if not self.get('ZoneType', Depth=1):
            Node(Name='ZoneType',Value='Structured',Type='ZoneType_t',Parent=self)

        if self.name() == 'Node': self.setName( 'Zone' )

    def save(self,*args,**kwargs):
        from .Tree import Tree
        t = Tree(Base=self)
        t.save(*args,**kwargs)

    def isStructured(self):
        return self.get('ZoneType',Depth=1).value() == 'Structured'

    def dim(self):
        if self.isStructured():
            try:
                return len(self.get('GridCoordinates',Depth=1).get('Coordinate*').value().shape)
            except:
                return self.value().shape[0]

    def numberOfPoints(self):
        return np.prod(self.value(), axis=0)[0]

    def numberOfCells(self):
        return np.prod(self.value(), axis=0)[1]

    def newFields(self, FieldNames, Container='FlowSolution',
                  GridLocation='auto', dtype=np.float, returned_type='list'):

        if GridLocation == 'auto':
            try: GridLocation = AutoGridLocation[ Container ]
            except KeyError: GridLocation = 'Vertex'

        FlowSolution = self.get(Container)
        if not FlowSolution:
            FlowSolution = Node(Parent=self, Name=Container, Type='FlowSolution_t',
                Children=[Node(Name='GridLocation', Type='GridLocation_t',
                               Value=GridLocation)])

        if isinstance(FieldNames,str): FieldNames = [ FieldNames ]

        shape = self.value()
        arrays = []
        if GridLocation == 'Vertex':
            for name in FieldNames:
                array = np.zeros(shape[:,0], dtype=dtype, order='F')
                arrays += [ array ]
                Node(Parent=FlowSolution, Name=name, Type='DataArray_t', Value=array)
        elif GridLocation == 'CellCenter':
            for name in FieldNames:
                array = np.zeros(shape[:,1], dtype=dtype, order='F')
                arrays += [ array ]
                Node(Parent=FlowSolution, Name=name, Type='DataArray_t', Value=array)
        else:
            raise AttributeError('GridLocation=%s not supported'%GridLocation)

        if returned_type == 'list':
            if len(arrays) == 1: arrays = array
            return arrays
        elif returned_type == 'dict':
            v = dict()
            for key, array in zip( FieldsNames, arrays):
                v[key] = array
            return v
        else:
            AttributeError('returned_type=%s not supported'%returned_type)            

    def getFields(self, FieldNames, Container='FlowSolution',
                  BehaviorIfNotFound='create', dtype=np.float, returned_type='list'):

        if GridLocation == 'auto':
            try: GridLocation = AutoGridLocation[ Container ]
            except KeyError: GridLocation = 'Vertex'

        if isinstance(FieldNames,str): FieldNames = [ FieldNames ]

        FlowSolution = self.get(Container)
        if not FlowSolution:
            if BehaviorIfNotFound == 'raise':
                raise ValueError('container %s not found in %s'%(Container,self.name()))
            elif BehaviorIfNotFound == 'create':
                return self.newFields(FieldNames, Container=Container,dtype=dtype)

        arrays = []
        for FieldName in FieldNames:
            try:
                array = FlowSolution.get( Name=FieldName ).value()
            except:
                array = self.newFields(FieldName,Container=Container,dtype=dtype)
            arrays += [ arrays ]

        if returned_type == 'list':
            if len(arrays) == 1: arrays = array
            return arrays
        elif returned_type == 'dict':
            v = dict()
            for key, array in zip( FieldsNames, arrays):
                v[key] = array
            return v
        else:
            AttributeError('returned_type=%s not supported'%returned_type)
