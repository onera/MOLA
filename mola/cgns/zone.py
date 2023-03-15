#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

'''
Implements class **Zone**, which inherits from :py:class:`Node`

21/12/2021 - L. Bernardos - first creation
'''

import numpy as np
import re
from .. import misc as m
from .node import Node


class Zone(Node):
    """docstring for Zone"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.setType('Zone_t')

        if self.value() is None:
            self.setValue(np.array([[2,1,0]],dtype=np.int32,order='F'))


        if not self.childNamed('ZoneType'):
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
                  GridLocation='guess', dtype=np.float64, return_type='list',
                  ravel=False):

        if GridLocation == 'guess':
            GridLocation = m.AutoGridLocation[ Container ]

        FlowSolution = self.get(Container,Depth=1)
        if not FlowSolution:
            FlowSolution = Node(Parent=self, Name=Container, Type='FlowSolution_t',
                Children=[Node(Name='GridLocation', Type='GridLocation_t',
                               Value=GridLocation)])
        ExistingLocation = self.inferLocation( Container )
        if GridLocation != ExistingLocation:
            MSG = ('you requested GridLocation={} at {}, but existing fields'
                   ' are located at {}.\n'
                   'Please adapt Container or GridLocation values').format(
                                    GridLocation,Container,ExistingLocation)
            raise ValueError(m.RED+MSG+m.ENDC)

        if isinstance(FieldNames,str): FieldNames = [ FieldNames ]

        arrays = []
        shape = self.value()
        if GridLocation == 'Vertex':
            for name in FieldNames:
                array = np.zeros(shape[:,0], dtype=dtype, order='F')
                arrays += [ array ]
                Node(Parent=FlowSolution, Name=name, Type='DataArray_t', Value=array)
                if ravel: array = array.ravel(order='K')
        elif GridLocation == 'CellCenter':
            for name in FieldNames:
                array = np.zeros(shape[:,1], dtype=dtype, order='F')
                arrays += [ array ]
                Node(Parent=FlowSolution, Name=name, Type='DataArray_t', Value=array)
                if ravel: array = array.ravel(order='K')
        else:
            raise AttributeError('GridLocation=%s not supported'%GridLocation)

        if return_type == 'list':
            if len(arrays) == 1: arrays = array
            return arrays
        elif return_type == 'dict':
            v = dict()
            for key, array in zip( FieldNames, arrays):
                v[key] = array
            return v
        else:
            AttributeError('return_type=%s not supported'%return_type)

    def fields(self, FieldNames, Container='FlowSolution',
             BehaviorIfNotFound='create', dtype=np.float64, return_type='list',
             ravel=False):
        if isinstance(FieldNames,str): FieldNames = [ FieldNames ]

        FlowSolution = self.childNamed( Container )

        if FlowSolution is None:
            if BehaviorIfNotFound == 'raise':
                raise ValueError('container %s not found in %s'%(Container,self.name()))
            elif BehaviorIfNotFound == 'create':
                return self.newFields(FieldNames, Container=Container,dtype=dtype)
            elif BehaviorIfNotFound == 'pass':
                return
            else:
                raise AttributeError(m.RED+'BehaviorIfNotFound=%s not supported'%BehaviorIfNotFound+m.ENDC)


        arrays = []
        for FieldName in FieldNames:
            FieldNode = FlowSolution.childNamed( FieldName )
            if FieldNode is None:
                if BehaviorIfNotFound == 'create':
                    array = self.newFields(FieldName,Container=Container,
                                           dtype=dtype, ravel=ravel)
                elif BehaviorIfNotFound == 'raise':
                    raise ValueError('%s not found in %s'%(FieldName,FlowSolution.path()))
                elif BehaviorIfNotFound == 'pass':
                    array = None

            else:
                array = FieldNode.value(ravel)

            arrays += [ array ]


        if return_type == 'list' or isinstance(return_type,list) or return_type is list:
            if len(arrays) == 1: return array
            return arrays
        elif return_type == 'dict' or isinstance(return_type,dict) or return_type is dict:
            v = dict()
            for key, array in zip( FieldNames, arrays):
                v[key] = array
            return v
        else:
            AttributeError('return_type=%s not supported'%return_type)

    def field(self, FieldName, **kwargs):
        if not isinstance(FieldName, str):
            raise AttributeError('FieldName must be of %s'%type(''))
        return self.fields( [FieldName], **kwargs )

    def xyz(self, Container='GridCoordinates', return_type='list',ravel=False):
        GC = self.get(Name='GridCoordinates', Depth=1)
        x = GC.get( Name='CoordinateX', Depth=1).value(ravel)
        y = GC.get( Name='CoordinateY', Depth=1).value(ravel)
        z = GC.get( Name='CoordinateZ', Depth=1).value(ravel)
        if return_type == 'list': return [x, y, z]
        else: return dict(CoordinateX=x, CoordinateY=y, CoordinateZ=z)

    def xy(self, Container='GridCoordinates', return_type='list',ravel=False):
        GC = self.get(Name='GridCoordinates', Depth=1)
        x = GC.get( Name='CoordinateX', Depth=1).value(ravel)
        y = GC.get( Name='CoordinateY', Depth=1).value(ravel)
        if return_type == 'list': return [x, y]
        else: return dict(CoordinateX=x, CoordinateY=y)

    def xz(self, Container='GridCoordinates', return_type='list',ravel=False):
        GC = self.get(Name='GridCoordinates', Depth=1)
        x = GC.get( Name='CoordinateX', Depth=1).value(ravel)
        z = GC.get( Name='CoordinateZ', Depth=1).value(ravel)
        if return_type == 'list': return [x, z]
        else: return dict(CoordinateX=x, CoordinateZ=z)

    def yz(self, Container='GridCoordinates', return_type='list',ravel=False):
        GC = self.get(Name='GridCoordinates', Depth=1)
        y = GC.get( Name='CoordinateY', Depth=1).value(ravel)
        z = GC.get( Name='CoordinateZ', Depth=1).value(ravel)
        if return_type == 'list': return [y, z]
        else: return dict(CoordinateY=y, CoordinateZ=z)

    def x(self, Container='GridCoordinates', return_type='list',ravel=False):
        GC = self.get(Name='GridCoordinates', Depth=1)
        x = GC.get( Name='CoordinateX', Depth=1).value(ravel)
        if return_type == 'list': return x
        else: return dict(CoordinateX=x)

    def y(self, Container='GridCoordinates', return_type='list',ravel=False):
        GC = self.get(Name='GridCoordinates', Depth=1)
        y = GC.get( Name='CoordinateY', Depth=1).value(ravel)
        if return_type == 'list': return y
        else: return dict(CoordinateY=y)

    def z(self, Container='GridCoordinates', return_type='list',ravel=False):
        GC = self.get(Name='GridCoordinates', Depth=1)
        z = GC.get( Name='CoordinateZ', Depth=1).value(ravel)
        if return_type == 'list': return z
        else: return dict(CoordinateZ=z)

    def allFields(self,include_coordinates=True,return_type='dict',ravel=False,
                  appendContainerToFieldName=False):

        arrays = self.xyz(ravel=ravel)
        FieldsNames = ['CoordinateX','CoordinateY','CoordinateZ']
        AllFlowSolutionNodes = self.group(Type='FlowSolution_t',Depth=1)
        NbOfContainers = len(AllFlowSolutionNodes)
        if NbOfContainers > 1 and not appendContainerToFieldName and return_type=='dict':
            MSG = ('allFields(): several containers where detected, use'
                ' appendContainerToFieldName=True for avoid keyword overriding')
            print(m.WARN+MSG+m.ENDC)
        for FlowSolution in AllFlowSolutionNodes:
            for child in FlowSolution.children():
                if child.type() != 'DataArray_t': continue
                if appendContainerToFieldName:
                    FieldsNames += [ FlowSolution.name()+'/'+child.name() ]
                else:
                    FieldsNames += [ child.name() ]
                arrays += [ child.value(ravel=ravel) ]

        if return_type == 'dict':
            v = dict()
            for key, array in zip( FieldsNames, arrays ):
                v[key] = array
            return v
        elif return_type == 'list':
            return arrays
        else:
            AttributeError('return_type=%s not supported'%return_type)

    def inferLocation(self, Container ):
        if Container == 'GridCoordinates': return 'Vertex'
        FlowSolution = self.get( Name=Container )
        if not FlowSolution:
            MSG = 'Container %s not found in %s'%(Container,self.path())
            raise ValueError(m.RED+MSG+m.ENDC)
        try:
            return FlowSolution.get( Name='GridLocation' ).value()
        except:
            if not FlowSolution.children(): return m.AutoGridLocation[ Container ]
            FieldSize = len(FlowSolution.children()[0].value(ravel=True))
            if FieldSize == zone.numberOfPoints(): return 'Vertex'
            elif FieldSize == zone.numberOfCells(): return 'CellCenter'
            else: raise ValueError('could not determine location of '+FlowSolution.path())

    def useEquation(self, equation, Container='FlowSolution', ravel=False):
        RegExpr = r"{[^}]*}"
        eqnVarsWithDelimiters = re.findall( RegExpr, equation )
        adaptedEqnVars = []

        v = self.allFields(ravel=ravel,appendContainerToFieldName=False)

        adaptedEqnVars, AllContainers = [], []
        for i, eqnVarWithDelimiters in enumerate(eqnVarsWithDelimiters):
            eqnVar = eqnVarWithDelimiters[1:-1]
            if eqnVar in ['CoordinateX','CoordinateY','CoordinateZ']:
                AllContainers.append( 'GridCoordinates' )
                adaptedEqnVars.append( "v['%s']"%eqnVar)

            elif len(eqnVar) == 1 and eqnVar.lower() in ['x','y','z']:
                AllContainers.append( 'GridCoordinates' )
                adaptedEqnVars.append( "v['Coordinate%s']"%eqnVar.upper() )

            else:
                eqnVarSplit = eqnVar.split('/')
                if len(eqnVarSplit) == 1:
                    eqnContainer = Container
                    eqnVarName = eqnVar
                else:
                    eqnContainer = eqnVarSplit[0]
                    eqnVarName = eqnVarSplit[1]

                eqnContainerAndName = eqnContainer+'/'+eqnVarName
                if eqnContainerAndName not in v:
                    if i ==0:
                        field = self.fields(eqnVarName, Container=eqnContainer,
                                    ravel=ravel, return_type='list')
                    else:
                        raise ValueError('unexpected')

                    v[eqnContainerAndName] = field
                AllContainers.append( eqnContainer )
                adaptedEqnVars.append( "v['%s']"%eqnContainerAndName )
        FirstMemberLocation = self.inferLocation(AllContainers[0])
        SecondMemberLocations = [self.inferLocation(c) for c in AllContainers[1:]]
        mixedLocations = False
        for i, SecondMemberLocation in enumerate(SecondMemberLocations):
            if FirstMemberLocation != SecondMemberLocation:
                mixedLocations = True
                break


        adaptedEqnVars[0] += '[:]'

        adaptedEquation = equation
        for keyword, adapted in zip(eqnVarsWithDelimiters, adaptedEqnVars):
            adaptedEquation = adaptedEquation.replace(keyword, adapted)

        try:
            exec(adaptedEquation,globals(),{'v':v,'np':np})
        except BaseException as e:
            print(m.RED+'ERROR : could not apply equation:\n'+adaptedEquation+m.ENDC)
            if mixedLocations:
                print(m.RED+'This may be due to mixing values located in both Vertex and CellCenter'+m.ENDC)
            raise e

    def hasFields(self):
        return bool( self.get( Type='FlowSolution_t', Depth=1 ) )
