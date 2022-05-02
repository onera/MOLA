'''
Implements class **Zone**, which inherits from :py:class:`Node`

21/12/2021 - L. Bernardos - first creation
'''

from .Core import (np,RED,GREEN,WARN,PINK,CYAN,ENDC,CGM,
                   AutoGridLocation, re)

from .Node import Node


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
                  GridLocation='auto', dtype=np.float, return_type='list',
                  ravel=False):

        if GridLocation == 'auto':
            try: GridLocation = AutoGridLocation[ Container ]
            except KeyError: GridLocation = 'Vertex'

        FlowSolution = self.get(Container)
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
            raise ValueError(RED+MSG+ENDC)

        if isinstance(FieldNames,str): FieldNames = [ FieldNames ]

        shape = self.value()
        arrays = []
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

    def getFields(self, FieldNames, Container='FlowSolution',
             BehaviorIfNotFound='create', dtype=np.float, return_type='list',
             ravel=False):
        if isinstance(FieldNames,str): FieldNames = [ FieldNames ]

        FlowSolution = self.get(Name=Container)
        if not FlowSolution:
            if BehaviorIfNotFound == 'raise':
                raise ValueError('container %s not found in %s'%(Container,self.name()))
            elif BehaviorIfNotFound == 'create':
                return self.newFields(FieldNames, Container=Container,dtype=dtype)
            elif BehaviorIfNotFound == 'pass':
                return
            else:
                raise AttributeError(RED+'BehaviorIfNotFound=%s not supported'%BehaviorIfNotFound+ENDC)


        arrays = []
        for FieldName in FieldNames:
            try:
                array = FlowSolution.get( Name=FieldName ).value(ravel=ravel)
            except:
                if BehaviorIfNotFound == 'create':
                    array = self.newFields(FieldName,Container=Container,
                                           dtype=dtype, ravel=ravel)
                elif BehaviorIfNotFound == 'raise':
                    raise ValueError('%s not found in %s'%(FieldName,FlowSolution.path()))
                elif BehaviorIfNotFound == 'pass':
                    array = None

            arrays += [ array ]

        if return_type == 'list':
            if len(arrays) == 1: return array
            return arrays
        elif return_type == 'dict':
            v = dict()
            for key, array in zip( FieldNames, arrays):
                v[key] = array
            return v
        else:
            AttributeError('return_type=%s not supported'%return_type)

    def getXYZ(self, Container='GridCoordinates', return_type='list',ravel=False):
        return self.getFields(['CoordinateX','CoordinateY','CoordinateZ'],
                                Container=Container, BehaviorIfNotFound='raise',
                                ravel=ravel)

    def getXY(self, Container='GridCoordinates', return_type='list',ravel=False):
        return self.getFields(['CoordinateX','CoordinateY'],
                                Container=Container, BehaviorIfNotFound='raise',
                                ravel=ravel)

    def getXZ(self, Container='GridCoordinates', return_type='list',ravel=False):
        return self.getFields(['CoordinateX','CoordinateZ'],
                                Container=Container, BehaviorIfNotFound='raise',
                                ravel=ravel)

    def getYZ(self, Container='GridCoordinates', return_type='list',ravel=False):
        return self.getFields(['CoordinateY','CoordinateZ'],
                                Container=Container, BehaviorIfNotFound='raise',
                                ravel=ravel)

    def getX(self, Container='GridCoordinates', return_type='list',ravel=False):
        return self.getFields('CoordinateX',
                                Container=Container, BehaviorIfNotFound='raise',
                                ravel=ravel)

    def getY(self, Container='GridCoordinates', return_type='list',ravel=False):
        return self.getFields('CoordinateY',
                                Container=Container, BehaviorIfNotFound='raise',
                                ravel=ravel)

    def getZ(self, Container='GridCoordinates', return_type='list',ravel=False):
        return self.getFields('CoordinateZ',
                                Container=Container, BehaviorIfNotFound='raise',
                                ravel=ravel)

    def getAllFields(self, include_coordinates=True, return_type='dict',
                           ravel=False):

        arrays = self.getXYZ(ravel=ravel)
        FieldsNames = ['CoordinateX','CoordinateY','CoordinateZ']
        for FlowSolution in self.group(Type='FlowSolution_t',Depth=1):
            for child in FlowSolution.children():
                if child.type() != 'DataArray_t': continue
                FieldsNames += [ FlowSolution.name()+'/'+child.name() ]
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
            raise ValueError(RED+MSG+ENDC)
        try:
            return FlowSolution.get( Name='GridLocation' ).value()
        except:
            if not FlowSolution.children(): return AutoGridLocation[ Container ]
            FieldSize = len(FlowSolution.children()[0].value(ravel=True))
            if FieldSize == zone.numberOfPoints(): return 'Vertex'
            elif FieldSize == zone.numberOfCells(): return 'CellCenter'
            else: raise ValueError('could not determine location of '+FlowSolution.path())


    def useEquation(self, equation, Container='FlowSolution', ravel=False):
        RegExpr = "\{[^}]*\}"
        eqnVarsWithDelimiters = re.findall( RegExpr, equation )
        adaptedEqnVars = []
        
        v = self.getAllFields(ravel=ravel)

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
                        field = self.getFields(eqnVarName, Container=eqnContainer,
                                    ravel=ravel, return_type='list')
                    else:
                        raise ValueError('unexpected')

                    v[eqnContainerAndName] = field
                AllContainers.append( eqnContainer )                
                adaptedEqnVars.append( "v['%s']"%eqnContainerAndName )
        FirstMemberLocation = self.inferLocation(AllContainers[0])
        SecondMemberLocations = [self.inferLocation(c) for c in AllContainers[1:]]
        for i, SecondMemberLocation in enumerate(SecondMemberLocations):
            if FirstMemberLocation != SecondMemberLocation:
                MSG = ('not implemented mix of locations {} and {}'
                       ' required by {} and {}').format(FirstMemberLocation,
                         SecondMemberLocation,eqnVarsWithDelimiters[0],
                         eqnVarsWithDelimiters[i+1])
                raise ValueError(RED+MSG+ENDC)


        adaptedEqnVars[0] += '[:]'

        adaptedEquation = equation
        for keyword, adapted in zip(eqnVarsWithDelimiters, adaptedEqnVars):
            adaptedEquation = adaptedEquation.replace(keyword, adapted)

        exec(adaptedEquation,globals(),{'v':v,'np':np})

