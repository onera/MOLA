'''
MOLA - StructuralModels.py

This module defines some handy shortcuts of the Cassiopee's
Converter.Internal module.

Furthermore it provides functions that interact with Code_Aster.

First creation:
31/05/2021 - M. Balmaseda
'''

FAIL  = '\033[91m'
GREEN = '\033[92m'
WARN  = '\033[93m'
MAGE  = '\033[95m'
CYAN  = '\033[96m'
ENDC  = '\033[0m'
try:
    #Code Aster:
    from code_aster.Cata.Commands import *
    from code_aster.Cata.Syntax import _F, CO
    from Utilitai import partition
except:
    print(WARN + 'Code_Aster modules are not loaded!!' + ENDC)


# System modules
import numpy as np
import scipy
from scipy.sparse import csr_matrix
# MOLA modules
import Converter.Internal as I
import Converter.PyTree as C

from .. import InternalShortcuts as J

from . import ShortCuts as SJ
from . import ModalAnalysis   as MA
from . import NonlinearForcesModels as NFM



#########################
# Mesh handeling Scripts:
#########################


def IsMeshInModels(t, MeshName):

    DictStructParam = J.get(t, '.StructuralParameters')
    IsInModel = False
    for MeshFamKey in  DictStructParam['MeshProperties']['MeshFamilies'].keys():
        #print(DictStructParam['MeshProperties']['MeshFamilies'])
        #print(MeshFamKey)
        #print(int(MeshName[1:]))
        #print(DictStructParam['MeshProperties']['MeshFamilies'][MeshFamKey]['Element'])
        if int(MeshName[1:]) in DictStructParam['MeshProperties']['MeshFamilies'][MeshFamKey]['Element']:
            IsInModel = True
    return IsInModel


def IsMeshInCaracteristics(t, MeshName):

    DictStructParam = J.get(t, '.StructuralParameters')
    IsInCara = False
    Caraddl  = None
    for MeshFamKey in  DictStructParam['MeshProperties']['MeshFamilies'].keys():
        if DictStructParam['MeshProperties']['Caracteristics'] is not None:
            for CaraKey in DictStructParam['MeshProperties']['Caracteristics'].keys():
                if MeshFamKey ==  DictStructParam['MeshProperties']['Caracteristics'][CaraKey]['MeshGroup']:
                    if MeshName in DictStructParam['MeshProperties']['MeshFamilies'][MeshFamKey]['Element']:
                        IsInCara = True
                        Caraddl = DictStructParam['MeshProperties']['Caracteristics'][CaraKey]['ddlName'].split(' ')
    return IsInCara, Caraddl



def CGNSElementType(Name):
    #print('NameASter%s'%NameAster)
    if Name == 'POI1':
        NameCGNS = 'NODE'
        NodeElem = 1
        CellDim = 1
    elif Name == 'SEG2':
        NameCGNS = 'BAR_2'
        NodeElem = 2
        CellDim = 2

    elif Name == 'SEG3':
        NameCGNS = 'BAR_3'
        NodeElem = 3
        CellDim = 1
    elif Name == 'QUAD4':
        NameCGNS = 'QUAD_4'
        NodeElem = 4
        CellDim = 2

    elif Name == 'QUAD8':
        NameCGNS = 'QUAD_8'
        NodeElem = 8
        CellDim = 2
    elif Name == 'HEXA8':
        NameCGNS = 'HEXA_8'
        NodeElem = 8
        CellDim = 3
    elif Name == 'HEXA20':
        NameCGNS = 'HEXA_20'
        NodeElem = 20
        CellDim = 3

    return NameCGNS, NodeElem, CellDim

#        def TranslateConnectivity2CGNS(Conectivity, NameAster):

#            if NameAster == 'HEXA8':
#
#                for i in range(int(numpy.size(mm.co)/8)):
            #print i  , '/' , numpy.size(mm.co[i][:]), n_nodes, numpy.size(mm.co)/8
            #if numpy.size(mm.co[i][:]) == 8:
            #    nelem = nelem + 1
            #    conect = numpy.zeros((nelem, 8))




                #var = numpy.zeros((nelem, 4))
                #var[:,:] = conect[:,-4:]
                #conect[:,-4:] = conect[:,12:16]
                #conect[:,12:16] = var
#
            #return Conectivity

def ExtractConnectivity(mm, DictElements, ElemType):

    Conectivity = mm.co
    Conectivity = [val + 1 for val in Conectivity]
    _, NodeElem,_ = CGNSElementType(ElemType)
    ConnectMatrix = np.zeros((len(DictElements['GroupOfElements'][ElemType]['ListOfElem']), NodeElem))
    #for Element, pos in zip(DictElements['GroupOfElements'][ElemType]['ListOfElem'], range(len(DictElements['GroupOfElements'][ElemType]['ListOfElem']))):
    #    print(Element, pos)
    #    elementPosition = Element - 1
    #    #print(DictElements['GroupOfElements'][ElemType]['ListOfElem'])
    #    #print(len(Conectivity[elementPosition][:]))
    #
    #    ConnectMatrix[pos, :] = Conectivity[elementPosition]
    #    ConnectMatrix = ConnectMatrix.astype(int)
    for Element, pos in zip(DictElements['GroupOfElements'][ElemType]['ListOfElem'], range(len(DictElements['GroupOfElements'][ElemType]['ListOfElem']))):
        if ElemType == 'POI1':
            ListPos =  range(len(DictElements['GroupOfElements'][ElemType]['ListOfElem']))
            ListElement = DictElements['GroupOfElements'][ElemType]['ListOfElem']
            SortedElNodes = np.sort(DictElements['GroupOfElements'][ElemType]['Nodes'])
            #print(list(SortedElNodes).index(Conectivity[pos]))
            ConnectMatrix[pos, :] = list(SortedElNodes).index(Conectivity[pos])+1
            ConnectMatrix = ConnectMatrix.astype(int)
            #ConnectMatrix[pos,:] = DictElements['M%s'%Element]['AsterConectivity']
            #ConnectMatrix = ConnectMatrix.astype(int)
            #print(DictElements['GroupOfElements'][ElemType]['Coordinates'])
            #print(ListPos)
            #print(ListElement)
            #XXX
            #print(ElNodes)
            #print(np.sort(ElNodes))
            ConnectMatrix[pos,:] = DictElements['M%s'%Element]['AsterConectivity']

        else:
            #print('Element %s'%Element)
            ConnectMatrix[pos,:] = DictElements['M%s'%Element]['AsterConectivity']
            ConnectMatrix = ConnectMatrix.astype(int)
            #print(ConnectMatrix[pos,:])
    #print(ConnectMatrix[0,:])
    #XXX
    if ElemType == 'HEXA20':
        print(GREEN+'Adapting the order of conectivity for %s elements'%ElemType+ENDC)
        #SwiftList[:,17:19] = [int(x - 1) for x in  [3, 4, 1, 2, 7, 8, 5, 6, 11, 12, 9, 10, 15, 16, 13, 14,19,20,17,18] ]
        ConnectMatrix[:,[15,18]] = ConnectMatrix[:, [18,15]]
        ConnectMatrix[:,[-1,12]] = ConnectMatrix[:, [12,-1]]
        ConnectMatrix[:,[14,17]] = ConnectMatrix[:, [17,14]]
        ConnectMatrix[:,[13,16]] = ConnectMatrix[:, [16,13]]

        #var = np.zeros((len(DictElements['GroupOfElements'][ElemType]['ListOfElem']), 4))
        #var[:,:] = ConnectMatrix[:,-4:]
        #ConnectMatrix[:,-4:] = ConnectMatrix[:,12:16]
        #ConnectMatrix[:,] = var
            #


    #if ElemType == 'HEXA8':
        #print(GREEN+'Adapting the order of conectivity for %s elements'%ElemType+ENDC)
        #ConnectMatrix = ConnectMatrix[:, [1, 2, 3, 0, 5, 6, 7, 4]]
        #ConnectMatrix = ConnectMatrix[:, [0, 3,4, 2, 6, 4, 5, 7]]



    #ConnectMatrix = TranslateConnectivity2CGNS(ConnectMatrix, ElemType)
    return ConnectMatrix

def ExtractCoordinates(mm, DictElements, ElemType):

    Coordinates = mm.cn
    if len(Coordinates[0,:]) == 2:
        Coordinates = np.append(Coordinates, np.zeros((len(Coordinates[:,0]),1)), axis = 1)

    ValidNodes = []
    for Element, pos in zip(DictElements['GroupOfElements'][ElemType]['ListOfElem'], range(len(DictElements['GroupOfElements'][ElemType]['ListOfElem']))):
        elementPosition = Element - 1

        for Node in mm.co[elementPosition]:
            if Node not in ValidNodes:
                ValidNodes.append(Node)
            #print(Node)
    #print(mm.cn)
    #print(ValidNodes)
    #print(Coordinates[ValidNodes])
    return np.array(Coordinates[np.sort(ValidNodes)]), np.sort(ValidNodes)

def DefineDimVectorFromElementName(DictElements,ElemName):
    if DictElements['GroupOfElements'][ElemName]['CellDimension'] == 1:
        DimVector = [[len(DictElements['GroupOfElements'][ElemName]['Coordinates'][:, 0]), 0,0]]
    if DictElements['GroupOfElements'][ElemName]['CellDimension'] == 2:
        DimVector = [[len(DictElements['GroupOfElements'][ElemName]['Coordinates'][:, 0]), len(DictElements['GroupOfElements'][ElemName]['Conectivity'][:,0]),0]]
    elif DictElements['GroupOfElements'][ElemName]['CellDimension'] == 3:
        DimVector = [[len(DictElements['GroupOfElements'][ElemName]['Coordinates'][:, 0]), len(DictElements['GroupOfElements'][ElemName]['Conectivity'][:,0]), 0]]
    return DimVector

def newElements(name='Elements', etype='UserDefined',
                econnectivity=None,
                erange=None, eboundary=0, parent=None):
    """Create a new Elements node."""
    if isinstance(etype, int): etp = etype
    else: etp, nnodes = I.eltName2EltNo(etype)
    #print('parent: %s'%parent)
    #print(etp)
    #print(etype)

    if parent is None:
        node = I.createNode(name, 'Elements_t', value=[etp,eboundary])
    else: node = I.createUniqueChild(parent, name, 'Elements_t',
                                   value=[etp,eboundary])
    I.newDataArray('ElementConnectivity', econnectivity, parent=node)
    #if erange is None: erange = numpy.ones(2, dtype=numpy.int32)
    I.newPointRange('ElementRange', erange, parent=node)
    return node
    
def CreateUnstructuredZone4ElemenType(Base, DictElements, ElemName):
    if True: #ElemName != 'SEG2':
      #zoneUns = I.createNode('InitialMesh_'+ElementName,ntype='Zone_t',value=np.array([[NPts, NElts,0]],dtype=np.int32,order='F'), parent= Base)
        #print(NPts, NElts)
        DimVector = DefineDimVectorFromElementName(DictElements, ElemName)

        zoneUns = I.newZone(name = 'InitialMesh_'+ElemName, zsize = DimVector  , ztype = 'Unstructured', parent= Base)
        zt_n = I.createNode('ZoneType', ntype='ZoneType_t',parent=zoneUns)
        I.setValue(zt_n,'Unstructured')


        g = I.newGridCoordinates(parent = zoneUns)
        I.newDataArray('CoordinateX', value = np.array(DictElements['GroupOfElements'][ElemName]['Coordinates'][:, 0],dtype = np.float32, order = 'F'), parent = g)  #np.array(mm.cn[:,0],dtype=np.float32,order='F'), parent=g) #np.array(CoordinatesE[:,0],dtype=np.float32,order='F'), parent=g) #CoordinatesE[:,0], parent = g)    #
        I.newDataArray('CoordinateY', value = np.array(DictElements['GroupOfElements'][ElemName]['Coordinates'][:, 1],dtype = np.float32, order = 'F'), parent = g)  #np.array(mm.cn[:,1],dtype=np.float32,order='F'), parent=g) #np.array(CoordinatesE[:,1],dtype=np.float32,order='F'), parent=g) #CoordinatesE[:,1], parent = g)    #
        I.newDataArray('CoordinateZ', value = np.array(DictElements['GroupOfElements'][ElemName]['Coordinates'][:, 2],dtype = np.float32, order = 'F'), parent = g)  #np.array(mm.cn[:,2],dtype=np.float32,order='F'), parent=g) #np.array(CoordinatesE[:,2],dtype=np.float32,order='F'), parent=g) #CoordinatesE[:,2], parent = g)    #
        #I.printTree(g)
        print(DimVector)
        if ElemName == 'POI1':
            I.newElements(name=ElemName, etype=DictElements['GroupOfElements'][ElemName]['CGNSType'].split('_')[0], econnectivity= [1], erange = np.array([1,0]), eboundary=0, parent =zoneUns) #np.array(ConectE,dtype=np.int32,order='F'), erange = np.array([1,NElts]), eboundary=0, parent =zoneUns)
        elif ElemName == 'SEG2':

            I.newElements(name=ElemName, etype=DictElements['GroupOfElements'][ElemName]['CGNSType'], econnectivity= DictElements['GroupOfElements'][ElemName]['Conectivity'].flatten(), erange = np.array([1,len(DictElements['GroupOfElements'][ElemName]['Conectivity'])]), parent = zoneUns)

            #I.createNode('ElementType', ntype='ElementType_t', value=DictElements['GroupOfElements'][ElemName]['CGNSType'], parent=aa)

        else: # ElemName == 'HEXA8':
            print('toto%s'%DictElements['GroupOfElements'][ElemName]['CGNSType'])
            print(DimVector)
            print(ElemName)
            print(DictElements['GroupOfElements'][ElemName]['Conectivity'].flatten()[:20])

            aa = newElements(name=ElemName, etype=DictElements['GroupOfElements'][ElemName]['CGNSType'], econnectivity=np.array(DictElements['GroupOfElements'][ElemName]['Conectivity'].flatten(),dtype = np.int32, order = 'F'), erange = [1,DimVector[0][1]], parent =zoneUns)
             #I.createNode('ElementType', ntype='ElementType_t', value=DictElements['GroupOfElements'][ElemName]['CGNSType'], parent=aa)

            #if ElemName == 'HEXA20':
            #
    #I.printTree(aa)
    #I.printTree(zoneUns)
    #I.addChild(zoneUns, aa)

    #C.convertPyTree2File(zoneUns,'/visu/mbalmase/Recherche/9_NewCouplingTheo/Test1.tp', 'fmt_tp')
    #C.convertPyTree2File(zoneUns,'/visu/mbalmase/Recherche/9_NewCouplingTheo/Test1.cgns', 'bin_adf')
    #FFF
    if ElemName == 'POI1':
        zoneUns = C.convertArray2Node(zoneUns)
    #print(ConectE)
    #import Generator as G
    #a = G.cart((0.,0.,0.), (0.1,0.1,0.2), (10,10,1))
    #b = C.convertArray2Node(a)
    #C.convertArrays2File([b], '/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/out.plt')
    #a = G.cart((0.,0.,0.), (0.1,0.1,0.2), (10,10,1))
    #a = C.convertArray2Node(a)
    #C.convertPyTree2File(a, '/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/out.cgns', 'bin_adf')
    #NEltsMin = 1

    #for ElemName in DictElements['GroupOfElements'].keys():
    #
    #    CoordinatesE, NodesElemType  = ExtractCoordinates(mm, DictElements, ElemName)
    #    #print(CoordinatesE == mm.cn)
    #    DictElements['GroupOfElements'][ElemName]['Coordinates'] = CoordinatesE
    #    DictElements['GroupOfElements'][ElemName]['NodesPosition'] = NodesElemType
    #    DictElements['GroupOfElements'][ElemName]['Nodes'] = np.array(NodesElemType) + 1
    #    NEltsMax = NEltsMin + len(DictElements['GroupOfElements'][ElemName]['ListOfElem'])
    #    ConectivityE  = ExtractConnectivity(mm, DictElements, ElemName)
    #    #print(ConectivityE)
    #    DictElements['GroupOfElements'][ElemName]['Connectivity'] = ConectivityE
    #
    ##
    #    #print(ElemName)
    #    #print(DictElements['GroupOfElements'].keys())
    #    #print(DictElements['GroupOfElements'][ElemName]['CGNSType'])
    #    aa = I.newElements(name=ElemName, etype=DictElements['GroupOfElements'][ElemName]['CGNSType'], econnectivity= np.array(ConectivityE.flatten(),dtype=np.int32,order='F'), erange = np.array([NEltsMin,NEltsMax]), eboundary=0, parent =zoneUns)#np.array(ConectE,dtype=np.int32,order='F'), erange = np.array([1,NElts]), eboundary=0, parent =zoneUns)
    #    NEltsMin = NEltsMax + 1
    #I.printTree(Base)

    return Base











def ModifySolidCGNS2Mesh(t):
    '''Read the structured SOLID node and adapt it to an unstructured Mesh node'''

    # Compute the number of nodes in the mesh and add it to t:
    DictStructParam = J.get(t, '.StructuralParameters')

    mm = partition.MAIL_PY()
    mm.FromAster('MAIL')

    DictStructParam['MeshProperties']['NNodes'] = len(list(mm.correspondance_noeuds))




    print(WARN+'Only one type of mesh element! or with same number of ddl'+ENDC)

    #DictStructParam['MeshProperties']['Nddl'] = int(DictStructParam['MeshProperties']['ddlElem']*DictStructParam['MeshProperties']['NNodes'])


    DictStructParam['MeshProperties']['NodesFamilies'] = {}
    for Family in mm.gno.keys():

      DictStructParam['MeshProperties']['NodesFamilies'][Family] = mm.gno[Family]

    DictStructParam['MeshProperties']['MeshFamilies'] = {}
    if mm.gma:
        for NameFamily in mm.gma.keys():

            DictStructParam['MeshProperties']['MeshFamilies'][NameFamily] = {}
            DictStructParam['MeshProperties']['MeshFamilies'][NameFamily]['Element'] = np.array([x + 1 for x in mm.gma[NameFamily]])
    else:
        print(WARN + 'The mesh does not have any associated MeshFamilies, we consider All the elements are the same. Create automatic groups depending on the mesh elements (HEXA8, SEG2...).'+ENDC)
        #print(mm.correspondance_mailles)

        for element in range(len(mm.tm)):
            ElType = mm.nom[mm.tm[element]]
            #print(ElType)
            try:
                DictStructParam['MeshProperties']['MeshFamilies'][ElType]['Element'].append(int(mm.correspondance_mailles[element][1:]))
            except:
                DictStructParam['MeshProperties']['MeshFamilies'][ElType] = {}
                DictStructParam['MeshProperties']['MeshFamilies'][ElType]['Element'] = [int(mm.correspondance_mailles[element][1:])]


#    for ModelName in DictStructParam['MeshProperties']['Models'].keys():
#
#        for MeshFamilyName in DictStructParam['MeshProperties']['MeshFamilies'].keys():
#
#            DictStructParam['MeshProperties']['MeshFamilies'][MeshFamilyName]['ddl'] =[]
#            if MeshFamilyName == DictStructParam['MeshProperties']['Models'][ModelName]['MeshGroup']:
#                for pos in DictStructParam['MeshProperties']['MeshFamilies'][MeshFamilyName]['ElementPosition']:
#
#                    DictStructParam['MeshProperties']['MeshFamilies'][MeshFamilyName]['ddl'].append(pos * DictStructParam['MeshProperties']['Models'][ModelName]['ddlElem'])
#                    DictStructParam['MeshProperties']['MeshFamilies'][MeshFamilyName]['ddl'].append(pos * DictStructParam['MeshProperties']['Models'][ModelName]['ddlElem'])
#                    DictStructParam['MeshProperties']['MeshFamilies'][MeshFamilyName]['ddl'].append(pos * DictStructParam['MeshProperties']['Models'][ModelName]['ddlElem'])

#    print(DictStructParam['MeshProperties']['MeshFamilies'])



#    for Family in mm.gno.keys():
#
#      DictStructParam['MeshProperties']['MeshGroup'][Family] = mm.gno[Family]
    #print(dir(mm))
    #print(mm.indice_noeuds)
    #print(mm.dime_maillage)
    #print(mm.ndim)
    #print(mm.nom)
    #print(mm.tm)

    #print(mm.correspondance_mailles)

    J.set(t, '.StructuralParameters', **DictStructParam)

 ######################  A ADAPTER POUR D'AUTRES FEM
    if I.getNodeByName(t, 'stack') == None:
        #Conectivity = mm.co
        #Conectivity = [val + 1 for val in Conectivity]
        #Coordinates = mm.cn




        DictElements = {}
        DictElements['GroupOfElements'] = {}

        Base = I.newCGNSBase('SOLID', parent=t)




        #print(mm.tm)
        #print(mm.nom)
        Nelem = 0
        for element in range(len(mm.tm)):

            #print(mm.nom[mm.tm[element]])

            if IsMeshInModels(t, mm.correspondance_mailles[element]):
                Nelem += 1
                ElemName = mm.correspondance_mailles[element].split(' ')[0]
                #print('OutLoop', ElemName)
                DictElements[ElemName] = {}
                DictElements[ElemName]['AsterType'] = mm.nom[mm.tm[element]]
                #print('Computing...%s'%DictElements[mm.correspondance_mailles[element]]['AsterType'])
                DictElements[ElemName]['CGNSType'],_, CellDim = CGNSElementType(DictElements[ElemName]['AsterType'])
                DictElements[ElemName]['AsterConectivity'] = mm.co[element] + 1

                #print(DictElements)
                try:
                    DictElements['GroupOfElements'][DictElements[ElemName]['AsterType']]['ListOfElem'].append(int(mm.correspondance_mailles[element][1:]))

                except:
                    DictElements['GroupOfElements'][DictElements[ElemName]['AsterType']] = {}
                    DictElements['GroupOfElements'][DictElements[ElemName]['AsterType']]['ListOfElem'] = [int(mm.correspondance_mailles[element][1:])]
                    print(FAIL+'eltype: %s'%DictElements[ElemName]['CGNSType']+ENDC)
                    DictElements['GroupOfElements'][DictElements[ElemName]['AsterType']]['CGNSType'] = DictElements[ElemName]['CGNSType']
                    DictElements['GroupOfElements'][DictElements[ElemName]['AsterType']]['CellDimension'] = CellDim


                DictElements['NbOfElementType'] = len(DictElements['GroupOfElements'].keys())


        for ElemName in DictElements['GroupOfElements'].keys():
            ##if ElemName != 'POI1':
                CoordinatesE, NodesElemType  = ExtractCoordinates(mm, DictElements, ElemName)
                #print(CoordinatesE == mm.cn)
                DictElements['GroupOfElements'][ElemName]['Coordinates'] = CoordinatesE
                DictElements['GroupOfElements'][ElemName]['NodesPosition'] = NodesElemType
                DictElements['GroupOfElements'][ElemName]['Nodes'] = np.array(NodesElemType) + 1

                ConectivityE  = ExtractConnectivity(mm, DictElements, ElemName)
                #print(ConectivityE)
                print(ConectivityE[0,:])

                DictElements['GroupOfElements'][ElemName]['Conectivity'] = ConectivityE

                Base = CreateUnstructuredZone4ElemenType(Base, DictElements, ElemName)


                #print(DictElements['GroupOfElements'])
        #Base = CreateUnstructuredZone4ElemenType(Base, DictElements, Nelem)
                #print(DictElements['GroupOfElements'][ElemName])

        DictStructParam['MeshProperties']['DictElements'] = {}
        DictStructParam['MeshProperties']['DictElements']['GroupOfElements'] = DictElements['GroupOfElements']
        J.set(t, '.StructuralParameters', **DictStructParam)
        J.set(t, '.ElementsDictionnary', **DictElements)

        existingelement = []

        #print(DictElements['GroupOfElements'].keys())
        #for elem in DictElements['GroupOfElements'].keys():
        #
        #    print(elem, len(DictElements['GroupOfElements'][elem]['Nodes']))
#
#
#
#        #    for node in DictElements['GroupOfElements'][elem]['Nodes']:
#        #        if node not in existingelement:
#        #            existingelement.append(node)
        #            print(node)


        #ddl = {}
        #ddl_vector = []
        #ddltot = 0
        #for maille, pos in zip(mm.correspondance_mailles, range(len(mm.correspondance_mailles))):
#
#        #    nomMaille = mm.nom[mm.tm[pos]]
#
#        #    for familleMesh in DictStructParam['MeshProperties']['MeshFamilies'].keys():
#        #
#        #        for element in DictStructParam['MeshProperties']['MeshFamilies'][familleMesh]['Element']:
#        #            #print(str(maille), str('M%s'%element), str(nomMaille))
#        #            #print('M%s'%element == str(maille))
#        #            if int(element) == int(maille[1:]):
#
#        #                for ModelName in DictStructParam['MeshProperties']['Models'].keys():
#        #
#        #                    if DictStructParam['MeshProperties']['Models'][ModelName]['MeshGroup'] == familleMesh:
#        #                        ddl_vector
#        #                        ddltot += DictStructParam['MeshProperties']['Models'][ModelName]['ddlElem']
#        #                        print(familleMesh, 'M%s'%element, maille, nomMaille, ddltot, familleMesh)
#
#
#        #
#        #                #print(maille,, nomMaille)
        #

        #print(DictStructParam['MeshProperties']['MeshFamilies'].keys())


        # Compute the number of ddl:


        # Be careful with hybrid meshes with more than 1 element!

        # Create the unstructured zone
#        NPts = len(Coordinates)
#
#
#        # 'HEXA': Elements
#        NElts = 0
#        for ConectValue in Conectivity:
#
#            if len(ConectValue) == 8:
#                NElts += 1
#                NValues = 8
#
#            if len(ConectValue) == 2:
#                NElts += 1
#                NValues = 2
#
#
#        conect = np.zeros((NElts, NValues), dtype=int)
#
#        li1 = -1
#        for ConectValue in Conectivity:
#
#            if len(ConectValue) == 8:
#
#                li1 += 1
#                conect[li1,:] = np.squeeze(np.asarray(ConectValue))
#
#
#            if len(ConectValue) == 2:
#
#                li1 += 1
#                conect[li1,:] = np.squeeze(np.asarray(ConectValue))
#
#
#    #####################
#
#        Base = I.newCGNSBase('SOLID', parent=t)
#        zoneUns = I.createNode('InitialMesh',ntype='Zone_t',value=np.array([[NPts, NElts,0]],dtype=np.int32,order='F'), parent= Base)
#        zt_n = I.createNode('ZoneType', ntype='ZoneType_t',parent=zoneUns)
#        I.setValue(zt_n,'Unstructured')
#
#
#        g = I.newGridCoordinates(parent = zoneUns)
#        I.newDataArray('CoordinateX', value=Coordinates[:,0], parent=g)
#        I.newDataArray('CoordinateY', value=Coordinates[:,1], parent=g)
#        I.newDataArray('CoordinateZ', value=Coordinates[:,2], parent=g)
#
#        aa = I.newElements(name='Elements', etype='HEXA_8', econnectivity=conect.flatten(), erange = np.array([1,NElts]), eboundary=0, parent =zoneUns)


        # Add grid coordinates
        #gc_n = I.newGridCoordinates(parent=zoneUns)
        #
        #I.createNode('CoordinateX',ntype='DataArray_t',value=Coordinates[:,0].reshape((NPts),order='F'), parent=gc_n)
        #I.createNode('CoordinateY',ntype='DataArray_t',value=Coordinates[:,1].reshape((NPts),order='F'), parent=gc_n)
        #I.createNode('CoordinateZ',ntype='DataArray_t',value=Coordinates[:,2].reshape((NPts),order='F'), parent=gc_n)

        # Add grid elements
        #GEval = 8 # If the element is HEXA8
        #ge_n = I.createNode('GridElements', ntype='Elements_t', value=np.array([GEval,0],dtype=np.int32, order='F'), parent=zoneUns)
        #I.createNode('ElementRange', ntype='IndexRange_t', value=np.array([1,NElts],dtype=np.int32, order='F'), parent=ge_n)
        #I.createNode('ElementConnectivity', ntype='DataArray_t', value=conect.flatten(), parent=ge_n)
        #I.createNode('ElementType', ntype='ElementType_t', value='HEXA_8', parent=ge_n)

        #print(zoneUns)

    else:
        # Convert the structured mesh into Hexa and erase the structured BCFamilies:
        I._renameNode(t, 'stack', 'InitialMesh')
        t = C.convertArray2Hexa(t)
        I._rmNodesByType(t, 'Family_t')

    # Add Auxiliary CGNSBases:
    #Node = I.createBase('StaticRotatorySolution')
    #I.addChild(t, Node)
    #Node = I.createBase('ModalBases')
    #I.addChild(t, Node)


    #t = I.merge([t, C.newPyTree(['StaticRotatorySolution', [], 'ModalBases', []])])
    #InitZone = I.getNodesFromNameAndType(t, 'InitialMesh', 'Zone_t')[0]
    #J._invokeFields(InitZone,['upx', 'upy', 'upz'])
    #C.convertPyTree2File(t,'/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/Test1.cgns', 'bin_adf')
    #C.convertPyTree2File(t,'/scratchm/mbalmase/Spiro/3_Update4MOLA/CouplingWF_NewMOLA/Test1.cgns', 'bin_adf')
    #C.convertPyTree2File(t,'/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/Test1.tp', 'bin_tp')
    #I.printTree(t)
    #C.convertPyTree2File(t,'/visu/mbalmase/Recherche/9_NewCouplingTheo/Test1.tp', 'fmt_tp')
    #C.convertPyTree2File(t,'/visu/mbalmase/Recherche/9_NewCouplingTheo/Test1.cgns', 'bin_adf')
    #XXX



    return t



def DefineFEMModels(t, Mesh):
    '''Affect several Materials defined in MaterialDict to the corresponding meshes'''

    DictStructParam = J.get(t, '.StructuralParameters')

    ModelDict = DictStructParam['MeshProperties']['Models']

    l_affe = []
    for LocModelName in ModelDict.keys():
        if ModelDict[LocModelName]['MeshGroup'] == 'All':
            ap = _F(TOUT = 'OUI', PHENOMENE = ModelDict[LocModelName]['Phenomene'], MODELISATION = ModelDict[LocModelName]['Modelling'])

        else:
            ap = _F(PHENOMENE = ModelDict[LocModelName]['Phenomene'], MODELISATION = ModelDict[LocModelName]['Modelling'], GROUP_MA = ModelDict[LocModelName]['MeshGroup'])
        l_affe.append(ap)

        print(GREEN+'Affecting %s model to %s mesh groups.'%(LocModelName, ModelDict[LocModelName]['MeshGroup'])+ENDC)

#MODELE=AFFE_MODELE(MAILLAGE=MAIL,
    #                   AFFE=_F(TOUT='OUI',
    #                           PHENOMENE='MECANIQUE',
    #                           MODELISATION='3D',),
    #                   )
#MODELE=AFFE_MODELE(MAILLAGE=MAIL,
    #                   AFFE=_F(TOUT='OUI',
    #                           PHENOMENE='MECANIQUE',
    #                           MODELISATION='3D',),
    #                   )


    MODELE = AFFE_MODELE(MAILLAGE = Mesh,
                         AFFE = l_affe)


    return MODELE


def AffectMaterialFromMaterialDictionary(MaterialDict, Mesh):
    '''Affect several Materials defined in MaterialDict to the corresponding meshes'''

    l_affe = []
    for LocMatName in MaterialDict.keys():
        if MaterialDict[LocMatName]['Mesh'] == 'All':
            ap = _F(TOUT = 'OUI', MATER = MaterialDict[LocMatName]['Properties'])

        else:
            ap = _F(GROUP_MA = MaterialDict[LocMatName]['Mesh'] , MATER = MaterialDict[LocMatName]['Properties'])
        l_affe.append(ap)

        print(GREEN+'Affecting %s material to %s mesh groups.'%(LocMatName, MaterialDict[LocMatName]['Mesh'])+ENDC)

    CHMAT=AFFE_MATERIAU(MAILLAGE=Mesh,
                        AFFE= l_affe,);
    return CHMAT


def DefineMaterials(t, Mesh):

    DictStructParam = J.get(t, '.StructuralParameters')

    MaterialDict = {}
    Ms = []
    M = [None]

    for LocMat, it in zip(DictStructParam['MaterialProperties'].keys(), range(len(DictStructParam['MaterialProperties'].keys()))):
        MaterialDict[LocMat] = {}
        M.append([None])

        M[it] = DEFI_MATERIAU(ELAS=_F(E= DictStructParam['MaterialProperties'][LocMat]['E'],
                                    NU=DictStructParam['MaterialProperties'][LocMat]['PoissonRatio'],
                                    RHO=DictStructParam['MaterialProperties'][LocMat]['Rho'],
                                    AMOR_ALPHA = DictStructParam['MaterialProperties'][LocMat]['XiAlpha'],
                                    AMOR_BETA =  4*np.pi*DictStructParam['MaterialProperties'][LocMat]['Freq4Dumping']*DictStructParam['MaterialProperties'][LocMat]['XiBeta'],
                                    ),);

        Ms.append(M[it])
        MaterialDict[LocMat]['Properties'] = M[it]

        MaterialDict[LocMat]['Mesh'] = DictStructParam['MaterialProperties'][LocMat]['MeshGroup']

        #DETRUIRE(CONCEPT = _F(NOM = MAT))



    CHMAT = AffectMaterialFromMaterialDictionary(MaterialDict, Mesh)



    return CHMAT, Ms

def DefineElementalCaracteristics(t, Mesh, Model):

    DictStructParam = J.get(t, '.StructuralParameters')


    DictCaracteristics = DictStructParam['MeshProperties']['Caracteristics']


    affe_cara = []

    affe_Poutre = []
    affe_Bar = []
    affe_Discret = []
    #print(DictCaracteristics)
    if DictCaracteristics is not None:
        for caraKey in DictCaracteristics.keys():

            if DictCaracteristics[caraKey]['KeyWord'] == 'POUTRE':

                affe_Poutre.append(_F(SECTION = DictCaracteristics[caraKey]['SectionType'],
                                      VARI_SECT = DictCaracteristics[caraKey]['SectionVariation'],
                                      CARA = DictCaracteristics[caraKey]['Properties'].split(' '),
                                      VALE = DictCaracteristics[caraKey]['PropValues'],
                                      GROUP_MA = DictCaracteristics[caraKey]['MeshGroup']
                                      ))

                print(GREEN+'Affecting %s caracteristics to %s mesh groups.'%(caraKey, DictCaracteristics[caraKey]['MeshGroup'])+ENDC)

            if DictCaracteristics[caraKey]['KeyWord'] == 'BARRE':
                affe_Bar = None
                pass

            if DictCaracteristics[caraKey]['KeyWord'] == 'DISCRET':
                affe_Discret.append(_F(SYME = DictCaracteristics[caraKey]['SectionSymetry'],
                                       CARA = DictCaracteristics[caraKey]['Properties'],
                                       VALE = DictCaracteristics[caraKey]['PropValues'],
                                       GROUP_MA = DictCaracteristics[caraKey]['MeshGroup']
                                       ))

                print(GREEN+'Affecting %s caracteristics to %s mesh groups.'%(caraKey, DictCaracteristics[caraKey]['MeshGroup'])+ENDC)




        CARELEM = AFFE_CARA_ELEM(MODELE = Model,
                                 BARRE = affe_Bar,
                                 POUTRE = affe_Poutre,
                                 DISCRET = affe_Discret)
    else:
        print(WARN+'Warning! No caracteristics are affected to the Mesh %s'%Mesh)
        CARELEM = None

    #CARELEM = None

    return CARELEM

def BuildFEmodel(t):
    '''Reads the mesh, creates the FE model in aster and computes the FOM matrices for the studied case.
    The Output is a cgns file of t with the FOM matrices and the model parameters.
       This program is inspired by the Load_FE_model.py function.
    '''

    DictStructParam = J.get(t, '.StructuralParameters')


    # Read the mesh in MED format:

    MAIL = LIRE_MAILLAGE(UNITE = 20,
                         FORMAT = 'MED')



    # Affect the mecanical model to all the mesh:

    MODELE = DefineFEMModels(t, MAIL)

    #MODELE=AFFE_MODELE(MAILLAGE=MAIL,
    #                   AFFE=_F(TOUT='OUI',
    #                           PHENOMENE='MECANIQUE',
    #                           MODELISATION='3D',),
    #                   )

    # Define the materials and affect them to their meshes:

    CHMAT, MAT = DefineMaterials(t, MAIL)

    # Define the elemental characteristics if needed:

    CARELEM = DefineElementalCaracteristics(t, MAIL, MODELE)

    # Modify the cgns in order to erase the SOLID node and to create the Mesh Node

    t = ModifySolidCGNS2Mesh(t)

    # Dictionary of aster objects:

    AsterObjs = dict(MAIL = MAIL,
                     MODELE = MODELE,
                     MAT= MAT,
                     CHMAT = CHMAT,
                     CARELEM = CARELEM)


    return t, AsterObjs

################################################################
################################################################

####################
# FOM Model        :
####################

def AsseMatricesFOM(Type_asse, **kwargs):

    sig_g = CREA_CHAMP(TYPE_CHAM = 'ELGA_SIEF_R',
                   OPERATION = 'EXTR',
                   RESULTAT = kwargs['SOLU'],
                   NOM_CHAM = 'SIEF_ELGA',
                   INST = 1.0)

    #RIGI1 = CO('RIGI1')
    #MASS1 = CO('MASS1')
    #C1 = CO('C1')
    #KASOU = CO('KASOU')
    #KGEO = CO('KGEO')
    #FE1 = CO('FE')
    #NUME = CO('NUME')

    try:
        RIGI1 = CO('RIGI1')
        MASS1 = CO('MASS1')
        C1 = CO('C1')
        KASOU = CO('KASOU')
        KGEO = CO('KGEO')
        FE1 = CO('FE')
        NUME = CO('NUME')

        ASSEMBLAGE(MODELE = kwargs['MODELE'],
               CHAM_MATER = kwargs['CHMAT'],
               CARA_ELEM  = kwargs['CARELEM'],
               CHARGE = kwargs['Cfd'],
               NUME_DDL = NUME,
               MATR_ASSE = (_F(MATRICE=RIGI1,
                               OPTION='RIGI_MECA',),
                            _F(MATRICE=MASS1,
                               OPTION='MASS_MECA',),
                            _F(MATRICE = KGEO,
                               OPTION = 'RIGI_GEOM',
                               SIEF_ELGA = sig_g),
                            _F(MATRICE=KASOU,
                               OPTION='RIGI_ROTA',),
                            _F(MATRICE = C1,
                               OPTION = 'AMOR_MECA'),
                             ),
               VECT_ASSE =(_F(VECTEUR = FE1,
                              OPTION = 'CHAR_MECA',
                              ),
                          ),
               INFO = 2
               )

        # Loading coefficients:

        if Type_asse == 'All':
            C_g = 1.0
            C_c = 1.0
        elif Type_asse == 'Kec':
            C_g = 0.0
            C_c = 1.0
        elif Type_asse == 'Keg':
            C_g = 1.0
            C_c = 0.0

        # Assembly of matrix:

        Komeg2 = COMB_MATR_ASSE( COMB_R = (_F( MATR_ASSE = RIGI1,
                                               COEF_R = 1.),
                                           _F(MATR_ASSE = KGEO,
                                              COEF_R = C_g),
                                           _F(MATR_ASSE = KASOU,
                                              COEF_R = C_c),),)
    except:
        print(FAIL+'WARNING! The matrix Kc was not computed! Only valid if Omega = 0 rpm'+ENDC)

        DETRUIRE (CONCEPT = _F (NOM = (RIGI1,MASS1, KGEO, KASOU, C1, FE1, NUME),
                            ),
              INFO = 1,
              )

        RIGI1 = CO('RIGI1')
        MASS1 = CO('MASS1')
        C1 = CO('C1')
        KASOU = None
        KGEO = CO('KGEO')
        FE1 = CO('FE')
        NUME = CO('NUME')

        ASSEMBLAGE(MODELE = kwargs['MODELE'],
               CHAM_MATER = kwargs['CHMAT'],
               CARA_ELEM  = kwargs['CARELEM'],
               CHARGE = kwargs['Cfd'],
               NUME_DDL = NUME,
               MATR_ASSE = (_F(MATRICE=RIGI1,
                               OPTION='RIGI_MECA',),
                            _F(MATRICE=MASS1,
                               OPTION='MASS_MECA',),
                            _F(MATRICE = KGEO,
                               OPTION = 'RIGI_GEOM',
                               SIEF_ELGA = sig_g),
                            _F(MATRICE = C1,
                               OPTION = 'AMOR_MECA'),
                             ),
               VECT_ASSE =(_F(VECTEUR = FE1,
                              OPTION = 'CHAR_MECA',
                              ),
                          ),
               INFO = 2
               )

        # Loading coefficients:

        if Type_asse == 'All':
            C_g = 1.0
            C_c = 1.0
        elif Type_asse == 'Kec':
            C_g = 0.0
            C_c = 1.0
        elif Type_asse == 'Keg':
            C_g = 1.0
            C_c = 0.0

        # Assembly of matrix:

        Komeg2 = COMB_MATR_ASSE( COMB_R = (_F( MATR_ASSE = RIGI1,
                                               COEF_R = 1.),
                                           _F(MATR_ASSE = KGEO,
                                              COEF_R = C_g),
                                           ),)


    DictMatrices = {}
    DictVectors = {}
    #AsterObjs =  dict(**kwargs , **dict(Komeg2 = Komeg2, MASS1 = MASS1, NUME = NUME))
    AsterObjs =  SJ.merge_dicts(kwargs , dict(Komeg2 = Komeg2, MASS1 = MASS1, NUME = NUME))

    DictMatrices['Ke'], P_Lagr = SJ.ExtrMatrixFromAster2Python(RIGI1, ComputeLagrange = True)

    DictMatrices['Kg'],_ = SJ.ExtrMatrixFromAster2Python(KGEO, ii = P_Lagr)
    if KASOU is not None:
        DictMatrices['Kc'],_ = SJ.ExtrMatrixFromAster2Python(KASOU, ii = P_Lagr)

    DictMatrices['Komeg'],_ = SJ.ExtrMatrixFromAster2Python(Komeg2, ii = P_Lagr)
    DictMatrices['C'],_ = SJ.ExtrMatrixFromAster2Python(C1, ii = P_Lagr)
    DictMatrices['M'],_ = SJ.ExtrMatrixFromAster2Python(MASS1, ii = P_Lagr)

    DictVectors['Fei'] = SJ.ExtrVectorFromAster2Python(FE1, P_Lagr)

    if KASOU  is not None:
        DETRUIRE (CONCEPT = _F (NOM = (sig_g, RIGI1, KGEO, KASOU, C1, FE1),
                            ),
              INFO = 1,
              )
    else:
        DETRUIRE (CONCEPT = _F (NOM = (sig_g, RIGI1, KGEO, C1, FE1),
                            ),
              INFO = 1,
              )

    #return Ke, Kg, Kc, Komeg, C, M, Fe, P_Lagr, AsterObjs
    return DictMatrices, DictVectors, P_Lagr, AsterObjs

def DefineBehaviourLaws(t):
    '''Affect several BehaviourLaws defined in ModelsDict to the corresponding meshes'''

    DictStructParam = J.get(t, '.StructuralParameters')

    ModelDict = DictStructParam['MeshProperties']['Models']

    l_Behaviour = []
    for LocModelName in ModelDict.keys():

        if ModelDict[LocModelName]['Strains'] == 'Green':
            Ty_Def = 'PETIT'
        elif ModelDict[LocModelName]['Strains'] == 'Green-Lagrange':
            Ty_Def = 'GROT_GDEP'

        if ModelDict[LocModelName]['MeshGroup'] == 'All':
            ap = _F(TOUT = 'OUI', RELATION = ModelDict[LocModelName]['BehaviourLaw'], DEFORMATION = Ty_Def)
        else:
            ap = _F(RELATION = ModelDict[LocModelName]['BehaviourLaw'], DEFORMATION = Ty_Def , GROUP_MA = ModelDict[LocModelName]['MeshGroup'])
        l_Behaviour.append(ap)

        print(GREEN+'Affecting %s behaviour to %s mesh groups.'%(LocModelName, ModelDict[LocModelName]['MeshGroup'])+ENDC)

    return l_Behaviour


def ComputeMatricesFOM(t, RPM, **kwargs):
    '''
    Computes the FOM matrices for a given velocity RPM (Ke, Kg, Kc, C, M).
    Returns K(Omega) and M
    '''

    DictStructParam = J.get(t, '.StructuralParameters')
    DictSimulaParam = J.get(t, '.SimulationParameters')

    #if DictStructParam['MaterialProperties']['BehaviourLaw'] == 'Green':
    #    DictStructParam['MaterialProperties']['TyDef'] = 'PETIT'
    #    sufix = '_L'
    #elif DictStructParam['MaterialProperties']['BehaviourLaw'] == 'Green-Lagrange':
    #    DictStructParam['MaterialProperties']['TyDef'] = 'GROT_GDEP'
    #    sufix = '_NL'

    J.set(t, '.StructuralParameters', **DictStructParam)

    # Define the external loading and solve the problem:


    Cfd = AFFE_CHAR_MECA(MODELE = kwargs['MODELE'],
                         ROTATION = _F(VITESSE = np.pi * RPM/30.,
                                       AXE = DictSimulaParam['RotatingProperties']['AxeRotation'],
                                       CENTRE = DictSimulaParam['RotatingProperties']['RotationCenter'],),
                         DDL_IMPO=(SJ.AffectImpoDDLByGroupType(t)
                                       ),
                                )


    RAMPE = DEFI_FONCTION(NOM_PARA = 'INST',
                                  VALE = (0.0,0.0,1.0,1.0),
                                  PROL_DROITE = 'CONSTANT',
                                  PROL_GAUCHE = 'CONSTANT',
                                  INTERPOL = 'LIN'
                                  );


    L_INST = DEFI_LIST_REEL(DEBUT = 0.0,
                            INTERVALLE = (_F(JUSQU_A = 1.0,
                                             NOMBRE = DictSimulaParam['IntegrationProperties']['Steps4CentrifugalForce'],),
                                          ),
                                  );

    SOLU = STAT_NON_LINE(MODELE = kwargs['MODELE'],
                         CHAM_MATER = kwargs['CHMAT'],
                         CARA_ELEM = kwargs['CARELEM'],
                         EXCIT =( _F(CHARGE = Cfd,
                                     FONC_MULT=RAMPE,),),
                         COMPORTEMENT = DefineBehaviourLaws(t),
                         CONVERGENCE=_F(RESI_GLOB_MAXI=2e-6,
                                        RESI_GLOB_RELA=1e-4,
                                        ITER_GLOB_MAXI=1000,
                                        ARRET = 'OUI',),
                         INCREMENT = _F( LIST_INST = L_INST,
                                        ),
                         INFO = 1,
                      ),

    # Add the model with matrices:
    #AsterObjs = dict(**kwargs, **dict(Cfd= Cfd, SOLU = SOLU[0], RAMPE = RAMPE, L_INST = L_INST))
    AsterObjs = SJ.merge_dicts(kwargs, dict(Cfd= Cfd, SOLU = SOLU[0], RAMPE = RAMPE, L_INST = L_INST))


    #Ke, Kg, Kc, Komeg, C, M, Fei, PointsLagrange, AsterObjs = AsseMatricesFOM('All', **AsterObjs)
    DictMatrices, DictVectors, PointsLagrange, AsterObjs = AsseMatricesFOM('All', **AsterObjs)

    DictStructParam['MeshProperties']['Nddl'] = np.shape(DictMatrices['Ke'][:,0])[0]
    DictStructParam['MeshProperties']['NodesFamilies']['LagrangeNodes'] = PointsLagrange
    J.set(t, '.StructuralParameters', **DictStructParam)

    for NameMV in DictMatrices.keys(): #Ke, Kg, Kc, Komeg, C, M, Fei
        #print(DictMatrices[NameMV])
        t = SJ.AddFOMVars2Tree(t, RPM, Vars = [DictMatrices[NameMV]], # Kg, Kc, Komeg, C, M],
                                   VarsName = [NameMV], #, 'Kg', 'Kc', 'Komeg', 'C', 'M'],
                                   Type = '.AssembledMatrices',
                                   )

    t = SJ.AddFOMVars2Tree(t, RPM, Vars = [DictVectors['Fei']],
                                   VarsName = ['Fei'],
                                   Type = '.AssembledVectors',
                                   )

    AsterObjs = SJ.merge_dicts(AsterObjs, dict(Fei= DictVectors['Fei']))

    return t, AsterObjs

def ExtrFromAsterSOLUwithOmegaFe(t, RPM, Instants = [1.0],ChampName = 'DEPL', **kwargs):
    DictStructParam = J.get(t, '.StructuralParameters')

    LIST = DEFI_LIST_REEL(VALE = Instants)
    try:
        tstaT = CREA_TABLE(RESU = _F(RESULTAT= kwargs['SOLU'],
                                             NOM_CHAM= ChampName,
                                             NOM_CMP= ('DX','DY','DZ', 'DRX', 'DRY', 'DRZ'),
                                             LIST_INST = LIST,
                                             TOUT = 'OUI',),
                                   TYPE_TABLE='TABLE',
                                   TITRE='Table_Depl_R',
                                   )
    except:
        tstaT = CREA_TABLE(RESU = _F(RESULTAT= kwargs['SOLU'],
                                             NOM_CHAM= ChampName,
                                             NOM_CMP= ('DX','DY','DZ'),
                                             LIST_INST = LIST,
                                             TOUT = 'OUI',),
                                   TYPE_TABLE='TABLE',
                                   TITRE='Table_Depl_R',
                                   )
        print(WARN+'No rotation dof  (DRX, DRY, DRZ) in the model. Computing only with displacements (DX, DY, DZ).'+ENDC)

    VectUpOmegaFe = VectFromAsterTable2Full(t, tstaT)

    DETRUIRE (CONCEPT = _F (NOM = (tstaT, LIST),
                            ),
              INFO = 1,
              )

    return VectUpOmegaFe

def ExtrVelocityFromAsterSOLUwithOmegaFe(t, RPM, Instants = [1.0], **kwargs):
    DictStructParam = J.get(t, '.StructuralParameters')

    LIST = DEFI_LIST_REEL(VALE = Instants)
    try:
        tstaT = CREA_TABLE(RESU = _F(RESULTAT= kwargs['SOLU'],
                                             NOM_CHAM='VITE',
                                             NOM_CMP= ('DX','DY','DZ', 'DRX', 'DRY', 'DRZ'),
                                             LIST_INST = LIST,
                                             TOUT = 'OUI',),
                                   TYPE_TABLE='TABLE',
                                   TITRE='Table_Depl_R',
                                   )
    except:
        tstaT = CREA_TABLE(RESU = _F(RESULTAT= kwargs['SOLU'],
                                             NOM_CHAM='VITE',
                                             NOM_CMP= ('DX','DY','DZ'),
                                             LIST_INST = LIST,
                                             TOUT = 'OUI',),
                                   TYPE_TABLE='TABLE',
                                   TITRE='Table_Depl_R',
                                   )
        print(WARN+'No rotation dof  (DRX, DRY, DRZ) in the model. Computing only with displacements (DX, DY, DZ).'+ENDC)

    VectUpOmegaFe = VectFromAsterTable2Full(t, tstaT)

    DETRUIRE (CONCEPT = _F (NOM = (tstaT, LIST),
                            ),
              INFO = 1,
              )

    return VectUpOmegaFe

#def ComputeDDLVector(SOLU):



#    return ddl, DictDDL

def ComputeTotalDDLFromAsterTable(t, Table):
    DictStructParam = J.get(t, '.StructuralParameters')

    depl_sta = Table.EXTR_TABLE()['NOEUD', 'DX', 'DY','DZ','DRX', 'DRY','DRZ'].values()
    if depl_sta == {}:
        depl_sta = Table.EXTR_TABLE()['NOEUD', 'DX', 'DY','DZ'].values()
        print(WARN+'No rotation dof  (DRX, DRY, DRZ) in the model. Computing only with displacements (DX, DY, DZ).'+ENDC)

    ddl_tot = 0

    for NodeKey, pos in zip(depl_sta['NOEUD'], range(len(depl_sta['NOEUD']))):

        for Var in depl_sta.keys():
            if Var not in ['NOEUD']:

                if depl_sta[Var][pos] != None:
                    ddl_tot += 1

    return ddl_tot, t

def ComputeTransformationLists(t, Table):
                # Tool lists to compute from XYZ to ddl vectors.

    DictStructParam = J.get(t, '.StructuralParameters')

    depl_sta = Table.EXTR_TABLE()['NOEUD', 'DX', 'DY','DZ','DRX', 'DRY','DRZ'].values()
    if depl_sta == {}:
        depl_sta = Table.EXTR_TABLE()['NOEUD', 'DX', 'DY','DZ'].values()
        print(WARN+'No rotation dof  (DRX, DRY, DRZ) in the model. Computing only with displacements (DX, DY, DZ).'+ENDC)


    n_ddl = 0
    l_ddl = []
    l_var = []
    l_SplitArray2Vars = []
    for NodeKey, pos in zip(depl_sta['NOEUD'], range(len(depl_sta['NOEUD']))):
        l_var = []
        l_SplitArray2Vars.append(n_ddl)
        for Var in depl_sta.keys():
            if Var not in ['NOEUD']:
                if depl_sta[Var][pos] != None:
                    n_ddl += 1
                    l_var.append(Var)

        l_ddl.append(np.array(l_var))
    DictStructParam['MeshProperties']['Transformations'] = {}
    DictStructParam['MeshProperties']['Transformations']['FOM2XYZ'] = l_SplitArray2Vars

    def CalcVectDDL(ArrayStr):

        VectNames = []
        DDLNum = []

        for comp in ArrayStr:
            if comp not in VectNames:
                VectNames.append(comp)

            DDLNum.append(VectNames.index(comp))

        VectNames = '.'.join(VectNames)

        return DDLNum, VectNames

    DictStructParam['MeshProperties']['Transformations']['VectDDLNum'],DictStructParam['MeshProperties']['Transformations']['VectDDLNames'] = CalcVectDDL(np.array(np.concatenate(l_ddl)))
    DictStructParam['MeshProperties']['Transformations']['VectDDL'] = np.array(np.concatenate(l_ddl))
    DictStructParam['MeshProperties']['Transformations']['DDLNodes'] = np.split(np.array(np.concatenate(l_ddl)), l_SplitArray2Vars[1:])

    ddl2Node = []
    for Node, posNode in zip(DictStructParam['MeshProperties']['Transformations']['DDLNodes'], range(DictStructParam['MeshProperties']['NNodes'][0])):
        for comp in Node:
            ddl2Node.append(posNode + 1)


    DictStructParam['MeshProperties']['Transformations']['DDL2Node'] = ddl2Node

    J.set(t, '.StructuralParameters', **DictStructParam)
    return t

def ComputeDDLandTransfMatrixFromAsterTable(t, Table):


    t = ComputeTransformationLists(t, Table)
    ddl, t = ComputeTotalDDLFromAsterTable(t, Table)

    return t

def VectFromAsterTable2Full(t, Table):

    depl_sta = Table.EXTR_TABLE()['NOEUD', 'DX', 'DY','DZ','DRX', 'DRY','DRZ'].values()
    if depl_sta == {}:
        depl_sta = Table.EXTR_TABLE()['NOEUD', 'DX', 'DY','DZ'].values()
        print(WARN+'No rotation dof  (DRX, DRY, DRZ) in the model. Computing only with displacements (DX, DY, DZ).'+ENDC)

    DictStructParam = J.get(t,'.StructuralParameters')
    NNodes = DictStructParam['MeshProperties']['NNodes'][0]

    n_ddl = 0
    l_ddl = []
    l_SplitArray2Vars = []
    for NodeKey, pos in zip(depl_sta['NOEUD'], range(NNodes)):
        l_var = []
        l_SplitArray2Vars.append(n_ddl)
        for Var in depl_sta.keys():
            if Var not in ['NOEUD']:

                if depl_sta[Var][pos] != None:
                    n_ddl += 1
                    l_var.append(depl_sta[Var][pos::NNodes])

        l_ddl.append(l_var)

    ConcatenatedArray = np.array(np.concatenate(l_ddl, axis = 0))
    if np.shape(ConcatenatedArray)[1] == 1:
        ConcatenatedArray = np.array(np.concatenate(l_ddl))
        ConcatenatedArray = ConcatenatedArray[:,0]

    return ConcatenatedArray

def ListXYZFromVectFull(t, VectFull):

    DictStructParam = J.get(t, '.StructuralParameters')
    #print(DictStructParam['MeshProperties']['Transformations']['FOM2XYZ'][1:])


    VectXYZ = np.split(np.array(np.concatenate(VectFull)), DictStructParam['MeshProperties']['Transformations']['FOM2XYZ'][1:])

    VectDDLNodes = SJ.BuildDDLVector(DictStructParam)

    #print(VectXYZ)
    DictXYZ = {}
    for Node in range(len(VectDDLNodes)):
        for Component, pos in zip(VectDDLNodes[Node],range(len(VectDDLNodes[Node]))):

            try:
                DictXYZ[Component].append(VectXYZ[Node][pos])
            except:
                DictXYZ[Component] = [VectXYZ[Node][pos]]

        for key in DictXYZ:
            #print(key, len(DictXYZ[key]))
            if len(DictXYZ[key]) < Node+1:
                DictXYZ[key].append(None)
    ListeXYZ = []
    for key in DictXYZ.keys():
        ListeXYZ.append(np.array(DictXYZ[key]))

    return ListeXYZ


def BuildRPMParametrisation(t, ):
    # Anadir y calculo de los parametricos: --> Calculo de K1...K3
                                    #      --> Calculo de PHIsvd
                                    #      --> Eliminar del arbol las dependencias RPM
                                    #      --> Guardar en el arbol el Dict 'Parametric'

    DictSimulaParam = J.get(t, '.SimulationParameters')
    DictStructParam = J.get(t, '.StructuralParameters')

    #Check if it's parametric
    RPMs=DictSimulaParam['RotatingProperties']['RPMs']
    #SJ.SaveModel(t, kwargs['FOMName'], Modes = True, StaticRotatorySolution = True)

    #Once every FOM matrix is created, we chck if it's a parametric model and then implement it
    if len(RPMs)==3 and (RPMs[1]-RPMs[0])==(RPMs[2]-RPMs[1]):
        print(GREEN + 'Parametric model'+ ENDC)


        NewFOMmatrices = {}
        NewFOMmatrices['Parametric']={}

        MatrFOM = J.get(t, '.AssembledMatrices')
        MatrFOM['Temporary']={}
        #Composing PHIAug
        PHIAug = []
        PHIAug = np.hstack((MatrFOM[str(np.round(RPMs[0],2))+'RPM']['PHI'],MatrFOM[str(np.round(RPMs[1],2))+'RPM']['PHI']))
        PHIAug = np.hstack((PHIAug,MatrFOM[str(np.round(RPMs[2],2))+'RPM']['PHI']))




        #SVD
        U,s,Vt = scipy.linalg.svd(PHIAug, full_matrices=True) ##CHECK SIZE!!!!
        print('SVD done')
        #print(type(PHIGrand))

        ## Single values choice (it depends on the type of the matrix: array in this case)
        # Svalue 0.01%max(sValue)         What about if all s are negative: CHANGE THIS CONDITION
        index=[ i for i in range(0,len(s)) if s[i]>=0.01/100.*max(s)]
        print(CYAN+'NewNumberOfModes: %s'%len(index)+ENDC)
        #The maximum number of single values is 3*r (where r is the number of modes that are chosen)
        U=U[:,index]
        NewFOMmatrices['Parametric']['PHI'] = U
        # Save the basis in the tree:

        #t = SJ.AddFOMVars2Tree(t, 0, Vars = [U], # Kg, Kc, Komeg, C, M],
        #                           VarsName = ['PHI'], #, 'Kg', 'Kc', 'Komeg', 'C', 'M'],
        #                           Type = '.AssembledMatrices',
        #                           )



        for indexVect in range(len(index)):
            ModeVect = U[:,index[indexVect]]

            ModZone = SJ.CreateNewSolutionFromNdArray(t, FieldDataArray = [ModeVect], ZoneName='Mode%s_Parametric'%indexVect,
                                               FieldName = 'ParametricMode%s'%indexVect
                                    )

            try:
              I._addChild(I.getNodeFromName(t, 'ModalBases'), ModZone)
            except:
              t = I.merge([t, C.newPyTree(['ModalBases', []])])
              I._addChild(I.getNodeFromName(t, 'ModalBases'), ModZone)
            #I._addChild(I.getNodeFromName(t, 'ModalBases'), I.createNode('Freq_%sRPM'%np.round(RPM,2), 'DataArray_t', value = np.sqrt(s[index[indexVect]])/(2.*np.pi))


        print(WARN + 'Warning! The requested number of modes has changed from %s to %s'%(DictStructParam['ROMProperties']['NModes'][0], len(index))+ENDC)

        DictStructParam['ROMProperties']['NModes'] = len(index)

        #FO model constants
        Kp0, _ = SJ.LoadSMatrixFromCGNS(t, RPMs[0], 'Komeg')
        Kp0Delta, _ =  SJ.LoadSMatrixFromCGNS(t, RPMs[1], 'Komeg')
        Kp02Delta, _ =  SJ.LoadSMatrixFromCGNS(t, RPMs[2], 'Komeg')
        Deltap=RPMs[1]-RPMs[0]

        NewFOMmatrices['Parametric']['Deltap'] = Deltap
        NewFOMmatrices['Parametric']['p0'] = RPMs[0]
        NewFOMmatrices['Parametric']['Range'] = [RPMs[0],RPMs[-1]]

        NewFOMmatrices['Parametric']['K0Parametric'] = Kp0
        NewFOMmatrices['Parametric']['K1Parametric'] = ((-1)*Kp02Delta+4*Kp0Delta-3*Kp0)/(2*Deltap)
        NewFOMmatrices['Parametric']['K2Parametric'] = (Kp02Delta-2*Kp0Delta+Kp0)/((Deltap)*(Deltap))

        NewFOMmatrices['Parametric']['C'], _=SJ.LoadSMatrixFromCGNS(t, RPMs[0], 'C')
        NewFOMmatrices['Parametric']['M'], _=SJ.LoadSMatrixFromCGNS(t, RPMs[0], 'M')


        #CGNS tree update and remove RPM dependencies
        #for i in range(0,len(RPMs)):
        #    MatrRed[str(RPMs[i])+'RPM']={}
        #    #I.rmNodeByPath(t,'.AssembledMatrices/'+str(RPMs[i])+'RPM')



        #WE DON'T PROJECT UNTIL WE ARE IN THE ROM, HERE ONL FOM ARE SAVED
        #MatrRed['Temporary']['K0FOMFD'] = K0FOMFD
        #MatrRed['Temporary']['K1FOMFD'] = K1FOMFD
        #MatrRed['Temporary']['K2FOMFD'] = K2FOMFD
        #MatrRed['Temporary']['M'] = M
        #MatrRed['Temporary']['C'] = C
        #MatrRed['Temporary']['PHI'] = U
        #MatrRed['Temporary']['p0'] = RPMs[0]
        #MatrRed['Temporary']['Range'] = [RPMs[0],RPMs[-1]]

        #SI ON N'EXECUTE PAS CETTE LIGNE, CES NOEUDS NE SONT PAS AJOUTES SUR L'ARBRE t
        J.set(t, '.AssembledMatrices', **NewFOMmatrices)
        J.set(t, '.StructuralParameters', **DictStructParam)

        for NameMV in NewFOMmatrices['Parametric'].keys(): #K0FD,K1FD,K2FD,M et C
            print(str(NameMV)+' being saved:')
            t = SJ.AddFOMVars2Tree(t, 'Parametric', Vars = [NewFOMmatrices['Parametric'][NameMV]], # Kg, Kc, Komeg, C, M],
                                   VarsName = [NameMV], #, 'Kg', 'Kc', 'Komeg', 'C', 'M'],
                                   Type = '.AssembledMatrices',
                                   )
            print(str(NameMV)+' saving done!')

        parametric = True

    else:
        print('Not a parametric model')
        parametric = False

    return t, parametric


def GetAsterTableOfStaticNodalForces(InstantsExtr = [1.0], **kwargs):

    LIST = DEFI_LIST_REEL(VALE = InstantsExtr)
    F_noda = CALC_CHAMP(MODELE = kwargs['MODELE'],
                        CHAM_MATER =kwargs['CHMAT'],
                        CARA_ELEM = kwargs['CARELEM'],
                        LIST_INST = LIST,
                        RESULTAT = kwargs['SOLU'],
                        FORCE = 'FORC_NODA'
                        );

    # Short Way: Unvalid for some reason...
    #GusZone,_ = SJ.CreateNewSolutionFromNdArray(t,
    #                                 FieldDataArray= [np.array(F_nodaX),
    #                                                  np.array(F_nodaY),
    #                                                  np.array(F_nodaZ)],
    #                                 ZoneName = 'G_sta'+str(np.round(RPM)),
    #                                 FieldNames = ['Gusx', 'Gusy', 'Gusz'],
    #                                 Depl = False,
    #                                 DefByField = UsField)
    # Long way:
    # CREA_TABLE --> EXTR_TABLE()['NOEUD', 'DX', 'DY', 'DZ'] --> SJ.CreateNewSolutionFromAsterTable

    try:
        tstaT2 = CREA_TABLE(RESU = _F(RESULTAT = F_noda,
                                    NOM_CHAM = 'FORC_NODA',
                                    NOM_CMP = ('DX','DY','DZ', 'DRX', 'DRY', 'DRZ'),
                                    TOUT='OUI',
                                    ),
                        TYPE_TABLE = 'TABLE',
                        TITRE = 'Table_Force_N',
                        )
    except:
        tstaT2 = CREA_TABLE(RESU = _F(RESULTAT = F_noda,
                                    NOM_CHAM = 'FORC_NODA',
                                    NOM_CMP = ('DX','DY','DZ'),
                                    TOUT='OUI',
                                    ),
                        TYPE_TABLE = 'TABLE',
                        TITRE = 'Table_Force_N',
                        )
        print(WARN + 'Only Fx, Fy and Fz are present'+ENDC)
    DETRUIRE(CONCEPT = _F(NOM = (F_noda, LIST)))

    return tstaT2

def ExtrUGStatRot(t, RPM, **kwargs):

    DictStructParam = J.get(t, '.StructuralParameters')

    try:
        tstaT = CREA_TABLE(RESU = _F(RESULTAT= kwargs['SOLU'],
                                             NOM_CHAM='DEPL',
                                             NOM_CMP= ('DX','DY','DZ', 'DRX', 'DRY', 'DRZ'),
                                             INST = 1.0,
                                             TOUT = 'OUI',),
                                   TYPE_TABLE='TABLE',
                                   TITRE='Table_Depl_R',
                                   )
    except:
        tstaT = CREA_TABLE(RESU = _F(RESULTAT= kwargs['SOLU'],
                                             NOM_CHAM='DEPL',
                                             NOM_CMP= ('DX','DY','DZ'),
                                             INST = 1.0,
                                             TOUT = 'OUI',),
                                   TYPE_TABLE='TABLE',
                                   TITRE='Table_Depl_R',
                                   )
        print(WARN+'No rotation dof  (DRX, DRY, DRZ) in the model. Computing only with displacements (DX, DY, DZ).'+ENDC)


    t = ComputeDDLandTransfMatrixFromAsterTable(t, tstaT)

    UsZones = SJ.CreateNewSolutionFromAsterTable(t,FieldDataTable= tstaT,
                                                   ZoneName = '%sRPM'%np.round(RPM,2),
                                                   FieldName = 'Us',
                                                   )

    #Computation of Us
    VectUsOmega = VectFromAsterTable2Full(t, tstaT)

    t = SJ.AddFOMVars2Tree(t, RPM, Vars = [VectUsOmega],
                                   VarsName = ['Us'],
                                   Type = '.AssembledVectors',
                                   )
    try:
        I._addChild(I.getNodeFromName(t, 'StaticRotatorySolution'), UsZones)
    except:
        t = I.merge([t, C.newPyTree(['StaticRotatorySolution', UsZones])])

    #C.convertPyTree2File(t,'/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/toto.cgns', 'bin_adf')
    #C.convertPyTree2File(t,'/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/toto.tp', 'bin_tp')

    #upx[:], upy[:], upz[:], = UsField[0], UsField[1], UsField[2]

    #upx, upy, upz, upthetax, upthetay, upthetaz  = J.getVars(Zone, ['upx', 'upy', 'upz', 'upthetax', 'upthetay', 'upthetaz'])
    #upx[:], upy[:], upz[:],upthetax[:], upthetay[:], upthetaz[:] = UsField[0][NodePosition], UsField[1][NodePosition], UsField[2][NodePosition], UsField[3][NodePosition], UsField[4][NodePosition], UsField[5][NodePosition]


    #VectFromAsterTable2Full(t, Table)

#VectUsOmega

    #if DictStructParam['MeshProperties']['ddlElem'][0] == 3:
#
#
#        tstaT = CREA_TABLE(RESU = _F(RESULTAT= kwargs['SOLU'],
#                                             NOM_CHAM='DEPL',
#                                             NOM_CMP= ('DX','DY','DZ'),
#                                             INST = 1.0,
#                                             TOUT = 'OUI',),
#                                   TYPE_TABLE='TABLE',
#                                   TITRE='Table_Depl_R',
#                                   )
#
#        # Tableau complet des deplacements, coordonnees modales,... :
#
#        depl_sta = tstaT.EXTR_TABLE()['NOEUD', 'DX', 'DY', 'DZ']
#
#        UsZone, UsField = SJ.CreateNewSolutionFromAsterTable(t, FieldDataTable= depl_sta,
#                                                             ZoneName = 'U_sta'+str(np.round(RPM)),
#                                                             FieldNames = ['Usx', 'Usy', 'Usz'],
#                                                             Depl = True)
#
#        J._invokeFields(UsZone, ['upx', 'upy', 'upz', 'ux', 'uy', 'uz'])
#        upx, upy, upz = J.getVars(UsZone, ['upx', 'upy', 'upz'])
#        upx[:], upy[:], upz[:], = UsField[0], UsField[1], UsField[2]
#
##            # Compute the Us vector and add it to the .AssembledVectors node:
##        DictStructParam = J.get(t, '.StructuralParameters')
##
##        VectUsOmega = np.zeros((DictStructParam['MeshProperties']['Nddl'][0]))
##        VectUsOmega[::DictStructParam['MeshProperties']['ddlElem'][0]] = upx
##        VectUsOmega[1::DictStructParam['MeshProperties']['ddlElem'][0]] = upy
##        VectUsOmega[2::DictStructParam['MeshProperties']['ddlElem'][0]] = upz
#
#    elif DictStructParam['MeshProperties']['ddlElem'][0] == 6:
#
#        VarExtr = ('DX','DY','DZ', 'DRX', 'DRY', 'DRZ')
#
#        tstaT = CREA_TABLE(RESU = _F(RESULTAT= kwargs['SOLU'],
#                                             NOM_CHAM='DEPL',
#                                             NOM_CMP= VarExtr,
#                                             INST = 1.0,
#                                             TOUT = 'OUI',),
#                                   TYPE_TABLE='TABLE',
#                                   TITRE='Table_Depl_R',
#                                   )
#
        # Tableau complet des deplacements, coordonnees modales,... :















#        UsZone, UsField = SJ.CreateNewSolutionFromAsterTable(t, FieldDataTable= depl_sta,
#                                                             ZoneName = 'U_sta'+str(np.round(RPM)),
#                                                             FieldNames = ['Usx', 'Usy', 'Usz', 'Usthetax', 'Usthetay', 'Usthetaz'],
#                                                             Depl = True)
#
#        for Zone in UsZone:
#            print(Zone[0])
#            NodePosition = J.getVars(Zone, ['NodesPosition'])
#            print(NodePosition)
#            J._invokeFields(Zone, ['upx', 'upy', 'upz', 'ux', 'uy', 'uz', 'upthetax', 'upthetay', 'upthetaz'])
#            upx, upy, upz, upthetax, upthetay, upthetaz  = J.getVars(Zone, ['upx', 'upy', 'upz', 'upthetax', 'upthetay', 'upthetaz'])
#            upx[:], upy[:], upz[:],upthetax[:], upthetay[:], upthetaz[:] = UsField[0][NodePosition], UsField[1][NodePosition], UsField[2][NodePosition], UsField[3][NodePosition], UsField[4][NodePosition], UsField[5][NodePosition]
#
#        # Compute the Us vector and add it to the .AssembledVectors node:
#        DictStructParam = J.get(t, '.StructuralParameters')
#
#        VectUsOmega = np.zeros((DictStructParam['MeshProperties']['Nddl'][0]))
#        VectUsOmega[::DictStructParam['MeshProperties']['ddlElem'][0]] = upx
#        VectUsOmega[1::DictStructParam['MeshProperties']['ddlElem'][0]] = upy
#        VectUsOmega[2::DictStructParam['MeshProperties']['ddlElem'][0]] = upz
#        VectUsOmega[3::DictStructParam['MeshProperties']['ddlElem'][0]] = upthetax
#        VectUsOmega[4::DictStructParam['MeshProperties']['ddlElem'][0]] = upthetay
#        VectUsOmega[5::DictStructParam['MeshProperties']['ddlElem'][0]] = upthetaz
#


    tstaT2 = GetAsterTableOfStaticNodalForces(**kwargs)
    # Table des forces nodales:

    GusZones = SJ.CreateNewSolutionFromAsterTable(t,
                                     FieldDataTable = tstaT2,
                                     ZoneName = str(np.round(RPM,2))+'RPM',
                                     FieldName = 'Gus')

    FeiZones = SJ.CreateNewSolutionFromNdArray(t,
                                     FieldDataArray = [kwargs['Fei']],
                                     ZoneName = str(np.round(RPM,2))+'RPM',
                                     FieldName = 'Fei')


    I.addChild(I.getNodeFromName(t, 'StaticRotatorySolution'), GusZones)
    I.addChild(I.getNodeFromName(t, 'StaticRotatorySolution'), FeiZones)


    DETRUIRE (CONCEPT = _F (NOM = (tstaT, tstaT2),
                            ),
              INFO = 1,
              )



    return t


def BuildFOM(t, **kwargs):

    DictStructParam = J.get(t, '.StructuralParameters')
    DictSimulaParam = J.get(t, '.SimulationParameters')

    SolverType0 = DictSimulaParam['IntegrationProperties']['SolverType']
    DictSimulaParam['IntegrationProperties']['SolverType'] = 'Static'
    J.set(t,'.SimulationParameters', **dict(DictSimulaParam))
    DictSimulaParam = J.get(t, '.SimulationParameters')

    ModesBase = I.createNode('ModalBases', 'CGNSBase_t')
    for RPM in DictSimulaParam['RotatingProperties']['RPMs']:

        t, AsterObjs = ComputeMatricesFOM(t, RPM, **kwargs['AsterObjs'])
        #C.convertPyTree2File(t,'/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/Mesh.cgns', 'bin_adf')
        #C.convertPyTree2File(t,'/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/Mesh.tp', 'bin_tp')

        t = ExtrUGStatRot(t, RPM, **AsterObjs)

        #C.convertPyTree2File(t,'/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/Mesh_Rotation.cgns', 'bin_adf')
        #C.convertPyTree2File(t,'/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/Mesh_Rotation.tp', 'bin_tp')
        #C.convertPyTree2File(t,'/scratchm/mbalmase/Spiro/3_Update4MOLA/CouplingWF_NewMOLA/Test1.cgns', 'bin_adf')
        #C.convertPyTree2File(t,'/scratchm/mbalmase/Spiro/3_Update4MOLA/CouplingWF_NewMOLA/Test1.tp', 'bin_tp')
        DictSimulaParam = J.get(t, '.SimulationParameters')


        t = MA.CalcLNM(t, RPM, **AsterObjs)
        #C.convertPyTree2File(t,'/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/Mesh_Rotation_LNM.cgns', 'bin_adf')



        SJ.DestroyAsterObjects(AsterObjs,
                               DetrVars = ['Cfd', 'SOLU', 'RAMPE', 'L_INST',
                                           'Komeg2', 'MASS1', 'NUME'])


    t, parametric = BuildRPMParametrisation(t)

    # Compute the Aij and Bijm Coefficients for the nonlinear forces:?????
    if not parametric:

        for RPM in DictSimulaParam['RotatingProperties']['RPMs']:

            if DictStructParam['ROMProperties']['ROMForceType'] != 'Linear':
                t = NFM.ComputeNLCoefficients(t, RPM, **AsterObjs)




    DictSimulaParam['IntegrationProperties']['SolverType'] = SolverType0
    print(DictSimulaParam['IntegrationProperties']['SolverType'])

    J.set(t,'.SimulationParameters', **dict(DictSimulaParam))

    SJ.SaveModel(t, kwargs['FOMName'], Modes = True, StaticRotatorySolution = True)

    return t

def COMPUTE_FOMmodel(t, FOMName):

    DictStructParam = J.get(t, '.StructuralParameters')

    t, AsterObjs = BuildFEmodel(t)

    t = BuildFOM(t, **SJ.merge_dicts(dict(AsterObjs = AsterObjs), dict(FOMName = FOMName)))

    return t



################################################################
################################################################



####################
# ROM Model        :
####################

def CreateNewROMTreeWithParametersAndBases(tFOM):

    tROM = I.newCGNSTree()

    DictStructParam = J.get(tFOM, '.StructuralParameters')
    DictSimulaParam = J.get(tFOM, '.SimulationParameters')
    #DictBases = J.get(tFOM, 'ModalBases')

    J.set(tROM, '.StructuralParameters',**DictStructParam)
    J.set(tROM, '.SimulationParameters',**DictSimulaParam)
    #J.set(tROM, 'ModalBases',**DictBases)


    I._addChild(tROM, I.getNodeByName(tFOM, 'ModalBases'))


    return tROM

def BuildROMMatrices(tFOM, tROM):
    DictSimulaParam = J.get(tFOM, '.SimulationParameters')

    DictAssembledMatrices = J.get(tFOM, '.AssembledMatrices')
    try:
        Check = DictAssembledMatrices['Parametric']
        Parametric = True
    except:
        Parametric = False

    if not Parametric:
        for RPM in DictSimulaParam['RotatingProperties']['RPMs']:
            #For a single value of RPM


            PHI =SJ.GetReducedBaseFromCGNS(tFOM, RPM) #    DictAssembledMatrices['PHI']
            PHIt = PHI.transpose()

            MatrRed = J.get(tROM, '.AssembledMatrices')
            MatrRed[str(np.round(RPM,2))+'RPM'] = {}
            MatrRed[str(np.round(RPM,2))+'RPM']['PHI'] = PHI
            for MatrixName in DictAssembledMatrices[str(np.round(RPM,2))+'RPM'].keys():
                if MatrixName != 'PHI':#print(MatrixName)
                    SFOMMatr, _ = SJ.LoadSMatrixFromCGNS(tFOM, RPM, MatrixName)
                    MatrRed[str(np.round(RPM,2))+'RPM'][MatrixName] = PHIt.dot(SFOMMatr.dot(PHI))

            J.set(tROM, '.AssembledMatrices', **MatrRed)
    else: # Parametric Model

        MatrRed = J.get(tROM, '.AssembledMatrices')
        MatrRed['Parametric']={}
        PHI = SJ.GetReducedBaseFromCGNS(tFOM, 'Parametric')
        MatrRed['Parametric']['PHI'] = PHI
        PHIt = PHI.transpose()

        for MatrixName in DictAssembledMatrices['Parametric'].keys():
            if MatrixName not in ['PHI', 'p0', 'Range','PHIAug', 'Deltap']:
                SFOMMatr, _ = SJ.LoadSMatrixFromCGNS(tFOM, 'Parametric', MatrixName)
                #Matrices are projected
                MatrRed['Parametric'][MatrixName] = PHIt.dot(SFOMMatr.dot(PHI))

        J.set(tROM, '.AssembledMatrices', **MatrRed)

#        Check if the parametric model is well created:
#
#        DictSimulaParam = J.get(tFOM, '.SimulationParameters')
#        RPMs = DictSimulaParam['RotatingProperties']['RPMs']
#
#        #Also full Komeg matrix from FOM are saved to make a later comparison
#        for RPMval in  RPMs:
#                MatrRed[str(np.round(RPMval,2))+'RPM']={}
#                MatrixName = 'Komeg'#print(MatrixName)
#                SFOMMatr, _ = SJ.LoadSMatrixFromCGNS(tFOM, RPMval, MatrixName)
#                MatrRed[str(np.round(RPMval,2))+'RPM'][MatrixName] = PHIt.dot(SFOMMatr.dot(PHI))
#                #The base PHI from reduced model is used
#

    return tROM


def copyInternalForcesCoefficients(tFOM, tROM, RPM):


    DictIntForces = J.get(tFOM, '.InternalForcesCoefficients')

    J.set(tROM, '.InternalForcesCoefficients', **DictIntForces)

    return tROM

def copyFromFOM2ROMDict(tFOM, tROM, Name):
    DictName = J.get(tFOM, Name)

    J.set(tROM, Name, **DictName)

    return tROM

def copyAssembledVectors(tFOM, tROM, RPM):

    DictIntForces = J.get(tFOM, '.AssembledVectors')

    J.set(tROM, '.AssembledVectors', **DictIntForces)

    return tROM


def COMPUTE_ROMmodel(tFOM, ROMName):

    #DictStructParam = J.get(tFOM, '.StructuralParameters')
    DictSimulaParam = J.get(tFOM, '.SimulationParameters')

    tROM = CreateNewROMTreeWithParametersAndBases(tFOM)
    I._addChild(tROM, I.getNodeByName(tFOM, 'SOLID'))
    #RPMs = DictSimulaParam['RotatingProperties']['RPMs']

    tROM = BuildROMMatrices(tFOM, tROM)


    for Name in ['.SimulationParameters', '.AerodynamicProperties', '.StructuralParameters', '.AssembledVectors', '.InternalForcesCoefficients']:

        tROM = copyFromFOM2ROMDict(tFOM,tROM, Name)

        #tROM = copyAssembledVectors(tFOM, tROM, RPM)

        #tROM = copyInternalForcesCoefficients(tFOM, tROM, RPM)

    SJ.SaveModel(tROM, ROMName)

    return tROM



def ComputeStaFullNodalF(t, **kwargs):

    tstaT2 = GetAsterTableOfStaticNodalForces(**kwargs)

    VectNodalF = VectFromAsterTable2Full(t, tstaT2)

#    ForceNodeTable = tstaT2.EXTR_TABLE()['NOEUD', 'DX', 'DY', 'DZ']
#
#    FieldCoordX = np.array(ForceNodeTable.values()['DX'][:])
#    FieldCoordY = np.array(ForceNodeTable.values()['DY'][:])
#    FieldCoordZ = np.array(ForceNodeTable.values()['DZ'][:])
#
#    DictStructParam = J.get(t, '.StructuralParameters')
#
#    VectFnl = np.zeros((DictStructParam['MeshProperties']['Nddl'][0]))
#    VectFnl[::3] = FieldCoordX
#    VectFnl[1::3] = FieldCoordY
#    VectFnl[2::3] = FieldCoordZ

    DETRUIRE(CONCEPT = _F(NOM = (tstaT2)))

    return VectNodalF

def ComputeFullNodalF(t, **kwargs):

    tstaT2 = GetAsterTableOfStaticNodalForces(InstantsExtr = kwargs['Instants'],**kwargs)

    VectNodalF = VectFromAsterTable2Full(t, tstaT2)

#    ForceNodeTable = tstaT2.EXTR_TABLE()['NOEUD', 'DX', 'DY', 'DZ']
#
#    FieldCoordX = np.array(ForceNodeTable.values()['DX'][:])
#    FieldCoordY = np.array(ForceNodeTable.values()['DY'][:])
#    FieldCoordZ = np.array(ForceNodeTable.values()['DZ'][:])
#
#    DictStructParam = J.get(t, '.StructuralParameters')
#
#    VectFnl = np.zeros((DictStructParam['MeshProperties']['Nddl'][0]))
#    VectFnl[::3] = FieldCoordX
#    VectFnl[1::3] = FieldCoordY
#    VectFnl[2::3] = FieldCoordZ

    DETRUIRE(CONCEPT = _F(NOM = (tstaT2)))

    return VectNodalF

def ComputeStaticU4GivenLoading(t, RPM, LoadVector, **kwargs):


    DictStructParam = J.get(t, '.StructuralParameters')
    DictSimulaParam = J.get(t, '.SimulationParameters')

    ListeLoading = SJ.TranslateNumpyLoadingVector2AsterList(t, LoadVector)

    RAMPE_r = DEFI_FONCTION(NOM_PARA = 'INST',
                            VALE = (-2.,0.,0.,1.),
                            PROL_DROITE = 'CONSTANT',
                            PROL_GAUCHE = 'CONSTANT',
                            INTERPOL = 'LIN'
                            );

    RAMPE = DEFI_FONCTION(NOM_PARA = 'INST',
                          VALE = (0.0,0.0,1.0,1.0),
                          PROL_DROITE = 'CONSTANT',
                          PROL_GAUCHE = 'CONSTANT',
                          INTERPOL = 'LIN'
                          );


    F_ext = AFFE_CHAR_MECA(MODELE = kwargs['MODELE'],
                         DDL_IMPO=SJ.AffectImpoDDLByGroupType(t),
                         FORCE_NODALE =  ListeLoading,
                         );

    F_rota = AFFE_CHAR_MECA(MODELE = kwargs['MODELE'],
                            ROTATION = _F(VITESSE = np.pi * RPM/30.,
                                       AXE = DictSimulaParam['RotatingProperties']['AxeRotation'] ,
                                       CENTRE = DictSimulaParam['RotatingProperties']['RotationCenter'],),
                           )

    L_INST = DEFI_LIST_REEL(VALE = SJ.ComputeTimeVector(t)[1]
                            );


    SOLU = STAT_NON_LINE(MODELE = kwargs['MODELE'],
                         CHAM_MATER = kwargs['CHMAT'],
                         CARA_ELEM  = kwargs['CARELEM'],
                         EXCIT =( _F(CHARGE = F_ext,
                                     FONC_MULT=RAMPE,),
                                  _F(CHARGE = F_rota,
                                     FONC_MULT = RAMPE_r),
                                ),
                         COMPORTEMENT = DefineBehaviourLaws(t),
                         CONVERGENCE=_F(RESI_GLOB_MAXI=DictSimulaParam['IntegrationProperties']['EpsConvergenceCriteria'][0],
                                        RESI_GLOB_RELA=1e-4,
                                        ITER_GLOB_MAXI=DictSimulaParam['IntegrationProperties']['NumberOfMaxIterations'][0],
                                        ARRET = 'OUI',),
                         INCREMENT = _F( LIST_INST = L_INST,
                                        ),
                         INFO = 1,
                        ),

    AsterObjs = SJ.merge_dicts(kwargs, dict(SOLU = SOLU, RAMPE = RAMPE, L_INST = L_INST))


    UpFromOmegaAndFe = ExtrFromAsterSOLUwithOmegaFe(t, RPM, **dict(SOLU = SOLU))

    GusFromOmegaAnfFe = ComputeStaFullNodalF(t, **AsterObjs)


    SJ.DestroyAsterObjects(dict(**dict(RAMPE_r = RAMPE_r, F_rota = F_rota, F_ext = F_ext, SOLU= SOLU, RAMPE = RAMPE, L_INST = L_INST)),
                           DetrVars = ['RAMPE_r', 'F_rota','F_ext', 'SOLU', 'RAMPE', 'L_INST',
                                      ])


    return  UpFromOmegaAndFe, GusFromOmegaAnfFe


def ComputeDynamic4GivenForceCoeffAndRPM(t, RPM, ForceCoeff, **kwargs):


    DictStructParam = J.get(t, '.StructuralParameters')
    DictSimulaParam = J.get(t, '.SimulationParameters')

    LoadingVector = ForceCoeff * DictSimulaParam['LoadingProperties']['ExternalForcesVector']['ShapeFunctionProj'][str(np.round(RPM,2))+'RPM'] * DictSimulaParam['LoadingProperties']['ExternalForcesVector']['Fmax']

    ListeLoading = SJ.TranslateNumpyLoadingVector2AsterList(t, LoadingVector)

    RAMPE_r = DEFI_FONCTION(NOM_PARA = 'INST',
                            VALE = (-2.,0.,0.,1.),
                            PROL_DROITE = 'CONSTANT',
                            PROL_GAUCHE = 'CONSTANT',
                            INTERPOL = 'LIN'
                            );



    timeCalcul = SJ.ComputeTimeVector(t)[1][DictSimulaParam['IntegrationProperties']['Steps4CentrifugalForce'][0]-1:]
    TimeFunction = DictSimulaParam['LoadingProperties']['ExternalForcesVector']['TimeFuntionVector']
    ValeAsterFunction = np.zeros((2*len(TimeFunction),))
    ValeAsterFunction[::2]  = timeCalcul
    ValeAsterFunction[1::2] = TimeFunction

    TFUNCEXT = DEFI_FONCTION(NOM_PARA = 'INST',
                             VALE = ValeAsterFunction,
                             PROL_DROITE = 'CONSTANT',
                             PROL_GAUCHE = 'CONSTANT',
                             INTERPOL = 'LIN'
                             );


    F_ext = AFFE_CHAR_MECA(MODELE = kwargs['MODELE'],
                           DDL_IMPO=SJ.AffectImpoDDLByGroupType(t),
                           FORCE_NODALE =  ListeLoading,
                           );

    F_rota = AFFE_CHAR_MECA(MODELE = kwargs['MODELE'],
                            ROTATION = _F(VITESSE = np.pi * RPM/30.,
                            AXE = DictSimulaParam['RotatingProperties']['AxeRotation'] ,
                            CENTRE = DictSimulaParam['RotatingProperties']['RotationCenter'],),
                           );

    L_INST = DEFI_LIST_REEL(VALE = SJ.ComputeTimeVector(t)[1]
                            );

    SOLU = DYNA_NON_LINE(MODELE = kwargs['MODELE'],
                         CHAM_MATER = kwargs['CHMAT'],
                         CARA_ELEM  = kwargs['CARELEM'],
                         EXCIT =( _F(CHARGE = F_ext,
                                     FONC_MULT=TFUNCEXT,),
                                  _F(CHARGE = F_rota,
                                     FONC_MULT = RAMPE_r),
                                ),
                         COMPORTEMENT = DefineBehaviourLaws(t),
                         CONVERGENCE=_F(RESI_GLOB_MAXI=DictSimulaParam['IntegrationProperties']['EpsConvergenceCriteria'][0],
                                        RESI_GLOB_RELA=1e-4,
                                        ITER_GLOB_MAXI=DictSimulaParam['IntegrationProperties']['NumberOfMaxIterations'][0],
                                        ARRET = 'OUI',),
                         INCREMENT = _F( LIST_INST = L_INST,
                                        ),
                         AMOR_RAYL_RIGI = 'ELASTIQUE',
                         SCHEMA_TEMPS = _F(SCHEMA = 'HHT',
                                           FORMULATION ='DEPLACEMENT',
                                           ALPHA = -1.*float(DictSimulaParam['IntegrationProperties']['IntegrationMethod']['Parameters']['Alpha']),
                                           MODI_EQUI = 'OUI'),
                         INFO = 1,
                        ),
    TimeSave = timeCalcul[::DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]]
    if TimeSave[-1] != timeCalcul[-1]:
        TimeSave.append(timeCalcul[-1])

    AsterObjs = SJ.merge_dicts(kwargs, dict(SOLU = SOLU, RAMPE = TFUNCEXT, L_INST = L_INST, Instants = TimeSave))




    UpFromOmegaAndFe = ExtrFromAsterSOLUwithOmegaFe(t, RPM,**AsterObjs)

    VelocityFromOmegaAndFe = ExtrFromAsterSOLUwithOmegaFe(t, RPM,ChampName = 'VITE', **AsterObjs)

    AccelerationFromOmegaAndFe = ExtrFromAsterSOLUwithOmegaFe(t, RPM,ChampName = 'ACCE',**AsterObjs)


    GusFromOmegaAnfFe = ComputeFullNodalF(t, **AsterObjs)

    # Keep the solution of the instants of interest:

    SJ.DestroyAsterObjects(dict(**dict(RAMPE_r = RAMPE_r, F_rota = F_rota, F_ext = F_ext, SOLU= SOLU, RAMPE = TFUNCEXT, L_INST = L_INST)),
                           DetrVars = ['RAMPE_r', 'F_rota','F_ext', 'SOLU', 'TFUNCEXT', 'L_INST',
                                      ])


    return  UpFromOmegaAndFe,VelocityFromOmegaAndFe,AccelerationFromOmegaAndFe,  GusFromOmegaAnfFe
