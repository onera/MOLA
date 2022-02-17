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

def ModifySolidCGNS2Mesh(t):
    '''Read the structured SOLID node and adapt it to an unstructured Mesh node'''

    # Compute the number of nodes in the mesh and add it to t:
    DictStructParam = J.get(t, '.StructuralParameters')

    mm = partition.MAIL_PY()
    mm.FromAster('MAIL')
    
    DictStructParam['MeshProperties']['NNodes'] = len(list(mm.correspondance_noeuds))




    print(WARN+'Only one type of mesh element! or with same number of ddl'+ENDC)

    DictStructParam['MeshProperties']['Nddl'] = int(DictStructParam['MeshProperties']['ddlElem']*DictStructParam['MeshProperties']['NNodes'])
    

    DictStructParam['MeshProperties']['NodesFamilies'] = {}
    for Family in mm.gno.keys():

      DictStructParam['MeshProperties']['NodesFamilies'][Family] = mm.gno[Family]

    J.set(t, '.StructuralParameters', **DictStructParam)

 ######################  A ADAPTER POUR D'AUTRES FEM
    if I.getNodeByName(t, 'stack') == None:
        #Conectivity = mm.co
        #Conectivity = [val + 1 for val in Conectivity]
        #Coordinates = mm.cn
        
        

        
        DictElements = {}
        DictElements['GroupOfElements'] = {}

        Base = I.newCGNSBase('SOLID', parent=t)

        def CGNSElementType(NameAster):

            if NameAster == 'POI1':
                NameCGNS = 'NODE'
                NodeElem = 1

            elif NameAster == 'SEG2':
                NameCGNS = 'BAR_2'
                NodeElem = 2


            return NameCGNS, NodeElem

        def ExtractConnectivity(mm, DictElements, ElemType):
            
            Conectivity = mm.co
            Conectivity = [val + 1 for val in Conectivity]

            _, NodeElem = CGNSElementType(ElemType)

            ConnectMatrix = np.zeros((len(DictElements['GroupOfElements'][ElemType]['ListOfElem']), NodeElem))
            for Element, pos in zip(DictElements['GroupOfElements'][ElemType]['ListOfElem'], range(len(DictElements['GroupOfElements'][ElemType]['ListOfElem']))):
                elementPosition = int(Element[1:]) - 1

                ConnectMatrix[pos, :] = np.squeeze(np.asarray(Conectivity[elementPosition]))

            return ConnectMatrix

        def ExtractCoordinates(mm, DictElements, ElemType):
            
            Coordinates = mm.cn
            if len(Coordinates[0,:]) == 2:
                Coordinates = np.append(Coordinates, np.zeros((len(Coordinates[:,0]),1)), axis = 1)
            
            ValidNodes = []
            for Element, pos in zip(DictElements['GroupOfElements'][ElemType]['ListOfElem'], range(len(DictElements['GroupOfElements'][ElemType]['ListOfElem']))):
                elementPosition = int(Element[1:]) - 1
                for Node in mm.co[elementPosition]:
                    if Node not in ValidNodes:
                        ValidNodes.append(Node)
            
            return np.array(Coordinates[ValidNodes]), ValidNodes

        def CreateUnstructuredZone4ElemenType(Base, ElementName, CGNSType, NPts, NElts, CoordinatesE, ConectE):

            #zoneUns = I.createNode('InitialMesh_'+ElementName,ntype='Zone_t',value=np.array([[NPts, NElts,0]],dtype=np.int32,order='F'), parent= Base)
            #print(NPts, NElts)
            zoneUns = I.newZone(name = 'InitialMesh_'+ElementName, zsize = [[NPts, NElts, 0]]  , ztype = 'Unstructured', parent= Base)
            zt_n = I.createNode('ZoneType', ntype='ZoneType_t',parent=zoneUns)
            I.setValue(zt_n,'Unstructured')
    
    
            g = I.newGridCoordinates(parent = zoneUns)
            I.newDataArray('CoordinateX', value=np.array(CoordinatesE[:,0],dtype=np.float32,order='F'), parent=g)
            I.newDataArray('CoordinateY', value=np.array(CoordinatesE[:,1],dtype=np.float32,order='F'), parent=g)
            I.newDataArray('CoordinateZ', value=np.array(CoordinatesE[:,2],dtype=np.float32,order='F'), parent=g)
            #I.printTree(g)

            
            aa = I.newElements(name='Elements', etype=CGNSType, econnectivity=np.array(ConectE.flatten(),dtype=np.int32,order='F'), erange = np.array([1,NElts]), eboundary=0, parent =zoneUns)
            #I.printTree(Base)
            return Base
        
        for element in range(len(mm.tm)):
            
            DictElements['E%s'%(element+1)] = {}

            DictElements['E%s'%(element+1)]['AsterType'] = mm.nom[mm.tm[element]]
            DictElements['E%s'%(element+1)]['CGNSType'],_ = CGNSElementType(DictElements['E%s'%(element+1)]['AsterType'])
            #print(DictElements)
            try:
                DictElements['GroupOfElements'][DictElements['E%s'%(element+1)]['AsterType']]['ListOfElem'].append('E%s'%(element + 1))  
            
            except:
                DictElements['GroupOfElements'][DictElements['E%s'%(element+1)]['AsterType']] = {}
                DictElements['GroupOfElements'][DictElements['E%s'%(element+1)]['AsterType']]['ListOfElem'] = ['E%s'%(element + 1)]
                DictElements['GroupOfElements'][DictElements['E%s'%(element+1)]['AsterType']]['CGNSType'] = DictElements['E%s'%(element+1)]['CGNSType']                

            DictElements['NbOfElementType'] = len(DictElements['GroupOfElements'].keys())

        
        for ElemName in DictElements['GroupOfElements'].keys():
            #if ElemName != 'POI1':
                CoordinatesE, NodesElemType  = ExtractCoordinates(mm, DictElements, ElemName)
                DictElements['GroupOfElements'][ElemName]['Coordinates'] = CoordinatesE 
                DictElements['GroupOfElements'][ElemName]['NodesPosition'] = NodesElemType
                DictElements['GroupOfElements'][ElemName]['Nodes'] = np.array(NodesElemType) + 1
                
                ConectivityE  = ExtractConnectivity(mm, DictElements, ElemName)
                DictElements['GroupOfElements'][ElemName]['Connectivity'] = ConectivityE 
                
                #print(DictElements['GroupOfElements'])
                Base = CreateUnstructuredZone4ElemenType(Base, ElemName, DictElements['GroupOfElements']['%s'%ElemName]['CGNSType'], len(CoordinatesE), len(DictElements['GroupOfElements'][ElemName]['ListOfElem']), CoordinatesE, ConectivityE)
                #print(DictElements['GroupOfElements'][ElemName])

        DictStructParam['MeshProperties']['DictElements'] = DictElements
        J.set(t, '.StructuralParameters', **DictStructParam)
        #C.convertPyTree2File(t, '/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/Test1.cgns', 'bin_adf')
        #C.convertPyTree2File(t, '/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/Test1.tp', 'bin_tp')
                


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
    

    t = I.merge([t, C.newPyTree(['StaticRotatorySolution', [], 'ModalBases', []])])
    #InitZone = I.getNodesFromNameAndType(t, 'InitialMesh', 'Zone_t')[0]
    #J._invokeFields(InitZone,['upx', 'upy', 'upz'])
    

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
    
    for LocMat, it in zip(DictStructParam['MaterialProperties']['Materials'].keys(), range(len(DictStructParam['MaterialProperties']['Materials'].keys()))):
        MaterialDict[LocMat] = {}
        M.append([None])

        M[it] = DEFI_MATERIAU(ELAS=_F(E= DictStructParam['MaterialProperties']['Materials'][LocMat]['E'],
                                    NU=DictStructParam['MaterialProperties']['Materials'][LocMat]['PoissonRatio'],
                                    RHO=DictStructParam['MaterialProperties']['Materials'][LocMat]['Rho'],
                                    AMOR_ALPHA = DictStructParam['MaterialProperties']['Materials'][LocMat]['XiAlpha'],
                                    AMOR_BETA =  4*np.pi*DictStructParam['MaterialProperties']['Materials'][LocMat]['Freq4Dumping']*DictStructParam['MaterialProperties']['Materials'][LocMat]['XiBeta'],
                                    ),);
        
        Ms.append(M[it])
        MaterialDict[LocMat]['Properties'] = M[it]

        MaterialDict[LocMat]['Mesh'] = DictStructParam['MaterialProperties']['Materials'][LocMat]['MeshGroup']

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

    for caraKey in DictCaracteristics.keys():

        if DictCaracteristics[caraKey]['KeyWord'] == 'POUTRE':
            affe_Poutre.append(_F(SECTION = DictCaracteristics[caraKey]['SectionType'], 
                                  VARI_SECT = DictCaracteristics[caraKey]['SectionVariation'], 
                                  CARA = DictCaracteristics[caraKey]['Properties'],
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


    #CARELEM = None

    return CARELEM

def BuildFEmodel(t):
    '''Reads the mesh, creates the FE model in aster and computes the FOM matrices for the studied case.
    The Output is a cgns file of t with the FOM matrices and the model parameters.
      -- This program is inspired by the Load_FE_model.py function.
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

#    ListDictCaracteristics = [dict(Section = 'CERCLE', CARA= (), VALE = (), GROUP_MA = '' ), 
#                              dict()
#                              ]
#
#    cara = AFFE_CARA_ELEM(POUTRE= (
#                                   _F(SECTION=’CERCLE’,CARA=(’R’,’EP’),VALE=(0.1,0.02),GROUP_MA=(’M1’,’M5’)),
#                                   _F(SECTION=’CERCLE’,CARA=(’R’,’EP’),VALE=(0.2,0.05),GROUP_MA= ’M3’),
#                                   _F(SECTION=’CERCLE’,CARA=(’R’,’EP’),VALE=(0.09,0.01),GROUP_MA= ’M6’),
#                                   _F(SECTION=’CERCLE’,CARA=(’R1’,’R2’),VALE=(0.1,0.2),GROUP_MA=(’M2’,’M4’)),
#                                   _F(SECTION=’CERCLE’,CARA=(’EP1’,’EP2’),VALE=(0.02,0.05),GROUP_MA=(’M2’,’M4’)
#                                  ),
#                                   ),
#                                  )
#



    
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




def ComputeMatricesFOM(t, RPM, **kwargs): 
    '''
    Computes the FOM matrices for a given velocity RPM (Ke, Kg, Kc, C, M).
    Returns K(Omega) and M
    '''

    DictStructParam = J.get(t, '.StructuralParameters')
    DictSimulaParam = J.get(t, '.SimulationParameters')

    if DictStructParam['MaterialProperties']['BehaviourLaw'] == 'Green':
        DictStructParam['MaterialProperties']['TyDef'] = 'PETIT'
        sufix = '_L'
    elif DictStructParam['MaterialProperties']['BehaviourLaw'] == 'Green-Lagrange':
        DictStructParam['MaterialProperties']['TyDef'] = 'GROT_GDEP'
        sufix = '_NL'

    J.set(t, '.StructuralParameters', **DictStructParam)

    # Define the external loading and solve the problem:

    
    Cfd = AFFE_CHAR_MECA(MODELE = kwargs['MODELE'],
                         ROTATION = _F(VITESSE = np.pi * RPM/30., 
                                       AXE = DictStructParam['RotatingProperties']['AxeRotation'], 
                                       CENTRE = DictStructParam['RotatingProperties']['RotationCenter'],),
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
                                             NOMBRE = DictSimulaParam['IntegrationProperties']['StaticSteps'],),
                                          ),
                                  );
          
    SOLU = STAT_NON_LINE(MODELE = kwargs['MODELE'],
                         CHAM_MATER = kwargs['CHMAT'],
                         CARA_ELEM = kwargs['CARELEM'], 
                         EXCIT =( _F(CHARGE = Cfd,
                                     FONC_MULT=RAMPE,),), 
                         COMPORTEMENT = _F(RELATION = 'ELAS',
                                         DEFORMATION = DictStructParam['MaterialProperties']['TyDef'], 
                                         TOUT = 'OUI',
                                        ),
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

def ExtrUpFromAsterSOLUwithOmegaFe(t, RPM, **kwargs):
    DictStructParam = J.get(t, '.StructuralParameters')

    if DictStructParam['MeshProperties']['ddlElem'][0] == 3:

        tstaT = CREA_TABLE(RESU = _F(RESULTAT= kwargs['SOLU'],
                                             NOM_CHAM='DEPL',
                                             NOM_CMP= ('DX','DY','DZ'),
                                             INST = 1.0,
                                             TOUT = 'OUI',),
                                   TYPE_TABLE='TABLE',
                                   TITRE='Table_Depl_R',
                                   )
                        
        # Tableau complet des deplacements, coordonnees modales,... :
        
        depl_sta = tstaT.EXTR_TABLE()['NOEUD', 'DX', 'DY', 'DZ']
    
        UsZone, UsField = SJ.CreateNewSolutionFromAsterTable(t, FieldDataTable= depl_sta,
                                                             ZoneName = 'U_sta'+str(np.round(RPM)), 
                                                             FieldNames = ['Usx', 'Usy', 'Usz'],
                                                             Depl = True)
        
        J._invokeFields(UsZone, ['upx', 'upy', 'upz', 'ux', 'uy', 'uz'])
        upx, upy, upz = J.getVars(UsZone, ['upx', 'upy', 'upz'])
        upx[:], upy[:], upz[:] = UsField[0], UsField[1], UsField[2] 

    elif DictStructParam['MeshProperties']['ddlElem'][0] == 6:
        tstaT = CREA_TABLE(RESU = _F(RESULTAT= kwargs['SOLU'],
                                             NOM_CHAM='DEPL',
                                             NOM_CMP= ('DX','DY','DZ', 'DRX', 'DRY', 'DRZ'),
                                             INST = 1.0,
                                             TOUT = 'OUI',),
                                   TYPE_TABLE='TABLE',
                                   TITRE='Table_Depl_R',
                                   )
                        
        # Tableau complet des deplacements, coordonnees modales,... :
        
        depl_sta = tstaT.EXTR_TABLE()['NOEUD', 'DX', 'DY', 'DZ','DRX', 'DRY', 'DRZ']

        UsZone, UsField = SJ.CreateNewSolutionFromAsterTable(t, FieldDataTable= depl_sta,
                                                             ZoneName = 'U_sta'+str(np.round(RPM)), 
                                                             FieldNames = ['Usx', 'Usy', 'Usz', 'Usthetax', 'Usthetay', 'Usthetaz'],
                                                             Depl = True)
        
        J._invokeFields(UsZone, ['upx', 'upy', 'upz', 'ux', 'uy', 'uz', 'upthetax', 'upthetay', 'upthetaz'])
        upx, upy, upz, upthetax, upthetay, upthetaz  = J.getVars(UsZone, ['upx', 'upy', 'upz', 'upthetax', 'upthetay', 'upthetaz'])
        upx[:], upy[:], upz[:],upthetax[:], upthetay[:], upthetaz[:] = UsField[0], UsField[1], UsField[2], UsField[3], UsField[4], UsField[5] 





    # Compute the Us vector and add it to the .AssembledVectors node:
    DictStructParam = J.get(t, '.StructuralParameters')
    
    VectUpOmegaFe = np.zeros((DictStructParam['MeshProperties']['Nddl'][0]))
    VectUpOmegaFe[::DictStructParam['MeshProperties']['ddlElem'][0]] = upx
    VectUpOmegaFe[1::DictStructParam['MeshProperties']['ddlElem'][0]] = upy
    VectUpOmegaFe[2::DictStructParam['MeshProperties']['ddlElem'][0]] = upz

    DETRUIRE (CONCEPT = _F (NOM = (tstaT),
                            ), 
              INFO = 1,
              )

    return VectUpOmegaFe

def ExtrUGStatRot(t, RPM, **kwargs):

    DictStructParam = J.get(t, '.StructuralParameters')
    C.convertPyTree2File(t, '/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/Test1.cgns', 'bin_adf')
    C.convertPyTree2File(t, '/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/Test1.tp', 'bin_tp')
                

    if DictStructParam['MeshProperties']['ddlElem'][0] == 3:
    
        tstaT = CREA_TABLE(RESU = _F(RESULTAT= kwargs['SOLU'],
                                             NOM_CHAM='DEPL',
                                             NOM_CMP= ('DX','DY','DZ'),
                                             INST = 1.0,
                                             TOUT = 'OUI',),
                                   TYPE_TABLE='TABLE',
                                   TITRE='Table_Depl_R',
                                   )
                        
        # Tableau complet des deplacements, coordonnees modales,... :
        
        depl_sta = tstaT.EXTR_TABLE()['NOEUD', 'DX', 'DY', 'DZ']

        UsZone, UsField = SJ.CreateNewSolutionFromAsterTable(t, FieldDataTable= depl_sta,
                                                             ZoneName = 'U_sta'+str(np.round(RPM)), 
                                                             FieldNames = ['Usx', 'Usy', 'Usz'],
                                                             Depl = True)
        
        J._invokeFields(UsZone, ['upx', 'upy', 'upz', 'ux', 'uy', 'uz'])
        upx, upy, upz = J.getVars(UsZone, ['upx', 'upy', 'upz'])
        upx[:], upy[:], upz[:], = UsField[0], UsField[1], UsField[2] 

            # Compute the Us vector and add it to the .AssembledVectors node:
        DictStructParam = J.get(t, '.StructuralParameters')
        
        VectUsOmega = np.zeros((DictStructParam['MeshProperties']['Nddl'][0]))
        VectUsOmega[::DictStructParam['MeshProperties']['ddlElem'][0]] = upx
        VectUsOmega[1::DictStructParam['MeshProperties']['ddlElem'][0]] = upy
        VectUsOmega[2::DictStructParam['MeshProperties']['ddlElem'][0]] = upz

    elif DictStructParam['MeshProperties']['ddlElem'][0] == 6:

        tstaT = CREA_TABLE(RESU = _F(RESULTAT= kwargs['SOLU'],
                                             NOM_CHAM='DEPL',
                                             NOM_CMP= ('DX','DY','DZ', 'DRX', 'DRY', 'DRZ'),
                                             INST = 1.0,
                                             TOUT = 'OUI',),
                                   TYPE_TABLE='TABLE',
                                   TITRE='Table_Depl_R',
                                   )
                        
        # Tableau complet des deplacements, coordonnees modales,... :
        
        depl_sta = tstaT.EXTR_TABLE()['NOEUD', 'DX', 'DY', 'DZ','DRX', 'DRY', 'DRZ']

        UsZone, UsField = SJ.CreateNewSolutionFromAsterTable(t, FieldDataTable= depl_sta,
                                                             ZoneName = 'U_sta'+str(np.round(RPM)), 
                                                             FieldNames = ['Usx', 'Usy', 'Usz', 'Usthetax', 'Usthetay', 'Usthetaz'],
                                                             Depl = True)
        
        for Zone in UsZone:
            print(Zone[0])
            NodePosition = J.getVars(Zone, ['NodesPosition'])
            print(NodePosition)
            J._invokeFields(Zone, ['upx', 'upy', 'upz', 'ux', 'uy', 'uz', 'upthetax', 'upthetay', 'upthetaz'])
            upx, upy, upz, upthetax, upthetay, upthetaz  = J.getVars(Zone, ['upx', 'upy', 'upz', 'upthetax', 'upthetay', 'upthetaz'])
            upx[:], upy[:], upz[:],upthetax[:], upthetay[:], upthetaz[:] = UsField[0][NodePosition], UsField[1][NodePosition], UsField[2][NodePosition], UsField[3][NodePosition], UsField[4][NodePosition], UsField[5][NodePosition] 
             
        # Compute the Us vector and add it to the .AssembledVectors node:
        DictStructParam = J.get(t, '.StructuralParameters')
        
        VectUsOmega = np.zeros((DictStructParam['MeshProperties']['Nddl'][0]))
        VectUsOmega[::DictStructParam['MeshProperties']['ddlElem'][0]] = upx
        VectUsOmega[1::DictStructParam['MeshProperties']['ddlElem'][0]] = upy
        VectUsOmega[2::DictStructParam['MeshProperties']['ddlElem'][0]] = upz
        VectUsOmega[3::DictStructParam['MeshProperties']['ddlElem'][0]] = upthetax
        VectUsOmega[4::DictStructParam['MeshProperties']['ddlElem'][0]] = upthetay
        VectUsOmega[5::DictStructParam['MeshProperties']['ddlElem'][0]] = upthetaz




    t = SJ.AddFOMVars2Tree(t, RPM, Vars = [VectUsOmega],
                                   VarsName = ['Us'],
                                   Type = '.AssembledVectors',
                                   )
 


    F_noda = CALC_CHAMP(MODELE = kwargs['MODELE'],
                        CHAM_MATER =kwargs['CHMAT'],
                        INST = 1.0,
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
    tstaT2 = CREA_TABLE(RESU = _F(RESULTAT = F_noda,
                                     NOM_CHAM = 'FORC_NODA',
                                     NOM_CMP = ('DX','DY','DZ'),
                                     TOUT='OUI',
                                     ),
                           TYPE_TABLE = 'TABLE',
                           TITRE = 'Table_Force_N',
                           )
        
    # Table des forces nodales:
        
    ForceNodeTable = tstaT2.EXTR_TABLE()['NOEUD', 'DX', 'DY', 'DZ']

    GusZone,_ = SJ.CreateNewSolutionFromAsterTable(t, 
                                     FieldDataTable = ForceNodeTable,
                                     ZoneName = 'G_sta'+str(np.round(RPM)),
                                     FieldNames = ['Gusx', 'Gusy', 'Gusz'], 
                                     Depl = False,
                                     DefByField = UsField)   
    
    FeiZone,_ = SJ.CreateNewSolutionFromNdArray(t, 
                                     FieldDataArray = [kwargs['Fei'][::DictStructParam['MeshProperties']['ddlElem'][0]],kwargs['Fei'][1::DictStructParam['MeshProperties']['ddlElem'][0]],kwargs['Fei'][2::DictStructParam['MeshProperties']['ddlElem'][0]], kwargs['Fei'][3::DictStructParam['MeshProperties']['ddlElem'][0]],kwargs['Fei'][4::DictStructParam['MeshProperties']['ddlElem'][0]],kwargs['Fei'][5::DictStructParam['MeshProperties']['ddlElem'][0]]],
                                     ZoneName = 'Fei'+str(np.round(RPM)),
                                     FieldNames = ['FeiX', 'FeiY', 'FeiZ', 'FeithetaX', 'FeithetaY', 'FeithetaZ'], 
                                     Depl = False,
                                     DefByField = UsField) 


    I.addChild(I.getNodeFromName(t, 'StaticRotatorySolution'), [UsZone, GusZone, FeiZone])

    
    DETRUIRE (CONCEPT = _F (NOM = (tstaT, F_noda, tstaT2),
                            ), 
              INFO = 1,
              )

    
    
    return t          
            

def BuildFOM(t, **kwargs):

    DictStructParam = J.get(t, '.StructuralParameters')
    DictSimulaParam = J.get(t, '.SimulationParameters')
    
    ModesBase = I.createNode('ModalBases', 'CGNSBase_t')
    for RPM in DictStructParam['RotatingProperties']['RPMs']:
    
        t, AsterObjs = ComputeMatricesFOM(t, RPM, **kwargs['AsterObjs'])

        t = ExtrUGStatRot(t, RPM, **AsterObjs)
        
        t = MA.CalcLNM(t, RPM, **AsterObjs)

        SJ.DestroyAsterObjects(AsterObjs, 
                               DetrVars = ['Cfd', 'SOLU', 'RAMPE', 'L_INST',
                                           'Komeg2', 'MASS1', 'NUME'])

        # Compute the Aij and Bijm Coefficients for the nonlinear forces:
     
        if DictStructParam['ROMProperties']['ROMForceType'] != 'Linear':
            t = NFM.ComputeNLCoefficients(t, RPM, **AsterObjs)

        
        
        
        
    SJ.SaveModel(t, kwargs['FOMName'])

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

def BuildROMMatrices(tFOM, tROM, RPM):

    DictAssembledMatrices = J.get(tFOM, '.AssembledMatrices')
    
    PHI =SJ.GetReducedBaseFromCGNS(tFOM, RPM) #    DictAssembledMatrices['PHI'] 
    PHIt = PHI.transpose()

    MatrRed = J.get(tROM, '.AssembledMatrices')
    MatrRed[str(np.round(RPM,2))+'RPM'] = {}
    MatrRed[str(np.round(RPM,2))+'RPM']['PHI'] = PHI
    for MatrixName in DictAssembledMatrices[str(np.round(RPM))+'RPM'].keys():
        print(MatrixName)
        SFOMMatr, _ = SJ.LoadSMatrixFromCGNS(tFOM, RPM, MatrixName)
        MatrRed[str(np.round(RPM,2))+'RPM'][MatrixName] = PHIt.dot(SFOMMatr.dot(PHI))
    
    J.set(tROM, '.AssembledMatrices', **MatrRed)

    
    return tROM

def copyInternalForcesCoefficients(tFOM, tROM, RPM):
    
    
    DictIntForces = J.get(tFOM, '.InternalForcesCoefficients')

    J.set(tROM, '.InternalForcesCoefficients', **DictIntForces)

    return tROM

def copyFromFOM2ROMDict(tFOM, tROM, RPM, Name):
    DictName = J.get(tFOM, Name)

    J.set(tROM, Name, **DictName)

    return tROM

def copyAssembledVectors(tFOM, tROM, RPM):
    
    DictIntForces = J.get(tFOM, '.AssembledVectors')

    J.set(tROM, '.AssembledVectors', **DictIntForces)

    return tROM


def COMPUTE_ROMmodel(tFOM, ROMName):

    DictStructParam = J.get(tFOM, '.StructuralParameters')


    tROM = CreateNewROMTreeWithParametersAndBases(tFOM)
    I._addChild(tROM, I.getNodeByName(tFOM, 'SOLID'))
   
    for RPM in DictStructParam['RotatingProperties']['RPMs']:
        
        tROM = BuildROMMatrices(tFOM, tROM, RPM)

        for Name in ['.SimulationParameters', '.AerodynamicProperties', '.StructuralParameters', '.AssembledVectors', '.InternalForcesCoefficients']:

            tROM = copyFromFOM2ROMDict(tFOM,tROM,RPM, Name)  

        #tROM = copyAssembledVectors(tFOM, tROM, RPM)

        #tROM = copyInternalForcesCoefficients(tFOM, tROM, RPM)

     
    SJ.SaveModel(tROM, ROMName)

    return tROM



def ComputeStaFullFnl(t, **kwargs):

    F_noda = CALC_CHAMP(MODELE = kwargs['MODELE'],
                    CHAM_MATER =kwargs['CHMAT'],
                    INST = 1.0,
                    RESULTAT = kwargs['SOLU'],
                    FORCE = 'FORC_NODA'
                    );

    tstaT2 = CREA_TABLE(RESU = _F(RESULTAT = F_noda,
                                 NOM_CHAM = 'FORC_NODA',
                                 NOM_CMP = ('DX','DY','DZ'),
                                 TOUT='OUI',
                                 ),
                       TYPE_TABLE = 'TABLE',
                       TITRE = 'Table_Force_N',
                       )

    ForceNodeTable = tstaT2.EXTR_TABLE()['NOEUD', 'DX', 'DY', 'DZ']
    
    FieldCoordX = np.array(ForceNodeTable.values()['DX'][:])
    FieldCoordY = np.array(ForceNodeTable.values()['DY'][:])
    FieldCoordZ = np.array(ForceNodeTable.values()['DZ'][:])
    
    DictStructParam = J.get(t, '.StructuralParameters')
        
    VectFnl = np.zeros((DictStructParam['MeshProperties']['Nddl'][0]))
    VectFnl[::3] = FieldCoordX
    VectFnl[1::3] = FieldCoordY
    VectFnl[2::3] = FieldCoordZ

    DETRUIRE(CONCEPT = _F(NOM = (F_noda,tstaT2)))

    return VectFnl
    
    
def ComputeStaticU4GivenLoading(t, RPM, LoadVector, **kwargs):
        

    DictStructParam = J.get(t, '.StructuralParameters')
    DictSimulaParam = J.get(t, '.SimulationParameters')

    ListeLoading = SJ.TranslateNumpyLoadingVector2AsterList(t, LoadVector)
    

    RAMPE = DEFI_FONCTION(NOM_PARA = 'INST',
                          VALE = (0.0,0.0,1.0,1.0),
                          PROL_DROITE = 'CONSTANT',
                          PROL_GAUCHE = 'CONSTANT',
                          INTERPOL = 'LIN'
                          );


    Cfd = AFFE_CHAR_MECA(MODELE = kwargs['MODELE'],
                         ROTATION = _F(VITESSE = RPM, 
                                       AXE = DictStructParam['RotatingProperties']['AxeRotation'] , 
                                       CENTRE = DictStructParam['RotatingProperties']['RotationCenter'],),
                         DDL_IMPO = _F(GROUP_NO='Node_Encastrement',
                                           DX=0.0,
                                           DY=0.0,
                                           DZ=0.0,),
                         FORCE_NODALE =  ListeLoading,
                         );


    L_INST = DEFI_LIST_REEL(DEBUT = 0.0,
                            INTERVALLE = (_F(JUSQU_A = 1.0,
                                             NOMBRE = DictSimulaParam['IntegrationProperties']['StaticSteps'][0],),
                                           ),
                            );


    SOLU = STAT_NON_LINE(MODELE = kwargs['MODELE'],
                         CHAM_MATER = kwargs['CHMAT'],
                         EXCIT =( _F( CHARGE = Cfd,
                                      FONC_MULT=RAMPE,),), 
                         COMPORTEMENT = _F(RELATION = 'ELAS',
                                           DEFORMATION =DictStructParam['MaterialProperties']['TyDef'], 
                                           TOUT = 'OUI',
                                          ),
                         CONVERGENCE=_F(RESI_GLOB_MAXI=2e-6,
                                        RESI_GLOB_RELA=1e-4,
                                        ITER_GLOB_MAXI=1000,
                                        ARRET = 'OUI',),
                         INCREMENT = _F( LIST_INST = L_INST,
                                        ),
                         INFO = 1,               
                        ),

    AsterObjs = SJ.merge_dicts(kwargs, dict(Cfd= Cfd, SOLU = SOLU[0], RAMPE = RAMPE, L_INST = L_INST))


    UpFromOmegaAndFe = ExtrUpFromAsterSOLUwithOmegaFe(t, RPM, **dict(SOLU = SOLU))
    
    GusFromOmegaAnfFe = ComputeStaFullFnl(t, **AsterObjs)
    

    SJ.DestroyAsterObjects(dict(**dict(Cfd = Cfd, SOLU= SOLU, RAMPE = RAMPE, L_INST = L_INST)),  
                           DetrVars = ['Cfd', 'SOLU', 'RAMPE', 'L_INST',
                                      ])

    return  UpFromOmegaAndFe, GusFromOmegaAnfFe
