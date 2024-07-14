#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################

FAIL  = '\033[91m'
GREEN = '\033[92m'
WARN  = '\033[93m'
MAGE  = '\033[95m'
CYAN  = '\033[96m'
ENDC  = '\033[0m'

# System modules
import numpy as np
import scipy
from scipy.sparse import csr_matrix
from scipy import linalg
from itertools import combinations_with_replacement

from code_aster.Cata.Syntax import _F
from code_aster.Cata.Language.DataStructure import CO
from code_aster.Cata.Commands import ASSEMBLAGE
import MOLA.Structure.DirectNormalForm as DNF

try:
    #Code Aster:  
    from code_aster.Commands import *

    #from code_aster.Cata.DataStructure import *
    #from code_aster.Cata.Language import *
    from code_aster.MacroCommands.Utils import partition
except:
    print(WARN + 'Code_Aster modules are not loaded!!' + ENDC)


def defineModelsAndMaterials(DictStructParam):

    print(CYAN + 'Loading mesh...'+ENDC)
    MAIL = LIRE_MAILLAGE(UNITE = 20,
                         FORMAT = 'MED',
                         );


    # Affect the mecanical model to all the mesh:
    print(CYAN + 'Loading FEMmodels...'+ENDC)


    MODELE = DefineFEMModels(MAIL, DictStructParam['MeshProperties']['Models'])

    # Define the materials and affect them to their meshes:
    print(CYAN + 'Loading materials...'+ENDC)
    CHMAT, MAT = DefineMaterials(MAIL, DictStructParam['MaterialProperties'])

    DictStructParam['MeshProperties']['Caracteristics'] = None #TODO: Adapt for beams

    # Define the elemental characteristics if needed:
    print(CYAN + 'Loading element caracteristics...'+ENDC)
    CARELEM = DefineElementalCaracteristics(MAIL, MODELE, DictStructParam['MeshProperties']['Caracteristics'])

    print(MAIL.getName())
    # Modify the cgns in order to erase the SOLID node and to create the Mesh Node
    print(CYAN + 'Extract the elements and connectivity the initial cgns...'+ENDC)


    DictStructParam = UpdateDictStructParamWithMeshData(DictStructParam, MAIL.getName())

    AsterObjs = dict(MAIL    = MAIL,
                     MODELE  = MODELE,
                     MAT     = MAT,
                     CHMAT   = CHMAT,
                     CARELEM = CARELEM)

    return AsterObjs, DictStructParam

def merge_dicts(a, b):
    m = a.copy()
    m.update(b)
    return m

def DestroyAsterObjects(AsterObjectsDictionnary, DetrVars = []):
    '''Function to erase the aster objects from the memory Jeveux'''

    for AsName in AsterObjectsDictionnary.keys():
        if (AsName in DetrVars) and (AsterObjectsDictionnary[AsName] is not None):
            print(FAIL+'Destroying %s...'%AsName+ENDC)
            DETRUIRE(CONCEPT = _F(NOM = AsterObjectsDictionnary[AsName]))



def DefineFEMModels(Mesh, ModelDict):
    '''Affect several Materials defined in MaterialDict to the corresponding meshes
    Inputs : 
        Mesh      : mesh obtained from LIRE_MAILLAGE(UNITE = 20, FORMAT = 'MED')
        ModelDict : corresponds to DictStructParam['MeshProperties']['Models'], see the DictMeshProperties from the setup
                    ex with only one type: DictMeshProperties = dict(Models = dict(Solid = dict(Phenomene    = 'MECANIQUE',
                                                                                                Modelling    = '3D', 
                                                                                                MeshGroup    = 'All', 
                                                                                                ddlElem      = 3,
                                                                                                BehaviourLaw = 'ELAS',
                                                                                                Strains      = 'Green-Lagrange'),
                                                                                                 ), 
                                                                                   Caracteristics = {}
                                                                                   ) 
    Outputs:
        MODELE : Aster object resulting from AFFE_MODELE
    '''

    l_affe = []
    for LocModelName in ModelDict.keys():
        if ModelDict[LocModelName]['MeshGroup'] == 'All':
            ap = _F(TOUT = 'OUI', PHENOMENE = ModelDict[LocModelName]['Phenomene'], MODELISATION = ModelDict[LocModelName]['Modelling'])

        else:
            ap = _F(PHENOMENE = ModelDict[LocModelName]['Phenomene'], MODELISATION = ModelDict[LocModelName]['Modelling'], GROUP_MA = ModelDict[LocModelName]['MeshGroup'])
        l_affe.append(ap)

        print(GREEN+'Affecting %s model to %s mesh groups.'%(LocModelName, ModelDict[LocModelName]['MeshGroup'])+ENDC)


    MODELE = AFFE_MODELE(MAILLAGE = Mesh,
                         AFFE = l_affe)

    return MODELE




def DefineMaterials(Mesh, DictMaterialProperties):
    '''Affect the material properties to the mesh
        Inputs : 
            Mesh                   : mesh obtained from LIRE_MAILLAGE(UNITE = 20, FORMAT = 'MED')
            DictMaterialProperties : corresponds to DictStructParam['MaterialProperties'], see the DictMaterialProperties from the setup
                                     ex with only one type: DictMaterialProperties = dict(Titane = dict(Rho           = 4500., 
                                                                                                        E             = 1.*110e9,
                                                                                                        PoissonRatio  = 0.318,
                                                                                                        ElasticStrain = 850e6,    # not used in finite deformations, small strains, large displacements large rotations                                     
                                                                                                        XiBeta        = 5e-3,
                                                                                                        XiAlpha       = 0, 
                                                                                                        Freq4Dumping  = 3.38,    # Hz
                                                                                                        MeshGroup     = 'All', 
                                                                                                        ),                  
                                     )
        Outputs:
            CHMAT : Aster object resulting from AFFE_MATERIAU, see the AffectMaterialFromMaterialDictionary 
            Ms    : list for all different material properties of the Aster definitions of materials DEFI_MATERIAU
        '''


    MaterialDict = {}
    Ms = []
    M = [None]

    for LocMat, it in zip(DictMaterialProperties.keys(), range(len(DictMaterialProperties.keys()))):
        MaterialDict[LocMat] = {}
        M.append([None])

        M[it] = DEFI_MATERIAU(ELAS=_F(E          = DictMaterialProperties[LocMat]['E'],
                                      NU         = DictMaterialProperties[LocMat]['PoissonRatio'],
                                      RHO        = DictMaterialProperties[LocMat]['Rho'],
                                      AMOR_ALPHA = DictMaterialProperties[LocMat]['XiAlpha'],
                                      AMOR_BETA  =  4*np.pi*DictMaterialProperties[LocMat]['Freq4Dumping']*DictMaterialProperties[LocMat]['XiBeta'],
                                      ),);

        Ms.append(M[it])
        MaterialDict[LocMat]['Properties'] = M[it]

        MaterialDict[LocMat]['Mesh'] = DictMaterialProperties[LocMat]['MeshGroup']

        #DETRUIRE(CONCEPT = _F(NOM = MAT))


    CHMAT = AffectMaterialFromMaterialDictionary(MaterialDict, Mesh)


    return CHMAT, Ms



def AffectMaterialFromMaterialDictionary(MaterialDict, Mesh):
    '''Affect several Materials defined in MaterialDict to the corresponding meshes
    Inputs :
        MaterialDict : dict of the material properties of obtained with DEFI_MATERIAU and the concerned meshgroups, see function DefineMaterials
        Mesh         : mesh obtained from LIRE_MAILLAGE(UNITE = 20, FORMAT = 'MED')
    Outputs : 
        CHMAT        : Aster object resulting from AFFE_MATERIAU to the mesh and the associated material properties
    '''

    l_affe = []
    for LocMatName in MaterialDict.keys():
        if MaterialDict[LocMatName]['Mesh'] == 'All':
            ap = _F(TOUT = 'OUI', MATER = MaterialDict[LocMatName]['Properties'])

        else:
            ap = _F(GROUP_MA = MaterialDict[LocMatName]['Mesh'] , MATER = MaterialDict[LocMatName]['Properties'])
        l_affe.append(ap)

        print(GREEN+'Affecting %s material to %s mesh groups.'%(LocMatName, MaterialDict[LocMatName]['Mesh'])+ENDC)

    CHMAT = AFFE_MATERIAU(MAILLAGE=Mesh,
                          AFFE= l_affe,);

    return CHMAT



def DefineElementalCaracteristics(Mesh, Model, DictCaracteristics):
    '''Affect the material caracteristics to the elements, 'POUTRE', 'BARRE' and 'DISCRET', return None else
        Inputs : 
            Mesh               : mesh obtained from LIRE_MAILLAGE(UNITE = 20, FORMAT = 'MED')
            Model              : Aster object MODELE resulting from AFFE_MODELE
            DictCaracteristics : corresponds to DictStructParam['MeshProperties']['Caracteristics'], can be {}, see the DictMeshProperties from the setup
                                     
        Outputs:
            CARELEM : Aster object resulting from  AFFE_CARA_ELEM(MODELE  = Model,
                                                                  BARRE   = affe_Bar,
                                                                  POUTRE  = affe_Poutre,
                                                                  DISCRET = affe_Discret)
                      None for other types of elements, for instance volumic elements
        '''

    affe_Poutre = []
    affe_Bar = []
    affe_Discret = []

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




        CARELEM = AFFE_CARA_ELEM(MODELE  = Model,
                                 BARRE   = affe_Bar,
                                 POUTRE  = affe_Poutre,
                                 DISCRET = affe_Discret)
    else:
        print(WARN+'Warning! No caracteristics are affected to the Mesh %s'%Mesh+ENDC)
        CARELEM = None

    return CARELEM





def UpdateDictStructParamWithMeshData(DictStructParam, MeshName):
    '''Only one type of mesh element! or with same number of ddl,
       Read the mesh and add informations about the number of nodes, the families and the dictionnary of elements

    Inputs:
        DictStructParam : dictionnary of the structural parameters
                          adding the keys 'Nnodes', 'NodesFamilies', 'MeshFamilies' to DictStructParam['MeshProperties']
                          and the dictionnary of the elements DictElements
        mm              : aster object mm = partition.MAIL_PY()

    Outputs :
        DictStructParam
    '''

    mm = partition.MAIL_PY()
    mm.FromAster(MeshName)
    
    #mm.FromAster('MAIL')

    print(WARN+'Only one type of mesh element! or with same number of ddl'+ENDC)
    
    DictStructParam['MeshProperties']['NNodes'] = len(list(mm.correspondance_noeuds)) 
    #DictStructParam['MeshProperties']['NNodes'] = MAIL.getNumberOfNodes
    
    DictStructParam['MeshProperties']['NodesFamilies'] = {}
    for Family in mm.gno: # Families
        DictStructParam['MeshProperties']['NodesFamilies'][Family] = mm.gno[Family]

    DictStructParam['MeshProperties']['MeshFamilies'] = {}
    if mm.gma: # Families names  
        for NameFamily in mm.gma.keys(): 
            DictStructParam['MeshProperties']['MeshFamilies'][NameFamily]            = {}
            DictStructParam['MeshProperties']['MeshFamilies'][NameFamily]['Element'] = np.array([x + 1 for x in mm.gma[NameFamily]])
    else:
        print(WARN + 'The mesh does not have any associated MeshFamilies, we consider All the elements are the same. Create automatic groups depending on the mesh elements (HEXA8, SEG2...).'+ENDC)
        for element in range(len(mm.tm)):
            ElType = mm.nom[mm.tm[element]]
            try:
                DictStructParam['MeshProperties']['MeshFamilies'][ElType]['Element'].append(int(mm.correspondance_mailles[element][1:]))
            except:
                DictStructParam['MeshProperties']['MeshFamilies'][ElType] = {}
                DictStructParam['MeshProperties']['MeshFamilies'][ElType]['Element'] = [int(mm.correspondance_mailles[element][1:])]


    # Build the dictionnary of Elements
    DictElements = {}
    DictElements['GroupOfElements'] = {}

    Nelem = 0
    for element in range(len(mm.tm)):
        Nelem += 1
        ElemName = mm.correspondance_mailles[element].split(' ')[0]
        DictElements[ElemName]                        = {}
        DictElements[ElemName]['AsterType']           = mm.nom[mm.tm[element]]
        DictElements[ElemName]['CGNSType'],_, CellDim = CGNSElementType(DictElements[ElemName]['AsterType'])
        DictElements[ElemName]['AsterConnectivity']   = mm.co[element] + 1
        #print(DictElements)
        try:
            DictElements['GroupOfElements'][DictElements[ElemName]['AsterType']]['ListOfElem'].append(int(mm.correspondance_mailles[element][1:]))
        except:
            DictElements['GroupOfElements'][DictElements[ElemName]['AsterType']]                  = {}
            DictElements['GroupOfElements'][DictElements[ElemName]['AsterType']]['ListOfElem']    = [int(mm.correspondance_mailles[element][1:])]
            print(FAIL+'eltype: %s'%DictElements[ElemName]['CGNSType']+ENDC)
            DictElements['GroupOfElements'][DictElements[ElemName]['AsterType']]['CGNSType']      = DictElements[ElemName]['CGNSType']
            DictElements['GroupOfElements'][DictElements[ElemName]['AsterType']]['CellDimension'] = CellDim
        DictElements['NbOfElementType'] = len(DictElements['GroupOfElements'].keys())
    
    
    for ElemName in DictElements['GroupOfElements']:
        CoordinatesE, NodesElemType  = ExtractCoordinates(mm, DictElements, ElemName)
        
        DictElements['GroupOfElements'][ElemName]['Coordinates']   = CoordinatesE
        DictElements['GroupOfElements'][ElemName]['NodesPosition'] = NodesElemType
        DictElements['GroupOfElements'][ElemName]['Nodes']         = np.array(NodesElemType) + 1
        ConnectivityE                                              = ExtractConnectivity(mm, DictElements, ElemName)
        DictElements['GroupOfElements'][ElemName]['Connectivity']  = ConnectivityE
            
    DictStructParam['MeshProperties']['DictElements']                    = {}
    DictStructParam['MeshProperties']['DictElements']['GroupOfElements'] = DictElements['GroupOfElements']

    return DictStructParam




def numpy_fillna(data):   
    """ data is a list of arrays with different lengths, create another list of arrays with same lengths 
    and fills them with the values of data and completes with -1 the other places (because no node has number -1) """
    
    # Get lengths of each row of data
    lens = np.array([len(i) for i in data])
    # Mask of valid places in each row
    mask = np.arange(lens.max()) < lens[:,None]
    # Setup output array and put elements from data into masked positions
    out = -1*np.ones(mask.shape, dtype=data.dtype)  
    out[mask] = np.concatenate(data)

    return out



def ExtractCoordinates(mm, DictElements, ElemType):
    """
    Extract the coordinates of the elements from the aster object mm = partition.MAIL_PY()
    """
    Coordinates = mm.cn
    if len(Coordinates[0,:]) == 2:
        Coordinates = np.append(Coordinates, np.zeros((len(Coordinates[:,0]),1)), axis = 1)   

    ValidNodes = []
    
    LinealConnectivity = numpy_fillna(np.array(mm.co))[np.array(DictElements['GroupOfElements'][ElemType]['ListOfElem'])-1,:].ravel()

    seen = set()
    ValidNodes = []
    for x in LinealConnectivity:
        if (x not in seen) and (x != -1):
            ValidNodes.append(x)
            seen.add(x)

    return np.array(Coordinates[np.sort(ValidNodes)]), np.sort(ValidNodes)




def ExtractConnectivity(mm, DictElements, ElemType):
    """
    Extract the connectivity of the elements from the aster object mm = partition.MAIL_PY()
    Works for elements of the types 'HEXA8', and a special treatment for elements 'POI1' and 'HEXA20', adaptations are maybe required for other types  
    """

    Connectivity   = mm.co
    Connectivity   = [val + 1 for val in Connectivity]
    _, NodeElem,_ = CGNSElementType(ElemType)
    ConnectMatrix = np.zeros((len(DictElements['GroupOfElements'][ElemType]['ListOfElem']), NodeElem))

    for Element, pos in zip(DictElements['GroupOfElements'][ElemType]['ListOfElem'], range(len(DictElements['GroupOfElements'][ElemType]['ListOfElem']))):

        if ElemType == 'POI1':
            #ListPos               = range(len(DictElements['GroupOfElements'][ElemType]['ListOfElem']))
            #ListElement           = DictElements['GroupOfElements'][ElemType]['ListOfElem']
            SortedElNodes         = np.sort(DictElements['GroupOfElements'][ElemType]['Nodes'])
            ConnectMatrix[pos, :] = list(SortedElNodes).index(Connectivity[pos])+1
            ConnectMatrix         = ConnectMatrix.astype(int)
            ConnectMatrix[pos,:]  = DictElements['M%s'%Element]['AsterConnectivity']

        else:
            ConnectMatrix[pos,:] = DictElements['M%s'%Element]['AsterConnectivity']
            ConnectMatrix = ConnectMatrix.astype(int)

    if ElemType == 'HEXA20':
        print(GREEN+'Adapting the order of connectivity for %s elements'%ElemType+ENDC)
        #SwiftList[:,17:19] = [int(x - 1) for x in  [3, 4, 1, 2, 7, 8, 5, 6, 11, 12, 9, 10, 15, 16, 13, 14,19,20,17,18] ]
        ConnectMatrix[:,[15,18]] = ConnectMatrix[:, [18,15]]
        ConnectMatrix[:,[-1,12]] = ConnectMatrix[:, [12,-1]]
        ConnectMatrix[:,[14,17]] = ConnectMatrix[:, [17,14]]
        ConnectMatrix[:,[13,16]] = ConnectMatrix[:, [16,13]]

    return ConnectMatrix



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




def AffectImpoDDLByGroupType(DictStructParam, ImposedVector = None):
    """Affect les ddl de blocage en fonction du type de famille predefinie (encastrement, rotule battement...)"""
    affe_impo = []

    for gno in DictStructParam['MeshProperties']['NodesFamilies']:
        if gno == 'Node_Encastrement':
            affe_impo.append(_F(GROUP_NO = gno, DX = 0.0, DY = 0.0, DZ = 0.0))

        if (gno  == 'Rotule_Battement') or (gno  == 'Rotule_Batement'):
            affe_impo.append(_F(GROUP_NO = gno, DX = 0.0, DY = 0.0, DZ = 0.0, DRX = 0.0, DRZ = 0.0))


        if ImposedVector is not None:
            print(len(ImposedVector))
            for n in range(len(ImposedVector)//3):
                if n in DictStructParam['MeshProperties']['NodesFamilies']['Imposed_Displacements']:
                    ap1 = _F(NOEUD = 'N'+ str(n + 1),DX = ImposedVector[n*3],DY = ImposedVector[n*3+ 1], DZ = ImposedVector[n*3 + 2],)
        
                    affe_impo.append(ap1)

    return affe_impo



def DefineBehaviourLaws(ModelDict):
    '''Affect several BehaviourLaws defined in ModelsDict to the corresponding meshes
    Input : 
        ModelDict : DictStructParam['MeshProperties']['Models']

    Output : 
        l_Behaviour : list of the aster behaviour law for all models defined in ModelDict
    '''

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





def computePrestress(RPM, DictStructParam, DictSimulaParam, **kwargs):
    '''
    Computes the FOM matrices (Ke, Kg, Kc, C, M) for a given velocity RPM.
    Inputs:
        RPM             : rotation speed in rounds per min (rpm) 
        DictStructParam : dictionnary of the structural parameters containing:
                             DictStructParam['MeshProperties']['Models']
        DictSimulaParam : dictionnary of the simulation parameters containing:
                            DictSimulaParam['RotatingProperties']['AxeRotation']
                            DictSimulaParam['RotatingProperties']['RotationCenter']
                            DictSimulaParam['IntegrationProperties']['Steps4CentrifugalForce']
                            DictSimulaParam['IntegrationProperties']['EpsConvergenceCriteria']
                            DictSimulaParam['IntegrationProperties']['NumberOfMaxIterations']
        **kwargs        : dictionnary ** AsterObjs, including especially the aster objects MODELE, CHMAT, CARELEM

    Outputs:
        AsterObjs       : aster object dictionnary in which are added Cfd, SOLU[0], RAMPE, L_INST, Fei, Komeg2, MASS1, NUME
        DictMatrices    : dictionnary of the matrices Ke, Kc, Komeg, C, M
        DictVectors     : dictionnary of the vector Fei of component of the centrifugal forces independant on the position, and Us the prestressed position due to the centrifugal rotation
        DictStructParam : dictionnary of the structural parameters in which is added:
                            DictStructParam['MeshProperties']['Nddl']
                            DictStructParam['MeshProperties']['NodesFamilies']['LagrangeNodes']
    '''

    Cfd = AFFE_CHAR_MECA(MODELE   = kwargs['MODELE'],
                         ROTATION = _F(VITESSE = np.pi * RPM/30.,  # in rad/s
                                       AXE     = DictSimulaParam['RotatingProperties']['AxeRotation'],
                                       CENTRE  = DictSimulaParam['RotatingProperties']['RotationCenter'],),
                         DDL_IMPO =(AffectImpoDDLByGroupType(DictStructParam)
                                        ),
                                )


    RAMPE = DEFI_FONCTION(NOM_PARA    = 'INST',
                          VALE        = (0.0,0.0,1.0,1.0),
                          PROL_DROITE = 'CONSTANT',
                          PROL_GAUCHE = 'CONSTANT',
                          INTERPOL    = 'LIN'
                          );


    L_INST = DEFI_LIST_REEL(DEBUT = 0.0,
                            INTERVALLE = (_F(JUSQU_A = 1.0,
                                             NOMBRE = DictSimulaParam['IntegrationProperties']['Steps4CentrifugalForce'],),
                                          ),
                                  );
                               
    # Newton without acceleration, default
    SOLU = STAT_NON_LINE(MODELE       = kwargs['MODELE'],
                         CHAM_MATER   = kwargs['CHMAT'],
                         CARA_ELEM    = kwargs['CARELEM'],
                         EXCIT        =( _F(CHARGE = Cfd,
                                            FONC_MULT=RAMPE,),
                                       ),
                         COMPORTEMENT = DefineBehaviourLaws(DictStructParam['MeshProperties']['Models']),
                         CONVERGENCE  =_F(RESI_GLOB_MAXI=DictSimulaParam['IntegrationProperties']['EpsConvergenceCriteria'],
                                        RESI_GLOB_RELA=1e-6,   # before 1e-4
                                        ITER_GLOB_MAXI=DictSimulaParam['IntegrationProperties']['NumberOfMaxIterations'],
                                        ARRET = 'OUI',),
                         INCREMENT    = _F( LIST_INST = L_INST,
                                           ),
                         INFO         = 1,               
                        ),
    """
    # GCPC LDLT_SP
    SOLU = STAT_NON_LINE(MODELE       = kwargs['MODELE'],
                         CHAM_MATER   = kwargs['CHMAT'],
                         CARA_ELEM    = kwargs['CARELEM'],
                         EXCIT        =( _F(CHARGE = Cfd,
                                            FONC_MULT=RAMPE,),
                                       ),
                        SOLVEUR       =_F(METHODE      ='GCPC',
                                          #ALGORITHME  ='CG',
                                          #SYME        ='OUI',
                                          PRE_COND     ='LDLT_SP',
                                          RENUM        ='SANS',
                                          RESI_RELA    = 1.0E-6,
                                          REAC_PRECOND =30,),
                         COMPORTEMENT = DefineBehaviourLaws(DictStructParam['MeshProperties']['Models']),
                         CONVERGENCE  =_F(RESI_GLOB_MAXI=DictSimulaParam['IntegrationProperties']['EpsConvergenceCriteria'],
                                          RESI_GLOB_RELA=1e-6,   # before 1e-4
                                          ITER_GLOB_MAXI=DictSimulaParam['IntegrationProperties']['NumberOfMaxIterations'],
                                          ARRET         = 'OUI',),
                         INCREMENT    = _F( LIST_INST = L_INST,
                                          ),
                         INFO         = 1,               
                        ),


    # Newton with not always actualisation of the tangent matrix
    
    SOLU = STAT_NON_LINE(MODELE       = kwargs['MODELE'],
                         CHAM_MATER   = kwargs['CHMAT'],
                         CARA_ELEM    = kwargs['CARELEM'],
                         EXCIT        =( _F(CHARGE = Cfd,
                                            FONC_MULT=RAMPE,),
                                       ), 
                         METHODE      = 'NEWTON', 
                         NEWTON       = _F(REAC_INCR = 10, 
                                           REAC_ITER = 3),
                         COMPORTEMENT = DefineBehaviourLaws(DictStructParam['MeshProperties']['Models']),
                         CONVERGENCE  =_F(RESI_GLOB_MAXI=DictSimulaParam['IntegrationProperties']['EpsConvergenceCriteria'],
                                          RESI_GLOB_RELA=1e-6,   # before 1e-4
                                          ITER_GLOB_MAXI=DictSimulaParam['IntegrationProperties']['NumberOfMaxIterations'],
                                          ARRET = 'OUI',),
                         INCREMENT    = _F( LIST_INST = L_INST,
                                           ),
                         INFO         = 1,               
                        ),"""
                        
    """
    SOLU = STAT_NON_LINE(MODELE       = kwargs['MODELE'],
                         CHAM_MATER   = kwargs['CHMAT'],
                         CARA_ELEM    = kwargs['CARELEM'],
                         EXCIT        =( _F(CHARGE = Cfd,
                                            FONC_MULT=RAMPE,),),
                         METHODE      = 'NEWTON', 
                         NEWTON       = _F(REAC_INCR = 10, 
                                           REAC_ITER = 3),
                         SOLVEUR      = _F(METHODE = 'MUMPS', 
                                           ACCELERATION = 'LR',
                                           LOW_RANK_SEUIL = 1e-9,
                                           POSTTRAITEMENTS = 'MINI',
                                           #MIXER_PRECISION = 'OUI',
                                           #FILTRAGE_MATRICE = 0.8,    
                                           ),
                         COMPORTEMENT = DefineBehaviourLaws(DictStructParam['MeshProperties']['Models']),
                         CONVERGENCE  =_F(RESI_GLOB_MAXI=DictSimulaParam['IntegrationProperties']['EpsConvergenceCriteria'],
                                          RESI_GLOB_RELA=1e-6,   # before 1e-4
                                          ITER_GLOB_MAXI=1000,
                                          ARRET = 'OUI',),
                         INCREMENT    = _F( LIST_INST = L_INST,
                                           ),
                         INFO         = 1,
                      ),"""


    # Add the model with matrices:
    AsterObjs = merge_dicts(kwargs, dict(Cfd= Cfd, SOLU = SOLU[0], RAMPE = RAMPE, L_INST = L_INST))

    #DictMatrices, Fei, PointsLagrange, AsterObjs = AsseMatricesFOM('All', **AsterObjs)
    
    ## Extract Us and update DictStructParam
    try:
        tstaT = CREA_TABLE(RESU = _F(RESULTAT = SOLU,
                                     NOM_CHAM = 'DEPL',
                                     NOM_CMP  = ('DX','DY','DZ', 'DRX', 'DRY', 'DRZ'),
                                     INST     = 1.0,   # we chose 1.0 for the last instant of the static solution
                                     TOUT     = 'OUI',),
                           TYPE_TABLE = 'TABLE',
                           TITRE      = 'Table_Depl_R',
                           )
    except:
        tstaT = CREA_TABLE(RESU = _F(RESULTAT = SOLU,
                                     NOM_CHAM = 'DEPL',
                                     NOM_CMP  = ('DX','DY','DZ'),
                                     INST     = 1.0,
                                     TOUT     = 'OUI',),
                           TYPE_TABLE = 'TABLE',
                           TITRE      = 'Table_Depl_R',
                           )
        print(WARN+'No rotation dof  (DRX, DRY, DRZ) in the model. Computing only with displacements (DX, DY, DZ).'+ENDC)

    # extract Us
    NNodes = DictStructParam['MeshProperties']['NNodes']
    Us = VectFromAsterTable2Full(tstaT, NNodes)

    # update DictStructParam
    DictStructParam = ComputeTransformationLists(DictStructParam, tstaT)

    DETRUIRE (CONCEPT = _F (NOM = (tstaT),
                            ),
              INFO = 1,
              )
    ################

    
    DictVectors        = {}
    DictVectors['Us']  = Us

    DictVectors['Gus'] = ComputeStaFullNodalF(NNodes, **AsterObjs)

    #AsterObjs = merge_dicts(AsterObjs, dict(Fei= DictVectors['Fei']))

    return AsterObjs, DictVectors, DictStructParam




def AsseMatricesFOM(Type_asse, **kwargs):
    """
    Computes the matrices of the structure
    Inputs:
        Type_asse : 'All', 'Kec' or 'Keg', used in the aster key COEF_R in COMB_MATR_ASSE 
        **kwargs  : dictionnary **AsterObjs containing especially the aster objects SOLU, MODELE, CHMAT, CARELEM, Cfd

    Outputs:
        DictMatrices   : dictionnary of the matrices Ke, Kc, Komeg, C, M
        Fei            : vector of the centrifugal efforts Fei independant of u
        PointsLagrange : lagrange degrees of freedom
        AsterObjs      : aster object AsterObjs of the input, in which are added the keys Komeg2, MASS1, NUME
    """

    sig_g = CREA_CHAMP(TYPE_CHAM = 'ELGA_SIEF_R',
                       OPERATION = 'EXTR',
                       RESULTAT  = kwargs['SOLU'],
                       NOM_CHAM  = 'SIEF_ELGA',
                       INST      = 1.0)

    
    try:
        #RIGI1 = CO('RIGI1')
        #MASS1 = CO('MASS1')
        #C1    = CO('C1')
        #KASOU = CO('KASOU')
        #KGEO  = CO('KGEO')
        #FE1   = CO('FE')
        #NUME  = CO('NUME')

        ASSEMBLAGE(MODELE     = kwargs['MODELE'],
                   CHAM_MATER = kwargs['CHMAT'],
                   CARA_ELEM  = kwargs['CARELEM'],
                   CHARGE     = kwargs['Cfd'],
                   NUME_DDL   = CO('NUME'),
                   MATR_ASSE  = (_F(MATRICE = CO('RIGI1'),
                                    OPTION  = 'RIGI_MECA',),
                                 _F(MATRICE = CO('MASS1'),
                                    OPTION  = 'MASS_MECA',),
                                 _F(MATRICE   = CO('KGEO'),
                                    OPTION    = 'RIGI_GEOM',
                                    SIEF_ELGA = sig_g),
                                 _F(MATRICE = CO('KASOU'),
                                    OPTION  = 'RIGI_ROTA',),
                                 _F(MATRICE = CO('C1'),
                                    OPTION  = 'AMOR_MECA'),
                                  ),
                   VECT_ASSE  =(_F(VECTEUR = CO('FEI'),
                                   OPTION  = 'CHAR_MECA',
                                   ),
                               ),
                   INFO       = 2
                   )

        # Loading coefficients:
        #print(RIGI1)
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

        Komeg2 = COMB_MATR_ASSE( COMB_R = (_F(MATR_ASSE = RIGI1,
                                              COEF_R    = 1.),
                                           _F(MATR_ASSE = KGEO,
                                              COEF_R    = C_g),
                                           _F(MATR_ASSE = KASOU,
                                              COEF_R    = C_c),),)
    except:
        print(FAIL+'WARNING! The matrix Kc was not computed! Only valid if Omega = 0 rpm'+ENDC)

        DETRUIRE (CONCEPT = _F (NOM = (RIGI1,MASS1, KGEO, KASOU, C1, FE1, NUME),
                            ),
              INFO = 1,
              )

        RIGI1 = CO('RIGI1')
        MASS1 = CO('MASS1')
        C1    = CO('C1')
        KASOU = None
        KGEO  = CO('KGEO')
        FE1   = CO('FE')
        NUME  = CO('NUME')

        ASSEMBLAGE(MODELE     = kwargs['MODELE'],
                   CHAM_MATER = kwargs['CHMAT'],
                   CARA_ELEM  = kwargs['CARELEM'],
                   CHARGE     = kwargs['Cfd'],
                   NUME_DDL   = NUME,
                   MATR_ASSE  = (_F(MATRICE = RIGI1,
                                    OPTION  = 'RIGI_MECA',),
                                 _F(MATRICE = MASS1,
                                    OPTION  = 'MASS_MECA',),
                                 _F(MATRICE   = KGEO,
                                    OPTION    = 'RIGI_GEOM',
                                    SIEF_ELGA = sig_g),
                                 _F(MATRICE = C1,
                                    OPTION  = 'AMOR_MECA'),
                                  ),
                   VECT_ASSE  =(_F(VECTEUR = FEI,
                                   OPTION  = 'CHAR_MECA',
                                   ),
                               ),
                   INFO       = 2
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

        Komeg2 = COMB_MATR_ASSE( COMB_R = (_F(MATR_ASSE = RIGI1,
                                              COEF_R    = 1.),
                                           _F(MATR_ASSE = KGEO,
                                              COEF_R    = C_g),
                                           ),)

    DictMatrices = {}
    
    AsterObjs    =  merge_dicts(kwargs , dict(Komeg2 = Komeg2, MASS1 = MASS1, NUME = NUME))

    DictMatrices['Ke'], P_Lagr = ExtrMatrixFromAster2Python(RIGI1, ComputeLagrange = True)
    DictMatrices['Kg'],_       = ExtrMatrixFromAster2Python(KGEO, ii = P_Lagr)

    if KASOU is not None:
        DictMatrices['Kc'],_ = ExtrMatrixFromAster2Python(KASOU, ii = P_Lagr)

    DictMatrices['Komeg'],_ = ExtrMatrixFromAster2Python(Komeg2, ii = P_Lagr)
    DictMatrices['C'],_     = ExtrMatrixFromAster2Python(C1, ii = P_Lagr)
    DictMatrices['M'],_     = ExtrMatrixFromAster2Python(MASS1, ii = P_Lagr)
    Fei                     = ExtrVectorFromAster2Python(FEI, P_Lagr)

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

    return DictMatrices, Fei, P_Lagr, AsterObjs



def ExtrMatrixFromAster2Python(MATRICE, **kwargs):
    """
    Extraction of the aster matrix to a csr sparse matrix and remove the lagrange degrees of freedom
    Inputs:
        MATRICE  : aster matrix to extract, ex: RIGI1 (=Ke), KGEO (=Kg), KASOU (=Kc), C1 (=C), MASS1 (=M) and Komeg2
        **kwargs : dictionnary containing the keys 'ComputeLagrange' and 'ii'
    
    Ouputs : 
        MatricePy : csr matrix of the aster MATRICE
        ii        : kwargs['ii']
    """
    try:
        if kwargs['ComputeLagrange'] == True:
            P_Lagr = np.array(MATRICE.sdj.CONL.get())
            ii = np.where(P_Lagr != 1.0)[0]
    except:
        ii = kwargs['ii']

    SparseTupl = MATRICE.EXTR_MATR(sparse=True)
    SparseMatr = csr_matrix((SparseTupl[0], (SparseTupl[1], SparseTupl[2])), shape=(SparseTupl[3], SparseTupl[3]))
    MatricePy = delete_from_csr(SparseMatr,row_indices=ii, col_indices=ii)

    return MatricePy, ii


def ExtrVectorFromAster2Python(VECTEUR, PointsLagrange):
    """
    Extraction of the aster vector to a numpy array and remove the lagrange degrees of freedom
    Inputs:
        VECTEUR        : aster vector to extract, ex: FE1 (=Fei)
        PointsLagrange : lagrange degrees of freedom, given by ExtrMatrixFromAster2Python
    
    Ouputs : 
        VectPy : numpy array of the aster VECTEUR
    """
    VectPy = np.array(VECTEUR.sdj.VALE.get())
    VectPy = np.delete(VectPy, PointsLagrange)
    return VectPy


def delete_from_csr(mat, row_indices=None, col_indices=None):
    """
    Remove the rows (denoted by ``row_indices``) and columns (denoted by ``col_indices``) from the CSR sparse matrix ``mat``.
    WARNING: Indices of altered axes are reset in the returned matrix
    """
    if not isinstance(mat, csr_matrix):
        raise ValueError("works only for CSR format -- use .tocsr() first")

    rows = []
    cols = []
    if row_indices is not None:
        rows = list(row_indices)
    if col_indices is not None:
        cols = list(col_indices)

    if len(rows) > 0 and len(cols) > 0:
        row_mask = np.ones(mat.shape[0], dtype=bool)
        row_mask[rows] = False
        col_mask = np.ones(mat.shape[1], dtype=bool)
        col_mask[cols] = False
        return mat[row_mask][:,col_mask]
    elif len(rows) > 0:
        mask = np.ones(mat.shape[0], dtype=bool)
        mask[rows] = False
        return mat[mask]
    elif len(cols) > 0:
        mask = np.ones(mat.shape[1], dtype=bool)
        mask[cols] = False
        return mat[:,mask]
    else:
        return mat
    




def TranslateNumpyLoadingVector2AsterList(DictMeshProperties, LoadVector):
    """
    Prepares the List of loading at the dofs for aster computations
    Inputs:
        DictMeshProperties : DictStructParam['MeshProperties']
        LoadVector         : array of shape(Ndofs) of the load at each dof
    
    Output:
        l_prepa : List that will be used in the keyword FORCE_NODALE of the aster function AFFE_CHAR_MECA
    """

    try:
        n_enc =  DictMeshProperties['NodesFamilies']['Node_Encastrement']
    except:
        print(WARN+'Node_Encastrement not found!'+ENDC)
        n_enc = []

    l_prepa = []
    for n in range(DictMeshProperties['NNodes']):
        if not(n in n_enc):
            ap = _F(NOEUD = 'N'+ str(n + 1),FX = LoadVector[n*3], FY = LoadVector[n*3+ 1], FZ = LoadVector[n*3 + 2],)
            l_prepa.append(ap)

    return l_prepa



def ComputeTimeVector(DictSimulaParam):
    """
    Prepares the list of the time instants, both for the centrifugal rotation and the calculations
    Inputs:
        DictSimulaParam : dictionnary of the simulation parameters, containing the information:
                            DictSimulaParam['IntegrationProperties']['Steps4CentrifugalForce']
                            DictSimulaParam['IntegrationProperties']['SolverType']
                            DictSimulaParam['IntegrationProperties']['StaticSteps']             if static computation
                            DictSimulaParam['LoadingProperties']['TimeProperties']['Time_max']  if dynamic computation
                            DictSimulaParam['LoadingProperties']['TimeProperties']['dt']        if dynamic computation
    
    Outputs : 
        DictSimulaParam : dictionnary DictSimulaParam in which DictSimulaParam['LoadingProperties']['Time'] was added
        TimeList        : list concatenating L_rota and L_calc of the time instants
    """

    L_rota = list(np.linspace(-2., 0., DictSimulaParam['IntegrationProperties']['Steps4CentrifugalForce']))[:-1]  # for applying the centrifugal force
    # L_rota is the list L_rota = [-2., ... , 0. (excluded)] # with Steps4CentrifugalForce - 1 elements

    if DictSimulaParam['IntegrationProperties']['SolverType'] == 'Static':
        print(GREEN + 'Computing the time increments for the static analysis...'+ENDC)

        L_calc = list(np.linspace( 0., 1., DictSimulaParam['IntegrationProperties']['StaticSteps']))
        
        TimeList = L_rota + L_calc
    
    elif DictSimulaParam['IntegrationProperties']['SolverType'] == 'Dynamic':
    
        L_calc = list(np.arange( 0., DictSimulaParam['LoadingProperties']['TimeProperties']['Time_max'], DictSimulaParam['LoadingProperties']['TimeProperties']['dt']))
        TimeList = L_rota + L_calc

    #            [<------L_rota----------------->|<---------------L_calc------------------>]
    # TimeList = [..|..|..| ... |..|..           |..|..|..|..|..|..| ... |..|..|..|..|..  |]
    #            [-2|..|..| ... |..|0. (excluded)|0.|..|..|..|..|..| ... |..|..|..|..|tmax|]

    else:
        print(FAIL+'Unknown SolverType!'+ENDC)

    DictSimulaParam['LoadingProperties']['Time'] = TimeList

    return TimeList, DictSimulaParam  # TimeList is a list concatenating L_rota and L_calc (the + for lists is concatenating)



def ComputeStaticU4GivenLoading(RPM, DictStructParam, DictSimulaParam, LoadVector, LoadType = 'Forces', **kwargs):
    """
    Function computing the nonlinear static solution in rotation at RPM, under the external load LoadVector applied at the degrees of freedom
    Inputs : 
        RPM             : rotation speed in rounds per min (rpm) 
        DictStructParam : dictionnary of the structural parameters, containing especially:
                            DictStructParam['MeshProperties']
                            DictStructParam['MeshProperties']['NNodes']
                            DictStructParam['MeshProperties']['Models']
                            + the ones needed in AffectImpoDDLByGroupType(DictStructParam)
        DictSimulaParam : dictionnary of the simulation parameters, containing especially:
                            DictSimulaParam['RotatingProperties']['AxeRotation']
                            DictSimulaParam['RotatingProperties']['RotationCenter']
                            DictSimulaParam['IntegrationProperties']['EpsConvergenceCriteria']
                            DictSimulaParam['IntegrationProperties']['NumberOfMaxIterations']
                            + the ones needed in ComputeTimeVector(DictSimulaParam)
        LoadVector      : numpy array of the loads at all degrees of freedom
        **kwargs        : **AsterObj, containg especially MODELE, CHMAT, CARELEM
    
    Outputs :
        UpFromOmegaAndFe  : numpy array of the total displacement of all the degrees of freedom, up = us +u
        GusFromOmegaAnfFe : numpy array of the internal forces of all the degrees of freedom
        DictSimulaParam   :
        AsterObjs         : AsterObjs in which are added SOLU, RAMPE and L_INST
    """

    TimeList, DictSimulaParam = ComputeTimeVector(DictSimulaParam)

    RAMPE_r = DEFI_FONCTION(NOM_PARA    = 'INST',     # for applying the centrifugal loading
                            VALE        = (-2.,0.,0.,1.),
                            PROL_DROITE = 'CONSTANT',
                            PROL_GAUCHE = 'CONSTANT',
                            INTERPOL    = 'LIN'
                            );

    RAMPE = DEFI_FONCTION(NOM_PARA    = 'INST',
                          VALE        = (0.0,0.0,1.0,1.0),
                          PROL_DROITE = 'CONSTANT',
                          PROL_GAUCHE = 'CONSTANT',
                          INTERPOL    = 'LIN'
                          );

    if LoadType == 'Forces':
        ListeLoading = TranslateNumpyLoadingVector2AsterList(DictStructParam['MeshProperties'], LoadVector)
        F_ext = AFFE_CHAR_MECA(MODELE       = kwargs['MODELE'],
                               DDL_IMPO     = AffectImpoDDLByGroupType(DictStructParam),
                               FORCE_NODALE =  ListeLoading,
                               );
    elif LoadType == 'Displacements':
        F_ext = AFFE_CHAR_MECA(MODELE = kwargs['MODELE'],
    		                   DDL_IMPO = (AffectImpoDDLByGroupType(DictStructParam, ImposedVector = LoadVector)),	
    		                          ),

    F_rota = AFFE_CHAR_MECA(MODELE   = kwargs['MODELE'],
                            ROTATION = _F(VITESSE = np.pi * RPM/30.,
                                          AXE     = DictSimulaParam['RotatingProperties']['AxeRotation'] ,
                                          CENTRE  = DictSimulaParam['RotatingProperties']['RotationCenter'],),
                           )

    L_INST = DEFI_LIST_REEL(VALE = TimeList
                            );
    
    print('TimeList', TimeList)
    print('L_INST', L_INST)

    # without acceleration, default
    
    SOLU = STAT_NON_LINE(MODELE       = kwargs['MODELE'],
                         CHAM_MATER   = kwargs['CHMAT'],
                         CARA_ELEM    = kwargs['CARELEM'],
                         EXCIT        =( _F(CHARGE = F_ext,
                                            FONC_MULT=RAMPE,),
                                         _F(CHARGE = F_rota,
                                            FONC_MULT = RAMPE_r),
                                       ), 
                         COMPORTEMENT = DefineBehaviourLaws(DictStructParam['MeshProperties']['Models']),
                         CONVERGENCE  =_F(RESI_GLOB_MAXI=DictSimulaParam['IntegrationProperties']['EpsConvergenceCriteria'],
                                          RESI_GLOB_RELA=1e-6,  # before 1e-4
                                          ITER_GLOB_MAXI=DictSimulaParam['IntegrationProperties']['NumberOfMaxIterations'],
                                          ARRET = 'OUI',),
                         INCREMENT    = _F( LIST_INST = L_INST,
                                           ),
                         INFO         = 1,               
                        ), 

    """                    
    # GCPC with LDLT_SP
    SOLU = STAT_NON_LINE(MODELE       = kwargs['MODELE'],
                         CHAM_MATER   = kwargs['CHMAT'],
                         CARA_ELEM    = kwargs['CARELEM'],
                         EXCIT        =( _F(CHARGE = F_ext,
                                            FONC_MULT=RAMPE,),
                                         _F(CHARGE = F_rota,
                                            FONC_MULT = RAMPE_r),
                                       ),
                         SOLVEUR      =_F(METHODE='GCPC',
                                          #ALGORITHME='CG',
                                          #SYME='OUI',
                                          PRE_COND='LDLT_SP',
                                          RENUM='SANS',
                                          RESI_RELA=1.0E-6,
                                          REAC_PRECOND=30,),
                         COMPORTEMENT = DefineBehaviourLaws(DictStructParam['MeshProperties']['Models']),
                         CONVERGENCE  =_F(RESI_GLOB_MAXI=DictSimulaParam['IntegrationProperties']['EpsConvergenceCriteria'],
                                          RESI_GLOB_RELA=1e-6,
                                          ITER_GLOB_MAXI=DictSimulaParam['IntegrationProperties']['NumberOfMaxIterations'],
                                          ARRET = 'OUI',),
                         INCREMENT    = _F( LIST_INST = L_INST,
                                           ),
                         INFO         = 1,
                        ),


    # Newton with not always actualisation of the tangent matrix
    
    SOLU = STAT_NON_LINE(MODELE       = kwargs['MODELE'],
                         CHAM_MATER   = kwargs['CHMAT'],
                         CARA_ELEM    = kwargs['CARELEM'],
                         EXCIT        =( _F(CHARGE = F_ext,
                                            FONC_MULT=RAMPE,),
                                         _F(CHARGE = F_rota,
                                            FONC_MULT = RAMPE_r),
                                       ), 
                         METHODE      = 'NEWTON', 
                         NEWTON       = _F(REAC_INCR = 10,   # only for dynamics
                                           REAC_ITER = 2),
                         COMPORTEMENT = DefineBehaviourLaws(DictStructParam['MeshProperties']['Models']),
                         CONVERGENCE  =_F(RESI_GLOB_MAXI=DictSimulaParam['IntegrationProperties']['EpsConvergenceCriteria'],
                                          RESI_GLOB_RELA=1e-6,   # before 1e-4
                                          ITER_GLOB_MAXI=DictSimulaParam['IntegrationProperties']['NumberOfMaxIterations'],
                                          ARRET = 'OUI',),
                         INCREMENT    = _F( LIST_INST = L_INST,
                                           ),
                         INFO         = 1,               
                        ),"""

    AsterObjs = merge_dicts(kwargs, dict(SOLU = SOLU, RAMPE = RAMPE, L_INST = L_INST))

    NNodes = DictStructParam['MeshProperties']['NNodes']
    UpFromOmegaAndFe  = ExtrFromAsterSOLUwithOmegaFe(NNodes, **dict(SOLU = SOLU))
    GupFromOmegaAnfFe = ComputeStaFullNodalF(NNodes, **AsterObjs)

    DestroyAsterObjects(dict(**dict(RAMPE_r = RAMPE_r, F_rota = F_rota, F_ext = F_ext, SOLU= SOLU, RAMPE = RAMPE, L_INST = L_INST)),
                        DetrVars = ['RAMPE_r', 'F_rota','F_ext', 'SOLU', 'RAMPE', 'L_INST',
                                   ])

    return  UpFromOmegaAndFe, GupFromOmegaAnfFe, DictSimulaParam, AsterObjs




def ExtrFromAsterSOLUwithOmegaFe(NNodes, Instants = [1.0], ChampName = 'DEPL', **kwargs):
    """
    Extract the displacements and rotations from the aster solution, accessed via aster Tables
    Inputs :
        NNodes    : number of nodes, can be obtained by DictStructParam['MeshProperties']['NNodes']
        Instants  : list of the instants desired for the extraction, the default value is [1.0] because we chose the last instant at 1.0 for static computations
        ChampName : name for NOM_CHAM in CREA_TABLE, the default value is 'DEPL', it can also be 'VITE' or 'ACCE' if the functions is used for dynamic cases
        **kwargs  : dictionnary containing the aster object SOLU
    
    Output : 
        VectUpOmegaFe :  array of the total displacements (and rotations if there are) of the degrees of freedom
    """

    LIST = DEFI_LIST_REEL(VALE = Instants)
    try:  # if there are displacements DX, DY, DZ and rotations DRX, DRY, DRZ
        tstaT = CREA_TABLE(RESU = _F(RESULTAT= kwargs['SOLU'],
                                     NOM_CHAM= ChampName,
                                     NOM_CMP= ('DX','DY','DZ', 'DRX', 'DRY', 'DRZ'),
                                     LIST_INST = LIST,
                                     TOUT = 'OUI',),
                                   TYPE_TABLE='TABLE',
                                   TITRE='Table_Depl_R',
                                   )
    except: # if there are only displacements DX, DY, DZ
        tstaT = CREA_TABLE(RESU = _F(RESULTAT= kwargs['SOLU'],
                                             NOM_CHAM= ChampName,
                                             NOM_CMP= ('DX','DY','DZ'),
                                             LIST_INST = LIST,
                                             TOUT = 'OUI',),
                                   TYPE_TABLE='TABLE',
                                   TITRE='Table_Depl_R',
                                   )
        print(WARN+'No rotation dof  (DRX, DRY, DRZ) in the model. Computing only with displacements (DX, DY, DZ).'+ENDC)

    VectUpOmegaFe = VectFromAsterTable2Full(tstaT, NNodes)

    DETRUIRE (CONCEPT = _F (NOM = (tstaT, LIST),
                            ),
              INFO = 1,
              )

    return VectUpOmegaFe




def VectFromAsterTable2Full(Table, NNodes):
    """
    Extraction of the displacements (and rotations if available) of the nodes in the aster Table 
    Inputs:
        Table  : aster table 
        NNodes : number of nodes, can be obtained by DictStructParam['MeshProperties']['NNodes']
    Output:
        ConcatenatedArray : vector of the all the degrees of freedom at the instants
    """
    
    depl_sta = Table.EXTR_TABLE()['NOEUD', 'DX', 'DY','DZ','DRX', 'DRY','DRZ'].values()
    if depl_sta == {}:
        depl_sta = Table.EXTR_TABLE()['NOEUD', 'DX', 'DY','DZ'].values()
        print(WARN+'No rotation dof  (DRX, DRY, DRZ) in the model. Computing only with displacements (DX, DY, DZ).'+ENDC)

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




def ComputeStaFullNodalF(NNodes, **kwargs):
    """
    Extraction of the internal forces at all the degrees of freedom resulting from a static computations and return them in an array for all the dofs
    Inputs : 
        NNodes   : number of nodes, can be obtained by DictStructParam['MeshProperties']['NNodes']
        **kwargs : dictionnary containing MODELE, CHMAT, CARELEM and SOLU
    
    Output: 
        VectNodalF : array of the internal forces at all the degrees of freedom
    """

    tstaT2 = GetAsterTableOfStaticNodalForces(**kwargs)

    VectNodalF = VectFromAsterTable2Full(tstaT2, NNodes)

    DETRUIRE(CONCEPT = _F(NOM = (tstaT2)))

    return VectNodalF




def GetAsterTableOfStaticNodalForces(InstantsExtr = [1.0], **kwargs):
    """
    Computes the nodal internal forces and returns the table, the latter will be extracted with the function ComputeStaFullNodalF
    Inputs:
        InstantsExtr : list of the instants desired for the extraction, the default value is [1.0] because we chose the last instant at 1.0 for static computations
        **kwargs     : dictionnary containing MODELE, CHMAT, CARELEM and SOLU

    Output:
        tstaT2 : Aster table of the nodal forces
    """

    LIST   = DEFI_LIST_REEL(VALE = InstantsExtr)
    F_noda = CALC_CHAMP(MODELE     = kwargs['MODELE'],
                        CHAM_MATER = kwargs['CHMAT'],
                        CARA_ELEM  = kwargs['CARELEM'],
                        LIST_INST  = LIST,
                        RESULTAT   = kwargs['SOLU'],
                        FORCE      = 'FORC_NODA'
                        );

    try: # if the model contains displacements and rotations degrees of freedom
        tstaT2 = CREA_TABLE(RESU = _F(RESULTAT = F_noda,
                                      NOM_CHAM = 'FORC_NODA',
                                      NOM_CMP  = ('DX','DY','DZ', 'DRX', 'DRY', 'DRZ'),
                                      TOUT     ='OUI',
                                      ),
                            TYPE_TABLE = 'TABLE',
                            TITRE      = 'Table_Force_N',
                            )
    except: # if the model contains only displacements degrees of freedom
        tstaT2 = CREA_TABLE(RESU = _F(RESULTAT = F_noda,
                                      NOM_CHAM = 'FORC_NODA',
                                      NOM_CMP  = ('DX','DY','DZ'),
                                      TOUT     ='OUI',
                                      ),
                            TYPE_TABLE = 'TABLE',
                            TITRE      = 'Table_Force_N',
                            )
        print(WARN + 'Only Fx, Fy and Fz are present'+ENDC)
    DETRUIRE(CONCEPT = _F(NOM = (F_noda, LIST)))


    return tstaT2




def ComputeTransformationLists(DictStructParam, Table):
    """
    Update the dictionnary of structural parameters with information about the degrees of freedom
    Inputs : 
        Table           : aster table of results, used to extract the number of degrees of freedom
        DictStructParam : dictionnary of the structural parameters
    
    Output : 
        DictStructParam : dictionnary of the structural parameters, enriched with:
                            DictStructParam['MeshProperties']['Transformations']
                            DictStructParam['MeshProperties']['Transformations']['FOM2XYZ']
                            DictStructParam['MeshProperties']['Transformations']['VectDDLNum']
                            DictStructParam['MeshProperties']['Transformations']['VectDDL']
                            DictStructParam['MeshProperties']['Transformations']['DDLNodes']
                            DictStructParam['MeshProperties']['Transformations']['DDL2Node']
    """

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

    ###############################
    def CalcVectDDL(ArrayStr):
        VectNames = []
        DDLNum = []
        for comp in ArrayStr:
            if comp not in VectNames:
                VectNames.append(comp)
            DDLNum.append(VectNames.index(comp))
        VectNames = '.'.join(VectNames)
        return DDLNum, VectNames
    ###############################

    DictStructParam['MeshProperties']['Transformations']['VectDDLNum'],DictStructParam['MeshProperties']['Transformations']['VectDDLNames'] = CalcVectDDL(np.array(np.concatenate(l_ddl)))
    DictStructParam['MeshProperties']['Transformations']['VectDDL'] = np.array(np.concatenate(l_ddl))
    DictStructParam['MeshProperties']['Transformations']['DDLNodes'] = np.split(np.array(np.concatenate(l_ddl)), l_SplitArray2Vars[1:])

    ddl2Node = []
    for Node, posNode in zip(DictStructParam['MeshProperties']['Transformations']['DDLNodes'], range(DictStructParam['MeshProperties']['NNodes'])):
        for comp in Node:
            ddl2Node.append(posNode + 1)

    DictStructParam['MeshProperties']['Transformations']['DDL2Node'] = ddl2Node

    return DictStructParam





# MODAL ANALYSIS: from ModalAnalysis.py
#----------------------------------------------------------------

def CalcLNM(RPM, DictStructParam, **kwargs):
    
    if not DictStructParam['ROMProperties']['RigidMotion']:

        MODESR = CALC_MODES(MATR_RIGI = kwargs['Komeg2'],
                            MATR_MASS = kwargs['MASS1'],
                            OPTION = 'PLUS_PETITE',
                            CALC_FREQ = _F(NMAX_FREQ = DictStructParam['ROMProperties']['NModes'],),
                            VERI_MODE = _F(SEUIL = 1.E-3, STOP_ERREUR = 'NON'),
                            )
    elif DictStructParam['ROMProperties']['RigidMotion']:

        MODESR = CALC_MODES(MATR_RIGI = kwargs['Komeg2'],
                            MATR_MASS = kwargs['MASS1'],
                            OPTION = 'BANDE',
                            CALC_FREQ = _F(FREQ = (-1,250.),),
                            VERI_MODE = _F(SEUIL = 1.E-3, STOP_ERREUR = 'NON'),
                            #SOLVEUR_MODAL = _F(METHODE = 'TRI_DIAG',
                            #                   MODE_RIGIDE = 'OUI',),
                          )


    #MODESR = CALC_MODES(affe_modes)
    MODE   = NORM_MODE(MODE = MODESR,
                       NORME = 'TRAN_ROTA',)

    PHImatrix, freq = Freq_Phi(RPM, DictStructParam, MODE)


    DETRUIRE (CONCEPT = _F (NOM = (MODESR, MODE),
                            ),
              INFO = 1,
              )

    return PHImatrix, freq



def Freq_Phi(RPM,DictStructParam,  MODE):

    freq = MODE.LIST_VARI_ACCES()['FREQ']
    freq = np.array(freq[:DictStructParam['ROMProperties']['NModes']])

    PHImatrix = np.zeros((DictStructParam['MeshProperties']['Nddl'],DictStructParam['ROMProperties']['NModes']))

    #ModZones = []
    for Mode in range(DictStructParam['ROMProperties']['NModes']):
        try:
            tabmod_T = CREA_TABLE(RESU=_F(RESULTAT= MODE,
                                  NOM_CHAM='DEPL',
                                  NOM_CMP= ('DX','DY','DZ', 'DRX', 'DRY', 'DRZ'),
                                  NUME_ORDRE=Mode+1,
                                  TOUT = 'OUI'),
                                  TYPE_TABLE='TABLE',
                                  TITRE='Table_Modes',);
        except:
            tabmod_T = CREA_TABLE(RESU=_F(RESULTAT= MODE,
                                  NOM_CHAM='DEPL',
                                  NOM_CMP= ('DX','DY','DZ'),
                                  NUME_ORDRE=Mode+1,
                                  TOUT = 'OUI'),
                                  TYPE_TABLE='TABLE',
                                  TITRE='Table_Modes',);
            print(WARN+'No rotation dof  (DRX, DRY, DRZ) in the model. Computing only with displacements (DX, DY, DZ).'+ENDC)

        #ModZone = SJ.CreateNewSolutionFromAsterTable(t, FieldDataTable= tabmod_T,
        #                                                ZoneName = 'Mode%s_'%Mode+str(np.round(RPM)),
        #                                                FieldName = 'Mode',
        #                                                )

        PHImatrix[:,Mode] = VectFromAsterTable2Full(tabmod_T, DictStructParam['MeshProperties']['NNodes'])

        DETRUIRE (CONCEPT = _F (NOM = (tabmod_T),
                            ),
              INFO = 1,
              )



        #ModZones.append(ModZone)

        #try:
        #  I._addChild(I.getNodeFromName(t, 'ModalBases'), ModZone)
        #except:
        #  t = I.merge([t, C.newPyTree(['ModalBases', []])])
        #  I._addChild(I.getNodeFromName(t, 'ModalBases'), ModZone)
        #I._addChild(I.getNodeFromName(t, 'ModalBases'), I.createNode('Freq_%sRPM'%np.round(RPM,2), 'DataArray_t', value = freq))





    #t = SJ.AddFOMVars2Tree(t, RPM, Vars = [PHImatrix], VarsName = ['PHI'], Type = '.AssembledMatrices')


    #C.convertPyTree2File(t,'/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/Test1.cgns', 'bin_adf')
    #C.convertPyTree2File(t,'/visu/mbalmase/Projets/VOLVER/0_FreeModalAnalysis/Test1.tp', 'bin_tp')

#    print(len(ModZones))
#    XX
#    VectUsOmega = VectFromAsterTable2Full(t, tabmod_T)
#
#    t = SJ.AddFOMVars2Tree(t, RPM, Vars = [VectUsOmega],
#                                   VarsName = ['Us'],
#                                   Type = '.AssembledVectors',
#                                   )
#

    # Tableau complet des modes, coordonnees modales,... :
#    print(tabmod_T.EXTR_TABLE().values().keys())
#    XXX

#    depl_mod = tabmod_T.EXTR_TABLE()['NOEUD', 'DX', 'DY', 'DZ']
#
#    # Liste des valeurs de la table:
#    depl_list = []
#    depl_list_Names = ['ModeX', 'ModeY', 'ModeZ']
#    depl_list.append(depl_mod.values()['DX'][:])
#    depl_list.append(depl_mod.values()['DY'][:])
#    depl_list.append(depl_mod.values()['DZ'][:])
#    #dep_md_X = depl_mod.values()['DX'][:]
#    #dep_md_Y = depl_mod.values()['DY'][:]
#    #dep_md_Z = depl_mod.values()['DZ'][:]
#
#    elif DictStructParam['MeshProperties']['ddlElem'][0] == 6:
#
#        tabmod_T = CREA_TABLE(RESU=_F(RESULTAT= MODE,
#                              NOM_CHAM='DEPL',
#                              NOM_CMP= ('DX','DY','DZ','DRX','DRY','DRZ'),
#                              TOUT='OUI',),
#                              TYPE_TABLE='TABLE',
#                              TITRE='Table_Modes',);
#
#        # Tableau complet des modes, coordonnees modales,... :
#
#        depl_mod = tabmod_T.EXTR_TABLE()['NOEUD', 'DX', 'DY', 'DZ','DRX','DRY','DRZ']
#
#        # Liste des valeurs de la table:
#        depl_list = []
#        depl_list.append(depl_mod.values()['DX'][:])
#        depl_list.append(depl_mod.values()['DY'][:])
#        depl_list.append(depl_mod.values()['DZ'][:])
#        depl_list.append(depl_mod.values()['DRX'][:])
#        depl_list.append(depl_mod.values()['DRY'][:])
#        depl_list.append(depl_mod.values()['DRZ'][:])
#        depl_list_Names = ['ModeX', 'ModeY', 'ModeZ','ModeThetaX','ModeThetaY','ModeThetaZ']
#
#    # Create a matrix with the base and save it into the .AssembledMatrices node:


#    t, PHI = SJ.BuildMatrixFromComponents(t, 'PHI', depl_list)
#
#    # Extract the nodes coordinates:
#    NodeZones = []
#    for NMode in range(DictStructParam['ROMProperties']['NModes'][0]):
#
#        Coord_list = []
#        for pos in range(len(depl_list)):
#
#            Coord_list.append(PHI[pos::len(depl_list), NMode])
#
#        #CoordX = dep_md_X[NMode * DictStructParam['MeshProperties']['NNodes'][0]:(NMode + 1) * DictStructParam['MeshProperties']['NNodes'][0]]
#        #CoordY = dep_md_Y[NMode * DictStructParam['MeshProperties']['NNodes'][0]:(NMode + 1) * DictStructParam['MeshProperties']['NNodes'][0]]
#        #CoordZ = dep_md_Z[NMode * DictStructParam['MeshProperties']['NNodes'][0]:(NMode + 1) * DictStructParam['MeshProperties']['NNodes'][0]]
#
#        NewZone,_ = SJ.CreateNewSolutionFromNdArray(t,
#                                       FieldDataArray= Coord_list,
#                                       ZoneName = str(np.round(RPM, 2))+'Mode'+str(NMode),
#                                       FieldNames = depl_list_Names,
#                                       Depl = True)
#
#        I.createChild(NewZone, 'Freq', 'DataArray_t', freq[NMode])
#
#        NodeZones.append(NewZone)
#
#
#    I.addChild(I.getNodeFromName(t, 'ModalBases'), NodeZones)
#
#    DETRUIRE (CONCEPT = _F (NOM = (tabmod_T),
#                            ),
#              INFO = 1,
#              )

    return PHImatrix, freq


# Parametrization
###################
def loadMatrixFromDict(AssembledMatrices, RPM, MatrixName):

    return AssembledMatrices['%sRPM'%np.round(RPM,2)][MatrixName]




def BuildRPMParametrisation(DictStructParam,DictSimulaParam, AssembledMatrices):
    # This function implements the parametric model:
        #      --> Compose the enlarged modal matrix 
        #      --> Compute the three constant matrices composing the model
        #      --> Remove the unnecessary terms from the tree (just to keep the parametric ones)
        #      --> Save on the tree the variables defining the model
    
    #To check if a parametric model is requested
    RPMs=DictSimulaParam['RotatingProperties']['RPMs']
    #SJ.SaveModel(t, kwargs['FOMName'], Modes = True, StaticRotatorySolution = True)
    if len(RPMs)==3 and (RPMs[1]-RPMs[0])==(RPMs[2]-RPMs[1]):
        print(GREEN + 'Parametric model (3 RPMs)'+ ENDC)
        
        
        NewFOMmatrices = {}
        NewFOMmatrices['Parametric']={}

        MatrFOM = AssembledMatrices
        MatrFOM['Temporary']={}
        
        #Composing enlarged modal matrix (PHIAug)
        PHIAug = []
        PHIAug = np.hstack((MatrFOM[str(np.round(RPMs[0],2))+'RPM']['PHI'],MatrFOM[str(np.round(RPMs[1],2))+'RPM']['PHI']))
        PHIAug = np.hstack((PHIAug,MatrFOM[str(np.round(RPMs[2],2))+'RPM']['PHI']))

        #Singluar values decomposition (SVD)
        U,s,Vt = linalg.svd(PHIAug, full_matrices=True) 
        print('SVD done')
        
        ## Single values choice (it depends on the matrix type: array in this case)
        # Svalue 0.01%max(sValue)         #What about if all s are negative: CHANGE THIS CONDITION
        index=[ i for i in range(0,len(s)) if s[i]>=0.01/100.*max(s)]
        print(CYAN+'NewNumberOfModes: %s'%len(index)+ENDC)
        #The maximum number of single values is 3*r (where r is the number of modes that are chosen)
        
        ## Criterion to avoid taking too many modes:
        #NModesMax=100
        #if len(index)>NModesMax:
        #    print(WARN  + 'Number of nodes has been truncated'+ ENDC)
        #    index=index[0:NMmodesMax] #The first NModesMax modes (they are already sorted in descending order)
        
        U=U[:,index]
        # Normalize to 1: 
        for pos in range(len(index)):
            U[:,pos] = U[:,pos]/np.max(abs(U[:,pos]))
        NewFOMmatrices['Parametric']['PHI'] = U   
       
        # Save the basis in the tree:  
#        for indexVect in range(len(index)):
#            ModeVect = U[:,index[indexVect]]
#
#            ModZone = SJ.CreateNewSolutionFromNdArray(t, FieldDataArray = [ModeVect], ZoneName='Mode%s_Parametric'%indexVect,
#                                               FieldName = 'ParametricMode%s'%indexVect
#                                    )
#
#            try:
#              I._addChild(I.getNodeFromName(t, 'ModalBases'), ModZone)
#            except:
#              t = I.merge([t, C.newPyTree(['ModalBases', []])])
#              I._addChild(I.getNodeFromName(t, 'ModalBases'), ModZone)
#            #I._addChild(I.getNodeFromName(t, 'ModalBases'), I.createNode('Freq_%sRPM'%np.round(RPM,2), 'DataArray_t', value = np.sqrt(s[index[indexVect]])/(2.*np.pi))
#

        print(WARN + 'Warning! The requested number of modes has changed from %s to %s'%(DictStructParam['ROMProperties']['NModes'], len(index))+ENDC)

        DictStructParam['ROMProperties']['NModes'] = len(index)
        
        #If this line is not executed, the node is not added to the tree
        #J.set(t, '.StructuralParameters', **DictStructParam)
        

        #FO model constants
        Kp0       = loadMatrixFromDict(AssembledMatrices, RPMs[0], 'Komeg')
        Kp0Delta  =  loadMatrixFromDict(AssembledMatrices, RPMs[1], 'Komeg')
        Kp02Delta =  loadMatrixFromDict(AssembledMatrices, RPMs[2], 'Komeg')
        Deltap=RPMs[1]-RPMs[0]

        NewFOMmatrices['Parametric']['Deltap'] = Deltap
        NewFOMmatrices['Parametric']['p0'] = RPMs[0]
        NewFOMmatrices['Parametric']['Range'] = [RPMs[0],RPMs[-1]]

        NewFOMmatrices['Parametric']['K0Parametric'] = Kp0
        NewFOMmatrices['Parametric']['K1Parametric'] = ((-1)*Kp02Delta+4*Kp0Delta-3*Kp0)/(2*Deltap)
        NewFOMmatrices['Parametric']['K2Parametric'] = (Kp02Delta-2*Kp0Delta+Kp0)/((Deltap)*(Deltap))

        NewFOMmatrices['Parametric']['C'] = loadMatrixFromDict(AssembledMatrices, RPMs[0], 'C')
        NewFOMmatrices['Parametric']['M'] = loadMatrixFromDict(AssembledMatrices, RPMs[0], 'M')


        #If this line is not executed, the node is not added to the tree
        #J.set(t, '.AssembledMatrices', **NewFOMmatrices)
        
        #for NameMV in NewFOMmatrices['Parametric'].keys(): #K0FD,K1FD,K2FD,M et C
        #    print(str(NameMV)+' being saved:')
        #    t = SJ.AddFOMVars2Tree(t, 'Parametric', Vars = [NewFOMmatrices['Parametric'][NameMV]], 
        #                           VarsName = [NameMV], 
        #                           Type = '.AssembledMatrices',
        #                           )
        #    print(str(NameMV)+' saving done!')

        return  NewFOMmatrices,  True

    elif len(RPMs)==4 and (RPMs[1]-RPMs[0])==(RPMs[2]-RPMs[1]) and (RPMs[1]-RPMs[0])==(RPMs[3]-RPMs[2]):
        print(GREEN + 'Parametric model (4 RPMs)'+ ENDC)
        
        
        NewFOMmatrices = {}
        NewFOMmatrices['Parametric']={}

        MatrFOM = AssembledMatrices
        MatrFOM['Temporary']={}
        
        #Composing enlarged modal matrix (PHIAug)
        PHIAug = []
        PHIAug = np.hstack((MatrFOM[str(np.round(RPMs[0],2))+'RPM']['PHI'],MatrFOM[str(np.round(RPMs[1],2))+'RPM']['PHI'])) 
        PHIAug = np.hstack((PHIAug,MatrFOM[str(np.round(RPMs[2],2))+'RPM']['PHI']))   
        PHIAug = np.hstack((PHIAug,MatrFOM[str(np.round(RPMs[3],2))+'RPM']['PHI'])) 
        
                

        #Singluar values decomposition (SVD)
        U,s,Vt = linalg.svd(PHIAug, full_matrices=True) 
        print('SVD done')


        ## Single values choice (it depends on the matrix type: array in this case)
        # Svalue 0.01%max(sValue)         #What about if all s are negative: CHANGE THIS CONDITION
        index=[ i for i in range(0,len(s)) if s[i]>=0.01/100.*max(s)]
        #The maximum number of single values is 4*r (where r is the number of modes that are chosen)
        
        
        ## Criterium to avoid taking too many modes
        #NModesMax=15
        #if len(index)>NModesMax:
        #    print(WARN  + 'Number of nodes has been truncated'+ ENDC)
        #    index=index[0:NMmodesMax] #The first NModesMax modes (they are already sorted in descending order)
        

        U=U[:,index]
        for pos in range(len(index)):
            U[:,pos] = U[:,pos]/np.max(abs(U[:,pos]))
        NewFOMmatrices['Parametric']['PHI'] = U   
        
        # Save the basis in the tree:
#        for indexVect in range(len(index)):
#            ModeVect = U[:,index[indexVect]]
#            
#            ModZone = SJ.CreateNewSolutionFromNdArray(t, FieldDataArray = [ModeVect], ZoneName='Mode%s_Parametric'%indexVect,
#                                               FieldName = 'ParametricMode%s'%indexVect
#                                    )
#
#            try:
#              I._addChild(I.getNodeFromName(t, 'ModalBases'), ModZone)
#            except:
#              t = I.merge([t, C.newPyTree(['ModalBases', []])])
#              I._addChild(I.getNodeFromName(t, 'ModalBases'), ModZone)
#            #I._addChild(I.getNodeFromName(t, 'ModalBases'), I.createNode('Freq_%sRPM'%np.round(RPM,2), 'DataArray_t', value = np.sqrt(s[index[indexVect]])/(2.*np.pi))
     

        print(WARN + 'Warning! The requested number of modes has changed from %s to %s'%(DictStructParam['ROMProperties']['NModes'], len(index))+ENDC)

        DictStructParam['ROMProperties']['NModes'] = len(index)
        
        #If this line is not executed, the node is not added to the tree
        #J.set(t, '.StructuralParameters', **DictStructParam)

        #FO model constants
        Kp0 = loadMatrixFromDict(AssembledMatrices, RPMs[0], 'Komeg')
        Kp0Delta = loadMatrixFromDict(AssembledMatrices, RPMs[1], 'Komeg')
        Kp02Delta = loadMatrixFromDict(AssembledMatrices, RPMs[2], 'Komeg')
        Kp03Delta = loadMatrixFromDict(AssembledMatrices, RPMs[3], 'Komeg')
        Deltap=RPMs[1]-RPMs[0]
        
        NewFOMmatrices['Parametric']['Deltap'] = Deltap
        NewFOMmatrices['Parametric']['p0'] = RPMs[0]
        NewFOMmatrices['Parametric']['Range'] = [RPMs[0],RPMs[-1]]

        NewFOMmatrices['Parametric']['K0Parametric'] = Kp0
        NewFOMmatrices['Parametric']['K1Parametric'] = ((-11/6)*Kp0+(3)*Kp0Delta+(-3/2)*Kp02Delta+(1/3)*Kp03Delta)/(Deltap)
        NewFOMmatrices['Parametric']['K2Parametric'] = ((2)*Kp0+(-5)*Kp0Delta+(4)*Kp02Delta+(-1)*Kp03Delta)/(Deltap**2)
        NewFOMmatrices['Parametric']['K3Parametric'] = ((-1)*Kp0+(3)*Kp0Delta+(-3)*Kp02Delta+(1)*Kp03Delta)/(Deltap**3)

        NewFOMmatrices['Parametric']['C'] = loadMatrixFromDict(AssembledMatrices, RPMs[0], 'C')
        NewFOMmatrices['Parametric']['M'] = loadMatrixFromDict(AssembledMatrices, RPMs[0], 'M')


        #If this line is not executed, the node is not added to the tree
        #J.set(t, '.AssembledMatrices', **NewFOMmatrices)
        #J.set(t, '.StructuralParameters', **DictStructParam)

        #for NameMV in NewFOMmatrices['Parametric'].keys(): #K0FD,K1FD,K2FD,M et C
        #    print(str(NameMV)+' being saved:')
        #    t = SJ.AddFOMVars2Tree(t, 'Parametric', Vars = [NewFOMmatrices['Parametric'][NameMV]],
        #                           VarsName = [NameMV], 
        #                           Type = '.AssembledMatrices',
        #                           )
        #    print(str(NameMV)+' saving done!')

        return NewFOMmatrices, True

    else:
        print('Not a parametric model')
        

        return AssembledMatrices, False



# NONLINEAR FORCES MODELS
#----------------------------------------------------------------

def ComputeNLCoefficients(DictStructParam, DictSimulaParam, DictOfVectorsAndMatrices, RPM, DictOfCoefficients, **kwargs):
    
    
    # ICE method:
    if (DictStructParam['ROMProperties']['ROMForceType'] == 'IC') or (DictStructParam['ROMProperties']['ROMForceType'] == 'ICE'):
        DictOfCoefficients = computeICECoefficients(DictStructParam, DictSimulaParam, DictOfVectorsAndMatrices, RPM, DictOfCoefficients, **kwargs) 

    # DNF methods: Warning, the system becomes into a single equation
    if 'DNF1' in DictStructParam['ROMProperties']['ROMForceType']:

        DictOfCoefficients = DNF.compute1ModeDNFCoefficients(DictStructParam, DictSimulaParam,DictOfVectorsAndMatrices, RPM, DictOfCoefficients,**kwargs)
   





    return DictOfCoefficients  


def computeICECoefficients(DictStructParam, DictSimulaParam, DictOfVectorsAndMatrices, RPM, DictOfCoefficients, **kwargs):

    
    # Calcul de tous les cas statiques avec les forces imposees:
    MatrULambda, MatrFLambda = ComputeMatrULambda4ICE(DictStructParam, DictSimulaParam, DictOfVectorsAndMatrices, RPM, **kwargs)
    
    # Compute the pseudo inverse of MatrULambda with respect to PHI:
    MatrQLambda = PseudoInverseWithModes(DictOfVectorsAndMatrices, RPM, MatrULambda)
    
    # Compute the pseudo inverse of MatrQlambda to solve the unknowns coefficient:
    PinvMatrQLambda = ComputeQMatrix4ICE(DictStructParam['ROMProperties']['NModes'], MatrQLambda)

    # Expansion, compute the ExpansionBase:
     
    ExpansionBase = ComputeExpansionBase(MatrULambda,MatrQLambda, DictOfVectorsAndMatrices['AsseMatrices']['%sRPM'%np.round(RPM,2)]['PHI'])

    # Compute the nonlinear forces matrice: 

    FnlLambdaMatrix = ComputeForceMatrix4ICE(DictOfVectorsAndMatrices, RPM, MatrULambda, MatrFLambda) 

    # Compute the Aij^k and the Bijm^k: 
    
    Aij, Bijm = ComputeAijAndBijmCoefficients(DictStructParam, DictOfVectorsAndMatrices['AsseMatrices']['%sRPM'%np.round(RPM,2)]['PHI'], FnlLambdaMatrix, PinvMatrQLambda)
    
    #DictOfCoefficients = J.get(t,'.InternalForcesCoefficients')
    DictOfCoefficients['%sRPM'%np.round(RPM,2)] = dict(Type = 'ICE',
                                                   Aij  = Aij,
                                                   Bijm = Bijm, 
                                                   ExpansionBase = ExpansionBase)



    #J.set(t,'.InternalForcesCoefficients', **dict(DictOfCoefficients))

    return DictOfCoefficients


def ComputeMatrULambda4ICE(DictStructParam, DictSimulaParam, DictOfVectorsAndMatrices, RPM, **kwargs):

    DictAsseVectors = DictOfVectorsAndMatrices['AsseVectors']

    lambda_vect = DictStructParam['ROMProperties']['lambda_vect']
    # If ICE generation as STEP logic:
    if DictStructParam['ROMProperties']['ICELoadingType'] == 'STEPType':
    
        nr = DictStructParam['ROMProperties']['NModes']
        len_quad = nr*(nr+1)/2
        len_cubic = (nr**3 + 3*nr**2 + 2*nr)/6
        lenQ = int(len_quad + len_cubic)
       
        MatrUpLambda = np.zeros((DictStructParam['MeshProperties']['Nddl'], lenQ))
        MatrFLambda = np.zeros((DictStructParam['MeshProperties']['Nddl'], lenQ))
        
        PHI = DictOfVectorsAndMatrices['AsseMatrices']['%sRPM'%np.round(RPM,2)]['PHI']

        # Combinaison d'un seul mode
        count = -1
        for i in range(DictStructParam['ROMProperties']['NModes']):
            count += 1         
            print(FAIL + 'Counter: %s/%s'%(count+1, lenQ) +ENDC)
            f1 = np.dot(PHI[: , i],  lambda_vect[i])
            f2 = -1. * f1 
            
            MatrUpLambda[:, count],_,_,_ = ComputeStaticU4GivenLoading(RPM,DictStructParam, DictSimulaParam, f1, **kwargs)
            MatrFLambda[:, count] = f1
            count += 1

            print(FAIL + 'Counter: %s/%s'%(count+1, lenQ) +ENDC)
            MatrUpLambda[:, count],_,_,_ = ComputeStaticU4GivenLoading(RPM,DictStructParam, DictSimulaParam, f2, **kwargs)
            MatrFLambda[:, count] = f2
         
         # Combinaison de deux modes   
                 
        for i in range(DictStructParam['ROMProperties']['NModes']):
            for j in range(i+1, DictStructParam['ROMProperties']['NModes']):
                count += 1         
                
                f3 = PHI[:,i] * lambda_vect[i] + PHI[: , j] * lambda_vect[j]   
                f4 = -1 *(PHI[:,i] * lambda_vect[i]) - PHI[: , j] * lambda_vect[j]
                f5 = PHI[:,i] * lambda_vect[i] - PHI[: , j] * lambda_vect[j]
 
                MatrUpLambda[:, count],_,_,_ = ComputeStaticU4GivenLoading(RPM, DictStructParam, DictSimulaParam, f3, **kwargs)
                MatrFLambda[:, count] = f3
                count += 1
                print(FAIL + 'Counter: %s/%s'%(count+1, lenQ) +ENDC)
                MatrUpLambda[:, count],_,_,_ = ComputeStaticU4GivenLoading(RPM, DictStructParam, DictSimulaParam, f4, **kwargs)
                MatrFLambda[:, count] = f4
                count += 1
                print(FAIL + 'Counter: %s/%s'%(count+1, lenQ) +ENDC)
                MatrUpLambda[:, count],_,_,_ = ComputeStaticU4GivenLoading(RPM, DictStructParam, DictSimulaParam, f5, **kwargs)
                MatrFLambda[:, count] = f5
             
                       
                     
        # Combinaison de 3 modes
     
        for i in range(DictStructParam['ROMProperties']['NModes']):
            for j in range(i+1, DictStructParam['ROMProperties']['NModes']):
                for k in range(j+1, DictStructParam['ROMProperties']['NModes']):
                    count += 1 
                    print(FAIL + 'Counter: %s/%s'%(count+1, lenQ) +ENDC)
                    f6 = PHI[:,i] * lambda_vect[i] + PHI[: , j] * lambda_vect[j] + PHI[: , k] * lambda_vect[k]   
                    MatrFLambda[:, count] = f6              
                    MatrUpLambda[:, count],_ ,_ ,_ = ComputeStaticU4GivenLoading(RPM, DictStructParam, DictSimulaParam, f6, **kwargs)
        

        # Compute rotating matrix to substract from MatrUpLambda:

        MatrUs = DictAsseVectors['%sRPM'%np.round(RPM,2)]['Us'].reshape((DictStructParam['MeshProperties']['Nddl'], 1)) * np.ones(np.shape(MatrUpLambda))
        
    return MatrUpLambda - MatrUs, MatrFLambda  

def PseudoInverseWithModes(DictOfVectorsAndMatrices, RPM, Matrice):
    '''Calcul de la pseudo inverse de Matrice en utilisant la base modale a RPM'''
    PHI = DictOfVectorsAndMatrices['AsseMatrices']['%sRPM'%np.round(RPM,2)]['PHI']
    # Recuperer la base modale a RPM
    return (np.linalg.inv((PHI.T).dot(PHI)).dot(PHI.T)).dot(Matrice)


def ComputeForceMatrix4ICE(DictOfVectorsAndMatrices, RPM, MatrULambda, MatrFLambda):
    '''Compute the nonlinear forces related to MatrULambda'''
    
    GnlULambda = np.zeros(np.shape(MatrULambda))

    Komeg = DictOfVectorsAndMatrices['AsseMatrices']['%sRPM'%np.round(RPM,2)]['Komeg']

    for i in range(len(MatrFLambda[0,:])):
        
        GnlULambda[:, i] = MatrFLambda[:, i] - Komeg.dot(MatrULambda[:,i])

    return GnlULambda

def ComputeExpansionBase(MatrULambda, MatrQLambda, PHI):
    """Calcul de la ExpansionBase pour la reconstruction ICE
    nf: Nombre des modes
    MatrULambda: Matrice avec les solutions u des calculs statiques
    """
    nf = np.shape(PHI)[1]
    nL = np.shape(MatrQLambda)[1]

    # construction de la matrice des produits des coordonnees generalisees
    indexes = [i for i in range(nf)]
    comb2 = combinations_with_replacement(indexes, 2) # as the list [1,2,3] is given in growing order, the combinations will be in ordered (ex (1,2) and not (2,1) )

    listcomb2 = list(comb2)
    
    Q_quad = np.zeros((nL, len(listcomb2)))
    for i in range(nL):
        qivect = MatrQLambda[:,i]  # line
        vect2 = combination_to_product(qivect, listcomb2)
        
        Q_quad[i,:] = vect2

    eta_s_mat = np.transpose(Q_quad) # matrix whose columns are quadratic combinations of the generalised coordinates
    PseudoInv_eta = np.linalg.pinv(eta_s_mat)

    # membrane modes rebuilt
    return  (MatrULambda - np.dot(PHI,MatrQLambda)).dot(PseudoInv_eta) 

def ComputeQMatrix4ICE(nr, qs_mat):
    '''nr: Number of modes, 
       qs_mat: Matrice Qlamda, MatrQLambda'''
    
    nsol = int(np.shape(qs_mat)[1])
    len_quad = int(nr*(nr+1)/2)
    len_cubic = int((nr**3 + 3*nr**2 + 2*nr)/6)
    lenQ = int(len_quad + len_cubic)
    Q = np.zeros((nsol, lenQ))
    for i in range(nsol):
        Q[i,0:len_quad] = quad_vect(qs_mat[:,i].reshape((nr,1)))
        Q[i, len_quad:lenQ] = cubic_vect(qs_mat[:,i].reshape((nr,1)))

    #rcondQ = np.linalg.cond(Q)
    #print('rcondQ', rcondQ)
    
    # Compute the pseudo inverse:

    return np.linalg.pinv(Q)

def quad_vect(qvect):
    '''Function that takes the column vector of the generalised coordinates
    and returns the vector [q1q1 q1q2 ... q1qn q2q2 q2q3 ... qnqn]'''

    n = np.shape(qvect)[0]
    lQ = int(n*(n+1)/2)
    Q2 = np.zeros(lQ)
    q = np.ravel(qvect)

    compteur = 0
    i = 0
    j = 0

    while i < n :
        while j < n :
            Q2[compteur] = q[i]*q[j]
            compteur += 1
            j += 1
        i += 1
        j = i

    return Q2


def cubic_vect(qvect):
    '''Function that takes the column vector of the generalised coordinates
    and returns the vector [q1q1q1 q1q1q2 ... q1qnqn q2q2q2 q2q2q3 ... qnqnqn]'''

    n = np.shape(qvect)[0]
    if n < 3:
        print('n should be bigger than 3')
        exit()
    
    lC = int((n**3 + 3*n**2 + 2*n)/6)
    Q3 = np.zeros(lC)
    q = np.ravel(qvect)

    compteur = 0
    i = 0
    j = 0
    m = 0

    while i < n:
        while j < n:
            while m < n:
                Q3[compteur] = q[i]*q[j]*q[m]
                compteur += 1
                m += 1
            j += 1
            m = j
        i += 1
        j = i
        m = j
    
    return Q3

def ComputeAijAndBijmCoefficients(DictStructParam, PHI, FnlLambdaMatrix, PinvMatrQLambda):
    
    NModes = DictStructParam['ROMProperties']['NModes']
    Aij, Bijm = np.zeros((NModes, NModes, NModes)) , np.zeros((NModes, NModes, NModes, NModes))

    for mode in range(DictStructParam['ROMProperties']['NModes']):

        bk = (PHI[:,mode].dot(FnlLambdaMatrix)).T
        
        Xk = PinvMatrQLambda.dot(bk)

        Aij, Bijm = extract_X(Xk,NModes,Aij,Bijm,mode)

    return Aij, Bijm

def combination_to_product(q, listcomb):
    ''' from the LINE vector qvect of generalised coordinates, and the list of combinations listcomb,
    returns the product of the generalised coordinates according to the indexes of the elements of listcomb in a numpy array LINE
    ex for combinations of 2: [(0,0), (0,1) , (1,1)] --> [q0*q0, q0*q1, q1*q1] '''

    lencomb = len(listcomb)
    Qvect = np.zeros(lencomb)
    nbelemcomb = len(listcomb[0]) # to get the number of elements of the combinations
    for i in range(lencomb):
        temp = 1
        comb = listcomb[i]
        for j in range(nbelemcomb):
            
            temp = temp*q[comb[j]]
        Qvect[i] = temp

    return Qvect

def extract_X(Xvect,n,Beta,Gamma,k):
    ''' The shape of the vector Xvect is : Xvect = np.transpose([ak11 ak12 ... ak22 ak23 ... aknn bk111 bk 112 ... bk11n ... bknnn])
    this function takes in argument the tables Beta and Gamma, completes them for k and returns their new version '''

    Betanew = Beta
    Gammanew = Gamma

    X = Xvect.ravel()

    lenX = np.shape(X)[0]

    len_quad = int(n*(n+1)/2)
    len_cubic = int((n**3 + 3*n**2 + 2*n)/6)

    # we decompose X as a vector of the quadratic components 
    # and a vector of the cubic components X = [X_quad, X_cubic]
    X_quad = X[0:len_quad]
    X_cubic = X[len_quad:lenX]

    # Extraction of the quadratic components
    compteur = 0
    i = 0
    j = 0

    while i < n:
        while j < n:
            Betanew[k,i,j] = X_quad[compteur]
            compteur += 1
            j += 1
        i += 1
        j = i
    
    # Extraction of the cubic components
    compteur = 0
    i = 0
    j = 0
    m = 0

    while i < n:
        while j < n:
            while m < n:
                Gammanew[k,i,j,m] = X_cubic[compteur]
                compteur += 1
                m += 1
            j += 1
            m = j
        i += 1
        j = i
        m = j

    return Betanew, Gammanew   
