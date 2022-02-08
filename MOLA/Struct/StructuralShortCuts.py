'''
MOLA - StructuralShortCuts.py

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

# System modules
import numpy as np
import vg
from scipy.sparse import csr_matrix, find, issparse
# MOLA and Cassiopee
import Converter.Internal as I
import Converter.PyTree as C
import MOLA.InternalShortcuts as J
import MOLA.Wireframe as W
import MOLA.LiftingLine as LL
try:
    #Code Aster:
    from code_aster.Cata.Commands import *
    from code_aster.Cata.Syntax import _F, CO
except:
    print(WARN + 'Code_Aster modules are not loaded!!' + ENDC)


def merge_dicts(a, b):
    m = a.copy()
    m.update(b)
    return m

def CreateNewSolutionFromNdArray(t, FieldDataArray = [], ZoneName=[],
                                    FieldNames = ['Ux', 'Uy', 'Uz'],
                                    Depl = True,
                                    DefByField = None):


    InitMesh = I.getNodesFromNameAndType(t, 'InitialMesh', 'Zone_t')[0]

    NewZone = I.copyTree(InitMesh)
    NewZone[0] = ZoneName

    XCoord, YCoord, ZCoord = J.getxyz(NewZone)

    FieldCoordX = np.array(FieldDataArray[0])
    FieldCoordY = np.array(FieldDataArray[1])
    FieldCoordZ = np.array(FieldDataArray[2])

    if Depl:
        XCoord[:] = XCoord + FieldCoordX
        YCoord[:] = YCoord + FieldCoordY
        ZCoord[:] = ZCoord + FieldCoordZ

    if (not Depl) and (DefByField is not None):
        XCoord[:] = XCoord + DefByField[0]
        YCoord[:] = YCoord + DefByField[1]
        ZCoord[:] = ZCoord + DefByField[2]

    Varx, Vary, Varz = J.invokeFields(NewZone, FieldNames)

    Varx[:], Vary[:],Varz[:], =  FieldCoordX, FieldCoordY, FieldCoordZ

    return NewZone, [Varx, Vary, Varz]

def CreateNewSolutionFromAsterTable(t, FieldDataTable, ZoneName,
                                    FieldNames = ['Ux', 'Uy', 'Uz'],
                                    Depl = True,
                                    DefByField = None):

    InitMesh = I.getNodesFromNameAndType(t, 'InitialMesh', 'Zone_t')[0]

    NewZone = I.copyTree(InitMesh)
    NewZone[0] = ZoneName


    XCoord, YCoord, ZCoord = J.getxyz(NewZone)

    FieldCoordX = np.array(FieldDataTable.values()['DX'][:])
    FieldCoordY = np.array(FieldDataTable.values()['DY'][:])
    FieldCoordZ = np.array(FieldDataTable.values()['DZ'][:])

    if Depl:
        XCoord[:] = XCoord + FieldCoordX
        YCoord[:] = YCoord + FieldCoordY
        ZCoord[:] = ZCoord + FieldCoordZ

    if (not Depl) and (DefByField is not None):
        XCoord[:] = XCoord + DefByField[0]
        YCoord[:] = YCoord + DefByField[1]
        ZCoord[:] = ZCoord + DefByField[2]

    Varx, Vary, Varz = J.invokeFields(NewZone, FieldNames)

    Varx[:], Vary[:],Varz[:], =  FieldCoordX, FieldCoordY, FieldCoordZ

    return NewZone, [Varx, Vary, Varz]


def getNameSufix(t):
    DictStructParam = J.get(t, '.StructuralParameters')
    if DictStructParam['MaterialProperties']['BehaviourLaw'] == 'Green':
        sufix = '_L'
    elif DictStructParam['MaterialProperties']['BehaviourLaw'] == 'Green-Lagrange':
        sufix = '_NL'

    return sufix

def SaveModel(t, MName):

    #FOMName += getNameSufix(t)

    C.convertPyTree2File(t, MName + '.cgns', 'bin_adf')
    #C.convertPyTree2File(t, MName + '.tp')

    return t


def getMatrixFromCGNS(t, MatrixName, RPM):
    DictMatrices = J.get(t, '.AssembledMatrices')
    return DictMatrices[str(np.round(RPM,2))+'RPM'][MatrixName]

def getVectorFromCGNS(t, VectorName, RPM):
    DictVector = J.get(t, '.AssembledVectors')
    return DictVector[str(np.round(RPM,2))+'RPM'][VectorName]

def getUsVectorFromCGNS(t, RPM):
    VectNode = I.getNodeByName(t, 'U_sta%s'%np.round(RPM,2))
    return J.getVars(VectNode, ['Usx', 'Usy', 'Usz'])

#def getPHIBaseFromCGNS(t, MatrixName, RPM): #TODO!!!
    #DictMatrices = J.get(t, '.AssembledMatrices')

#    return #DictMatrices[str(np.round(RPM))+'RPM'][MatrixName][0]


def AddFOMVars2Tree(t, RPM, Vars = [], VarsName = [], Type = '.AssembledMatrices'):
    ''' Function for saving FOM size matrices into the cgsn file
        Type = ['Matrices'  | 'Vectors']
    '''

    DictVars = J.get(t, Type)


    try:
        for Var, VarName in zip(Vars, VarsName):
            DictVars[str(np.round(RPM,2))+'RPM'][VarName] = Var
    except:
        DictVars[str(np.round(RPM,2))+'RPM'] = {}
        for Var, VarName in zip(Vars, VarsName):

            if issparse(Var):
                DictVars[str(np.round(RPM,2))+'RPM'][VarName] = {}
                DictVars[str(np.round(RPM,2))+'RPM'][VarName]['rows'], DictVars[str(np.round(RPM,2))+'RPM'][VarName]['cols'], DictVars[str(np.round(RPM,2))+'RPM'][VarName]['data'] = find(Var)
                DictVars[str(np.round(RPM,2))+'RPM'][VarName]['Matrice'] = None
            else:
                DictVars[str(np.round(RPM,2))+'RPM'][VarName] = Var


    J.set(t, Type, **DictVars
          )

    return t


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



def Matrices(self, name , lagrange):
    from SD.sd_stoc_morse import sd_stoc_morse
    import scipy.sparse as scsp
    # Pre-traitement des informations: 
    
    refa = np.array(self.sdj.REFA.get())     # Composantes des dicctionaires 
    nu=refa[1]                            # Recuperation de la sd_num_ddl
    smos = sd_stoc_morse(nu[:14]+'.SMOS') # Recuperation de la sd_stoc_morse
    
    # Acquisition des valeurs des matrices et positionement:
    
    valm=self.sdj.VALM.get()   # Valeurs de MATASSR1
    smhc=smos.SMHC.get()       # Ligne dans MATASS des elements de valm
    smdi=smos.SMDI.get()       # Position dans valm des elements diagonaux
    
    
    # Transformation des segments de valeurs en vecteurs Python:
    
    A=np.array(valm[1])   # Valeurs matrice
    B=np.array(smhc)-1    # Renumerotation Python des elements lignes
    C=np.array(smdi)      # Diagonal values position
    E=np.array(lagrange)      # Position des multiplicateurs de Lagrange
    
    # Creation du vecteur D contenant les positions des colonnes:
    
    D=np.zeros(np.shape(A))
    for k in range(np.size(C)):
        D[C[k]-1]=k
     
    for k in range(np.size(D)-1):
        if D[-k-1]==0:
            D[-k-1]=D[-k]
    
    
    # Enregistrement des matrices stockees dans Aster sous forme sparse:
    
    m = np.size(C) # Dime_tot : Multiplicateur de lagrange et ddl probleme compris
    
    # Definition des matrices sparse avec la fonction "scipy.sparse":
    
    K0 = scsp.csr_matrix((A,(B,D)),shape=(m,m)) 
    
    # Obtention des noeuds qui ne sont pas de multiplicateurs de Lagrange:
    
    i=0
    ddl_int=[] # Indices dans K0 et M0 des ddls physiques
    while i < np.size(E):
        if E[i]!=1:
            i=i+3
        else:
            ddl_int=np.concatenate((ddl_int,[i,i+1,i+2]))
            i=i+3
    
    # Supression des lignes et colonnes des multiplicateurs de Lagrange
    ddl_int = [int(x) for x in ddl_int]
    K = (K0[:,ddl_int])[ddl_int]
    
    # Partie inferieure des matrices:
    
    n = np.size(ddl_int) # Dime_tot : Sans prendre en compte les multiplicateurs
    
    # L = [0,1,2..n-1]. Matrices diag contenant les diag de K et M
    
    L=np.zeros(n) 
    for i in range(1,n):
        L[i]=L[i-1]+1
    
    Kdiag=scsp.csr_matrix((K.diagonal(),(L,L)),shape=(n,n))
    
    K = K+K.transpose()-Kdiag
    
    return K 


def ExtrMatrixFromAster2Python(MATRICE, **kwargs):

    try:
        if kwargs['ComputeLagrange'] == True:
            P_Lagr = np.array(MATRICE.sdj.CONL.get())
            ii = np.where(P_Lagr != 1.0)[0]
    except:
        ii = kwargs['ii']

    SparseTupl = MATRICE.EXTR_MATR(sparse=True)
    SparseMatr = csr_matrix((SparseTupl[0], (SparseTupl[1], SparseTupl[2])), shape=(SparseTupl[3], SparseTupl[3]))
    MatricePy = delete_from_csr(SparseMatr,row_indices=ii, col_indices=ii)

#    import scipy.sparse as scsp
#    toto = scsp.csr_matrix(MATRICE.EXTR_MATR())
#    MatricePy2 = delete_from_csr(SparseMatr,row_indices=ii, col_indices=ii)
#
#    print(np.amax(abs(MatricePy2-MatricePy)))
#    
#
#    Matrice = Matrices(MATRICE, 'Ke', kwargs['P_Lagr'])
#
#    print(np.amax(abs(MatricePy2-Matrice)))
#    #Matrice = MATRICE.EXTR_MATR()
#    #print(Matrice)
#    #A = MatricePy.toarray()
#    #for i in range(np.shape(Matrice)[0]):
#
    #    print(np.max(np.max((abs(A[i,:] - Matrice[i,:])))))
    

    return MatricePy, ii


def ExtrVectorFromAster2Python(VECTEUR, PointsLagrange):

    VectPy = np.array(VECTEUR.sdj.VALE.get())
    VectPy = np.delete(VectPy, PointsLagrange)

    return VectPy

def DestroyAsterObjects(AsterObjectsDictionnary, DetrVars = []):
    '''Function to erase the aster objects from the memory Jeveux'''

    for AsName in AsterObjectsDictionnary.keys():

        if AsName in DetrVars:

            DETRUIRE(CONCEPT = _F(NOM = AsterObjectsDictionnary[AsName]))


def SetSparseMatrixFromCGNS(t, RPM, MatrixName, Type = '.AssembledMatrices'):
    
    DictAssembledMatrices = J.get(t, Type)
    DictStructParam = J.get(t, '.StructuralParameters')

    NNodes = DictStructParam['MeshProperties']['NNodes'][0]


    data = DictAssembledMatrices[str(np.round(RPM,2))+'RPM'][MatrixName]['data']
    rows = DictAssembledMatrices[str(np.round(RPM,2))+'RPM'][MatrixName]['rows']
    cols = DictAssembledMatrices[str(np.round(RPM,2))+'RPM'][MatrixName]['cols']

    DictAssembledMatrices[str(np.round(RPM,2))+'RPM'][MatrixName]['Matrice'] = csr_matrix((data, (rows, cols)), shape = (3*NNodes,3*NNodes))

    J.set(t, Type, **DictAssembledMatrices)

    return t

def GetSparseMatrixFromCGNS(t, RPM, MatrixName, Type = '.AssembledMatrices'):

    DictAssembledMatrices = J.get(t, Type)

    return DictAssembledMatrices[str(np.round(RPM,2))+'RPM'][MatrixName]['Matrice'][0]


def LoadSMatrixFromCGNS(t, RPM, MatrixName, Type = '.AssembledMatrices' ):

    DictAssembledMatrices = J.get(t, Type)

    if DictAssembledMatrices[str(np.round(RPM,2))+'RPM'][MatrixName]['Matrice'] == None:
        t = SetSparseMatrixFromCGNS(t, RPM, MatrixName, Type)

    SMatrice = GetSparseMatrixFromCGNS(t, RPM, MatrixName, Type)

    return SMatrice, t


def GetReducedBaseFromCGNS(t, RPM):

    DictStructParam = J.get(t, '.StructuralParameters')

    NNodes = DictStructParam['MeshProperties']['NNodes'][0]
    NModes = DictStructParam['ROMProperties']['NModes'][0]


    PHI = np.zeros((3*NNodes, NModes))

    for Mode in range(NModes):

        ModeZone = I.getNodeFromName(t, str(np.round(RPM,2))+'Mode'+str(Mode))
        
        PHI[::3, Mode], PHI[1::3, Mode], PHI[2::3, Mode] = J.getVars(ModeZone,['ModeX', 'ModeY', 'ModeZ'])

    return PHI

def VectFromROMtoFULL(PHI, qVect):
    return PHI.dot(qVect)

def PseudoInverseWithMatrix(Matrice):
    '''Calcul de la pseudo inverse de la Matrice donnee:
       A: m x n matrix
        Square: A^(-1)
        Tall (m>n): A^(-L) = (A'A)^(-1)A'
        Wide (m<n): A^(-R) = A'(AA')^(-1)
    '''
    m, n = np.shape(Matrice)
    if m == n: 
        Pinv = np.linalg.inv(Matrice)
    elif m > n:
        Pinv = np.linalg.inv((Matrice.T).dot(Matrice)).dot(Matrice.T)
    else:
        Pinv = (Matrice.T).dot(np.linalg.inv(Matrice.dot(Matrice.T)))
    return Pinv

def PseudoInverseWithModes(t, RPM, Matrice):
    '''Calcul de la pseudo inverse de Matrice en utilisant la base modale a RPM'''
    PHI = GetReducedBaseFromCGNS(t, RPM) 
    # Recuperer la base modale a RPM
    return (np.linalg.inv((PHI.T).dot(PHI)).dot(PHI.T)).dot(Matrice)


def SaveSolution2PythonDict(Solution, ForceCoeff, RPM, PHI, q_qp_qpp, fnl_q, fext_q ):
    '''Prends une solution dynamique ou statique et la garde sur un dictionnaire 
    apres avoir calcule les grandeurs dans le FOM'''

    if len(q_qp_qpp) == 1:
        Solution['%sRPM'%np.round(RPM,2)]['FCoeff%s'%ForceCoeff]['q'] = q_qp_qpp[0] 
    else:
        Solution['%sRPM'%np.round(RPM,2)]['FCoeff%s'%ForceCoeff]['q'] = q_qp_qpp[0] 
        Solution['%sRPM'%np.round(RPM,2)]['FCoeff%s'%ForceCoeff]['qp'] = q_qp_qpp[1] 
        Solution['%sRPM'%np.round(RPM,2)]['FCoeff%s'%ForceCoeff]['qpp'] = q_qp_qpp[2] 
   
    Solution['%sRPM'%np.round(RPM,2)]['FCoeff%s'%ForceCoeff]['fnl_q'] = fnl_q
    Solution['%sRPM'%np.round(RPM,2)]['FCoeff%s'%ForceCoeff]['fext_q'] = fext_q

    NameOfFullVariables = ['Displacement', 'Velocity', 'Acceleration']
    for key, it in zip(Solution['%sRPM'%np.round(RPM,2)]['FCoeff%s'%ForceCoeff].keys(), range(len(q_qp_qpp)+1)):
        if 'fnl' not in key:
            Solution['%sRPM'%np.round(RPM,2)]['FCoeff%s'%ForceCoeff][NameOfFullVariables[it]] = VectFromROMtoFULL(PHI, Solution['%sRPM'%np.round(RPM,2)]['FCoeff%s'%ForceCoeff][key])
        else:
            Solution['%sRPM'%np.round(RPM,2)]['FCoeff%s'%ForceCoeff]['fnlFull'] = PseudoInverseWithMatrix(PHI.T).dot(fnl_q)
            Solution['%sRPM'%np.round(RPM,2)]['FCoeff%s'%ForceCoeff]['fextFull'] = PseudoInverseWithMatrix(PHI.T).dot(fext_q)
    
    return Solution




def ExtForcesFromBEMT(t):

    fLE, fTE = FrocesAndMomentsFromLiftingLine2ForcesAtLETE(LiftingLine, t)

    fVect = LETEvector2FullDim(t, fLE = fLE, fTE = fTE)

    return fVect

def LETEvector2FullDim(t, fLE, fTE):

    DictStructParam = J.get(t, '.StructuralParameters')
    NLE = DictStructParam['MeshProperties']['NodesFamilies']['LeadingEdge']
    NTE = DictStructParam['MeshProperties']['NodesFamilies']['TrailingEdge']
    NNodes = DictStructParam['MeshProperties']['NNodes'][0]

    fVect = np.zeros((3*NNodes, ))

    fVect[3*NLE]   = fLE[0]
    fVect[3*NLE+1] = fLE[1]
    fVect[3*NLE+2] = fLE[2]
    fVect[3*NTE]   = fTE[0]
    fVect[3*NTE+1] = fTE[1]
    fVect[3*NTE+2] = fTE[2]

    return fVect

def GetCoordsOfTEandLE(t, RPM = None):


    InitZone = I.getNodesFromNameAndType(t, 'InitialMesh', 'Zone_t')[0]
    XCoords, YCoords, ZCoords = J.getxyz(InitZone)
    

    if RPM is not None:
        UsZone = I.getNodesFromNameAndType(t, 'U_sta'+str(np.round(RPM,2)), 'Zone_t')[0]
        upx, upy, upz = J.getVars(UsZone,['upx', 'upy', 'upz'])
        XCoords += upx
        YCoords += upy
        ZCoords += upz

    DictStructParam = J.get(t, '.StructuralParameters')
    NLE = DictStructParam['MeshProperties']['NodesFamilies']['LeadingEdge']
    NTE = DictStructParam['MeshProperties']['NodesFamilies']['TrailingEdge']

    # Leading EdgeCoords:
    XCoordsLE, YCoordsLE, ZCoordsLE =  XCoords[NLE], YCoords[NLE], ZCoords[NLE]

    # Trailing EdgeCoods:
    XCoordsTE, YCoordsTE, ZCoordsTE =  XCoords[NTE], YCoords[NTE], ZCoords[NTE]

    
    return [XCoordsLE, YCoordsLE, ZCoordsLE], [XCoordsTE, YCoordsTE, ZCoordsTE]

def GetCoordsOfTEandLEWithROMq(t, RPM = None, q = None):

    DictStructParam = J.get(t, '.StructuralParameters')
    NLE = DictStructParam['MeshProperties']['NodesFamilies']['LeadingEdge']
    NTE = DictStructParam['MeshProperties']['NodesFamilies']['TrailingEdge']

    # Get PHI:
    PHI = GetReducedBaseFromCGNS(t, RPM)
    # Get u vector u = PHI*q
    u = PHI.dot(q)

    # Get the mesh coordinates:
    InitZone = I.getNodesFromNameAndType(t, 'InitialMesh', 'Zone_t')[0]
    XCoords, YCoords, ZCoords = J.getxyz(InitZone)

    us = getVectorFromCGNS(t, 'Us', RPM)
    
    # LE Up coordintes:
    XCoordsLE = XCoords[NLE] + us[::3][NLE] + u[3*NLE]
    YCoordsLE = YCoords[NLE] + us[1::3][NLE] + u[3*NLE+1]
    ZCoordsLE = ZCoords[NLE] + us[2::3][NLE] + u[3*NLE+2] 

    # TE Up coordinates:
    XCoordsTE = XCoords[NTE] + us[::3][NTE] + u[3*NTE]
    YCoordsTE = YCoords[NTE] + us[1::3][NTE] + u[3*NTE+1]
    ZCoordsTE = ZCoords[NTE] + us[2::3][NTE] + u[3*NTE+2] 

    
    return [XCoordsLE, YCoordsLE, ZCoordsLE], [XCoordsTE, YCoordsTE, ZCoordsTE]



def GetCoordsOfTEandLEWithFOMu(t, RPM = None, u = None):

    DictStructParam = J.get(t, '.StructuralParameters')
    NLE = DictStructParam['MeshProperties']['NodesFamilies']['LeadingEdge']
    NTE = DictStructParam['MeshProperties']['NodesFamilies']['TrailingEdge']

    # Get the mesh coordinates:
    InitZone = I.getNodesFromNameAndType(t, 'InitialMesh', 'Zone_t')[0]
    XCoords, YCoords, ZCoords = J.getxyz(InitZone)

    us = getVectorFromCGNS(t, 'Us', RPM)
    
    # LE Up coordintes:
    XCoordsLE = XCoords[NLE] + us[::3][NLE] + u[3*NLE]
    YCoordsLE = YCoords[NLE] + us[1::3][NLE] + u[3*NLE+1]
    ZCoordsLE = ZCoords[NLE] + us[2::3][NLE] + u[3*NLE+2] 

    # TE Up coordinates:
    XCoordsTE = XCoords[NTE] + us[::3][NTE] + u[3*NTE]
    YCoordsTE = YCoords[NTE] + us[1::3][NTE] + u[3*NTE+1]
    ZCoordsTE = ZCoords[NTE] + us[2::3][NTE] + u[3*NTE+2] 

    
    return [XCoordsLE, YCoordsLE, ZCoordsLE], [XCoordsTE, YCoordsTE, ZCoordsTE]


def FrocesAndMomentsFromLiftingLine2ForcesAtLETE(t, RPM, q = None):

    LiftingLine = I.getNodeFromName(t, 'LiftingLine')

    # Compute the geometrical variables (Coordinates and s):
    try:
        LECoord, TECoord = GetCoordsOfTEandLE(t, RPM)
    except:
        DictStructParam = J.get(t, '.StructuralParameters')
        if len(q) == DictStructParam['ROMProperties']['NModes'][0]:

            LECoord, TECoord = GetCoordsOfTEandLEWithROMq(t, RPM, q)
        else:
            LECoord, TECoord = GetCoordsOfTEandLEWithFOMu(t, RPM, q)

    LEZone = J.createZone('LEZone', LECoord, ['CoordinateX', 'CoordinateY', 'CoordinateZ'])
    sLE = W.gets(LEZone)
    # Implement --> W.getS using a Zone as input
    #_,sLE,_ = J.getDistributionFromHeterogeneousInput__(abs(LECoord[0])) # Span Along X
    #_,sTE,_ = J.getDistributionFromHeterogeneousInput__(abs(TECoord[2])) # Span Along Z

    Span, s = J.getVars(LiftingLine, ['Span', 's'])

    sLL=J.interpolate__(AbscissaRequest=abs(LECoord[0]), AbscissaData=Span, ValuesData=s,
                  Law='akima')


    # First the LL forces (fx, fy, fz) are transferred to the LE and TE

    # Get and interpolate the forces in the LL:
    DictInterpolatedVars = dict()
    InterpolatedNames = ['Chord','Twist','fx','fy','fz', 'mx','my','mz',
                         'tx', 'ty', 'tz', 'bx', 'by', 'bz', 'nx', 'ny', 'nz']


    for VarName in InterpolatedNames:
        VariableData = J.getVars(LiftingLine,[VarName])[0]

        DictInterpolatedVars[VarName] = J.interpolate__(AbscissaRequest=sLE,
                                                    AbscissaData=s,
                                                    ValuesData=VariableData,
                                                    Law='akima'
                                                    )


    # Transfer the LL forces towards the LE and TE (No moment forces):

    FxLE, FxTE = 3./4. * DictInterpolatedVars['fx'], 1./4. * DictInterpolatedVars['fx']
    FyLE, FyTE = 3./4. * DictInterpolatedVars['fy'], 1./4. * DictInterpolatedVars['fy']
    FzLE, FzTE = 3./4. * DictInterpolatedVars['fz'], 1./4. * DictInterpolatedVars['fz']

    # Second, the LL moments (mx, my, mz) are transferred to the LE and TE as forces

    Mosc = np.sqrt(DictInterpolatedVars['mx']*DictInterpolatedVars['mx'] + DictInterpolatedVars['my']*DictInterpolatedVars['my'] + DictInterpolatedVars['mz']*DictInterpolatedVars['mz'])

    Cosdir = [DictInterpolatedVars['mx']/Mosc, DictInterpolatedVars['my']/Mosc, DictInterpolatedVars['mz']/Mosc]
    #print(Cosdir)

    #print('t')
    #print(DictInterpolatedVars['tx'])
    #print(DictInterpolatedVars['ty'])
    #print(DictInterpolatedVars['tz'])
    #
    #print('b')
    #print(DictInterpolatedVars['bx'])
    #print(DictInterpolatedVars['by'])
    #print(DictInterpolatedVars['bz'])
    #
    #print('n')
    #print(DictInterpolatedVars['nx'])
    #print(DictInterpolatedVars['ny'])
    #print(DictInterpolatedVars['nz'])
    
    
    # Perpendicular forces with respect to the chord at TE and LE:

    FoscLE,  FoscTE = -Mosc / DictInterpolatedVars['Chord'] , Mosc / DictInterpolatedVars['Chord']



    #import matplotlib.pyplot as plt
#    import StructuralShortCuts as SJ
#    #plt.plot(sLE, FoscLE, label = 'LE')
#    #plt.plot(sLE, FoscTE, label = 'TE')
#    #plt.plot(sLE, Mosc, label = 'Mosc')
#    #plt.grid()
#    #plt.legend()
#    #plt.savefig('/scratchm/mbalmase/Spiro/3_Update4MOLA/CouplingWF/Fosc.pdf')
#    #plt.close()
#    # Change the coord system to b and n:
#    #print(DictInterpolatedVars['Twist'])
    FbLE, FnLE  =  FoscLE*np.sin(DictInterpolatedVars['Twist']*np.pi/180. ), FoscLE*np.cos(DictInterpolatedVars['Twist']*np.pi/180. ),
    FbTE, FnTE  =  FoscTE*np.sin(DictInterpolatedVars['Twist']*np.pi/180. ), FoscTE*np.cos(DictInterpolatedVars['Twist']*np.pi/180. )




    FxLEm = FbLE * DictInterpolatedVars['bx'] + FnLE * DictInterpolatedVars['nx']
    FyLEm = FbLE * DictInterpolatedVars['by'] + FnLE * DictInterpolatedVars['ny']
    FzLEm = FbLE * DictInterpolatedVars['bz'] + FnLE * DictInterpolatedVars['nz']
    FxTEm = FbTE * DictInterpolatedVars['bx'] + FnTE * DictInterpolatedVars['nx']
    FyTEm = FbTE * DictInterpolatedVars['by'] + FnTE * DictInterpolatedVars['ny']
    FzTEm = FbTE * DictInterpolatedVars['bz'] + FnTE * DictInterpolatedVars['nz']


    return [FxLE+FxLEm, FyLE+FyLEm, FzLE+FzLEm],[FxTE+FxTEm, FyTE+FyTEm, FzTE+FzTEm], [[FxLE, FyLE, FzLE],[FxTE, FyTE, FzTE],[FxLEm, FyLEm, FzLEm],[FxTEm, FyTEm, FzTEm]]


def LLCoordsFromLETE(LECoord, TECoord, s,sLE):

    #_,sLE,_ = J.getDistributionFromHeterogeneousInput__(abs(LECoord[0])) # Span along X

    LECoordLLx = J.interpolate__(AbscissaRequest=s,
                                AbscissaData=sLE,
                                ValuesData=LECoord[0],
                                Law='akima'
                                )

    TECoordLLx = J.interpolate__(AbscissaRequest=s,
                                AbscissaData=sLE,
                                ValuesData=TECoord[0],
                                Law='akima'
                                )
    LECoordLLy = J.interpolate__(AbscissaRequest=s,
                                AbscissaData=sLE,
                                ValuesData=LECoord[1],
                                Law='akima'
                                )

    TECoordLLy = J.interpolate__(AbscissaRequest=s,
                                AbscissaData=sLE,
                                ValuesData=TECoord[1],
                                Law='akima'
                                )
    LECoordLLz = J.interpolate__(AbscissaRequest=s,
                                AbscissaData=sLE,
                                ValuesData=LECoord[2],
                                Law='akima'
                                )

    TECoordLLz = J.interpolate__(AbscissaRequest=s,
                                AbscissaData=sLE,
                                ValuesData=TECoord[2],
                                Law='akima'
                                )
    LLx = TECoordLLx + 0.75 * (LECoordLLx - TECoordLLx)
    LLy = TECoordLLy + 0.75 * (LECoordLLy - TECoordLLy)
    LLz = TECoordLLz + 0.75 * (LECoordLLz - TECoordLLz)

    return LLx, LLy, LLz, sLE

def updateLLKinematics(t, RPM):

    LiftingLine = I.getNodeFromName(t, 'LiftingLine')

    DictStructParam = J.get(t, '.StructuralParameters')

    LL.setKinematicsUsingConstantRPM(LiftingLine, RotationCenter=DictStructParam['RotatingProperties']['RotationCenter'],
                                  RotationAxis=DictStructParam['RotatingProperties']['AxeRotation'], RPM=RPM,
                                  RightHandRuleRotation=DictStructParam['RotatingProperties']['RightHandRuleRotation'])

    I._addChild(t, LiftingLine)

    return t


def updateLiftingLineFromStructureLETE(t, RPM, q = None):

    LiftingLine = I.getNodeFromName(t, 'LiftingLine')
    # Load the coodinates of the LE and TE:
    Span, Chord, Twist, s, Dihedral, Sweep = J.getVars(LiftingLine, ["Span", "Chord", "Twist", "s", "Dihedral", "Sweep"])

    try:
        LECoord, TECoord = GetCoordsOfTEandLE(t, RPM)
    except:
        DictStructParam = J.get(t, '.StructuralParameters')
        if len(q) == DictStructParam['ROMProperties']['NModes'][0]:

            LECoord, TECoord = GetCoordsOfTEandLEWithROMq(t, RPM, q)
        else:
            LECoord, TECoord = GetCoordsOfTEandLEWithFOMu(t, RPM, q)

    LEZone = J.createZone('LEZone', LECoord, ['CoordinateX', 'CoordinateY', 'CoordinateZ'])
    sLE = W.gets(LEZone)

    x,y,z = J.getxyz(LiftingLine)
    LLx, LLy, LLz, sLE = LLCoordsFromLETE(LECoord, TECoord,s, sLE)
    
    #x,y,z = J.getxyz(LiftingLine)
    
    x[:], y[:], z[:] = LLx, LLy, LLz
    J.invokeFields(LiftingLine, ['tx','ty','tz', 'nx', 'ny', 'nz', 'bx', 'by', 'bz'])
    
    RotationAxis, RotationCenter, Dir = LL.getRotationAxisCenterAndDirFromKinematics(LiftingLine)
   
    
    _,bxyz,_ = LL.updateLocalFrame(LiftingLine)


    #_,sLE,_ = J.getDistributionFromHeterogeneousInput__(abs(LECoord[0])) # Span along X
    
    ChordSMagLL = np.sqrt((LECoord[0]-TECoord[0])*(LECoord[0]-TECoord[0])+
                        (LECoord[1]-TECoord[1])*(LECoord[1]-TECoord[1])+
                        (LECoord[2]-TECoord[2])*(LECoord[2]-TECoord[2]))
    VectChord = np.array([LECoord[0]-TECoord[0], LECoord[1]-TECoord[1],
                          LECoord[2]-TECoord[2]])


    # Compute the local frame:
    

    

    # Translate the structural LL towards the real LL and compute Twist:

    ChordLL = J.interpolate__(AbscissaRequest=s,
                              AbscissaData=sLE,
                              ValuesData=ChordSMagLL,
                              Law='akima'
                              )

    VectChordLLx = J.interpolate__(AbscissaRequest=s,
                                   AbscissaData=sLE,
                                   ValuesData=VectChord[0],
                                   Law='akima'
                                   )
    VectChordLLy = J.interpolate__(AbscissaRequest=s,
                                   AbscissaData=sLE,
                                   ValuesData=VectChord[1],
                                   Law='akima'
                                   )
    VectChordLLz = J.interpolate__(AbscissaRequest=s,
                                   AbscissaData=sLE,
                                   ValuesData=VectChord[2],
                                   Law='akima'
                                   )


    UnitVectCordLL = [VectChordLLx, VectChordLLy, VectChordLLz]/ChordLL


    TwstVg = []
    for i in range(len(bxyz[0])):
        TwstVg.append(90.-vg.angle(np.array([bxyz[0][i], bxyz[1][i], bxyz[2][i]]), np.array([UnitVectCordLL[0][i], UnitVectCordLL[1][i], UnitVectCordLL[2][i]])))
        
    
    Chord[:], Twist[:] = ChordLL, TwstVg
    Dihedral[:], Sweep[:] = LLx, -LLy


    I._addChild(t, LiftingLine)

    return t

def totuple(a):
    try:
        return tuple(totuple(i) for i in a)
    except TypeError:
        return a



def locVector2FullDimension(t, Vector, NodeNumber):

    DictStructParam = J.get(t, '.StructuralParameters')
    NNodes = DictStructParam['MeshProperties']['NNodes'][0]
    
    # Initialize the vector:

    FullVector = np.zeros((3*NNodes, ))
    
    FullVector[3*NodeNumber] = Vector[0]
    FullVector[3*NodeNumber+1] = Vector[1]
    FullVector[3*NodeNumber+2] = Vector[2]

    return FullVector



def VectorXYZ2FullDimension(t, Vector):
    '''Vector = [[CoordX], [CoordY], [CoordZ]]'''

    DictStructParam = J.get(t, '.StructuralParameters')
    NNodes = DictStructParam['MeshProperties']['NNodes'][0]
    
    # Initialize the vector:

    FullVector = np.zeros((3*NNodes, ))
    
    FullVector[0::3] = Vector[0]
    FullVector[1::3] = Vector[1]
    FullVector[2::3] = Vector[2]

    return FullVector


def Solution2RotatoryFrame(t, Solution):

    for RPMKey in Solution.keys():

        RPM = float(RPMKey[:-3])

        try:
            UsVect = getUsVectorFromCGNS(t, RPM)
            UsVectFull = VectorXYZ2FullDimension(t, UsVect)
        except:
            print(WARN+'Us vector nor present in the tree within a Zone_t'+ENDC)
            UsVectFull = getVectorFromCGNS(t, 'Us', RPM)


        for FcoeffKey in Solution[RPMKey].keys():
            Displ = np.zeros(np.shape(Solution[RPMKey][FcoeffKey]['Displacement']))
            for column in range(len(Solution[RPMKey][FcoeffKey]['Displacement'][0,:])):
                Displ[:,column] =  UsVectFull + Solution[RPMKey][FcoeffKey]['Displacement'][:,column]
            Solution[RPMKey][FcoeffKey]['UpDisplacement'] = Displ
        
    return Solution

def SolutionFromRotatoryFrame(t, Solution):

    for RPMKey in Solution.keys():

        RPM = float(RPMKey[:-3])
        
        try:
            UsVect = getUsVectorFromCGNS(t, RPM)
            UsVectFull = VectorXYZ2FullDimension(t, UsVect)
        except:
            print(WARN+'Us vector nor present in the tree within a Zone_t'+ENDC)
            UsVectFull = getVectorFromCGNS(t, 'Us', RPM)

       
        for FcoeffKey in Solution[RPMKey].keys():
            Displ = np.zeros(np.shape(Solution[RPMKey][FcoeffKey]['Displacement']))
            try:
                for column in range(len(Solution[RPMKey][FcoeffKey]['Displacement'][0,:])):
                    Displ[:,column] =  Solution[RPMKey][FcoeffKey]['Displacement'][:,column] - UsVectFull
            except:
                Displ =  Solution[RPMKey][FcoeffKey]['Displacement'] - UsVectFull
        
            Solution[RPMKey][FcoeffKey]['UpDisplacement'] = Solution[RPMKey][FcoeffKey]['Displacement']
            Solution[RPMKey][FcoeffKey]['Displacement'] = Displ
        
    return Solution




def TranslateNumpyLoadingVector2AsterList(t, LoadVector):
    
    DictStructParam = J.get(t, '.StructuralParameters')
    
    n_enc =  DictStructParam['MeshProperties']['NodesFamilies']['Node_Encastrement']

    l_prepa = [] 
    for n in range(DictStructParam['MeshProperties']['NNodes'][0]):
        if not(n in n_enc):
            ap = _F(NOEUD = 'N'+ str(n + 1),FX = LoadVector[n*3], FY = LoadVector[n*3+ 1], FZ = LoadVector[n*3 + 2],)
            l_prepa.append(ap)

    return l_prepa




def ComputeLoadingType(t):

    DictSimulaParam = J.get(t, '.SimulationParameters')
      
    TypeOfLoading = DictSimulaParam['LoadingProperties']['LoadingType']
    TypeOfSolver  = DictSimulaParam['IntegrationProperties']['SolverType']
    

    if TypeOfSolver == 'Static':

        NincrIter = DictSimulaParam['IntegrationProperties']['StaticSteps'][0]
    
    elif TypeOfSolver == 'Dynamic':

        NincrIter = DictSimulaParam['LoadingProperties']['TimeProperties']['NItera'][0]


    # Compute the form of the f(t) function with max value 1.

    if TypeOfLoading == 'Ramp':
        
        LoadingTypeVector = np.linspace(0,1,NincrIter)

    elif TypeOfLoading == 'Sinusoidal':
        # Sinusoidal loading
        pass

    elif TypeOfLoading == 'Aeroelastic':

        pass


    return LoadingTypeVector      

def ComputeShapeVectorAndMaxForce(t, LoadingVectorLocation = 'FromFile'):

    if LoadingVectorLocation == 'FromFile':
        FullLoadingVector = C.convertFile2PyTree('InputData/Loading/FullLoadingVector.cgns','bin_adf')
    
    DictFullLoadingVector = J.get(FullLoadingVector, '.LoadingVector')

    Fmax = np.max(abs(DictFullLoadingVector['LoadingVector']))
    ShapeFunction = DictFullLoadingVector['LoadingVector']/Fmax
    
    return Fmax, ShapeFunction


def BuildExterForcesShapeVectorAndLoadingTypeVector(t, FOM = False):
    '''Building the components for the loading vector:
          fe = f(t)*PHI.T*Psi*Fmax*alphaF

          fe: External force
          f(t): Loading Type (Ramp, Sinusoidal...)
          PHI : Reduced basis
          PSI : Shape of the loading vector (normalized with respect to the maximum value)
          Fmax: Maximum value of the loading
          alphaF: Force intensity coefficient # Not here, implemented on the N-R procedure'''
   

    DictStructParam = J.get(t, '.StructuralParameters')
    DictSimulaParam = J.get(t, '.SimulationParameters')

    DictOfLoading = {}
    DictOfLoading['TimeFuntionVector'] = ComputeLoadingType(t)
    DictOfLoading['Fmax'], DictOfLoading['ShapeFunction'] = ComputeShapeVectorAndMaxForce(t)
    DictOfLoading['ShapeFunctionProj'] = {}    
    for RPMValue in  DictStructParam['RotatingProperties']['RPMs']:
        if not FOM:
            PHI = GetReducedBaseFromCGNS(t, RPMValue)
    
            DictOfLoading['ShapeFunctionProj'][str(int(RPMValue))+'RPM'] = (PHI.T).dot(DictOfLoading['ShapeFunction'])
    
        else:
            DictOfLoading['ShapeFunctionProj'][str(int(RPMValue))+'RPM'] = DictOfLoading['ShapeFunction']
            
    DictSimulaParam['LoadingProperties']['ExternalForcesVector'] = dict(**DictOfLoading)

    J.set(t,'.SimulationParameters', **dict(DictSimulaParam)
                                            )

    return t      

def ComputeLoadingFromTimeOrIncrement(t, RPM, TimeIncr):

    DictSimulaParam = J.get(t, '.SimulationParameters')
    LoadingVector = DictSimulaParam['LoadingProperties']['ExternalForcesVector']['TimeFuntionVector'][TimeIncr] * DictSimulaParam['LoadingProperties']['ExternalForcesVector']['ShapeFunctionProj'][str(int(RPM))+'RPM'] * DictSimulaParam['LoadingProperties']['ExternalForcesVector']['Fmax']

    return LoadingVector



def ComputeSolutionCGNS(t, Solution):
    tSol = I.copyTree(t)
    J.set(tSol, '.Solution', **Solution)
    
    try:
        Calc = range(len(Solution[list(Solution.keys())[0]][list(Solution[list(Solution.keys())[0]].keys())[0]]['Displacement'][0,:]))
        Matrice = True
    except:
        Calc = range(len(Solution[list(Solution.keys())[0]][list(Solution[list(Solution.keys())[0]].keys())[0]]['Displacement'][:]))
        Matrice = False

    MaxIt = range(len(Solution[list(Solution.keys())[0]][list(Solution[list(Solution.keys())[0]].keys())[0]]))
        
    for Iteration in MaxIt:
        NewBase = I.newCGNSBase()
        NewBase[0] = 'Iteration%s'%Iteration
        
        for RPMKey in Solution.keys():
            for FcoeffKey in Solution[RPMKey].keys():
    
                ZoneName = 'It%s_'%Iteration + RPMKey +'_'+ FcoeffKey
                
                if Matrice:
                    NewZone, _ = CreateNewSolutionFromNdArray(t, FieldDataArray = [Solution[RPMKey][FcoeffKey]['UpDisplacement'][::3,Iteration], Solution[RPMKey][FcoeffKey]['UpDisplacement'][1::3,Iteration], Solution[RPMKey][FcoeffKey]['UpDisplacement'][2::3,Iteration]],
                                        ZoneName=ZoneName,
                                        FieldNames = ['Upx', 'Upy', 'Upz'],
                                        Depl = True,
                                        DefByField = None)
                else:
                    NewZone, _ = CreateNewSolutionFromNdArray(t, FieldDataArray = [Solution[RPMKey][FcoeffKey]['UpDisplacement'][::3], Solution[RPMKey][FcoeffKey]['UpDisplacement'][1::3], Solution[RPMKey][FcoeffKey]['UpDisplacement'][2::3]],
                                        ZoneName=ZoneName,
                                        FieldNames = ['Upx', 'Upy', 'Upz'],
                                        Depl = True,
                                        DefByField = None)

                VarNames = ['Ux', 'Uy', 'Uz', 'fnlX','fnlY','fnlZ', 'fextX','fextY','fextZ', 'Vx', 'Vy', 'Vz', 'Ax', 'Ay', 'Az']
                SolNames = ['Displacement', 'fnlFull', 'fextFull', 'Velocity', 'Acceleration']
                Vars = J.invokeFields(NewZone, VarNames)
                
                if len(Solution[RPMKey][FcoeffKey].keys()) > 7:
                    LenVars = len(SolNames)
                else:
                    LenVars = len(SolNames) - 2
                
                if Matrice:
                    for Sol, loc in zip(SolNames, range(LenVars)):
        
                        Vars[3*loc][:]    = Solution[RPMKey][FcoeffKey][SolNames[loc]][::3,Iteration]
                        Vars[3*loc +1][:] = Solution[RPMKey][FcoeffKey][SolNames[loc]][1::3,Iteration]
                        Vars[3*loc +2][:] = Solution[RPMKey][FcoeffKey][SolNames[loc]][2::3,Iteration]
                else:
                    for Sol, loc in zip(SolNames, range(LenVars)):
        
                        Vars[3*loc][:]    = Solution[RPMKey][FcoeffKey][SolNames[loc]][::3]
                        Vars[3*loc +1][:] = Solution[RPMKey][FcoeffKey][SolNames[loc]][1::3]
                        Vars[3*loc +2][:] = Solution[RPMKey][FcoeffKey][SolNames[loc]][2::3]
                    
                I._addChild(NewBase, NewZone)
    
        I._addChild(tSol, NewBase)
    
    return tSol 