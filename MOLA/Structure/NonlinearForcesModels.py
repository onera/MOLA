'''
MOLA - NonlinearForcesModels.py 

This module defins some functions to compute the NL ROM models Offline Stage.

Furthermore it provides functions that interact with Code_Aster.

First creation:
30/09/2021 - M. Balmaseda et T. Flament
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
import MOLA.InternalShortcuts as J

import MOLA.Structure.ShortCuts as SJ
import MOLA.Structure.Models as SM



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



    UpFromOmegaAndFe = SM.ExtrUpFromAsterSOLUwithOmegaFe(t, RPM, **dict(SOLU = SOLU))

    SJ.DestroyAsterObjects(dict(**dict(Cfd = Cfd, SOLU= SOLU, RAMPE = RAMPE, L_INST = L_INST)),  
                           DetrVars = ['Cfd', 'SOLU', 'RAMPE', 'L_INST',
                                      ])

    
    return  UpFromOmegaAndFe

def ComputeMatrULambda4ICE(t, RPM, **kwargs):

    DictStructParam = J.get(t, '.StructuralParameters')
    DictAsseVectors = J.get(t, '.AssembledVectors')

    lambda_vect = DictStructParam['ROMProperties']['lambda_vect']
# If ICE generation as STEP logic:
    if DictStructParam['ROMProperties']['ICELoadingType'] == 'STEPType':
    
        nr = DictStructParam['ROMProperties']['NModes'][0]
        len_quad = nr*(nr+1)/2
        len_cubic = (nr**3 + 3*nr**2 + 2*nr)/6
        lenQ = int(len_quad + len_cubic)
       
        MatrUpLambda = np.zeros((DictStructParam['MeshProperties']['Nddl'][0], lenQ))
        MatrFLambda = np.zeros((DictStructParam['MeshProperties']['Nddl'][0], lenQ))
        
        PHI = SJ.GetReducedBaseFromCGNS(t, RPM)

        # Combinaison d'un seul mode
        count = -1
        for i in range(DictStructParam['ROMProperties']['NModes'][0]):
            count += 1         
            print(FAIL + 'Counter: %s/%s'%(count+1, lenQ) +ENDC)
            f1 = np.dot(PHI[: , i],  lambda_vect[i])
            f2 = -1. * f1 
            
            MatrUpLambda[:, count] = ComputeStaticU4GivenLoading(t, RPM, f1, **kwargs)
            MatrFLambda[:, count] = f1
            count += 1
            print(FAIL + 'Counter: %s/%s'%(count+1, lenQ) +ENDC)
            MatrUpLambda[:, count] = ComputeStaticU4GivenLoading(t, RPM, f2, **kwargs)
            MatrFLambda[:, count] = f2
         
         # Combinaison de deux modes   
                 
        for i in range(DictStructParam['ROMProperties']['NModes'][0]):
            for j in range(i+1, DictStructParam['ROMProperties']['NModes'][0]):
                count += 1         
                
                f3 = PHI[:,i] * lambda_vect[i] + PHI[: , j] * lambda_vect[j]   
                f4 = -1 *(PHI[:,i] * lambda_vect[i]) - PHI[: , j] * lambda_vect[j]
                f5 = PHI[:,i] * lambda_vect[i] - PHI[: , j] * lambda_vect[j]
 
                MatrUpLambda[:, count] = ComputeStaticU4GivenLoading(t, RPM, f3, **kwargs)
                MatrFLambda[:, count] = f3
                count += 1
                print(FAIL + 'Counter: %s/%s'%(count+1, lenQ) +ENDC)
                MatrUpLambda[:, count] = ComputeStaticU4GivenLoading(t, RPM, f4, **kwargs)
                MatrFLambda[:, count] = f4
                count += 1
                print(FAIL + 'Counter: %s/%s'%(count+1, lenQ) +ENDC)
                MatrUpLambda[:, count] = ComputeStaticU4GivenLoading(t, RPM, f5, **kwargs)
                MatrFLambda[:, count] = f5
             
                       
                     
        # Combinaison de 3 modes
     
        for i in range(DictStructParam['ROMProperties']['NModes'][0]):
            for j in range(i+1, DictStructParam['ROMProperties']['NModes'][0]):
                for k in range(j+1, DictStructParam['ROMProperties']['NModes'][0]):
                    count += 1 
                    print(FAIL + 'Counter: %s/%s'%(count+1, lenQ) +ENDC)
                    f6 = PHI[:,i] * lambda_vect[i] + PHI[: , j] * lambda_vect[j] + PHI[: , k] * lambda_vect[k]   
                    MatrFLambda[:, count] = f6              
                    MatrUpLambda[:, count] = ComputeStaticU4GivenLoading(t, RPM, f6, **kwargs)
        

        # Compute rotating matrix to substract from MatrUpLambda:

        MatrUs = DictAsseVectors[str(np.round(RPM)) + 'RPM']['Us'].reshape((DictStructParam['MeshProperties']['Nddl'][0], 1)) * np.ones(np.shape(MatrUpLambda))
        
    return MatrUpLambda - MatrUs, MatrFLambda  

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


def ComputeForceMatrix4ICE(t, RPM, MatrULambda, MatrFLambda):
    '''Compute the nonlinear forces related to MatrULambda'''
    
    GnlULambda = np.zeros(np.shape(MatrULambda))

    Komeg,_ = SJ.LoadSMatrixFromCGNS(t, RPM, 'Komeg', Type = '.AssembledMatrices' )

    for i in range(len(MatrFLambda[0,:])):
        
        GnlULambda[:, i] = MatrFLambda[:, i] - Komeg.dot(MatrULambda[:,i])

    return GnlULambda


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

def ComputeAijAndBijmCoefficients(t, RPM, FnlLambdaMatrix, PinvMatrQLambda):
    DictStructParam = J.get(t, '.StructuralParameters')
    PHI = SJ.GetReducedBaseFromCGNS(t,RPM)
    NModes = DictStructParam['ROMProperties']['NModes'][0]
    Aij, Bijm = np.zeros((NModes, NModes, NModes)) , np.zeros((NModes, NModes, NModes, NModes))

    for mode in range(DictStructParam['ROMProperties']['NModes'][0]):

        bk = (PHI[:,mode].dot(FnlLambdaMatrix)).T
        
        Xk = PinvMatrQLambda.dot(bk)

        Aij, Bijm = extract_X(Xk,NModes,Aij,Bijm,mode)

    return Aij, Bijm


def ComputeULambda4ICE(t, RPM, **kwargs):

    DictStructParam = J.get(t, '.StructuralParameters')
    DictSimulaParam = J.get(t, '.StructuralParameters')

       
    # Calcul de tous les cas statiques avec les forces imposees:
    MatrULambda, MatrFLambda = ComputeMatrULambda4ICE(t, RPM, **kwargs)

    # Compute the pseudo inverse of MatrULambda with respect to PHI:
    MatrQLambda = SJ.PseudoInverseWithModes(t, RPM, MatrULambda)
    
    # Compute the pseudo inverse of MatrQlambda to solve the unknowns coefficient:
    PinvMatrQLambda = ComputeQMatrix4ICE(DictStructParam['ROMProperties']['NModes'][0], MatrQLambda)

    # Compute the nonlinear forces matrice: 

    FnlLambdaMatrix = ComputeForceMatrix4ICE(t, RPM, MatrULambda, MatrFLambda) 

    # Compute the Aij^k and the Bijm^k: 
    
    Aij, Bijm = ComputeAijAndBijmCoefficients(t, RPM, FnlLambdaMatrix, PinvMatrQLambda)

    DictOfCoefficients = J.get(t,'.InternalForcesCoefficients')
    DictOfCoefficients[str(int(RPM))+'RPM'] = dict(Type = 'ICE',
                                                   Aij  = Aij,
                                                   Bijm = Bijm)



    J.set(t,'.InternalForcesCoefficients', **dict(DictOfCoefficients)
                                            )
         

    return t



def ComputeNLCoefficients(t, RPM, **kwargs):
    
    DictStructParam = J.get(t, '.StructuralParameters')
    # ICE method:
    if DictStructParam['ROMProperties']['ROMForceType'] == 'ICE':
        t = ComputeULambda4ICE(t, RPM, **kwargs) 
        
    return t    



def CalcFnl_IC(Beta,Gamma,qvect):
    ''' function computing the projected purely non-linear forces fnl as a 3rd order polynomial function of the generalised coordinates
    qvect is a column vector
        Note: 
               Beta = Aij
               Gamma = Bijm '''
    
    #nq = np.shape(qvect)[0]

    q = np.ravel(qvect) # becomes a line vector
    
    nq = len(q)

    fnl_IC = np.zeros((nq,1))

    for k in range(0,nq):
        fnl_k = 0

        for i in range(0,nq):
            for j in range(i,nq):
                qi = q[i]
                qj = q[j]
                fnl_k = fnl_k + Beta[k,i,j]*qi*qj
        
        for i in range(0,nq):
            for j in range(i,nq):
                for m in range(j,nq):
                    qi = q[i]
                    qj = q[j]
                    qm = q[m]
                    fnl_k = fnl_k + Gamma[k,i,j,m]*qi*qj*qm

        fnl_IC[k,0] = fnl_k

    return fnl_IC
        

def CalcKnl_IC_kl(Beta,Gamma,qvect,k,l):
    ''' Function computing the term of the line k and the column l of the matrix derivative of the projected IC non-linear forces,
    qvect is the column vector of the generalised coordinates 
        Note: 
               Beta = Aij
               Gamma = Bijm'''

    Knl_IC_kl = 0
    #nq = np.shape(qvect)[0]

    q =  qvect.ravel()
    nq = len(q)

    # derivatives of the quadratic terms ========================

    # of the terms in ql*qi (l in first position)
    if l < nq - 1:
        for j in range(l+1,nq): # until nq-1 included
            Knl_IC_kl = Knl_IC_kl + Beta[k,l,j]*q[j]
    
    # of the terms in qj*ql (l in second position)
    if l > 0:
        for i in range(0,l):  # until l-1 included
            Knl_IC_kl = Knl_IC_kl + Beta[k,i,l]*q[i]
    
    # of the terms in ql**2
    Knl_IC_kl = Knl_IC_kl + 2*Beta[k,l,l]*q[l]

    # derivatives of the cubic terms ===========================

    # of the terms ql*qj*qm (l < j < m)
    if l < nq - 2: 
        for j in range(l+1,nq-1): # until nq - 2 included
            for m in range(j+1,nq): # until nq - 1 included
                Knl_IC_kl = Knl_IC_kl + Gamma[k,l,j,m]*q[j]*q[m]
    
    # of the terms qi*ql*qm (i < l < m)
    if (l > 0) and (l < nq - 1):
        for i in range(0,l): # until l-1 included
            for m in range(l+1,nq): # until nq - 1 included
                Knl_IC_kl = Knl_IC_kl + Gamma[k,i,l,m]*q[i]*q[m]
    
    # of the terms qi*qj*ql (i < j < l)
    if l > 1:
        for i in range(0,l-1): # until l-2 included
            for j in range(i+1,l): # until l-1 included
                Knl_IC_kl = Knl_IC_kl + Gamma[k,i,j,l]*q[i]*q[j]
    
    # of the doublets (ql**2)*qm
    if l < nq - 1:
        for m in range(l+1,nq): # until nq-1 included
            Knl_IC_kl = Knl_IC_kl + 2*Gamma[k,l,l,m]*q[m]*q[l]
    
    # of the doublets qi*(ql**2)
    if l > 0:
        for i in range(0,l): # until l-1 included
            Knl_IC_kl = Knl_IC_kl + 2*Gamma[k,i,l,l]*q[i]*q[l]
    
    # of the triplets ql**3
    Knl_IC_kl = Knl_IC_kl + 3*Gamma[k,l,l,l]*q[l]**2

    return Knl_IC_kl


def CalcKNLproj_IC(Beta,Gamma,q):
    ''' Function returning the matrix derivative of the projected IC non-linear forces,
    qvect is the column vector of the generalised coordinates 
        Note: 
               Beta = Aij
               Gamma = Bijm         '''
    
    #nq = np.shape(q)[0]
    
    nq = len(q)

    KNLproj_IC = np.zeros((nq,nq))

    for k in range(0,nq):
        for l in range(0,nq):
            KNLproj_IC[k,l] = CalcKnl_IC_kl(Beta,Gamma,q,k,l)

    return KNLproj_IC


def fnl_Proj(t, q, Aij, Bijm):

    DictStructParam = J.get(t, '.StructuralParameters') 
    
    ROMForceType = DictStructParam['ROMProperties']['ROMForceType']
    

    if ROMForceType == 'Linear':

       fnl_Proj = np.zeros((len(q),1)) 

    elif ROMForceType == 'ICE':

       fnl_Proj = CalcFnl_IC(Aij,Bijm,q)   

    return fnl_Proj


def Knl_Jacobian_Proj(t, q, Aij, Bijm):
    
    DictStructParam = J.get(t, '.StructuralParameters') 
    
    ROMForceType = DictStructParam['ROMProperties']['ROMForceType']
    
    if ROMForceType == 'Linear':

       Knl_Jacob_Proj = np.zeros((len(q),len(q))) 

    elif ROMForceType == 'ICE':

       Knl_Jacob_Proj = CalcKNLproj_IC(Aij,Bijm,q)

    return Knl_Jacob_Proj