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
from itertools import combinations_with_replacement

# MOLA modules
import Converter.Internal as I
import Converter.PyTree as C

from .. import InternalShortcuts as J

from . import ShortCuts as SJ
from . import Models as SM

#
#  --> Moved to Models.py
#
#def ComputeStaticU4GivenLoading(t, RPM, LoadVector, **kwargs):
#    
#
#    DictStructParam = J.get(t, '.StructuralParameters')
#    DictSimulaParam = J.get(t, '.SimulationParameters')
#
#    ListeLoading = SJ.TranslateNumpyLoadingVector2AsterList(t, LoadVector)
#    
#
#    RAMPE = DEFI_FONCTION(NOM_PARA = 'INST',
#                          VALE = (0.0,0.0,1.0,1.0),
#                          PROL_DROITE = 'CONSTANT',
#                          PROL_GAUCHE = 'CONSTANT',
#                          INTERPOL = 'LIN'
#                          );
#
#
#    Cfd = AFFE_CHAR_MECA(MODELE = kwargs['MODELE'],
#                         ROTATION = _F(VITESSE = RPM, 
#                                       AXE = DictStructParam['RotatingProperties']['AxeRotation'] , 
#                                       CENTRE = DictStructParam['RotatingProperties']['RotationCenter'],),
#                         DDL_IMPO = _F(GROUP_NO='Node_Encastrement',
#                                           DX=0.0,
#                                           DY=0.0,
#                                           DZ=0.0,),
#                         FORCE_NODALE =  ListeLoading,
#                         );
#
#
#    L_INST = DEFI_LIST_REEL(DEBUT = 0.0,
#                            INTERVALLE = (_F(JUSQU_A = 1.0,
#                                             NOMBRE = DictSimulaParam['IntegrationProperties']['StaticSteps'][0],),
#                                           ),
#                            );
#
#
#    SOLU = STAT_NON_LINE(MODELE = kwargs['MODELE'],
#                         CHAM_MATER = kwargs['CHMAT'],
#                         EXCIT =( _F( CHARGE = Cfd,
#                                      FONC_MULT=RAMPE,),), 
#                         COMPORTEMENT = _F(RELATION = 'ELAS',
#                                           DEFORMATION =DictStructParam['MaterialProperties']['TyDef'], 
#                                           TOUT = 'OUI',
#                                          ),
#                         CONVERGENCE=_F(RESI_GLOB_MAXI=2e-6,
#                                        RESI_GLOB_RELA=1e-4,
#                                        ITER_GLOB_MAXI=1000,
#                                        ARRET = 'OUI',),
#                         INCREMENT = _F( LIST_INST = L_INST,
#                                        ),
#                         INFO = 1,               
#                        ),
#
#
#
#    UpFromOmegaAndFe = SM.ExtrUpFromAsterSOLUwithOmegaFe(t, RPM, **dict(SOLU = SOLU))
#
#    SJ.DestroyAsterObjects(dict(**dict(Cfd = Cfd, SOLU= SOLU, RAMPE = RAMPE, L_INST = L_INST)),  
#                           DetrVars = ['Cfd', 'SOLU', 'RAMPE', 'L_INST',
#                                      ])

    
#    return  UpFromOmegaAndFe

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
            
            MatrUpLambda[:, count],_ = SM.ComputeStaticU4GivenLoading(t, RPM, f1, **kwargs)
            MatrFLambda[:, count] = f1
            count += 1

            print(FAIL + 'Counter: %s/%s'%(count+1, lenQ) +ENDC)
            MatrUpLambda[:, count],_ = SM.ComputeStaticU4GivenLoading(t, RPM, f2, **kwargs)
            MatrFLambda[:, count] = f2
         
         # Combinaison de deux modes   
                 
        for i in range(DictStructParam['ROMProperties']['NModes'][0]):
            for j in range(i+1, DictStructParam['ROMProperties']['NModes'][0]):
                count += 1         
                
                f3 = PHI[:,i] * lambda_vect[i] + PHI[: , j] * lambda_vect[j]   
                f4 = -1 *(PHI[:,i] * lambda_vect[i]) - PHI[: , j] * lambda_vect[j]
                f5 = PHI[:,i] * lambda_vect[i] - PHI[: , j] * lambda_vect[j]
 
                MatrUpLambda[:, count],_ = SM.ComputeStaticU4GivenLoading(t, RPM, f3, **kwargs)
                MatrFLambda[:, count] = f3
                count += 1
                print(FAIL + 'Counter: %s/%s'%(count+1, lenQ) +ENDC)
                MatrUpLambda[:, count],_ = SM.ComputeStaticU4GivenLoading(t, RPM, f4, **kwargs)
                MatrFLambda[:, count] = f4
                count += 1
                print(FAIL + 'Counter: %s/%s'%(count+1, lenQ) +ENDC)
                MatrUpLambda[:, count],_ = SM.ComputeStaticU4GivenLoading(t, RPM, f5, **kwargs)
                MatrFLambda[:, count] = f5
             
                       
                     
        # Combinaison de 3 modes
     
        for i in range(DictStructParam['ROMProperties']['NModes'][0]):
            for j in range(i+1, DictStructParam['ROMProperties']['NModes'][0]):
                for k in range(j+1, DictStructParam['ROMProperties']['NModes'][0]):
                    count += 1 
                    print(FAIL + 'Counter: %s/%s'%(count+1, lenQ) +ENDC)
                    f6 = PHI[:,i] * lambda_vect[i] + PHI[: , j] * lambda_vect[j] + PHI[: , k] * lambda_vect[k]   
                    MatrFLambda[:, count] = f6              
                    MatrUpLambda[:, count],_ = SM.ComputeStaticU4GivenLoading(t, RPM, f6, **kwargs)
        

        # Compute rotating matrix to substract from MatrUpLambda:

        MatrUs = DictAsseVectors[str(np.round(RPM,2)) + 'RPM']['Us'].reshape((DictStructParam['MeshProperties']['Nddl'][0], 1)) * np.ones(np.shape(MatrUpLambda))
        
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

def QuadraticCombinationMatrix(q_Save):

    nf = np.shape(q_Save)[0]
    try:
        nL = np.shape(q_Save)[1]
    except:
        nL = 1
    indexes = [i for i in range(nf)]
    comb2 = combinations_with_replacement(indexes, 2) # as the list [1,2,3] is given in growing order, the combinations will be in ordered (ex (1,2) and not (2,1) )
    listcomb2 = list(comb2)

    QuadraticComMatrix = np.zeros((len(listcomb2),nL))
    for i in range(nL):
        try:
            qf = q_Save[:,i].reshape((nf,1))
        except:
            qf = q_Save.reshape(nf,1)
        QuadraticComMatrix[:,i] = combination_to_product(qf, listcomb2).reshape((len(listcomb2),))
    
    return QuadraticComMatrix


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


def ComputeULambda4ICE(t, RPM, **kwargs):

    DictStructParam = J.get(t, '.StructuralParameters')
    DictSimulaParam = J.get(t, '.SimulationParameters')
    
    # Calcul de tous les cas statiques avec les forces imposees:
    MatrULambda, MatrFLambda = ComputeMatrULambda4ICE(t, RPM, **kwargs)
    
    # Compute the pseudo inverse of MatrULambda with respect to PHI:
    MatrQLambda = SJ.PseudoInverseWithModes(t, RPM, MatrULambda)
    
    # Compute the pseudo inverse of MatrQlambda to solve the unknowns coefficient:
    PinvMatrQLambda = ComputeQMatrix4ICE(DictStructParam['ROMProperties']['NModes'][0], MatrQLambda)

    # Expansion, compute the ExpansionBase:
     
    ExpansionBase = ComputeExpansionBase(MatrULambda,MatrQLambda, SJ.GetReducedBaseFromCGNS(t, RPM))

    # Compute the nonlinear forces matrice: 

    FnlLambdaMatrix = ComputeForceMatrix4ICE(t, RPM, MatrULambda, MatrFLambda) 

    # Compute the Aij^k and the Bijm^k: 
    
    Aij, Bijm = ComputeAijAndBijmCoefficients(t, RPM, FnlLambdaMatrix, PinvMatrQLambda)
    
    DictOfCoefficients = J.get(t,'.InternalForcesCoefficients')
    DictOfCoefficients['%sRPM'%np.round(RPM,2)] = dict(Type = 'ICE',
                                                   Aij  = Aij,
                                                   Bijm = Bijm, 
                                                   ExpansionBase = ExpansionBase)



    J.set(t,'.InternalForcesCoefficients', **dict(DictOfCoefficients))

    return t



def ComputeNLCoefficients(t, RPM, **kwargs):
    
    DictStructParam = J.get(t, '.StructuralParameters')
    
    # ICE method:
    if (DictStructParam['ROMProperties']['ROMForceType'] == 'IC') or (DictStructParam['ROMProperties']['ROMForceType'] == 'ICE'):
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
    qvect is the column vector of the generalised coordinates '''

    Knl2_IC_kl = 0
    Knl3_IC_kl = 0
    nq = np.shape(qvect)[0] #like len(qvect)

    q =  qvect.ravel()

    # derivatives of the quadratic terms ========================
    for i in range(nq):
        if i < l:
            Knl2_IC_kl += Beta[k,i,l]*q[i]
        elif i == l :
            Knl2_IC_kl += 2.*Beta[k,l,l]*q[l]
        else:  # i > l
            Knl2_IC_kl += Beta[k,l,i]*q[i]
    
    # derivatives of the cubic terms ============================
    for i in range(nq):
        for j in range(i, nq):
            if j < l:
                Knl3_IC_kl += Gamma[k,i,j,l]*q[i]*q[j]
            elif j == l:
                if i < l:
                    Knl3_IC_kl += 2.*Gamma[k,i,l,l]*q[i]*q[j]
                elif i == l:
                    Knl3_IC_kl += 3.*Gamma[k,l,l,l]*q[i]*q[j]
            else:  # j > l
                if i < l:
                    Knl3_IC_kl += Gamma[k,i,l,j]*q[i]*q[j]
                elif i == l:
                    Knl3_IC_kl += 2.*Gamma[k,l,l,j]*q[i]*q[j]
                else: # i > l
                    Knl3_IC_kl += Gamma[k,l,i,j]*q[i]*q[j]

    Knl_kl = Knl2_IC_kl + Knl3_IC_kl

    return Knl_kl



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

    elif (ROMForceType == 'IC') or (ROMForceType == 'ICE'):

       fnl_Proj = CalcFnl_IC(Aij,Bijm,q)   

    return fnl_Proj


def Knl_Jacobian_Proj(t, q, Aij, Bijm):
    
    DictStructParam = J.get(t, '.StructuralParameters') 
    
    ROMForceType = DictStructParam['ROMProperties']['ROMForceType']
    
    if ROMForceType == 'Linear':

       Knl_Jacob_Proj = np.zeros((len(q),len(q))) 

    elif (ROMForceType == 'IC') or (ROMForceType == 'ICE') :

       Knl_Jacob_Proj = CalcKNLproj_IC(Aij,Bijm,q)

    return Knl_Jacob_Proj