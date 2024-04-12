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
#from scipy.sparse import csr_matrix
#from scipy import linalg
import scipy.sparse.linalg
from itertools import combinations_with_replacement

from code_aster.Cata.Syntax import _F
from code_aster.Cata.Language.DataStructure import CO
from code_aster.Cata.Commands import ASSEMBLAGE

try:
    #Code Aster:  
    from code_aster.Commands import *

    #from code_aster.Cata.DataStructure import *
    #from code_aster.Cata.Language import *
    from code_aster.MacroCommands.Utils import partition
except:
    print(WARN + 'Code_Aster modules are not loaded!!' + ENDC)

from . import Functions4ASTER as myFunct4Aster
    
    
def solveGnluFromImposedDisplacements(DictStructParam, DictSimulaParam,DictOfVectorsAndMatrices, RPM, ImposeDisplacement, **kwargs):

    # Compute G(us + u)
    _, Gusu, _, _ = myFunct4Aster.ComputeStaticU4GivenLoading(RPM, DictStructParam, DictSimulaParam, ImposeDisplacement, LoadType = 'Displacements', **kwargs)

    # Compute gnl(u):
    Ks = DictOfVectorsAndMatrices['AsseMatrices']['%sRPM'%np.round(RPM,2)]['Ke'] + DictOfVectorsAndMatrices['AsseMatrices']['%sRPM'%np.round(RPM,2)]['Kg']

    gnlu =  Gusu - DictOfVectorsAndMatrices['AsseVectors']['%sRPM'%np.round(RPM,2)]['Gus'] - Ks.dot(ImposeDisplacement-DictOfVectorsAndMatrices['AsseVectors']['%sRPM'%np.round(RPM,2)]['Us'])

    return gnlu


def compute_STEP_1vect(DictStructParam, DictSimulaParam,DictOfVectorsAndMatrices, RPM, PHI1 = [] , qi = [], us = [], **kwargs):
    """ 
    Function computing the coefficients Aii associated to one vector vect
    Inputs :
        shapeVect     : vector of shape(ndofs,1)
        VectAmplitude : generalized coordinate for the step method
        **kwargs : dictionnary of the beam mesh, properties and matrices
    Output :
        G_ii  : quadratic coefficients G(vect,vect)
        H_iii : cubic coefficient H(vect,vect,vect)
    """


    ImposedDisplacement =  PHI1 * qi + us

    Fnl_qivect       = solveGnluFromImposedDisplacements(DictStructParam, DictSimulaParam,DictOfVectorsAndMatrices, RPM, ImposedDisplacement, **kwargs)

    ImposedDisplacement =  -1.*PHI1 * qi + us
    Fnl_minus_qivect = solveGnluFromImposedDisplacements(DictStructParam, DictSimulaParam,DictOfVectorsAndMatrices, RPM, ImposedDisplacement, **kwargs)

    #print('\n')

    G_ii  = 1./(2*qi**2) * (Fnl_qivect + Fnl_minus_qivect)
    H_iii = 1./(2*qi**3) * (Fnl_qivect - Fnl_minus_qivect)

    return G_ii, H_iii 

def compute_STEP_2vect(DictStructParam, DictSimulaParam,DictOfVectorsAndMatrices, RPM,us, vect1, vect2, q1, q2, G_vect1_vect1, G_vect2_vect2, **kwargs):
    """ 
    Function computing the coefficients G(vect1,vect2)
    Inputs :
        vect1         : vector of shape(ndofs,1)
        vect2         : vector of shape(ndofs,1)
        q1            : generalized coordinate associated to vect1
        q2            : generalized coordinate associated to vect2
        G_vect1_vect1 : G(vect1,vect1) computed with compute_STEP_1vect(vect1,q1,...) or None if not available
        G_vect2_vect2 : G(vect2,vect2) computed with compute_STEP_1vect(vect2,q2,...) or None if not available
        nitermax : maximal number of Newton-Raphson iterations for the computation of the nonlinear static solution
        nincr    : number of increments to apply the load for the computation of the nonlinear static solution
        **kwargs : dictionnary of the beam mesh, properties and matrices
    Output :
        G_vect1_vect2  : quadratic coefficients G(vect1,vect2)
    """


    ImposedDisplacement1 =  vect1 * q1 + vect2 * q2 + us

    Fnl_plusq1_plusq2       = solveGnluFromImposedDisplacements(DictStructParam, DictSimulaParam,DictOfVectorsAndMatrices, RPM, ImposedDisplacement1, **kwargs)

    ImposedDisplacement2 =  -vect1 * q1 - vect2 * q2 + us
    Fnl_minusq1_minusq2 = solveGnluFromImposedDisplacements(DictStructParam, DictSimulaParam,DictOfVectorsAndMatrices, RPM, ImposedDisplacement2, **kwargs)

    
    G_vect1_vect2 = 1./(4*q1*q2) * (Fnl_plusq1_plusq2 + Fnl_minusq1_minusq2 - 2*q1**2*G_vect1_vect1 - 2*q2**2*G_vect2_vect2)

    return G_vect1_vect2


def compute1ModeDNFCoefficients(DictStructParam, DictSimulaParam,DictOfVectorsAndMatrices, RPM, DictOfCoefficients,**kwargs):
    '''
      Warning! The name dof the DNF considers two different assumptions:

      First number informs about the number of modal conbinations. Second number considers the master mode number.

      i.e.   'DNF1_Mode5': DNF (with one mode combination) using the 5th mode as master
      '''

    # Normalize first vector of the basis:
    MasterModeNumber = int(DictStructParam['ROMProperties']['ROMForceType'].split('_')[-1][4:]) 
    

    PHI = DictOfVectorsAndMatrices['AsseMatrices']['%sRPM'%np.round(RPM,2)]['PHI']
    M   = DictOfVectorsAndMatrices['AsseMatrices']['%sRPM'%np.round(RPM,2)]['M']
    Komeg = DictOfVectorsAndMatrices['AsseMatrices']['%sRPM'%np.round(RPM,2)]['Komeg']
    us = DictOfVectorsAndMatrices['AsseVectors']['%sRPM'%np.round(RPM,2)]['Us']

    PHI1 = PHI[:, MasterModeNumber-1]
    
    ReducedMass = M.dot(PHI1)
    ReducedMass = PHI1.T.dot(ReducedMass) 

    PHI1 = PHI1/np.sqrt(ReducedMass) 

    
    lambda_qi = DictStructParam['ROMProperties']['lambda_vect'][MasterModeNumber-1]

    # Solve G_11 and H_111:

    G_11, H_111  = compute_STEP_1vect(DictStructParam, DictSimulaParam,DictOfVectorsAndMatrices, RPM, PHI1 = PHI1 , qi = lambda_qi, us = us, **kwargs)
 
    # Solve Rayleigh dumping coefficients:

    zetaM = 4*np.pi*DictStructParam['MaterialProperties']['Titane']['Freq4Dumping']*DictStructParam['MaterialProperties']['Titane']['XiBeta']
    zetaK = DictStructParam['MaterialProperties']['Titane']['XiAlpha']

    w1 = DictOfVectorsAndMatrices['AsseVectors']['%sRPM'%np.round(RPM,2)]['ModalFrequencies'][MasterModeNumber - 1]
    
    Zs11_barre = scipy.sparse.linalg.spsolve((4*w1**2)*M - Komeg  ,G_11)
    Zd11_barre = scipy.sparse.linalg.spsolve(-Komeg  ,G_11)

    a11_barre  = 0.5*(Zd11_barre + Zs11_barre)

    b11_barre     = 1./(2*w1**2) * (Zd11_barre - Zs11_barre)
    gamma11_barre = 2*Zs11_barre
    
    MZs11  = M.dot(Zs11_barre)
    Zss11_barre = scipy.sparse.linalg.spsolve((4*w1**2)*M - Komeg , MZs11)

    c11_barre     = 0.5*Zd11_barre*(zetaM/(w1**2) + zetaK) - 0.5*Zs11_barre*(zetaM/(w1**2) + 5.*zetaK) + 2*Zss11_barre*(2*zetaK*w1**2 - zetaM)
    alpha11_barre = -(w1**2)*c11_barre
    beta11_barre  = c11_barre - (zetaM/w1 + zetaK*w1)*b11_barre


    q1_a11_barre = 1.*lambda_qi/np.max(abs(a11_barre))
    q1_b11_barre = 1.*lambda_qi/np.max(abs(b11_barre))
    q1_c11_barre = 1.*lambda_qi/np.max(abs(c11_barre))
    

    G_a11barre_a11barre,_ = compute_STEP_1vect(DictStructParam, DictSimulaParam,DictOfVectorsAndMatrices, RPM, PHI1 = a11_barre , qi = q1_a11_barre, us = us, **kwargs)
    G_Phi1_a11barre       = compute_STEP_2vect(DictStructParam, DictSimulaParam,DictOfVectorsAndMatrices, RPM, us, PHI1, a11_barre, lambda_qi, q1_a11_barre, G_11, G_a11barre_a11barre, **kwargs) 
    G_b11barre_b11barre,_ = compute_STEP_1vect(DictStructParam, DictSimulaParam,DictOfVectorsAndMatrices, RPM, PHI1 = b11_barre , qi = q1_b11_barre, us = us, **kwargs)
    G_Phi1_b11barre       = compute_STEP_2vect(DictStructParam, DictSimulaParam,DictOfVectorsAndMatrices, RPM, us, PHI1, b11_barre, lambda_qi, q1_b11_barre, G_11, G_b11barre_b11barre, **kwargs)
    G_c11barre_c11barre,_ = compute_STEP_1vect(DictStructParam, DictSimulaParam,DictOfVectorsAndMatrices, RPM, PHI1 = c11_barre , qi = q1_c11_barre, us = us, **kwargs)
    G_Phi1_c11barre       = compute_STEP_2vect(DictStructParam, DictSimulaParam,DictOfVectorsAndMatrices, RPM, us, PHI1, c11_barre, lambda_qi, q1_c11_barre, G_11, G_c11barre_c11barre, **kwargs)  

    A111 = 2*np.dot(PHI1.T, G_Phi1_a11barre)
    B111 = 2*np.dot(PHI1.T, G_Phi1_b11barre)
    C111 = 2*np.dot(PHI1.T, G_Phi1_c11barre)
    h111 = np.dot(PHI1.T, H_111)


    DictOfCoefficients = dict(a11_barre =a11_barre,
                              b11_barre =b11_barre,
                              gamma11_barre =gamma11_barre,
                              c11_barre =c11_barre,
                              alpha11_barre =alpha11_barre,
                              beta11_barre =beta11_barre,
                              A111 =A111,
                              B111 =B111,
                              C111 =C111,
                              h111 =h111,
                              PHI1 = PHI1
                             )

    print(DictOfCoefficients)

    return DictOfCoefficients

    # computeNonlinearForcesFromImposedDisplacements(DictStructParam, DictSimulaParam, RPM, ImposeDisplacement,**kwargs)

    # compute_STEP_1vect(shapeVect, VectAmplitude, **kwargs)