'''StructuralAnalysis.py

   Structural module compatible with Cassiopee and MOLA




      The objective of this module is to provide cgns based structural functions to perform the
   static or dynamic analysis of structures. This module is fully based in Python.

   Author: Mikel Balmaseda 

   1. 26/05/2021  Mikel Balmaseda  cgns adaptation of the previous Scripts developed durin the PhD

'''
import sys
import numpy as np
from numpy.linalg import norm

import Converter.PyTree as C
import Converter.Internal as I

from .. import InternalShortcuts as J
from .. import PropellerAnalysis as PA

from . import ShortCuts as SJ
from . import ModalAnalysis   as MA
from . import NonlinearForcesModels as NFM

FAIL  = '\033[91m'
GREEN = '\033[92m' 
WARN  = '\033[93m'
MAGE  = '\033[95m'
CYAN  = '\033[96m'
ENDC  = '\033[0m'


def Macro_BEMT(t, PolarsInterpFuns, RPM):

    print (CYAN+'Launching 3D BEMT computation...'+ENDC)

    DictAerodynamicProperties = J.get(t, '.AerodynamicProperties')

    LiftingLine = I.getNodeFromName(t, 'LiftingLine')

    ResultsDict = PA.computeBEMTaxial3D(LiftingLine, PolarsInterpFuns,
    NBlades=DictAerodynamicProperties['BladeParameters']['NBlades'],
    Constraint=DictAerodynamicProperties['BladeParameters']['Constraint'],
    ConstraintValue=DictAerodynamicProperties['BladeParameters']['ConstraintValue'],
    AttemptCommandGuess=[],

    Velocity=[0.,0.,-DictAerodynamicProperties['FlightConditions']['Velocity']],  # Propeller's advance velocity (m/s)
    RPM=RPM,              # Propellers angular speed (rev per min.)
    Temperature = DictAerodynamicProperties['FlightConditions']['Temperature'],     # Temperature (Kelvin)
    Density=DictAerodynamicProperties['FlightConditions']['Density'],          # Air density (kg/m3)
    model=DictAerodynamicProperties['BEMTParameters']['model'],          # BEMT kind (Drela, Adkins or Heene)
    TipLosses=DictAerodynamicProperties['BEMTParameters']['TipLosses'],

    FailedAsNaN=True,
    )

    print("RPM: %g rpm, Thrust: %g N,  Power: %g W,  Prop. Eff.: %g, | Pitch: %g deg"%(RPM, ResultsDict['Thrust'],ResultsDict['Power'],ResultsDict['PropulsiveEfficiency'],ResultsDict['Pitch']))
    print(WARN + '3D BEMT computation COMPLETED'+ENDC)

    I._addChild(t, LiftingLine)

    return ResultsDict, t



def ComputeExternalForce(t):

    if ForceType == 'CoupledRotatoryEquilibrium':
        
        LiftingLine = I.getNodeFromNameAndType(t, 'LiftingLine', 'Zone_t')
        
        LiftingLine = SJ.updateLiftingLineFromStructureLETE(LiftingLine, tFOM, RPM)
        LiftingLine, ResultsDict = Macro_BEMT(LiftingLine, PolarsInterpFuns, tFOM, RPM)
    
        FTE, FLE = SJ.FrocesAndMomentsFromLiftingLine2ForcesAtLETE(LiftingLine, tFOM, RPM)
        
        return  SJ.LETEvector2FullDim(tFOM, FLE, FTE)



def Compute_IntFnl(t):

    
    


    return Fnl


def SolveTimeIntegration(t, RPM):
    '''MacroFunction of Time Integration Methods

        Implemented methods:

               - 'Li2020': 2 step algorithm https://link.springer.com/article/10.1007/s00419-019-01637-7
               - 'HHT-alpha':
               - 'Newmark':

    '''
    DictSimulaParam = J.get(t, '.SimulationParameters')

    

    if DictSimulaParam['IntegrationProperties']['IntegrationMethod'] == 'Li2020':

        
        
        # Load time

        InitialTime, dt, itmax, NItera = DictSimulaParam['LoadingProperties']['TimeProperties']['InitialTime'][0], DictSimulaParam['LoadingProperties']['TimeProperties']['dt'][0], DictSimulaParam['IntegrationProperties']['NumberOfMaxIterations'][0], DictSimulaParam['LoadingProperties']['TimeProperties']['NItera'][0]

        time = np.linspace(InitialTime, InitialTime + dt*(NItera - 1), NItera)

        # Load matrices of the problem, M, C, K(Omega)  --> Either full or reduced version
        Matrices={}
        for MatrixName in ['M', 'Komeg', 'C']:
            Matrices[MatrixName] = SJ.LoadSMatrixFromCGNS(t, RPM, MatrixName, Type = '.AssembledMatrices' )

        # Load and compute the integration method parameters:

        rconv = DictSimulaParam['IntegrationProperties']['EpsConvergenceCriteria']

        try:
            rho_inf = J.getVars(zone, ['rho_inf_Li'])
        except:
            rho_inf = 0.5
            print(WARN+'rho_inf_Li is NOT initialised, default value: 0.5'+ENDC)

        #gamma = (2-sqrt(2*(1-rho_inf)))/(1+rho_inf)   #0.731 #, 2 - sqrt(2)    #Gamma 2
        gamma = (2-np.sqrt(2*(1+rho_inf)))/(1-rho_inf)
    
        C1 = (3. * gamma - gamma**2  - 1.)/(2.*gamma)
        C2 = (1. - gamma)/(2.*gamma)
        C3 = gamma/2.


        try:
            M,_ = Matrices['M']
        except:
            print(MAGE+'M not defined!!!'+ENDC)
            sys.exit()

        try:
            Komeg,_ = Matrices['Komeg']
        except:
            print(MAGE+'Komeg not defined!!!'+ENDC)
            sys.exit()
        
        try:
            C,_ = Matrices['C']
        except:
            print(MAGE+'C not defined!!!'+ENDC)
            sys.exit()


        K1 = (gamma/2. * dt)**2 * Komeg + gamma/2. * dt * C  + M


        # Initialize:
        DimProblem = Komeg.shape[0]
        #print(DimProblem)
        TempData = J.createZone('.temporalData',[np.zeros((DimProblem,)),np.zeros((DimProblem,)),np.zeros((DimProblem,))],
                                            ['u','v', 'z']
                                )


        for loc_t, it in zip(time, range(NItera)):
            
            #it += 1
            print('Structural Iteration: %s, time: %s'%(t,time[it]))


            # Compute the external force: fe(t, u, v)

            #fe = fe_t() # Provides the value of fe_t at instant t


            # 1st Predidction:

            Ug = u + gamma* dt* v + (gamma * dt)**2 /2. * a


            for ni in range(itmax):
            
                Vg = 2./(gamma * dt) * (Ug - u) - v
                Ag = 2./(gamma * dt) * (Vg - v) - a

                Rg = Compute_IntFnl(t) # Computes the internal nonlinear force
                Kp = Compute_TangentMatrix(t) # Computes the tangent matrix 
                
                Fg = ComputeExternalForce(t) # fe_t ug...
                
                
                Residue = Fg - Rg - dot(Komeg, Ug) - dot(C, Vg) - dot(M, Ag)
                Nresid =  amax(abs(Residue))
                
                if Nresid < rconv:
                    break
                else:
                    S = 4./(gamma**2 * dt**2) * M + 2./(gamma*dt) * C + Kp
                    DU = solve(S, Residue)
                    Ug += DU

                if ni == itmax - 1:
                    print(WARN+'Careful, it %s first loop not converged'%(it)+ENDC)

            t = UpdateSolution(t, Ug, Vg, Ag)
            # 2nd Prediction:

            Uh = 1./gamma**3 * ((gamma**3 - 3*gamma + 2)* u + (3. * gamma - 2. )* Ug + (gamma - 1.) * gamma * dt * (( gamma - 1.) * v - Vg))

            for ni in range(itmax):

              Vh = 1./(C3 * dt) * (Uh - u) - 1./C3 * (C1 * v + C2 * Vg)
              Ah = 1./(C3 * dt) * (Vh - v) - 1./C3 * (C1 * a + C2 * Ag)

              Rh = Compute_Fnl(t) # Computes the internal nonlinear force
              Kp = Compute_Tangent(t) # Computes the tangent matrix 

              S_r = 1./(C3**2 * dt**2) * M + 1./(C3 * dt) * C + Kp
              
              Fg = f_ex(loc_t+dt  ) # fe_t uh...
  
              Resid = Fh - Rh - dot(K, Uh) - dot(C, Vh) - dot(M, Ah)
  
              Nresid =  amax(abs(Resid))
              if code1 == 'Inflation':
                  savetxt(pathsol + 'Computation_time/Iteration.txt', ['{0}/{1}, t= {2}s, Residu: {3}, Step2 N-R it {4}'.format(i+1,len(temp), t, Nresid, it+1)],fmt='%s')
  
              #print 'Nresid', Nresid
  
              DU = solve(S_r, Resid)
  
              if Nresid < rconv:
                  U[:, i+ 1] = Uh
                  V[:, i+ 1] = Vh
                  A[:, i+ 1] = Ah
                  R[:, i+ 1] = Rh 
                  #print(it, Nresid, Resid)
                  break
              else:
                  Uh = Uh + DU
  
              if it == N_itmax -1:
                  U[:, i+ 1] = Uh
                  V[:, i+ 1] = Vh
                  A[:, i+ 1] = Ah
                  R[:, i+ 1] = Rh
                  print(WARN+'WARNING: Iteration '+str(i)+' did not converge! Residue: '+str(Nresid)+ENDC, rconv)







def StaticSolver_Newton_Raphson(t, RPM, ForceIntensityC):
    "Function returning the reduced static non-linear solution using the IC non-linear function"

    DictStructParam = J.get(t, '.StructuralParameters')
    DictInternalForcesCoefficients = J.get(t, '.InternalForcesCoefficients')
    DictSimulaParam = J.get(t, '.SimulationParameters')
    
    # Get the needed variables from the cgns tree:

    V = SJ.GetReducedBaseFromCGNS(t, RPM)  # Base de reduction PHI

    nq       = DictStructParam['ROMProperties']['NModes'][0]

    nitermax = DictSimulaParam['IntegrationProperties']['NumberOfMaxIterations'][0]
    nincr    = DictSimulaParam['IntegrationProperties']['StaticSteps'][0]
    try:
        Aij      = DictInternalForcesCoefficients[str(int(RPM))+'RPM']['Aij']
        Bijm     = DictInternalForcesCoefficients[str(int(RPM))+'RPM']['Bijm']
    except:
        Aij, Bijm = 0, 0 

        print(WARN + 'Warning!! Aij and Bijm not readed!'+ENDC)

    Kproj = SJ.getMatrixFromCGNS(t, 'Komeg', RPM)
    
    # Initialisation des vecteurs du calcul:
    q = np.zeros((nq,1))
    Fextproj = np.zeros((nq,1))
    Fnlproj = np.zeros((nq,1))

    # Initialisation des vecteurs de sauvegarde:
    if int(DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]) == 0:
        q_Save   = np.zeros((nq, 1)) 
        Fnl_Save = np.zeros((nq, 1)) 
        Fext_Save     = np.zeros((nq, 1)) 
    else:
        if int((nincr - 1)%DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]) == 0 :
            q_Save   = np.zeros((nq, 1 + int((nincr-1)//DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]))) 
            Fnl_Save = np.zeros((nq, 1 + int((nincr-1)//DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]))) 
            Fext_Save     = np.zeros((nq, 1+ int((nincr-1)//DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]))) 
        else:
            q_Save   = np.zeros((nq, 2+int((nincr-1)//DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]))) 
            Fnl_Save = np.zeros((nq, 2+int((nincr-1)//DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]))) 
            Fext_Save     = np.zeros((nq, 2+int((nincr-1)//DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]))) 
            
    it2 =-1
    for incr in range(1,nincr+1):
            
        Fextproj[:,0] = ForceIntensityC * SJ.ComputeLoadingFromTimeOrIncrement(t, RPM, incr-1)
        
        Resi = np.dot(Kproj,q) + NFM.fnl_Proj(t, q, Aij, Bijm) - Fextproj
        niter=0
    
        while np.linalg.norm(Resi,2) > DictSimulaParam['IntegrationProperties']['EpsConvergenceCriteria'][0]:
        
            niter=niter+1
            
            if niter>=nitermax:
                print(FAIL+'Too much N.L. iterations for increment number %s'%str(incr)+ENDC)
                break

            #Compute tangent stiffness matrix
            
            Ktanproj = Kproj + NFM.Knl_Jacobian_Proj(t, q, Aij, Bijm)
            # Solve displacement increment
            dq = -np.linalg.solve(Ktanproj,Resi)
            q = q + dq
            
            #Compute internal forces vector
            Fnlproj = NFM.fnl_Proj(t, q, Aij, Bijm)

           
            #compute residual
            Resi = np.dot(Kproj,q) + Fnlproj - Fextproj
            
        
        # Save the data in the matrices:
        if DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0] == 0: 
            if incr == nincr: 
                it2 += 1
                q_Save[:, it2] = q.ravel()
                Fnl_Save[:, it2] = Fnlproj.ravel()
                Fext_Save[:, it2] = Fextproj.ravel()
                
        elif (not (incr-1)%DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]) or (incr == nincr):
            
            it2 += 1
            q_Save[:, it2] = q.ravel()
            Fnl_Save[:, it2] = Fnlproj.ravel()
            Fext_Save[:, it2] = Fextproj.ravel()
             
    
    return q_Save, Fnl_Save, Fext_Save 

def StaticSolver_Newton_Raphson1IncrFext(t, RPM, fext):
    "Function returning the reduced static non-linear solution using the IC non-linear function"

    DictStructParam = J.get(t, '.StructuralParameters')
    DictInternalForcesCoefficients = J.get(t, '.InternalForcesCoefficients')
    DictSimulaParam = J.get(t, '.SimulationParameters')
    
    # Get the needed variables from the cgns tree:

    V = SJ.GetReducedBaseFromCGNS(t, RPM)  # Base de reduction PHI
    
    nq       = DictStructParam['ROMProperties']['NModes'][0]
    nitermax = DictSimulaParam['IntegrationProperties']['NumberOfMaxIterations'][0]
    nincr    = DictSimulaParam['IntegrationProperties']['StaticSteps'][0]
    try:
        Aij      = DictInternalForcesCoefficients[str(int(RPM))+'RPM']['Aij']
        Bijm     = DictInternalForcesCoefficients[str(int(RPM))+'RPM']['Bijm']
    except:
        Aij, Bijm = 0, 0 

        print(WARN + 'Warning!! Aij and Bijm not readed!'+ENDC)

    Kproj = SJ.getMatrixFromCGNS(t, 'Komeg', RPM)
    

    # Initialisation des vecteurs du calcul:
    q = np.zeros((nq,1))
    Fextproj = np.zeros((nq,1))
    #Fnlproj_IC = np.zeros((nq,1))

    # Initialisation des vecteurs de sauvegarde:
    q_Save   = np.zeros((nq,)) 
    Fnl_Save = np.zeros((nq,)) 
    Fext     = np.zeros((nq,)) 
            
    Fextproj[:,0] = (V.T).dot(fext)

    Resi = np.dot(Kproj,q) + NFM.fnl_Proj(t, q, Aij, Bijm) - Fextproj
    niter=0
    
    while np.linalg.norm(Resi,2) > DictSimulaParam['IntegrationProperties']['EpsConvergenceCriteria'][0]:
    
        niter=niter+1
        
        if niter>=nitermax:
            print(FAIL+'Too much N.L. iterations for increment number %s'%str(incr)+ENDC)
            break
        #Compute tangent stiffness matrix
        
        Ktanproj = Kproj + NFM.Knl_Jacobian_Proj(t, q, Aij, Bijm)
        # Solve displacement increment
        dq = -np.linalg.solve(Ktanproj,Resi)
        q = q + dq
        
        #Compute internal forces vector
        Fnlproj = NFM.fnl_Proj(t, q, Aij, Bijm)
       
        #compute residual
        Resi = np.dot(Kproj,q) + Fnlproj - Fextproj
        
    
    # Save the data in the matrices:
   
    print(GREEN + 'Nb iterations = %s, for increment: 1, residual =  %0.4f'%(niter, norm(Resi,2))+ENDC)
    
    
    
    return q, Fnlproj, Fextproj 




def SolveStatic(t, RPM, ForceCoeff=1.):
    '''MacroFunction of Time Integration Methods

        Implemented methods:

               - 'Newton-Raphson
               - 'FixPoint'
    '''
    DictSimulaParam = J.get(t, '.SimulationParameters')

    
    if DictSimulaParam['IntegrationProperties']['IntegrationMethod'] == 'Newton_Raphson':
        
        q, fnl_q , Fext_q =  StaticSolver_Newton_Raphson(t, RPM, ForceCoeff)

    if DictSimulaParam['IntegrationProperties']['IntegrationMethod'] == 'AEL':
        if DictSimulaParam['IntegrationProperties']['TypeAEL'] == 'Static_Newton1':
            # NewtonRhapson with one iteration:
            q, fnl_q , Fext_q =  StaticSolver_Newton_Raphson1IncrFext(t, RPM, ForceCoeff)

        if DictSimulaParam['IntegrationProperties']['TypeAEL'] == 'FOM':
            pass # ComputeStaticU4GivenLoading(t, RPM, LoadVector, **kwargs)


    # Manque save to Tree


    return [q], fnl_q, Fext_q



def SolveROM(tROM, InputRPM = None, InputForceCoeff = None): 

    DictSimulaParam = J.get(tROM, '.SimulationParameters')
    DictStructParam = J.get(tROM, '.StructuralParameters')

    TypeOfLoading = DictSimulaParam['LoadingProperties']['LoadingType']
    TypeOfSolver  = DictSimulaParam['IntegrationProperties']['SolverType']
 
    Solution = {}

    if InputRPM == None: 
        InputRPM = DictSimulaParam['RotatingProperties']['RPMs']
        InputForceCoeff = DictSimulaParam['LoadingProperties']['ForceIntensityCoeff']


    for RPM in InputRPM:
        Solution['%sRPM'%np.round(RPM,2)] = {}
        PHI = SJ.GetReducedBaseFromCGNS(tROM, RPM)

        for ForceCoeff in InputForceCoeff:
            Solution['%sRPM'%np.round(RPM,2)]['FCoeff%s'%ForceCoeff] = {}

            if TypeOfSolver == 'Static':
        
                q_qp_qpp, fnl_q, fext_q  = SolveStatic(tROM, RPM, ForceCoeff)
                
            
            elif TypeOfSolver == 'Dynamic':
                pass

 
        
            # Save the reduced q_qp_qpp:
            Solution = SJ.SaveSolution2PythonDict(Solution, ForceCoeff, RPM, PHI, q_qp_qpp, fnl_q, fext_q)

    return Solution
                