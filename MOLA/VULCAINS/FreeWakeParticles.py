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

import numpy as np
import Converter.Internal as I

from .. import InternalShortcuts as J
from . import Main as V

####################################################################################################
####################################################################################################
############################################### VPM ################################################
####################################################################################################
####################################################################################################
def initialiseVPM(tE = [], tH = [], tLL = [], VPMParameters = {}, HybridParameters = {},
    LiftingLineParameters = {}, PerturbationFieldParameters = {}):
    '''
    Creates a Tree initialised with the given checked and updated parameters and initialise the
    particles according to the given Lifting Lines or Eulerian Mesh. Also generate Hybrid
    Domains and/or Lifting Lines in the Tree.

    Parameters
    ----------
        tE : Tree
            Eulerian field.

        tH : Tree
            Hybrid Domain.

        tLL : Tree
            Lifting Lines.

        VPMParameters : :py:class:`dict`
            Parameters of the VPM solver.

        HybridParameters : :py:class:`dict`
            Parameters of the Hybrid solver.

        LiftingLineParameters : :py:class:`dict`
            Parameters of the Lifting Lines coupling.

        PerturbationFieldParameters : :py:class:`dict`
            Parameters relative to the Perturbation velocity field.
    Returns
    -------
        tL : Tree
            Lagrangian field.
    '''
    _tLL, _tE, _tH = V.getTrees([tLL, tE, tH], ['LiftingLines', 'Eulerian', 'Hybrid'])

    if not(VPMParameters['Resolution'].all()):
        raise ValueError(J.FAIL + 'The Resolution can not be computed. The Resolution or a ' + \
                                                          'Lifting Line must be specified' + J.ENDC)
    if not(VPMParameters['TimeStep']):
        raise ValueError(J.FAIL + 'The TimeStep can not be computed. The TimeStep or a Lifting ' + \
                                                                  'Line must be specified' + J.ENDC)
    tL = V.buildEmptyVPMTree()
    Particles = V.getFreeParticles(tL)
    if _tLL:
        V.show(f"{'||':>57}\r" + '||' + '{:=^53}'.format(' Initialisation of Lifting Lines '))
        V.initialiseParticlesOnLitingLine(tL, _tLL, VPMParameters)
        V.show(f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Done '))
    if _tE:
        V.show(f"{'||':>57}\r" + '||' + '{:=^53}'.format(' Initialisation of Hybrid Domain '))
        V.initialiseHybridParticles(tL, _tE, _tH, VPMParameters, HybridParameters)
        tL[2][1][2] += V.generateBEMParticles(_tE, _tH, VPMParameters, HybridParameters)
        # HybridParameters['BEMMatrix'] = np.zeros((HybridParameters['NumberOfBEMUnknown'][0]*\
                        # VPMParameters['NumberOfBEMSources'][0])**2, dtype = np.float64, order = 'F')
        tL[2][1][2] += V.generateImmersedParticles(_tE, _tH, VPMParameters, HybridParameters)
        J.set(Particles, '.Hybrid#Parameters', **HybridParameters)
        I._sortByName(I.getNodeFromName1(Particles, '.Hybrid#Parameters'))

    J.set(Particles, '.VPM#Parameters', **VPMParameters)
    I._sortByName(I.getNodeFromName1(Particles, '.VPM#Parameters'))
    return tL

def initialisePerturbationfield(tL = [], tP = []):
    '''
    Initialises the FMM tree capsule of the perturbation field.

    Parameters
    ----------
        tL : Tree
            Lagrangian field.

        tP : Tree
            Perturbation field.
    Returns
    -------
        tP : Tree
            Perturbation field.
    '''
    _tL = V.getTrees([tL], ['Particles'])
    if tP:
        if isinstance(tP, str):
            print(f"{'||':>57}\r" + '||', end='')
            tP = V.load(tP)
            V.deletePrintedLines()

        V.show(f"{'||':>57}\r" + '||' + '{:=^53}'.format(' Initialisation of Perturbation Field '))
        NumberOfNodes = V.getParameter(_tL, 'NumberOfNodes')
        V.tP_Capsule[0] = V.build_perturbation_velocity_capsule(tP, NumberOfNodes)
        V.show(f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Done '))
        return tP
    
    return []

def induceVPMField(tL = [], tP = []):
    '''
    Computes the current velocity, velocity gradients, vorticity, diffusion and stretching of
    the VPM particles.

    Parameters
    ----------
        tL : Tree
            Lagrangian field.

        tP : Tree
            Perturbation field.
    Returns
    -------
        IterationInfo : :py:class:`dict`
            VPM solver information on the current iteration.
    '''
    _tL, _tP = V.getTrees([tL, tP], ['Particles', 'Perturbation'], fillEmptyTrees = True)
    Scheme = V.Scheme_str2int[V.getParameter(_tL, 'VorticityEquationScheme')]
    Diffusion = V.DiffusionScheme_str2int[V.getParameter(_tL, 'DiffusionScheme')]
    solveVorticityEquationInfo = V.induce_vpm_field(_tL, _tP, V.tP_Capsule[0], Scheme,
                                                                                 Diffusion).tolist()
    IterationInfo = {}
    if V.tP_Capsule[0]: IterationInfo['Perturbation time'] = solveVorticityEquationInfo.pop()
        
    IterationInfo['FMM time'] = solveVorticityEquationInfo.pop()
    if solveVorticityEquationInfo:
        IterationInfo['Rel. err. of Velocity'] = solveVorticityEquationInfo.pop(0)
        IterationInfo['Rel. err. of Vorticity'] = solveVorticityEquationInfo.pop(0)
        IterationInfo['Rel. err. of Velocity Gradient'] = solveVorticityEquationInfo.pop(0)
    if solveVorticityEquationInfo: 
        IterationInfo['Rel. err. of PSE'] = solveVorticityEquationInfo.pop(0)
    if solveVorticityEquationInfo:
        IterationInfo['Rel. err. of Diffusion Velocity'] = solveVorticityEquationInfo.pop(0)

    return IterationInfo

def updateSmagorinskyConstantAndComputeTurbulentViscosity(tL = []):
    '''
    Updates the eddy viscosity constant of the LES eddy viscosity model for each particles. Also
    computes the eddy viscosity of the particles.

    Parameters
    ----------
        tL : Tree
            Lagrangian field.
    '''
    _tL = V.getTrees([tL], ['Particles'], fillEmptyTrees = True)
    V.update_smagorinsky_constant_and_compute_turbulent_viscosity(_tL,
                                V.EddyViscosityModel_str2int[V.getParameter(_tL, 'EddyViscosityModel')])

def computeTurbulentViscosity(tL = []):
    '''
    Computes the eddy viscosity of the particles.

    Parameters
    ----------
        tL : Tree
            Lagrangian field.
    '''
    _tL = V.getTrees([tL], ['Particles'], fillEmptyTrees = True)
    V.compute_turbulent_viscosity(_tL, V.EddyViscosityModel_str2int[V.getParameter(_tL,
                                                                             'EddyViscosityModel')])

def computeLagrangianNextTimeStep(tL = [], tLL = [], tP = []):
    '''
    Advances the Lagrangian VPM field one timestep forward.

    Parameters
    ----------
        tL : Tree
            Lagrangian field.

        tLL : Tree
            Lifting Lines.

        tP : Tree
            Perturbation field.
    '''
    _tL, _tLL, _tP = V.getTrees([tL, tLL, tP], ['Particles', 'LiftingLines', 'Perturbation'],
                                                                              fillEmptyTrees = True)
    if not _tL: return
    time, dt, it, IntegOrder, lowstorage = V.getParameters(_tL,
               ['Time','TimeStep', 'CurrentIteration', 'IntegrationOrder', 'LowStorageIntegration'])
    if lowstorage:
        if IntegOrder == 1:
            a = np.array([0.], dtype = np.float64)
            b = np.array([1.], dtype = np.float64)
            #c = np.array([1.], dtype = np.float64)
        elif IntegOrder == 2:
            a = np.array([0., -0.5], dtype = np.float64)
            b = np.array([0.5, 1.], dtype = np.float64)
            #c = np.array([1./2., 1.], dtype = np.float64)
        elif IntegOrder == 3:
            a = np.array([0., -5./9., -153./128.], dtype = np.float64)
            b = np.array([1./3., 15./16., 8./15.], dtype = np.float64)
            #c = np.array([1./3., 3./4., 1.], dtype = np.float64)
        elif IntegOrder == 4:
            a = np.array([0., -1., -0.5, -4.], dtype = np.float64)
            b = np.array([1./2., 1./2., 1., 1./6.], dtype = np.float64)
            #c = np.array([1./2., 1./2., 3./2., 1.], dtype = np.float64)
        else:
            raise AttributeError('This Integration Scheme Has Not Been Implemented Yet')

        c = [0., b[0]]
        for i in range(1, len(b)):
            c += [b[i]*(a[i]/b[i - 1]*(c[-1] - c[-2]) + 1) + c[-1]]
        
        c = np.array(c, dtype = np.float64)

    else:
        if IntegOrder == 1:
            a = np.array([], dtype = np.float64)
            b = np.array([1.], dtype = np.float64)
        elif IntegOrder == 2:
            a = np.array([[0.5]], dtype = np.float64)
            b = np.array([0, 1.], dtype = np.float64)
        elif IntegOrder == 3:
            a = np.array([[0.5, 0.], [-1., 2.]], dtype = np.float64)
            b = np.array([1./6., 2./3., 1./6.], dtype = np.float64)
        elif IntegOrder == 4:
            a = np.array([[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 1.]], dtype = np.float64)
            b = np.array([1./6., 1./3., 1./3., 1./6.], dtype = np.float64)
        else:
            raise AttributeError('This Integration Scheme Has Not Been Implemented Yet')

        c = np.array([np.sum(ai) for ai in a] + [1.], dtype = np.float64)

    Scheme = V.Scheme_str2int[V.getParameter(_tL, 'VorticityEquationScheme')]
    Diffusion = V.DiffusionScheme_str2int[V.getParameter(_tL, 'DiffusionScheme')]
    EddyViscosityModel = V.EddyViscosityModel_str2int[V.getParameter(_tL, 'EddyViscosityModel')]
    if lowstorage:
        V.runge_kutta_low_storage(_tL, _tP, V.tP_Capsule[0], a, b, c, Scheme, Diffusion,
                                                                                 EddyViscosityModel)
    else:
        V.runge_kutta(_tL, _tP, V.tP_Capsule[0], a, b, c, Scheme, Diffusion, EddyViscosityModel)
    
    for LiftingLine in I.getZones(_tLL):
        TimeShed = I.getNodeFromName(LiftingLine, 'TimeSinceLastShedding')
        TimeShed[1][0] += dt[0]

    time += dt
    it += 1

def populationControl(tL = [], tLL = [], tH = []):
    '''
    Split, merge, resize and erase particles when necessary.

    Parameters
    ----------
        tL : Tree
            Lagrangian field.

        tLL : Tree
            Lifting Lines.

        tH : Tree
            Hybrid Domain.
    Returns
    -------
        IterationInfo : :py:class:`dict`
            VPM solver information on the current iteration.
    '''
    _tL, _tLL, _tH = V.getTrees([tL, tLL, tH], ['Particles', 'LiftingLines', 'Hybrid'],
                                                                              fillEmptyTrees = True)
    if not _tL: return {}

    IterationInfo = {'Population Control time': J.tic()}
    Particles = V.getFreeParticles(_tL)
    HybridInterface = V.getHybridDomainOuterInterface(_tH)
    Np = V.getParticlesNumber(Particles, pointer = True)
    AABB = []
    for Zone in I.getZones(V.SafeZones) + I.getZones(HybridInterface) + I.getZones(_tLL):
        x, y, z = J.getxyz(Zone)
        AABB += [[np.min(x), np.min(y), np.min(z), np.max(x), np.max(y), np.max(z)]]

    AABB = np.array(AABB, dtype = np.float64)
    RedistributionKernel = V.RedistributionKernel_str2int[V.getParameter(_tL, 'RedistributionKernel')]
    N0 = Np[0]
    populationControlInfo = np.array([0]*5, dtype = np.int32)
    RedistributedParticles = V.population_control(_tL, AABB, RedistributionKernel,
                                                                          populationControlInfo)
    if RedistributedParticles.any():
        V.adjustTreeSize(_tL, NewSize = len(RedistributedParticles[0]), OldSize = N0)
        X, Y, Z = J.getxyz(Particles)
        AX, AY, AZ, S, Cs, Nu, Enstrophy, Age = J.getVars(Particles, V.vectorise('Alpha') + \
                                                       ['Sigma', 'Cvisq', 'Nu', 'Enstrophy', 'Age'])
        X[:]         = RedistributedParticles[0][:]
        Y[:]         = RedistributedParticles[1][:]
        Z[:]         = RedistributedParticles[2][:]
        AX[:]        = RedistributedParticles[3][:]
        AY[:]        = RedistributedParticles[4][:]
        AZ[:]        = RedistributedParticles[5][:]
        S[:]         = RedistributedParticles[6][:]
        Cs[:]        = RedistributedParticles[7][:]
        Nu[:]        = RedistributedParticles[8][:]
        Enstrophy[:] = RedistributedParticles[9][:]
        Age[:]       = np.array([int(a) for a in RedistributedParticles[10]], dtype = np.int32)
    else:
       V.adjustTreeSize(_tL, NewSize = Np[0], OldSize = N0)

    IterationInfo['Number of depleted particles'] = populationControlInfo[0]
    IterationInfo['Number of particles beyond cutoff'] = populationControlInfo[1]
    IterationInfo['Number of resized particles'] = populationControlInfo[2]
    IterationInfo['Number of split particles'] = populationControlInfo[3]
    IterationInfo['Number of merged particles'] = populationControlInfo[4]
    IterationInfo['Population Control time'] = J.tic() - IterationInfo['Population Control time']
    return IterationInfo

####################################################################################################
####################################################################################################
########################################### Vortex Rings ###########################################
####################################################################################################
####################################################################################################
def createLambOseenVortexRing(t = [], VPMParameters = {}, VortexParameters = {}):
    '''
    Initialises a Lamb Oseen vortex ring.

    Parameters
    ----------
        t : Tree
            Lagrangian field.

        VPMParameters : :py:class:`dict`
            Parameters of the VPM solver.

        VortexParameters : :py:class:`dict`
            Parameters of the vortex.
    '''
    Gamma = VortexParameters['Intensity'][0]
    sigma = VPMParameters['Sigma0'][0]
    lmbd_s = VPMParameters['SmoothingRatio'][0]
    nu = VPMParameters['KinematicViscosity'][0]
    R = VortexParameters['RingRadius'][0]
    nc = [0]
    h = VPMParameters['Resolution'][0]
    
    if 'CoreRadius' in VortexParameters:
        a = VortexParameters['CoreRadius'][0]
        if nu != 0.: tau = a*a/4./nu
        else: tau = 0.
        VortexParameters['Tau'] = np.array([tau], dtype = np.float64, order = 'F')
    elif 'Tau' in VortexParameters:
        tau = VortexParameters['Tau'][0]
        a = np.sqrt(4.*nu*tau)
        VortexParameters['CoreRadius'] = np.array([a], dtype = np.float64, order = 'F')

    a = VortexParameters['CoreRadius'][0]
    w = lambda r : Gamma/(np.pi*a*a)*np.exp(-r*r/(a*a))

    if 'MinimumVorticityFraction' in VortexParameters:
        r = a*np.sqrt(-np.log(VortexParameters['MinimumVorticityFraction'][0]*np.pi*a**2/Gamma))
        nc = int(r/h)
        VortexParameters['NumberLayers'] = np.array([nc], dtype = np.int32, order = 'F')
    elif 'NumberLayers' in VortexParameters:
        nc = VortexParameters['NumberLayers'][0]
        frac = w(nc*h) + 1e-15
        VortexParameters['MinimumVorticityFraction'] = \
                                              np.array([frac], dtype = np.float64, order = 'F')

    N_s = 1 + 3*nc*(nc + 1)
    N_phi = int(2.*np.pi*R/h)
    N_phi += N_phi%4
    Np = N_s*N_phi
    V.extend(t, Np)
    Particles = V.getFreeParticles(t)
    px, py, pz = J.getxyz(Particles)
    AlphaX, AlphaY, AlphaZ, VorticityX, VorticityY, VorticityZ, Sigma, Nu = \
                       J.getVars(Particles, V.vectorise(['Alpha', 'Vorticity']) + ['Sigma', 'Nu'])
    Nu[:] = nu
    
    r0 = h/2.
    rc = r0*(2*nc + 1)
    if (Np != N_phi*N_s): V.show("Achtung Bicyclette")
    if (R - rc < 0): V.show("Beware of the initial ring radius " , R , " < " , rc)
    else: V.show("R=", R, ", rc=", rc, ", a=", a, ", sigma=", sigma, ", nc=", nc, ", N_phi=",
                                                               N_phi, ", N_s=", N_s, ", N=", Np)

    X = [R]
    Z = [0.]
    W = [Gamma/(np.pi*a*a)]
    Vol = [2.*np.pi*np.pi*R/N_phi*r0*r0]
    for n in range(1, nc + 1):
        for j in range(6*n):
            theta = np.pi*(2.*j + 1.)/6./n
            r = r0*(1. + 12.*n*n)/6./n
            X.append(R + r*np.cos(theta))
            Z.append(r*np.sin(theta))
            Vol.append(4./3.*4.*np.pi*r0*r0/N_phi*(np.pi*R/2. + (np.sin(np.pi*(j + 1)/4./n) - \
                                                   np.sin(np.pi*j/4./n))*(4.*n*n + 1./3.)*r0))
            W.append(W[0]*np.exp(-r*r/(a*a)))

    V.show(W[0])
    S = [lmbd_s*h]*len(X)
    for i in range(N_phi):
        phi = 2.*np.pi*i/N_phi
        for j in range(N_s):
            px[i*N_s + j] = X[j]*np.cos(phi)
            py[i*N_s + j] = X[j]*np.sin(phi)
            pz[i*N_s + j] = Z[j]
            Sigma[i*N_s + j] = S[j]
            VorticityX[i*N_s + j] = -W[j]*np.sin(phi)
            VorticityY[i*N_s + j] = W[j]*np.cos(phi)
            VorticityZ[i*N_s + j] = 0.
            AlphaX[i*N_s + j] = Vol[j]*VorticityX[i*N_s + j]
            AlphaY[i*N_s + j] = Vol[j]*VorticityY[i*N_s + j]
            AlphaZ[i*N_s + j] = Vol[j]*VorticityZ[i*N_s + j]

    VortexParameters['Nphi'] = N_phi
    VortexParameters['Ns'] = N_s
    VortexParameters['NumberLayers'] = nc
    V.adjust_vortex_ring(t, N_s, N_phi, Gamma, np.pi*r0*r0, np.pi*r0*r0*4./3., nc)

def createLambOseenVortexBlob(t, VPMParameters, VortexParameters):
    '''
    Initialises a Lamb Oseen vortex discretised within a cylinder.

    Parameters
    ----------
        t : Tree
            Lagrangian field.

        VPMParameters : :py:class:`dict`
            Parameters of the VPM solver.

        VortexParameters : :py:class:`dict`
            Parameters of the vortex.
    '''
    Gamma = VortexParameters['Intensity'][0]
    sigma = VPMParameters['Sigma0'][0]
    lmbd_s = VPMParameters['SmoothingRatio'][0]
    nu = VPMParameters['KinematicViscosity'][0]
    L = VortexParameters['Length'][0]
    frac = VortexParameters['MinimumVorticityFraction'][0]
    h = VPMParameters['Resolution'][0]
    if 'CoreRadius' in VortexParameters:
        a = VortexParameters['CoreRadius'][0]
        tau = a*a/4./nu
        VortexParameters['Tau'] = np.array([tau], dtype = np.float64, order = 'F')
    elif 'Tau' in VortexParameters:
        tau = VortexParameters['Tau'][0]
        a = np.sqrt(4.*nu*tau)
        VortexParameters['CoreRadius'] = np.array([a], dtype = np.float64, order = 'F')

    w = lambda r : Gamma/(4.*np.pi*nu*tau)*np.exp(-r*r/(4.*nu*tau))
    minw = w(0)*frac
    l = a*np.sqrt(-np.log(frac*np.pi*a**2/Gamma))
    l = int(l/h)*h
    L = int(L/h)*h
    NL = 2*int(L/h) + 1
    nc = int(l/h + 0.5) + 1
    Ns = 1 + 3*nc*(nc + 1)

    l = nc*h
    L = NL*h
    VortexParameters['RingRadius'] = l
    VortexParameters['Length'] = L
    VortexParameters['Nphi'] = NL
    VortexParameters['Ns'] = Ns
    VortexParameters['NumberLayers'] = nc
    Np = NL*Ns
    V.extend(t, Np)
    Particles = V.getFreeParticles(t)
    px, py, pz = J.getxyz(Particles)
    AlphaX, AlphaY, AlphaZ, VorticityX, VorticityY, VorticityZ, Sigma, Nu = \
                       J.getVars(Particles, V.vectorise(['Alpha', 'Vorticity']) + ['Sigma', 'Nu'])
    Nu[:] = nu
    
    r0 = h/2.
    rc = r0*(2*nc + 1)
    V.show("L=", L, ", tau=", tau, ", rc=", l, ", a=", a, ", sigma=", sigma, ", NL=", NL, \
                                                                        ", Ns=", Ns, ", N=", Np)

    X = [0]
    Z = [0.]
    W = [Gamma/np.pi/a*a]
    V = [h*np.pi*r0*r0]
    for n in range(1, nc + 1):
        for j in range(6*n):
            theta = np.pi*(2.*j + 1.)/6./n
            r = r0*(1. + 12.*n*n)/6./n
            X.append(r*np.cos(theta))
            Z.append(r*np.sin(theta))
            V.append(V[0]*4./3.)
            W.append(W[0]*np.exp(-r*r/(a*a)))
    
    V.show("W in", w(0), np.min(W))
    S = [lmbd_s*h]*len(X)
    px[:Ns] = X[:]
    py[:Ns] = 0.
    pz[:Ns] = Z[:]
    Sigma[:Ns] = S[:]
    VorticityX[:Ns] = 0.
    VorticityY[:Ns] = W[:]
    VorticityZ[:Ns] = 0.
    AlphaX[:Ns] = 0.
    AlphaY[:Ns] = np.array(V[:])*np.array(W[:])
    AlphaZ[:Ns] = 0.
    for i in range(1, NL, 2):
        for j in range(Ns):
            px[i*Ns + j] = X[j]
            py[i*Ns + j] = (i//2 + 1)*h
            pz[i*Ns + j] = Z[j]
            Sigma[i*Ns + j] = S[j]
            VorticityX[i*Ns + j] = 0.
            VorticityY[i*Ns + j] = W[j]
            VorticityZ[i*Ns + j] = 0.
            AlphaX[i*Ns + j] = 0.
            AlphaY[i*Ns + j] = V[j]*VorticityY[i*Ns + j]
            AlphaZ[i*Ns + j] = 0.
        for j in range(Ns):
            px[(i + 1)*Ns + j] = X[j]
            py[(i + 1)*Ns + j] = -(i//2 + 1)*h
            pz[(i + 1)*Ns + j] = Z[j]
            Sigma[(i + 1)*Ns + j] = S[j]
            VorticityX[(i + 1)*Ns + j] = 0.
            VorticityY[(i + 1)*Ns + j] = W[j]
            VorticityZ[(i + 1)*Ns + j] = 0.
            AlphaX[(i + 1)*Ns + j] = 0.
            AlphaY[(i + 1)*Ns + j] = V[j]*VorticityY[(i + 1)*Ns + j]
            AlphaZ[(i + 1)*Ns + j] = 0.
    
    V.adjust_vortex_tube(t, Ns, NL, Gamma, np.pi*r0*r0, np.pi*r0*r0*4./3., nc)
