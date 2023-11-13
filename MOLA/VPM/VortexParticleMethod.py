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

import sys
import os
import numpy as np
import VortexParticleMethod.vortexparticlemethod as vpm_cpp
import Converter.PyTree as C
import Geom.PyTree as D
import Converter.Internal as I
import Generator.PyTree as G
import Transform.PyTree as T
import Connector.PyTree as CX
import Post.PyTree as P
import CPlot.PyTree as CPlot
from .. import LiftingLine as LL
from .. import Wireframe as W
from .. import InternalShortcuts as J
from .. import ExtractSurfacesProcessor as ESP

__version__ = '0.3'
__author__ = 'Johan VALENTIN'

if True:
####################################################################################################
####################################################################################################
######################################## CGNS tree control #########################################
####################################################################################################
####################################################################################################
    def delete(t = [], mask = []):
        Particles = pickParticlesZone(t)
        GridCoordinatesNode = I.getNodeFromName1(Particles, 'GridCoordinates')
        FlowSolutionNode = I.getNodeFromName1(Particles, 'FlowSolution')
        Np = Particles[1].ravel(order = 'F')[0]
        if Np != len(mask): raise ValueError('The length of the mask (%d) must be the same as the \
                                                          number of particles (%d)'%(len(mask), Np))
        mask = np.logical_not(mask)
        for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
            if node[3] == 'DataArray_t':
                node[1] = node[1][mask]

        Particles[1].ravel(order = 'F')[0] = len(node[1])

    def extend(t = [], ExtendSize = 0, Offset = 0, ExtendAtTheEnd = True):
        Particles = pickParticlesZone(t)
        Cvisq = getParameter(Particles, 'EddyViscosityConstant')
        if not Cvisq: Cvisq = 0.
        GridCoordinatesNode = I.getNodeFromName1(Particles, 'GridCoordinates')
        FlowSolutionNode = I.getNodeFromName1(Particles, 'FlowSolution')
        Np = len(J.getx(Particles))
        if Np < abs(Offset): raise ValueError('Offset (%d) cannot be greater than existing number \
                                                                    of particles (%d)'%(Offset, Np))
        if ExtendAtTheEnd:
            for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
                if node[3] == 'DataArray_t':
                    zeros = np.array(np.zeros(ExtendSize) + (node[0] == 'Active') + Cvisq*(node[0] \
                                                    == 'Cvisq'), dtype = node[1].dtype, order = 'F')
                    node[1] = np.append(np.append(node[1][:Np -Offset], zeros), node[1][Np-Offset:])

        else:
            for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
                if node[3] == 'DataArray_t':
                    zeros = np.array(np.zeros(ExtendSize) + (node[0] == 'Active') + Cvisq*(node[0] \
                                                    == 'Cvisq'), dtype = node[1].dtype, order = 'F')
                    node[1] = np.append(np.append(node[1][:Offset], zeros), node[1][Offset:])

        Particles[1].ravel(order = 'F')[0] = len(node[1])

    def addParticlesToTree(t, NewX = [], NewY = [], NewZ = [], NewAX = [], NewAY = [], NewAZ = [],
        NewSigma = [], Offset = 0, ExtendAtTheEnd = False):
        Nnew = len(NewX)
        extend(t, ExtendSize = Nnew, Offset = Offset, ExtendAtTheEnd = ExtendAtTheEnd)
        Particles = pickParticlesZone(t)
        Nnew += Offset
        px, py, pz = J.getxyz(Particles)
        px[Offset: Nnew] = NewX
        py[Offset: Nnew] = NewY
        pz[Offset: Nnew] = NewZ
        AX, AY, AZ, WX, WY, WZ, A, W, Nu, Volume, Sigma = J.getVars(Particles, \
                                     ['Alpha'+i for i in 'XYZ'] + ['Vorticity'+i for i in 'XYZ'] + \
                               ['StrengthMagnitude', 'VorticityMagnitude', 'Nu', 'Volume', 'Sigma'])
        AX[Offset: Nnew] = NewAX
        AY[Offset: Nnew] = NewAY
        AZ[Offset: Nnew] = NewAZ
        A[Offset: Nnew] = np.linalg.norm(np.vstack([NewAX, NewAY, NewAZ]),axis=0)
        Sigma[Offset: Nnew] = NewSigma
        #Volume[Offset: Nnew] = NewSigma**3
        #Nu[Offset: Nnew] = getParameter(Particles, 'KinematicViscosity')
        #WX[Offset: Nnew] = NewAX/Volume[Offset: Nnew]
        #WY[Offset: Nnew] = NewAY/Volume[Offset: Nnew]
        #WZ[Offset: Nnew] = NewAZ/Volume[Offset: Nnew]

    def trim(t = [], NumberToTrim = 0, Offset = 0, TrimAtTheEnd = True):
        Particles = pickParticlesZone(t)
        GridCoordinatesNode = I.getNodeFromName1(Particles, 'GridCoordinates')
        FlowSolutionNode = I.getNodeFromName1(Particles, 'FlowSolution')
        Np = len(J.getx(Particles))
        if Np < abs(Offset): raise ValueError('Offset (%d) cannot be greater than existing number \
                                                                    of particles (%d)'%(Offset, Np))
        if TrimAtTheEnd:
            for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
                if node[3] == 'DataArray_t':
                    node[1] = np.append(node[1][:Np - Offset - NumberToTrim], node[1][Np - Offset:])
        else:
            for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
                if node[3] == 'DataArray_t':
                    node[1] = np.append(node[1][:Offset], node[1][Offset + NumberToTrim:])

        Particles[1].ravel(order = 'F')[0] = len(node[1])

    def adjustTreeSize(t = [], NewSize = 0, OldSize = -1, Offset = 0, AtTheEnd = True):
        Particles = pickParticlesZone(t)
        if OldSize == -1: OldSize = len(J.getx(Particles))
        SizeDiff = NewSize - OldSize

        if 0 < SizeDiff:extend(t, ExtendSize = SizeDiff, Offset = Offset, ExtendAtTheEnd = AtTheEnd)
        else: trim(t, NumberToTrim = -SizeDiff, Offset = Offset, TrimAtTheEnd = AtTheEnd)

    def roll(t = [], PivotNumber = 0):
        Particles = pickParticlesZone(t)
        Np = Particles[1].ravel(order = 'F')[0]
        if PivotNumber == 0 or PivotNumber == Np: return
        if PivotNumber > Np:
            raise ValueError('PivotNumber (%d) cannot be greater than existing number of particles\
                                                                            (%d)'%(PivotNumber, Np))
        GridCoordinatesNode = I.getNodeFromName1(Particles, 'GridCoordinates')
        FlowSolutionNode = I.getNodeFromName1(Particles, 'FlowSolution')
        for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
            if node[3] == 'DataArray_t':
                node[1] = np.roll(node[1], PivotNumber)

    def checkParametersTypes(ParametersList = [], int_Params = [], float_Params = [],
        bool_Params = []):
        for Parameters in ParametersList:
            for key in Parameters:
                if key in int_Params:
                    Parameters[key] = np.atleast_1d(np.array(Parameters[key],
                                                            dtype = np.int32))
                elif key in float_Params:
                    Parameters[key] = np.atleast_1d(np.array(Parameters[key],
                                                            dtype = np.float64))
                elif key in bool_Params:
                    Parameters[key] = np.atleast_1d(np.array(Parameters[key],
                                                            dtype = np.int32))

    def getPerturbationFieldParameters(t = []):
        return J.get(pickParticlesZone(t), '.PerturbationField#Parameters')

    def getParameter(t = [], Name = ''):
        Particles = pickParticlesZone(t)
        Node = getVPMParameters(Particles)
        if Name in Node:
            ParameterNode = Node[Name]
            if (isinstance(ParameterNode, list) or isinstance(ParameterNode, np.ndarray)):
                return ParameterNode
        else :ParameterNode = None
        if ParameterNode == None:
            Node = getLiftingLineParameters(Particles)
            if Node and Name in Node: ParameterNode = Node[Name]
        if ParameterNode == None:
            Node = getHybridParameters(Particles)
            if Node and Name in Node: ParameterNode = Node[Name]
        if ParameterNode == None:
            Node = getPerturbationFieldParameters(Particles)
            if Node and Name in Node: ParameterNode = Node[Name]
        return ParameterNode

    def getParameters(t = [], Names = []):
        Particles = pickParticlesZone(t)
        return [getParameter(Particles, Name) for Name in Names]

####################################################################################################
####################################################################################################
############################################### VPM ################################################
####################################################################################################
####################################################################################################
    Kernel_str2int = {'Gaussian': 0, 'Gaussian2': 0, 'G': 0, 'G2': 0, 'HOA': 1, 'HighOrderAlgebraic'
         : 1, 'Gaussian3': 2, 'G3': 2, 'LOA': 3, 'LowOrderAlgebraic': 3,'SuperGaussian': 4, 'SP': 4}
    Scheme_str2int = {'Transpose': 0, 'T': 0, 'Mixed': 1, 'M': 1, 'Classical': 2, 'C': 2}
    EddyViscosityModel_str2int = {'Mansour': 0, 'Smagorinsky': 1, 'Vreman': 2, None: -1, 'None': -1,
                                                                                     'Mansour2': -2}
    RedistributionKernel_str2int = {'M4Prime': 5, 'M4': 4, 'M3': 3, 'M2': 2, 'M1': 1, None: 0,
                                                                                          'None': 0}
    DiffusionScheme_str2int = {'PSE': 1, 'ParticleStrengthExchange': 1, 'pse': 1, 'DVM': 2,'dvm': 2,
        'DiffusionVelocityMethod': 2, 'CSM': 3, 'CS': 3, 'csm': 3, 'cs': 3, 'CoreSpreading': 3,
        'CoreSpreadingMethod': 3, 'None': 0, None: 0}

    def buildEmptyVPMTree():
        Particles = C.convertArray2Node(D.line((0., 0., 0.), (0., 0., 0.), 2))
        Particles[0] = 'Particles'


        FieldNames = ['VelocityInduced' + v for v in 'XYZ'] + ['Vorticity' + v for v in 'XYZ'] + \
                     ['Alpha' + v for v in 'XYZ'] + ['gradxVelocity' + v for v in 'XYZ'] + \
                     ['gradyVelocity' + v for v in 'XYZ']+ ['gradzVelocity' + v for v in 'XYZ'] + \
                     ['PSE' + v for v in 'XYZ'] + ['Stretching' + v for v in 'XYZ'] + \
                     ['VelocityDiffusion' + v for v in 'XYZ'] + ['Active', 'Age', 'HybridFlag', \
                      'Nu', 'Sigma', 'StrengthMagnitude', 'Volume', 'VorticityMagnitude', 'divUd', \
                      'Enstrophy', 'Enstrophyf', 'KineticEnergy', 'KineticEnergyf', 'SFS', 'Cvisq',\
                      'KineticEnergyVariation', 'EnstrophyVariation'] + ['VelocityPerturbation' + \
                      v for v in 'XYZ']
        J.invokeFieldsDict(Particles, FieldNames)

        IntegerFieldNames = ['Active', 'Age', 'HybridFlag']
        GridCoordinatesNode = I.getNodeFromName1(Particles, 'GridCoordinates')
        FlowSolutionNode = I.getNodeFromName1(Particles, 'FlowSolution')
        for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
            if node[3] == 'DataArray_t':
                if node[0] in IntegerFieldNames:
                    node[1] = np.array([], dtype = np.int32, order = 'F')
                else:
                    node[1] = np.array([], dtype = np.float64, order = 'F')

        Particles[1].ravel(order = 'F')[0] = 0
        I._sortByName(I.getNodeFromName1(Particles, 'FlowSolution'))
        return C.newPyTree(['ParticlesBase', Particles])

    def initialiseVPM(EulerianMesh = [], LiftingLineTree = [], PerturbationField = [],
        PolarInterpolator = [], VPMParameters = {}, HybridParameters = {},
        LiftingLineParameters = {}, PerturbationFieldParameters = {}):
        int_Params =['MaxLiftingLineSubIterations', 'MaximumSubIteration','StrengthRampAtbeginning', 
            'MinNbShedParticlesPerLiftingLine', 'CurrentIteration', 'NumberOfHybridInterfaces',
            'MaximumAgeAllowed', 'RedistributionPeriod', 'NumberOfThreads', 'IntegrationOrder',
            'IterationTuningFMM', 'IterationCounter', 'NumberOfNodes',
            'NumberOfParticlesPerInterface', 'ClusterSize',
            'FarFieldApproximationOrder', 'NumberOfLiftingLineSources', 'NumberOfHybridSources']

        float_Params = ['Density', 'EddyViscosityConstant', 'Temperature', 'ResizeParticleFactor',
            'Time', 'CutoffXmin', 'CutoffZmin', 'MaximumMergingVorticityFactor', 
            'SFSContribution', 'SmoothingRatio', 'CirculationThreshold', 'RPM','KinematicViscosity',
            'CirculationRelaxation', 'Pitch', 'CutoffXmax', 'CutoffYmin', 'CutoffYmax', 'Sigma0',
            'CutoffZmax', 'ForcedDissipation','MaximumAngleForMerging', 'MinimumVorticityFactor', 
            'RelaxationFactor', 'MinimumOverlapForMerging', 'VelocityFreestream', 'AntiStretching',
            'RelaxationThreshold', 'RedistributeParticlesBeyond', 'RedistributeParticleSizeFactor',
            'TimeStep', 'Resolution', 'VelocityTranslation', 'NearFieldOverlappingRatio', 'TimeFMM',
            'RemoveWeakParticlesBeyond', 'OuterDomainToWallDistance', 'InnerDomainToWallDistance',
            'MagnitudeRelaxationFactor', 'EddyViscosityRelaxationFactor', 'TimeVelPert', 'Limitor',
            'FactorOfCFDVorticityShed', 'RealignmentRelaxationFactor', 'MachLimitor']

        bool_Params = ['MonitorDiagnostics', 'LowStorageIntegration']

        defaultParameters = {
            ############################################################################################
            ################################### Simulation Conditions ##################################
            ############################################################################################
                'Density'                       : 1.225,          #]0., +inf[, in kg.m^-3
                'EddyViscosityConstant'         : 0.15,           #[0., +inf[, constant for the eddy viscosity model, Cm(Mansour) around 0.1, Cs(Smagorinsky) around 0.15, Cr(Vreman) around 0.07
                'EddyViscosityModel'            : 'Vreman',       #Mansour, Mansour2, Smagorinsky, Vreman or None, select a LES model to compute the eddy viscosity
                'KinematicViscosity'            : 1.46e-5,        #[0., +inf[, in m^2.s^-1
                'Temperature'                   : 288.15,         #]0., +inf[, in K
                'Time'                          : 0.,             #in s, keep track of the physical time
            ############################################################################################
            ###################################### VPM Parameters ######################################
            ############################################################################################
                'AntiStretching'                : 0.,             #between 0 and 1, 0 means particle strength fully takes vortex stretching, 1 means the particle size fully takes the vortex stretching
                'DiffusionScheme'               : 'PSE',          #PSE, CSM or None. gives the scheme used to compute the diffusion term of the vorticity equation
                'RegularisationKernel'          : 'Gaussian',     #The available smoothing kernels are Gaussian, HOA, LOA, Gaussian3 and SuperGaussian
                'SFSContribution'               : 0.,             #between 0 and 1, the closer to 0, the more the viscosity affects the particle strength, the closer to 1, the more it affects the particle size
                'SmoothingRatio'                : 2.,             #in m, anywhere between 1.5 and 2.5, the higher the NumberSource, the smaller the Resolution and the higher the SmoothingRatio should be to avoid blowups, the HOA kernel requires a higher smoothing
                'VorticityEquationScheme'       : 'Transpose',    #Classical, Transpose or Mixed, The schemes used to compute the vorticity equation are the classical scheme, the transpose scheme (conserves total vorticity) and the mixed scheme (a fusion of the previous two)
                'MachLimitor'                   : 0.9,            #[0, +in[, gives the maximum velocity a particle can have
                'StrengthVariationLimitor'      : 2.,             #[0, +in[, gives the maximum ratio a particle can grow/shrink when updated with the vorticity equation
            ############################################################################################
            ################################### Numerical Parameters ###################################
            ############################################################################################
                'CurrentIteration'              : 0,              #follows the current iteration
                'IntegrationOrder'              : 3,              #[|1, 4|]1st, 2nd, 3rd or 4th order Runge Kutta. In the hybrid case, there must be at least as much Interfaces in the hybrid domain as the IntegrationOrder of the time integration scheme
                'LowStorageIntegration'         : True,           #[|0, 1|], states if the classical or the low-storage Runge Kutta is used
                'MonitorDiagnostics'            : True,           #[|0, 1|], allows or not the computation of the diagnostics (kinetic energy, enstrophy, divergence-free kinetic energy, divergence-free enstrophy)
            ############################################################################################
            ##################################### Particles Control ####################################
            ############################################################################################
                'CutoffXmin'                    : -np.inf,        #in m, spatial Cutoff
                'CutoffXmax'                    : +np.inf,        #in m, spatial Cutoff
                'CutoffYmin'                    : -np.inf,        #in m, spatial Cutoff
                'CutoffYmax'                    : +np.inf,        #in m, spatial Cutoff
                'CutoffZmin'                    : -np.inf,        #in m, spatial Cutoff
                'CutoffZmax'                    : +np.inf,        #in m, spatial Cutoff
                'ForcedDissipation'             : 0.,             #in %/s, gives the % of strength particles looses per sec, usefull to kill unnecessary particles without affecting the LLs
                'MaximumAgeAllowed'             : 0,              #0 <=,  particles are eliminated after MaximumAgeAllowed iterations, if MaximumAgeAllowed == 0, they are not deleted
                'MaximumAngleForMerging'        : 0.,             #[0., 180.[ in deg,   maximum angle   allowed between two particles to be merged
                'MaximumMergingVorticityFactor' : 0.,             #in %, particles can be merged if their combined strength is below the given poucentage of the maximum strength on the blades
                'MinimumOverlapForMerging'      : 0.,             #[0., +inf[, if two particles have at least an overlap of MinimumOverlapForMerging*SigmaRatio, they are considered for merging
                'MinimumVorticityFactor'        : 0.,             #in %, sets the minimum strength kept as a percentage of the maximum strength on the blades
                'RedistributeParticlesBeyond'   : np.inf,         #do not redistribute particles if closer than RedistributeParticlesBeyond*Resolution from a LL
                'RedistributionKernel'          : None,           #M4Prime, M4, M3, M2, M1 or None, redistribution kernel used. the number gives the order preserved by the kernel, if None local splitting/merging is used
                'RedistributionPeriod'          : 0,              #frequency at which particles are redistributed, if 0 the particles are never redistributed
                'RealignmentRelaxationFactor'   : 0.,             #[0., 1.[, is used during the relaxation process to realign the particles with their voticity and avoid having a non null divergence of the vorticity field
                'MagnitudeRelaxationFactor'     : 0.,             #[0., 1.[, is used during the relaxation process to change the magnitude of the particles to avoid having a non null divergence of the vorticity field
                'EddyViscosityRelaxationFactor' : 0.,             #[0., 1.[, is used during the relaxation process when updating the eddy viscosity constant to satisfy the transfert of enstrophy to the kinetic energy
                'RemoveWeakParticlesBeyond'     : np.inf,         #do not remove weak particles if closer than RemoveWeakParticlesBeyond*Resolution from a LL
                'ResizeParticleFactor'          : 0.,             #[0, +inf[, resize particles that grow/shrink RedistributeParticleSizeFactor * Sigma0 (i.e. Resolution*SmoothingRatio), if 0 then no resizing is done
                'StrengthRampAtbeginning'       : 25,             #[|0, +inf [|, limit the vorticity shed for the StrengthRampAtbeginning first iterations for the wake to stabilise
            ############################################################################################
            ###################################### FMM Parameters ######################################
            ############################################################################################
                'FarFieldApproximationOrder'    : 8,              #[|6, 12|], order of the polynomial which approximates the far field interactions, the higher the more accurate and the more costly
                'IterationTuningFMM'            : 50,             #frequency at which the FMM is compared to the direct computation, gives the relative L2 error
                'NearFieldOverlappingRatio'     : 0.5,            #[0., 1.], Direct computation of the interactions between clusters that overlap by NearFieldOverlappingRatio, the smaller the more accurate and the more costly
                'NumberOfThreads'               : 'auto',         #number of threads of the machine used. If 'auto', the highest number of threads is set
                'TimeFMM'                       : 0.,             #in s, keep track of the CPU time spent for the FMM
                'ClusterSize'                   : 2**9,           #[|0, +inf[|, maximum number of particles per FMM cluster, better as a power of 2
        }
        defaultHybridParameters = {
            ############################################################################################
            ################################ Hybrid Domain Parameters ################################
            ############################################################################################
                #'BCFarFieldName'                   : 'farfield',#the name of the farfield boundary condition from which the Outer Interface is searched
                'MaximumSubIteration'              : 100,       #[|0, +inf[|, gives the maximum number of sub-iterations when computing the strength of the particles generated from the vorticity on the Interfaces
                'NumberOfHybridInterfaces'         : 4.,         #|]0, +inf[|, number of interfaces in the Hybrid Domain from which hybrid particles are generated
                'NumberOfParticlesPerInterface'    : 500,        #[|0, +inf[|, number of particles generated per hybrid interface
                'OuterDomainToWallDistance'        : 0.3,        #]0, +inf[ in m, distance between the wall and the end of the Hybrid Domain
                'FactorOfCFDVorticityShed'         : 75.,        #[0, 100] in %, gives the percentage of vorticity from the Hybrid Domain that is converted into hybrid particles
                'RelaxationFactor'                 : 0.5,       #[0, +inf[, gives the relaxtion factor used for the relaxation process when computing the strength of the particles generated from the vorticity on the Interface
                'RelaxationThreshold'              : 1e-6,      #[0, +inf[ in m^3.s^-1, gives the convergence criteria for the relaxtion process when computing the strength of the particles generated from the vorticity on the Interface
        }
        defaultLiftingLineParameters = {
            ############################################################################################
            ################################# Lifting Lines Parameters #################################
            ############################################################################################
                'CirculationThreshold'             : 1e-4,     #convergence criteria for the circulation sub-iteration process, somewhere between 1e-3 and 1e-6 is ok
                'CirculationRelaxation'            : 1./3.,    #relaxation parameter of the circulation sub-iterations, somwhere between 0.1 and 1 is good, the more unstable the simulation, the lower it should be
                'IntegralLaw'                      : 'linear', #uniform, tanhOneSide, tanhTwoSides or ratio, gives the type of interpolation of the circulation on the lifting lines
                'MaxLiftingLineSubIterations'      : 100,      #max number of sub iteration when computing the LL circulations
                'MinNbShedParticlesPerLiftingLine' : 25,       #minimum number of station for every LL from which particles are shed
                'NumberOfLiftingLineSources'       : 0,        #total number of particle sources on the lifting lines
                'Pitch'                            : 0.,       #]-180., 180[ in deg, gives the pitch given to all the lifting lines, if 0 no pitch id added
        }
        defaultVelocityPertParameters = {
            ############################################################################################
            ############################## Perturbation Field Parameters ###############################
            ############################################################################################
                'NumberOfNodes'             : 0,
                'NearFieldOverlappingRatio' : 0.5,
                'TimeVelocityPerturbation'  : 0.,
        }
        if True:
            defaultParameters.update(VPMParameters)
            if PerturbationField: defaultVelocityPertParameters.update(PerturbationFieldParameters)
            else: defaultVelocityPertParameters = {}
            if EulerianMesh: defaultHybridParameters.update(HybridParameters)
            else: defaultHybridParameters = {}
            if LiftingLineTree: defaultLiftingLineParameters.update(LiftingLineParameters)
            else: defaultLiftingLineParameters = {}

            if defaultParameters['NumberOfThreads'] == 'auto':
                NbOfThreads = int(os.getenv('OMP_NUM_THREADS',len(os.sched_getaffinity(0))))
                defaultParameters['NumberOfThreads'] = NbOfThreads
            else: NbOfThreads = defaultParameters['NumberOfThreads']
            os.environ['OMP_NUM_THREADS'] = str(NbOfThreads)
            #vpm_cpp.mpi_init(defaultParameters['NumberOfThreads']);
            checkParametersTypes([defaultParameters, defaultHybridParameters,
                               defaultLiftingLineParameters], int_Params, float_Params, bool_Params)
            renameLiftingLineTree(LiftingLineTree, defaultParameters, defaultLiftingLineParameters)
            updateParameters(LiftingLineTree, defaultParameters, defaultLiftingLineParameters)
            updateLiftingLines(LiftingLineTree, defaultLiftingLineParameters, defaultParameters['TimeStep'])
            tE = []
            t = buildEmptyVPMTree()
            Particles = pickParticlesZone(t)
            if LiftingLineTree:
                print('||' + '{:=^50}'.format(' Initialisation of Lifting Lines '))
                LiftingLines = I.getZones(LiftingLineTree)
                initialiseParticlesOnLitingLine(t, LiftingLines, PolarInterpolator,
                                                    defaultParameters, defaultLiftingLineParameters)
                J.set(Particles, '.LiftingLine#Parameters', **defaultLiftingLineParameters)
                I._sortByName(I.getNodeFromName1(Particles, '.LiftingLine#Parameters'))
                print('||' + '{:-^50}'.format(' Done '))
            if EulerianMesh:
                print('||' + '{:=^50}'.format(' Initialisation of Hybrid Domain '))
                NumberOfLiftingLineSources = 0
                if 'NumberOfLiftingLineSources' in defaultLiftingLineParameters:
                    NumberOfLiftingLineSources = \
                                        defaultLiftingLineParameters['NumberOfLiftingLineSources'][0]
                if type(EulerianMesh) == str: tE = open(EulerianMesh)
                #tE = updateField(tE, defaultParameters, defaultHybridParameters)
                tE = generateMirrorWing(tE, defaultParameters, defaultHybridParameters)
                HybridDomain = generateHybridDomain(tE, defaultParameters, defaultHybridParameters)
                initialiseHybridParticles(t, tE, defaultParameters, 
                                        defaultHybridParameters, Offset = NumberOfLiftingLineSources)
                J.set(Particles, '.Hybrid#Parameters', **defaultHybridParameters)
                I._sortByName(I.getNodeFromName1(Particles, '.Hybrid#Parameters'))
                print('||' + '{:-^50}'.format(' Done '))

            J.set(Particles, '.VPM#Parameters', **defaultParameters)
            I._sortByName(I.getNodeFromName1(Particles, '.VPM#Parameters'))
            if defaultParameters['MonitorDiagnostics']:
                J.set(Particles, '.VPM#Diagnostics',
                    AngularImpulse = np.zeros(3, dtype = np.float64, order = 'F'),
                    Enstrophy = np.zeros(1, dtype = np.float64, order = 'F'),
                    EnstrophyDivFree = np.zeros(1, dtype = np.float64, order = 'F'),
                    LinearImpulse = np.zeros(3, dtype = np.float64, order = 'F'),
                    Omega = np.zeros(3, dtype = np.float64, order = 'F'))

            if PerturbationField:
                print('||' + '{:=^50}'.format(' Initialisation of Perturbation Field '))
                if type(PerturbationField) == str: PerturbationField = open(PerturbationField)
                NumberOfNodes = np.array([0], dtype = np.int32, order = 'F')
                defaultVelocityPertParameters['NumberOfNodes'] = NumberOfNodes
                J.set(Particles, '.PerturbationField#Parameters', **defaultVelocityPertParameters)
                t = I.merge([t, PerturbationField])

                PerturbationFieldBase = I.newCGNSBase('PerturbationField', cellDim=1, physDim=3)
                PerturbationFieldBase[2] = I.getZones(PerturbationField)
                PerturbationFieldCapsule = vpm_cpp.build_perturbation_velocity_capsule(\
                                                               PerturbationFieldBase, NumberOfNodes)
                print('||' + '{:-^50}'.format(' Done '))
            else: PerturbationFieldCapsule = None
            if LiftingLineTree:
                print('||' + '{:=^50}'.format(' Generate Lifting Lines Particles '))
                t = I.merge([t, LiftingLineTree])
                ShedVorticitySourcesFromLiftingLines(t, PolarInterpolator, 
                                                PerturbationFieldCapsule = PerturbationFieldCapsule)
        
        if EulerianMesh:
            print('||' + '{:=^50}'.format(' Generate Hybrid Particles '))
            t = I.merge([t, HybridDomain])
            Nh = getParameter(t, 'NumberOfHybridSources')
            Nhcurrent = splitHybridParticles(t)
            Nh[0] = Nhcurrent
            print('||' + '{:27}'.format('Number of Hybrid particles') + ': ' + '{:d}'.format(Nh[0]))
            print('||' + '{:-^50}'.format(' Inverse BEM Matrix '))
            updateBEMMatrix(t)
            print('||' + '{:-^50}'.format(' Enforce BC on Solid '))
            updateBEMSources(t)
            print('||' + '{:-^50}'.format(' Done '))

        solveVorticityEquation(t, PerturbationFieldCapsule = PerturbationFieldCapsule)
        IterationCounter = I.getNodeFromName(t, 'IterationCounter')
        IterationCounter[1][0] = defaultParameters['IterationTuningFMM']*\
                                                               defaultParameters['IntegrationOrder']
        return t, tE, PerturbationFieldCapsule

    def pickParticlesZone(t = []):
        for z in I.getZones(t):
            if z[0] == 'Particles':
                return z
        return []

    def pickPerturbationFieldZone(t = []):
        for z in I.getZones(t):
            if z[0] == 'PerturbationField':
                return [z]
        return []

    def getVPMParameters(t = []): return J.get(pickParticlesZone(t), '.VPM#Parameters')

    def solveParticleStrength(t = [], VorticityX = [], VorticityY = [], VorticityZ = [],Offset = 0):
        return vpm_cpp.solve_particle_strength(t, VorticityX, VorticityY, VorticityZ, Offset)

    def maskParticlesInsideShieldBoxes(t = [], Boxes = []):
        BoxesBase = I.newCGNSBase('ShieldBoxes', cellDim=1, physDim=3)
        BoxesBase[2] = I.getZones(Boxes)
        return vpm_cpp.box_interaction(t, BoxesBase)

    def getInducedVelocityFromWake(t = [], Targets = []):
        TargetsBase = I.newCGNSBase('Targets', cellDim=1, physDim=3)
        TargetsBase[2] = I.getZones(Targets)
        Kernel = Kernel_str2int[getParameter(t, 'RegularisationKernel')]
        return vpm_cpp.induce_velocity_from_wake(t, TargetsBase, Kernel)

    def findMinimumDistanceBetweenParticles(X = [], Y = [], Z = []):
        return vpm_cpp.find_minimum_distance_between_particles(X, Y, Z)

    def findParticleClusters(X = [], Y = [], Z = [], ClusterSize = 0.):
        return vpm_cpp.find_particle_clusters(X, Y, Z, ClusterSize)

    def solveVorticityEquation(t = [], IterationInfo = {}, PerturbationFieldCapsule = []):
        Kernel = Kernel_str2int[getParameter(t, 'RegularisationKernel')]
        Scheme = Scheme_str2int[getParameter(t, 'VorticityEquationScheme')]
        Diffusion = DiffusionScheme_str2int[getParameter(t, 'DiffusionScheme')]
        EddyViscosityModel = EddyViscosityModel_str2int[getParameter(t, 'EddyViscosityModel')]
        MonitorDiagnostics = getParameters(t, ['MonitorDiagnostics'])[0]
        PertubationFieldBase = I.newCGNSBase('PertubationFieldBase', cellDim=1, physDim=3)
        PertubationFieldBase[2] = pickPerturbationFieldZone(t)
        solveVorticityEquationInfo = vpm_cpp.wrap_vpm_solver(t, PertubationFieldBase,
                                                PerturbationFieldCapsule, Kernel, Scheme, Diffusion,
                                                        EddyViscosityModel, int(MonitorDiagnostics))
        IterationInfo['Number of threads'] = int(solveVorticityEquationInfo[0])
        IterationInfo['SIMD vectorisation'] = int(solveVorticityEquationInfo[1])
        IterationInfo['Near field overlapping ratio'] = solveVorticityEquationInfo[2]
        IterationInfo['Far field polynomial order'] = int(solveVorticityEquationInfo[3])
        IterationInfo['FMM time'] = solveVorticityEquationInfo[4]
        if PerturbationFieldCapsule:
            IterationInfo['Perturbation time'] = solveVorticityEquationInfo[-1]
            solveVorticityEquationInfo = solveVorticityEquationInfo[:-1]
        if len(solveVorticityEquationInfo) != 5:
            IterationInfo['Rel. err. of Velocity'] = solveVorticityEquationInfo[5]
            IterationInfo['Rel. err. of Vorticity'] = solveVorticityEquationInfo[6]
            IterationInfo['Rel. err. of Velocity Gradient'] = solveVorticityEquationInfo[7]
            if len(solveVorticityEquationInfo) == 9: 
                IterationInfo['Rel. err. of PSE'] = solveVorticityEquationInfo[8]
            if len(solveVorticityEquationInfo) == 10: 
                IterationInfo['Rel. err. of PSE'] = solveVorticityEquationInfo[8]
                IterationInfo['Rel. err. of Diffusion Velocity'] = solveVorticityEquationInfo[9]

        return IterationInfo

    def computeNextTimeStep(t = [], NoDissipationRegions=[], PerturbationFieldCapsule = []):
        Particles = pickParticlesZone(t)
        LiftingLines = LL.getLiftingLines(t)
        HybridInterface = pickHybridDomainOuterInterface(t)
        NoDissipationRegions.extend(LiftingLines)
        NoDissipationRegions.extend(HybridInterface)
        
        time, dt, it, IntegOrder, lowstorage = getParameters(t,
                ['Time','TimeStep', 'CurrentIteration', 'IntegrationOrder','LowStorageIntegration'])
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
        Kernel = Kernel_str2int[getParameter(t, 'RegularisationKernel')]
        Scheme = Scheme_str2int[getParameter(t, 'VorticityEquationScheme')]
        Diffusion = DiffusionScheme_str2int[getParameter(t, 'DiffusionScheme')]
        EddyViscosityModel = EddyViscosityModel_str2int[getParameter(t, 'EddyViscosityModel')]
        PertubationFieldBase = I.newCGNSBase('PertubationFieldBase', cellDim=1, physDim=3)
        PertubationFieldBase[2] = pickPerturbationFieldZone(t)
        if lowstorage:
            vpm_cpp.runge_kutta_low_storage(t, PertubationFieldBase, PerturbationFieldCapsule, a, b,
                                                      Kernel, Scheme, Diffusion, EddyViscosityModel)
        else:
            vpm_cpp.runge_kutta(t, PertubationFieldBase, PerturbationFieldCapsule, a, b, Kernel,
                                                               Scheme, Diffusion,EddyViscosityModel)
        for LiftingLine in LiftingLines:
            TimeShed = I.getNodeFromName(LiftingLine, 'TimeSinceLastShedding')
            TimeShed[1][0] += dt[0]
        time += dt
        it += 1

    def populationControl(t = [], NoRedistributionRegions = [], IterationInfo = {}):
        IterationInfo['Population Control time'] = J.tic()
        LiftingLines = LL.getLiftingLines(t)
        Particles = pickParticlesZone(t)
        HybridInterface = pickHybridDomainOuterInterface(t)
        Np = Particles[1][0]
        AABB = []
        for LiftingLine in I.getZones(LiftingLines):
            x, y, z = J.getxyz(LiftingLine)
            AABB += [[min(x), min(y), min(z), max(x), max(y), max(z)]]

        for BC in I.getZones(HybridInterface):
            x, y, z = J.getxyz(BC)
            AABB += [[min(x), min(y), min(z), max(x), max(y), max(z)]]

        AABB = np.array(AABB, dtype = np.float64)
        RedistributionKernel = RedistributionKernel_str2int[getParameter(t, 'RedistributionKernel')]
        N0 = Np[0]
        populationControlInfo = np.array([0, 0, 0, 0], dtype = np.int32)
        RedistributedParticles = vpm_cpp.population_control(t, AABB, RedistributionKernel,
                                                                              populationControlInfo)
        if RedistributedParticles.any():
            adjustTreeSize(t, NewSize = len(RedistributedParticles[0]), OldSize = N0)
            X, Y, Z = J.getxyz(Particles)
            AX, AY, AZ, AMag, Vol, S, Age, Cvisq, Enstrophy = J.getVars(Particles, \
                                              ['Alpha' + v for v in 'XYZ'] + ['StrengthMagnitude', \
                                                    'Volume', 'Sigma', 'Age', 'Cvisq', 'Enstrophy'])

            X[:]         = RedistributedParticles[0][:]
            Y[:]         = RedistributedParticles[1][:]
            Z[:]         = RedistributedParticles[2][:]
            AX[:]        = RedistributedParticles[3][:]
            AY[:]        = RedistributedParticles[4][:]
            AZ[:]        = RedistributedParticles[5][:]
            AMag[:]      = RedistributedParticles[6][:]
            Vol[:]       = RedistributedParticles[7][:]
            S[:]         = RedistributedParticles[8][:]
            Age[:]       = np.array([int(a) for a in RedistributedParticles[9]], dtype = np.int32)
            Cvisq[:]     = RedistributedParticles[10][:]
            Enstrophy[:] = RedistributedParticles[11][:]
        else:
           adjustTreeSize(t, NewSize = Np[0], OldSize = N0)

        IterationInfo['Number of particles beyond cutoff'] = populationControlInfo[0]
        IterationInfo['Number of split particles'] = populationControlInfo[1]
        IterationInfo['Number of depleted particles'] = populationControlInfo[2]
        IterationInfo['Number of merged particles'] = populationControlInfo[3]
        IterationInfo['Population Control time'] = J.tic() -IterationInfo['Population Control time']
        return IterationInfo

####################################################################################################
####################################################################################################
############################################## Hybrid ##############################################
####################################################################################################
####################################################################################################
    def generateMirrorWing(tE = [], VPMParameters = {}, HybridParameters = {}):
        if type(tE) == str: tE = I.getNodeFromName(open(tE), 'BaseWing')
        Zones = I.getZones(tE)
        rmNodes = ['Momentum' + v for v in 'XYZ'] + ['Density', 'TurbulentEnergyKineticDensity',
               'TurbulentDissipationRateDensity', 'ViscosityMolecular', 'Pressure', 'Mach', 'cellN',
               'EnergyStagnationDensity', 'q_criterion', 'Viscosity_EddyMolecularRatio', 'indicm']
        
        Zones_m = []
        reverse = ['VorticityX', 'VorticityZ', 'VelocityY', 'CenterY']
        for i, Zone in enumerate(Zones):
            FlowSolutionNode = I.getNodeFromName1(Zone, 'FlowSolution#Init')
            FlowSolutionNode[0] = 'FlowSolution#Centers'

            x, y, z = J.getxyz(C.node2Center(Zone))
            xc, yc, zc = J.invokeFields(Zone, ['CenterX', 'CenterY', 'CenterZ'],
              locationTag = 'centers:')
            xc[:], yc[:], zc[:] = x, y, z

            C._initVars(Zone, 'centers:Zone', i)
            C._initVars(Zone, 'centers:Index', 0)
            C._initVars(Zone, 'centers:VelocityX={MomentumX}/{Density}')
            C._initVars(Zone, 'centers:VelocityY={MomentumY}/{Density}')
            C._initVars(Zone, 'centers:VelocityZ={MomentumZ}/{Density}')
            Index = I.getNodeFromName(Zone, 'Index')
            cmpt = 0
            for l in range(len(Index[1][0][0])):
                for m in range(len(Index[1][0])):
                    for n in range(len(Index[1])):
                        Index[1][n][m][l] = cmpt
                        cmpt += 1

            for name in rmNodes: I._rmNodesByName(Zone, name)
            
            FlowSolutionNode = I.getNodeFromName1(Zone, 'FlowSolution#Centers')
            x, y, z = J.getxyz(Zone)
            Zone_m = J.createZone(Zone[0] + '_m', [x, -y, z], 'xyz')
            for FlowSolution in FlowSolutionNode[2][1:]:
                FlowSolution_m = J.invokeFields(Zone_m, [FlowSolution[0]],
                                                                        locationTag = 'centers:')[0]
                if FlowSolution[0] in reverse: FlowSolution_m[:] = -FlowSolution[1]
                else: FlowSolution_m[:] = FlowSolution[1]
                if FlowSolution[0] == 'Zone': FlowSolution_m[:] += len(Zones)
            Zones_m += [Zone_m]

        MeshRadius = -1.
        for Zone in Zones:
            TurbDist = I.getNodeFromName(Zone, 'TurbulentDistance')[1]
            MeshRadius = max(MeshRadius, np.max(TurbDist))

        NumberOfHybridInterfaces = HybridParameters['NumberOfHybridInterfaces'][0]
        OuterDomainToWallDistance = HybridParameters['OuterDomainToWallDistance'][0]
        Sigma = VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]
        InnerDomain = OuterDomainToWallDistance - NumberOfHybridInterfaces*Sigma
        if InnerDomain < Sigma:
            raise ValueError('The Hybrid Domain radius (NumberOfHybridInterfaces*Sigma = %.5f m) \
                is too close to the solid (InnerDomainToWallDistance = %.5f m < Sigma = %.5f) for \
                the selected OuterDomainToWallDistance = %.5f m. Either reduce the \
                NumberOfHybridInterfaces, the Resolution or the SmoothingRatio, or increase the \
                OuterDomainToWallDistance.'%(NumberOfHybridInterfaces*Sigma, InnerDomain, Sigma, \
                                                                         OuterDomainToWallDistance))
        if MeshRadius <= OuterDomainToWallDistance:
            raise ValueError('The Hybrid Domain ends beyond the mesh (OuterDomainToWallDistance = \
                %.5f m). The furthest cell is %.5f m from the wall.'%(OuterDomainToWallDistance, \
                                                                                        MeshRadius))

        if Zones_m: tE = C.newPyTree([Zones + Zones_m])
        else: tE = C.newPyTree([Zones])
        return I.correctPyTree(tE)

    def updateField(tE = [], VPMParameters = {}, HybridParameters = {}):
        if type(tE) == str: tE = I.getNodeFromName(open(tE), 'Sphere')
        Zones = I.getZones(tE)
        rmNodes = ['Momentum' + v for v in 'XYZ'] + ['Density', 'TurbulentEnergyKineticDensity',
               'TurbulentDissipationRateDensity', 'ViscosityMolecular', 'Pressure', 'Mach', 'cellN',
               'EnergyStagnationDensity', 'q_criterion', 'Viscosity_EddyMolecularRatio', 'indicm']
        
        reverse = ['VorticityX', 'VorticityZ', 'VelocityY', 'CenterY']
        for i, Zone in enumerate(Zones):
            FlowSolutionNode = I.getNodeFromName1(Zone, 'FlowSolution#Init')
            FlowSolutionNode[0] = 'FlowSolution#Centers'

            x, y, z = J.getxyz(C.node2Center(Zone))
            xc, yc, zc = J.invokeFields(Zone, ['CenterX', 'CenterY', 'CenterZ'],
              locationTag = 'centers:')
            xc[:], yc[:], zc[:] = x, y, z

            C._initVars(Zone, 'centers:Zone', i)
            C._initVars(Zone, 'centers:Index', 0)
            C._initVars(Zone, 'centers:VelocityX={MomentumX}/{Density}')
            C._initVars(Zone, 'centers:VelocityY={MomentumY}/{Density}')
            C._initVars(Zone, 'centers:VelocityZ={MomentumZ}/{Density}')
            Index = I.getNodeFromName(Zone, 'Index')
            cmpt = 0
            for l in range(len(Index[1][0][0])):
                for m in range(len(Index[1][0])):
                    for n in range(len(Index[1])):
                        Index[1][n][m][l] = cmpt
                        cmpt += 1

            for name in rmNodes: I._rmNodesByName(Zone, name)

        MeshRadius = -1.
        for zone in I.getZones(tE):
            MeshRadius = np.max(MeshRadius, np.max(J.getVars(zone, ['TurbulentDistance'])[0]))

        NumberOfHybridInterfaces = HybridParameters['NumberOfHybridInterfaces'][0]
        OuterDomainToWallDistance = HybridParameters['OuterDomainToWallDistance'][0]
        Sigma = VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]
        InnerDomain = OuterDomainToWallDistance - NumberOfHybridInterfaces*Sigma
        if InnerDomain < Sigma:
            raise ValueError('The Hybrid Domain radius (NumberOfHybridInterfaces*Sigma = %.5f m) \
                 is too close to the solid (InnerDomainToWallDistance = %.5f m < Sigma = %.5f) for \
                 the selected OuterDomainToWallDistance = %.5f m. Either reduce the \
                 NumberOfHybridInterfaces, the Resolution or the SmoothingRatio, or increase the \
                 OuterDomainToWallDistance.'%(NumberOfHybridInterfaces*Sigma, InnerDomain, Sigma, \
                                                                         OuterDomainToWallDistance))
        if MeshRadius <= OuterDomainToWallDistance:
            raise ValueError('The Hybrid Domain ends beyond the mesh (OuterDomainToWallDistance = \
                  %.5f m). The furthest cell is %.5f m from the wall.'%(OuterDomainToWallDistance, \
                                                                                        MeshRadius))

        return I.correctPyTree(C.newPyTree([Zones]))

    def getTurbulentDistances(tE = []):
        d = np.array([])
        for zone in I.getZones(tE):
            FlowSolutionNode = I.getNodeFromName1(zone, 'FlowSolution#Centers')
            d = np.append(d, I.getNodeFromName(FlowSolutionNode, 'TurbulentDistance')[1])
        return np.mean(d)      

    def generateHybridDomainInterfaces(tE = [], VPMParameters = {}, HybridParameters = {}):
        print('||'+'{:-^50}'.format(' Generate Hybrid Interfaces '))
        OuterDomainToWallDistance = HybridParameters['OuterDomainToWallDistance'][0]
        h = VPMParameters['Resolution'][0]
        InterfaceGap = h*VPMParameters['SmoothingRatio'][0]
        NumberOfHybridInterfaces = HybridParameters['NumberOfHybridInterfaces'][0]
        InnerDomainToWallDistance = OuterDomainToWallDistance - InterfaceGap*\
                                                                            NumberOfHybridInterfaces
        HybridParameters['InnerDomainToWallDistance'] = np.array([InnerDomainToWallDistance],
                                                                    order = 'F', dtype = np.float64)
        tE_joint = T.join(C.convertArray2Hexa(tE))
        remove = ['GridLocation', '.Solver#Param', '.MOLA#Offset', '.MOLA#Trim', 'FamilyName',
                             'AdditionalFamilyName', 'ZoneGridConnectivity', 'FlowSolution#Centers']
        Interfaces = []
        names = ['OuterInterface', 'InnerInterface', 'BEMInterface']
        for d in [OuterDomainToWallDistance, InnerDomainToWallDistance, h/100.]:
            Zones = I.getZones(P.isoSurfMC(tE_joint, 'TurbulentDistance', d))
            for Zone in Zones:
                for rm in remove: I._rmNodesByName(Zone, rm)

            Interface = T.join(C.convertArray2Hexa(C.newPyTree([Zones])))
            Interface[0] = names.pop(0)
            G._getNormalMap(Interface)
            Interface = C.center2Node(Interface, ['centers:sx', 'centers:sy', 'centers:sz'])
            I._rmNodesByName(Interface, 'FlowSolution#Centers')
            Interfaces += [Interface]

        msg =  '||' + '{:27}'.format('Outer Interface distance') + ': ' + \
                                                   '{:.4f}'.format(OuterDomainToWallDistance) + '\n'
        msg += '||' + '{:27}'.format('Inner Interface distance')     + ': ' + \
                                                   '{:.4f}'.format(InnerDomainToWallDistance) + '\n'
        msg += '||' + '{:-^50}'.format(' Done ')
        print(msg)
        return Interfaces

    def getRegularGridInHybridDomain(OuterInterface = [], InnerInterface = [], Resolution = 0.):
        bbox = np.array(G.bbox(OuterInterface))
        Ni = int(np.ceil((bbox[3] - bbox[0])/Resolution)) + 4
        Nj = int(np.ceil((bbox[4] - bbox[1])/Resolution)) + 4
        Nk = int(np.ceil((bbox[5] - bbox[2])/Resolution)) + 4
        cart = G.cart(np.ceil(bbox[:3]/Resolution)*Resolution - 2*Resolution, (Resolution,
                                                              Resolution, Resolution), (Ni, Nj, Nk))
        t_cart = C.newPyTree(['Base', cart])
        maskInnerSurface = CX.blankCells(t_cart, [[InnerInterface]], np.array([[1]]),
                            blankingType = 'center_in', delta = 0, dim = 3, tol = 0.)
        maskOuterSurface = CX.blankCells(t_cart, [[OuterInterface]], np.array([[1]]),
                            blankingType = 'center_in', delta = Resolution, dim = 3, tol = 0.)
        maskInnerSurface  = C.node2Center(I.getZones(maskInnerSurface)[0])
        maskOuterSurface = C.node2Center(I.getZones(maskOuterSurface)[0])
        cellInnerSurface = J.getVars(maskInnerSurface, ['cellN'], 'FlowSolution')[0]
        cellOuterSurface = J.getVars(maskOuterSurface, ['cellN'], 'FlowSolution')[0]
        inside = (cellOuterSurface == 0)*(cellInnerSurface == 1)
        x, y, z = J.getxyz(maskOuterSurface)
        return C.convertArray2Node(J.createZone('Grid', [x[inside], y[inside], z[inside]], 'xyz'))

    def findDonorsIndex(Mesh = [], Target = [], returnUniqueList = False):
        x, y, z = J.getxyz(Target)
        t_centers = C.node2Center(Mesh)
        zones_centers = I.getZones(t_centers)
        hook, indir = C.createGlobalHook(t_centers, function = 'nodes', indir = 1)
        nodes, dist = C.nearestNodes(hook, J.createZone('Zone', [x, y, z], 'xyz'))
        nodes, unique = np.unique(nodes, return_index = True)
        x = x[unique]

        cumul = 0
        cumulated = []
        for z in zones_centers:
            cumulated += [cumul]
            cumul += C.getNPts(z)

        receivers = []
        donors = []
        for p in range(len(x)):
            ind = nodes[p] - 1
            closest_index = ind - cumulated[indir[ind]]
            receivers += [indir[ind]]
            donors += [closest_index]

        donors = np.array(donors, order = 'F', dtype = np.int32)
        receivers = np.array(receivers, order = 'F', dtype = np.int32)
        if returnUniqueList: return donors, receivers, unique
        
        return donors, receivers
  
    def findDonorFields(Mesh = [], donors = [], receivers = [], FieldNames = []):
        Nh = len(donors)
        Fields = {}
        for name in FieldNames:
            Fields[name] = np.zeros(Nh, dtype = np.float64, order = 'F')

        for iz, zone in enumerate(I.getZones(Mesh)):
            receiver_slice = receivers == iz
            DonorFields = J.getVars(zone, FieldNames, 'FlowSolution#Centers')

            donor_slice = donors[receiver_slice]
            for n, name in enumerate(FieldNames):
                Fields[name][receiver_slice] = np.ravel(DonorFields[n], order = 'F')[donor_slice]

        return Fields

    def setHybridDonors(Mesh = [], Interfaces = [], VPMParameters = {}, HybridParameters = {}):
        print('||'+'{:-^50}'.format(' Generate Hybrid Sources '))
        Sigma0 = VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]
        Grid = getRegularGridInHybridDomain(Interfaces[0], Interfaces[-1], Sigma0)
        donors, receivers = findDonorsIndex(Mesh, Grid)
        Fields = findDonorFields(Mesh, donors, receivers, ['Center' + v for v in 'XYZ'] + \
                                                                              ['TurbulentDistance'])
        OuterDomainToWallDistance = HybridParameters['OuterDomainToWallDistance'][0]
        NumberOfHybridInterfaces = HybridParameters['NumberOfHybridInterfaces'][0]
        InnerDomainToWallDistance = OuterDomainToWallDistance - Sigma0*NumberOfHybridInterfaces
        flags = []
        for n in range(NumberOfHybridInterfaces + 1):
            d = OuterDomainToWallDistance - n*Sigma0
            flags += [Fields['TurbulentDistance'] <= d]#flags everything inside current interface

        Domain_Flag = flags[0]*np.logical_not(flags[-1])#Particles inside OuterInterface and outside InnerInterface
        Resolution = findMinimumDistanceBetweenParticles(Fields['CenterX'][Domain_Flag], Fields['CenterY'][Domain_Flag], Fields['CenterZ'][Domain_Flag])
        while np.min(Resolution) < Sigma0/2.:
            CurrentFlag = Domain_Flag[Domain_Flag]
            hmin = np.unique(Resolution[Resolution < Sigma0/2.])#get one sample of each h that is too small
            for i in range(len(hmin)):
                CurrentFlag[np.min(np.where(Resolution == hmin[i]))] = False#get rid of one particle for each h that is too small. that is because if one h is too small, then at least one other has the same value. when the 1st h is taken out, the others will increase.

            Domain_Flag[Domain_Flag] = CurrentFlag
            Resolution = findMinimumDistanceBetweenParticles(Fields['CenterX'][Domain_Flag], Fields['CenterY'][Domain_Flag], Fields['CenterZ'][Domain_Flag])

        InterfacesFlags = {}
        for i in range(len(flags) - 1):
            InterfacesFlags['Interface_' + str(i)] = np.array(flags[i][Domain_Flag]*\
                           np.logical_not(flags[i + 1][Domain_Flag]), dtype = np.int32, order = 'F')#seperates the particles for each interface

        HybridParameters['NumberOfHybridSources'] = np.array([0], order = 'F', dtype = np.int32)
        HybridParameters['HybridDonors']          = np.array(donors[Domain_Flag],    order = 'F',
                                                                                   dtype = np.int32)
        HybridParameters['HybridReceivers']       = np.array(receivers[Domain_Flag], order = 'F',
                                                                                   dtype = np.int32)
        HybridParameters['ParticleSeparationPerInterface'] = InterfacesFlags
        HybridParameters['HybridSigma'] = np.array(Resolution, order = 'F', dtype = np.float64)

        msg = '||' + '{:27}'.format('Number of Hybrid sources') + ': '+ '{:d}'.format(len(Resolution))   + '\n'
        msg += '||' + '{:27}'.format('Targeted Particle spacing') + ': '+'{:.4f}'.format(\
                         VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]) + ' m\n'
        msg += '||' + '{:27}'.format('Mean Particle spacing')     + ': ' + \
                                                       '{:.4f}'.format(np.mean(Resolution)) + ' m\n'
        msg += '||' +'{:27}'.format('Particle spacing deviation') + ': ' + \
                                                        '{:.4f}'.format(np.std(Resolution)) + ' m\n'
        msg += '||' + '{:27}'.format('Maximum Particle spacing')  + ': ' + \
                                                        '{:.4f}'.format(np.max(Resolution)) + ' m\n'
        msg += '||' + '{:27}'.format('Minimum Particle spacing')  + ': ' + \
                                                        '{:.4f}'.format(np.min(Resolution)) + ' m\n'
        msg += '||' + '{:-^50}'.format(' Done ')
        print(msg)

    def setBEMDonors(Mesh = [], Interface = [], VPMParameters = {}, HybridParameters = {}):
        print('||'+'{:-^50}'.format(' Generate BEM Panels '))
        Sigma0 = VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]
        donors, receivers, unique = findDonorsIndex(Mesh, Interface, returnUniqueList = True)
        Fields = findDonorFields(Mesh, donors, receivers, ['Center' + v for v in 'XYZ'])
        sx, sy, sz = J.getVars(Interface, ['s' + v for v in 'xyz'])
        sx = sx[unique]
        sy = sy[unique]
        sz = sz[unique]
        Zone = C.convertArray2Node(J.createZone('Zone', [Fields['CenterX'], Fields['CenterY'], \
                                                                         Fields['CenterZ']], 'xyz'))
        surf = vpm_cpp.find_panel_clusters(Zone, VPMParameters['Resolution'][0])
        flag = surf != 0.

        Resolution = findMinimumDistanceBetweenParticles(Fields['CenterX'][flag], \
                                                   Fields['CenterY'][flag], Fields['CenterZ'][flag])
        while np.min(Resolution) < Sigma0/2.:
            CurrentFlag = flag[flag]
            hmin = np.unique(Resolution[Resolution < Sigma0/2.])#get one sample of each h that is too small
            for i in range(len(hmin)):
                CurrentFlag[np.min(np.where(Resolution == hmin[i]))] = False#get rid of one particle for each h that is too small. that is because if one h is too small, then at least one other has the same value. when the 1st h is taken out, the others will increase.
            
            flag[flag] = CurrentFlag
            Resolution = findMinimumDistanceBetweenParticles(Fields['CenterX'][flag], \
                                                   Fields['CenterY'][flag], Fields['CenterZ'][flag])

        sx = sx[flag]
        sy = sy[flag]
        sz = sz[flag]
        s = np.linalg.norm(np.vstack([sx, sy, sz]), axis=0)
        nx = -sx/s
        ny = -sy/s
        nz = -sz/s
        #t1 = ez vec n 
        t1x = -ny
        t1y = nx
        t1z = 0.*nz
        t1 = np.linalg.norm(np.vstack([t1x, t1y, t1z]), axis=0)
        #t2 = n vec t1
        t2x = ny*t1z - nz*t1y
        t2y = nz*t1x - nx*t1z
        t2z = nx*t1y - ny*t1x
        t2 = np.linalg.norm(np.vstack([t2x, t2y, t2z]), axis=0)

        HybridParameters['NormalBEMX'] = np.array(nx, dtype = np.float64, order = 'F')
        HybridParameters['NormalBEMY'] = np.array(ny, dtype = np.float64, order = 'F')
        HybridParameters['NormalBEMZ'] = np.array(nz, dtype = np.float64, order = 'F')
        HybridParameters['Tangential1BEMX'] = np.array(t1x/t1, dtype = np.float64, order = 'F')
        HybridParameters['Tangential1BEMY'] = np.array(t1y/t1, dtype = np.float64, order = 'F')
        HybridParameters['Tangential1BEMZ'] = np.array(t1z/t1, dtype = np.float64, order = 'F')
        HybridParameters['Tangential2BEMX'] = np.array(t2x/t2, dtype = np.float64, order = 'F')
        HybridParameters['Tangential2BEMY'] = np.array(t2y/t2, dtype = np.float64, order = 'F')
        HybridParameters['Tangential2BEMZ'] = np.array(t2z/t2, dtype = np.float64, order = 'F')
        HybridParameters['BEMDonors'] = np.array(donors[flag], order = 'F', dtype = np.int32)
        HybridParameters['BEMReceivers'] = np.array(receivers[flag], order = 'F', dtype = np.int32)
        HybridParameters['NumberOfBEMSources'] = np.array([len(nx)], dtype = np.int32, order = 'F')
        HybridParameters['SurfaceBEM'] = np.array(Resolution**2, order = 'F', dtype = np.float64)
        msg =  '||' + '{:27}'.format('Number of BEM panels')     + ': '+ '{:d}'.format(len(nx))+'\n'
        msg += '||' + '{:27}'.format('Targeted Particle spacing') + ': '+'{:.4f}'.format(\
                         VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]) + ' m\n'
        msg += '||' + '{:27}'.format('Mean Particle spacing')     + ': ' + \
                                                       '{:.4f}'.format(np.mean(Resolution)) + ' m\n'
        msg += '||' +'{:27}'.format('Particle spacing deviation') + ': ' + \
                                                        '{:.4f}'.format(np.std(Resolution)) + ' m\n'
        msg += '||' + '{:27}'.format('Maximum Particle spacing')  + ': ' + \
                                                        '{:.4f}'.format(np.max(Resolution)) + ' m\n'
        msg += '||' + '{:27}'.format('Minimum Particle spacing')  + ': ' + \
                                                        '{:.4f}'.format(np.min(Resolution)) + ' m\n'
        msg += '||' + '{:-^50}'.format(' Done ')
        print(msg)

    def setCFDDonors(Mesh = [], Interface = [], VPMParameters = {}, HybridParameters = {}):
        print('||'+'{:-^50}'.format(' Generate Eulerian Panels '))
        Sigma0 = VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]
        donors, receivers, unique = findDonorsIndex(Mesh, Interface, returnUniqueList = True)
        Fields = findDonorFields(Mesh, donors, receivers, ['Center' + v for v in 'XYZ'])
        sx, sy, sz = J.getVars(Interface, ['s' + v for v in 'xyz'])
        sx = sx[unique]
        sy = sy[unique]
        sz = sz[unique]
        Zone = C.convertArray2Node(J.createZone('Zone', [Fields['CenterX'], Fields['CenterY'], \
                                                                         Fields['CenterZ']], 'xyz'))
        surf = vpm_cpp.find_panel_clusters(Zone, VPMParameters['Resolution'][0])
        flag = surf != 0.

        Resolution = findMinimumDistanceBetweenParticles(Fields['CenterX'][flag], Fields['CenterY'][flag], \
                                                                            Fields['CenterZ'][flag])
        while np.min(Resolution) < Sigma0/2.:
            CurrentFlag = flag[flag]
            hmin = np.unique(Resolution[Resolution < Sigma0/2.])#get one sample of each h that is too small
            for i in range(len(hmin)):
                CurrentFlag[np.min(np.where(Resolution == hmin[i]))] = False#get rid of one particle for each h that is too small. that is because if one h is too small, then at least one other has the same value. when the 1st h is taken out, the others will increase.
            
            flag[flag] = CurrentFlag
            Resolution = findMinimumDistanceBetweenParticles(Fields['CenterX'][flag], \
                                                   Fields['CenterY'][flag], Fields['CenterZ'][flag])

        sx = -sx[flag]
        sy = -sy[flag]
        sz = -sz[flag]

        surf = Resolution**2/np.linalg.norm(np.vstack([sx, sy, sz]), axis = 0)
        HybridParameters['SurfaceCFDX'] = np.array(sx*surf, dtype = np.float64, order = 'F')
        HybridParameters['SurfaceCFDY'] = np.array(sy*surf, dtype = np.float64, order = 'F')
        HybridParameters['SurfaceCFDZ'] = np.array(sz*surf, dtype = np.float64, order = 'F')
        HybridParameters['CFDDonors'] = np.array(donors[flag], order = 'F', dtype = np.int32)
        HybridParameters['CFDReceivers'] = np.array(receivers[flag], order = 'F', dtype = np.int32)
        HybridParameters['NumberOfCFDSources'] = np.array([len(sx)], dtype = np.int32, order = 'F')
        msg  = '||' + '{:27}'.format('Number of CFD panels') + ': '+ '{:d}'.format(len(sx)) + '\n'
        msg += '||' + '{:27}'.format('Targeted Particle spacing') + ': '+'{:.4f}'.format(\
                         VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]) + ' m\n'
        msg += '||' + '{:27}'.format('Mean Particle spacing')     + ': ' + \
                                                       '{:.4f}'.format(np.mean(Resolution)) + ' m\n'
        msg += '||' +'{:27}'.format('Particle spacing deviation') + ': ' + \
                                                        '{:.4f}'.format(np.std(Resolution)) + ' m\n'
        msg += '||' + '{:27}'.format('Maximum Particle spacing')  + ': ' + \
                                                        '{:.4f}'.format(np.max(Resolution)) + ' m\n'
        msg += '||' + '{:27}'.format('Minimum Particle spacing')  + ': ' + \
                                                        '{:.4f}'.format(np.min(Resolution)) + ' m\n'
        msg += '||' + '{:-^50}'.format(' Done ')
        print(msg)

    def generateHybridDomain(tE = [], VPMParameters = {}, HybridParameters = {}):
        Interfaces = generateHybridDomainInterfaces(tE, VPMParameters, HybridParameters)
        setBEMDonors(tE, Interfaces[-1], VPMParameters, HybridParameters)
        setCFDDonors(tE, Interfaces[-2], VPMParameters, HybridParameters)
        setHybridDonors(tE, Interfaces, VPMParameters, HybridParameters)
        HybridParameters['BEMMatrix'] = np.array([0.]*9*\
                                             HybridParameters['NumberOfBEMSources'][0]**2,
                                                                    dtype = np.float64, order = 'F')
        return C.newPyTree(['HybridDomain', Interfaces])

    def pickHybridDomain(t = []):
        return I.getNodeFromName1(t, 'HybridDomain')

    def pickHybridDomainOuterInterface(t = []):
        HybridDomain = pickHybridDomain(t)
        if HybridDomain: return I.getZones(HybridDomain)[0]

        return []

    def pickHybridDomainInnerInterface(t = []):
        HybridDomain = pickHybridDomain(t)
        if HybridDomain: return I.getZones(HybridDomain)[-2]

        return []

    def pickBEMInterface(t = []):
        HybridDomain = pickHybridDomain(t)
        if HybridDomain: return I.getZones(HybridDomain)[-1]

        return []

    def getHybridParameters(t = []): return J.get(pickParticlesZone(t), '.Hybrid#Parameters')

    def findMinHybridParticleVorticityToShed(ω = [], VorticityPercentageToShed = 0.):
        ω = np.sort(ω)
        ωsum = [ω[-1]]
        for i in range(len(ω) - 2, -1, -1): ωsum += [ω[i] + ωsum[-1]]

        Nshed = np.argmin(np.abs(np.array(ωsum) - ωsum[-1]*VorticityPercentageToShed/100.))#keep the Nshed strongest
        return ω[-min(Nshed + 2, len(ω))]

    def initialiseHybridParticles(tL = [], tE = [], VPMParameters = {}, HybridParameters = {},
        Offset = 0):
        Ramp = np.sin(min(1./VPMParameters['StrengthRampAtbeginning'][0], 1.)*np.pi/2.)
        #First BEM particles
        Fields = findDonorFields(tE, HybridParameters['BEMDonors'],
                              HybridParameters['BEMReceivers'], ['Center' + v for v in 'XYZ'] + \
                                                                    ['Velocity' + v for v in 'XYZ'])
        s  = HybridParameters['SurfaceBEM']
        nx = HybridParameters['NormalBEMX']
        ny = HybridParameters['NormalBEMY']
        nz = HybridParameters['NormalBEMZ']
        tx = HybridParameters['Tangential1BEMX'] + HybridParameters['Tangential2BEMX']
        ty = HybridParameters['Tangential1BEMY'] + HybridParameters['Tangential2BEMY']
        tz = HybridParameters['Tangential1BEMZ'] + HybridParameters['Tangential2BEMZ']        
        at = (Fields['VelocityX']*tx + Fields['VelocityY']*ty + Fields['VelocityZ']*tz)*s*Ramp
        addParticlesToTree(tL, Fields['CenterX'], Fields['CenterY'], Fields['CenterZ'], at*tx, \
                                                                   at*ty, at*tz, np.sqrt(s), Offset)
        HybridParameters['AlphaBEMN'] = (nx*Fields['VelocityX'] + ny*Fields['VelocityY'] + \
                                                                   nz*Fields['VelocityZ'])*s*Ramp
        Nbem = len(Fields['CenterX'])
        #Second CFD particles
        Fields = findDonorFields(tE, HybridParameters['CFDDonors'],
                              HybridParameters['CFDReceivers'], ['Center' + v for v in 'XYZ'] + \
                                                                    ['Velocity' + v for v in 'XYZ'])
        sx = HybridParameters['SurfaceCFDX']
        sy = HybridParameters['SurfaceCFDY']
        sz = HybridParameters['SurfaceCFDZ']
        s  = np.linalg.norm(np.vstack([sx, sy, sz]), axis = 0)
        addParticlesToTree(tL, Fields['CenterX'], Fields['CenterY'], Fields['CenterZ'], \
                  (Fields['VelocityY']*sz - Fields['VelocityZ']*sy)*Ramp, \
                  (Fields['VelocityZ']*sx - Fields['VelocityX']*sz)*Ramp, \
                  (Fields['VelocityX']*sy - Fields['VelocityY']*sx)*Ramp, np.sqrt(s), Offset + Nbem)
        HybridParameters['AlphaCFDN'] = (Fields['VelocityX']*sx + Fields['VelocityY']*sy + Fields['VelocityZ']*sz)*Ramp
        Ncfd = len(Fields['CenterX'])
        #Last Hybrid particles
        Fields = findDonorFields(tE, HybridParameters['HybridDonors'],
                              HybridParameters['HybridReceivers'], ['Center' + v for v in 'XYZ'] + \
                                                                   ['Vorticity' + v for v in 'XYZ'])
        wp = np.linalg.norm(np.vstack([Fields['VorticityX'], Fields['VorticityY'], \
                                                                     Fields['VorticityZ']]), axis=0)
        wtot = np.sum(wp)#total particle vorticity in the Hybrid Domain
        ParticleSeparationPerInterface = HybridParameters['ParticleSeparationPerInterface']
        VorticityFactor = HybridParameters['FactorOfCFDVorticityShed'][0]
        hybrid = np.array([False]*len(ParticleSeparationPerInterface['Interface_0']))
        if 0. < VorticityFactor:
            for i, InterfaceFlag in enumerate(ParticleSeparationPerInterface):
                flag = (ParticleSeparationPerInterface[InterfaceFlag] == 1)
                wi = wp[flag]
                wmin = findMinHybridParticleVorticityToShed(wi, VorticityFactor)
                flag[flag] = wmin < wi#only the strongest will survive
                hybrid += np.array(flag)
        else:
            Ni = HybridParameters['NumberOfParticlesPerInterface'][0]
            for i, InterfaceFlag in enumerate(ParticleSeparationPerInterface):
                flag = (ParticleSeparationPerInterface[InterfaceFlag] == 1)
                wi = wp[flag]
                Ncurrent = min(max(Ni, 1), len(wi))
                wmin = np.sort(wi)[-Ncurrent]#vorticity cutoff
                flagi = wmin < wi
                Nflagi = np.sum(flagi)
                if Ncurrent != Nflagi:#if several cells have the same vort and it happens to be wmin, Ncurrent will be below Ni, so i need to add somme cells until Ncurrent = Ni
                    allwmin = np.where(wmin == wi)[0]
                    flagi[allwmin[: Ncurrent - Nflagi]] = True

                flag[flag] = flagi#only the strongest will survive
                hybrid += np.array(flag)

        Nh = np.sum(hybrid)
        sigma = HybridParameters['HybridSigma']
        s3 = sigma**3
        apx = Fields['VorticityX']*s3*Ramp
        apy = Fields['VorticityY']*s3*Ramp
        apz = Fields['VorticityZ']*s3*Ramp
        addParticlesToTree(tL, Fields['CenterX'][hybrid], Fields['CenterY'][hybrid], \
                                                Fields['CenterZ'][hybrid], apx[hybrid], apy[hybrid],
                                                   apz[hybrid], sigma[hybrid], Offset + Nbem + Ncfd)
        HybridParameters['AlphaHybridX'] = apx
        HybridParameters['AlphaHybridY'] = apy
        HybridParameters['AlphaHybridZ'] = apz
        HybridParameters['NumberOfHybridSources'][0] = Nh

        HybridFlag, s = J.getVars(pickParticlesZone(tL), ['HybridFlag', 'Sigma'])
        HybridFlag[:Offset]                           = 0
        HybridFlag[Offset: Offset + Nbem + Ncfd + Nh] = 1
        HybridFlag[Offset + Nbem + Ncfd + Nh:]        = 0

    def eraseParticlesInHybridDomain(t = []):
        Particles = pickParticlesZone(t)
        Nll, Nbem, Ncfd = getParameters(Particles, ['NumberOfLiftingLineSources',
                                                        'NumberOfBEMSources', 'NumberOfCFDSources'])
        if not Nll : Nll  = np.array([0])
        Offset = Nll[0] + Nbem[0] + Ncfd[0]
        OuterInterface = pickHybridDomainOuterInterface(t)
        box = [np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf]
        for BC in I.getZones(OuterInterface):
            x, y, z = J.getxyz(BC)
            box = [min(box[0], np.min(x)), min(box[1], np.min(y)), min(box[2], np.min(z)),
                             max(box[3], np.max(x)), max(box[4], np.max(y)), max(box[5], np.max(z))]

        x, y, z = J.getxyz(Particles)
        inside = (box[0] < x)*(box[1] < y)*(box[2] < z)*(x < box[3])*(y < box[4])*(z < box[5])#does a first cleansing to avoid checking far away particles
        x, y, z = x[inside], y[inside], z[inside]
        mask = C.convertArray2Node(J.createZone('Zone', [x, y, z], 'xyz'))
        mask = I.getZones(CX.blankCells(C.newPyTree(['Base', mask]), [[OuterInterface]],
                      np.array([[1]]),  blankingType = 'node_in', delta = 0., dim = 3, tol = 0.))[0]
        cellN = J.getVars(mask, ['cellN'], 'FlowSolution')[0]
        inside[inside] = (cellN == 0)
        inside[:Offset] = False
        delete(t, inside)
        return np.sum(inside)

    def splitHybridParticles(t = []):
        Particles = pickParticlesZone(t)
        splitParticles = vpm_cpp.split_hybrid_particles(t)
        Nll, Nbem, Ncfd, Nh = getParameters(Particles, ['NumberOfLiftingLineSources',
                               'NumberOfBEMSources', 'NumberOfCFDSources', 'NumberOfHybridSources'])
        if not Nll : Nll  = np.array([0])
        Offset = Nll[0] + Nbem[0] + Ncfd[0]

        if (splitParticles is not None):
            Nsplit = len(splitParticles[0])
            adjustTreeSize(t, NewSize = Nsplit, OldSize =  Nh, AtTheEnd = False, Offset = Offset)
            X, Y, Z = J.getxyz(Particles)
            AX, AY, AZ, A, Vol, S, HybridFlag = J.getVars(Particles, ['Alpha' + v for v in 'XYZ'] +\
                                             ['StrengthMagnitude', 'Volume', 'Sigma', 'HybridFlag'])
            HybridFlag[Offset: Nsplit + Offset] = 1
            X[Offset: Offset + Nsplit]          = splitParticles[0][:]
            Y[Offset: Offset + Nsplit]          = splitParticles[1][:]
            Z[Offset: Offset + Nsplit]          = splitParticles[2][:]
            AX[Offset: Offset + Nsplit]         = splitParticles[3][:]
            AY[Offset: Offset + Nsplit]         = splitParticles[4][:]
            AZ[Offset: Offset + Nsplit]         = splitParticles[5][:]
            A[Offset: Offset + Nsplit]          = splitParticles[6][:]
            Vol[Offset: Offset + Nsplit]        = splitParticles[7][:]
            S[Offset: Offset + Nsplit]          = splitParticles[8][:]
        else :Nsplit = Nh#case where Resolution = Sigma

        return Nsplit

    def updateBEMMatrix(t = []):
        HybridParameters = getHybridParameters(t)
        HybridParameters['BEMMatrix'][:] = vpm_cpp.inverse_bem_matrix(t)

    def updateBEMSources(tL = []): return vpm_cpp.update_bem_strength(tL)

    def updateCFDSources(tL = [], tE = []):
        Particles = pickParticlesZone(tL)
        HybridParameters = getHybridParameters(Particles)
        it, Ramp = getParameters(tL, ['CurrentIteration', 'StrengthRampAtbeginning'])
        Ramp = np.sin(min(it[0]/Ramp[0], 1.)*np.pi/2.)
        Nll, Nbem, Ncfd, U0 = getParameters(Particles, ['NumberOfLiftingLineSources',
                                  'NumberOfBEMSources', 'NumberOfCFDSources', 'VelocityFreestream'])
        if not Nll : Nll  = np.array([0])
        Offset = Nll[0] + Nbem[0]

        Fields = findDonorFields(tE, HybridParameters['CFDDonors'],
                                  HybridParameters['CFDReceivers'], ['Velocity' + v for v in 'XYZ'])
        Fields['VelocityX'] -= U0[0]
        Fields['VelocityY'] -= U0[1]
        Fields['VelocityZ'] -= U0[2]
        SX = HybridParameters['SurfaceCFDX']
        SY = HybridParameters['SurfaceCFDY']
        SZ = HybridParameters['SurfaceCFDZ']
        AX, AY, AZ, A = J.getVars(Particles, ['Alpha' + v for v in 'XYZ'] + ['StrengthMagnitude'])
        Ncfd = Offset + Ncfd[0]
        AX[Offset: Ncfd] = (SY*Fields['VelocityZ'] - SZ*Fields['VelocityY'])*Ramp
        AY[Offset: Ncfd] = (SZ*Fields['VelocityX'] - SX*Fields['VelocityZ'])*Ramp
        AZ[Offset: Ncfd] = (SX*Fields['VelocityY'] - SY*Fields['VelocityX'])*Ramp
        A[Offset: Ncfd]  = np.linalg.norm(np.vstack([AX[Offset: Ncfd], AY[Offset: Ncfd],
                                                                         AZ[Offset: Ncfd]]), axis=0)
        HybridParameters['AlphaCFDN'][:] = Ramp*(SX*Fields['VelocityX'] + SY*Fields['VelocityY'] + \
                                                                              SZ*Fields['VelocityZ'])

    def updateHybridSources(tL = [], tE = [], IterationInfo = {}):
        Particles = pickParticlesZone(tL)
        HybridParameters = getHybridParameters(Particles)
        VPMParameters = getVPMParameters(Particles)
        Ramp = np.sin(min(VPMParameters['CurrentIteration'][0]/VPMParameters['StrengthRampAtbeginning'][0], 1.)*np.pi/2.)
        Nll, Nbem, Ncfd = getParameters(Particles, ['NumberOfLiftingLineSources',
                                                        'NumberOfBEMSources', 'NumberOfCFDSources'])
        if not Nll : Nll  = np.array([0])
        Nbem = Nbem[0]
        Ncfd = Ncfd[0]
        Offset = Nll[0]
        Fields = findDonorFields(tE, HybridParameters['HybridDonors'],
                              HybridParameters['HybridReceivers'], ['Center' + v for v in 'XYZ'] + \
                                                                   ['Vorticity' + v for v in 'XYZ'])
        wp = np.linalg.norm(np.vstack([Fields['VorticityX'], Fields['VorticityY'], Fields['VorticityZ']]), axis=0)
        wtot = np.sum(wp)#total particle vorticity in the Hybrid Domain

        ParticleSeparationPerInterface = HybridParameters['ParticleSeparationPerInterface']
        VorticityFactor = HybridParameters['FactorOfCFDVorticityShed'][0]
        hybrid = np.array([False]*len(ParticleSeparationPerInterface['Interface_0']))
        if 0. < VorticityFactor:
            for i, InterfaceFlag in enumerate(ParticleSeparationPerInterface):
                flag = (ParticleSeparationPerInterface[InterfaceFlag] == 1)
                wi = wp[flag]
                wmin = findMinHybridParticleVorticityToShed(wi, VorticityFactor)
                flag[flag] = wmin < wi#only the strongest will survive
                hybrid += np.array(flag)
        else:
            Ni = HybridParameters['NumberOfParticlesPerInterface'][0]
            for i, InterfaceFlag in enumerate(ParticleSeparationPerInterface):
                flag = (ParticleSeparationPerInterface[InterfaceFlag] == 1)
                wi = wp[flag]
                Ncurrent = min(max(Ni, 1), len(wi))
                wmin = np.sort(wi)[-Ncurrent]#vorticity cutoff
                flagi = wmin < wi
                Nflagi = np.sum(flagi)
                if Ncurrent != Nflagi:#if several cells have the same vort and it happens to be wmin, Ncurrent will be below Ni, so i need to add somme cells until Ncurrent = Ni
                    allwmin = np.where(wmin == wi)[0]
                    flagi[allwmin[: Ncurrent - Nflagi]] = True

                flag[flag] = flagi#only the strongest will survive
                hybrid += np.array(flag)

        Nh = np.sum(hybrid)
        sigma = VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]
        apx = HybridParameters['AlphaHybridX']
        apy = HybridParameters['AlphaHybridY']
        apz = HybridParameters['AlphaHybridZ']
        IterationInfo['Number of Hybrids Generated'] = -eraseParticlesInHybridDomain(tL)

        addParticlesToTree(tL, Fields['CenterX'][hybrid], Fields['CenterY'][hybrid], Fields['CenterZ'][hybrid], apx[hybrid], apy[hybrid],
                         apz[hybrid], HybridParameters['HybridSigma'][hybrid], Offset + Nbem + Ncfd)

        HybridFlag = J.getVars(Particles, ['HybridFlag'])[0]
        HybridFlag[:Offset]                           = 0
        HybridFlag[Offset: Offset + Nbem + Ncfd + Nh] = 1
        HybridFlag[Offset + Nbem + Ncfd + Nh:]        = 0

        solveParticleStrength(tL, Fields['VorticityX'][hybrid]*Ramp, Fields['VorticityY'][hybrid]*Ramp, Fields['VorticityZ'][hybrid]*Ramp, Offset + Nbem + Ncfd)
        ax, ay, az = J.getVars(Particles, ['Alpha' + v for v in 'XYZ'])
        Offset += Nbem + Ncfd
        HybridParameters['AlphaHybridX'][hybrid] = ax[Offset: Offset + Nh]
        HybridParameters['AlphaHybridY'][hybrid] = ay[Offset: Offset + Nh]
        HybridParameters['AlphaHybridZ'][hybrid] = az[Offset: Offset + Nh]

        HybridParameters['NumberOfHybridSources'][0] = Nh
        HybridParameters['NumberOfHybridSources'][0] = splitHybridParticles(tL)
        IterationInfo['Number of Hybrids Generated'] += HybridParameters['NumberOfHybridSources'][0]

        wtot = np.sum(wp)
        wp = wp[hybrid]
        wkept = np.sum(wp)#total particle voticity kept
        IterationInfo['Minimum Eulerian Vorticity'] = np.min(wp)
        IterationInfo['Eulerian Vorticity lost'] = wtot - wkept
        IterationInfo['Eulerian Vorticity lost per'] = (wtot - wkept)/wtot*100
        return IterationInfo

    def updateHybridDomainAndSources(tL = [], tE = [], IterationInfo = {}):
        if not tE: return IterationInfo
        IterationInfo['Hybrid Computation time'] = J.tic()

        updateCFDSources(tL, tE)
        updateBEMSources(tL)
        IterationInfo = updateHybridSources(tL, tE, IterationInfo)

        IterationInfo['Hybrid Computation time'] = J.tic() -IterationInfo['Hybrid Computation time']
        return IterationInfo
    
####################################################################################################
####################################################################################################
########################################### Lifting Lines ##########################################
####################################################################################################
####################################################################################################
    def setTimeStepFromShedParticles(t = [], LiftingLines = [], NumberParticlesShedAtTip = 5.):
        if not LiftingLines: raise AttributeError('The time step is not given and can not be \
                     computed without a Lifting Line. Specify the time step or give a Lifting Line')
        LL.computeKinematicVelocity(LiftingLines)
        LL.assembleAndProjectVelocities(LiftingLines)

        if type(t) == dict:
            Resolution = t['Resolution']
            U0         = t['VelocityFreestream']
        else:
            Particles  = pickParticlesZone(t)
            Resolution = I.getNodeFromName(Particles, 'Resolution')[1][0]
            U0         = I.getValue(I.getNodeFromName(Particles, 'VelocityFreestream'))

        Urelmax = 0.
        for LiftingLine in LiftingLines:
            Ukin = np.vstack(J.getVars(LiftingLine, ['VelocityKinematic' + i for i in 'XYZ']))
            ui   = np.vstack(J.getVars(LiftingLine, ['VelocityInduced' + i for i in 'XYZ']))
            Urel = U0 + ui.T - Ukin.T
            Urel = max([np.linalg.norm(urel) for urel in Urel])
            if (Urelmax < Urel): Urelmax = Urel

        if Urelmax == 0:
            raise ValueError('Maximum velocity is zero. Set non-zero kinematic or freestream \
                                                                                         velocity.')

        if type(t) == dict:
            t['TimeStep'] = NumberParticlesShedAtTip*Resolution/Urel
        else:
            VPMParameters = getVPMParameters(t)
            VPMParameters['TimeStep'] = NumberParticlesShedAtTip*Resolution/Urel

    def setTimeStepFromBladeRotationAngle(t = [], LiftingLines = [], BladeRotationAngle = 5.):
        if not LiftingLines: raise AttributeError('The time step is not given and can not be \
                     computed without a Lifting Line. Specify the time step or give a Lifting Line')

        RPM = 0.
        for LiftingLine in LiftingLines:
            RPM = max(RPM, I.getValue(I.getNodeFromName(LiftingLine, 'RPM')))
        
        if type(t) == dict:
            t['TimeStep'] = 1./6.*BladeRotationAngle/RPM
        else:
            VPMParameters = getVPMParameters(t)
            VPMParameters['TimeStep'] = 1./6.*BladeRotationAngle/RPM

    TimeStepFunction_str2int = {'setTimeStepFromBladeRotationAngle':
    setTimeStepFromBladeRotationAngle, 'shed': setTimeStepFromShedParticles,
    'BladeRotationAngle': setTimeStepFromBladeRotationAngle, 'setTimeStepFromShedParticles':
    setTimeStepFromShedParticles, 'ShedParticles': setTimeStepFromShedParticles, 'Angle':
    setTimeStepFromBladeRotationAngle, 'angle': setTimeStepFromBladeRotationAngle, 'Shed':
    setTimeStepFromShedParticles}

    def setMinNbShedParticlesPerLiftingLine(LiftingLines = [], Parameters = {},
        NumberParticlesShedAtTip = 5):
        LL.computeKinematicVelocity(LiftingLines)
        LL.assembleAndProjectVelocities(LiftingLines)
        Urelmax = 0.
        L = 0.
        flag = False
        if 'VelocityFreestream' not in Parameters: flag = True
        else: U0 = Parameters['VelocityFreestream']
        for LiftingLine in LiftingLines:
            Ukin = np.array(J.getVars(LiftingLine, ['VelocityKinematicX',
                                                       'VelocityKinematicY', 'VelocityKinematicZ']))
            ui = np.array(J.getVars(LiftingLine, ['VelocityInducedX',
                                                           'VelocityInducedY', 'VelocityInducedZ']))
            if flag:
                U0 = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityFreestream'))
            Urel= np.linalg.norm(U0 + ui - Ukin)
            if (Urelmax < Urel):
                Urelmax = Urel
                L = np.minimum(L, W.getLength(LiftingLine))
        Parameters['MinNbShedParticlesPerLiftingLine'] = int(round(2. + NumberParticlesShedAtTip*L/\
                                                                       Urel/Parameters['TimeStep']))

    def renameLiftingLineTree(LiftingLineTree = [], VPMParameters = {}, LiftingLineParameters = {}):
        if not LiftingLineTree: return

        TypeOfInput = I.isStdNode(LiftingLineTree)
        ERRMSG = J.FAIL + 'LiftingLines must be a tree, a list of bases or a list of zones' + J.ENDC
        if TypeOfInput == -1:# is a standard CGNS node
            if I.isTopTree(LiftingLineTree):
                LiftingLineBases = I.getBases(LiftingLineTree)
                if len(LiftingLineBases) == 1 and LiftingLineBases[0][0] == 'Base':
                    LiftingLineBases[0][0] = 'LiftingLines'
            elif LiftingLineTree[3] == 'CGNSBase_t':
                LiftingLineBase = LiftingLineTree
                if LiftingLineBase[0] == 'Base': LiftingLineBase[0] = 'LiftingLines'
                LiftingLineTree = C.newPyTree([])
                LiftingLineTree[2] = [LiftingLineBase]
            elif LiftingLineTree[3] == 'Zone_t':
                LiftingLineTree = C.newPyTree(['LiftingLines', [LiftingLineTree]])
            else:
                raise AttributeError(ERRMSG)
        elif TypeOfInput == 0:# is a list of CGNS nodes
            if LiftingLineTree[0][3] == 'CGNSBase_t':
                LiftingLineTreeBases = I.getBases(LiftingLineTree)
                LiftingLineTree = C.newPyTree([])
                LiftingLineTree[2] = LiftingLineTreeBases
            elif LiftingLineTree[0][3] == 'Zone_t':
                LiftingLineTreeZones = I.getZones(LiftingLineTree)
                LiftingLineTree = C.newPyTree(['LiftingLines', LiftingLineTreeZones])
            else:
                raise AttributeError(ERRMSG)

        else:
            raise AttributeError(ERRMSG)

    def updateParameters(LiftingLineTree = [], Parameters = {}, LiftingLineParameters = {}):
        LiftingLines = I.getZones(LiftingLineTree)
        flag = False
        if LiftingLines and LiftingLineParameters['MinNbShedParticlesPerLiftingLine']%2:
            if 'ParticleDistribution' in LiftingLineParameters and \
                                       LiftingLineParameters['ParticleDistribution']['Symmetrical']:
                flag = True
            else:
                for LiftingLine in LiftingLines:
                    LLParameters = J.get(LiftingLine, '.VPM#Parameters')
                    if LLParameters and 'ParticleDistribution' in LLParameters and 'Symmetrical' in\
                                  LLParameters['ParticleDistribution'] and \
                                                LLParameters['ParticleDistribution']['Symmetrical']:
                        flag = True

            if flag:
                LiftingLineParameters['MinNbShedParticlesPerLiftingLine'] += 1
                print('||' + '{:=^50}'.format(''))
                print('||Odd number of embedded particles on at least one ')
                print('||Lifting Line dispite its symmetry. Number of ')
                print('||particle sources changed to ' + 
                                  str(LiftingLineParameters['MinNbShedParticlesPerLiftingLine'][0]))
                print('||' + '{:=^50}'.format(''))

        Nsmin = LiftingLineParameters['MinNbShedParticlesPerLiftingLine'][0]
        if 'Resolution' not in Parameters and LiftingLines:
            Resolution = W.getLength(LiftingLines[0])/max(Nsmin, \
                                I.getNodeFromName(LiftingLines[0], 'NumberOfParticleSources')[1][0])
            hmax, hmin = Resolution, Resolution
            for LiftingLine in LiftingLines[1:]:
                Resolution =  W.getLength(LiftingLine)/max(Nsmin, \
                                    I.getNodeFromName(LiftingLine, 'NumberOfParticleSources')[1][0])
                hmax = max(Resolution, hmax)
                hmin = min(Resolution, hmin)
            
            Parameters['Resolution'] = np.array([hmin, hmax], dtype = np.float64, order = 'F')
        elif 'Resolution' in Parameters:
            Parameters['Resolution'] = np.array([min(Parameters['Resolution']),
                                    max(Parameters['Resolution'])], dtype = np.float64, order = 'F')
        elif 'Resolution' not in Parameters:
            raise ValueError('The Resolution can not be computed. The Resolution or a Lifting Line \
                                                                                 must be specified')

        if 'VelocityFreestream' not in Parameters: Parameters['VelocityFreestream'] = \
                                                   np.array([0.]*3, dtype = np.float64, order = 'F')

        if 'TimeStep' not in Parameters: setTimeStepFromShedParticles(Parameters, LiftingLines,
                                                                      NumberParticlesShedAtTip = 1.)

        Parameters['Sigma0'] = np.array(Parameters['Resolution']*Parameters['SmoothingRatio'],
                                                                    dtype = np.float64, order = 'F')
        Parameters['IterationCounter'] = np.array([0], dtype = np.int32, order = 'F')
        Parameters['StrengthRampAtbeginning'][0] = max(Parameters['StrengthRampAtbeginning'], 1)
        Parameters['MinimumVorticityFactor'][0] = max(0., Parameters['MinimumVorticityFactor'])

        if 'RPM' in LiftingLineParameters:
            RPM = LiftingLineParameters['RPM']
            LL.setRPM(LiftingLines, LiftingLineParameters['RPM'])

        if 'Pitch' in LiftingLineParameters:
            for LiftingLine in LiftingLines:
                FlowSolution = I.getNodeFromName(LiftingLine, 'FlowSolution')
                Twist = I.getNodeFromName(FlowSolution, 'Twist')
                Twist[1] += LiftingLineParameters['Pitch']

        for LiftingLine in LiftingLines:
            J.set(LiftingLine, '.Conditions',
                    Density=Parameters['Density'],
                    Temperature=Parameters['Temperature'],
                    VelocityFreestream=Parameters['VelocityFreestream'])

        if 'VelocityTranslation' in LiftingLineParameters:
            for LiftingLine in LiftingLines:
                Kinematics = I.getNodeFromName(LiftingLine, '.Kinematics')
                VelocityTranslation = I.getNodeFromName(Kinematics, 'VelocityTranslation')
                VelocityTranslation[1] = np.array(LiftingLineParameters['VelocityTranslation'],
                                                                    dtype = np.float64, order = 'F')

    def updateLiftingLines(LiftingLineTree = [], LiftingLineParameters = {}, dt = 0.):
        for LiftingLine in I.getZones(LiftingLineTree):
            LLParameters = J.get(LiftingLine, '.VPM#Parameters')
            Ns = I.getNodeFromName(LiftingLine, 'NumberOfParticleSources')
            if Ns:
                LLParameters['NumberOfParticleSources'] = max(Ns[1], 
                                   LiftingLineParameters['MinNbShedParticlesPerLiftingLine'][0])
            else:
                LLParameters['NumberOfParticleSources'] = \
                                    LiftingLineParameters['MinNbShedParticlesPerLiftingLine'][0]
            h = W.getLength(LiftingLine)/LLParameters['NumberOfParticleSources']
            if not LLParameters:
                LL.setVPMParameters(LiftingLine)
                LLParameters = J.get(LiftingLine, '.VPM#Parameters')
            if 'IntegralLaw' in LiftingLineParameters:
                LLParameters['IntegralLaw'] = LiftingLineParameters['IntegralLaw']
            if 'ParticleDistribution' in LiftingLineParameters:
                ParticleDistributionOld = LiftingLineParameters['ParticleDistribution']
            else: ParticleDistributionOld = LLParameters['ParticleDistribution']

            ParticleDistribution = {'kind' : ParticleDistributionOld['kind']}

            if 'FirstSegmentRatio' in ParticleDistributionOld:
                ParticleDistribution['FirstCellHeight'] = \
                            ParticleDistributionOld['FirstSegmentRatio']*h
            if 'LastSegmentRatio' in ParticleDistributionOld:
                ParticleDistribution['LastCellHeight'] =\
                             ParticleDistributionOld['LastSegmentRatio']*h
            if 'growthRatio' in ParticleDistributionOld:
                ParticleDistribution['growth'] = ParticleDistributionOld['growthRatio']*h
            if 'parameter' in ParticleDistributionOld:
                ParticleDistribution['parameter'] = ParticleDistributionOld['parameter']
            if 'Symmetrical' in ParticleDistributionOld:
                ParticleDistribution['Symmetrical'] = ParticleDistributionOld['Symmetrical']

            LLParameters['ParticleDistribution'] = ParticleDistribution

            J.set(LiftingLine, '.VPM#Parameters',
                            IntegralLaw = LLParameters['IntegralLaw'],
                            ParticleDistribution = LLParameters['ParticleDistribution'],
                            NumberOfParticleSources = LLParameters['NumberOfParticleSources'],
                            TimeSinceLastShedding = dt[0])

    def initialiseParticlesOnLitingLine(t = [], LiftingLines = [], PolarInterpolator = {},
        VPMParameters = {}, LiftingLineParameters = {}):
        if not LiftingLines: return

        LL.computeKinematicVelocity(LiftingLines)
        LL.assembleAndProjectVelocities(LiftingLines)
        LL._applyPolarOnLiftingLine(LiftingLines, PolarInterpolator, ['Cl', 'Cd','Cm'])
        LL.computeGeneralLoadsOfLiftingLine(LiftingLines)

        X, Y, Z, S = [], [], [], []
        X0, Y0, Z0, S0 = [], [], [], []
        for LiftingLine in LiftingLines:
            #Gamma, GammaM1 = J.getVars(LiftingLine, ['Gamma', 'GammaM1'])
            #GammaM1[:] = Gamma[:]
            h = W.getLength(LiftingLine)/\
                                     I.getNodeFromName(LiftingLine, 'NumberOfParticleSources')[1][0]
            L = W.getLength(LiftingLine)
            LLParameters = J.get(LiftingLine, '.VPM#Parameters')
            ParticleDistribution = LLParameters['ParticleDistribution']
            if ParticleDistribution['Symmetrical']:
                HalfStations = int(LLParameters['NumberOfParticleSources']/2 + 1)
                SemiWing = W.linelaw(P1 = (0., 0., 0.), P2 = (L/2., 0., 0.), N = HalfStations,
                                                                Distribution = ParticleDistribution)# has to give +1 point because one point is lost with T.symetrize()
                WingDiscretization = J.getx(T.join(T.symetrize(SemiWing, (0, 0, 0), (0, 1, 0), \
                                                                              (0, 0, 1)), SemiWing))
                WingDiscretization += L/2.
                ParticleDistribution = WingDiscretization/L
            else:
                WingDiscretization = J.getx(W.linelaw(P1 = (0., 0., 0.), P2 = (L, 0., 0.),
                                                        N = LLParameters['NumberOfParticleSources'][0],
                                                            Distribution = ParticleDistribution))
                ParticleDistribution = WingDiscretization/L

            LLParameters = J.get(LiftingLine, '.VPM#Parameters')
            LLParameters['ParticleDistribution'] = ParticleDistribution
            J.set(LiftingLine, '.VPM#Parameters', **LLParameters)
            Source = LL.buildVortexParticleSourcesOnLiftingLine(LiftingLine,
                                                          AbscissaSegments = [ParticleDistribution],
                                                          IntegralLaw = LLParameters['IntegralLaw'])
            SourceX = I.getValue(I.getNodeFromName(Source, 'CoordinateX'))
            SourceY = I.getValue(I.getNodeFromName(Source, 'CoordinateY'))
            SourceZ = I.getValue(I.getNodeFromName(Source, 'CoordinateZ'))
            dy = ((SourceX[2:-1] - SourceX[1:-2])**2 + (SourceY[2:-1] - SourceY[1:-2])**2 +\
                                (SourceZ[2:-1] - SourceZ[1:-2])**2)**0.5
            X0 = np.append(X0, 0.5*(SourceX[2:-1] + SourceX[1:-2]))
            Y0 = np.append(Y0, 0.5*(SourceY[2:-1] + SourceY[1:-2]))
            Z0 = np.append(Z0, 0.5*(SourceZ[2:-1] + SourceZ[1:-2]))
            S0 = np.append(S0, dy*VPMParameters['SmoothingRatio'][0])
            Kinematics = J.get(LiftingLine, '.Kinematics')
            VelocityRelative = VPMParameters['VelocityFreestream']-Kinematics['VelocityTranslation']
            Dpsi = Kinematics['RPM']*6.*VPMParameters['TimeStep']
            #if (Dpsi == 0. and VelocityRelative == 0.): VelocityRelative = np.array(VPMParameters['Resolution'][0], VPMParameters['Resolution'][0], VPMParameters['Resolution'][0])
            if not Kinematics['RightHandRuleRotation']: Dpsi *= -1
            T._rotate(Source, Kinematics['RotationCenter'], Kinematics['RotationAxis'], -Dpsi[0])
            T._translate(Source, VelocityRelative/(np.linalg.norm(VelocityRelative) + 1e-12)*h)

            SourceX = I.getValue(I.getNodeFromName(Source, 'CoordinateX'))
            SourceY = I.getValue(I.getNodeFromName(Source, 'CoordinateY'))
            SourceZ = I.getValue(I.getNodeFromName(Source, 'CoordinateZ'))
            dy = 0.5*((SourceX[2:] - SourceX[:-2])**2 + (SourceY[2:] - SourceY[:-2])**2 +\
                                (SourceZ[2:] - SourceZ[:-2])**2)**0.5
            X = np.append(X, SourceX[1:-1])
            Y = np.append(Y, SourceY[1:-1])
            Z = np.append(Z, SourceZ[1:-1])
            S = np.append(S, dy*VPMParameters['SmoothingRatio'][0])
            
        Nbound = len(X0)
        LiftingLineParameters['NumberOfLiftingLineSources'] = Nbound
        extend(t, len(X) + len(X0), ExtendAtTheEnd = False, Offset = 0)
        Particles = pickParticlesZone(t)
        x, y, z = J.getxyz(Particles)
        x[:] = np.array(np.append(X0, X), dtype = np.float64, order = 'F')
        y[:] = np.array(np.append(Y0, Y), dtype = np.float64, order = 'F')
        z[:] = np.array(np.append(Z0, Z), dtype = np.float64, order = 'F')
        Nu, Cvisq, Sigma, Volume = J.getVars(Particles, ['Nu', 'Cvisq', 'Sigma', 'Volume'])
        Nu[:Nbound] = 0.
        Nu[Nbound:] = VPMParameters['KinematicViscosity']
        Cvisq[:Nbound] = 0.
        Cvisq[Nbound:] = VPMParameters['EddyViscosityConstant']
        Sigma[:] = np.array(np.append(S0, S), dtype = np.float64, order = 'F')
        Volume[:Nbound] = 0.
        Volume[Nbound:] = Sigma[Nbound:]**3

        LL.computeGeneralLoadsOfLiftingLine(LiftingLines,
                UnsteadyData={'IterationNumber':0,
                              'Time':0,
                              'CirculationSubiterations':0,
                              'CirculationError':0},
                                UnsteadyDataIndependentAbscissa='IterationNumber')

    def computeInducedVelocityOnLiftinLines(Target, x, y, z, ax, ay, az, s, VelocityLimitor,
        WakeInducedVelocity, Nshed):
        TargetBase = I.newCGNSBase('LiftingLine', cellDim=1, physDim=3)
        TargetBase[2] = I.getZones(Target)
        return vpm_cpp.induce_total_velocity_on_lifting_line(TargetBase, x, y, z, ax, ay, az, s, 
                                                        VelocityLimitor, WakeInducedVelocity, Nshed)

    def setShedParticleStrength(Dir, VeciX, VeciY, VeciZ, SheddingDistance, FilamentDistance, ax,
        ay, az, Sources, SourcesM1, NumberParticlesShedPerStation, NumberOfLiftingLineSources,
        NumberOfSources, TimeShed):
        SourcesBase = I.newCGNSBase('Sources', cellDim=1, physDim=3)
        SourcesBase[2] = I.getZones(Sources)
        SourcesBaseM1 = I.newCGNSBase('SourcesM1', cellDim=1, physDim=3)
        SourcesBaseM1[2] = I.getZones(SourcesM1)
        return vpm_cpp.shed_particles_from_lifting_lines(Dir, VeciX, VeciY, VeciZ, SheddingDistance,
            FilamentDistance, ax, ay, az, SourcesBase, SourcesBaseM1, NumberParticlesShedPerStation,
                                              NumberOfLiftingLineSources, NumberOfSources, TimeShed)

    def getLiftingLineParameters(t = []): return J.get(pickParticlesZone(t), 
                                                                          '.LiftingLine#Parameters')

    def relaxCirculationAndGetImbalance(GammaOld = [], GammaRelax = 0., Sources = []):
        GammaError = 0
        for i in range(len(Sources)):
            GammaNew, = J.getVars(Sources[i],['Gamma'])
            GammaError = max(GammaError, max(abs(GammaNew - GammaOld[i]))/max(1e-12, \
                                                                            np.mean(abs(GammaNew))))
            GammaNew[:] = (1. - GammaRelax)*GammaOld[i] + GammaRelax*GammaNew
            GammaOld[i][:] = GammaNew
        return GammaError

    def moveAndUpdateLiftingLines(t, LiftingLines, dt, PerturbationFieldCapsule = []):
        LL.computeKinematicVelocity(LiftingLines)
        LL.moveLiftingLines(LiftingLines, dt)
        extractperturbationField(t = t, Targets = LiftingLines,
                                                PerturbationFieldCapsule = PerturbationFieldCapsule)
        LL.assembleAndProjectVelocities(LiftingLines)

    def initialiseShedParticles(LiftingLines, Particles, Sources, dt, Ramp, ratio,
        NumberOfLiftingLineSources, NumberOfSources, ParticleDistribution):
        SheddingLiftingLine = []
        px, py, pz = J.getxyz(Particles)
        pos = NumberOfSources
        index = 0
        for Source, LiftingLine in zip(Sources, LiftingLines):#checks if their is enough space for particles to be shed for this LL
            h = W.getLength(LiftingLine)/\
                                     I.getNodeFromName(LiftingLine, 'NumberOfParticleSources')[1][0]
            sx, sy, sz = J.getxyz(Source)
            dy = 0
            for i in range(1, len(sx) - 1):
                dy += np.linalg.norm(np.array([sx[i]  , sy[i]  , sz[i]]) - 
                                     np.array([px[pos], py[pos],  pz[pos]]))
                pos += 1

            if dy/(len(sx) - 2) < h: SheddingLiftingLine += [index]

            index += 1

        ParticlesShedPerStation = []
        ShedParticlesX, ShedParticlesY, ShedParticlesZ = [], [], []
        FirstRowParticlesX, FirstRowParticlesY, FirstRowParticlesZ = [], [], []
        BoundParticlesX, BoundParticlesY, BoundParticlesZ = [], [], []
        ShedParticlesSigma, FirstRowParticlesSigma, BoundParticlesSigma = [], [], []
        VeciX, VeciY, VeciZ, = [], [], []
        Dir = []
        SheddingDistance, FilamentDistance = [], []
        px, py, pz = J.getxyz(Particles)
        Sigma = J.getVars(Particles, ['Sigma'])[0]
        index = 0
        for Source, LiftingLine in zip(Sources, LiftingLines):
            h = W.getLength(LiftingLine)/\
                                     I.getNodeFromName(LiftingLine, 'NumberOfParticleSources')[1][0]
            Dir += [1 if I.getValue(I.getNodeFromName(LiftingLine,'RightHandRuleRotation')) else -1]
            sx, sy, sz = J.getxyz(Source)
            s = 0.5*((sx[2:] - sx[:-2])**2 + (sy[2:] - sy[:-2])**2 + (sz[2:] - sz[:-2])**2)**0.5
            FirstRowParticlesSigma.extend(np.array(s, order='F', dtype=np.float64))
            BoundParticlesX.extend(0.5*(sx[2:-1] + sx[1:-2]))
            BoundParticlesY.extend(0.5*(sy[2:-1] + sy[1:-2]))
            BoundParticlesZ.extend(0.5*(sz[2:-1] + sz[1:-2]))
            s = ratio*((sx[2:-1] - sx[1:-2])**2 + (sy[2:-1] - sy[1:-2])**2 + \
                                                                      (sz[2:-1] - sz[1:-2])**2)**0.5
            BoundParticlesSigma.extend(np.array(s, order='F', dtype=np.float64))
            if index in SheddingLiftingLine:
                for i in range(1, len(sx) - 1):
                    VeciX += [0.]
                    VeciY += [0.]
                    VeciZ += [0.]
                    SheddingDistance += [0.]
                    FilamentDistance += [0.]
                    FirstRowParticlesX += [px[len(ParticlesShedPerStation) + NumberOfSources]]
                    FirstRowParticlesY += [py[len(ParticlesShedPerStation) + NumberOfSources]]
                    FirstRowParticlesZ += [pz[len(ParticlesShedPerStation) + NumberOfSources]]
                    ParticlesShedPerStation += [-1]
            else:
                for i in range(1, len(sx) - 1):
                    xm = np.array([sx[i], sy[i], sz[i]])
                    vecj = np.array([px[len(ParticlesShedPerStation) + NumberOfSources], \
                                               py[len(ParticlesShedPerStation) + NumberOfSources], \
                                               pz[len(ParticlesShedPerStation) + NumberOfSources]]) - xm
                    dy = np.linalg.norm(vecj)
                    VeciX += [0.5*(sx[i + 1] - sx[i - 1])]
                    VeciY += [0.5*(sy[i + 1] - sy[i - 1])]
                    VeciZ += [0.5*(sz[i + 1] - sz[i - 1])]
                    SheddingDistance += [dy]
                    FilamentDistance += [np.linalg.norm([VeciX[-1], VeciY[-1], VeciZ[-1]])]
                    Nshed = max(int(round(dy/h - 0.95)), 0) + 1
                    FirstRowParticlesX += [xm[0]]
                    FirstRowParticlesY += [xm[1]]
                    FirstRowParticlesZ += [xm[2]]
                    if Nshed != 0:
                        ShedParticlesSigma += [(VeciX[-1]**2 + VeciY[-1]**2 + VeciZ[-1]**2)**0.5]*(Nshed -1)
                        for j in range(1, Nshed):
                            ShedParticlesX += [xm[0] + j/Nshed*vecj[0]]
                            ShedParticlesY += [xm[1] + j/Nshed*vecj[1]]
                            ShedParticlesZ += [xm[2] + j/Nshed*vecj[2]]

                    ParticlesShedPerStation += [Nshed]

            index += 1

        for index in SheddingLiftingLine[::-1]:
            Sources.pop(index)
            LiftingLines.pop(index)
            ParticleDistribution.pop(index)
            Dir.pop(index)

        ParticlesShedPerStation = np.array(ParticlesShedPerStation, dtype=np.int32, order = 'F')
        Ramp = Ramp/(ParticlesShedPerStation + 2*(ParticlesShedPerStation == -1))
        Dir = np.array(Dir, dtype = np.int32, order = 'F')
        VeciX = np.array(VeciX, dtype = np.float64, order = 'F')*Ramp
        VeciY = np.array(VeciY, dtype = np.float64, order = 'F')*Ramp
        VeciZ = np.array(VeciZ, dtype = np.float64, order = 'F')*Ramp
        SheddingDistance = np.array(SheddingDistance, dtype = np.float64, order = 'F')*Ramp + 1e-12
        FilamentDistance = np.array(FilamentDistance, dtype = np.float64, order = 'F')*Ramp + 1e-12
        Nshed = np.sum(ParticlesShedPerStation) + 2*np.sum(ParticlesShedPerStation == -1)
        extend(Particles, ExtendSize = Nshed, Offset = NumberOfSources, ExtendAtTheEnd = False)
        Sigma = J.getVars(Particles, ['Sigma'])[0]
        Sigma[:NumberOfLiftingLineSources] = np.array(BoundParticlesSigma, dtype = np.float64,
                                                                                        order = 'F')
        Sigma[NumberOfSources: Nshed + NumberOfSources] = np.array(np.append(FirstRowParticlesSigma,
                                               ShedParticlesSigma), dtype = np.float64, order = 'F')
        x, y, z = J.getxyz(Particles)
        x[: NumberOfLiftingLineSources] = np.array(BoundParticlesX, dtype = np.float64, order = 'F')
        y[: NumberOfLiftingLineSources] = np.array(BoundParticlesY, dtype = np.float64, order = 'F')
        z[: NumberOfLiftingLineSources] = np.array(BoundParticlesZ, dtype = np.float64, order = 'F')
        x[NumberOfSources: Nshed + NumberOfSources] = np.array(np.append(FirstRowParticlesX,
                                                   ShedParticlesX), dtype = np.float64, order = 'F')
        y[NumberOfSources: Nshed + NumberOfSources] = np.array(np.append(FirstRowParticlesY,
                                                   ShedParticlesY), dtype = np.float64, order = 'F')
        z[NumberOfSources: Nshed + NumberOfSources] = np.array(np.append(FirstRowParticlesZ,
                                                   ShedParticlesZ), dtype = np.float64, order = 'F')


        return ParticlesShedPerStation, Dir, VeciX, VeciY, VeciZ, SheddingDistance, FilamentDistance

    def ShedParticlesFromLiftingLines(Particles, LiftingLines, PolarsInterpolatorDict,
        WakeInducedVelocity, GammaThreshold, GammaRelax, MaxIte, dt, ratio, Ramp,
        KinematicViscosity, EddyViscosityConstant, VelocityLimitor, NumberOfLiftingLineSources,
        NumberOfSources):
        it = getParameter(Particles, 'CurrentIteration')[0]

        #C.convertPyTree2File(Particles, 't' + str(it) + '_av.cgns')
        ParticleDistribution = [I.getNodeFromName(LiftingLine, 'ParticleDistribution')[1] for \
                                                                        LiftingLine in LiftingLines]
        Sources = LL.buildVortexParticleSourcesOnLiftingLine(LiftingLines, AbscissaSegments = \
                                                       ParticleDistribution, IntegralLaw = 'linear')
        x, y, z = J.getxyz(Particles)
        #print('tot before ini', len(x), len(Sources), [ll[0] for ll in LiftingLines])
        ParticlesShedPerStation, Dir, VeciX, VeciY, VeciZ, SheddingDistance, FilamentDistance = \
                                     initialiseShedParticles(LiftingLines, Particles, Sources,
                 dt, Ramp, ratio, NumberOfLiftingLineSources, NumberOfSources, ParticleDistribution)
        TimeShed = np.array([I.getNodeFromName(LiftingLine, 'TimeSinceLastShedding')[1][0] for \
                                      LiftingLine in LiftingLines], order = 'F', dtype = np.float64)
        Nshed = np.sum(ParticlesShedPerStation) + 2*np.sum(ParticlesShedPerStation == -1)
        SourcesM1 = [I.copyTree(Source) for Source in Sources]
        GammaOld = [I.getNodeFromName3(Source, 'Gamma')[1] for Source in Sources]
        x, y, z = J.getxyz(Particles)
        #print('tot after ini', len(x), len(Sources), [ll[0] for ll in LiftingLines])
        ax, ay, az, s = J.getVars(Particles, ['Alpha' + v for v in 'XYZ'] + ['Sigma'])

        ni = 0
        for _ in range(MaxIte):
            #print(ax[: NumberOfLiftingLineSources])
            setShedParticleStrength(Dir, VeciX, VeciY, VeciZ, SheddingDistance, FilamentDistance,
                                            ax, ay, az, Sources, SourcesM1, ParticlesShedPerStation,
                                              NumberOfLiftingLineSources, NumberOfSources, TimeShed)
            computeInducedVelocityOnLiftinLines(LiftingLines, x, y, z, ax, ay, az, s, \
                           VelocityLimitor, WakeInducedVelocity, Nshed + NumberOfLiftingLineSources)
            LL.assembleAndProjectVelocities(LiftingLines)
            LL._applyPolarOnLiftingLine(LiftingLines, PolarsInterpolatorDict, ['Cl', 'Cd','Cm'])
            IntegralLoads = LL.computeGeneralLoadsOfLiftingLine(LiftingLines)
            Sources = LL.buildVortexParticleSourcesOnLiftingLine(LiftingLines, AbscissaSegments = \
                                                       ParticleDistribution, IntegralLaw = 'linear')
            GammaError = relaxCirculationAndGetImbalance(GammaOld, GammaRelax, Sources)

            ni += 1
            if GammaError < GammaThreshold: break

        #C.convertPyTree2File(Particles, 't' + str(it) + '_ap.cgns')

        wx, wy, wz, w, a, Volume, Nu, Cvisq = J.getVars(Particles, \
                                            ['Vorticity'+i for i in 'XYZ'] + ['VorticityMagnitude',\
                                                      'StrengthMagnitude', 'Volume', 'Nu', 'Cvisq'])
        a[: NumberOfLiftingLineSources] = np.linalg.norm(np.vstack([
                                                        ax[:NumberOfLiftingLineSources],
                                                        ay[:NumberOfLiftingLineSources],
                                                        az[:NumberOfLiftingLineSources]]), axis = 0)
        #print('a bound', a[: NumberOfLiftingLineSources])
        a[NumberOfSources: NumberOfSources + Nshed] = np.linalg.norm(np.vstack([\
                                           ax[NumberOfSources: NumberOfSources + Nshed],
                                           ay[NumberOfSources: NumberOfSources + Nshed],
                                           az[NumberOfSources: NumberOfSources + Nshed]]), axis = 0)
        s[NumberOfSources: NumberOfSources + Nshed] *= ratio
        Volume[NumberOfSources: NumberOfSources + Nshed] = s[NumberOfSources: NumberOfSources + \
                                                                                           Nshed]**3
        Nu[NumberOfSources: NumberOfSources + Nshed] = KinematicViscosity
        Cvisq[NumberOfSources: NumberOfSources + Nshed] = EddyViscosityConstant
        flag = [False]*len(a)
        for i in range(len(ParticlesShedPerStation)):
            if ParticlesShedPerStation[i] == -1:
                a[NumberOfSources + i]  = a[NumberOfSources + Nshed + i]
                ax[NumberOfSources + i] = ax[NumberOfSources + Nshed + i]
                ay[NumberOfSources + i] = ay[NumberOfSources + Nshed + i]
                az[NumberOfSources + i] = az[NumberOfSources + Nshed + i]
                x[NumberOfSources + i]  =  x[NumberOfSources + Nshed + i]
                y[NumberOfSources + i]  =  y[NumberOfSources + Nshed + i]
                z[NumberOfSources + i]  =  z[NumberOfSources + Nshed + i]
                flag[NumberOfSources + Nshed + i] = True
        delete(Particles, np.array(flag))
        #print('a shed', a[NumberOfSources: NumberOfSources + NumberOfSources])

        for LiftingLine in LiftingLines:
            TimeShed = I.getNodeFromName(LiftingLine, 'TimeSinceLastShedding')
            TimeShed[1][0] = 0

        #C.convertPyTree2File(Particles, 't' + str(it) + '_del.cgns')

        #for LiftingLine in LiftingLines:
        #    Gamma, GammaM1, dGammadt = J.getVars(LiftingLine, ['Gamma', 'GammaM1', 'dGammadt'])
        #    dGammadt[:] = (Gamma[:] - GammaM1[:])/dt

        return GammaError, ni, Nshed - np.sum(flag)

    def ShedVorticitySourcesFromLiftingLines(t = [], PolarsInterpolatorDict = {},
        IterationInfo = {}, PerturbationFieldCapsule = []):
        timeLL = J.tic()
        LiftingLines = LL.getLiftingLines(t)
        if not LiftingLines: return IterationInfo

        #for LiftingLine in LiftingLines:
        #    Gamma, GammaM1 = J.getVars(LiftingLine, ['Gamma', 'GammaM1'])
        #    GammaM1[:] = Gamma[:].copy()

        Particles = pickParticlesZone(t)
        if not Particles: raise ValueError('"Particles" zone not found in ParticlesTree')
        ratio, dt, time, it, Ramp, GammaThreshold, GammaRelax, MaxIte, \
        KinematicViscosity, EddyViscosityConstant, MachLimitor, Temperature, \
        NumberOfLiftingLineSources, \
        NumberOfHybridSources, NumberOfBEMSources, NumberOfCFDSources = getParameters(t, \
                ['SmoothingRatio', 'TimeStep', 'Time', 'CurrentIteration', \
                 'StrengthRampAtbeginning', 'CirculationThreshold', 'CirculationRelaxation', \
                 'MaxLiftingLineSubIterations', 'KinematicViscosity', 'EddyViscosityConstant', \
                 'MachLimitor', 'Temperature', \
                 'NumberOfLiftingLineSources', 'NumberOfHybridSources', 'NumberOfBEMSources', \
                 'NumberOfCFDSources'])
        VelLimitor = MachLimitor*np.sqrt(1.4*287.053*Temperature)
        if not NumberOfHybridSources: NumberOfHybridSources = [0]
        if not NumberOfBEMSources: NumberOfBEMSources = [0]
        if not NumberOfCFDSources: NumberOfCFDSources = [0]
        Ramp = np.sin(min(it/Ramp, 1.)*np.pi/2.)
        moveAndUpdateLiftingLines(t, LiftingLines, dt, PerturbationFieldCapsule)
        WakeInducedVelocity = getInducedVelocityFromWake(t, Targets = LiftingLines)

        Error, ni, Nshed = ShedParticlesFromLiftingLines(Particles, LiftingLines,
              PolarsInterpolatorDict, WakeInducedVelocity, GammaThreshold, GammaRelax, MaxIte[0],
              dt[0], ratio, Ramp, KinematicViscosity, EddyViscosityConstant, VelLimitor[0],
              NumberOfLiftingLineSources[0], NumberOfLiftingLineSources[0] + \
              NumberOfHybridSources[0] + NumberOfCFDSources[0] + NumberOfBEMSources[0])

        LL.computeGeneralLoadsOfLiftingLine(LiftingLines,
                UnsteadyData={'IterationNumber':it[0],
                              'Time':time[0],
                              'CirculationSubiterations':ni,
                              'CirculationError':Error},
                                UnsteadyDataIndependentAbscissa='IterationNumber')

        IterationInfo['Circulation error'] = Error
        IterationInfo['Number of sub-iterations (LL)'] = ni
        IterationInfo['Number of shed particles'] = Nshed
        IterationInfo['Lifting Line time'] = J.tic() - timeLL
        return IterationInfo

####################################################################################################
####################################################################################################
########################################### Vortex Rings ###########################################
####################################################################################################
####################################################################################################
    def createLambOseenVortexRing(t, VPMParameters, VortexParameters):
        Gamma = VortexParameters['Intensity'][0]
        sigma = VPMParameters['Sigma0'][0]
        lmbd_s = VPMParameters['SmoothingRatio'][0]
        nu = VPMParameters['KinematicViscosity'][0]
        R = VortexParameters['RingRadius'][0]
        nc = [0]
        h = VPMParameters['Resolution'][0]
        
        if 'CoreRadius' in VortexParameters:
            a = VortexParameters['CoreRadius'][0]
            if nu != 0.: tau = a**2/4./nu
            else: tau = 0.
            VortexParameters['Tau'] = np.array([tau], dtype = np.float64, order = 'F')
        elif 'Tau' in VortexParameters:
            tau = VortexParameters['Tau'][0]
            a = (4.*nu*tau)**0.5
            VortexParameters['CoreRadius'] = np.array([a], dtype = np.float64, order = 'F')

        a = VortexParameters['CoreRadius'][0]
        w = lambda r : Gamma/(np.pi*a**2)*np.exp(-(r/a)**2)

        if 'MinimumVorticityFraction' in VortexParameters:
            r = a*(-np.log(VortexParameters['MinimumVorticityFraction'][0]))**0.5
            nc = int(r/h)
            VortexParameters['NumberLayers'] = np.array([nc], dtype = np.int32, order = 'F')
        elif 'NumberLayers' in VortexParameters:
            nc = VortexParameters['NumberLayers'][0]
            frac = w(nc*h)
            VortexParameters['MinimumVorticityFraction'] = \
                                                  np.array([frac], dtype = np.float64, order = 'F')

        N_s = 1 + 3*nc*(nc + 1)
        N_phi = int(2.*np.pi*R/h)
        N_phi += N_phi%4
        Np = N_s*N_phi
        extend(t, Np)
        Particles = pickParticlesZone(t)
        px, py, pz = J.getxyz(Particles)
        AlphaX, AlphaY, AlphaZ, VorticityX, VorticityY, VorticityZ, Volume, Sigma, Nu = \
                                       J.getVars(Particles, ['Alpha' + v for v in 'XYZ'] + \
                                       ['Vorticity' + v for v in 'XYZ'] + ['Volume', 'Sigma', 'Nu'])
        Nu[:] = nu
        
        r0 = h/2.
        rc = r0*(2*nc + 1)
        if (Np != N_phi*N_s): print("Achtung Bicyclette")
        if (R - rc < 0): print("Beware of the initial ring radius " , R , " < " , rc)
        else: print("R=", R, ", rc=", rc, ", a=", a, ", sigma=", sigma, ", nc=", nc, ", N_phi=",
                                                                   N_phi, ", N_s=", N_s, ", N=", Np)

        X = [R]
        Z = [0.]
        W = [Gamma/(np.pi*a**2)]
        V = [2.*np.pi**2*R/N_phi*r0**2]
        for n in range(1, nc + 1):
            for j in range(6*n):
                theta = np.pi*(2.*j + 1.)/6./n
                r = r0*(1. + 12.*n*n)/6./n
                X.append(R + r*np.cos(theta))
                Z.append(r*np.sin(theta))
                V.append(4./3.*4.*np.pi*r0*r0/N_phi*(np.pi*R/2. + (np.sin(np.pi*(j + 1)/4./n) - \
                                                         np.sin(np.pi*j/4./n))*(4.*n*n + 1./3.)*r0))
                W.append(W[0]*np.exp(-(r/a)**2))

        print(W[0])
        S = [lmbd_s*h]*len(X)
        for i in range(N_phi):
            phi = 2.*np.pi*i/N_phi
            for j in range(N_s):
                px[i*N_s + j] = X[j]*np.cos(phi)
                py[i*N_s + j] = X[j]*np.sin(phi)
                pz[i*N_s + j] = Z[j]
                Volume[i*N_s + j] = V[j]
                Sigma[i*N_s + j] = S[j]
                VorticityX[i*N_s + j] = -W[j]*np.sin(phi)
                VorticityY[i*N_s + j] = W[j]*np.cos(phi)
                VorticityZ[i*N_s + j] = 0.
                AlphaX[i*N_s + j] = V[j]*VorticityX[i*N_s + j]
                AlphaY[i*N_s + j] = V[j]*VorticityY[i*N_s + j]
                AlphaZ[i*N_s + j] = V[j]*VorticityZ[i*N_s + j]

        VortexParameters['Nphi'] = N_phi
        VortexParameters['Ns'] = N_s
        VortexParameters['NumberLayers'] = nc
        vpm_cpp.adjust_vortex_ring(t, N_s, N_phi, Gamma, np.pi*r0**2, np.pi*r0**2*4./3., nc)

    def createLambOseenVortexBlob(t, VPMParameters, VortexParameters):
        Gamma = VortexParameters['Intensity'][0]
        sigma = VPMParameters['Sigma0'][0]
        lmbd_s = VPMParameters['SmoothingRatio'][0]
        nu = VPMParameters['KinematicViscosity'][0]
        L = VortexParameters['Length'][0]
        frac = VortexParameters['MinimumVorticityFraction'][0]
        h = VPMParameters['Resolution'][0]
        if 'CoreRadius' in VortexParameters:
            a = VortexParameters['CoreRadius'][0]
            tau = a**2/4./nu
            VortexParameters['Tau'] = np.array([tau], dtype = np.float64, order = 'F')
        elif 'Tau' in VortexParameters:
            tau = VortexParameters['Tau'][0]
            a = (4.*nu*tau)**0.5
            VortexParameters['CoreRadius'] = np.array([a], dtype = np.float64, order = 'F')

        w = lambda r : Gamma/(4.*np.pi*nu*tau)*np.exp(-r**2/(4.*nu*tau))
        minw = w(0)*frac
        l = a*(-np.log(frac))**0.5
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
        extend(t, Np)
        Particles = pickParticlesZone(t)
        px, py, pz = J.getxyz(Particles)
        AlphaX, AlphaY, AlphaZ, VorticityX, VorticityY, VorticityZ, Volume, Sigma, Nu = \
                                       J.getVars(Particles, ['Alpha' + v for v in 'XYZ'] + \
                                       ['Vorticity' + v for v in 'XYZ'] + ['Volume', 'Sigma', 'Nu'])
        Nu[:] = nu
        
        r0 = h/2.
        rc = r0*(2*nc + 1)
        print("L=", L, ", tau=", tau, ", rc=", l, ", a=", a, ", sigma=", sigma, ", NL=", NL, \
                                                                            ", Ns=", Ns, ", N=", Np)

        X = [0]
        Z = [0.]
        W = [Gamma/np.pi/a**2]
        V = [h*np.pi*r0**2]
        for n in range(1, nc + 1):
            for j in range(6*n):
                theta = np.pi*(2.*j + 1.)/6./n
                r = r0*(1. + 12.*n*n)/6./n
                X.append(r*np.cos(theta))
                Z.append(r*np.sin(theta))
                V.append(V[0]*4./3.)
                W.append(W[0]*np.exp(-(r/a)**2))
        
        print("W in", w(0), np.min(W))
        S = [lmbd_s*h]*len(X)
        px[:Ns] = X[:]
        py[:Ns] = 0.
        pz[:Ns] = Z[:]
        Volume[:Ns] = V[:]
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
                Volume[i*Ns + j] = V[j]
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
                Volume[(i + 1)*Ns + j] = V[j]
                Sigma[(i + 1)*Ns + j] = S[j]
                VorticityX[(i + 1)*Ns + j] = 0.
                VorticityY[(i + 1)*Ns + j] = W[j]
                VorticityZ[(i + 1)*Ns + j] = 0.
                AlphaX[(i + 1)*Ns + j] = 0.
                AlphaY[(i + 1)*Ns + j] = V[j]*VorticityY[(i + 1)*Ns + j]
                AlphaZ[(i + 1)*Ns + j] = 0.
        
        vpm_cpp.adjust_vortex_tube(t, Ns, NL, Gamma, np.pi*r0**2, np.pi*r0**2*4./3., nc)
    
    def computeVortexRing(VPMParameters, VortexParameters, NumberOfIterations = 10000,
        SaveVPMPeriod = 10, DIRECTORY_OUTPUT = 'OUTPUT', LeapFrog = False):
        int_Params =['StrengthRampAtbeginning', 
            'CurrentIteration', 'ClusterSize', 
            'MaximumAgeAllowed', 'RedistributionPeriod', 'NumberOfThreads', 'IntegrationOrder',
            'IterationTuningFMM', 'IterationCounter', 
            'FarFieldApproximationOrder', 'NumberLayers']

        float_Params = ['Density', 'EddyViscosityConstant', 'Temperature', 'ResizeParticleFactor',
            'Time', 'CutoffXmin', 'CutoffZmin', 'MaximumMergingVorticityFactor',
            'MagnitudeRelaxationFactor', 'SFSContribution', 'SmoothingRatio', 'RPM','KinematicViscosity',
            'Pitch', 'CutoffXmax', 'CutoffYmin', 'CutoffYmax', 'Sigma0',
            'CutoffZmax', 'ForcedDissipation','MaximumAngleForMerging', 'MinimumVorticityFactor', 
            'MinimumOverlapForMerging', 'VelocityFreestream', 'AntiStretching',
            'RedistributeParticlesBeyond', 'RedistributeParticleSizeFactor', 'MachLimitor',
            'TimeStep', 'Resolution', 'NearFieldOverlappingRatio', 'TimeFMM',
            'RemoveWeakParticlesBeyond', 'Intensity', 'CoreRadius', 'RingRadius', 'Length', 'Tau',
            'MinimumVorticityFraction', 'EddyViscosityRelaxationFactor', 'StrengthVariationLimitor',
            'RealignmentRelaxationFactor']

        bool_Params = ['MonitorDiagnostics', 'LowStorageIntegration']
        
        defaultParameters = {
            ############################################################################################
            ################################### Simulation conditions ##################################
            ############################################################################################
                'Density'                       : 1.225,          #]0., +inf[, in kg.m^-3
                'EddyViscosityConstant'         : 0.15,            #[0., +inf[, constant for the eddy viscosity model, Cm(Mansour) around 0.1, Cs(Smagorinsky) around 0.15, Cr(Vreman) around 0.07
                'EddyViscosityModel'            : 'Vreman',       #Mansour, Mansour2, Smagorinsky, Vreman or None, select a LES model to compute the eddy viscosity
                'KinematicViscosity'            : 1.46e-5,        #[0., +inf[, in m^2.s^-1
                'Temperature'                   : 288.15,         #]0., +inf[, in K
                'Time'                          : 0.,             #in s, keep track of the physical time
            ############################################################################################
            ###################################### VPM parameters ######################################
            ############################################################################################
                'AntiStretching'                : 0.,             #between 0 and 1, 0 means particle strength fully takes vortex stretching, 1 means the particle size fully takes the vortex stretching
                'DiffusionScheme'               : 'PSE',          #PSE, CSM or None. gives the scheme used to compute the diffusion term of the vorticity equation
                'RegularisationKernel'          : 'Gaussian',     #The available smoothing kernels are Gaussian, HOA, LOA, Gaussian3 and SuperGaussian
                'SFSContribution'               : 0.,             #between 0 and 1, the closer to 0, the more the viscosity affects the particle strength, the closer to 1, the more it affects the particle size
                'SmoothingRatio'                : 2.,             #in m, anywhere between 1.5 and 2.5, the higher the NumberSource, the smaller the Resolution and the higher the SmoothingRatio should be to avoid blowups, the HOA kernel requires a higher smoothing
                'VorticityEquationScheme'       : 'Transpose',    #Classical, Transpose or Mixed, The schemes used to compute the vorticity equation are the classical scheme, the transpose scheme (conserves total vorticity) and the mixed scheme (a fusion of the previous two)
                'Sigma0'                        : 0.1,
                'MachLimitor'                   : 0.9,            #[0, +in[, gives the maximum velocity a particle can have
                'StrengthVariationLimitor'      : 2.,             #[0, +in[, gives the maximum ratio a particle can grow/shrink when updated with the vorticity equation
            ############################################################################################
            ################################### Numerical Parameters ###################################
            ############################################################################################
                'CurrentIteration'              : 0,              #follows the current iteration
                'IntegrationOrder'              : 3,              #[|1, 4|]1st, 2nd, 3rd or 4th order Runge Kutta. In the hybrid case, there must be at least as much Interfaces in the hybrid domain as the IntegrationOrder of the time integration scheme
                'LowStorageIntegration'         : True,           #[|0, 1|], states if the classical or the low-storage Runge Kutta is used
                'MonitorDiagnostics'            : True,           #[|0, 1|], allows or not the computation of the diagnostics (kinetic energy, enstrophy, divergence-free kinetic energy, divergence-free enstrophy)
            ############################################################################################
            ##################################### Particles Control ####################################
            ############################################################################################
                'CutoffXmin'                    : -np.inf,        #in m, spatial Cutoff
                'CutoffXmax'                    : +np.inf,        #in m, spatial Cutoff
                'CutoffYmin'                    : -np.inf,        #in m, spatial Cutoff
                'CutoffYmax'                    : +np.inf,        #in m, spatial Cutoff
                'CutoffZmin'                    : -np.inf,        #in m, spatial Cutoff
                'CutoffZmax'                    : +np.inf,        #in m, spatial Cutoff
                'ForcedDissipation'             : 0.,             #in %/s, gives the % of strength particles looses per sec, usefull to kill unnecessary particles without affecting the LLs
                'MaximumAgeAllowed'             : 0,              #0 <=,  particles are eliminated after MaximumAgeAllowed iterations, if MaximumAgeAllowed == 0, they are not deleted
                'MaximumAngleForMerging'        : 0.,             #[0., 180.[ in deg,   maximum angle   allowed between two particles to be merged
                'MaximumMergingVorticityFactor' : 0.,             #in %, particles can be merged if their combined strength is below the given poucentage of the maximum strength on the blades
                'MinimumOverlapForMerging'      : 0.,             #[0., +inf[, if two particles have at least an overlap of MinimumOverlapForMerging*SigmaRatio, they are considered for merging
                'MinimumVorticityFactor'        : 0.,             #in %, sets the minimum strength kept as a percentage of the maximum strength on the blades
                'RedistributeParticlesBeyond'   : np.inf,         #do not redistribute particles if closer than RedistributeParticlesBeyond*Resolution from a LL
                'RedistributionKernel'          : None,           #M4Prime, M4, M3, M2, M1 or None, redistribution kernel used. the number gives the order preserved by the kernel, if None local splitting/merging is used
                'RedistributionPeriod'          : 0,              #frequency at which particles are redistributed, if 0 the particles are never redistributed
                'RealignmentRelaxationFactor'   : 0.,             #[0., 1.[, is used during the relaxation process to realign the particles with their voticity and avoid having a non null divergence of the vorticity field
                'MagnitudeRelaxationFactor'     : 0.,             #[0., 1.[, is used during the relaxation process to change the magnitude of the particles to avoid having a non null divergence of the vorticity field
                'EddyViscosityRelaxationFactor' : 0.,             #[0., 1.[, is used during the relaxation process when updating the eddy viscosity constant to satisfy the transfert of enstrophy to the kinetic energy
                'RemoveWeakParticlesBeyond'     : np.inf,         #do not remove weak particles if closer than RemoveWeakParticlesBeyond*Resolution from a LL
                'ResizeParticleFactor'          : 0.,             #[0, +inf[, resize particles that grow/shrink RedistributeParticleSizeFactor * Sigma0 (i.e. Resolution*SmoothingRatio), if 0 then no resizing is done
                'StrengthRampAtbeginning'       : 25,             #[|0, +inf [|, limit the vorticity shed for the StrengthRampAtbeginning first iterations for the wake to stabilise
            ############################################################################################
            ###################################### FMM parameters ######################################
            ############################################################################################
                'FarFieldApproximationOrder'    : 8,              #[|6, 12|], order of the polynomial which approximates the far field interactions, the higher the more accurate and the more costly
                'IterationTuningFMM'            : 50,             #frequency at which the FMM is compared to the direct computation, gives the relative L2 error
                'NearFieldOverlappingRatio'     : 0.5,            #[0., 1.], Direct computation of the interactions between clusters that overlap by NearFieldOverlappingRatio, the smaller the more accurate and the more costly
                'NumberOfThreads'               : 'auto',         #number of threads of the machine used. If 'auto', the highest number of threads is set
                'TimeFMM'                       : 0.,             #in s, keep track of the CPU time spent for the FMM
                'ClusterSize'                   : 2**9,           #[|0, +inf[|, maximum number of particles per FMM cluster, better as a power of 2
        }
        defaultVortexParameters = {
            ############################################################################################
            ################################## Vortex Rings parameters #################################
            ############################################################################################
                'Intensity'      : 1.,
                'NumberLayers'   : 6,
        }
        defaultParameters.update(VPMParameters)
        defaultVortexParameters.update(VortexParameters)

        if defaultParameters['NumberOfThreads'] == 'auto':
            NbOfThreads = int(os.getenv('OMP_NUM_THREADS',len(os.sched_getaffinity(0))))
            defaultParameters['NumberOfThreads'] = NbOfThreads
        else:
            NbOfThreads = defaultParameters['NumberOfThreads']
        os.environ['OMP_NUM_THREADS'] = str(NbOfThreads)
        #vpm_cpp.mpi_init(defaultParameters['NumberOfThreads']);
        checkParametersTypes([defaultParameters, defaultVortexParameters], int_Params, float_Params,
                                                                                        bool_Params)
        defaultParameters['VelocityFreestream'] = np.array([0.]*3, dtype = float)
        defaultParameters['Sigma0'] = np.array(defaultParameters['Resolution']*\
                               defaultParameters['SmoothingRatio'], dtype = np.float64, order = 'F')
        defaultParameters['IterationCounter'] = np.array([0], dtype = np.int32, order = 'F')
        defaultParameters['StrengthRampAtbeginning'][0] = max(\
                                                    defaultParameters['StrengthRampAtbeginning'], 1)
        defaultParameters['MinimumVorticityFactor'][0] = max(0., \
                                                        defaultParameters['MinimumVorticityFactor'])
        t = buildEmptyVPMTree()
        if 'Length' in defaultVortexParameters:
            createLambOseenVortexBlob(t, defaultParameters, defaultVortexParameters)
        else:
            createLambOseenVortexRing(t, defaultParameters, defaultVortexParameters)
            if LeapFrog:
                Particles = pickParticlesZone(t)
                Np = Particles[1][0][0]
                extend(Particles, Np)
                ax, ay, az, a, s, v, c, nu = J.getVars(Particles, ['Alpha' + v for v in 'XYZ'] + \
                                          ['StrengthMagnitude', 'Sigma', 'Volume', 'Cvisq', 'Nu', ])
                ax[Np:], ay[Np:], az[Np:], a[Np:], s[Np:], v[Np:], c[Np:], nu[Np:] = ax[:Np], \
                                           ay[:Np], az[:Np], a[:Np], s[:Np], v[:Np], c[:Np], nu[:Np]
                x, y, z = J.getxyz(Particles)
                x[Np:], y[Np:], z[Np:] = x[:Np], y[:Np], z[:Np] + defaultVortexParameters['RingRadius']

        
        Particles = pickParticlesZone(t)
        J.set(Particles, '.VPM#Parameters', **defaultParameters)
        J.set(Particles, '.VortexRing#Parameters', **defaultVortexParameters)
        I._sortByName(I.getNodeFromName1(Particles, '.VPM#Parameters'))
        I._sortByName(I.getNodeFromName1(Particles, '.VortexRing#Parameters'))
        if defaultParameters['MonitorDiagnostics']:
            J.set(Particles, '.VPM#Diagnostics', Omega = [0., 0., 0.], LinearImpulse = [0., 0., 0.],
                                   AngularImpulse = [0., 0., 0.], Helicity = 0., KineticEnergy = 0.,
                                   KineticEnergyDivFree = 0., Enstrophy = 0., EnstrophyDivFree = 0.)

        solveVorticityEquation(t)

        IterationCounter = I.getNodeFromName(t, 'IterationCounter')
        IterationCounter[1][0] = defaultParameters['IterationTuningFMM']*\
                                                               defaultParameters['IntegrationOrder']

        C.convertPyTree2File(t, DIRECTORY_OUTPUT + '.cgns')
        compute(VPMParameters = {}, HybridParameters = {}, LiftingLineParameters = {},
            PolarsFilename = None, EulerianPath = None, LiftingLinePath = None,
            NumberOfIterations = NumberOfIterations, RestartPath = DIRECTORY_OUTPUT + '.cgns',
            DIRECTORY_OUTPUT = DIRECTORY_OUTPUT,
            VisualisationOptions = {'addLiftingLineSurfaces':False}, SaveVPMPeriod = SaveVPMPeriod,
            Verbose = True, SaveImageOptions={}, Surface = 0., FieldsExtractionGrid = [],
            SaveFieldsPeriod = np.inf, SaveImagePeriod = np.inf)

####################################################################################################
####################################################################################################
######################################### Coeff/Loads Aero #########################################
####################################################################################################
####################################################################################################
    def getAerodynamicCoefficientsOnLiftingLine(LiftingLines = [], StdDeviationSample = 50,
        IterationInfo = {}, Freestream = True, Wings = False, Surface = 0.):
        if LiftingLines:
            if Wings: IterationInfo = getAerodynamicCoefficientsOnWing(LiftingLines, Surface,
                      StdDeviationSample = StdDeviationSample, IterationInfo = IterationInfo)
            else:
                if Freestream: IterationInfo = getAerodynamicCoefficientsOnPropeller(LiftingLines,
                      StdDeviationSample = StdDeviationSample, IterationInfo = IterationInfo)
                else: IterationInfo = getAerodynamicCoefficientsOnRotor(LiftingLines,
                      StdDeviationSample = StdDeviationSample, IterationInfo = IterationInfo)
        return IterationInfo

    def getAerodynamicCoefficientsOnPropeller(LiftingLines = [], StdDeviationSample = 50,
        IterationInfo = {}):
        LiftingLine = I.getZones(LiftingLines)[0]
        RotationCenter = I.getValue(I.getNodeFromName(LiftingLine, 'RotationCenter'))
        RPM = I.getValue(I.getNodeFromName(LiftingLine, 'RPM'))
        Rho = I.getValue(I.getNodeFromName(LiftingLine, 'Density'))
        U0 = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityFreestream'))
        V = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityTranslation'))
        x, y, z = J.getxyz(LiftingLine)
        D = 2*max((x - RotationCenter[0])**2 + (y - RotationCenter[1])**2 + \
                                                                    (z - RotationCenter[2])**2)**0.5
        IntegralLoads = LL.computeGeneralLoadsOfLiftingLine(LiftingLines)
        n   = RPM/60.
        T = IntegralLoads['Total']['Thrust'][0]
        P = IntegralLoads['Total']['Power'][0]
        P = np.sign(P)*np.maximum(1e-12,np.abs(P))
        cT = T/(Rho*n**2*D**4)
        cP = P/(Rho*n**3*D**5)
        Uinf = np.linalg.norm(U0 - V)
        Eff = Uinf*T/P
        std_Thrust, std_Power = getStandardDeviationBlade(LiftingLines = LiftingLines,
                                                            StdDeviationSample = StdDeviationSample)
        IterationInfo['Thrust'] = T
        IterationInfo['Thrust Standard Deviation'] = std_Thrust/(T + np.sign(T)*1e-12)*100.
        IterationInfo['Power'] = P
        IterationInfo['Power Standard Deviation'] = std_Power/(P + np.sign(P)*1e-12)*100.
        IterationInfo['cT'] = cT
        IterationInfo['cP'] = cP
        IterationInfo['Eff'] = Eff
        return IterationInfo

    def getAerodynamicCoefficientsOnRotor(LiftingLines = [], StdDeviationSample = 50,
        IterationInfo = {}):
        LiftingLine = I.getZones(LiftingLines)[0]
        RotationCenter = I.getValue(I.getNodeFromName(LiftingLine, 'RotationCenter'))
        RPM = I.getValue(I.getNodeFromName(LiftingLine, 'RPM'))
        Rho = I.getValue(I.getNodeFromName(LiftingLine, 'Density'))
        x, y, z = J.getxyz(LiftingLine)
        D = 2*max((x - RotationCenter[0])**2 + (y - RotationCenter[1])**2 +\
                    (z - RotationCenter[2])**2)**0.5
        IntegralLoads = LL.computeGeneralLoadsOfLiftingLine(LiftingLines)
        n   = RPM/60.
        T = IntegralLoads['Total']['Thrust'][0]
        P = IntegralLoads['Total']['Power'][0]
        cT = T/(Rho*n**2*D**4)
        cP = P/(Rho*n**3*D**5)
        cP = np.sign(cP)*np.maximum(1e-12,np.abs(cP))
        Eff = np.sqrt(2./np.pi)*np.abs(cT)**1.5/cP

        std_Thrust, std_Power = getStandardDeviationBlade(LiftingLines = LiftingLines,
                                                            StdDeviationSample = StdDeviationSample)
        IterationInfo['Thrust'] = T
        IterationInfo['Thrust Standard Deviation'] = std_Thrust/(T + np.sign(T)*1e-12)*100.
        IterationInfo['Power'] = P
        IterationInfo['Power Standard Deviation'] = std_Power/(P + np.sign(P)*1e-12)*100.
        IterationInfo['cT'] = cT
        IterationInfo['cP'] = cP
        IterationInfo['Eff'] = Eff
        return IterationInfo

    def getAerodynamicCoefficientsOnWing(LiftingLines = [], Surface = 0., StdDeviationSample = 50,
        IterationInfo = {}):
        Fx, Fz, cL, cD = 0., 0., 0., 0.
        for LiftingLine in LiftingLines:
            Rho = I.getValue(I.getNodeFromName(LiftingLine, 'Density'))
            U0 = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityFreestream'))
            V = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityTranslation'))
            IntegralLoads = LL.computeGeneralLoadsOfLiftingLine(LiftingLine)
            Fz += IntegralLoads['ForceZ'][0]
            Fx += IntegralLoads['ForceX'][0]
            q0 = 0.5*Rho*Surface*np.linalg.norm(U0 - V)**2
            cL += IntegralLoads['ForceZ'][0]/(q0 + 1e-12)
            cD += IntegralLoads['ForceX'][0]/(q0 + 1e-12)

        std_Thrust, std_Drag = getStandardDeviationWing(LiftingLines = LiftingLines,
                                                            StdDeviationSample = StdDeviationSample)

        IterationInfo['Lift'] = Fz
        IterationInfo['Lift Standard Deviation'] = std_Thrust/(Fz + np.sign(Fz)*1e-12)*100.
        IterationInfo['Drag'] = Fx
        IterationInfo['Drag Standard Deviation'] = std_Drag/(Fx + np.sign(Fx)*1e-12)*100.
        IterationInfo['cL'] = cL
        IterationInfo['cD'] = cD
        IterationInfo['f'] = Fz/Fx
        return IterationInfo

    def getStandardDeviationWing(LiftingLines = [], StdDeviationSample = 50):
        LiftingLine = I.getZones(LiftingLines)[0]
        UnsteadyLoads = I.getNodeFromName(LiftingLine, '.UnsteadyLoads')
        Thrust = I.getValue(I.getNodeFromName(UnsteadyLoads, 'Thrust'))
        if type(Thrust) == np.ndarray or type(Thrust) == list:
            StdDeviationSample = max(min(StdDeviationSample,len(Thrust)), 1)
        else: return 0., 0.

        Thrust = np.array([0.]*StdDeviationSample)
        Drag = np.array([0.]*StdDeviationSample)
        for LiftingLine in LiftingLines:
            UnsteadyLoads = I.getNodeFromName(LiftingLine, '.UnsteadyLoads')
            Thrust += I.getNodeFromName(UnsteadyLoads, 'ForceZ')[1][-StdDeviationSample:]
            Drag += I.getNodeFromName(UnsteadyLoads, 'ForceX')[1][-StdDeviationSample:]
        meanThrust = sum(Thrust)/StdDeviationSample
        meanDrag = sum(Drag)/StdDeviationSample

        std_Thrust = (sum((Thrust - meanThrust)**2)/StdDeviationSample)**0.5
        std_Drag = (sum((Drag - meanDrag)**2)/StdDeviationSample)**0.5
        return std_Thrust, std_Drag

    def getStandardDeviationBlade(LiftingLines = [], StdDeviationSample = 50):
        LiftingLine = I.getZones(LiftingLines)[0]
        UnsteadyLoads = I.getNodeFromName(LiftingLine, '.UnsteadyLoads')
        Thrust = I.getValue(I.getNodeFromName(UnsteadyLoads, 'Thrust'))
        if type(Thrust) == np.ndarray or type(Thrust) == list:
            StdDeviationSample = max(min(StdDeviationSample,len(Thrust)), 1)
        else: return 0., 0.

        Thrust = np.array([0.]*StdDeviationSample)
        Power = np.array([0.]*StdDeviationSample)
        for LiftingLine in LiftingLines:
            UnsteadyLoads = I.getNodeFromName(LiftingLine, '.UnsteadyLoads')
            Thrust += I.getNodeFromName(UnsteadyLoads, 'Thrust')[1][-StdDeviationSample:]
            Power += I.getNodeFromName(UnsteadyLoads, 'Power')[1][-StdDeviationSample:]
        meanThrust = sum(Thrust)/StdDeviationSample
        meanPower = sum(Power)/StdDeviationSample

        stdThrust = (sum((Thrust - meanThrust)**2)/StdDeviationSample)**0.5
        stdPower = (sum((Power - meanPower)**2)/StdDeviationSample)**0.5
        return stdThrust, stdPower

####################################################################################################
####################################################################################################
######################################### IO/Visualisation #########################################
####################################################################################################
####################################################################################################
    def setVisualization(t = [], ParticlesColorField = 'VorticityMagnitude',
        ParticlesRadius = '{Sigma}/8', addLiftingLineSurfaces = True, AirfoilPolarsFilename = None):
        Particles = pickParticlesZone(t)
        Sigma = I.getValue(I.getNodeFromName(Particles, 'Sigma'))
        C._initVars(Particles, 'radius=' + ParticlesRadius)
        if not ParticlesColorField: ParticlesColorField = 'VorticityMagnitude'

        CPlot._addRender2Zone(Particles, material = 'Sphere',
                 color = 'Iso:' + ParticlesColorField, blending = 0.6, shaderParameters = [0.04, 0])
        LiftingLines = LL.getLiftingLines(t)
        for zone in LiftingLines:
            CPlot._addRender2Zone(zone, material = 'Flat', color = 'White')

        if addLiftingLineSurfaces:
            if not AirfoilPolarsFilename:
                ERRMSG = J.FAIL + ('production of surfaces from lifting-line requires'
                    ' attribute AirfoilPolars') + J.ENDC
                raise AttributeError(ERRMSG)
            LiftingLineSurfaces = []
            for ll in LiftingLines:
                surface = LL.postLiftingLine2Surface(ll, AirfoilPolarsFilename)
                deletePrintedLines()
                surface[0] = ll[0] + '.surf'
                CPlot._addRender2Zone(surface, material = 'Solid', color = '#ECF8AB')
                LiftingLineSurfaces += [surface]
            I.createUniqueChild(t, 'LiftingLineSurfaces', 'CGNSBase_t',
                value = np.array([2, 3], order = 'F'), children = LiftingLineSurfaces)

        for zone in I.getZones(pickHybridDomain(t)):
            CPlot._addRender2Zone(zone, material = 'Glass', color = 'White', blending = 0.6)
        CPlot._addRender2PyTree(t, mode = 'Render', colormap = 'Blue2Red', isoLegend=1,
                                                                  scalarField = ParticlesColorField)


    def saveImage(t = [], ShowInScreen = False, ImagesDirectory = 'FRAMES', **DisplayOptions):
        if 'mode' not in DisplayOptions: DisplayOptions['mode'] = 'Render'
        if 'displayInfo' not in DisplayOptions: DisplayOptions['displayInfo'] = 0
        if 'colormap' not in DisplayOptions: DisplayOptions['colormap'] = 0
        if 'win' not in DisplayOptions: DisplayOptions['win'] = (700, 700)
        DisplayOptions['exportResolution'] = '%gx%g'%DisplayOptions['win']

        try: os.makedirs(ImagesDirectory)
        except: pass

        sp = getVPMParameters(t)

        DisplayOptions['export'] = os.path.join(ImagesDirectory,
            'frame%05d.png'%sp['CurrentIteration'])

        if ShowInScreen:
            DisplayOptions['offscreen'] = 0
        else:
            DisplayOptions['offscreen'] = 1

        if 'backgroundFile' not in DisplayOptions:
            MOLA = os.getenv('MOLA')
            MOLASATOR = os.getenv('MOLASATOR')
            for MOLAloc in [MOLA, MOLASATOR]:
                backgroundFile = os.path.join(MOLAloc, 'MOLA', 'GUIs', 'background.png')
                if os.path.exists(backgroundFile):
                    DisplayOptions['backgroundFile']=backgroundFile
                    DisplayOptions['bgColor']=13
                    break

        CPlot.display(t, **DisplayOptions)
        if DisplayOptions['offscreen']:
            CPlot.finalizeExport(DisplayOptions['offscreen'])



    def open(filename = ''):
        t = C.convertFile2PyTree(filename)
        Particles = pickParticlesZone(t)
        if Particles:
            FieldNames = ['gradxVelocity' + v for v in 'XYZ'] + \
                         ['gradyVelocity' + v for v in 'XYZ'] + \
                         ['gradzVelocity' + v for v in 'XYZ'] + \
                         ['PSE' + v for v in 'XYZ'] + ['Stretching' + v for v in 'XYZ'] + \
                         ['VelocityDiffusion' + v for v in 'XYZ'] + ['Active', 'Age', \
                          'HybridFlag', 'Nu', 'divUd', 'Enstrophy', 'Enstrophyf', 'KineticEnergy', \
                          'KineticEnergyf', 'SFS', 'KineticEnergyVariation', \
                          'EnstrophyVariation'] + ['VelocityPerturbation' + v for v in 'XYZ']
            J.invokeFieldsDict(Particles, FieldNames)
            solveVorticityEquation(t)
            HybridParameters = I.getNodeFromName(t, '.Hybrid#Parameters')
            if HybridParameters:
                Nbem = getParameter(Particles, 'NumberOfBEMSources')[0]
                HybridParameters[2] += [['BEMMatrix', np.array([0.]*9*Nbem*Nbem, dtype = np.float64,
                                                                   order = 'F'), [], 'DataArray_t']]
                updateBEMMatrix(t)

        deletePrintedLines()
        return t

    def save(t = [], filename = '', VisualisationOptions = {}):
        tref = I.copyRef(t)
        rmNodesFlowSolution = ['gradxVelocity' + v for v in 'XYZ'] + \
                     ['gradyVelocity' + v for v in 'XYZ'] + ['gradzVelocity' + v for v in 'XYZ'] + \
                     ['PSE' + v for v in 'XYZ'] + ['Stretching' + v for v in 'XYZ'] + \
                     ['VelocityDiffusion' + v for v in 'XYZ'] + ['Active', 'Age', 'HybridFlag', \
                      'Nu', 'divUd', 'Enstrophy', 'Enstrophyf', 'KineticEnergy',\
                      'KineticEnergyf', 'SFS', 'KineticEnergyVariation', \
                      'EnstrophyVariation'] + ['VelocityPerturbation' + v for v in 'XYZ']
        Particles = pickParticlesZone(tref)
        if Particles:
            I._rmNodesByName(Particles, 'BEMMatrix')
            FlowSolution = I.getNodeFromName(Particles, 'FlowSolution')
            for node in rmNodesFlowSolution: I._rmNodesByName(FlowSolution, node)
            # an, nx, ny, nz, t1x, t1y, t1z, t2x, t2y, t2z = J.invokeFields(Particles, ['AlphaN'] + ['Normal' + v for v in 'XYZ'] + ['Tangent1' + v for v in 'XYZ'] + ['Tangent2' + v for v in 'XYZ'])
            # acfdn = np.append(I.getNodeFromName(Particles, 'AlphaBEMN')[1], \
            #                                          I.getNodeFromName(Particles, 'AlphaCFDN')[1])
            # an[:len(acfdn)] = acfdn
            # s = np.linalg.norm(np.vstack([I.getNodeFromName(Particles, 'SurfaceCFDX')[1], I.getNodeFromName(Particles, 'SurfaceCFDY')[1], I.getNodeFromName(Particles, 'SurfaceCFDZ')[1]]), axis = 0)
            # ncfdx = np.append(I.getNodeFromName(Particles, 'NormalBEMX')[1], \
            #                                          I.getNodeFromName(Particles, 'SurfaceCFDX')[1]/s)
            # ncfdy = np.append(I.getNodeFromName(Particles, 'NormalBEMY')[1], \
            #                                          I.getNodeFromName(Particles, 'SurfaceCFDY')[1]/s)
            # ncfdz = np.append(I.getNodeFromName(Particles, 'NormalBEMZ')[1], \
            #                                          I.getNodeFromName(Particles, 'SurfaceCFDZ')[1]/s)
            # nx[:len(ncfdx)] = ncfdx
            # ny[:len(ncfdy)] = ncfdy
            # nz[:len(ncfdz)] = ncfdz
            # t1cfdx = I.getNodeFromName(Particles, 'Tangential1BEMX')[1]
            # t1cfdy = I.getNodeFromName(Particles, 'Tangential1BEMY')[1]
            # t1cfdz = I.getNodeFromName(Particles, 'Tangential1BEMZ')[1]
            # t1x[:len(t1cfdx)] = t1cfdx
            # t1y[:len(t1cfdy)] = t1cfdy
            # t1z[:len(t1cfdz)] = t1cfdz
            # t2cfdx = I.getNodeFromName(Particles, 'Tangential2BEMX')[1]
            # t2cfdy = I.getNodeFromName(Particles, 'Tangential2BEMY')[1]
            # t2cfdz = I.getNodeFromName(Particles, 'Tangential2BEMZ')[1]
            # t2x[:len(t2cfdx)] = t2cfdx
            # t2y[:len(t2cfdy)] = t2cfdy
            # t2z[:len(t2cfdz)] = t2cfdz
        try:
            if os.path.islink(filename):
                os.unlink(filename)
            else:
                os.remove(filename)
        except:
            pass

        if VisualisationOptions: setVisualization(tref, **VisualisationOptions)

        C.convertPyTree2File(tref, filename)
        deletePrintedLines()

    def loadAirfoilPolars(filename = ''): return LL.loadPolarsInterpolatorDict(filename)

    def printIterationInfo(IterationInfo = {}, PSE = False, DVM = False, Wings = False):
        msg = '||' + '{:-^50}'.format(' Iteration ' + '{:d}'.format(IterationInfo['Iteration']) + \
                        ' (' + '{:.1f}'.format(IterationInfo['Percentage']) + '%) ') + '\n'
        msg += '||' + '{:34}'.format('Physical time') + \
                        ': ' + '{:.5f}'.format(IterationInfo['Physical time']) + ' s' + '\n'
        msg += '||' + '{:34}'.format('Number of particles') + \
                        ': ' + '{:d}'.format(IterationInfo['Number of particles']) + '\n'
        msg += '||' + '{:34}'.format('Total iteration time') + \
                        ': ' + '{:.2f}'.format(IterationInfo['Total iteration time']) + ' s' + '\n'
        msg += '||' + '{:34}'.format('Total simulation time') + \
                        ': ' + '{:.1f}'.format(IterationInfo['Total simulation time']) + ' s' + '\n'
        msg += '||' + '{:-^50}'.format(' Loads ') + '\n'
        if (Wings and 'Lift' in IterationInfo) or (not Wings and 'Thrust' in IterationInfo):
            if (Wings):
                msg += '||' + '{:34}'.format('Lift') + \
                      ': ' + '{:.4g}'.format(IterationInfo['Lift']) + ' N' + '\n'
                msg += '||' + '{:34}'.format('Lift Standard Deviation') + \
                      ': ' + '{:.2f}'.format(IterationInfo['Lift Standard Deviation']) + ' %' + '\n'
                msg += '||' + '{:34}'.format('Drag') + \
                      ': ' + '{:.4g}'.format(IterationInfo['Drag']) + ' N' + '\n'
                msg += '||' + '{:34}'.format('Drag Standard Deviation') + \
                      ': ' + '{:.2f}'.format(IterationInfo['Drag Standard Deviation']) + ' %' + '\n'
                msg += '||' + '{:34}'.format('cL') + \
                      ': ' + '{:.4f}'.format(IterationInfo['cL']) + '\n'
                msg += '||' + '{:34}'.format('cD') + \
                      ': ' + '{:.5f}'.format(IterationInfo['cD']) + '\n'
                msg += '||' + '{:34}'.format('f') + \
                      ': ' + '{:.4f}'.format(IterationInfo['f']) + '\n'
            else:
                msg += '||' + '{:34}'.format('Thrust') + \
                    ': ' + '{:.5g}'.format(IterationInfo['Thrust']) + ' N' + '\n'
                msg += '||' + '{:34}'.format('Thrust Standard Deviation') + \
                    ': ' + '{:.2f}'.format(IterationInfo['Thrust Standard Deviation']) + ' %' + '\n'
                msg += '||' + '{:34}'.format('Power') + \
                    ': ' + '{:.3g}'.format(IterationInfo['Power']) + ' W' + '\n'
                msg += '||' + '{:34}'.format('Power Standard Deviation') + \
                    ': ' + '{:.2f}'.format(IterationInfo['Power Standard Deviation']) + ' %' + '\n'
                msg += '||' + '{:34}'.format('cT') + \
                    ': ' + '{:.5f}'.format(IterationInfo['cT']) + '\n'
                msg += '||' + '{:34}'.format('Cp') + \
                    ': ' + '{:.5f}'.format(IterationInfo['cP']) + '\n'
                msg += '||' + '{:34}'.format('Eff') + \
                    ': ' + '{:.5f}'.format(IterationInfo['Eff']) + '\n'
        msg += '||' + '{:-^50}'.format(' Population Control ') + '\n'
        msg += '||' + '{:34}'.format('Number of particles beyond cutoff') + \
                     ': ' + '{:d}'.format(IterationInfo['Number of particles beyond cutoff']) + '\n'
        msg += '||' + '{:34}'.format('Number of split particles') + \
                     ': ' + '{:d}'.format(IterationInfo['Number of split particles']) + '\n'
        msg += '||' + '{:34}'.format('Number of depleted particles') + \
                     ': ' + '{:d}'.format(IterationInfo['Number of depleted particles']) + '\n'
        msg += '||' + '{:34}'.format('Number of merged particles') + \
                     ': ' + '{:d}'.format(IterationInfo['Number of merged particles']) + '\n'
        msg += '||' + '{:34}'.format('Control Computation time') + \
                     ': ' + '{:.2f}'.format(IterationInfo['Population Control time']) + ' s (' + \
                                          '{:.1f}'.format(IterationInfo['Population Control time']/\
                                          IterationInfo['Total iteration time']*100.) + '%) ' + '\n'
        if 'Lifting Line time' in IterationInfo:
            msg += '||' + '{:-^50}'.format(' Lifting Line ') + '\n'
            msg += '||' + '{:34}'.format('Circulation error') + \
                         ': ' + '{:.5e}'.format(IterationInfo['Circulation error']) + '\n'
            msg += '||' + '{:34}'.format('Number of sub-iterations') + \
                         ': ' + '{:d}'.format(IterationInfo['Number of sub-iterations (LL)']) + '\n'
            msg += '||' + '{:34}'.format('Number of shed particles') + \
                         ': ' + '{:d}'.format(IterationInfo['Number of shed particles']) + '\n'
            msg += '||' + '{:34}'.format('Lifting Line Computation time') + \
                         ': ' + '{:.2f}'.format(IterationInfo['Lifting Line time']) + ' s (' + \
                                                '{:.1f}'.format(IterationInfo['Lifting Line time']/\
                                          IterationInfo['Total iteration time']*100.) + '%) ' + '\n'

        if 'Hybrid Computation time' in IterationInfo:
            msg += '||' + '{:-^50}'.format(' Hybrid Solver ') + '\n'
            msg += '||' + '{:34}'.format('Eulerian Vorticity lost') + \
                          ': ' + '{:.1g}'.format(IterationInfo['Eulerian Vorticity lost']) + \
                          ' s-1 (' + '{:.1f}'.format(IterationInfo['Eulerian Vorticity lost per'])+\
                                                                                        '%) ' + '\n'
            msg += '||' + '{:34}'.format('Minimum Eulerian Vorticity') + \
                          ': ' + '{:.2g}'.format(IterationInfo['Minimum Eulerian Vorticity']) + '\n'
            msg += '||' + '{:34}'.format('Number of Hybrids Generated') + \
                          ': ' + '{:d}'.format(IterationInfo['Number of Hybrids Generated']) + '\n'
            msg += '||' + '{:34}'.format('Hybrid Computation time') + \
                          ': ' + '{:.2f}'.format(IterationInfo['Hybrid Computation time']) + \
                               ' s (' + '{:.1f}'.format(IterationInfo['Hybrid Computation time']/\
                                          IterationInfo['Total iteration time']*100.) + '%) ' + '\n'
            
        msg += '||' + '{:-^50}'.format(' FMM ') + '\n'
        msg += '||' + '{:34}'.format('Number of threads') + \
                        ': ' + '{:d}'.format(IterationInfo['Number of threads']) + '\n'
        msg += '||' + '{:34}'.format('SIMD vectorisation') + \
                        ': ' + '{:d}'.format(IterationInfo['SIMD vectorisation']) + '\n'
        msg += '||' + '{:34}'.format('Near field overlapping ratio') + \
                        ': ' + '{:.2f}'.format(IterationInfo['Near field overlapping ratio']) + '\n'
        msg += '||' + '{:34}'.format('Far field polynomial order') + \
                        ': ' + '{:d}'.format(IterationInfo['Far field polynomial order']) + '\n'
        msg += '||' + '{:34}'.format('Rel. err. of Velocity') + \
                        ': ' + '{:e}'.format(IterationInfo['Rel. err. of Velocity']) + '\n'
        msg += '||' + '{:34}'.format('Rel. err. of Velocity Gradient') + \
                        ': ' + '{:e}'.format(IterationInfo['Rel. err. of Velocity']) + '\n'
        msg += '||' + '{:34}'.format('Rel. err. of Vorticity') + \
                        ': ' + '{:e}'.format(IterationInfo['Rel. err. of Velocity Gradient']) + '\n'
        if PSE: msg += '||' + '{:34}'.format('Rel. err. of PSE') + \
                        ': ' + '{:e}'.format(IterationInfo['Rel. err. of PSE']) + '\n'
        if DVM:
            msg += '||' + '{:34}'.format('Rel. err. of PSE') + \
                        ': ' + '{:e}'.format(IterationInfo['Rel. err. of PSE']) + '\n'
            msg += '||' + '{:34}'.format('Rel. err. of Diffusion Velocity') + \
                        ': ' + '{:e}'.format(IterationInfo['Rel. err. of Diffusion Velocity']) +'\n'
        msg += '||' + '{:34}'.format('FMM Computation time') + \
                        ': ' + '{:.2f}'.format(IterationInfo['FMM time']) + ' s (' + \
                                                         '{:.1f}'.format(IterationInfo['FMM time']/\
                                          IterationInfo['Total iteration time']*100.) + '%) ' + '\n'
        if "Perturbation time" in IterationInfo:
            msg += '||' + '{:-^50}'.format(' Perturbation Field ') + '\n'
            msg += '||' + '{:34}'.format('Interpolation time') + \
                        ': ' + '{:.2f}'.format(IterationInfo['Perturbation time']) + ' s (' + \
                                                '{:.1f}'.format(IterationInfo['Perturbation time']/\
                                          IterationInfo['Total iteration time']*100.) + '%) ' + '\n'
        msg += '||' + '{:=^50}'.format('')
        print(msg)

    def deletePrintedLines(NumberOfLineToDelete = 1):
        for i in range(NumberOfLineToDelete):
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')

####################################################################################################
####################################################################################################
############################################## Solver ##############################################
####################################################################################################
####################################################################################################
    def compute(VPMParameters = {}, HybridParameters = {}, LiftingLineParameters = {},
        PerturbationFieldParameters = {}, PolarsFilename = None, EulerianPath = None,
        PerturbationFieldPath = None, LiftingLinePath = None, NumberOfIterations = 1000,
        RestartPath = None, DIRECTORY_OUTPUT = 'OUTPUT',
        VisualisationOptions = {'addLiftingLineSurfaces':True}, StdDeviationSample = 50,
        SaveVPMPeriod = 100, Verbose = True, SaveImageOptions={}, Surface = 0.,
        FieldsExtractionGrid = [], SaveFieldsPeriod = np.inf, SaveImagePeriod = np.inf):
        try: os.makedirs(DIRECTORY_OUTPUT)
        except: pass

        if PolarsFilename: AirfoilPolars = loadAirfoilPolars(PolarsFilename)
        else: AirfoilPolars = None

        if RestartPath:
            t = open(RestartPath)
            if PerturbationFieldPath:
                PerturbationField = open(PerturbationFieldPath)
                if VPMParameters['NumberOfThreads'] == 'auto':
                    NbOfThreads = int(os.getenv('OMP_NUM_THREADS',len(os.sched_getaffinity(0))))
                    VPMParameters['NumberOfThreads'] = NbOfThreads
                else: NbOfThreads = VPMParameters['NumberOfThreads']
                os.environ['OMP_NUM_THREADS'] = str(NbOfThreads)
                PerturbationFieldCapsule = vpm_cpp.build_perturbation_velocity_capsule(PerturbationField, NbOfThreads)
            else: PerturbationFieldCapsule = []

            if EulerianPath:
                try:
                    tE = open(EulerianPath)
                    tE = generateMirrorWing(tE, getVPMParameters(t), getHybridParameters(t))
                except: tE = []
            else: tE = []
        else:
            if LiftingLinePath: LiftingLine = open(LiftingLinePath) # LB: TODO dangerous; rather use os.path.isfile()
            else: LiftingLine = []
            #if EulerianPath: EulerianMesh = open(EulerianPath)
            #else: EulerianMesh = []
            t, tE, PerturbationFieldCapsule = initialiseVPM(EulerianMesh = EulerianPath,
                PerturbationField = PerturbationFieldPath, HybridParameters = HybridParameters,
                LiftingLineTree = LiftingLine, LiftingLineParameters = LiftingLineParameters,
                PerturbationFieldParameters = PerturbationFieldParameters, PolarInterpolator = \
                                                       AirfoilPolars, VPMParameters = VPMParameters)

        
        IterationInfo = {'Rel. err. of Velocity': 0, 'Rel. err. of Velocity Gradient': 0,
                                        'Rel. err. of PSE': 0, 'Rel. err. of Diffusion Velocity': 0}
        TotalTime = J.tic()
        sp = getVPMParameters(t)
        Np = pickParticlesZone(t)[1][0]
        LiftingLines = LL.getLiftingLines(t)

        it = sp['CurrentIteration']
        simuTime = sp['Time']
        PSE = DiffusionScheme_str2int[sp['DiffusionScheme']] < 2
        DVM = DiffusionScheme_str2int[sp['DiffusionScheme']] == 2
        Freestream = (np.linalg.norm(sp['VelocityFreestream']) != 0.)
        try:
            Wing = (I.getValue(I.getNodeFromName(LiftingLines, 'RPM')) == 0)
        except:
            Wing = True
        if AirfoilPolars: VisualisationOptions['AirfoilPolarsFilename'] = PolarsFilename
        else: VisualisationOptions['addLiftingLineSurfaces'] = False

        filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_It%d.cgns'%it[0])
        save(t, filename, VisualisationOptions)
        J.createSymbolicLink(filename,  DIRECTORY_OUTPUT + '.cgns')
        for _ in range(3): print('||' + '{:=^50}'.format(''))
        print('||' + '{:=^50}'.format(' Begin VPM Computation '))
        for _ in range(3): print('||' + '{:=^50}'.format(''))
        while it[0] < NumberOfIterations:
            SAVE_ALL = J.getSignal('SAVE_ALL')
            SAVE_FIELDS = ((it + 1)%SaveFieldsPeriod==0 and it>0) or J.getSignal('SAVE_FIELDS')
            SAVE_IMAGE = ((it + 1)%SaveImagePeriod==0 and it>0) or J.getSignal('SAVE_IMAGE')
            SAVE_VPM = ((it + 1)%SaveVPMPeriod==0) or J.getSignal('SAVE_VPM')
            CONVERGED = J.getSignal('CONVERGED')
            if CONVERGED: SAVE_ALL = True

            if SAVE_ALL:
                SAVE_FIELDS = SAVE_VPM = True
            
            IterationTime = J.tic()
            computeNextTimeStep(t, PerturbationFieldCapsule = PerturbationFieldCapsule)
            IterationInfo['Iteration'] = it[0]
            IterationInfo['Percentage'] = it[0]/NumberOfIterations*100.
            IterationInfo['Physical time'] = simuTime[0]
            IterationInfo = updateHybridDomainAndSources(t, tE, IterationInfo)
            IterationInfo = populationControl(t, [], IterationInfo)
            IterationInfo = ShedVorticitySourcesFromLiftingLines(t, AirfoilPolars, IterationInfo,
                                                PerturbationFieldCapsule = PerturbationFieldCapsule)
            IterationInfo['Number of particles'] = Np[0]
            IterationInfo = solveVorticityEquation(t, IterationInfo = IterationInfo,
                                                PerturbationFieldCapsule = PerturbationFieldCapsule)
            IterationInfo['Total iteration time'] = J.tic() - IterationTime
            IterationInfo = getAerodynamicCoefficientsOnLiftingLine(LiftingLines, Wings = Wing,
                                   StdDeviationSample = StdDeviationSample, Freestream = Freestream, 
                                                   IterationInfo = IterationInfo, Surface = Surface)
            IterationInfo['Total simulation time'] = J.tic() - TotalTime
            if Verbose: printIterationInfo(IterationInfo, PSE = PSE, DVM = DVM, Wings = Wing)

            if (SAVE_FIELDS or SAVE_ALL) and FieldsExtractionGrid:
                extract(t, FieldsExtractionGrid, 5000)
                filename = os.path.join(DIRECTORY_OUTPUT, 'fields_It%d.cgns'%it)
                save(FieldsExtractionGrid, filename)
                J.createSymbolicLink(filename, 'fields.cgns')

            if SAVE_IMAGE or SAVE_ALL:
                setVisualization(t, **VisualisationOptions)
                saveImage(t, **SaveImageOptions)    

            if SAVE_VPM or SAVE_ALL:
                filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_It%d.cgns'%it)
                save(t, filename, VisualisationOptions)
                J.createSymbolicLink(filename,  DIRECTORY_OUTPUT + '.cgns')

            if CONVERGED: break
            
        filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_It%d.cgns'%it)
        save(t, filename, VisualisationOptions)
        save(t, DIRECTORY_OUTPUT + '.cgns', VisualisationOptions)
        for _ in range(3): print('||' + '{:=^50}'.format(''))
        print('||' + '{:-^50}'.format(' End of VPM computation '))
        for _ in range(3): print('||' + '{:=^50}'.format(''))

        return t

    def computePolar(VPMParameters = {}, LiftingLineParameters = {}, PolarData = {},
        PolarsFilename = '', LiftingLinePath = 'LiftingLine.cgns', DIRECTORY_OUTPUT = 'POLARS',
        RestartPath = None, MaxNumberOfIterationsPerPolar = 200, MinNumberOfIterationsPerPolar = 0,
        NumberOfIterationsForTransition = 0, StdDeviationSample = 10, Surface = 1., Verbose = True,
        MaxThrustStandardDeviation = 1, MaxPowerStandardDeviation = 100,
        VisualisationOptions = {'addLiftingLineSurfaces':True}):
        try: os.makedirs(DIRECTORY_OUTPUT)
        except: pass

        AirfoilPolars = loadAirfoilPolars(PolarsFilename)
        if RestartPath:
            PolarsTree = open(RestartPath)
            Polars = I.getNodeFromName(PolarsTree, 'Polars')[2]
            OldPolarData = {}
            for z in Polars:
                if type(z[1][0]) == np.bytes_:
                    OldPolarData[z[0]] = ''
                    for zi in z[1]: OldPolarData[z[0]] += zi.decode('UTF-8')
                else: OldPolarData[z[0]] = z[1]
            
            for v in OldPolarData['Variables']:
                i = 0
                while i < len(PolarData['Variables']):
                    if PolarData['Variables'][i] == v:
                        PolarData['Variables'] = np.delete(PolarData['Variables'], i)
                        i -= 1
                    i += 1
            PolarData['Variables'] = np.append(OldPolarData['Variables'], PolarData['Variables'])
            OldPolarData.update(PolarData)
            PolarData = OldPolarData
            if PolarData['VariableName'] == 'Pitch':
                PolarData['VariableName'] = 'Twist'
                PolarData['Pitch'] = True

            N0 = 0
            while (N0 < len(OldPolarData['Variables']) and
                                      OldPolarData['Variables'][N0] != OldPolarData['OldVariable']):
                N0 += 1

            if N0 == len(OldPolarData['Variables']): N0 = 0
            else: N0 += 1

            t = open(os.path.join(DIRECTORY_OUTPUT, PolarData['LastPolar']))

        else:
            LiftingLine = open(LiftingLinePath)
            t, tE = initialiseVPM(LiftingLineTree = LiftingLine, VPMParameters = VPMParameters,
                   LiftingLineParameters = LiftingLineParameters, PolarInterpolator = AirfoilPolars)
            if PolarData['VariableName'] == 'Pitch':
                PolarData['VariableName'] = 'Twist'
                PolarData['Pitch'] = True

            PolarData['Offset'] = I.getValue(I.getNodeFromName(t, PolarData['VariableName']))
            if PolarData['overwriteVPMWithVariables']: PolarData['Offset'] *= 0.

            if not (isinstance(PolarData['Offset'], list) or \
                                                       isinstance(PolarData['Offset'], np.ndarray)):
                PolarData['Offset'] = np.array([PolarData['Offset']])

            PolarData['OldVariable'] = PolarData['Variables'][0]
            if len(I.getZones(LiftingLine)) == 1:
                PolarData['Lift'] = np.array([], dtype = np.float64)
                PolarData['Drag'] = np.array([], dtype = np.float64)
                PolarData['cL'] = np.array([], dtype = np.float64)
                PolarData['cD'] = np.array([], dtype = np.float64)
                PolarData['f'] = np.array([], dtype = np.float64)
                PolarData['LiftStandardDeviation'] = np.array([], dtype = np.float64)
                PolarData['DragStandardDeviation'] = np.array([], dtype = np.float64)
            else:
                PolarData['Thrust'] = np.array([], dtype = np.float64)
                PolarData['Power'] = np.array([], dtype = np.float64)
                PolarData['cT'] = np.array([], dtype = np.float64)
                PolarData['cP'] = np.array([], dtype = np.float64)
                PolarData['Efficiency'] = np.array([], dtype = np.float64)
                PolarData['ThrustStandardDeviation'] = np.array([], dtype = np.float64)
                PolarData['PowerStandardDeviation'] = np.array([], dtype = np.float64)

            PolarsTree = C.newPyTree()
            J.set(PolarsTree, 'Polars', **PolarData)
            N0 = 0

        sp = getVPMParameters(t)
        Particles = pickParticlesZone(t)
        Np = Particles[1][0]
        LiftingLines = LL.getLiftingLines(t)

        f = sp['RedistributionPeriod'][0]
        it = sp['CurrentIteration']
        dt = sp['TimeStep']
        U0 = np.linalg.norm(sp['VelocityFreestream'])
        simuTime = sp['Time']
        NbLL = len(I.getZones(LiftingLines))
        VisualisationOptions['AirfoilPolarsFilename'] = PolarsFilename

        for _ in range(3): print('||' + '{:=^50}'.format(''))
        print('||' + '{:=^50}'.format(' Begin VPM Polar '))
        for _ in range(3): print('||' + '{:=^50}'.format(''))

        TotalTime = J.tic()
        for i, Variable in enumerate(PolarData['Variables'][N0:]):
            i += N0
            if NumberOfIterationsForTransition or N0 == i:
                for n in range(1, NumberOfIterationsForTransition + 1):
                    computeNextTimeStep(t)

                    NewVariable = PolarData['OldVariable'] + (Variable - PolarData['OldVariable'])\
                                                                  *n/NumberOfIterationsForTransition
                    for LiftingLine in LiftingLines:
                        LLVariable = I.getNodeFromName(LiftingLine, PolarData['VariableName'])
                        LLVariable[1] = PolarData['Offset'] + NewVariable

                    VPMVariable = I.getNodeFromName(Particles, PolarData['VariableName'])
                    if VPMVariable != None: VPMVariable[1] = PolarData['Offset'] + NewVariable
                    if 'TimeStepFunction' in PolarData:
                        TimeStepFunction_str2int[PolarData['TimeStepFunction']](t, LiftingLines
                                                           , PolarData['TimeStepFunctionParameter'])

                    populationControl(t, NoRedistributionRegions=[])
                    ShedVorticitySourcesFromLiftingLines(t, AirfoilPolars)
                    solveVorticityEquation(t)

                    if Verbose:
                        if n != 1: deletePrintedLines()
                        print('||' + '{:-^50}'.format(' Transition ' + '{:.1f}'.format(n/\
                                              NumberOfIterationsForTransition*100.) + '% ') + ' \r')

            for LiftingLine in LiftingLines:
                LLVariable = I.getNodeFromName(LiftingLine, PolarData['VariableName'])
                LLVariable[1] = PolarData['Offset'] + Variable
            VPMVariable = I.getNodeFromName(Particles, PolarData['VariableName'])
            if VPMVariable != None: VPMVariable[1] = PolarData['Offset'] + Variable
            if 'TimeStepFunction' in PolarData:
                TimeStepFunction_str2int[PolarData['TimeStepFunction']](t, LiftingLines, 
                                                             PolarData['TimeStepFunctionParameter'])
            it0 = it[0]
            stdThrust = MaxThrustStandardDeviation + 1
            stdPower = MaxPowerStandardDeviation + 1

            while (it[0] - it0 < MaxNumberOfIterationsPerPolar and (MaxThrustStandardDeviation < \
                stdThrust or MaxPowerStandardDeviation < stdPower)) or (it[0] - it0 < \
                               MinNumberOfIterationsPerPolar) or (it[0] - it0 < StdDeviationSample):
                msg = '||' + '{:-^50}'.format(' Iteration ' + '{:d}'.format(it[0] - it0) + ' ')+'\n'
                computeNextTimeStep(t)
                populationControl(t, NoRedistributionRegions=[])
                IterationInfo = ShedVorticitySourcesFromLiftingLines(t, AirfoilPolars)
                solveVorticityEquation(t)

                msg += '||' + '{:34}'.format('Circulation error') + \
                             ': ' + '{:.5e}'.format(IterationInfo['Circulation error']) + '\n'
                msg += '||' + '{:34}'.format('Number of sub-iterations') + \
                             ': ' + '{:d}'.format(IterationInfo['Number of sub-iterations (LL)']) +\
                                                                                                '\n'
                if NbLL == 1:
                    IterationInfo = getAerodynamicCoefficientsOnWing(LiftingLines, Surface,
                                                     StdDeviationSample = StdDeviationSample)
                    msg += '||' + '{:34}'.format('Lift')
                    msg += ': ' + '{:.3f}'.format(IterationInfo['Lift']) + ' N' + '\n'
                    msg += '||' + '{:34}'.format('Lift Standard Deviation')
                    msg += ': '+ '{:.2f}'.format(IterationInfo['Lift Standard Deviation'])+' %'+'\n'
                    msg += '||' + '{:34}'.format('Drag')
                    msg += ': ' + '{:.3f}'.format(IterationInfo['Drag']) + ' N' + '\n'
                    msg += '||' + '{:34}'.format('Drag Standard Deviation')
                    msg += ': '+ '{:.2f}'.format(IterationInfo['Drag Standard Deviation'])+' %'+'\n'
                    msg += '||'+'{:34}'.format('cL')+': '+ '{:.4f}'.format(IterationInfo['cL'])+'\n'
                    msg += '||'+'{:34}'.format('cD')+': '+ '{:.5f}'.format(IterationInfo['cD'])+'\n'
                    msg += '||'+'{:34}'.format('f') +': ' + '{:.4f}'.format(IterationInfo['f'])+'\n'
                    stdThrust = IterationInfo['Lift Standard Deviation']
                    stdPower = IterationInfo['Drag Standard Deviation']
                else:
                    U0 = np.linalg.norm(I.getNodeFromName(LiftingLines, 'VelocityFreestream')[1])
                    if U0 == 0:
                        IterationInfo = getAerodynamicCoefficientsOnRotor(LiftingLines,
                                                     StdDeviationSample = StdDeviationSample)
                    else:
                        IterationInfo = getAerodynamicCoefficientsOnPropeller(LiftingLines,
                                                     StdDeviationSample = StdDeviationSample)
                    msg += '||' + '{:34}'.format('Thrust')
                    msg += ': ' + '{:.3f}'.format(IterationInfo['Thrust']) + ' N' + '\n'
                    msg += '||' + '{:34}'.format('Thrust Standard Deviation')
                    msg +=': '+'{:.2f}'.format(IterationInfo['Thrust Standard Deviation'])+' %'+'\n'
                    msg += '||' + '{:34}'.format('Power')
                    msg += ': ' + '{:.3f}'.format(IterationInfo['Power']) + ' W' + '\n'
                    msg += '||' + '{:34}'.format('Power Standard Deviation')
                    msg += ': '+'{:.2f}'.format(IterationInfo['Power Standard Deviation'])+' %'+'\n'
                    msg += '||'+'{:34}'.format('cT') +': '+'{:.5f}'.format(IterationInfo['cT'])+'\n'
                    msg += '||'+'{:34}'.format('Cp') +': '+'{:.5f}'.format(IterationInfo['cP'])+'\n'
                    msg +='||'+'{:34}'.format('Eff')+': '+'{:.5f}'.format(IterationInfo['Eff'])+'\n'
                    stdThrust = IterationInfo['Thrust Standard Deviation']
                    stdPower = IterationInfo['Power Standard Deviation']
                if Verbose:
                    if it[0] != it0 + 1: deletePrintedLines(11)
                    print(msg)

            PolarData['OldVariable'] = Variable
            if NbLL == 1:
                PolarData['Lift'] = np.append(PolarData['Lift'], IterationInfo['Lift'])
                PolarData['Drag'] = np.append(PolarData['Drag'], IterationInfo['Drag'])
                PolarData['cL'] = np.append(PolarData['cL'], IterationInfo['cL'])
                PolarData['cD'] = np.append(PolarData['cD'], IterationInfo['cD'])
                PolarData['f'] = np.append(PolarData['f'], IterationInfo['f'])
                PolarData['LiftStandardDeviation'] = np.append(PolarData['LiftStandardDeviation'],
                                                           IterationInfo['Lift Standard Deviation'])
                PolarData['DragStandardDeviation'] = np.append(PolarData['DragStandardDeviation'], 
                                                           IterationInfo['Drag Standard Deviation'])
            else:
                PolarData['Thrust'] = np.append(PolarData['Thrust'], IterationInfo['Thrust'])
                PolarData['Power'] = np.append(PolarData['Power'], IterationInfo['Power'])
                PolarData['cT'] = np.append(PolarData['cT'], IterationInfo['cT'])
                PolarData['cP'] = np.append(PolarData['cP'], IterationInfo['cP'])
                PolarData['Efficiency'] = np.append(PolarData['Efficiency'], IterationInfo['Eff'])
                PolarData['ThrustStandardDeviation'] =np.append(PolarData['ThrustStandardDeviation']
                                                       , IterationInfo['Thrust Standard Deviation'])
                PolarData['PowerStandardDeviation'] = np.append(PolarData['PowerStandardDeviation']
                                                        , IterationInfo['Power Standard Deviation'])

            if Verbose:
                deletePrintedLines(3)
                print('||' + '{:-^50}'.format(''))

            if (it[0] - it0 == MaxNumberOfIterationsPerPolar):
                msg = ' Maximum number of iteration reached for '
            else:
                msg = ' Convergence criteria met for '

            if 'Pitch' not in PolarData:
                PolarData['LastPolar'] = 'VPM_Polar_' + PolarData['VariableName'] + '_'+ str(round(\
                                                                             Variable, 2)) + '.cgns'
                filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_Polars_' + PolarData['VariableName']\
                                                                                          + '.cgns')
                msg += PolarData['VariableName'] + ' = ' + str(round(Variable, 2)) + ' '
            else:
                PolarData['LastPolar'] = 'VPM_Polar_Pitch_' + str(round(Variable, 2)) + '.cgns'
                filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_Polars_Pitch.cgns')
                msg += 'Pitch = ' + str(round(Variable, 2)) + ' '
            if Verbose: print('||' + '{:-^50}'.format(msg))

            J.set(PolarsTree, 'Polars', **PolarData)
            save(PolarsTree, filename)

            filename = os.path.join(DIRECTORY_OUTPUT, PolarData['LastPolar'])
            save(I.merge([PolarsTree, t]), filename, VisualisationOptions)
            J.createSymbolicLink(filename,  DIRECTORY_OUTPUT + '.cgns')
            
            if Verbose:
                print('||' + '{:=^50}'.format('') + '\n||' + '{:=^50}'.format('') +\
                        '\n||' + '{:=^50}'.format(''))

        
        TotalTime = J.tic() - TotalTime
        print('||' + '{:=^50}'.format(' Total time spent: ' +str(int(round(TotalTime//60)))+' min '\
                                           + str(int(round(TotalTime - TotalTime//60*60))) + ' s '))
        for _ in range(3): print('||' + '{:=^50}'.format(''))

    def extract(t = [], ExctractionTree = [], NbOfParticlesUsedForPrecisionEvaluation = 1000,
        FarFieldApproximationOrder = 12, NearFieldOverlappingRatio = 0.4):
        if not ExctractionTree: return
        newFieldNames = ['Velocity' + v for v in 'XYZ'] + ['Vorticity' + v for v in 'XYZ'] + \
                        ['gradxVelocity' + v for v in 'XYZ']+ ['gradyVelocity' + v for v in 'XYZ']+\
                        ['gradzVelocity' + v for v in 'XYZ'] + ['PSE' + v for v in 'XYZ'] + \
                        ['VelocityMagnitude', 'VorticityMagnitude', 'divVelocity', 'QCriterion',
                                                                            'Nu', 'Sigma', 'Volume']
        [C._initVars(ExctractionTree, fn, 0.) for fn in newFieldNames]
        ExtractionZones = I.getZones(ExctractionTree)
        NPtsPerZone = [0] + [C.getNPts(z) for z in ExtractionZones]
        tmpZone = D.line((0, 0, 0), (1, 0, 0), np.sum(NPtsPerZone))
        [C._initVars(tmpZone, fn, 0.) for fn in newFieldNames]
        coordst = J.getxyz(tmpZone)
        for i, zone in enumerate(ExtractionZones):
            coords = J.getxyz(zone)
            for ct, c in zip(coordst, coords):
                ct[NPtsPerZone[i]:NPtsPerZone[i+1]] = c.ravel(order = 'F')
        tmpTree = C.newPyTree(['Base', tmpZone])
        Kernel = Kernel_str2int[getParameter(t, 'RegularisationKernel')]
        EddyViscosityModel = EddyViscosityModel_str2int[getParameter(t, 'EddyViscosityModel')]
        vpm_cpp.wrap_extract_plane(t, tmpTree, int(NbOfParticlesUsedForPrecisionEvaluation), Kernel,
                                           EddyViscosityModel, np.int32(FarFieldApproximationOrder),
                                                              np.float64(NearFieldOverlappingRatio))
        #TODO: add the extractperturbationField here
        tmpFields = J.getVars(I.getZones(tmpTree)[0], newFieldNames)

        for i, zone in enumerate(ExtractionZones):
            fields = J.getVars(zone, newFieldNames)
            for ft, f in zip(tmpFields, fields):
                fr = f.ravel(order = 'F')
                fr[:] = ft[NPtsPerZone[i]:NPtsPerZone[i+1]]

        return ExctractionTree

    def extractperturbationField(t = [], Targets = [], PerturbationFieldCapsule = []):
        if PerturbationFieldCapsule:
            TargetsBase = I.newCGNSBase('Targets', cellDim=1, physDim=3)
            TargetsBase[2] = I.getZones(Targets)
            PertubationFieldBase = I.newCGNSBase('PertubationFieldBase', cellDim=1, physDim=3)
            PertubationFieldBase[2] = pickPerturbationFieldZone(t)
            Theta, NumberOfNodes, TimeVelPert = getParameters(t, ['NearFieldOverlappingRatio',
                                                       'NumberOfNodes', 'TimeVelocityPerturbation'])
            TimeVelPert[0] += vpm_cpp.extract_perturbation_velocity_field(TargetsBase,
                            PertubationFieldBase, PerturbationFieldCapsule, NumberOfNodes, Theta)[0]