import sys
import os
import numpy as np
import VortexParticleMethod.vortexparticlemethod as vpm_cpp
import Converter.PyTree as C
import Geom.PyTree as D
import Converter.Internal as I
import Generator.PyTree as G
import Transform.PyTree as T
import CPlot.PyTree as CPlot
from . import particles as LLparticles
from .. import LiftingLine as MLL
from .. import Wireframe as W
from .. import InternalShortcuts as J
from time import time, sleep

Kernel_str2int = {'Gaussian': 0, 'Gaussian2': 0, 'G': 0, 'G2': 0, 'HOA': 1, 'HighOrderAlgebraic': 1,
                  'Gaussian3': 2, 'G3': 2, 'LOA': 3, 'LowOrderAlgebraic': 3, 'SuperGaussian': 4,
                  'SP': 4}
Scheme_str2int = {'Transpose': 0, 'T': 0, 'Mixed': 1, 'M': 1, 'Classical': 2, 'C': 2}
EddyViscosityModel_str2int = {'Mansour': 0, 'Smagorinsky': 1, 'Vreman': 2, None: -1, 'None': -1,
                              'Mansour2': -2}
RedistributionKernel_str2int = {'M4Prime': 5, 'M4': 4, 'M3': 3, 'M2': 2, 'M1': 1, None: 0, 'None': 0}
DiffusionScheme_str2int = {'PSE': 1, 'ParticleStrengthExchange': 1, 'pse': 1, 'CSM': 2, 'CS': 2,
                           'csm': 2, 'cs': 2, 'CoreSpreading': 2, 'CoreSpreadingMethod': 2, 'None': 0,
                           None: 0}

def setTimeStepFromShedParticles(t, LiftingLines, NumberParticlesShedAtTip = 5.):
    MLL.computeKinematicVelocity(LiftingLines)
    MLL.assembleAndProjectVelocities(LiftingLines)

    if type(t) == dict:
        Resolution = t['Resolution']
        U0         = t['VelocityFreestream']
    else:
        Particles  = pickParticlesZone(t)
        Resolution = I.getValue(I.getNodeFromName(Particles, 'Resolution'))
        U0         = I.getValue(I.getNodeFromName(Particles, 'VelocityFreestream'))

    Urelmax = 0.
    for LiftingLine in LiftingLines:
        Ukin = np.vstack(J.getVars(LiftingLine, ['VelocityKinematic'+i for i in 'XYZ']))
        ui   = np.vstack(J.getVars(LiftingLine, ['VelocityInduced'+i for i in 'XYZ']))
        Urel = U0 + ui.T - Ukin.T
        Urel = max([np.linalg.norm(urel) for urel in Urel])
        if (Urelmax < Urel): Urelmax = Urel

    if Urelmax == 0:
        raise ValueError('Maximum velocity is zero. Please set non-zero kinematic or ' +
                                                                    'freestream velocity.')

    if type(t) == dict:
        t['TimeStep'] = NumberParticlesShedAtTip*Resolution/Urel
    else:
        TimeStep = I.getNodeFromName(Particles, 'TimeStep')
        TimeStep[1][0] = NumberParticlesShedAtTip*Resolution/Urel

def setTimeStepFromBladeRotationAngle(t, LiftingLines, BladeRotationAngle = 5.):
    TimeStep = I.getNodeFromName(pickParticlesZone(t), 'TimeStep')
    RPM = 0.
    for LiftingLine in LiftingLines:
        RPM = max(RPM, I.getValue(I.getNodeFromName(LiftingLine, 'RPM')))
    TimeStep[1][0] = 1.*BladeRotationAngle/RPM/6.

TimeStepFunction_str2int = {'setTimeStepFromBladeRotationAngle': setTimeStepFromShedParticles,
                            'BladeRotationAngle': setTimeStepFromShedParticles,
                            'Angle': setTimeStepFromShedParticles,
                            'angle': setTimeStepFromShedParticles,
                            'setTimeStepFromShedParticles': setTimeStepFromShedParticles,
                            'ShedParticles': setTimeStepFromShedParticles,
                            'Shed': setTimeStepFromShedParticles, 'shed': setTimeStepFromShedParticles}

def maskParticlesInsideShieldBoxes(t, Boxes):
    BoxesBase = I.newCGNSBase('ShieldBoxes',cellDim=1,physDim=3)
    BoxesBase[2] = I.getZones(Boxes)
    return vpm_cpp.box_interaction(t, BoxesBase)

def addParticlesFromLiftingLineSources(t, Sources, SourcesM1, NumberParticlesShedPerStation,
                                            NumberSource):
    SourcesBase = I.newCGNSBase('Sources',cellDim=1,physDim=3)
    SourcesBase[2] = I.getZones(Sources)
    SourcesBaseM1 = I.newCGNSBase('SourcesM1',cellDim=1,physDim=3)
    SourcesBaseM1[2] = I.getZones(SourcesM1)
    return vpm_cpp.generate_particles(t, SourcesBase, SourcesBaseM1, NumberParticlesShedPerStation,
                                        NumberSource)

def getInducedVelocityFromWake(t, Target, TargetSigma):
    TargetBase = I.newCGNSBase('LiftingLine',cellDim=1,physDim=3)
    TargetBase[2] = I.getZones(Target)
    Kernel = Kernel_str2int[getParameter(t, 'RegularisationKernel')]
    return vpm_cpp.induce_velocity_from_wake(t, TargetBase, Kernel, TargetSigma)

def computeInducedVelocityOnLiftinLines(t, Nsource, Target, TargetSigma, WakeInducedVelocity):
    TargetBase = I.newCGNSBase('LiftingLine',cellDim=1,physDim=3)
    TargetBase[2] = I.getZones(Target)
    Kernel = Kernel_str2int[getParameter(t, 'RegularisationKernel')]
    return vpm_cpp.induce_total_velocity_on_lifting_line(t, Nsource, Kernel, TargetBase, TargetSigma,
                                                        WakeInducedVelocity)

def solveVorticityEquation(t, IterationInfo = {}):
    Kernel = Kernel_str2int[getParameter(t, 'RegularisationKernel')]
    Scheme = Scheme_str2int[getParameter(t, 'VorticityEquationScheme')]
    Diffusion = DiffusionScheme_str2int[getParameter(t, 'DiffusionScheme')]
    EddyViscosityModel = EddyViscosityModel_str2int[getParameter(t, 'EddyViscosityModel')]
    solveVorticityEquationInfo = vpm_cpp.wrap_vpm_solver(t, Kernel, Scheme, Diffusion,
                                                            EddyViscosityModel)
    IterationInfo['Number of threads'] = int(solveVorticityEquationInfo[0])
    IterationInfo['SIMD vectorisation'] = int(solveVorticityEquationInfo[1])
    IterationInfo['Near field overlapping ratio'] = solveVorticityEquationInfo[2]
    IterationInfo['Far field polynomial order'] = int(solveVorticityEquationInfo[3])
    IterationInfo['FMM time'] = solveVorticityEquationInfo[4]
    if len(solveVorticityEquationInfo) != 5:
        IterationInfo['Rel. err. of Velocity'] = solveVorticityEquationInfo[5]
        IterationInfo['Rel. err. of Velocity Gradient'] = solveVorticityEquationInfo[6]
        if len(solveVorticityEquationInfo) == 8: 
            IterationInfo['Rel. err. of PSE'] = solveVorticityEquationInfo[7]

    return IterationInfo

def computeNextTimeStep(t, NoDissipationRegions=[]):
    LiftingLines = MLL.getLiftingLines(t)
    NoDissipationRegions.extend(LiftingLines)
    Particles           = I.getNodeFromName2(t, 'Particles')
    IntegrationOrder, time, dt, it, lowstorage = getParameters(t,
        ['IntegrationOrder','Time', 'TimeStep', 'CurrentIteration', 'LowStorageIntegration'])
    NumberOfSources = 0
    for LiftingLine in LiftingLines:
        VPM_Parameters = J.get(LiftingLine,'.VPM#Parameters')
        NumberOfSources += len(VPM_Parameters['SigmaDistribution'])
    
    if lowstorage:
        if IntegrationOrder == 1:
            a = np.array([0.], dtype = np.float64)
            b = np.array([1.], dtype = np.float64)
        elif IntegrationOrder == 2:
            a = np.array([0., -0.5], dtype = np.float64)
            b = np.array([0.5, 1.], dtype = np.float64)
        elif IntegrationOrder == 3:
            a = np.array([0., -5./9., -153./128.], dtype = np.float64)
            b = np.array([1./3., 15./16., 8./15.], dtype = np.float64)
        elif IntegrationOrder == 4:
            a = np.array([0., -1., -0.5, -4.], dtype = np.float64)
            b = np.array([1./2., 1./2., 1., 1./6.], dtype = np.float64)
        else:
            raise AttributeError('This Integration Scheme Has Not Been Implemented Yet')
    else:
        if IntegrationOrder == 1:
            a = np.array([], dtype = np.float64)
            b = np.array([1.], dtype = np.float64)
        elif IntegrationOrder == 2:
            a = np.array([[0.5]], dtype = np.float64)
            b = np.array([0, 1.], dtype = np.float64)
        elif IntegrationOrder == 3:
            a = np.array([[0.5, 0.], [-1., 2.]], dtype = np.float64)
            b = np.array([1./6., 2./3., 1./6.], dtype = np.float64)
        elif IntegrationOrder == 4:
            a = np.array([[0.5, 0., 0.], [0., 0.5, 0.], [0., 0., 1.]], dtype = np.float64)
            b = np.array([1./6., 1./3., 1./3., 1./6.], dtype = np.float64)
        else:
            raise AttributeError('This Integration Scheme Has Not Been Implemented Yet')

    Kernel = Kernel_str2int[getParameter(t, 'RegularisationKernel')]
    Scheme = Scheme_str2int[getParameter(t, 'VorticityEquationScheme')]
    Diffusion = DiffusionScheme_str2int[getParameter(t, 'DiffusionScheme')]
    EddyViscosityModel = EddyViscosityModel_str2int[getParameter(t, 'EddyViscosityModel')]
    if lowstorage: vpm_cpp.runge_kutta_low_storage(t, a, b, Kernel, Scheme, Diffusion, 
                                                EddyViscosityModel, NumberOfSources)
    else: vpm_cpp.runge_kutta(t, a, b, Kernel, Scheme, Diffusion, EddyViscosityModel, NumberOfSources)

    time += dt
    it += 1

def populationControl(t, NoRedistributionRegions = [], IterationInfo = {}):
    tcontrol = time()
    LiftingLines = MLL.getLiftingLines(t)
    Particles = pickParticlesZone(t)
    Np = Particles[1][0]
    RotationCenter = I.getValue(I.getNodeFromName(LiftingLines, 'RotationCenter'))
    AABB = [RotationCenter[0], RotationCenter[1], RotationCenter[2], RotationCenter[0],
                    RotationCenter[1], RotationCenter[2]]
    for LiftingLine in I.getZones(LiftingLines):
        x, y, z = J.getxyz(LiftingLine)
        AABB[0] = min(min(x), AABB[0])
        AABB[1] = min(min(y), AABB[1])
        AABB[2] = min(min(z), AABB[2])
        AABB[3] = max(max(x), AABB[3])
        AABB[4] = max(max(y), AABB[4])
        AABB[5] = max(max(z), AABB[5])
    AABB = np.array(AABB, dtype = np.float64)

    NumberOfSources = 0
    for LiftingLine in LiftingLines:
        VPM_Parameters = J.get(LiftingLine,'.VPM#Parameters')
        NumberOfSources += len(VPM_Parameters['SigmaDistribution'])

    RedistributionKernel = RedistributionKernel_str2int[getParameter(t, 'RedistributionKernel')]
    Diffusion = DiffusionScheme_str2int[getParameter(t, 'DiffusionScheme')]
    N0 = Np[0]
    populationControlInfo = np.array([0, 0, 0, 0], dtype = np.int32)
    RedistributedParticles = vpm_cpp.populationControl(t, NumberOfSources, AABB, RedistributionKernel,
                                                        Diffusion, populationControlInfo)
    if len(RedistributedParticles) != 0:
        _adjustTreeSize(t, len(RedistributedParticles[0]))

        CoordinateX = I.getNodeFromName3(Particles, 'CoordinateX')
        CoordinateY = I.getNodeFromName3(Particles, 'CoordinateY')
        CoordinateZ = I.getNodeFromName3(Particles, 'CoordinateZ')
        AlphaX = I.getNodeFromName3(Particles, 'AlphaX')
        AlphaY = I.getNodeFromName3(Particles, 'AlphaY')
        AlphaZ = I.getNodeFromName3(Particles, 'AlphaZ')
        VorticityX = I.getNodeFromName3(Particles, 'VorticityX')
        VorticityY = I.getNodeFromName3(Particles, 'VorticityY')
        VorticityZ = I.getNodeFromName3(Particles, 'VorticityZ')
        VorticityMagnitude = I.getNodeFromName3(Particles, 'VorticityMagnitude')
        StrengthMagnitude = I.getNodeFromName3(Particles, 'StrengthMagnitude')
        Volume = I.getNodeFromName3(Particles, 'Volume')
        Sigma = I.getNodeFromName3(Particles, 'Sigma')
        Age = I.getNodeFromName3(Particles, 'Age')

        CoordinateX[1]        = RedistributedParticles[0][:]
        CoordinateY[1]        = RedistributedParticles[1][:]
        CoordinateZ[1]        = RedistributedParticles[2][:]
        AlphaX[1]             = RedistributedParticles[3][:]
        AlphaY[1]             = RedistributedParticles[4][:]
        AlphaZ[1]             = RedistributedParticles[5][:]
        StrengthMagnitude[1]  = RedistributedParticles[6][:]
        Volume[1]             = RedistributedParticles[7][:]
        Sigma[1]              = RedistributedParticles[8][:]
        Age[1]                = np.array([int(age) for age in RedistributedParticles[9]],
                                            dtype = np.int32)
    else:
        if Np[0] != N0:
            GridCoordinatesNode = I.getNodeFromName1(Particles,'GridCoordinates')
            FlowSolutionNode = I.getNodeFromName1(Particles,'FlowSolution')
            for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
                if node[3] == 'DataArray_t':
                    node[1] = node[1][:Np[0]]

    IterationInfo['Number of particles beyond cutoff'] = populationControlInfo[0]
    IterationInfo['Number of split particles'] = populationControlInfo[1]
    IterationInfo['Number of depleted particles'] = populationControlInfo[2]
    IterationInfo['Number of merged particles'] = populationControlInfo[3]
    IterationInfo['Population Control time'] = time() - tcontrol
    return IterationInfo

###############################################################################
# ------------------------------- added by LB ------------------------------- #

def initialise(LiftingLinesTree, PolarInterpolator, **solverParameters):
    int_Params =['MaxLiftingLineSubIterations', 'MinNbShedParticlesPerLiftingLine',
                 'StrengthRampAtbeginning', 'CurrentIteration', 'IntegrationOrder',
                 'MaximumAgeAllowed', 'RedistributionPeriod', 'NumberOfThreads',
                 'FarFieldApproximationOrder', 'IterationTuningFMM', 'IterationCounter',
                 'ShedParticlesIndex']

    float_Params = ['Density', 'EddyViscosityConstant', 'KinematicViscosity', 'Temperature', 'Time',
                    'AntiStretching', 'SFSContribution', 'SmoothingRatio', 'CirculationThreshold',
                    'CirculationRelaxation', 'Pitch', 'CutoffXmin', 'CutoffXmax', 'CutoffYmin',
                    'CutoffYmax', 'CutoffZmin', 'CutoffZmax', 'ForcedDissipation',
                    'MaximumAngleForMerging', 'MaximumMergingVorticityFactor',
                    'MinimumOverlapForMerging', 'MinimumVorticityFactor', 'RedistributeParticlesBeyond',
                    'RelaxationCutoff', 'RedistributeParticleSizeFactor', 'RemoveWeakParticlesBeyond',
                    'ResizeParticleFactor', 'NearFieldOverlappingRatio', 'TimeFMM', 'Resolution',
                    'RPM', 'VelocityFreestream', 'VelocityTranslation', 'TimeStep', 'Sigma0']

    bool_Params = ['GammaZeroAtRoot', 'GammaZeroAtTip', 'GhostParticleAtRoot', 'GhostParticleAtTip',
                   'LowStorageIntegration', 'MonitorInvariants']

    defaultParameters = {
    ##############################################################################################
    ############################## Atmospheric/Freestream conditions #############################
    ##############################################################################################
        'Density'                            : 1.225,                 #in kg.m^-3, 1.01325e5/((273.15 + 15)*287.05)
        'EddyViscosityConstant'              : 0.07,                  #constant for the eddy viscosity model, Cm(Mansour) around 0.1, Cs(Smagorinsky) around 0.15, Cr(Vreman) around 0.7
        'EddyViscosityModel'                 : 'Vreman',              #Mansour, Smagorinsky, Vreman or None, select a LES model to compute the eddy viscosity
        'KinematicViscosity'                 : 1.46e-5,               #in m^2.s^-1, kinematic viscosity, TODO must be inferred from atmospheric conditions
        'Temperature'                        : 288.15,                #in K, 273.15 + 15.
        'Time'                               : 0.,                    #in s, keep track of the physical time
    ##############################################################################################
    ####################################### VPM parameters #######################################
    ##############################################################################################
        'AntiStretching'                     : 0.,                    #between 0 and 1, 0 means particle strength fully takes vortex stretching, 1 means the particle size fully takes the vortex stretching
        'DiffusionScheme'                    : 'PSE',                 #PSE, CSM or None. gives the scheme used to compute the diffusion term of the vorticity equation
        'RegularisationKernel'               : 'Gaussian',            #The available smoothing kernels are Gaussian, HOA, LOA, Gaussian3 and SuperGaussian
        'SFSContribution'                    : 0.,                    #between 0 and 1, the closer to 0, the more the viscosity affects the particle strength, the closer to 1, the more it affects the particle size
        'SmoothingRatio'                     : 1.5,                   #in m, anywhere between 1.5 and 2.5, the higher the NumberSource, the smaller the Resolution and the higher the SmoothingRatio should be to avoid blowups, the HOA kernel requires a higher smoothing
        'VorticityEquationScheme'            : 'Transpose',           #Classical, Transpose or Mixed, The schemes used to compute the vorticity equation are the classical scheme, the transpose scheme (conserves total vorticity) and the mixed scheme (a fusion of the previous two)
    ##############################################################################################
    ################################## Lifting Lines parameters ##################################
    ##############################################################################################
        'CirculationThreshold'               : 1e-4,                  #convergence criteria for the circulation sub-iteration process, somewhere between 1e-3 and 1e-6 is ok
        'CirculationRelaxation'              : 1./5.,                 #relaxation parameter of the circulation sub-iterations, somwhere between 0.1 and 1 is good, the more unstable the simulation, the lower it should be
        'GammaZeroAtRoot'                    : 1,                     #[|0, 1|], sets the circulation of the root of the blade to zero
        'GammaZeroAtTip'                     : 1,                     #[|0, 1|], sets the circulation of the tip  of the blade to zero
        'GhostParticleAtRoot'                : 0,                     #[|0, 1|], add a particles at after the root of the blade
        'GhostParticleAtTip'                 : 0,                     #[|0, 1|], add a particles at after the tip  of the blade
        'IntegralLaw'                        : 'linear',              #uniform, tanhOneSide, tanhTwoSides or ratio, gives the type of interpolation of the circulation on the lifting lines
        'MaxLiftingLineSubIterations'        : 100,                   #max number of sub iteration when computing the LL circulations
        'MinNbShedParticlesPerLiftingLine'   : 27,                    #minimum number of station for every LL from which particles are shed
        'ParticleDistribution'               : dict(kind = 'uniform', #uniform, tanhOneSide, tanhTwoSides or ratio, repatition law of the particles on the Lifting Lines
                                               FirstSegmentRatio = 1.,#size of the particles at the root of the blades relative to Sigma0 (i.e. Resolution*SmoothingRatio)
                                                LastSegmentRatio = 1.,#size of the particles at the tip  of the blades relative to Sigma0 (i.e. Resolution*SmoothingRatio)
                                                 Symmetrical = False),#[|0, 1|], gives a symmetrical repartition of particles along the blades or not
        'Pitch'                              : 0.,                    #]-180., 180[ in deg, gives the pitch given to all the lifting lines, if 0 no pitch id added
        'StrengthRampAtbeginning'            : 25,                    #[|0, +inf [|, limit the vorticity shed for the StrengthRampAtbeginning first iterations for the wake to stabilise
    ##############################################################################################
    #################################### Simulation Parameters ###################################
    ##############################################################################################
        'CurrentIteration'                   : 0,                     #follows the current iteration
        'IntegrationOrder'                   : 1,                     #[|1, 4|]1st, 2nd, 3rd or 4th order Runge Kutta
        'LowStorageIntegration'              : 1,                     #[|0, 1|], states if the classical or the low-storage Runge Kutta is used
        'MonitorInvariants'                  : False,                 #must be linked with the invariants function
    ##############################################################################################
    ###################################### Particles Control #####################################
    ##############################################################################################
        'CutoffXmin'                         : -np.inf,               #in m, spatial Cutoff
        'CutoffXmax'                         : +np.inf,               #in m, spatial Cutoff
        'CutoffYmin'                         : -np.inf,               #in m, spatial Cutoff
        'CutoffYmax'                         : +np.inf,               #in m, spatial Cutoff
        'CutoffZmin'                         : -np.inf,               #in m, spatial Cutoff
        'CutoffZmax'                         : +np.inf,               #in m, spatial Cutoff
        'ForcedDissipation'                  : 0.,                    #in %/s, gives the % of strength particles looses per sec, usefull to kill unnecessary particles without affecting the LLs
        'MaximumAgeAllowed'                  : 0,                     #0 <=,  particles are eliminated after MaximumAgeAllowed iterations, if MaximumAgeAllowed == 0, they are not deleted
        'MaximumAngleForMerging'             : 0.,                    #[0., 180.[ in deg,   maximum angle   allowed between two particles to be merged
        'MaximumMergingVorticityFactor'      : 0.,                    #in %, particles can be merged if their combined strength is below the given poucentage of the maximum strength on the blades
        'MinimumOverlapForMerging'           : 0.,                    #[0., +inf[, if two particles have at least an overlap of MinimumOverlapForMerging*SigmaRatio, they are considered for merging
        'MinimumVorticityFactor'             : 0.,                    #in %, sets the minimum strength kept as a percentage of the maximum strength on the blades
        'RedistributeParticlesBeyond'        : np.inf,                #do not redistribute particles if closer than RedistributeParticlesBeyond*Resolution from a LL
        'RedistributionKernel'               : None,                  #M4Prime, M4, M3, M2, M1 or None, redistribution kernel used. the number gives the order preserved by the kernel, if None local splitting/merging is used
        'RedistributionPeriod'               : 0,                     #frequency at which particles are redistributed, if 0 the particles are never redistributed
        'RelaxationCutoff'                   : 0.,                    #in Hz, is used during the relaxation process to realign the particles with the vorticity
        'RemoveWeakParticlesBeyond'          : np.inf,                #do not remove weak particles if closer than RemoveWeakParticlesBeyond*Resolution from a LL
        'ResizeParticleFactor'               : 0.,                    #[0, +inf[, resize particles that grow/shrink RedistributeParticleSizeFactor * Sigma0 (i.e. Resolution*SmoothingRatio), if 0 then no resizing is done
    ##############################################################################################
    ####################################### FMM parameters #######################################
    ##############################################################################################
        'FarFieldApproximationOrder'         : 8,                     #[|6, 12|], order of the polynomial which approximates the far field interactions, the higher the more accurate and the more costly
        'IterationTuningFMM'                 : 50,                    #frequency at which the FMM is compared to the direct computation, gives the relative L2 error
        'NearFieldOverlappingRatio'          : 0.5,                   #[0., 1.], Direct computation of the interactions between clusters that overlap by NearFieldOverlappingRatio, the smaller the more accurate and the more costly
        'NumberOfThreads'                    : int(os.getenv('NPROCMPI', 48)),#number of threads of the machine used, does not matter if above the total number of threads of the machine, just slows down the simulation
        'TimeFMM'                            : 0.,                    #in s, keep track of the CPU time spent for the FMM
    }

    defaultParameters.update(solverParameters)
    for key in defaultParameters:
        if key in int_Params:
            defaultParameters[key] = np.atleast_1d(np.array(defaultParameters[key],
                                                    dtype = np.int32))
        elif key in float_Params:
            defaultParameters[key] = np.atleast_1d(np.array(defaultParameters[key],
                                                    dtype = np.float64))
        elif key in bool_Params:
            defaultParameters[key] = np.atleast_1d(np.array(defaultParameters[key],
                                                    dtype = np.int32))

    # renaming
    TypeOfInput = I.isStdNode(LiftingLinesTree)
    ERRMSG = J.FAIL+'LiftingLinesTree must be a tree, a list of bases or a list of zones'+J.ENDC
    if TypeOfInput == -1: # is a standard CGNS node
        if I.isTopTree(LiftingLinesTree):
            LiftingLineBases = I.getBases(LiftingLinesTree)
            if len(LiftingLineBases) == 1 and LiftingLineBases[0][0] == 'Base':
                LiftingLineBases[0][0] = 'LiftingLines'
        elif LiftingLinesTree[3] == 'CGNSBase_t':
            LiftingLineBase = LiftingLinesTree
            if LiftingLineBase[0] == 'Base': LiftingLineBase[0] = 'LiftingLines'
            LiftingLinesTree = C.newPyTree([])
            LiftingLinesTree[2] = [LiftingLineBase]
        elif LiftingLinesTree[3] == 'Zone_t':
            LiftingLinesTree = C.newPyTree(['LiftingLines', [LiftingLinesTree]])
        else:
            raise AttributeError(ERRMSG)
    elif TypeOfInput == 0: # is a list of CGNS nodes
        if LiftingLinesTree[0][3] == 'CGNSBase_t':
            LiftingLinesBases = I.getBases(LiftingLinesTree)
            LiftingLinesTree = C.newPyTree([])
            LiftingLinesTree[2] = LiftingLinesBases
        elif LiftingLinesTree[0][3] == 'Zone_t':
            LiftingLinesZones = I.getZones(LiftingLinesTree)
            LiftingLinesTree = C.newPyTree(['LiftingLines',LiftingLinesZones])
        else:
            raise AttributeError(ERRMSG)

    else:
        raise AttributeError(ERRMSG)

    LiftingLines = I.getZones(LiftingLinesTree)


    if 'Resolution' not in defaultParameters:
        ShortestLiftingLineSpan = np.inf
        for LiftingLine in LiftingLines:
            ShortestLiftingLineSpan = np.minimum(ShortestLiftingLineSpan,
                                                 W.getLength(LiftingLine))
        defaultParameters['Resolution'] = ShortestLiftingLineSpan / \
                (defaultParameters['MinNbShedParticlesPerLiftingLine'] - 2)

    if 'RPM' in defaultParameters:
        RPM = defaultParameters['RPM']
        MLL.setRPM(LiftingLines, defaultParameters['RPM'])
    else: defaultParameters['RPM'] = 0

    if defaultParameters['Pitch'] != 0.:
        for LiftingLine in LiftingLines:
            FlowSolution = I.getNodeFromName(LiftingLine, 'FlowSolution')
            Twist = I.getNodeFromName(FlowSolution, 'Twist')
            Twist[1] += defaultParameters['Pitch']

    if 'VelocityFreestream' not in defaultParameters: 
        defaultParameters['VelocityFreestream'] = np.array([0.]*3,dtype = float)

    for LiftingLine in LiftingLines:
        J.set(LiftingLine, '.Conditions',
                Density=defaultParameters['Density'],
                Temperature=defaultParameters['Temperature'],
                VelocityFreestream=defaultParameters['VelocityFreestream'])

    if 'VelocityTranslation' in defaultParameters:
        VelocityTranslation = defaultParameters['VelocityTranslation']
        for LiftingLine in LiftingLines:
            Kinematics = I.getNodeFromName(LiftingLine, '.Kinematics')
            CurrentVelocityTranslation = I.getNodeFromName(Kinematics, 'VelocityTranslation')
            CurrentVelocityTranslation[1] = np.array(VelocityTranslation, dtype = np.float64,
                                                        order = 'F')
    else: defaultParameters['VelocityTranslation'] = np.array([0.]*3, dtype = np.float64,
                                                                order = 'F')

    MLL.computeKinematicVelocity(LiftingLines)
    MLL.assembleAndProjectVelocities(LiftingLines)

    ShieldBoxes = LLparticles._initialiseLiftingLinesAndGetShieldBoxes(LiftingLines,
                        PolarInterpolator, defaultParameters['Resolution'])
    #ShieldBoxesTree = C.newPyTree(['ShieldsBase', I.getZones(ShieldBoxes)])

    if 'TimeStep' not in defaultParameters:
        setTimeStepFromShedParticles(defaultParameters, LiftingLines, NumberParticlesShedAtTip = 1.)

    os.environ['OMP_NUM_THREADS'] = str(defaultParameters['NumberOfThreads'])
    defaultParameters['Sigma0'] = defaultParameters['Resolution']* \
                                 defaultParameters['SmoothingRatio']
    defaultParameters['IterationCounter'] = 0
    defaultParameters['StrengthRampAtbeginning'] = max(defaultParameters['StrengthRampAtbeginning'], 1)
    defaultParameters['MinimumVorticityFactor'] = max(0., defaultParameters['MinimumVorticityFactor'])

    Sources = []
    SourcesM1 = []
    TotalNumberOfSources = 0
    for LiftingLine in LiftingLines:
        VPM_Parameters = J.get(LiftingLine,'.VPM#Parameters')
        if not VPM_Parameters:
            MLL.setVPMParameters(LiftingLine)
            VPM_Parameters = J.get(LiftingLine,'.VPM#Parameters')
        if 'GammaZeroAtRoot' in defaultParameters:
            VPM_Parameters['GammaZeroAtRoot'][0] = defaultParameters['GammaZeroAtRoot']
        if 'GammaZeroAtTip'  in defaultParameters:
            VPM_Parameters['GammaZeroAtTip'][0]  = defaultParameters['GammaZeroAtTip']
        if 'GhostParticleAtRoot' in defaultParameters:
            VPM_Parameters['GhostParticleAtRoot'][0] = defaultParameters['GhostParticleAtRoot']
        if 'GhostParticleAtTip'  in defaultParameters:
            VPM_Parameters['GhostParticleAtTip'][0]  = defaultParameters['GhostParticleAtTip']
        if 'IntegralLaw' in defaultParameters:
            VPM_Parameters['IntegralLaw'] = defaultParameters['IntegralLaw']
        if 'ParticleDistribution' in defaultParameters:
            ParticleDistributionOld = defaultParameters['ParticleDistribution']
        else: ParticleDistributionOld = VPM_Parameters['ParticleDistribution']

        ParticleDistribution = {'kind' : ParticleDistributionOld['kind']}

        if 'FirstSegmentRatio' in ParticleDistributionOld:
            ParticleDistribution['FirstCellHeight'] = ParticleDistributionOld['FirstSegmentRatio']*\
                                                        defaultParameters['Resolution']
        if 'LastSegmentRatio' in ParticleDistributionOld:
            ParticleDistribution['LastCellHeight'] = ParticleDistributionOld['LastSegmentRatio']*\
                                                        defaultParameters['Resolution']
        if 'growthRatio' in ParticleDistributionOld:
            ParticleDistribution['growth'] = ParticleDistributionOld['growthRatio']*\
                                                defaultParameters['Resolution']
        if 'parameter' in ParticleDistributionOld:
            ParticleDistribution['parameter'] = ParticleDistributionOld['parameter']
        if 'Symmetrical' in ParticleDistributionOld:
            ParticleDistribution['Symmetrical'] = ParticleDistributionOld['Symmetrical']

        L = W.getLength(LiftingLine)
        NumberOfSources = int(np.round(L/defaultParameters['Resolution'])) + 2#n segments gives n + 1 stations, each station is surrounded by two particles, thus because of the extremities, there are n + 2 particles shed

        if ParticleDistribution['Symmetrical']:
            SemiWing = W.linelaw(P1 = (0., 0., 0.), P2 = (L/2., 0., 0.), N = int(NumberOfSources/2),
                                    Distribution = ParticleDistribution)
            WingDiscretization = J.getx(T.join(T.symetrize(SemiWing, (0, 0, 0), (0, 1, 0), (0, 0, 1)),
                                                            SemiWing))
            WingDiscretization += L/2.
            ParticleDistribution = WingDiscretization/L
        else:
            WingDiscretization = J.getx(W.linelaw(P1 = (0., 0., 0.), P2 = (L, 0., 0.),
                                                    N = NumberOfSources - 1,
                                                    Distribution = ParticleDistribution))
            ParticleDistribution = WingDiscretization/L

        TotalNumberOfSources += len(WingDiscretization) - 1 + VPM_Parameters['GhostParticleAtRoot'][0]\
                                + VPM_Parameters['GhostParticleAtTip'][0]
        Source = MLL.buildVortexParticleSourcesOnLiftingLine(LiftingLine,
                                                    AbscissaSegments = [ParticleDistribution],
                                                    IntegralLaw = VPM_Parameters['IntegralLaw'])
        Sources += [Source]

        Kinematics = J.get(LiftingLine,'.Kinematics')
        VelocityRelative = defaultParameters['VelocityFreestream'] - Kinematics['VelocityTranslation']
        Dpsi = Kinematics['RPM']*6.*defaultParameters['TimeStep']
        if not Kinematics['RightHandRuleRotation']: Dpsi *= -1
        T._rotate(Source, Kinematics['RotationCenter'], Kinematics['RotationAxis'], -Dpsi[0])
        T._translate(Source, defaultParameters['TimeStep']*VelocityRelative)
        SourcesM1 += [Source]

        CoordinateX = I.getValue(I.getNodeFromName(Sources[-1], 'CoordinateX'))
        CoordinateY = I.getValue(I.getNodeFromName(Sources[-1], 'CoordinateY'))
        CoordinateZ = I.getValue(I.getNodeFromName(Sources[-1], 'CoordinateZ'))
        dy = ((CoordinateX[1:] - CoordinateX[:-1])**2 + (CoordinateY[1:] - CoordinateY[:-1])**2 +\
                            (CoordinateZ[1:] - CoordinateZ[:-1])**2)**0.5
        SigmaDistribution = dy*defaultParameters['SmoothingRatio']#np.array([defaultParameters['Sigma0'][0]]*len(dy), order = 'F', dtype = np.float64)
        x, y, z = J.getxyz(LiftingLine)
        dy = ((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2 + (z[1:] - z[:-1])**2)**0.5
        mean_dy = np.insert((dy[1:] + dy[:-1])/2., [0, len(dy)-1], [dy[0], dy[-1]])
        SigmaDistributionOnLiftingLine = mean_dy*defaultParameters['SmoothingRatio']

        J.set(LiftingLine, '.VPM#Parameters',
                GammaZeroAtRoot = VPM_Parameters['GammaZeroAtRoot'],
                GammaZeroAtTip = VPM_Parameters['GammaZeroAtTip'],
                GhostParticleAtRoot = VPM_Parameters['GhostParticleAtRoot'],
                GhostParticleAtTip = VPM_Parameters['GhostParticleAtTip'],
                IntegralLaw = VPM_Parameters['IntegralLaw'],
                ParticleDistribution = ParticleDistribution,
                SigmaDistribution = SigmaDistribution,
                SigmaDistributionOnLiftingLine = SigmaDistributionOnLiftingLine)

    defaultParameters['ShedParticlesIndex'] = np.array([i for i in range(TotalNumberOfSources,
                                                    2*TotalNumberOfSources)], dtype = np.int32)
    RotationAngle = defaultParameters['RPM']*60.*defaultParameters['TimeStep']
    ParticlesTree = _initialiseParticles(defaultParameters, Sources, SourcesM1, SigmaDistribution)
    Particles = I.getZones(ParticlesTree)[0]
    J.set(Particles, 'SolverParameters', **defaultParameters)
    SolverParametersNode = I.getNodeFromName1(Particles, 'SolverParameters')
    I._sortByName(SolverParametersNode)
    MLL.moveLiftingLines(LiftingLines, -defaultParameters['TimeStep'])#for the simulation to start with the propeller at phi = 0
    t = I.merge([ParticlesTree, LiftingLinesTree])#, ShieldBoxesTree])
    LLparticles.addParticles(t, PolarInterpolator)
    solveVorticityEquation(t)
    IterationCounter = I.getNodeFromName(t, 'IterationCounter')
    IterationCounter[1][0] = defaultParameters['IterationTuningFMM']*defaultParameters['IntegrationOrder']

    return t

def _initialiseParticles(solverParameters, Sources, SourcesM1, SigmaDistribution):
    X, Y, Z, S = [], [], [], []
    Np = 0
    for i, Source in enumerate(Sources):
        SourceX = I.getValue(I.getNodeFromName(Source, 'CoordinateX'))
        SourceY = I.getValue(I.getNodeFromName(Source, 'CoordinateY'))
        SourceZ = I.getValue(I.getNodeFromName(Source, 'CoordinateZ'))
        X += [(SourceX[1:] + SourceX[:-1])/2.]
        Y += [(SourceY[1:] + SourceY[:-1])/2.]
        Z += [(SourceZ[1:] + SourceZ[:-1])/2.]
        S += [SigmaDistribution]
        Np += len(X[-1])

    for i, Source in enumerate(SourcesM1):
        SourceX = I.getValue(I.getNodeFromName(Source, 'CoordinateX'))
        SourceY = I.getValue(I.getNodeFromName(Source, 'CoordinateY'))
        SourceZ = I.getValue(I.getNodeFromName(Source, 'CoordinateZ'))
        X += [(SourceX[1:] + SourceX[:-1])/2.]
        Y += [(SourceY[1:] + SourceY[:-1])/2.]
        Z += [(SourceZ[1:] + SourceZ[:-1])/2.]
        S += [SigmaDistribution]
        Np += len(X[-1])

    X, Y, Z = np.array(X).reshape(Np), np.array(Y).reshape(Np), np.array(Z).reshape(Np)
    S = np.array(S).reshape(Np)
    Particles = C.convertArray2Node(D.line((0., 0., 0.), (0., 0., 0.), Np))
    Particles[0] = 'Particles'
    FieldNames = ['Active', 'Age', 'AlphaX', 'AlphaY', 'AlphaZ', 'gradxVelocityX', 'gradyVelocityX',
                  'gradzVelocityX', 'gradxVelocityY', 'gradyVelocityY', 'gradzVelocityY',
                  'gradxVelocityZ', 'gradyVelocityZ', 'gradzVelocityZ', 'Nu', 'PSEX', 'PSEY', 'PSEZ',
                  'Sigma', 'StrengthMagnitude', 'StretchingX', 'StretchingY', 'StretchingZ',
                  'VelocityInducedX', 'VelocityInducedY', 'VelocityInducedZ', 'Volume',
                  'VorticityMagnitude', 'VorticityX', 'VorticityY', 'VorticityZ']
    fields = J.invokeFieldsDict(Particles, FieldNames)
    CoordinateX = I.getNodeFromName(Particles, 'CoordinateX')
    CoordinateX[1] = X
    CoordinateY = I.getNodeFromName(Particles, 'CoordinateY')
    CoordinateY[1] = Y
    CoordinateZ = I.getNodeFromName(Particles, 'CoordinateZ')
    CoordinateZ[1] = Z
    Sigma = I.getNodeFromName(Particles, 'Sigma')
    Sigma[1] = S
    Volume = I.getNodeFromName(Particles, 'Volume')
    Volume[1] = 4./3.*np.pi*Sigma[1]**3

    IntegerFieldNames = ['Active', 'Age']
    for intFieldName in IntegerFieldNames:
        IntegerFieldNode = I.getNodeFromName2(Particles, intFieldName)
        IntegerFieldNode[1] = np.ones(len(fields[intFieldName]),dtype = np.int32)
    fields['Nu'][:] = solverParameters['KinematicViscosity']
    fields['Sigma'][:] = solverParameters['Sigma0']

    try: addInvariants = solverParameters['MonitorInvariants']
    except KeyError: addInvariants = False

    if addInvariants:
        J.set(Particles, 'Invariants', Omega=[0.,0.,0.], LinearImpulse=[0.,0.,0.],
            AngularImpulse=[0.,0.,0.], Helicity=0., KineticEnergy=0.,
            KineticEnergyDivFree=0., Enstrophy=0., EnstrophyDivFree=0.)

    return C.newPyTree(['ParticlesBase', Particles])

def pickParticlesZone(t):
    for z in I.getZones(t):
        if z[0] == 'Particles':
            return z

def _getValueFromParameterNode(node):
    if node is None: raise ValueError('requested node was not found')
    n = node[1]
    if isinstance(n, np.ndarray):
        if n.dtype.char == 'S' or n.dtype.char == 'c':
            return I.getValue(node)
    return n

def getParameter(t, Parameter):
    Particles = pickParticlesZone(t)
    SolverParametersNode = I.getNodeFromName1(Particles, 'SolverParameters')
    ParameterNode = I.getNodeFromName1(SolverParametersNode, Parameter)
    return _getValueFromParameterNode(ParameterNode)

def getParameters(t, Parameters):
    Particles = pickParticlesZone(t)
    sp = I.getNodeFromName1(Particles, 'SolverParameters')
    return [_getValueFromParameterNode(I.getNodeFromName1(sp,p)) for p in Parameters]

def getSolverParameters(t):
    Particles = pickParticlesZone(t)
    return J.get(Particles,'SolverParameters')

def _adjustTreeSize(t, NewSize):
    Particles = pickParticlesZone(t)
    GridCoordinatesNode = I.getNodeFromName1(Particles,'GridCoordinates')
    FlowSolutionNode = I.getNodeFromName1(Particles,'FlowSolution')
    for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
        if node[3] == 'DataArray_t':
            if node[0] in ['Active', 'Age']:
                node[1] = np.zeros(NewSize,dtype = node[1].dtype) + (node[0] == 'Active')
            else:
                node[1] = np.zeros(NewSize,dtype = node[1].dtype)
    Particles[1].ravel(order = 'F')[0] = NewSize

def _trim(function, t, *args, **kwargs):
    Particles = pickParticlesZone(t)
    GridCoordinatesNode = I.getNodeFromName1(Particles,'GridCoordinates')
    CoordinateX = I.getNodeFromName1(GridCoordinatesNode,'CoordinateX')[1]
    NumberOfParticlesBeforeLimitor = len(CoordinateX)
    function(t, *args, **kwargs)
    Np = Particles[1].ravel(order = 'F')[0]
    if Np != NumberOfParticlesBeforeLimitor:
        FlowSolutionNode = I.getNodeFromName1(Particles,'FlowSolution')
        for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
            if node[3] == 'DataArray_t':
                node[1] = node[1][:Np]

def _extend(t, AdditionalNumberOfParticles):
    Particles = pickParticlesZone(t)
    GridCoordinatesNode = I.getNodeFromName1(Particles,'GridCoordinates')
    FlowSolutionNode = I.getNodeFromName1(Particles,'FlowSolution')
    for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
        if node[3] == 'DataArray_t':
            if node[0] in ['Active', 'Age']:
                node[1] = np.append(node[1], np.zeros(AdditionalNumberOfParticles,
                                                    dtype = node[1].dtype) + (node[0] == 'Active'))
            else:
                node[1] = np.append(node[1], np.zeros(AdditionalNumberOfParticles,
                                                    dtype = node[1].dtype))
    Particles[1].ravel(order = 'F')[0] = len(node[1])

def _roll(t, PivotNumber):
    Particles = pickParticlesZone(t)
    Np = Particles[1].ravel(order = 'F')[0]
    if PivotNumber == 0 or PivotNumber == Np: return
    if PivotNumber > Np:
        raise ValueError('PivotNumber (%d) cannot be greater than existing number of particles (%d)'%(PivotNumber,Np))
    GridCoordinatesNode = I.getNodeFromName1(Particles,'GridCoordinates')
    FlowSolutionNode = I.getNodeFromName1(Particles,'FlowSolution')
    for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
        if node[3] == 'DataArray_t':
            node[1] = np.roll(node[1], PivotNumber)

def setVisualization(t, ParticlesColorField='VorticityMagnitude',
                        ParticlesRadius='{Sigma}/10',
                        addLiftingLineSurfaces=True, AirfoilPolarsFilename=None):
    Particles = pickParticlesZone(t)
    Sigma = I.getValue(I.getNodeFromName(Particles, 'Sigma'))
    #Sigma0 = I.getValue(I.getNodeFromName(Particles, 'Sigma0'))
    #maxS, minS, meanS = max(Sigma), min(Sigma), sum(Sigma)/len(Sigma)
    #ParticlesRadius = ((3./2.*Sigma0 - 2./3.*Sigma0)*Sigma + 2./3.*Sigma0*maxS - 3./2.*Sigma0*minS)/(maxS - minS)/10.
    #C._initVars(Particles,'radius', 0)
    #radius = I.getNodeFromName(Particles, 'radius')
    #radius[1] = ParticlesRadius

    C._initVars(Particles,'radius='+ParticlesRadius)
    CPlot._addRender2Zone(Particles, material='Sphere',
        color='Iso:'+ParticlesColorField, blending=0.6, shaderParameters=[0.04, 0])
    LiftingLines = MLL.getLiftingLines(t)
    for zone in LiftingLines:
        CPlot._addRender2Zone(zone, material='Flat', color='White')
    Shields = I.getZones(I.getNodeFromName2(t,'ShieldsBase'))
    for zone in Shields:
        CPlot._addRender2Zone(zone, material='Glass', color='White', blending=0.6)
    if addLiftingLineSurfaces:
        if not AirfoilPolarsFilename:
            ERRMSG = J.FAIL+('production of surfaces from lifting-line requires'
                ' attribute AirfoilPolars')+J.ENDC
            raise AttributeError(ERRMSG)
        LiftingLineSurfaces = []
        for ll in LiftingLines:
            surface = MLL.postLiftingLine2Surface(ll,AirfoilPolarsFilename)
            surface[0] = ll[0]+'.surf'
            CPlot._addRender2Zone(surface, material='Solid', color='#ECF8AB')
            LiftingLineSurfaces += [surface]
        I.createUniqueChild(t,'LiftingLineSurfaces','CGNSBase_t',
            value=np.array([2,3],order = 'F'),children=LiftingLineSurfaces)

    CPlot._addRender2PyTree(t, mode='Render', colormap='Blue2Red', isoLegend=1,
                               scalarField=ParticlesColorField)

def saveImage(t, ShowInScreen=False, ImagesDirectory='FRAMES', **DisplayOptions):

    if 'mode' not in DisplayOptions: DisplayOptions['mode'] = 'Render'
    if 'displayInfo' not in DisplayOptions: DisplayOptions['displayInfo'] = 0
    if 'colormap' not in DisplayOptions: DisplayOptions['colormap'] = 0
    if 'win' not in DisplayOptions: DisplayOptions['win'] = (700,700)
    DisplayOptions['exportResolution'] = '%gx%g'%DisplayOptions['win']


    try: os.makedirs(ImagesDirectory)
    except: pass

    sp = getSolverParameters(t)

    DisplayOptions['export'] = os.path.join(ImagesDirectory,
        'frame%05d.png'%sp['CurrentIteration'])

    if ShowInScreen:
        DisplayOptions['offscreen']=0
    else:
        machine = os.getenv('MAC', 'ld')
        if 'spiro' in machine or 'sator' in machine:
            DisplayOptions['offscreen']=1 # TODO solve bug https://elsa.onera.fr/issues/10536
        else:
            DisplayOptions['offscreen']=2

    CPlot.display(t, **DisplayOptions)
    from time import sleep; sleep(0.1)
    if 'backgroundFile' not in DisplayOptions:
        MOLA = os.getenv('MOLA')
        MOLASATOR = os.getenv('MOLASATOR')
        for MOLAloc in [MOLA, MOLASATOR]:
            backgroundFile = os.path.join(MOLAloc,'MOLA','GUIs','background.png')
            if os.path.exists(backgroundFile):
                CPlot.setState(backgroundFile=backgroundFile)
                CPlot.setState(bgColor=13)
                break

def open(*args,**kwargs): return C.convertFile2PyTree(*args,**kwargs)

def save(*args, **kwargs):
    try:
        if os.path.islink(args[1]):
            os.unlink(args[1])
        else:
            os.remove(args[1])
    except:
        pass

    return C.convertPyTree2File(*args, **kwargs)

def loadAirfoilPolars(filename): return MLL.loadPolarsInterpolatorDict(filename)

def addParticlesFromLiftingLines(*args, **kwargs): return LLparticles.addParticles(*args, **kwargs)

def exit(): os._exit(0)

def extract(t, ExctractionTree, NbOfParticlesUsedForPrecisionEvaluation = 1000):
    if not ExctractionTree: return
    newFieldNames = ['divVelocity', 'gradxVelocityX', 'gradyVelocityX', 'gradzVelocityX',
                     'gradxVelocityY', 'gradyVelocityY', 'gradzVelocityY', 'gradxVelocityZ',
                     'gradyVelocityZ', 'gradzVelocityZ', 'Nu', 'PSEX', 'PSEY', 'PSEZ',
                     'QCriterion', 'Sigma', 'VelocityMagnitude', 'VelocityX', 'VelocityY',
                     'VelocityZ', 'Volume', 'VorticityMagnitude', 'VorticityX', 'VorticityY',
                     'VorticityZ']
    [C._initVars(ExctractionTree, fn, 0.) for fn in newFieldNames]
    ExtractionZones = I.getZones(ExctractionTree)
    NPtsPerZone = [0] + [C.getNPts(z) for z in ExtractionZones]
    tmpZone = D.line((0,0,0),(1,0,0), np.sum(NPtsPerZone))
    [C._initVars(tmpZone, fn, 0.) for fn in newFieldNames]
    coordst = J.getxyz(tmpZone)
    for i, zone in enumerate(ExtractionZones):
        coords = J.getxyz(zone)
        for ct, c in zip(coordst, coords):
            ct[NPtsPerZone[i]:NPtsPerZone[i+1]] = c.ravel(order = 'F')
    tmpTree = C.newPyTree(['Base',tmpZone])
    Kernel = Kernel_str2int[getParameter(t, 'RegularisationKernel')]
    EddyViscosityModel = EddyViscosityModel_str2int[getParameter(t, 'EddyViscosityModel')]
    vpm_cpp.wrap_extract_plane(t, tmpTree, int(NbOfParticlesUsedForPrecisionEvaluation), Kernel,
                                EddyViscosityModel)
    tmpFields = J.getVars(I.getZones(tmpTree)[0], newFieldNames)

    for i, zone in enumerate(ExtractionZones):
        fields = J.getVars(zone, newFieldNames)
        for ft, f in zip(tmpFields, fields):
            fr = f.ravel(order = 'F')
            fr[:] = ft[NPtsPerZone[i]:NPtsPerZone[i+1]]

    return ExctractionTree

def compute(Parameters, PolarsFilename, LiftingLinePath = 'LiftingLine.cgns', 
            DIRECTORY_OUTPUT = 'OUTPUT', NumberOfIterations = 1000, RestartPath = None,
            SaveVPMPeriod = 100, VisualisationOptions = {'addLiftingLineSurfaces':True},
            FieldsExtractionGrid = [], SaveFieldsPeriod = np.inf, Verbose = True,
            NumberOfSampleForStdDeviation = 50, SaveImageOptions = {}, SaveImagePeriod = np.inf,
            Surface = 0.,):
    try: os.makedirs(DIRECTORY_OUTPUT)
    except: pass

    AirfoilPolars = loadAirfoilPolars(PolarsFilename)
    if RestartPath:
        t = open(RestartPath)
        IterationInfo = {'Rel. err. of Velocity': 0, 'Rel. err. of Velocity Gradient': 0,
                            'Rel. err. of PSE': 0}
    else:
        LiftingLine = open(LiftingLinePath)
        t = initialise(LiftingLine, AirfoilPolars, **Parameters)
        IterationInfo = {}

    TotalTime = time()
    sp = getSolverParameters(t)
    Np = pickParticlesZone(t)[1][0]
    LiftingLines = MLL.getLiftingLines(t)

    h = sp['Resolution'][0]
    f = sp['RedistributionPeriod'][0]
    it = sp['CurrentIteration']
    dt = sp['TimeStep']
    U0 = np.linalg.norm(sp['VelocityFreestream'])
    simuTime = sp['Time']
    Ramp = sp['StrengthRampAtbeginning'][0]
    PSE = DiffusionScheme_str2int[sp['DiffusionScheme']] == 1
    NbLL = len(I.getZones(LiftingLines))
    VisualisationOptions['AirfoilPolarsFilename'] = PolarsFilename

    filename = os.path.join(DIRECTORY_OUTPUT,'VPM_AfterIt%d.cgns'%it[0])
    setVisualization(t, **VisualisationOptions)
    save(t,filename)
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
        
        IterationTime = time()
        computeNextTimeStep(t)

        IterationInfo['Iteration'] = it[0]
        IterationInfo['Percentage'] = it[0]/NumberOfIterations*100.
        IterationInfo['Physical time'] = simuTime[0]
        IterationInfo = populationControl(t, NoRedistributionRegions=[], IterationInfo = IterationInfo)
        IterationInfo = addParticlesFromLiftingLines(t, AirfoilPolars, IterationInfo = IterationInfo)
        IterationInfo['Number of Particles'] = Np[0]
        IterationInfo = solveVorticityEquation(t, IterationInfo = IterationInfo)
        IterationInfo['Total iteration time'] = time() - IterationTime
        if NbLL == 1: IterationInfo = getAerodynamicCoefficientsOnWing(LiftingLines, Surface,
                                        NumberOfSampleForStdDeviation = NumberOfSampleForStdDeviation,
                                        IterationInfo = IterationInfo)
        else:   
            if U0 == 0: IterationInfo = getAerodynamicCoefficientsOnRotor(LiftingLines,
                                        NumberOfSampleForStdDeviation = NumberOfSampleForStdDeviation,
                                        IterationInfo = IterationInfo)
            else: IterationInfo = getAerodynamicCoefficientsOnPropeller(LiftingLines,
                                    NumberOfSampleForStdDeviation = NumberOfSampleForStdDeviation,
                                    IterationInfo = IterationInfo)
        IterationInfo['Total simulation time'] = time() - TotalTime
        if Verbose: printIterationInfo(IterationInfo, PSE = PSE, Wings = (NbLL == 1))

        if (SAVE_FIELDS or SAVE_ALL) and FieldsExtractionGrid:
            extract(t, FieldsExtractionGrid, 5000)
            filename = os.path.join(DIRECTORY_OUTPUT, 'fields_AfterIt%d.cgns'%it)
            save(FieldsExtractionGrid, filename)
            J.createSymbolicLink(filename, 'fields.cgns')

        if SAVE_IMAGE or SAVE_ALL:
            setVisualization(t, **VisualisationOptions)
            saveImage(t, **SaveImageOptions)

        if SAVE_VPM or SAVE_ALL:
            setVisualization(t, **VisualisationOptions)
            filename = os.path.join(DIRECTORY_OUTPUT,'VPM_AfterIt%d.cgns'%it)
            save(t,filename)
            J.createSymbolicLink(filename,  DIRECTORY_OUTPUT + '.cgns')

        if CONVERGED: break

    setVisualization(t, **VisualisationOptions)
    filename = os.path.join(DIRECTORY_OUTPUT,'VPM_AfterIt%d.cgns'%it)
    save(t, DIRECTORY_OUTPUT + '.cgns')
    J.createSymbolicLink(filename,  DIRECTORY_OUTPUT + '.cgns')
    for _ in range(3): print('||' + '{:=^50}'.format(''))
    print('||' + '{:-^50}'.format(' End of VPM computation '))
    for _ in range(3): print('||' + '{:=^50}'.format(''))

    # exit()

def computePolar(Parameters, dictPolar, PolarsFilename, LiftingLinePath = 'LiftingLine.cgns',
                DIRECTORY_OUTPUT = 'POLARS', RestartPath = None, MaxNumberOfIterationsPerPolar = 200,
                NumberOfIterationsForTransition = 0, NumberOfSampleForStdDeviation = 10,
                MaxThrustStandardDeviation = 1, MaxPowerStandardDeviation = 100, 
                VisualisationOptions = {'addLiftingLineSurfaces':True},
                Verbose = True, Surface = 1.):

    try: os.makedirs(DIRECTORY_OUTPUT)
    except: pass

    AirfoilPolars = loadAirfoilPolars(PolarsFilename)
    if RestartPath:
        PolarsTree = open(RestartPath)
        Polars = I.getNodeFromName(PolarsTree, 'Polars')[2]
        OlddictPolar = {}
        for z in Polars:
            if type(z[1][0]) == np.bytes_:
                OlddictPolar[z[0]] = ''
                for zi in z[1]: OlddictPolar[z[0]] += zi.decode('UTF-8')
            else: OlddictPolar[z[0]] = z[1]
        
        for v in OlddictPolar['Variables']:
            i = 0
            while i < len(dictPolar['Variables']):
                if dictPolar['Variables'][i] == v:
                    dictPolar['Variables'] = np.delete(dictPolar['Variables'], i)
                    i -= 1
                i += 1

        dictPolar['Variables'] = np.append(OlddictPolar['Variables'], dictPolar['Variables'])
        OlddictPolar.update(dictPolar)
        dictPolar = OlddictPolar
        if dictPolar['VariableName'] == 'Pitch':
            dictPolar['VariableName'] = 'Twist'
            dictPolar['Pitch'] = True

        N0 = 0
        while (N0 < len(OlddictPolar['Variables']) and
                                    OlddictPolar['Variables'][N0] != OlddictPolar['OldVariable']):
            N0 += 1

        if N0 == len(OlddictPolar['Variables']): N0 = 0
        else: N0 += 1

        t = C.convertFile2PyTree(os.path.join(DIRECTORY_OUTPUT, dictPolar['LastPolar']))
    else:
        LiftingLine = open(LiftingLinePath)
        t = initialise(LiftingLine, AirfoilPolars, **Parameters)
        if dictPolar['VariableName'] == 'Pitch':
            dictPolar['VariableName'] = 'Twist'
            dictPolar['Pitch'] = True

        dictPolar['Offset'] = I.getValue(I.getNodeFromName(t, dictPolar['VariableName']))
        if dictPolar['overwriteVPMWithVariables']: dictPolar['Offset'] *= 0.

        if not (isinstance(dictPolar['Offset'], list) or isinstance(dictPolar['Offset'], np.ndarray)):
            dictPolar['Offset'] = np.array([dictPolar['Offset']])

        dictPolar['OldVariable'] = dictPolar['Variables'][0]
        if len(I.getZones(LiftingLine)) == 1:
            dictPolar['Lift'] = np.array([], dtype = np.float64)
            dictPolar['Drag'] = np.array([], dtype = np.float64)
            dictPolar['cL'] = np.array([], dtype = np.float64)
            dictPolar['cD'] = np.array([], dtype = np.float64)
            dictPolar['f'] = np.array([], dtype = np.float64)
            dictPolar['LiftStandardDeviation'] = np.array([], dtype = np.float64)
            dictPolar['DragStandardDeviation'] = np.array([], dtype = np.float64)
        else:
            dictPolar['Thrust'] = np.array([], dtype = np.float64)
            dictPolar['Power'] = np.array([], dtype = np.float64)
            dictPolar['cT'] = np.array([], dtype = np.float64)
            dictPolar['cP'] = np.array([], dtype = np.float64)
            dictPolar['Efficiency'] = np.array([], dtype = np.float64)
            dictPolar['ThrustStandardDeviation'] = np.array([], dtype = np.float64)
            dictPolar['PowerStandardDeviation'] = np.array([], dtype = np.float64)

        PolarsTree = C.newPyTree()
        J.set(PolarsTree, 'Polars', **dictPolar)
        N0 = 0

    sp = getSolverParameters(t)
    Particles = pickParticlesZone(t)
    Np = Particles[1][0]
    LiftingLines = MLL.getLiftingLines(t)

    h = sp['Resolution'][0]
    f = sp['RedistributionPeriod'][0]
    it = sp['CurrentIteration']
    dt = sp['TimeStep']
    U0 = np.linalg.norm(sp['VelocityFreestream'])
    simuTime = sp['Time']
    Ramp = sp['StrengthRampAtbeginning'][0]
    PSE = DiffusionScheme_str2int[sp['DiffusionScheme']] == 1
    NbLL = len(I.getZones(LiftingLines))
    VisualisationOptions['AirfoilPolarsFilename'] = PolarsFilename

    for _ in range(3): print('||' + '{:=^50}'.format(''))
    print('||' + '{:=^50}'.format(' Begin VPM Polar '))
    for _ in range(3): print('||' + '{:=^50}'.format(''))

    TotalTime = time()
    for i, Variable in enumerate(dictPolar['Variables'][N0:]):
        i += N0
        if NumberOfIterationsForTransition or N0 == i:
            for n in range(1, NumberOfIterationsForTransition + 1):
                computeNextTimeStep(t)

                NewVariable = dictPolar['OldVariable'] + (Variable - dictPolar['OldVariable'])\
                                                                *n/NumberOfIterationsForTransition
                for LiftingLine in LiftingLines:
                    LLVariable = I.getNodeFromName(LiftingLine, dictPolar['VariableName'])
                    LLVariable[1] = dictPolar['Offset'] + NewVariable

                VPMVariable = I.getNodeFromName(Particles, dictPolar['VariableName'])
                if VPMVariable != None: VPMVariable[1] = dictPolar['Offset'] + NewVariable
                if 'TimeStepFunction' in dictPolar:
                    TimeStepFunction_str2int[dictPolar['TimeStepFunction']](t, LiftingLines
                                                        , dictPolar['TimeStepFunctionParameter'])

                populationControl(t, NoRedistributionRegions=[])
                addParticlesFromLiftingLines(t, AirfoilPolars)
                solveVorticityEquation(t)

                if Verbose:
                    if n != 1: deletePrintedLines()
                    print('||' + '{:-^50}'.format(' Transition ' + \
                            '{:.1f}'.format(n/NumberOfIterationsForTransition*100.) + '% ') + ' \r')

        for LiftingLine in LiftingLines:
            LLVariable = I.getNodeFromName(LiftingLine, dictPolar['VariableName'])
            LLVariable[1] = dictPolar['Offset'] + Variable
        VPMVariable = I.getNodeFromName(Particles, dictPolar['VariableName'])
        if VPMVariable != None: VPMVariable[1] = dictPolar['Offset'] + Variable
        if 'TimeStepFunction' in dictPolar:
            TimeStepFunction_str2int[dictPolar['TimeStepFunction']](t, LiftingLines
                                                        , dictPolar['TimeStepFunctionParameter'])
        it0 = it[0]
        stdThrust = MaxThrustStandardDeviation + 1
        stdPower = MaxPowerStandardDeviation + 1

        while (it[0] - it0 < MaxNumberOfIterationsPerPolar and (MaxThrustStandardDeviation < stdThrust 
            or MaxPowerStandardDeviation < stdPower)) or (it[0] - it0 < NumberOfSampleForStdDeviation):
            msg = '||' + '{:-^50}'.format(' Iteration ' + '{:d}'.format(it[0] - it0) + ' ') + '\n'
            computeNextTimeStep(t)
            populationControl(t, NoRedistributionRegions=[])
            addParticlesFromLiftingLines(t, AirfoilPolars)
            solveVorticityEquation(t)

            msg += '||' + '{:-^50}'.format(' Loads ') + '\n'
            if NbLL == 1:
                IterationInfo = getAerodynamicCoefficientsOnWing(LiftingLines, Surface,
                                    NumberOfSampleForStdDeviation = NumberOfSampleForStdDeviation)
                msg += '||' + '{:34}'.format('Lift')
                msg += ': ' + '{:.3f}'.format(IterationInfo['Lift']) + ' N' + '\n'
                msg += '||' + '{:34}'.format('Lift Standard Deviation')
                msg += ': ' + '{:.3f}'.format(IterationInfo['Lift Standard Deviation']) + ' N' + '\n'
                msg += '||' + '{:34}'.format('Drag')
                msg += ': ' + '{:.3f}'.format(IterationInfo['Drag']) + ' N' + '\n'
                msg += '||' + '{:34}'.format('Drag Standard Deviation')
                msg += ': ' + '{:.3f}'.format(IterationInfo['Drag Standard Deviation']) + ' N' + '\n'
                msg += '||' + '{:34}'.format('cL') + ': ' + '{:.4f}'.format(IterationInfo['cL']) + '\n'
                msg += '||' + '{:34}'.format('cD') + ': ' + '{:.5f}'.format(IterationInfo['cD']) + '\n'
                msg += '||' + '{:34}'.format('f') + ': ' + '{:.4f}'.format(IterationInfo['f']) + '\n'
                stdThrust = IterationInfo['Lift Standard Deviation']
                stdPower = IterationInfo['Drag Standard Deviation']
            else:
                U0 = np.linalg.norm(I.getValue(I.getNodeFromName(LiftingLines, 'VelocityFreestream')))
                if U0 == 0:
                    IterationInfo = getAerodynamicCoefficientsOnRotor(LiftingLines,
                                        NumberOfSampleForStdDeviation = NumberOfSampleForStdDeviation)
                else:
                    IterationInfo = getAerodynamicCoefficientsOnPropeller(LiftingLines,
                                        NumberOfSampleForStdDeviation = NumberOfSampleForStdDeviation)
                msg += '||' + '{:34}'.format('Thrust')
                msg += ': ' + '{:.3f}'.format(IterationInfo['Thrust']) + ' N' + '\n'
                msg += '||' + '{:34}'.format('Thrust Standard Deviation')
                msg += ': ' + '{:.3f}'.format(IterationInfo['Thrust Standard Deviation']) + ' N' + '\n'
                msg += '||' + '{:34}'.format('Power')
                msg += ': ' + '{:.3f}'.format(IterationInfo['Power']) + ' W' + '\n'
                msg += '||' + '{:34}'.format('Power Standard Deviation')
                msg += ': ' + '{:.3f}'.format(IterationInfo['Power Standard Deviation']) + ' W' + '\n'
                msg += '||' + '{:34}'.format('cT') + ': ' + '{:.5f}'.format(IterationInfo['cT']) + '\n'
                msg += '||' + '{:34}'.format('Cp') + ': ' + '{:.5f}'.format(IterationInfo['cP']) + '\n'
                msg += '||' + '{:34}'.format('Eff') + ': ' + '{:.5f}'.format(IterationInfo['Eff']) + '\n'
                stdThrust = IterationInfo['Thrust Standard Deviation']
                stdPower = IterationInfo['Power Standard Deviation']
            
            if Verbose:
                if it[0] != it0 + 1: deletePrintedLines(10)
                print(msg)

        dictPolar['OldVariable'] = Variable
        if NbLL == 1:
            dictPolar['Lift'] = np.append(dictPolar['Lift'], IterationInfo['Lift'])
            dictPolar['Drag'] = np.append(dictPolar['Drag'], IterationInfo['Drag'])
            dictPolar['cL'] = np.append(dictPolar['cL'], IterationInfo['cL'])
            dictPolar['cD'] = np.append(dictPolar['cD'], IterationInfo['cD'])
            dictPolar['f'] = np.append(dictPolar['f'], IterationInfo['f'])
            dictPolar['LiftStandardDeviation'] = np.append(dictPolar['LiftStandardDeviation']
                                                        , IterationInfo['Lift Standard Deviation'])
            dictPolar['DragStandardDeviation'] = np.append(dictPolar['DragStandardDeviation']
                                                        , IterationInfo['Drag Standard Deviation'])
        else:
            dictPolar['Thrust'] = np.append(dictPolar['Thrust'], IterationInfo['Thrust'])
            dictPolar['Power'] = np.append(dictPolar['Power'], IterationInfo['Power'])
            dictPolar['cT'] = np.append(dictPolar['cT'], IterationInfo['cT'])
            dictPolar['cP'] = np.append(dictPolar['cP'], IterationInfo['cP'])
            dictPolar['Efficiency'] = np.append(dictPolar['Efficiency'], IterationInfo['Eff'])
            dictPolar['ThrustStandardDeviation'] = np.append(dictPolar['ThrustStandardDeviation']
                                                        , IterationInfo['Thrust Standard Deviation'])
            dictPolar['PowerStandardDeviation'] = np.append(dictPolar['PowerStandardDeviation']
                                                        , IterationInfo['Power Standard Deviation'])

        if Verbose:
            deletePrintedLines(1)
            print('||' + '{:-^50}'.format(''))

        if (it[0] - it0 == MaxNumberOfIterationsPerPolar):
            msg = ' Maximum number of iteration reached for '
        else:
            msg = ' Convergence criteria met for '

        if 'Pitch' not in dictPolar:
            dictPolar['LastPolar'] = 'VPM_Polar_' + dictPolar['VariableName'] + '_' + str(round(Variable, 2)) + '.cgns'
            filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_Polars_' + dictPolar['VariableName'] + '.cgns')
            msg += dictPolar['VariableName'] + ' = ' + str(round(Variable, 2)) + ' '
        else:
            dictPolar['LastPolar'] = 'VPM_Polar_Pitch_' + str(round(Variable, 2)) + '.cgns'
            filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_Polars_Pitch.cgns')
            msg += 'Pitch = ' + str(round(Variable, 2)) + ' '
        if Verbose: print('||' + '{:-^50}'.format(msg))

        J.set(PolarsTree, 'Polars', **dictPolar)
        save(PolarsTree, filename)

        filename = os.path.join(DIRECTORY_OUTPUT, dictPolar['LastPolar'])
        setVisualization(t, **VisualisationOptions)
        save(I.merge([PolarsTree, t]), filename)
        J.createSymbolicLink(filename,  DIRECTORY_OUTPUT + '.cgns')
        
        if Verbose:
            print('||' + '{:=^50}'.format('') + '\n||' + '{:=^50}'.format('') +\
                    '\n||' + '{:=^50}'.format(''))

    
    TotalTime = time() - TotalTime
    print('||' + '{:=^50}'.format(' Total time spent: ' + str(int(round(TotalTime//60))) + ' min ' +\
             str(int(round(TotalTime - TotalTime//60*60))) + ' s '))
    for _ in range(3): print('||' + '{:=^50}'.format(''))


def getAerodynamicCoefficientsOnPropeller(LiftingLines, NumberOfSampleForStdDeviation = 50,
                                                IterationInfo = {}):
    LiftingLine = I.getZones(LiftingLines)[0]
    RotationCenter = I.getValue(I.getNodeFromName(LiftingLine, 'RotationCenter'))
    RPM = I.getValue(I.getNodeFromName(LiftingLine, 'RPM'))
    Rho = I.getValue(I.getNodeFromName(LiftingLine, 'Density'))
    U0 = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityFreestream'))
    V = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityTranslation'))
    x, y, z = J.getxyz(LiftingLine)
    D = 2*max((x - RotationCenter[0])**2 + (y - RotationCenter[1])**2 + (z - RotationCenter[2])**2)**0.5
    IntegralLoads = MLL.computeGeneralLoadsOfLiftingLine(LiftingLines)
    n   = RPM/60.
    T = IntegralLoads['Total']['Thrust'][0]
    P = IntegralLoads['Total']['Power'][0]
    cT = T/(Rho*n**2*D**4)
    cP = P/(Rho*n**3*D**5)
    Uinf = np.linalg.norm(U0 - V)
    Eff = T/P*Uinf
    std_Thrust, std_Power = getStandardDeviationBlade(LiftingLines = LiftingLines, 
                                NumberOfSampleForStdDeviation = NumberOfSampleForStdDeviation)
    IterationInfo['Thrust'] = T
    IterationInfo['Thrust Standard Deviation'] = std_Thrust
    IterationInfo['Power'] = P
    IterationInfo['Power Standard Deviation'] = std_Power
    IterationInfo['cT'] = cT
    IterationInfo['cP'] = cP
    IterationInfo['Eff'] = Eff
    return IterationInfo

def getAerodynamicCoefficientsOnRotor(LiftingLines, NumberOfSampleForStdDeviation = 50,
                                        IterationInfo = {}):
    LiftingLine = I.getZones(LiftingLines)[0]
    RotationCenter = I.getValue(I.getNodeFromName(LiftingLine, 'RotationCenter'))
    RPM = I.getValue(I.getNodeFromName(LiftingLine, 'RPM'))
    Rho = I.getValue(I.getNodeFromName(LiftingLine, 'Density'))
    x, y, z = J.getxyz(LiftingLine)
    D = 2*max((x - RotationCenter[0])**2 + (y - RotationCenter[1])**2 +\
                (z - RotationCenter[2])**2)**0.5
    IntegralLoads = MLL.computeGeneralLoadsOfLiftingLine(LiftingLines)
    n   = RPM/60.
    T = IntegralLoads['Total']['Thrust'][0]
    P = IntegralLoads['Total']['Power'][0]
    cT = T/(Rho*n**2*D**4)
    cP = P/(Rho*n**3*D**5)
    Eff = np.sqrt(2./np.pi)*cT**1.5/cP

    std_Thrust, std_Power = getStandardDeviationBlade(LiftingLines = LiftingLines,
                                    NumberOfSampleForStdDeviation = NumberOfSampleForStdDeviation)
    IterationInfo['Thrust'] = T
    IterationInfo['Thrust Standard Deviation'] = std_Thrust
    IterationInfo['Power'] = P
    IterationInfo['Power Standard Deviation'] = std_Power
    IterationInfo['cT'] = cT
    IterationInfo['cP'] = cP
    IterationInfo['Eff'] = Eff
    return IterationInfo

def getAerodynamicCoefficientsOnWing(LiftingLines, Surface, NumberOfSampleForStdDeviation = 50,
                                        IterationInfo = {}):
    LiftingLine = I.getZones(LiftingLines)[0]
    Rho = I.getValue(I.getNodeFromName(LiftingLine, 'Density'))
    U0 = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityFreestream'))
    V = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityTranslation'))
    x, y, z = J.getxyz(LiftingLine)
    IntegralLoads = MLL.computeGeneralLoadsOfLiftingLine(LiftingLine)
    Rmax = (x[0]**2 + y[0]**2 + z[0]**2)**0.5
    Rmin = (x[1]**2 + y[1]**2 + z[1]**2)**0.5
    Fz = IntegralLoads['ForceZ'][0]
    Fx = IntegralLoads['ForceX'][0]
    q0 = 0.5*Rho*Surface*np.linalg.norm(U0 - V)**2
    cL = Fz/q0
    cD = Fx/q0
    std_Thrust, std_Drag = getStandardDeviationWing(LiftingLines = LiftingLines,
                                    NumberOfSampleForStdDeviation = NumberOfSampleForStdDeviation)
    IterationInfo['Lift'] = Fz
    IterationInfo['Lift Standard Deviation'] = std_Thrust
    IterationInfo['Drag'] = Fx
    IterationInfo['Drag Standard Deviation'] = std_Drag
    IterationInfo['cL'] = cL
    IterationInfo['cD'] = cD
    IterationInfo['f'] = cL/cD
    return IterationInfo

def getStandardDeviationWing(LiftingLines, NumberOfSampleForStdDeviation = 50):
    LiftingLine = I.getZones(LiftingLines)[0]
    UnsteadyLoads = I.getNodeFromName(LiftingLine, '.UnsteadyLoads')
    NumberOfSampleForStdDeviation = min(NumberOfSampleForStdDeviation,
                                        len(I.getValue(I.getNodeFromName(UnsteadyLoads, 'Thrust'))))
    Thrust = np.array([0.]*NumberOfSampleForStdDeviation)
    Drag = np.array([0.]*NumberOfSampleForStdDeviation)
    for LiftingLine in LiftingLines:
        UnsteadyLoads = I.getNodeFromName(LiftingLine, '.UnsteadyLoads')
        Thrust += I.getValue(I.getNodeFromName(UnsteadyLoads, 'ForceZ'))[-NumberOfSampleForStdDeviation:]
        Drag += I.getValue(I.getNodeFromName(UnsteadyLoads, 'ForceX'))[-NumberOfSampleForStdDeviation:]
    meanThrust = sum(Thrust)/NumberOfSampleForStdDeviation
    meanDrag = sum(Drag)/NumberOfSampleForStdDeviation

    std_Thrust = (sum((Thrust - meanThrust)**2)/NumberOfSampleForStdDeviation)**0.5
    std_Drag = (sum((Drag - meanDrag)**2)/NumberOfSampleForStdDeviation)**0.5
    return std_Thrust, std_Drag

def getStandardDeviationBlade(LiftingLines, NumberOfSampleForStdDeviation = 50):
    LiftingLine = I.getZones(LiftingLines)[0]
    UnsteadyLoads = I.getNodeFromName(LiftingLine, '.UnsteadyLoads')
    NumberOfSampleForStdDeviation = min(NumberOfSampleForStdDeviation,
                                        len(I.getValue(I.getNodeFromName(UnsteadyLoads, 'Thrust'))))
    Thrust = np.array([0.]*NumberOfSampleForStdDeviation)
    Power = np.array([0.]*NumberOfSampleForStdDeviation)
    for LiftingLine in LiftingLines:
        UnsteadyLoads = I.getNodeFromName(LiftingLine, '.UnsteadyLoads')
        Thrust += I.getValue(I.getNodeFromName(UnsteadyLoads, 'Thrust'))[-NumberOfSampleForStdDeviation:]
        Power += I.getValue(I.getNodeFromName(UnsteadyLoads, 'Power'))[-NumberOfSampleForStdDeviation:]
    meanThrust = sum(Thrust)/NumberOfSampleForStdDeviation
    meanPower = sum(Power)/NumberOfSampleForStdDeviation

    stdThrust = (sum((Thrust - meanThrust)**2)/NumberOfSampleForStdDeviation)**0.5
    stdPower = (sum((Power - meanPower)**2)/NumberOfSampleForStdDeviation)**0.5
    return stdThrust, stdPower

def setMinNbShedParticlesPerLiftingLine(LiftingLines, Parameters,
                                                        NumberParticlesShedAtTip = 5):
    MLL.computeKinematicVelocity(LiftingLines)
    MLL.assembleAndProjectVelocities(LiftingLines)
    Urelmax = 0.
    L = 0.
    flag = False
    if 'VelocityFreestream' not in defaultParameters: flag = True
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
                                                                    Urel/Paramters['TimeStep']))
    return Paramters

def printIterationInfo(IterationInfo, PSE = False, Wings = False):
    msg = '||' + '{:-^50}'.format(' Iteration ' + '{:d}'.format(IterationInfo['Iteration']) +\
          ' (' + '{:.1f}'.format(IterationInfo['Percentage']) + '%) ') + '\n'
    msg += '||' + '{:34}'.format('Physical time') + ': ' +\
           '{:.5f}'.format(IterationInfo['Physical time']) + ' s' + '\n'
    msg += '||' + '{:34}'.format('Number of Particles') + ': ' +\
           '{:d}'.format(IterationInfo['Number of Particles']) + '\n'
    msg += '||' + '{:34}'.format('Total iteration time') + ': ' +\
           '{:.2f}'.format(IterationInfo['Total iteration time']) + ' s' + '\n'
    msg += '||' + '{:34}'.format('Total simulation time') + ': ' +\
           '{:.1f}'.format(IterationInfo['Total simulation time']) + ' s' + '\n'
    msg += '||' + '{:-^50}'.format(' Loads ') + '\n'
    if (Wings):
        msg += '||' + '{:34}'.format('Lift') + ': ' + '{:g}'.format(IterationInfo['Lift']) +\
               ' N' + '\n'
        msg += '||' + '{:34}'.format('Lift Standard Deviation') + ': ' +\
               '{:g}'.format(IterationInfo['Lift Standard Deviation']) + ' N' + '\n'
        msg += '||' + '{:34}'.format('Drag') + ': ' + '{:g}'.format(IterationInfo['Drag']) +\
               ' N' + '\n'
        msg += '||' + '{:34}'.format('Drag Standard Deviation') + ': ' +\
               '{:g}'.format(IterationInfo['Drag Standard Deviation']) + ' N' + '\n'
        msg += '||' + '{:34}'.format('cL')     + ': ' + '{:g}'.format(IterationInfo['cL']) + '\n'
        msg += '||' + '{:34}'.format('cD')     + ': ' + '{:g}'.format(IterationInfo['cD']) + '\n'
        msg += '||' + '{:34}'.format('f')    + ': ' + '{:g}'.format(IterationInfo['f']) + '\n'
    else:
        msg += '||' + '{:34}'.format('Thrust') + ': ' + '{:g}'.format(IterationInfo['Thrust']) +\
               ' N' + '\n'
        msg += '||' + '{:34}'.format('Thrust Standard Deviation') + ': ' +\
               '{:g}'.format(IterationInfo['Thrust Standard Deviation']) + ' N' + '\n'
        msg += '||' + '{:34}'.format('Power') + ': ' + '{:g}'.format(IterationInfo['Power']) +\
               ' W' + '\n'
        msg += '||' + '{:34}'.format('Power Standard Deviation') + ': ' +\
               '{:g}'.format(IterationInfo['Power Standard Deviation']) + ' W' + '\n'
        msg += '||' + '{:34}'.format('cT')     + ': ' + '{:g}'.format(IterationInfo['cT']) + '\n'
        msg += '||' + '{:34}'.format('Cp')     + ': ' + '{:g}'.format(IterationInfo['cP']) + '\n'
        msg += '||' + '{:34}'.format('Eff')    + ': ' + '{:g}'.format(IterationInfo['Eff']) + '\n'
    msg += '||' + '{:-^50}'.format(' Population Control ') + '\n'
    msg += '||' + '{:34}'.format('Number of particles beyond cutoff') + ': ' +\
           '{:d}'.format(IterationInfo['Number of particles beyond cutoff']) + '\n'
    msg += '||' + '{:34}'.format('Number of split particles') + ': ' +\
           '{:d}'.format(IterationInfo['Number of split particles']) + '\n'
    msg += '||' + '{:34}'.format('Number of depleted particles') + ': ' +\
           '{:d}'.format(IterationInfo['Number of depleted particles']) + '\n'
    msg += '||' + '{:34}'.format('Number of merged particles') + ': ' +\
           '{:d}'.format(IterationInfo['Number of merged particles']) + '\n'
    msg += '||' + '{:34}'.format('Control Computation time')  + ': ' +\
           '{:.2f}'.format(IterationInfo['Population Control time']) + ' s (' +\
           '{:.1f}'.format(IterationInfo['Population Control time']/\
            IterationInfo['Total iteration time']*100.) + '%) ' + '\n'
    msg += '||' + '{:-^50}'.format(' Lifting Line ') + '\n'
    msg += '||' + '{:34}'.format('Circulation Error') + ': ' +\
           '{:.5e}'.format(IterationInfo['Circulation Error']) + '\n'
    msg += '||' + '{:34}'.format('Number of sub-iterations') + ': ' +\
           '{:d}'.format(IterationInfo['Number of sub-iterations']) + '\n'
    msg += '||' + '{:34}'.format('Number of shed particles')  + ': ' +\
           '{:d}'.format(IterationInfo['Number of shed particles']) + '\n'
    msg += '||' + '{:34}'.format('Lifting Line Computation time')  + ': ' +\
           '{:.2f}'.format(IterationInfo['Lifting Line time']) + ' s (' +\
           '{:.1f}'.format(IterationInfo['Lifting Line time']/\
            IterationInfo['Total iteration time']*100.) + '%) ' + '\n'
    msg += '||' + '{:-^50}'.format(' FMM ') + '\n'
    msg += '||' + '{:34}'.format('Number of threads') + ': ' +\
           '{:d}'.format(IterationInfo['Number of threads']) + '\n'
    msg += '||' + '{:34}'.format('SIMD vectorisation') + ': ' +\
           '{:d}'.format(IterationInfo['SIMD vectorisation']) + '\n'
    msg += '||' + '{:34}'.format('Near field overlapping ratio') + ': ' +\
           '{:.2f}'.format(IterationInfo['Near field overlapping ratio']) + '\n'
    msg += '||' + '{:34}'.format('Far field polynomial order')  + ': ' +\
           '{:d}'.format(IterationInfo['Far field polynomial order']) + '\n'
    msg += '||' + '{:34}'.format('Rel. err. of Velocity')  + ': ' +\
           '{:e}'.format(IterationInfo['Rel. err. of Velocity']) + '\n'
    msg += '||' + '{:34}'.format('Rel. err. of Velocity Gradient')  + ': ' +\
           '{:e}'.format(IterationInfo['Rel. err. of Velocity Gradient']) + '\n'
    if PSE: msg += '||' + '{:34}'.format('Rel. err. of PSE')  + ': ' +\
           '{:e}'.format(IterationInfo['Rel. err. of PSE']) + '\n'
    msg += '||' + '{:34}'.format('FMM Computation time')  + ': ' +\
           '{:.2f}'.format(IterationInfo['FMM time']) + ' s (' +\
           '{:.1f}'.format(IterationInfo['FMM time']/\
            IterationInfo['Total iteration time']*100.) + '%) ' + '\n'
    msg += '||' + '{:=^50}'.format('')
    print(msg)

def deletePrintedLines(NumberOfLineToDelete = 1):
    for i in range(NumberOfLineToDelete):
        sys.stdout.write('\x1b[1A')
        sys.stdout.write('\x1b[2K')