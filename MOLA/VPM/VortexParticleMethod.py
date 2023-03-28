import sys
import vortexparticlemethod as VPM
import os
import numpy as np
import Converter.PyTree as C
import Geom.PyTree as D
import Converter.Internal as I
import Generator.PyTree as G
import Transform.PyTree as T
import CPlot.PyTree as CPlot
import Connector.PyTree as CX
import Post.PyTree as P
from .. import LiftingLine as LL
from .. import Wireframe as W
from .. import InternalShortcuts as J
from .. import ExtractSurfacesProcessor as ESP

__version__ = '0.2'
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
                                                          number of Particles (%d)'%(len(mask), Np))
        mask = np.logical_not(mask)
        for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
            if node[3] == 'DataArray_t':
                node[1] = node[1][mask]

        Particles[1].ravel(order = 'F')[0] = len(node[1])

    def extend(t = [], ExtendSize = 0, Offset = 0, ExtendAtTheEnd = True):
        Particles = pickParticlesZone(t)
        GridCoordinatesNode = I.getNodeFromName1(Particles, 'GridCoordinates')
        FlowSolutionNode = I.getNodeFromName1(Particles, 'FlowSolution')
        Np = len(J.getx(Particles))
        if Np < abs(Offset): raise ValueError('Offset (%d) cannot be greater than existing number \
                                                                    of Particles (%d)'%(Offset, Np))
        if ExtendAtTheEnd:
            for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
                if node[3] == 'DataArray_t':
                    zeros = np.zeros(ExtendSize, dtype = node[1].dtype, order = 'F') + (node[0] == \
                                                                                           'Active')
                    node[1] = np.append(np.append(node[1][:Np -Offset], zeros), node[1][Np-Offset:])

        else:
            for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
                if node[3] == 'DataArray_t':
                    zeros = np.zeros(ExtendSize, dtype = node[1].dtype, order = 'F') + (node[0] == \
                                                                                           'Active')
                    node[1] = np.append(np.append(node[1][:Offset], zeros), node[1][Offset:])

        Particles[1].ravel(order = 'F')[0] = len(node[1])

    def trim(t = [], NumberToTrim = 0, Offset = 0, TrimAtTheEnd = True):
        Particles = pickParticlesZone(t)
        GridCoordinatesNode = I.getNodeFromName1(Particles, 'GridCoordinates')
        FlowSolutionNode = I.getNodeFromName1(Particles, 'FlowSolution')
        Np = len(J.getx(Particles))
        if Np < abs(Offset): raise ValueError('Offset (%d) cannot be greater than existing number \
                                                                    of Particles (%d)'%(Offset, Np))
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
        NewSize = NewSize - OldSize

        if 0 < NewSize: extend(t, NewSize, Offset = Offset, ExtendAtTheEnd = AtTheEnd)
        else: trim(t, -NewSize, Offset = Offset, TrimAtTheEnd = AtTheEnd)

    def roll(t = [], PivotNumber = 0):
        Particles = pickParticlesZone(t)
        Np = Particles[1].ravel(order = 'F')[0]
        if PivotNumber == 0 or PivotNumber == Np: return
        if PivotNumber > Np:
            raise ValueError('PivotNumber (%d) cannot be greater than existing number of Particles\
                                                                            (%d)'%(PivotNumber, Np))
        GridCoordinatesNode = I.getNodeFromName1(Particles, 'GridCoordinates')
        FlowSolutionNode = I.getNodeFromName1(Particles, 'FlowSolution')
        for node in GridCoordinatesNode[2] + FlowSolutionNode[2]:
            if node[3] == 'DataArray_t':
                node[1] = np.roll(node[1], PivotNumber)

    def checkParametersTypes(ParametersList = [], int_Params = [], float_Params = [], bool_Params = []):
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

    def getParameter(t = [], Name = ''):
        Particles = pickParticlesZone(t)
        Node = getVPMParameters(Particles)
        if Name in Node: ParameterNode = Node[Name]
        else :ParameterNode = None
        if not ParameterNode:
            Node = getLiftingLineParameters(Particles)
            if Node and Name in Node: ParameterNode = Node[Name]
        if not ParameterNode:
            Node = getHybridParameters(Particles)
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
    DiffusionScheme_str2int = {'PSE': 1, 'ParticleStrengthExchange': 1, 'pse': 1, 'CSM': 2, 'CS': 2,
        'csm': 2, 'cs': 2, 'CoreSpreading': 2, 'CoreSpreadingMethod': 2, 'None': 0, None: 0}

    def buildEmptyVPMTree():
        Particles = C.convertArray2Node(D.line((0., 0., 0.), (0., 0., 0.), 2))
        Particles[0] = 'Particles'


        FieldNames = ['VelocityInduced' + v for v in 'XYZ'] + ['Vorticity' + v for v in 'XYZ'] +\
                        ['Alpha' + v for v in 'XYZ'] + ['gradxVelocity' + v for v in 'XYZ'] + \
                        ['gradyVelocity' + v for v in 'XYZ']+ ['gradzVelocity' + v for v in 'XYZ']+\
                        ['PSE' + v for v in 'XYZ'] + ['Stretching' + v for v in 'XYZ'] + ['Active',
                        'Age', 'HybridFlag', 'Nu', 'Sigma', 'StrengthMagnitude', 'Volume',
                                                                               'VorticityMagnitude']
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

    def initialiseVPM(EulerianMesh = [], LiftingLineTree = [], PolarInterpolator = [], VPMParameters = {}, HybridParameters = {}, LiftingLineParameters = {}):
        int_Params =['MaxLiftingLineSubIterations', 'MaximumSubIteration','StrengthRampAtbeginning', 
            'MinNbShedParticlesPerLiftingLine', 'CurrentIteration', 'NumberOfHybridInterfaces',
            'MaximumAgeAllowed', 'RedistributionPeriod', 'NumberOfThreads', 'IntegrationOrder',
            'IterationTuningFMM', 'IterationCounter', 'ShedParticlesIndex', 'OuterInterfaceCell',
            'FarFieldApproximationOrder', 'NumberOfParticlesPerInterface', 'NumberOfSources']

        float_Params = ['Density', 'EddyViscosityConstant', 'Temperature', 'ResizeParticleFactor',
            'Time', 'CutoffXmin', 'CutoffZmin', 'MaximumMergingVorticityFactor', 'RelaxationCutoff',
            'SFSContribution', 'SmoothingRatio', 'CirculationThreshold', 'RPM','KinematicViscosity',
            'CirculationRelaxation', 'Pitch', 'CutoffXmax', 'CutoffYmin', 'CutoffYmax', 'Sigma0',
            'CutoffZmax', 'ForcedDissipation','MaximumAngleForMerging', 'MinimumVorticityFactor', 
            'RelaxationFactor', 'MinimumOverlapForMerging', 'VelocityFreestream', 'AntiStretching',
            'RelaxationThreshold', 'RedistributeParticlesBeyond', 'RedistributeParticleSizeFactor',
            'TimeStep', 'Resolution', 'VelocityTranslation', 'NearFieldOverlappingRatio', 'TimeFMM',
            'RemoveWeakParticlesBeyond', 'OuterDomainToWallDistance', 'InnerDomainToWallDistance']

        bool_Params = ['GammaZeroAtRoot', 'GammaZeroAtTip','MonitorInvariants','GhostParticleAtTip',
            'LowStorageIntegration', 'GhostParticleAtRoot']

        defaultParameters = {
            ############################################################################################
            ################################### Simulation conditions ##################################
            ############################################################################################
                'Density'                       : 1.225,          #in kg.m^-3, 1.01325e5/((273.15 + 15)*287.05)
                'EddyViscosityConstant'         : 0.1,            #constant for the eddy viscosity model, Cm(Mansour) around 0.1, Cs(Smagorinsky) around 0.15, Cr(Vreman) around 0.7
                'EddyViscosityModel'            : 'Vreman',       #Mansour, Smagorinsky, Vreman or None, select a LES model to compute the eddy viscosity
                'KinematicViscosity'            : 1.46e-5,        #in m^2.s^-1, kinematic viscosity, TODO must be inferred from atmospheric conditions
                'Temperature'                   : 288.15,         #in K, 273.15 + 15.
                'Time'                          : 0.,             #in s, keep track of the physical time
            ############################################################################################
            ###################################### VPM parameters ######################################
            ############################################################################################
                'AntiStretching'                : 0.,             #between 0 and 1, 0 means Particle strength fully takes vortex stretching, 1 means the Particle size fully takes the vortex stretching
                'DiffusionScheme'               : 'PSE',          #PSE, CSM or None. gives the scheme used to compute the diffusion term of the vorticity equation
                'RegularisationKernel'          : 'Gaussian',     #The available smoothing kernels are Gaussian, HOA, LOA, Gaussian3 and SuperGaussian
                'SFSContribution'               : 0.,             #between 0 and 1, the closer to 0, the more the viscosity affects the Particle strength, the closer to 1, the more it affects the Particle size
                'SmoothingRatio'                : 2.,             #in m, anywhere between 1.5 and 2.5, the higher the NumberSource, the smaller the Resolution and the higher the SmoothingRatio should be to avoid blowups, the HOA kernel requires a higher smoothing
                'VorticityEquationScheme'       : 'Transpose',    #Classical, Transpose or Mixed, The schemes used to compute the vorticity equation are the classical scheme, the transpose scheme (conserves total vorticity) and the mixed scheme (a fusion of the previous two)
            ############################################################################################
            ################################### Numerical Parameters ###################################
            ############################################################################################
                'CurrentIteration'              : 0,              #follows the current iteration
                'IntegrationOrder'              : 3,              #[|1, 4|]1st, 2nd, 3rd or 4th order Runge Kutta. In the hybrid case, there must be at least as much Interfaces in the hybrid domain as the IntegrationOrder of the time integration scheme
                'LowStorageIntegration'         : True,           #[|0, 1|], states if the classical or the low-storage Runge Kutta is used
                'MonitorInvariants'             : False,          #must be linked with the invariants function
            ############################################################################################
            ##################################### Particles Control ####################################
            ############################################################################################
                'CutoffXmin'                    : -np.inf,        #in m, spatial Cutoff
                'CutoffXmax'                    : +np.inf,        #in m, spatial Cutoff
                'CutoffYmin'                    : -np.inf,        #in m, spatial Cutoff
                'CutoffYmax'                    : +np.inf,        #in m, spatial Cutoff
                'CutoffZmin'                    : -np.inf,        #in m, spatial Cutoff
                'CutoffZmax'                    : +np.inf,        #in m, spatial Cutoff
                'ForcedDissipation'             : 0.,             #in %/s, gives the % of strength Particles looses per sec, usefull to kill unnecessary Particles without affecting the LLs
                'MaximumAgeAllowed'             : 0,              #0 <=,  Particles are eliminated after MaximumAgeAllowed iterations, if MaximumAgeAllowed == 0, they are not deleted
                'MaximumAngleForMerging'        : 0.,             #[0., 180.[ in deg,   maximum angle   allowed between two Particles to be merged
                'MaximumMergingVorticityFactor' : 0.,             #in %, Particles can be merged if their combined strength is below the given poucentage of the maximum strength on the blades
                'MinimumOverlapForMerging'      : 0.,             #[0., +inf[, if two Particles have at least an overlap of MinimumOverlapForMerging*SigmaRatio, they are considered for merging
                'MinimumVorticityFactor'        : 0.,             #in %, sets the minimum strength kept as a percentage of the maximum strength on the blades
                'RedistributeParticlesBeyond'   : np.inf,         #do not redistribute Particles if closer than RedistributeParticlesBeyond*Resolution from a LL
                'RedistributionKernel'          : None,           #M4Prime, M4, M3, M2, M1 or None, redistribution kernel used. the number gives the order preserved by the kernel, if None local splitting/merging is used
                'RedistributionPeriod'          : 0,              #frequency at which Particles are redistributed, if 0 the Particles are never redistributed
                'RelaxationCutoff'              : 0.,             #in Hz, is used during the relaxation process to realign the Particles with the vorticity
                'RemoveWeakParticlesBeyond'     : np.inf,         #do not remove weak Particles if closer than RemoveWeakParticlesBeyond*Resolution from a LL
                'ResizeParticleFactor'          : 0.,             #[0, +inf[, resize Particles that grow/shrink RedistributeParticleSizeFactor * Sigma0 (i.e. Resolution*SmoothingRatio), if 0 then no resizing is done
                'StrengthRampAtbeginning'       : 25,             #[|0, +inf [|, limit the vorticity shed for the StrengthRampAtbeginning first iterations for the wake to stabilise
            ############################################################################################
            ###################################### FMM parameters ######################################
            ############################################################################################
                'FarFieldApproximationOrder'    : 8,              #[|6, 12|], order of the polynomial which approximates the far field interactions, the higher the more accurate and the more costly
                'IterationTuningFMM'            : 50,             #frequency at which the FMM is compared to the direct computation, gives the relative L2 error
                'NearFieldOverlappingRatio'     : 0.5,            #[0., 1.], Direct computation of the interactions between clusters that overlap by NearFieldOverlappingRatio, the smaller the more accurate and the more costly
                'NumberOfThreads'               : "`nproc --all`",#number of threads of the machine used, does not matter if above the total number of threads of the machine, just slows down the simulation
                'TimeFMM'                       : 0.,             #in s, keep track of the CPU time spent for the FMM
        }
        defaultHybridParameters = {
            ############################################################################################
            ################################ Hybrid Domain parameters ################################
            ############################################################################################
                'BCFarFieldName'                   : 'farfield',#the name of the farfield boundary condition from which the Outer Interface is searched
                'MaximumSubIteration'              : 100,       #[|0, +inf[|, gives the maximum number of sub-iterations when computing the strength of the Particles generated from the vorticity on the Interfaces
                'NumberOfHybridInterfaces'         : 4.,        #|]0, +inf[|, number of interfaces in the Hybrid Domain from which hybrid particles are generated
                'OuterDomainToWallDistance'        : 0.3,       #]0, +inf[ in m, distance between the wall and the end of the Hybrid Domain
                'OuterInterfaceCell'               : 0,         #[|0, +inf[|, the Outer Interface is searched starting at the OuterInterfaceCell cell from the given BCFarFieldName, one row of cells at a time, until OuterDomainToWallDistance is reached
                'NumberOfParticlesPerInterface'    : 300,      #[|0, +inf[|, number of particles generated per hybrid interface
                'RelaxationFactor'                 : 0.5,       #[0, +inf[, gives the relaxtion factor used for the relaxation process when computing the strength of the Particles generated from the vorticity on the Interface
                'RelaxationThreshold'              : 1e-6,      #[0, +inf[ in m^3.s^-1, gives the convergence criteria for the relaxtion process when computing the strength of the Particles generated from the vorticity on the Interface
        }
        defaultLiftingLineParameters = {
            ############################################################################################
            ################################# Lifting Lines parameters #################################
            ############################################################################################
                'CirculationThreshold'             : 1e-4,                     #convergence criteria for the circulation sub-iteration process, somewhere between 1e-3 and 1e-6 is ok
                'CirculationRelaxation'            : 1./5.,                    #relaxation parameter of the circulation sub-iterations, somwhere between 0.1 and 1 is good, the more unstable the simulation, the lower it should be
                'GammaZeroAtRoot'                  : True,                     #[|0, 1|], sets the circulation of the root of the blade to zero
                'GammaZeroAtTip'                   : True,                     #[|0, 1|], sets the circulation of the tip  of the blade to zero
                'GhostParticleAtRoot'              : False,                    #[|0, 1|], add a Particles at after the root of the blade
                'GhostParticleAtTip'               : False,                    #[|0, 1|], add a Particles at after the tip  of the blade
                'IntegralLaw'                      : 'linear',                 #uniform, tanhOneSide, tanhTwoSides or ratio, gives the type of interpolation of the circulation on the lifting lines
                'MaxLiftingLineSubIterations'      : 100,                      #max number of sub iteration when computing the LL circulations
                'MinNbShedParticlesPerLiftingLine' : 27,                       #minimum number of station for every LL from which Particles are shed
                'ParticleDistribution'             : dict(kind = 'uniform',    #uniform, tanhOneSide, tanhTwoSides or ratio, repatition law of the Particles on the Lifting Lines
                                                        FirstSegmentRatio = 1.,#size of the Particles at the root of the blades relative to Sigma0 (i.e. Resolution*SmoothingRatio)
                                                        LastSegmentRatio = 1., #size of the Particles at the tip  of the blades relative to Sigma0 (i.e. Resolution*SmoothingRatio)
                                                        Symmetrical = False),  #[|0, 1|], gives a symmetrical repartition of Particles along the blades or not
                'Pitch'                            : 0.,                       #]-180., 180[ in deg, gives the pitch given to all the lifting lines, if 0 no pitch id added
        }
        defaultParameters.update(VPMParameters)
        if EulerianMesh: defaultHybridParameters.update(HybridParameters)
        else: defaultHybridParameters = {}
        if LiftingLineTree: defaultLiftingLineParameters.update(LiftingLineParameters)
        else: defaultLiftingLineParameters = {}

        if type(defaultParameters['NumberOfThreads']) != str: defaultParameters['NumberOfThreads']=\
                                                    int(round(defaultParameters['NumberOfThreads']))

        os.popen("export OMP_NUM_THREADS=" + str(defaultParameters['NumberOfThreads']))
        defaultParameters['NumberOfThreads'] = int(os.getenv("OMP_NUM_THREADS"))
        VPM.mpi_init(defaultParameters['NumberOfThreads']);
        checkParametersTypes([defaultParameters, defaultHybridParameters,
                               defaultLiftingLineParameters], int_Params, float_Params, bool_Params)
        renameLiftingLineTree(LiftingLineTree, defaultParameters, defaultLiftingLineParameters)
        updateParametersWithLiftingLines(LiftingLineTree, defaultParameters,
                                                                       defaultLiftingLineParameters)
        updateLiftingLines(LiftingLineTree, defaultParameters, defaultLiftingLineParameters)
        tE = []
        if EulerianMesh: tE = generateMirrorWing(EulerianMesh, defaultParameters,
                                                                            defaultHybridParameters)
        t = buildEmptyVPMTree()
        Particles = pickParticlesZone(t)

        if LiftingLineTree:
            print('||' + '{:=^50}'.format(' Initialisation of Lifting Lines '))
            LiftingLines = I.getZones(LiftingLineTree)
            LL.moveLiftingLines(LiftingLines, -defaultParameters['TimeStep'])#for the simulation to start with the propeller at phi = 0
            initialiseParticlesOnLitingLine(t, LiftingLines, PolarInterpolator,
                                                    defaultParameters, defaultLiftingLineParameters)
            J.set(Particles, '.LiftingLine#Parameters', **defaultLiftingLineParameters)
            I._sortByName(I.getNodeFromName1(Particles, '.LiftingLine#Parameters'))
            print('||' + '{:-^50}'.format(' Done '))
        if EulerianMesh:
            print('||' + '{:=^50}'.format(' Initialisation of Hybrid Domain '))
            NumberOfSources = 0
            if 'NumberOfSources' in defaultLiftingLineParameters: NumberOfSources = \
                                                     defaultLiftingLineParameters['NumberOfSources']
            HybridDomain = generateHybridDomain(tE, defaultParameters, defaultHybridParameters)
            initialiseHybridParticles(t, tE, HybridDomain, defaultParameters, 
                                                  defaultHybridParameters, Offset = NumberOfSources)
            J.set(Particles, '.Hybrid#Parameters', **defaultHybridParameters)
            I._sortByName(I.getNodeFromName1(Particles, '.Hybrid#Parameters'))
            print('||' + '{:-^50}'.format(' Done '))

        J.set(Particles, '.VPM#Parameters', **defaultParameters)
        I._sortByName(I.getNodeFromName1(Particles, '.VPM#Parameters'))
        if defaultParameters['MonitorInvariants']:
            J.set(Particles, '.VPM#Invariants', Omega = [0., 0., 0.], LinearImpulse = [0., 0., 0.],
                                   AngularImpulse = [0., 0., 0.], Helicity = 0., KineticEnergy = 0.,
                                   KineticEnergyDivFree = 0., Enstrophy = 0., EnstrophyDivFree = 0.)

        if LiftingLineTree:
            print('||' + '{:=^50}'.format(' Generate Lifting Lines Particles '))
            t = I.merge([t, LiftingLineTree])
            (t, PolarInterpolator)
        if EulerianMesh:
            print('||' + '{:=^50}'.format(' Generate Hybrid Particles '))
            t = I.merge([t, HybridDomain])

            solveParticleStrength(t)
            splitHybridParticles(t)
        
        solveVorticityEquation(t)
        IterationCounter = I.getNodeFromName(t, 'IterationCounter')
        IterationCounter[1][0] = defaultParameters['IterationTuningFMM']*\
                                                               defaultParameters['IntegrationOrder']
        return t, tE

    def pickParticlesZone(t = []):
        for z in I.getZones(t):
            if z[0] == 'Particles':
                return z
        return []

    def getVPMParameters(t = []): return J.get(pickParticlesZone(t), '.VPM#Parameters')

    def solveParticleStrength(t = []):
        Particles = pickParticlesZone(t)
        Np = Particles[1][0][0]
        Offset = getParameter(Particles, 'NumberOfSources')
        if not Offset: Offset = 0

        roll(t, Np - Offset)
        solverInfo = VPM.solve_particle_strength(t)
        roll(t, Offset)
        return solverInfo

    def maskParticlesInsideShieldBoxes(t = [], Boxes = []):
        BoxesBase = I.newCGNSBase('ShieldBoxes', cellDim=1, physDim=3)
        BoxesBase[2] = I.getZones(Boxes)
        return VPM.box_interaction(t, BoxesBase)

    def getInducedVelocityFromWake(t = [], Target = [], TargetSigma = []):
        TargetBase = I.newCGNSBase('Targets', cellDim=1, physDim=3)
        TargetBase[2] = I.getZones(Target)
        Kernel = Kernel_str2int[getParameter(t, 'RegularisationKernel')]
        return VPM.induce_velocity_from_wake(t, TargetBase, Kernel, TargetSigma)

    def findMinimumDistanceBetweenParticles(X = [], Y = [], Z = []):
        return VPM.find_minimum_distance_between_particles(X, Y, Z)

    def findParticleClusters(X = [], Y = [], Z = [], ClusterSize = 0.):
        return VPM.find_particle_clusters(X, Y, Z, ClusterSize)

    def solveVorticityEquation(t = [], IterationInfo = {}):
        Kernel = Kernel_str2int[getParameter(t, 'RegularisationKernel')]
        Scheme = Scheme_str2int[getParameter(t, 'VorticityEquationScheme')]
        Diffusion = DiffusionScheme_str2int[getParameter(t, 'DiffusionScheme')]
        EddyViscosityModel = EddyViscosityModel_str2int[getParameter(t, 'EddyViscosityModel')]
        solveVorticityEquationInfo = VPM.wrap_vpm_solver(t, Kernel, Scheme, Diffusion,
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

    def computeNextTimeStep(t = [], NoDissipationRegions=[]):
        Particles = pickParticlesZone(t)
        LiftingLines = LL.getLiftingLines(t)
        HybridInterface = pickHybridDomainOuterInterface(t)
        NoDissipationRegions.extend(LiftingLines)
        NoDissipationRegions.extend(HybridInterface)
        time, dt, it, IntegOrder, lowstorage, NumberOfSources = getParameters(t,
                 ['Time','TimeStep', 'CurrentIteration', 'IntegrationOrder','LowStorageIntegration',
                                                                                 'NumberOfSources'])
        if not NumberOfSources: NumberOfSources = 0

        if lowstorage:
            if IntegOrder == 1:
                a = np.array([0.], dtype = np.float64)
                b = np.array([1.], dtype = np.float64)
            elif IntegOrder == 2:
                a = np.array([0., -0.5], dtype = np.float64)
                b = np.array([0.5, 1.], dtype = np.float64)
            elif IntegOrder == 3:
                a = np.array([0., -5./9., -153./128.], dtype = np.float64)
                b = np.array([1./3., 15./16., 8./15.], dtype = np.float64)
            elif IntegOrder == 4:
                a = np.array([0., -1., -0.5, -4.], dtype = np.float64)
                b = np.array([1./2., 1./2., 1., 1./6.], dtype = np.float64)
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
        if lowstorage: VPM.runge_kutta_low_storage(t, a, b, Kernel, Scheme, Diffusion,
                                                    EddyViscosityModel, NumberOfSources)
        else: VPM.runge_kutta(t, a, b, Kernel, Scheme, Diffusion,EddyViscosityModel,NumberOfSources)

        time += dt
        it += 1

    def populationControl(t = [], NoRedistributionRegions = [], IterationInfo = {}):
        IterationInfo['Population Control time'] = time()
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
        RedistributedParticles = VPM.population_control(t, AABB, RedistributionKernel,
                                                                              populationControlInfo)
        if RedistributedParticles.any():
            adjustTreeSize(t, NewSize = len(RedistributedParticles[0]), OldSize = N0)
            X, Y, Z = J.getxyz(Particles)
            AX, AY, AZ, AMag, Vol, S, Age = J.getVars(Particles, ['Alpha' + v for v in 'XYZ'] + \
                                                    ['StrengthMagnitude', 'Volume', 'Sigma', 'Age'])

            X[:]    = RedistributedParticles[0][:]
            Y[:]    = RedistributedParticles[1][:]
            Z[:]    = RedistributedParticles[2][:]
            AX[:]   = RedistributedParticles[3][:]
            AY[:]   = RedistributedParticles[4][:]
            AZ[:]   = RedistributedParticles[5][:]
            AMag[:] = RedistributedParticles[6][:]
            Vol[:]  = RedistributedParticles[7][:]
            S[:]    = RedistributedParticles[8][:]

            

            Age[:]  = np.array([int(age) for age in RedistributedParticles[9]], dtype = np.int32)
        else:
           adjustTreeSize(t, NewSize = Np[0], OldSize = N0)

        IterationInfo['Number of Particles beyond cutoff'] = populationControlInfo[0]
        IterationInfo['Number of split Particles'] = populationControlInfo[1]
        IterationInfo['Number of depleted Particles'] = populationControlInfo[2]
        IterationInfo['Number of merged Particles'] = populationControlInfo[3]
        IterationInfo['Population Control time'] = time() - IterationInfo['Population Control time']
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
        'TurbulentDissipationRateDensity', 'ViscosityMolecular', 'Pressure', 'EnergyStagnationDensity',
        'q_criterion', 'Viscosity_EddyMolecularRatio', 'Mach', 'cellN', 'indicm']
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
                FlowSolution_m = J.invokeFields(Zone_m, [FlowSolution[0]], locationTag = 'centers:')[0]
                if FlowSolution[0] in reverse: FlowSolution_m[:] = -FlowSolution[1]
                else: FlowSolution_m[:] = FlowSolution[1]
                if FlowSolution[0] == 'Zone': FlowSolution_m[:] += len(Zones)
            Zones_m += [Zone_m]
            
        
        MeshRadius = getTurbulentDistances(ESP.extractSurfacesByOffsetCellsFromBCFamilyName(tE,
                               BCFamilyName = HybridParameters['BCFarFieldName'], NCellsOffset = 0))
        NumberOfHybridInterfaces = HybridParameters['NumberOfHybridInterfaces'][0]
        OuterDomainToWallDistance = HybridParameters['OuterDomainToWallDistance'][0]
        Sigma = VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]
        InnerDomain = OuterDomainToWallDistance - NumberOfHybridInterfaces*Sigma
        if InnerDomain < Sigma:
            raise ValueError('The Hybrid Domain radius (NumberOfHybridInterfaces*Sigma = %.5f m) \
                is too close to the solid (%.5f m < Sigma = %.5f) for the selected \
                OuterDomainToWallDistance = %.5f m. Either reduce the NumberOfHybridInterfaces, \
                the Resolution or the SmoothingRatio, or increase the OuterDomainToWallDistance.'%(\
                NumberOfHybridInterfaces*Sigma, InnerDomain, Sigma, OuterDomainToWallDistance))
        if MeshRadius <= OuterDomainToWallDistance:
            raise ValueError('The Hybrid Domain ends beyond the mesh (OuterDomainToWallDistance = \
                %.5f m). The furthest cell is %.5f m from the wall.'%(MeshRadius))

        tE = C.newPyTree([Zones + Zones_m])
        return I.correctPyTree(tE)

    def getTurbulentDistances(tE = []):
        d = np.array([])
        for zone in I.getZones(tE):
            FlowSolutionNode = I.getNodeFromName1(zone, 'FlowSolution#Centers')
            d = np.append(d, I.getNodeFromName(FlowSolutionNode, 'TurbulentDistance')[1])
        return np.mean(d)

    def generateSourceInterfaceForHybridDomain(tE = [], VPMParameters = {}, HybridParameters = {}):
        print('||'+'{:-^50}'.format(' Generate Hybrid Interfaces '))
        OuterDomainToWallDistance = HybridParameters['OuterDomainToWallDistance'][0]
        InterfaceGap = VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]
        NumberOfHybridInterfaces = HybridParameters['NumberOfHybridInterfaces'][0]
        tE_joint = T.join(C.convertArray2Hexa(tE))
        remove = ['GridLocation', '.Solver#Param', '.MOLA#Offset', '.MOLA#Trim', 'FamilyName',
                             'AdditionalFamilyName', 'ZoneGridConnectivity', 'FlowSolution#Centers']
        Interfaces = []
        for n in range(NumberOfHybridInterfaces + 2):
            d = OuterDomainToWallDistance - n*InterfaceGap
            Zones = I.getZones(P.isoSurfMC(tE_joint, 'TurbulentDistance', d))
            for Zone in Zones:
                for rm in remove: I._rmNodesByName(Zone, rm)

            Interface = T.join(C.convertArray2Hexa(C.newPyTree([Zones])))
            Interface[0] = 'Interface_' + str(n)
            Interfaces += [Interface]

        HybridParameters['InnerDomainToWallDistance'] = np.array([d + InterfaceGap], dtype = 
                                                                            np.float64, order = 'F')
        return Interfaces

    def getRegularGridInHybridDomain(OuterHybridDomain = [], InnerHybridDomain = [], Resolution = 0):

        bbox = np.array(G.bbox(OuterHybridDomain))
        Ni = int(np.ceil((bbox[3] - bbox[0])/Resolution)) + 4
        Nj = int(np.ceil((bbox[4] - bbox[1])/Resolution)) + 4
        Nk = int(np.ceil((bbox[5] - bbox[2])/Resolution)) + 4
        cart = G.cart(np.ceil(bbox[:3]/Resolution)*Resolution - 2*Resolution, (Resolution, Resolution, Resolution),
                                                                                       (Ni, Nj, Nk))
        t_cart = C.newPyTree(['Base', cart])
        maskInnerSurface = CX.blankCells(t_cart, [[InnerHybridDomain]], np.array([[1]]),
                             blankingType = 'cell_intersect', delta = 0, dim = 3, tol = 0.)
        maskOuterSurface = CX.blankCells(t_cart, [[OuterHybridDomain]], np.array([[1]]),
                             blankingType = 'cell_intersect', delta = Resolution, dim = 3, tol = 0.)
        maskInnerSurface  = C.node2Center(I.getZones(maskInnerSurface)[0])
        maskOuterSurface = C.node2Center(I.getZones(maskOuterSurface)[0])
        cellInnerSurface = J.getVars(maskInnerSurface, ['cellN'], 'FlowSolution')[0]
        cellOuterSurface = J.getVars(maskOuterSurface, ['cellN'], 'FlowSolution')[0]
        inside = (cellOuterSurface == 0)*(cellInnerSurface == 1)
        x, y, z = J.getxyz(maskOuterSurface)
        return C.convertArray2Node(J.createZone('Grid', [x[inside], y[inside], z[inside]], 'xyz'))

    def setDonorsFromEulerianMesh(Donor = [], Interfaces = [[], []], VPMParameters = {}, HybridParameters = {}):
        Sigma0 = VPMParameters['Resolution'][0]*VPMParameters['SmoothingRatio'][0]
        Grid = getRegularGridInHybridDomain(Interfaces[0], Interfaces.pop(), Sigma0)
        InnerDomainToWallDistance = HybridParameters['InnerDomainToWallDistance'][0]
        OuterDomainToWallDistance = HybridParameters['OuterDomainToWallDistance'][0]
        NumberOfHybridInterfaces = HybridParameters['NumberOfHybridInterfaces'][0]
        px, py, pz = J.getxyz(Grid)
        t_centers = C.node2Center(Donor)
        zones_centers = I.getZones(t_centers)
        hook, indir = C.createGlobalHook(t_centers, function = 'nodes', indir = 1)
        nodes, dist = C.nearestNodes(hook, J.createZone('Particles', [px, py, pz], 'xyz'))
        nodes, unique = np.unique(nodes, return_index=True)
        px, py, pz = px[unique], py[unique], pz[unique]

        cumul = 0
        cumulated = []
        for z in zones_centers:
            cumulated += [cumul]
            cumul += C.getNPts(z)

        receivers = []
        donors = []
        for p in range(len(px)):
            ind = nodes[p] - 1
            closest_index = ind - cumulated[indir[ind]]
            receivers += [indir[ind]]
            donors += [closest_index]

        HybridParameters['Donors'] = np.array(donors, order = 'F', dtype = np.int32)
        HybridParameters['Receivers'] = np.array(receivers, order = 'F', dtype = np.int32)

        Nh = len(HybridParameters['Donors'])
        px = np.zeros(Nh, dtype = np.float64)
        py = np.zeros(Nh, dtype = np.float64)
        pz = np.zeros(Nh, dtype = np.float64)
        for iz, zone in enumerate(I.getZones(Donor)):
            receiver_slice = HybridParameters['Receivers'] == iz
            donor_slice = HybridParameters['Donors'][receiver_slice]
            xc, yc, zc = J.getVars(zone, ['Center' + v for v in 'XYZ'], 'FlowSolution#Centers')
            px[receiver_slice] = np.ravel(xc, order = 'F')[donor_slice]
            py[receiver_slice] = np.ravel(yc, order = 'F')[donor_slice]
            pz[receiver_slice] = np.ravel(zc, order = 'F')[donor_slice]

        particles = C.newPyTree([C.convertArray2Node(J.createZone('Zone', [px, py, pz], 'xyz'))])
        flags = []
        for Interface in Interfaces:
            mask = I.getZones(CX.blankCells(particles, [[Interface]], np.array([[1]]), 
                                        blankingType = 'node_in', delta = 0., dim = 3, tol = 0.))[0]
            flags += [J.getVars(mask, ['cellN'], 'FlowSolution')[0]]

        domain = (flags[0] == 0)*(flags[-1] == 1)#0 means in, 1 means out, outer at the beginning, inner at the end
        HybridParameters['Donors'] = HybridParameters['Donors'][domain]
        HybridParameters['Receivers'] = HybridParameters['Receivers'][domain]
        px, py, pz = px[domain], py[domain], pz[domain]
        flags = [(flags[i][domain] == 0)*(flags[i + 1][domain] == 1) for i in range(len(flags) - 1)]
        InterfacesFlags = {}
        for Interface, flag in zip(Interfaces, flags):
            InterfacesFlags[Interface[0]] = np.array(flag, dtype = np.int32, order = 'F')

        HybridParameters['Interfaces'] = InterfacesFlags
        '''
        Sigma = findMinimumDistanceBetweenParticles(px, py, pz)
        TooClose = Sigma < 0.5*Sigma0
        while TooClose.any():
            SmallSigma = Sigma[TooClose]
            FusedParticles = []
            i = 0
            while i < len(SmallSigma):
                while i < len(SmallSigma) and (SmallSigma[i] in FusedParticles): i += 1

                if i != len(SmallSigma):
                    FusedParticles += [SmallSigma[i]]
                    TooClose[TooClose][i] = False

            TooClose = np.logical_not(TooClose)
            px, py, pz = px[TooClose], py[TooClose], pz[TooClose]
            HybridParameters['Donors'] = HybridParameters['Donors'][TooClose]
            HybridParameters['Receivers'] = HybridParameters['Receivers'][TooClose]
            HybridParameters['Source'] = HybridParameters['Source'][TooClose]
            HybridParameters['Domain'] = HybridParameters['Domain'][TooClose]

            Sigma = findMinimumDistanceBetweenParticles(px, py, pz)
            TooClose = Sigma < 0.5*Sigma0
        '''

    def generateHybridDomain(tE = [], VPMParameters = {}, HybridParameters = {}):
        Interfaces = generateSourceInterfaceForHybridDomain(tE, VPMParameters, HybridParameters)
        setDonorsFromEulerianMesh(tE, Interfaces, VPMParameters, HybridParameters)
        Interfaces.pop()
        Interfaces = ['HybridDomain'] + Interfaces
        return C.newPyTree(Interfaces)

    def pickHybridDomainSourceInterface(t = []):
        for z in I.getZones(t):
            if z[0] == 'SourceInterface':
                return z
        return []

    def pickHybridDomainOuterInterface(t = []):
        HybridDomain = I.getNodeFromName1(t, 'HybridDomain')
        if HybridDomain is not None: return I.getZones(HybridDomain)[0]

        return []

    def pickHybridDomainInnerInterface(t = []):
        HybridDomain = I.getNodeFromName1(t, 'HybridDomain')
        if HybridDomain is not None: return I.getZones(HybridDomain)[-1]

        return []

    def getHybridParameters(t = []): return J.get(pickParticlesZone(t), '.Hybrid#Parameters')

    def initialiseHybridParticles(tL = [], tE = [], HybridDomain = [], VPMParameters = {}, HybridParameters = {}, Offset = 0):
        ite = VPMParameters['CurrentIteration'][0]
        Ramp = VPMParameters['StrengthRampAtbeginning'][0]
        Ramp = np.sin(min((ite + 1.)/Ramp, 1.)*np.pi/2.)
        Nh = len(HybridParameters['Donors'])
        px = np.zeros(Nh, dtype = np.float64)
        py = np.zeros(Nh, dtype = np.float64)
        pz = np.zeros(Nh, dtype = np.float64)
        ωpx = np.zeros(Nh, dtype = np.float64)
        ωpy = np.zeros(Nh, dtype = np.float64)
        ωpz = np.zeros(Nh, dtype = np.float64)
        for iz, zone in enumerate(I.getZones(tE)):
            receiver_slice = HybridParameters['Receivers'] == iz
            xc, yc, zc = J.getVars(zone, ['Center' + v for v in 'XYZ'], 'FlowSolution#Centers')
            ωxc, ωyc, ωzc = J.getVars(zone, ['Vorticity' + v for v in 'XYZ'], 'FlowSolution#Centers')

            donor_slice = HybridParameters['Donors'][receiver_slice]
            px[receiver_slice] = np.ravel(xc, order = 'F')[donor_slice]
            py[receiver_slice] = np.ravel(yc, order = 'F')[donor_slice]
            pz[receiver_slice] = np.ravel(zc, order = 'F')[donor_slice]
            ωpx[receiver_slice] = np.ravel(ωxc, order = 'F')[donor_slice]
            ωpy[receiver_slice] = np.ravel(ωyc, order = 'F')[donor_slice]
            ωpz[receiver_slice] = np.ravel(ωzc, order = 'F')[donor_slice]

        ωpx *= Ramp
        ωpy *= Ramp
        ωpz *= Ramp
        ωp = np.linalg.norm(np.vstack([ωpx, ωpy, ωpz]), axis=0)
        ωtot = np.sum(ωp)#total particle voticity in the Hybrid Domain

        Np = 0
        Interfaces = HybridParameters['Interfaces']
        source = np.array([False]*len(Interfaces['Interface_0']))
        for i, Interface in enumerate(Interfaces):
            flag = (Interfaces[Interface] == 1)
            ωi = ωp[flag]
            Ni = min(max(HybridParameters['NumberOfParticlesPerInterface'][0], 1), len(ωi))
            Np += Ni
            ωmin = np.sort(ωi)[-Ni]#vorticity cutoff        
            SourceFlag = ωmin < ωi#only the strongest will survive
            NSourceFlag = np.sum(SourceFlag)
            if Ni != NSourceFlag:
                allωmin = np.where(ωmin == ωi)[0]
                SourceFlag[allωmin[:Ni - NSourceFlag]] = True

            flag[flag] = SourceFlag
            source += np.array(flag)

        extend(tL, Np, ExtendAtTheEnd = False, Offset = Offset)
        Np += Offset
        Particles = pickParticlesZone(tL)
        x, y, z = J.getxyz(Particles)
        VortX, VortY, VortZ, VortMag, Nu, Volume, Sigma, HybridFlag = J.getVars(Particles, 
                                                        ['Vorticity' + v for v in 'XYZ'] + \
                                        ['VorticityMagnitude', 'Nu', 'Volume', 'Sigma', 'HybridFlag'])
        x[Offset: Np], y[Offset: Np], z[Offset: Np] = px[source], py[source], pz[source]
        VortX[Offset: Np], VortY[Offset: Np], VortZ[Offset: Np] = ωpx[source], ωpy[source], ωpz[source]
        VortMag[Offset: Np] = ωp[source]
        HybridFlag[Offset: Np] = 1
        Nu[Offset: Np] = VPMParameters['KinematicViscosity']

        s = findMinimumDistanceBetweenParticles(x, y, z)[Offset: Np]
        Sigma[Offset: Np] = s
        h3 = s**3
        Volume[Offset: Np] = (2.*np.pi)**1.5*h3

        HybridParameters['AlphaX'] = VortX[Offset: Np]*h3
        HybridParameters['AlphaY'] = VortY[Offset: Np]*h3
        HybridParameters['AlphaZ'] = VortZ[Offset: Np]*h3
        HybridParameters['Sigma'] = s
        HybridParameters['Volume'] = Volume[Offset: Np]

        Np -= Offset
        msg = '||'+'{:27}'.format('Number of cells') + ': '+ '{:d}'.format(Nh) + '\n'
        msg += '||' + '{:27}'.format('Number of hybrid Particles') + ': '+ '{:d}'.format(Np) +\
                                                    ' (' + '{:.1f}'.format(Np/Nh*100.) + '%)\n'
        msg += '||' + '{:27}'.format('Mean Particle spacing') + ': '+'{:.3f}'.format(np.mean(s)) +' m\n'
        msg += '||' +'{:27}'.format('Particle spacing deviation')+': '+'{:.3f}'.format(np.std(s))+' m\n'
        msg += '||' + '{:27}'.format('Maximum Particle spacing') +': '+'{:.3f}'.format(np.max(s))+' m\n'
        msg += '||' + '{:27}'.format('Minimum Particle spacing') +': '+'{:.3f}'.format(np.min(s))+' m\n'
        print(msg)

    def eraseParticlesInHybridDomain(t = [], Offset = 0):
        Particles = pickParticlesZone(t)
        HybridDomain = pickHybridDomainOuterInterface(t)
        box = [np.inf, np.inf, np.inf, -np.inf, -np.inf, -np.inf]
        for BC in I.getZones(HybridDomain):
            x, y, z = J.getxyz(BC)
            box = [min(box[0], np.min(x)), min(box[1], np.min(y)), min(box[2], np.min(z)),
                             max(box[3], np.max(x)), max(box[4], np.max(y)), max(box[5], np.max(z))]

        x, y, z = J.getxyz(Particles)
        inside = (box[0] < x)*(box[1] < y)*(box[2] < z)*(x < box[3])*(y < box[4])*(z < box[5])
        x, y, z = x[inside], y[inside], z[inside]
        mask = C.convertArray2Node(J.createZone('Zone', [x, y, z], 'xyz'))
        mask = I.getZones(CX.blankCells(C.newPyTree(['Base', mask]), [[HybridDomain]],
                      np.array([[1]]),  blankingType = 'node_in', delta = 0., dim = 3, tol = 0.))[0]
        cellN = J.getVars(mask, ['cellN'], 'FlowSolution')[0]
        inside[inside] = (cellN == 0)
        inside[:Offset] = False
        Nerase = np.sum(inside)
        delete(t, inside)
        return Nerase

    def splitHybridParticles(t = []):
        Particles = pickParticlesZone(t)
        splitParticles = VPM.split_hybrid_particles(t)
        Offset = getParameter(Particles, 'NumberOfSources')
        if not Offset: Offset = 0
        HybridParameters = getHybridParameters(t)
        Nh = HybridParameters['NumberOfParticlesPerInterface'][0]*\
                                                     HybridParameters['NumberOfHybridInterfaces'][0]
        if (splitParticles is not None):
            Nsplit = len(splitParticles[0])
            adjustTreeSize(t, NewSize = Nsplit, OldSize =  Nh, AtTheEnd = False, Offset = Offset)
            
            X, Y, Z = J.getxyz(Particles)
            AX, AY, AZ, AMag, Vol, S, HybridFlag = J.getVars(Particles, ['Alpha' + v for v in 'XYZ'] +\
                                                 ['StrengthMagnitude', 'Volume', 'Sigma', 'HybridFlag'])
            HybridFlag[Offset: Nsplit + Offset] = 1
            flag = (HybridFlag == 1)
            X[flag]    = splitParticles[0][:]
            Y[flag]    = splitParticles[1][:]
            Z[flag]    = splitParticles[2][:]
            AX[flag]   = splitParticles[3][:]
            AY[flag]   = splitParticles[4][:]
            AZ[flag]   = splitParticles[5][:]
            AMag[flag] = splitParticles[6][:]
            Vol[flag]  = splitParticles[7][:]
            S[flag]    = splitParticles[8][:]
        else :Nsplit = Nh

        return Nsplit

    def generateParticlesInHybridInterfaces(tL = [], tE = [], IterationInfo = {}):
        if not tE: return IterationInfo
        IterationInfo['Strength computation time'] = time()
        HybridParameters = getHybridParameters(tL)
        VPMParameters = getVPMParameters(tL) 
        Offset = getParameter(tL, 'NumberOfSources')
        if not Offset: Offset = 0

        t0 = J.tic()
        IterationInfo['Number of Hybrids Generated'] = -eraseParticlesInHybridDomain(tL, Offset)
        #print('erase', J.tic() - t0)

        t0 = J.tic()
        ite = VPMParameters['CurrentIteration'][0]
        Ramp = VPMParameters['StrengthRampAtbeginning'][0]
        Ramp = np.sin(min((ite + 1.)/Ramp, 1.)*np.pi/2.)
        Nh = len(HybridParameters['Donors'])
        px = np.zeros(Nh, dtype = np.float64)
        py = np.zeros(Nh, dtype = np.float64)
        pz = np.zeros(Nh, dtype = np.float64)
        ωpx = np.zeros(Nh, dtype = np.float64)
        ωpy = np.zeros(Nh, dtype = np.float64)
        ωpz = np.zeros(Nh, dtype = np.float64)
        for iz, zone in enumerate(I.getZones(tE)):
            receiver_slice = HybridParameters['Receivers'] == iz
            xc, yc, zc = J.getVars(zone, ['Center' + v for v in 'XYZ'], 'FlowSolution#Centers')
            ωxc, ωyc, ωzc = J.getVars(zone, ['Vorticity' + v for v in 'XYZ'], 'FlowSolution#Centers')

            donor_slice = HybridParameters['Donors'][receiver_slice]
            px[receiver_slice] = np.ravel(xc, order = 'F')[donor_slice]
            py[receiver_slice] = np.ravel(yc, order = 'F')[donor_slice]
            pz[receiver_slice] = np.ravel(zc, order = 'F')[donor_slice]
            ωpx[receiver_slice] = np.ravel(ωxc, order = 'F')[donor_slice]
            ωpy[receiver_slice] = np.ravel(ωyc, order = 'F')[donor_slice]
            ωpz[receiver_slice] = np.ravel(ωzc, order = 'F')[donor_slice]

        ωpx *= Ramp
        ωpy *= Ramp
        ωpz *= Ramp
        ωp = np.linalg.norm(np.vstack([ωpx, ωpy, ωpz]), axis=0)
        ωtot = np.sum(ωp)#total particle voticity in the Hybrid Domain
        
        Np = 0
        Interfaces = HybridParameters['Interfaces']
        source = np.array([False]*len(Interfaces['Interface_0']))
        for i, Interface in enumerate(Interfaces):
            flag = (Interfaces[Interface] == 1)
            ωi = ωp[flag]
            Ni = min(max(HybridParameters['NumberOfParticlesPerInterface'][0], 1), len(ωi))
            Np += Ni
            ωmin = np.sort(ωi)[-Ni]#vorticity cutoff        
            SourceFlag = ωmin < ωi#only the strongest will survive
            NSourceFlag = np.sum(SourceFlag)
            if Ni != NSourceFlag:
                allωmin = np.where(ωmin == ωi)[0]
                SourceFlag[allωmin[:Ni - NSourceFlag]] = True

            flag[flag] = SourceFlag
            source += np.array(flag)

        extend(tL, Np, ExtendAtTheEnd = False, Offset = Offset)
        Np += Offset
        Particles = pickParticlesZone(tL)
        x, y, z = J.getxyz(Particles)
        VortX, VortY, VortZ, VortMag, Nu, Volume, Sigma, HybridFlag = J.getVars(Particles, 
                                                        ['Vorticity' + v for v in 'XYZ'] + \
                                        ['VorticityMagnitude', 'Nu', 'Volume', 'Sigma', 'HybridFlag'])
        x[Offset: Np], y[Offset: Np], z[Offset: Np] = px[source], py[source], pz[source]
        VortX[Offset: Np], VortY[Offset: Np], VortZ[Offset: Np] = ωpx[source], ωpy[source], ωpz[source]
        VortMag[Offset: Np] = ωp[source]
        Nu[Offset: Np] = VPMParameters['KinematicViscosity']

        HybridFlag[:Offset], HybridFlag[Offset: Np], HybridFlag[Np:] = 0, 1, 0

        Sigma[Offset: Np] = HybridParameters['Sigma']
        Volume[Offset: Np] = HybridParameters['Volume']

        #print('gen', J.tic() - t0)

        t0 = J.tic()
        solverInfo = solveParticleStrength(tL)
        #print('solv', J.tic() - t0)
        t0 = J.tic()
        IterationInfo['Number of Hybrids Generated'] += splitHybridParticles(tL)
        #print('split', J.tic() - t0)
        
        ωkept = np.sum(VortMag[Offset: Np])#total particle voticity kept
        IterationInfo['Number of sub-iterations (E)'] = int(round(solverInfo[0]))
        IterationInfo['Rel. err. of Vorticity'] = solverInfo[1]
        IterationInfo['Number of Hybrids Generated'] += np.sum(HybridFlag)
        IterationInfo['Minimum Eulerian Vorticity'] = np.min(VortMag[Offset: Np])
        IterationInfo['Eulerian Vorticity lost'] = ωtot - ωkept
        IterationInfo['Eulerian Vorticity lost per'] = (ωtot - ωkept)/ωtot*100
        IterationInfo['Strength computation time'] = time() - IterationInfo['Strength computation time']
        return IterationInfo

####################################################################################################
####################################################################################################
########################################### LiftingLines ###########################################
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
            Resolution = I.getValue(I.getNodeFromName(Particles, 'Resolution'))
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
            TimeStep = I.getNodeFromName(Particles, 'TimeStep')
            TimeStep[1][0] = NumberParticlesShedAtTip*Resolution/Urel

    def setTimeStepFromBladeRotationAngle(t = [], LiftingLines = [], BladeRotationAngle = 5.):
        if not LiftingLines: raise AttributeError('The time step is not given and can not be \
                     computed without a Lifting Line. Specify the time step or give a Lifting Line')
        RPM = 0.
        for LiftingLine in LiftingLines:
            RPM = max(RPM, I.getValue(I.getNodeFromName(LiftingLine, 'RPM')))

        if type(t) == dict:
            t['TimeStep'] = 1./6.*BladeRotationAngle/RPM
        else:
            TimeStep = I.getNodeFromName(Particles, 'TimeStep')
            TimeStep[1][0] = 1./6.*BladeRotationAngle/RPM

    TimeStepFunction_str2int = {'setTimeStepFromBladeRotationAngle':
    setTimeStepFromBladeRotationAngle, 'shed': setTimeStepFromShedParticles,
    'BladeRotationAngle': setTimeStepFromBladeRotationAngle, 'setTimeStepFromShedParticles':
    setTimeStepFromShedParticles, 'ShedParticles': setTimeStepFromShedParticles, 'Angle':
    setTimeStepFromBladeRotationAngle, 'angle': setTimeStepFromBladeRotationAngle, 'Shed':
    setTimeStepFromShedParticles}

    def setMinNbShedParticlesPerLiftingLine(LiftingLines = [], Parameters = {}, NumberParticlesShedAtTip = 5):
        LL.computeKinematicVelocity(LiftingLines)
        LL.assembleAndProjectVelocities(LiftingLines)
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
                LiftingLineTree = C.newPyTree(['LiftingLines', LiftingLinesZones])
            else:
                raise AttributeError(ERRMSG)

        else:
            raise AttributeError(ERRMSG)

    def updateParametersWithLiftingLines(LiftingLineTree = [], Parameters = {}, LiftingLineParameters = {}):
        LiftingLines = I.getZones(LiftingLineTree)
        if 'Resolution' not in Parameters and LiftingLines:
            ShortestLiftingLineSpan = np.inf
            for LiftingLine in LiftingLines:
                ShortestLiftingLineSpan = np.minimum(ShortestLiftingLineSpan,
                                                     W.getLength(LiftingLine))
            Parameters['Resolution'] = ShortestLiftingLineSpan/\
                                    LiftingLineParameters['MinNbShedParticlesPerLiftingLine']
        elif 'MinNbShedParticlesPerLiftingLine' not in LiftingLineParameters and 'Resolution' in \
                                                                        Parameters and LiftingLines:
            ShortestLiftingLineSpan = np.inf
            for LiftingLine in LiftingLines:
                ShortestLiftingLineSpan = np.minimum(ShortestLiftingLineSpan,
                                                     W.getLength(LiftingLine))
            LiftingLineParameters['MinNbShedParticlesPerLiftingLine'] = \
                                int(round(ShortestLiftingLineSpan/Parameters['Resolution']))#n segments gives n + 1 stations, and each Particles is surrounded by two stations, thus shedding n Particles. One has to add up to that the presence of ghost Particles at the tips
        elif 'Resolution' not in Parameters:
            raise ValueError('The Resolution can not be computed, the Resolution itself or a \
                                Lifting Line with a MinNbShedParticlesPerLiftingLine must be given')

        if 'VelocityFreestream' not in Parameters:Parameters['VelocityFreestream'] =np.array([0.]*3,
                                                                                      dtype = float)

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

    def updateLiftingLines(LiftingLineTree = [], VPMParameters = {}, LiftingLineParameters = {}):
        for LiftingLine in I.getZones(LiftingLineTree):
            LLParameters = J.get(LiftingLine, '.VPM#Parameters')
            if not LLParameters:
                LL.setVPMParameters(LiftingLine)
                LLParameters = J.get(LiftingLine, '.VPM#Parameters')
            if 'GammaZeroAtRoot' in LiftingLineParameters:
                LLParameters['GammaZeroAtRoot'][0] = LiftingLineParameters['GammaZeroAtRoot'][0]
            if 'GammaZeroAtTip'  in LiftingLineParameters:
                LLParameters['GammaZeroAtTip'][0]  = LiftingLineParameters['GammaZeroAtTip'][0]
            if 'GhostParticleAtRoot' in LiftingLineParameters:
                LLParameters['GhostParticleAtRoot'][0] = \
                                                    LiftingLineParameters['GhostParticleAtRoot'][0]
            if 'GhostParticleAtTip'  in LiftingLineParameters:
                LLParameters['GhostParticleAtTip'][0]  = \
                                                    LiftingLineParameters['GhostParticleAtTip'][0]
            if 'IntegralLaw' in LiftingLineParameters:
                LLParameters['IntegralLaw'] = LiftingLineParameters['IntegralLaw']
            if 'ParticleDistribution' in LiftingLineParameters:
                ParticleDistributionOld = LiftingLineParameters['ParticleDistribution']
            else: ParticleDistributionOld = LLParameters['ParticleDistribution']

            ParticleDistribution = {'kind' : ParticleDistributionOld['kind']}

            if 'FirstSegmentRatio' in ParticleDistributionOld:
                ParticleDistribution['FirstCellHeight'] = \
                            ParticleDistributionOld['FirstSegmentRatio']*VPMParameters['Resolution']
            if 'LastSegmentRatio' in ParticleDistributionOld:
                ParticleDistribution['LastCellHeight'] =\
                             ParticleDistributionOld['LastSegmentRatio']*VPMParameters['Resolution']
            if 'growthRatio' in ParticleDistributionOld:
                ParticleDistribution['growth'] = ParticleDistributionOld['growthRatio']*\
                                                                         VPMParameters['Resolution']
            if 'parameter' in ParticleDistributionOld:
                ParticleDistribution['parameter'] = ParticleDistributionOld['parameter']
            if 'Symmetrical' in ParticleDistributionOld:
                ParticleDistribution['Symmetrical'] = ParticleDistributionOld['Symmetrical']
            LLParameters['ParticleDistribution'] = ParticleDistribution

            J.set(LiftingLine, '.VPM#Parameters',
                            GammaZeroAtRoot = LLParameters['GammaZeroAtRoot'],
                            GammaZeroAtTip = LLParameters['GammaZeroAtTip'],
                            GhostParticleAtRoot = LLParameters['GhostParticleAtRoot'],
                            GhostParticleAtTip = LLParameters['GhostParticleAtTip'],
                            IntegralLaw = LLParameters['IntegralLaw'],
                            ParticleDistribution = LLParameters['ParticleDistribution'])

    def initialiseParticlesOnLitingLine(t = [], LiftingLines = [], PolarInterpolator = {}, VPMParameters = {}, LiftingLineParameters = {}):
        if not LiftingLines: return

        LL.computeKinematicVelocity(LiftingLines)
        LL.assembleAndProjectVelocities(LiftingLines)
        LL._applyPolarOnLiftingLine(LiftingLines, PolarInterpolator, ['Cl', 'Cd'])
        LL.computeGeneralLoadsOfLiftingLine(LiftingLines)

        Sources = []
        SourcesM1 = []
        TotalNumberOfSources = 0
        X, Y, Z, S = [], [], [], []
        X0, Y0, Z0, S0 = [], [], [], []
        Np = 0
        for LiftingLine in LiftingLines:
            L = W.getLength(LiftingLine)
            LLParameters = J.get(LiftingLine, '.VPM#Parameters')
            ParticleDistribution = LLParameters['ParticleDistribution']
            NumberOfStations = int(np.round(L/VPMParameters['Resolution'])) + 1#n segments gives n + 1 stations on the LL

            if ParticleDistribution['Symmetrical']:
                HalfStations = int(NumberOfStations/2 + 1)
                SemiWing = W.linelaw(P1 = (0., 0., 0.), P2 = (L/2., 0., 0.), N = HalfStations,
                                                                Distribution = ParticleDistribution)# has to give +1 point because one point is lost with T.symetrize()
                WingDiscretization = J.getx(T.join(T.symetrize(SemiWing, (0, 0, 0), (0, 1, 0), \
                                                                                (0, 0, 1)), SemiWing))
                WingDiscretization += L/2.
                ParticleDistribution = WingDiscretization/L
                if not (NumberOfStations % 2):
                    N = len(ParticleDistribution)//2
                    array = ParticleDistribution
                    ParticleDistribution = np.append([(array[N] + array[N - 2])/2., (array[N + 2] +\
                                                                       array[N])/2.], array[N + 2:])
                    ParticleDistribution = np.append(array[:N - 1], ParticleDistribution)
            else:
                WingDiscretization = J.getx(W.linelaw(P1 = (0., 0., 0.), P2 = (L, 0., 0.),
                                                        N = NumberOfStations,
                                                        Distribution = ParticleDistribution))
                ParticleDistribution = WingDiscretization/L

            TotalNumberOfSources += len(ParticleDistribution) - 1
            Source = LL.buildVortexParticleSourcesOnLiftingLine(LiftingLine,
                                                        AbscissaSegments = [ParticleDistribution],
                                                        IntegralLaw = LLParameters['IntegralLaw'])

            SourceX = I.getValue(I.getNodeFromName(Source, 'CoordinateX'))
            SourceY = I.getValue(I.getNodeFromName(Source, 'CoordinateY'))
            SourceZ = I.getValue(I.getNodeFromName(Source, 'CoordinateZ'))
            dy = ((SourceX[1:] - SourceX[:-1])**2 + (SourceY[1:] - SourceY[:-1])**2 +\
                                (SourceZ[1:] - SourceZ[:-1])**2)**0.5
            SigmaDistribution = dy*VPMParameters['SmoothingRatio']#np.array([Parameters['Sigma0'][0]]*len(dy), order = 'F', dtype = np.float64)
            x, y, z = J.getxyz(LiftingLine)
            dy = ((x[1:] - x[:-1])**2 + (y[1:] - y[:-1])**2 + (z[1:] - z[:-1])**2)**0.5
            mean_dy = np.insert((dy[1:] + dy[:-1])/2., [0, len(dy)-1], [dy[0], dy[-1]])
            SigmaDistributionOnLiftingLine = mean_dy*VPMParameters['SmoothingRatio']
            LLParameters = J.get(LiftingLine, '.VPM#Parameters')
            LLParameters['ParticleDistribution'] = ParticleDistribution
            LLParameters['SigmaDistribution'] = SigmaDistribution
            LLParameters['SigmaDistributionOnLiftingLine'] = SigmaDistributionOnLiftingLine
            J.set(LiftingLine, '.VPM#Parameters', **LLParameters)

            X0 = np.append(X0, (SourceX[1:] + SourceX[:-1])/2.)
            Y0 = np.append(Y0, (SourceY[1:] + SourceY[:-1])/2.)
            Z0 = np.append(Z0, (SourceZ[1:] + SourceZ[:-1])/2.)
            S0 = np.append(S0, SigmaDistribution)
            Np += len(SourceX) - 1

            Kinematics = J.get(LiftingLine, '.Kinematics')
            VelocityRelative = VPMParameters['VelocityFreestream']-Kinematics['VelocityTranslation']
            Dpsi = Kinematics['RPM']*6.*VPMParameters['TimeStep']
            if not Kinematics['RightHandRuleRotation']: Dpsi *= -1
            T._rotate(Source, Kinematics['RotationCenter'], Kinematics['RotationAxis'], -Dpsi[0])
            T._translate(Source, VPMParameters['TimeStep']*VelocityRelative)

            SourceX = I.getValue(I.getNodeFromName(Source, 'CoordinateX'))
            SourceY = I.getValue(I.getNodeFromName(Source, 'CoordinateY'))
            SourceZ = I.getValue(I.getNodeFromName(Source, 'CoordinateZ'))
            X = np.append(X, (SourceX[1:] + SourceX[:-1])/2.)
            Y = np.append(Y, (SourceY[1:] + SourceY[:-1])/2.)
            Z = np.append(Z, (SourceZ[1:] + SourceZ[:-1])/2.)
            S = np.append(S, SigmaDistribution)
            Np += len(SourceX) - 1

        LiftingLineParameters['NumberOfSources'] = TotalNumberOfSources#is also Np/2
        LiftingLineParameters['ShedParticlesIndex'] = np.array([i for i in range(
                                   TotalNumberOfSources, 2*TotalNumberOfSources)], dtype = np.int32)

        extend(t, Np, ExtendAtTheEnd = False, Offset = 0)
        Particles = pickParticlesZone(t)
        x, y, z = J.getxyz(Particles)
        x[:Np] = np.array(np.append(X0, X), dtype = np.float64, order = 'F')
        y[:Np] = np.array(np.append(Y0, Y), dtype = np.float64, order = 'F')
        z[:Np] = np.array(np.append(Z0, Z), dtype = np.float64, order = 'F')
        Nu, Sigma, Volume = J.getVars(Particles, ['Nu', 'Sigma', 'Volume'])
        Nu[:Np] = VPMParameters['KinematicViscosity']
        Sigma[:Np] = np.array(np.append(S0, S), dtype = np.float64, order = 'F').reshape(Np)
        Volume[:Np] = 4./3.*np.pi*Sigma[:]**3

    def computeInducedVelocityOnLiftinLines(t = [], Nsource = 0, Target = [], TargetSigma = [], WakeInducedVelocity = []):
        TargetBase = I.newCGNSBase('LiftingLine', cellDim=1, physDim=3)
        TargetBase[2] = I.getZones(Target)
        Kernel = Kernel_str2int[getParameter(t, 'RegularisationKernel')]
        return VPM.induce_total_velocity_on_lifting_line(t, Nsource, Kernel, TargetBase,TargetSigma,
                                                                                WakeInducedVelocity)

    def addParticlesFromLiftingLineSources(t = [], Sources = [], SourcesM1 = [], NumberParticlesShedPerStation = 0, NumberSource = 0):
        SourcesBase = I.newCGNSBase('Sources', cellDim=1, physDim=3)
        SourcesBase[2] = I.getZones(Sources)
        SourcesBaseM1 = I.newCGNSBase('SourcesM1', cellDim=1, physDim=3)
        SourcesBaseM1[2] = I.getZones(SourcesM1)
        return VPM.generate_particles_on_lifting_lines(t, SourcesBase, SourcesBaseM1,
                                                        NumberParticlesShedPerStation, NumberSource)

    def getLiftingLineParameters(t = []): return J.get(pickParticlesZone(t), '.LiftingLine#Parameters')

    def relaxCirculationAndGetImbalance(GammaOld = [], GammaRelax = 0., Sources = []):
        GammaError = 0
        for i in range(len(Sources)):
            GammaNew, = J.getVars(Sources[i],['Gamma'])
            GammaError = max(GammaError, max(abs(GammaNew - GammaOld[i]))/max(1e-12,np.mean(abs(GammaNew))))
            GammaNew[:] = (1. - GammaRelax)*GammaOld[i] + GammaRelax*GammaNew
            GammaOld[i][:] = GammaNew
        return GammaError

    def shedParticlesFromLiftingLines(t = [], PolarsInterpolatorDict = {}, IterationInfo = {}):
        timeLL = time()
        LiftingLines = LL.getLiftingLines(t)
        if not LiftingLines: return IterationInfo
        ShieldsBase = I.getNodeFromName2(t, 'ShieldsBase')
        ShieldBoxes = I.getZones(ShieldsBase)
        ParticlesBase = I.getNodeFromName2(t, 'ParticlesBase')
        Particles = pickParticlesZone(t)

        if not Particles: raise ValueError('"Particles" zone not found in ParticlesTree')

        VPM_Params = getVPMParameters(Particles)
        LL_Params = getLiftingLineParameters(Particles)

        LiftingLinesM1 = [I.copyTree(ll) for ll in LiftingLines]
        h = VPM_Params['Resolution'][0]
        Sigma0 = VPM_Params['Sigma0'][0]
        dt = VPM_Params['TimeStep']
        U0 = VPM_Params['VelocityFreestream']
        MaskShedParticles = LL_Params['ShedParticlesIndex']
        for LiftingLineM1 in LiftingLinesM1:
            x, y, z = J.getxyz(LiftingLineM1)
            ui, vi, wi = J.getVars(LiftingLineM1, ['VelocityInduced' + i for i in 'XYZ'])
            x += dt*(U0[0] + ui)
            y += dt*(U0[1] + vi)
            z += dt*(U0[2] + wi)

        LL.computeKinematicVelocity(LiftingLinesM1)
        LL.moveLiftingLines(LiftingLines, dt)
        LL.assembleAndProjectVelocities(LiftingLines)

        AllAbscissaSegments, SigmaDistributionOnLiftingLine, SigmaDistribution = [], [], []
        for LiftingLine in LiftingLines:
            VPM_Parameters = J.get(LiftingLine,'.VPM#Parameters')
            AllAbscissaSegments += [VPM_Parameters['ParticleDistribution']]
            SigmaDistributionOnLiftingLine.extend(np.array(VPM_Parameters['SigmaDistributionOnLiftingLine'], order='F', dtype=np.float64))
            SigmaDistribution.extend(np.array(VPM_Parameters['SigmaDistribution'], order='F', dtype=np.float64))

        Sources = LL.buildVortexParticleSourcesOnLiftingLine(LiftingLines, AbscissaSegments=AllAbscissaSegments, IntegralLaw='linear')
        SourcesM1 = LL.buildVortexParticleSourcesOnLiftingLine(LiftingLinesM1, AbscissaSegments=AllAbscissaSegments, IntegralLaw='linear')

        px, py, pz = J.getxyz(Particles)
        NumberParticlesShedPerStation, NewSigma = [], []
        NumberSource = 0
        for Source in Sources:
            sx, sy, sz = J.getxyz(Source)
            NumberSourceM1 = NumberSource
            NumberSource = NumberSourceM1 + len(sx) - 1
            px[NumberSourceM1: NumberSource] = (sx[1:] + sx[:-1])/2.
            py[NumberSourceM1: NumberSource] = (sy[1:] + sy[:-1])/2.
            pz[NumberSourceM1: NumberSource] = (sz[1:] + sz[:-1])/2.
            mask = MaskShedParticles[NumberSourceM1: NumberSource]
            TrailingDistance = np.linalg.norm(np.vstack([px[NumberSourceM1: NumberSource] - px[mask],
                                                         py[NumberSourceM1: NumberSource] - py[mask],
                                                         pz[NumberSourceM1: NumberSource] - pz[mask]]), axis = 0)
            NumberParticlesShedPerStation += [[max(int(round(dy/h - 0.95)), 0) for dy in TrailingDistance]]
            for i in range(len(sx) - 1): NewSigma += [SigmaDistribution[i + NumberSourceM1]]*NumberParticlesShedPerStation[-1][i]

        NumberParticlesShedPerStation = np.hstack(np.array(NumberParticlesShedPerStation, dtype=np.int32, order = 'F'))
        NumberParticlesShed = np.sum(NumberParticlesShedPerStation)

        SigmaDistributionOnLiftingLine = np.hstack(np.array(SigmaDistributionOnLiftingLine, dtype=np.float64))
        WakeInducedVelocity = getInducedVelocityFromWake(t, LiftingLines, SigmaDistributionOnLiftingLine)

        N0 = Particles[1][0][0] * 1

        roll(Particles, N0 - NumberSource)
        extend(Particles, NumberParticlesShed)


        Particles[1][0][0] = N0
        KinematicViscosity, Volume, Sigma = J.getVars(Particles, ['Nu', 'Volume', 'Sigma'])
        Sigma[N0 - NumberSource:] = SigmaDistribution + NewSigma
        GammaOld = [I.getNodeFromName3(Source, 'Gamma')[1] for Source in Sources]

        GammaThreshold = LL_Params['CirculationThreshold']
        GammaRelax = LL_Params['CirculationRelaxation']
        GammaError = GammaThreshold + 1.

        ni = 0
        for Ni in range(int(LL_Params['MaxLiftingLineSubIterations'])):
            Particles[1][0][0] = N0
            addParticlesFromLiftingLineSources(t, Sources, SourcesM1, NumberParticlesShedPerStation, NumberSource)
            computeInducedVelocityOnLiftinLines(t, NumberParticlesShed + NumberSource, LiftingLines, SigmaDistributionOnLiftingLine, WakeInducedVelocity)
            LL.assembleAndProjectVelocities(LiftingLines)
            LL._applyPolarOnLiftingLine(LiftingLines, PolarsInterpolatorDict, ['Cl', 'Cd'])
            IntegralLoads = LL.computeGeneralLoadsOfLiftingLine(LiftingLines)
            Sources = LL.buildVortexParticleSourcesOnLiftingLine(LiftingLines, AbscissaSegments=AllAbscissaSegments, IntegralLaw='linear')
            GammaError = relaxCirculationAndGetImbalance(GammaOld, GammaRelax, Sources)

            ni += 1
            if GammaError < GammaThreshold: break


        Particles[1][0][0] = N0 + NumberParticlesShed
        LL.computeGeneralLoadsOfLiftingLine(LiftingLines,
                UnsteadyData={'IterationNumber':VPM_Params['CurrentIteration'],
                              'Time':VPM_Params['Time'],
                              'CirculationSubiterations':ni,
                              'CirculationError':GammaError},
                                UnsteadyDataIndependentAbscissa='IterationNumber')

        AlphaXYZ = np.vstack(J.getVars(Particles, ['Alpha'+i for i in 'XYZ']))
        AlphaNorm = np.linalg.norm(AlphaXYZ[:, N0 - NumberSource:],axis=0)
        StrengthMagnitude = J.getVars(Particles, ['StrengthMagnitude'])[0]
        StrengthMagnitude[N0 - NumberSource:] = AlphaNorm[:]

        Volume[N0 - NumberSource:] = (2.*np.pi)**1.5*Sigma[N0 - NumberSource:]
        KinematicViscosity[N0 - NumberSource:] = VPM_Params['KinematicViscosity'][0] + \
            (Sigma0*VPM_Params['EddyViscosityConstant'][0])**2*2**0.5*\
            AlphaNorm[:]/Volume[N0 - NumberSource:]

        VorticityX, VorticityY, VorticityZ = J.getVars(Particles, ['Vorticity'+i for i in 'XYZ'])
        VorticityX[N0 - NumberSource:] = AlphaXYZ[0,N0 - NumberSource:]/Volume[N0 - NumberSource:]
        VorticityY[N0 - NumberSource:] = AlphaXYZ[1,N0 - NumberSource:]/Volume[N0 - NumberSource:]
        VorticityZ[N0 - NumberSource:] = AlphaXYZ[2,N0 - NumberSource:]/Volume[N0 - NumberSource:]
        VorticityXYZ = np.vstack([VorticityX, VorticityY, VorticityZ])
        VorticityNorm = np.linalg.norm(VorticityXYZ[:, N0 - NumberSource:],axis=0)
        VorticityMagnitude = J.getVars(Particles, ['VorticityMagnitude'])[0]
        VorticityMagnitude[N0 - NumberSource:] = VorticityNorm[:]

        Ns = 0
        posNumberParticlesShed = 0
        for Source in Sources:
            SourceX = I.getValue(I.getNodeFromName(Source, 'CoordinateX'))
            for i in range(Ns, Ns + len(SourceX) - 1):
                if NumberParticlesShedPerStation[posNumberParticlesShed]:
                    MaskShedParticles[i] = sum(NumberParticlesShedPerStation[:posNumberParticlesShed]) + NumberSource
                else: 
                    MaskShedParticles[i] += NumberParticlesShed

                posNumberParticlesShed += 1
            Ns += len(SourceX) - 1
        roll(Particles, NumberSource + NumberParticlesShed)
        
        IterationInfo['Circulation error'] = GammaError
        IterationInfo['Number of sub-iterations (LL)'] = ni
        IterationInfo['Number of shed particles'] = NumberParticlesShed
        IterationInfo['Lifting Line time'] = time() - timeLL
        return IterationInfo

####################################################################################################
####################################################################################################
######################################### Coeff/Loads Aero #########################################
####################################################################################################
####################################################################################################
    def getAerodynamicCoefficientsOnLiftingLine(LiftingLines = [], StdDeviationSample = 50, IterationInfo = {}, Freestream = True, Wings = False, Surface = 0.):
        if LiftingLines:
            if Wings: IterationInfo = getAerodynamicCoefficientsOnWing(LiftingLines, Surface,
                                            StdDeviationSample = StdDeviationSample, IterationInfo =
                                                                                      IterationInfo)
            else:
                if Freestream: IterationInfo = getAerodynamicCoefficientsOnPropeller(LiftingLines,
                                            StdDeviationSample = StdDeviationSample, IterationInfo =
                                                                                      IterationInfo)
                else: IterationInfo = getAerodynamicCoefficientsOnRotor(LiftingLines,
                                            StdDeviationSample = StdDeviationSample, IterationInfo =
                                                                                      IterationInfo)
        return IterationInfo

    def getAerodynamicCoefficientsOnPropeller(LiftingLines = [], StdDeviationSample = 50, IterationInfo = {}):
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
        cT = T/(Rho*n**2*D**4)
        cP = P/(Rho*n**3*D**5)
        Uinf = np.linalg.norm(U0 - V)
        Eff = T/P*Uinf
        std_Thrust, std_Power = getStandardDeviationBlade(LiftingLines = LiftingLines,
                                                            StdDeviationSample = StdDeviationSample)
        IterationInfo['Thrust'] = T
        IterationInfo['Thrust Standard Deviation'] = std_Thrust
        IterationInfo['Power'] = P
        IterationInfo['Power Standard Deviation'] = std_Power
        IterationInfo['cT'] = cT
        IterationInfo['cP'] = cP
        IterationInfo['Eff'] = Eff
        return IterationInfo

    def getAerodynamicCoefficientsOnRotor(LiftingLines = [], StdDeviationSample = 50,IterationInfo = {}):
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
        Eff = np.sqrt(2./np.pi)*cT**1.5/cP

        std_Thrust, std_Power = getStandardDeviationBlade(LiftingLines = LiftingLines, StdDeviationSample = StdDeviationSample)
        IterationInfo['Thrust'] = T
        IterationInfo['Thrust Standard Deviation'] = std_Thrust
        IterationInfo['Power'] = P
        IterationInfo['Power Standard Deviation'] = std_Power
        IterationInfo['cT'] = cT
        IterationInfo['cP'] = cP
        IterationInfo['Eff'] = Eff
        return IterationInfo

    def getAerodynamicCoefficientsOnWing(LiftingLines = [], Surface = 0., StdDeviationSample = 50, IterationInfo = {}):
        LiftingLine = I.getZones(LiftingLines)[0]
        Rho = I.getValue(I.getNodeFromName(LiftingLine, 'Density'))
        U0 = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityFreestream'))
        V = I.getValue(I.getNodeFromName(LiftingLine, 'VelocityTranslation'))
        x, y, z = J.getxyz(LiftingLine)
        IntegralLoads = LL.computeGeneralLoadsOfLiftingLine(LiftingLine)
        Rmax = (x[0]**2 + y[0]**2 + z[0]**2)**0.5
        Rmin = (x[1]**2 + y[1]**2 + z[1]**2)**0.5
        Fz = IntegralLoads['ForceZ'][0]
        Fx = IntegralLoads['ForceX'][0]
        q0 = 0.5*Rho*Surface*np.linalg.norm(U0 - V)**2
        cL = Fz/q0
        cD = Fx/q0
        std_Thrust, std_Drag = getStandardDeviationWing(LiftingLines = LiftingLines,
                                                            StdDeviationSample = StdDeviationSample)
        IterationInfo['Lift'] = Fz
        IterationInfo['Lift Standard Deviation'] = std_Thrust
        IterationInfo['Drag'] = Fx
        IterationInfo['Drag Standard Deviation'] = std_Drag
        IterationInfo['cL'] = cL
        IterationInfo['cD'] = cD
        IterationInfo['f'] = cL/cD
        return IterationInfo

    def getStandardDeviationWing(LiftingLines = [], StdDeviationSample = 50):
        LiftingLine = I.getZones(LiftingLines)[0]
        UnsteadyLoads = I.getNodeFromName(LiftingLine, '.UnsteadyLoads')
        Thrust = I.getValue(I.getNodeFromName(UnsteadyLoads, 'Thrust'))
        if type(Thrust) == np.ndarray or type(Thrust) == list:
            StdDeviationSample = min(StdDeviationSample,len(Thrust))
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
            StdDeviationSample = min(StdDeviationSample,len(Thrust))
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
    def setVisualization(t = [], ParticlesColorField = 'VorticityMagnitude', ParticlesRadius = '{Sigma}/10', addLiftingLineSurfaces = True, AirfoilPolarsFilename = None):
        Particles = pickParticlesZone(t)
        Sigma = I.getValue(I.getNodeFromName(Particles, 'Sigma'))
        C._initVars(Particles, 'radius=' + ParticlesRadius)
        if not ParticlesColorField: ParticlesColorField = 'VorticityMagnitude'
        CPlot._addRender2Zone(Particles, material = 'Sphere',
            color = 'Iso:' + ParticlesColorField, blending=0.6, shaderParameters=[0.04, 0])
        LiftingLines = LL.getLiftingLines(t)
        for zone in LiftingLines:
            CPlot._addRender2Zone(zone, material = 'Flat', color = 'White')
        Shields = I.getZones(I.getNodeFromName2(t, 'ShieldsBase'))
        for zone in Shields:
            CPlot._addRender2Zone(zone, material = 'Glass', color = 'White', blending=0.6)
        if addLiftingLineSurfaces:
            if not AirfoilPolarsFilename:
                ERRMSG = J.FAIL+('production of surfaces from lifting-line requires'
                    ' attribute AirfoilPolars')+J.ENDC
                raise AttributeError(ERRMSG)
            LiftingLineSurfaces = []
            for ll in LiftingLines:
                surface = LL.postLiftingLine2Surface(ll, AirfoilPolarsFilename)
                deletePrintedLines()
                surface[0] = ll[0]+'.surf'
                CPlot._addRender2Zone(surface, material = 'Solid', color = '#ECF8AB')
                LiftingLineSurfaces += [surface]
            I.createUniqueChild(t, 'LiftingLineSurfaces', 'CGNSBase_t',
                value=np.array([2, 3], order = 'F'), children=LiftingLineSurfaces)

        CPlot._addRender2PyTree(t, mode = 'Render', colormap = 'Blue2Red', isoLegend=1,
                                   scalarField=ParticlesColorField)

    def saveImage(t = [], ShowInScreen=False, ImagesDirectory = 'FRAMES', **DisplayOptions):
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
            machine = os.getenv('MAC', 'ld')
            if 'spiro' in machine or 'sator' in machine:
                DisplayOptions['offscreen'] = 1 # TODO solve bug https://elsa.onera.fr/issues/10536
            else:
                DisplayOptions['offscreen'] = 2

        CPlot.display(t, **DisplayOptions)
        sleep(0.1)
        if 'backgroundFile' not in DisplayOptions:
            MOLA = os.getenv('MOLA')
            MOLASATOR = os.getenv('MOLASATOR')
            for MOLAloc in [MOLA, MOLASATOR]:
                backgroundFile = os.path.join(MOLAloc, 'MOLA', 'GUIs', 'background.png')
                if os.path.exists(backgroundFile):
                    CPlot.setState(backgroundFile=backgroundFile)
                    CPlot.setState(bgColor =13)
                    break

    def open(filename = ''):
        t = C.convertFile2PyTree(filename)
        deletePrintedLines()
        return t

    def save(t = [], filename = '', VisualisationOptions = {}):
        try:
            if os.path.islink(filename):
                os.unlink(filename)
            else:
                os.remove(filename)
        except:
            pass

        if VisualisationOptions: setVisualization(t, **VisualisationOptions)

        C.convertPyTree2File(t, filename)
        deletePrintedLines()
        if VisualisationOptions:
            Particles = pickParticlesZone(t)
            I._rmNodesByName(Particles, 'radius')

    def loadAirfoilPolars(filename = ''): return LL.loadPolarsInterpolatorDict(filename)

    def printIterationInfo(IterationInfo = {}, PSE = False, Wings = False):
        msg = '||' + '{:-^50}'.format(' Iteration ' + '{:d}'.format(IterationInfo['Iteration']) + \
                        ' (' + '{:.1f}'.format(IterationInfo['Percentage']) + '%) ') + '\n'
        msg += '||' + '{:34}'.format('Physical time') + \
                        ': ' + '{:.5f}'.format(IterationInfo['Physical time']) + ' s' + '\n'
        msg += '||' + '{:34}'.format('Number of Particles') + \
                        ': ' + '{:d}'.format(IterationInfo['Number of Particles']) + '\n'
        msg += '||' + '{:34}'.format('Total iteration time') + \
                        ': ' + '{:.2f}'.format(IterationInfo['Total iteration time']) + ' s' + '\n'
        msg += '||' + '{:34}'.format('Total simulation time') + \
                        ': ' + '{:.1f}'.format(IterationInfo['Total simulation time']) + ' s' + '\n'
        msg += '||' + '{:-^50}'.format(' Loads ') + '\n'
        if (Wings and 'Lift' in IterationInfo) or (not Wings and 'Thrust' in IterationInfo):
            if (Wings):
                msg += '||' + '{:34}'.format('Lift') + \
                      ': ' + '{:.3f}'.format(IterationInfo['Lift']) + ' N' + '\n'
                msg += '||' + '{:34}'.format('Lift Standard Deviation') + \
                      ': ' + '{:.3g}'.format(IterationInfo['Lift Standard Deviation']) + ' N' + '\n'
                msg += '||' + '{:34}'.format('Drag') + \
                      ': ' + '{:.3f}'.format(IterationInfo['Drag']) + ' N' + '\n'
                msg += '||' + '{:34}'.format('Drag Standard Deviation') + \
                      ': ' + '{:.3g}'.format(IterationInfo['Drag Standard Deviation']) + ' N' + '\n'
                msg += '||' + '{:34}'.format('cL') + \
                      ': ' + '{:.4f}'.format(IterationInfo['cL']) + '\n'
                msg += '||' + '{:34}'.format('cD') + \
                      ': ' + '{:.5f}'.format(IterationInfo['cD']) + '\n'
                msg += '||' + '{:34}'.format('f') + \
                      ': ' + '{:.4f}'.format(IterationInfo['f']) + '\n'
            else:
                msg += '||' + '{:34}'.format('Thrust') + \
                    ': ' + '{:.3f}'.format(IterationInfo['Thrust']) + ' N' + '\n'
                msg += '||' + '{:34}'.format('Thrust Standard Deviation') + \
                    ': ' + '{:.3g}'.format(IterationInfo['Thrust Standard Deviation']) + ' N' + '\n'
                msg += '||' + '{:34}'.format('Power') + \
                    ': ' + '{:.3f}'.format(IterationInfo['Power']) + ' W' + '\n'
                msg += '||' + '{:34}'.format('Power Standard Deviation') + \
                    ': ' + '{:.3g}'.format(IterationInfo['Power Standard Deviation']) + ' W' + '\n'
                msg += '||' + '{:34}'.format('cT') + \
                    ': ' + '{:.5f}'.format(IterationInfo['cT']) + '\n'
                msg += '||' + '{:34}'.format('Cp') + \
                    ': ' + '{:.5f}'.format(IterationInfo['cP']) + '\n'
                msg += '||' + '{:34}'.format('Eff') + \
                    ': ' + '{:.5f}'.format(IterationInfo['Eff']) + '\n'
        msg += '||' + '{:-^50}'.format(' Population Control ') + '\n'
        msg += '||' + '{:34}'.format('Number of Particles beyond cutoff') + \
                     ': ' + '{:d}'.format(IterationInfo['Number of Particles beyond cutoff']) + '\n'
        msg += '||' + '{:34}'.format('Number of split Particles') + \
                     ': ' + '{:d}'.format(IterationInfo['Number of split Particles']) + '\n'
        msg += '||' + '{:34}'.format('Number of depleted Particles') + \
                     ': ' + '{:d}'.format(IterationInfo['Number of depleted Particles']) + '\n'
        msg += '||' + '{:34}'.format('Number of merged Particles') + \
                     ': ' + '{:d}'.format(IterationInfo['Number of merged Particles']) + '\n'
        msg += '||' + '{:34}'.format('Control Computation time') + \
                     ': ' + '{:.2f}'.format(IterationInfo['Population Control time']) + ' s (' + \
                                          '{:.1f}'.format(IterationInfo['Population Control time']/\
                                          IterationInfo['Total iteration time']*100.) + '%) ' + '\n'
        if 'Circulation error' in IterationInfo:
            msg += '||' + '{:-^50}'.format(' Lifting Line ') + '\n'
            msg += '||' + '{:34}'.format('Circulation error') + \
                         ': ' + '{:.5e}'.format(IterationInfo['Circulation error']) + '\n'
            msg += '||' + '{:34}'.format('Number of sub-iterations') + \
                         ': ' + '{:d}'.format(IterationInfo['Number of sub-iterations (LL)']) + '\n'
            msg += '||' + '{:34}'.format('Number of shed Particles') + \
                         ': ' + '{:d}'.format(IterationInfo['Number of shed particles']) + '\n'
            msg += '||' + '{:34}'.format('Lifting Line Computation time') + \
                         ': ' + '{:.2f}'.format(IterationInfo['Lifting Line time']) + ' s (' + \
                                                '{:.1f}'.format(IterationInfo['Lifting Line time']/\
                                          IterationInfo['Total iteration time']*100.) + '%) ' + '\n'

        if 'Rel. err. of Vorticity' in IterationInfo:
            msg += '||' + '{:-^50}'.format(' Hybrid Solver ') + '\n'
            msg += '||' + '{:34}'.format('Eulerian Vorticity lost') + \
                          ': ' + '{:.1g}'.format(IterationInfo['Eulerian Vorticity lost']) + \
                          ' s-1 (' + '{:.1f}'.format(IterationInfo['Eulerian Vorticity lost per'])+\
                                                                                        '%) ' + '\n'
            msg += '||' + '{:34}'.format('Minimum Eulerian Vorticity') + \
                          ': ' + '{:.2g}'.format(IterationInfo['Minimum Eulerian Vorticity']) + '\n'
            msg += '||' + '{:34}'.format('Number of sub-iterations') + \
                          ': ' + '{:d}'.format(IterationInfo['Number of sub-iterations (E)']) + '\n'
            msg += '||' + '{:34}'.format('Number of Hybrids Generated') + \
                          ': ' + '{:d}'.format(IterationInfo['Number of Hybrids Generated']) + '\n'
            msg += '||' + '{:34}'.format('Rel. err. of Vorticity') + \
                          ': ' + '{:.5e}'.format(IterationInfo['Rel. err. of Vorticity']) + '\n'
            msg += '||' + '{:34}'.format('Strength Computation time') + \
                          ': ' + '{:.2f}'.format(IterationInfo['Strength computation time']) + \
                               ' s (' + '{:.1f}'.format(IterationInfo['Strength computation time']/\
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
                        ': ' + '{:e}'.format(IterationInfo['Rel. err. of Velocity Gradient']) + '\n'
        if PSE: msg += '||' + '{:34}'.format('Rel. err. of PSE') + \
                        ': ' + '{:e}'.format(IterationInfo['Rel. err. of PSE']) + '\n'
        msg += '||' + '{:34}'.format('FMM Computation time') + \
                        ': ' + '{:.2f}'.format(IterationInfo['FMM time']) + ' s (' + \
                                                         '{:.1f}'.format(IterationInfo['FMM time']/\
                                          IterationInfo['Total iteration time']*100.) + '%) ' + '\n'
        msg += '||' + '{:=^50}'.format('')
        print(msg)

    def deletePrintedLines(NumberOfLineToDelete = 1):
        for i in range(NumberOfLineToDelete):
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')

    if __name__ == '__main__':
        main()

####################################################################################################
####################################################################################################
############################################## Solver ##############################################
####################################################################################################
####################################################################################################
    def compute(VPMParameters = {}, HybridParameters = {}, LiftingLineParameters = {}, PolarsFilename = None, EulerianPath = None, LiftingLinePath = None, NumberOfIterations = 1000, RestartPath = None, DIRECTORY_OUTPUT = 'OUTPUT', VisualisationOptions = {'addLiftingLineSurfaces':True}, StdDeviationSample = 50, SaveVPMPeriod = 100, Verbose = True, SaveImageOptions={}, Surface = 0., FieldsExtractionGrid = [], SaveFieldsPeriod = np.inf, SaveImagePeriod = np.inf):
        try: os.makedirs(DIRECTORY_OUTPUT)
        except: pass

        if PolarsFilename: AirfoilPolars = loadAirfoilPolars(PolarsFilename)
        else: AirfoilPolars = None

        if RestartPath:
            t = open(RestartPath)
            try: tE = open('tE.cgns')
            except: tE = []
        else:
            if LiftingLinePath: LiftingLine = open(LiftingLinePath)
            else: LiftingLine = []
            #if EulerianPath: EulerianMesh = open(EulerianPath)
            #else: EulerianMesh = []
            EulerianMesh = EulerianPath
            t, tE = initialiseVPM(EulerianMesh = EulerianMesh, HybridParameters = HybridParameters,
                        LiftingLineTree = LiftingLine,LiftingLineParameters = LiftingLineParameters,
                        PolarInterpolator = AirfoilPolars, VPMParameters = VPMParameters)

        
        IterationInfo = {'Rel. err. of Velocity': 0, 'Rel. err. of Velocity Gradient': 0,
                            'Rel. err. of PSE': 0}
        TotalTime = time()
        sp = getVPMParameters(t)
        Np = pickParticlesZone(t)[1][0]
        LiftingLines = LL.getLiftingLines(t)

        h = sp['Resolution'][0]
        it = sp['CurrentIteration']
        simuTime = sp['Time']
        PSE = DiffusionScheme_str2int[sp['DiffusionScheme']] == 1
        Freestream = (np.linalg.norm(sp['VelocityFreestream']) != 0.)
        Wing = (len(I.getZones(LiftingLines)) == 1)
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
            
            IterationTime = time()
            computeNextTimeStep(t)
            IterationInfo['Iteration'] = it[0]
            IterationInfo['Percentage'] = it[0]/NumberOfIterations*100.
            IterationInfo['Physical time'] = simuTime[0]
            IterationInfo = generateParticlesInHybridInterfaces(t, tE, IterationInfo)
            IterationInfo = populationControl(t, [], IterationInfo)
            IterationInfo = shedParticlesFromLiftingLines(t, AirfoilPolars, IterationInfo)
            IterationInfo['Number of Particles'] = Np[0]
            IterationInfo = solveVorticityEquation(t, IterationInfo = IterationInfo)
            IterationInfo['Total iteration time'] = time() - IterationTime
            IterationInfo = getAerodynamicCoefficientsOnLiftingLine(LiftingLines, Wings = Wing,
                                   StdDeviationSample = StdDeviationSample, Freestream = Freestream, 
                                                   IterationInfo = IterationInfo, Surface = Surface)
            IterationInfo['Total simulation time'] = time() - TotalTime
            if Verbose: printIterationInfo(IterationInfo, PSE = PSE, Wings = Wing)

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
            
        save(t, DIRECTORY_OUTPUT + '.cgns', VisualisationOptions)
        for _ in range(3): print('||' + '{:=^50}'.format(''))
        print('||' + '{:-^50}'.format(' End of VPM computation '))
        for _ in range(3): print('||' + '{:=^50}'.format(''))

        exit()

    def computePolar(Parameters = {}, dictPolar = {}, PolarsFilename = '', LiftingLinePath = 'LiftingLine.cgns', DIRECTORY_OUTPUT = 'POLARS', RestartPath = None, MaxNumberOfIterationsPerPolar = 200, NumberOfIterationsForTransition = 0, StdDeviationSample = 10, Surface = 1., Verbose = True, MaxThrustStandardDeviation = 1, MaxPowerStandardDeviation = 100, VisualisationOptions = {'addLiftingLineSurfaces':True}):
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

            t = open(os.path.join(DIRECTORY_OUTPUT, dictPolar['LastPolar']))
        else:
            LiftingLine = open(LiftingLinePath)
            t = initialiseVPM(LiftingLineTree = LiftingLine, VPMParameters = Parameters,
                    HybridParameters = {}, LiftingLineParameters = {}, PolarInterpolator = 
                                                                                      AirfoilPolars)
            if dictPolar['VariableName'] == 'Pitch':
                dictPolar['VariableName'] = 'Twist'
                dictPolar['Pitch'] = True

            dictPolar['Offset'] = I.getValue(I.getNodeFromName(t, dictPolar['VariableName']))
            if dictPolar['overwriteVPMWithVariables']: dictPolar['Offset'] *= 0.

            if not (isinstance(dictPolar['Offset'], list) or \
                                                       isinstance(dictPolar['Offset'], np.ndarray)):
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

        sp = getVPMParameters(t)
        Particles = pickParticlesZone(t)
        Np = Particles[1][0]
        LiftingLines = LL.getLiftingLines(t)

        h = sp['Resolution'][0]
        f = sp['RedistributionPeriod'][0]
        it = sp['CurrentIteration']
        dt = sp['TimeStep']
        U0 = np.linalg.norm(sp['VelocityFreestream'])
        simuTime = sp['Time']
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
                    shedParticlesFromLiftingLines(t, AirfoilPolars)
                    solveVorticityEquation(t)

                    if Verbose:
                        if n != 1: deletePrintedLines()
                        print('||' + '{:-^50}'.format(' Transition ' + '{:.1f}'.format(n/\
                                              NumberOfIterationsForTransition*100.) + '% ') + ' \r')

            for LiftingLine in LiftingLines:
                LLVariable = I.getNodeFromName(LiftingLine, dictPolar['VariableName'])
                LLVariable[1] = dictPolar['Offset'] + Variable
            VPMVariable = I.getNodeFromName(Particles, dictPolar['VariableName'])
            if VPMVariable != None: VPMVariable[1] = dictPolar['Offset'] + Variable
            if 'TimeStepFunction' in dictPolar:
                TimeStepFunction_str2int[dictPolar['TimeStepFunction']](t, LiftingLines, 
                                                             dictPolar['TimeStepFunctionParameter'])
            it0 = it[0]
            stdThrust = MaxThrustStandardDeviation + 1
            stdPower = MaxPowerStandardDeviation + 1

            while (it[0] - it0 < MaxNumberOfIterationsPerPolar and (MaxThrustStandardDeviation < \
                stdThrust or MaxPowerStandardDeviation < stdPower)) or \
                                                                 (it[0] - it0 < StdDeviationSample):
                msg = '||' + '{:-^50}'.format(' Iteration ' + '{:d}'.format(it[0] - it0) + ' ')+'\n'
                computeNextTimeStep(t)
                populationControl(t, NoRedistributionRegions=[])
                shedParticlesFromLiftingLines(t, AirfoilPolars)
                solveVorticityEquation(t)

                msg += '||' + '{:-^50}'.format(' Loads ') + '\n'
                if NbLL == 1:
                    IterationInfo = getAerodynamicCoefficientsOnWing(LiftingLines, Surface,
                                                            StdDeviationSample = StdDeviationSample)
                    msg += '||' + '{:34}'.format('Lift')
                    msg += ': ' + '{:.3f}'.format(IterationInfo['Lift']) + ' N' + '\n'
                    msg += '||' + '{:34}'.format('Lift Standard Deviation')
                    msg += ': '+ '{:.3g}'.format(IterationInfo['Lift Standard Deviation'])+' N'+'\n'
                    msg += '||' + '{:34}'.format('Drag')
                    msg += ': ' + '{:.3f}'.format(IterationInfo['Drag']) + ' N' + '\n'
                    msg += '||' + '{:34}'.format('Drag Standard Deviation')
                    msg += ': '+ '{:.3g}'.format(IterationInfo['Drag Standard Deviation'])+' N'+'\n'
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
                    msg +=': '+'{:.3g}'.format(IterationInfo['Thrust Standard Deviation'])+' N'+'\n'
                    msg += '||' + '{:34}'.format('Power')
                    msg += ': ' + '{:.3f}'.format(IterationInfo['Power']) + ' W' + '\n'
                    msg += '||' + '{:34}'.format('Power Standard Deviation')
                    msg += ': '+'{:.3g}'.format(IterationInfo['Power Standard Deviation'])+' W'+'\n'
                    msg += '||'+'{:34}'.format('cT') +': '+'{:.5f}'.format(IterationInfo['cT'])+'\n'
                    msg += '||'+'{:34}'.format('Cp') +': '+'{:.5f}'.format(IterationInfo['cP'])+'\n'
                    msg +='||'+'{:34}'.format('Eff')+': '+'{:.5f}'.format(IterationInfo['Eff'])+'\n'
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
                dictPolar['LiftStandardDeviation'] = np.append(dictPolar['LiftStandardDeviation'],
                                                           IterationInfo['Lift Standard Deviation'])
                dictPolar['DragStandardDeviation'] = np.append(dictPolar['DragStandardDeviation'], 
                                                           IterationInfo['Drag Standard Deviation'])
            else:
                dictPolar['Thrust'] = np.append(dictPolar['Thrust'], IterationInfo['Thrust'])
                dictPolar['Power'] = np.append(dictPolar['Power'], IterationInfo['Power'])
                dictPolar['cT'] = np.append(dictPolar['cT'], IterationInfo['cT'])
                dictPolar['cP'] = np.append(dictPolar['cP'], IterationInfo['cP'])
                dictPolar['Efficiency'] = np.append(dictPolar['Efficiency'], IterationInfo['Eff'])
                dictPolar['ThrustStandardDeviation'] =np.append(dictPolar['ThrustStandardDeviation']
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
                dictPolar['LastPolar'] = 'VPM_Polar_' + dictPolar['VariableName'] + '_'+ str(round(\
                                                                             Variable, 2)) + '.cgns'
                filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_Polars_' + dictPolar['VariableName']\
                                                                                          + '.cgns')
                msg += dictPolar['VariableName'] + ' = ' + str(round(Variable, 2)) + ' '
            else:
                dictPolar['LastPolar'] = 'VPM_Polar_Pitch_' + str(round(Variable, 2)) + '.cgns'
                filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_Polars_Pitch.cgns')
                msg += 'Pitch = ' + str(round(Variable, 2)) + ' '
            if Verbose: print('||' + '{:-^50}'.format(msg))

            J.set(PolarsTree, 'Polars', **dictPolar)
            save(PolarsTree, filename)

            filename = os.path.join(DIRECTORY_OUTPUT, dictPolar['LastPolar'])
            save(I.merge([PolarsTree, t]), filename, VisualisationOptions)
            J.createSymbolicLink(filename,  DIRECTORY_OUTPUT + '.cgns')
            
            if Verbose:
                print('||' + '{:=^50}'.format('') + '\n||' + '{:=^50}'.format('') +\
                        '\n||' + '{:=^50}'.format(''))

        
        TotalTime = time() - TotalTime
        print('||' + '{:=^50}'.format(' Total time spent: ' +str(int(round(TotalTime//60)))+' min '\
                                           + str(int(round(TotalTime - TotalTime//60*60))) + ' s '))
        for _ in range(3): print('||' + '{:=^50}'.format(''))

    def extract(t = [], ExctractionTree = [], NbOfParticlesUsedForPrecisionEvaluation = 1000):
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
        VPM.wrap_extract_plane(t, tmpTree, int(NbOfParticlesUsedForPrecisionEvaluation), Kernel,
                                    EddyViscosityModel)
        tmpFields = J.getVars(I.getZones(tmpTree)[0], newFieldNames)

        for i, zone in enumerate(ExtractionZones):
            fields = J.getVars(zone, newFieldNames)
            for ft, f in zip(tmpFields, fields):
                fr = f.ravel(order = 'F')
                fr[:] = ft[NPtsPerZone[i]:NPtsPerZone[i+1]]

        return ExctractionTree

    def exit(): os._exit(0)





    
    







































'''
def initialiseLiftingLinesAndGetShieldBoxes(LiftingLines, PolarsInterpolatorDict, Resolution):
    LL.computeKinematicVelocity(LiftingLines)
    LL.assembleAndProjectVelocities(LiftingLines)
    LL._applyPolarOnLiftingLine(LiftingLines, PolarsInterpolatorDict, ['Cl', 'Cd'])
    LL.computeGeneralLoadsOfLiftingLine(LiftingLines)
    ShieldBoxes = buildShieldBoxesAroundLiftingLines(LiftingLines, Resolution)

    return ShieldBoxes

def buildShieldBoxesAroundLiftingLines(LiftingLines, Resolution):
    ShieldBoxes = []
    h = 2*Resolution
    for LiftingLine in I.getZones(LiftingLines):
        tx,ty,tz,bx,by,bz,nx,ny,nz = J.getVars(LiftingLine,
            ['tx','ty','tz','bx','by','bz','nx','ny','nz'])
        x,y,z = J.getxyz(LiftingLine)
        quads = []
        for i in range(len(tx)):
            quad = G.cart((-h/2.,-h/2.,0),(h,h,1),(2,2,1))
            T._rotate(quad,(0,0,0), ((1,0,0),(0,1,0),(0,0,1)),
                ((nx[i],ny[i],nz[i]), (bx[i],by[i],bz[i]), (tx[i],ty[i],tz[i])))
            T._translate(quad,(x[i],y[i],z[i]))
            quads += [ quad ]
        I._correctPyTree(quads, level=3)
        ShieldBox = G.stack(quads)
        ShieldBox[0] = LiftingLine[0] + '.shield'

        # in theory, this get+set is a copy by reference (e.g.: in-place
        # modification of RPM in LiftingLine will produce a modification of the
        # RPM of its associated ShieldBoxes)
        for paramsName in ['.Conditions','.Kinematics']:
            params = J.get(LiftingLine, paramsName)
            J.set(ShieldBox, paramsName, **params)
        ShieldBoxes += [ ShieldBox ]

    return ShieldBoxes

def updateParticlesStrength(Particles, MaskShedParticles, Sources, SourcesM1, NumberParticlesShedPerStation, NumberSource):
    Np = Particles[1][0]
    CoordinateX       = I.getNodeFromName3(Particles, "CoordinateX")
    CoordinateY       = I.getNodeFromName3(Particles, "CoordinateY")
    CoordinateZ       = I.getNodeFromName3(Particles, "CoordinateZ")
    AlphaX            = I.getNodeFromName3(Particles, "AlphaX")
    AlphaY            = I.getNodeFromName3(Particles, "AlphaY")
    AlphaZ            = I.getNodeFromName3(Particles, "AlphaZ")

    Ns = 0
    posEmbedded = Np[0] - NumberSource
    for k in range(len(Sources)):
        LLXj                  = I.getValue(I.getNodeFromName3(Sources[k], "CoordinateX"))
        LLYj                  = I.getValue(I.getNodeFromName3(Sources[k], "CoordinateY"))
        LLZj                  = I.getValue(I.getNodeFromName3(Sources[k], "CoordinateZ"))
        Gamma                = I.getValue(I.getNodeFromName3(Sources[k], "Gamma"))
        V2DXj                 = I.getValue(I.getNodeFromName3(Sources[k], "Velocity2DX"))
        V2DYj                 = I.getValue(I.getNodeFromName3(Sources[k], "Velocity2DY"))
        V2DZj                 = I.getValue(I.getNodeFromName3(Sources[k], "Velocity2DZ"))
        GammaM1              = I.getValue(I.getNodeFromName3(SourcesM1[k], "Gamma"))

        NsCurrent = len(LLXj)

        for i in range(NsCurrent - 1):
            Nshed = NumberParticlesShedPerStation[Ns + i] + 1
            vecj2D = -np.array([V2DXj[i + 1] + V2DXj[i], V2DYj[i + 1] + V2DYj[i], V2DZj[i + 1] + V2DZj[i]])
            dy = np.linalg.norm(vecj2D)
            vecj2D /= dy

            veci = np.array([(LLXj[i + 1] - LLXj[i]), (LLYj[i + 1] - LLYj[i]), (LLZj[i + 1] - LLZj[i])])
            dx = np.linalg.norm(veci)
            veci /= dx

            pos = 0.5*np.array([LLXj[i + 1] + LLXj[i], LLYj[i + 1] + LLYj[i], LLZj[i + 1] + LLZj[i]])
            vecj = np.array([CoordinateX[1][MaskShedParticles[Ns + i] - NumberSource], CoordinateY[1][MaskShedParticles[Ns + i] - NumberSource], CoordinateZ[1][MaskShedParticles[Ns + i] - NumberSource]]) - pos
            dy = np.linalg.norm(vecj)
            vecj /= Nshed
            
            GammaTrailing = (Gamma[i + 1] - Gamma[i])*dy/Nshed
            GammaShedding = (GammaM1[i + 1] + GammaM1[i] - (Gamma[i + 1] + Gamma[i]))/2.*dx/Nshed
            GammaBound = [GammaTrailing*vecj2D[0] + GammaShedding*veci[0], GammaTrailing*vecj2D[1] + GammaShedding*veci[1], GammaTrailing*vecj2D[2] + GammaShedding*veci[2]]
            CoordinateX[1][posEmbedded] = pos[0]
            CoordinateY[1][posEmbedded] = pos[1]
            CoordinateZ[1][posEmbedded] = pos[2]
            AlphaX[1][posEmbedded] = GammaBound[0]
            AlphaY[1][posEmbedded] = GammaBound[1]
            AlphaZ[1][posEmbedded] = GammaBound[2]
            posEmbedded += 1
            for j in range(1, Nshed):
                CoordinateX[1][Np[0]] = pos[0] + (j + 0.)*vecj[0]
                CoordinateY[1][Np[0]] = pos[1] + (j + 0.)*vecj[1]
                CoordinateZ[1][Np[0]] = pos[2] + (j + 0.)*vecj[2]
                AlphaX[1][Np[0]] = GammaBound[0]
                AlphaY[1][Np[0]] = GammaBound[1]
                AlphaZ[1][Np[0]] = GammaBound[2]

                Np[0] += 1

        Ns += NsCurrent - 1
        
        #for i in range(NsCurrent - 1):
        #    Nshed = NumberParticlesShedPerStation[Ns + i] + 1
        #    vecj2D = -dt*0.5*np.array([V2DXj[i + 1] + V2DXj[i], V2DYj[i + 1] + V2DYj[i], V2DZj[i + 1] + V2DZj[i]])
        #    dy = np.linalg.norm(vecj2D)
        #    vecj2D /= dy

        #    veci = np.array([(LLXj[i + 1] - LLXj[i]), (LLYj[i + 1] - LLYj[i]), (LLZj[i + 1] - LLZj[i])])
        #    dx = np.linalg.norm(veci)
        #    veci /= dx

        #    pos = 0.5*np.array([LLXj[i + 1] + LLXj[i], LLYj[i + 1] + LLYj[i], LLZj[i + 1] + LLZj[i]])
        #    vecj = np.array([CoordinateX[1][MaskShedParticles[Ns + i] - NumberSource], CoordinateY[1][MaskShedParticles[Ns + i] - NumberSource], CoordinateZ[1][MaskShedParticles[Ns + i] - NumberSource]]) - pos
        #    dy = np.linalg.norm(vecj)
        #    vecj /= dy
        #    
        #    GammaTrailing = (Gamma[i + 1] - Gamma[i])*dy
        #    GammaShedding = (GammaM1[i + 1] + GammaM1[i] - (Gamma[i + 1] + Gamma[i]))/2.*dx
        #    MeanWeightX = (GammaTrailing*vecj2D[0] + GammaShedding*veci[0] + AlphaX[1][MaskShedParticles[Ns + i] - NumberSource])/(Nshed + 1)
        #    slopeX = 2.*(AlphaX[1][MaskShedParticles[Ns + i] - NumberSource] - MeanWeightX)/dy
        #    heightX = 2.*MeanWeightX - AlphaX[1][MaskShedParticles[Ns + i] - NumberSource]
        #    MeanWeightY = (GammaTrailing*vecj2D[1] + GammaShedding*veci[1] + AlphaY[1][MaskShedParticles[Ns + i] - NumberSource])/(Nshed + 1)
        #    slopeY = 2.*(AlphaY[1][MaskShedParticles[Ns + i] - NumberSource] - MeanWeightY)/dy
        #    heightY = 2.*MeanWeightY - AlphaY[1][MaskShedParticles[Ns + i] - NumberSource]
        #    MeanWeightZ = (GammaTrailing*vecj2D[2] + GammaShedding*veci[2] + AlphaZ[1][MaskShedParticles[Ns + i] - NumberSource])/(Nshed + 1)
        #    slopeZ = 2.*(AlphaZ[1][MaskShedParticles[Ns + i] - NumberSource] - MeanWeightZ)/dy
        #    heightZ = 2.*MeanWeightZ - AlphaZ[1][MaskShedParticles[Ns + i] - NumberSource]
        #    CoordinateX[1][posEmbedded] = pos[0]
        #    CoordinateY[1][posEmbedded] = pos[1]
        #    CoordinateZ[1][posEmbedded] = pos[2]
        #    AlphaX[1][posEmbedded] = 0.*slopeX + heightX
        #    AlphaY[1][posEmbedded] = 0.*slopeY + heightY
        #    AlphaZ[1][posEmbedded] = 0.*slopeZ + heightZ
        #    posEmbedded += 1
        #    for j in range(1, Nshed):
        #        CoordinateX[1][Np[0]] = pos[0] + j*dy/Nshed*vecj[0]
        #        CoordinateY[1][Np[0]] = pos[1] + j*dy/Nshed*vecj[1]
        #        CoordinateZ[1][Np[0]] = pos[2] + j*dy/Nshed*vecj[2]
        #        AlphaX[1][Np[0]] = j*dy/Nshed*slopeX + heightX
        #        AlphaY[1][Np[0]] = j*dy/Nshed*slopeY + heightY
        #        AlphaZ[1][Np[0]] = j*dy/Nshed*slopeZ + heightZ
        #        Np[0] += 1
        #Ns += NsCurrent - 1
 '''
