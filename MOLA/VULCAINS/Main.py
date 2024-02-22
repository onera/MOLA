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
import scipy
import numpy as np
import VULCAINS.vulcains as V
import Converter.PyTree as C
import Geom.PyTree as D
import Converter.Internal as I
import Generator.PyTree as G
import Transform.PyTree as T
import Connector.PyTree as CX
import Post.PyTree as P
import CPlot.PyTree as CPlot

from . import FreeWakeParticles as VPM
from . import LiftingLineCoupling as VPMLL
from . import EulerianCoupling as H

from .. import LiftingLine as LL
from .. import Wireframe as W
from .. import InternalShortcuts as J
from .. import ExtractSurfacesProcessor as ESP

from VULCAINS.__init__ import __version__, __author__

Scheme_str2int = VPM.Scheme_str2int
EddyViscosityModel_str2int = VPM.EddyViscosityModel_str2int
RedistributionKernel_str2int = VPM.RedistributionKernel_str2int
DiffusionScheme_str2int = VPM.DiffusionScheme_str2int
Vector_VPM_FlowSolution = VPM.Vector_VPM_FlowSolution
Scalar_VPM_FlowSolution = VPM.Scalar_VPM_FlowSolution
VPM_FlowSolution = VPM.VPM_FlowSolution

####################################################################################################
####################################################################################################
############################################## Solver ##############################################
####################################################################################################
####################################################################################################
def compute(VPMParameters = {}, HybridParameters = {}, LiftingLineParameters = {},
    PerturbationFieldParameters = {}, Polars  = [], EulerianMesh = None,
    PerturbationField = [], LiftingLineTree = [], NumberOfIterations = 1000,
    RestartPath = None, DIRECTORY_OUTPUT = 'OUTPUT', SaveFields = ['all'],
    VisualisationOptions = {'addLiftingLineSurfaces':True}, StdDeviationSample = 53,
    SaveVPMPeriod = 100, Verbose = True, SaveImageOptions={}, Surface = 0.,
    FieldsExtractionGrid = [], SaveFieldsPeriod = np.inf, SaveImagePeriod = np.inf,
    NoRedistributionZones = []):
    '''
    Launches the VPM solver.

    Parameters
    ----------
        VPMParameters : :py:class:`dict` of :py:class:`float`, :py:class:`int`, :py:class:`bool` and :py:class:`str`
            Containes user given parameters for the VPM solver (if any).

        HybridParameters : :py:class:`dict` of :py:class:`float` and :py:class:`int`
            Containes user given parameters for the Hybrid solver (if any).

        LiftingLineParameters : :py:class:`dict` of :py:class:`float`, :py:class:`int` and
            :py:class:`str`
            Containes user given parameters for the Lifting Line(s) (if any). 

        PerturbationFieldParameters : :py:class:`dict` of :py:class:`float` and :py:class:`int`
            Containes user given parameters for the Perturbation Velocity Field (if any).

        Polars : :py:func:`list` of :py:func:`zone` or :py:class:`str`
            Enhanced **Polars** for each airfoil, containing also foilwise
            distributions fields (``Cp``, ``theta``, ``delta1``...).

            .. note::
              if input type is a :py:class:`str`, then **Polars** is
              interpreted as a CGNS file name containing the airfoil polars data

        EulerianMesh : Tree, Base, Zone or :py:class:`str`
            Location of the Eulerian mesh (if any).

        PerturbationField : Tree, Base, Zone or :py:class:`str`
            Location of the Perturbation mesh (if any).

        LiftingLineTree : Tree or :py:class:`str`
            Containes the Lifting Lines.

        NumberOfIterations : :py:class:`int`
            Number of time iteration to do.

        RestartPath : :py:class:`str`
            Location of the VPM tree from where the simulation must start from (if any).

        DIRECTORY_OUTPUT : :py:class:`str`
            Location where the simulation CGNS are written.

        SaveFields : :py:class:`list` or numpy.ndarray of :py:class:`str`
            Fields to save. if 'all', then they are all saved.

        VisualisationOptions : :py:class:`dict`
            CPlot options for the visualisation (if any).

            addLiftingLineSurfaces : :py:class:`bool`
                States whether the Lifting Line(s) surfaces are to be visualised. Requires to
                give a Polars.

            Polars : :py:func:`list` of :py:func:`zone` or :py:class:`str`
                Polars for addLiftingLineSurfaces.

        StdDeviationSample : :py:class:`int`
            Number of samples for the standard deviation.

        SaveVPMPeriod : :py:class:`int`
            Saving frequency of the VPM simulation.

        Verbose : :py:class:`bool`
            States whether the VPM solver prompts the VPM information during the simulation.

        SaveImageOptions : :py:class:`dict`
            CPlot visualisation options (if any).

        Surface : :py:class:`float`
            Surface of wing Lifting Line(s) for the computation of aerodynamic coefficients (if 
            any).

        FieldsExtractionGrid : Tree, Base, Zone
            Probes of the VPM field.

        SaveFieldsPeriod : :py:class:`int`
            Frequency at which the FieldsExtractionGrid is extracted and saved.

        SaveImagePeriod : :py:class:`int`
            Frequency at which an image of the VPM simulation is saved.

    '''
    try: os.makedirs(DIRECTORY_OUTPUT)
    except: pass

    if isinstance(Polars, str): Polars = load(Polars)
    if Polars:
        PolarsInterpolators = LL.buildPolarsInterpolatorDict(Polars,
                                                                  InterpFields = ['Cl', 'Cd', 'Cm'])
    if RestartPath:

        if isinstance(RestartPath, str): t = load(RestartPath)

        if PerturbationField:
            PerturbationField = VPM.pickPerturbationFieldZone(t)
            NumberOfNodes = VPM.getParameter(VPM.pickParticlesZone(t), 'NumberOfNodes')
            PerturbationFieldCapsule = V.build_perturbation_velocity_capsule(\
                                          C.newPyTree(I.getZones(PerturbationField)), NumberOfNodes)
        else: PerturbationFieldCapsule = None
        if isinstance(EulerianMesh, str):
                tE = load(EulerianMesh)
                #tE = H.generateMirrorWing(tE, VPM.getVPMParameters(t), H.getHybridParameters(t))
        else: tE = []
    else:
        if isinstance(LiftingLineTree, str): LiftingLineTree = C.newPyTree(I.getZones(load(LiftingLineTree)))
        elif LiftingLineTree: LiftingLineTree = C.newPyTree(I.getZones(LiftingLineTree))
        else: LiftingLineTree = None
        if isinstance(PerturbationField, str): PerturbationField = load(PerturbationField)
        elif PerturbationField: PerturbationField = C.newPyTree(I.getZones(PerturbationField))
        else: PerturbationField = None
        #if EulerianMesh: EulerianMesh = load(EulerianMesh)
        #else: EulerianMesh = []
        t, tE, PerturbationFieldCapsule = VPM.initialiseVPM(EulerianMesh = EulerianMesh,
            PerturbationField = PerturbationField, HybridParameters = HybridParameters,
            LiftingLineTree = LiftingLineTree, LiftingLineParameters = LiftingLineParameters,
            PerturbationFieldParameters = PerturbationFieldParameters, PolarsInterpolators = \
                                                 PolarsInterpolators, VPMParameters = VPMParameters)
    SaveFields = checkSaveFields(SaveFields)
    IterationInfo = {'Rel. err. of Velocity': 0, 'Rel. err. of Velocity Gradient': 0,
       'Rel. err. of Vorticity': 0, 'Rel. err. of PSE': 0, 'Rel. err. of Diffusion Velocity': 0}
    TotalTime = J.tic()
    sp = VPM.getVPMParameters(t)
    Np = VPM.pickParticlesZone(t)[1][0]
    LiftingLines = LL.getLiftingLines(t)
    it = sp['CurrentIteration']
    simuTime = sp['Time']
    PSE = DiffusionScheme_str2int[sp['DiffusionScheme']] < 2
    DVM = DiffusionScheme_str2int[sp['DiffusionScheme']] == 2
    Freestream = (np.linalg.norm(sp['VelocityFreestream'], axis = 0) != 0.)
    try:
        Wing = (I.getValue(I.getNodeFromName(LiftingLines, 'RPM')) == 0)
    except:
        Wing = True
    if Polars: VisualisationOptions['AirfoilPolars'] = Polars
    else: VisualisationOptions['addLiftingLineSurfaces'] = False
    filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_It%d.cgns'%it[0])
    save(t, filename, VisualisationOptions, SaveFields)
    J.createSymbolicLink(filename,  DIRECTORY_OUTPUT + '.cgns')
    for _ in range(3): print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))
    print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(' Begin VPM Computation '))
    for _ in range(3): print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))
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
        VPM.updateSmagorinskyConstantAndComputeTurbulentViscosity(t)
        VPM.computeNextTimeStep(t, PerturbationFieldCapsule = PerturbationFieldCapsule)
        IterationInfo['Iteration'] = it[0]
        IterationInfo['Percentage'] = it[0]/NumberOfIterations*100.
        IterationInfo['Physical time'] = simuTime[0]
        IterationInfo = H.updateHybridDomainAndSources(t, tE, IterationInfo)
        IterationInfo = VPM.populationControl(t, IterationInfo,
                                                  NoRedistributionZones = NoRedistributionZones)
        IterationInfo = VPMLL.ShedVorticitySourcesFromLiftingLines(t, PolarsInterpolators, IterationInfo,
                                            PerturbationFieldCapsule = PerturbationFieldCapsule)
        IterationInfo['Number of particles'] = Np[0]
        IterationInfo = VPM.induceVPMField(t, IterationInfo = IterationInfo,
                                            PerturbationFieldCapsule = PerturbationFieldCapsule)
        IterationInfo['Total iteration time'] = J.tic() - IterationTime
        IterationInfo = VPMLL.getAerodynamicCoefficientsOnLiftingLine(LiftingLines, Wings = Wing,
                               StdDeviationSample = StdDeviationSample, Freestream = Freestream, 
                                               IterationInfo = IterationInfo, Surface = Surface)
        IterationInfo['Total simulation time'] = J.tic() - TotalTime
        if Verbose: printIterationInfo(IterationInfo, PSE = PSE, DVM = DVM, Wings = Wing)

        if (SAVE_FIELDS or SAVE_ALL) and FieldsExtractionGrid:
            extract(t, FieldsExtractionGrid, 5300)
            filename = os.path.join(DIRECTORY_OUTPUT, 'fields_It%d.cgns'%it)
            save(FieldsExtractionGrid, filename, SaveFields)
            J.createSymbolicLink(filename, 'fields.cgns')

        if SAVE_IMAGE or SAVE_ALL:
            setVisualization(t, **VisualisationOptions)
            saveImage(t, **SaveImageOptions)

        if SAVE_VPM or SAVE_ALL:
            filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_It%d.cgns'%it)
            save(t, filename, VisualisationOptions, SaveFields)
            J.createSymbolicLink(filename,  DIRECTORY_OUTPUT + '.cgns')

        if CONVERGED: break
        
    if FieldsExtractionGrid:
        extract(t, FieldsExtractionGrid, 5300)
        filename = os.path.join(DIRECTORY_OUTPUT, 'fields_It%d.cgns'%it)
        save(FieldsExtractionGrid, filename, SaveFields)
        save(FieldsExtractionGrid, 'fields.cgns', SaveFields)

    filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_It%d.cgns'%it)
    save(t, filename, VisualisationOptions, SaveFields)
    save(t, DIRECTORY_OUTPUT + '.cgns', VisualisationOptions, SaveFields)
    for _ in range(3): print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))
    print(f"{'||':>57}\r" + '||' + '{:-^53}'.format(' End of VPM computation '))
    for _ in range(3): print(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))

    return t

def computeVortexRing(VPMParameters = {}, VortexParameters = {}, NumberOfIterations = 10000,
    SaveVPMPeriod = 10, DIRECTORY_OUTPUT = 'OUTPUT', LeapFrog = False):
    int_Params = ['StrengthRampAtbeginning', 'EnstrophyControlRamp',
        'CurrentIteration', 'ClusterSize', 
        'MaximumAgeAllowed', 'RedistributionPeriod', 'NumberOfThreads', 'IntegrationOrder',
        'IterationTuningFMM', 'IterationCounter', 
        'FarFieldApproximationOrder', 'NumberLayers']

    float_Params = ['Density', 'EddyViscosityConstant', 'Temperature', 'ResizeParticleFactor',
        'Time', 'CutoffXmin', 'CutoffZmin', 'MaximumMergingVorticityFactor',
        'MagnitudeRelaxationFactor', 'AntiDiffusion', 'SmoothingRatio', 'RPM',
        'CutoffXmax', 'CutoffYmin', 'CutoffYmax', 'Sigma0','KinematicViscosity',
        'CutoffZmax', 'ForcedDissipation','MaximumAngleForMerging', 'MinimumVorticityFactor', 
        'MinimumOverlapForMerging', 'VelocityFreestream', 'AntiStretching',
        'RedistributeParticlesBeyond', 'RedistributeParticleSizeFactor', 'MachLimitor',
        'TimeStep', 'Resolution', 'NearFieldOverlappingRatio', 'TimeFMM',
        'RemoveWeakParticlesBeyond', 'Intensity', 'CoreRadius', 'RingRadius', 'Length', 'Tau',
        'MinimumVorticityFraction', 'EddyViscosityRelaxationFactor', 'StrengthVariationLimitor',
        'RealignmentRelaxationFactor', 'ParticleSizeVariationLimitor']

    bool_Params = ['LowStorageIntegration']
    
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
            'AntiDiffusion'                 : 0.,             #between 0 and 1, the closer to 0, the more the viscosity affects the particle strength, the closer to 1, the more it affects the particle size
            'SmoothingRatio'                : 2.,             #in m, anywhere between 1.5 and 2.5, the higher the NumberSource, the smaller the Resolution and the higher the SmoothingRatio should be to avoid blowups, the HOA kernel requires a higher smoothing
            'VorticityEquationScheme'       : 'Transpose',    #Classical, Transpose or Mixed, The schemes used to compute the vorticity equation are the classical scheme, the transpose scheme (conserves total vorticity) and the mixed scheme (a fusion of the previous two)
            'Sigma0'                        : 0.1,
            'MachLimitor'                   : 0.,             #[0, +inf[, sets the maximum/minimun induced velocity a particle can have
            'StrengthVariationLimitor'      : 0.,             #[1, +inf[, gives the maximum variation the strength of a particle can have during an iteration
            'ParticleSizeVariationLimitor'  : 0.,             #[1, +inf[, gives the maximum a particle can grow/shrink during an iteration
        ############################################################################################
        ################################### Numerical Parameters ###################################
        ############################################################################################
            'CurrentIteration'              : 0,              #follows the current iteration
            'IntegrationOrder'              : 3,              #[|1, 4|]1st, 2nd, 3rd or 4th order Runge Kutta. In the hybrid case, there must be at least as much Interfaces in the hybrid domain as the IntegrationOrder of the time integration scheme
            'LowStorageIntegration'         : True,           #[|0, 1|], states if the classical or the low-storage Runge Kutta is used
            'NumberOfLiftingLines'          : 0,              #[0, +inf[, number of LiftingLines
            'NumberOfLiftingLineSources'    : 0,              #[0, +inf[, total number of embedded source particles on the LiftingLines
            'NumberOfBEMSources'            : 0,              #[0, +inf[, total number of embedded Boundary Element Method particles on the solid boundaries
            'NumberOfCFDSources'            : 0,              #[0, +inf[, total number of embedded cfd particles on the Hybrid Inner Interface
            'NumberOfHybridSources'         : 0,              #[0, +inf[, total number of hybrid particles generated in the hybrid Domain
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
            'EnstrophyControlRamp'          : 100,            #[|0, +inf[|, number of iteration before the enstrophy relaxation is fully applied. If propeller -> nbr of iteration to make 1 rotation (60/(dt*rpm)). If wing -> nbr of iteration for the freestream to travel one wingspan (L/(Uinf*dt)).
        ############################################################################################
        ###################################### FMM parameters ######################################
        ############################################################################################
            'FarFieldApproximationOrder'    : 8,              #[|6, 12|], order of the polynomial which approximates the far field interactions, the higher the more accurate and the more costly
            'IterationTuningFMM'            : 53,             #frequency at which the FMM is compared to the direct computation, gives the relative L2 error
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
        defaultParameters['NumberOfThreads'] = int(os.getenv('OMP_NUM_THREADS', \
                                                                  len(os.sched_getaffinity(0))))
    else:
        defaultParameters['NumberOfThreads'] = min(defaultParameters['NumberOfThreads'], \
                                int(os.getenv('OMP_NUM_THREADS', len(os.sched_getaffinity(0)))))
    VPM.checkParametersTypes([defaultParameters, defaultVortexParameters], int_Params,
                                                                      float_Params, bool_Params)
    architecture = V.mpi_init(defaultParameters['NumberOfThreads'][0])

    defaultParameters['VelocityFreestream'] = np.array([0.]*3, dtype = float)
    defaultParameters['Sigma0'] = np.array(defaultParameters['Resolution']*\
                           defaultParameters['SmoothingRatio'], dtype = np.float64, order = 'F')
    defaultParameters['IterationCounter'] = np.array([0], dtype = np.int32, order = 'F')
    defaultParameters['StrengthRampAtbeginning'][0] = max(\
                                                defaultParameters['StrengthRampAtbeginning'], 1)
    defaultParameters['MinimumVorticityFactor'][0] = max(0., \
                                                    defaultParameters['MinimumVorticityFactor'])
    t = VPM.buildEmptyVPMTree()
    if 'Length' in defaultVortexParameters:
        VPM.createLambOseenVortexBlob(t, defaultParameters, defaultVortexParameters)
    else:
        VPM.createLambOseenVortexRing(t, defaultParameters, defaultVortexParameters)
        if LeapFrog:
            Particles = VPM.pickParticlesZone(t)
            Np = Particles[1][0][0]
            VPM.extend(Particles, Np)
            ax, ay, az, s, c, nu = J.getVars(Particles, VPM.vectorise('Alpha') + \
                                                ['Sigma', 'Cvisq', 'Nu', ])
            ax[Np:], ay[Np:], az[Np:], s[Np:], c[Np:], nu[Np:] = ax[:Np], \
                                               ay[:Np], az[:Np], s[:Np], c[:Np], nu[:Np]
            x, y, z = J.getxyz(Particles)
            x[Np:], y[Np:], z[Np:] = x[:Np], y[:Np], z[:Np] + \
                                                           defaultVortexParameters['RingRadius']

    
    Particles = VPM.pickParticlesZone(t)
    J.set(Particles, '.VPM#Parameters', **defaultParameters)
    J.set(Particles, '.VortexRing#Parameters', **defaultVortexParameters)
    I._sortByName(I.getNodeFromName1(Particles, '.VPM#Parameters'))
    I._sortByName(I.getNodeFromName1(Particles, '.VortexRing#Parameters'))

    VPM.induceVPMField(t)

    IterationCounter = I.getNodeFromName(t, 'IterationCounter')
    IterationCounter[1][0] = defaultParameters['IterationTuningFMM']*\
                                                           defaultParameters['IntegrationOrder']

    C.convertPyTree2File(t, DIRECTORY_OUTPUT + '.cgns')
    deletePrintedLines()
    compute(VPMParameters = {}, HybridParameters = {}, LiftingLineParameters = {},
        AirfoilPolars = [], EulerianMesh = None, LiftingLineTree = [],
        NumberOfIterations = NumberOfIterations, RestartPath = DIRECTORY_OUTPUT + '.cgns',
        DIRECTORY_OUTPUT = DIRECTORY_OUTPUT,
        VisualisationOptions = {'addLiftingLineSurfaces':False}, SaveVPMPeriod = SaveVPMPeriod,
        Verbose = True, SaveImageOptions={}, Surface = 0., FieldsExtractionGrid = [],
        SaveFieldsPeriod = np.inf, SaveImagePeriod = np.inf)

def extract(t = [], ExctractionTree = [], NbOfParticlesUsedForPrecisionEvaluation = 1000,
    FarFieldApproximationOrder = 12, NearFieldOverlappingRatio = 0.4):
    '''
    Extract the VPM field onto given nodes.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        ExctractionTree : Tree, Base, Zone
            Probes of the VPM field.

        NbOfParticlesUsedForPrecisionEvaluation : :py:class:`int`
            Number of nodes where the solution computed by the FMM is checked.

        FarFieldApproximationOrder : :py:class:`int`
            Order of the polynomial used to approximate far field interaction by the FMM (the
            higher the more accurate).

        NearFieldOverlappingRatio : :py:class:`float`
            Ratio at which interactions between close particle clusters are directly computed
            rather than approximated by the FMM (the lower the more accurate).
    '''
    if not ExctractionTree: return
    newFieldNames = ['Velocity' + v for v in 'XYZ'] + ['Vorticity' + v for v in 'XYZ'] + \
                    ['gradxVelocity' + v for v in 'XYZ']+ ['gradyVelocity' + v for v in 'XYZ']+\
                    ['gradzVelocity' + v for v in 'XYZ'] + ['PSE' + v for v in 'XYZ'] + \
                    ['VelocityMagnitude', 'VorticityMagnitude', 'divVelocity', 'QCriterion',
                                                                                  'Nu', 'Sigma']
    for fn in newFieldNames: C._initVars(ExctractionTree, fn, 0.)

    ExtractionZones = I.getZones(ExctractionTree)
    NPtsPerZone = [0] + [C.getNPts(z) for z in ExtractionZones]
    tmpZone = D.line((0, 0, 0), (1, 0, 0), np.sum(NPtsPerZone))
    for fn in newFieldNames: C._initVars(tmpZone, fn, 0.)

    coordst = J.getxyz(tmpZone)
    for i, zone in enumerate(ExtractionZones):
        coords = J.getxyz(zone)
        for ct, c in zip(coordst, coords):
            ct[NPtsPerZone[i]:NPtsPerZone[i+1]] = c.ravel(order = 'F')

    tmpTree = C.newPyTree(['Base', tmpZone])
    EddyViscosityModel = EddyViscosityModel_str2int[VPM.getParameter(t, 'EddyViscosityModel')]

    inside = H.flagNodesInsideSurface(Surface = H.pickHybridDomainInnerInterface(t),
                             X = np.ravel(I.getNodeFromName(ExctractionTree, 'CoordinateX')[1]),
                             Y = np.ravel(I.getNodeFromName(ExctractionTree, 'CoordinateY')[1]),
                             Z = np.ravel(I.getNodeFromName(ExctractionTree, 'CoordinateZ')[1]))

    V.wrap_extract_plane(t, tmpTree, int(NbOfParticlesUsedForPrecisionEvaluation),
                                       EddyViscosityModel, np.int32(FarFieldApproximationOrder),
                                                  np.float64(NearFieldOverlappingRatio), inside)
    #TODO: add the VPM.extractperturbationField here
    tmpFields = J.getVars(I.getZones(tmpTree)[0], newFieldNames)

    for i, zone in enumerate(ExtractionZones):
        fields = J.getVars(zone, newFieldNames)
        for ft, f in zip(tmpFields, fields):
            fr = f.ravel(order = 'F')
            fr[:] = ft[NPtsPerZone[i]:NPtsPerZone[i+1]]

    return ExctractionTree

####################################################################################################
####################################################################################################
######################################### IO/Visualisation #########################################
####################################################################################################
####################################################################################################
def setVisualization(t = [], ParticlesColorField = 'VorticityMagnitude',
    ParticlesRadius = '{Sigma}/5', addLiftingLineSurfaces = True, AirfoilPolars = []):
    '''
    Set the visualisation options for CPlot when the CGNS are saved.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        ParticlesColorField : :py:class:`list`
            VPM field to color.

        ParticlesRadius : :py:class:`float`
            Radius of the visualised particles.

        addLiftingLineSurfaces : :py:class:`bool`
            States whether Lifting Line(s) surfaces are added for visualisation. Requires a
            valid AirfoilPolars.

        Polars : :py:func:`list` of :py:func:`zone` or :py:class:`str`
            Enhanced **Polars** for each airfoil, containing also foilwise
            distributions fields (``Cp``, ``theta``, ``delta1``...).

            .. note::
              if input type is a :py:class:`str`, then **Polars** is
              interpreted as a CGNS file name containing the airfoil polars data
    '''
    Particles = VPM.pickParticlesZone(t)
    Sigma = I.getValue(I.getNodeFromName(Particles, 'Sigma'))
    C._initVars(Particles, 'radius=' + ParticlesRadius)
    FlowSolution = I.getNodeFromName1(Particles, 'FlowSolution')
    if not ParticlesColorField: ParticlesColorField = 'VorticityMagnitude'

    flag = True
    for Field in FlowSolution[2]:
        if ParticlesColorField == Field[0]: flag = False

    if flag:
        if ParticlesColorField == 'VorticityMagnitude': C._initVars(Particles,
                    'VorticityMagnitude=({VorticityX}**2 + {VorticityY}**2 + {VorticityZ}**2)**0.5')
        elif ParticlesColorField == 'StrengthMagnitude': C._initVars(Particles,
                                 'StrengthMagnitude=({AlphaX}**2 + {AlphaY}**2 + {AlphaZ}**2)**0.5')
        elif ParticlesColorField == 'VelocityMagnitude':
            U0 = VPM.getParameter(Particles, 'VelocityFreestream')
            C._initVars(Particles, 'U0X', U0[0])
            C._initVars(Particles, 'U0Y', U0[1])
            C._initVars(Particles, 'U0Z', U0[2])
            C._initVars(Particles, 'VelocityMagnitude=(\
          ({UX0}+{VelocityInducedX}+{VelocityPerturbationX}+{VelocityBEMX}+{VelocityInterfaceX})**2\
         +({UY0}+{VelocityInducedY}+{VelocityPerturbationY}+{VelocityBEMY}+{VelocityInterfaceY})**2\
         +({UZ0}+{VelocityInducedZ}+{VelocityPerturbationZ}+{VelocityBEMZ}+{VelocityInterfaceZ})**2\
                                                                                            )**0.5')
        elif ParticlesColorField == 'RotU':
            C._initVars(Particles, 'RotU=(({gradyVelocityZ} - {gradzVelocityY})**2 + \
                ({gradzVelocityX} - {gradxVelocityZ})**2 + ({gradxVelocityY} - {gradyVelocityX})**2\
                                                                                            )**0.5')
    CPlot._addRender2Zone(Particles, material = 'Sphere',
             color = 'Iso:' + ParticlesColorField, blending = 0.6, shaderParameters = [0.04, 0])
    LiftingLines = LL.getLiftingLines(t)
    for zone in LiftingLines:
        CPlot._addRender2Zone(zone, material = 'Flat', color = 'White', blending = 0.2)

    if addLiftingLineSurfaces:
        if not AirfoilPolars:
            ERRMSG = J.FAIL + ('production of surfaces from lifting-line requires'
                ' attribute AirfoilPolars') + J.ENDC
            raise AttributeError(ERRMSG)
        LiftingLineSurfaces = []
        for ll in LiftingLines:
            surface = LL.postLiftingLine2Surface(ll, AirfoilPolars)
            surface[0] = ll[0] + '.surf'
            CPlot._addRender2Zone(surface, material = 'Solid', color = 'Grey',
                                                      meshOverlay = 1, shaderParameters=[1.,1.])
            LiftingLineSurfaces += [surface]
        I.createUniqueChild(t, 'LiftingLineSurfaces', 'CGNSBase_t',
            value = np.array([2, 3], order = 'F'), children = LiftingLineSurfaces)

    for zone in I.getZones(H.pickHybridDomain(t)):
        CPlot._addRender2Zone(zone, material = 'Glass', color = 'White', blending = 0.6,
                                                      meshOverlay = 1, shaderParameters=[1.,1.])
    CPlot._addRender2PyTree(t, mode = 'Render', colormap = 'Blue2Red', isoLegend=1,
                                                              scalarField = ParticlesColorField)

def saveImage(t = [], ShowInScreen = False, ImagesDirectory = 'FRAMES', **DisplayOptions):
    '''
    Saves an image from the t tree.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone
            Containes a zone of particles named 'Particles'.

        ShowInScreen : :py:class:`bool`
            CPlot option.

        ImagesDirectory : :py:class:`str`
            Location where the image is written.

        DisplayOptions : :py:class:`dict`
                mode : :py:class:`dict`
                    CPlot Display mode (Render, Scalar, Vector, ...).

                displayInfo : :py:class:`int`
                
                colormap : :py:class:`str`
                    Color used for the visulation of the particles.
                
                win : :py:class:`tuple`
                    Image resolution.

                ... other CPlot options
    '''
    if 'mode' not in DisplayOptions: DisplayOptions['mode'] = 'Render'
    if 'displayInfo' not in DisplayOptions: DisplayOptions['displayInfo'] = 0
    if 'colormap' not in DisplayOptions: DisplayOptions['colormap'] = 0
    if 'win' not in DisplayOptions: DisplayOptions['win'] = (700, 700)
    DisplayOptions['exportResolution'] = '%gx%g'%DisplayOptions['win']

    try: os.makedirs(ImagesDirectory)
    except: pass

    sp = VPM.getVPMParameters(t)

    DisplayOptions['export'] = os.path.join(ImagesDirectory,
        'frame%05d.png'%sp['CurrentIteration'])

    if ShowInScreen:
        DisplayOptions['offscreen'] = 0
    else:
        DisplayOptions['offscreen'] = 1

    CPlot.display(t, **DisplayOptions)
    if DisplayOptions['offscreen']:
        CPlot.finalizeExport(DisplayOptions['offscreen'])

def load(filename = ''):
    '''
    Opens the CGNS file designated by the user. If the CGNS containes particles, the VPM field
    is updated.

    Parameters
    ----------
        filename : :py:class:`str`
            Location of the CGNS to open.

    Returns
    ----------
        t : Tree, Base, Zone or list of Zone
    '''
    t = C.convertFile2PyTree(filename)
    deletePrintedLines()
    Particles = VPM.pickParticlesZone(t)
    if Particles:
        rmNodes = ['RotU', 'VelocityMagnitude', 'AlphaN'] + ['RotU' + v for v in 'XYZ'] + \
                                                    ['Velocity' + v for v in 'XYZ'] + ['radius']
        for Nodes in rmNodes: I._rmNodesByName(Particles, Nodes)

        RequiredFlowSolution = VPM_FlowSolution.copy()
        FlowSolution = I.getNodeFromName1(Particles, 'FlowSolution')
        for Field in FlowSolution[2]:
            if Field[0] in RequiredFlowSolution: RequiredFlowSolution.remove(Field[0])
        
        J.invokeFieldsDict(Particles, RequiredFlowSolution)
        I._sortByName(Particles)
        IterationCounter = VPM.getParameter(t, 'IterationCounter')
        IterationCounter[0] -= 1
        NumberOfNodes = VPM.getParameter(VPM.pickParticlesZone(t), 'NumberOfNodes')
        if NumberOfNodes:
            PerturbationFieldCapsule = V.build_perturbation_velocity_capsule(\
                           C.newPyTree(I.getZones(VPM.pickPerturbationFieldZone(t))), NumberOfNodes)
        else:
            PerturbationFieldCapsule = None
        Old_esM1 = I.getValue(I.getNodeFromName(t, 'EnstrophyM1'))
        VPM.induceVPMField(t, PerturbationFieldCapsule = PerturbationFieldCapsule)
        esM1 = I.getNodeFromName(t, 'EnstrophyM1')
        esM1[1] = Old_esM1
        HybridParameters = I.getNodeFromName(t, '.Hybrid#Parameters')
        if HybridParameters:
            Nbem = VPM.getParameter(Particles, 'NumberOfBEMSources')[0]
            HybridParameters[2] += [['BEMMatrix', np.array([0.]*9*Nbem*Nbem, dtype = np.float64,
                                                               order = 'F'), [], 'DataArray_t']]
            H.updateBEMMatrix(t)

    return t

def checkSaveFields(SaveFields = ['all']):
    '''
    Updates the VPM fields to conserve when the particle zone is saved.

    Parameters
    ----------
        SaveFields : :py:class:`list` or numpy.ndarray of :py:class:`str`
            Fields to save. if 'all', then they are all saved.

    Returns
    ----------
        FieldNames : :py:class:`list` or numpy.ndarray of :py:class:`str`
            Fields to save. if 'all', then they are all saved.
    ''' 
    VectorFieldNames = Vector_VPM_FlowSolution + VPM.vectorise(['RotU']) + VPM.vectorise(['Velocity'])
    ScalarFieldNames = Scalar_VPM_FlowSolution + ['AlphaN', 'StrengthMagnitude', 'RotU', \
                                                      'VelocityMagnitude', 'VorticityMagnitude']

    if not (isinstance(SaveFields, list) or isinstance(SaveFields, np.ndarray)):
        SaveFields = [SaveFields]

    SaveFields = np.array(SaveFields)
    FieldNames = []
    if 'all' in SaveFields:
        for VectorFieldName in VectorFieldNames:
            FieldNames += [VectorFieldName]

        for ScalarFieldName in ScalarFieldNames:
            FieldNames += [ScalarFieldName]
    else:
        for VectorFieldName in VectorFieldNames:
            if (VectorFieldName[:-1] in SaveFields) or (VectorFieldName in SaveFields):
                FieldNames += [VectorFieldName]

        for ScalarFieldName in ScalarFieldNames:
            if ScalarFieldName in SaveFields:
                FieldNames += [ScalarFieldName]

    FieldNames += VPM.vectorise('Alpha') + ['Age', 'Sigma', 'Nu', 'Cvisq', 'EnstrophyM1']
    return np.unique(FieldNames)

def save(t = [], filename = '', VisualisationOptions = {}, SaveFields = checkSaveFields()):
    '''
    Saves the CGNS file designated by the user. If the CGNS containes particles, the VPM field
    saved are the one given by the user.

    Parameters
    ----------
        t : Tree, Base, Zone or list of Zone

        filename : :py:class:`str`
            Location of the where t is saved.

        VisualisationOptions : :py:class:`dict`
            CPlot visualisation options.

        SaveFields : :py:class:`list` or numpy.ndarray of :py:class:`str`
            Particles fields to save (if any). if 'all', then they are all saved.
    '''
    tref = I.copyRef(t)
    if VisualisationOptions:
        setVisualization(tref, **VisualisationOptions)
        SaveFields = np.append(SaveFields, ['radius'])

    Particles = VPM.pickParticlesZone(tref)
    if Particles:
        I._rmNodesByName(Particles, 'BEMMatrix')
        if ('AlphaN' in SaveFields) and H.pickHybridDomain(t):
            an = J.invokeFields(Particles, ['AlphaN'])[0]
            acfdn = np.append(I.getNodeFromName(Particles, 'AlphaBEMN')[1], \
                                                   I.getNodeFromName(Particles, 'AlphaCFDN')[1])
            an[:len(acfdn)] = acfdn

        if 'VelocityX' in SaveFields:
            ux, uy, uz = J.invokeFields(Particles, ['Velocity' + v for v in 'XYZ'])
            u0 = VPM.getParameter(Particles, 'VelocityFreestream')
            uix, uiy, uiz, upertx, uperty, upertz, udiffx, udiffy, udiffz, ubemx, ubemy, ubemz,\
                                                   usurfx, usurfy, usurfz = J.getVars(Particles,
                                                  ['VelocityInduced'      + v for v in 'XYZ'] +\
                                                  ['VelocityPerturbation' + v for v in 'XYZ'] +\
                                                  ['VelocityDiffusion'    + v for v in 'XYZ'] +\
                                                  ['VelocityBEM'          + v for v in 'XYZ'] +\
                                                  ['VelocityInterface'    + v for v in 'XYZ'])
            ux[:] = u0[0] + uix + upertx + ubemx + usurfx# + udiffx#the diffusion velocity is not a velocity actually present in the flow but still moves the particles, should it be put here ?
            uy[:] = u0[1] + uiy + uperty + ubemy + usurfy# + udiffy
            uz[:] = u0[2] + uiz + upertz + ubemz + usurfz# + udiffz

        if 'VelocityMagnitude' in SaveFields:
            u = J.invokeFields(Particles, ['VelocityMagnitude'])[0]
            u0 = VPM.getParameter(Particles, 'VelocityFreestream')
            uix, uiy, uiz, upertx, uperty, upertz, udiffx, udiffy, udiffz, ubemx, ubemy, ubemz,\
                                                   usurfx, usurfy, usurfz = J.getVars(Particles,
                                                  ['VelocityInduced'      + v for v in 'XYZ'] +\
                                                  ['VelocityPerturbation' + v for v in 'XYZ'] +\
                                                  ['VelocityDiffusion'    + v for v in 'XYZ'] +\
                                                  ['VelocityBEM'          + v for v in 'XYZ'] +\
                                                  ['VelocityInterface'    + v for v in 'XYZ'])
            u[:] = np.linalg.norm(
                          np.vstack([u0[0] + uix + upertx + udiffx + ubemx + usurfx,
                                     u0[1] + uiy + uperty + udiffy + ubemy + usurfy,
                                     u0[2] + uiz + upertz + udiffz + ubemz + usurfz]), axis = 0)

        if 'RotUX' in SaveFields:
            rotux, rotuy, rotuz = J.invokeFields(Particles, ['RotU' + v for v in 'XYZ'])
            duxdx, duydx, duzdx, duxdy, duydy, duzdy, duxdz, duydz, duzdz = J.getVars(Particles,
                 ['gradxVelocity' + v for v in 'XYZ'] + ['gradyVelocity' + v for v in 'XYZ'] + \
                                                           ['gradzVelocity' + v for v in 'XYZ'])
            rotux[:] = duzdy - duydz
            rotuy[:] = duxdz - duzdx
            rotuz[:] = duydx - duxdy

        if 'RotU' in SaveFields:
            rotu = J.invokeFields(Particles, ['RotU'])[0]
            duxdx, duydx, duzdx, duxdy, duydy, duzdy, duxdz, duydz, duzdz = J.getVars(Particles,
                 ['gradxVelocity' + v for v in 'XYZ'] + ['gradyVelocity' + v for v in 'XYZ'] + \
                                                           ['gradzVelocity' + v for v in 'XYZ'])
            rotu[:] = np.linalg.norm(np.vstack([duzdy - duydz, duxdz - duzdx, duydx - duxdy]), \
                                                                                       axis = 0)

        if 'VorticityMagnitude' in SaveFields:
            w = J.invokeFields(Particles, ['VorticityMagnitude'])[0]
            wx, wy, wz = J.getVars(Particles, VPM.vectorise('Vorticity'))
            w[:] = np.linalg.norm(np.vstack([wx, wy, wz]), axis = 0)
            Nll, Nbem, Nsurf = VPM.getParameters(Particles, ['NumberOfLiftingLineSources', \
                                                    'NumberOfBEMSources', 'NumberOfCFDSources'])
            if Nbem + Nsurf:
                duxdx, duydx, duzdx, duxdy, duydy, duzdy, duxdz, duydz, duzdz = \
                    J.getVars(Particles, VPM.vectorise('gradxVelocity') + \
                                        VPM.vectorise('gradyVelocity') + VPM.vectorise('gradzVelocity'))
                n0 = Nll[0]
                n1 = Nll[0] + Nbem[0] + Nsurf[0]
                w[n0: n1] = np.linalg.norm(np.vstack([duzdy[n0: n1] - duydz[n0: n1], \
                       duxdz[n0: n1] - duzdx[n0: n1], duydx[n0: n1] - duxdy[n0: n1]]), axis = 0)

        if 'StrengthMagnitude' in SaveFields:
            C._initVars(Particles, 'StrengthMagnitude=({AlphaX}**2 + {AlphaY}**2 + {AlphaZ}**2)\
                                                                                         **0.5')

        FlowSolution = I.getNodeFromName(Particles, 'FlowSolution')
        rmNodes = []
        for Field in FlowSolution[2]:
            if Field[0] not in SaveFields: rmNodes += [Field[0]]

        for Node in rmNodes: I._rmNodesByName(FlowSolution, Node)

        # C._initVars(Particles, 'Theta={Enstrophy}/({StrengthMagnitude}*{RotU})')
        # Theta = I.getNodeFromName(Particles, 'Theta')
        # Theta[1] = 180./np.pi*np.arccos(Theta[1])
        I._sortByName(Particles)

    try:
        if os.path.islink(filename):
            os.unlink(filename)
        else:
            os.remove(filename)
    except:
        pass

    C.convertPyTree2File(tref, filename)
    deletePrintedLines()

def printIterationInfo(IterationInfo = {}, PSE = False, DVM = False, Wings = False):
    '''
    Prints the current iteration information.

    Parameters
    ----------
        IterationInfo : :py:class:`dict` of :py:class:`str`
            VPM solver information on the current iteration.

        PSE : :py:class:`bool`
            States whether the PSE was used.

        DVM : :py:class:`bool`
            States whether the DVM was used.

        Wings : :py:class:`bool`
            States whether the Lifting Line(s) Wings were used.
    '''
    msg = f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Iteration ' + '{:d}'.format(IterationInfo['Iteration']) + \
                    ' (' + '{:.1f}'.format(IterationInfo['Percentage']) + '%) ') + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Physical time') + \
                    ': ' + '{:.5f}'.format(IterationInfo['Physical time']) + ' s' + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of particles') + \
                    ': ' + '{:d}'.format(IterationInfo['Number of particles']) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Total iteration time') + \
                    ': ' + '{:.2f}'.format(IterationInfo['Total iteration time']) + ' s' + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Total simulation time') + \
                    ': ' + '{:.1f}'.format(IterationInfo['Total simulation time']) + ' s' + '\n'
    msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Loads ') + '\n'
    if (Wings and 'Lift' in IterationInfo) or (not Wings and 'Thrust' in IterationInfo):
        if (Wings):
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Lift') + \
                  ': ' + '{:.4g}'.format(IterationInfo['Lift']) + ' N' + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Lift Standard Deviation') + \
                  ': ' + '{:.2f}'.format(IterationInfo['Lift Standard Deviation']) + ' %' + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Drag') + \
                  ': ' + '{:.4g}'.format(IterationInfo['Drag']) + ' N' + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Drag Standard Deviation') + \
                  ': ' + '{:.2f}'.format(IterationInfo['Drag Standard Deviation']) + ' %' + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('cL') + \
                  ': ' + '{:.4f}'.format(IterationInfo['cL']) + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('cD') + \
                  ': ' + '{:.5f}'.format(IterationInfo['cD']) + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('f') + \
                  ': ' + '{:.4f}'.format(IterationInfo['f']) + '\n'
        else:
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Thrust') + \
                ': ' + '{:.5g}'.format(IterationInfo['Thrust']) + ' N' + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Thrust Standard Deviation') + \
                ': ' + '{:.2f}'.format(IterationInfo['Thrust Standard Deviation']) + ' %' + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Power') + \
                ': ' + '{:.5g}'.format(IterationInfo['Power']) + ' W' + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Power Standard Deviation') + \
                ': ' + '{:.2f}'.format(IterationInfo['Power Standard Deviation']) + ' %' + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('cT') + \
                ': ' + '{:.5f}'.format(IterationInfo['cT']) + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Cp') + \
                ': ' + '{:.5f}'.format(IterationInfo['cP']) + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Eff') + \
                ': ' + '{:.5f}'.format(IterationInfo['Eff']) + '\n'
    msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Population Control ') + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of resized particles') + \
                 ': ' + '{:d}'.format(IterationInfo['Number of resized particles']) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of particles beyond cutoff') + \
                 ': ' + '{:d}'.format(IterationInfo['Number of particles beyond cutoff']) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of depleted particles') + \
                 ': ' + '{:d}'.format(IterationInfo['Number of depleted particles']) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of split particles') + \
                 ': ' + '{:d}'.format(IterationInfo['Number of split particles']) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of merged particles') + \
                 ': ' + '{:d}'.format(IterationInfo['Number of merged particles']) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Control Computation time') + \
                 ': ' + '{:.2f}'.format(IterationInfo['Population Control time']) + ' s (' + \
                                      '{:.1f}'.format(IterationInfo['Population Control time']/\
                                      IterationInfo['Total iteration time']*100.) + '%) ' + '\n'
    if 'Lifting Line time' in IterationInfo:
        msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Lifting Line ') + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Circulation error') + \
                     ': ' + '{:.5e}'.format(IterationInfo['Circulation error']) + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of sub-iterations') + \
                     ': ' + '{:d}'.format(IterationInfo['Number of sub-iterations (LL)']) + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of shed particles') + \
                     ': ' + '{:d}'.format(IterationInfo['Number of shed particles']) + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Lifting Line Computation time') + \
                     ': ' + '{:.2f}'.format(IterationInfo['Lifting Line time']) + ' s (' + \
                                            '{:.1f}'.format(IterationInfo['Lifting Line time']/\
                                      IterationInfo['Total iteration time']*100.) + '%) ' + '\n'

    if 'Hybrid Computation time' in IterationInfo:
        msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Hybrid Solver ') + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Eulerian Vorticity lost') + \
                      ': ' + '{:.1g}'.format(IterationInfo['Eulerian Vorticity lost']) + \
                      ' s-1 (' + '{:.1f}'.format(IterationInfo['Eulerian Vorticity lost per'])+\
                                                                                    '%) ' + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Minimum Eulerian Vorticity') + \
                      ': ' + '{:.2g}'.format(IterationInfo['Minimum Eulerian Vorticity']) + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of Hybrids Generated') + \
                      ': ' + '{:d}'.format(IterationInfo['Number of Hybrids Generated']) + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Hybrid Computation time') + \
                      ': ' + '{:.2f}'.format(IterationInfo['Hybrid Computation time']) + \
                           ' s (' + '{:.1f}'.format(IterationInfo['Hybrid Computation time']/\
                                      IterationInfo['Total iteration time']*100.) + '%) ' + '\n'
        
    msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' FMM ') + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Rel. err. of Velocity') + \
                    ': ' + '{:e}'.format(IterationInfo['Rel. err. of Velocity']) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Rel. err. of Velocity Gradient') + \
                    ': ' + '{:e}'.format(IterationInfo['Rel. err. of Velocity Gradient']) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Rel. err. of Vorticity') + \
                    ': ' + '{:e}'.format(IterationInfo['Rel. err. of Vorticity']) + '\n'
    if PSE: msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Rel. err. of PSE') + \
                    ': ' + '{:e}'.format(IterationInfo['Rel. err. of PSE']) + '\n'
    if DVM:
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Rel. err. of PSE') + \
                    ': ' + '{:e}'.format(IterationInfo['Rel. err. of PSE']) + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Rel. err. of Diffusion Velocity') + \
                    ': ' + '{:e}'.format(IterationInfo['Rel. err. of Diffusion Velocity']) +'\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('FMM Computation time') + \
                    ': ' + '{:.2f}'.format(IterationInfo['FMM time']) + ' s (' + \
                                                     '{:.1f}'.format(IterationInfo['FMM time']/\
                                      IterationInfo['Total iteration time']*100.) + '%) ' + '\n'
    if "Perturbation time" in IterationInfo:
        msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Perturbation Field ') + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Interpolation time') + \
                    ': ' + '{:.2f}'.format(IterationInfo['Perturbation time']) + ' s (' + \
                                            '{:.1f}'.format(IterationInfo['Perturbation time']/\
                                      IterationInfo['Total iteration time']*100.) + '%) ' + '\n'
    msg += f"{'||':>57}\r" + '||' + '{:=^53}'.format('')
    print(msg)

def deletePrintedLines(NumberOfLineToDelete = 1):
    '''
    Deletes the last printed lines on the teminal.

    Parameters
    ----------
        NumberOfLineToDelete : :py:class:`int`
            Number of lines to delete.
    '''
    for i in range(NumberOfLineToDelete):
       sys.stdout.write('\x1b[1A')
       sys.stdout.write('\x1b[2K')
