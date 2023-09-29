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
MOLA - StructuralAnalysis.py

STRUCTURAL MODULE compatible with Cassiopee and MOLA

The objective of this module is to provide cgns based structural functions to perform the
static or dynamic analysis of structures. This module is fully based in Python.

Author: Mikel Balmaseda 

1. 26/05/2021  Mikel Balmaseda :  cgns adaptation of the previous Scripts developed durin the PhD
2. 12/05/2022  Miguel Garcia   :  Comments are introduced to make its reading easier and compatible with GitLab documentation

'''

# System modules
import sys, os
import numpy as np
from numpy.linalg import norm

import Converter.PyTree as C
import Converter.Internal as I

from .. import InternalShortcuts as J
from .. import PropellerAnalysis as PA

from . import ShortCuts as SJ
from . import ModalAnalysis   as MA
from . import NonlinearForcesModels as NFM

import MOLA.Data as D
import MOLA.Data.BEMT as BEMT
#from MOLA.Data.LiftingLine import LiftingLine
import MOLA.VPM.VortexParticleMethod as VPM
import MOLA.LiftingLine as LL

import Transform.PyTree as T


# Notices/Warnings colors
FAIL  = '\033[91m'
GREEN = '\033[92m' 
WARN  = '\033[93m'
MAGE  = '\033[95m'
CYAN  = '\033[96m'
ENDC  = '\033[0m'


#### AERODYNAMICS:
def Macro_BEMT(t, LiftingLine, RPM):
    '''
    This is a macro-function used for getting the values calculated by the BEMT theory 

    Parameters
    ----------

        t : cgns tree 
            Contains the LiftingLine information

        RPM : float
            rotational speed of the blade [revolutions per minute]

        PolarsInterpFuns: 
        
    Returns
    -------

        DictOfIntegralData : :py:class:`dict`
            dictionary including predictions

        Prop.Loads.Data : PUMA object

        SectionalLoadsLL : PUMA object
    '''

    print (CYAN+'Launching 3D BEMT computation...'+ENDC)

    DictAerodynamicProperties = J.get(t, '.AerodynamicProperties')

    #LiftingLine = I.getNodeFromName(t, 'LiftingLine')
    ResultsDict = BEMT.compute(LiftingLine, model='Drela',
                             AxialVelocity=DictAerodynamicProperties['FlightConditions']['Velocity'], RPM=RPM, Pitch=DictAerodynamicProperties['BladeParameters']['Pitch']['PitchAngle'], NumberOfBlades=DictAerodynamicProperties['BladeParameters']['NBlades'],
                             Density=DictAerodynamicProperties['FlightConditions']['Density'], Temperature=DictAerodynamicProperties['FlightConditions']['Temperature'])


#    ResultsDict = PA.computeBEMTaxial3D(LiftingLine, PolarsInterpFuns,
#    NBlades=DictAerodynamicProperties['BladeParameters']['NBlades'],
#    Constraint=DictAerodynamicProperties['BladeParameters']['Constraint'],
#    ConstraintValue=DictAerodynamicProperties['BladeParameters']['ConstraintValue'],
#    AttemptCommandGuess=[],
#
#    Velocity=[0.,0.,-DictAerodynamicProperties['FlightConditions']['Velocity']],  # Propeller's advance velocity (m/s)
#    RPM=RPM,              # Propellers angular speed (rev per min.)
#    Temperature = DictAerodynamicProperties['FlightConditions']['Temperature'],     # Temperature (Kelvin)
#    Density=DictAerodynamicProperties['FlightConditions']['Density'],          # Air density (kg/m3)
#    model=DictAerodynamicProperties['BEMTParameters']['model'],          # BEMT kind (Drela, Adkins or Heene)
#    TipLosses=DictAerodynamicProperties['BEMTParameters']['TipLosses'],
#
#    FailedAsNaN=True,
#    )

    print("RPM: %g rpm, Thrust: %g N,  Power: %g W,  Prop. Eff.: %g, | Pitch: %g deg"%(RPM, ResultsDict['Thrust'],ResultsDict['Power'],ResultsDict['PropulsiveEfficiency'],ResultsDict['Pitch']))
    print(WARN + '3D BEMT computation COMPLETED'+ENDC)

    I._addChild(t, LiftingLine)
 

    return ResultsDict, t , LiftingLine



def Macro_VPM(tStructure, LiftingLine, RPM):
    '''
    This is a macro-function used for getting the values calculated by the VPM theory 

    Parameters
    ----------

        t : cgns tree 
            Contains the LiftingLine information

        RPM : float
            rotational speed of the blade [revolutions per minute]

        PolarsInterpFuns: 
        
    Returns
    -------

        DictOfIntegralData : :py:class:`dict`
            dictionary including predictions

        Prop.Loads.Data : PUMA object

        SectionalLoadsLL : PUMA object
    '''
    #LiftingLine[0] = 'HAD1'
    
    #C.convertPyTree2File(LiftingLine, 'LLBeforeProper.cgns')
    Propeller = LL.buildPropeller(LiftingLine, NBlades = 4, InitialAzimutDirection = [1, 0, 0])
    #Propeller[0] = 'HAD1'

    C.convertPyTree2File(Propeller, 'LiftingLineVPM.cgns', 'bin_adf')

    
    VPMParameters = {
    ##############################################################################################
    ############################## Atmospheric/Freestream conditions #############################
    ##############################################################################################
        'Density'                       : 1.225,              #]0., +inf[, in kg.m^-3
        'EddyViscosityConstant'         : 0.1,                #[0., +inf[, constant for the eddy viscosity model, Cm(Mansour) around 0.1, Cs(Smagorinsky) around 0.15, Cr(Vreman) around 0.07
        'EddyViscosityModel'            : 'Vreman',           #Mansour, Mansour2, Smagorinsky, Vreman or None, select a LES model to compute the eddy viscosity
        'KinematicViscosity'            : 1.46e-5,            #[0., +inf[, in m^2.s^-1
        'Temperature'                   : 288.15,             #]0., +inf[, in K
        'VelocityFreestream'            : [0., 0., 0.], #in m.s^-1, freestream velocity
    ##############################################################################################
    ####################################### VPM parameters #######################################
    ##############################################################################################
        'AntiStretching'                : 0.6,                 #[0., 1.], 0 means particle strength fully takes vortex stretching, 1 means the particle size fully takes the vortex stretching
        'DiffusionScheme'               : 'CSM',              #PSE, CSM or None. gives the scheme used to compute the diffusion term of the vorticity equation
        'RegularisationKernel'          : 'Gaussian',         #The only available smoothing kernel for now is Gaussian. The others are on their way ;-)
        'SFSContribution'               : 0.,                 #[0., 1.], the closer to 0, the more the viscosity affects the particle strength, the closer to 1, the more it affects the particle size
        'SmoothingRatio'                : 2.,                 #[1., +inf[, in m, anywhere between 1. and 2.5 is good, the higher the NumberSource or the smaller the Resolution and the higher the SmoothingRatio should be to avoid blowups, the HOA kernel requires a higher smoothing
        'VorticityEquationScheme'       : 'Transpose',        #Classical, Transpose or Mixed, The schemes used to compute the vorticity equation are the classical scheme, the transpose scheme (conserves total vorticity) and the mixed scheme (a fusion of the previous two)
    ##############################################################################################
    #################################### Simulation Parameters ###################################
    ##############################################################################################
        'IntegrationOrder'              : 2,                  #[|1, 4|]1st, 2nd, 3rd or 4th order Runge Kutta
        'LowStorageIntegration'         : 1,                  #[|0, 1|], states if the classical or the low-storage Runge Kutta is used
    ##############################################################################################
    ###################################### Particles Control #####################################
    ##############################################################################################
        'CutoffXmin'                    : -1.4*10,           #]-inf, +inf[, in m, spatial Cutoff
        'CutoffXmax'                    : +1.4*10,           #]-inf, +inf[, in m, spatial Cutoff
        'CutoffYmin'                    : -1.4*10,           #]-inf, +inf[, in m, spatial Cutoff
        'CutoffYmax'                    : +1.4*10,           #]-inf, +inf[, in m, spatial Cutoff
        'CutoffZmin'                    : -1.4*10,           #]-inf, +inf[, in m, spatial Cutoff
        'CutoffZmax'                    : +0.30,           #]-inf, +inf[, in m, spatial Cutoff
        'ForcedDissipation'             : 0.,                #[0., +inf[, in %/s, gives the % of strength particles looses per sec, usefull to kill unnecessary particles without affecting the LLs
        'MaximumAgeAllowed'             : 72,                 #[|0., +inf[|,  particles are eliminated after MaximumAgeAllowed iterations, if MaximumAgeAllowed == 0, the age is not checked
        'MaximumAngleForMerging'        : 25.,               #[0., 180.[ in deg,   maximum angle   allowed between two particles to be merged
        'MaximumMergingVorticityFactor' : 100.,              #[0., +inf[, in %, particles can be merged if their combined strength is below the given poucentage of the maximum strength on the blades
        'MinimumOverlapForMerging'      : 2.,                #]0., +inf[, if two particles have at least an overlap of MinimumOverlapForMerging*SigmaRatio, they are considered for merging
        'MinimumVorticityFactor'        : 0.1,               #[0., +inf[, in %, sets the minimum strength kept as a percentage of the maximum strength on the blades
        'RedistributeParticlesBeyond'   : 0.7,               #[0., +inf[, do not redistribute particles if closer than RedistributeParticlesBeyond*Resolution from a LL
        'RedistributionKernel'          : 'None',            #M4Prime, M4, M3, M2, M1 or None, redistribution kernel used. the number gives the order preserved by the kernel
        'RedistributionPeriod'          : 5,                 #[|0., +inf[|, frequency at which particles are redistributed
        'RelaxationCutoff'              : 0.10,              #[0., +inf[, in Hz, is used during the relaxation process to realign the particles with their voticity and avoid having a non null divergence of the vorticity field
        'RemoveWeakParticlesBeyond'     : 0.7,               #[0., +inf[, do not remove weak particles if closer than RemoveWeakParticlesBeyond*Resolution from a LL
        'ResizeParticleFactor'          : 3.,                #[0, +inf[, resize particles that grow/shrink ResizeParticleFactor * Sigma0 (Sigma0 = Resolution*SmoothingRatio), if 0 then no resizing is done
        'StrengthRampAtbeginning'       : 72,                #[|0, +inf [|, limit the vorticity shed for the StrengthRampAtbeginning first iterations for the wake to stabilise
    ##############################################################################################
    ####################################### FMM parameters #######################################
    ##############################################################################################
        'FarFieldApproximationOrder'    : 6,                 #[|6, 12|], order of the polynomial which approximates the far field interactions, the higher the more accurate and the more costly
        'IterationTuningFMM'            : 50,                #[|0., +inf[|, frequency at which the FMM is compared to the direct computation, gives the relative L2 error
        'NearFieldOverlappingRatio'     : 0.5,               #[0., 1.], gives the overlap beyond which the interactions between groups of particles are approximated by the FMM. The smaller the more accurate and the more costly
        'NumberOfThreads'               : 'auto',            #number of threads of the machine used. If 'auto', the highest number of threads is set
    }
    LiftingLineParameters = {
    ##############################################################################################
    ################################## Lifting Lines parameters ##################################
    ##############################################################################################
        'CirculationThreshold'             : 1e-4,                     #]0., +inf[, convergence criteria for the circulation sub-iteration process, somewhere between 1e-3 and 1e-6 is ok
        'CirculationRelaxation'            : 1./4.,                    #]0., 1.], relaxation parameter of the circulation sub-iterations, somwhere between 0.1 and 1 is good, the more unstable the simulation, the lower it should be
        'GammaZeroAtRoot'                  : 1,                        #[|0, 1|], sets the circulation of the root of the blade to zero
        'GammaZeroAtTip'                   : 1,                        #[|0, 1|], sets the circulation of the tip  of the blade to zero
        'GhostParticleAtRoot'              : 0,                        #[|0, 1|], add a particles after the root of the blade
        'GhostParticleAtTip'               : 0,                        #[|0, 1|], add a particles after the tip  of the blade
        'IntegralLaw'                      : 'linear',                 #linear, pchip, interp1d or akima, gives the type of interpolation of the circulation on the lifting lines
        'MaxLiftingLineSubIterations'      : 100,                      #[|0, +inf[|, max number of sub iteration when computing the LL circulations
        'MinNbShedParticlesPerLiftingLine' : 40,                       #[|10, +inf[|, minimum number of station for every LL from which particles are shed
        'ParticleDistribution'             : dict(kind='tanhTwoSides', #uniform, tanhOneSide, tanhTwoSides or ratio, repatition law of the particles on the Lifting Lines
                                                  #FirstCellHeight = 0.004,
                                                  FirstSegmentRatio=2.,#]0., +inf[, size of the particles at the root of the blades relative to Sigma0 (i.e. Resolution*SmoothingRatio)
                                                  LastSegmentRatio=0.5,#]0., +inf[, size of the particles at the tip  of the blades relative to Sigma0 (i.e. Resolution*SmoothingRatio)
                                                  Symmetrical=False),  #[|0, 1|], gives a symmetrical repartition of particles along the blades or not, if symmetrical, MinNbShedParticlesPerLiftingLine should be even
        'Pitch'                            : 0.,                      #]-180, 180[ in deg, gives the pitch added to all the lifting lines, if 0 no pitch is added
        'RPM'                              : RPM,                    #]-inf, inf[in tr.min^-1, rotation per minute
        'VelocityTranslation'              : [0., 0., 0.],             #E |R^3, in m.s^-1, kinematic velocity of the Lifting Lines
    }

    #b = 0.8 - 0.12
    #VPMParameters['TimeStep'] = 5.*b/(VPMParameters['MinNbShedParticlesPerLiftingLine'] - 2.)\
    #                            /(np.linalg.norm(VPMParameters['VelocityFreestream'] - LiftingLineParameters['VelocityTranslation'])**2\
    #                            + (VPMParameters['RPM']*np.pi/30.*0.8)**2)**0.5
    #b = 0.8
    VPMParameters['TimeStep'] = 5./LiftingLineParameters['RPM']/6.

    DIRECTORY_OUTPUT = 'OUTPUT_VPM_%s/'%RPM
    RESTART = 0
    RestartPath = None
    EulerianPath = None
    HybridParameters = {}

    LiftingLinePath = 'LiftingLineVPM.cgns'
    PolarsFilename = 'INPUT/Polars/Polars.cgns'

    NumberOfIterations = 1080
    VisualisationOptions = {'addLiftingLineSurfaces':True}
    StdDeviationSample = 50
    SaveVPMPeriod = 120
    Verbose = True 
    SaveImageOptions={'ImagesDirectory':'FRAMES_%s'%RPM}
    Surface = 0. 
    FieldsExtractionGrid = [] 
    SaveFieldsPeriod = np.inf 
    SaveImagePeriod = np.inf 


    try: os.makedirs(DIRECTORY_OUTPUT)
    except: pass
    if PolarsFilename: AirfoilPolars = VPM.loadAirfoilPolars(PolarsFilename)
    else: AirfoilPolars = None
    if RestartPath:
        t = open(RestartPath)
        try: tE = open('tE.cgns') # LB: TODO dangerous; rather use os.path.isfile()
        except: tE = []
    else:
        #LiftingLine = LiftingLine
        if LiftingLinePath: LiftingLine = VPM.open(LiftingLinePath) # LB: TODO dangerous; rather use os.path.isfile()
        else: LiftingLine = []
        #if EulerianPath: EulerianMesh = open(EulerianPath)
        #else: EulerianMesh = []
        
        EulerianMesh = EulerianPath
        t, tE = VPM.initialiseVPM(EulerianMesh = EulerianMesh, HybridParameters = HybridParameters,
                    LiftingLineTree = LiftingLine,LiftingLineParameters = LiftingLineParameters,
                    PolarInterpolator = AirfoilPolars, VPMParameters = VPMParameters)
    
    IterationInfo = {'Rel. err. of Velocity': 0, 'Rel. err. of Velocity Gradient': 0,
                        'Rel. err. of PSE': 0}
    TotalTime = J.tic()
    sp = VPM.getVPMParameters(t)
    Np = VPM.pickParticlesZone(t)[1][0]
    LiftingLines = LL.getLiftingLines(t)
    h = sp['Resolution'][0]
    it = sp['CurrentIteration']
    simuTime = sp['Time']
    maxAge = sp['MaximumAgeAllowed']
    PSE = VPM.DiffusionScheme_str2int[sp['DiffusionScheme']] == 1
    Freestream = (np.linalg.norm(sp['VelocityFreestream']) != 0.)
    Wing = (len(I.getZones(LiftingLines)) == 1)
    if AirfoilPolars: VisualisationOptions['AirfoilPolarsFilename'] = PolarsFilename
    else: VisualisationOptions['addLiftingLineSurfaces'] = False
    #filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_It%d.cgns'%it[0])
    #VPM.save(t, filename, VisualisationOptions, 'bin_adf')
    #filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_It%d.tp'%it[0])
    #VPM.save(t, filename, VisualisationOptions, 'bin_tp')
    #J.createSymbolicLink(filename,  DIRECTORY_OUTPUT + '.cgns')
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
        VPM.computeNextTimeStep(t)
        IterationInfo['Iteration'] = it[0] 
        IterationInfo['Percentage'] = it[0]/NumberOfIterations*100.
        IterationInfo['Physical time'] = simuTime[0]
        IterationInfo = VPM.generateParticlesInHybridInterfaces(t, tE, IterationInfo)
        
        # Modify the age of the maximum old particles to ease the convergence 
        if it[0]%2: maxAge[0] += 1

        IterationInfo = VPM.populationControl(t, [], IterationInfo)
        IterationInfo = VPM.shedParticlesFromLiftingLines(t, AirfoilPolars, IterationInfo)
        IterationInfo['Number of particles'] = Np[0]
        IterationInfo = VPM.solveVorticityEquation(t, IterationInfo = IterationInfo)
        IterationInfo['Total iteration time'] = J.tic() - IterationTime
        IterationInfo = VPM.getAerodynamicCoefficientsOnLiftingLine(LiftingLines, Wings = Wing,
                               StdDeviationSample = StdDeviationSample, Freestream = Freestream, 
                                               IterationInfo = IterationInfo, Surface = Surface)
        IterationInfo['Total simulation time'] = J.tic() - TotalTime
        if Verbose: VPM.printIterationInfo(IterationInfo, PSE = PSE, Wings = Wing)
        if (SAVE_FIELDS or SAVE_ALL) and FieldsExtractionGrid:
            VPM.extract(t, FieldsExtractionGrid, 5000)
            filename = os.path.join(DIRECTORY_OUTPUT, 'fields_It%d.cgns'%it)
            VPM.save(FieldsExtractionGrid, filename)
            J.createSymbolicLink(filename, 'fields.cgns')
        if SAVE_IMAGE or SAVE_ALL:
            VPM.setVisualization(t, **VisualisationOptions)
            VPM.saveImage(t, **SaveImageOptions)
        if SAVE_VPM or SAVE_ALL:
            filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_It%d.cgns'%it)
            VPM.save(t, filename, VisualisationOptions)
            J.createSymbolicLink(filename,  DIRECTORY_OUTPUT + '.cgns')
        if CONVERGED: break
        
        


    VPM.save(t, DIRECTORY_OUTPUT + '.cgns', VisualisationOptions)
    for _ in range(3): print('||' + '{:=^50}'.format(''))
    print('||' + '{:-^50}'.format(' End of VPM computation '))
    for _ in range(3): print('||' + '{:=^50}'.format(''))

    
    LiftingLineSingle = I.copyTree(I.getNodesFromName(t, '*blade1')[0])
    Azimuth = sp['Time'][-1]/60.*RPM*360 % 360
    
    LiftingLineSingle = T.rotate(LiftingLineSingle, (0,0,0),(0,0,1),-Azimuth, vectors = [['fx', 'fy', 'fz'], ['mx', 'my', 'mz'], ['bx', 'by', 'bz'], ['nx', 'ny', 'nz'], ['tx', 'ty', 'tz']])
    print(Azimuth)
    
    LiftingLineSingle[0] = 'LiftingLine'

    return LiftingLineSingle






    def preprocessVPMforAEL():
        try: os.makedirs(DIRECTORY_OUTPUT)
        except: pass

        if PolarsFilename: AirfoilPolars = loadAirfoilPolars(PolarsFilename)
        else: AirfoilPolars = None

        if RestartPath:
            t = open(RestartPath)
            try: tE = open('tE.cgns') # LB: TODO dangerous; rather use os.path.isfile()
            except: tE = []
        else:
            if LiftingLinePath: LiftingLine = open(LiftingLinePath) # LB: TODO dangerous; rather use os.path.isfile()
            else: LiftingLine = []
            #if EulerianPath: EulerianMesh = open(EulerianPath)
            #else: EulerianMesh = []
            EulerianMesh = EulerianPath
            t, tE = initialiseVPM(EulerianMesh = EulerianMesh, HybridParameters = HybridParameters,
                        LiftingLineTree = LiftingLine,LiftingLineParameters = LiftingLineParameters,
                        PolarInterpolator = AirfoilPolars, VPMParameters = VPMParameters)

        
        IterationInfo = {'Rel. err. of Velocity': 0, 'Rel. err. of Velocity Gradient': 0,
                            'Rel. err. of PSE': 0}
        TotalTime = J.tic()
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
            
            IterationTime = J.tic()
            computeNextTimeStep(t)
            IterationInfo['Iteration'] = it[0]
            IterationInfo['Percentage'] = it[0]/NumberOfIterations*100.
            IterationInfo['Physical time'] = simuTime[0]
            IterationInfo = generateParticlesInHybridInterfaces(t, tE, IterationInfo)
            IterationInfo = populationControl(t, [], IterationInfo)
            IterationInfo = shedParticlesFromLiftingLines(t, AirfoilPolars, IterationInfo)
            IterationInfo['Number of particles'] = Np[0]
            IterationInfo = solveVorticityEquation(t, IterationInfo = IterationInfo)
            IterationInfo['Total iteration time'] = J.tic() - IterationTime
            IterationInfo = getAerodynamicCoefficientsOnLiftingLine(LiftingLines, Wings = Wing,
                                   StdDeviationSample = StdDeviationSample, Freestream = Freestream, 
                                                   IterationInfo = IterationInfo, Surface = Surface)
            IterationInfo['Total simulation time'] = J.tic() - TotalTime
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


    XXXXX







    print (CYAN+'Launching VPM computation...'+ENDC)

    DictAerodynamicProperties = J.get(t, '.AerodynamicProperties')

    #LiftingLine = I.getNodeFromName(t, 'LiftingLine')

    ResultsDict = BEMT.compute(LL, model='Drela',
                             AxialVelocity=DictAerodynamicProperties['FlightConditions']['Velocity'], RPM=RPM, Pitch=DictAerodynamicProperties['BladeParameters']['Pitch']['PitchAngle'], NumberOfBlades=DictAerodynamicProperties['BladeParameters']['NBlades'],
                             Density=DictAerodynamicProperties['FlightConditions']['Density'], Temperature=DictAerodynamicProperties['FlightConditions']['Temperature'])


    print("RPM: %g rpm, Thrust: %g N,  Power: %g W,  Prop. Eff.: %g, | Pitch: %g deg"%(RPM, ResultsDict['Thrust'],ResultsDict['Power'],ResultsDict['PropulsiveEfficiency'],ResultsDict['Pitch']))
    print(WARN + '3D BEMT computation COMPLETED'+ENDC)

    I._addChild(t, LL)
 

    return ResultsDict, t , LL



def ComputeExternalForce(t):

    if ForceType == 'CoupledRotatoryEquilibrium':
        
        LiftingLine = I.getNodeFromNameAndType(t, 'LiftingLine', 'Zone_t')
        
        LiftingLine = SJ.updateLiftingLineFromStructureLETE(LiftingLine, tFOM, RPM)
        LiftingLine, ResultsDict = Macro_BEMT(LiftingLine, PolarsInterpFuns, tFOM, RPM)
    
        FTE, FLE = SJ.FrocesAndMomentsFromLiftingLine2ForcesAtLETE(LiftingLine, tFOM, RPM)
        
        return  SJ.LETEvector2FullDim(tFOM, FLE, FTE)



def Compute_IntFnl(t):

    
    


    return Fnl




#STRUCTURE:
def DynamicSolver_Li2020(t, RPM, fext):
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













#        Static Solvers:

def StaticSolver_Newton_Raphson(t, RPM, ForceCoeff):
    "Function returning the reduced static non-linear solution using the IC non-linear function"

    DictStructParam = J.get(t, '.StructuralParameters')
    DictInternalForcesCoefficients = J.get(t, '.InternalForcesCoefficients')
    DictSimulaParam = J.get(t, '.SimulationParameters')
    
    # Get the needed variables from the cgns tree:

    V = SJ.GetReducedBaseFromCGNS(t, RPM)  # Base de reduction PHI

    nq       = DictStructParam['ROMProperties']['NModes'][0]
    time = SJ.ComputeTimeVector(t)[1][DictSimulaParam['IntegrationProperties']['Steps4CentrifugalForce'][0]-1:]
    time_Save = []

    nitermax = DictSimulaParam['IntegrationProperties']['NumberOfMaxIterations'][0]
    nincr    = DictSimulaParam['IntegrationProperties']['StaticSteps'][0]
    try:
        Aij      = DictInternalForcesCoefficients['%sRPM'%np.round(RPM,2)]['Aij']
        Bijm     = DictInternalForcesCoefficients['%sRPM'%np.round(RPM,2)]['Bijm']
    except:
        Aij, Bijm = 0, 0 

        print(WARN + 'Warning!! Aij and Bijm not read!'+ENDC)

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
            
        Fextproj[:,0] = ForceCoeff * SJ.ComputeLoadingFromTimeOrIncrement(t, RPM, incr-1)
        
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
                time_Save.append(time[incr-1])
                
        elif (not (incr-1)%DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]) or (incr == nincr):
            
            it2 += 1
            q_Save[:, it2] = q.ravel()
            Fnl_Save[:, it2] = Fnlproj.ravel()
            Fext_Save[:, it2] = Fextproj.ravel()
            time_Save.append(time[incr-1])
             
    
    return q_Save, Fnl_Save, Fext_Save, np.array(time_Save)

def StaticSolver_Newton_Raphson1IncrFext(t, RPM, fext):
    "Function returning the reduced static non-linear solution using the IC non-linear function"

    DictStructParam = J.get(t, '.StructuralParameters')
    DictInternalForcesCoefficients = J.get(t, '.InternalForcesCoefficients')
    DictSimulaParam = J.get(t, '.SimulationParameters')
    
    # Get the needed variables from the cgns tree:

    V = SJ.GetReducedBaseFromCGNS(t, RPM)  # Base de reduction PHI (if parametric model, we give PHIAug o U)
    
    nq       = DictStructParam['ROMProperties']['NModes'][0]
    nitermax = DictSimulaParam['IntegrationProperties']['NumberOfMaxIterations'][0]
    nincr    = DictSimulaParam['IntegrationProperties']['StaticSteps'][0]
    try:
        Aij      = DictInternalForcesCoefficients['%sRPM'%np.round(RPM,2)]['Aij']
        Bijm     = DictInternalForcesCoefficients['%sRPM'%np.round(RPM,2)]['Bijm']
    except:
        Aij, Bijm = 0, 0 

        print(WARN + 'Warning!! Aij and Bijm not read!'+ENDC)

    Kproj = SJ.getMatrixFromCGNS(t, 'Komeg', RPM)
    

    # Initialisation des vecteurs du calcul:
    q = np.zeros((nq,1))
    Fextproj = np.zeros((nq,1))
    Fnlproj = np.zeros((nq,1))
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
        # Solve displacement increment (linear matrix equation)
        dq = -np.linalg.solve(Ktanproj,Resi)
        q = q + dq
        
        #Compute internal forces vector
        Fnlproj = NFM.fnl_Proj(t, q, Aij, Bijm)
       
        #compute residual
        Resi = np.dot(Kproj,q) + Fnlproj - Fextproj
        
    
    # Save the data in the matrices:
   
    print(GREEN + 'Nb iterations = %s, for increment: 1, residual =  %0.4f'%(niter, norm(Resi,2))+ENDC)
    
    
    
    return q, Fnlproj, Fextproj 


def DynamicSolver_HHTalpha(t, RPM, ForceCoeff):

    

    DictStructParam = J.get(t, '.StructuralParameters')
    DictInternalForcesCoefficients = J.get(t, '.InternalForcesCoefficients')
    DictSimulaParam = J.get(t, '.SimulationParameters')
    
    # Define and get the time vector:
    time = SJ.ComputeTimeVector(t)[1][DictSimulaParam['IntegrationProperties']['Steps4CentrifugalForce'][0]-1:]
    nincr = len(time)
    dt = DictSimulaParam['LoadingProperties']['TimeProperties']['dt'][0]
    time_Save = []
    
    # Get the needed variables from the cgns tree:

    V = SJ.GetReducedBaseFromCGNS(t, RPM)  # Base de reduction PHI

    nq       = DictStructParam['ROMProperties']['NModes'][0]

    nitermax = DictSimulaParam['IntegrationProperties']['NumberOfMaxIterations'][0]
    
    try:
        Aij      = DictInternalForcesCoefficients['%sRPM'%np.round(RPM,2)]['Aij']
        Bijm     = DictInternalForcesCoefficients['%sRPM'%np.round(RPM,2)]['Bijm']
    except:
        Aij, Bijm = 0, 0 

        print(WARN + 'Warning!! Aij and Bijm not read!'+ENDC)

    Kproj = SJ.getMatrixFromCGNS(t, 'Komeg', RPM)
    Cproj = SJ.getMatrixFromCGNS(t, 'C', RPM)
    Mproj = SJ.getMatrixFromCGNS(t, 'M', RPM)
    
    # Initialisation des vecteurs du calcul:
    q = np.zeros((nq,1))
    qp = np.zeros((nq,1))
    qpp = np.zeros((nq,1))

    q_m1 = np.zeros((nq,1))
    qp_m1 = np.zeros((nq,1))
    qpp_m1 = np.zeros((nq,1))
    
    Fextproj = np.zeros((nq,1))
    Fnlproj = np.zeros((nq,1))

    Fextproj_m1 = np.zeros((nq,1))
    Fnlproj_m1 = np.zeros((nq,1))

    # Initialisation des vecteurs de sauvegarde:
    if int(DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]) == 0:
        q_Save   = np.zeros((nq, 1))
        qp_Save   = np.zeros((nq, 1))
        qpp_Save   = np.zeros((nq, 1)) 
        Fnl_Save = np.zeros((nq, 1)) 
        Fext_Save     = np.zeros((nq, 1)) 
    else:
        if int((nincr - 1)%DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]) == 0 :
            q_Save   = np.zeros((nq, 1 + int((nincr-1)//DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]))) 
            qp_Save   = np.zeros((nq, 1 + int((nincr-1)//DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]))) 
            qpp_Save   = np.zeros((nq, 1 + int((nincr-1)//DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]))) 
            Fnl_Save = np.zeros((nq, 1 + int((nincr-1)//DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]))) 
            Fext_Save     = np.zeros((nq, 1+ int((nincr-1)//DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]))) 
        else:
            q_Save   = np.zeros((nq, 2+int((nincr-1)//DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]))) 
            qp_Save   = np.zeros((nq, 2+int((nincr-1)//DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]))) 
            qpp_Save   = np.zeros((nq, 2+int((nincr-1)//DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]))) 
            Fnl_Save = np.zeros((nq, 2+int((nincr-1)//DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]))) 
            Fext_Save     = np.zeros((nq, 2+int((nincr-1)//DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]))) 

    # HHT-alpha method parameters:
    alpha = DictSimulaParam['IntegrationProperties']['IntegrationMethod']['Parameters']['Alpha'][0]

    beta = 0.25 * (1+alpha)**2
    gamma = 0.5 * (1+2*alpha)

    it2 =-1
    for incr in range(1,nincr+1):
            
        Fextproj[:,0] = ForceCoeff * SJ.ComputeLoadingFromTimeOrIncrement(t, RPM, incr-1)
        
        CsteTerm_previous = Cproj.dot(qp_m1) + Kproj.dot(q_m1) + Fnlproj_m1 - Fextproj_m1

        # Initial prediction:

        q = q + dt*qp + 0.5*(1-2*beta)*(dt**2)*qpp
        qp = qp + (1-gamma)*dt*qpp
        qpp = np.zeros((nq,1))

        Fnlproj = NFM.fnl_Proj(t, q, Aij, Bijm)
       
        Resi = Mproj.dot(qpp) + (1. - alpha)*(Cproj.dot(qp) + Kproj.dot(q) + Fnlproj - Fextproj) + alpha*CsteTerm_previous

        niter=0
    
        while np.linalg.norm(Resi,2) > DictSimulaParam['IntegrationProperties']['EpsConvergenceCriteria'][0]:
        
            niter=niter+1
            
            if niter>=nitermax:
                print(FAIL+'Too much N.L. iterations for increment number %s'%str(incr)+ENDC)
                break

            #Compute tangent stiffness matrix
            
            Ktanproj = Kproj + NFM.Knl_Jacobian_Proj(t, q, Aij, Bijm)
            Ctanproj = Cproj
            
            # Compute the Jacobian:

            Jacobian = (1./(beta*dt**2))*Mproj   + (1. - alpha)*(gamma/(beta*dt))*Ctanproj + (1. - alpha)*Ktanproj 
            
            # Solve displacement increment
            
            dq = -np.linalg.solve(Jacobian,Resi)
            q = q + dq
            qp = qp + (gamma/(beta*dt))*dq
            qpp = qpp + (1./(beta*dt**2))*dq
            
            #Compute internal forces vector
            Fnlproj = NFM.fnl_Proj(t, q, Aij, Bijm)

           
            #compute residual
            Resi = Mproj.dot(qpp) + (1. - alpha)*(Cproj.dot(qp) + Kproj.dot(q) + Fnlproj - Fextproj) + alpha*CsteTerm_previous

        q_m1 = q
        qp_m1 = qp
        qpp_m1 = qpp
        Fextproj_m1 = Fextproj
        Fnlproj_m1 = Fnlproj
        # Save the data in the matrices:
        if DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0] == 0: 
            if incr == nincr: 
                it2 += 1
                q_Save[:, it2] = q.ravel()
                qp_Save[:, it2] = qp.ravel()
                qpp_Save[:, it2] = qpp.ravel()
                
                Fnl_Save[:, it2] = Fnlproj.ravel()
                Fext_Save[:, it2] = Fextproj.ravel()
                time_Save.append(time[incr-1]) 

        elif (not (incr-1)%DictSimulaParam['IntegrationProperties']['SaveEveryNIt'][0]) or (incr == nincr):
            
            it2 += 1
            q_Save[:, it2] = q.ravel()
            qp_Save[:, it2] = qp.ravel()
            qpp_Save[:, it2] = qpp.ravel()
            Fnl_Save[:, it2] = Fnlproj.ravel()
            Fext_Save[:, it2] = Fextproj.ravel()
            time_Save.append(time[incr-1]) 
                
    return q_Save, qp_Save, qpp_Save, Fnlproj, Fext_Save, np.array(time_Save)


def SolveStatic(t, RPM, ForceCoeff=1.):
    '''MacroFunction of Time Integration Methods

        Implemented methods:

               - 'Newton_Raphson
               - 'FixPoint'
    '''
    DictSimulaParam = J.get(t, '.SimulationParameters')
    
    if DictSimulaParam['IntegrationProperties']['IntegrationMethod']['MethodName'] == 'Newton_Raphson':
        
        q, fnl_q , Fext_q, time =  StaticSolver_Newton_Raphson(t, RPM, ForceCoeff)
        
    if DictSimulaParam['IntegrationProperties']['IntegrationMethod']['MethodName'] == 'AEL':
        if DictSimulaParam['IntegrationProperties']['IntegrationMethod']['Parameters']['Type'] == 'Static_Newton1':
            # NewtonRhapson with one iteration:
            q, fnl_q , Fext_q =  StaticSolver_Newton_Raphson1IncrFext(t, RPM, ForceCoeff)
            time = None
            
        if DictSimulaParam['IntegrationProperties']['IntegrationMethod']['Parameters']['Type'] == 'FOM':
            pass # ComputeStaticU4GivenLoading(t, RPM, LoadVector, **kwargs)
            

    # Manque save to Tree


    return [q], fnl_q, Fext_q, time



def SolveDynamic(t, RPM, ForceCoeff=1.):
    '''MacroFunction of Time marching Integration Methods

        Implemented methods:

               - ''
               - ''
    '''
    DictSimulaParam = J.get(t, '.SimulationParameters')

    if DictSimulaParam['IntegrationProperties']['IntegrationMethod']['MethodName'] == 'HHT-Alpha':
        
        q,qp,qpp, fnl_q , Fext_q, time =  DynamicSolver_HHTalpha(t, RPM, ForceCoeff)

    if DictSimulaParam['IntegrationProperties']['IntegrationMethod']['MethodName'] == 'Li2020':
        pass
    

    # Manque save to Tree


    return [q, qp, qpp], fnl_q, Fext_q, time








def SolveROM(tROM, InputRPM = None, InputForceCoeff = None): 

    DictSimulaParam = J.get(tROM, '.SimulationParameters')
    DictStructParam = J.get(tROM, '.StructuralParameters')
    DictInternalForcesCoefficients = J.get(tROM, '.InternalForcesCoefficients')

    TypeOfSolver  = DictSimulaParam['IntegrationProperties']['SolverType']
 
    Solution = {}

    if InputRPM == None: 
        InputRPM = DictSimulaParam['RotatingProperties']['RPMs']
        InputForceCoeff = DictSimulaParam['LoadingProperties']['ForceIntensityCoeff']
        
    for RPM in InputRPM:

        try:
            ExpansionBase = DictInternalForcesCoefficients['%sRPM'%np.round(RPM,2)]['ExpansionBase']
        except:
            ExpansionBase = None
        
        Solution['%sRPM'%np.round(RPM,2)] = {}
        PHI = SJ.GetReducedBaseFromCGNS(tROM, RPM)

        for ForceCoeff in InputForceCoeff:
            Solution['%sRPM'%np.round(RPM,2)]['FCoeff%s'%ForceCoeff] = {}

            if TypeOfSolver == 'Static':
                q_qp_qpp, fnl_q, fext_q, time  = SolveStatic(tROM, RPM, ForceCoeff)
                
            
            elif TypeOfSolver == 'Dynamic':
                q_qp_qpp, fnl_q, fext_q, time  = SolveDynamic(tROM, RPM, ForceCoeff)
                
                
            
            # Save the reduced q_qp_qpp:

            Solution = SJ.SaveSolution2PythonDict(Solution, ForceCoeff, RPM, PHI, q_qp_qpp, fnl_q, fext_q, DictSimulaParam['LoadingProperties']['ExternalForcesVector'] , time, DictStructParam['ROMProperties']['ROMForceType'] == 'ICE',ExpansionBase )

    return Solution
                