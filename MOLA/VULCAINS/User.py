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
VULCAINS (Viscous Unsteady Lagrangian Code for Aerodynamics with Incompressible Navier-Stokes)

This module is used as a launcher for the VULCAINS simulations. VPM-only, coupled VPM-Lifting Line
and hybrid Lagrangian-Eulerian (VPM-URANS) simulations are available.

Version:
0.5

Author:
Johan VALENTIN
'''

import os
import numpy as np
import Converter.PyTree as C
import Converter.Internal as I
import Transform.PyTree as T
from .. import InternalShortcuts as J
from . import Main as V

####################################################################################################
####################################################################################################
############################################## Solver ##############################################
####################################################################################################
####################################################################################################
def compute(Parameters = {}, Polars  = [], EulerianMesh = None, PerturbationField = [],
    LiftingLines = [], NumberOfIterations = 1000, RestartPath = None, DIRECTORY_OUTPUT = 'OUTPUT',
    SaveFields = ['all'], StdDeviationSample = 50, SaveVPMPeriod = 100, Verbose = True,
    VisualisationOptions = {'addLiftingLineSurfaces':True}, SaveImageOptions = {}, Surface = 0.,
    FieldsExtractionGrid = [], SaveFieldsPeriod = np.inf, SaveImagePeriod = np.inf,
    NoRedistributionZones = []):
    '''
    Launches the VPM solver.

    Parameters
    ----------
        Parameters : :py:class:`dict` of :py:class:`dict`
            User-provided parameters for the VULCAINS simulation. If a parameter is not filled out,
            a default value will be provided.

                FluidParameters : :py:class:`dict`
                    Fluid-related parameters as established in
                    :py:func:`~MOLA.VULCAINS.Main.getDefaultFluidParameters`
                
                ModelingParameters : :py:class:`dict`
                    Method-related parameters as established in
                    :py:func:`~MOLA.VULCAINS.Main.getDefaultModelingParameters`
                
                NumericalParameters : :py:class:`dict`
                    Simulation-related parameters as established in
                    :py:func:`~MOLA.VULCAINS.Main.getDefaultNumericalParameters`
                
                LiftingLineParameters : :py:class:`dict`
                    LiftingLine-related parameters as established in
                    :py:func:`~MOLA.VULCAINS.Main.getDefaultLiftingLineParameters`
                
                HybridParameters : :py:class:`dict`
                    Hybrid-related parameters as established in
                    :py:func:`~MOLA.VULCAINS.Main.getDefaultHybridParameters`

        Polars : :py:class:`list` of :py:class:`~MOLA.Data.Zone.Zone` or :py:class:`str`
            Enhanced **Polars** for each airfoil, containing also foilwise
            distributions fields (``Cp``, ``theta``, ``delta1``...).
            As provided by :py:func:`MOLA.WorkflowAirfoil.buildPolar` or
            :py:func:`MOLA.LiftingLine.convertHOSTPolarFile2PyZonePolar`

            .. note::
              if input type is a :py:class:`str`, then **Polars** is
              interpreted as a CGNS file name containing the airfoil polars data

        EulerianMesh : :py:class:`~MOLA.Data.Tree.Tree` or :py:class:`str` or :py:obj:`None`
            Eulerian mesh or its file path.

        PerturbationField : :py:class:`~MOLA.Data.Tree.Tree` or :py:class:`str` or :py:obj:`None`
            Perturbation field or its file path. Must contain

        LiftingLines : :py:class:`~MOLA.Data.Tree.Tree` or :py:class:`str` or :py:obj:`None`
            Lifting Lines or its file path.

        NumberOfIterations : int
            Number of time iteration to perform.

        RestartPath : :py:class:`str` or :py:obj:`None`
            File path of the complete VULCAINS tree from where the simulation must
            start from (if any). if ``RestartPath == None``, a new simulation is launched.

        DIRECTORY_OUTPUT : str
            Directory path where the simulation CGNS output files will be written.

        SaveFields : :py:class:`list` of :py:class:`str`, or :py:class:`str`
            Fields to save at each particle.
            If ``'all'``, then they are all saved.
            Possible fields are:

            * ``VelocityInduced``
                description

            * ``VelocityPerturbation``
                description

            * ``VelocityDiffusion``
                description

            * ``gradxVelocity``
                description

            * ``gradyVelocity``
                description

            * ``gradzVelocity``
                description

            * ``PSE``
                description

            * ``Vorticity``
                description

            * ``Alpha``
                description

            * ``Stretching``
                description

            * ``rotU``
                description

            * ``Velocity``
                description

            * ``Age``
                description

            * ``Sigma``
                description

            * ``Cvisq``
                description

            * ``Nu``
                description

            * ``divUd``
                description

            * ``Enstrophyf``
                description

            * ``Enstrophy``
                description

            * ``EnstrophyM1``
                description

            * ``StrengthMagnitude``
                description

            * ``VelocityMagnitude`` 
                description

            * ``VorticityMagnitude``
                description

            .. hint::
                Use *SaveFields* = ``"all"`` for saving all fields

        VisualisationOptions : :py:class:`dict`
            Keyword-argument parameters provided to :py:func:`~MOLA.VULCAINS.Main.setVisualization`,
            used for setting the scene of the visualisation

        StdDeviationSample : int
            Number of iterations for computing the standard deviation.

        SaveVPMPeriod : :py:class:`int`
            Period for saving the entire VPM tree.

        Verbose : bool
            States whether the VPM solver prompts the VPM information during the simulation.

        SaveImageOptions : :py:class:`dict`
            Keyword-argument parameters provided to :py:func:`~MOLA.VULCAINS.Main.setVisualization`
            used for creating the images (see also **VisualisationOptions** argument)

        Surface : float
            Surface of Lifting Line wing for the computation of its aerodynamic coefficients (if 
            any).

        FieldsExtractionGrid : Tree
            Probes of the VPM field. Will extract the simulation fields onto the given grids,
            surfaces, points...

        SaveFieldsPeriod : int
            Period at which the **FieldsExtractionGrid** is extracted and saved.

        SaveImagePeriod : :py:class:`int`
            Frequency at which an image of the VPM simulation is saved (see **SaveImageOptions** argument).

        NoRedistributionZones: :py:class:`list` of :py:class:`~MOLA.Data.Zone.Zone`
            Particles cannot be deleted or redistributed within the coordinates of the zones in
            NoRedistributionZones. These zones must be rectangular parallelepipeds.
    '''
    if Verbose: V.enablePrint()
    else: V.blockPrint()

    if not V.printedlogo[0]:
        # V.show(logo)
        V.printedlogo[0] = True
    
    try: os.makedirs(DIRECTORY_OUTPUT)
    except: pass

    if isinstance(Polars, str): Polars = V.load(Polars)
    V.buildPolarsInterpolator(Polars)
    V.addSafeZones(NoRedistributionZones)
    if RestartPath: t = restartComputation(path = RestartPath, Parameters = Parameters)
    else: t = initialiseComputation(Parameters = Parameters, EulerianMesh = EulerianMesh,
                                 LiftingLines = LiftingLines, PerturbationField = PerturbationField)


    SaveFields = V.checkSaveFields(SaveFields)

    it = V.getParameter(t, 'CurrentIteration')
    if Polars: VisualisationOptions['AirfoilPolars'] = Polars
    else: VisualisationOptions['addLiftingLineSurfaces'] = False
    filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_It%d.cgns'%it[0])
    V.save(t, filename, VisualisationOptions, SaveFields)
    J.createSymbolicLink(filename,  DIRECTORY_OUTPUT + '.cgns')
    for _ in range(3): V.show(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))
    V.show(f"{'||':>57}\r" + '||' + '{:=^53}'.format(' Begin VPM Computation '))
    for _ in range(3): V.show(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))

    iterateVPM(t, SaveFields, NumberOfIterations, DIRECTORY_OUTPUT,
        VisualisationOptions, SaveImageOptions, SaveFieldsPeriod, SaveImagePeriod, SaveVPMPeriod,
                                                  StdDeviationSample, FieldsExtractionGrid, Surface)
    if FieldsExtractionGrid:
        extractFields(t, FieldsExtractionGrid, 5300)
        filename = os.path.join(DIRECTORY_OUTPUT, 'fields_It%d.cgns'%it)
        V.save(FieldsExtractionGrid, filename, VisualisationOptions, SaveFields)
        V.save(FieldsExtractionGrid, 'fields.cgns', VisualisationOptions, SaveFields)

    filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_It%d.cgns'%it)
    V.save(t, filename, VisualisationOptions, SaveFields)
    V.save(t, DIRECTORY_OUTPUT + '.cgns', VisualisationOptions, SaveFields)
    for _ in range(3): V.show(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))
    V.show(f"{'||':>57}\r" + '||' + '{:-^53}'.format(' End of VPM computation '))
    for _ in range(3): V.show(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))

    return t

def runVPMTrees(tL = [], tLL = [], tE = [], tH = [], tP = []):
    '''
    Runs one VULCAINS iteration.

    Parameters
    ----------
        tL : Tree
            Lagrangian (particles) field.

        tLL : Tree
            Lifting Lines.

        tE : Tree
            Eulerian (Fast CFD) field.

        tH : Tree
            Hybrid Domain.

        tP : Tree
            Perturbation field
    Returns
    -------
        IterationInfo : :py:class:`dict`
            VULCAINS information on the current iteration.
    '''
    IterationInfo = {}
    V.updateSmagorinskyConstantAndComputeTurbulentViscosity(tL)
    V.computeLagrangianNextTimeStep(tL, tLL, tP)
    IterationInfo.update(V.populationControl(tL, tLL, tH))
    IterationInfo.update(V.shedVorticitySourcesFromLiftingLines(tL, tLL, tP))
    IterationInfo.update(V.computeEulerianNextTimeStep(tL, tE, tH))
    IterationInfo.update(V.shedVorticitySourcesFromHybridDomain(tL, tE, tH))
    IterationInfo.update(V.induceVPMField(tL, tP))
    return IterationInfo

def runVPM(t = []):
    '''
    Runs one VULCAINS iteration.

    Parameters
    ----------
        t : Tree
            Contains the Lagrangian field, Lifting Lines, Eulerian field, Hybrid Domain and
            Perturbation field.
    Returns
    -------
        IterationInfo : :py:class:`dict`
            VULCAINS information on the current iteration.
    '''
    IterationInfo = {}
    V.updateSmagorinskyConstantAndComputeTurbulentViscosity(t)
    V.computeLagrangianNextTimeStep(t)
    IterationInfo.update(V.populationControl(t))
    IterationInfo.update(V.shedVorticitySourcesFromLiftingLines(t))
    IterationInfo.update(V.computeEulerianNextTimeStep(t))
    IterationInfo.update(V.shedVorticitySourcesFromHybridDomain(t))
    IterationInfo.update(V.induceVPMField(t))
    return IterationInfo

def iterateVPM(t = [], SaveFields = [], NumberOfIterations = 1, DIRECTORY_OUTPUT = '',
    VisualisationOptions = {}, SaveImageOptions = {}, SaveFieldsPeriod = 0, SaveImagePeriod = 0,
    SaveVPMPeriod = 0, StdDeviationSample = 100, FieldsExtractionGrid = [], Surface = 0.,
                                                                        mainFunction = runVPMTrees):
    '''
    Loops over the VULCAINS iterations.

    Parameters
    ----------
        t : Tree
            Contains the Lagrangian field, Lifting Lines, Eulerian field, Hybrid Domain and
            Perturbation field.

        SaveFields : list or numpy.ndarray of :py:class:`str`
            Names of the Lagrangian FlowSolution fields to save, as in :py:func:`compute`

        NumberOfIterations : :py:class:`int`
            Number of iterations

        DIRECTORY_OUTPUT : :py:class:`str`
            Saved CGNS folder location.

        VisualisationOptions : :py:class:`dict`
            CPlot visualisation options for the saved files.

        SaveImageOptions : :py:class:`dict`
            CPlot visualisation options for the saved images.

        SaveFieldsPeriod : :py:class:`int`
            Frequency at wich fields are extracted from the simulaton.

        SaveImagePeriod : :py:class:`int`
            Frequency at wich the simulation screenshots are saved.

        SaveVPMPeriod : :py:class:`int`
            Frequency at wich the CGNS files are saved.

        StdDeviationSample : :py:class:`int`
            Number of iterations for the standard deviation computation.

        FieldsExtractionGrid : Tree
            Extraction grids.

        Surface : :py:class:`float`
            Surface of the simulated solid for the computation of the aerodynamic coefficients.

        mainFunction : :py:func:
            Function over which VULCAINS loops.
    Returns
    -------
        t : Tree
            Contains the Lagrangian field, Lifting Lines, Eulerian field, Hybrid Domain and
            Perturbation field.
    '''
    IterationInfo = {'Rel. err. of Velocity': 0, 'Rel. err. of Velocity Gradient': 0,
       'Rel. err. of Vorticity': 0, 'Rel. err. of PSE': 0, 'Rel. err. of Diffusion Velocity': 0}
    TotalTime = J.tic()

    tL = V.getParticlesTree(t)
    tE = V.getEulerianTree(t)
    tLL = V.getLiftingLinesTree(t)
    tH = V.getHybridDomainTree(t)
    tP = V.getPerturbationFieldTree(t)
    Parameters = V.getAllParameters(t)
    Np = V.getParticlesNumber(t, pointer = True)
    it = Parameters['PrivateParameters']['CurrentIteration']
    simuTime = Parameters['PrivateParameters']['Time']
    PSE = V.DiffusionScheme_str2int[Parameters['ModelingParameters']['DiffusionScheme']] < 2
    DVM = V.DiffusionScheme_str2int[Parameters['ModelingParameters']['DiffusionScheme']] == 2
    Freestream = (np.linalg.norm(Parameters['FluidParameters']['VelocityFreestream']) != 0.)
    try: Wing = (I.getValue(I.getNodeFromName(t, 'RPM')) == 0)
    except: Wing = True

    while it[0] < NumberOfIterations:
        SAVE_ALL = J.getSignal('SAVE_ALL')
        SAVE_FIELDS = ((it + 1)%SaveFieldsPeriod == 0 and 0 < it) or J.getSignal('SAVE_FIELDS')
        SAVE_IMAGE  = ((it + 1)%SaveImagePeriod == 0 and 0 < it) or J.getSignal('SAVE_IMAGE')
        SAVE_VPM    = ((it + 1)%SaveVPMPeriod == 0) or J.getSignal('SAVE_VPM')
        CONVERGED   = J.getSignal('CONVERGED')
        if CONVERGED: SAVE_ALL = True
        if SAVE_ALL: SAVE_FIELDS = SAVE_VPM = True
        
        IterationTime = J.tic()
        
        newInfo = mainFunction(tL, tLL, tE, tH, tP)
        # newInfo = mainFunction(t)

        if newInfo: IterationInfo.update(newInfo)
        IterationInfo['Iteration'] = it[0]
        IterationInfo['Percentage'] = it[0]/NumberOfIterations*100.
        IterationInfo['Physical time'] = simuTime[0]
        IterationInfo['Number of particles'] = Np[0]
        IterationInfo['Total iteration time'] = J.tic() - IterationTime
        IterationInfo.update(V.getAerodynamicCoefficientsOnLiftingLine(t, Wings = Wing,
               StdDeviationSample = StdDeviationSample, Freestream = Freestream, Surface = Surface))
        IterationInfo['Total simulation time'] = J.tic() - TotalTime
        printIterationInfo(IterationInfo, PSE = PSE, DVM = DVM, Wings = Wing)

        if (SAVE_FIELDS or SAVE_ALL) and FieldsExtractionGrid:
            extractFields(t, FieldsExtractionGrid, 5300)
            filename = os.path.join(DIRECTORY_OUTPUT, 'fields_It%d.cgns'%it)
            V.save(FieldsExtractionGrid, filename, VisualisationOptions, SaveFields)
            J.createSymbolicLink(filename, 'fields.cgns')

        if SAVE_IMAGE or SAVE_ALL:
            V.setVisualization(t, **VisualisationOptions)
            V.saveImage(t, **SaveImageOptions)

        if SAVE_VPM or SAVE_ALL:
            filename = os.path.join(DIRECTORY_OUTPUT, 'VPM_It%d.cgns'%it)
            V.save(t, filename, VisualisationOptions, SaveFields)
            J.createSymbolicLink(filename,  DIRECTORY_OUTPUT + '.cgns')

        if CONVERGED: break
    
    return t

def initialiseComputation(Parameters = {}, LiftingLines = [], EulerianMesh = [],
                                                                            PerturbationField = []):
    '''
    Initialises all the trees used for the VULCAINS simulation.

    Parameters
    ----------
        Parameters : :py:class:`dict` of :py:class:`dict`
            User-provided VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.Main.compute`

        EulerianMesh : Tree
            like in :py:func:`compute`

        PerturbationField : Tree
            like in :py:func:`compute`

        LiftingLines : Tree
            like in :py:func:`compute`
    Returns
    -------
        t : Tree
            Contains the Lagrangian field, Lifting Lines, Eulerian field, Hybrid Domain and
            Perturbation field.
    '''
    if 'NumberOfThreads' not in Parameters['NumericalParameters']:
        Parameters['NumericalParameters']['NumberOfThreads'] = 'auto'
    
    Parameters['NumericalParameters']['NumberOfThreads'] = V.initialiseThreads(\
                                               Parameters['NumericalParameters']['NumberOfThreads'])
    for field in [f + 'Parameters' for f in ['Fluid', 'Hybrid', 'Modeling', 'Numerical','Private']]:
        if field not in Parameters: Parameters[field] = dict()

    if not 'LiftingLineParameters' in Parameters: Parameters['LiftingLineParameters'] = dict()
    V.checkParameters(Parameters)
    tLL = V.initialiseLiftingLines(LiftingLines, Parameters)
    tE = V.initialiseEulerianDomain(EulerianMesh, Parameters)
    tH = V.generateHybridDomain(tE, Parameters)
    tP = V.initialisePerturbationfield(PerturbationField, Parameters)
    tL = V.initialiseVPM(tE = tE, tH = tH, tLL = tLL, Parameters = Parameters)
    t = C.newPyTree()
    t[2] = [I.getNodeFromName1(t, 'CGNSLibraryVersion')] + I.getBases(tL) + I.getBases(tE) + \
                                                   I.getBases(tH) + I.getBases(tLL) + I.getBases(tP)
    V.induceVPMField(t)
    V.getParameter(t, 'IterationCounter')[0] = Parameters['ModelingParameters']['IntegrationOrder']\
                           *Parameters['NumericalParameters']['FMMParameters']['IterationTuningFMM']
                                                
    return t

def restartComputation(path = 'OUTPUT.cgns', Parameters = {}):
    '''
    Restarts and checks all the trees used for the VULCAINS simulation.

    Parameters
    ----------
        path : Tree or :py:class:`str`
            Restart tree (VPM) or its file path

        Parameters : :py:class:`dict` of :py:class:`dict`
            User-provided VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.Main.compute`
    Returns
    -------
        t : Tree
            Contains the Lagrangian field, Lifting Lines, Eulerian field, Hybrid Domain and
            Perturbation field.
    '''
    if isinstance(path, str): t = V.load(path)
    else: t = path

    if 'NumericalParameters' not in Parameters: Parameters['NumericalParameters'] = dict()
    if 'NumberOfThreads' in Parameters['NumericalParameters']:
        OMP_NUM_THREADS = Parameters['NumericalParameters']['NumberOfThreads']
    else: OMP_NUM_THREADS = V.getParameter(t, 'NumberOfThreads')
    
    Parameters['NumericalParameters']['NumberOfThreads'] = V.initialiseThreads(OMP_NUM_THREADS)
    V.checkTrees(t, Parameters)
    tL, tLL, tE, tH, tP = V.getTrees([t], ['Particles', 'LiftingLines', 'Eulerian', 'Hybrid',
                                                                                    'Perturbation'])
    if tP:
        NumberOfNodes = V.getParameter(tL, 'NumberOfNodes')
        V.tP_Capsule[0] = V.build_perturbation_velocity_capsule(tP, NumberOfNodes)
    
    IterationCounter = V.getParameter(tL, 'IterationCounter')
    IterationCounter[0] -= 1
    I.getNodeFromName(tL, 'Enstrophy')[1][:] = I.getValue(I.getNodeFromName(tL, 'EnstrophyM1'))
    V.induceVPMField(tL, tP)
    V.computeFastMetrics(tE)
    t = C.newPyTree()
    t[2] = [I.getNodeFromName1(t, 'CGNSLibraryVersion')] + I.getBases(tL) + I.getBases(tE) + \
                                                   I.getBases(tH) + I.getBases(tLL) + I.getBases(tP)
    return t

def computeFreeVortex(Parameters = {}, VortexParameters = {}, NumberOfIterations = 10000,
    SaveVPMPeriod = 10, DIRECTORY_OUTPUT = 'OUTPUT', SaveFields = ['all']):
    '''
    Initialises all the trees used for the VULCAINS simulation of unbounded vortex rings.

    Parameters
    ----------
        Parameters : :py:class:`dict` of :py:class:`dict`
            User-provided VULCAINS parameters as established in
            :py:func:`~MOLA.VULCAINS.Main.compute`

        VortexParameters : :py:class:`dict`
            Describe the vortex to simulate.

                Intensity : :py:class:`float`
                    Intensity of the vortex :math:``m^2.s^{-1}``.

                MinimumVorticityFraction : :py:class:`float`
                    Keep adding layers to discretise the vortex until the lowest vorticity of the
                    layer is just above ``MinimumVorticityFraction`` times the vorticity at the
                    center of the vortex.

                RingRadius : :py:class:`float`
                    Radius of the vortex ring (if any).

                InitialSpacing : :py:class:`float`
                    Initial center-to-center distance between the two coaxial vortex rings.

                Length : :py:class:`float`
                    Length of the vortex tube (if any).

                NumberLayers : :py:class:`int`
                    Gives the number of layers used to discretise the vortex.


        NumberOfIterations : :py:class:`int`
            same as in :py:func:`compute`

        DIRECTORY_OUTPUT : :py:class:`str`
            same as in :py:func:`compute`

        SaveVPMPeriod : :py:class:`int`
            same as in :py:func:`compute`

        SaveFields : :py:class:`list` or numpy.ndarray of :py:class:`str`
            same as in :py:func:`compute`

    Returns
    -------
        t : Tree
            Contains the Lagrangian field, Lifting Lines, Eulerian field, Hybrid Domain and
            Perturbation field.
    '''
    if 'NumberOfThreads' not in Parameters['NumericalParameters']:
        Parameters['NumericalParameters']['NumberOfThreads'] = 'auto'
    
    Parameters['NumericalParameters']['NumberOfThreads'] = V.initialiseThreads(\
                                               Parameters['NumericalParameters']['NumberOfThreads'])
    for field in [f + 'Parameters' for f in ['Fluid', 'Hybrid', 'Modeling', 'Numerical','Private']]:
        if field not in Parameters: Parameters[field] = dict()

    V.checkParameters(Parameters)
    t = V.buildEmptyVPMTree()
    if 'Length' in VortexParameters:
        createLambOseenVortexBlob(t, Parameters, VortexParameters)
    else:
        createLambOseenVortexRing(t, Parameters, VortexParameters)
        if 'InitialSpacing' in VortexParameters:
            Particles = V.getParticles(t)
            Np = V.getParticlesNumber(Particles)
            V.extend(Particles, Np)
            ax, ay, az, s, c, nu = J.getVars(Particles, V.vectorise('Alpha') + \
                                                ['Sigma', 'Cvisq', 'Nu', ])
            ax[Np:], ay[Np:], az[Np:], s[Np:], c[Np:], nu[Np:] = ax[:Np], \
                                               ay[:Np], az[:Np], s[:Np], c[:Np], nu[:Np]
            x, y, z = J.getxyz(Particles)
            x[Np:], y[Np:], z[Np:] = x[:Np], y[:Np], z[:Np] + VortexParameters['InitialSpacing']

    
    Particles = V.getParticles(t)
    for field in ['Fluid', 'Hybrid', 'Modeling', 'Numerical', 'Private']:
        name = field + 'Parameters'
        if name in Parameters and Parameters[name]:
            J.set(Particles, '.' + field + '#Parameters', **Parameters[name])
            I._sortByName(I.getNodeFromName1(Particles, '.' + field + '#Parameters'))

    V.induceVPMField(t)
    V.getParameter(t, 'IterationCounter')[0] = Parameters['ModelingParameters']['IntegrationOrder']\
                           *Parameters['NumericalParameters']['FMMParameters']['IterationTuningFMM']
    V.compute(RestartPath = t, NumberOfIterations = NumberOfIterations,
        DIRECTORY_OUTPUT = DIRECTORY_OUTPUT, SaveFields = SaveFields,
        VisualisationOptions = {'addLiftingLineSurfaces':False}, SaveVPMPeriod = SaveVPMPeriod)

def extractFields(Targets = [], tL = [], tE = [], tH = [], FarFieldPolynomialOrder = 12,
    NearFieldOverlapingFactor = 4, NbOfParticlesForPrecisionEvaluation = 1000):
    '''
    Extract fields from a VULCAINS simulation onto given grids, surfaces,
    points... If a Hybrid case is given, the solution is interpolated on the
    overlapping regions between tL and tE.

    Parameters
    ----------
        Targets : Tree
            Probes of the VPM field.

        tL : Tree
            Lagrangian field.

        tE : Tree
            Eulerian field.

        tH : Tree
            Hybrid Domain.

        NbOfParticlesForPrecisionEvaluation : :py:class:`int`
            Number of nodes where the solution approximated by the FMM is checked.

        FarFieldPolynomialOrder : int
            must be :math:`\in [4, 12]`. It is the order of the polynomial which
            approximates the long distance particle interactions by the FMM.
            The higher it is, the more accurate and the more costly the extraction
            becomes.

        NearFieldOverlapingFactor : :py:class:`float`
            must be :math:`[1, +\infty)`. The particle interactions are
            approximated by the FMM as soon as two clusters of particles are
            separated by at least **NearFieldOverlapingFactor** the size
            of the particles in the cluster.
            The higher it is, the more accurate and the more costly the extraction
            becomes.
    '''
    #initialise targets
    if not Targets: return
    _tL, _tE, _tH = V.getTrees([tL, tE, tH], ['Particles', 'Eulerian', 'Hybrid'])
    if not _tL: return
    V.show(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))
    V.show(f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Extract VPM Solution '))

    newFieldNames = V.vectorise('Velocity') + V.vectorise('Vorticity') + V.vectorise('gradxVelocity') + \
                    V.vectorise('gradyVelocity') + V.vectorise('gradzVelocity') + V.vectorise('rotU') + \
                    ['VelocityMagnitude', 'VorticityMagnitude', 'rotU', 'divVelocity', 'QCriterion']
    for fn in newFieldNames: C._initVars(Targets, fn, 0.)

    rho0, T0 = V.getParameters(_tL, ['Density', 'Temperature'])
    C._initVars(Targets, 'Density', rho0[0])
    C._initVars(Targets, 'Temperature', T0[0])
    TargetsZones = I.getZones(Targets)
    NbNodes = [0]
    for z in TargetsZones: NbNodes += [C.getNPts(z) + NbNodes[-1]]

    V.show(f"{'||':>57}\r" + '|| ' + '{:32}'.format('Number of targets') + ': ' +
                                                                       '{:.4g}'.format(NbNodes[-1]))
    V.show(f"{'||':>57}\r" + '|| ' + '{:32}'.format('Number of VPM particles') + ': ' +
                                                           '{:.4g}'.format(V.getParticlesNumber(_tL)))
    #transform it in particles
    LagrangianGrid = V.buildEmptyVPMTree(NbNodes[-1], newFieldNames)
    xyzLagrangian = J.getxyz(I.getZones(LagrangianGrid)[0])
    for i, Zone in enumerate(TargetsZones):
        xyzZone = J.getxyz(Zone)
        for coordReceiver, coordDonor in zip(xyzLagrangian, xyzZone):
            coordReceiver[NbNodes[i]:NbNodes[i+1]] = coordDonor.ravel(order = 'F')

    #compute VPM solution
    # updateCFDSources(_tL, _tE, _tH)#in that order
    # updateBEMSources(_tL, _tE, _tH)
    fmm_info = V.extract_vpm_fields(LagrangianGrid, _tL,
                   np.int32(NbOfParticlesForPrecisionEvaluation), np.int32(FarFieldPolynomialOrder),
                                                              np.float64(NearFieldOverlapingFactor))
    msg = f"{'||':>57}\r" + '||' + '{:-^53}'.format(' FMM ') + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Rel. err. of Velocity') + ': ' + \
                                                                   '{:e}'.format(fmm_info[0]) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Rel. err. of Velocity Gradient') + ': ' + \
                                                                   '{:e}'.format(fmm_info[2]) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Rel. err. of Vorticity') + ': ' + \
                                                                   '{:e}'.format(fmm_info[1]) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('FMM Computation time') + ': ' + \
                                                               '{:.2f}'.format(fmm_info[3]) + ' s'
    V.show(msg)

    #extract VPM solution to targets
    Donors = J.getVars(I.getZones(LagrangianGrid)[0], newFieldNames)
    for i, Zone in enumerate(TargetsZones):
        Receivers = J.getVars(Zone, newFieldNames)
        for Receivers, Donor in zip(Receivers, Donors):
            Receivers.ravel(order = 'F')[:] = Donor[NbNodes[i]:NbNodes[i+1]]

    if not _tE:
        V.show(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))
        return
    V.show(f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Extract Eulerian Solution '))

    V.computeFastMetrics(_tE)
    V.computeEulerianVelocityGradients(_tE)
    t, tc = V.getEulerianBases(_tE)
    tref = I.copyRef(t)
    #get rid of unnecessary flow fields
    AllFields = [Field[0] for Field in I.getNodeFromName(tref, 'FlowSolution#Centers')[2]]
    InterpFieldNames = ['Density', 'Temperature'] + V.vectorise('Velocity')
    for dir in 'xyz': InterpFieldNames += V.vectorise('grad' + dir + 'Velocity')
    for Field in AllFields:
        if Field not in InterpFieldNames:
            I._rmNodesByName(tref, Field)
    #interpolate the Eulerian solution
    EulerianGrid = I.copyTree(P.extractMesh(tref, Targets, order=2, extrapOrder=0, constraint=0.))
    #mix both Eulerian and Lagrangian solutions
    vars = V.vectorise('Vorticity') + V.vectorise('rotU') + ['VelocityMagnitude', 'VorticityMagnitude',\
                                                                'rotU', 'divVelocity', 'QCriterion']
    InterpFieldNames += vars
    formulas = ['{gradyVelocityZ} - {gradzVelocityY}',
                '{gradzVelocityX} - {gradxVelocityZ}',
                '{gradxVelocityY} - {gradyVelocityX}',
                '{VorticityX}', '{VorticityY}', '{VorticityZ}',
                '({VelocityX}**2 + {VelocityY}**2 + {VelocityZ}**2)**0.5',
                '({VorticityX}**2 + {VorticityY}**2 + {VorticityZ}**2)**0.5',
                '{VorticityMagnitude}',
                '{gradxVelocityX} + {gradyVelocityY} + {gradzVelocityZ}',
                '   {gradxVelocityX}*{gradyVelocityY} + {gradxVelocityX}*{gradzVelocityZ}'\
                ' + {gradyVelocityY}*{gradzVelocityZ} - ({gradxVelocityY}*{gradyVelocityX}'\
                ' + {gradxVelocityZ}*{gradzVelocityX} + {gradyVelocityZ}*{gradzVelocityY})']
    for var, formula in zip(vars, formulas):
        C._initVars(EulerianGrid, '{' + var + '}=' + formula)

    V.show(f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Interpolate Hybrid Solution '))
    #compute TurbulentDistance & get interpolation coeffs
    import Dist2Walls.PyTree as Dist2Walls
    Boundary = Dist2Walls.distance2Walls(EulerianGrid, V.getHybridDomainOuterInterface(_tH),
                                                                          signed = 1, loc = 'nodes')
    dMax = V.getMeanTurbulentDistance(T.subzone(I.rmNodesByName(t, 'cellN'), (1, 1, -1), \
                        (-1, -1, -1))) - I.getValue(I.getNodeFromName3(t, 'OuterInterfaceDistance'))
    #create the interpolation weight got the Eulerian and Lagrangian solutions
    Nh = 0
    
    for Zone, BC in zip(I.getZones(EulerianGrid), I.getZones(Boundary)):
        Distance0 = I.getNodeFromName(BC, 'TurbulentDistance')
        if Distance0:
            Distance = Distance0[1]
            InnerCells = Distance < 1e-12
            OuterCells = dMax < Distance
            Distance /= dMax + 1e-12
            Distance[InnerCells] = 0.
            Distance[OuterCells] = 1.
            Nh += np.sum(np.logical_not(InnerCells + OuterCells))
            J.invokeFields(Zone, ['InterpolationWeight'])[0][:] = np.square(np.cos(\
                                                                        np.pi/2.*Distance))
            # Distance0[1] = Distance

    if Nh:
        V.show(f"{'||':>57}\r" + '|| ' + '{:32}'.format('Number of hybrids') + ': ' + \
                                                                                '{:.4g}'.format(Nh))
        for zone, zoneEulerian in zip(I.getZones(Targets), I.getZones(EulerianGrid)):
            fieldsL = J.getVars(zone, InterpFieldNames)
            fieldsE = J.getVars(zoneEulerian, InterpFieldNames + ['InterpolationWeight'])
            weightE = fieldsE.pop()
            weightE[weightE < 0.] = 0.
            weightE[1. < weightE] = 1.
            weightL = 1. - weightE
            for fieldL, fieldE in zip(fieldsL, fieldsE): fieldL[:] = fieldL*weightL + fieldE*weightE

    V.show(f"{'||':>57}\r" + '||' + '{:=^53}'.format(''))

####################################################################################################
####################################################################################################
######################################### IO/Visualisation #########################################
####################################################################################################
####################################################################################################
def printIterationInfo(IterationInfo = {}, PSE = False, DVM = False, Wings = False):
    '''
    Prints the current iteration information.

    Parameters
    ----------
        IterationInfo : :py:class:`dict`
            VPM solver information on the current iteration.

        PSE : :py:class:`bool`
            States whether the PSE was used.

        DVM : :py:class:`bool`
            States whether the DVM was used.

        Wings : :py:class:`bool`
            States whether the Lifting Line(s) Wings were used.
    '''
    msg = f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Iteration ' + \
                                                '{:d}'.format(IterationInfo['Iteration']) + ' (' + \
                                        '{:.1f}'.format(IterationInfo['Percentage']) + '%) ') + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Physical time') + ': ' + \
                                       '{:.5f}'.format(IterationInfo['Physical time']) + ' s' + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of particles') + ': ' + \
                                          '{:d}'.format(IterationInfo['Number of particles']) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Total iteration time') + ': ' + \
                                '{:.2f}'.format(IterationInfo['Total iteration time']) + ' s' + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Total simulation time') + ': ' + \
                               '{:.1f}'.format(IterationInfo['Total simulation time']) + ' s' + '\n'
    msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Loads ') + '\n'
    if (Wings and 'Lift' in IterationInfo) or (not Wings and 'Thrust' in IterationInfo):
        if (Wings):
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Lift') + ': ' + \
                                                '{:.4g}'.format(IterationInfo['Lift']) + ' N' + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Lift Standard Deviation') + ': ' + \
                             '{:.2f}'.format(IterationInfo['Lift Standard Deviation']) + ' %' + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Drag') + ': ' + \
                                                '{:.4g}'.format(IterationInfo['Drag']) + ' N' + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Drag Standard Deviation') + ': ' + \
                             '{:.2f}'.format(IterationInfo['Drag Standard Deviation']) + ' %' + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('cL') + ': ' + \
                                                         '{:.4f}'.format(IterationInfo['cL']) + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('cD') + ': ' + \
                                                         '{:.5f}'.format(IterationInfo['cD']) + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('f') + ': ' + \
                                                          '{:.4f}'.format(IterationInfo['f']) + '\n'
        else:
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Thrust') + ': ' + \
                                              '{:.5g}'.format(IterationInfo['Thrust']) + ' N' + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Thrust Standard Deviation') + ': ' + \
                           '{:.2f}'.format(IterationInfo['Thrust Standard Deviation']) + ' %' + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Power') + ': ' + \
                                               '{:.5g}'.format(IterationInfo['Power']) + ' W' + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Power Standard Deviation') + ': ' + \
                            '{:.2f}'.format(IterationInfo['Power Standard Deviation']) + ' %' + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('cT') + ': ' + \
                                                         '{:.5f}'.format(IterationInfo['cT']) + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Cp') + ': ' + \
                                                         '{:.5f}'.format(IterationInfo['cP']) + '\n'
            msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Eff') + ': ' + \
                                                        '{:.5f}'.format(IterationInfo['Eff']) + '\n'
    msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Population Control ') + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of resized particles') + ': ' + \
                                  '{:d}'.format(IterationInfo['Number of resized particles']) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of particles beyond cutoff') + ': ' + \
                   '{:d}'.format(IterationInfo['Number of particles beyond cutoff']) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of depleted particles') + ': ' + \
                                 '{:d}'.format(IterationInfo['Number of depleted particles']) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of split particles') + ': ' + \
                                    '{:d}'.format(IterationInfo['Number of split particles']) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of merged particles') + ': ' + \
                                   '{:d}'.format(IterationInfo['Number of merged particles']) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Control Computation time') + ': ' + \
                              '{:.2f}'.format(IterationInfo['Population Control time']) + ' s (' + \
                                          '{:.1f}'.format(IterationInfo['Population Control time']/\
                                                IterationInfo['Total iteration time']*100.) + '%)\n'
    if 'Lifting Line time' in IterationInfo:
        msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Lifting Line ') + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Circulation error') + ': ' + \
                                          '{:.5e}'.format(IterationInfo['Circulation error']) + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of sub-iterations') + ': ' + \
                                '{:d}'.format(IterationInfo['Number of sub-iterations (LL)']) + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of shed particles') + ': ' + \
                                     '{:d}'.format(IterationInfo['Number of shed particles']) + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Lifting Line Computation time') + ': ' + \
                                    '{:.2f}'.format(IterationInfo['Lifting Line time']) + ' s (' + \
                                            '{:.1f}'.format(IterationInfo['Lifting Line time']/\
                                                IterationInfo['Total iteration time']*100.) + '%)\n'

    if 'Eulerian time' in IterationInfo:
        msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Hybrid Solver ') + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of shed particles') + ': ' + \
                            '{:d}'.format(IterationInfo['Number of shed particles Eulerian']) + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Number of iterations') + ': ' + \
                                         '{:d}'.format(IterationInfo['Number of iterations']) + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Vorticity Rel. err.') + ': ' + \
                              '{:e}'.format(IterationInfo['Rel. err. of Vorticity Eulerian']) + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Iterative methode time') + ': ' + \
                                     '{:2f}'.format(IterationInfo['Iterative methode time']) + 's\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Eulerian time') + ': ' + \
                                        '{:.2f}'.format(IterationInfo['Eulerian time']) + ' s (' + \
                                                    '{:.1f}'.format(IterationInfo['Eulerian time']/\
                                                IterationInfo['Total iteration time']*100.) + '%)\n'
        
    msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' FMM ') + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Rel. err. of Velocity') + ': ' + \
                                        '{:e}'.format(IterationInfo['Rel. err. of Velocity']) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Rel. err. of Velocity Gradient') + ': ' + \
                               '{:e}'.format(IterationInfo['Rel. err. of Velocity Gradient']) + '\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Rel. err. of Vorticity') + ': ' + \
                                       '{:e}'.format(IterationInfo['Rel. err. of Vorticity']) + '\n'
    if PSE: msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Rel. err. of PSE') + ': ' + \
                                             '{:e}'.format(IterationInfo['Rel. err. of PSE']) + '\n'
    if DVM:
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Rel. err. of PSE') + ': ' + \
                                             '{:e}'.format(IterationInfo['Rel. err. of PSE']) + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Rel. err. of Diffusion Velocity') + ': ' +\
                               '{:e}'.format(IterationInfo['Rel. err. of Diffusion Velocity']) +'\n'
    msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('FMM Computation time') + ': ' + \
                                             '{:.2f}'.format(IterationInfo['FMM time']) + ' s (' + \
                                                         '{:.1f}'.format(IterationInfo['FMM time']/\
                                                IterationInfo['Total iteration time']*100.) + '%)\n'
    if "Perturbation time" in IterationInfo:
        msg += f"{'||':>57}\r" + '||' + '{:-^53}'.format(' Perturbation Field ') + '\n'
        msg += f"{'||':>57}\r" + '|| ' + '{:34}'.format('Interpolation time') + ': ' + \
                                    '{:.2f}'.format(IterationInfo['Perturbation time']) + ' s (' + \
                                                '{:.1f}'.format(IterationInfo['Perturbation time']/\
                                                IterationInfo['Total iteration time']*100.) + '%)\n'
    msg += f"{'||':>57}\r" + '||' + '{:=^53}'.format('')
    V.show(msg)

