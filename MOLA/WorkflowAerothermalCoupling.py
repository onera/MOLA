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
MOLA - WorkflowAerothermalCoupling.py

WORKFLOW AEROTHERMAL COUPLING

Collection of functions designed for Workflow Aerothermal Coupling
This workflow may be used to run coupled simulations between elsA and Zset.
The communication between solvers is performed with CWIPI.
CWIPI is not in the default MOLA environment and must be sourced before using
this workflow.

File history:
07/04/2022 - T. Bontemps - Creation
'''

import MOLA

if not MOLA.__ONLY_DOC__:
    import sys
    import os
    import numpy as np
    import copy

    import Converter.PyTree    as C
    import Converter.Internal  as I

from . import InternalShortcuts as J
from . import Preprocess        as PRE
from . import WorkflowCompressor as WC
from . import JobManager as JM

def checkDependencies():
    '''
    Make a series of functional tests in order to determine if the user
    environment is correctly set for using the Workflow Aerothermal Coupling
    '''
    from . import JobManager as JM
    JM.checkDependencies()

    print('Checking ETC...')
    if WC.ETC is None:
        MSG = 'Fail to import ETC module: Some functions of {} are unavailable'.format(__name__)
        print(J.FAIL + MSG + J.ENDC)
    else:
        print(J.GREEN+'ETC module is available'+J.ENDC)

    print('Checking MOLA.ParametrizeChannelHeight...')
    if  WC.ParamHeight is None:
        MSG = 'Fail to import MOLA.ParametrizeChannelHeight module: Some functions of {} are unavailable'.format(__name__)
        print(J.FAIL + MSG + J.ENDC)
    else:
        print(J.GREEN+'MOLA.ParametrizeChannelHeight module is available'+J.ENDC)

    print('Checking CWIPI...')
    try:
        import CWIPI.CGNS as C2
        print(J.GREEN+'CWIPI.CGNS module is available'+J.ENDC)
    except ImportError:
        MSG = 'Fail to import CWIPI.CGNS module: Some functions of {} are unavailable'.format(__name__)
        print(J.FAIL + MSG + J.ENDC)

    print('\nVERIFICATIONS TERMINATED')


def prepareMesh4ElsA(mesh, kwargs):
    '''
    Exactly like :py:func:`MOLA.WorkflowCompressor.prepareMesh4ElsA`
    '''
    return WC.prepareMesh4ElsA(mesh, **kwargs)

def prepareMainCGNS4ElsA(mesh='mesh.cgns', ReferenceValuesParams={},
        NumericalParams={}, OverrideSolverKeys={},
        TurboConfiguration={}, Extractions=[], BoundaryConditions=[],
        BodyForceInputData=[], writeOutputFields=True, bladeFamilyNames=['Blade'],
        Initialization={'method':'uniform'}, JobInformation={}, SubmitJob=False,
        FULL_CGNS_MODE=True, templates=dict(), secondOrderRestart=False):
    '''
    This is mainly a function similar to :func:`MOLA.WorkflowCompressor.prepareMainCGNS4ElsA`
    but adapted to aerothermal simulations with CWIPI coupling.

    Parameters
    ----------

        mesh : :py:class:`str` or PyTree
            if the input is a :py:class:`str`, then such string specifies the
            path to file (usually named ``mesh.cgns``) where the result of
            function :py:func:`prepareMesh4ElsA` has been writen. Otherwise,
            **mesh** can directly be the PyTree resulting from :func:`prepareMesh4ElsA`

        ReferenceValuesParams : dict
            Python dictionary containing the
            Reference Values and other relevant data of the specific case to be
            run using elsA. For information on acceptable values, please
            see the documentation of function :func:`computeReferenceValues`.

            .. note:: internally, this dictionary is passed as *kwargs* as follows:

                >>> MOLA.Preprocess.computeReferenceValues(arg, **ReferenceValuesParams)

        NumericalParams : dict
            dictionary containing the numerical
            settings for elsA. For information on acceptable values, please see
            the documentation of function :func:`MOLA.Preprocess.getElsAkeysNumerics`

            .. note:: internally, this dictionary is passed as *kwargs* as follows:

                >>> MOLA.Preprocess.getElsAkeysNumerics(arg, **NumericalParams)

        OverrideSolverKeys : :py:class:`dict` of maximum 3 :py:class:`dict`
            exactly the same as in :py:func:`MOLA.Preprocess.prepareMainCGNS4ElsA`

        TurboConfiguration : dict
            Dictionary concerning the compressor properties.
            For details, refer to documentation of :func:`getTurboConfiguration`

        Extractions : :py:class:`list` of :py:class:`dict`
            List of extractions to perform during the simulation. See
            documentation of :func:`MOLA.Preprocess.prepareMainCGNS4ElsA`

        BoundaryConditions : :py:class:`list` of :py:class:`dict`
            List of boundary conditions to set on the given mesh.
            For details, refer to documentation of :func:`setBoundaryConditions`

        BodyForceInputData : :py:class:`list` of :py:class:`dict`

        writeOutputFields : bool
            if :py:obj:`True`, write initialized fields overriding
            a possibly existing ``OUTPUT/fields.cgns`` file. If :py:obj:`False`, no
            ``OUTPUT/fields.cgns`` file is writen, but in this case the user must
            provide a compatible ``OUTPUT/fields.cgns`` file to elsA (for example,
            using a previous computation result).

        bladeFamilyNames : :py:class:`list` of :py:class:`str`
            list of patterns to find families related to blades.

        Initialization : dict
            dictionary defining the type of initialization, using the key
            **method**. See documentation of :func:`MOLA.Preprocess.initializeFlowSolution`

        JobInformation : dict
            Dictionary containing information to update the job file. For
            information on acceptable values, please see the documentation of
            function :func:`MOLA.JobManager.updateJobFile`

        SubmitJob : bool
            if :py:obj:`True`, submit the SLURM job based on information contained
            in **JobInformation**

        FULL_CGNS_MODE : bool
            if :py:obj:`True`, put all elsA keys in a node ``.Solver#Compute``
            to run in full CGNS mode.

        templates : dict
            Main files to copy for the workflow. 
            By default, it is filled with the following values:

            .. code-block::python

                templates = dict(
                    job_template = '$MOLA/TEMPLATES/job_template.sh',
                    compute = '$MOLA/TEMPLATES/<WORKFLOW>/compute.py',
                    coprocess = '$MOLA/TEMPLATES/<WORKFLOW>/coprocess.py',
                    otherWorkflowFiles = ['monitor_perfos.py'],
                )
        
        secondOrderRestart : bool
            If :py:obj:`True`, and if NumericalParams['time_algo'] is 'gear' or 'DualTimeStep' 
            (second order time integration schemes), prepare a second order restart, and allow 
            the automatic restart of such a case. By default, the value is :py:obj:`False`.

            .. important:: 
            
                This behavior works only if elsA reaches the final iteration given by ``niter``.
                If the simulation stops because of the time limit or because all convergence criteria
                have been reached, then the restart will be done at the first order, without raising an error.

    Returns
    -------

        files : None
            A number of files are written:

            * ``main.cgns``
                main CGNS file to be read directly by elsA

            * ``OUTPUT/fields.cgns``
                file containing the initial fields (if ``writeOutputFields=True``)

            * ``setup.py``
                ultra-light file containing all relevant info of the simulation
    '''
    toc = J.tic()
    def addFieldExtraction(fieldname):
        try:
            FieldsExtr = ReferenceValuesParams['FieldsAdditionalExtractions']
            if fieldname not in FieldsExtr.split():
                FieldsExtr += ' '+fieldname
        except:
            ReferenceValuesParams['FieldsAdditionalExtractions'] = fieldname

    if isinstance(mesh,str):
        t = C.convertFile2PyTree(mesh)
    elif I.isTopTree(mesh):
        t = mesh
    else:
        raise ValueError('parameter mesh must be either a filename or a PyTree')

    hasBCOverlap = True if C.extractBCOfType(t, 'BCOverlap') else False


    if hasBCOverlap: addFieldExtraction('ChimeraCellType')
    if BodyForceInputData: addFieldExtraction('Temperature')

    IsUnstructured = PRE.hasAnyUnstructuredZones(t)

    TurboConfiguration = WC.getTurboConfiguration(t, **TurboConfiguration)
    FluidProperties = PRE.computeFluidProperties()
    if not 'Surface' in ReferenceValuesParams:
        ReferenceValuesParams['Surface'] = WC.getReferenceSurface(t, BoundaryConditions, TurboConfiguration)

    if 'PeriodicTranslation' in TurboConfiguration:
        MainDirection = np.array([1,0,0]) # Strong assumption here
        YawAxis = np.array(TurboConfiguration['PeriodicTranslation'])
        YawAxis /= np.sqrt(np.sum(YawAxis**2))
        PitchAxis = np.cross(YawAxis, MainDirection)
        ReferenceValuesParams.update(dict(PitchAxis=PitchAxis, YawAxis=YawAxis))

    ReferenceValues = WC.computeReferenceValues(FluidProperties, **ReferenceValuesParams)
    PRE.appendAdditionalFieldExtractions(ReferenceValues, Extractions)

    if I.getNodeFromName(t, 'proc'):
        JobInformation['NumberOfProcessors'] = int(max(PRE.getProc(t))+1)
        Splitter = None
    else:
        Splitter = 'PyPart'

    elsAkeysCFD      = PRE.getElsAkeysCFD(nomatch_linem_tol=1e-6, unstructured=IsUnstructured)
    elsAkeysModel    = PRE.getElsAkeysModel(FluidProperties, ReferenceValues, unstructured=IsUnstructured)
    if BodyForceInputData: 
        NumericalParams['useBodyForce'] = True
        PRE.tag_zones_with_sourceterm(t)
    if not 'NumericalScheme' in NumericalParams:
        NumericalParams['NumericalScheme'] = 'roe'
    elsAkeysNumerics = PRE.getElsAkeysNumerics(ReferenceValues,
                            unstructured=IsUnstructured, **NumericalParams)

    if secondOrderRestart:
        secondOrderRestart = True if elsAkeysNumerics['time_algo'] in ['gear', 'dts'] else False
    PRE.initializeFlowSolution(t, Initialization, ReferenceValues, secondOrderRestart=secondOrderRestart)

    if not 'PeriodicTranslation' in TurboConfiguration and \
        any([rowParams['NumberOfBladesSimulated'] > rowParams['NumberOfBladesInInitialMesh'] \
            for rowParams in TurboConfiguration['Rows'].values()]):
        t = WC.duplicateFlowSolution(t, TurboConfiguration)

    WC.setMotionForRowsFamilies(t, TurboConfiguration)
    WC.setBoundaryConditions(t, BoundaryConditions, TurboConfiguration,
                            FluidProperties, ReferenceValues,
                            bladeFamilyNames=bladeFamilyNames)

    WC.computeFluxCoefByRow(t, ReferenceValues, TurboConfiguration)

    if not 'PeriodicTranslation' in TurboConfiguration:
        WC.addMonitoredRowsInExtractions(Extractions, TurboConfiguration)

    allowed_override_objects = ['cfdpb','numerics','model']
    for v in OverrideSolverKeys:
        if v == 'cfdpb':
            elsAkeysCFD.update(OverrideSolverKeys[v])
        elif v == 'numerics':
            elsAkeysNumerics.update(OverrideSolverKeys[v])
        elif v == 'model':
            elsAkeysModel.update(OverrideSolverKeys[v])
        else:
            raise AttributeError('OverrideSolverKeys "%s" must be one of %s'%(v,
                                                str(allowed_override_objects)))

    AllSetupDicts = dict(Workflow='AerothermalCoupling',
                        Splitter=Splitter,
                        JobInformation=JobInformation,
                        TurboConfiguration=TurboConfiguration,
                        FluidProperties=FluidProperties,
                        ReferenceValues=ReferenceValues,
                        elsAkeysCFD=elsAkeysCFD,
                        elsAkeysModel=elsAkeysModel,
                        elsAkeysNumerics=elsAkeysNumerics,
                        Extractions=Extractions)

    if BodyForceInputData: AllSetupDicts['BodyForceInputData'] = BodyForceInputData

    
    ReferenceValues['BCExtractions']['BCWallViscousIsothermal'] = \
        ReferenceValues['BCExtractions']['BCWall'] \
        + ['tsta', 'normalheatflux', 'thrm_cndy_lam',
           'hpar', 'ro', 'visclam', 'viscrapp']

    PRE.addTrigger(t)

    is_unsteady = AllSetupDicts['elsAkeysNumerics']['time_algo'] != 'steady'
    avg_requested = AllSetupDicts['ReferenceValues']['CoprocessOptions']['FirstIterationForFieldsAveraging'] is not None

    if is_unsteady and not avg_requested:
        msg =('WARNING: You are setting an unsteady simulation, but no field averaging\n'
              'will be done since CoprocessOptions key "FirstIterationForFieldsAveraging"\n'
              'is set to None. If you want fields average extraction, please set a finite\n'
              'positive value to "FirstIterationForFieldsAveraging" and relaunch preprocess')
        print(J.WARN+msg+J.ENDC)

    PRE.addExtractions(t, AllSetupDicts['ReferenceValues'],
                      AllSetupDicts['elsAkeysModel'],
                      extractCoords=False,
                      BCExtractions=ReferenceValues['BCExtractions'],
                      add_time_average= is_unsteady and avg_requested,
                      secondOrderRestart=secondOrderRestart)


    PRE.addReferenceState(t, AllSetupDicts['FluidProperties'],
                         AllSetupDicts['ReferenceValues'])
    dim = int(AllSetupDicts['elsAkeysCFD']['config'][0])
    PRE.addGoverningEquations(t, dim=dim)
    AllSetupDicts['ReferenceValues']['NumberOfProcessors'] = int(max(PRE.getProc(t))+1)
    PRE.writeSetup(AllSetupDicts)

    if FULL_CGNS_MODE:
        PRE.addElsAKeys2CGNS(t, [AllSetupDicts['elsAkeysCFD'],
                                 AllSetupDicts['elsAkeysModel'],
                                 AllSetupDicts['elsAkeysNumerics']])

    addExchangeSurfaces(t, AllSetupDicts['ReferenceValues']['CoprocessOptions']['CoupledSurfaces'])
    renameElementsNodesInElsAHybrid(t)

    PRE.saveMainCGNSwithLinkToOutputFields(t,writeOutputFields=writeOutputFields)

    if not Splitter:
        print('REMEMBER : configuration shall be run using %s%d%s procs'%(J.CYAN,
                                                   JobInformation['NumberOfProcessors'],J.ENDC))
    else:
        print('REMEMBER : configuration shall be run using %s'%(J.CYAN + \
            Splitter + J.ENDC))

    templates.setdefault('otherWorkflowFiles', [])
    if 'monitor_perfos.py' not in templates['otherWorkflowFiles']:
        templates['otherWorkflowFiles'].append('monitor_perfos.py')
    JM.getTemplates('AerothermalCoupling', templates, JobInformation=JobInformation)
    if 'DIRECTORY_WORK' in JobInformation:
        PRE.sendSimulationFiles(JobInformation['DIRECTORY_WORK'], overrideFields=writeOutputFields)

    for i in range(SubmitJob):
        singleton = False if i==0 else True
        JM.submitJob(JobInformation['DIRECTORY_WORK'], singleton=singleton)

    J.printElapsedTime('prepareMainCGNS4ElsA took ', toc)


def addExchangeSurfaces(t, coupledSurfaces, couplingScript='coprocess.py'):
    '''
    Modify in-place **t** to prepare surfaces coupled with CWIPI.

    Parameters
    ----------

        t : PyTree
            input tree

        coupledSurfaces : list
            List of family names of surfaces coupled with CWIPI

        couplingScript : str
            Name of the script used for CWIPI coupling.
    '''

    # Create template for ExchangeSurface node
    ExchangeSurfaceTemplate = I.newFamily(name='ExchangeSurface')
    n = I.createChild(ExchangeSurfaceTemplate, 'DataInput', 'UserDefinedData_t')
    I.createChild(n, 'Density', 'DataArray_t')
    n = I.createChild(ExchangeSurfaceTemplate, 'DataOutput', 'UserDefinedData_t')
    I.createChild(n, 'HeatFlux', 'DataArray_t')
    I.createChild(ExchangeSurfaceTemplate, 'Mode', 'UserDefinedData_t')

    base = I.getBases(t)[0]  # WARNING: Only one base
    for i, famBCTrigger in enumerate(coupledSurfaces):
        # Create Family for cwipi
        ExchangeSurface = I.copyRef(ExchangeSurfaceTemplate)
        surfaceName = 'ExchangeSurface{}'.format(i)
        I.setName(ExchangeSurface, surfaceName)
        I.addChild(base, ExchangeSurface)
        # Add AdditionnalFamilyName in concerned BC nodes
        for BC in C.getFamilyBCs(base, famBCTrigger):
            I.createChild(BC, 'SurfaceName', 'AdditionalFamilyName_t', value=surfaceName)

        Family = I.getNodeFromName1(base, famBCTrigger)
        # This Trigger node is mandatory for pyC2, even the zone is already tagged with a trigger node
        J.set(Family, '.Solver#Trigger', next_iteration=1, next_state=19, file=couplingScript)


def renameElementsNodesInElsAHybrid(t):
    ####################################################
    # WARNING : pyC2 manages only quad unstructured mesh
    ####################################################
    I._renameNode(t, 'InternalElts', 'InternalQuads')
    I._renameNode(t, 'ExternalElts', 'ExternalQuads')

def locateCouplingSurfaces(t):
    '''
    Locate all the surfaces in **t** that will be coupled with CWIPI.

    Parameters
    ----------

        t : PyTree
            elsA tree

    Returns
    -------

        CouplingSurfaces : dict
            Dictionary of coupling surfaces. Each key is the name of a coupling
            surface. The corresponding value is a sub-dictionary, with the
            following elements:

            * ParentBase : str
              Name of the base where the coupling surface is located.

            * ParentZones : list
              Names of all the zones where the coupling surface is located.

            * ExchangeSurface : str
              Name of the ``Family_t`` node used by CWIPI. It should be
              ``ExchangeSurface<N>``, with ``<N>`` an integer.
    '''
    CouplingSurfacesList = []
    for family in I.getNodesFromType2(t, 'Family_t'):
        if not I.getNodesFromType(family, 'FamilyBC_t'):
            continue
        trigger = I.getNodeFromName(family, '.Solver#Trigger')
        if trigger:
            CouplingSurfacesList.append(I.getName(family))

    # Initialize dictionary
    CouplingSurfaces = dict()
    for Family in CouplingSurfacesList:
        CouplingSurfaces[Family] = dict(ExchangeSurface=None, ParentZones=[], ParentBase=[])

    for FamilyName, CouplingSurface in CouplingSurfaces.items():
        for base in I.getBases(t):
            for zone in I.getZones(t):
                if not I.getNodeFromType1(zone, 'GridCoordinates_t'): continue  # Skeleton zone
                zoneName = I.getName(zone)
                bcs = C.getFamilyBCs(zone, FamilyName)
                if len(bcs) > 0:
                    CouplingSurface['ParentZones'].append(zoneName)
                    for bc in bcs:
                        CouplingSurface['ExchangeSurface'] = I.getValue(I.getNodeFromType(bc, 'AdditionalFamilyName_t'))
            if len(CouplingSurface['ParentZones']) > 0:
                CouplingSurface['ParentBase'].append(I.getName(base))

    for FamilyName, CouplingSurface in CouplingSurfaces.items():
        if len(CouplingSurface['ParentBase']) == 0:
            CouplingSurface['ParentBase'] = None
        elif len(CouplingSurface['ParentBase']) == 1:
            CouplingSurface['ParentBase'] = CouplingSurface['ParentBase'][0]
        else:
            MSG = 'Coupling surface must be located in only one base, but {} \
                   is found in the following bases: {}'.format(FamilyName,
                   CouplingSurface['ParentBase'])
            raise ValueError(MSG)

    return CouplingSurfaces

def initializeCWIPIConnections(t, Distribution, CouplingSurfaces, tol=0.01):
    '''
    Initialize CWIPI. This function must be called in ``compute.py``.

    .. important::
        CWIPI is not in the default MOLA environment and must be sourced before
        using this function.

    Parameters
    ----------

        t : PyTree
            elsA tree

        Distribution : dict
            Same dictionary than the one used to impose the distribution of zones
            on processors for the elsA simulation.

        CouplingSurfaces : dict
            Dictionary of coupling surfaces, as returned by
            :py:func:`locateCouplingSurfaces`.
            Each key is the name of a coupling surface. The corresponding value
            is a sub-dictionary, with the following elements:

            * ParentBase : str
              Name of the base where the coupling surface is located.

            * ExchangeSurface : str
              Name of the ``Family_t`` node used by CWIPI. It should be
              ``ExchangeSurface<N>``, with ``<N>`` an integer.

            * ParentZones : list
              Names of all the zones where the coupling surface is located.

            * tolerance : float
              Geometrical tolerance for CWIPI interpolations. If not given, the
              default value **tol** is taken.

        tol : float
            Default geometrical tolerance for CWIPI interpolations.

    Returns
    -------

        fwk : CWIPI Framework object

        pyC2Connections : dict
            dictionary with all the CWIPI Connection objects. Each key
            is the name of a coupling surface.

    '''
    import CWIPI.CGNS as C2

    fwk = C2.Framework('elsA', distribution=Distribution)
    fwk.trace('init elsA connection')

    pyC2Connections = dict()
    for surface, CouplingSurface in CouplingSurfaces.items():
        if CouplingSurface['ExchangeSurface']:
            if not 'tolerance' in CouplingSurface:
                CouplingSurface['tolerance'] = tol
            pyC2Connections[surface] = C2.Connection(fwk,
                                            surface,
                                            'Zebulon',
                                            t,
                                            CouplingSurface['ParentBase'],
                                            surfacename=CouplingSurface['ExchangeSurface'],
                                            zonelist=CouplingSurface['ParentZones'],
                                            mode=C2.PARAPART,
                                            solver=C2.CELLCENTER,
                                            tolerance=CouplingSurface['tolerance'],
                                            debug=1
                                            )

    fwk.trace('end init elsA connection')
    return fwk, pyC2Connections


################################################################################
# Functions for coprocess (coupling with CWIPI)
################################################################################

def computeOptimalAlpha(FluidData, dt, problem='DirichletRobin'):
    '''
    Compute the optimal value of the coupling coefficient alpha.

    Parameters
    ----------

        FluidData : dict
            Fluid data on coupled elsA BCs. Must contain the following keys:

            * thrm_cndy_lam: Thermal conductivity (m/s)

            * Density: density (m/s)

            * cp: Specific heat at constant pressure (J/K/kg)

            * hpar: size of the first cell in the normal to the wall direction (m)

        # SolidConductivity : float
        #     Thermal conductivity of the solid (W/K/m)
        #
        # SolidLength : float
        #     Characteristic length of the solid for thermal transfer

        dt : float
            Timestep between 2 coupling exchanges (CWIPI calls)

        problem : str
            Type of problem to solve. Default value is 'DirichletRobin',
            corresponding to a Dirichlet condition on the fluid side,
            and a Robin condition on the solid side.

    Returns
    -------

        alphaopt : float
            Optimal coupling coefficient in Robin condition
    '''

    if problem == 'DirichletRobin':
        alphaopt = computeOptimalAlphaDirichletRobin(FluidData, dt)
    else:
        raise ValueError('method is unknown')

    alphaopt[alphaopt<0] = 0.

    return alphaopt

def computeOptimalAlphaDirichletRobin(FluidData, dt):
    '''
    Compute the optimal value of the coupling coefficient alpha for a
    Dirichlet/Robin problem. (see eq. 2.49 in Rocco Moretti PhD thesis)

    Parameters
    ----------

        FluidData : dict
            Fluid data on coupled elsA BCs. Must contain the following keys:

            * thrm_cndy_lam: Thermal conductivity (m/s)

            * Density: density (m/s)

            * cp: Specific heat at constant pressure (J/K/kg)

            * hpar: size of the first cell in the normal to the wall direction (m)

        dt : float
            Timestep between 2 coupling exchanges (CWIPI calls)

    Returns
    -------

        alphaopt : float
            Optimal coupling coefficient in Robin condition
    '''
    Kf        = 2. * FluidData['thrm_cndy_lam'] / FluidData['hpar']
    FluidDiffusivity = FluidData['thrm_cndy_lam'] / FluidData['Density'] / FluidData['cp']
    Df        = FluidDiffusivity * dt / (FluidData['hpar']**2)
    Dfbar     = np.divide(Df, 1 + Df + np.sqrt(1+2*Df))
    alphaopt  = Kf * (1-Dfbar)/2.

    return alphaopt

def computeLocalTimestep(FluidData, setup, CFL=None):
    '''
    Compute the local timestep, used to compute the optimal alpha
    coefficient. It is the minimum between the diffusive timestep and
    the convective timestep.

    .. note::
        See the following paper for more details:
        Rami Salem, Marc Errera, Julien Marty, *Adaptive diffusive time-step in
        conjugate heat transfer interface conditions for thermal-barrier-coated
        applications*, International Journal of Thermal Sciences, 2019

    Parameters
    ----------

        FluidData : dict
            Fluid data on coupled elsA BCs. Must contain the following keys:

            * Density: density (m/s)

            * hpar: size of the first cell in the normal to the wall direction (m)

            * ViscosityMolecular (Pa.s)

            * Viscosity_EddyMolecularRatio (-)

        setup : module
            Python module object as obtained from command

            >>> import setup

        CFL : float
            Value of the CFL number

    Returns
    -------

        dt : float
            diffusive timestep
    '''
    if not CFL:
        try:
            solverFunction = setup.elsAkeysNumerics['.Solver#Function']
            assert solverFunction['name'] == 'f_cfl'
            CFL = solverFunction['valf']  # Last (so maximum) value of CFL
                                          # -> increase the computed timestep
                                          # -> increase the coupling stability
        except:
            raise Exception('Cannot extract the CFL value from setup')
    dy = FluidData['hpar']
    rho = FluidData['Density']
    mu = FluidData['ViscosityMolecular']
    mut_mu = FluidData['Viscosity_EddyMolecularRatio']
    gamma = setup.FluidProperties['Gamma']
    Pr = setup.FluidProperties['Prandtl']
    Prt = setup.FluidProperties['PrandtlTurbulence']
    sound_speed =(FluidData['cp'] * (gamma -1) * FluidData['Temperature']) ** 0.5

    # Diffusive timestep
    dt_diff = 0.5 * CFL * dy**2 * rho * Pr / mu / (1. + mut_mu * Pr/Prt) # / gamma in the paper of Rami
    # Convective timestep
    dt_conv = CFL * dy / sound_speed  # c+u=c because u=0 at the wall
    #
    dt_fluid = np.minimum(dt_diff, dt_conv)

    return dt_fluid

def cwipiCoupling(to, pyC2Connections, setup, CurrentIteration):
    '''
    Perform the CWIPI coupling with Zset to update coupled boundary conditions.

    .. important::
        For the moment, the problem is solved with a Dirichlet-Robin set-up.

    Parameters
    ----------

        to : PyTree
            Coupling tree as obtained from :py:func:`adaptEndOfRun`

        pyC2Connections : dict
            dictionary with all the CWIPI Connection objects. Each key
            is the name of a coupling surface. As returned by
            :py:func:`MOLA.WorkflowAerothermalCoupling.initializeCWIPIConnections`

        setup : module
            Python module object as obtained from command

            >>> import setup

        CurrentIteration : int
            Current iteration in the simulation

    returns
    -------

        CWIPIdata : dict
            Data exchanged with CWIPI. The structure of this :py:class:`dict` is
            the following:

            >>> CWIPIdata[COM][CoupledSurface][VariableName] = np.array

            where 'COM' is ``SEND`` or ``RECV``, 'CoupledSurface' is the
            family name of the coupled BC surface, and 'VariableName' is the
            name of the exchanged quantity.

    '''
    import Coprocess as CO

    SentVariables     = ['NormalHeatFlux', 'Temperature']
    ReceivedVariables = ['Temperature']
    VariablesForAlpha = ['Density', 'thrm_cndy_lam', 'hpar', 'ViscosityMolecular', 'Viscosity_EddyMolecularRatio']
    AllNeededVariables = SentVariables + VariablesForAlpha

    stepForCwipiCommunication = setup.ReferenceValues['CoprocessOptions']['UpdateCWIPICouplingFrequency']
    if 'timestep' in setup.elsAkeysNumerics:
        timestep = setup.elsAkeysNumerics['timestep']
        dtCoupling = timestep * stepForCwipiCommunication
    MultiplicativeRampForAlpha = setup.ReferenceValues['CoprocessOptions'].get('MultiplicativeRampForAlpha', None)

    CWIPIdata = dict(SEND={}, RECV={})
    for CPLsurf in pyC2Connections:
        CWIPIdata['SEND'][CPLsurf] = dict()
        CWIPIdata['RECV'][CPLsurf] = dict()

    for CPLsurf, cplConnection in pyC2Connections.items():
        CO.printCo('CWIPI coupling on {}'.format(CPLsurf), 0, color=CO.MAGE)
        #___________________________________________________________________________
        # Get all needed data at coupled BCs
        #___________________________________________________________________________
        BCnodes = C.getFamilyBCs(to, CPLsurf)
        if len(BCnodes) == 0:
            continue
        elif len(BCnodes) > 1:
            raise Exception('Could be only one coupled BC per Family on each processor')
        else:
            BCnode = BCnodes[0]

        BCDataSet = dict()
        for var in AllNeededVariables:
            varNode = I.getNodeFromName(BCnode, var)
            if varNode:
                BCDataSet[var] = I.getValue(varNode).flatten()

        #___________________________________________________________________________
        # SEND DATA
        #___________________________________________________________________________
        for var in SentVariables:
            CWIPIdata['SEND'][CPLsurf][var] = BCDataSet[var]
            print('Sending {}...'.format(var))
            print('shape {}'.format(BCDataSet[var].shape))
            cplConnection.publish(CWIPIdata['SEND'][CPLsurf][var], iteration=CurrentIteration, stride=1, tag=100)
            print("Send {} with mean value = {}".format(var, np.mean(CWIPIdata['SEND'][CPLsurf][var])))

        #___________________________________________________________________________
        # Compute alpha_opt
        #___________________________________________________________________________
        BCDataSet['cp'] = setup.FluidProperties['cp']

        if 'timestep' not in setup.elsAkeysNumerics:
            localTimestep = computeLocalTimestep(BCDataSet, setup)
            dtCoupling = localTimestep * stepForCwipiCommunication

        alphaOpt = computeOptimalAlpha(BCDataSet, dtCoupling)
        if MultiplicativeRampForAlpha:
            alphaOpt *= J.rampFunction(**MultiplicativeRampForAlpha)(CurrentIteration)
        CWIPIdata['SEND'][CPLsurf]['alpha'] = alphaOpt
        print('Sending {}...'.format('alpha'))
        cplConnection.publish(alphaOpt, iteration=CurrentIteration, stride=1, tag=100)
        print("Send {} with mean value = {}".format('alpha', np.mean(alphaOpt)))
    #___________________________________________________________________________
    # RECEIVE DATA
    #___________________________________________________________________________
    for CPLsurf, cplConnection in pyC2Connections.items():
        print('Receiving...')
        remote_data = cplConnection.retrieve(iteration=CurrentIteration, stride=len(ReceivedVariables), tag=100)
        i1 = 0
        dataLength = remote_data.size / len(ReceivedVariables)
        for var in ReceivedVariables:
            CWIPIdata['RECV'][CPLsurf][var] = remote_data[i1:i1+dataLength]
            print("Receive {} with mean value = {}".format(var, np.mean(CWIPIdata['RECV'][CPLsurf][var])))
            i1 += dataLength

    #___________________________________________________________________________
    # UPDATE BCs IN ELSA TREE
    #___________________________________________________________________________
    for CPLsurf, cplConnection in pyC2Connections.items():
        BCnode = C.getFamilyBCs(to, CPLsurf)[0] # The test of the lenght of the
                                                # list (=1) has already been done
                                                # before in this function
        for var in ReceivedVariables:
            varNode = I.getNodeFromName(BCnode, var)
            if varNode:
                newDataOnBC = CWIPIdata['RECV'][CPLsurf][var].reshape(I.getValue(varNode).shape)
                I.setValue(varNode, newDataOnBC)

    #___________________________________________________________________________
    # UPDATE RUNTIME TREE
    #___________________________________________________________________________
    I._rmNodesByType(to, 'FlowSolution_t')
    CO.Cmpi.barrier()

    return to, CWIPIdata

def appendCWIPIDict2Arrays(arrays, CWIPIdata, CurrentIteration, RequestedStatistics=[]):
    import Coprocess as CO

    for CPLsurf in CWIPIdata['SEND']:
        SENDdata2Arrays = dict(
            IterationNumber = CurrentIteration, #-1,  # Because extraction before current iteration (next_state=16)
            TemperatureMax  = np.amax(CWIPIdata['SEND'][CPLsurf]['Temperature']),
            HeatFluxAbsMax  = np.amax(np.abs(CWIPIdata['SEND'][CPLsurf]['NormalHeatFlux'])),
            AlphaMin        = np.amin(CWIPIdata['SEND'][CPLsurf]['alpha'])
        )
        RECVdata2Arrays = dict(
            IterationNumber = CurrentIteration, #-1,  # Because extraction before current iteration (next_state=16)
            TemperatureMax  = np.amax(CWIPIdata['RECV'][CPLsurf]['Temperature']),
        )

        CO.appendDict2Arrays(arrays, SENDdata2Arrays, 'SEND_{}'.format(CPLsurf))
        CO.appendDict2Arrays(arrays, RECVdata2Arrays, 'RECV_{}'.format(CPLsurf))

        CO._extendArraysWithStatistics(arrays, 'SEND_{}'.format(CPLsurf), RequestedStatistics)
        CO._extendArraysWithStatistics(arrays, 'RECV_{}'.format(CPLsurf), RequestedStatistics)

    arraysTree = CO.arraysDict2PyTree(arrays)
    return arraysTree
