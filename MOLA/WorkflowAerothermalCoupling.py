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

import sys
import os
import numpy as np
import copy

import Converter.PyTree    as C
import Converter.Internal  as I

from . import InternalShortcuts as J
from . import Preprocess        as PRE
from . import WorkflowCompressor as WC

def prepareMesh4ElsA(mesh, kwargs):
    '''
    Exactly like :py:func:`MOLA.WorkflowCompressor.prepareMesh4ElsA`
    '''
    return WC.prepareMesh4ElsA(mesh, **kwargs)

def prepareMainCGNS4ElsA(mesh='mesh.cgns', ReferenceValuesParams={},
        NumericalParams={}, TurboConfiguration={}, Extractions={}, BoundaryConditions={},
        BodyForceInputData=[], writeOutputFields=True, bladeFamilyNames=['Blade'],
        Initialization={'method':'uniform'}, FULL_CGNS_MODE=True):
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

        FULL_CGNS_MODE : bool
            if :py:obj:`True`, put all elsA keys in a node ``.Solver#Compute``
            to run in full CGNS mode.

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

    if I.getNodeFromName(t, 'proc'):
        NProc = max([I.getNodeFromName(z,'proc')[1][0][0] for z in I.getZones(t)])+1
        ReferenceValues['NProc'] = int(NProc)
        ReferenceValuesParams['NProc'] = int(NProc)
        Splitter = None
    else:
        ReferenceValues['NProc'] = 0
        Splitter = 'PyPart'

    elsAkeysCFD      = PRE.getElsAkeysCFD(nomatch_linem_tol=1e-6, unstructured=IsUnstructured)
    elsAkeysModel    = PRE.getElsAkeysModel(FluidProperties, ReferenceValues, unstructured=IsUnstructured)
    if BodyForceInputData: NumericalParams['useBodyForce'] = True
    if not 'NumericalScheme' in NumericalParams:
        NumericalParams['NumericalScheme'] = 'roe'
    elsAkeysNumerics = PRE.getElsAkeysNumerics(ReferenceValues,
                            unstructured=IsUnstructured, **NumericalParams)

    PRE.initializeFlowSolution(t, Initialization, ReferenceValues)

    if not 'PeriodicTranslation' in TurboConfiguration and \
        any([rowParams['NumberOfBladesSimulated'] > rowParams['NumberOfBladesInInitialMesh'] \
            for rowParams in TurboConfiguration['Rows'].values()]):
        t = WC.duplicateFlowSolution(t, TurboConfiguration)

    WC.setBoundaryConditions(t, BoundaryConditions, TurboConfiguration,
                            FluidProperties, ReferenceValues,
                            bladeFamilyNames=bladeFamilyNames)

    WC.computeFluxCoefByRow(t, ReferenceValues, TurboConfiguration)

    if not 'PeriodicTranslation' in TurboConfiguration:
        WC.addMonitoredRowsInExtractions(Extractions, TurboConfiguration)

    AllSetupDics = dict(FluidProperties=FluidProperties,
                        ReferenceValues=ReferenceValues,
                        elsAkeysCFD=elsAkeysCFD,
                        elsAkeysModel=elsAkeysModel,
                        elsAkeysNumerics=elsAkeysNumerics,
                        TurboConfiguration=TurboConfiguration,
                        Extractions=Extractions,
                        Splitter=Splitter)
    if BodyForceInputData: AllSetupDics['BodyForceInputData'] = BodyForceInputData

    BCExtractions = dict(
        BCWall = ['normalvector', 'frictionvectorx', 'frictionvectory', 'frictionvectorz','psta', 'bl_quantities_2d', 'yplusmeshsize'],
        BCInflow = ['convflux_ro'],
        BCOutflow = ['convflux_ro'],
        )
    BCExtractions['BCWallViscousIsothermal'] = BCExtractions['BCWall'] \
        + ['tsta', 'normalheatflux', 'thrm_cndy_lam', 'hpar', 'ro', 'visclam', 'viscrapp']

    PRE.addTrigger(t)
    PRE.addExtractions(t, AllSetupDics['ReferenceValues'],
                      AllSetupDics['elsAkeysModel'],
                      extractCoords=False, BCExtractions=BCExtractions)

    if elsAkeysNumerics['time_algo'] != 'steady':
        PRE.addAverageFieldExtractions(t, AllSetupDics['ReferenceValues'],
            AllSetupDics['ReferenceValues']['CoprocessOptions']['FirstIterationForAverage'])

    PRE.addReferenceState(t, AllSetupDics['FluidProperties'],
                         AllSetupDics['ReferenceValues'])
    dim = int(AllSetupDics['elsAkeysCFD']['config'][0])
    PRE.addGoverningEquations(t, dim=dim)
    AllSetupDics['ReferenceValues']['NProc'] = int(max(PRE.getProc(t))+1)
    PRE.writeSetup(AllSetupDics)

    if FULL_CGNS_MODE:
        PRE.addElsAKeys2CGNS(t, [AllSetupDics['elsAkeysCFD'],
                                 AllSetupDics['elsAkeysModel'],
                                 AllSetupDics['elsAkeysNumerics']])

    addExchangeSurfaces(t, AllSetupDics['ReferenceValues']['CoprocessOptions']['CoupledSurfaces'], couplingScript='CouplingScript.py')
    renameElementsNodesInElsAHybrid(t)

    PRE.saveMainCGNSwithLinkToOutputFields(t,writeOutputFields=writeOutputFields)

    if not Splitter:
        print('REMEMBER : configuration shall be run using %s%d%s procs'%(J.CYAN,
                                                   ReferenceValues['NProc'],J.ENDC))
    else:
        print('REMEMBER : configuration shall be run using %s'%(J.CYAN + \
            Splitter + J.ENDC))


def addExchangeSurfaces(t, coupledSurfaces, couplingScript='CouplingScript.py'):
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
    i = 0
    for surface, CouplingSurface in CouplingSurfaces.items():
        if CouplingSurface['ExchangeSurface']:
            if not 'tolerance' in CouplingSurface:
                CouplingSurface['tolerance'] = tol
            pyC2Connections[surface] = C2.Connection(fwk,
                                            'raccord{}'.format(i),
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
            i += 1
    fwk.trace('end init elsA connection')
    return fwk, pyC2Connections

def rampFunction(x1, x2, y1, y2):
    '''
    Create a ramp function, going from **y1** to **y2** between **x1** to **x2**.

    Parameters
    ----------

        x1, x2, y1, y2 : float

    Returns
    -------

        f : function
            Ramp function. Could be called in ``x`` with:

            >> f(x)
    '''
    slope = (y2-y1) / (x2-x1)
    if y1 == y2:
        f = lambda x: y1*np.ones(np.shape(x))
    elif y1 < y2:
        f = lambda x: np.maximum(y1, np.minimum(y2, slope*(x-x1)+y1))
    else:
        f = lambda x: np.minimum(y1, np.maximum(y2, slope*(x-x1)+y1))
    return f

def computeOptimalAlpha(FluidData, dt, problem='DR'):
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
            Type of problem to solve. Default value is 'DR', corresponding to
            a Dirichlet condition on the fluid side, and a Robin condition on
            the solid side.

    Returns
    -------

        alphaopt : float
            Optimal coupling coefficient in Robin condition
    '''

    if problem == 'DR':
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


def getNumberOfNodesInBC(BCNode):
    '''
    Get the number of nodes on a BC patch. Works for structured and
    unstructured meshes.

    Parameters
    ----------

        BCNode : PyTree
            Node of type ``'BC_t'``

    Results
    -------

        size : int
            Number of nodes on **BCNode**
    '''
    BCIndexNode = I.getNodeFromType(BCNode, 'IndexArray_t') # May be PointRange if structured or PointList if unstructured
    dim = I.getValue(BCIndexNode)
    if I.getName(BCIndexNode) == 'PointRange':
        # Structured zone
        i = dim[0][1] - dim[0][0]
        j = dim[1][1] - dim[1][0]
        k = dim[2][1] - dim[2][0]
        if i == 0: size = (j+1)*(k+1)
        if j == 0: size = (i+1)*(k+1)
        if k == 0: size = (i+1)*(j+1)
    elif I.getName(BCIndexNode) == 'PointList':
        # Unstructured zone
        size = dim.size
    else:
        raise ValueError
    return size