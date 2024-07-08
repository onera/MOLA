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
MOLA - WorkflowCompressor.py

WORKFLOW COMPRESSOR

Collection of functions designed for Workflow Compressor

31/08/2021 - T. Bontemps - Creation
'''

import MOLA
from . import InternalShortcuts as J
from . import Preprocess        as PRE
from . import JobManager        as JM
from . import BodyForceTurbomachinery as BF

if not MOLA.__ONLY_DOC__:
    import sys
    import os
    import numpy as np
    import pprint
    import copy
    import scipy.optimize

    import Converter.PyTree    as C
    import Converter.Internal  as I
    import Distributor2.PyTree as D2
    import Post.PyTree         as P
    import Generator.PyTree    as G
    import Transform.PyTree    as T
    import Connector.PyTree    as X
    import Converter.elsAProfile as elsAProfile


def checkDependencies():
    '''
    Make a series of functional tests in order to determine if the user
    environment is correctly set for using the Workflow Compressor
    '''
    JM.checkDependencies()

    print('Checking ETC...')
    try:
        import etc.transform
        print(J.GREEN+'ETC module is available'+J.ENDC)
    except ImportError:
        MSG = 'Fail to import ETC module: Some functions of {} are unavailable'.format(__name__)
        print(J.FAIL + MSG + J.ENDC)

    print('\nVERIFICATIONS TERMINATED')


def prepareMesh4ElsA(mesh, InputMeshes=None, splitOptions=None, 
                    duplicationInfos={}, zonesToRename={},
                    scale=1., rotation='fromAG5', tol=1e-8, PeriodicTranslation=None,
                    BodyForceRows=None, families2Remove=[], saveGeometricalDataForBodyForce=True):
    '''
    This is a macro-function used to prepare the mesh for an elsA computation
    from a CGNS file provided by Autogrid 5.

    The sequence of performed operations is the following:

    #. load and clean the mesh from Autogrid 5
    #. apply transformations
    #. add grid connectivities
    #. duplicate the mesh in rotation (if needed)
    #. split the mesh (only if PyPart is not used)
    #. distribute the mesh (only if PyPart is not used)
    #. make final elsA-specific adaptations of CGNS data

    
    .. warning::
        The following assumptions on the input mesh are made:
        
        * it does not need any scaling

        * the shaft axis is the Z-axis, pointing downstream (convention in
          Autogrid 5). The mesh will be rotated to follow the elsA convention,
          thus the shaft axis will be the X-axis, pointing downstream.

    Parameters
    ----------

        mesh : :py:class:`str` or PyTree
            Name of the CGNS mesh file from Autogrid 5 or already read PyTree.

        InputMeshes : :py:class:`list` of :py:class:`dict`
            User-provided data. See documentation of Preprocess.prepareMesh4ElsA

        splitOptions : dict
            All optional parameters passed to function
            :py:func:`MOLA.Preprocess.splitAndDistribute`

            .. important::
                If **splitOptions** is an empty :py:class:`dict` (default value),
                no split nor distribution are done. This behavior required to
                use PyPart in the simulation.
                If you want to split the mesh with default parameters, you have
                to specify one of them. For instance, use:

                >> splitOptions=dict(mode='auto')

        duplicationInfos : dict
            User-provided data related to domain duplication.
            Each key corresponds to a row FamilyName.
            The associated element is a dictionary with the following parameters:

                * NumberOfBlades: number of blades in the row (in reality)

                * NumberOfDuplications: number of duplications to make of the
                  input row domain.

                * MergeBlocks: boolean, if True the duplicated blocks are merged.

        zonesToRename : dict
            Each key corresponds to the name of a zone to modify, and the associated
            value is the new name to give.

        scale : float
            Homothety factor to apply on the mesh. Default is 1.

        rotation : :py:class:`str` or :py:class:`list`
            List of rotations to apply on the mesh. If **rotation** =
            ``'fromAG5'``, then default rotations are applied:

            >>> rotation = [((0,0,0), (0,1,0), 90), ((0,0,0), (1,0,0), 90)]

            Else, **rotation** must be a list of rotation to apply to the grid
            component. Each rotation is defined by 3 elements:

                * a 3-tuple corresponding to the center coordinates
                * a 3-tuple corresponding to the rotation axis
                * a float (or integer) defining the angle of rotation in
                  degrees

        PeriodicTranslation : :py:obj:`None` or :py:class:`list` of :py:class:`float`
            If not :py:obj:`None`, the configuration is considered to be with
            a periodicity in the direction **PeriodicTranslation**. This argument
            has to be used for linear cascade configurations.

        tol : float
            Tolerance for connectivities matching (including periodic connectivities).

        BodyForceRows : :py:class:`dict` or :py:obj:`None`
            If not :py:obj:`None`, this parameters allows to replace user-defined
            row domains with meshes adapted to body-force modelling.
            See documentation of py:func:`MOLA.BodyForceTurbomachinery.replaceRowWithBodyForceMesh`.

        families2Remove : list
            Families to remove in the tree when using body-force. Should be a list 
            of families related to interstage interfaces between a stator row and 
            a BFM row, or to BFM rows. It allows to force a matching mesh at the interface
            instead having a mixing plane.

        saveGeometricalDataForBodyForce : bool
            If :py:obj:`True`, save the intermediate files ``BodyForceData_{row}.cgns`` for each row.
            These files contain a CGNS tree with:

                * 4 lines (1D zones) corresponding to Hub, Shroud, Leading edge and Trailing Edge.
                * The zone'Skeleton' with geometrical data on blade profile (used for interpolation later). 

    Returns
    -------

        t : PyTree
            the pre-processed mesh tree (usually saved as ``mesh.cgns``)

            .. important:: This tree is **NOT** ready for elsA computation yet !
                The user shall employ function :py:func:`prepareMainCGNS4ElsA`
                as next step
    '''
    toc = J.tic()
    if isinstance(mesh,str):
        filename = mesh
        t = J.load(mesh)
    elif I.isTopTree(mesh):
        filename = None
        t = mesh
    else:
        raise ValueError('parameter mesh must be either a filename or a PyTree')


    if PRE.hasAnyUnstructuredZones(t):
        t = PRE.convertUnstructuredMeshToNGon(t,
                mergeZonesByFamily=False if splitOptions else True)

    if InputMeshes is None:
        InputMeshes = generateInputMeshesFromAG5(t,
            scale=scale, rotation=rotation, tol=tol, PeriodicTranslation=PeriodicTranslation)
        for InputMesh in InputMeshes: 
            InputMesh['file'] = filename

    PRE.checkFamiliesInZonesAndBC(t)
    PRE.transform(t, InputMeshes)

    if BodyForceRows:
        # Remesh rows to model with body-force
        t, newRowMeshes = BF.replaceRowWithBodyForceMesh(
            t, BodyForceRows, saveGeometricalDataForBodyForce=saveGeometricalDataForBodyForce)
        
        for row, BodyForceParams in BodyForceRows.items():
            if 'AzimutalAngleDeg' in BodyForceParams:
                InputMeshes[0]['Connection'].append(
                    dict(type='PeriodicMatch', tolerance=tol, rotationAngle=[BodyForceParams['AzimutalAngleDeg'],0.,0.])
                )
        # Default value
        InputMeshes[0]['Connection'].append(
                dict(type='PeriodicMatch', tolerance=tol, rotationAngle=[2.,0.,0.])
            )

    t = cleanMeshFromAutogrid(t, basename=InputMeshes[0]['baseName'], zonesToRename=zonesToRename)

    if BodyForceRows:
         #Add body-force domains in the main mesh
        base = I.getBases(t)[0]
        for newRowMesh in newRowMeshes:
            for zone in I.getZones(newRowMesh):
                I.addChild(base, zone)
        if families2Remove == []:
            warning = 'WARNING: families2Remove is empty although body force is used.'
            warning+= 'Please have a look to the documentation and double check that is not a mistake.'
            print(J.WARN+warning+J.ENDC)
        else:
            for family in families2Remove:
                for bc in C.getFamilyBCs(t, family):
                    I._rmNode(t, bc)
                I._rmNodesByName(t, family)

    for row, rowParams in duplicationInfos.items():
        try: MergeBlocks = rowParams['MergeBlocks']
        except: MergeBlocks = False
        duplicate(t, row, rowParams['NumberOfBlades'],
                nDupli=rowParams['NumberOfDuplications'], merge=MergeBlocks)

    t = PRE.connectMesh(t, InputMeshes)
    if splitOptions is not None:
        t = PRE.splitAndDistribute(t, InputMeshes, **splitOptions)
    
    # WARNING: Names of BC_t nodes must be unique to use PyPart on globborders
    for l in [2,3,4]: I._correctPyTree(t, level=l)

    for base, meshInfo in zip(I.getBases(t), InputMeshes):
        J.set(base,'.MOLA#InputMesh',**meshInfo)

    PRE.adapt2elsA(t, InputMeshes)
    J.checkEmptyBC(t)

    J.printElapsedTime('prepareMesh took ', toc)

    return t

def prepareMainCGNS4ElsA(mesh='mesh.cgns', ReferenceValuesParams={},
        NumericalParams={}, OverrideSolverKeys= {}, 
        TurboConfiguration={}, Extractions=[], BoundaryConditions=[],
        PostprocessOptions={}, BodyForceInputData=[], writeOutputFields=True,
        bladeFamilyNames=['BLADE', 'AUBE'], Initialization={'method':'uniform'},
        JobInformation={}, SubmitJob=False,
        FULL_CGNS_MODE=False, templates=dict(), secondOrderRestart=False):
    '''
    This is mainly a function similar to :func:`MOLA.Preprocess.prepareMainCGNS4ElsA`
    but adapted to compressor computations. Its purpose is adapting the CGNS to
    elsA.

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

        PostprocessOptions : dict
            Dictionary for post-processing.

        BodyForceInputData : list
            if provided, each key in this :py:class:`list` is a dictionay that activate a body-force model,
            described with the following entries:

                * Family: the name of the row family on which the model is applied.

                * BodyForceParameters: a dict to provide the parameter of the model
                  (see dedicated functions in :mod:`MOLA.BodyForceTurbomachinery`).


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
    if isinstance(mesh,str):
        t = J.load(mesh)
    elif I.isTopTree(mesh):
        t = mesh
    else:
        raise ValueError('parameter mesh must be either a filename or a PyTree')

    IsUnstructured = PRE.hasAnyUnstructuredZones(t)

    TurboConfiguration = getTurboConfiguration(t, BodyForceInputData=BodyForceInputData, **TurboConfiguration)
    FluidProperties = PRE.computeFluidProperties()
    if not 'Surface' in ReferenceValuesParams:
        ReferenceValuesParams['Surface'] = getReferenceSurface(t, BoundaryConditions, TurboConfiguration)

    if 'PeriodicTranslation' in TurboConfiguration:
        MainDirection = np.array([1,0,0]) # Strong assumption here
        YawAxis = np.array(TurboConfiguration['PeriodicTranslation'])
        YawAxis /= np.sqrt(np.sum(YawAxis**2))
        PitchAxis = np.cross(YawAxis, MainDirection)
        ReferenceValuesParams.update(dict(PitchAxis=PitchAxis, YawAxis=YawAxis))

    ReferenceValues = computeReferenceValues(FluidProperties, **ReferenceValuesParams)
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
 
    if ('ChorochronicInterface' or 'stage_choro') in (bc['type'] for bc in BoundaryConditions):
        CHORO_TAG = True
        MSG = 'Chorochronic BC detected.'
        print(J.WARN + MSG + J.ENDC)
        ChoroInterfaceNumber = 0 
        for bc in BoundaryConditions :
            if bc['type'] == 'ChorochronicInterface' or bc['type'] == 'stage_choro':
                ChoroInterfaceNumber += 1
        if ChoroInterfaceNumber > 1:
            MSG = 'Warning: more than one chorochronic interface has been detected: multichorochronic simulation is not available yet.'
            raise Exception(J.FAIL + MSG + J.ENDC)       
        updateChoroTimestep(t, Rows = TurboConfiguration['Rows'], NumericalParams = NumericalParams)
    else:
        CHORO_TAG = False
    
    elsAkeysNumerics = PRE.getElsAkeysNumerics(ReferenceValues,
                            unstructured=IsUnstructured, **NumericalParams)
    
    # Restart with second order automatically for unsteady simulation
    if secondOrderRestart:
        secondOrderRestart = True if elsAkeysNumerics['time_algo'] in ['gear', 'dts'] else False

    if Initialization['method'] == 'turbo':
        t = initializeFlowSolutionWithTurbo(t, FluidProperties, ReferenceValues, TurboConfiguration)
        if secondOrderRestart:
            for zone in I.getZones(t):
                FSnode = I.copyTree(I.getNodeFromName1(zone, 'FlowSolution#Init'))
                I.setName(FSnode, 'FlowSolution#Init-1')
                I.addChild(zone, FSnode)
    else:
        if CHORO_TAG and Initialization['method'] != 'copy':
            MSG = 'Flow initialization failed. No initial solution provided. Chorochronic simulations must be initialized from a mixing plane solution obtained on the same mesh'
            print(J.FAIL + MSG + J.ENDC)
            raise Exception(J.FAIL + MSG + J.ENDC)

        PRE.initializeFlowSolution(t, Initialization, ReferenceValues, secondOrderRestart=secondOrderRestart)

    if not 'PeriodicTranslation' in TurboConfiguration and \
        any([rowParams['NumberOfBladesSimulated'] > rowParams['NumberOfBladesInInitialMesh'] \
            for rowParams in TurboConfiguration['Rows'].values()]):
        t = duplicateFlowSolution(t, TurboConfiguration)

    setMotionForRowsFamilies(t, TurboConfiguration)

    setBoundaryConditions(t, BoundaryConditions, TurboConfiguration,
                            FluidProperties, ReferenceValues,
                            bladeFamilyNames=bladeFamilyNames)

    # TODO: improvement => via ReferenceValues['WallDistance'] => distinction elsA/Cassiopee
    WallDistance = ReferenceValues.get('WallDistance',None)
    if isinstance(WallDistance,dict):
        walldistperiodic = WallDistance.get('periodic',None)
        walldistsoftware = WallDistance.get('software','elsa')
        if walldistperiodic:
            setZoneParamForPeriodicDistanceByRowsFamilies(t, TurboConfiguration)
            if 'elsa' in walldistsoftware.lower():
                setBCFamilyParamForPeriodicDistance(t, ReferenceValues, bladeFamilyNames=bladeFamilyNames)
            elif 'cassiopee' in walldistsoftware.lower():
                hubFamilyNames=['HUB', 'SPINNER', 'MOYEU']
                shroudFamilyNames=['SHROUD', 'CARTER']
                computeDistance2Walls(t, WallFamilies=bladeFamilyNames+hubFamilyNames+shroudFamilyNames, verbose=True, wallFilename='wall.hdf')
            else: raise ValueError('WallDistance: value for software key must be "elsa" or "cassiopee"')

    computeFluxCoefByRow(t, ReferenceValues, TurboConfiguration)

    addMonitoredRowsInExtractions(Extractions, TurboConfiguration)

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

    AllSetupDicts = dict(Workflow='Compressor',
                        Splitter=Splitter,
                        JobInformation=JobInformation,
                        TurboConfiguration=TurboConfiguration,
                        FluidProperties=FluidProperties,
                        ReferenceValues=ReferenceValues,
                        elsAkeysCFD=elsAkeysCFD,
                        elsAkeysModel=elsAkeysModel,
                        elsAkeysNumerics=elsAkeysNumerics,
                        Extractions=Extractions, 
                        PostprocessOptions=PostprocessOptions)
    if BodyForceInputData: 
        AllSetupDicts['BodyForceInputData'] = BodyForceInputData


    # WARNING: BCInflow and BCOutflow are also used for rotor/stator interfaces. However, extracting other
    # quantities on them, such as 'psta', is not possible and would raise the following error:
    # BaseException: Error: boundary BCOutflow is not implemented yet.



    PRE.addTrigger(t)

    is_unsteady = AllSetupDicts['elsAkeysNumerics']['time_algo'] != 'steady'
    avg_requested = AllSetupDicts['ReferenceValues']['CoprocessOptions']['FirstIterationForFieldsAveraging'] is not None

    if is_unsteady:
        if avg_requested:
            if PostprocessOptions is not None:  # Otherwise postprocessing is not activated
                containers = ['FlowSolution#InitV', 'FlowSolution#AverageV']
                try:
                    containers_at_vertex = PostprocessOptions['container_at_vertex']
                    for c in containers:
                        if c not in containers_at_vertex: 
                            containers_at_vertex += [ c ]
                except KeyError:
                    PostprocessOptions['container_at_vertex'] = containers

        else:
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
    PRE.writeSetup(AllSetupDicts)

    if FULL_CGNS_MODE:
        PRE.addElsAKeys2CGNS(t, [AllSetupDicts['elsAkeysCFD'],
                                 AllSetupDicts['elsAkeysModel'],
                                 AllSetupDicts['elsAkeysNumerics']])

    PRE.adaptFamilyBCNamesToElsA(t)
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
    JM.getTemplates('Compressor', templates, JobInformation=JobInformation)
    if 'DIRECTORY_WORK' in JobInformation:
        PRE.sendSimulationFiles(JobInformation['DIRECTORY_WORK'], overrideFields=writeOutputFields)

    for i in range(SubmitJob):
        singleton = False if i==0 else True
        JM.submitJob(JobInformation['DIRECTORY_WORK'], singleton=singleton)

    J.printElapsedTime('prepareMainCGNS4ElsA took ', toc)

def parametrizeChannelHeight(t, lin_axis=None, method=2):
    '''
    Compute the variable *ChannelHeight* from a mesh PyTree **t**. This function
    relies on the turbo module.

    .. important::

        Dependency to *turbo* module. See file:///stck/jmarty/TOOLS/turbo/doc/html/index.html

    Parameters
    ----------

        t : PyTree
            input mesh tree

        lin_axis : :py:obj:`None` or :py:class:`str`
            Axis for linear configuration.
            If :py:obj:`None`, the configuration is annular (default case), else
            the configuration is linear.
            'XY' means that X-axis is the streamwise direction and Y-axis is the
            spanwise direction.(see turbo documentation)
        
        method : int
            Method used for ``turbo.height.generateHLinesAxial()``. Default value is 2.

    Returns
    -------

        t : PyTree
            modified tree

    '''
    import turbo.height as TH

    def plot_hub_and_shroud_lines(t):
        # Get geometry
        hub     = I.getNodeFromName(t, 'Hub')
        xHub    = I.getValue(I.getNodeFromName(hub, 'CoordinateX'))
        yHub    = I.getValue(I.getNodeFromName(hub, 'CoordinateY'))
        shroud  = I.getNodeFromName(t, 'Shroud')
        xShroud = I.getValue(I.getNodeFromName(shroud, 'CoordinateX'))
        yShroud = I.getValue(I.getNodeFromName(shroud, 'CoordinateY'))
        # Import matplotlib
        import matplotlib.pyplot as plt
        # Plot
        plt.figure()
        plt.plot(xHub, yHub, '-', label='Hub')
        plt.plot(xShroud, yShroud, '-', label='Shroud')
        plt.axis('equal')
        plt.grid()
        plt.xlabel('x (m)')
        plt.ylabel('y (m)')
        # Save
        plt.savefig('shroud_hub_lines.png', dpi=150, bbox_inches='tight')
        return 0

    print(J.CYAN + 'Add ChannelHeight in the mesh...' + J.ENDC)
    OLD_FlowSolutionNodes = I.__FlowSolutionNodes__
    I.__FlowSolutionNodes__ = 'FlowSolution#Height'

    silence_stdout = J.OutputGrabber(stream=sys.stdout)
    silence_stderr = J.OutputGrabber(stream=sys.stderr)
    message = None

    with silence_stdout:

        if not lin_axis:
            endlinesTree = TH.generateHLinesAxial(t, filename='shroud_hub_lines.plt', method=method)
            try: 
                plot_hub_and_shroud_lines(endlinesTree)
            except: 
                pass

            # - Generation of the mask file
            with silence_stderr:
                m = TH.generateMaskWithChannelHeight(t, 'shroud_hub_lines.plt')
            os.remove('shroud_hub_lines.plt')
        else:
            m = TH.generateMaskWithChannelHeightLinear(t, lin_axis=lin_axis)

        # - Generation of the ChannelHeight field
        TH._computeHeightFromMask(t, m, writeMask='mask.cgns', lin_axis=lin_axis)
    
    if message: print(message)

    I.__FlowSolutionNodes__ = OLD_FlowSolutionNodes
    print(J.GREEN + 'done.' + J.ENDC)
    return t

def generateInputMeshesFromAG5(mesh, scale=1., rotation='fromAG5', tol=1e-8, PeriodicTranslation=None):
    '''
    Generate automatically the :py:class:`list` **InputMeshes** with a default
    parametrization adapted to Autogrid 5 meshes.

    Parameters
    ----------

        mesh : :py:class:`str` or PyTree
            Name of the CGNS mesh file from Autogrid 5 or already read PyTree.

        scale : float
            Homothety factor to apply on the mesh. Default is 1.

        rotation : :py:class:'str' or :py:class:`list`
            List of rotations to apply on the mesh. If **rotation** =
            ``fromAG5``, then default rotations are applied:

            >>> rotation = [((0,0,0), (0,1,0), 90), ((0,0,0), (1,0,0), 90)]

            Else, **rotation** must be a list of rotation to apply to the grid
            component. Each rotation is defined by 3 elements:

                * a 3-tuple corresponding to the center coordinates

                * a 3-tuple corresponding to the rotation axis

                * a float (or integer) defining the angle of rotation in
                  degrees

        tol : float
            Tolerance for connectivities matching (including periodic connectivities).

        PeriodicTranslation : :py:obj:'None' or :py:class:`list` of :py:class:`float`
            If not :py:obj:'None', the configuration is considered to be with
            a periodicity in the direction **PeriodicTranslation**. This argument
            has to be used for linear cascade configurations.

    Returns
    -------

        InputMeshes : list

    '''

    if isinstance(mesh,str):
        t = J.load(mesh)
    elif I.isTopTree(mesh):
        t = mesh
    else:
        raise ValueError('parameter mesh must be either a filename or a PyTree')

    if not I.getNodeFromName(t, 'BladeNumber'):
        if PeriodicTranslation is None:
            MSG = 'There must be a BladeNumber node for each row Family. '
            MSG += 'Otherwise, the option PeriodicTranslation must not be None '
            MSG += 'to indicate a configuration with a periodicity by translation'
            raise Exception(J.FAIL + MSG + J.ENDC)
        angles = []

    if rotation == 'fromAG5':
        rotation = [((0,0,0), (0,1,0), 90),((0,0,0), (1,0,0), 90)]

    InputMeshes = [dict(
                    baseName=I.getName(I.getNodeByType(t, 'CGNSBase_t')),
                    Transform=dict(scale=scale, rotate=rotation),
                    Connection=[dict(type='Match', tolerance=tol)],
                    )]
    # Set automatic periodic connections
    InputMesh = InputMeshes[0]
    if I.getNodeFromName(t, 'BladeNumber'):
        BladeNumberList = [I.getValue(bn) for bn in I.getNodesFromName(t, 'BladeNumber')]
        angles = list(set([360./float(bn) for bn in BladeNumberList]))
        for angle in angles:
            print('  angle = {:g} deg ({} blades)'.format(angle, int(360./angle)))
            InputMesh['Connection'].append(
                    dict(type='PeriodicMatch', tolerance=tol, rotationAngle=[angle,0.,0.])
                    )
    if PeriodicTranslation:
        print('  translation = {} m'.format(PeriodicTranslation))
        InputMesh['Connection'].append(
                dict(type='PeriodicMatch', tolerance=tol, translation=PeriodicTranslation)
                )

    return InputMeshes

def cleanMeshFromAutogrid(t, basename='Base#1', zonesToRename={}):
    '''
    Clean a CGNS mesh from Autogrid 5.
    The sequence of operations performed are the following:

    #. remove useless nodes specific to AG5
    #. rename base
    #. rename zones
    #. clean Joins & Periodic Joins
    #. clean Rotor/Stator interfaces
    #. join HUB and SHROUD families

    Parameters
    ----------

        t : PyTree
            CGNS mesh from Autogrid 5

        basename: str
            Name of the base. Will replace the default AG5 name.

        zonesToRename : dict
            Each key corresponds to the name of a zone to modify, and the associated
            value is the new name to give.

    Returns
    -------

        t : PyTree
            modified mesh tree

    '''

    I._rmNodesByName(t, 'Numeca*')
    I._rmNodesByName(t, 'blockName')
    I._rmNodesByName(t, 'meridional_base')
    I._rmNodesByName(t, 'tools_base')

    # Clean Names
    # - Recover BladeNumber and Clean Families
    for fam in I.getNodesFromType(t, 'Family_t'):
        I._rmNodesByName(fam, 'RotatingCoordinates')
        I._rmNodesByName(fam, 'Periodicity')
        I._rmNodesByName(fam, 'DynamicData')
    I._rmNodesByName(t, 'FamilyProperty')

    # - Rename base
    base = I.getNodeFromType(t, 'CGNSBase_t')
    I.setName(base, basename)

    # - Rename Zones
    for zone in I.getNodesFromType(t, 'Zone_t'):
        name = I.getName(zone)
        if name in zonesToRename:
            newName = zonesToRename[name]
            print("Zone {} is renamed: {}".format(name,newName))
            I._renameNode(t, name, newName)
            continue
        # Delete some usual patterns in AG5
        new_name = name
        for pattern in ['_flux_1', '_flux_2', '_flux_3', '_Main_Blade']:
            new_name = new_name.replace(pattern, '')
        I._renameNode(t, name, new_name)

    # Clean Joins & Periodic Joins
    I._rmNodesByType(t, 'ZoneGridConnectivity_t')
    periodicFamilyNames = [I.getName(fam) for fam in I.getNodesFromType(t, "Family_t") if 'PER' in I.getName(fam)]

    for fname in periodicFamilyNames:
        # print('|- delete PeriodicBC family of name {}'.format(name))
        C._rmBCOfType(t, 'FamilySpecified:%s'%fname)
        fbc = I.getNodeFromName2(t, fname)
        I.rmNode(t, fbc)

    # Clean RS interfaces
    I._rmNodesByType(t,'InterfaceType')
    I._rmNodesByType(t,'DonorFamily')

    # Join HUB and SHROUD families
    J.joinFamilies(t, 'HUB')
    J.joinFamilies(t, 'SHROUD')
    return t

def convert2Unstructured(t, merge=True, tol=1e-6):
    '''
    Same that :func:`MOLA.Preprocess.convert2Unstructured`
    '''
    return PRE.convert2Unstructured(t, merge, tol)

def duplicate(tree, rowFamily, nBlades, nDupli=None, merge=False, axis=(1,0,0),
    verbose=1, container='FlowSolution#Init',
    vectors2rotate=[['VelocityX','VelocityY','VelocityZ'],['MomentumX','MomentumY','MomentumZ']]):
    '''
    Duplicate **nDupli** times the domain attached to the family **rowFamily**
    around the axis of rotation.

    Parameters
    ----------

        tree : PyTree
            tree to modify

        rowFamily : str
            Name of the CGNS family attached to the row domain to Duplicate

        nBlades : int
            Number of blades in the row. Used to compute the azimuthal length of
            a blade sector.

        nDupli : int
            Number of duplications to make

            .. warning:: This is the number of duplication of the input mesh
                domain, not the wished number of simulated blades. Keep this
                point in mind if there is already more than one blade in the
                input mesh.

        merge : bool
            if :py:obj:`True`, merge all the blocks resulting from the
            duplication.

            .. tip:: This option is useful is the mesh is to split and if a
                globborder will be defined on a BC of the duplicated domain. It
                allows the splitting procedure to provide a 'matricial' ordering
                (see `elsA Tutorial about globborder <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/globborder.html>`_)

        axis : tuple
            axis of rotation given as a 3-tuple of integers or floats

        verbose : int
            level of verbosity:

                * 0: no print

                * 1: print the number of duplications for row **rowFamily** and
                  the total number of blades.

                * 2: print also the name of all duplicated zones

        container : str
            Name of the FlowSolution container to rotate. Default is 'FlowSolution#Init'

        vectors2rotate : :py:class:`list` of :py:class:`list` of :py:class:`str`
            list of vectors to rotate. Each vector is a list of three strings,
            corresponding to each components.
            The default value is:

            >>> vectors2rotate = [['VelocityX','VelocityY','VelocityZ'],
            >>>                   ['MomentumX','MomentumY','MomentumZ']]

            .. note:: 
            
                Rotation of vectors is done with Cassiopee function Transform.rotate. 
                However, it is not useful to put the prefix 'centers:'. It will be 
                added automatically in the function.

    '''
    OLD_FlowSolutionCenters = I.__FlowSolutionCenters__
    I.__FlowSolutionCenters__ = container

    if nDupli is None:
        nDupli = nBlades # for a 360 configuration
    if nDupli == nBlades:
        if verbose>0: print('Duplicate {} over 360 degrees ({} blades in row)'.format(rowFamily, nBlades))
    else:
        if verbose>0: print('Duplicate {} on {} blades ({} blades in row)'.format(rowFamily, nDupli, nBlades))

    check = False
    vectors = []
    for vec in vectors2rotate:
        vectors.append(vec)
        vectors.append(['centers:'+v for v in vec])

    if I.getType(tree) == 'CGNSBase_t':
        bases = [tree]
    else:
        bases = I.getBases(tree)

    for base in bases:
        for zone in I.getZones(base):
            zone_name = I.getName(zone)
            FamilyNameNode = I.getNodeFromName1(zone, 'FamilyName')
            if not FamilyNameNode: continue
            zone_family = I.getValue(FamilyNameNode)
            if zone_family == rowFamily:
                if verbose>1: print('  > zone {}'.format(zone_name))
                check = True
                zones2merge = [zone]
                for n in range(nDupli-1):
                    ang = 360./nBlades*(n+1)
                    rot = T.rotate(I.copyNode(zone),(0.,0.,0.), axis, ang, vectors=vectors)
                    I.setName(rot, "{}_{}".format(zone_name, n+2))
                    I._addChild(base, rot)
                    zones2merge.append(rot)
                if merge:
                    for node in zones2merge:
                        I.rmNode(base, node)
                    tree_dist = T.merge(zones2merge, tol=1e-8)
                    for i, node in enumerate(I.getZones(tree_dist)):
                        I._addChild(base, node)
                        disk_block = I.getNodeFromName(base, I.getName(node))
                        disk_block[0] = '{}_{:02d}'.format(zone_name, i)
                        I.createChild(disk_block, 'FamilyName', 'FamilyName_t', value=rowFamily)
    if merge: PRE.autoMergeBCs(tree)

    I.__FlowSolutionCenters__ = OLD_FlowSolutionCenters
    assert check, 'None of the zones was duplicated. Check the name of row family'

def duplicateFlowSolution(t, TurboConfiguration):
    '''
    Duplicated the input PyTree **t**, already initialized.
    This function perform the following operations:

    #. Duplicate the mesh

    #. Initialize the different blade sectors by rotating the ``FlowSolution#Init``
       node available in the original sector(s)

    #. Update connectivities and periodic boundary conditions

    .. warning:: This function does not rotate vectors in BCDataSet nodes.

    Parameters
    ----------

        t : PyTree
            input tree already initialized, but before setting boundary conditions

        TurboConfiguration : dict
            dictionary as provided by :py:func:`getTurboConfiguration`

    Returns
    -------

        t : PyTree
            tree after duplication
    '''
    # Remove connectivities and periodic BCs
    I._rmNodesByType(t, 'GridConnectivity1to1_t')

    angles4ConnectMatchPeriodic = []
    for row, rowParams in TurboConfiguration['Rows'].items():
        nBlades = rowParams['NumberOfBlades']
        nDupli = rowParams['NumberOfBladesSimulated']
        nMesh = rowParams['NumberOfBladesInInitialMesh']
        if nDupli > nMesh:
            duplicate(t, row, nBlades, nDupli=nDupli, axis=(1,0,0))

        angle = 360. / nBlades * nDupli
        if not np.isclose(angle, 360.):
            angles4ConnectMatchPeriodic.append(angle)

    # Connectivities
    X.connectMatch(t, tol=1e-8)
    for angle in angles4ConnectMatchPeriodic:
        # Not full 360 simulation: periodic BC must be restored
        t = X.connectMatchPeriodic(t, rotationAngle=[angle, 0., 0.], tol=1e-8)

    # WARNING: Names of BC_t nodes must be unique to use PyPart on globborders
    for l in [2,3,4]: I._correctPyTree(t, level=l)

    return t

def computeAzimuthalExtensionFromFamily(t, FamilyName):
    '''
    Compute the azimuthal extension in radians of the mesh **t** for the row **FamilyName**.

    .. warning:: This function needs to calculate the surface of the slice in X
                 at Xmin + 5% (Xmax - Xmin). If this surface is crossed by a
                 solid (e.g. a blade) or by the inlet boundary, the function
                 will compute a wrong value of the number of blades inside the
                 mesh.

    Parameters
    ----------

        t : PyTree
            mesh tree

        FamilyName : str
            Name of the row, identified by a ``FamilyName``.

    Returns
    -------

        deltaTheta : float
            Azimuthal extension in radians

    '''
    # Extract zones in family
    zonesInFamily = C.getFamilyZones(t, FamilyName)
    # Slice in x direction at middle range
    xmin = C.getMinValue(zonesInFamily, 'CoordinateX')
    xmax = C.getMaxValue(zonesInFamily, 'CoordinateX')
    sliceX = P.isoSurfMC(zonesInFamily, 'CoordinateX', value=xmin+0.05*(xmax-xmin))
    # Compute Radius
    C._initVars(sliceX, '{Radius}=({CoordinateY}**2+{CoordinateZ}**2)**0.5')
    Rmin = C.getMinValue(sliceX, 'Radius')
    Rmax = C.getMaxValue(sliceX, 'Radius')
    # Compute surface
    SurfaceTree = C.convertArray2Tetra(sliceX)
    SurfaceTree = C.initVars(SurfaceTree, 'ones=1')
    Surface = P.integ(SurfaceTree, var='ones')[0]
    # Compute deltaTheta
    deltaTheta = 2* Surface / (Rmax**2 - Rmin**2)
    return deltaTheta

def getNumberOfBladesInMeshFromFamily(t, FamilyName, NumberOfBlades):
    '''
    Compute the number of blades for the row **FamilyName** in the mesh **t**.

    .. warning:: This function needs to calculate the surface of the slice in X
                 at Xmin + 5% (Xmax - Xmin). If this surface is crossed by a
                 solid (e.g. a blade) or by the inlet boundary, the function
                 will compute a wrong value of the number of blades inside the
                 mesh.

    Parameters
    ----------

        t : PyTree
            mesh tree

        FamilyName : str
            Name of the row, identified by a ``FamilyName``.

        NumberOfBlades : int
            Number of blades of the row **FamilyName** on 360 degrees.

    Returns
    -------

        Nb : int
            Number of blades in **t** for row **FamilyName**

    '''
    deltaTheta = computeAzimuthalExtensionFromFamily(t, FamilyName)
    # Compute number of blades in the mesh
    Nb = NumberOfBlades * deltaTheta / (2*np.pi)
    Nb = int(np.round(Nb))
    print('Number of blades in initial mesh for {}: {}'.format(FamilyName, Nb))
    return Nb

def computeReferenceValues(FluidProperties, PressureStagnation,
                           TemperatureStagnation, Surface, MassFlow=None, Mach=None,
                           YawAxis=[0.,0.,1.], PitchAxis=[0.,1.,0.],
                           TurbulenceCutoff=1e-8,
                           FieldsAdditionalExtractions=[
                                'ViscosityMolecular',
                                'Viscosity_EddyMolecularRatio',
                                'Pressure',
                                'Temperature',
                                'PressureStagnation',
                                'TemperatureStagnation',
                                'Mach',
                                'Entropy'],
                           BCExtractions=dict(
                             BCWall = ['normalvector', 'frictionvector','psta', 'bl_quantities_2d', 'yplusmeshsize'],
                             BCInflow = ['convflux_ro'],
                             BCOutflow = ['convflux_ro'],
                             BCGilesMxPl = ['convflux_ro']),
                            **kwargs):
    '''
    This function is the Compressor's equivalent of :func:`MOLA.Preprocess.computeReferenceValues`.
    The main difference is that in this case reference values are set through
    ``MassFlow``, total Pressure ``PressureStagnation``, total Temperature
    ``TemperatureStagnation`` and ``Surface``.

    You can also give the Mach number instead of massflow (but not both).

    Please, refer to :func:`MOLA.Preprocess.computeReferenceValues` doc for more details.
    '''
    ASSERTION_ERR = 'For this workflow, you must provide PressureStagnation, TemperatureStagnation and MassFlow (or Mach). '
    ASSERTION_ERR+= 'You cannot provide Density, Temperature and Velocity'
    assert all([not var in kwargs for var in ['Density', 'Temperature', 'Velocity']]), \
        J.FAIL + ASSERTION_ERR + J.ENDC
    
    # Fluid properties local shortcuts
    Gamma   = FluidProperties['Gamma']
    IdealGasConstant = FluidProperties['IdealGasConstant']
    cv      = FluidProperties['cv']
    cp      = FluidProperties['cp']

    # Compute variables
    assert not(MassFlow and Mach), 'MassFlow and Mach cannot be given together in ReferenceValues. Choose one'
    if MassFlow:
        Mach  = machFromMassFlow(MassFlow, Surface, Pt=PressureStagnation,
                                Tt=TemperatureStagnation, r=IdealGasConstant,
                                gamma=Gamma)
    else:
        MassFlow  = massflowFromMach(Mach, Surface, Pt=PressureStagnation,
                            Tt=TemperatureStagnation, r=IdealGasConstant,
                            gamma=Gamma)
    Temperature  = TemperatureStagnation / (1. + 0.5*(Gamma-1.) * Mach**2)
    Pressure  = PressureStagnation / (1. + 0.5*(Gamma-1.) * Mach**2)**(Gamma/(Gamma-1))
    Density = Pressure / (Temperature * IdealGasConstant)
    SoundSpeed  = np.sqrt(Gamma * IdealGasConstant * Temperature)
    Velocity  = Mach * SoundSpeed

    # REFERENCE VALUES COMPUTATION
    mus = FluidProperties['SutherlandViscosity']
    Ts  = FluidProperties['SutherlandTemperature']
    S   = FluidProperties['SutherlandConstant']
    ViscosityMolecular = mus * (Temperature/Ts)**1.5 * ((Ts + S)/(Temperature + S))

    # if not 'AveragingIterations' in CoprocessOptions:
    #     CoprocessOptions['AveragingIterations'] = 1000

    TurboStatistics = ['rsd-{}'.format(var) for var in ['MassFlowIn', 'MassFlowOut',
        'PressureStagnationRatio', 'TemperatureStagnationRatio', 'EfficiencyIsentropic',
        'PressureStagnationLossCoeff']]
    
    try: 
        CoprocessOptions = kwargs.pop('CoprocessOptions')
    except KeyError:
        CoprocessOptions = dict()
    try:
        RequestedStatistics = CoprocessOptions['RequestedStatistics']
        for stat in TurboStatistics:
            if stat not in CoprocessOptions:
                RequestedStatistics.append( stat )
    except KeyError:
        CoprocessOptions['RequestedStatistics'] = TurboStatistics

    CoprocessOptions.setdefault('BodyForceComputeFrequency', 1)

    ReferenceValues = PRE.computeReferenceValues(FluidProperties,
        Density=Density,
        Velocity=Velocity,
        Temperature=Temperature,
        Surface=Surface,
        CoprocessOptions=CoprocessOptions,
        FieldsAdditionalExtractions=FieldsAdditionalExtractions,
        BCExtractions=BCExtractions,
        YawAxis=YawAxis,
        PitchAxis=PitchAxis,
        TurbulenceCutoff=TurbulenceCutoff,
        **kwargs)

    ReferenceValues.update(
        dict(
            PressureStagnation    = PressureStagnation,
            TemperatureStagnation = TemperatureStagnation,
            MassFlow              = MassFlow,
        )
    )

    return ReferenceValues

def computeFluxCoefByRow(t, ReferenceValues, TurboConfiguration):
    '''
    Compute the parameter **FluxCoef** for boundary conditions (except wall BC)
    and rotor/stator intefaces (``GridConnectivity_t`` nodes).
    **FluxCoef** will be used later to normalize the massflow.

    Modify **ReferenceValues** by adding:

    >>> ReferenceValues['NormalizationCoefficient'][<FamilyName>]['FluxCoef'] = FluxCoef

    for <FamilyName> in the list of BC families, except families of type 'BCWall*'.

    Parameters
    ----------

        t : PyTree
            Mesh tree with boudary conditions families, with a BCType.

        ReferenceValues : dict
            as produced by :py:func:`computeReferenceValues`

        TurboConfiguration : dict
            as produced by :py:func:`getTurboConfiguration`

    '''
    for zone in I.getZones(t):
        FamilyNode = I.getNodeFromType1(zone, 'FamilyName_t')
        if FamilyNode is None:
            continue
        if 'PeriodicTranslation' in TurboConfiguration:
            fluxcoeff = 1.
        else:
            row = I.getValue(FamilyNode)
            try:
                rowParams = TurboConfiguration['Rows'][row]
                fluxcoeff = rowParams['NumberOfBlades'] / float(rowParams['NumberOfBladesSimulated'])
            except KeyError:
                # since a FamilyNode does not necessarily belong to a row
                fluxcoeff = 1.

        for bc in I.getNodesFromType2(zone, 'BC_t')+I.getNodesFromType2(zone, 'GridConnectivity_t'):
            FamilyNameNode = I.getNodeFromType1(bc, 'FamilyName_t')
            if FamilyNameNode is None:
                continue
            FamilyName = I.getValue(FamilyNameNode)
            BCType = PRE.getFamilyBCTypeFromFamilyBCName(t, FamilyName)
            if BCType is None or 'BCWall' in BCType:
                continue
            if not 'NormalizationCoefficient' in ReferenceValues:
                ReferenceValues['NormalizationCoefficient'] = dict()
            ReferenceValues['NormalizationCoefficient'][FamilyName] = dict(FluxCoef=fluxcoeff)

def getTurboConfiguration(t, ShaftRotationSpeed=0., HubRotationSpeed=[], Rows={},
    PeriodicTranslation=None, BodyForceInputData=[]):
    '''
    Construct a dictionary concerning the compressor properties.

    Parameters
    ----------

        t : PyTree
            input tree

        ShaftRotationSpeed : :py:class:`float`
            Shaft speed in rad/s

            .. attention:: only for single shaft configuration

            .. attention:: Pay attention to the sign of **ShaftRotationSpeed**

        HubRotationSpeed : :py:class:`list` of :py:class:`tuple`
            Hub rotation speed. Each tuple (``xmin``, ``xmax``) corresponds to a
            ``CoordinateX`` interval where the speed at hub wall is
            **ShaftRotationSpeed**. It is zero outside these intervals.

        Rows : :py:class:`dict`
            This dictionary has one entry for each row domain. The key names
            must be the family names in the CGNS Tree.
            For each family name, the following entries are expected:

                * RotationSpeed : :py:class:`float` or :py:class:`str`
                    Rotation speed in rad/s. Set ``'auto'`` to automatically
                    set **ShaftRotationSpeed**.

                    .. attention::
                        Use **RotationSpeed** = ``'auto'`` for rotors only.

                    .. attention::
                        Pay attention to the sign of **RotationSpeed**

                * NumberOfBlades : :py:class:`int`
                    The number of blades in the row

                * NumberOfBladesSimulated : :py:class:`int`
                    The wanted number of blades in the computational domain at
                    the end of the set up process. Set to **NumberOfBlades** for
                    a full 360 simulation.
                    If not given, the default value is 1.

                * NumberOfBladesInInitialMesh : :py:class:`int`
                    The number of blades in the provided mesh ``mesh.cgns``.
                    If not given, it is computed automatically.

                * InletPlane : :py:class:`float`, optional
                    Position (in ``CoordinateX``) of the inlet plane for this
                    row. This plane is used for post-processing and convergence
                    monitoring.

                * OutletPlane : :py:class:`float`, optional
                    Position of the outlet plane for this row.

        PeriodicTranslation : :py:obj:'None' or :py:class:`list` of :py:class:`float`
            If not :py:obj:'None', the configuration is considered to be with
            a periodicity in the direction **PeriodicTranslation**. This argument
            has to be used for linear cascade configurations.
        
        BodyForceInputData : list
            see :py:func:`prepareMainCGNS4ElsA`

    Returns
    -------

        TurboConfiguration : :py:class:`dict`
            set of compressor properties
    '''
    if PeriodicTranslation:
        TurboConfiguration = dict(
            PeriodicTranslation = PeriodicTranslation,
            Rows                = Rows
            )
    else:
        TurboConfiguration = dict(
            ShaftRotationSpeed = ShaftRotationSpeed,
            HubRotationSpeed   = HubRotationSpeed,
            Rows               = Rows
            )
        for row, rowParams in TurboConfiguration['Rows'].items():
            for key, value in rowParams.items():
                if key == 'RotationSpeed' and value == 'auto':
                    rowParams[key] = ShaftRotationSpeed
                ERR_MSG = f'The key RotationSpeed is not found for the Family {row}'
                assert 'RotationSpeed' in rowParams, J.FAIL+ERR_MSG+J.ENDC
            for BodyForceComponent in BodyForceInputData:
                if row == BodyForceComponent['Family']:
                    # Replace the number of blades to be consistant with the body-force mesh
                    deltaTheta = computeAzimuthalExtensionFromFamily(t, row)
                    rowParams['NumberOfBlades'] = int(2*np.pi / deltaTheta)
                    rowParams['NumberOfBladesInInitialMesh'] = 1
                    print(f'Number of blades for {row}: {rowParams["NumberOfBlades"]} (got from the body-force mesh)')
            if not 'NumberOfBladesSimulated' in rowParams:
                rowParams['NumberOfBladesSimulated'] = 1
            if not 'NumberOfBladesInInitialMesh' in rowParams:
                rowParams['NumberOfBladesInInitialMesh'] = getNumberOfBladesInMeshFromFamily(t, row, rowParams['NumberOfBlades'])
    return TurboConfiguration

def getReferenceSurface(t, BoundaryConditions, TurboConfiguration):
    '''
    Compute the reference surface (**Surface** parameter in **ReferenceValues**
    :py:class:`dict`) from the inflow family.

    Parameters
    ----------

        t : PyTree
            Input tree

        BoundaryConditions : list
            Boundary conditions to set on the given mesh,
            as given to :py:func:`prepareMainCGNS4ElsA`.

        TurboConfiguration : dict
            Compressor properties, as given to :py:func:`prepareMainCGNS4ElsA`.

    Returns
    -------

        Surface : float
            Reference surface
    '''
    # Get inflow BCs
    InflowBCs = [bc for bc in BoundaryConditions \
        if bc['type'] == 'InflowStagnation' or bc['type'].startswith('inj') or bc['type'] == 'InflowGiles']
    # Check unicity
    if len(InflowBCs) != 1:
        MSG = 'Please provide a reference surface as "Surface" in '
        MSG += 'ReferenceValues or provide a unique inflow BC in BoundaryConditions'
        raise Exception(J.FAIL + MSG + J.ENDC)
    # Compute surface of the inflow BC
    InflowFamily = InflowBCs[0]['FamilyName']
    zones = C.extractBCOfName(t, 'FamilySpecified:'+InflowFamily)
    SurfaceTree = C.convertArray2Tetra(zones)
    SurfaceTree = C.initVars(SurfaceTree, 'ones=1')
    Surface = P.integ(SurfaceTree, var='ones')[0]
    if 'PeriodicTranslation' not in TurboConfiguration:
        # Compute normalization coefficient
        zoneName = I.getName(zones[0]).split('/')[0]
        zone = I.getNodeFromName2(t, zoneName)
        row = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
        rowParams = TurboConfiguration['Rows'][row]
        fluxcoeff = rowParams['NumberOfBlades'] / float(rowParams['NumberOfBladesInInitialMesh'])
        # Compute reference surface
        Surface *= fluxcoeff
    print('Reference surface = {} m^2 (computed from family {})'.format(Surface, InflowFamily))

    return Surface

def addMonitoredRowsInExtractions(Extractions, TurboConfiguration):
    '''
    Get the positions of inlet and outlet planes for each row (identified by
    keys **InletPlane** and **OutletPlane** in the row sub-dictionary of
    **TurboConfiguration**) and add them to **Extractions**.

    Parameters
    ----------

        Extractions : list
            List of extractions, each of them being a dictionary.

        TurboConfiguration : dict
            Compressor properties, as given to :py:func:`prepareMainCGNS4ElsA`.

    '''
    # Get the positions of inlet and outlet planes for each row
    # and add them to Extractions
    for row, rowParams in TurboConfiguration['Rows'].items():
        for plane in ['InletPlane', 'OutletPlane']:
            if plane in rowParams:
                planeAlreadyInExtractions = False
                for extraction in Extractions:
                    if extraction['type'] == 'IsoSurface' \
                        and extraction['field'] == 'CoordinateX' \
                        and np.isclose(extraction['value'], rowParams[plane]):
                        planeAlreadyInExtractions = True
                        extraction.update(dict(ReferenceRow=row, tag=plane))
                        break
                if not planeAlreadyInExtractions:
                    Extractions.append(dict(type='IsoSurface', field='CoordinateX', \
                        value=rowParams[plane], ReferenceRow=row, tag=plane, family=row))

def computeDistance2Walls(t, WallFamilies=[], verbose=True, wallFilename=None):
    '''
    Identical to :func:`MOLA.Preprocess.computeDistance2Walls`, except that the
    list **WallFamilies** is automatically filled with with the following
    patterns:
    'WALL', 'HUB', 'SHROUD', 'BLADE', 'MOYEU', 'CARTER', 'AUBE'.
    Names are not case-sensitive (automatic conversion to lower, uper and
    capitalized cases). Others patterns might be added with the argument
    **WallFamilies**.
    '''
    WallFamilies += ['WALL', 'HUB', 'SHROUD', 'BLADE', 'MOYEU', 'CARTER', 'AUBE']
    PRE.computeDistance2Walls(t, WallFamilies=WallFamilies, verbose=verbose, wallFilename=wallFilename)

def massflowFromMach(Mx, S, Pt=101325.0, Tt=288.25, r=287.053, gamma=1.4):
    '''
    Compute the massflow rate through a section.

    Parameters
    ----------

        Mx : :py:class:`float`
            Mach number in the normal direction to the section.

        S : :py:class:`float`
            Surface of the section.

        Pt : :py:class:`float`
            Stagnation pressure of the flow.

        Tt : :py:class:`float`
            Stagnation temperature of the flow.

        r : :py:class:`float`
            Specific gas constant.

        gamma : :py:class:`float`
            Ratio of specific heats of the gas.


    Returns
    -------

        massflow : :py:class:`float`
            Value of massflow through the section.
    '''
    return S * Pt * (gamma/r/Tt)**0.5 * Mx / (1. + 0.5*(gamma-1.) * Mx**2) ** ((gamma+1) / 2 / (gamma-1))

def machFromMassFlow(massflow, S, Pt=101325.0, Tt=288.25, r=287.053, gamma=1.4):
    '''
    Compute the Mach number normal to a section from the massflow rate.

    Parameters
    ----------

        massflow : :py:class:`float`
            MassFlow rate through the section.

        S : :py:class:`float`
            Surface of the section.

        Pt : :py:class:`float`
            Stagnation pressure of the flow.

        Tt : :py:class:`float`
            Stagnation temperature of the flow.

        r : :py:class:`float`
            Specific gas constant.

        gamma : :py:class:`float`
            Ratio of specific heats of the gas.


    Returns
    -------

        Mx : :py:class:`float`
            Value of the Mach number in the normal direction to the section.
    '''
    if isinstance(massflow, (list, tuple, np.ndarray)):
        Mx = []
        for i, MF in enumerate(massflow):
            Mx.append(machFromMassFlow(MF, S, Pt=Pt, Tt=Tt, r=r, gamma=gamma))
        if isinstance(massflow, np.ndarray):
            Mx = np.array(Mx)
        return Mx
    else:
        # Check that massflow is lower than the chocked massflow
        chocked_massflow = massflowFromMach(1., S, Pt=Pt, Tt=Tt, r=r, gamma=gamma)
        assert massflow < chocked_massflow, "MassFlow ({:6.3f}kg/s) is greater than the chocked massflow ({:6.3f}kg/s)".format(massflow, chocked_massflow)
        # MassFlow as a function of Mach number
        f = lambda Mx: massflowFromMach(Mx, S, Pt, Tt, r, gamma)
        # Objective function
        g = lambda Mx: f(Mx) - massflow
        # Search for the corresponding Mach Number between 0 and 1
        Mx = scipy.optimize.brentq(g, 0, 1)
        return Mx

def setMotionForRowsFamilies(t, TurboConfiguration):
    '''
    Set the rotation speed for all families related to row domains. It is defined in:

        >>> TurboConfiguration['Rows'][rowName]['RotationSpeed'] = float

    Parameters
    ----------

        t : PyTree
            Tree to modify

        TurboConfiguration : dict
            as produced :py:func:`getTurboConfiguration`
            
    '''
    # Add info on row movement (.Solver#Motion)
    for row, rowParams in TurboConfiguration['Rows'].items():
        famNode = I.getNodeFromNameAndType(t, row, 'Family_t')
        try: 
            omega = float(rowParams['RotationSpeed'])
        except KeyError:
            # No RotationSpeed --> zones attached to this family are not moving
            continue

        # Test if zones in that family are modelled with Body Force
        for zone in C.getFamilyZones(t, row):
            if I.getNodeFromName1(zone, 'FlowSolution#DataSourceTerm'):
                # If this node is present, body force is used
                # Then the frame of this row must be the absolute frame
                omega = 0.
                break
        
        print(f'setting .Solver#Motion at family {row} (omega={omega}rad/s)')
        if famNode is None:
            MSG = 'did not find family node for row\n'
            MSG+= str(row)
            MSG+= '\ncheck debug.cgns'
            try: C.convertPyTree2File(t,'debug.cgns')
            except: pass
            raise Exception(J.FAIL+MSG+J.ENDC)
        J.set(famNode, '.Solver#Motion',
                motion='mobile',
                omega=omega,
                axis_pnt_x=0., axis_pnt_y=0., axis_pnt_z=0.,
                axis_vct_x=1., axis_vct_y=0., axis_vct_z=0.)

def setZoneParamForPeriodicDistanceByRowsFamilies(t, TurboConfiguration):
    '''
    Set the zone parameters for all families related to row domains if periodicity in accounting for in wall distance computation.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        TurboConfiguration : dict
            as produced :py:func:`getTurboConfiguration`

    '''
    # Add info on Zone_t nodes (.Solver#Param)
    for row, rowParams in TurboConfiguration['Rows'].items():
        for zone in C.getFamilyZones(t, row):
            elsAProfile._addPeriodicDataInSolverParam(zone,rotationAngle=[360./rowParams['NumberOfBlades'],0.,0.],NAzimutalSectors=rowParams['NumberOfBlades'])
            axis_ang2 = I.getNodeFromNameAndType(zone,'axis_ang_2','DataArray_t') # always 1 even in case of duplicated configuration
            I.setValue(axis_ang2,rowParams['NumberOfBladesSimulated'])

def setBCFamilyParamForPeriodicDistance(t, ReferenceValues,
                                        bladeFamilyNames=['BLADE','AUBE'],
                                        hubFamilyNames=['HUB', 'SPINNER', 'MOYEU'],
                                        shroudFamilyNames=['SHROUD', 'CARTER']):
    '''
    Set the BC family parameters for all families related to BCWall* conditions if periodicity in accounting for in wall distance computation.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        ReferenceValues : dict
            as produced by :py:func:`computeReferenceValues`

        bladeFamilyNames : :py:class:`list` of :py:class:`str`
            list of patterns to find families related to blades.

        hubFamilyNames : :py:class:`list` of :py:class:`str`
            list of patterns to find families related to hub.

        shroudFamilyNames : :py:class:`list` of :py:class:`str`
            list of patterns to find families related to shroud.

    '''
    # Add info on Family_t nodes (in .Solver#BC)
    bladeFamilyNames = PRE.extendListOfFamilies(bladeFamilyNames)
    hubFamilyNames = PRE.extendListOfFamilies(hubFamilyNames)
    shroudFamilyNames = PRE.extendListOfFamilies(shroudFamilyNames)

    # print('setBCFamilyParamForPeriodicDistance')
    for family in bladeFamilyNames+hubFamilyNames+shroudFamilyNames:
        for famNode in I.getNodesFromNameAndType(t, '*{}*'.format(family), 'Family_t'):
            famName = I.getName(famNode)
            famBC = I.getNodeFromType1(famNode,'FamilyBC_t')
            if 'BCWall' in I.getValue(famBC) or 'UserDefined' in I.getValue(famBC):
                # I.printTree(famNode)
                solver_bc_data = I.getNodeFromName(famNode,'.Solver#BC')
                if not solver_bc_data: # does not exists
                    solver_bc_data = I.newUserDefinedData(name='.Solver#BC', value=None, parent=famNode)
                I.newDataArray('pangle', value=1, parent=solver_bc_data)
                I.newDataArray('ptype', value='rot2dir', parent=solver_bc_data)

################################################################################
################# Boundary Conditions Settings  ################################
################################################################################

def setBoundaryConditions(t, BoundaryConditions, TurboConfiguration,
    FluidProperties, ReferenceValues, bladeFamilyNames=['BLADE','AUBE']):
    '''
    Set all BCs defined in the dictionary **BoundaryConditions**.

    .. important::

        Wall BCs are defined automatically given the dictionary **TurboConfiguration**


    Parameters
    ----------

        t : PyTree
            preprocessed tree as performed by :py:func:`prepareMesh4ElsA`

        BoundaryConditions : :py:class:`list` of :py:class:`dict`
            User-provided list of boundary conditions. Each element is a
            dictionary with the following keys:

                * FamilyName :
                    Name of the family on which the boundary condition will be imposed

                * type :
                  BC type among the following:

                  * Farfield

                  * InflowStagnation

                  * InflowMassFlow

                  * OutflowPressure

                  * OutflowMassFlow

                  * OutflowRadialEquilibrium

                  * MixingPlane

                  * UnsteadyRotorStatorInterface

                  * ChorochronicInterface

                  * WallViscous

                  * WallViscousIsothermal

                  * WallInviscid

                  * SymmetryPlane

                  .. note::

                    elsA names are also available (``nref``, ``inj1``, ``injfmr1``,
                    ``outpres``, ``outmfr2``, ``outradeq``,
                    ``stage_mxpl``, ``stage_red``,
                    ``walladia``, ``wallisoth``, ``wallslip``,
                    ``sym``)

                * option (optional) : add a specification for type
                  InflowStagnation (could be 'uniform' or 'file')

                * other keys depending on type. They will be passed as an
                  unpacked dictionary of arguments to the BC type-specific
                  function.

        TurboConfiguration : dict
            as produced by :py:func:`getTurboConfiguration`

        FluidProperties : dict
            as produced by :py:func:`computeFluidProperties`

        ReferenceValues : dict
            as produced by :py:func:`computeReferenceValues`

        bladeFamilyNames : :py:class:`list` of :py:class:`str`
            list of patterns to find families related to blades.

    See also
    --------

    setBC_Walls, setBC_walladia, setBC_wallisoth, setBC_wallslip, setBC_sym,
    setBC_nref,
    setBC_inj1, setBC_inj1_uniform, setBC_inj1_imposeFromFile,
    setBC_outpres, setBC_outmfr2,
    setBC_outradeq, setBC_outradeqhyb,
    setBC_stage_mxpl, setBC_stage_mxpl_hyb,
    setBC_stage_red, setBC_stage_red_hyb,
    setBC_stage_choro, setBC_stage_choro_hyb,
    setBCwithImposedVariables

    Examples
    --------

    The following list defines classical boundary conditions for a compressor
    stage:

    .. code-block:: python

        BoundaryConditions = [
            dict(type='InflowStagnation', option='uniform', FamilyName='row_1_INFLOW'),
            dict(type='OutflowRadialEquilibrium', FamilyName='row_2_OUTFLOW', valve_type=4, valve_ref_pres=0.75*Pt, valve_relax=0.3*Pt),
            dict(type='MixingPlane', left='Rotor_stator_10_left', right='Rotor_stator_10_right')
        ]

    Each type of boundary conditions currently available in MOLA is detailed below.

    **Wall boundary conditions**

    These BCs are automatically defined based on the rotation speeds in the
    :py:class:`dict` **TurboConfiguration**. There is a strong requirement on the
    names of families defining walls:

    * for the shroud: all family names must contain the pattern 'SHROUD' or 'CARTER'
      (in lower, upper or capitalized case)

    * for the hub: all family names must contain the pattern 'HUB' or 'MOYEU'
      (in lower, upper or capitalized case)

    * for the blades: all family names must contain the pattern 'BLADE' or 'AUBE'
      (in lower, upper or capitalized case). If names differ from that ones, it
      is still possible to give a list of patterns that are enought to find all
      blades (adding 'BLADE' or 'AUBE' if necessary). It is done with the
      argument **bladeFamilyNames** of :py:func:`prepareMainCGNS4ElsA`.

    If needed, these boundary conditions may be overwritten to impose other kinds
    of conditions. For instance, the following :py:class:`dict` may be used as
    an element of the :py:class:`list` **BoundaryConditions** to change the
    family 'SHROUD' into an inviscid wall:

    >>> dict(type='WallInviscid', FamilyName='SHROUD')

    The following py:class:`dict` change the family 'SHROUD' into a symmetry plane:

    >>> dict(type='SymmetryPlane', FamilyName='SHROUD')


    **Inflow boundary conditions**

    For the example, it is assumed that there is only one inflow family called
    'row_1_INFLOW'. The following types can be used as elements of the
    :py:class:`list` **BoundaryConditions**:

    >>> dict(type='Farfield', FamilyName='row_1_INFLOW')

    It defines a 'nref' condition based on the **ReferenceValues**
    :py:class:`dict`.

    >>> dict(type='InflowStagnation', FamilyName='row_1_INFLOW')

    It defines a uniform inflow condition imposing stagnation quantities ('inj1' in
    *elsA*) based on the **ReferenceValues**  and **FluidProperties**
    :py:class:`dict`. To impose values not based on **ReferenceValues**, additional
    optional parameters may be given (see the dedicated documentation for the function).

    >>> dict(type='InflowStagnation', FamilyName='row_1_INFLOW', PressureStagnation=100000.)

    Same that before but imposing a PressureStagnation different from that in **ReferenceValues**.

    >>> dict(type='InflowStagnation', FamilyName='row_1_INFLOW', PressureStagnation=funPt)

    Same that before but imposing a radial profile of PressureStagnation given by the function 'funPt'
    (must be defined before by the user). The function may be an analytical function or a interpoland 
    computed by the user from a data set. If not given, the function argument is the variable 'ChannelHeight'.
    Otherwise, the function argument has to be precised with the optional argument `variableForInterpolation`.
    It must be one of 'ChannelHeight' (default value), 'Radius', 'CoordinateX', 'CoordinateY' or 'CoordinateZ'.

    >>> dict(type='InflowStagnation', option='file', FamilyName='row_1_INFLOW', filename='inflow.cgns')

    It defines an inflow condition imposing stagnation quantities ('inj1' in
    *elsA*) given a 2D map written in the given file (must be given at cell centers, 
    in the container 'FlowSolution#Centers'). The flow field will be just copied, there is no 
    interpolation (if needed, the user has to done that before and provide a ready-to-copy file).

    >>> dict(type='InflowMassFlow', FamilyName='row_1_INFLOW')

    It defines a uniform inflow condition imposing the massflow ('inj1mfr1' in
    *elsA*) based on the **ReferenceValues**  and **FluidProperties**
    :py:class:`dict`. To impose values not based on **ReferenceValues**, additional
    optional parameters may be given (see the dedicated documentation for the function).
    In particular, either the massflow (``MassFlow``) or the surfacic massflow
    (``SurfacicMassFlow``) may be specified.

    **Outflow boundary conditions**

    For the example, it is assumed that there is only one outflow family called
    'row_2_OUTFLOW'. The following types can be used as elements of the
    :py:class:`list` **BoundaryConditions**:

    >>> dict(type='OutflowPressure', FamilyName='row_2_OUTFLOW', Pressure=20e3)

    It defines an outflow condition imposing a uniform static pressure ('outpres' in
    *elsA*).

    >>> dict(type='OutflowMassFlow', FamilyName='row_2_OUTFLOW', MassFlow=5.)

    It defines an MassFlow condition imposing the massflow ('outmfr2' in *elsA*).
    Be careful, **assFlow** should be the massflow through the given family BC
    *in the simulated domain* (not the 360 degrees configuration, except if it
    is simulated).
    If **MassFlow** is not given, the massflow given in the **ReferenceValues**
    is automatically taken and normalized by the appropriate section.

    >>> dict(type='OutflowRadialEquilibrium', FamilyName='row_2_OUTFLOW', valve_type=4, valve_ref_pres=0.75*Pt, valve_ref_mflow=5., valve_relax=0.3*Pt)

    It defines an outflow condition imposing a radial equilibrium ('outradeq' in
    *elsA*). The arguments have the same names that *elsA* keys. Valve law types
    from 1 to 5 are available. The radial equilibrium without a valve law (with
    **valve_type** = 0, which is the default value) is also available. To be
    consistant with the condition 'OutflowPressure', the argument
    **valve_ref_pres** may also be named **Pressure**.


    **Interstage boundary conditions**

    For the example, it is assumed that there is only one interstage with both
    families 'Rotor_stator_10_left' and 'Rotor_stator_10_right'. The following
    types can be used as elements of the :py:class:`list` **BoundaryConditions**:

    >>> dict(type='MixingPlane', left='Rotor_stator_10_left', right='Rotor_stator_10_right')

    It defines a mixing plane ('stage_mxpl' in *elsA*).

    >>> dict(type='UnsteadyRotorStatorInterface', left='Rotor_stator_10_left', right='Rotor_stator_10_right', stage_ref_time=1e-5)

    It defines an unsteady interpolating interface (RNA interface, 'stage_red'
    in *elsA*). If **stage_ref_time** is not provided, it is automatically
    computed assuming a 360 degrees rotor/stator interface:

    >>> stage_ref_time = 2*np.pi / abs(TurboConfiguration['ShaftRotationSpeed'])


    >>> dict(type='ChorochronicInterface', left='Rotor_stator_10_left', right='Rotor_stator_10_right', stage_ref_time=1e-5) 

    It defines a chorochronic  interface ('stage_choro' in *elsA*)

    '''
    PreferedBoundaryConditions = dict(
        Farfield                     = 'nref',
        InflowStagnation             = 'inj1',
        InflowMassFlow               = 'injmfr1',
        OutflowPressure              = 'outpres',
        OutflowMassFlow              = 'outmfr2',
        OutflowRadialEquilibrium     = 'outradeq',
        MixingPlane                  = 'stage_mxpl',
        GilesMixingPlane             = 'giles_stage_mxpl',
        UnsteadyRotorStatorInterface = 'stage_red',
        ChorochronicInterface        = 'stage_choro',
        WallViscous                  = 'walladia',
        WallViscousIsothermal        = 'wallisoth',
        WallInviscid                 = 'wallslip',
        SymmetryPlane                = 'sym',
        OutflowGiles                 = 'giles_out',
        InflowGiles                  = 'giles_in'
    )

    print(J.CYAN + 'set BCs at walls' + J.ENDC)
    setBC_Walls(t, TurboConfiguration, bladeFamilyNames=bladeFamilyNames)

    GilesMonitoringFlag = 1     # Initialization of index for monitoring flag with Giles BC

    for BCparam in BoundaryConditions:

        BCkwargs = {key:BCparam[key] for key in BCparam if key not in ['type', 'option']}
        if BCparam['type'] in PreferedBoundaryConditions:
            BCparam['type'] = PreferedBoundaryConditions[BCparam['type']]

        if BCparam['type'] == 'nref':
            if 'option' not in BCparam:
                if 'filename' in BCkwargs:
                    BCparam['option'] = 'file'
                else:
                    BCparam['option'] = 'uniform'

            if BCparam['option'] == 'uniform':
                print(J.CYAN + 'set BC nref on ' + BCparam['FamilyName'] + J.ENDC)
                setBC_nref(t, **BCkwargs)

            elif BCparam['option'] == 'file':
                print('{}set BC nref (from file {}) on {}{}'.format(J.CYAN,
                    BCparam['filename'], BCparam['FamilyName'], J.ENDC))
                setBC_nref_imposeFromFile(t, ReferenceValues, **BCkwargs)

        elif BCparam['type'] == 'inj1':

            if 'option' not in BCparam:
                if 'bc' in BCkwargs:
                    BCparam['option'] = 'bc'
                elif 'filename' in BCkwargs:
                    BCparam['option'] = 'file'
                else:
                    BCparam['option'] = 'uniform'

            if BCparam['option'] == 'uniform':
                print(J.CYAN + 'set BC inj1 (uniform) on ' + BCparam['FamilyName'] + J.ENDC)
                setBC_inj1_uniform(t, FluidProperties, ReferenceValues, **BCkwargs)

            elif BCparam['option'] == 'file':
                print('{}set BC inj1 (from file {}) on {}{}'.format(J.CYAN,
                    BCparam['filename'], BCparam['FamilyName'], J.ENDC))
                setBC_inj1_imposeFromFile(t, FluidProperties, ReferenceValues, **BCkwargs)

            elif BCparam['option'] == 'bc':
                print('set BC inj1 on {}'.format(J.CYAN, BCparam['FamilyName'], J.ENDC))
                setBC_inj1(t, ReferenceValues, **BCkwargs)

        elif BCparam['type'] == 'injmfr1':
            print(J.CYAN + 'set BC injmfr1 on ' + BCparam['FamilyName'] + J.ENDC)
            setBC_injmfr1(t, FluidProperties, ReferenceValues, **BCkwargs)

        elif BCparam['type'] == 'outpres':
            print(J.CYAN + 'set BC outpres on ' + BCparam['FamilyName'] + J.ENDC)
            setBC_outpres(t, **BCkwargs)

        elif BCparam['type'] == 'outmfr2':
            print(J.CYAN + 'set BC outmfr2 on ' + BCparam['FamilyName'] + J.ENDC)
            BCkwargs['ReferenceValues'] = ReferenceValues
            BCkwargs['TurboConfiguration'] = TurboConfiguration
            setBC_outmfr2(t, **BCkwargs)

        elif BCparam['type'] == 'OutflowMassFlowFromMach':
            print(J.CYAN + 'set BC outmfr2 on ' + BCparam['FamilyName'] + J.ENDC)
            BCkwargs['ReferenceValues'] = ReferenceValues
            BCkwargs['TurboConfiguration'] = TurboConfiguration
            BCkwargs['FluidProperties'] = FluidProperties
            setBC_MachFromMassFlow(t, **BCkwargs)


        elif BCparam['type'] == 'outradeq':
            print(J.CYAN + 'set BC outradeq on ' + BCparam['FamilyName'] + J.ENDC)
            BCkwargs['ReferenceValues'] = ReferenceValues
            BCkwargs['TurboConfiguration'] = TurboConfiguration
            setBC_outradeq(t, **BCkwargs)

        elif BCparam['type'] == 'outradeqhyb':
            print(J.CYAN + 'set BC outradeqhyb on ' + BCparam['FamilyName'] + J.ENDC)
            BCkwargs['ReferenceValues'] = ReferenceValues
            BCkwargs['TurboConfiguration'] = TurboConfiguration
            setBC_outradeqhyb(t, **BCkwargs)

        elif BCparam['type'] == 'giles_out':
            print(J.CYAN + 'set BC nscbc giles_out on ' + BCparam['FamilyName'] + ' (MonitoringFlag=%i)'%GilesMonitoringFlag + J.ENDC)
            
            # add the node FamilyBC_t in the Family Node
            FamilyNode = I.getNodeFromNameAndType(t, BCparam['FamilyName'], 'Family_t')
            I._rmNodesByName(FamilyNode, '.Solver#BC')
            I._rmNodesByType(FamilyNode, 'FamilyBC_t')
            I.newFamilyBC(value='BCOutflowSubsonic', parent=FamilyNode)

            if not 'VelocityScale' in BCkwargs:
                # computation of sound velocity for Giles
                BCkwargs['VelocityScale'] =  (FluidProperties['Gamma']*FluidProperties['IdealGasConstant']*ReferenceValues['TemperatureStagnation'])**0.5 

            BCkwargs['GilesMonitoringFlag'] = GilesMonitoringFlag
            if 'option' in BCparam:
                BCkwargs['option'] = BCparam['option']
            else:
                BCkwargs['option'] = 'RadialEquilibrium'
                
            for bc in C.getFamilyBCs(t,BCparam['FamilyName']):
                setBC_giles_outlet(t, bc, **BCkwargs)
            GilesMonitoringFlag += 1

        elif BCparam['type'] == 'giles_in':
            print(J.CYAN + 'set BC nscbc giles_in on ' + BCparam['FamilyName'] + ' (MonitoringFlag=%i)'%GilesMonitoringFlag + J.ENDC)
            
            # add the node FamilyBC_t in the Family Node
            FamilyNode = I.getNodeFromNameAndType(t, BCparam['FamilyName'], 'Family_t')
            I._rmNodesByName(FamilyNode, '.Solver#BC')
            I._rmNodesByType(FamilyNode, 'FamilyBC_t')
            I.newFamilyBC(value='BCInflowSubsonic', parent=FamilyNode)

            if not 'VelocityScale' in BCkwargs:
                # computation of sound velocity for Giles
                BCkwargs['VelocityScale'] =  (FluidProperties['Gamma']*FluidProperties['IdealGasConstant']*ReferenceValues['TemperatureStagnation'])**0.5 

            BCkwargs['GilesMonitoringFlag'] = GilesMonitoringFlag
            if 'option' in BCparam:
                BCkwargs['option'] = BCparam['option']
            else:
                BCkwargs['option'] = 'uniform'

            for bc in C.getFamilyBCs(t,BCparam['FamilyName']):
                setBC_giles_inlet(t, bc, FluidProperties, ReferenceValues, **BCkwargs)
            GilesMonitoringFlag += 1

        elif BCparam['type'] == 'stage_mxpl':
            print('{}set BC stage_mxpl between {} and {}{}'.format(J.CYAN,
                BCparam['left'], BCparam['right'], J.ENDC))
            setBC_stage_mxpl(t, **BCkwargs)

        elif BCparam['type'] == 'stage_mxpl_hyb':
            print('{}set BC stage_mxpl_hyb between {} and {}{}'.format(J.CYAN,
                BCparam['left'], BCparam['right'], J.ENDC))
            setBC_stage_mxpl_hyb(t, **BCkwargs)

        elif BCparam['type'] == 'stage_red':
            print('{}set BC stage_red between {} and {}{}'.format(J.CYAN,
                BCparam['left'], BCparam['right'], J.ENDC))
            if not 'stage_ref_time' in BCkwargs:
                # Assume a 360 configuration
                BCkwargs['stage_ref_time'] = 2*np.pi / abs(TurboConfiguration['ShaftRotationSpeed'])
            setBC_stage_red(t, **BCkwargs)

        elif BCparam['type'] == 'stage_red_hyb':
            print('{}set BC stage_red_hyb between {} and {}{}'.format(J.CYAN,
                BCparam['left'], BCparam['right'], J.ENDC))
            if not 'stage_ref_time' in BCkwargs:
                # Assume a 360 configuration
                BCkwargs['stage_ref_time'] = 2*np.pi / abs(TurboConfiguration['ShaftRotationSpeed'])
            setBC_stage_red_hyb(t, **BCkwargs)

        elif BCparam['type'] == 'giles_stage_mxpl':
            print('{}set BC giles_stage_mxpl between {} and {} ({}-{}) {}'.format(J.CYAN,
                BCparam['left'], BCparam['right'], GilesMonitoringFlag, GilesMonitoringFlag+1, J.ENDC))

            if not 'VelocityScale' in BCkwargs:
                # computation of sound velocity for Giles
                BCkwargs['VelocityScale'] =  (FluidProperties['Gamma']*FluidProperties['IdealGasConstant']*ReferenceValues['TemperatureStagnation'])**0.5 

            BCkwargs['nscbc_mxpl_flag'] = GilesMonitoringFlag                   # index gathering left and right BCs for one given Mxpl interface
            BCkwargs['GilesMonitoringFlag_left'] = GilesMonitoringFlag         # index gathering all BCs "left" for one given Mxpl interface
            BCkwargs['GilesMonitoringFlag_right'] = GilesMonitoringFlag+1      # index gathering all BCs "right" for one given Mxpl interface
            
            setBC_giles_stage_mxpl(t, **BCkwargs)
            GilesMonitoringFlag += 2



        elif BCparam['type'] == 'stage_choro':
            print('{}set BC stage_choro between {} and {}{}'.format(J.CYAN,
                BCparam['left'], BCparam['right'], J.ENDC))
            BCkwargs['Rows'] = TurboConfiguration['Rows']
            setChorochronic(t, **BCkwargs)

        # TODO
        # elif BCparam['type'] == 'stage_choro_hyb':
        #     print('{}set BC stage_red_hyb between {} and {}{}'.format(J.CYAN,
        #         BCparam['left'], BCparam['right'], J.ENDC))
        #     setBC_stage_choro_hyb(t, **BCkwargs)

        elif BCparam['type'] == 'sym':
            print(J.CYAN + 'set BC sym on ' + BCparam['FamilyName'] + J.ENDC)
            setBC_sym(t, **BCkwargs)

        elif BCparam['type'] == 'walladia':
            print(J.CYAN + 'set BC walladia on ' + BCparam['FamilyName'] + J.ENDC)
            setBC_walladia(t, **BCkwargs)

        elif BCparam['type'] == 'wallslip':
            print(J.CYAN + 'set BC wallslip on ' + BCparam['FamilyName'] + J.ENDC)
            setBC_wallslip(t, **BCkwargs)

        elif BCparam['type'] == 'wallisoth':
            print(J.CYAN + 'set BC wallisoth on ' + BCparam['FamilyName'] + J.ENDC)
            setBC_wallisoth(t, **BCkwargs)

        else:
            raise AttributeError('BC type %s not implemented'%BCparam['type'])


def setBC_Walls(t, TurboConfiguration,
                    bladeFamilyNames=['BLADE', 'AUBE'],
                    hubFamilyNames=['HUB', 'SPINNER', 'MOYEU'],
                    shroudFamilyNames=['SHROUD', 'CARTER']):
    '''
    Set all the wall boundary conditions in a turbomachinery context, by making
    the following operations:

        * set BCs related to each blade.
        * set BCs related to hub. The intervals where the rotation speed is the
          shaft speed (for rotor platforms) are set in the following form:

            >>> TurboConfiguration['HubRotationSpeed'] = [(xmin1, xmax1), ..., (xminN, xmaxN)]

        * set BCs related to shroud. Rotation speed is set to zero.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        TurboConfiguration : dict
            as produced :py:func:`getTurboConfiguration`

        bladeFamilyNames : :py:class:`list` of :py:class:`str`
            list of patterns to find families related to blades. Not sensible
            to string case. By default, search patterns 'BLADE' and 'AUBE'.

        hubFamilyNames : :py:class:`list` of :py:class:`str`
            list of patterns to find families related to hub. Not sensible
            to string case. By default, search patterns 'HUB' and 'MOYEU'.

        shroudFamilyNames : :py:class:`list` of :py:class:`str`
            list of patterns to find families related to shroud. Not sensible
            to string case. By default, search patterns 'SHROUD' and 'CARTER'.

    '''
    bladeFamilyNames = PRE.extendListOfFamilies(bladeFamilyNames)
    hubFamilyNames = PRE.extendListOfFamilies(hubFamilyNames)
    shroudFamilyNames = PRE.extendListOfFamilies(shroudFamilyNames)

    if 'PeriodicTranslation' in TurboConfiguration:
        # For linear cascade configuration: all blocks and wall are motionless
        wallFamily = []
        for wallFamily in bladeFamilyNames + hubFamilyNames + shroudFamilyNames:
            for famNode in I.getNodesFromNameAndType(t, '*{}*'.format(wallFamily), 'Family_t'):
                I._rmNodesByType(famNode, 'FamilyBC_t')
                I.newFamilyBC(value='BCWallViscous', parent=famNode)
        return

    def omegaHubAtX(x):
        omega = np.zeros(x.shape, dtype=float)
        for (x1, x2) in TurboConfiguration['HubRotationSpeed']:
            omega[(x1<=x) & (x<=x2)] = TurboConfiguration['ShaftRotationSpeed']
        return np.asfortranarray(omega).ravel(order='K')

    def getZoneFamilyNameWithFamilyNameBC(zones, FamilyNameBC):
        ZoneFamilyName = None
        for zone in zones:
            ZoneBC = I.getNodeFromType1(zone, 'ZoneBC_t')
            if not ZoneBC: continue
            FamiliesNames = I.getNodesFromType2(ZoneBC, 'FamilyName_t')
            for FamilyName_n in FamiliesNames:
                if I.getValue(FamilyName_n) == FamilyNameBC:
                    ZoneFamilyName = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
            if ZoneFamilyName: break

        try:
            assert ZoneFamilyName is not None, 'Cannot determine associated row for family {}. '.format(FamilyNameBC)
        except AssertionError as e:
            J.save(zones,'debug.cgns')
            raise e
        return ZoneFamilyName
        
    # BLADES
    zones = I.getZones(t)
    families = I.getNodesFromType2(t,'Family_t')
    for blade_family in bladeFamilyNames:
        for famNode in I.getNodesFromNameAndType(t, '*{}*'.format(blade_family), 'Family_t'):
            famName = I.getName(famNode)
            if famName.startswith('F_OV_') or famName.endswith('Zones'): continue
            ZoneFamilyName = getZoneFamilyNameWithFamilyNameBC(zones, famName)
            family_with_bcwall, = [f for f in families if f[0]==ZoneFamilyName]
            solver_motion_data = J.get(family_with_bcwall,'.Solver#Motion')
            setBC_walladia(t, famName, omega=solver_motion_data['omega'])

    # HUB
    for hub_family in hubFamilyNames:
        for famNode in I.getNodesFromNameAndType(t, '*{}*'.format(hub_family), 'Family_t'):
            famName = I.getName(famNode)
            if famName.startswith('F_OV_') or famName.endswith('Zones'): continue
            setBC_walladia(t, famName, omega=0.)

            # TODO request initVars of BCDataSet
            wallHubBC = C.extractBCOfName(t, 'FamilySpecified:{0}'.format(famName))
            wallHubBC = C.node2Center(wallHubBC)
            for w in wallHubBC:
                xw = I.getValue(I.getNodeFromName(w,'CoordinateX'))
                zname, wname = I.getName(w).split(os.sep)
                znode = I.getNodeFromNameAndType(t,zname,'Zone_t')
                wnode = I.getNodeFromNameAndType(znode,wname,'BC_t')
                BCDataSet = I.newBCDataSet(name='BCDataSet#Init', value='Null',
                    gridLocation='FaceCenter', parent=wnode)
                J.set(BCDataSet, 'NeumannData', childType='BCData_t', omega=omegaHubAtX(xw))

    # SHROUD
    for shroud_family in shroudFamilyNames:
        for famNode in I.getNodesFromNameAndType(t, '*{}*'.format(shroud_family), 'Family_t'):
            famName = I.getName(famNode)
            if famName.startswith('F_OV_') or famName.endswith('Zones'): continue
            setBC_walladia(t, famName, omega=0.)

def setBC_walladia(t, FamilyName, omega=None):
    '''
    Set a viscous wall boundary condition.

    .. note:: see `elsA Tutorial about wall conditions <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#wall-conditions/>`_

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        omega : float
            Rotation speed imposed at the wall. If :py:obj:`None`, it is not specified 
            in the Family node (but same behavior that zero).

    '''
    wall = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    if not wall: 
        J.save(t,'debug.cgns')
        raise ValueError(J.FAIL+f'could not find family "{FamilyName}" for wall. Check debug.cgns'+J.ENDC)
    I._rmNodesByName(wall, '.Solver#BC')
    I._rmNodesByType(wall, 'FamilyBC_t')
    I.newFamilyBC(value='BCWallViscous', parent=wall)
    if omega is not None:
        J.set(wall, '.Solver#BC',
                    type='walladia',
                    data_frame='user',
                    omega=omega,
                    axis_pnt_x=0., axis_pnt_y=0., axis_pnt_z=0.,
                    axis_vct_x=1., axis_vct_y=0., axis_vct_z=0.)

def setBC_wallslip(t, FamilyName):
    '''
    Set an inviscid wall boundary condition.

    .. note:: see `elsA Tutorial about wall conditions <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#wall-conditions/>`_

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

    '''
    wall = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    I._rmNodesByName(wall, '.Solver#BC')
    I._rmNodesByType(wall, 'FamilyBC_t')
    I.newFamilyBC(value='BCWallInviscid', parent=wall)

def setBC_wallisoth(t, FamilyName, Temperature, bc=None):
    '''
    Set an isothermal wall boundary condition.

    .. note:: see `elsA Tutorial about wall conditions <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#wall-conditions/>`_

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        Temperature : :py:class:`float` or :py:class:`numpy.ndarray` or :py:class:`dict`
            Value of temperature to impose on the boundary condition. May be:

                * either a scalar: in that case it is imposed once for the
                  family **FamilyName** in the corresponding ``Family_t`` node.

                * or a numpy array: in that case it is imposed for the ``BC_t``
                  node **bc**.

            Alternatively, **Temperature** may be a :py:class:`dict` of the form:

            >>> Temperature = dict(wall_temp=value)

            In that case, the same requirements that before stands for *value*.

        bc : PyTree
            ``BC_t`` node on which the boundary condition will be imposed. Must
            be :py:obj:`None` if the condition must be imposed once in the
            ``Family_t`` node.

    '''
    if isinstance(Temperature, dict):
        assert 'wall_temp' in Temperature
        assert len(Temperature.keys() == 1)
        ImposedVariables = Temperature
    else:
        ImposedVariables = dict(wall_temp=Temperature)
    setBCwithImposedVariables(t, FamilyName, ImposedVariables,
        FamilyBC='BCWallViscousIsothermal', BCType='wallisoth', bc=bc)

def setBC_sym(t, FamilyName):
    '''
    Set a symmetry boundary condition.

    .. note:: see `elsA Tutorial about symmetry condition <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#symmetry/>`_

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

    '''
    symmetry = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    I._rmNodesByName(symmetry, '.Solver#BC')
    I._rmNodesByType(symmetry, 'FamilyBC_t')
    I.newFamilyBC(value='BCSymmetryPlane', parent=symmetry)

def setBC_nref(t, FamilyName):
    '''
    Set a nref boundary condition.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

    '''
    farfield = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    I._rmNodesByName(farfield, '.Solver#BC')
    I._rmNodesByType(farfield, 'FamilyBC_t')
    I.newFamilyBC(value='BCFarfield', parent=farfield)

def setBC_nref_imposeFromFile(t, ReferenceValues, FamilyName, filename, fileformat=None):

    def setBC_nref_on_bc(t, FamilyName, ImposedVariables, bc=None):
        if not bc and not all([np.ndim(v)==0 and not callable(v) for v in ImposedVariables.values()]):
            for bc in C.getFamilyBCs(t, FamilyName):
                setBCwithImposedVariables(t, FamilyName, ImposedVariables,
                    FamilyBC='BCFarfield', BCType='nref', bc=bc)
        else:
            setBCwithImposedVariables(t, FamilyName, ImposedVariables,
                FamilyBC='BCFarfield', BCType='nref', bc=bc)
            
    var2interp = ['Density', 'MomentumX', 'MomentumY', 'MomentumZ', 'EnergyStagnationDensity']
    var2interp += ReferenceValues['FieldsTurbulence']

    donor_tree = C.convertFile2PyTree(filename, format=fileformat)
    inlet_BC_nodes = C.extractBCOfName(t, f'FamilySpecified:{FamilyName}', reorder=False)

    I._adaptZoneNamesForSlash(inlet_BC_nodes)
    I._rmNodesByType(inlet_BC_nodes,'FlowSolution_t')
    J.migrateFields(donor_tree, inlet_BC_nodes)

    for w in inlet_BC_nodes:
        bcLongName = I.getName(w)  # from C.extractBCOfName: <zone>\<bc>
        zname, wname = bcLongName.split('\\')
        znode = I.getNodeFromNameAndType(t, zname, 'Zone_t')
        bcnode = I.getNodeFromNameAndType(znode, wname, 'BC_t')
        ImposedVariables = dict()
        for var in var2interp:
            FS = I.getNodeFromName(w, I.__FlowSolutionCenters__)
            varNode = I.getNodeFromName(FS, var) 
            if varNode:
                ImposedVariables[var] = np.asfortranarray(I.getValue(varNode))
            else:
                raise TypeError('variable {} not found in {}'.format(var, filename))
        
        setBC_nref_on_bc(t, FamilyName, ImposedVariables, bc=bcnode)


def setBC_inj1(t, FamilyName, ImposedVariables, bc=None, variableForInterpolation='ChannelHeight'):
    '''
    Generic function to impose a Boundary Condition ``inj1``. The following
    functions are more specific:

        * :py:func:`setBC_inj1_uniform`

        * :py:func:`setBC_inj1_imposeFromFile`

    .. note::
        see `elsA Tutorial about inj1 condition <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#inj1/>`_

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        ImposedVariables : dict
            Dictionary of variables to imposed on the boudary condition. Keys
            are variable names and values must be:

                * either scalars: in that case they are imposed once for the
                  family **FamilyName** in the corresponding ``Family_t`` node.

                * or numpy arrays: in that case they are imposed for the ``BC_t``
                  node **bc**.

        bc : PyTree
            ``BC_t`` node on which the boundary condition will be imposed. Must
            be :py:obj:`None` if the condition must be imposed once in the
            ``Family_t`` node.
        
        variableForInterpolation : str
            When using a function to impose the radial profile of one or several quantities, 
            it defines the variable used as the argument of this function.
            Must be 'ChannelHeight' (default value) or 'Radius'.

    See also
    --------

    setBC_inj1_uniform, setBC_inj1_imposeFromFile
    '''
    if not bc and not all([np.ndim(v)==0 and not callable(v) for v in ImposedVariables.values()]):
        for bc in C.getFamilyBCs(t, FamilyName):
            setBCwithImposedVariables(t, FamilyName, ImposedVariables,
                FamilyBC='BCInflowSubsonic', BCType='inj1', bc=bc, variableForInterpolation=variableForInterpolation)
    else:
        setBCwithImposedVariables(t, FamilyName, ImposedVariables,
            FamilyBC='BCInflowSubsonic', BCType='inj1', bc=bc, variableForInterpolation=variableForInterpolation)

def getPrimitiveTurbulentFieldForInjection(FluidProperties, ReferenceValues, **kwargs):
        '''
        Get the primitive (without the Density factor) turbulent variables (names and values) 
        to inject in an inflow boundary condition.

        For RSM models, see issue https://elsa.onera.fr/issues/5136 for the naming convention.

        Parameters
        ----------
        ReferenceValues : dict
            as obtained from :py:func:`computeReferenceValues`

        kwargs : dict
            Optional parameters, taken from **ReferenceValues** if not given.

        Returns
        -------
        dict
            Imposed turbulent variables
        '''
        if 'TurbulenceLevel' in kwargs or 'Viscosity_EddyMolecularRatio' in kwargs:   
            print('  recomputing turbulent variables for this BC...')        
            ReferenceValuesForTurbulence = computeReferenceValues(
                FluidProperties,
                MassFlow=kwargs.get('MassFlow', ReferenceValues['MassFlow']),
                PressureStagnation=kwargs.get('PressureStagnation', ReferenceValues['PressureStagnation']),
                TemperatureStagnation=kwargs.get('TemperatureStagnation', ReferenceValues['TemperatureStagnation']),
                Surface=kwargs.get('Surface', ReferenceValues['Surface']),
                TurbulenceLevel=kwargs.get('TurbulenceLevel', ReferenceValues['TurbulenceLevel']),
                Viscosity_EddyMolecularRatio=kwargs.get('Viscosity_EddyMolecularRatio', ReferenceValues['Viscosity_EddyMolecularRatio']),
                VelocityUsedForScalingAndTurbulence=kwargs.get('VelocityUsedForScalingAndTurbulence', None),
                TurbulenceModel=ReferenceValues['TurbulenceModel']
                )
        else:
            ReferenceValuesForTurbulence = ReferenceValues

        turbDict = dict()
        for name, value in zip(ReferenceValuesForTurbulence['FieldsTurbulence'], ReferenceValuesForTurbulence['ReferenceStateTurbulence']):
            # If the 'conservative' value is given in kwargs
            value = kwargs.get(name, value)

            if name.endswith('Density'):
                name = name.replace('Density', '')
                value /= ReferenceValuesForTurbulence['Density']
            elif name == 'ReynoldsStressDissipationScale':
                name = 'TurbulentDissipationRate'
                value /= ReferenceValuesForTurbulence['Density']
            elif name.startswith('ReynoldsStress'):
                name = name.replace('ReynoldsStress', 'VelocityCorrelation')
                value /= ReferenceValuesForTurbulence['Density']

            # If the 'primitive' value is given in kwargs
            turbDict[name] = kwargs.get(name, value)
            
        return turbDict

def setBC_inj1_uniform(t, FluidProperties, ReferenceValues, FamilyName, **kwargs):
    '''
    Set a Boundary Condition ``inj1`` with uniform inflow values. These values
    are them in **ReferenceValues**.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FluidProperties : dict
            as obtained from :py:func:`computeFluidProperties`

        ReferenceValues : dict
            as obtained from :py:func:`computeReferenceValues`

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        kwargs : dict
            Optional parameters, taken from **ReferenceValues** if not given:
            PressureStagnation, TemperatureStagnation, EnthalpyStagnation,
            VelocityUnitVectorX, VelocityUnitVectorY, VelocityUnitVectorZ, 
            and primitive turbulent variables

    See also
    --------

    setBC_inj1, setBC_inj1_imposeFromFile, setBC_injmfr1

    '''
    # HACK to handle this function called by a workflow for external aerodynamics
    ReferenceValues.setdefault('PressureStagnation', None)
    ReferenceValues.setdefault('TemperatureStagnation', None)

    PressureStagnation    = kwargs.get('PressureStagnation', ReferenceValues['PressureStagnation'])
    TemperatureStagnation = kwargs.get('TemperatureStagnation', ReferenceValues['TemperatureStagnation'])
    EnthalpyStagnation    = kwargs.get('EnthalpyStagnation', FluidProperties['cp'] * TemperatureStagnation)
    VelocityUnitVectorX   = kwargs.get('VelocityUnitVectorX', ReferenceValues['DragDirection'][0])
    VelocityUnitVectorY   = kwargs.get('VelocityUnitVectorY', ReferenceValues['DragDirection'][1])
    VelocityUnitVectorZ   = kwargs.get('VelocityUnitVectorZ', ReferenceValues['DragDirection'][2])
    variableForInterpolation = kwargs.get('variableForInterpolation', 'ChannelHeight')   

    ImposedVariables = dict(
        PressureStagnation  = PressureStagnation,
        EnthalpyStagnation  = EnthalpyStagnation,
        VelocityUnitVectorX = VelocityUnitVectorX,
        VelocityUnitVectorY = VelocityUnitVectorY,
        VelocityUnitVectorZ = VelocityUnitVectorZ,
        **getPrimitiveTurbulentFieldForInjection(FluidProperties, ReferenceValues, **kwargs)
        )

    setBC_inj1(t, FamilyName, ImposedVariables, variableForInterpolation=variableForInterpolation)

def setBC_inj1_imposeFromFile(t, FluidProperties, ReferenceValues, FamilyName, filename, fileformat=None):
    '''
    Set a Boundary Condition ``inj1`` using the field map in the file
    **filename**. It is expected to be a surface with the following variables
    defined at cell centers (in the container 'FlowSolution#Centers'):

        * the coordinates

        * the stagnation pressure ``'PressureStagnation'``

        * the stagnation enthalpy ``'EnthalpyStagnation'``

        * the three components of the unit vector for the velocity direction:
          ``'VelocityUnitVectorX'``, ``'VelocityUnitVectorY'``, ``'VelocityUnitVectorZ'``

        * the primitive turbulent variables (so not multiplied by density)
          comptuted from ``ReferenceValues['FieldsTurbulence']`` and
          depending on the turbulence model.
          For example: ``'TurbulentEnergyKinetic'`` and
          ``'TurbulentDissipationRate'`` for a k-omega model.

    Field variables are just read in **filename** and written in
    BCs of **t** attached to the family **FamilyName**.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        ReferenceValues : dict
            as obtained from :py:func:`computeReferenceValues`

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        filename : str
            name of the input filename

        fileformat : optional, str
            format of the input file to be passed to Converter.convertFile2PyTree
            Cassiopee function.

            .. note:: see `available file formats <http://elsa.onera.fr/Cassiopee/Converter.html?highlight=initvars#fileformats>`_

    See also
    --------

    setBC_inj1, setBC_inj1_uniform

    '''

    var2interp = ['PressureStagnation', 'EnthalpyStagnation',
        'VelocityUnitVectorX', 'VelocityUnitVectorY', 'VelocityUnitVectorZ']
    turbDict = getPrimitiveTurbulentFieldForInjection(FluidProperties, ReferenceValues)
    var2interp += list(turbDict)

    donor_tree = C.convertFile2PyTree(filename, format=fileformat)
    inlet_BC_nodes = C.extractBCOfName(t, f'FamilySpecified:{FamilyName}', reorder=False)

    I._adaptZoneNamesForSlash(inlet_BC_nodes)
    I._rmNodesByType(inlet_BC_nodes,'FlowSolution_t')
    J.migrateFields(donor_tree, inlet_BC_nodes)

    for w in inlet_BC_nodes:
        bcLongName = I.getName(w)  # from C.extractBCOfName: <zone>\<bc>
        zname, wname = bcLongName.split('\\')
        znode = I.getNodeFromNameAndType(t, zname, 'Zone_t')
        bcnode = I.getNodeFromNameAndType(znode, wname, 'BC_t')
        ImposedVariables = dict()
        for var in var2interp:
            FS = I.getNodeFromName(w, I.__FlowSolutionCenters__)
            varNode = I.getNodeFromName(FS, var) 
            if varNode:
                ImposedVariables[var] = np.asfortranarray(I.getValue(varNode))
            else:
                raise TypeError('variable {} not found in {}'.format(var, filename))
        
        setBC_inj1(t, FamilyName, ImposedVariables, bc=bcnode)

def setBC_injmfr1(t, FluidProperties, ReferenceValues, FamilyName, **kwargs):
    '''
    Set a Boundary Condition ``injmfr1`` with uniform inflow values. These values
    are them in **ReferenceValues**.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FluidProperties : dict
            as obtained from :py:func:`computeFluidProperties`

        ReferenceValues : dict
            as obtained from :py:func:`computeReferenceValues`

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        kwargs : dict
            Optional parameters, taken from **ReferenceValues** if not given:
            MassFlow, SurfacicMassFlow, Surface, TemperatureStagnation, EnthalpyStagnation,
            VelocityUnitVectorX, VelocityUnitVectorY, VelocityUnitVectorZ,
            TurbulenceLevel, Viscosity_EddyMolecularRatio and primitive turbulent variables

    See also
    --------

    setBC_inj1, setBC_inj1_imposeFromFile

    '''
    Surface = kwargs.get('Surface', None)
    if not Surface:
        # Compute surface of the inflow BC
        zones = C.extractBCOfName(t, 'FamilySpecified:'+FamilyName)
        assert len(zones) != 0, f'{J.FAIL}Cannot extract BC {FamilyName}. Please check the name of the family.{J.ENDC}'
        SurfaceTree = C.convertArray2Tetra(zones)
        SurfaceTree = C.initVars(SurfaceTree, 'ones=1')
        Surface = P.integ(SurfaceTree, var='ones')[0]

    MassFlow              = kwargs.get('MassFlow', ReferenceValues['MassFlow'])
    SurfacicMassFlow      = kwargs.get('SurfacicMassFlow', MassFlow / Surface)
    TemperatureStagnation = kwargs.get('TemperatureStagnation', ReferenceValues['TemperatureStagnation'])
    EnthalpyStagnation    = kwargs.get('EnthalpyStagnation', FluidProperties['cp'] * TemperatureStagnation)
    VelocityUnitVectorX   = kwargs.get('VelocityUnitVectorX', ReferenceValues['DragDirection'][0])
    VelocityUnitVectorY   = kwargs.get('VelocityUnitVectorY', ReferenceValues['DragDirection'][1])
    VelocityUnitVectorZ   = kwargs.get('VelocityUnitVectorZ', ReferenceValues['DragDirection'][2])
    variableForInterpolation = kwargs.get('variableForInterpolation', 'ChannelHeight')  
    if not 'MassFlow' in kwargs:
        # used for getPrimitiveTurbulentFieldForInjection
        kwargs['MassFlow'] = SurfacicMassFlow * Surface

    ImposedVariables = dict(
        SurfacicMassFlow    = SurfacicMassFlow,
        EnthalpyStagnation  = EnthalpyStagnation,
        VelocityUnitVectorX = VelocityUnitVectorX,
        VelocityUnitVectorY = VelocityUnitVectorY,
        VelocityUnitVectorZ = VelocityUnitVectorZ,
        **getPrimitiveTurbulentFieldForInjection(FluidProperties, ReferenceValues, **kwargs)
        )

    setBCwithImposedVariables(t, FamilyName, ImposedVariables,
        FamilyBC='BCInflowSubsonic', BCType='injmfr1', variableForInterpolation=variableForInterpolation)

def setBC_outpres(t, FamilyName, Pressure, bc=None, variableForInterpolation='ChannelHeight'):
    '''
    Impose a Boundary Condition ``outpres``.

    .. note::
        see `elsA Tutorial about outpres condition <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#outpres/>`_

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        Pressure : :py:class:`float` or :py:class:`numpy.ndarray` or :py:class:`dict`
            Value of pressure to impose on the boundary conditions. May be:

                * either a scalar: in that case it is imposed once for the
                  family **FamilyName** in the corresponding ``Family_t`` node.

                * or a numpy array: in that case it is imposed for the ``BC_t``
                  node **bc**.

            Alternatively, **Pressure** may be a :py:class:`dict` of the form:

            >>> Pressure = dict(Pressure=value)

            In that case, the same requirements that before stands for *value*.

        bc : PyTree
            ``BC_t`` node on which the boundary condition will be imposed. Must
            be :py:obj:`None` if the condition must be imposed once in the
            ``Family_t`` node.
        
        variableForInterpolation : str
            When using a function to impose the radial profile of one or several quantities, 
            it defines the variable used as the argument of this function.
            Must be 'ChannelHeight' (default value) or 'Radius'.

    '''
    if isinstance(Pressure, dict):
        assert 'Pressure' in Pressure or 'pressure' in Pressure
        assert len(Pressure.keys() == 1)
        ImposedVariables = Pressure
    else:
        ImposedVariables = dict(Pressure=Pressure)

    if not bc and not all([np.ndim(v) == 0 and not callable(v) for v in ImposedVariables.values()]):
        for bc in C.getFamilyBCs(t, FamilyName):
            setBCwithImposedVariables(t, FamilyName, ImposedVariables,
                                      FamilyBC='BCOutflowSubsonic', BCType='outpres', bc=bc, variableForInterpolation=variableForInterpolation)
    else:
        setBCwithImposedVariables(t, FamilyName, ImposedVariables,
                                FamilyBC='BCOutflowSubsonic', BCType='outpres', bc=bc, variableForInterpolation=variableForInterpolation)

def setBC_giles_outlet(t, bc, FamilyName,**kwargs):
    '''
    Impose a Boundary Condition ``giles_out``.

    .. note::
    
        see theoretical report: /home/bfrancoi/NSCBC/RapportsONERA/SONICE-TF-S2.1.4.1.2_NSCBCgilesInletSteadyStructured_Final3.pdf

    Parameters
    ----------

            t : PyTree
                Tree to modify

            bc : CGNS node of type BC_t
                 BC node attached to the family in the which the boundary condition is applied

            FamilyName : str
                Name of the family on which the boundary condition will be imposed
            
            kwargs : dict
                Parameters defined by the user: FamilyName, Pressure, NbModesFourierGiles, monitoring_flag, option

            TO DO: 
               1. keep a similar structure as outpres and outradeq
               2. add cartography
    '''

    # creation of dictionnary of keys for Giles outlet BC  
    DictKeysGilesOutlet = {}

    # keys relative to NSCBC
    DictKeysGilesOutlet['type'] = 'nscbc_out'                                                                   # mandatory key to have NSCBC-Giles treatment
    DictKeysGilesOutlet['nscbc_giles'] = 'statio'                                                               # mandatory key to have NSCBC-Giles treatment
    DictKeysGilesOutlet['nscbc_interpbc'] = 'linear'                                                            # mandatory value
    DictKeysGilesOutlet['nscbc_fluxt'] = kwargs.get('nscbc_fluxt', 'fluxBothTransv')                            # recommended value - possible keys : 'classic'; 'fluxInviscidTransv'; 'fluxBothTransv'
    DictKeysGilesOutlet['nscbc_surf'] = kwargs.get('nscbc_surf',  'revolution')                                 # recommended value - possible keys : 'flat', 'revolution'
    DictKeysGilesOutlet['nscbc_outwave'] = kwargs.get('nscbc_outwave',  'grad_etat')                            # recommended value - possible keys : 'grad_etat'; 'extrap_flux'
    DictKeysGilesOutlet['nscbc_velocity_scale'] = kwargs.get('nscbc_velocity_scale',kwargs['VelocityScale'])    # default value - reference sound velocity 
    DictKeysGilesOutlet['nscbc_viscwall_len'] = kwargs.get('nscbc_viscwall_len', 5.e-4)                         # default value, could be updated by the user if convergence issue

    if kwargs.get('nscbc_viscwall_len_hub') is not None:
        DictKeysGilesOutlet['nscbc_viscwall_len_hub'] = kwargs.get('nscbc_viscwall_len_hub')                    # value of nscbc_viscwall_len for the hub only
    if kwargs.get('nscbc_viscwall_len_carter') is not None:
        DictKeysGilesOutlet['nscbc_viscwall_len_carter'] = kwargs.get('nscbc_viscwall_len_carter')              # value of nscbc_viscwall_len for the hub only           


    # keys relative to the Giles treatment 
    DictKeysGilesOutlet['giles_opt'] = 'relax'                                                        # mandatory key for NSCBC-Giles treatment
    DictKeysGilesOutlet['giles_restric_relax'] = 'inactive'                                           # mandatory key for NSCBC-Giles treatment
    DictKeysGilesOutlet['giles_exact_lodi'] = kwargs.get('giles_exact_lodi',  'active')               # recommended value - possible keys: 'inactive', 'partial', 'active'
    DictKeysGilesOutlet['giles_nbMode'] = kwargs.get('NbModesFourierGiles')                           # given by the user - recommended value : ncells_theta/2 + 1 (odd_value)

    # keys relative to the monitoring and radii calculus - monitoring data stored in LOGS
    DictKeysGilesOutlet['bnd_monitoring'] = 'active'                                                  # recommended value
    DictKeysGilesOutlet['monitoring_comp_rad'] = 'auto'                                               # recommended value - possible keys: 'from_file', 'monofenetre'
    DictKeysGilesOutlet['monitoring_tol_rad'] = kwargs.get('monitoring_tol_rad',  1e-6)               # recommended value
    DictKeysGilesOutlet['monitoring_var'] = 'psta'
    DictKeysGilesOutlet['monitoring_file'] = 'LOGS/%s_'%FamilyName
    DictKeysGilesOutlet['monitoring_period'] = kwargs.get('monitoring_period',  20)                   # recommended value   
    DictKeysGilesOutlet['monitoring_flag'] = kwargs['GilesMonitoringFlag']                            # automatically computed

    # keys relative to the outlet NSCBC/Giles
    DictKeysGilesOutlet['nscbc_relaxo'] = kwargs.get('nscbc_relaxo',  200.)                           # recommended value
    DictKeysGilesOutlet['giles_relaxo'] = kwargs.get('giles_relaxo',  200.)                           # recommended value

    # default option: RadialEquilibrium
    if kwargs['option'] == 'RadialEquilibrium':
        DictKeysGilesOutlet['monitoring_indpiv'] = kwargs.get('IndexPivot',1)                         # default value
        DictKeysGilesOutlet['monitoring_pressure'] = kwargs.get('Pressure',None)                      # given by the user

        # Case for Valve Law
        valve_ref_type = kwargs.get('valve_ref_type',0)
        if valve_ref_type!=0:
            DictKeysGilesOutlet['monitoring_valve_ref_type'] = valve_ref_type
            DictKeysGilesOutlet['monitoring_valve_ref_pres'] = valve_ref_pres
            DictKeysGilesOutlet['monitoring_valve_ref_mflow'] = valve_ref_mflow
            DictKeysGilesOutlet['monitoring_valve_relax'] = valve_relax
            
    # imposed cartography from a CGNS file
    elif kwargs['option'] == 'file':

        # get the data from the file
        bnd_data = C.convertFile2PyTree(kwargs['filename'])

        # get Node FlowSolutionCenters
        # we suppose here that the variable names are correctly set for Giles outpres
        FS = I.getNodeFromName(bnd_data, I.__FlowSolutionCenters__)

        # store data in a dictionnary
        ImposedVariables = dict()
        for child in I.getChildren(FS):
            childname = I.getName(child)            
            if childname != 'GridLocation':
                ImposedVariables[childname] = np.asfortranarray(I.getValue(child))

        #print(ImposedVariables)
        
        # build node for BCDataSet
        BCDataSet = I.newBCDataSet(name='BCDataSet#Init', value='Null',
            gridLocation='FaceCenter', parent=bc)
        
        # add the data in BCDataSet
        J.set(BCDataSet, 'DirichletData', childType='BCData_t', **ImposedVariables)


        #raise Exception('Giles BC with file imposed not implemented yet')


    # set the BC with keys
    J.set(bc, '.Solver#BC',**DictKeysGilesOutlet)



def setBC_giles_inlet(t, bc, FluidProperties, ReferenceValues, FamilyName, **kwargs):
    '''
    Impose a Boundary Condition ``giles_in``.

    .. note::
    
        see theoretical report: /home/bfrancoi/NSCBC/RapportsONERA/SONICE-TF-S2.1.4.1.2_NSCBCgilesInletSteadyStructured_Final3.pdf

    Parameters
    ----------

            t : PyTree
                Tree to modify

            bc : CGNS node of type BC_t
                 BC node attached to the family in the which the boundary condition is applied

            FluidProperties : dict
                as obtained from :py:func:`computeFluidProperties`

            ReferenceValues : dict
                as obtained from :py:func:`computeReferenceValues`

            FamilyName : str
                Name of the family on which the boundary condition will be imposed

            kwargs : dict
                Parameters defined by the user: FamilyName, NbModesFourierGiles, monitoring_flag, option
            
    '''

    # creation of dictionnary of keys for Giles inlet BC  
    DictKeysGilesInlet = {}

    # keys relative to NSCBC
    DictKeysGilesInlet['type'] = 'nscbc_in'                                                                   # mandatory key to have NSCBC-Giles treatment
    DictKeysGilesInlet['nscbc_giles'] = 'statio'                                                              # mandatory key to have NSCBC-Giles treatment
    DictKeysGilesInlet['nscbc_interpbc'] = 'linear'                                                           # mandatory value
    DictKeysGilesInlet['nscbc_fluxt'] = kwargs.get('nscbc_fluxt', 'fluxBothTransv')                           # recommended value - possible keys : 'classic'; 'fluxInviscidTransv'; 'fluxBothTransv' 
    DictKeysGilesInlet['nscbc_surf'] = kwargs.get('nscbc_surf',  'revolution')                                # recommended value - possible keys : 'flat', 'revolution' 
    DictKeysGilesInlet['nscbc_outwave'] = kwargs.get('nscbc_outwave',  'grad_etat')                           # recommended value - possible keys : 'grad_etat'; 'extrap_flux'
    DictKeysGilesInlet['nscbc_velocity_scale'] = kwargs.get('nscbc_velocity_scale',kwargs['VelocityScale'])   # default value - sound velocity
    DictKeysGilesInlet['nscbc_viscwall_len'] = kwargs.get('nscbc_viscwall_len', 5.e-4)                        # default value, could be updated by the user if convergence issue

    if kwargs.get('nscbc_viscwall_len_hub') is not None:
        DictKeysGilesInlet['nscbc_viscwall_len_hub'] = kwargs.get('nscbc_viscwall_len_hub')                    # value of nscbc_viscwall_len for the hub only
    if kwargs.get('nscbc_viscwall_len_carter') is not None:
        DictKeysGilesInlet['nscbc_viscwall_len_carter'] = kwargs.get('nscbc_viscwall_len_carter')              # value of nscbc_viscwall_len for the hub only           


    # keys relative to the Giles treatment 
    DictKeysGilesInlet['giles_opt'] = kwargs.get('giles_relax_opt', 'relax')                             # mandatory key for NSCBC-Giles treatment
    DictKeysGilesInlet['giles_restric_relax'] = 'inactive'                                               # mandatory key for NSCBC-Giles treatment
    DictKeysGilesInlet['giles_exact_lodi'] = kwargs.get('giles_exact_lodi',  'active')                   # recommended value - possible keys: 'inactive', 'partial', 'active'
    DictKeysGilesInlet['giles_nbMode'] = kwargs.get('NbModesFourierGiles')                               # to be given by the user - recommended value : ncells_theta/2 + 1 (odd_value)

    # keys relative to the monitoring and radii calculus - monitoring data stored in LOGS
    DictKeysGilesInlet['bnd_monitoring'] = 'active'                                                      # recommended value
    DictKeysGilesInlet['monitoring_comp_rad'] = 'auto'                                                   # recommended value - possible keys: 'from_file', 'monofenetre'
    DictKeysGilesInlet['monitoring_tol_rad'] = kwargs.get('monitoring_tol_rad',  1e-6)                   # recommended value
    DictKeysGilesInlet['monitoring_var'] = 'psta pgen Tgen ux uy uz diffPgen diffTgen diffVel'
    DictKeysGilesInlet['monitoring_file'] = 'LOGS/%s_'%FamilyName
    DictKeysGilesInlet['monitoring_period'] = kwargs.get('monitoring_period',  20)                       # recommended value
    DictKeysGilesInlet['monitoring_flag'] = kwargs['GilesMonitoringFlag']                                # automatically computed

    # keys relative to the inlet BC
    DictKeysGilesInlet['nscbc_in_type'] = 'htpt'                                                         # mandatory key to have NSCBC-Giles treatment
    # - numerics -
    DictKeysGilesInlet['nscbc_relaxi1'] = kwargs.get('nscbc_relaxi1',  500.)                             # recommended value
    DictKeysGilesInlet['nscbc_relaxi2'] = kwargs.get('nscbc_relaxi2',  500.)                             # recommended value
    giles_relax_in = kwargs.get('giles_relax_in',  [200.,  500.,  1000.,  1000.])                        # recommended value
    DictKeysGilesInlet['giles_relax_in1'] = giles_relax_in[0]
    DictKeysGilesInlet['giles_relax_in2'] = giles_relax_in[1]
    DictKeysGilesInlet['giles_relax_in3'] = giles_relax_in[2]
    DictKeysGilesInlet['giles_relax_in4'] = giles_relax_in[3]    
    if kwargs['option'] == 'uniform':
        # - physical quantities -
        DictKeysGilesInlet['stagnation_enthalpy'] = kwargs.get('stagnation_enthalpy',FluidProperties['cp'] * ReferenceValues['TemperatureStagnation'])   # to be given by the user
        DictKeysGilesInlet['stagnation_pressure'] = kwargs.get('stagnation_pressure',ReferenceValues['PressureStagnation'])                              # to be given by the user
        DictKeysGilesInlet['vtx'] = kwargs.get('VelocityUnitVectorX',1.)                                                                                 # to be given by the user
        DictKeysGilesInlet['vtr'] = kwargs.get('VelocityUnitVectorR',0.)                                                                                 # to be given by the user
        DictKeysGilesInlet['vtt'] = kwargs.get('VelocityUnitVectorTheta',0.)                                                                             # to be given by the user
        # - turbulent quantities -
        turbDict = getPrimitiveTurbulentFieldForInjection(FluidProperties, ReferenceValues)
        for CGNSTurbVariable in turbDict.keys():
            elsATurbVariable = translateVariablesFromCGNS2Elsa([CGNSTurbVariable])[0]
            DictKeysGilesInlet[elsATurbVariable] = turbDict[CGNSTurbVariable]

    elif kwargs['option'] == 'file':

        # get the data from the file
        bnd_data = C.convertFile2PyTree(kwargs['filename'])

        # get Node FlowSolutionCenters
        # we suppose here that the variable names are correctly set for Giles inj1
        FS = I.getNodeFromName(bnd_data, I.__FlowSolutionCenters__)

        # store data in a dictionnary
        ImposedVariables = dict()
        for child in I.getChildren(FS):
            childname = I.getName(child)            
            if childname != 'GridLocation':
                ImposedVariables[childname] = np.asfortranarray(I.getValue(child))

        #print(ImposedVariables)
        
        # build node for BCDataSet
        BCDataSet = I.newBCDataSet(name='BCDataSet#Init', value='Null',
            gridLocation='FaceCenter', parent=bc)
        
        # add the data in BCDataSet
        J.set(BCDataSet, 'DirichletData', childType='BCData_t', **ImposedVariables)

        

        
        
        #raise Exception('Giles BC with file imposed not implemented yet')

    # set the BC with keys
    J.set(bc, '.Solver#BC',**DictKeysGilesInlet)



def setBC_outmfr2(t, FamilyName, MassFlow=None, groupmassflow=1, ReferenceValues=None, TurboConfiguration=None):
    '''
    Set an outflow boundary condition of type ``outmfr2``.

    .. note:: see `elsA Tutorial about outmfr2 condition <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#outmfr2/>`_

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        MassFlow : :py:class:`float` or :py:obj:`None`
            Total massflow on the family (with the same **groupmassflow**).
            If :py:obj:`None`, the reference massflow in **ReferenceValues**
            divided by the appropriate fluxcoeff is taken.

            .. attention::
                It has to be the massflow through the simulated section only,
                not on the full 360 degrees configuration (except if the full
                circonference is simulated).

        groupmassflow : int
            Index used to link participating patches to this boundary condition.
            If several BC ``outmfr2`` are defined, **groupmassflow** has to be
            incremented for each family.

        ReferenceValues : :py:class:`dict` or :py:obj:`None`
            dictionary as obtained from :py:func:`computeReferenceValues`. Can
            be :py:obj:`None` only if **MassFlow** is not :py:obj:`None`.

        TurboConfiguration : :py:class:`dict` or :py:obj:`None`
            dictionary as obtained from :py:func:`getTurboConfiguration`. Can
            be :py:obj:`None` only if **MassFlow** is not :py:obj:`None`.

    '''
    if MassFlow is None:
        bc = C.getFamilyBCs(t, FamilyName)[0]
        zone = I.getParentFromType(t, bc, 'Zone_t')
        row = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
        rowParams = TurboConfiguration['Rows'][row]
        fluxcoeff = rowParams['NumberOfBlades'] / float(rowParams['NumberOfBladesSimulated'])
        MassFlow = ReferenceValues['MassFlow'] / fluxcoeff
    else:
        bc = None

    ImposedVariables = dict(globalmassflow=MassFlow, groupmassflow=groupmassflow)

    setBCwithImposedVariables(t, FamilyName, ImposedVariables,
        FamilyBC='BCOutflowSubsonic', BCType='outmfr2', bc=bc)

def setBC_MachFromMassFlow(t, FamilyName, Mach=None, PressureStagnation=None, TemperatureStagnation=None, Surface=None, groupmassflow=1, ReferenceValues=None, TurboConfiguration=None, FluidProperties=None):
    '''
    Set an outflow boundary condition of type ``outmfr2``.

    .. note:: see `elsA Tutorial about outmfr2 condition <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#outmfr2/>`_

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        Mach : :py:class:`float` or :py:obj:`None`
            If :py:obj:`None`, the reference Mach in **ReferenceValues** is taken.
        
        PressureStagnation : :py:class:`float` or :py:obj:`None`
            If :py:obj:`None`, the reference PressureStagnation in **ReferenceValues** is taken.

        TemperatureStagnation : :py:class:`float` or :py:obj:`None`
            If :py:obj:`None`, the reference TemperatureStagnation in **ReferenceValues** is taken.

        Surface : :py:class:`float` or :py:obj:`None`
            If :py:obj:`None`, the reference Surface in **ReferenceValues** is taken.

        groupmassflow : int
            Index used to link participating patches to this boundary condition.
            If several BC ``outmfr2`` are defined, **groupmassflow** has to be
            incremented for each family.

        ReferenceValues : :py:class:`dict` or :py:obj:`None`
            dictionary as obtained from :py:func:`computeReferenceValues`.
        
        TurboConfiguration : :py:class:`dict` or :py:obj:`None`
            dictionary as obtained from :py:func:`getTurboConfiguration`. Can
            be :py:obj:`None` only if **MassFlow** is not :py:obj:`None`.

        FluidProperties : :py:class:`dict` or :py:obj:`None`
            dictionary as obtained from :py:func:`computeFluidProperties`. 
    '''
    if Mach is None:
        Mach = ReferenceValues['Mach']
    if PressureStagnation is None:
        PressureStagnation = ReferenceValues['PressureStagnation']
    if TemperatureStagnation is None:
        TemperatureStagnation = ReferenceValues['TemperatureStagnation']
    if Surface is None:
        Surface = ReferenceValues['Surface']

    # Massflow on 360°
    MassFlow = massflowFromMach(Mach, S=Surface, Pt=PressureStagnation, Tt=TemperatureStagnation, r=FluidProperties['IdealGasConstant'], gamma=FluidProperties['Gamma'])
    # Compute massflow on the required section on Family
    bc = C.getFamilyBCs(t, FamilyName)[0]
    zone = I.getParentFromType(t, bc, 'Zone_t')
    row = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
    try:
        rowParams = TurboConfiguration['Rows'][row]
        fluxcoeff = rowParams['NumberOfBlades'] / float(rowParams['NumberOfBladesSimulated'])
    except KeyError:
        # To handle workflows without TurboConfiguration
        fluxcoeff = 1
    MassFlow /= fluxcoeff

    setBC_outmfr2(t, FamilyName, MassFlow=MassFlow, groupmassflow=groupmassflow)

def setBCwithImposedVariables(t, FamilyName, ImposedVariables, FamilyBC, BCType,
    bc=None, BCDataSetName='BCDataSet#Init', BCDataName='DirichletData', variableForInterpolation='ChannelHeight'):
    '''
    Generic function to impose quantities on a Boundary Condition.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        ImposedVariables : str
            When using a function to impose the radial profile of one or several quantities, 
            it defines the variable used as the argument of this function.
            Must be 'ChannelHeight' (default value) or 'Radius'.riable names and values must be either:

                * scalars: in that case they are imposed once for the
                  family **FamilyName** in the corresponding ``Family_t`` node.

                * numpy arrays: in that case they are imposed for the ``BC_t``
                  node **bc**.

                * functions: in that case the function defined a profile depending on radius.
                  It is evaluated in each cell on the **bc**.
            
            They may be a combination of three.

        bc : PyTree
            ``BC_t`` node on which the boundary condition will be imposed. Must
            be :py:obj:`None` if the condition must be imposed once in the
            ``Family_t`` node.

        BCDataSetName : str
            Name of the created node of type ``BCDataSet_t``. Default value is
            'BCDataSet#Init'

        BCDataName : str
            Name of the created node of type ``BCData_t``. Default value is
            'DirichletData'
        
        variableForInterpolation : str
            When using a function to impose the radial profile of one or several quantities, 
            it defines the variable used as the argument of this function.
            Must be 'ChannelHeight' (default value), 'Radius', 'CoordinateX', 'CoordinateY' or 'CoordinateZ'.

    See also
    --------

    setBC_inj1, setBC_outpres, setBC_outmfr2

    '''
    # MANDOTORY to prevent this function modifying ImposedVariables.
    # Otherwise it does not work when imposing a radial profile on several BC nodes in the same Family
    ImposedVariables = copy.deepcopy(ImposedVariables)

    FamilyNode = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    I._rmNodesByName(FamilyNode, '.Solver#BC')
    I._rmNodesByType(FamilyNode, 'FamilyBC_t')
    I.newFamilyBC(value=FamilyBC, parent=FamilyNode)

    if all([np.ndim(v)==0 and not callable(v) for v in ImposedVariables.values()]):
        #  All data are scalars (not arrays or functions)
        # The implementation of the BC is fully in the Family node
        checkVariables(ImposedVariables)
        ImposedVariables = translateVariablesFromCGNS2Elsa(ImposedVariables)
        J.set(FamilyNode, '.Solver#BC', type=BCType, **ImposedVariables)

    else:
        #  Somme data are arrays or functions
        # The implementation of the BC is in each BC nodes
        assert bc is not None
        J.set(bc, '.Solver#BC', type=BCType)

        zone = I.getParentFromType(t, bc, 'Zone_t') 
        zone = I.copyRef(zone)
        I._rmNodesFromName(zone, 'FlowSolution#Init')
        I._renameNode(zone, 'FlowSolution#Height', 'FlowSolution')
        extractedBC = C.extractBCOfName(zone, I.getName(bc),reorder=False)
        extractedBC = C.node2Center(extractedBC)

        # Check if some data are functions to interpolate, to prepare the dedicated treatment
        varForInterp = None
        for var, value in ImposedVariables.items():
            if callable(value): # it is a function
                if variableForInterpolation in ['Radius', 'radius']:
                    varForInterp, _ = J.getRadiusTheta(extractedBC)
                elif variableForInterpolation == 'ChannelHeight':
                    try:
                        varForInterp = I.getValue(I.getNodeFromName(extractedBC, 'ChannelHeight'))
                    except:
                        ERR_MSG = 'ChannelHeight is mandatory to impose a radial profile based on this quantity, ' 
                        ERR_MSG+= 'but it has not been computed yet. '
                        ERR_MSG+= 'Please compute it earlier in the process or change varForInterpolation.'
                        raise Exception(J.FAIL + ERR_MSG + J.ENDC)
                elif variableForInterpolation.startsWith('Coordinate'):
                    varForInterp = I.getValue(I.getNodeFromName(extractedBC, variableForInterpolation))
                else:
                    raise ValueError('varForInterpolation must be ChannelHeight, Radius, CoordinateX, CoordinateY or CoordinateZ')

                break

        if varForInterp is None:
            varForInterp = I.getValue(I.getNodeFromName(extractedBC, 'CoordinateX'))

        
        for var, value in ImposedVariables.items():
            ImposedVariables[var] = np.atleast_1d(np.copy(varForInterp,order='K'))
            # Reshape data if needed to have a 2D map to impose on the BC at face centers
            if callable(value):
                # value is a function to evaluate in each cell for the quantity varForInterp
                ImposedVariables[var][:] = value(varForInterp)
            else:
                ImposedVariables[var][:] = value

            # data shall be 1D https://elsa.onera.fr/issues/11219
            ImposedVariables[var] = ImposedVariables[var].ravel(order='K')
        
        checkVariables(ImposedVariables)

        BCDataSet = I.newBCDataSet(name=BCDataSetName, value='Null',
            gridLocation='FaceCenter', parent=bc)

        J.set(BCDataSet, BCDataName, childType='BCData_t', **ImposedVariables)

def checkVariables(ImposedVariables):
    '''
    Check that variables in the input dictionary are well defined. Raise a
    ``ValueError`` if not.

    Parameters
    ----------

        ImposedVariables : dict
            Each key is a variable name. Based on this name, the value (float or
            numpy.array) is checked.
            For instance:

                * Variables such as pressure, temperature or turbulent quantities
                  must be strictly positive.

                * Components of a unit vector must be between -1 and 1.

    '''
    posiviteVars = ['PressureStagnation', 'EnthalpyStagnation',
        'stagnation_pressure', 'stagnation_enthalpy', 'stagnation_temperature',
        'Pressure', 'pressure', 'Temperature', 'wall_temp',
        'TurbulentEnergyKinetic', 'TurbulentDissipationRate', 'TurbulentDissipation', 'TurbulentLengthScale',
        'TurbulentSANuTilde', 'globalmassflow', 'MassFlow', 'surf_massflow']
    unitVectorComponent = ['VelocityUnitVectorX', 'VelocityUnitVectorY', 'VelocityUnitVectorZ',
        'txv', 'tyv', 'tzv']

    def positive(value):
        if isinstance(value, np.ndarray): return np.all(value>0)
        else: return value>0

    def unitComponent(value):
        if isinstance(value, np.ndarray): return np.all(np.absolute(value)<=1)
        else: return abs(value)<=1

    for var, value in ImposedVariables.items():
        if var in posiviteVars and not positive(value):
            raise ValueError('{} must be positive, but here it is equal to {}'.format(var, value))
        elif var in unitVectorComponent and not unitComponent(value):
            raise ValueError('{} must be between -1 and +1, but here it is equal to {}'.format(var, value))

def translateVariablesFromCGNS2Elsa(Variables):
    '''
    Translate names in **Variables** from CGNS standards to elsA names for
    boundary conditions.

    Parameters
    ----------

        Variables : :py:class:`dict` or :py:class:`list` or :py:class:`str`
            Could be eiter:

                * a :py:class:`dict` with keys corresponding to variables names

                * a :py:class:`list` of variables names

                * a :py:class:`str` as a single variable name

    Returns
    -------

        NewVariables : :py:class:`dict` or :py:class:`list` or :py:class:`str`
            Depending on the input type, return the same object with variable
            names translated to elsA standards.

    '''
    CGNS2ElsaDict = dict(
        PressureStagnation       = 'stagnation_pressure',
        EnthalpyStagnation       = 'stagnation_enthalpy',
        TemperatureStagnation    = 'stagnation_temperature',
        Pressure                 = 'pressure',
        MassFlow                 = 'globalmassflow',
        SurfacicMassFlow         = 'surf_massflow',
        VelocityUnitVectorX      = 'txv',
        VelocityUnitVectorY      = 'tyv',
        VelocityUnitVectorZ      = 'tzv',
        TurbulentSANuTilde       = 'inj_tur1',
        TurbulentEnergyKinetic   = 'inj_tur1',
        TurbulentDissipationRate = 'inj_tur2',
        TurbulentDissipation     = 'inj_tur2',
        TurbulentLengthScale     = 'inj_tur2',
        VelocityCorrelationXX    = 'inj_tur1',
        VelocityCorrelationXY    = 'inj_tur2', 
        VelocityCorrelationXZ    = 'inj_tur3',
        VelocityCorrelationYY    = 'inj_tur4', 
        VelocityCorrelationYZ    = 'inj_tur5', 
        VelocityCorrelationZZ    = 'inj_tur6',
        Intermittency            = 'inj_tur3',
        MomentumThicknessReynolds= 'inj_tur4',
    )
    if 'VelocityCorrelationXX' in Variables:
        # For RSM models
        CGNS2ElsaDict['TurbulentDissipationRate'] = 'inj_tur7'

    elsAVariables = CGNS2ElsaDict.values()

    if isinstance(Variables, dict):
        NewVariables = dict()
        for var, value in Variables.items():
            if var == 'groupmassflow':
                NewVariables[var] = int(value)                    
            elif var in elsAVariables:
                NewVariables[var] = float(value)
            elif var in CGNS2ElsaDict:
                NewVariables[CGNS2ElsaDict[var]] = float(value)
            else:
                NewVariables[var] = float(value)
        return NewVariables
    elif isinstance(Variables, list):
        NewVariables = []
        for var in Variables:
            if var in elsAVariables:
                NewVariables.append(var)
            else:
                NewVariables.append(CGNS2ElsaDict[var])
        return NewVariables
    elif isinstance(Variables, str):
        if Variables in elsAVariables:
            return CGNS2ElsaDict[Variables]
    else:
        raise TypeError('Variables must be of type dict, list or string')


@J.mute_stdout
def setBC_stage_mxpl(t, left, right, method='globborder_dict'):
    '''
    Set a mixing plane condition between families **left** and **right**.

    .. important : This function has a dependency to the ETC module.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        left : str
            Name of the family on the left side.

        right : str
            Name of the family on the right side.

        method : optional, str
            Method used to compute the globborder. The default value is
            ``'globborder_dict'``, it corresponds to the ETC topological
            algorithm.
            Another possible value is ``'poswin'`` to use the geometrical
            algorithm in *turbo* (in this case, *turbo* environment must be
            sourced).
    '''

    import etc.transform as trf

    if method == 'globborder_dict':
        t = trf.defineBCStageFromBC(t, left)
        t = trf.defineBCStageFromBC(t, right)
        t, stage = trf.newStageMxPlFromFamily(t, left, right)

    elif method == 'poswin':
        t = trf.defineBCStageFromBC(t, left)
        t = trf.defineBCStageFromBC(t, right)

        gbdu = computeGlobborderPoswin(t, left)
        # print("newStageMxPlFromFamily(up): gbdu = {}".format(gbdu))
        ups = []
        for bc in C.getFamilyBCs(t, left):
          bcpath = I.getPath(t, bc)
          bcu = trf.BCStageMxPlUp(t, bc)
          globborder = bcu.glob_border(left, opposite=right)
          globborder.i_poswin = gbdu[bcpath]['i_poswin']
          globborder.j_poswin = gbdu[bcpath]['j_poswin']
          globborder.glob_dir_i = gbdu[bcpath]['glob_dir_i']
          globborder.glob_dir_j = gbdu[bcpath]['glob_dir_j']
          ups.append(bcu)

        # Downstream BCs declaration
        gbdd = computeGlobborderPoswin(t, right)
        # print("newStageMxPlFromFamily(down): gbdd = {}".format(gbdd))
        downs = []
        for bc in C.getFamilyBCs(t, right):
          bcpath = I.getPath(t, bc)
          bcd = trf.BCStageMxPlDown(t, bc)
          globborder = bcd.glob_border(right, opposite=left)
          globborder.i_poswin = gbdd[bcpath]['i_poswin']
          globborder.j_poswin = gbdd[bcpath]['j_poswin']
          globborder.glob_dir_i = gbdd[bcpath]['glob_dir_i']
          globborder.glob_dir_j = gbdd[bcpath]['glob_dir_j']
          downs.append(bcd)

        # StageMxpl declaration
        stage = trf.BCStageMxPl(t, up=ups, down=downs)
    else:
        raise Exception

    stage.jtype = 'nomatch_rad_line'
    stage.create()

    setRotorStatorFamilyBC(t, left, right)

def setBC_giles_stage_mxpl(t, left, right, method = 'Robust', **kwargs):
    '''
    Set a mixing plane condition between families **left** and **right** using the Giles treatment.
    The setting of the mixing plane with giles is different from classical mixing plane. Therefore, 
    a dedicated function is used. 

    .. note:: 
     
        see theoretical report: /home/bfrancoi/NSCBC/RapportsONERA/SONICE-TF-S2.1.4.1.2_NSCBCgilesMxplStructured.pdf

    Parameters
    ----------

        t : PyTree
            Tree to modify

        left : str
            Name of the family on the left side (upstream).

        right : str
            Name of the family on the right side (downstream).

        method : optional, str
            Add the type of mxpl treatment here.

            * Robust : 
              simplified notation to indicate the Alpha method (see technical report above). Non conservative treatement but non reflective. 
              Averaging of the quantities: PsPtHt and direction of speed (as inlet and outlet). Works as inlet and outlet are combined.

            * Conservative: 
              simplified notation to indicate the Cons2 method (see technical report above). Conservative and non-reflective treatment.
              Maybe tricky to converge. Avoid coarse grid.

    '''

    # add BCGilesMxpl in family left and family right
    MxplLeftNode = I.getNodeFromNameAndType(t, left, 'Family_t')
    I.createChild(MxplLeftNode,'FamilyBC','FamilyBC_t',value='BCGilesMxPl')
    MxplRightNode = I.getNodeFromNameAndType(t, right, 'Family_t')
    I.createChild(MxplRightNode,'FamilyBC','FamilyBC_t',value='BCGilesMxPl')

    # creation of dictionnary of keys for Giles mxpl left 
    DictKeysGilesMxpl = {}

    # keys relative to NSCBC
    DictKeysGilesMxpl['type'] = 'nscbc_mxpl'                                                        # mandatory key to have NSCBC-Giles treatment
    DictKeysGilesMxpl['nscbc_giles'] = 'statio'                                                     # mandatory key to have NSCBC-Giles treatment
    #DictKeysGilesMxpl['nscbc_interpbc'] = 'linear'                                                 # necessary for Mxpl?
    DictKeysGilesMxpl['nscbc_fluxt'] = kwargs.get('nscbc_fluxt', 'fluxInviscidTransv')              # recommended value - possible keys : 'classic'; 'fluxInviscidTransv'; 'fluxBothTransv'
    DictKeysGilesMxpl['nscbc_surf'] = kwargs.get('nscbc_surf',  'revolution')                       # recommended value - possible keys : 'flat', 'revolution'
    DictKeysGilesMxpl['nscbc_outwave'] = kwargs.get('nscbc_outwave',  'grad_etat')                  # recommended value - possible keys : 'grad_etat'; 'extrap_flux'
    DictKeysGilesMxpl['nscbc_velocity_scale'] = kwargs.get('nscbc_velocity_scale',kwargs['VelocityScale'])    # default value - reference sound velocity 
    DictKeysGilesMxpl['nscbc_viscwall_len'] = kwargs.get('nscbc_viscwall_len', 5.e-4)               # default value, could be updated by the user if convergence issue

    if kwargs.get('nscbc_viscwall_len_hub') is not None:
        DictKeysGilesMxpl['nscbc_viscwall_len_hub'] = kwargs.get('nscbc_viscwall_len_hub')                    # value of nscbc_viscwall_len for the hub only
    if kwargs.get('nscbc_viscwall_len_carter') is not None:
        DictKeysGilesMxpl['nscbc_viscwall_len_carter'] = kwargs.get('nscbc_viscwall_len_carter')              # value of nscbc_viscwall_len for the hub only           


    # keys relative to the Giles treatment 
    DictKeysGilesMxpl['giles_opt'] = 'relax'                                                        # mandatory key for NSCBC-Giles treatment
    DictKeysGilesMxpl['giles_restric_relax'] = 'inactive'                                           # mandatory key for NSCBC-Giles treatment
    DictKeysGilesMxpl['giles_exact_lodi'] = kwargs.get('giles_exact_lodi',  'partial')               # recommended value - possible keys: 'inactive', 'partial', 'active'
    DictKeysGilesMxpl['giles_nbMode'] = kwargs.get('NbModesFourierGiles')                           # given by the user - recommended value : ncells_theta/2 + 1 (odd_value)

    # keys relative to the mxpl NSCBC/Giles
    if method == 'Robust':
        DictKeysGilesMxpl['nscbc_mxpl_type'] = kwargs.get('nscbc_mxpl_type',  'pshtpt')                 
        DictKeysGilesMxpl['nscbc_mxpl_avermean'] = kwargs.get('nscbc_mxpl_avermean',  'pshtpt')         
    elif method == 'Conservative':
        DictKeysGilesMxpl['nscbc_mxpl_type'] = kwargs.get('nscbc_mxpl_type',  'flux')                 
        DictKeysGilesMxpl['nscbc_mxpl_avermean'] = kwargs.get('nscbc_mxpl_avermean',  'flux')         
    DictKeysGilesMxpl['nscbc_mxpl_flag'] = kwargs['nscbc_mxpl_flag']                                # automatically computed, different for each pair of Mxpl planes
    DictKeysGilesMxpl['nscbc_relaxi1'] = kwargs.get('nscbc_relaxi1',  20.)                          # recommended value
    DictKeysGilesMxpl['nscbc_relaxi2'] = kwargs.get('nscbc_relaxi2',  20.)                          # recommended value
    DictKeysGilesMxpl['nscbc_relaxo'] = kwargs.get('nscbc_relaxo',  20.)                            # recommended value
    DictKeysGilesMxpl['giles_relax_in1'] = kwargs.get('giles_relax_in1',  50.)                      # recommended value
    DictKeysGilesMxpl['giles_relax_in2'] = kwargs.get('giles_relax_in2',  50.)                      # recommended value
    DictKeysGilesMxpl['giles_relax_in3'] = kwargs.get('giles_relax_in3',  50.)                      # recommended value
    DictKeysGilesMxpl['giles_relax_in4'] = kwargs.get('giles_relax_in4',  50.)                      # recommended value
    DictKeysGilesMxpl['giles_relax_out'] = kwargs.get('giles_relax_out',  50.)                      # recommended value 

    # keys relative to the monitoring and radii calculus - monitoring data stored in LOGS
    DictKeysGilesMxpl['bnd_monitoring'] = 'active'                                                  # recommended value
    DictKeysGilesMxpl['monitoring_comp_rad'] = 'auto'                                               # recommended value - possible keys: 'from_file', 'monofenetre'
    DictKeysGilesMxpl['monitoring_tol_rad'] = kwargs.get('monitoring_tol_rad',  1e-6)               # recommended value - decrease value if the mesh is coarse
    DictKeysGilesMxpl['monitoring_var'] = 'psta  pgen Tgen ux uy uz diffPgen diffTgen diffVel'
    DictKeysGilesMxpl['monitoring_period'] = kwargs.get('monitoring_period',  20)                   # recommended value   

    # define parameter for left and right interface
    
    LogRootName = 'Mxpl_%i_%i'%(kwargs['GilesMonitoringFlag_left'],kwargs['GilesMonitoringFlag_right']) # give a common LogRootName for the Mxpl interface (upstream and downstream)
    DictKeysGilesMxpl_left = DictKeysGilesMxpl.copy()
    DictKeysGilesMxpl_left['monitoring_flag'] = kwargs['GilesMonitoringFlag_left']                  # automatically computed, must be different from other Giles BC, including right BC of Mxpl
    DictKeysGilesMxpl_left['monitoring_file'] = 'LOGS/%s_%i_'%(LogRootName,kwargs['GilesMonitoringFlag_left'])
    DictKeysGilesMxpl_right = DictKeysGilesMxpl.copy()
    DictKeysGilesMxpl_right['monitoring_flag'] = kwargs['GilesMonitoringFlag_right']                # automatically computed, must be different from other Giles BC, including left BC of Mxpl
    DictKeysGilesMxpl_right['monitoring_file'] = 'LOGS/%s_%i_'%(LogRootName,kwargs['GilesMonitoringFlag_right'])

     # set the BCs left with keys
    ListBCNodes_left = C.getFamilyBCs(t,left)
    for BCNode_left in ListBCNodes_left:
        J.set(BCNode_left, '.Solver#BC',**DictKeysGilesMxpl_left)

    # set the BCs right with keys
    ListBCNodes_right = C.getFamilyBCs(t,right)
    for BCNode_right in ListBCNodes_right:
        J.set(BCNode_right, '.Solver#BC',**DictKeysGilesMxpl_right)



@J.mute_stdout
def setBC_stage_mxpl_hyb(t, left, right, nbband=100, c=0.3):
    '''
    Set a hybrid mixing plane condition between families **left** and **right**.

    .. important : This function has a dependency to the ETC module.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        left : str
            Name of the family on the left side.

        right : str
            Name of the family on the right side.

        nbband : int
            Number of points in the radial distribution to compute.

        c : float
            Parameter for the distribution of radial points.

    '''

    import etc.transform as trf

    t = trf.defineBCStageFromBC(t, left)
    t = trf.defineBCStageFromBC(t, right)

    t, stage = trf.newStageMxPlHybFromFamily(t, left, right)
    stage.jtype = 'nomatch_rad_line'
    stage.hray_tolerance = 1e-16
    for stg in stage.down:
        filename = "state_radius_{}_{}.plt".format(right, nbband)
        radius = stg.repartition(mxpl_dirtype='axial',
                                 filename=filename, fileformat="bin_tp")
        radius.compute(t, nbband=nbband, c=c)
        radius.write()
    for stg in stage.up:
        filename = "state_radius_{}_{}.plt".format(left, nbband)
        radius = stg.repartition(mxpl_dirtype='axial',
                                 filename=filename, fileformat="bin_tp")
        radius.compute(t, nbband=nbband, c=c)
        radius.write()
    stage.create()

    setRotorStatorFamilyBC(t, left, right)


@J.mute_stdout
def setBC_stage_red(t, left, right, stage_ref_time):
    '''
    Set a RNA condition between families **left** and **right**.

    .. important : This function has a dependency to the ETC module.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        left : str
            Name of the family on the left side.

        right : str
            Name of the family on the right side.

        stage_ref_time : float
            Reference period on the simulated azimuthal extension.
    '''

    import etc.transform as trf

    t = trf.defineBCStageFromBC(t, left)
    t = trf.defineBCStageFromBC(t, right)

    t, stage = trf.newStageRedFromFamily(
        t, left, right, stage_ref_time=stage_ref_time)
    stage.create()

    setRotorStatorFamilyBC(t, left, right)


@J.mute_stdout
def setBC_stage_red_hyb(t, left, right, stage_ref_time):
    '''
    Set a hybrid RNA condition between families **left** and **right**.

    .. important : This function has a dependency to the ETC module.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        left : str
            Name of the family on the left side.

        right : str
            Name of the family on the right side.

        stage_ref_time : float
            Reference period on the simulated azimuthal extension.

    '''

    import etc.transform as trf

    t = trf.defineBCStageFromBC(t, left)
    t = trf.defineBCStageFromBC(t, right)

    t, stage = trf.newStageRedHybFromFamily(
        t, left, right, stage_ref_time=stage_ref_time)
    stage.create()

    for gc in I.getNodesFromType(t, 'GridConnectivity_t'):
        I._rmNodesByType(gc, 'FamilyBC_t')



@J.mute_stdout
def setBC_stage_choro(t, left, right, method='globborder_dict', stage_choro_type='characteristic', harm_freq_comp=1, jtype = 'nomatch_rad_line'):
    '''
    Set a chorochronic interface condition between families **left** and **right**.

    .. important : This function has a dependency to the ETC module.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        left : str
            Name of the family on the left side.

        right : str
            Name of the family on the right side.

        method : optional, str
            Method used to compute the globborder. The default value is
            ``'globborder_dict'``, it corresponds to the ETC topological
            algorithm.
            Another possible value is ``'poswin'`` to use the geometrical
            algorithm in *turbo* (in this case, *turbo* environment must be
            sourced).

        stage_choro_type : str
            Type of chorochronic interface:
                characteristic (default) : condition based on characteristic relations.
                half_sum : condition based on half-sum of values.

        harm_freq_comp : str
            Frequency of harmonics computation

        jtype : str
            Specifies the type of join:
                match_rad_line : coincident radial match along lines (for “stage”-like turbomachine conditions);
                nomatch_rad_line : non-coincident radial match along lines (for “stage”-like turbomachine conditions).

    '''

    import etc.transform as trf

    if method == 'globborder_dict':
        t = trf.defineBCStageFromBC(t, left)
        t = trf.defineBCStageFromBC(t, right)
        t, stage = trf.newStageChoroFromFamily(t, left, right)

    elif method == 'poswin':
        t = trf.defineBCStageFromBC(t, left)
        t = trf.defineBCStageFromBC(t, right)

        gbdu = computeGlobborderPoswin(t, left)
        # print("newStageMxPlFromFamily(up): gbdu = {}".format(gbdu))
        ups = []
        for bc in C.getFamilyBCs(t, left):
          bcpath = I.getPath(t, bc)
          bcu = trf.BCStageChoroUp(t, bc)
          globborder = bcu.glob_border(left, opposite=right)
          globborder.i_poswin = gbdu[bcpath]['i_poswin']
          globborder.j_poswin = gbdu[bcpath]['j_poswin']
          globborder.glob_dir_i = gbdu[bcpath]['glob_dir_i']
          globborder.glob_dir_j = gbdu[bcpath]['glob_dir_j']
          ups.append(bcu)

        # Downstream BCs declaration
        gbdd = computeGlobborderPoswin(t, right)
        # print("newStageMxPlFromFamily(down): gbdd = {}".format(gbdd))
        downs = []
        for bc in C.getFamilyBCs(t, right):
          bcpath = I.getPath(t, bc)
          bcd = trf.BCStageChoroDown(t, bc)
          globborder = bcd.glob_border(right, opposite=left)
          globborder.i_poswin = gbdd[bcpath]['i_poswin']
          globborder.j_poswin = gbdd[bcpath]['j_poswin']
          globborder.glob_dir_i = gbdd[bcpath]['glob_dir_i']
          globborder.glob_dir_j = gbdd[bcpath]['glob_dir_j']
          downs.append(bcd)

        # StageMxpl declaration
        stage = trf.BCStageChoro(t, up=ups, down=downs)
    else:
        raise Exception

    stage.jtype = jtype
    stage.stage_choro_type = stage_choro_type
    stage.harm_freq_comp = harm_freq_comp
    stage.choro_file_up = 'None'
    stage.file_up = None
    stage.choro_file_down = 'None'
    stage.file_down = None
    stage.nomatch_special = 'None'
    stage.format = 'CGNS'

    stage.create()

    setRotorStatorFamilyBC(t, left, right)

###TODO: DEVELOP THE FUNCTION FOR THE HYBRID CHORO


@J.mute_stdout
def setBC_outradeq(t, FamilyName, valve_type=0, valve_ref_pres=None,
    valve_ref_mflow=None, valve_relax=0.1, indpiv=1, 
    ReferenceValues=None, TurboConfiguration=None, method='globborder_dict'):
    '''
    Set an outflow boundary condition of type ``outradeq``.

    .. important : This function has a dependency to the ETC module.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        valve_type : int
            Valve law type. See `elsA documentation about valve laws <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/STB-97020/Textes/Boundary/Valve.html>`_.
            If 0, not valve law is used. In that case, **valve_ref_pres** corresponds
            to the prescribed static pressure at the pivot index, and **valve_ref_mflow**
            and **valve_relax** are not used.

        valve_ref_pres : :py:class:`float` or :py:obj:`None`
            Reference static pressure at the pivot index.
            If :py:obj:`None`, the value ``ReferenceValues['Pressure']`` is taken.

        valve_ref_mflow : :py:class:`float` or :py:obj:`None`
            Reference mass flow rate.
            If :py:obj:`None`, the value ``ReferenceValues['MassFlow']`` is taken
            and normalized using information in **TurboConfiguration** to get
            the corresponding mass flow rate on the section of **FamilyName**
            actually simulated.

        valve_relax : float
            'Relaxation' parameter of the valve law. The default value is 0.1.
            Be careful:

            * for laws 1, 2 and 5, it is a real Relaxation coefficient without
              dimension.

            * for law 3, it is a value homogeneous with a pressure divided
              by a mass flow.

            * for law 4, it is a value homogeneous with a pressure.
        
        indpiv : int
            Index of the cell where the pivot value is imposed.

        ReferenceValues : :py:class:`dict` or :py:obj:`None`
            as produced by :py:func:`computeReferenceValues`

        TurboConfiguration : :py:class:`dict` or :py:obj:`None`
            as produced by :py:func:`getTurboConfiguration`

        method : optional, str
            Method used to compute the globborder. The default value is
            ``'globborder_dict'``, it corresponds to the ETC topological
            algorithm.
            Another possible value is ``'poswin'`` to use the geometrical
            algorithm in *turbo* (in this case, *turbo* environment must be
            sourced).

    '''

    import etc.transform as trf

    if valve_ref_pres is None:
        try:
            valve_ref_pres = ReferenceValues['Pressure']
        except:
            MSG = 'valve_ref_pres or ReferenceValues must be not None'
            raise Exception(J.FAIL+MSG+J.ENDC)
    if valve_type != 0 and valve_ref_mflow is None:
        try:
            bc = C.getFamilyBCs(t, FamilyName)[0]
            zone = I.getParentFromType(t, bc, 'Zone_t')
            row = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
            rowParams = TurboConfiguration['Rows'][row]
            fluxcoeff = rowParams['NumberOfBlades'] / \
                float(rowParams['NumberOfBladesSimulated'])
            valve_ref_mflow = ReferenceValues['MassFlow'] / fluxcoeff
        except:
            MSG = 'Either valve_ref_mflow or both ReferenceValues and TurboConfiguration must be not None'
            raise Exception(J.FAIL+MSG+J.ENDC)

    # Delete previous BC if it exists
    for bc in C.getFamilyBCs(t, FamilyName):
        I._rmNodesByName(bc, '.Solver#BC')
    # Create Family BC
    family_node = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    I._rmNodesByName(family_node, '.Solver#BC')
    I.newFamilyBC(value='BCOutflowSubsonic', parent=family_node)

    # Outflow (globborder+outradeq, valve 4)
    if method == 'globborder_dict':
        from etc.globborder.globborder_dict import globborder_dict
        gbd = globborder_dict(t, FamilyName, config="axial")
    elif method == 'poswin':
        gbd = computeGlobborderPoswin(t, FamilyName)
    else:
        raise Exception
    for bcn in C.getFamilyBCs(t, FamilyName):
        bcpath = I.getPath(t, bcn)
        bc = trf.BCOutRadEq(t, bcn)
        bc.indpiv = indpiv
        bc.dirorder = -1
        # Valve laws:
        # <bc>.valve_law(valve_type, pref, Qref, valve_relax=relax, valve_file=None, valve_file_freq=1) # v4.2.01 pour valve_file*
        # valvelaws = [(1, 'SlopePsQ'),     # p(it+1) = p(it) + relax*( pref * (Q(it)/Qref) -p(it)) # relax = sans dim. # isoPs/Q
        #              (2, 'QTarget'),      # p(it+1) = p(it) + relax*pref * (Q(it)/Qref-1)         # relax = sans dim. # debit cible
        #              (3, 'QLinear'),      # p(it+1) = pref + relax*(Q(it)-Qref)                  # relax = Pascal    # lin en debit
        #              (4, 'QHyperbolic'),  # p(it+1) = pref + relax*(Q(it)/Qref)**2               # relax = Pascal    # comp. exp.
        #              (5, 'SlopePiQ')]     # p(it+1) = p(it) + relax*( pref * (Q(it)/Qref) -pi(it)) # relax = sans dim. # isoPi/Q
        # for law 5, pref = reference total pressure
        if valve_type == 0:
            bc.prespiv = valve_ref_pres
        else:
            valve_law_dict = {1: 'SlopePsQ', 2: 'QTarget',
                              3: 'QLinear', 4: 'QHyperbolic'}
            bc.valve_law(valve_law_dict[valve_type], valve_ref_pres,
                         valve_ref_mflow, valve_relax=valve_relax, valve_file=f'prespiv_{FamilyName}.log')
        globborder = bc.glob_border(current=FamilyName)
        globborder.i_poswin = gbd[bcpath]['i_poswin']
        globborder.j_poswin = gbd[bcpath]['j_poswin']
        globborder.glob_dir_i = gbd[bcpath]['glob_dir_i']
        globborder.glob_dir_j = gbd[bcpath]['glob_dir_j']
        globborder.azi_orientation = gbd[bcpath]['azi_orientation']
        globborder.h_orientation = gbd[bcpath]['h_orientation']
        bc.create()


@J.mute_stdout
def setBC_outradeqhyb(t, FamilyName, valve_type=0, valve_ref_pres=None,
                      valve_ref_mflow=None, valve_relax=0.1, indpiv=1, nbband=100, c=0.3, 
                      ReferenceValues=None, TurboConfiguration=None):
    '''
    Set an outflow boundary condition of type ``outradeqhyb``.

    .. important : This function has a dependency to the ETC module.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        valve_type : int
            Valve law type. See `elsA documentation about valve laws <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/STB-97020/Textes/Boundary/Valve.html>`_.
            Cannot be 0.

        valve_ref_pres : float
            Reference static pressure at the pivot index.

        valve_ref_mflow : float
            Reference mass flow rate.

        valve_relax : float
            'Relaxation' parameter of the valve law. The default value is 0.1.
            Be careful:

            * for laws 1, 2 and 5, it is a real Relaxation coefficient without
              dimension.

            * for law 3, it is a value homogeneous with a pressure divided
              by a mass flow.

            * for law 4, it is a value homogeneous with a pressure.
        
        indpiv : int
            Index of the cell where the pivot value is imposed.

        nbband : int
            Number of points in the radial distribution to compute.

        c : float
            Parameter for the distribution of radial points.
        
        ReferenceValues : :py:class:`dict` or :py:obj:`None`
            as produced by :py:func:`computeReferenceValues`

        TurboConfiguration : :py:class:`dict` or :py:obj:`None`
            as produced by :py:func:`getTurboConfiguration`


    '''

    import etc.transform as trf

    if valve_ref_pres is None:
        try:
            valve_ref_pres = ReferenceValues['Pressure']
        except:
            MSG = 'valve_ref_pres or ReferenceValues must be not None'
            raise Exception(J.FAIL+MSG+J.ENDC)
    if valve_type != 0 and valve_ref_mflow is None:
        try:
            bc = C.getFamilyBCs(t, FamilyName)[0]
            zone = I.getParentFromType(t, bc, 'Zone_t')
            row = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
            rowParams = TurboConfiguration['Rows'][row]
            fluxcoeff = rowParams['NumberOfBlades'] / \
                float(rowParams['NumberOfBladesSimulated'])
            valve_ref_mflow = ReferenceValues['MassFlow'] / fluxcoeff
        except:
            MSG = 'Either valve_ref_mflow or both ReferenceValues and TurboConfiguration must be not None'
            raise Exception(J.FAIL+MSG+J.ENDC)

    # Delete previous BC if it exists
    for bc in C.getFamilyBCs(t, FamilyName):
        I._rmNodesByName(bc, '.Solver#BC')
    # Create Family BC
    family_node = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    I._rmNodesByName(family_node, '.Solver#BC')
    I.newFamilyBC(value='BCOutflowSubsonic', parent=family_node)

    bc = trf.BCOutRadEqHyb(
        t, I.getNodeFromNameAndType(t, FamilyName, 'Family_t'))
    bc.glob_border()
    bc.indpiv = indpiv
    valve_law_dict = {1: 'SlopePsQ', 2: 'QTarget',
                      3: 'QLinear', 4: 'QHyperbolic'}
    bc.valve_law(valve_law_dict[valve_type], valve_ref_pres,
                 valve_ref_mflow, valve_relax=valve_relax, valve_file=f'prespiv_{FamilyName}.log')
    bc.dirorder = -1
    radius_filename = "state_radius_{}_{}.plt".format(FamilyName, nbband)
    radius = bc.repartition(filename=radius_filename, fileformat="bin_tp")
    radius.compute(t, nbband=nbband, c=c)
    radius.write()
    bc.create()


def updatePressurePivotForRestart(t, FamilyName):
    '''
    Based on previous log files named ``'LOGS/prespiv_<FamilyName>-*.log'``, update the pivot pressure 
    for an outradeq/outradeqhyb condition for a restart.

    Parameters
    ----------
    t : PyTree
        main tree

    FamilyName : str
        Name of the BC Family
    '''
    # HACK add prespiv_restart argument to bc.valve_law in etc
    # Find all prespiv log files for the current Family
    import glob
    prespiv_files = glob.glob(f'LOGS/prespiv_{FamilyName}-*.log')
    if len(prespiv_files) == 0: 
        return

    prespiv_file = sorted(prespiv_files, key=lambda s: int(s.replace(f'LOGS/prespiv_{FamilyName}-', '').replace('.log', '')))[-1]

    prespiv_data_tree = C.convertFile2PyTree(prespiv_file)
    zone = I.getZones(prespiv_data_tree)[0]
    iteration, pres_piv = J.getVars(zone,['iteration', 'pres_piv'])
    print(J.CYAN + f'Update prespiv_restart for Family {FamilyName} to {pres_piv[-1]}' + J.ENDC)
    # For outradeq --> in BC_t
    for bc in C.getFamilyBCs(t, FamilyName):
        solverBC = I.getNodeFromName(bc, '.Solver#BC')
        I._rmNodesByName(solverBC, 'prespiv_restart')
        I.newDataArray(name='prespiv_restart', value=pres_piv[-1], parent=solverBC)
    # For outradeqhyb --> in Family_t
    for bc in I.getNodesFromNameAndType(t, FamilyName, 'Family_t'):
        solverBC = I.getNodeFromName(bc, '.Solver#BC')
        if solverBC:
            I._rmNodesByName(solverBC, 'prespiv_restart')
            I.newDataArray(name='prespiv_restart', value=pres_piv[-1], parent=solverBC)


def setRotorStatorFamilyBC(t, left, right):
    for gc in I.getNodesFromType(t, 'GridConnectivity_t'):
        I._rmNodesByType(gc, 'FamilyBC_t')

    leftFamily = I.getNodeFromNameAndType(t, left, 'Family_t')
    rightFamily = I.getNodeFromNameAndType(t, right, 'Family_t')
    I.newFamilyBC(value='BCOutflow', parent=leftFamily)
    I.newFamilyBC(value='BCInflow', parent=rightFamily)


def computeGlobborderPoswin(tree, win):
    from turbo.poswin import computePosWin
    gbd = computePosWin(tree, win)
    for path, obj in gbd.items():
        gbd.pop(path)
        bc = I.getNodeFromPath(tree, path)
        gdi, gdj = getGlobDir(tree, bc)
        gbd['CGNSTree/'+path] = dict(glob_dir_i=gdi, glob_dir_j=gdj,
                                     i_poswin=obj.i, j_poswin=obj.j,
                                     azi_orientation=gdi, h_orientation=gdj)
    return gbd


def getGlobDir(tree, bc):
    # Remember: glob_dir_i is the opposite of theta, which is positive when it goes from Y to Z
    # Remember: glob_dir_j is as the radius, which is positive when it goes from hub to shroud

    # Check if the BC is in i, j or k constant: need pointrage of BC
    ptRi = I.getValue(I.getNodeFromName(bc, 'PointRange'))[0]
    ptRj = I.getValue(I.getNodeFromName(bc, 'PointRange'))[1]
    ptRk = I.getValue(I.getNodeFromName(bc, 'PointRange'))[2]
    x, y, z = J.getxyz(I.getParentFromType(tree, bc, 'Zone_t'))
    y0 = y[0, 0, 0]
    z0 = z[0, 0, 0]

    if ptRi[0] == ptRi[1]:
        dir1 = 2  # j
        dir2 = 3  # k
        y1 = y[0, -1, 0]
        z1 = z[0, -1, 0]
        y2 = y[0, 0, -1]
        z2 = y[0, 0, -1]

    elif ptRj[0] == ptRj[1]:
        dir1 = 1  # i
        dir2 = 3  # k
        y1 = y[-1, 0, 0]
        z1 = z[-1, 0, 0]
        y2 = y[0, 0, -1]
        z2 = y[0, 0, -1]

    elif ptRk[0] == ptRk[1]:
        dir1 = 1  # i
        dir2 = 2  # j
        y1 = y[-1, 0, 0]
        z1 = z[-1, 0, 0]
        y2 = y[0, -1, 0]
        z2 = y[0, -1, 0]

    rad0 = np.sqrt(y0**2+z0**2)
    rad1 = np.sqrt(y1**2+z1**2)
    rad2 = np.sqrt(y2**2+z2**2)
    tet0 = np.arctan2(z0, y0)
    tet1 = np.arctan2(z1, y1)
    tet2 = np.arctan2(z2, y2)

    ang1 = np.arctan2(rad1-rad0, rad1*tet1-rad0*tet0)
    ang2 = np.arctan2(rad2-rad0, rad2*tet2-rad0*tet0)

    if abs(np.sin(ang2)) < abs(np.sin(ang1)):
        # dir2 is more vertical than dir1
        # => globDirJ = +/- dir2
        if np.cos(ang1) > 0:
            # dir1 points towards theta>0
            globDirI = -dir1
        else:
            # dir1 points towards thetaw0
            globDirI = dir1
        if np.sin(ang2) > 0:
            # dir2 points towards r>0
            globDirJ = dir2
        else:
            # dir2 points towards r<0
            globDirJ = -dir2
    else:
        # dir1 is more vertical than dir2
        # => globDirJ = +/- dir1
        if np.cos(ang2) > 0:
            # dir2 points towards theta>0
            globDirI = -dir2
        else:
            # dir2 points towards thetaw0
            globDirI = dir2
        if np.sin(ang1) > 0:
            # dir1 points towards r>0
            globDirJ = dir1
        else:
            # dir1 points towards r<0
            globDirJ = -dir1

    print('  * glob_dir_i = %s\n  * glob_dir_j = %s' % (globDirI, globDirJ))
    assert globDirI != globDirJ
    return globDirI, globDirJ


################################################################################
#######  Boundary conditions without ETC dependency  ###########################
#######         WARNING: VALIDATION REQUIRED         ###########################
################################################################################

# def setBC_outradeqhyb(t, FamilyName, valve_type, valve_ref_pres,
#     valve_ref_mflow, valve_relax=0.1, nbband=100):
#     '''
#     Set an outflow boundary condition of type ``outradeqhyb``.
#
#     .. important:: The hybrid globborder conditions are availble since elsA v5.0.03.
#
#     .. note:: see `elsA Tutorial about valve laws <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/STB-97020/Textes/Boundary/Valve.html>`_
#
#     Parameters
#     ----------
#
#         t : PyTree
#             Tree to modify
#
#         FamilyName : str
#             Name of the family on which the boundary condition will be imposed
#
#         valve_type : int
#             Type of valve law
#
#         valve_ref_pres : float
#             Reference pressure for the valve boundary condition.
#
#         valve_ref_mflow : float
#             Reference massflow for the valve boundary condition.
#
#         valve_relax : float
#             Relaxation coefficient for the valve boundary condition.
#
#             .. warning:: This is a real relaxation coefficient for valve laws 1
#                 and 2, but it has the dimension of a pressure for laws 3, 4 and
#                 5
#
#         nbband : int
#             Number of points in the radial distribution to compute.
#
#     See also
#     --------
#
#         computeRadialDistribution
#
#     '''
#
#     # Delete previous BC if it exists
#     for bc in C.getFamilyBCs(t, FamilyName):
#         I._rmNodesByName(bc, '.Solver#BC')
#     # Create Family BC
#     family_node = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
#     I._rmNodesByName(family_node, '.Solver#BC')
#     I.newFamilyBC(value='BCOutflowSubsonic', parent=family_node)
#
#     radDistFile = 'radialDist_{}_{}.plt'.format(FamilyName, nbband)
#     radDist = computeRadialDistribution(t, FamilyName, nbband)
#     C.convertPyTree2File(radDist, radDistFile)
#
#     J.set(family_node, '.Solver#BC',
#             type            = 'outradeqhyb',
#             indpiv          = 1,
#             hray_tolerance  = 1e-12,
#             valve_type      = valve_type,
#             valve_ref_pres  = valve_ref_pres,
#             valve_ref_mflow = valve_ref_mflow,
#             valve_relax     = valve_relax,
#             glob_border_cur = FamilyName,
#             file            = radDistFile,
#             format          = 'bin_tp',
#         )
#
# def setBC_MxPl_hyb(t, left, right, nbband=100):
#     '''
#     Set a hybrid mixing plane condition between families **left** and **right**.
#
#     .. important:: The hybrid globborder conditions are availble since elsA v5.0.03.
#
#     Parameters
#     ----------
#
#         t : PyTree
#             tree to modify
#
#         left : str
#             Name of the first family corresponding to one side of the interface.
#
#         right : str
#             Name of the second family, see **left**
#
#         nbband : int
#             Number of points in the radial distribution to compute.
#
#     See also
#     --------
#
#         setBC_MxPl_hyb_OneSide, computeRadialDistribution
#
#     '''
#
#     setBC_MxPl_hyb_OneSide(t, left, right, nbband)
#     setBC_MxPl_hyb_OneSide(t, right, left, nbband)
#     for gc in I.getNodesFromType(t, 'GridConnectivity_t'):
#         I._rmNodesByType(gc, 'FamilyBC_t')
#
# def setBC_MxPl_hyb_OneSide(t, FamCur, FamOpp, nbband):
#     '''
#     Set a hybrid mixing plane condition for the family **FamCur**.
#
#     .. important:: This function is intended to be called twice by
#         :py:func:`setBC_MxPl_hyb`, once for **FamCur** (with the opposite family
#         **FamOpp**) and once for **FamOpp** (with the opposite family **FamCur**)
#
#     Parameters
#     ----------
#
#         t : PyTree
#             tree to modify
#
#         FamCur : str
#             Name of the first family corresponding to one side of the interface.
#
#         FamOpp : str
#             Name of the second family, on the opposite side of **FamCur**.
#
#         nbband : int
#             Number of points in the radial distribution to compute.
#
#     See also
#     --------
#     setBC_MxPl_hyb, computeRadialDistribution
#     '''
#
#     for bc in C.getFamilyBCs(t, FamCur):
#         bcName = I.getName(bc)
#         PointRange = I.getNodeFromType(bc, 'IndexRange_t')
#         zone = I.getParentFromType(t, bc, 'Zone_t')
#         I.rmNode(t, bc)
#         zgc = I.getNodeFromType(zone, 'ZoneGridConnectivity_t')
#         gc = I.newGridConnectivity(name=bcName, donorName=I.getName(zone),
#                                     ctype='Abutting', parent=zgc)
#         I._addChild(gc, PointRange)
#         I.createChild(gc, 'FamilyName', 'FamilyName_t', value=FamCur)
#
#     radDistFileFamCur = 'radialDist_{}_{}.plt'.format(FamCur, nbband)
#     radDistFamCur = computeRadialDistribution(t, FamCur, nbband)
#     C.convertPyTree2File(radDistFamCur, radDistFileFamCur)
#
#     for gc in I.getNodesFromType(t, 'GridConnectivity_t'):
#         fam = I.getValue(I.getNodeFromType(gc, 'FamilyName_t'))
#         if fam == FamCur:
#             J.set(gc, '.Solver#Property',
#                     globborder      = FamCur,
#                     globborderdonor = FamOpp,
#                     file            = radDistFileFamCur,
#                     type            = 'stage_mxpl_hyb',
#                     mxpl_dirtype    = 'axial',
#                     mxpl_avermean   = 'riemann',
#                     mxpl_avertur    = 'conservative',
#                     mxpl_num        = 'characteristic',
#                     mxpl_ari_sensor = 0.5,
#                     hray_tolerance  = 1e-12,
#                     jtype           = 'nomatch_rad_line',
#                     nomatch_special = 'none',
#                     format          = 'bin_tp'
#                 )
#
# def computeRadialDistribution(t, FamilyName, nbband):
#     '''
#     Compute a distribution of radius values according the density of cells for
#     the BCs of family **FamilyName**.
#
#     Parameters
#     ----------
#
#         t : PyTree
#             mesh tree with families
#
#         FamilyName : str
#             Name of the BC family to extract to compute the radial repartition.
#
#         nbband : int
#             Number of values in the returned repartition. It is used to decimate
#             the list of the radii at the center of each cell. For a structured
#             grid, should be ideally the number of cells in the radial direction.
#
#     Returns
#     -------
#
#         zone : PyTree
#             simple tree containing only a one dimension array called 'radius'
#
#     '''
#     bcNodes = C.extractBCOfName(t, 'FamilySpecified:{0}'.format(FamilyName))
#     # Compute radius and put this value at cell centers
#     C._initVars(bcNodes, '{radius}=({CoordinateY}**2+{CoordinateZ}**2)**0.5')
#     bcNodes = C.node2Center(bcNodes, 'radius')
#     I._rmNodesByName(bcNodes, I.__FlowSolutionNodes__)
#     # Put all the radii values in a list
#     radius = []
#     for bc in bcNodes:
#         radius += list(I.getValue(I.getNodeFromName(bc, 'radius')).flatten())
#     # Sort and transform to numpy array
#     step = (len(radius)-1) / float(nbband-1)
#     ind = [int(np.ceil(step*n)) for n in range(nbband)]
#     radius = np.array(sorted(radius))[ind]
#     assert radius.size == nbband
#     # Convert to PyTree
#     zone = I.newZone('Zone1', [[len(radius)],[1],[1]], 'Structured')
#     FS = I.newFlowSolution(parent=zone)
#     I.newDataArray('radius', value=radius, parent=FS)
#
#     return zone

################################################################################
#############  Multiple jobs submission  #######################################
################################################################################

def getRangesOfIsospeedLines(config):
    ThrottleRange = sorted(list(set([float(case['CASE_LABEL'].split('_')[0]) for case in config.JobsQueues])))
    RotationSpeedRange = sorted(list(set(case['TurboConfiguration']['ShaftRotationSpeed'] for case in config.JobsQueues)))
    return ThrottleRange, RotationSpeedRange

def launchIsoSpeedLines(machine, DIRECTORY_WORK,
                    ThrottleRange, RotationSpeedRange=None, **kwargs):
    '''
    User-level function designed to launch iso-speed lines.

    Parameters
    ----------

        machine : str
            name of the machine ``'sator'``, ``'spiro'``, ``'eos'``...

            .. warning:: only ``'sator'`` has been tested

        DIRECTORY_WORK : str
            the working directory at computation server.

            .. note:: indicated directory may not exist. In this case, it will be created.

        ThrottleRange : list
            Throttle values to consider (depend on the valve law)

        RotationSpeedRange : list, optional
            RotationSpeed values to consider. If not given, then the value in **TurboConfiguration** is taken.

        kwargs : dict
            same arguments than prepareMainCGNS4ElsA, except that 'mesh' may be
            a :py:class:`list`. In that case, the list must have the same length
            that **RotationSpeedRange**, and each element corresponds to the
            mesh used for each rotation speed. It may be useful to make the
            geometry vary with the rotation speed (updating shape, tip gaps, etc).

    Returns
    -------

        None : None
            File ``JobsConfiguration.py`` is writen and polar builder job is
            launched
    '''
    IsJobInformationGiven = 'JobInformation' in kwargs and \
        all([key in kwargs['JobInformation'] for key in ['JobName', 'AER', 'NumberOfProcessors']])
    assert IsJobInformationGiven, 'JobInformation is required with not default values for JobName, AER and NumberOfProcessors'

    if not RotationSpeedRange:
        RotationSpeedRange = [kwargs['TurboConfiguration']['ShaftRotationSpeed']]

    try:
        prev_config = JM.getJobsConfiguration('./',useLocalConfig=True)
        prev_ThrottleRange, prev_RotationSpeedRange = getRangesOfIsospeedLines(prev_config) #WARNING : la presence du fichier config ne garantit pas que la config a ete lancee... Il vaudrait peut etre mieux checker si des fichiers sont presents sur le cluster...
        ThrottleRange = list(ThrottleRange) + prev_ThrottleRange
        RotationSpeedRange = list(RotationSpeedRange) + prev_RotationSpeedRange 
        EXTEND = True
        MSG = 'Extending previous iso-speed line.'
        print(J.WARN + MSG + J.ENDC)  
    except:
        MSG = 'Building new iso-speed line from scratch.'
        print(J.CYAN + MSG + J.ENDC) 
        EXTEND = False

    ThrottleRange = list(np.unique(ThrottleRange))
    RotationSpeedRange = list(np.unique(RotationSpeedRange))

    ThrottleRange = sorted(ThrottleRange)
    # Sort Rotation speeds (and mesh files, if a list is given) 
    index2sortRSpeed = sorted(range(len(RotationSpeedRange)), key=lambda i: abs(RotationSpeedRange[i]))
    RotationSpeedRange = [RotationSpeedRange[i] for i in index2sortRSpeed]
    if isinstance(kwargs['mesh'], list):
        kwargs['mesh'] = [kwargs['mesh'][i] for i in index2sortRSpeed]

    ThrottleMatrix, RotationSpeedMatrix  = np.meshgrid(ThrottleRange, RotationSpeedRange)

    Throttle_       = ThrottleMatrix.ravel(order='K')
    RotationSpeed_  = RotationSpeedMatrix.ravel(order='K')
    NewJobs         = Throttle_ == ThrottleRange[0]

    def adaptPathForDispatcher(filename):
        # This is a path: remove it for writing in JobConfiguration.py
        return os.path.join('..', '..', 'DISPATCHER', filename.split(os.path.sep)[-1])

    JobsQueues = []
    for i, (Throttle, RotationSpeed, NewJob) in enumerate(zip(Throttle_, RotationSpeed_, NewJobs)):

        print('Assembling run {} Throttle={} RotationSpeed={} | NewJob = {}'.format(
                i, Throttle, RotationSpeed, NewJob))

        WorkflowParams = copy.deepcopy(kwargs)

        if NewJob:
            JobName = WorkflowParams['JobInformation']['JobName']+'%d'%i
            WorkflowParams['writeOutputFields'] = True
        else:
            WorkflowParams['writeOutputFields'] = False

        CASE_LABEL = '{:08.2f}_{}'.format(abs(Throttle), JobName)
        if Throttle < 0: CASE_LABEL = 'M'+CASE_LABEL

        WorkflowParams['TurboConfiguration']['ShaftRotationSpeed'] = RotationSpeed

        for BC in WorkflowParams['BoundaryConditions']:
            if any([condition in BC['type'] for condition in ['outradeq', 'OutflowRadialEquilibrium']]):
                if BC['valve_type'] == 0:
                    if 'prespiv' in BC: BC['prespiv'] = Throttle
                    elif 'valve_ref_pres' in BC: BC['valve_ref_pres'] = Throttle
                    else: raise Exception('valve_ref_pres must be given explicitely')
                elif BC['valve_type'] in [1, 5]:
                    if 'valve_ref_pres' in BC: BC['valve_ref_pres'] = Throttle
                    else: raise Exception('valve_ref_pres must be given explicitely')
                elif BC['valve_type'] == 2:
                    if 'valve_ref_mflow' in BC: BC['valve_ref_mflow'] = Throttle
                    else: raise Exception('valve_ref_mflow must be given explicitely')
                elif BC['valve_type'] in [3, 4]:
                    if 'valve_relax' in BC: BC['valve_relax'] = Throttle
                    else: raise Exception('valve_relax must be given explicitely')
                else:
                    raise Exception('valve_type={} not taken into account yet'.format(BC['valve_type']))

            if 'filename' in BC:
                BC['filename'] = adaptPathForDispatcher(BC['filename'])

        if 'Initialization' in WorkflowParams:
            if i != 0:
                WorkflowParams.pop('Initialization')
            elif 'file' in WorkflowParams['Initialization']:
                WorkflowParams['Initialization']['file'] = adaptPathForDispatcher(WorkflowParams['Initialization']['file'])

        if isinstance(WorkflowParams['mesh'], list):
            speedIndex = RotationSpeedRange.index(RotationSpeed)
            WorkflowParams['mesh'] = WorkflowParams['mesh'][speedIndex]
        
        WorkflowParams['mesh'] = adaptPathForDispatcher(WorkflowParams['mesh'])

        JobsQueues.append(
            dict(ID=i, CASE_LABEL=CASE_LABEL, NewJob=NewJob, JobName=JobName, **WorkflowParams)
            )

        # adapt custom templates
        if 'templates' in WorkflowParams:
            for key, filename in WorkflowParams['templates'].items():
                if isinstance(filename, str):
                    WorkflowParams['templates'][key] = adaptPathForDispatcher(filename)
                elif isinstance(filename, list):
                    WorkflowParams['templates'][key] = [adaptPathForDispatcher(f) for f in filename]

    JM.saveJobsConfiguration(JobsQueues, machine, DIRECTORY_WORK)

    def findElementsInCollection(collec, searchKey, elements=[]):
        '''
        In the nested collection **collec** (may be a dictionary or a list),
        find all the values corresponding to the key **searchKey**.

        Parameters
        ----------

            collec : :py:class:`dict` or :py:class:`list`
                Nested dictionary or list where **searchKey** is searched

            searchKey : str
                Key to find in **collec**

            elements : list
                accumulated list of found values. Works as an accumulator in the
                recursive function

        Returns
        -------

            elements : list
                list of the found values correspondingto **searchKey**
        '''
        if isinstance(collec, dict):
            for key, value in collec.items():
                if isinstance(value, (dict, list)):
                    findElementsInCollection(value, searchKey, elements=elements)
                elif key == searchKey:
                    elements.append(value)
        elif isinstance(collec, list):
            for elem in collec:
                findElementsInCollection(elem, searchKey, elements=elements)
        return elements

    otherFiles = findElementsInCollection(kwargs, 'file') + findElementsInCollection(kwargs, 'filename')
    if isinstance(kwargs['mesh'], list):
        otherFiles.extend(kwargs['mesh'])
    else:
        otherFiles.append(kwargs['mesh'])
    if 'Initialization' in kwargs:
        if 'method' in kwargs['Initialization']:
            if 'turbo' in kwargs['Initialization']['method']:
                otherFiles.append('mask.cgns')
    # get custom templates
    if 'templates' in kwargs:
        for key, filename in kwargs['templates'].items():
            if isinstance(filename, str):
                otherFiles.append(filename)
            elif isinstance(filename, list):
                otherFiles.extend(filename)
            else: 
                raise TypeError(f'Values in the dict templates must be either strings or list of strings. Current value of template is: {kwargs["templates"]}')

    JM.launchJobsConfiguration(templatesFolder=MOLA.__MOLA_PATH__+'/TEMPLATES/WORKFLOW_COMPRESSOR', otherFiles=otherFiles, ExtendPreviousConfig=EXTEND)

def printConfigurationStatus(DIRECTORY_WORK, useLocalConfig=False):
    '''
    Print the current status of a IsoSpeedLines computation.

    Parameters
    ----------

        DIRECTORY_WORK : str
            directory where ``JobsConfiguration.py`` file is located

        useLocalConfig : bool
            if :py:obj:`True`, use the local ``JobsConfiguration.py``
            file instead of retreiving it from **DIRECTORY_WORK**

    '''
    config = JM.getJobsConfiguration(DIRECTORY_WORK, useLocalConfig)
    Throttle = np.array(sorted(list(set([float(case['CASE_LABEL'].split('_')[0]) for case in config.JobsQueues]))))
    RotationSpeed = np.array(sorted(list(set([case['TurboConfiguration']['ShaftRotationSpeed'] for case in config.JobsQueues]))))

    nThrottle = Throttle.size
    nRotationSpeed = RotationSpeed.size
    NcolMax = 79
    FirstCol = 15
    Ndigs = int((NcolMax-FirstCol)/nRotationSpeed)
    ColFmt = r'{:^'+str(Ndigs)+'g}'
    ColStrFmt = r'{:^'+str(Ndigs)+'s}'
    TagStrFmt = r'{:>'+str(FirstCol)+'s}'
    TagFmt = r'{:>'+str(FirstCol-2)+'g} |'

    def getCaseLabel(config, throttle, rotSpeed):
        for case in config.JobsQueues:
            if np.isclose(float(case['CASE_LABEL'].split('_')[0]), throttle) and \
                np.isclose(case['TurboConfiguration']['ShaftRotationSpeed'], rotSpeed):

                return case['CASE_LABEL']

    JobNames = [getCaseLabel(config, Throttle[0], m).split('_')[-1] for m in RotationSpeed]
    print('')
    print(TagStrFmt.format('JobName |')+''.join([ColStrFmt.format(j) for j in JobNames]))
    print(TagStrFmt.format('RotationSpeed |')+''.join([ColFmt.format(r) for r in RotationSpeed]))
    print(TagStrFmt.format('Throttle |')+''.join(['_' for m in range(NcolMax-FirstCol)]))

    for throttle in Throttle:
        Line = TagFmt.format(throttle)
        for rotSpeed in RotationSpeed:
            CASE_LABEL = getCaseLabel(config, throttle, rotSpeed)
            status = JM.statusOfCase(config, CASE_LABEL)
            if status == 'COMPLETED':
                msg = J.GREEN+ColStrFmt.format('OK')+J.ENDC
            elif status == 'FAILED':
                msg = J.FAIL+ColStrFmt.format('KO')+J.ENDC
            elif status == 'TIMEOUT':
                msg = J.WARN+ColStrFmt.format('TO')+J.ENDC
            elif status == 'RUNNING':
                msg = ColStrFmt.format('GO')
            else:
                msg = ColStrFmt.format('PD') # Pending
            Line += msg
        print(Line)

def printConfigurationStatusWithPerfo(monitoredRow, DIRECTORY_WORK='.'):
    '''
    Print the current status of a IsoSpeedLines computation and display
    performance of the monitored row for completed jobs.

    Parameters
    ----------

        monitoredRow : str
            Name of the row whose performance will be displayed
        
        DIRECTORY_WORK : str
            directory where ``JobsConfiguration.py`` file is located

    Returns
    -------

        perfo : list
            list with performance of **monitoredRow** for completed
            simulations. Each element is a dict corresponding to one rotation speed.
            This dict contains the following keys:

            * RotationSpeed (float)

            * Throttle (numpy.array)

            * MassFlow (numpy.array)

            * PressureStagnationRatio (numpy.array)

            * EfficiencyIsentropic (numpy.array)

    '''
    print(J.CYAN+'Repatriating data...'+J.ENDC)
    os.system("mola_repatriate --arrays")
    print(J.GREEN+'  Repatriating data done.'+J.ENDC)

    config = J.load_source('config', os.path.join(DIRECTORY_WORK, 'JobsConfiguration.py'))
    Throttle = np.array(sorted(list(set([float(case['CASE_LABEL'].split('_')[0]) for case in config.JobsQueues]))))
    RotationSpeed = np.array(sorted(list(set([case['TurboConfiguration']['ShaftRotationSpeed'] for case in config.JobsQueues]))))
    root_path = os.path.join(DIRECTORY_WORK, config.DIRECTORY_WORK.split('/')[-2])

    nThrottle = Throttle.size
    nCol = 4
    NcolMax = 79
    FirstCol = 15
    Ndigs = int((NcolMax-FirstCol)/nCol)
    ColFmt = r'{:^'+str(Ndigs)+'g}'
    ColStrFmt = r'{:^'+str(Ndigs)+'s}'
    TagStrFmt = r'{:>'+str(FirstCol)+'s}'
    TagFmt = r'{:>'+str(FirstCol-2)+'g} |'

    def getCaseLabel(config, throttle, rotSpeed):
        for case in config.JobsQueues:
            if np.isclose(float(case['CASE_LABEL'].split('_')[0]), throttle) and \
                np.isclose(case['TurboConfiguration']['ShaftRotationSpeed'], rotSpeed):

                return case['CASE_LABEL']
    
    def getStatus(path):
        Output = os.listdir(path)
        if 'COMPLETED' in Output:
            return 'COMPLETED'
        if 'FAILED' in Output:
            return 'FAILED'
        for o in Output:
            if o.startswith('core') or o.startswith('elsA.x'):
                return 'TIMEOUT'
        if 'coprocess.log' in Output:
            return 'RUNNING'
        return 'PENDING'

    perfo = []
    lines = ['']

    JobNames = [getCaseLabel(config, Throttle[0], r).split('_')[-1] for r in RotationSpeed]
    for idSpeed, rotationSpeed in enumerate(RotationSpeed):

        lines.append(TagStrFmt.format('JobName |')+''.join([ColStrFmt.format(JobNames[idSpeed])] + [ColStrFmt.format('') for j in range(nCol-1)]))
        lines.append(TagStrFmt.format('RotationSpeed |')+''.join([ColFmt.format(rotationSpeed)] + [ColStrFmt.format('') for j in range(nCol-1)]))
        lines.append(TagStrFmt.format(' |')+''.join([ColStrFmt.format(''), ColStrFmt.format('MFR'), ColStrFmt.format('RPI'), ColStrFmt.format('ETA')]))
        lines.append(TagStrFmt.format('Throttle |')+''.join(['_' for m in range(NcolMax-FirstCol)]))
        
        perfoOverIsospeedLine = dict(
            RotationSpeed = rotationSpeed,
            Throttle = [],
            MassFlow = [],
            PressureStagnationRatio = [],
            EfficiencyIsentropic = []
        )

        for throttle in Throttle:
            Line = TagFmt.format(throttle)
            CASE_LABEL = getCaseLabel(config, throttle, rotationSpeed)
            interm_dir = '_'.join(CASE_LABEL.split('_')[1:])
            case_path = f'{root_path}/{interm_dir}/{CASE_LABEL}'
            status = getStatus(case_path)
            if status == 'COMPLETED':
                msg = J.GREEN+ColStrFmt.format('OK')+J.ENDC
            elif status == 'FAILED':
                msg = J.FAIL+ColStrFmt.format('KO')+J.ENDC
            elif status == 'TIMEOUT':
                msg = J.WARN+ColStrFmt.format('TO')+J.ENDC
            elif status == 'RUNNING':
                msg = ColStrFmt.format('GO')
            else:
                msg = ColStrFmt.format('PD') # Pending

            if os.path.isfile(f'{case_path}/COMPLETED'):
                ArraysTree = C.convertFile2PyTree(f'{case_path}/OUTPUT/arrays.cgns')
                ArraysZone = I.getNodeFromName2(ArraysTree, 'PERFOS_{}'.format(monitoredRow))
                lastarrays = J.getVars2Dict(ArraysZone,C.getVarNames(ArraysZone,excludeXYZ=True)[0])
                for v in lastarrays: lastarrays[v] = lastarrays[v][-1]

                perfoOverIsospeedLine['Throttle'].append(throttle)
                perfoOverIsospeedLine['MassFlow'].append(lastarrays['MassFlowIn'])
                perfoOverIsospeedLine['PressureStagnationRatio'].append(lastarrays['PressureStagnationRatio'])
                perfoOverIsospeedLine['EfficiencyIsentropic'].append(lastarrays['EfficiencyIsentropic'])
    
                msg += ''.join([ColFmt.format(lastarrays['MassFlowIn']), 
                                ColFmt.format(lastarrays['PressureStagnationRatio']), 
                                ColFmt.format(lastarrays['EfficiencyIsentropic'])
                                ])
            else:
                msg += ''.join([ColStrFmt.format('') for n in range(nCol-1)])
            Line += msg
            lines.append(Line)

        lines.append('')
        for key, value in perfoOverIsospeedLine.items():
            perfoOverIsospeedLine[key] = np.array(value)
        perfo.append(perfoOverIsospeedLine)

    for line in lines: print(line)

    return perfo

def getPostprocessQuantities(basename, DIRECTORY_WORK='.', rename=True):
    '''
    Print the current status of a IsoSpeedLines computation and display
    performance of the monitored row for completed jobs.

    Parameters
    ----------
        basename : str
            Name of the base to get

        DIRECTORY_WORK : str
            directory where ``JobsConfiguration.py`` file is located

        rename : bool 
            if :py:obj:`True`, rename variables with CGNS names (or inspired CGNS names, already used in MOLA)

    Returns
    -------

        perfo : list
            list with data contained in the base **baseName** for completed
            simulations. Each element is a dict corresponding to one rotation speed.
            This dict contains the following keys:

            * RotationSpeed (float)

            * Throttle (numpy.array)

            * and all quantities found in **baseName** (numpy.array)

    '''
    current_directory = os.getcwd()
    os.chdir(DIRECTORY_WORK)
    print(J.CYAN+'Repatriating data...'+J.ENDC)
    os.system("mola_repatriate --light")
    print(J.GREEN+'  Repatriating data done.'+J.ENDC)
    os.chdir(current_directory)

    config = J.load_source('config', os.path.join(DIRECTORY_WORK, 'JobsConfiguration.py'))
    root_path = os.path.join(DIRECTORY_WORK, config.DIRECTORY_WORK.split('/')[-2])

    perfo = getPostprocessQuantitiesLocal(basename, config.JobsQueues, root_path, rename=rename)

    return perfo

def getPostprocessQuantitiesLocal(basename, configJobsQueues, root_path, rename=True):
    '''
    Print the current status of a IsoSpeedLines computation and display
    performance of the monitored row for completed jobs.

    Parameters
    ----------
        basename : str
            Name of the base to get
        
        configJobsQueues : list

        root_path : str

        rename : bool 
            if :py:obj:`True`, rename variables with CGNS names (or inspired CGNS names, already used in MOLA)

    Returns
    -------

        perfo : list
            list with data contained in the base **baseName** for completed
            simulations. Each element is a dict corresponding to one rotation speed.
            This dict contains the following keys:

            * RotationSpeed (float)

            * Throttle (numpy.array)

            * and all quantities found in **baseName** (numpy.array)

    '''
    Throttle = np.array(sorted(list(set([float(case['CASE_LABEL'].split('_')[0]) for case in configJobsQueues]))))
    RotationSpeed = np.array(sorted(list(set([case['TurboConfiguration']['ShaftRotationSpeed'] for case in configJobsQueues]))))

    def getCaseLabel(configJobsQueues, throttle, rotSpeed):
        for case in configJobsQueues:
            if np.isclose(float(case['CASE_LABEL'].split('_')[0]), throttle) and \
                np.isclose(case['TurboConfiguration']['ShaftRotationSpeed'], rotSpeed):

                return case['CASE_LABEL']

    def getStatus(path):
        Output = os.listdir(path)
        if 'COMPLETED' in Output:
            return 'COMPLETED'
        if 'FAILED' in Output:
            return 'FAILED'
        for o in Output:
            if o.startswith('core') or o.startswith('elsA.x'):
                return 'TIMEOUT'
        if 'coprocess.log' in Output:
            return 'RUNNING'
        return 'PENDING'
    
    perfo = []
    for rotationSpeed in RotationSpeed:
        perfoOverIsospeedLine = dict(RotationSpeed=rotationSpeed, Throttle=[])

        for idThrottle, throttle in enumerate(Throttle):
            CASE_LABEL = getCaseLabel(configJobsQueues, throttle, rotationSpeed)
            interm_dir = '_'.join(CASE_LABEL.split('_')[1:])
            case_path = f'{root_path}/{interm_dir}/{CASE_LABEL}'
            status = getStatus(case_path)

            if status == 'COMPLETED':
                perfoOverIsospeedLine['Throttle'].append(throttle)

                ArraysTree = C.convertFile2PyTree(f'{case_path}/OUTPUT/arrays.cgns')
                ArraysZone = I.getNodeFromName2(ArraysTree, basename)
                lastarrays = J.getVars2Dict(ArraysZone,C.getVarNames(ArraysZone,excludeXYZ=True)[0])
                for v in lastarrays: lastarrays[v] = lastarrays[v][-1]
                if not 'Massflow' in lastarrays:
                    try:
                        ArraysZone = I.getNodeFromName2(ArraysTree, '#'.join(basename.split('#')[1:]))
                        lastarrays['Massflow'] = J.getVars(ArraysZone, ['Massflow'])[0][-1]
                    except:
                        pass

                for key, value in lastarrays.items():
                    if idThrottle == 0:
                        perfoOverIsospeedLine[key] = [value]
                    else:
                        perfoOverIsospeedLine[key].append(value)

        for key, value in perfoOverIsospeedLine.items():
            perfoOverIsospeedLine[key] = np.array(value)
        perfo.append(perfoOverIsospeedLine)

    if rename:
        VarsToRename = [
            ('Massflow', 'MassFlow'), 
            ('StagnationPressureRatio', 'PressureStagnationRatio'), 
            ('IsentropicEfficiency', 'EfficiencyIsentropic')
            ]
        for (oldName, newName) in VarsToRename:
            for perfoOverIsospeedLine in perfo: 
                if oldName in perfoOverIsospeedLine:
                    perfoOverIsospeedLine[newName] = perfoOverIsospeedLine[oldName]

    return perfo


def convertPeriodic2Chorochrono(t):
    '''
    Convert the periodic boundary condition from a PyTree t to a chorochrono boundary condition.
    
    Parameters
    ----------

        t : PyTree
            Tree to modify

    '''
    import etc.transform as trf
    gcnodes = []
    for zone_node in I.getZones(t):
        zgc_node = I.getNodeFromType1(zone_node,"ZoneGridConnectivity_t")
        if zgc_node:
            for gc_node in I.getNodesFromType1(zgc_node,"GridConnectivity_t")+I.getNodesFromType1(zgc_node,"GridConnectivity1to1_t"):
                gcp_node = I.getNodeFromType1(gc_node,"GridConnectivityProperty_t")
                if gcp_node:
                    periodic_node = I.getNodeFromType1(gcp_node,"Periodic_t")
                    if periodic_node:
                        gcnodes.append(gc_node)

    for gcnode in gcnodes:
        gc = trf.BCChoroChrono(t, gcnode, choro_file = 'None')
        gc.choro_file   = 'None'
        gc.file   = None
        gc.format = 'CGNS'
        gc.create()

def updateChoroTimestep(t, Rows, NumericalParams):
    '''
    Compute the timestep for chorochronic simulations if not provided.
    
    Parameters
    ----------

        t : PyTree
            Tree to modify

        Rows : :py:class:`dict`
            Dictionary of Rows as provided in TurboConfiguration for the prepareMainCGNS function.

        NumericalParams : :py:class:`dict`
            dictionary containing the numerical settings for elsA. Similar to that required in prepareMainCGNS function.

    '''   

    rowNameList = list(Rows.keys())
    
    Nblade_Row1 = Rows[rowNameList[0]]['NumberOfBlades']
    Nblade_Row2 = Rows[rowNameList[1]]['NumberOfBlades']
    omega_Row1 = Rows[rowNameList[0]]['RotationSpeed']
    omega_Row2 = Rows[rowNameList[1]]['RotationSpeed']

    per_Row1 = (2*np.pi)/(Nblade_Row2*np.abs(omega_Row1-omega_Row2))
    per_Row2 = (2*np.pi)/(Nblade_Row1*np.abs(omega_Row1-omega_Row2))

    gcd =np.gcd(Nblade_Row1,Nblade_Row2)
    
    DeltaT = gcd*2*np.pi/(np.abs(omega_Row1-omega_Row2)*Nblade_Row1*Nblade_Row2) #Largest time step that is a fraction of the period of both Row1 and Row2.
    MSG = 'DeltaT : %s'%(DeltaT)
    print(J.WARN + MSG + J.ENDC)
    
    if 'timestep' not in NumericalParams.keys():
        MSG = 'Time-step not provided by the user. Computing a suitable time-step based on stage properties.'
        print(J.WARN + MSG + J.ENDC)
        Nquo = 10
        time_step = DeltaT/Nquo
    
        NewNquo = Nquo
        while time_step> 5*10**-7:
            NewNquo = NewNquo+10
            time_step = DeltaT/NewNquo

    
        NumericalParams['timestep'] = time_step

    else:
        MSG = 'Time-step provided by the user.'
        print(J.WARN + MSG + J.ENDC)
        NewNquo = DeltaT/NumericalParams['timestep']
        Nquo_round = np.round(NewNquo)
        if np.absolute(NewNquo-Nquo_round)>1e-08:
            MSG = 'Choice of time-step does no seem to be suited for the case. Check the following parameters:'
            print(J.WARN + MSG + J.ENDC)


    MSG = 'Nquo : %s'%(NewNquo)
    print(J.WARN + MSG + J.ENDC)    
    
    MSG = 'Time step : %s'%(NumericalParams['timestep'])
    print(J.WARN + MSG + J.ENDC)

    MSG = 'Number of time step per period for row 1 : %s'%(per_Row1/NumericalParams['timestep'])
    print(J.WARN + MSG + J.ENDC)

    MSG = 'Number of time step per period for row 2 : %s'%(per_Row2/NumericalParams['timestep'])
    print(J.WARN + MSG + J.ENDC)   

def computeChoroParameters(t, Rows, Nharm_Row1, Nharm_Row2):
    '''
    Compute the parameters to run a chorochronic computation.
    
    Parameters
    ----------

        t : PyTree
            Tree to modify

        Rows : :py:class:`dict`
            Dictionary of Rows as provided in TurboConfiguration for the prepareMainCGNS function.

        Nharm_Row1 : float
            Number of harmonics of the first row.

        Nharm_Row2 : float
            Number of harmonics of the second row.

    Returns
    -------

        choroParamsStage : :py:class:`dict`
           Dictionary containing chorochronic parameters for each row. The dictionary keys correspond to the row names referenced in the TurboConfiguration dictionary.

    '''       
    rowNameList = list(Rows.keys())

    Nblade_Row1 = Rows[rowNameList[0]]['NumberOfBlades']
    Nblade_Row2 = Rows[rowNameList[1]]['NumberOfBlades']
    omega_Row1 = Rows[rowNameList[0]]['RotationSpeed']
    omega_Row2 = Rows[rowNameList[1]]['RotationSpeed']

    freq_Row1 = Nblade_Row2*np.abs(omega_Row1-omega_Row2)/(2*np.pi)
    freq_Row2 = Nblade_Row1*np.abs(omega_Row1-omega_Row2)/(2*np.pi)
    omgRel_Row2  = omega_Row1 - omega_Row2
    omgRel_Row1  = omega_Row2 - omega_Row1

    gcd =np.gcd(Nblade_Row1,Nblade_Row2)
    if Nharm_Row1 < Nblade_Row1/gcd:
        MSG = 'The number of chorochronic harmonics for the first row is too low (%s). Recomputing...\n '%(Nharm_Row1)
        print(J.WARN + MSG + J.ENDC)     
        Nharm_Row1 = float(Nblade_Row2)


    if Nharm_Row2 < Nblade_Row2/gcd:
        MSG = 'The number of chorochronic harmonics for the second row is too low (%s). Recomputing...\n '%(Nharm_Row2)
        print(J.WARN + MSG + J.ENDC)
        Nharm_Row2 = float(Nblade_Row1)
        MSG = 'New number of harmonics for row 2 : %s'%(Nharm_Row2)
        print(J.WARN + MSG + J.ENDC)

    relax  = 1.0 

    # PeriodRow1 = 2*np.pi/(Nblade_Row1*np.abs(omega_Row1  - omega_Row2))
    # PeriodRow2 = 2*np.pi/(Nblade_Row2*np.abs(omega_Row1  - omega_Row2))

    choroParamsRow1 = dict( freq = freq_Row1, omega = omgRel_Row1, harm = Nharm_Row1, relax = relax, axis_ang_1 = Nblade_Row1, axis_ang_2 = 1)
    choroParamsRow2 = dict( freq = freq_Row2, omega = omgRel_Row2, harm = Nharm_Row2, relax = relax, axis_ang_1 = Nblade_Row2, axis_ang_2 = 1)
    choroParamsStage = {rowNameList[0]:choroParamsRow1, rowNameList[1]:choroParamsRow2}
    
    MSG = 'Number of harmonics for %s : %s'%(rowNameList[0], Nharm_Row1)
    print(J.WARN + MSG + J.ENDC)

    MSG = 'Number of harmonics for %s : %s'%(rowNameList[1], Nharm_Row2)
    print(J.WARN + MSG + J.ENDC)
    
    return choroParamsStage


def add_choro_data(t,rowName,freq,omega,Nharm,relax,axis_ang_1,axis_ang_2):
    '''
    Add the chorochronic parameters computed using computeChoroParameters() to the PyTree t.
    
    Parameters
    ----------

        t : PyTree
            Tree to modify

        rowName : str
            Name of the considered row. Correspond to an element of TurboConfiguration['Rows'].keys()

        freq : float
            Frequency of blade passage to next wheel, as provided by computeChoroParameters().

        Nharm : float
            Number of harmonics of the considered row, as provided by computeChoroParameters().

        omega : float
            rotation speed in rad/s relative to the other row, as provided by computeChoroParameters().

        relax : float
            Relaxation coefficient for multichoro condition, as provided by computeChoroParameters(). Equals 1.0 for a single stage rotor/stator stage.

        axis_ang_1 : float
           Number of blades in the considered row, as provided by computeChoroParameters().

        axis_ang_2 : float
            Number of simulated passages for the considered row, as provided by computeChoroParameters().

    ''' 
    zones = C.getFamilyZones(t,rowName)
    fam_node = I.getNodeFromName(t,rowName)
    motion_node = I.getNodeFromName(fam_node,'.Solver#Motion')

    for z in zones:
        # I.printTree(z)
        sp = I.getNodeFromName1(z,'.Solver#Param')
        if not isinstance(sp,list): sp = I.createChild(z,'.Solver#Param','UserDefinedData_t')
        I.newDataArray('f_freq', value=float(freq), parent=sp)
        I.newDataArray('f_omega',value=float(omega),parent=sp)
        I.newDataArray('f_harm', value=float(Nharm), parent=sp)
        I.newDataArray('f_relax',value=float(relax),parent=sp)
        I.newDataArray('axis_ang_1',value=axis_ang_1,parent=sp)
        I.newDataArray('axis_ang_2',value=axis_ang_2,parent=sp)
        for node in I.getChildren(motion_node):
            if 'axis' in I.getName(node):
                I.newDataArray(I.getName(node),value=I.getValue(node),parent=sp)

    print('Adding axis ang to Motion node')
    I.newDataArray('axis_ang_1',value=axis_ang_1,parent=motion_node)
    I.newDataArray('axis_ang_2',value=axis_ang_2,parent=motion_node)
    


def computeChoroAndAddParameters(t, Rows, Nharm_Row1 = 20., Nharm_Row2 = 20.):
    '''
    Compute the parameters to run a chorochronic computation an add them to the PyTree t.
    
    Parameters
    ----------

        t : PyTree
            Tree to modify

        Rows : :py:class:`dict`
            Dictionary of Rows as provided in TurboConfiguration for the prepareMainCGNS function.

        Nharm_Row1 : float
            Number of harmonics of the first row.

        Nharm_Row2 : float
            Number of harmonics of the second row.
    '''  
    
    choroParamsStage = computeChoroParameters(t, Rows, Nharm_Row1 = Nharm_Row1, Nharm_Row2 = Nharm_Row2)
    RowsL=[]
    for row_fam in Rows.keys():
        RowsL.append(row_fam)
    add_choro_data(t,RowsL[0],freq = choroParamsStage[RowsL[0]]['freq'], omega = choroParamsStage[RowsL[0]]['omega'], Nharm = choroParamsStage[RowsL[0]]['harm'], relax = choroParamsStage[RowsL[0]]['relax'], axis_ang_1 = choroParamsStage[RowsL[0]]['axis_ang_1'],axis_ang_2 = choroParamsStage[RowsL[0]]['axis_ang_2'])
    add_choro_data(t,RowsL[1],freq = choroParamsStage[RowsL[1]]['freq'], omega = choroParamsStage[RowsL[1]]['omega'], Nharm = choroParamsStage[RowsL[1]]['harm'], relax = choroParamsStage[RowsL[1]]['relax'], axis_ang_1 = choroParamsStage[RowsL[1]]['axis_ang_1'],axis_ang_2 = choroParamsStage[RowsL[1]]['axis_ang_2'])


def setChorochronic(t, Rows, left, right, method='globborder_dict', stage_choro_type='characteristic', harm_freq_comp=1, jtype = 'nomatch_rad_line', Nharm_Row1 = 20., Nharm_Row2 = 20.):
    '''
    Compute the parameters to run a chorochronic computation.
    
    Parameters
    ----------

        t : PyTree
            Tree to modify

        Rows : :py:class:`dict`
            Dictionary of Rows as provided in TurboConfiguration for the prepareMainCGNS function.

        left : str
            Name of the family on the left side of the chorochronic interface.

        right : str
            Name of the family on the right side of the chorochronic interface.

        method : optional, str
            Method used to compute the globborder of the chorochronic interface. 
            The default value is``'globborder_dict'``, it corresponds to the ETC topological
            algorithm.
            Another possible value is ``'poswin'`` to use the geometrical
            algorithm in *turbo* (in this case, *turbo* environment must be
            sourced).

        stage_choro_type : str
            Type of chorochronic interface:
                characteristic (default) : condition based on characteristic relations.
                half_sum : condition based on half-sum of values.

        harm_freq_comp : str
            Frequency of harmonics computation

        jtype : str
            Specifies the type of join:
                match_rad_line : coincident radial match along lines (for “stage”-like turbomachine conditions);
                nomatch_rad_line : non-coincident radial match along lines (for “stage”-like turbomachine conditions).

        Nharm_Row1 : float
            Number of harmonics of the first row.

        Nharm_Row2 : float
            Number of harmonics of the second row.
    '''   

    setBC_stage_choro(t, left, right, method='globborder_dict', stage_choro_type='characteristic', harm_freq_comp=1, jtype = 'nomatch_rad_line')
    convertPeriodic2Chorochrono(t)
    computeChoroAndAddParameters(t, Rows, Nharm_Row1 = Nharm_Row1, Nharm_Row2 = Nharm_Row2)


def plotIsoSpeedLine(perfo, filename='isoSpeedLines.png'):
    '''Plot performances in **perfo** (total pressure ratio and isentropic efficiency depending on massflow)

    Parameters
    ----------
    perfo : list
        as got from :py:func:`printConfigurationStatusWithPerfo` or :py:func:`getPostprocessQuantities`
    '''
    import matplotlib.pyplot as plt

    linestyles = [dict(linestyle=ls, marker=mk) for mk in ['o', 's', 'd', 'h']
                                            for ls in ['-', ':', '--', '-.']]
    fig, ax1 = plt.subplots()

    # Total pressure ratio
    color = 'teal'
    ax1.set_xlabel('MassFlow (kg/s)')
    ax1.set_ylabel('Total pressure ratio (-)', color=color)
    for i, perfo_iso in enumerate(perfo):
        speed = perfo_iso['RotationSpeed'] * 30./np.pi # in RPM
        ax1.plot(perfo_iso['MassFlow'], perfo_iso['PressureStagnationRatio'],
                color=color, 
                label=f'{speed} rpm', 
                **linestyles[i])
    ax1.tick_params(axis='y', labelcolor=color)

    # Isentropic efficiency
    color = 'firebrick'
    ax2 = ax1.twinx()
    ax2.set_ylabel('Isentropic efficiency (-)', color=color)
    for i, perfo_iso in enumerate(perfo):
        speed = perfo_iso['RotationSpeed'] * 30./np.pi # in RPM
        ax2.plot(perfo_iso['MassFlow'], perfo_iso['EfficiencyIsentropic'],
                color=color, 
                label=f'{speed} rpm', 
                **linestyles[i])
        # To display legend in black
        ax2.plot([], [], color='k', label=f'{speed} rpm', **linestyles[i])
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(top=1)

    if len(perfo) > 1:
        ax2.legend(loc='center left', bbox_to_anchor=(1.1, 0.5))

    fig.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.show()


def initializeFlowSolutionWithTurbo(t, FluidProperties, ReferenceValues, TurboConfiguration, mask=None):
    '''
    Initialize the flow solution of **t** with the module ``turbo``.
    The initial flow is computed analytically in the 2D-throughflow plane
    based on:

    * radial equilibrium in the radial direction.

    * Euler theorem between rows in the axial direction.

    The values **FlowAngleAtRoot** and **FlowAngleAtTip** (relative angles 'beta')
    must be provided for each row in **TurboConfiguration**.

    .. note::
        See also documentation of the related function in ``turbo`` module
        `<file:///stck/jmarty/TOOLS/turbo/doc/html/initial.html>`_

    .. important::
        Dependency to ``turbo``

    .. danger::
        Works only in Python3, considering that dictionaries conserve order.
        Rows in TurboConfiguration must be list in the downstream order.

    Parameters
    ----------

        t : PyTree
            Tree to initialize

        FluidProperties : dict
            as produced by :py:func:`computeFluidProperties`

        ReferenceValues : dict
            as produced by :py:func:`computeReferenceValues`

        TurboConfiguration : dict

        mask : PyTree

    Returns
    -------

        t : PyTree
            Modified PyTree
    '''
    import turbo.initial as TI

    if not mask:
        if os.path.isfile('mask.cgns'):
            mask = C.convertFile2PyTree('mask.cgns')
        elif os.path.isfile('../../DISPATCHER/mask.cgns'):
            mask = C.convertFile2PyTree('../../DISPATCHER/mask.cgns')
        else: raise NameError("File 'mask.cgns' not found")

    def getInletPlane(t, row, rowParams):
        try: 
            return rowParams['InletPlane']
        except KeyError:
            zones = C.getFamilyZones(t, row)
            return C.getMinValue(zones, 'CoordinateX')

    def getOutletPlane(t, row, rowParams):
        try: 
            return rowParams['OutletPlane']
        except KeyError:
            zones = C.getFamilyZones(t, row)
            return C.getMaxValue(zones, 'CoordinateX')

    class RefState(object):
        def __init__(self):
          self.Gamma = FluidProperties['Gamma']
          self.Rgaz  = FluidProperties['IdealGasConstant']
          self.Pio   = ReferenceValues['PressureStagnation']
          self.Tio   = ReferenceValues['TemperatureStagnation']
          self.roio  = self.Pio / self.Tio / self.Rgaz
          self.aio   = (self.Gamma * self.Rgaz * self.Tio)**0.5
          self.Lref  = 1.

    # Get turbulent variables names and values
    turbDict = dict(zip(ReferenceValues['FieldsTurbulence'],  ReferenceValues['ReferenceStateTurbulence']))

    planes_data = []

    row, rowParams = list(TurboConfiguration['Rows'].items())[0]
    xIn = getInletPlane(t, row, rowParams)
    alpha = ReferenceValues['AngleOfAttackDeg']
    planes_data.append(
        dict(
            omega = 0.,
            beta = [alpha, alpha],
            Pt = 1.,
            Tt = 1.,
            massflow = ReferenceValues['MassFlow'],
            plane_points = [[xIn,-999.],[xIn,999.]],
            plane_name = '{}_InletPlane'.format(row)
        )
    )

    for row, rowParams in TurboConfiguration['Rows'].items():
        xOut = getOutletPlane(t, row, rowParams)
        omega = rowParams['RotationSpeed']
        beta1 = rowParams.get('FlowAngleAtRoot', 0.)
        beta2 = rowParams.get('FlowAngleAtTip', 0.)
        Csir = 1. if omega == 0 else 0.95  # Total pressure loss is null for a rotor, 5% for a stator
        planes_data.append(
            dict(
                omega = rowParams['RotationSpeed'],
                beta = [beta1, beta2],
                Csir = Csir,
                plane_points = [[xOut,-999.],[xOut,999.]],
                plane_name = '{}_OutletPlane'.format(row)
                )
        )

    # > Initialization
    print(J.CYAN + 'Initialization with turbo...' + J.ENDC)
    silence = J.OutputGrabber()
    with silence:
        t = TI.initialize(t, mask, RefState(), planes_data,
                nbslice=10,
                constant_data=turbDict,
                turbvarsname=list(turbDict),
                velocity='absolute',
                useSI=True,
                keepTmpVars=False,
                keepFS=True  # To conserve other FlowSolution_t nodes, as FlowSolution#Height
                )
    print('..done.')

    return t


def postprocess_turbomachinery(surfaces, stages=[], 
                                var4comp_repart=None, var4comp_perf=None, var2keep=None, 
                                computeRadialProfiles=True, 
                                heightListForIsentropicMach='all',
                                config='annular', 
                                lin_axis='XY',
                                RowType='compressor',
                                container_at_vertex='FlowSolution#InitV'):
    '''
    Perform a series of classical postprocessings for a turbomachinery case : 

    #. Compute extra variables, in relative and absolute frames of reference

    #. Compute averaged values for all iso-X planes, and
       compare inlet and outlet planes for each row if available, to get row performance (total 
       pressure ratio, isentropic efficiency, etc) 

    #. Compute radial profiles for all iso-X planes (results are in the `RadialProfiles` base), and
       compare inlet and outlet planes for each row if available, to get row performance (total 
       pressure ratio, isentropic efficiency, etc) 

    #. Compute isentropic Mach number on blades, slicing at constant height, for all values of height 
       already extracted as iso-surfaces.

    Parameters
    ----------

        surfaces : PyTree
            extracted surfaces

        stages : :py:class:`list` of :py:class:`tuple`, optional
            List of row stages, of the form:

            >>> stages = [('rotor1', 'stator1'), ('rotor2', 'stator2')] 

            For each tuple of rows, the inlet plane of row 1 is compared with the outlet plane of row 2.

        var4comp_repart : :py:class:`list`, optional
            List of variables computed for radial distributions. If not given, all possible variables are computed.

        var4comp_perf : :py:class:`list`, optional
            List of variables computed for row performance (plane to plane comparison). If not given, 
            the same variables as in **var4comp_repart** are computed, plus `Power`.

        var2keep : :py:class:`list`, optional
            List of variables to keep in the saved file. If not given, the following variables are kept:
            
            .. code-block:: python

                var2keep = [
                    'Pressure', 'Temperature', 'PressureStagnation', 'TemperatureStagnation',
                    'StagnationPressureRelDim', 'StagnationTemperatureRelDim',
                    'Entropy',
                    'Viscosity_EddyMolecularRatio',
                    'VelocitySoundDim', 'StagnationEnthalpyAbsDim',
                    'MachNumberAbs', 'MachNumberRel',
                    'AlphaAngleDegree',  'BetaAngleDegree', 'PhiAngleDegree',
                    'VelocityXAbsDim', 'VelocityRadiusAbsDim', 'VelocityThetaAbsDim',
                    'VelocityMeridianDim', 'VelocityRadiusRelDim', 'VelocityThetaRelDim',
                    ]
        
        computeRadialProfiles : bool
            Choose or not to compute radial profiles.
        
        heightListForIsentropicMach : list or str, optional
            List of heights to make slices on blades. 
            If 'all' (by default), the list is got by taking the values of the existing 
            iso-height surfaces in the input tree.
        
        config : str
            see :py:func:`MOLA.PostprocessTurbo.compute1DRadialProfiles`

        lin_axis : str
            see :py:func:`MOLA.PostprocessTurbo.compute1DRadialProfiles`

        RowType : str
            see parameter 'config' of :py:func:`MOLA.PostprocessTurbo.compareRadialProfilesPlane2Plane`
        
        container_at_vertex : :py:class:`str` or :py:class:`list` of :py:class:`str`
            specifies the *FlowSolution* container located at 
            vertex where postprocess will be applied. 

            .. hint::
                provide a :py:class:`list` of :py:class:`str` so that the 
                postprocess will be applied to each of the provided containers.
                This is useful for making post-processing on e.g. both
                instantaneous and averaged flow fields
        
    '''
    import Converter.Mpi as Cmpi
    import MOLA.PostprocessTurbo as Post
    import turbo.user as TUS

    Post.setup = J.load_source('setup', 'setup.py')

    # prepare auxiliary surfaces tree, with flattened FlowSolution container
    # located at Vertex including ChannelHeight
    previous_vertex_container = I.__FlowSolutionNodes__
    turbo_required_vertex_container = 'FlowSolution'
    turbo_new_centers_container = 'FlowSolution#Centers'

    if isinstance(container_at_vertex, str):
        containers_at_vertex = [container_at_vertex]
    elif not isinstance(container_at_vertex, list):
        raise TypeError('container_at_vertex must be str or list of str')
    else:
        containers_at_vertex = container_at_vertex

    suffixes = [c.replace('FlowSolution','') for c in containers_at_vertex]

    for container_at_vertex in containers_at_vertex:
        I.__FlowSolutionNodes__ = container_at_vertex
        for zone in I.getZones(surfaces):
            fs_container = I.getNodeFromName1(zone, container_at_vertex)
            if not fs_container: continue
            channel_height = I.getNodeFromName2(zone, 'ChannelHeight')
            if not channel_height: continue
            fs_container[2] += [ channel_height ]
            fs_container[0] = turbo_required_vertex_container

        #______________________________________________________________________________
        # Variables
        #______________________________________________________________________________
        allVariables = TUS.getFields(config=config)
        if not var4comp_repart:
            var4comp_repart = ['StagnationEnthalpyDelta',
                            'StagnationPressureRatio', 'StagnationTemperatureRatio',
                            'StaticPressureRatio', 'Static2StagnationPressureRatio',
                            'IsentropicEfficiency', 'PolytropicEfficiency',
                            'StaticPressureCoefficient', 'StagnationPressureCoefficient',
                            'StagnationPressureLoss1', 'StagnationPressureLoss2',
                            ]
        if not var4comp_perf:
            var4comp_perf = var4comp_repart + ['Power']
        if not var2keep:
            var2keep = [
                'Pressure', 'Temperature', 'PressureStagnation', 'TemperatureStagnation',
                'StagnationPressureRelDim', 'StagnationTemperatureRelDim',
                'Entropy',
                'Viscosity_EddyMolecularRatio',
                'VelocitySoundDim', 'StagnationEnthalpyAbsDim',
                'MachNumberAbs', 'MachNumberRel',
                'AlphaAngleDegree',  'BetaAngleDegree', 'PhiAngleDegree',
                'VelocityXAbsDim', 'VelocityRadiusAbsDim', 'VelocityThetaAbsDim',
                'VelocityMeridianDim', 'VelocityRadiusRelDim', 'VelocityThetaRelDim',
            ]

        variablesByAverage = Post.sortVariablesByAverage(allVariables)

        #______________________________________________________________________________#
        Post.computeVariablesOnIsosurface(surfaces, allVariables, config=config, lin_axis=lin_axis)
        Post.compute0DPerformances(surfaces, variablesByAverage)
        if computeRadialProfiles: 
            Post.compute1DRadialProfiles(
                surfaces, variablesByAverage, config=config, lin_axis=lin_axis)
        if config == 'annular' and heightListForIsentropicMach:
            # TODO compute Machis also for linear cascade. Is this available in turbo ? 
            Post.computeVariablesOnBladeProfiles(surfaces, height_list=heightListForIsentropicMach)
        #______________________________________________________________________________#

        if Cmpi.rank == 0:
            Post.comparePerfoPlane2Plane(surfaces, var4comp_perf, stages)
            if computeRadialProfiles: 
                Post.compareRadialProfilesPlane2Plane(
                    surfaces, var4comp_repart, stages, config=RowType)

        Post.cleanSurfaces(surfaces, var2keep=var2keep)

        suffix = container_at_vertex.replace('FlowSolution','')
        for zone in I.getZones(surfaces):
            for fs_container in I.getNodesFromType1(zone, 'FlowSolution_t'):
                fs_name = fs_container[0]
                is_turbo_container = fs_name in [turbo_required_vertex_container,
                                                turbo_new_centers_container]
                is_new_comparison = fs_name.startswith('Comparison') and not \
                                    fs_name.endswith(suffix)

                if is_turbo_container or is_new_comparison: 
                    if not any([fs_container[0].endswith(s) for s in suffixes]):
                        fs_container[0] += suffix
                        if fs_container[0].startswith(turbo_new_centers_container):
                            fs_container[0]=fs_container[0].replace(turbo_new_centers_container,
                                                                    'FlowSolution')

        I.__FlowSolutionNodes__ = previous_vertex_container

