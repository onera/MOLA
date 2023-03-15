#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

'''
MOLA - WorkflowCompressor.py

WORKFLOW COMPRESSOR

Collection of functions designed for Workflow Compressor

File history:
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


    try:
        from . import parametrize_channel_height as ParamHeight
    except ImportError:
        MSG = 'Fail to import parametrize_channel_height: function parametrize_channel_height is unavailable'.format(__name__)
        print(J.WARN + MSG + J.ENDC)
        ParamHeight = None

def checkDependencies():
    '''
    Make a series of functional tests in order to determine if the user
    environment is correctly set for using the Workflow Compressor
    '''
    JM.checkDependencies()

    print('Checking ETC...')
    try:
        import etc.transform.__future__
        print(J.GREEN+'ETC module is available'+J.ENDC)
    except ImportError:
        MSG = 'Fail to import ETC module: Some functions of {} are unavailable'.format(__name__)
        print(J.FAIL + MSG + J.ENDC)

    print('Checking MOLA.parametrize_channel_height...')
    if ParamHeight is None:
        MSG = 'Fail to import MOLA.parametrize_channel_height module: Some functions of {} are unavailable'.format(__name__)
        print(J.FAIL + MSG + J.ENDC)
    else:
        print(J.GREEN+'MOLA.parametrize_channel_height module is available'+J.ENDC)

    print('\nVERIFICATIONS TERMINATED')


def prepareMesh4ElsA(mesh, InputMeshes=None, splitOptions=None, #dict(SplitBlocks=False),
                    duplicationInfos={}, zonesToRename={},
                    scale=1., rotation='fromAG5', tol=1e-8, PeriodicTranslation=None,
                    BodyForceRows=None, families2Remove=[], saveGeometricalDataForBodyForce=True):
    '''
    This is a macro-function used to prepare the mesh for an elsA computation
    from a CGNS file provided by Autogrid 5.

    The sequence of operations performed are the following:

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
          Autgrid 5). The mesh will be rotated to follow the elsA convention,
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


        PeriodicTranslation : :py:obj:'None' or :py:class:`list` of :py:class:`float`
            If not :py:obj:'None', the configuration is considered to be with
            a periodicity in the direction **PeriodicTranslation**. This argument
            has to be used for linear cascade configurations.

        tol : float
            Tolerance for connectivities matching (including periodic connectivities).

        BodyForceRows : :py:class:`dict` or :py:obj:`None`
            If not :py:obj:`None`, this parameters allows to replace user-defined
            row domains with meshes adapted to body-force modelling.
            See documentation of 
            :py:func:`MOLA.BodyForceTurbomachinery.replaceRowWithBodyForceMesh`.

        families2Remove : list
            Families to remove in the tree when using body-force. Should be a list 
            of families related to interstage interfaces between a stator row and 
            a BFM row, or to BFM rows. It allows to force a matching mesh at the interface
            instead having a mixing plane.

        saveGeometricalDataForBodyForce : bool
            If :py:obj:`True`, save the intermediate files 'BodyForceData_{row}.cgns' for each row.
            These files contain a CGNS tree with :
                #. 4 lines (1D zones) corresponding to Hub, Shroud, Leading edge and Trailing Edge.
                #. The zone'Skeleton' with geometrical data on blade profile (used for interpolation later). 

    Returns
    -------

        t : PyTree
            the pre-processed mesh tree (usually saved as ``mesh.cgns``)

            .. important:: This tree is **NOT** ready for elsA computation yet !
                The user shall employ function :py:func:`prepareMainCGNS4ElsA`
                as next step
    '''
    if isinstance(mesh,str):
        filename = mesh
        t = C.convertFile2PyTree(mesh)
    elif I.isTopTree(mesh):
        filename = None
        t = mesh
    else:
        raise ValueError('parameter mesh must be either a filename or a PyTree')

    I._fixNGon(t) # Needed for an unstructured mesh

    if InputMeshes is None:
        InputMeshes = generate_input_meshes_from_autogrid(t,
            scale=scale, rotation=rotation, tol=tol, PeriodicTranslation=PeriodicTranslation)
        for InputMesh in InputMeshes: 
            InputMesh['file'] = filename

    PRE.checkFamiliesInZonesAndBC(t)
    PRE.transform(t, InputMeshes)

    if BodyForceRows:
        # Remesh rows to model with body-force
        t, newRowMeshes = BF.replaceRowWithBodyForceMesh(
            t, BodyForceRows, saveGeometricalDataForBodyForce=saveGeometricalDataForBodyForce)

    t = clean_mesh_from_autogrid(t, basename=InputMeshes[0]['baseName'], zonesToRename=zonesToRename)

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

    if splitOptions is not None:
        t = PRE.splitAndDistribute(t, InputMeshes, **splitOptions)
    else:
        t = PRE.connectMesh(t, InputMeshes)
    # WARNING: Names of BC_t nodes must be unique to use PyPart on globborders
    for l in [2,3,4]: I._correctPyTree(t, level=l)

    for base, meshInfo in zip(I.getBases(t), InputMeshes):
        J.set(base,'.MOLA#InputMesh',**meshInfo)

    PRE.adapt2elsA(t, InputMeshes)
    J.checkEmptyBC(t)

    return t

def prepareMainCGNS4ElsA(mesh='mesh.cgns', ReferenceValuesParams={},
        NumericalParams={}, OverrideSolverKeys= {}, 
        TurboConfiguration={}, Extractions=[], BoundaryConditions=[],
        PostprocessOptions={}, BodyForceInputData={}, writeOutputFields=True,
        bladeFamilyNames=['BLADE', 'AUBE'], Initialization={'method':'uniform'},
        JobInformation={}, SubmitJob=False,
        FULL_CGNS_MODE=False, COPY_TEMPLATES=True):
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

        BodyForceInputData : :py:class:`dict`
            if provided, each key in this :py:class:`dict` is the name of a row family to model
            with body-force. The associated value is a sub-dictionary, with the following 
            potential entries:

                * model (:py:class:`dict`): the name of the body-force model to apply. Available models 
                  are: 'hall', 'blockage', 'Tspread', 'constant'.

                * rampIterations (:py:class:`dict`): The number of iterations to apply a ramp on source terms, 
                  starting from `BodyForceInitialIteration` (in `ReferenceValues['CoprocessOptions']`). 
                  If not given, there is no ramp (source terms are fully applied from the `BodyForceInitialIteration`).

                * other optional parameters depending on the **model** 
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

            .. note::
                only relevant if **COPY_TEMPLATES** is py:obj:`True` and
                **JobInformation** is provided

        FULL_CGNS_MODE : bool
            if :py:obj:`True`, put all elsA keys in a node ``.Solver#Compute``
            to run in full CGNS mode.

        COPY_TEMPLATES : bool
            If :py:obj:`True` (default value), copy templates files in the
            current directory.

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
        t = C.convertFile2PyTree(mesh)
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

    if I.getNodeFromName(t, 'proc'):
        JobInformation['NumberOfProcessors'] = int(max(PRE.getProc(t))+1)
        Splitter = None
    else:
        Splitter = 'PyPart'

    elsAkeysCFD      = PRE.getElsAkeysCFD(nomatch_linem_tol=1e-6, unstructured=IsUnstructured)
    elsAkeysModel    = PRE.getElsAkeysModel(FluidProperties, ReferenceValues, unstructured=IsUnstructured)
    if BodyForceInputData: 
        NumericalParams['useBodyForce'] = True
    if not 'NumericalScheme' in NumericalParams:
        NumericalParams['NumericalScheme'] = 'roe'
    elsAkeysNumerics = PRE.getElsAkeysNumerics(ReferenceValues,
                            unstructured=IsUnstructured, **NumericalParams)

    if Initialization['method'] == 'turbo':
        t = initializeFlowSolutionWithTurbo(t, FluidProperties, ReferenceValues, TurboConfiguration)
    else:
        PRE.initializeFlowSolution(t, Initialization, ReferenceValues)

    if not 'PeriodicTranslation' in TurboConfiguration and \
        any([rowParams['NumberOfBladesSimulated'] > rowParams['NumberOfBladesInInitialMesh'] \
            for rowParams in TurboConfiguration['Rows'].values()]):
        t = duplicate_flow_solution(t, TurboConfiguration)

    setMotionForRowsFamilies(t, TurboConfiguration)
    setBoundaryConditions(t, BoundaryConditions, TurboConfiguration,
                            FluidProperties, ReferenceValues,
                            bladeFamilyNames=bladeFamilyNames)

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
    PRE.addExtractions(t, AllSetupDicts['ReferenceValues'],
                          AllSetupDicts['elsAkeysModel'],
                          extractCoords=False,
                          BCExtractions=ReferenceValues['BCExtractions'])

    if elsAkeysNumerics['time_algo'] != 'steady':
        PRE.addAverageFieldExtractions(t, AllSetupDicts['ReferenceValues'],
            AllSetupDicts['ReferenceValues']['CoprocessOptions']['FirstIterationForAverage'])

    PRE.addReferenceState(t, AllSetupDicts['FluidProperties'],
                         AllSetupDicts['ReferenceValues'])
    dim = int(AllSetupDicts['elsAkeysCFD']['config'][0])
    PRE.addGoverningEquations(t, dim=dim)
    PRE.writeSetup(AllSetupDicts)

    if FULL_CGNS_MODE:
        PRE.addElsAKeys2CGNS(t, [AllSetupDicts['elsAkeysCFD'],
                                 AllSetupDicts['elsAkeysModel'],
                                 AllSetupDicts['elsAkeysNumerics']])

    PRE.saveMainCGNSwithLinkToOutputFields(t,writeOutputFields=writeOutputFields)

    if not Splitter:
        print('REMEMBER : configuration shall be run using %s%d%s procs'%(J.CYAN,
                                                   JobInformation['NumberOfProcessors'],J.ENDC))
    else:
        print('REMEMBER : configuration shall be run using %s'%(J.CYAN + \
            Splitter + J.ENDC))

    if COPY_TEMPLATES:
        JM.getTemplates('Compressor', otherWorkflowFiles=['monitor_perfos.py'],
                JobInformation=JobInformation)
        if 'DIRECTORY_WORK' in JobInformation:
            PRE.sendSimulationFiles(JobInformation['DIRECTORY_WORK'],
                                    overrideFields=writeOutputFields)

        for i in range(SubmitJob):
            singleton = False if i==0 else True
            JM.submitJob(JobInformation['DIRECTORY_WORK'], singleton=singleton)

    ElapsedTime = str(PRE.datetime.timedelta(seconds=J.tic()-toc))
    hours, minutes, seconds = ElapsedTime.split(':')
    ElapsedTimeHuman = hours+' hours '+minutes+' minutes and '+seconds+' seconds'
    msg = 'prepareMainCGNS took '+ElapsedTimeHuman
    print(J.BOLD+msg+J.ENDC)



################################################################################
#############  Multiple jobs submission  #######################################
################################################################################

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

            .. note:: indicated directory may not exist. In this case, it will
                be created.

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

    ThrottleRange = sorted(list(ThrottleRange))
    # Sort Rotation speeds (and mesh files, if a list is given) 
    RotationSpeedRange = list(RotationSpeedRange)
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

    JM.launchJobsConfiguration(templatesFolder=MOLA.__MOLA_PATH__+'/TEMPLATES/WORKFLOW_COMPRESSOR', otherFiles=otherFiles)

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

def printConfigurationStatusWithPerfo(DIRECTORY_WORK, useLocalConfig=False,
                                        monitoredRow='row_1'):
    '''
    Print the current status of a IsoSpeedLines computation and display
    performance of the monitored row for completed jobs.

    Parameters
    ----------

        DIRECTORY_WORK : str
            directory where ``JobsConfiguration.py`` file is located

        useLocalConfig : bool
            if :py:obj:`True`, use the local ``JobsConfiguration.py``
            file instead of retreiving it from **DIRECTORY_WORK**

        monitoredRow : str
            Name of the row whose performance will be displayed

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
    config = JM.getJobsConfiguration(DIRECTORY_WORK, useLocalConfig)
    Throttle = np.array(sorted(list(set([float(case['CASE_LABEL'].split('_')[0]) for case in config.JobsQueues]))))
    RotationSpeed = np.array(sorted(list(set([case['TurboConfiguration']['ShaftRotationSpeed'] for case in config.JobsQueues]))))

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

            if status == 'COMPLETED':
                lastarrays = JM.getCaseArrays(config, CASE_LABEL, basename='PERFOS_{}'.format(monitoredRow))
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

def getPostprocessQuantities(DIRECTORY_WORK, basename, useLocalConfig=False, rename=True):
    '''
    Print the current status of a IsoSpeedLines computation and display
    performance of the monitored row for completed jobs.

    Parameters
    ----------

        DIRECTORY_WORK : str
            directory where ``JobsConfiguration.py`` file is located

        basename : str
            Name of the base to get

        useLocalConfig : bool
            if :py:obj:`True`, use the local ``JobsConfiguration.py``
            file instead of retreiving it from **DIRECTORY_WORK**

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
    config = JM.getJobsConfiguration(DIRECTORY_WORK, useLocalConfig)
    Throttle = np.array(sorted(list(set([float(case['CASE_LABEL'].split('_')[0]) for case in config.JobsQueues]))))
    RotationSpeed = np.array(sorted(list(set([case['TurboConfiguration']['ShaftRotationSpeed'] for case in config.JobsQueues]))))

    def getCaseLabel(config, throttle, rotSpeed):
        for case in config.JobsQueues:
            if np.isclose(float(case['CASE_LABEL'].split('_')[0]), throttle) and \
                np.isclose(case['TurboConfiguration']['ShaftRotationSpeed'], rotSpeed):

                return case['CASE_LABEL']

    perfo = []
    for rotationSpeed in RotationSpeed:
        perfoOverIsospeedLine = dict(RotationSpeed=rotationSpeed, Throttle=[])

        for idThrottle, throttle in enumerate(Throttle):
            CASE_LABEL = getCaseLabel(config, throttle, rotationSpeed)
            status = JM.statusOfCase(config, CASE_LABEL)

            if status == 'COMPLETED':
                perfoOverIsospeedLine['Throttle'].append(throttle)
                lastarrays = JM.getCaseArrays(config, CASE_LABEL, basename=basename)
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
            ('MassFlow', 'Massflow'), 
            ('PressureStagnationRatio', 'StagnationPressureRatio'), 
            ('EfficiencyIsentropic', 'IsentropicEfficiency')
            ]
        for (name1, name2) in VarsToRename:
            for perfoOverIsospeedLine in perfo: 
                perfoOverIsospeedLine[name1] = perfoOverIsospeedLine[name2]

    return perfo

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


