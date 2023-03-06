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
        from . import ParametrizeChannelHeight as ParamHeight
    except ImportError:
        MSG = 'Fail to import ParametrizeChannelHeight: function parametrizeChannelHeight is unavailable'.format(__name__)
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

    print('Checking MOLA.ParametrizeChannelHeight...')
    if ParamHeight is None:
        MSG = 'Fail to import MOLA.ParametrizeChannelHeight module: Some functions of {} are unavailable'.format(__name__)
        print(J.FAIL + MSG + J.ENDC)
    else:
        print(J.GREEN+'MOLA.ParametrizeChannelHeight module is available'+J.ENDC)

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
        InputMeshes = generateInputMeshesFromAG5(t,
            scale=scale, rotation=rotation, tol=tol, PeriodicTranslation=PeriodicTranslation)
        for InputMesh in InputMeshes: 
            InputMesh['file'] = filename

    PRE.checkFamiliesInZonesAndBC(t)

    if BodyForceRows:
        # Remesh rows to model with body-force
        t, newRowMeshes = BF.replaceRowWithBodyForceMesh(
            t, BodyForceRows, saveGeometricalDataForBodyForce=saveGeometricalDataForBodyForce)

    t = cleanMeshFromAutogrid(t, basename=InputMeshes[0]['baseName'], zonesToRename=zonesToRename)
    PRE.transform(t, InputMeshes)

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

    TurboConfiguration = getTurboConfiguration(t, **TurboConfiguration)
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
        t = duplicateFlowSolution(t, TurboConfiguration)

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


def parametrizeChannelHeight(t, nbslice=101, fsname='FlowSolution#Height',
    hlines='hub_shroud_lines.plt', subTree=None):
    '''
    Compute the variable *ChannelHeight* from a mesh PyTree **t**. This function
    relies on the ETC module.

    Parameters
    ----------

        t : PyTree
            input mesh tree

        nbslice : int
            Number of axial positions used to compute the iso-lines in
            *ChannelHeight*. Change the axial discretization.

        fsname : str
            Name of the ``FlowSolution_t`` container to stock the variable at
            nodes *ChannelHeight*.

        hlines : str
            Name of the intermediate file that contains (x,r) coordinates of hub
            and shroud lines.

        subTree : PyTree
            Part of the main tree **t** used to compute *ChannelHeigth*. For
            zones in **t** but not in **subTree**, *ChannelHeigth* will be equal
            to -1. This option is useful to exclude irelevant zones for height
            computation, for example the domain (if it exists) around the
            nacelle with the external flow. To extract **subTree** based on a
            Family, one may use:

            >>> subTree = C.getFamilyZones(t, Family)

    Returns
    -------

        t : PyTree
            modified tree

    '''
    print(J.CYAN + 'Add ChannelHeight in the mesh...' + J.ENDC)
    excludeZones = True
    if not subTree:
        subTree = t
        excludeZones = False

    ParamHeight.generateHLinesAxial(subTree, hlines, nbslice=nbslice)
    try: ParamHeight.plot_hub_and_shroud_lines(hlines)
    except: pass
    I._rmNodesByName(t, fsname)
    t = ParamHeight.computeHeight(t, hlines, fsname=fsname, writeMask='mask.cgns')

    if excludeZones:
        OLD_FlowSolutionNodes = I.__FlowSolutionNodes__
        I.__FlowSolutionNodes__ = fsname
        zonesInSubTree = [I.getName(z) for z in I.getZones(subTree)]
        for zone in I.getZones(t):
            if I.getName(zone) not in zonesInSubTree:
                C._initVars(zone, 'ChannelHeight=-1')
        I.__FlowSolutionNodes__ = OLD_FlowSolutionNodes

    print(J.GREEN + 'done.' + J.ENDC)
    return t

def parametrizeChannelHeight_future(t, nbslice=101, tol=1e-10, offset=1e-10,
                                elines='shroud_hub_lines.plt', lin_axis=None):
    '''
    Compute the variable *ChannelHeight* from a mesh PyTree **t**. This function
    relies on the turbo module.

    .. important::

        Dependency to *turbo* module. See file:///stck/jmarty/TOOLS/turbo/doc/html/index.html

    Parameters
    ----------

        t : PyTree
            input mesh tree

        nbslice : int
            Number of axial positions used to compute the iso-lines in
            *ChannelHeight*. Change the axial discretization.

        tol : float
            Tolerance to offset the min (+tol) / max (-tol) value for CoordinateX

        offset : float
            Offset value to add an articifial point (not based on real geometry)
            to be sure that the mesh is fully included. 'tol' and 'offset' must
            be consistent.

        elines : str
            Name of the intermediate file that contains (x,r) coordinates of hub
            and shroud lines.

        lin_axis : :py:obj:`None` or :py:class:`str`
            Axis for linear configuration.
            If :py:obj:`None`, the configuration is annular (default case), else
            the configuration is linear.
            'XY' means that X-axis is the streamwise direction and Y-axis is the
            spanwise direction.(see turbo documentation)

    Returns
    -------

        t : PyTree
            modified tree

    '''
    import turbo.height as TH

    print(J.CYAN + 'Add ChannelHeight in the mesh...' + J.ENDC)
    OLD_FlowSolutionNodes = I.__FlowSolutionNodes__
    I.__FlowSolutionNodes__ = 'FlowSolution#Height'

    silence = J.OutputGrabber()
    with silence:
        if not lin_axis:
            # - Generation of hub/shroud lines (axial configuration only)
            endlinesTree = TH.generateHLinesAxial(t, elines, nbslice=nbslice, tol=tol, offset=offset)

            try:
                import matplotlib.pyplot as plt
                # Get geometry
                xHub, yHub = J.getxy(I.getNodeFromName(endlinesTree, 'Hub'))
                xShroud, yShroud = J.getxy(I.getNodeFromName(endlinesTree, 'Shroud'))
                # Plot
                plt.figure()
                plt.plot(xHub, yHub, '-', label='Hub')
                plt.plot(xShroud, yShroud, '-', label='Shroud')
                plt.axis('equal')
                plt.grid()
                plt.xlabel('x (m)')
                plt.ylabel('y (m)')
                plt.savefig(elines.replace('.plt', '.png'), dpi=150, bbox_inches='tight')
            except:
                pass

            # - Generation of the mask file
            m = TH.generateMaskWithChannelHeight(t, elines, 'bin_tp')
        else:
            m = TH.generateMaskWithChannelHeightLinear(t, lin_axis=lin_axis)
        # - Generation of the ChannelHeight field
        t = TH.computeHeightFromMask(t, m, writeMask='mask.cgns')

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
        t = C.convertFile2PyTree(mesh)
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

            .. note:: Rotation of vectors is done with Cassiopee function
                      Transform.rotate. However, it is not useful to put the
                      prefix 'centers:'. It will be added automatically in the
                      function.

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
    # Compute number of blades in the mesh
    Nb = NumberOfBlades * deltaTheta / (2*np.pi)
    Nb = int(np.round(Nb))
    print('Number of blades in initial mesh for {}: {}'.format(FamilyName, Nb))
    return Nb

def computeReferenceValues(FluidProperties, PressureStagnation,
                           TemperatureStagnation, Surface, MassFlow=None, Mach=None, TurbulenceLevel=0.001,
        Viscosity_EddyMolecularRatio=0.1, TurbulenceModel='Wilcox2006-klim',
        TurbulenceCutoff=1e-8, TransitionMode=None, CoprocessOptions={},
        Length=1.0, TorqueOrigin=[0., 0., 0.],
        FieldsAdditionalExtractions=['ViscosityMolecular', 'Viscosity_EddyMolecularRatio', 'Pressure', 'Temperature', 'PressureStagnation', 'TemperatureStagnation', 'Mach', 'Entropy'],
        BCExtractions=dict(
            BCWall = ['normalvector', 'frictionvector','psta', 'bl_quantities_2d', 'yplusmeshsize'],
            BCInflow = ['convflux_ro'],
            BCOutflow = ['convflux_ro']),
        AngleOfAttackDeg=0.,
        YawAxis=[0.,0.,1.],
        PitchAxis=[0.,1.,0.]):
    '''
    This function is the Compressor's equivalent of :func:`MOLA.Preprocess.computeReferenceValues`.
    The main difference is that in this case reference values are set through
    ``MassFlow``, total Pressure ``PressureStagnation``, total Temperature
    ``TemperatureStagnation`` and ``Surface``.

    You can also give the Mach number instead of massflow (but not both).

    Please, refer to :func:`MOLA.Preprocess.computeReferenceValues` doc for more details.
    '''
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
        RequestedStatistics = CoprocessOptions['RequestedStatistics']
        for stat in TurboStatistics:
            if stat not in CoprocessOptions:
                RequestedStatistics.append( stat )
    except KeyError:
        CoprocessOptions['RequestedStatistics'] = TurboStatistics


    ReferenceValues = PRE.computeReferenceValues(FluidProperties,
        Density=Density,
        Velocity=Velocity,
        Temperature=Temperature,
        AngleOfAttackDeg=AngleOfAttackDeg,
        AngleOfSlipDeg = 0.0,
        YawAxis=YawAxis,
        PitchAxis=PitchAxis,
        TurbulenceLevel=TurbulenceLevel,
        Surface=Surface,
        Length=Length,
        TorqueOrigin=TorqueOrigin,
        TurbulenceModel=TurbulenceModel,
        Viscosity_EddyMolecularRatio=Viscosity_EddyMolecularRatio,
        TurbulenceCutoff=TurbulenceCutoff,
        TransitionMode=TransitionMode,
        CoprocessOptions=CoprocessOptions,
        FieldsAdditionalExtractions=FieldsAdditionalExtractions,
        BCExtractions=BCExtractions)

    addKeys = dict(
        PressureStagnation = PressureStagnation,
        TemperatureStagnation = TemperatureStagnation,
        MassFlow = MassFlow,
        )

    ReferenceValues.update(addKeys)

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
    PeriodicTranslation=None):
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
        if bc['type'] == 'InflowStagnation' or bc['type'].startswith('inj')]
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
                        value=rowParams[plane], ReferenceRow=row, tag=plane))

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
            omega = rowParams['RotationSpeed']
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
        J.set(famNode, '.Solver#Motion',
                motion='mobile',
                omega=omega,
                axis_pnt_x=0., axis_pnt_y=0., axis_pnt_z=0.,
                axis_vct_x=1., axis_vct_y=0., axis_vct_z=0.)


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
    setBC_inj1, setBC_inj1_uniform, setBC_inj1_interpFromFile,
    setBC_outpres, setBC_outmfr2,
    setBC_outradeq, setBC_outradeqhyb,
    setBC_stage_mxpl, setBC_stage_mxpl_hyb,
    setBC_stage_red, setBC_stage_red_hyb,
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

    >>> dict(type='InflowStagnation', option='uniform', FamilyName='row_1_INFLOW')

    It defines a uniform inflow condition imposing stagnation quantities ('inj1' in
    *elsA*) based on the **ReferenceValues**  and **FluidProperties**
    :py:class:`dict`. To impose values not based on **ReferenceValues**, additional
    optional parameters may be given (see the dedicated documentation for the function).

    >>> dict(type='InflowStagnation', option='file', FamilyName='row_1_INFLOW', filename='inflow.cgns')

    It defines an inflow condition imposing stagnation quantities ('inj1' in
    *elsA*) interpolating a 2D map written in the given file (must be given at cell centers, 
    in the container 'FlowSolution#Centers'). 

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

    >>> dict(type='OutflowMassflow', FamilyName='row_2_OUTFLOW', Massflow=5.)

    It defines an outflow condition imposing the massflow ('outmfr2' in *elsA*).
    Be careful, **Massflow** should be the massflow through the given family BC
    *in the simulated domain* (not the 360 degrees configuration, except if it
    is simulated).
    If **Massflow** is not given, the massflow given in the **ReferenceValues**
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

    '''
    PreferedBoundaryConditions = dict(
        Farfield                     = 'nref',
        InflowStagnation             = 'inj1',
        InflowMassFlow               = 'injmfr1',
        OutflowPressure              = 'outpres',
        OutflowMassFlow              = 'outmfr2',
        OutflowRadialEquilibrium     = 'outradeq',
        MixingPlane                  = 'stage_mxpl',
        UnsteadyRotorStatorInterface = 'stage_red',
        WallViscous                  = 'walladia',
        WallViscousIsothermal        = 'wallisoth',
        WallInviscid                 = 'wallslip',
        SymmetryPlane                = 'sym',
    )

    print(J.CYAN + 'set BCs at walls' + J.ENDC)
    setBC_Walls(t, TurboConfiguration, bladeFamilyNames=bladeFamilyNames)

    for BCparam in BoundaryConditions:

        BCkwargs = {key:BCparam[key] for key in BCparam if key not in ['type', 'option']}
        if BCparam['type'] in PreferedBoundaryConditions:
            BCparam['type'] = PreferedBoundaryConditions[BCparam['type']]

        if BCparam['type'] == 'nref':
            print(J.CYAN + 'set BC nref on ' + BCparam['FamilyName'] + J.ENDC)
            setBC_nref(t, **BCkwargs)

        elif BCparam['type'] == 'inj1':

            if 'option' not in BCparam:
                if 'bc' in BCkwargs:
                    BCparam['option'] = 'bc'
                else:
                    BCparam['option'] = 'uniform'

            if BCparam['option'] == 'uniform':
                print(J.CYAN + 'set BC inj1 (uniform) on ' + BCparam['FamilyName'] + J.ENDC)
                setBC_inj1_uniform(t, FluidProperties, ReferenceValues, **BCkwargs)

            elif BCparam['option'] == 'file':
                print('{}set BC inj1 (from file {}) on {}{}'.format(J.CYAN,
                    BCparam['filename'], BCparam['FamilyName'], J.ENDC))
                setBC_inj1_interpFromFile(t, ReferenceValues, **BCkwargs)

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
    def extendListOfFamilies(FamilyNames):
        '''
        For each <NAME> in the list **FamilyNames**, add Name, name and NAME.
        '''
        ExtendedFamilyNames = copy.deepcopy(FamilyNames)
        for fam in FamilyNames:
            newNames = [fam.lower(), fam.upper(), fam.capitalize()]
            for name in newNames:
                if name not in ExtendedFamilyNames:
                    ExtendedFamilyNames.append(name)
        return ExtendedFamilyNames

    bladeFamilyNames = extendListOfFamilies(bladeFamilyNames)
    hubFamilyNames = extendListOfFamilies(hubFamilyNames)
    shroudFamilyNames = extendListOfFamilies(shroudFamilyNames)

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
        return np.asfortranarray(omega)

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

        assert ZoneFamilyName is not None, 'Cannot determine associated row for family {}. '.format(FamilyNameBC)
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

def setBC_inj1(t, FamilyName, ImposedVariables, bc=None, variableForInterpolation='ChannelHeight'):
    '''
    Generic function to impose a Boundary Condition ``inj1``. The following
    functions are more specific:

        * :py:func:`setBC_inj1_uniform`

        * :py:func:`setBC_inj1_interpFromFile`

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

    setBC_inj1_uniform, setBC_inj1_interpFromFile
    '''
    if not bc and not all([np.ndim(v)==0 and not callable(v) for v in ImposedVariables.values()]):
        for bc in C.getFamilyBCs(t, FamilyName):
            setBCwithImposedVariables(t, FamilyName, ImposedVariables,
                FamilyBC='BCInflowSubsonic', BCType='inj1', bc=bc, variableForInterpolation=variableForInterpolation)
    else:
        setBCwithImposedVariables(t, FamilyName, ImposedVariables,
            FamilyBC='BCInflowSubsonic', BCType='inj1', bc=bc, variableForInterpolation=variableForInterpolation)

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
            VelocityUnitVectorX, VelocityUnitVectorY, VelocityUnitVectorZ

    See also
    --------

    setBC_inj1, setBC_inj1_interpFromFile, setBC_injmfr1

    '''

    PressureStagnation    = kwargs.get('PressureStagnation', ReferenceValues['PressureStagnation'])
    TemperatureStagnation = kwargs.get('TemperatureStagnation', ReferenceValues['TemperatureStagnation'])
    EnthalpyStagnation    = kwargs.get('EnthalpyStagnation', FluidProperties['cp'] * TemperatureStagnation)
    VelocityUnitVectorX   = kwargs.get('VelocityUnitVectorX', ReferenceValues['DragDirection'][0])
    VelocityUnitVectorY   = kwargs.get('VelocityUnitVectorY', ReferenceValues['DragDirection'][1])
    VelocityUnitVectorZ   = kwargs.get('VelocityUnitVectorZ', ReferenceValues['DragDirection'][2])
    variableForInterpolation = kwargs.get('variableForInterpolation', 'ChannelHeight')

    # Get turbulent variables names and values
    turbVars = ReferenceValues['FieldsTurbulence']
    turbVars = [var.replace('Density', '') for var in turbVars]
    turbValues = [val/ReferenceValues['Density'] for val in ReferenceValues['ReferenceStateTurbulence']]
    turbDict = dict(zip(turbVars, turbValues))

    ImposedVariables = dict(
        PressureStagnation  = PressureStagnation,
        EnthalpyStagnation  = EnthalpyStagnation,
        VelocityUnitVectorX = VelocityUnitVectorX,
        VelocityUnitVectorY = VelocityUnitVectorY,
        VelocityUnitVectorZ = VelocityUnitVectorZ,
        **turbDict
        )

    setBC_inj1(t, FamilyName, ImposedVariables, variableForInterpolation=variableForInterpolation)

def setBC_inj1_interpFromFile(t, ReferenceValues, FamilyName, filename, fileformat=None):
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

    Field variables will be extrapolated on the BCs attached to the family
    **FamilyName**, except if:

    * the file can be converted in a PyTree

    * with zone names like: ``<ZONE>\<BC>``, as obtained from function
      :py:func:`Converter.PyTree.extractBCOfName`

    * and all zone names and BC names are consistent with the current tree **t**

    In that case, field variables are just read in **filename** and written in
    BCs of **t**.

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
    turbVars = ReferenceValues['FieldsTurbulence']
    turbVars = [var.replace('Density', '') for var in turbVars]
    var2interp += turbVars

    donor_tree = C.convertFile2PyTree(filename, format=fileformat)
    inlet_BC_nodes = C.extractBCOfName(t, 'FamilySpecified:{0}'.format(FamilyName))
    I._adaptZoneNamesForSlash(inlet_BC_nodes)
    for w in inlet_BC_nodes:
        bcLongName = I.getName(w)  # from C.extractBCOfName: <zone>\<bc>
        zname, wname = bcLongName.split('\\')
        znode = I.getNodeFromNameAndType(t, zname, 'Zone_t')
        bcnode = I.getNodeFromNameAndType(znode, wname, 'BC_t')

        print('Interpolate Inflow condition on BC {}...'.format(bcLongName))
        I._rmNodesByType(w, 'FlowSolution_t')
        donor_BC = P.extractMesh(donor_tree, w, mode='accurate')

        ImposedVariables = dict()
        for var in var2interp:
            varNode = I.getNodeFromName(donor_BC, var)
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
            TurbulenceLevel, Viscosity_EddyMolecularRatio

    See also
    --------

    setBC_inj1, setBC_inj1_interpFromFile

    '''
    Surface = kwargs.get('Surface', None)
    if not Surface:
        # Compute surface of the inflow BC
        zones = C.extractBCOfName(t, 'FamilySpecified:'+FamilyName)
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

    TurbulenceLevel = kwargs.get('TurbulenceLevel', None)
    Viscosity_EddyMolecularRatio = kwargs.get('Viscosity_EddyMolecularRatio', None)
    if TurbulenceLevel and Viscosity_EddyMolecularRatio:
        ReferenceValuesForTurbulence = computeReferenceValues(FluidProperties,
                MassFlow, ReferenceValues['PressureStagnation'],
                TemperatureStagnation, Surface,
                TurbulenceLevel=TurbulenceLevel,
                Viscosity_EddyMolecularRatio=Viscosity_EddyMolecularRatio,
                TurbulenceModel=ReferenceValues['TurbulenceModel'])
    else:
        ReferenceValuesForTurbulence = ReferenceValues

    # Get turbulent variables names and values
    turbVars = ReferenceValues['FieldsTurbulence']
    turbVars = [var.replace('Density', '') for var in turbVars]
    turbValues = [val/ReferenceValues['Density'] for val in ReferenceValues['ReferenceStateTurbulence']]
    turbDict = dict(zip(turbVars, turbValues))

    # Convert names to inj_tur1 and (if needed) inj_tur2
    if 'TurbulentSANuTilde' in turbDict:
        turbDict = dict(inj_tur1=turbDict['TurbulentSANuTilde'])
    else:
        turbDict['inj_tur1'] = turbDict['TurbulentEnergyKinetic']
        turbDict.pop('TurbulentEnergyKinetic')
        inj_tur2 = [var for var in turbDict if var != 'inj_tur1']
        assert len(inj_tur2) == 1, \
            'Turbulent models with more than 2 equations are not supported yet'
        inj_tur2 = inj_tur2[0]
        turbDict['inj_tur2'] = turbDict[inj_tur2]
        turbDict.pop(inj_tur2)

    ImposedVariables = dict(
        surf_massflow       = SurfacicMassFlow,
        stagnation_enthalpy = EnthalpyStagnation,
        txv                 = VelocityUnitVectorX,
        tyv                 = VelocityUnitVectorY,
        tzv                 = VelocityUnitVectorZ,
        **turbDict
        )

    setBCwithImposedVariables(t, FamilyName, ImposedVariables,
        FamilyBC='BCInflowSubsonic', BCType='injmfr1')

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

def setBCwithImposedVariables(t, FamilyName, ImposedVariables, FamilyBC, BCType,
    bc=None, BCDataSetName='BCDataSet#Init', BCDataName='DirichletData', variableForInterpolation='ChannelHeight'):
    '''
    Generic function to impose a Boundary Condition ``inj1``. The following
    functions are more specific:

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        ImposedVarvariableForInterpolation : str
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
            Must be 'ChannelHeight' (default value) or 'Radius'.

    See also
    --------

    setBC_inj1, setBC_outpres, setBC_outmfr2

    '''
    FamilyNode = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    I._rmNodesByName(FamilyNode, '.Solver#BC')
    I._rmNodesByType(FamilyNode, 'FamilyBC_t')
    I.newFamilyBC(value=FamilyBC, parent=FamilyNode)

    if all([np.ndim(v)==0 and not callable(v) for v in ImposedVariables.values()]):
        checkVariables(ImposedVariables)
        ImposedVariables = translateVariablesFromCGNS2Elsa(ImposedVariables)
        J.set(FamilyNode, '.Solver#BC', type=BCType, **ImposedVariables)
    else:
        assert bc is not None
        J.set(bc, '.Solver#BC', type=BCType)

        zone = I.getParentFromType(t, bc, 'Zone_t') 
        if variableForInterpolation in ['Radius', 'radius']:
            radius, theta = J.getRadiusTheta(zone)
        elif variableForInterpolation == 'ChannelHeight':
            radius = I.getValue(I.getNodeFromName(zone, 'ChannelHeight'))
        else:
            raise ValueError('varForInterpolation must be Radius or ChannelHeight')

        PointRangeNode = I.getNodeFromType(bc, 'IndexRange_t')
        if PointRangeNode:
            # Structured mesh
            PointRange = I.getValue(PointRangeNode)
            bc_shape = PointRange[:, 1] - PointRange[:, 0]
            if bc_shape[0] == 0:
                bc_shape = (bc_shape[1], bc_shape[2])
                radius = radius[PointRange[0, 0]-1,
                                PointRange[1, 0]-1:PointRange[1, 1]-1, 
                                PointRange[2, 0]-1:PointRange[2, 1]-1]
            elif bc_shape[1] == 0:
                bc_shape = (bc_shape[0], bc_shape[2])
                radius = radius[PointRange[0, 0]-1:PointRange[0, 1]-1,
                                PointRange[1, 0]-1, 
                                PointRange[2, 0]-1:PointRange[2, 1]-1]
            elif bc_shape[2] == 0:
                bc_shape = (bc_shape[0], bc_shape[1])
                radius = radius[PointRange[0, 0]-1:PointRange[0, 1]-1,
                                PointRange[1, 0]-1:PointRange[1, 1]-1,
                                PointRange[2, 0]-1]
            else:
                raise ValueError('Wrong BC shape {} in {}'.format(bc_shape, I.getPath(t, bc)))
        
        else: 
            # Unstructured mesh
            PointList = I.getValue(I.getNodeFromType(bc, 'IndexArray_t'))
            bc_shape = PointList.size
            radius = radius[PointList-1]

        for var, value in ImposedVariables.items():
            if callable(value):
                ImposedVariables[var] = value(radius) 
            elif np.ndim(value)==0:
                # scalar value --> uniform data
                ImposedVariables[var] = value * np.ones(radius.shape)
            assert ImposedVariables[var].shape == bc_shape, \
                'Wrong shape for variable {}: {} (shape {} for {})'.format(
                    var, ImposedVariables[var].shape, bc_shape, I.getPath(t, bc))
        
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
        TurbulentSANuTilde       = 'inj_tur1',
        TurbulentEnergyKinetic   = 'inj_tur1',
        TurbulentDissipationRate = 'inj_tur2',
        TurbulentDissipation     = 'inj_tur2',
        TurbulentLengthScale     = 'inj_tur2',
        VelocityUnitVectorX      = 'txv',
        VelocityUnitVectorY      = 'tyv',
        VelocityUnitVectorZ      = 'tzv',
    )

    elsAVariables = CGNS2ElsaDict.values()

    if isinstance(Variables, dict):
        NewVariables = dict()
        for var, value in Variables.items():
            if var in elsAVariables:
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

    import etc.transform.__future__ as trf

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

    import etc.transform.__future__ as trf

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

    import etc.transform.__future__ as trf

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

    import etc.transform.__future__ as trf

    t = trf.defineBCStageFromBC(t, left)
    t = trf.defineBCStageFromBC(t, right)

    t, stage = trf.newStageRedHybFromFamily(
        t, left, right, stage_ref_time=stage_ref_time)
    stage.create()

    for gc in I.getNodesFromType(t, 'GridConnectivity_t'):
        I._rmNodesByType(gc, 'FamilyBC_t')


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

    import etc.transform.__future__ as trf

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
                         valve_ref_mflow, valve_relax=valve_relax)
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

    import etc.transform.__future__ as trf

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
                 valve_ref_mflow, valve_relax=valve_relax)
    bc.dirorder = -1
    radius_filename = "state_radius_{}_{}.plt".format(FamilyName, nbband)
    radius = bc.repartition(filename=radius_filename, fileformat="bin_tp")
    radius.compute(t, nbband=nbband, c=c)
    radius.write()
    bc.create()


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

        perfo : :py:class:`dict` of :py:class:`list`
            dictionary with performance of **monitoredRow** for completed
            simulations. It contains the following keys:

            * MassFlow

            * PressureStagnationRatio

            * EfficiencyIsentropic

            * RotationSpeed

            * Throttle

            Each list corresponds to one rotation speed. Each sub-list
            corresponds to the different operating points on a iso-speed line.

    '''
    from . import Coprocess as CO

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

    perfo = dict(
        MassFlow = [],
        PressureStagnationRatio = [],
        EfficiencyIsentropic = [],
        RotationSpeed = [],
        Throttle = []
    )
    lines = ['']

    JobNames = [getCaseLabel(config, Throttle[0], r).split('_')[-1] for r in RotationSpeed]
    for idSpeed, rotationSpeed in enumerate(RotationSpeed):

        lines.append(TagStrFmt.format('JobName |')+''.join([ColStrFmt.format(JobNames[idSpeed])] + [ColStrFmt.format('') for j in range(nCol-1)]))
        lines.append(TagStrFmt.format('RotationSpeed |')+''.join([ColFmt.format(rotationSpeed)] + [ColStrFmt.format('') for j in range(nCol-1)]))
        lines.append(TagStrFmt.format(' |')+''.join([ColStrFmt.format(''), ColStrFmt.format('MFR'), ColStrFmt.format('RPI'), ColStrFmt.format('ETA')]))
        lines.append(TagStrFmt.format('Throttle |')+''.join(['_' for m in range(NcolMax-FirstCol)]))
        MFR = []
        RPI = []
        ETA = []
        ROT = []
        THR = []

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
                lastarrays = JM.getCaseArrays(config, CASE_LABEL,
                                        basename='PERFOS_{}'.format(monitoredRow))
                MFR.append(lastarrays['MassFlowIn'])
                RPI.append(lastarrays['PressureStagnationRatio'])
                ETA.append(lastarrays['EfficiencyIsentropic'])
                ROT.append(rotationSpeed)
                THR.append(throttle)
                msg += ''.join([ColFmt.format(MFR[-1]), ColFmt.format(RPI[-1]), ColFmt.format(ETA[-1])])
            else:
                msg += ''.join([ColStrFmt.format('') for n in range(nCol-1)])
            Line += msg
            lines.append(Line)

        lines.append('')
        perfo['MassFlow'].append(MFR)
        perfo['PressureStagnationRatio'].append(RPI)
        perfo['EfficiencyIsentropic'].append(ETA)
        perfo['RotationSpeed'].append(ROT)
        perfo['Throttle'].append(THR)

    for line in lines: print(line)

    return perfo

def getPostprocessQuantities(DIRECTORY_WORK, basename, useLocalConfig=False):
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

    Returns
    -------

        perfo : :py:class:`dict` of :py:class:`list`
            dictionary with data contained in the base **baseName** for completed
            simulations. 

            Each list corresponds to one rotation speed. Each sub-list
            corresponds to the different operating points on a iso-speed line.

    '''
    config = JM.getJobsConfiguration(DIRECTORY_WORK, useLocalConfig)
    Throttle = np.array(sorted(list(set([float(case['CASE_LABEL'].split('_')[0]) for case in config.JobsQueues]))))
    RotationSpeed = np.array(sorted(list(set([case['TurboConfiguration']['ShaftRotationSpeed'] for case in config.JobsQueues]))))

    def getCaseLabel(config, throttle, rotSpeed):
        for case in config.JobsQueues:
            if np.isclose(float(case['CASE_LABEL'].split('_')[0]), throttle) and \
                np.isclose(case['TurboConfiguration']['ShaftRotationSpeed'], rotSpeed):

                return case['CASE_LABEL']

    perfo = dict()

    for idSpeed, rotationSpeed in enumerate(RotationSpeed):
        perfoOnCarac = dict(RotationSpeed=[], Throttle=[])

        for idThrottle, throttle in enumerate(Throttle):
            CASE_LABEL = getCaseLabel(config, throttle, rotationSpeed)
            status = JM.statusOfCase(config, CASE_LABEL)

            if status == 'COMPLETED':
                lastarrays = JM.getCaseArrays(config, CASE_LABEL, basename=basename)
                for key, value in lastarrays.items():
                    if idThrottle == 0:
                        perfoOnCarac[key] = [value]
                    else:
                        perfoOnCarac[key].append(value)
                perfoOnCarac['RotationSpeed'].append(rotationSpeed)
                perfoOnCarac['Throttle'].append(throttle)

        for key, value in perfoOnCarac.items():
            if idSpeed == 0:
                perfo[key] = [value]
            else:
                perfo[key].append(value)

    return perfo



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
        mask = C.convertFile2PyTree('mask.cgns')

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
        for (beta, paramName) in [(beta1, 'FlowAngleAtRoot'), (beta2, 'FlowAngleAtTip')]:
            if beta * omega < 0:
                MSG=f'WARNING: {paramName} ({beta} deg) has not the same sign that the rotation speed in {row} ({omega} rad/s).\n'
                MSG += '        Double check it is not a mistake.'
                print(J.WARN + MSG + J.ENDC)
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
                                config='annular', 
                                lin_axis='XY',
                                RowType='compressor'):
    '''
    Perform a series of classical postprocessings for a turbomachinery case : 

    #. Compute extra variables, in relative and absolute frames of reference

    #. Compute averaged values for all iso-X planes (results are in the `.Average` node), and
       compare inlet and outlet planes for each row if available, to get row performance (total 
       pressure ratio, isentropic efficiency, etc) (results are in the `.Average#ComparisonXX` of
       the inlet plane, `XX` being the numerotation starting at `01`)

    #. Compute radial profiles for all iso-X planes (results are in the `.RadialProfile` node), and
       compare inlet and outlet planes for each row if available, to get row performance (total 
       pressure ratio, isentropic efficiency, etc) (results are in the `.RadialProfile#ComparisonXX` of
       the inlet plane, `XX` being the numerotation starting at `01`)

    #. Compute isentropic Mach number on blades, slicing at constant height, for all values of height 
       already extracted as iso-surfaces. Results are in the `.Iso_H_XX` nodes.

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
        
        config : str
            see :py:func:`MOLA.Postprocess.compute1DRadialProfiles`

        lin_axis : str
            see :py:func:`MOLA.Postprocess.compute1DRadialProfiles`

        RowType : str
            see parameter 'config' of :py:func:`MOLA.Postprocess.compareRadialProfilesPlane2Plane`
        
    '''
    import Converter.Mpi as Cmpi
    import MOLA.PostprocessTurbo as Post
    import turbo.user as TUS

    Post.setup = J.load_source('setup', 'setup.py')

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
    Post.computeVariablesOnIsosurface(surfaces, allVariables)
    Post.compute0DPerformances(surfaces, variablesByAverage)
    if computeRadialProfiles: 
        Post.compute1DRadialProfiles(
            surfaces, variablesByAverage, config=config, lin_axis=lin_axis)
    # Post.computeVariablesOnBladeProfiles(surfaces, hList='all')
    #______________________________________________________________________________#

    if Cmpi.rank == 0:
        Post.comparePerfoPlane2Plane(surfaces, var4comp_perf, stages)
        if computeRadialProfiles: 
            Post.compareRadialProfilesPlane2Plane(
                surfaces, var4comp_repart, stages, config=RowType)

    Post.cleanSurfaces(surfaces, var2keep=var2keep)
