'''
MOLA - WorkflowCompressor.py

WORKFLOW COMPRESSOR

Collection of functions designed for Workflow Compressor

File history:
31/08/2021 - T. Bontemps - Creation
'''

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

from . import InternalShortcuts as J
from . import Preprocess        as PRE
from . import JobManager        as JM

try:
    from . import ParametrizeChannelHeight as ParamHeight
except ImportError:
    MSG = 'Fail to import ParametrizeChannelHeight: function parametrizeChannelHeight is unavailable'.format(__name__)
    print(J.WARN + MSG + J.ENDC)
    ParamHeight = None

try:
    from . import WorkflowCompressorETC as ETC
except ImportError:
    MSG = 'Fail to import ETC module: Some functions of {} are unavailable'.format(__name__)
    print(J.WARN + MSG + J.ENDC)
    ETC = None


def checkDependencies():
    '''
    Make a series of functional tests in order to determine if the user
    environment is correctly set for using the Workflow Compressor
    '''
    JM.checkDependencies()

    print('Checking ETC...')
    if ETC is None:
        MSG = 'Fail to import ETC module: Some functions of {} are unavailable'.format(__name__)
        print(J.FAIL + MSG + J.ENDC)
    else:
        print(J.GREEN+'ETC module is available'+J.ENDC)

    print('Checking MOLA.ParametrizeChannelHeight...')
    if ParamHeight is None:
        MSG = 'Fail to import MOLA.ParametrizeChannelHeight module: Some functions of {} are unavailable'.format(__name__)
        print(J.FAIL + MSG + J.ENDC)
    else:
        print(J.GREEN+'MOLA.ParametrizeChannelHeight module is available'+J.ENDC)

    print('\nVERIFICATIONS TERMINATED')


def prepareMesh4ElsA(mesh, InputMeshes=None, NProcs=None, ProcPointsLoad=100000,
                    duplicationInfos={}, blocksToRename={}, SplitBlocks=False,
                    scale=1., rotation='fromAG5', PeriodicTranslation=None):
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

        NProcs : int
            If a positive integer is provided, then the
            distribution of the tree (and eventually the splitting) will be done in
            order to satisfy a total number of processors provided by this value.
            If not provided (:py:obj:`None`) then the number of procs is automatically
            determined using as information **ProcPointsLoad** variable.

        ProcPointsLoad : int
            this is the desired number of grid points
            attributed to each processor. If **SplitBlocks** = :py:obj:`True`, then it is used to
            split zones that have more points than **ProcPointsLoad**. If
            **NProcs** = :py:obj:`None` , then **ProcPointsLoad** is used to determine
            the **NProcs** to be used.

        duplicationInfos : dict
            User-provided data related to domain duplication.
            Each key corresponds to a row FamilyName.
            The associated element is a dictionary with the following parameters:

                * NumberOfBlades: number of blades in the row (in reality)

                * NumberOfDuplications: number of duplications to make of the
                  input row domain.

                * MergeBlocks: boolean, if True the duplicated blocks are merged.

        blocksToRename : dict
            Each key corresponds to the name of a zone to modify, and the associated
            value is the new name to give.

        SplitBlocks : bool
            if :py:obj:`False`, do not split and distribute the mesh (use this
            option if the simulation will run with PyPart).

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

    Returns
    -------

        t : PyTree
            the pre-processed mesh tree (usually saved as ``mesh.cgns``)

            .. important:: This tree is **NOT** ready for elsA computation yet !
                The user shall employ function :py:func:`prepareMainCGNS4ElsA`
                as next step
    '''
    if isinstance(mesh,str):
        t = C.convertFile2PyTree(mesh)
    elif I.isTopTree(mesh):
        t = mesh
    else:
        raise ValueError('parameter mesh must be either a filename or a PyTree')

    if InputMeshes is None:
        InputMeshes = generateInputMeshesFromAG5(t, SplitBlocks=SplitBlocks,
            scale=scale, rotation=rotation, PeriodicTranslation=PeriodicTranslation)

    PRE.checkFamiliesInZonesAndBC(t)
    t = cleanMeshFromAutogrid(t, basename=InputMeshes[0]['baseName'], blocksToRename=blocksToRename)
    PRE.transform(t, InputMeshes)
    for row, rowParams in duplicationInfos.items():
        try: MergeBlocks = rowParams['MergeBlocks']
        except: MergeBlocks = False
        duplicate(t, row, rowParams['NumberOfBlades'],
                nDupli=rowParams['NumberOfDuplications'], merge=MergeBlocks)
    if not any([InputMesh['SplitBlocks'] for InputMesh in InputMeshes]):
        t = PRE.connectMesh(t, InputMeshes)
    else:
        t = splitAndDistribute(t, InputMeshes, NProcs=NProcs,
                                    ProcPointsLoad=ProcPointsLoad)
    # WARNING: Names of BC_t nodes must be unique to use PyPart on globborders
    for l in [2,3,4]: I._correctPyTree(t, level=l)
    PRE.adapt2elsA(t, InputMeshes)
    J.checkEmptyBC(t)

    return t

def prepareMainCGNS4ElsA(mesh='mesh.cgns', ReferenceValuesParams={},
        NumericalParams={}, TurboConfiguration={}, Extractions={}, BoundaryConditions={},
        BodyForceInputData=[], writeOutputFields=True, bladeFamilyNames=['Blade'],
        Initialization={'method':'uniform'}):
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

        Initialization : dict
            dictionary defining the type of initialization, using the key
            **method**. See documentation of :func:`MOLA.Preprocess.initializeFlowSolution`

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


    if hasBCOverlap: PRE.addFieldExtraction('ChimeraCellType')
    if BodyForceInputData: PRE.addFieldExtraction('Temperature')

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
        NProc = max([I.getNodeFromName(z,'proc')[1][0][0] for z in I.getZones(t)])+1
        ReferenceValues['NProc'] = int(NProc)
        ReferenceValuesParams['NProc'] = int(NProc)
        Splitter = None
    else:
        ReferenceValues['NProc'] = 0
        Splitter = 'PyPart'
    elsAkeysCFD      = PRE.getElsAkeysCFD(nomatch_linem_tol=1e-6)
    elsAkeysModel    = PRE.getElsAkeysModel(FluidProperties, ReferenceValues)
    if BodyForceInputData: NumericalParams['useBodyForce'] = True

    if not 'NumericalScheme' in NumericalParams:
        NumericalParams['NumericalScheme'] = 'roe'
    elsAkeysNumerics = PRE.getElsAkeysNumerics(ReferenceValues, **NumericalParams)

    PRE.initializeFlowSolution(t, Initialization, ReferenceValues)

    if not 'PeriodicTranslation' in TurboConfiguration and \
        any([rowParams['NumberOfBladesSimulated'] > rowParams['NumberOfBladesInInitialMesh'] \
            for rowParams in TurboConfiguration['Rows'].values()]):
        t = duplicateFlowSolution(t, TurboConfiguration)

    setBoundaryConditions(t, BoundaryConditions, TurboConfiguration,
                            FluidProperties, ReferenceValues,
                            bladeFamilyNames=bladeFamilyNames)

    computeFluxCoefByRow(t, ReferenceValues, TurboConfiguration)

    if not 'PeriodicTranslation' in TurboConfiguration:
        addMonitoredRowsInExtractions(Extractions, TurboConfiguration)

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
        BCWall = ['normalvector', 'frictionvector','psta', 'bl_quantities_2d', 'yplusmeshsize'],
        BCInflow = ['convflux_ro'],
        BCOutflow = ['convflux_ro']
    )

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

    PRE.saveMainCGNSwithLinkToOutputFields(t,writeOutputFields=writeOutputFields)

    if not Splitter:
        print('REMEMBER : configuration shall be run using %s%d%s procs'%(J.CYAN,
                                                   ReferenceValues['NProc'],J.ENDC))
    else:
        print('REMEMBER : configuration shall be run using %s'%(J.CYAN + \
            Splitter + J.ENDC))

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
                                elines='shroud_hub_lines.plt'):
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
            Rmax = np.amax(yShroud)
            plt.ylim(-0.05*Rmax, 1.05*Rmax)
            plt.savefig(elines.replace('.plt', '.png'), dpi=150, bbox_inches='tight')
        except:
            pass

        # - Generation of the mask file
        m = TH.generateMaskWithChannelHeight(t, elines, 'bin_tp')
        # - Generation of the ChannelHeight field
        t = TH.computeHeightFromMask(t, m, writeMask='mask.cgns', writeMaskCart ='maskCart.cgns')

    I.__FlowSolutionNodes__ = OLD_FlowSolutionNodes
    print(J.GREEN + 'done.' + J.ENDC)
    return t

def generateInputMeshesFromAG5(mesh, SplitBlocks=False, scale=1., rotation='fromAG5', PeriodicTranslation=None):
    '''
    Generate automatically the :py:class:`list` **InputMeshes** with a default
    parametrization adapted to Autogrid 5 meshes.

    Parameters
    ----------

        mesh : :py:class:`str` or PyTree
            Name of the CGNS mesh file from Autogrid 5 or already read PyTree.

        SplitBlocks : bool
            if :py:obj:`False`, do not split and distribute the mesh (use this
            option if the simulation will run with PyPart).

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
                    Connection=[dict(type='Match', tolerance=1e-8)],
                    SplitBlocks=SplitBlocks,
                    )]
    # Set automatic periodic connections
    InputMesh = InputMeshes[0]
    if I.getNodeFromName(t, 'BladeNumber'):
        BladeNumberList = [I.getValue(bn) for bn in I.getNodesFromName(t, 'BladeNumber')]
        angles = list(set([360./float(bn) for bn in BladeNumberList]))
        for angle in angles:
            print('  angle = {:g} deg ({} blades)'.format(angle, int(360./angle)))
            InputMesh['Connection'].append(
                    dict(type='PeriodicMatch', tolerance=1e-8, rotationAngle=[angle,0.,0.])
                    )
    if PeriodicTranslation:
        print('  translation = {} m'.format(PeriodicTranslation))
        InputMesh['Connection'].append(
                dict(type='PeriodicMatch', tolerance=1e-8, translation=PeriodicTranslation)
                )

    return InputMeshes

def prepareTree(t, rowParams):
    '''
    Rename wall families, add row family to each zone and add BladeNumber
    property to row Family.

    .. attention:: There must be only one row in **t**.

    Parameters
    ----------

        t : newPyTree

        rowParams : dict
            Dictionary to rename wall families, add families to rows and add
            BladeNumber property to rows.
            Should follow the following form:

            .. code-block:: python

                rowParams = dict(
                    rowName     = <RowFamily>,
                    blade       = <BladeFamily>,
                    hub         = <HubFamily>,
                    shroud      = <ShroudFamily>,
                    BladeNumber = <Nb>,
                )

    '''
    I._renameNode(t, rowParams['hub'], '{}_HUB'.format(row))
    I._renameNode(t, rowParams['shroud'],'{}_SHROUD'.format(row))
    I._renameNode(t, rowParams['blade'],  '{}_Main_Blade'.format(row))
    if not I.getNodesFromNameAndType2(t, rowParams['rowName'], 'Family_t'):
        C._tagWithFamily(t, rowParams['rowName'], add=True)
        C._addFamily2Base(t, rowParams['rowName'])
        J.set(n, 'Periodicity', BladeNumber=rowParams['BladeNumber'])

def cleanMeshFromAutogrid(t, basename='Base#1', blocksToRename={}):
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

        blocksToRename : dict
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
        for block in blocksToRename:
            if block in name:
                newName = name.replace(block, blocksToRename[block])
                print("Zone {} is renamed: {}".format(name,newName))
                I._renameNode(tree, name, newName)
        # Delete some usual patterns in AG5
        new_name = name
        for pattern in ['_flux_1', '_flux_2', '_flux_3', '_Main_Blade']:
            new_name = new_name.replace(pattern, '')
        I._renameNode(t, name, new_name)

    # Clean Joins & Periodic Joins
    I._rmNodesByType(t, 'ZoneGridConnectivity_t')
    periodicFamilyNames = [I.getName(fam) for fam in I.getNodesFromType(t, "Family_t")
        if 'PER' in I.getName(fam)]
    for fname in periodicFamilyNames:
        # print('|- delete PeriodicBC family of name {}'.format(name))
        C._rmBCOfType(t, 'FamilySpecified:%s'%fname)
        fbc = I.getNodeFromName2(t, fname)
        I.rmNode(t, fbc)

    # Clean RS interfaces
    I._rmNodesByType(t,'InterfaceType')
    I._rmNodesByType(t,'DonorFamily')

    # Join HUB and SHROUD families
    joinFamilies(t, 'HUB')
    joinFamilies(t, 'SHROUD')
    return t

def joinFamilies(t, pattern):
    '''
    In the CGNS tree t, gather all the Families <ROW_I>_<PATTERN>_<SUFFIXE> into
    Families <ROW_I>_<PATTERN>, so as many as rows.
    Useful to join all the row_i_HUB* or (row_i_SHROUD*) together

    Parameters
    ----------

        t : PyTree
            A PyTree read by Cassiopee

        pattern : str
            The pattern used to gather CGNS families. Should be for example 'HUB' or 'SHROUD'
    '''
    fam2remove = set()
    fam2keep = set()
    # Loop on the BCs in the tree
    for bc in I.getNodesFromType(t, 'BC_t'):
        # Get BC family name
        famBC_node = I.getNodeFromType(bc, 'FamilyName_t')
        famBC = I.getValue(famBC_node)
        # Check if the pattern is present in FamilyBC name
        if pattern not in famBC:
            continue
        # Split to get the short name based on pattern
        split_fanBC = famBC.split(pattern)
        assert len(split_fanBC) == 2, 'The pattern {} is present more than once in the FamilyBC {}. It must be more selective.'.format(pattern, famBC)
        preffix, suffix = split_fanBC
        # Add the short name to the set fam2keep
        short_name = '{}{}'.format(preffix, pattern)
        fam2keep |= {short_name}
        if suffix != '':
            # Change the family name
            I.setValue(famBC_node, '{}'.format(short_name))
            fam2remove |= {famBC}

    # Remove families
    for fam in fam2remove:
        print('Remove family {}'.format(fam))
        I._rmNodesByNameAndType(t, fam, 'Family_t')

    # Check that families to keep still exist
    base = I.getNodeFromType(t,'CGNSBase_t')
    for fam in fam2keep:
        fam_node = I.getNodeFromNameAndType(t, fam, 'Family_t')
        if fam_node is None:
            print('Add family {}'.format(fam))
            I.newFamily(fam, parent=base)

def duplicate(tree, rowFamily, nBlades, nDupli=None, merge=False, axis=(1,0,0),
    verbose=1, container='FlowSolution#Init',
    vectors2rotate=[['VelocityX','VelocityY','VelocityZ'],['MomentumX','MomentumY','MomentumZ']]):
    '''
    Duplicate **nDupli** times the domain attached to the family **rowFamily**
    around the axis of rotation.

    .. warning:: This function works only for empty meshes. It can be used on a
        PyTree with FlowSolution containers, but the vectors will not be
        rotated !

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

    if I.getType(tree) == 'CGNSBase_t':
        base = tree
    else:
        base = I.getNodeFromType(tree, 'CGNSBase_t')

    check = False
    vectors = []
    for vec in vectors2rotate:
        vectors.append(vec)
        vectors.append(['centers:'+v for v in vec])

    for zone in I.getZones(base):
        zone_name = I.getName(zone)
        zone_family = I.getValue(I.getNodeFromName1(zone, 'FamilyName'))
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
                PRE.autoMergeBCs(tree)

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
    Surface = abs(P.integNorm(SurfaceTree, var='ones')[0][0])
    # Compute deltaTheta
    deltaTheta = 2* Surface / (Rmax**2 - Rmin**2)
    # Compute number of blades in the mesh
    Nb = NumberOfBlades * deltaTheta / (2*np.pi)
    Nb = int(np.round(Nb))
    print('Number of blades in initial mesh for {}: {}'.format(FamilyName, Nb))
    return Nb

def splitAndDistribute(t, InputMeshes, NProcs, ProcPointsLoad):
    '''
    Split a PyTree **t** using the desired proc points load **ProcPointsLoad**.
    Distribute the PyTree **t** using a user-provided **NProcs**. If **NProcs**
    is not provided, then it is automatically computed.

    Returns a new split and distributed PyTree.

    .. note:: only **InputMeshes** where ``'SplitBlocks':True`` are split.

    Parameters
    ----------

        t : PyTree
            assembled tree

        InputMeshes : :py:class:`list` of :py:class:`dict`
            user-provided preprocessing
            instructions as described in :py:func:`prepareMesh4ElsA` doc

        NProcs : int
            If a positive integer is provided, then the
            distribution of the tree (and eventually the splitting) will be done in
            order to satisfy a total number of processors provided by this value.
            If not provided (:py:obj:`None`) then the number of procs is automatically
            determined using as information **ProcPointsLoad** variable.

        ProcPointsLoad : int
            this is the desired number of grid points
            attributed to each processor. If **SplitBlocks** = :py:obj:`True`, then it is used to
            split zones that have more points than **ProcPointsLoad**. If
            **NProcs** = :py:obj:`None` , then **ProcPointsLoad** is used to determine
            the **NProcs** to be used.

    Returns
    -------

        t : PyTree
            new distributed *(and possibly split)* tree

    '''
    if InputMeshes[0]['SplitBlocks']:
        t = T.splitNParts(t, NProcs, dirs=[1,2,3], recoverBC=True)
        for l in [2,3,4]: I._correctPyTree(t, level=l)
        t = PRE.connectMesh(t, InputMeshes)
    #
    InputMeshesNoSplit = []
    for InputMesh in InputMeshes:
        InputMeshNoSplit = dict()
        for meshInfo in InputMesh:
            if meshInfo == 'SplitBlocks':
                InputMeshNoSplit['SplitBlocks'] = False
            else:
                InputMeshNoSplit[meshInfo] = InputMesh[meshInfo]
        InputMeshesNoSplit.append(InputMeshNoSplit)
    # Just to distribute zones on procs
    t = PRE.splitAndDistribute(t, InputMeshesNoSplit, NProcs=NProcs, ProcPointsLoad=ProcPointsLoad)
    return t

def computeReferenceValues(FluidProperties, MassFlow, PressureStagnation,
        TemperatureStagnation, Surface, TurbulenceLevel=0.001,
        Viscosity_EddyMolecularRatio=0.1, TurbulenceModel='Wilcox2006-klim',
        TurbulenceCutoff=1e-8, TransitionMode=None, CoprocessOptions={},
        Length=1.0, TorqueOrigin=[0., 0., 0.],
        FieldsAdditionalExtractions=['ViscosityMolecular', 'Viscosity_EddyMolecularRatio', 'Pressure', 'Temperature', 'PressureStagnation', 'TemperatureStagnation', 'Mach', 'Entropy'],
        AngleOfAttackDeg=0.,
        YawAxis=[0.,0.,1.],
        PitchAxis=[0.,1.,0.]):
    '''
    This function is the Compressor's equivalent of :func:`MOLA.Preprocess.computeReferenceValues`.
    The main difference is that in this case reference values are set through
    ``MassFlow``, total Pressure ``PressureStagnation``, total Temperature
    ``TemperatureStagnation`` and ``Surface``.

    Please, refer to :func:`MOLA.Preprocess.computeReferenceValues` doc for more details.
    '''
    # Fluid properties local shortcuts
    Gamma   = FluidProperties['Gamma']
    IdealGasConstant = FluidProperties['IdealGasConstant']
    cv      = FluidProperties['cv']
    cp      = FluidProperties['cp']

    # Compute variables
    Mach  = machFromMassFlow(MassFlow, Surface, Pt=PressureStagnation,
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

    if not 'AveragingIterations' in CoprocessOptions:
        CoprocessOptions['AveragingIterations'] = 1000

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
        FieldsAdditionalExtractions=FieldsAdditionalExtractions)

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
            rowParams = TurboConfiguration['Rows'][row]
            fluxcoeff = rowParams['NumberOfBlades'] / float(rowParams['NumberOfBladesSimulated'])
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
    Surface = abs(P.integNorm(SurfaceTree, var='ones')[0][0])
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

                * type :
                  BC type among the following:

                  * Farfield

                  * InflowStagnation

                  * OutflowPressure

                  * OutflowMassFlow

                  * OutflowRadialEquilibrium

                  * MixingPlane

                  * UnsteadyRotorStatorInterface

                  * WallViscous

                  * WallInviscid

                  * SymmetryPlane

                  elsA names are also available (``nref``, ``inj1``,
                  ``outpres``, ``outmfr2``, ``outradeq``, ``stage_mxpl``,
                  ``stage_red``, ``walladia``, ``wallslip``, ``sym``)

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

    setBC_Walls, setBC_walladia, setBC_wallslip, setBC_sym,
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
    :py:class:`dict`.

    >>> dict(type='InflowStagnation', option='file', FamilyName='row_1_INFLOW', filename='inflow.cgns')

    It defines an inflow condition imposing stagnation quantities ('inj1' in
    *elsA*) interpolating a 2D map written in the given file.


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
        OutflowPressure              = 'outpres',
        OutflowMassFlow              = 'outmfr2',
        OutflowRadialEquilibrium     = 'outradeq',
        MixingPlane                  = 'stage_mxpl',
        UnsteadyRotorStatorInterface = 'stage_red',
        WallViscous                  = 'walladia',
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
                print(J.CYAN + 'set BC inj1 on ' + BCparam['FamilyName'] + J.ENDC)
                setBC_inj1(t, **BCkwargs)

            elif BCparam['option'] == 'uniform':
                print(J.CYAN + 'set BC inj1 (uniform) on ' + BCparam['FamilyName'] + J.ENDC)
                setBC_inj1_uniform(t, FluidProperties, ReferenceValues, **BCkwargs)

            elif BCparam['option'] == 'file':
                print('{}set BC inj1 (from file {}) on {}{}'.format(J.CYAN,
                    BCparam['filename'], BCparam['FamilyName'], J.ENDC))
                setBC_inj1_interpFromFile(t, ReferenceValues, **BCkwargs)

        elif BCparam['type'] == 'outpres':
            print(J.CYAN + 'set BC outpres on ' + BCparam['FamilyName'] + J.ENDC)
            setBC_outpres(t, **BCkwargs)

        elif BCparam['type'] == 'outmfr2':
            print(J.CYAN + 'set BC outmfr2 on ' + BCparam['FamilyName'] + J.ENDC)
            BCkwargs['ReferenceValues'] = ReferenceValues
            setBC_outmfr2(t, **BCkwargs)

        elif BCparam['type'] == 'outradeq':
            print(J.CYAN + 'set BC outradeq on ' + BCparam['FamilyName'] + J.ENDC)
            BCkwargs['ReferenceValues'] = ReferenceValues
            BCkwargs['TurboConfiguration'] = TurboConfiguration
            setBC_outradeq(t, **BCkwargs)

        elif BCparam['type'] == 'outradeqhyb':
            print(J.CYAN + 'set BC outradeqhyb on ' + BCparam['FamilyName'] + J.ENDC)
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

        else:
            raise AttributeError('BC type %s not implemented'%BCparam['type'])

def setBC_Walls(t, TurboConfiguration,
                    bladeFamilyNames=['BLADE', 'AUBE'],
                    hubFamilyNames=['HUB', 'MOYEU'],
                    shroudFamilyNames=['SHROUD', 'CARTER']):
    '''
    Set all the wall boundary conditions in a turbomachinery context, by making
    the following operations:

        * set the rotation speed for all families related to row domains. It is
          defined in:

            >>> TurboConfiguration['Rows'][rowName]['RotationSpeed'] = float

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

    # Add info on row movement (.Solver#Motion)
    for row, rowParams in TurboConfiguration['Rows'].items():
        famNode = I.getNodeFromNameAndType(t, row, 'Family_t')
        J.set(famNode, '.Solver#Motion',
                motion='mobile',
                omega=rowParams['RotationSpeed'],
                axis_pnt_x=0., axis_pnt_y=0., axis_pnt_z=0.,
                axis_vct_x=1., axis_vct_y=0., axis_vct_z=0.)

    # BLADES
    for blade_family in bladeFamilyNames:
        for famNode in I.getNodesFromNameAndType(t, '*{}*'.format(blade_family), 'Family_t'):
            famName = I.getName(famNode)
            row_omega = None
            for row, rowParams in TurboConfiguration['Rows'].items():
                if row in famName:
                    row_omega = rowParams['RotationSpeed']
                    break
            assert row_omega is not None, 'Cannot determine associated row for family {}. '.format(famName)

            I.newFamilyBC(value='BCWallViscous', parent=famNode)
            J.set(famNode, '.Solver#BC',
                    type='walladia',
                    data_frame='user',
                    omega=row_omega,
                    axis_pnt_x=0., axis_pnt_y=0., axis_pnt_z=0.,
                    axis_vct_x=1., axis_vct_y=0., axis_vct_z=0.)

    # HUB
    for hub_family in hubFamilyNames:
        for famNode in I.getNodesFromNameAndType(t, '*{}*'.format(hub_family), 'Family_t'):
            famName = I.getName(famNode)
            I.newFamilyBC(value='BCWallViscous', parent=famNode)
            J.set(famNode, '.Solver#BC',
                    type='walladia',
                    data_frame='user',
                    omega=0.,
                    axis_pnt_x=0., axis_pnt_y=0., axis_pnt_z=0.,
                    axis_vct_x=1., axis_vct_y=0., axis_vct_z=0.)

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
            I.newFamilyBC(value='BCWallViscous', parent=famNode)
            J.set(famNode, '.Solver#BC',
                    type='walladia',
                    data_frame='user',
                    omega=0.,
                    axis_pnt_x=0., axis_pnt_y=0., axis_pnt_z=0.,
                    axis_vct_x=1., axis_vct_y=0., axis_vct_z=0.)

def setBC_walladia(t, FamilyName):
    '''
    Set a viscous wall boundary condition.

    .. note:: see `elsA Tutorial about wall conditions <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#wall-conditions/>`_

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

    '''
    wall = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    I._rmNodesByType(wall, 'FamilyBC_t')
    I.newFamilyBC(value='BCWallViscous', parent=wall)

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
    I._rmNodesByType(wall, 'FamilyBC_t')
    I.newFamilyBC(value='BCWallInviscid', parent=wall)

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
    I._rmNodesByType(farfield, 'FamilyBC_t')
    I.newFamilyBC(value='BCFarfield', parent=farfield)

def setBC_inj1(t, FamilyName, ImposedVariables, bc=None):
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

    See also
    --------

    setBC_inj1_uniform, setBC_inj1_interpFromFile
    '''
    setBCwithImposedVariables(t, FamilyName, ImposedVariables,
        FamilyBC='BCInflowSubsonic', BCType='inj1', bc=bc)

def setBC_inj1_uniform(t, FluidProperties, ReferenceValues, FamilyName):
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

    See also
    --------

    setBC_inj1, setBC_inj1_interpFromFile

    '''

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
        PressureStagnation  = ReferenceValues['PressureStagnation'],
        stagnation_enthalpy = FluidProperties['cp'] * ReferenceValues['TemperatureStagnation'],
        txv                 = ReferenceValues['DragDirection'][0],
        tyv                 = ReferenceValues['DragDirection'][1],
        tzv                 = ReferenceValues['DragDirection'][2],
        **turbDict
        )

    setBC_inj1(t, FamilyName, ImposedVariables)

def setBC_inj1_interpFromFile(t, ReferenceValues, FamilyName, filename, fileformat=None):
    '''
    Set a Boundary Condition ``inj1`` using the field map in the file
    **filename**. It is expected to be a surface with the following variables
    defined at cell centers:

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

def setBC_outpres(t, FamilyName, Pressure, bc=None):
    '''
    Impose a Boundary Condition ``outpres``. The following
    functions are more specific:

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
    '''
    if isinstance(Pressure, dict):
        assert 'Pressure' in Pressure or 'pressure' in Pressure
        assert len(Pressure.keys() == 1)
        ImposedVariables = Pressure
    else:
        ImposedVariables = dict(Pressure=Pressure)
    setBCwithImposedVariables(t, FamilyName, ImposedVariables,
        FamilyBC='BCOutflowSubsonic', BCType='outpres', bc=bc)

def setBC_outmfr2(t, FamilyName, MassFlow=None, groupmassflow=1, ReferenceValues=None):
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

    '''
    if MassFlow is None and ReferenceValues is not None:
        bc = C.getFamilyBCs(t, FamilyName)[0]
        zone = I.getParentFromType(t, bc, 'Zone_t')
        row = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
        rowParams = TurboConfiguration['Rows'][row]
        fluxcoeff = rowParams['NumberOfBlades'] / float(rowParams['NumberOfBladesSimulated'])
        MassFlow = ReferenceValues['MassFlow'] / fluxcoeff

    ImposedVariables = dict(globalmassflow=MassFlow, groupmassflow=groupmassflow)

    setBCwithImposedVariables(t, FamilyName, ImposedVariables,
        FamilyBC='BCOutflowSubsonic', BCType='outmfr2', bc=bc)

def setBCwithImposedVariables(t, FamilyName, ImposedVariables, FamilyBC, BCType,
    bc=None, BCDataSetName='BCDataSet#Init', BCDataName='DirichletData',
    gridLocation='FaceCenter'):
    '''
    Generic function to impose a Boundary Condition ``inj1``. The following
    functions are more specific:

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

        BCDataSetName : str
            Name of the created node of type ``BCDataSet_t``. Default value is
            'BCDataSet#Init'

        BCDataName : str
            Name of the created node of type ``BCData_t``. Default value is
            'DirichletData'

    See also
    --------

    setBC_inj1, setBC_outpres, setBC_outmfr2

    '''
    checkVariables(ImposedVariables)
    FamilyNode = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    I._rmNodesByType(FamilyNode, 'FamilyBC_t')
    I.newFamilyBC(value=FamilyBC, parent=FamilyNode)

    if all([np.ndim(v)==0 for v in ImposedVariables.values()]):
        ImposedVariables = translateVariablesFromCGNS2Elsa(ImposedVariables)
        J.set(FamilyNode, '.Solver#BC', type=BCType, **ImposedVariables)
    else:
        assert bc is not None
        J.set(bc, '.Solver#BC', type=BCType)

        PointRange = I.getValue(I.getNodeFromType(bc, 'IndexRange_t'))
        bc_shape = PointRange[:, 1] - PointRange[:, 0]
        if bc_shape[0] == 0:
            bc_shape = (bc_shape[1], bc_shape[2])
        elif bc_shape[1] == 0:
            bc_shape = (bc_shape[0], bc_shape[2])
        elif bc_shape[2] == 0:
            bc_shape = (bc_shape[0], bc_shape[1])
        else:
            raise ValueError('Wrong BC shape {} in {}'.format(bc_shape, I.getPath(t, bc)))

        for var, value in ImposedVariables.items():
            assert value.shape == bc_shape, \
                'Wrong shape for variable {}: {} (shape {} for {})'.format(var, value.shape, bc_shape, I.getPath(t, bc))

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
        'Pressure', 'pressure', 'Temperature',
        'TurbulentEnergyKinetic', 'TurbulentDissipationRate', 'TurbulentDissipation', 'TurbulentLengthScale',
        'TurbulentSANuTilde', 'globalmassflow']
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
                NewVariables[var] = value
            else:
                NewVariables[CGNS2ElsaDict[var]] = value
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
            return CGNS2ElsaDict[var]
    else:
        raise TypeError('Variables must be of type dict, list or string')



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

    ETC.setBC_stage_mxpl(t, left, right, method=method)

def setBC_stage_mxpl_hyb(t, left, right, nbband=100, c=None):
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

    ETC.setBC_stage_mxpl_hyb(t, left, right, nbband=nbband, c=c)

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

    ETC.setBC_stage_red(t, left, right, stage_ref_time)

def setBC_stage_red_hyb(t, left, right, stage_ref_time, nbband=100, c=None):
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

        nbband : int
            Number of points in the radial distribution to compute.

        c : float
            Parameter for the distribution of radial points.

    '''

    ETC.setBC_stage_red_hyb(t, left, right, stage_ref_time, nbband=nbband, c=c)

def setBC_outradeq(t, FamilyName, valve_type=0, valve_ref_pres=None,
    valve_ref_mflow=None, valve_relax=0.1, ReferenceValues=None,
    TurboConfiguration=None, method='globborder_dict'):
    '''
    Set an outflow boundary condition of type ``outradeq``.
    The pivot index is 1 and cannot be changed.

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

    ETC.setBC_outradeq(t, FamilyName, valve_type=valve_type, valve_ref_pres=valve_ref_pres,
        valve_ref_mflow=valve_ref_mflow, valve_relax=valve_relax, ReferenceValues=ReferenceValues,
        TurboConfiguration=TurboConfiguration, method=method)

def setBC_outradeqhyb(t, FamilyName, valve_type, valve_ref_pres,
    valve_ref_mflow, valve_relax=0.1, nbband=100, c=None):
    '''
    Set an outflow boundary condition of type ``outradeqhyb``.
    The pivot index is 1 and cannot be changed.

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

        nbband : int
            Number of points in the radial distribution to compute.

        c : float
            Parameter for the distribution of radial points.

    '''

    ETC.setBC_outradeqhyb(t, FamilyName, valve_type, valve_ref_pres,
        valve_ref_mflow, valve_relax=valve_relax, nbband=nbband, c=c)


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

def launchIsoSpeedLines(PREFIX_JOB, AER, NProc, machine, DIRECTORY_WORK,
                    ThrottleRange, RotationSpeedRange, **kwargs):
    '''
    User-level function designed to launch iso-speed lines.

    Parameters
    ----------

        PREFIX_JOB : str
            an arbitrary prefix for the jobs

        AER : str
            full AER code for launching simulations on SATOR

        NProc : int
            Number of processors for each job.

        machine : str
            name of the machine ``'sator'``, ``'spiro'``, ``'eos'``...

            .. warning:: only ``'sator'`` has been tested

        DIRECTORY_WORK : str
            the working directory at computation server.

            .. note:: indicated directory may not exist. In this case, it will
                be created.

        ThrottleRange : list
            Throttle values to consider (depend on the valve law)

        RotationSpeedRange : list
            RotationSpeed numbers to consider

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
    ThrottleMatrix, RotationSpeedMatrix  = np.meshgrid(ThrottleRange, RotationSpeedRange)

    Throttle_       = ThrottleMatrix.ravel(order='K')
    RotationSpeed_  = RotationSpeedMatrix.ravel(order='K')
    NewJobs         = Throttle_ == ThrottleRange[0]

    JobsQueues = []
    for i, (Throttle, RotationSpeed, NewJob) in enumerate(zip(Throttle_, RotationSpeed_, NewJobs)):

        print('Assembling run {} Throttle={} RotationSpeed={} | NewJob = {}'.format(
                i, Throttle, RotationSpeed, NewJob))

        if NewJob:
            JobName = PREFIX_JOB+'%d'%i
            writeOutputFields = True
        else:
            writeOutputFields = False

        CASE_LABEL = '{:08.2f}_{}'.format(abs(Throttle), JobName)
        if Throttle < 0: CASE_LABEL = 'M'+CASE_LABEL

        WorkflowParams = copy.deepcopy(kwargs)

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

        if 'Initialization' in WorkflowParams and i != 0:
            WorkflowParams.pop('Initialization')

        if isinstance(WorkflowParams['mesh'], list):
            speedIndex = RotationSpeedRange.index(RotationSpeed)
            WorkflowParams['mesh'] = WorkflowParams['mesh'][speedIndex]

        JobsQueues.append(
            dict(ID=i, CASE_LABEL=CASE_LABEL, NewJob=NewJob, JobName=JobName, **WorkflowParams)
            )

    JM.saveJobsConfiguration(JobsQueues, AER, machine, DIRECTORY_WORK, NProc=NProc)

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
    for filename in otherFiles:
        if filename.startswith('/') or filename.startswith('../') \
            or len(filename.split('/'))>1:
            MSG = 'Input files must be inside the submission repository (not the case for {})'.format(filename)
            raise Exception(J.FAIL + MSG + J.ENDC)
    templatesFolder = os.getenv('MOLA') + '/TEMPLATES/WORKFLOW_COMPRESSOR'
    JM.launchJobsConfiguration(templatesFolder=templatesFolder, otherFiles=otherFiles)

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
