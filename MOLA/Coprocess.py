'''
MOLA Coprocess module - designed to be used in coupling (trigger) elsA context

Recommended syntax for use:

::

    import MOLA.Coprocess as CO

23/12/2020 - L. Bernardos - creation by recycling
'''


import sys
import os
import time
import timeit
from datetime import datetime
import numpy as np
from scipy.ndimage.filters import uniform_filter1d
import shutil
import psutil
import pprint
import glob
import copy
from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
NProcs = comm.Get_size()
nbOfDigitsOfNProcs = int(np.ceil(np.log10(NProcs+1)))

import Converter.PyTree as C
import Converter.Internal as I
import Converter.Filter as Filter
import Converter.Mpi as Cmpi
import Transform.PyTree as T
import Post.PyTree as P

from . import InternalShortcuts as J
from . import Preprocess as PRE
from . import JobManager as JM


# ------------------------------------------------------------------ #
# Following variables should be overridden using compute.py and coprocess.py
# scripts
FULL_CGNS_MODE   = False
FILE_SETUP       = 'setup.py'
FILE_CGNS        = 'main.cgns'
FILE_SURFACES    = 'surfaces.cgns'
FILE_ARRAYS      = 'arrays.cgns'
FILE_FIELDS      = 'fields.cgns'
FILE_COLOG       = 'coprocess.log'
FILE_BODYFORCESRC= 'bodyforce.cgns'
DIRECTORY_OUTPUT = 'OUTPUT'
DIRECTORY_LOGS   = 'LOGS'
setup            = None
CurrentIteration = 0
elsAxdt          = None
PyPartBase       = None
# ------------------------------------------------------------------- #
FAIL  = '\033[91m'
GREEN = '\033[92m'
WARN  = '\033[93m'
MAGE  = '\033[95m'
CYAN  = '\033[96m'
ENDC  = '\033[0m'


def invokeCoprocessLogFile():
    '''
    This function creates the ``coprocess.log`` file used for monitoring the
    progress of the simulation.
    '''
    if rank > 0: return
    DATEANDTIME=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open(FILE_COLOG, 'w') as f:
        f.write('COPROCESS LOG FILE STARTED AT %s\n'%DATEANDTIME)

def printCo(message, proc=None, color=None):
    '''
    This function is used for easily writing messages in ``coproces.log`` file.
    It is designed to be used in a MPI context (coprocessing).

    Parameters
    ----------

        message : str
            Message to be written in ``coprocess.log`` file

        proc : int or None
            if provided, only local MPI rank will write the message.
            If :py:obj:`None`, all procs will write the message.

        color : str
            endscape code for terminal colored output.
            For example, red output is obtained like this: ``color='\\033[91m'``
    '''
    if proc is not None and rank != proc: return
    preffix = ('[{:0%d}]: '%nbOfDigitsOfNProcs).format(rank)
    if color:
        message = color+message+ENDC
    with open(FILE_COLOG, 'a') as f:
        f.write(preffix+message+'\n')

def extractFields(Skeleton):
    '''
    Extract the coupling CGNS PyTree from elsAxdt *OUTPUT_TREE* and make
    necessary adaptions, including migration of coordinates fields to
    GridCoordinates_t nodes, renaming of conventional fields names and
    adding the tree's Skeleton.

    Parameters
    ----------

        Skeleton : PyTree
            Skeleton tree as obtained from :py:func:`Converter.Mpi.convertFile2SkeletonTree`

    Returns
    -------

        t : PyTree
            Coupling adapted PyTree

    '''
    t = elsAxdt.get(elsAxdt.OUTPUT_TREE)
    adaptEndOfRun(t)
    t = I.merge([Skeleton, t])

    return t

def extractArrays(t, arrays, RequestedStatistics=[], Extractions=[],
                  addMemoryUsage=True, addResiduals=True):
    '''
    Extract the arrays (1D data) as a PyTree, including additional
    optional information requested by optional arguments of the function.

    Parameters
    ----------

        t : PyTree
            Output PyTree as obtained from :py:func:`extractFields`

        arrays : dict
            Dictionary of arrays, as created using :py:func:`invokeArrays`

        RequestedStatistics : :py:class:`list` of :py:class:`str`
            same argument as :py:func:`extractIntegralData`

        Extractions : :py:class:`list` of :py:class:`dict`
            .. note:: to be implemented

        addMemoryUsage : bool
            if :py:obj:`True`, register and add the current memory usage

        addResiduals : bool
            if :py:obj:`True`, add the residuals information
    '''

    if addResiduals: extractResiduals(t, arrays)
    if addMemoryUsage: addMemoryUsage2Arrays(arrays)
    extractIntegralData(t, arrays, RequestedStatistics=RequestedStatistics,
                         Extractions=Extractions)
    arraysTree = arraysDict2PyTree(arrays)

    return arraysTree


def extractSurfaces(t, Extractions):
    '''
    Extracts flowfield data as surfacic (or curvilinear) zones from input
    fields **t** and requested extraction information provided in **Extractions**.

    Parameters
    ----------

        t : PyTree
            Output PyTree as obtained from :py:func:`extractFields`

        Extractions : :py:class:`list` of :py:class:`dict`
            Each element of this list is a dictionary that specifies the kind of
            requested extraction. Possible keys of the dictionary are:

            * ``type`` : mandatory
                Can be one of :

                * ``AllBC<type>``
                    Extract all BC windows corresponding to a given *<type>*,
                    and stores them in separated CGNSBases using their families'
                    names.

                * ``BC<type>``
                    Extract all BC windows corresponding to a given *<type>*,
                    not necessarily defined by means of a family, and stores
                    them in a single CGNSBase.

                * ``FamilySpecified:<FamilyName>``
                    Extract BC windows defined by a family named *<FamilyName>*,
                    and stores them in a single CGNSBase.

                * ``IsoSurface``
                    Slice the input flowfields following a given field name and
                    value

                * ``Sphere``
                    Slice the input flowfields using a sphere defined using a
                    center and a radius.

                * ``Plane``
                    Slice the input flowfields using a plane defined using a
                    point and a normal direction.

            * ``name`` : :py:class:`str` (optional)
                If provided, this name replaces the default name of the CGNSBase
                container of the surfaces

                ..note::
                  not relevant if ``type`` starts with  ``AllBC``

            * ``field`` : :py:class:`str` (contextual)
                Name of the field employed for slicing if ``type`` = ``IsoSurface``

            * ``value`` : :py:class:`str` (contextual)
                Value of the field employed for slicing if ``type`` = ``IsoSurface``

            * ``center`` : :py:class:`list` of 3 :py:class:`float` (optional, contextual)
                Coordinates of the sphere center employed for slicing if
                ``type`` = ``Sphere``

            * ``radius`` : :py:class:`float` (contextual)
                Radius of the sphere employed for slicing if ``type`` = ``Sphere``

            * ``point`` : :py:class:`list` of 3 :py:class:`float` (contextual)
                Coordinates of the point employed for slicing if
                ``type`` = ``Plane``

            * ``normal`` : :py:class:`list` of 3 :py:class:`float` (contextual)
                Normal vector employed for slicing if ``type`` = ``Plane``

    Returns
    -------

        SurfacesTree : PyTree
            Tree containing all requested surfaces as a set of zones stored
            in possibly different CGNSBases

    '''

    cellDimOutputTree = I.getZoneDim(I.getZones(t)[0])[-1]

    def addBase2SurfacesTree(basename):
        if not zones: return
        base = I.newCGNSBase(basename, cellDim=cellDimOutputTree-1, physDim=3,
            parent=SurfacesTree)
        I._addChild(base, zones)
        J.set(base, '.ExtractionInfo', **ExtractionInfo)
        return base

    t = I.renameNode(t, 'FlowSolution#Init', 'FlowSolution#Centers')
    I._renameNode(t, 'FlowSolution#Height', 'FlowSolution')
    I._rmNodesByName(t, 'FlowSolution#EndOfRun*')
    reshapeBCDatasetNodes(t)
    DictBCNames2Type = C.getFamilyBCNamesDict(t)
    SurfacesTree = I.newCGNSTree()
    PartialTree = Cmpi.convert2PartialTree(t)

    # See Anomaly 8784 https://elsa.onera.fr/issues/8784
    for BCDataSetNode in I.getNodesFromType(PartialTree, 'BCDataSet_t'):
        for node in I.getNodesFromType(BCDataSetNode, 'DataArray_t'):
            if I.getValue(node) is None:
                I.rmNode(BCDataSetNode, node)

    for Extraction in Extractions:
        TypeOfExtraction = Extraction['type']
        ExtractionInfo = copy.deepcopy(Extraction)

        if TypeOfExtraction.startswith('AllBC'):
            BCFilterName = TypeOfExtraction.replace('AllBC','')
            for BCFamilyName in DictBCNames2Type:
                BCType = DictBCNames2Type[BCFamilyName]
                if BCFilterName.lower() in BCType.lower():
                    zones = C.extractBCOfName(PartialTree,'FamilySpecified:'+BCFamilyName, extrapFlow=False)
                    ExtractionInfo['type'] = 'BC'
                    ExtractionInfo['BCType'] = BCType
                    addBase2SurfacesTree(BCFamilyName)

        elif TypeOfExtraction.startswith('BC'):
            zones = C.extractBCOfType(PartialTree, TypeOfExtraction, extrapFlow=False)
            try: basename = Extraction['name']
            except KeyError: basename = TypeOfExtraction
            ExtractionInfo['type'] = 'BC'
            ExtractionInfo['BCType'] = TypeOfExtraction
            addBase2SurfacesTree(basename)

        elif TypeOfExtraction.startswith('FamilySpecified:'):
            zones = C.extractBCOfName(PartialTree, TypeOfExtraction, extrapFlow=False)
            try: basename = Extraction['name']
            except KeyError: basename = TypeOfExtraction.replace('FamilySpecified:','')
            ExtractionInfo['type'] = 'BC'
            ExtractionInfo['BCType'] = TypeOfExtraction
            addBase2SurfacesTree(basename)

        elif TypeOfExtraction == 'IsoSurface':
            zones = P.isoSurfMC(PartialTree, Extraction['field'], Extraction['value'])
            try: basename = Extraction['name']
            except KeyError:
                FieldName = Extraction['field'].replace('Coordinate','').replace('Radius', 'R').replace('ChannelHeight', 'H')
                basename = 'Iso_%s_%g'%(FieldName,Extraction['value'])
            addBase2SurfacesTree(basename)

        elif TypeOfExtraction == 'Sphere':
            try: center = Extraction['center']
            except KeyError: center = 0,0,0
            Eqn = '({x}-{x0})**2+({y}-{y0})**2+({z}-{z0})**2-{r}**2'.format(
                x='{CoordinateX}',y='{CoordinateY}',z='{CoordinateZ}',
                r=Extraction['radius'], x0=center[0],
                y0=center[1], z0=center[2])
            C._initVars(PartialTree,'Slice=%s'%Eqn)
            zones = P.isoSurfMC(PartialTree, 'Slice', 0.0)
            try: basename = Extraction['name']
            except KeyError: basename = 'Sphere_%g'%Extraction['radius']
            addBase2SurfacesTree(basename)

        elif TypeOfExtraction == 'Plane':
            n = np.array(Extraction['normal'])
            Pt = np.array(Extraction['point'])
            PlaneCoefs = n[0],n[1],n[2],-n.dot(Pt)
            C._initVars(PartialTree,'Slice=%0.12g*{CoordinateX}+%0.12g*{CoordinateY}+%0.12g*{CoordinateZ}+%0.12g'%PlaneCoefs)
            zones = P.isoSurfMC(PartialTree, 'Slice', 0.0)
            try: basename = Extraction['name']
            except KeyError: basename = 'Plane'
            addBase2SurfacesTree(basename)

    Cmpi._convert2PartialTree(SurfacesTree)
    J.forceZoneDimensionsCoherency(SurfacesTree)
    Cmpi.barrier()
    restoreFamilies(SurfacesTree, t)
    Cmpi.barrier()

    return SurfacesTree

def extractIntegralData(to, arrays, Extractions=[],
                        RequestedStatistics=['std-CL', 'std-CD']):
    '''
    Extract integral data from coupling tree **to**, and update **arrays** Python
    dictionary adding statistics requested by the user.

    Parameters
    ----------

        to : PyTree
            Coupling tree as obtained from :py:func:`adaptEndOfRun`

        arrays :dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array

            ..note:: **arrays** is modified in-place

        RequestedStatistics : :py:class:`list` of :py:class:`str`
            Here, the user requests the additional statistics to be computed.
            The syntax of each quantity must be as follows:

            ::

                '<preffix>-<integral_quantity_name>'

            `<preffix>` can be ``'avg'`` (for cumulative average) or ``'std'``
            (for standard deviation). ``<integral_quantity_name>`` can be any
            quantity contained in arrays, including other statistics.

            .. hint:: chaining preffixes is perfectly accepted, like
                ``'std-std-CL'`` which would compute the cumulative standard
                deviation of the cumulative standard deviation of the
                lift coefficient (:math:`\sigma(\sigma(C_L))`)

    '''
    IntegralDataNodes = I.getNodesFromType2(to, 'IntegralData_t')
    for IntegralDataNode in IntegralDataNodes:
        IntegralDataName = getIntegralDataName(IntegralDataNode)
        _appendIntegralDataNode2Arrays(arrays, IntegralDataNode)
        _extendArraysWithProjectedLoads(arrays, IntegralDataName)
        _normalizeMassFlowInArrays(arrays, IntegralDataName)
    for IntegralDataName in arrays:
        _extendArraysWithStatistics(arrays, IntegralDataName, RequestedStatistics)
    Cmpi.barrier()

def extractResiduals(to, arrays):
    '''
    Extract residuals from coupling tree **to**, and update **arrays** Python
    dictionary.

    Parameters
    ----------

        to : PyTree
            Coupling tree as obtained from :py:func:`adaptEndOfRun`

        arrays : dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array

            ..note:: **arrays** is modified in-place

    '''
    ConvergenceHistoryNodes = I.getNodesByType(to, 'ConvergenceHistory_t')
    for ConvergenceHistory in ConvergenceHistoryNodes:
        ConvergenceDict = dict()
        for DataArrayNode in I.getNodesFromType(ConvergenceHistory, 'DataArray_t'):
            DataArrayValue = I.getValue(DataArrayNode)
            if isinstance(DataArrayValue, int) or isinstance(DataArrayValue, float):
                DataArrayValue = np.array([DataArrayValue],dtype=type(DataArrayValue))
            if len(DataArrayValue.shape) == 1:
                ConvergenceDict[I.getName(DataArrayNode)] = DataArrayValue
        appendDict2Arrays(arrays, ConvergenceDict, I.getName(ConvergenceHistory))
        break # we ignore possibly multiple ConvergenceHistory_t nodes

def save(t, filename, tagWithIteration=False):
    '''
    Generic function to save a PyTree **t** in parallel. Works whatever the
    dimension of the PyTree. Use it to save ``'fields.cgns'``,
    ``'surfaces.cgns'`` or ``'arrays.cgns'``.

    .. important::
        If the mesh was split with PyPart and if the function is called to save
        *FILE_FIELDS*, the tree is automatically merged and saved using PyPart.
        In that case, the variable **PyPartBase** should be defined (normally,
        in ``compute.py``)

    Parameters
    ----------

        t : PyTree
            tree to save

        filename : str
            Name of the file

        tagWithIteration : bool
            if :py:obj:`True`, adds a suffix ``_AfterIter<iteration>``
            to the saved filename (creates a copy)
    '''
    if PyPartBase and filename.endswith(FILE_FIELDS):
        saveWithPyPart(t, filename, tagWithIteration=tagWithIteration)
        return

    t = I.copyRef(t) if I.isTopTree(t) else C.newPyTree(['Base', J.getZones(t)])
    Cmpi._convert2PartialTree(t)
    I._adaptZoneNamesForSlash(t)
    for z in I.getZones(t):
        SolverParam = I.getNodeFromName(z,'.Solver#Param')
        if not SolverParam or not I.getNodeFromName(SolverParam,'proc'):
            Cmpi._setProc(z, Cmpi.rank)
    I._rmNodesByName(t,'ID_*')
    I._rmNodesByType(t,'IntegralData_t')

    Skeleton = J.getStructure(t)

    UseMerge = False
    try:
        Skeletons = Cmpi.KCOMM.gather(Skeleton,root=0)
    except SystemError:
        UseMerge = True
        printCo('Cmpi.KCOMM.gather FAILED. Using merge=True', color=J.WARN)
        UseMerge = comm.bcast(UseMerge,root=Cmpi.rank)

    if not UseMerge:
        if Cmpi.rank == 0:
            trees = [s if s else I.newCGNSTree() for s in Skeletons]
            trees.insert(0,t)
            tWithSkel = I.merge(trees)
        else:
            tWithSkel = t
        Cmpi.barrier()
        renameTooLongZones(tWithSkel)
        for l in 2,3: I._correctPyTree(tWithSkel,l) # unique base and zone names
    else:
        tWithSkel = t

    Cmpi.barrier()
    if Cmpi.rank==0:
        try:
            if os.path.islink(filename):
                os.unlink(filename)
            else:
                os.remove(filename)
        except:
            pass
    Cmpi.barrier()

    printCo('will save %s ...'%filename,0, color=J.CYAN)
    Cmpi.convertPyTree2File(tWithSkel, filename, merge=UseMerge)
    printCo('... saved %s'%filename,0, color=J.CYAN)
    Cmpi.barrier()
    if tagWithIteration and Cmpi.rank == 0: copyOutputFiles(filename)

def saveWithPyPart(t, filename, tagWithIteration=False):
    '''
    Function to save a PyTree **t** with PyPart. The PyTree must have been
    splitted with PyPart in ``compute.py``. An important point is the presence
    in every zone of **t** of the special node ``:CGNS#Ppart``.

    Use this function to save ``'fields.cgns'``.

    .. note:: For more details on PyPart, see the dedicated pages on elsA
        support:
        `PyPart alone <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/PreprocessTutorials/etc_pypart_alone.html>`_
        and
        `PyPart with elsA <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/PreprocessTutorials/etc_pypart_elsa.html>`_

    Parameters
    ----------

        t : PyTree
            tree to save

        filename : str
            Name of the file

        tagWithIteration : bool
            if :py:obj:`True`, adds a suffix ``_AfterIter<iteration>``
            to the saved filename (creates a copy)
    '''
    t = I.copyRef(t)
    Cmpi._convert2PartialTree(t)
    I._rmNodesByName(t, '.Solver#Param')
    I._rmNodesByType(t,'IntegralData_t')
    Cmpi.barrier()
    printCo('will save %s ...'%filename,0, color=J.CYAN)
    PyPartBase.mergeAndSave(t, 'PyPart_fields')
    Cmpi.barrier()
    if rank == 0:
        t = C.convertFile2PyTree('PyPart_fields_all.hdf')
        C.convertPyTree2File(t, filename)
        for fn in glob.glob('PyPart_fields_*.hdf'):
            try: os.remove(fn)
            except: pass
    printCo('... saved %s'%filename,0, color=J.CYAN)
    Cmpi.barrier()
    if tagWithIteration and rank == 0: copyOutputFiles(filename)

def moveTemporaryFile(temporary_file):
    '''
    Removes ``tmp-`` characters of given **temporary_file** using a *move*
    operation.

    Parameters
    ----------

        temporary_file : str
            path of the file whose name contains characters ``tmp-``

    '''
    final_file = temporary_file.replace('tmp-','')
    if final_file == temporary_file: return
    if Cmpi.rank == 0:
        printCo('deleting %s ...'%final_file)
        try:
            os.remove(final_file)
            WillReplace = True
        except:
            WillReplace = False

        if WillReplace:
            printCo('deleting %s ... OK'%final_file,color=GREEN)
            printCo('moving %s to %s ...'%(temporary_file,final_file))
            try:
                shutil.move(temporary_file,final_file)
                printCo('moving %s to %s ... OK'%(temporary_file,final_file),
                        color=GREEN)
            except:
                printCo('moving %s to %s ... FAILED'%(temporary_file,final_file),
                        color=FAIL)
        else:
            printCo('deleting %s ... FAILED'%final_file,color=FAIL)
    Cmpi.barrier()


def restoreFamilies(surfaces, skeleton):
    '''
    Restore families in the PyTree **surfaces** (e.g read from
    ``'surfaces.cgns'``) based on information in **skeleton** (e.g read from
    ``'main.cgns'``). Also add the ReferenceState to be able to use function
    computeVariables from Cassiopee Post module.

    .. tip:: **skeleton** may be a skeleton tree.

    Parameters
    ----------

        surfaces : PyTree
            tree where zone names are the same as in **skeleton** (or with a
            suffix in '\\<bcname>'), but without information on families and
            ReferenceState.

        skeleton : PyTree
            tree of the full 3D domain with zones, families and ReferenceState.
            No data is needed so **skeleton** may be a skeleton tree.
    '''
    I._adaptZoneNamesForSlash(surfaces)

    ReferenceState = I.getNodeFromType2(skeleton, 'ReferenceState_t') # To compute variables

    FamilyNodes = I.getNodesFromType2(skeleton, 'Family_t')

    for base in I.getNodesFromType1(surfaces, 'CGNSBase_t'):
        I.addChild(base, ReferenceState)
        familiesInBase = []
        for zone in I.getZones(base):
            zoneName = I.getName(zone).split('\\')[0]  # There might be a \ in zone name
                                                       # if it is a result of C.ExtractBCOfType
            zoneInFullTree = I.getNodeFromNameAndType(skeleton, zoneName, 'Zone_t')
            if not zoneInFullTree:
                # Zone may have been renamed by C.correctPyTree in <zone>.N with N an integer
                zoneName = '.'.join(zoneName.split('.')[:-1])
                zoneInFullTree = I.getNodeFromNameAndType(skeleton, zoneName, 'Zone_t')
                if not zoneInFullTree:
                    raise Exception('Zone {} not found in skeleton'.format(zoneName))
            fam = I.getNodeFromType1(zoneInFullTree, 'FamilyName_t')
            I.addChild(zone, fam)
            familiesInBase.append(I.getValue(fam))
        for family in FamilyNodes:
            if I.getName(family) in familiesInBase:
                I.addChild(base, family)

def monitorTurboPerformance(surfaces, arrays, RequestedStatistics=[], tagWithIteration=False):
    '''
    Monitor performance (massflow in/out, total pressure ratio, total
    temperature ratio, isentropic efficiency) for each row in a compressor
    simulation. This processing is triggered if at least two bases in the PyTree
    **surfaces** fill the following requirements:

    #. there is a node ``'.ExtractionInfo'`` of type ``'UserDefinedData_t'``

    #. it contains a node ``'ReferenceRow'``, whose value is a :py:class:`str`
       corresponding to a row Family in ``'main.cgns'``.

    #. it contains a node ``'tag'``, whose value is a :py:class:`str` equal
       to ``'InletPlane'`` or ``'OutletPlane'``.

    .. note:: For one ``'ReferenceRow'``, the monitor is processed only if both
        ``'InletPlane'`` and ``'OutletPlane'`` are found.

    .. note:: These bases must contain variables ``'PressureStagnation'`` and
        ``'TemperatureStagnation'``

    .. important:: This function is adapted only to **Workflow Compressor** cases.

    Parameters
    ----------

        surfaces : PyTree
            as produced by :py:func:`extractSurfaces`

        arrays :dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array

        RequestedStatistics : :py:class:`list` of :py:class:`str`
            Here, the user requests the additional statistics to be computed.
            See documentation of function :py:func:`extractIntegralData` for
            more details.

    '''
    # FIXME: Segmentation fault bug when this function is used after
    #        POST.absolute2Relative (in co -proccessing only)
    def massflowWeightedIntegral(t, var):
        t = C.initVars(t, 'rou_var={MomentumX}*{%s}'%(var))
        integ  = abs(P.integNorm(t, 'rou_var')[0][0])
        return integ

    def surfaceWeightedIntegral(t, var):
        integ  = abs(P.integNorm(t, var)[0][0])
        return integ

    for row, rowParams in setup.TurboConfiguration['Rows'].items():

        planeUpstream   = I.newCGNSTree()
        planeDownstream = I.newCGNSTree()
        if not 'RotationSpeed' in rowParams:
            continue
        elif rowParams['RotationSpeed'] != 0:
            IsRotor = True
        else:
            IsRotor = False

        for base in I.getNodesFromType(surfaces, 'CGNSBase_t'):
            try:
                ExtractionInfo = I.getNodeFromNameAndType(base, '.ExtractionInfo', 'UserDefinedData_t')
                ReferenceRow = I.getValue(I.getNodeFromName(ExtractionInfo, 'ReferenceRow'))
                tag = I.getValue(I.getNodeFromName(ExtractionInfo, 'tag'))
                if ReferenceRow == row and tag == 'InletPlane':
                    planeUpstream = base
                elif ReferenceRow == row and tag == 'OutletPlane':
                    planeDownstream = base
            except:
                pass

        if IsRotor:
            VarAndMeanList = [
                (['PressureStagnation', 'TemperatureStagnation'], massflowWeightedIntegral)
            ]
        else:
            VarAndMeanList = [
                (['PressureStagnation'], massflowWeightedIntegral),
                (['Pressure'], surfaceWeightedIntegral)
            ]

        dataUpstream   = integrateVariablesOnPlane(  planeUpstream, VarAndMeanList)
        dataDownstream = integrateVariablesOnPlane(planeDownstream, VarAndMeanList)

        if not dataUpstream or not dataDownstream:
            continue

        if rank == 0:
            fluxcoeff = rowParams['NumberOfBlades'] / float(rowParams['NumberOfBladesSimulated'])
            if IsRotor:
                perfos = computePerfoRotor(dataUpstream, dataDownstream, fluxcoeff=fluxcoeff)
            else:
                perfos = computePerfoStator(dataUpstream, dataDownstream, fluxcoeff=fluxcoeff)
            appendDict2Arrays(arrays, perfos, 'PERFOS_{}'.format(row))
            _extendArraysWithStatistics(arrays, 'PERFOS_{}'.format(row), RequestedStatistics)

    arraysTree = arraysDict2PyTree(arrays)
    save(arraysTree, os.path.join(DIRECTORY_OUTPUT, FILE_ARRAYS), tagWithIteration=tagWithIteration)

def computePerfoRotor(dataUpstream, dataDownstream, fluxcoeff=1., fluxcoeffOut=None):

    if not fluxcoeffOut:
        fluxcoeffOut = fluxcoeff

    gamma = setup.FluidProperties['Gamma']

    # Compute total quantities ratio between in/out planes
    meanPtIn  =      dataUpstream['PressureStagnation'] /   dataUpstream['MassFlow']
    meanPtOut =    dataDownstream['PressureStagnation'] / dataDownstream['MassFlow']
    meanTtIn  =   dataUpstream['TemperatureStagnation'] /   dataUpstream['MassFlow']
    meanTtOut = dataDownstream['TemperatureStagnation'] / dataDownstream['MassFlow']
    PtRatio = meanPtOut / meanPtIn
    TtRatio = meanTtOut / meanTtIn
    # Compute Isentropic Efficiency
    etaIs = (PtRatio**((gamma-1.)/gamma) - 1.) / (TtRatio - 1.)

    perfos = dict(
        IterationNumber            = CurrentIteration-1,  # Because extraction before current iteration (next_state=16)
        MassFlowIn                 = dataUpstream['MassFlow']*fluxcoeff,
        MassFlowOut                = dataDownstream['MassFlow']*fluxcoeffOut,
        PressureStagnationRatio    = PtRatio,
        TemperatureStagnationRatio = TtRatio,
        EfficiencyIsentropic       = etaIs
    )

    return perfos

def computePerfoStator(dataUpstream, dataDownstream, fluxcoeff=1., fluxcoeffOut=None):

    if not fluxcoeffOut:
        fluxcoeffOut = fluxcoeff

    meanPtIn  =   dataUpstream['PressureStagnation'] /   dataUpstream['MassFlow']
    meanPtOut = dataDownstream['PressureStagnation'] / dataDownstream['MassFlow']
    meanPsIn  =             dataUpstream['Pressure'] /       dataUpstream['Area']

    perfos = dict(
        IterationNumber         = CurrentIteration-1,  # Because extraction before current iteration (next_state=16)
        MassFlowIn              = dataUpstream['MassFlow']*fluxcoeff,
        MassFlowOut             = dataDownstream['MassFlow']*fluxcoeffOut,
        PressureStagnationRatio = meanPtOut / meanPtIn,
        PressureStagnationLossCoeff = (meanPtIn - meanPtOut) / (meanPtIn - meanPsIn)
    )

    return perfos

def integrateVariablesOnPlane(surface, VarAndMeanList):
    '''
    Integrate variables on a surface.

    Parameters
    ----------

        surface : PyTree
            Surface for integration. Required variables must be present already
            in **surface**.

        VarAndMeanList : :py:class:`list` of :py:class:`tuple`
            List of 2-tuples. Each tuple associates:

                * a list of variables, that must be found in **surface**

                * a function to perform the weighted integration wished for
                  variables

            For example:

            ::

                VarAndMeanList = [([var1, var2], meanFunction1),
                                  ([var3], meanFunction2), ...]

            Example of function for a massflow weighted integration:

            ::

                import Converter.PyTree as C
                import Post.PyTree      as P
                def massflowWeightedIntegral(t, var):
                    t = C.initVars(t, 'rou_var={MomentumX}*{%s}'%(var))
                    integ  = abs(P.integNorm(t, 'rou_var')[0][0])
                    return integ

    Returns
    -------

        data : :py:class:`dict` or :py:obj:`None`
            dictionary that contains integrated values of variables. If
            **surface** is empty, does not contain a ``FlowSolution`` node or
            does not contains required variables, the function returns
            :py:obj:`None`.

    '''
    # Convert to Tetra arrays for integration # TODO identify bug and notify
    surface = C.convertArray2Tetra(surface)
    check =  True
    data = dict()

    if I.getNodesFromType(surface, 'FlowSolution_t') == []:
        for var in ['MassFlow', 'Area']:
            data[var] = 0
        for varList, meanFunction in VarAndMeanList:
            for var in varList:
                data[var] = 0
    else:
        C._initVars(surface, 'ones=1')
        data['Area']     = abs(P.integNorm(surface, var='ones')[0][0])
        data['MassFlow'] = abs(P.integNorm(surface, var='MomentumX')[0][0])
        try:
            for varList, meanFunction in VarAndMeanList:
                for var in varList:
                    data[var] = meanFunction(surface, var)
        except NameError:
            # Variables cannot be found
            check = False

    # Check if the needed variables were extracted
    Cmpi.barrier()
    check = comm.allreduce(check, op=MPI.LAND) #LAND = Logical AND
    data['MassFlow'] = comm.allreduce(data['MassFlow'], op=MPI.SUM)
    data['Area'] = comm.allreduce(data['Area'], op=MPI.SUM)
    Cmpi.barrier()
    if not check or data['Area']==0: return None

    # MPI Reduction to sum quantities on proc 0
    Cmpi.barrier()
    for varList, meanFunction in VarAndMeanList:
        for var in varList:
            data[var] = comm.reduce(data[var], op=MPI.SUM, root=0)
    Cmpi.barrier()
    return data

def writeSetup(setup):
    '''
    Write the ``setup.py`` file using as input the setup module object.

    .. warning:: This function will be replaced by :py:func:`MOLA.Preprocess.writeSetup`
        and :py:func:`MOLA.Preprocess.writeSetupFromModuleObject` functions

    Parameters
    ---------

        setup : module
            Python module object as obtained from command

            >>> import setup
    '''

    Lines  = ['"""\n%s file automatically generated in COPROCESS\n"""\n'%FILE_SETUP]

    Lines += ["FluidProperties=" +pprint.pformat(setup.FluidProperties)+"\n"]
    Lines += ["ReferenceValues=" +pprint.pformat(setup.ReferenceValues)+"\n"]
    Lines += ["elsAkeysCFD="     +pprint.pformat(setup.elsAkeysCFD)+"\n"]
    Lines += ["elsAkeysModel="   +pprint.pformat(setup.elsAkeysModel)+"\n"]
    Lines += ["elsAkeysNumerics="+pprint.pformat(setup.elsAkeysNumerics)+"\n"]

    try:
        Lines += ["BodyForceInputData="+pprint.pformat(setup.BodyForceInputData)+"\n"]
    except:
        pass

    AllLines = '\n'.join(Lines)

    with open(FILE_SETUP,'w') as f: f.write(AllLines)

def updateAndWriteSetup(setup):
    '''
    This function is used for adapting ``setup.py`` information for a new run.

    Parameters
    ---------

        setup : module
            Python module object as obtained from command

            >>> import setup
    '''
    if rank == 0:
        printCo('updating setup.py ...', proc=0, color=GREEN)
        setup.elsAkeysNumerics['inititer'] = CurrentIteration
        if 'itime' in setup.elsAkeysNumerics:
            setup.elsAkeysNumerics['itime'] = CurrentIteration * setup.elsAkeysNumerics['timestep']
        PRE.writeSetupFromModuleObject(setup)
        printCo('updating setup.py ... OK', proc=0, color=GREEN)
    comm.Barrier()

def invokeArrays():
    '''
    Create **arrays** Python dictionary by reading any pre-existing data
    contained in ``OUTPUT/arrays.cgns``

    .. note:: an empty dictionary is returned if no ``OUTPUT/arrays.cgns`` file
        is found

    Returns
    -------

        arrays :dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array
    '''
    Cmpi.barrier()
    arrays = dict()
    FullPathArraysFile = os.path.join(DIRECTORY_OUTPUT, FILE_ARRAYS)
    ExistingArraysFile = os.path.exists(FullPathArraysFile)
    Cmpi.barrier()
    inititer = setup.elsAkeysNumerics['inititer']
    if ExistingArraysFile and inititer>1:
        t = Cmpi.convertFile2SkeletonTree(FullPathArraysFile)
        t = Cmpi.readZones(t, FullPathArraysFile, rank=rank)
        Cmpi._convert2PartialTree(t, rank=rank)

        for zone in I.getZones(t):
            ZoneName = I.getName(zone)
            VarNames, = C.getVarNames(zone, excludeXYZ=True)
            FlowSol_n = I.getNodeFromName1(zone, 'FlowSolution')
            arrays[ZoneName] = dict()
            arraysSubset = arrays[ZoneName]
            if FlowSol_n:
                for VarName in VarNames:
                    Var_n = I.getNodeFromName1(FlowSol_n, VarName)
                    if Var_n:
                        arraysSubset[VarName] = Var_n[1]

            try:
                iters = np.copy(arraysSubset['IterationNumber'])
                for VarName in arraysSubset:
                    arraysSubset[VarName] = arraysSubset[VarName][iters<inititer]
            except KeyError:
                pass

    Cmpi.barrier()

    return arrays

def addMemoryUsage2Arrays(arrays):
    '''
    This function adds or updates a component in **arrays** for monitoring the
    employed memory. Only nodes are monitored (not every single proc, as this
    would produce redundant information). The number of cores contained in each
    computational node is retreived from the environment variable
    ``SLURM_CPUS_ON_NODE``.

    If this information does not exist, a value of ``28`` is taken by default.

    Parameters
    ----------

        arrays :dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array

            parameter **arrays** is modified
    '''
    CoreNumberPerNode = int(os.getenv('SLURM_CPUS_ON_NODE', 28))

    if rank%CoreNumberPerNode == 0:
        ZoneName = 'MemoryUsageOfProc%d'%rank
        UsedMemory = psutil.virtual_memory().used
        UsedMemoryPctg = psutil.virtual_memory().percent
        try:
            ArraysItem = arrays[ZoneName]
        except KeyError:
            arrays[ZoneName] = dict(IterationNumber=np.array([],dtype=int),
                                   UsedMemoryInPercent=np.array([],dtype=float),
                                   UsedMemory=np.array([],dtype=float),)
            ArraysItem = arrays[ZoneName]

        try:
            ArraysItem['IterationNumber'] = np.hstack((ArraysItem['IterationNumber'],
                                                      int(CurrentIteration)))
            ArraysItem['UsedMemoryInPercent'] = np.hstack((ArraysItem['UsedMemoryInPercent'],
                                                          float(UsedMemoryPctg)))
            ArraysItem['UsedMemory'] = np.hstack((ArraysItem['UsedMemory'],
                                             float(UsedMemory)))
        except KeyError:
            del arrays[ZoneName]
    Cmpi.barrier()

def arraysDict2PyTree(arrays):
    '''
    This function converts the **arrays** Python dictionary to a PyTree (CGNS)
    structure **t**.

    Parameters
    ----------

        arrays :dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array

    Returns
    -------

        t : PyTree
            same information as input, but structured in a PyTree CGNS form

    .. warning:: after calling the function, **arrays** and **t** do *NOT*
        share memory, which means that modifications on **arrays** will not
        affect **t** and vice-versa
    '''
    zones = []
    for ZoneName in arrays:
        arraysSubset = arrays[ZoneName]
        Arrays, Vars = [], []
        OrderedVars = [var for var in arraysSubset]

        OrderedVars.sort()
        for var in OrderedVars:
            Vars.append(var)
            Arrays.append(arraysSubset[var])

        zone = J.createZone(ZoneName, Arrays, Vars)
        if zone: zones.append(zone)

    if zones:
        Cmpi._setProc(zones, rank)
        t = C.newPyTree(['Base', zones])
    else:
        t = C.newPyTree(['Base'])
    Cmpi.barrier()

    return t

def appendDict2Arrays(arrays, dictToAppend, basename):
    '''
    This function add data defined in **dictToAppend** in the base **basename**
    of **arrays**.

    Parameters
    ----------

        arrays :dict
            Contains integral data in the following form:

            >>> arrays[basename]['VariableName'] = np.array

        dictToAppend : dict
            Contains data to append in **arrays**. For each element:
                * key is the variable name
                * value is the associated value

        basename : str
            Name of the base in which values will be appended.

    '''
    if not basename in arrays:
        arrays[basename] = dict()

    for var, value in dictToAppend.items():
        if var in arrays[basename]:
            arrays[basename][var] = np.append(arrays[basename][var], value)
        else:
            arrays[basename][var] = np.array([value],ndmin=1)


def _appendIntegralDataNode2Arrays(arrays, IntegralDataNode):
    '''
    Beware: this is a private function, employed by :py:func:`extractIntegralData`

    This function converts the CGNS IntegralDataNode (as provided by elsA)
    into the Python dictionary structure of arrays dictionary, and append it
    to the latter.

    Parameters
    ----------

        arrays :dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array

        IntegralDataNode : node
            Contains integral data as provided by elsA
    '''
    IntegralData = dict()
    for DataArrayNode in I.getChildren(IntegralDataNode):
        IntegralData[I.getName(DataArrayNode)] = I.getValue(DataArrayNode)
    IntegralDataName = getIntegralDataName(IntegralDataNode)
    IterationNumber = IntegralData['IterationNumber']

    try:
        arraysSubset = arrays[IntegralDataName]
    except KeyError:
        arrays[IntegralDataName] = dict()
        arraysSubset = arrays[IntegralDataName]


    try: RegisteredIterations = arraysSubset['IterationNumber']
    except KeyError: RegisteredIterations = np.array([])
    if len(RegisteredIterations) > 0:
        PreviousRegisteredArrays = True
        eps = 1e-12
        UpdatePortion = IterationNumber > (RegisteredIterations[-1] + eps)
        try: FirstIndex2Update = np.where(UpdatePortion)[0][0]
        except IndexError: return
    else:
        PreviousRegisteredArrays = False

    for integralKey in IntegralData:
        if PreviousRegisteredArrays:
            PreviousArray = arraysSubset[integralKey]
            AppendArray = IntegralData[integralKey][FirstIndex2Update:]
            arraysSubset[integralKey] = np.hstack((PreviousArray, AppendArray))
        else:
            arraysSubset[integralKey] = np.array(IntegralData[integralKey],
                                               order='F', ndmin=1)

def _normalizeMassFlowInArrays(arrays, IntegralDataName):
    '''
    Beware: this is a private function, employed by :py:func:`extractIntegralData`

    If:

        * the variable 'convflux_ro' is in **arrays[IntegralDataName]** (massflow
          extracted by elsA on the Family **IntegralDataName**),

        * and it exists a **FluxCoef** associted to **IntegralDataName** in
          ** ReferenceValues**, written in:

          >>> setup.ReferenceValues['NormalizationCoefficient'][IntegralDataName]['FluxCoef']

    Then the variable 'MassFlow' is added in **arrays[IntegralDataName]** by
    multiplying 'convflux_ro' by this **FluxCoef**.

    Parameters
    ----------

        arrays : dict
            Contains integral data in the following form:

            >>> np.array = arrays['FamilyBCNameOrElementName']['VariableName']

        IntegralDataName : str
            Name of the IntegralDataNode (CGNS) provided by elsA. It is used as
            key for arrays dictionary.
    '''
    arraysSubset = arrays[IntegralDataName]
    try:
        FluxCoef = setup.ReferenceValues['NormalizationCoefficient'][IntegralDataName]['FluxCoef']
        arraysSubset['MassFlow'] = arraysSubset['convflux_ro'] * FluxCoef
    except:
        return

def _extendArraysWithProjectedLoads(arrays, IntegralDataName):
    '''
    Beware: this is a private function, employed by :py:func:`extractIntegralData`

    This function is employed for adding aerodynamic-relevant coefficients to
    the arrays dictionary. The new quantites are the following :

        elsA Extractions  ---->   New projections (aero coefficients)
        -------------------------------------------------------------
        MomentumXFlux               CL   (force coef following LiftDirection)
        MomentumYFlux               CD   (force coef following DragDirection)
        MomentumZFlux               CY   (force coef following SideDirection)
        TorqueX                     Cn   (torque coef around LiftDirection)
        TorqueY                     Cl   (torque coef around DragDirection)
        TorqueZ                     Cm   (torque coef around SideDirection)


    The aerodynamic coefficients are dimensionless. They are computed using
    the scales and origins 'FluxCoef', 'TorqueCoef' and 'TorqueOrigin' contained
    in the setup.py dictionary,
                ReferenceValues['NormalizationCoefficient'][<IntegralDataName>]

    where <IntegralDataName> is the name of the component.
    If they are not provided, then global scaling factors are taken from
    ReferenceValues of setup.py,

                        ReferenceValues['FluxCoef']
                        ReferenceValues['TorqueCoef']
                        ReferenceValues['TorqueOrigin']


    ***************************************************************************
    * VERY IMPORTANT NOTE : in order to obtain meaningful Cn, Cl and Cm       *
    * coefficients, elsA integral torques (TorqueX, TorqueY and TorqueZ) must *
    * be applied at absolute zero coordinates (0,0,0), which means that       *
    * special elsA CGNS keys xtorque, ytorque and ztorque must all be 0.      *
    * Also, all Momentum and Torque fluxes shall be dimensional, which means  *
    * that special elsA CGNS keys fluxcoeff and torquecoeff must be exactly 1.*
    * Indeed, proper scaling and normalization is done using the setup.py     *
    * ReferenceValues, as indicated previously.                               *
    ***************************************************************************

    Parameters
    ----------

        arrays : dict
            Contains integral data in the following form:

            >>> np.array = arrays['FamilyBCNameOrElementName']['VariableName']

        IntegralDataName : str
            Name of the IntegralDataNode (CGNS) provided by elsA. It is used as
            key for arrays dictionary.
    '''


    DragDirection=np.array(setup.ReferenceValues['DragDirection'],dtype=np.float)
    SideDirection=np.array(setup.ReferenceValues['SideDirection'],dtype=np.float)
    LiftDirection=np.array(setup.ReferenceValues['LiftDirection'],dtype=np.float)

    arraysSubset = arrays[IntegralDataName]

    try:
        FluxCoef = setup.ReferenceValues['NormalizationCoefficient'][IntegralDataName]['FluxCoef']
        TorqueCoef = setup.ReferenceValues['NormalizationCoefficient'][IntegralDataName]['TorqueCoef']
        TorqueOrigin = setup.ReferenceValues['NormalizationCoefficient'][IntegralDataName]['TorqueOrigin']
    except:
        FluxCoef = setup.ReferenceValues['FluxCoef']
        TorqueCoef = setup.ReferenceValues['TorqueCoef']
        TorqueOrigin = setup.ReferenceValues['TorqueOrigin']

    try:
        FX = arraysSubset['MomentumXFlux']
        FY = arraysSubset['MomentumYFlux']
        FZ = arraysSubset['MomentumZFlux']
        MX = arraysSubset['TorqueX']
        MY = arraysSubset['TorqueY']
        MZ = arraysSubset['TorqueZ']
    except KeyError:
        return # no required fields for computing external aero coefficients

    # Pole change
    # TODO make ticket for elsA concerning CGNS parsing of xtorque ytorque ztorque
    TX = MX-(TorqueOrigin[1]*FZ - TorqueOrigin[2]*FY)
    TY = MY-(TorqueOrigin[2]*FX - TorqueOrigin[0]*FZ)
    TZ = MZ-(TorqueOrigin[0]*FY - TorqueOrigin[1]*FX)

    arraysSubset['CL']=FX*LiftDirection[0]+FY*LiftDirection[1]+FZ*LiftDirection[2]
    arraysSubset['CD']=FX*DragDirection[0]+FY*DragDirection[1]+FZ*DragDirection[2]
    arraysSubset['CY']=FX*SideDirection[0]+FY*SideDirection[1]+FZ*SideDirection[2]
    arraysSubset['Cn']=TX*LiftDirection[0]+TY*LiftDirection[1]+TZ*LiftDirection[2]
    arraysSubset['Cl']=TX*DragDirection[0]+TY*DragDirection[1]+TZ*DragDirection[2]
    arraysSubset['Cm']=TX*SideDirection[0]+TY*SideDirection[1]+TZ*SideDirection[2]

    # Normalize forces and moments
    for Force in ('CL','CD','CY'):  arraysSubset[Force]  *= FluxCoef
    for Torque in ('Cn','Cl','Cm'): arraysSubset[Torque] *= TorqueCoef


def _extendArraysWithStatistics(arrays, IntegralDataName, RequestedStatistics):
    '''
    Beware: this is a private function, employed by :py:func:`extractIntegralData`

    Add to arrays dictionary the relevant statistics requested by the user
    through the RequestedStatistics list of special named strings.

    Parameters
    ----------

        arrays : dict
            Contains integral data in the following form:
            ::
                >>> np.array = arrays['FamilyBCNameOrElementName']['VariableName']

        IntegralDataName : str
            Name of the IntegralDataNode (CGNS) provided by elsA. It is used as
            key for loads dictionary.

        RequestedStatistics : :py:class:`list` of :py:class:`str`
            Desired statistics to infer from loads dictionary. For more
            information see documentation of function
            :py:func:`extractIntegralData`
    '''

    AvgIt = setup.ReferenceValues["CoprocessOptions"]["AveragingIterations"]

    arraysSubset = arrays[IntegralDataName]
    try:
        IterationNumber = arraysSubset['IterationNumber']
    except BaseException as e:
        return # this is the case for GlobalConvergenceHistory at present
    IterationWindow = len(IterationNumber[IterationNumber>(IterationNumber[-1]-AvgIt)])
    if IterationWindow < 2: return

    for StatKeyword in RequestedStatistics:
        KeywordsSplit = StatKeyword.split('-')
        StatType = KeywordsSplit[0]
        VarName = '-'.join(KeywordsSplit[1:])
        if VarName not in arraysSubset: continue

        try:
            InstantaneousArray = arraysSubset[VarName]
            InvalidValues = np.logical_not(np.isfinite(InstantaneousArray))
            InstantaneousArray[InvalidValues] = 0.
        except:
            continue

        if StatType.lower() == 'avg':
            StatisticArray = sliddingAverage(InstantaneousArray, IterationWindow)

        elif StatType.lower() == 'std':
            avg = sliddingAverage(InstantaneousArray, IterationWindow)
            arraysSubset['avg-'+VarName] = avg
            StatisticArray = sliddingSTD(InstantaneousArray, IterationWindow, avg=avg)

        elif StatType.lower() == 'rsd':
            avg = sliddingAverage(InstantaneousArray, IterationWindow)
            arraysSubset['avg-'+VarName] = avg
            std = sliddingSTD(InstantaneousArray, IterationWindow, avg=avg)
            arraysSubset['std-'+VarName] = std
            StatisticArray = sliddingRSD(InstantaneousArray, IterationWindow,
                                        avg=avg, std=std)

        arraysSubset[StatKeyword] = StatisticArray

def sliddingAverage(array, window):
    '''
    Compute the slidding average of the signal

    Parameters
    ----------

        array : numpy.ndarray
            input signal

        window : int
            length of the slidding window

    Returns
    -------

        average : numpy.ndarray
            sliding average
    '''
    average = uniform_filter1d(array, size=window)
    InvalidValues = np.logical_not(np.isfinite(average))
    average[InvalidValues] = 0.
    return average

def sliddingSTD(array, window, avg=None):
    '''
    Compute the slidding standard deviation of the signal

    Parameters
    ----------

        array : numpy.ndarray
            input signal

        window : int
            length of the slidding window

        avg : numpy.ndarray or :py:obj:`None`
            slidding average of **array** on the same **window**. If
            :py:obj:`None`, it is computed

    Returns
    -------

        std : numpy.ndarray
            sliding standard deviation
    '''
    if avg is None:
        avg = sliddingAverage(array, window)

    AvgSqrd = uniform_filter1d(array**2, size=window)

    InvalidValues = np.logical_not(np.isfinite(AvgSqrd))
    AvgSqrd[InvalidValues] = 0.
    AvgSqrd[AvgSqrd<0] = 0.

    std = np.sqrt(np.abs(AvgSqrd - avg**2))

    return std

def sliddingRSD(array, window, avg=None, std=None):
    '''
    Compute the relative slidding standard deviation of the signal

    .. math::

        rsd = std / avg

    Parameters
    ----------

        array : numpy.ndarray
            input signal

        window : int
            length of the slidding window

        average : numpy.ndarray or :py:obj:`None`
            slidding average of **array** on the same **window**. If
            :py:obj:`None`, it is computed

        std : numpy.ndarray or :py:obj:`None`
            slidding standard deviation of **array** on the same **window**. If
            :py:obj:`None`, it is computed

    Returns
    -------

        rsd : numpy.ndarray
            sliding relative standard deviation
    '''
    if avg is None:
        avg = sliddingAverage(array, window)
    if std is None:
        std = sliddingSTD(array, window, avg)

    rsd = std / avg

    InvalidValues = np.logical_not(np.isfinite(rsd))
    rsd[InvalidValues] = 0.

    return rsd

def getIntegralDataName(IntegralDataNode):
    '''
    Transforms the elsA provided **IntegralDataNode** name into a suitable name
    for further storing it at **arrays** dictionary.

    Parameters
    ----------

        IntegralDataNode : node
            Contains integral data as provided by elsA

    Returns
    -------

        IntegralDataName : str
            name of the integral quantity
    '''
    return I.getName(IntegralDataNode).split('-')[0]


def isConverged(ConvergenceCriteria):
    '''
    This method is used to determine if the current simulation is converged by
    looking at user-provided convergence criteria.
    If converged, the signal returns :py:obj:`True` to all ranks and writes a
    message to ``coprocess.log`` file.

    Parameters
    ----------

        ConvergenceCriteria : :py:class:`list` of :py:class:`dict`
            Each :py:class:`dict` corresponds to a criterion. Its has the
            following keys:

            * ``Family``: Name of the zone to monitor (shall exist in
              ``arrays.cgns``)

            * ``Variable``: Name of the variable to monitor on ``Family``

            * ``Threshold``: Value of the threshold to consider. The current
              criterion is satisfied if the value of the last element of
              ``Variable`` on ``Family`` in ``arrays.cgns`` is lower than
              ``Threshold``.

            * ``Condition`` (optinal, 'Necessary' by default): logical
              requirement for the current criterion. To verify convergence,
              criteria tagged 'Necessary' must all be satisfied simultaneously
              and at least one criterion tagged 'Sufficient' must be satisfied.
              For instance, if CN1 and CN2 are 'Necessary' and CS1 and CS2 are
              'Sufficient', convergence is reached when:
              (CN1 AND CN2) AND (CS1 OR CS2)

    Returns
    -------

        CONVERGED : bool
            :py:obj:`True` if the convergence criteria are satisfied
    '''
    if not ConvergenceCriteria:
        return False

    CONVERGED = False
    if rank == 0:
        AllNecessaryCriteria = True
        OneSufficientCriterion = True
        # Default value of Condition = 'Necessary'
        for criterion in ConvergenceCriteria:
            if 'Condition' not in criterion:
                criterion['Condition'] = 'Necessary'
            elif criterion['Condition'] == 'Sufficient':
                OneSufficientCriterion = False
        try:
            arraysTree = C.convertFile2PyTree(os.path.join(DIRECTORY_OUTPUT, FILE_ARRAYS))
            arraysZones = I.getZones(arraysTree)
            for criterion in ConvergenceCriteria:
                if OneSufficientCriterion and criterion['Condition'] == 'Sufficient':
                    continue
                zone, = [z for z in arraysZones if z[0] == criterion['Family']]
                Flux, = J.getVars(zone, [criterion['Variable']])
                if Flux is None and criterion['Condition'] == 'Necessary':
                    printCo('WARNING: requested convergence variable %s not found in %s'%(criterion['Variable'],criterion['Family']),color=WARN)
                    AllNecessaryCriteria = False
                    continue
                criterion['FoundValue'] = Flux[-1]
                IsSatisfied = criterion['FoundValue'] < criterion['Threshold']
                if criterion['Condition'] == 'Necessary' and not IsSatisfied:
                    AllNecessaryCriteria = False
                    break
                elif criterion['Condition'] == 'Sufficient' and IsSatisfied:
                    OneSufficientCriterion = criterion['Variable']

            CONVERGED = OneSufficientCriterion and AllNecessaryCriteria
            if CONVERGED:
                MSG = 'CONVERGED at iteration {} since:'.format(CurrentIteration-1)
                for criterion in ConvergenceCriteria:
                    if criterion['Condition'] == 'Necessary' \
                        or criterion['Variable'] == OneSufficientCriterion:
                        MSG += '\n  {}={} < {} on {} ({})'.format(criterion['Variable'],
                                                               criterion['FoundValue'],
                                                               criterion['Threshold'],
                                                               criterion['Family'],
                                                               criterion['Condition'])
                printCo('*******************************************',color=GREEN)
                printCo(MSG, color=GREEN)
                printCo('*******************************************',color=GREEN)

        except BaseException as e:
            printCo("isConverged failed: {}".format(e),color=FAIL)


    Cmpi.barrier()
    CONVERGED = comm.bcast(CONVERGED,root=0)

    return CONVERGED


def hasReachedTimeOutMargin(ElapsedTime, TimeOut, MarginBeforeTimeOut):
    '''
    This function returns :py:obj:`True` to all processors if the margin before
    time-out is satisfied. Otherwise, it returns :py:obj:`False` to all processors.

    Parameters
    ----------

        ElapsedTime : float
            total elapsed time in seconds since the launch of ``compute.py``
            elsA script.

        TimeOut : float
            total time-out in seconds. It shall correspond to the
            remaining time of a slurm job before forced termination.

        MarginBeforeTimeOut : float
            Margin in seconds before time out is reached. Shall account for safe
            postprocessing operations before the job triggers forced termination

    Returns
    -------

        ReachedTimeOutMargin : bool
            :py:obj:`True` if

            ::

                ElapsedTime >= (TimeOut - MarginBeforeTimeOut)

            Otherwise, returns :py:obj:`False`.

            .. note:: the same value (:py:obj:`True` or :py:obj:`False`) is sent to *all*
                processors.
    '''
    ReachedTimeOutMargin = False
    if rank == 0:
        ReachedTimeOutMargin = ElapsedTime >= (TimeOut - MarginBeforeTimeOut)
        if ReachedTimeOutMargin:
            printCo('REACHED MARGIN BEFORE TIMEOUT at %s'%datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                            proc=0, color=WARN)
    comm.Barrier()
    ReachedTimeOutMargin = comm.bcast(ReachedTimeOutMargin,root=0)

    return ReachedTimeOutMargin


def copyOutputFiles(*files2copy):
    '''
    Copy the files provided as input *(comma-separated variables)* by addding to
    their name ``'_AfterIter<X>'`` where ``<X>`` will be replaced with the
    corresponding interation

    Parameters
    ----------

        file2copy : comma-separated :py:class:`str`
            file(s) name(s) to copy at ``OUTPUT`` directory.

    Examples
    --------

    ::

        copyOutputFiles('surfaces.cgns','arrays.cgns')

    '''
    for file2copy in files2copy:
        f2cSplit = file2copy.split('.')
        newFileName = '{name}_AfterIter{it}.{fmt}'.format(
                        name='.'.join(f2cSplit[:-1]),
                        it=CurrentIteration-1,
                        fmt=f2cSplit[-1])
        try:
            shutil.copy2(file2copy, newFileName)
        except:
            pass


def computeTransitionOnsets(to):
    '''
    Extracts the airfoil's top and bottom side transition onset locations.

    .. important:: This function is adapted only to **Workflow Airfoil** cases.

    Parameters
    ----------

        to : PyTree
            Coupling tree as obtained from `adaptEndOfRun`

    Returns
    -------

        XtrTop : :py:class:`float`
            X-coordinate location of the transition onset location
            at the top side of the airfoil

        XtrBottom : :py:class:`float`
            X-coordinate location of the transition onset location
            at the bottom side of the airfoil
    '''
    XtrTop, XtrBottom = np.nan, np.nan
    surf   = boundaryConditions2Surfaces(to, onlyWalls=True)
    Cmpi._convert2PartialTree(surf)
    surf   = T.merge(surf)
    Slices = P.isoSurfMC(surf,'intermittency',value=0.5)

    # Really required ?
    TransitionPoints = T.splitConnexity(Slices) if len(Slices) > 0 else None

    # Each proc send to rank=0 the transition points found
    comm.Barrier()
    if rank == 0:
        TransitionPoints = comm.gather(TransitionPoints,0)
    else:
        comm.gather(TransitionPoints,0)

    # Post process the transition onset (rank=0)
    if rank == 0:
        TransPts2postProcess = []
        for p in TransitionPoints:
            if p is not None:
                isStdNode = I.isStdNode(p)
                if isStdNode == 0:
                    TransPts2postProcess += p
                elif isStdNode == -1:
                    TransPts2postProcess += [p]
        QtyOfTransitionOnsetPoints = len(TransPts2postProcess)
        if QtyOfTransitionOnsetPoints != 2:
            print(J.WARN+'WARNING: unexpected number of Transition onset points (%d, expected 2). Replacing by NaN'%QtyOfTransitionOnsetPoints+J.ENDC)
            TopPointXY = BottomPointXY = [np.array([np.nan]), np.array([np.nan])]
        else:
            TopPointXY    = J.getxy(TransPts2postProcess[0])
            BottomPointXY = J.getxy(TransPts2postProcess[1])

            if TopPointXY[1][0] < BottomPointXY[1][0]:
                TopPointXY, BottomPointXY = BottomPointXY, TopPointXY

        XtrTop    = TopPointXY[0][0]
        XtrBottom = BottomPointXY[0][0]
    comm.Barrier()

    return XtrTop, XtrBottom


def addArrays(arrays, ZoneName, ListOfArraysNames, NumpyArrays):
    '''
    This function is an interface for adding new user-defined arrays into the
    **arrays** Python dictionary.

    Parameters
    ----------

        arrays :dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array

            parameter **arrays** is modified

        ZoneName : str
            Name of the existing or new component where new arrays are going
            to be added. (FamilyBC or Component name)

        ListOfArraysNames : :py:class:`list` of :py:class:`str`
            Each element of this list is a name of the new arrays to be added.
            For example:

            ::

                ['MyFirstLoad', 'AnotherLoad']

        NumpyArrays : :py:class:`list` of numpy 1d arrays
            Values to be added to **arrays**

            .. attention::
                All arrays provided to **NumpyArrays** *(which belongs to the
                same component)* must have exactly the same number of elements
    '''
    try:
        arraysSubset = arrays[ZoneName]
    except KeyError:
        arrays[ZoneName] = {}
        arraysSubset = arrays[ZoneName]
        for array, name in zip(NumpyArrays, ListOfArraysNames):
            arraysSubset[name] = array.flatten()
        return

    for array, name in zip(NumpyArrays, ListOfArraysNames):
        try:
            ExistingArray = arraysSubset[name]
        except KeyError:
            arraysSubset[name] = array.flatten()
            continue

        arraysSubset[name] = np.hstack((ExistingArray.flatten(), array.flatten()))


def addBodyForcePropeller2Arrays(arrays, BodyForceDisks):
    '''
    This function is an interface adapted to body-force computations.
    It transfers the integral information of each body-force disk into
    the **arrays** dictionary.

    The fields that are appended to **arrays** are:

    ::

        ['Thrust', 'RPM', 'Power', 'Pitch']

    hence, these values must exist in ``.Info`` CGNS node of each CGNS zone
    contained in **BodyForceDisks**

    .. note:: The new component of load dictionary has the same name as its
        corresponding body-force disk.

    Parameters
    ----------

        arrays :dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array

            parameter **arrays** is modified

        BodyForceDisks : :py:class:`list` of zone
            Current bodyforce disks as obtained from
            :py:func:`MOLA.LiftingLine.computePropellerBodyForce`
    '''
    Cmpi.barrier()
    for BodyForceDisk in BodyForceDisks:
        BodyForceDiskName = I.getName( BodyForceDisk )
        Loads = J.get(BodyForceDisk, '.AzimutalAveragedLoads')
        if Loads:
            LoadsNames = [k for k in Loads]
            LoadsArrays = [np.array([Loads[k]]) for k in Loads]
            Commands = J.get(BodyForceDisk, '.Commands')
            if Commands:
                CommandsNames = [k for k in Commands]
                CommandsArrays = [np.array([Commands[k]]) for k in Commands]
                LoadsNames.extend(CommandsNames)
                LoadsArrays.extend(CommandsArrays)

            LoadsNames.append('IterationNumber')
            LoadsArrays.append(np.array([CurrentIteration]))
            addArrays(arrays, BodyForceDiskName, LoadsNames, LoadsArrays)
    Cmpi.barrier()


def getSignal(filename):
    '''
    Get a signal using an temporary auxiliar file technique.

    If the intermediary file exists (signal received) then it is removed, and
    the function returns :py:obj:`True` to all processors. Otherwise, it returns
    :py:obj:`False` to all processors.

    This function is employed for controling a simulation in a simple manner,
    for example using UNIX command ``touch``:

    .. code-block:: bash

        touch filename

    at the same directory where :py:func:`getSignal` is called.

    Parameters
    ----------

        filename : str
            the name of the file (the signal keyword)

    Returns
    -------

        isOrder : bool
            :py:obj:`True` if the signal is received, otherwise :py:obj:`False`, to all
            processors
    '''
    isOrder = False
    if rank == 0:
        try:
            os.remove(filename)
            isOrder = True
            printCo(CYAN+"Received signal %s"%filename+ENDC, proc=0)
        except:
            pass
    comm.Barrier()
    isOrder = comm.bcast(isOrder,root=0)
    return isOrder


def adaptEndOfRun(to):
    '''
    This function is used to make adaptations of the coupling trigger tree
    provided by elsA. The following operations are performed:

    * ``GridCoordinates`` node is created from ``FlowSolution#EndOfRun#Coords``
    * adapt name of masking field (``cellnf`` is renamed as ``cellN``)
    * rename ``FlowSolution#EndOfRun`` as ``FlowSolution#Init``

    Parameters
    ----------

         to : PyTree
            Coupling tree as obtained from function

            >>> elsAxdt.get(elsAxdt.OUTPUT_TREE)

            .. note:: tree **to** is modified
    '''
    moveCoordsFromEndOfRunToGridCoords(to)
    I._renameNode(to, 'cellnf', 'cellN')
    I._renameNode(to, 'FlowSolution#EndOfRun', 'FlowSolution#Init')


def moveCoordsFromEndOfRunToGridCoords(to):
    '''
    This function is used to make adaptations of the coupling trigger tree
    provided by elsA. The following operations are performed:

    * ``GridCoordinates`` node is created from ``FlowSolution#EndOfRun#Coords``


    Parameters
    ----------

         to : PyTree
            Coupling tree as obtained from function

            >>> elsAxdt.get(elsAxdt.OUTPUT_TREE)

            .. note:: tree **to** is modified
    '''
    FScoords = I.getNodeFromName(to, 'FlowSolution#EndOfRun#Coords')
    if FScoords:
        I._renameNode(to,'FlowSolution#EndOfRun#Coords','GridCoordinates')
        for GridCoordsNode in I.getNodesFromName3(to, 'GridCoordinates'):
            GridLocationNode = I.getNodeFromType1(GridCoordsNode, 'GridLocation_t')
            if I.getValue(GridLocationNode) != 'Vertex':
                zone = I.getParentOfNode(to, GridCoordsNode)
                ERRMSG = ('Extracted coordinates of zone '
                          '%s must be located in Vertex')%I.getName(zone)
                raise ValueError(FAIL+ERRMSG+ENDC)
            I.rmNode(to, GridLocationNode)
            I.setType(GridCoordsNode, 'GridCoordinates_t')
    Cmpi.barrier()

def boundaryConditions2Surfaces(to, onlyWalls=True):
    '''
    Extract the BC data contained in the coupling tree as a PyTree.

    Parameters
    ----------

        to : PyTree
            Coupling tree as obtained from :py:func:`adaptEndOfRun`

        onlyWalls : bool
            if :py:obj:`True`, only BC with keyword ``'wall'`` contained in
            their type are extracted. Otherwise, all BC are extracted,
            regardless of their type.

    Returns
    -------

        BCs : PyTree
            PyTree with one base by BC Family. Include fields stored in
            FlowSolution containers
    '''
    Cmpi.barrier()
    tR = I.renameNode(to, 'FlowSolution#Init', 'FlowSolution#Centers')
    I.__FlowSolutionCenters__ = 'FlowSolution#Centers'
    # TODO make it multi-container of FlowSolution
    DictBCNames2Type = C.getFamilyBCNamesDict(tR)

    BCs = I.newCGNSTree()
    for FamilyName in DictBCNames2Type:
        BCType = DictBCNames2Type[FamilyName]
        BC = None
        if onlyWalls:
            if 'wall' in BCType.lower():
                BC = C.extractBCOfName(tR,'FamilySpecified:'+FamilyName)
        else:
            BC = C.extractBCOfName(tR,'FamilySpecified:'+FamilyName)
        if BC:
            base = I.newCGNSBase(FamilyName, 2, 3, parent=BCs)
            for bc in BC:
                I.addChild(base, bc)

    Cmpi.barrier()

    return BCs

def renameTooLongZones(to, n=25):
    '''
    .. warning:: this is a private function, employed by :py:func:`saveSurfaces`

    This function rename zones in a PyTree **to** if their names are too long
    to be save in a CGNS file (maximum length = 32 characters).

    The new name of a zone follows this format:
    ``<NewName>`` = ``<First <n> characters of old name>_<ID>``
    with ``<n>`` an integer and ``<ID>`` the lowest integer (starting form 0) such as
    ``<NewName>`` does not already exist in the PyTree.

    Parameters
    ----------

        to : PyTree
            PyTree to check. Zones with a too long name will be renamed.

            .. note:: tree **to** is modified

        n : int
            Number of characters to keep in the old zone name.
    '''
    for zone in I.getZones(to):
        zoneName = I.getName(zone)
        if len(zoneName) > 32:
            CurrentZoneNames = [I.getName(z) for z in I.getZones(to)]
            c = 0
            newName = '{}_{}'.format(zoneName[:n+1], c)
            while newName in CurrentZoneNames and c < 1000:
                c += 1
                newName = '{}_{}'.format(zoneName[:n+1], c)
            if c == 1000:
                ERRMSG = 'Zone {} has not been renamed by renameTooLongZones() but its length ({}) is greater than maximum authorized length (32)'.format(zoneName, len(zoneName))
                raise ValueError(FAIL+ERRMSG+ENDC)
            I.setName(zone, newName)

def reshapeBCDatasetNodes(to):
    '''
    ..warning:: this is a private function, employed by :py:func:`saveSurfaces`

    This function checks the shape of DataArray in all ``BCData_t`` nodes in a
    PyTree **to**.
    For some unknown reason, the extraction of BCData throught a BCDataSet is
    done sometimes in unstructured 1D shape, so it is not consistent with BC
    PointRange. If so, this function reshape the DataArray to the BC shape.
    Link to ``Anomaly #6186`` (see <https://elsa-e.onera.fr/issues/6186>`_)

    Parameters
    ----------

        to : PyTree
            PyTree to check.

            .. note:: tree **to** is modified
    '''
    def _getBCShape(bc):
        PointRange_n = I.getNodeFromName(bc, 'PointRange')
        if not PointRange_n: return
        PointRange = I.getValue(PointRange_n)
        imin = PointRange[0, 0]
        imax = PointRange[0, 1]
        jmin = PointRange[1, 0]
        jmax = PointRange[1, 1]
        kmin = PointRange[2, 0]
        kmax = PointRange[2, 1]
        if imin == imax:
            bc_shape = (jmax-jmin, kmax-kmin)
        elif jmin == jmax:
            bc_shape = (imax-imin, kmax-kmin)
        else:
            bc_shape = (imax-imin, jmax-jmin)
        return bc_shape

    for bc in I.getNodesFromType(to, 'BC_t'):
        bc_shape = _getBCShape(bc)
        if not bc_shape: continue
        for BCData in I.getNodesFromType(bc, 'BCData_t'):
            for node in I.getNodesFromType(BCData, 'DataArray_t'):
                value = I.getValue(node)
                if isinstance(value, np.ndarray):
                    if value.shape != bc_shape:
                        I.setValue(node, value.reshape(bc_shape, order='F'))


def getOption(OptionName, default=None):
    '''
    This function is an interface for easily accessing the values of dictionary
    ``ReferenceValues['CoprocessOptions']`` contained in ``setup.py``.

    Parameters
    ----------

        OptionName : str
            Name of the key to return.

        default
            value to return if **OptionName** key is not present in
            ``ReferenceValues['CoprocessOptions']`` dictionary

    Returns
    -------

        value
            the value contained in
            ``ReferenceValues['CoprocessOptions'][<OptionName>]`` or the default
            value if key is not found
    '''
    try:
        value = setup.ReferenceValues['CoprocessOptions'][OptionName]
    except KeyError:
        value = default

    return value

def write4Debug(MSG):
    '''
    This function allows for writing into separated files (one per processor)
    named ``LOGS/rank<P>.log`` where ``<P>`` is replaced with the proc number.
    It is useful for debugging.

    Parameters
    ----------

        MSG : str
            Message to append into appropriate file following the local
            proc number.
    '''
    with open('LOGS/rank%d.log'%rank,'a') as f: f.write('%s\n'%MSG)

def loadSkeleton(Skeleton=None, PartTree=None):
    '''
    Load the skeleton tree (if not given) and add nodes that are required for
    coprocessing.

    Parameters
    ----------

        Skeleton : PyTree or :py:obj:`None`
            Skeleton tree, got from ``Cmpi.convertFile2SkeletonTree`` with
            Cassiopee or ``PyPartBase.getPyPartSkeletonTree`` with PyPart.
            If :py:obj:`None`, load the Skeleton tree with
            ``Cmpi.convertFile2SkeletonTree(FILE_CGNS)``.

        PartTree : PyTree or :py:obj:`None`
            Partial tree, got from ``Cmpi.convertFile2PyTree(..., proc=rank)``
            with Cassiopee or ``PyPartBase.runPyPart`` with PyPart.
            If :py:obj:`None`, only the needed nodes are read with
            ``Converter.Filter``.

    Returns
    -------

        Skeleton : PyTree
            Skeleton Tree with additional information, required for Coprocess
            functions.
    '''
    addCoordinates = True
    if not Skeleton: Skeleton = Cmpi.convertFile2SkeletonTree(FILE_CGNS)

    FScoords = I.getNodeFromName1(Skeleton, 'FlowSolution#EndOfRun#Coords')
    if FScoords: addCoordinates = False

    I._rmNodesByName(Skeleton, 'FlowSolution*')
    I._rmNodesByName(Skeleton, 'ID_*')

    if PartTree:
        # Needed nodes are read from PartTree
        def readNodesFromPaths(path):
            split_path = path.split('/')
            path_begining = '/'.join(split_path[:-1])
            name = split_path[-1]
            parent = I.getNodeFromPath(PartTree, path_begining)
            return I.getNodesFromName(parent, name)

        FScoords = I.getNodeFromName1(PartTree, 'FlowSolution#EndOfRun#Coords')
        if FScoords: addCoordinates = False
    else:
        # Needed nodes are read from FILE_CGNS with Converter.Filter
        def readNodesFromPaths(path):
            return Filter.readNodesFromPaths(FILE_CGNS, [path])


    def replaceNodeByName(parent, parentPath, name):
        oldNode = I.getNodeFromName1(parent, name)
        newNode = readNodesFromPaths('{}/{}'.format(parentPath, name))
        I._rmNode(parent, oldNode)
        I._addChild(parent, newNode)

    for base in I.getBases(Skeleton):
        basename = I.getName(base)
        for zone in I.getNodesFromType1(base, 'Zone_t'):
            # Only for local zones on proc
            proc = I.getValue(I.getNodeFromName(zone, 'proc'))
            if proc != rank: continue

            zonePath = '{}/{}'.format(basename, I.getName(zone))

            # Coordinates
            if addCoordinates: replaceNodeByName(zone, zonePath, 'GridCoordinates')

            replaceNodeByName(zone, zonePath, 'FlowSolution#Height')

            replaceNodeByName(zone, zonePath, ':CGNS#Ppart')

            # For unstructured mesh
            if I.getZoneType(zone) == 2: # unstructured zone
                replaceNodeByName(zone, zonePath, ':elsA#Hybrid')
                # TODO: Add other types of Elements_t nodes if needed
                replaceNodeByName(zone, zonePath, 'NGonElements')
                replaceNodeByName(zone, zonePath, 'NFaceElements')
                # PointList in BCs and GridConnectivities
                for BC in I.getNodesFromType2(zone, 'BC_t'):
                    BCpath = '{}/ZoneBC/{}'.format(zonePath, I.getName(BC))
                    replaceNodeByName(BC, BCpath, 'PointList')
                for GC in I.getNodesFromType2(zone, 'GridConnectivity_t'):
                    GCpath = '{}/ZoneGridConnectivity/{}'.format(zonePath, I.getName(GC))
                    replaceNodeByName(GC, GCpath, 'PointList')

    return Skeleton

def splitWithPyPart():
    '''
    Use PyPart to split the mesh in ``main.cgns``. This function should be use
    in ``compute.py`` to prepare the mesh before calling ``elsAxdt.XdtCGNS()``.

    .. note:: For more details on PyPart, see the dedicated pages on elsA
        support:
        `PyPart alone <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/PreprocessTutorials/etc_pypart_alone.html>`_
        and
        `PyPart with elsA <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/PreprocessTutorials/etc_pypart_elsa.html>`_

    .. important:: Dependence to ETC module

    Returns
    -------

        t : PyTree
            Split tree, merged with the skeleton. It will be the **tree**
            argument of ``elsAxdt.XdtCGNS()`` in ``compute.py``

        Skeleton : PyTree
            Skeleton tree to use in ``coprocess.py``

        PyPartBase : PyPart object
            PyPart objet that is mandatory to use its method mergeAndSave latter

        Distribution : dict
            Correspondence between zones and processors.

    '''

    import etc.pypart.PyPart     as PPA

    PyPartBase = PPA.PyPart(FILE_CGNS,
                            lksearch=[DIRECTORY_OUTPUT, '.'],
                            loadoption='partial',
                            mpicomm=comm,
                            LoggingInFile=False,
                            LoggingFile='{}/partTree'.format(DIRECTORY_LOGS),
                            LoggingVerbose=40  # Filter: None=0, DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
                            )
    PartTree = PyPartBase.runPyPart(method=2, partN=1, reorder=[4, 3])
    PyPartBase.finalise(PartTree, savePpart=True, method=1)
    Skeleton = PyPartBase.getPyPartSkeletonTree()
    Distribution = PyPartBase.getDistribution()

    # Put Distribution into the Skeleton
    for zone in I.getZones(Skeleton):
        zonePath = I.getPath(Skeleton, zone, pyCGNSLike=True)[1:]
        Cmpi._setProc(zone, Distribution[zonePath])

    t = I.merge([Skeleton, PartTree])

    Skeleton = loadSkeleton(Skeleton, PartTree)
    # Add empty Coordinates for skeleton zones
    # Needed to make Cmpi.convert2PartialTree work
    for zone in I.getZones(Skeleton):
        GC = I.getNodeFromType1(zone, 'GridCoordinates_t')
        if not GC:
            J.set(zone, 'GridCoordinates', childType='GridCoordinates_t',
                CoordinateX=None, CoordinateY=None, CoordinateZ=None)
        elif I.getZoneType(zone) == 2:
            # For unstructured zone, correct the node NFaceElements/ElementConnectivity
            # Problem with PyPart: see issue https://elsa-e.onera.fr/issues/9002
            # C._convertArray2NGon(zone)
            NFaceElements = I.getNodeFromName(zone, 'NFaceElements')
            if NFaceElements:
                node = I.getNodeFromName(NFaceElements, 'ElementConnectivity')
                I.setValue(node, np.abs(I.getValue(node)))

    return t, Skeleton, PyPartBase, Distribution

def moveLogFiles():
    if Cmpi.rank == 0:
        try: os.makedirs(DIRECTORY_LOGS)
        except: pass

        for fn in glob.glob('*.log'):
            FilenameBase = fn[:-4]
            i = 1
            NewFilename = FilenameBase+'-%d'%i+'.log'
            while JM.fileExists('LOGS', NewFilename):
                i += 1
                NewFilename = FilenameBase+'-%d'%i+'.log'

            shutil.move(fn, os.path.join('LOGS', NewFilename))
        for fn in glob.glob('elsA_MPI*'):
            shutil.move(fn, os.path.join('LOGS', fn))

    Cmpi.barrier()
