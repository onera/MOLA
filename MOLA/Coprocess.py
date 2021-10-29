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
from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
NProcs = comm.Get_size()


import Converter.PyTree as C
import Converter.Internal as I
import Converter.Filter as Filter
import Converter.Mpi as Cmpi
import Transform.PyTree as T
import Post.PyTree as P

from . import InternalShortcuts as J
from . import Preprocess as PRE

# ------------------------------------------------------------------ #
# Following variables should be overridden using compute.py and coprocess.py
# scripts
FULL_CGNS_MODE   = False
FILE_SETUP       = 'setup.py'
FILE_CGNS        = 'main.cgns'
FILE_SURFACES    = 'surfaces.cgns'
FILE_LOADS       = 'loads.cgns'
FILE_FIELDS      = 'fields.cgns'
FILE_COLOG       = 'coprocess.log'
FILE_BODYFORCESRC= 'BodyForceSources.cgns'
DIRECTORY_OUTPUT = 'OUTPUT'
setup            = None
CurrentIteration = 0
elsAxdt = None
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
    preffix = '[%d]: '%rank
    if color:
        message = color+message+ENDC
    with open(FILE_COLOG, 'a') as f:
        f.write(preffix+message+'\n')

def extractSurfaces(OutputTreeWithSkeleton, Extractions):

    cellDimOutputTree = I.getZoneDim(I.getZones(OutputTreeWithSkeleton)[0])[-1]

    def addBase2SurfacesTree(basename):
        if not zones: return
        base = I.newCGNSBase(basename, cellDim=cellDimOutputTree-1, physDim=3,
            parent=SurfacesTree)
        I._addChild(base, zones)
        J.set(base, '.ExtractionInfo', **Extraction)
        return base

    t = I.renameNode(OutputTreeWithSkeleton, 'FlowSolution#Init', 'FlowSolution#Centers')
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

        if TypeOfExtraction.startswith('AllBC'):
            BCFilterName = TypeOfExtraction.replace('AllBC','')
            for BCFamilyName in DictBCNames2Type:
                BCType = DictBCNames2Type[BCFamilyName]
                if BCFilterName.lower() in BCType.lower():
                    zones = C.extractBCOfName(t,'FamilySpecified:'+BCFamilyName)
                    addBase2SurfacesTree(BCFamilyName)

        elif TypeOfExtraction.startswith('BC'):
            zones = C.extractBCOfType(t, TypeOfExtraction)
            try: basename = Extraction['name']
            except KeyError: basename = TypeOfExtraction
            addBase2SurfacesTree(basename)

        elif TypeOfExtraction.startswith('FamilySpecified:'):
            zones = C.extractBCOfName(t, TypeOfExtraction)
            try: basename = Extraction['name']
            except KeyError: basename = TypeOfExtraction.replace('FamilySpecified:','')
            addBase2SurfacesTree(basename)

        elif TypeOfExtraction == 'IsoSurface':
            zones = P.isoSurfMC(PartialTree, Extraction['field'], Extraction['value'])
            try: basename = Extraction['name']
            except KeyError:
                FieldName = Extraction['field'].replace('Coordinate','')
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
    restoreFamilies(SurfacesTree, OutputTreeWithSkeleton)
    Cmpi.barrier()

    return SurfacesTree

def extractIntegralData(to, loads, Extractions=[],
                        DesiredStatistics=['std-CL', 'std-CD']):
    '''
    Extract integral data from coupling tree **to**, and update **loads** Python
    dictionary adding statistics requested by the user.

    Parameters
    ----------

        to : PyTree
            Coupling tree as obtained from :py:func:`adaptEndOfRun`

        loads : dict
            Contains integral data in the following form:

            >>> loads['FamilyBCNameOrElementName']['VariableName'] = np.array

        DesiredStatistics : :py:class:`list` of :py:class:`str`
            Here, the user requests the additional statistics to be computed.
            The syntax of each quantity must be as follows:

            ::

                '<preffix>-<integral_quantity_name>'

            `<preffix>` can be ``'avg'`` (for cumulative average) or ``'std'``
            (for standard deviation). ``<integral_quantity_name>`` can be any
            quantity contained in loads, including other statistics.

            .. hint:: chaining preffixes is perfectly accepted, like
                ``'std-std-CL'`` which would compute the cumulative standard
                deviation of the cumulative standard deviation of the
                lift coefficient (:math:`\sigma(\sigma(C_L))`)

    '''
    IntegralDataNodes = I.getNodesFromType2(to, 'IntegralData_t')
    for IntegralDataNode in IntegralDataNodes:
        IntegralDataName = getIntegralDataName(IntegralDataNode)
        _appendIntegralDataNode2Loads(loads, IntegralDataNode)
        _extendLoadsWithProjectedLoads(loads, IntegralDataName)
        _extendLoadsWithStatistics(loads, IntegralDataName, DesiredStatistics)

    return loads

def save(t, filename, tagWithIteration=False):
    '''
    Generic function to save a PyTree **t** in parallel. Works whatever the
    dimension of the PyTree. Use it to save ``'fields.cgns'``,
    ``'surfaces.cgns'`` or ``'loads.cgns'``.

    Parameters
    ----------

        t : PyTree
            tree to save

        filename : str
            Name of the file

        tagWithIteration : bool
            if ``True``, adds a suffix ``_AfterIter<iteration>``
            to the saved filename (creates a copy)
    '''
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

def monitorTurboPerformance(surfaces, loads, DesiredStatistics=[]):
    '''
    Monitor performance (massflow in/out, total pressure ratio, total
    temperature ratio, isentropic efficiency) for each row in a compressor
    simulation. This processing is triggered if at least two bases in the PyTree
    **surfaces** fill the following requirements:

        #. there is a node ``'.ExtractionInfo'`` of type ``'UserDefinedData_t'``
        #. it contains a node ``'ReferenceRow'``, whose value is a
            :py:class:`str` corresponding to a row Family in ``'main.cgns'``.
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

        loads : dict
            Contains integral data in the following form:

            >>> loads['FamilyBCNameOrElementName']['VariableName'] = np.array

        DesiredStatistics : :py:class:`list` of :py:class:`str`
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
        if rowParams['RotationSpeed'] != 0:
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
            fluxcoeff = rowParams['NumberOfBlades'] / rowParams['NumberOfBladesSimulated']
            if IsRotor:
                perfos = computePerfoRotor(dataUpstream, dataDownstream, fluxcoeff=fluxcoeff)
            else:
                perfos = computePerfoStator(dataUpstream, dataDownstream, fluxcoeff=fluxcoeff)
            appendDict2Loads(loads, perfos, 'PERFOS_{}'.format(row))
            _extendLoadsWithStatistics(loads, 'PERFOS_{}'.format(row), DesiredStatistics)

    loadsTree = loadsDict2PyTree(loads)
    save(loadsTree, os.path.join(DIRECTORY_OUTPUT, FILE_LOADS))

def computePerfoRotor(dataUpstream, dataDownstream, fluxcoeff=1., fluxcoeffOut=None):

    if not fluxcoeffOut:
        fluxcoeffOut = fluxcoeff

    gamma = setup.FluidProperties['Gamma']

    # Compute total quantities ratio between in/out planes
    meanPtIn  =      dataUpstream['PressureStagnation'] /   dataUpstream['Massflow']
    meanPtOut =    dataDownstream['PressureStagnation'] / dataDownstream['Massflow']
    meanTtIn  =   dataUpstream['TemperatureStagnation'] /   dataUpstream['Massflow']
    meanTtOut = dataDownstream['TemperatureStagnation'] / dataDownstream['Massflow']
    PtRatio = meanPtOut / meanPtIn
    TtRatio = meanTtOut / meanTtIn
    # Compute Isentropic Efficiency
    etaIs = (PtRatio**((gamma-1.)/gamma) - 1.) / (TtRatio - 1.)

    perfos = dict(
        IterationNumber            = CurrentIteration-1,  # Because extraction before current iteration (next_state=16)
        MassflowIn                 = dataUpstream['Massflow']*fluxcoeff,
        MassflowOut                = dataDownstream['Massflow']*fluxcoeffOut,
        PressureStagnationRatio    = PtRatio,
        TemperatureStagnationRatio = TtRatio,
        EfficiencyIsentropic       = etaIs
    )

    return perfos

def computePerfoStator(dataUpstream, dataDownstream, fluxcoeff=1., fluxcoeffOut=None):

    if not fluxcoeffOut:
        fluxcoeffOut = fluxcoeff

    meanPtIn  =   dataUpstream['PressureStagnation'] /   dataUpstream['Massflow']
    meanPtOut = dataDownstream['PressureStagnation'] / dataDownstream['Massflow']
    meanPsIn  =             dataUpstream['Pressure'] /       dataUpstream['Area']

    perfos = dict(
        IterationNumber         = CurrentIteration-1,  # Because extraction before current iteration (next_state=16)
        MassflowIn              = dataUpstream['Massflow']*fluxcoeff,
        MassflowOut             = dataDownstream['Massflow']*fluxcoeffOut,
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
                >>> VarAndMeanList = [([var1, var2], meanFunction1), ([var3], meanFunction2), ...]

            Example of function for a massflow weighted integration:

            ::
                >>> import Converter.PyTree as C
                >>> import Post.PyTree      as P
                >>> def massflowWeightedIntegral(t, var):
                >>>     t = C.initVars(t, 'rou_var={MomentumX}*{%s}'%(var))
                >>>     integ  = abs(P.integNorm(t, 'rou_var')[0][0])
                >>>     return integ

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
        for var in ['Massflow', 'Area']:
            data[var] = 0
        for varList, meanFunction in VarAndMeanList:
            for var in varList:
                data[var] = 0
    else:
        C._initVars(surface, 'ones=1')
        data['Area']     = abs(P.integNorm(surface, var='ones')[0][0])
        data['Massflow'] = abs(P.integNorm(surface, var='MomentumX')[0][0])
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
    data['Massflow'] = comm.allreduce(data['Massflow'], op=MPI.SUM)
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
        PRE.writeSetupFromModuleObject(setup)
        printCo('updating setup.py ... OK', proc=0, color=GREEN)
    comm.Barrier()

def invokeLoads():
    '''
    Create **loads** Python dictionary by reading any pre-existing data
    contained in ``OUTPUT/loads.cgns``

    .. note:: an empty dictionary is returned if no ``OUTPUT/loads.cgns`` file
        is found

    Returns
    -------

        loads : dict
            Contains integral data in the following form:

            >>> loads['FamilyBCNameOrElementName']['VariableName'] = np.array
    '''
    Cmpi.barrier()
    loads = dict()
    FullPathLoadsFile = os.path.join(DIRECTORY_OUTPUT, FILE_LOADS)
    ExistingLoadsFile = os.path.exists(FullPathLoadsFile)
    Cmpi.barrier()
    inititer = setup.elsAkeysNumerics['inititer']
    if ExistingLoadsFile and inititer>1:
        t = Cmpi.convertFile2SkeletonTree(FullPathLoadsFile)
        t = Cmpi.readZones(t, FullPathLoadsFile, rank=rank)
        Cmpi._convert2PartialTree(t, rank=rank)

        for zone in I.getZones(t):
            ZoneName = I.getName(zone)
            VarNames, = C.getVarNames(zone, excludeXYZ=True)
            FlowSol_n = I.getNodeFromName1(zone, 'FlowSolution')
            loads[ZoneName] = dict()
            loadsSubset = loads[ZoneName]
            if FlowSol_n:
                for VarName in VarNames:
                    Var_n = I.getNodeFromName1(FlowSol_n, VarName)
                    if Var_n:
                        loadsSubset[VarName] = Var_n[1]

            try:
                iters = np.copy(loadsSubset['IterationNumber'])
                for VarName in loadsSubset:
                    loadsSubset[VarName] = loadsSubset[VarName][iters<inititer]
            except KeyError:
                pass

    Cmpi.barrier()

    return loads

def addMemoryUsage2Loads(loads):
    '''
    This function adds or updates a component in **loads** for monitoring the
    employed memory. Only nodes are monitored (not every single proc, as this
    would produce redundant information). The number of cores contained in each
    computational node is retreived from the user-specified variable contained
    in ``setup``:

    ::

        setup.ReferenceValues['CoreNumberPerNode']

    If this information does not exist, a value of ``28`` is taken by default.

    Parameters
    ----------

        loads : dict
            Contains integral data in the following form:

            >>> loads['FamilyBCNameOrElementName']['VariableName'] = np.array

            parameter **loads** is modified
    '''

    try: CoreNumberPerNode = setup.ReferenceValues['CoreNumberPerNode']
    except: CoreNumberPerNode = 28

    if rank%CoreNumberPerNode == 0:
        ZoneName = 'MemoryUsageOfProc%d'%rank
        UsedMemory = psutil.virtual_memory().used
        UsedMemoryPctg = psutil.virtual_memory().percent
        try:
            LoadsItem = loads[ZoneName]
        except KeyError:
            loads[ZoneName] = dict(IterationNumber=np.array([],dtype=int),
                                   UsedMemoryInPercent=np.array([],dtype=float),
                                   UsedMemory=np.array([],dtype=float),)
            LoadsItem = loads[ZoneName]

        try:
            LoadsItem['IterationNumber'] = np.hstack((LoadsItem['IterationNumber'],
                                                      int(CurrentIteration)))
            LoadsItem['UsedMemoryInPercent'] = np.hstack((LoadsItem['UsedMemoryInPercent'],
                                                          float(UsedMemoryPctg)))
            LoadsItem['UsedMemory'] = np.hstack((LoadsItem['UsedMemory'],
                                             float(UsedMemory)))
        except KeyError:
            del loads[ZoneName]
    Cmpi.barrier()

def loadsDict2PyTree(loads):
    '''
    This function converts the **loads** Python dictionary to a PyTree (CGNS)
    structure **t**.

    Parameters
    ----------

        loads : dict
            Contains integral data in the following form:

            >>> loads['FamilyBCNameOrElementName']['VariableName'] = np.array

    Returns
    -------

        t : PyTree
            same information as input, but structured in a PyTree CGNS form

    .. warning:: after calling the function, **loads** and **t** do *NOT*
        share memory, which means that modifications on **loads** will not
        affect **t** and vice-versa
    '''
    zones = []
    for ZoneName in loads:
        loadsSubset = loads[ZoneName]
        Arrays, Vars = [], []
        OrderedVars = [var for var in loadsSubset]

        OrderedVars.sort()
        for var in OrderedVars:
            Vars.append(var)
            Arrays.append(loadsSubset[var])

        zone = J.createZone(ZoneName, Arrays, Vars)
        if zone: zones.append(zone)

    if zones:
        Cmpi._setProc(zones, rank)
        t = C.newPyTree(['Base', zones])
    else:
        t = C.newPyTree(['Base'])
    Cmpi.barrier()

    return t

def appendDict2Loads(loads, dictToAppend, basename):
    '''
    This function add data defined in **dictToAppend** in the base **basename**
    of **loads**.

    Parameters
    ----------

        loads : dict
            Contains integral data in the following form:

            >>> loads[basename]['VariableName'] = np.array

        dictToAppend : dict
            Contains data to append in **loads**. For each element:
                * key is the variable name
                * value is the associated value

        basename : str
            Name of the base in which values will be appended.

    '''
    if not basename in loads:
        loads[basename] = dict()

    for var, value in dictToAppend.items():
        if var in loads[basename]:
            loads[basename][var] = np.append(loads[basename][var], value)
        else:
            loads[basename][var] = np.array([value])


def _appendIntegralDataNode2Loads(loads, IntegralDataNode):
    '''
    Beware: this is a private function, employed by updateAndSaveLoads()

    This function converts the CGNS IntegralDataNode (as provided by elsA)
    into the Python dictionary structure of loads dictionary, and append it
    to the latter.

    Parameters
    ----------

        loads : dict
            Contains integral data in the following form:

            >>> loads['FamilyBCNameOrElementName']['VariableName'] = np.array

        IntegralDataNode : node
            Contains integral data as provided by elsA
    '''
    IntegralData = dict()
    for DataArrayNode in I.getChildren(IntegralDataNode):
        IntegralData[I.getName(DataArrayNode)] = I.getValue(DataArrayNode)
    IntegralDataName = getIntegralDataName(IntegralDataNode)
    IterationNumber = IntegralData['IterationNumber']

    try:
        loadsSubset = loads[IntegralDataName]
    except KeyError:
        loads[IntegralDataName] = dict()
        loadsSubset = loads[IntegralDataName]


    try: RegisteredIterations = loadsSubset['IterationNumber']
    except KeyError: RegisteredIterations = np.array([])
    if len(RegisteredIterations) > 0:
        PreviousRegisteredLoads = True
        eps = 1e-12
        UpdatePortion = IterationNumber > (RegisteredIterations[-1] + eps)
        try: FirstIndex2Update = np.where(UpdatePortion)[0][0]
        except IndexError: return
    else:
        PreviousRegisteredLoads = False

    for integralKey in IntegralData:
        if PreviousRegisteredLoads:
            PreviousArray = loadsSubset[integralKey]
            AppendArray = IntegralData[integralKey][FirstIndex2Update:]
            loadsSubset[integralKey] = np.hstack((PreviousArray, AppendArray))
        else:
            loadsSubset[integralKey] = np.array(IntegralData[integralKey],
                                               order='F', ndmin=1)


def _extendLoadsWithProjectedLoads(loads, IntegralDataName):
    '''
    Beware: this is a private function, employed by :py:func:`updateAndSaveLoads`

    This function is employed for adding aerodynamic-relevant coefficients to
    the loads dictionary. The new quantites are the following :

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
                ReferenceValues['IntegralScales'][<IntegralDataName>]

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

    INPUTS

    loads - (Python dictionary) - Contains integral data in the following form:
        np.array = loads['FamilyBCNameOrElementName']['VariableName']

    IntegralDataName - (string) - Name of the IntegralDataNode (CGNS) provided
        by elsA. It is used as key for loads dictionary.
    '''


    DragDirection=np.array(setup.ReferenceValues['DragDirection'],dtype=np.float)
    SideDirection=np.array(setup.ReferenceValues['SideDirection'],dtype=np.float)
    LiftDirection=np.array(setup.ReferenceValues['LiftDirection'],dtype=np.float)

    loadsSubset = loads[IntegralDataName]

    try:
        FluxCoef = setup.ReferenceValues['IntegralScales'][IntegralDataName]['FluxCoef']
        TorqueCoef = setup.ReferenceValues['IntegralScales'][IntegralDataName]['TorqueCoef']
        TorqueOrigin = setup.ReferenceValues['IntegralScales'][IntegralDataName]['TorqueOrigin']
    except:
        FluxCoef = setup.ReferenceValues['FluxCoef']
        TorqueCoef = setup.ReferenceValues['TorqueCoef']
        TorqueOrigin = setup.ReferenceValues['TorqueOrigin']

    try:
        FX = loadsSubset['MomentumXFlux']
        FY = loadsSubset['MomentumYFlux']
        FZ = loadsSubset['MomentumZFlux']
        MX = loadsSubset['TorqueX']
        MY = loadsSubset['TorqueY']
        MZ = loadsSubset['TorqueZ']
    except KeyError:
        return # no required fields for computing external aero coefficients

    # Pole change
    # TODO make ticket for elsA concerning CGNS parsing of xtorque ytorque ztorque
    TX = MX-(TorqueOrigin[1]*FZ - TorqueOrigin[2]*FY)
    TY = MY-(TorqueOrigin[2]*FX - TorqueOrigin[0]*FZ)
    TZ = MZ-(TorqueOrigin[0]*FY - TorqueOrigin[1]*FX)

    loadsSubset['CL']=FX*LiftDirection[0]+FY*LiftDirection[1]+FZ*LiftDirection[2]
    loadsSubset['CD']=FX*DragDirection[0]+FY*DragDirection[1]+FZ*DragDirection[2]
    loadsSubset['CY']=FX*SideDirection[0]+FY*SideDirection[1]+FZ*SideDirection[2]
    loadsSubset['Cn']=TX*LiftDirection[0]+TY*LiftDirection[1]+TZ*LiftDirection[2]
    loadsSubset['Cl']=TX*DragDirection[0]+TY*DragDirection[1]+TZ*DragDirection[2]
    loadsSubset['Cm']=TX*SideDirection[0]+TY*SideDirection[1]+TZ*SideDirection[2]

    # Normalize forces and moments
    for Force in ('CL','CD','CY'):  loadsSubset[Force]  *= FluxCoef
    for Torque in ('Cn','Cl','Cm'): loadsSubset[Torque] *= TorqueCoef


def _extendLoadsWithStatistics(loads, IntegralDataName, DesiredStatistics):
    '''
    Beware: this is a private function, employed by updateAndSaveLoads()

    Add to loads dictionary the relevant statistics requested by the user
    through the DesiredStatistics list of special named strings.

    Parameters
    ----------

        loads : dict
            Contains integral data in the following form:
            ::
                >>> np.array = loads['FamilyBCNameOrElementName']['VariableName']

        IntegralDataName : str
            Name of the IntegralDataNode (CGNS) provided by elsA. It is used as
            key for loads dictionary.

        DesiredStatistics : :py:class:`list` of :py:class:`str`
            Desired statistics to infer from loads dictionary. For more
            information see documentation of function
            :py:func:`extractIntegralData`
    '''

    AvgIt = setup.ReferenceValues["CoprocessOptions"]["AveragingIterations"]

    loadsSubset = loads[IntegralDataName]
    IterationNumber = loadsSubset['IterationNumber']
    IterationWindow = len(IterationNumber[IterationNumber>(IterationNumber[-1]-AvgIt)])
    if IterationWindow < 2: return

    for StatKeyword in DesiredStatistics:
        KeywordsSplit = StatKeyword.split('-')
        StatType = KeywordsSplit[0]
        VarName = '-'.join(KeywordsSplit[1:])

        try:
            InstantaneousArray = loadsSubset[VarName]
            InvalidValues = np.logical_not(np.isfinite(InstantaneousArray))
            InstantaneousArray[InvalidValues] = 0.

        except KeyError:
            WARNMSG = ('WARNING: user requested statistic for variable {}, but '
                       ' this variable was not found in loads.\n'
                       'Please choose one of: {}').format(VarName,
                                                    str(loadsSubset.keys()))
            printCo(WARN+WARNMSG+ENDC, proc=rank)
            return

        if StatType.lower() == 'avg':
            StatisticArray = uniform_filter1d(InstantaneousArray,
                                              size=IterationWindow)

            InvalidValues = np.logical_not(np.isfinite(StatisticArray))
            StatisticArray[InvalidValues] = 0.

        elif StatType.lower() == 'std':
            AverageArray = uniform_filter1d(InstantaneousArray,
                                            size=IterationWindow)


            InvalidValues = np.logical_not(np.isfinite(AverageArray))
            AverageArray[InvalidValues] = 0.

            loadsSubset['avg-'+VarName] = AverageArray

            FilteredInstantaneousSqrd = uniform_filter1d(InstantaneousArray**2,
                                                      size=IterationWindow)

            InvalidValues = np.logical_not(np.isfinite(FilteredInstantaneousSqrd))
            FilteredInstantaneousSqrd[InvalidValues] = 0.
            FilteredInstantaneousSqrd[FilteredInstantaneousSqrd<0] = 0.

            StatisticArray = np.sqrt(np.abs(FilteredInstantaneousSqrd-AverageArray**2))
        loadsSubset[StatKeyword] = StatisticArray


def getIntegralDataName(IntegralDataNode):
    '''
    Transforms the elsA provided **IntegralDataNode** name into a suitable name
    for further storing it at **loads** dictionary.

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


def isConverged(ZoneName='AIRFOIL', FluxName='std-CL', FluxThreshold=0.001):
    '''
    This method is used to determine if a given load is converged by looking
    at its standard deviation and comparing it to a user-provided threshold.
    If converged, the signal returns :py:obj:`True` to all ranks and writes a message
    to ``coprocess.log`` file.

    To give several criteria, all the arguments must be lists of the same length.

    Parameters
    ----------

        ZoneName : :py:class:`str` or :py:class:`list`
            Component name (shall exist in **loads** dictionary)

        FluxName : :py:class:`str` or :py:class:`list`
            Name of the load quantity (typically, standard deviation statistic
            of some effort) used for convergence determination.

        FluxThreshold : :py:class:`str` or :py:class:`list`
            if the last element of the flux named **FluxName** is less than the
            user-provided **FluxThreshold**, then the convergence
            criterion is satisfied

    Returns
    -------

        ConvergedCriterion : bool
            :py:obj:`True` if the convergence criteria are satisfied
    '''
    ConvergedCriterion = False
    if rank == 0:
        try:
            if isinstance(ZoneName, str):
                assert isinstance(FluxName, str) and isinstance(FluxThreshold, str)
                ZoneName = [ZoneName]
                FluxName = [FluxName]
                FluxThreshold = [FluxThreshold]
            else:
                assert len(ZoneName) == len(FluxName) == len(FluxThreshold)
            loadsTree = C.convertFile2PyTree(os.path.join(DIRECTORY_OUTPUT,
                                                           FILE_LOADS))
            loadsZones = I.getZones(loadsTree)
            ConvergedCriteria = []
            for zoneCur, fluxCur, thresholdCur in zip(ZoneName, FluxName, FluxThreshold):
                zone, = [z for z in loadsZones if z[0] == zoneCur]
                Flux, = J.getVars(zone, [fluxCur])
                ConvergedCriteria.append(Flux[-1] < thresholdCur)
            ConvergedCriterion = all(ConvergedCriteria)
            if ConvergedCriterion:
                MSG = 'CONVERGED at iteration {} since:'.format(CurrentIteration-1)
                for zoneCur, fluxCur, thresholdCur in zip(ZoneName, FluxName, FluxThreshold):
                    MSG += '\n  {} < {} on {}'.format(fluxCur, thresholdCur, zoneCur)
                printCo('*******************************************',color=GREEN)
                printCo(MSG, color=GREEN)
                printCo('*******************************************',color=GREEN)

        except:
            ConvergedCriterion = False

    comm.Barrier()
    ConvergedCriterion = comm.bcast(ConvergedCriterion,root=0)

    return ConvergedCriterion


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

        copyOutputFiles('surfaces.cgns','loads.cgns')

    '''
    for file2copy in files2copy:
        f2cSplit = file2copy.split('.')
        newFileName = f2cSplit[0]+'_AfterIter%d.'%(CurrentIteration-1)+f2cSplit[1]
        try:
            shutil.copy2(os.path.join(DIRECTORY_OUTPUT,file2copy),
                         os.path.join(DIRECTORY_OUTPUT,newFileName))
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


def addLoads(loads, ZoneName, ListOfLoadsNames, NumpyArrays):
    '''
    This function is an interface for adding new user-defined loads into the
    **loads** Python dictionary.

    Parameters
    ----------

        loads : dict
            Contains integral data in the following form:

            >>> loads['FamilyBCNameOrElementName']['VariableName'] = np.array

            parameter **loads** is modified

        ZoneName : str
            Name of the existing or new component where new loads are going
            to be added. (FamilyBC or Component name)

        ListOfLoadsNames : :py:class:`list` of :py:class:`str`
            Each element of this list is a name of the new loads to be added.
            For example:

            ::

                ['MyFirstLoad', 'AnotherLoad']

        NumpyArrays : :py:class:`list` of numpy 1d arrays
            Values to be added to **loads**

            .. attention::
                All arrays provided to **NumpyArrays** *(which belongs to the
                same component)* must have exactly the same number of elements
    '''
    try:
        loadsSubset = loads[ZoneName]
    except KeyError:
        loads[ZoneName] = {}
        loadsSubset = loads[ZoneName]
        for array, name in zip(NumpyArrays, ListOfLoadsNames):
            loadsSubset[name] = array
        return

    for array, name in zip(NumpyArrays, ListOfLoadsNames):
        try:
            ExistingArray = loadsSubset[name]
        except KeyError:
            loadsSubset[name] = array
            continue

        loadsSubset[name] = np.hstack((ExistingArray, array))


def addBodyForcePropeller2Loads(loads, BodyForceDisks):
    '''
    This function is an interface adapted to body-force computations.
    It transfers the integral information of each body-force disk into
    the **loads** dictionary.

    The fields that are appended to **loads** are:

    ::

        ['Thrust', 'RPM', 'Power', 'Pitch']

    hence, these values must exist in ``.Info`` CGNS node of each CGNS zone
    contained in **BodyForceDisks**

    .. note:: The new component of load dictionary has the same name as its
        corresponding body-force disk.

    Parameters
    ----------

        loads : dict
            Contains integral data in the following form:

            >>> loads['FamilyBCNameOrElementName']['VariableName'] = np.array

            parameter **loads** is modified

        BodyForceDisks : :py:class:`list` of zone
            Current bodyforce disks as obtained from
            :py:func:`MOLA.LiftingLine.computePropellerBodyForce`
    '''
    Cmpi.barrier()
    for BodyForceDisk in BodyForceDisks:
        BodyForceDiskName = I.getName( BodyForceDisk )
        Info_n = I.getNodeFromName3(BodyForceDisk, '.Info')
        try:
            PropLoadsNames = ['Thrust','RPM','Power','Pitch']
            PropLoads = [I.getNodeFromName1(Info_n, ln)[1] for ln in PropLoadsNames]
            PropLoadsNames.append('IterationNumber')
            PropLoads.append(np.array([CurrentIteration]))
            addLoads(loads, BodyForceDiskName, PropLoadsNames, PropLoads)
        except:
            pass
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


#=================== Functions that will be deprecated soon ===================#
@J.deprecated(1.12, 1.13)
def saveAll(CouplingTreeWithSkeleton, CouplingTree,
            loads, DesiredStatistics,
            BodyForceInputData, BodyForceDisks,
            quit=False):
    '''
    This method is used to save all relevant data of the simulation, e.g.:

    * ``setup.py``
    * ``OUTPUT/fields.cgns``
    * ``OUTPUT/surfaces.cgns``
    * ``OUTPUT/loads.cgns``
    * ``OUTPUT/BodyForceSources.cgns``


    .. note:: The method can be used to stop the elsA simulation after saving all data
        by providing the argument **quit** = :py:obj:`True`

    .. deprecated:: 1.12

    Parameters
    ----------

        CouplingTreeWithSkeleton : PyTree
            This is the partial tree as obtained using :py:func:`adaptEndOfRun`,
            but including also the entire Skeleton.

            .. tip:: **CouplingTreeWithSkeleton** is the result of adding the
                skeleton to elsA's output tree **CouplingTree** as follows:
                ::

                    CouplingTreeWithSkeleton = I.merge([Skeleton, CouplingTree])

        CouplingTree : PyTree
            This is the partial tree as obtained using :py:func:`adaptEndOfRun`

        loads : dict
            It contains the integral data that will be saved as
            ``OUTPUT/loads.cgns``. Its structure is:
            ``loads['FamilyBCNameOrElementName']['VariableName'] = np.array``

        DesiredStatistics : :py:class:`list` of :py:class:`str`
            Desired statistics to infer from loads dictionary. For more
            information see documentation of function :py:func:`updateAndSaveLoads`

        BodyForceInputData : list
            This object contains the user-provided information contained in
            **BodyForceInputData** list of ``setup.py`` file

        BodyForceDisks : :py:class:`list` of zone
            Current bodyforce disks as obtained from function
            :py:func:`MOLA.LiftingLine.computePropellerBodyForce`

        quit : bool
            if :py:obj:`True`, force quit the simulation after saving relevant data.
    '''
    printCo('SAVING ALL', proc=0, color=GREEN)
    Cmpi.barrier()
    updateAndWriteSetup(setup)

    saveDistributedPyTree(CouplingTreeWithSkeleton, FILE_FIELDS)

    saveSurfaces(CouplingTreeWithSkeleton, loads, DesiredStatistics, tagWithIteration=True)

    updateAndSaveLoads(CouplingTree, loads, DesiredStatistics,
                       tagWithIteration=True)

    if BodyForceInputData:
        distributeAndSavePyTree(BodyForceDisks, FILE_BODYFORCESRC,
                                tagWithIteration=True)

    Cmpi.barrier()
    if quit:
        printCo('QUIT', proc=0, color=FAIL)
        Cmpi.barrier()
        elsAxdt.free("xdt-runtime-tree")
        elsAxdt.free("xdt-output-tree")
        Cmpi.barrier()
        # if elsAxdt: elsAxdt.safeInterrupt()
        # else: os._exit(0)
        os._exit(0)

@J.deprecated(1.12, 1.13, 'Use extractSurfaces instead')
def saveSurfaces(to, loads, DesiredStatistics, tagWithIteration=False,
                 onlyWalls=True):
    '''
    Save the ``OUTPUT/surfaces.cgns`` file.

    Extract:

    * Boundary Conditions
    * IsoSurfaces if the entry PostParameters is present in setup.py, else
       do nothing

    For a turbomachinery case, monitor the performance of each row and save the
    results in loads.cgns

    .. deprecated:: 1.12

    Parameters
    ----------

        to : PyTree
            Coupling tree as obtained from :py:func:`adaptEndOfRun`

        loads : :py:class:`dict`
            Contains integral data in the following form:

            >>> loads['FamilyBCNameOrElementName']['VariableName'] = np.array

        DesiredStatistics : :py:class:`list` of :py:class:`str`
            Here, the user requests the additional statistics to be computed.
            See documentation of function updateAndSaveLoads for more details.

        tagWithIteration : :py:class:`bool`
            if ``True``, adds a suffix ``_AfterIter<iteration>``
            to the saved filename (creates a copy)

        onlyWalls : :py:class:`bool`
            if ``True``, only BC defined with ``BCWall*`` type are extracted.
            Otherwise, all BC (but not GridConnectivity) are extracted
    '''
    to = I.renameNode(to, 'FlowSolution#Init', 'FlowSolution#Centers')
    to = I.renameNode(to, 'FlowSolution#Height', 'FlowSolution') # BEWARE if there is already a FlowSolution node
    reshapeBCDatasetNodes(to)
    BCs = boundaryConditions2Surfaces(to, onlyWalls=onlyWalls)
    isosurf = extractIsoSurfaces(to)
    surfaces = I.merge([BCs, isosurf])
    renameTooLongZones(surfaces)
    try:
        Cmpi._setProc(surfaces, rank)
        I._adaptZoneNamesForSlash(surfaces)
    except:
        pass
    Cmpi.barrier()
    printCo('saving surfaces', proc=0, color=CYAN)
    Cmpi.barrier()
    Cmpi.convertPyTree2File(surfaces, os.path.join(DIRECTORY_OUTPUT, FILE_SURFACES))
    printCo('surfaces saved OK', proc=0, color=GREEN)
    Cmpi.barrier()
    if tagWithIteration and rank == 0: copyOutputFiles(FILE_SURFACES)

    if 'TurboConfiguration' in dir(setup):
        monitorTurboPerformance(surfaces, loads, DesiredStatistics,
                                tagWithIteration=tagWithIteration)

@J.deprecated(1.12, 1.13, 'Use extractSurfaces with Extractions of type IsoSurface instead')
def extractIsoSurfaces(to):
    '''
    Extract IsoSurfaces in the PyTree **to**. The parametrization is done with the
    entry PostParameters in setup.py, that must contain keys:

    * ``IsoSurfaces``
        a dictionary whose keys are variable names and values are
        lists of associated levels.

    * ``Variables``
        a list of strings to compute extra variables on the extracted
        surfaces.

    .. note:: If ``PostParameters`` is not present in ``setup.py``, or if both
        ``IsoSurfaces`` and ``Variables`` are not present in ``PostParameters``,
        then the function return an empty PyTree.

    .. deprecated:: 1.12
        Use :py:func:`extractSurfaces` with Extractions of type IsoSurface instead

    Parameters
    ----------

        to : PyTree
            Coupling tree as obtained from :py:func:`adaptEndOfRun`

    Returns
    -------

        surfaces : PyTree
            PyTree with extracted ``IsoSurfaces``. There is one base for each, with
            the following naming convention : ``ISO_<Variable>_<Value>``
    '''
    surfaces = I.newCGNSTree()

    if not 'PostParameters' in dir(setup):
        return surfaces
    elif not ('IsoSurfaces' in setup.PostParameters \
                and 'Variables' in setup.PostParameters):
        return surfaces

    # See Anomaly 8784 https://elsa.onera.fr/issues/8784
    for BCDataSetNode in I.getNodesFromType(pto, 'BCDataSet_t'):
        for node in I.getNodesFromType(BCDataSetNode, 'DataArray_t'):
            if I.getValue(node) is None:
                I.rmNode(BCDataSetNode, node)

    # EXTRACT ISO-SURFACES
    pto = Cmpi.convert2PartialTree(to)
    for varname, values in setup.PostParameters['IsoSurfaces'].items():
        for value in values:
            iso = P.isoSurfMC(pto, varname, value)
            if iso != []:
                P._computeVariables(iso, setup.PostParameters['Variables'])
            base = I.newCGNSBase('ISO_{}_{}'.format(varname, value), 2, 3, parent=surfaces)
            I.addChild(base, iso)
    return surfaces

@J.deprecated(1.12, 1.13, 'Use save instead')
def distributeAndSavePyTree(ListOfZones, filename, tagWithIteration=False):
    '''
    Given a :py:class:`list` of zone (possibly empty list at some
    ranks), this function assigns a rank number to each zone and then
    saves the provided zones in a single CGNS file.

    .. deprecated:: 1.12

    Parameters
    ----------

        ListOfZones : :py:class:`list` of zone
            List of CGNS zones to be saved in parallel

        filename : str
            filename where zones will be writen

            .. attention:: **filename** extension must be ``.cgns``

        tagWithIteration : bool
            if :py:obj:`True`, adds a suffix ``_AfterIter<iteration>``
            to the saved filename (creates a copy)
    '''
    DetectedZones = I.getZones(ListOfZones)
    Cmpi._setProc(DetectedZones, rank)
    I._adaptZoneNamesForSlash(DetectedZones)
    printCo('saving '+filename, proc=0, color=CYAN)
    Cmpi.barrier()
    Cmpi.convertPyTree2File(DetectedZones, os.path.join(DIRECTORY_OUTPUT, filename))
    printCo('%s saved OK'%filename, proc=0, color=GREEN)
    if tagWithIteration and rank == 0: copyOutputFiles(filename)

@J.deprecated(1.12, 1.13, 'Use save instead')
def saveDistributedPyTree(t, filename, tagWithIteration=False):
    '''
    Given an already distributed PyTree (with coherent *proc* number), save it
    in a single CGNS file.

    .. deprecated:: 1.12

    Parameters
    ----------

        t : PyTree
            Distributed PyTree

        filename : str
            Name of the file where data will be writen.

            .. attention:: **filename** extension must be ``.cgns``

        tagWithIteration : bool
            if :py:obj:`True`, adds a suffix ``_AfterIter<iteration>``
            to the saved filename (creates a copy)
    '''
    printCo('saving '+filename, proc=0, color=CYAN)
    Cmpi.barrier()
    FullPath = os.path.join(DIRECTORY_OUTPUT, filename)
    Cmpi.barrier()
    if rank==0:
        try:
            if os.path.islink(FullPath):
                os.unlink(FullPath)
            else:
                os.remove(FullPath)
        except:
            pass
    Cmpi.barrier()
    if t is None: t = []
    Cmpi.convertPyTree2File(t, FullPath)
    Cmpi.barrier()
    printCo('%s saved OK'%filename, proc=0, color=GREEN)
    if tagWithIteration and rank == 0: copyOutputFiles(filename)
    Cmpi.barrier()

@J.deprecated(1.12, 1.13, 'Use save extractIntegralData and save instead')
def updateAndSaveLoads(to, loads, DesiredStatistics=['std-CL', 'std-CD'],
                       tagWithIteration=False, monitorMemory=False):
    '''
    Extract integral data from coupling tree **to**, and update **loads** Python
    dictionary adding statistics requested by the user.
    Then, write ``OUTPUT/loads.cgns`` file.

    .. deprecated:: 1.12

    Parameters
    ----------

        to : PyTree
            Coupling tree as obtained from :py:func:`adaptEndOfRun`

        loads : dict
            Contains integral data in the following form:

            >>> loads['FamilyBCNameOrElementName']['VariableName'] = np.array

        DesiredStatistics : :py:class:`list` of :py:class:`str`
            Here, the user requests the additional statistics to be computed.
            The syntax of each quantity must be as follows:

            ::

                '<preffix>-<integral_quantity_name>'

            `<preffix>` can be ``'avg'`` (for cumulative average) or ``'std'``
            (for standard deviation). ``<integral_quantity_name>`` can be any
            quantity contained in loads, including other statistics.

            .. hint:: chaining preffixes is perfectly accepted, like
                ``'std-std-CL'`` which would compute the cumulative standard
                deviation of the cumulative standard deviation of the
                lift coefficient (:math:`\sigma(\sigma(C_L))`)

        tagWithIteration : bool
            if :py:obj:`True`, adds a suffix ``_AfterIter<iteration>``
            to the saved filename (creates a copy)

        monitorMemory : bool
            if :py:obj:`True`, function :py:func:`addMemoryUsage2Loads` is
            called, which adds memory usage information into **loads**
    '''
    IntegralDataNodes = I.getNodesFromType2(to, 'IntegralData_t')
    for IntegralDataNode in IntegralDataNodes:
        IntegralDataName = getIntegralDataName(IntegralDataNode)
        _appendIntegralDataNode2Loads(loads, IntegralDataNode)
        _extendLoadsWithProjectedLoads(loads, IntegralDataName)
        _extendLoadsWithStatistics(loads, IntegralDataName, DesiredStatistics)
    if monitorMemory: addMemoryUsage2Loads(loads)
    saveLoads(loads, tagWithIteration)

@J.deprecated(1.12, 1.13, 'Use save instead')
def saveLoads(loads, tagWithIteration=False):
    '''
    Save the ``OUTPUT/loads.cgns`` file.

    Parameters
    ----------

        loads : :py:class:`dict`
            Contains integral data in the following form:

            >>> loads['FamilyBCNameOrElementName']['VariableName'] = np.array

        tagWithIteration : :py:class:`bool`
            if ``True``, adds a suffix ``_AfterIter<iteration>``
            to the saved filename (creates a copy)
    '''
    loadsPyTree = loadsDict2PyTree(loads)
    printCo(CYAN+'saving loads...'+ENDC, proc=0)
    Cmpi.barrier()
    Cmpi.convertPyTree2File(loadsPyTree,
                            os.path.join(DIRECTORY_OUTPUT, FILE_LOADS))
    Cmpi.barrier()
    printCo(GREEN+'loads saved OK'+ENDC, proc=0)
    if tagWithIteration and rank == 0: copyOutputFiles(FILE_LOADS)
