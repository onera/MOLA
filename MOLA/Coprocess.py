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
            If ``None``, all procs will write the message.

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
        by providing the argument **quit** = ``True``

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
            if ``True``, force quit the simulation after saving relevant data.
    '''
    printCo('SAVING ALL', proc=0, color=GREEN)
    Cmpi.barrier()
    updateAndWriteSetup(setup)

    saveDistributedPyTree(CouplingTreeWithSkeleton, FILE_FIELDS)

    saveSurfaces(CouplingTreeWithSkeleton, tagWithIteration=True)

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



def saveSurfaces(to, tagWithIteration=False, onlyWalls=True):
    '''
    Save the ``OUTPUT/surfaces.cgns`` file.

    Parameters
    ----------

        to : PyTree
            Coupling tree as obtained from :py:func:`adaptEndOfRun`

        tagWithIteration : bool
            if ``True``, adds a suffix ``_AfterIter<iteration>``
            to the saved filename (creates a copy)

        onlyWalls : bool
            if ``True``, only BC defined with ``BCWall*`` type are extracted.
            Otherwise, all BC (but not GridConnectivity) are extracted
    '''
    BCs = boundaryConditions2Surfaces(to, onlyWalls=onlyWalls)
    try:
        Cmpi._setProc(BCs,rank)
        I._adaptZoneNamesForSlash(BCs)
    except:
        pass
    Cmpi.barrier()
    printCo('saving surfaces', proc=0, color=CYAN)
    Cmpi.barrier()
    Cmpi.convertPyTree2File(BCs, os.path.join(DIRECTORY_OUTPUT, FILE_SURFACES))
    printCo('surfaces saved OK', proc=0, color=GREEN)
    Cmpi.barrier()
    if tagWithIteration and rank == 0: copyOutputFiles(FILE_SURFACES)


def distributeAndSavePyTree(ListOfZones, filename, tagWithIteration=False):
    '''
    Given a :py:class:`list` of zone (possibly empty list at some
    ranks), this function assigns a rank number to each zone and then
    saves the provided zones in a single CGNS file.

    Parameters
    ----------

        ListOfZones : :py:class:`list` of zone
            List of CGNS zones to be saved in parallel

        filename : str
            filename where zones will be writen

            .. attention:: **filename** extension must be ``.cgns``

        tagWithIteration : bool
            if ``True``, adds a suffix ``_AfterIter<iteration>``
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



def saveDistributedPyTree(t, filename, tagWithIteration=False):
    '''
    Given an already distributed PyTree (with coherent *proc* number), save it
    in a single CGNS file.

    Parameters
    ----------

        t : PyTree
            Distributed PyTree

        filename : str
            Name of the file where data will be writen.

            .. attention:: **filename** extension must be ``.cgns``

        tagWithIteration : bool
            if ``True``, adds a suffix ``_AfterIter<iteration>``
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
        writeSetup(setup)
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
    if ExistingLoadsFile and setup.elsAkeysNumerics['inititer']>1:
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
    Cmpi.barrier()

    return loads


def updateAndSaveLoads(to, loads, DesiredStatistics=['std-CL', 'std-CD'],
                       tagWithIteration=False, monitorMemory=False):
    '''
    Extract integral data from coupling tree **to**, and update **loads** Python
    dictionary adding statistics requested by the user.
    Then, write ``OUTPUT/loads.cgns`` file.

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
            if ``True``, adds a suffix ``_AfterIter<iteration>``
            to the saved filename (creates a copy)

        monitorMemory : bool
            if ``True``, function :py:func:`addMemoryUsage2Loads` is
            called, which adds memory usage information into **loads**
    '''
    IntegralDataNodes = I.getNodesFromType2(to, 'IntegralData_t')
    for IntegralDataNode in IntegralDataNodes:
        IntegralDataName = getIntegralDataName(IntegralDataNode)
        _appendIntegralDataNode2Loads(loads, IntegralDataNode)
        _extendLoadsWithProjectedLoads(loads, IntegralDataName)
        _extendLoadsWithStatistics(loads, IntegralDataName, DesiredStatistics)
    if monitorMemory: addMemoryUsage2Loads(loads)
    loadsPyTree = loadsDict2PyTree(loads)
    printCo(CYAN+'saving loads...'+ENDC, proc=0)
    Cmpi.barrier()
    Cmpi.convertPyTree2File(loadsPyTree,
                            os.path.join(DIRECTORY_OUTPUT, FILE_LOADS))
    Cmpi.barrier()
    printCo(GREEN+'loads saved OK'+ENDC, proc=0)
    if tagWithIteration and rank == 0: copyOutputFiles(FILE_LOADS)



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
        LoadsItem['IterationNumber'] = np.hstack((LoadsItem['IterationNumber'],
                                                  int(CurrentIteration)))
        LoadsItem['UsedMemoryInPercent'] = np.hstack((LoadsItem['UsedMemoryInPercent'],
                                                      float(UsedMemoryPctg)))
        LoadsItem['UsedMemory'] = np.hstack((LoadsItem['UsedMemory'],
                                             float(UsedMemory)))
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
        zones.append(zone)

    if zones:
        Cmpi._setProc(zones, rank)
        t = C.newPyTree(['Base', zones])
    else:
        t = C.newPyTree(['Base'])
    Cmpi.barrier()

    return t


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
            loadsSubset[integralKey] = np.copy(IntegralData[integralKey],
                                               order='F')



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

    FX = loadsSubset['MomentumXFlux']
    FY = loadsSubset['MomentumYFlux']
    FZ = loadsSubset['MomentumZFlux']
    MX = loadsSubset['TorqueX']
    MY = loadsSubset['TorqueY']
    MZ = loadsSubset['TorqueZ']

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

    INPUTS

    loads - (Python dictionary) - Contains integral data in the following form:
        np.array = loads['FamilyBCNameOrElementName']['VariableName']

    IntegralDataName - (string) - Name of the IntegralDataNode (CGNS) provided
        by elsA. It is used as key for loads dictionary.

    DesiredStatistics - (List of strings) - Desired statistics to infer from
        loads dictionary. For more information see documentation of
        function updateAndSaveLoads()
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
    If converged, the signal returns ``True`` to all ranks and writes a message
    to ``coprocess.log`` file.

    Parameters
    ----------

        ZoneName : str
            Component name (shall exist in **loads** dictionary)

        FluxName : str
            Name of the load quantity (typically, standard deviation statistic
            of some effort) used for convergence determination.

        FluxThreshold : float
            if the last element of the flux named **FluxName** is less than the
            user-provided **FluxThreshold**, then the convergence
            criterion is satisfied

    Returns
    -------

        ConvergedCriterion : bool
            ``True`` if the convergence criterion is satisfied
    '''
    ConvergedCriterion = False
    if rank == 0:
        try:
            loadsTree = C.convertFile2PyTree(os.path.join(DIRECTORY_OUTPUT,
                                                           FILE_LOADS))
            loadsZones = I.getZones(loadsTree)
            zone, = [z for z in loadsZones if z[0] == ZoneName]
            Flux, = J.getVars(zone, [FluxName])
            ConvergedCriterion = Flux[-1] < FluxThreshold
            if ConvergedCriterion:
                MSG = 'CONVERGED at iteration {} since {} < {}'.format(
                        CurrentIteration-1, FluxName, FluxThreshold)
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
    This function returns ``True`` to all processors if the margin before
    time-out is satisfied. Otherwise, it returns ``False`` to all processors.

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
            ``True`` if

            ::

                ElapsedTime >= (TimeOut - MarginBeforeTimeOut)

            Otherwise, returns ``False``.

            .. note:: the same value (``True`` or ``False``) is sent to *all*
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
    the function returns ``True`` to all processors. Otherwise, it returns
    ``False`` to all processors.

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
            ``True`` if the signal is received, otherwise ``False``, to all
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
    for EoRnode in I.getNodesFromName3(to, 'FlowSolution#EndOfRun#Coords'):
        GridLocationNode = I.getNodeFromType1(EoRnode, 'GridLocation_t')
        if I.getValue(GridLocationNode) != 'Vertex':
            zone = I.getParentOfNode(to, EoRnode)
            ERRMSG = ('Extracted coordinates of zone '
                      '%s must be located in Vertex')%I.getName(zone)
            raise ValueError(FAIL+ERRMSG+ENDC)
        I.rmNode(to, GridLocationNode)
        I.setName(EoRnode, 'GridCoordinates')
        I.setType(EoRnode, 'GridCoordinates_t')
    Cmpi.barrier()

def boundaryConditions2Surfaces(to, onlyWalls=True):
    '''
    Extract the BC data contained in the coupling tree as a list of CGNS zones.

    Parameters
    ----------

        to : PyTree
            Coupling tree as obtained from :py:func:`adaptEndOfRun`

        onlyWalls : bool
            if ``True``, only BC with keyword ``'wall'`` contained in
            their type are extracted. Otherwise, all BC are extracted,
            regardless of their type.

    Returns
    -------

        BCs : :py:class:`list` of zone
            List of surfaces, including fields stored in FlowSolution containers
    '''
    Cmpi.barrier()
    tR = I.renameNode(to, 'FlowSolution#Init', 'FlowSolution#Centers')
    DictBCNames2Type = C.getFamilyBCNamesDict(tR)

    BCs = []
    for FamilyName in DictBCNames2Type:
        BCType = DictBCNames2Type[FamilyName]
        if onlyWalls:
            if 'wall' in BCType.lower():
                BC = C.extractBCOfName(tR,'FamilySpecified:'+FamilyName)
                BCs.extend(BC)
        else:
            BC = C.extractBCOfName(tR,'FamilySpecified:'+FamilyName)
            BCs.extend(BC)

    Cmpi.barrier()

    return BCs


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
