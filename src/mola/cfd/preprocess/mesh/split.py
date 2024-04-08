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

import numpy as np
from treelab import cgns
from mola import misc
from mola.logging import mola_logger, MolaException, MolaAssertionError, redirect_streams_to_null, print, GREEN, ENDC

def apply(workflow):
    '''
    Distribute a PyTree **t**, with optional splitting.

    Returns a new split and distributed PyTree.

    .. important:: only **InputMeshes** where ``'SplitBlocks':True`` are split.

    Parameters
    ----------

        t : PyTree
            assembled tree

        InputMeshes : :py:class:`list` of :py:class:`dict`
            user-provided preprocessing
            instructions as described in :py:func:`prepareMesh4ElsA` doc

        mode : str
            choose the mode of splitting and distribution among these possibilities:

            * ``'auto'``
                automatically search for the optimum distribution verifying
                the constraints given by **maximum_allowed_nodes** and
                **maximum_number_of_points_per_node**

                .. note:: **NumberOfProcessors** is ignored if **mode** = ``'auto'``, as it
                    is automatically computed by the function. The resulting
                    **NumberOfProcessors** is a multiple of **cores_per_node**

            * ``'imposed'``
                the number of processors is imposed using parameter **NumberOfProcessors**.

                .. note:: **cores_per_node** and **maximum_allowed_nodes**
                    parameters are ignored.

        cores_per_node : int
            number of available CPU cores per node.

            .. note:: only relevant if **mode** = ``'auto'``

        minimum_number_of_nodes : int
            Establishes the minimum number of nodes for the automatic research of
            **NumberOfProcessors**.

            .. note:: only relevant if **mode** = ``'auto'``

        maximum_allowed_nodes : int
            Establishes a boundary of maximum usable nodes. The resulting
            number of processors is the product **cores_per_node** :math:`\\times`
            **maximum_allowed_nodes**

            .. note:: only relevant if **mode** = ``'auto'``

        maximum_number_of_points_per_node : int
            Establishes a boundary of maximum points per node. This value is
            important in order to reduce the required RAM memory for each one
            of the nodes. It raises a :py:obj:`ValueError` if at least one node
            does not satisfy this condition.

        only_consider_full_node_nproc : bool
            if :py:obj:`True` and **mode** = ``'auto'``, then the number of
            processors considered for the optimum search distribution is a
            multiple of **cores_per_node**, in order to employ each node at its
            full capacity. If :py:obj:`False`, then any processor number from
            **cores_per_node** up to **cores_per_node** :math:`\\times` **maximum_allowed_nodes**
            is explored

            .. note:: only relevant if **mode** = ``'auto'``

        NumberOfProcessors : int
            number of processors to be imposed when **mode** = ``'imposed'``

            .. attention:: if **mode** = ``'auto'``, this parameter is ignored

        SplitBlocks : bool
            default value of **SplitBlocks** if it does not exist in the InputMesh
            component.


    Returns
    -------

        t : PyTree
            new distributed *(and possibly split)* tree

    '''
    workflow.SplittingAndDistribution = set_default_splitting_parameters(workflow.SplittingAndDistribution)
    
    if not workflow.SplittingAndDistribution['Strategy'].lower() == 'atpreprocess': 
        return
        
    mola_logger.info('splitting and distributing mesh...')
    mode = get_and_check_splitting_mode(workflow.SplittingAndDistribution)
    if mode == 'auto':
        split_with_auto_mode(workflow)
    else:
        split_with_imposed_mode(workflow)

    showStatisticsAndCheckDistribution(workflow.tree, CoresPerNode=workflow.SplittingAndDistribution['CoresPerNode'])
    set_default_NumberOfProcessors_in_RunManagement(workflow)
    workflow.tree = cgns.castNode(workflow.tree)

def set_default_splitting_parameters(SplittingAndDistribution):
    
    if isinstance(SplittingAndDistribution, str):
        if SplittingAndDistribution.lower() == 'pypart':
            SplittingAndDistribution = dict(
                Strategy='AtComputation', 
                Splitter='PyPart', 
                Distributor='PyPart', 
                ComponentsToSplit='all',
                )
        elif SplittingAndDistribution.lower() == 'maia':
            SplittingAndDistribution = dict(
                Strategy='AtComputation', 
                Splitter='maia', 
                Distributor='maia', 
                ComponentsToSplit='all',
                )
        else:
            raise MolaException(f'More parameters must be given with the splitter {SplittingAndDistribution}. See the doc.')
        
    default_splitAndDist = dict(
            Strategy='AtPreprocess', # "AtPreprocess" or "AtComputation"
            Splitter='Cassiopee', # or 'maia', 'PyPart' etc..
            Distributor='Cassiopee', 
            ComponentsToSplit='all', # 'all', or None or ['first', 'second'...]
            NumberOfProcessors='auto', 
            MinimumAllowedNodes=1,
            MaximumAllowedNodes=1,
            MaximumNumberOfPointsPerNode=1e9,
            CoresPerNode=48, # FIXME Should depend on the machine, and so on the Network. Otherwise, don't set a default value
            DistributeExclusivelyOnFullNodes=True
            )

    for key, value in default_splitAndDist.items():
        SplittingAndDistribution.setdefault(key, value)
    
    return SplittingAndDistribution


def get_and_check_splitting_mode(SplittingParameters):

    the_number_of_processors_is_given = \
        'NumberOfProcessors' in SplittingParameters \
        and isinstance(SplittingParameters['NumberOfProcessors'], int)
    if the_number_of_processors_is_given:
        return 'imposed'

    if SplittingParameters['MinimumAllowedNodes'] == SplittingParameters['MaximumAllowedNodes']:
        if SplittingParameters['DistributeExclusivelyOnFullNodes']:
            SplittingParameters['NumberOfProcessors'] = SplittingParameters['MinimumAllowedNodes'] * SplittingParameters['CoresPerNode']
            mola_logger.warning(f'User constrained to NumberOfProcessors={SplittingParameters["NumberOfProcessors"]}, switching to mode="imposed"')
            return 'imposed'

    elif SplittingParameters['MinimumAllowedNodes'] > SplittingParameters['MaximumAllowedNodes']:
        raise MolaException('minimum_number_of_nodes > maximum_allowed_nodes')

    elif SplittingParameters['MinimumAllowedNodes'] < 1:
        raise MolaException('minimum_number_of_nodes must be at least equal to 1')

    return 'auto'

def split_with_auto_mode(workflow):

    t = workflow.tree
    splitAndDistribUser = workflow.SplittingAndDistribution

    cores_per_node = splitAndDistribUser['CoresPerNode']
    minimum_number_of_nodes = splitAndDistribUser['MinimumAllowedNodes']
    maximum_allowed_nodes = splitAndDistribUser['MaximumAllowedNodes']
    only_consider_full_node_nproc = splitAndDistribUser['DistributeExclusivelyOnFullNodes']
    maximum_number_of_points_per_node = splitAndDistribUser['MaximumNumberOfPointsPerNode']

    TotalNPts = t.numberOfPoints()
    startNProc = cores_per_node*minimum_number_of_nodes+1
    if not only_consider_full_node_nproc: startNProc -= cores_per_node 
    endNProc = maximum_allowed_nodes*cores_per_node+1

    if only_consider_full_node_nproc:
        NProcCandidates = np.array(list(range(startNProc-1,
                                                (endNProc-1)+cores_per_node,
                                                cores_per_node)))
    else:
        NProcCandidates = np.array(list(range(startNProc, endNProc)))

    EstimatedAverageNodeLoad = TotalNPts / (NProcCandidates / cores_per_node)
    NProcCandidates = NProcCandidates[EstimatedAverageNodeLoad < maximum_number_of_points_per_node]

    if len(NProcCandidates) < 1:
        raise MolaException('maximum_number_of_points_per_node is too likely to be exceeded.\nTry increasing maximum_allowed_nodes and/or maximum_number_of_points_per_node')

    Title1= ' number of  | number of  | max pts at | max pts at | percent of | average pts|'
    Title = ' processors | zones      | any proc   | any node   | imbalance  | per proc   |'
    
    Ncol = len(Title)
    print('-'*Ncol)
    print(Title1)
    print(Title)
    print('-'*Ncol)
    Ndigs = len(Title.split('|')[0]) + 1
    ColFmt = r'{:^'+str(Ndigs)+'g}'

    AllNZones = []
    AllVarMax = []
    AllAvgPts = []
    AllMaxPtsPerNode = []
    AllMaxPtsPerProc = []
    for i, NumberOfProcessors in enumerate(NProcCandidates):
        _, NZones, varMax, meanPtsPerProc, MaxPtsPerNode, MaxPtsPerProc = \
            _splitAndDistributeUsingNProcs(workflow, NumberOfProcessors,
                                            raise_error=False)
        AllNZones.append( NZones )
        AllVarMax.append( varMax )
        AllAvgPts.append( meanPtsPerProc )
        AllMaxPtsPerNode.append( MaxPtsPerNode )
        AllMaxPtsPerProc.append( MaxPtsPerProc )

        log_level = 'INFO'
        Line = ColFmt.format(NumberOfProcessors)
        try:
            Line += ColFmt.format(AllNZones[i])
            Line += ColFmt.format(AllMaxPtsPerProc[i])
            Line += ColFmt.format(AllMaxPtsPerNode[i])
            Line += ColFmt.format(AllVarMax[i] * 100)
            Line += ColFmt.format(AllAvgPts[i])
        except IndexError:
            log_level = 'ERROR'
            Line += f'  <== EXCEEDED nb. pts. per node with {AllMaxPtsPerNode[i]}'
        
        print(Line, level=log_level)
        if cores_per_node>1 and (NumberOfProcessors%cores_per_node==0):
            print('-'*Ncol)
        

    BestOption = np.argmin( AllMaxPtsPerProc )

    for i, NumberOfProcessors in enumerate(NProcCandidates):
        if i == BestOption and AllNZones[i] > 0:
            Line = GREEN + ColFmt.format(NumberOfProcessors)
            Line += ColFmt.format(AllNZones[i])
            Line += ColFmt.format(AllMaxPtsPerProc[i])
            Line += ColFmt.format(AllMaxPtsPerNode[i])
            Line += ColFmt.format(AllVarMax[i] * 100)
            Line += ColFmt.format(AllAvgPts[i])
            Line += '  <== BEST'+ENDC
            print(Line)
            break
    tRef = _splitAndDistributeUsingNProcs(workflow, NumberOfProcessors,
                                            raise_error=True)[0]

    tRef.setUniqueZoneNames()
    # tRef = connectMesh(tRef, InputMeshes) # TODO do it after split&dist

    # update NumberOfProcessors in workflow
    # This is mandatory for function set_default_NumberOfProcessors_in_RunManagement
    workflow.SplittingAndDistribution['NumberOfProcessors'] = NumberOfProcessors

    workflow.tree = cgns.castNode(tRef)

def split_with_imposed_mode(workflow):
    NumberOfProcessors = workflow.SplittingAndDistribution['NumberOfProcessors']
    tRef = _splitAndDistributeUsingNProcs(workflow, NumberOfProcessors, raise_error=True)[0]
    tRef.setUniqueZoneNames()
    # tRef = connectMesh(tRef, InputMeshes) # TODO do it after split&dist
    workflow.tree = cgns.castNode(tRef)
    
def set_default_NumberOfProcessors_in_RunManagement(workflow):
    if 'NumberOfProcessors' not in workflow.RunManagement \
        or workflow.RunManagement['NumberOfProcessors'] is None:
        workflow.RunManagement['NumberOfProcessors'] = workflow.SplittingAndDistribution['NumberOfProcessors']

def _splitAndDistributeUsingNProcs(workflow, NumberOfProcessors, raise_error=False):

    import Distributor2.PyTree as D2
    import Transform.PyTree as T

    t = workflow.tree
    tRef = t.copy()
    TotalNPts = t.numberOfCells()

    ProcPointsLoad = TotalNPts / NumberOfProcessors
    basesToSplit, basesNotToSplit = _getBasesBasedOnSplitPolicy(tRef, workflow)
    remainingNProcs = NumberOfProcessors * 1
    baseName2NProc = dict()

    for base in basesNotToSplit:
        baseNPts = base.numberOfCells()
        baseNProc = int( baseNPts / ProcPointsLoad )
        baseName2NProc[base[0]] = baseNProc
        remainingNProcs -= baseNProc


    if basesToSplit:

        tToSplit = cgns.merge([b.copy() for b in basesToSplit])
        splitter = workflow.SplittingAndDistribution['Splitter']
        if splitter.lower() == 'cassiopee':
            tToSplit.findAndRemoveNodes(Type='GridConnectivity1to1_t')
            tToSplit.findAndRemoveNodes(Type='GridConnectivity_t', Value='Abbuting')

            tSplit = T.splitSize(tToSplit, 0, type=0, R=remainingNProcs,
                                minPtsPerDir=5)
            tSplit = cgns.castNode(tSplit)
        else:
            raise ValueError(f'splitter {splitter} not implemented yet')

        NbOfZonesAfterSplit = tSplit.numberOfZones()
        HasDegeneratedZones = False
        if NbOfZonesAfterSplit < remainingNProcs:
            mola_logger.warning(f'Number of zones after split ({NbOfZonesAfterSplit}) is less than expected procs ({remainingNProcs})')
            if splitter.lower() == 'cassiopee':
                mola_logger.debug('attempting T.splitNParts()...')
                tSplit = T.splitNParts(tToSplit, remainingNProcs)
                tSplit = cgns.castNode(tSplit)
            else:
                raise MolaException(f'splitter {splitter} not implemented yet')

            splitZones = tSplit.zones()
            if len(splitZones) < remainingNProcs:
                raise MolaException(('could not split sufficiently. Try manually splitting '
                                   'mesh and set SplittingAndDistribution["ComponentsToSplit"]=None'))

            for zone in splitZones:
                if zone.isStructured():
                    dims = zone.shape()
                    for NPts, dir in zip(dims, ['i', 'j', 'k']):
                        if NPts < 5:
                            if NPts < 3:
                                raise MolaException('zone {zone[0]} has {NPts} pts in {dir} direction', exit=False)
                                HasDegeneratedZones = True
                            else:
                                mola_logger.warning('zone {zone[0]} has {NPts} pts in {dir} direction')

        if HasDegeneratedZones:
            raise MolaException('grid has degenerated zones. See previous print error messages')

        for splitbase in tSplit.bases():
            base = tRef.get(Name=splitbase.name(), Type='CGNSBase_t', Depth=1)
            if not base: raise ValueError(f'unexpected ! could not find base {splitbase.name()}')
            base.swap(splitbase)
        
        NZones = tRef.numberOfZones()
        if NumberOfProcessors > NZones:
            if raise_error:
                raise MolaException((f'Requested number of procs ({NumberOfProcessors}) is higher than the final number of zones ({NZones}).\n'
                       'You may try the following:\n'
                       ' - Reduce the number of procs\n'
                       ' - increase the number of grid points'))
            return tRef, 0, np.inf, np.inf, np.inf, np.inf

    NZones = tRef.numberOfZones()
    if NumberOfProcessors > NZones:
        if raise_error:
            raise MolaException((f'Requested number of procs ({NumberOfProcessors}) is higher than the final number of zones ({NZones}).\n'
                   'You may try the following:\n'
                   ' - set SplitBlocks=True to more grid components\n'
                   ' - Reduce the number of procs\n'
                   ' - increase the number of grid points'))
        else:
            return tRef, 0, np.inf, np.inf, np.inf, np.inf

    distributor = workflow.SplittingAndDistribution['Distributor']
    stats = dict()
    if distributor.lower() == 'cassiopee':
        # NOTE see Cassiopee BUG #8244 -> need algorithm='fast'
        with redirect_streams_to_null():
            tRef, stats = D2.distribute(tRef, NumberOfProcessors, algorithm='fast', useCom='all')
            tRef = cgns.castNode(tRef)
        stats.update(stats)
    else: 
        raise MolaException(f'distributor {distributor} not implemented yet')
   
    behavior = 'raise' if raise_error else 'silent'

    if hasAnyEmptyProc(tRef, NumberOfProcessors, behavior=behavior):
        return tRef, 0, np.inf, np.inf, np.inf, np.inf

    splitAndDistribUser = workflow.SplittingAndDistribution

    cores_per_node = splitAndDistribUser['CoresPerNode']
    maximum_number_of_points_per_node = splitAndDistribUser['MaximumNumberOfPointsPerNode']

    HighestLoad = getNbOfPointsOfHighestLoadedNode(tRef, cores_per_node)
    HighestLoadProc = getNbOfPointsOfHighestLoadedProc(tRef)

    if HighestLoad > maximum_number_of_points_per_node:
        if raise_error:
            raise MolaException(f'exceeded maximum_number_of_points_per_node ({HighestLoad}>{maximum_number_of_points_per_node})')
        return tRef, 0, np.inf, np.inf, np.inf, np.inf


    return tRef, NZones, stats['varMax'], stats['meanPtsPerProc'], HighestLoad, HighestLoadProc

def getNbOfPointsOfHighestLoadedNode(t, cores_per_node):
    NPtsPerNode = {}
    for zone in t.zones():
        Proc, = getProc(zone)
        Node = int(Proc)//cores_per_node
        try: NPtsPerNode[Node] += zone.numberOfCells()
        except KeyError: NPtsPerNode[Node] = zone.numberOfCells()

    nodes = list(NPtsPerNode)
    NodesLoad = np.zeros(max(nodes)+1, dtype=int)
    for node in NPtsPerNode: NodesLoad[node] = NPtsPerNode[node]
    HighestLoad = np.max(NodesLoad)

    return HighestLoad

def getNbOfPointsOfHighestLoadedProc(t):
    NPtsPerProc = {}
    for zone in t.zones():
        Proc, = getProc(zone)
        try: NPtsPerProc[Proc] += zone.numberOfCells()
        except KeyError: NPtsPerProc[Proc] = zone.numberOfCells()

    procs = list(NPtsPerProc)
    ProcsLoad = np.zeros(max(procs)+1, dtype=int)
    for proc in NPtsPerProc: ProcsLoad[proc] = NPtsPerProc[proc]
    HighestLoad = np.max(ProcsLoad)

    return HighestLoad

def hasAnyEmptyProc(t, NumberOfProcessors, behavior='raise', debug_filename=''):
    '''
    Check the proc distribution of a tree and raise an error (or print message)
    if there are any empty proc.

    Parameters
    ----------

        t : PyTree
            tree with node ``.Solver#Param/proc``

        NumberOfProcessors : int
            initially requested number of processors for distribution

        behavior : str
            if empty processors are found, this parameter specifies the behavior
            of the function:

            * ``'raise'``
                Raises a :py:obj:`ValueError`, stopping execution

            * ``'print'``
                Prints a message onto the termina, execution continues

            * ``'silent'``
                No error, no print; execution continues

        debug_filename : str
            if given, then writes the input tree **t** before the designed
            exceptions are raised or in case some proc is empty.

    Returns
    -------

        hasAnyEmptyProc : bool
            :py:obj:`True` if any processor has no attributed zones
    '''
    if behavior not in ['raise', 'print', 'silent']:
        raise MolaException('behavior %s not recognized'%behavior)
    
    Proc2Zones = dict()
    UnaffectedProcs = list(range(NumberOfProcessors))

    for z in t.zones():
        proc = int(getProc(z))

        if proc < 0:
            raise ValueError('zone %s is not distributed'%z[0])

        if proc in Proc2Zones:
            Proc2Zones[proc].append( z.name() )
        else:
            Proc2Zones[proc] = [ z.name() ]

        try: UnaffectedProcs.remove( proc )
        except ValueError: pass


    if UnaffectedProcs:
        hasAnyEmptyProc = True
        MSG = 'THERE ARE UNAFFECTED PROCS IN DISTRIBUTION!!\n'
        MSG+= 'Empty procs: %s'%str(UnaffectedProcs)
        raise MolaException(MSG, exit=(behavior == 'raise'))
    else:
        hasAnyEmptyProc = False

    return hasAnyEmptyProc

def showStatisticsAndCheckDistribution(tNew, CoresPerNode=48):
    '''
    Print statistics on the distribution of a PyTree and also indicates the load
    attributed to each computational node.

    Parameters
    ----------

        tNew : PyTree
            tree where distribution was done.

        CoresPerNode : int
            number of processors per node.

    '''
    ProcDistributed = getProc(tNew)
    ResultingNProc = max(ProcDistributed)+1

    NPtsPerProc = {}
    for zone in tNew.zones():
        Proc, = getProc(zone)
        try: NPtsPerProc[Proc] += zone.numberOfCells()
        except KeyError: NPtsPerProc[Proc] = zone.numberOfCells()

    NPtsPerNode = {}
    for zone in tNew.zones():
        Proc, = getProc(zone)
        Node = (Proc//CoresPerNode)+1
        try: NPtsPerNode[Node] += zone.numberOfCells()
        except KeyError: NPtsPerNode[Node] = zone.numberOfCells()


    ListOfProcs = list(NPtsPerProc.keys())
    ListOfNPts = [NPtsPerProc[p] for p in ListOfProcs]
    ArgNPtsMin = np.argmin(ListOfNPts)
    ArgNPtsMax = np.argmax(ListOfNPts)

    MSG = f'''Statistics of distribution on processors:
  Total number of processors is {ResultingNProc}
  Total number of zones is {tNew.numberOfZones()}
  Proc {ListOfProcs[ArgNPtsMin]} has lowest nb. of points with {ListOfNPts[ArgNPtsMin]}
  Proc {ListOfProcs[ArgNPtsMax]} has highest nb. of points with {ListOfNPts[ArgNPtsMax]}
'''
    for node in NPtsPerNode:
        MSG += f'    Node {node} has {NPtsPerNode[node]} points\n'
    MSG += '  '+'-'*29 + '\n'
    MSG += f'  TOTAL NUMBER OF POINTS: {tNew.numberOfCells():,}'.replace(',',' ')
    mola_logger.info(MSG)

    for p in range(ResultingNProc):
        if p not in ProcDistributed:
            raise MolaException(f'Bad proc distribution! Rank {p} is empty')


def _getComponentsNamesBasedOnSplitPolicy(workflow):
    splitUserData = workflow.SplittingAndDistribution
    splitCompsUserData = splitUserData['ComponentsToSplit']
    ComponentsToSplit = []
    ComponentsNotToSplit = []
    for component in workflow.RawMeshComponents:
        if isinstance(splitCompsUserData, str) and splitCompsUserData.lower()=='all':
            ComponentsToSplit += [ component['Name']  ]
        elif isinstance(splitCompsUserData, list):
            if component['Name'] in splitCompsUserData:
                ComponentsToSplit += [ component['Name']  ]
            else:
                ComponentsNotToSplit += [ component['Name']  ]
        else:
            ComponentsNotToSplit += [ component['Name']  ]
    
    return ComponentsToSplit, ComponentsNotToSplit


def _getBasesBasedOnSplitPolicy(t, workflow):
    toSplit, notToSplit = _getComponentsNamesBasedOnSplitPolicy(workflow)
    basesToSplit = []
    basesNotToSplit = []
    for base in t.bases():
        if base.name() in toSplit:
            basesToSplit += [ base ]
        elif base.name() in notToSplit:
            basesNotToSplit += [ base ]
        else:
            msg = f'FATAL: base {base.name()} was neither in toSplit:\n'
            msg+= str(toSplit)+'\n'
            msg+= 'nor in notToSplit:\n'
            msg+= str(notToSplit)+'\n'
            msg+= 'please contact the support'
            raise MolaException(msg)
    return basesToSplit, basesNotToSplit

def getProc(t):
    procs = []
    for zone in cgns.getZones(t):
        solverParam = zone.get(Name='.Solver#Param',Depth=1)
        procs += [ int(solverParam.get(Name='proc').value()) ]
    return np.array(procs, order='F', ndmin=1)

def splitWithPyPart(comm=None):
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
    import Converter.Internal as I
    import etc.pypart.PyPart as PPA
    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    PyPartBase = PPA.PyPart('main.cgns',
                            lksearch=['OUTPUT', '.'],
                            loadoption='partial',
                            mpicomm=comm,
                            LoggingInFile=False,
                            LoggingFile='LOGS/partTree',
                            LoggingVerbose=40  # Filter: None=0, DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
                            )
    # reorder=[6, 2] is recommended by CLEF, mostly for unstructured mesh
    # with modernized elsA. It is also mandatory to use lussorscawf on
    # unstructured mesh.
    PartTree = PyPartBase.runPyPart(method=2, partN=1, reorder=[6, 2])
    PyPartBase.finalise(PartTree, savePpart=True, method=1)
    Skeleton = PyPartBase.getPyPartSkeletonTree()
    Distribution = PyPartBase.getDistribution()

    # # Put Distribution into the Skeleton
    # for zone in I.getZones(Skeleton):
    #     zonePath = I.getPath(Skeleton, zone, pyCGNSLike=True)[1:]
    #     Cmpi._setProc(zone, Distribution[zonePath])

    t = I.merge([Skeleton, PartTree])

    # Skeleton = loadSkeleton(Skeleton, PartTree)
    # # Add empty Coordinates for skeleton zones
    # # Needed to make Cmpi.convert2PartialTree work
    # for zone in I.getZones(Skeleton):
    #     GC = I.getNodeFromType1(zone, 'GridCoordinates_t')
    #     if not GC:
    #         J.set(zone, 'GridCoordinates', childType='GridCoordinates_t',
    #             CoordinateX=None, CoordinateY=None, CoordinateZ=None)
    #     elif I.getZoneType(zone) == 2:
    #         # For unstructured zone, correct the node NFaceElements/ElementConnectivity
    #         # Problem with PyPart: see issue https://elsa-e.onera.fr/issues/9002
    #         # C._convertArray2NGon(zone)
    #         NFaceElements = I.getNodeFromName(zone, 'NFaceElements')
    #         if NFaceElements:
    #             node = I.getNodeFromName(NFaceElements, 'ElementConnectivity')
    #             I.setValue(node, np.abs(I.getValue(node)))

    # if 'CoupledSurfaces' in setup.ReferenceValues['CoprocessOptions']:
    #     # This part is linked to the WorkflowAerothermalCoupling
    #     # For unstructured zones, AdditionnalFamilyName nodes are lost
    #     # See Anomaly #10494 on elsA support
    #     # We need to restore them
    #     for i, famBCTrigger in enumerate(setup.ReferenceValues['CoprocessOptions']['CoupledSurfaces']):
    #         surfaceName = 'ExchangeSurface{}'.format(i)
    #         for zone in I.getZones(t):
    #             if I.getZoneType(zone) == 2:
    #                 for BC in C.getFamilyBCs(t, famBCTrigger):
    #                     I.createChild(BC, 'SurfaceName', 'AdditionalFamilyName_t', value=surfaceName)


    return t, Skeleton, PyPartBase, Distribution

def splitWithMaia(comm=None):
    '''
    Use Maia to split the mesh in ``main.cgns``. This function should be use
    in ``compute.py`` to prepare the mesh before calling ``elsAxdt.XdtCGNS()``.

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
    import maia
    if comm is None:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD

    dist_tree = maia.io. file_to_dist_tree('main.cgns', comm)
    # zone_to_parts = maia.factory.partitioning.compute_balanced_weights(dist_tree, comm)
    part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
    maia.io.part_tree_to_file(part_tree, 'part_tree.cgns', comm)

    return part_tree, Distribution
