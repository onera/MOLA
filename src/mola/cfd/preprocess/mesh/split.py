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

import os
import numpy as np
from treelab import cgns
import mola.naming_conventions as names
from mola.logging import mola_logger, MolaException, MolaAssertionError, redirect_streams_to_null, print, GREEN, ENDC

def apply(workflow):
    '''
    Distribute a PyTree **t**, with optional splitting.
    '''
    if not workflow.SplittingAndDistribution['Strategy'].lower() == 'atpreprocess':
        return
    
    if workflow.SplittingAndDistribution['Splitter'].lower() == 'cassiopee' and not workflow.tree.isStructured():
        raise MolaAssertionError('Incompatibility of SplittingAndDistribution with mesh: Cassiopee cannot be used to split unstructured mesh.')

    mola_logger.info("  🪚 splitting and distributing mesh", rank=0)

    nproc = workflow.RunManagement['NumberOfProcessors']
    workflow.SplittingAndDistribution.setdefault('NumberOfParts',nproc)
    workflow.SplittingAndDistribution.setdefault('ComponentsToSplit', [])

    is_partitioned = bool(workflow.tree.get(':CGNS#GlobalNumbering'))
    if is_partitioned:
        from mpi4py import MPI
        size = MPI.COMM_WORLD.Get_size()
        if size>1 and size != nproc:
            raise MolaException(f'MPI preprocess is being executed using {size} ranks, but it does not match the requested NumberOfProcessors in RunManagement ({nproc})')
        mola_logger.info('mesh already split and distributed: skip splitting', rank=0)
        return

    mola_logger.info(f'splitting and distributing mesh...', rank=0)
    split_with_imposed_mode(workflow)
    showStatisticsAndCheckDistribution(workflow.tree, CoresPerNode=workflow.SplittingAndDistribution['CoresPerNode'])
    workflow.tree = cgns.castNode(workflow.tree)


def split_with_imposed_mode(workflow):
    NumberOfParts = workflow.SplittingAndDistribution['NumberOfParts']
    NumberOfProcessors = workflow.RunManagement['NumberOfProcessors']
    if NumberOfParts == 1: # only distribute and return
        for z in workflow.tree.zones(): z.setParameters('.Solver#Param',proc=0)
        return
    splitter = workflow.SplittingAndDistribution['Splitter'].lower()
    if splitter == 'cassiopee':
        tRef = _splitAndDistributeUsingNPartsAndNProcsWithCassiopee(workflow, NumberOfParts, NumberOfProcessors, raise_error=True)
    elif splitter == 'maia':
        tRef = _splitAndDistributeUsingNPartsWithMaia(workflow)
    tRef.setUniqueZoneNames()
    
    workflow.tree = tRef

def _splitAndDistributeUsingNPartsAndNProcsWithCassiopee(workflow, NumberOfParts, NumberOfProcessors, raise_error=False):

    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() > 1:
        raise MolaException('cannot use Splitter="cassiopee" in MPI parallel mode. Use maia or relaunch in sequential mode.')

    # Checks on splitter and distributor
    splitter = workflow.SplittingAndDistribution['Splitter']
    if splitter.lower() != 'cassiopee':
        raise MolaException(f'splitter {splitter} not implemented yet')
    distributor = workflow.SplittingAndDistribution['Distributor']
    if distributor.lower() != 'cassiopee':
        raise MolaException(f'distributor {distributor} not implemented yet')


    t = workflow.tree
    tRef = t.copy()

    TotalNPts = tRef.numberOfCells()
    ProcPointsLoad = TotalNPts / NumberOfParts
    basesToSplit, basesNotToSplit = _getBasesBasedOnSplitPolicy(tRef, workflow)

    if basesToSplit: 
        tRef = _split_with_cassiopee(tRef, basesToSplit, basesNotToSplit, ProcPointsLoad, NumberOfParts, NumberOfProcessors, raise_error)

    tRef = _distribute_with_cassiopee(tRef, NumberOfProcessors, workflow.SplittingAndDistribution['CoresPerNode'])
    
    tRef = cgns.castNode(tRef)
    return tRef

def _split_with_cassiopee(tRef, basesToSplit, basesNotToSplit, ProcPointsLoad, NumberOfParts, NumberOfProcessors, raise_error=False):

    import Converter.PyTree as C
    import Transform.PyTree as T
    
    remainingNProcs = NumberOfParts * 1
    baseName2NProc = dict()

    for base in basesNotToSplit:
        baseNPts = base.numberOfCells()
        baseNProc = int( baseNPts / ProcPointsLoad )
        baseName2NProc[base[0]] = baseNProc
        remainingNProcs -= baseNProc

    if basesToSplit:

        tToSplit = cgns.add([b.copy() for b in basesToSplit])
        C.registerAllNames(tToSplit) # HACK https://gitlab.onera.net/numerics/mola/-/issues/143
        tSplit = T.splitSize(tToSplit, 0, type=0, R=remainingNProcs,
                             minPtsPerDir=5)

        tSplit = cgns.castNode(tSplit)
        NbOfZonesAfterSplit = tSplit.numberOfZones()
        HasDegeneratedZones = False
        if NbOfZonesAfterSplit < remainingNProcs:
            mola_logger.user_warning(f'Number of zones after split ({NbOfZonesAfterSplit}) is less than expected procs ({remainingNProcs})')
            mola_logger.debug('attempting T.splitNParts()...')
            tSplit = T.splitNParts(tToSplit, remainingNProcs)
            tSplit = cgns.castNode(tSplit)

            splitZones = tSplit.zones()
            if len(splitZones) < remainingNProcs:
                raise MolaException(('could not split sufficiently. Try manually splitting '
                                   'mesh and set SplittingAndDistribution["ComponentsToSplit"]=None'))

            for zone in splitZones:
                if zone.isStructured():
                    dims = zone.shape()
                    for NPts, dir in zip(dims, ['i', 'j', 'k']):
                        if NPts < 5:
                            warning_message = f'zone {zone[0]} has {NPts} pts in {dir} direction'
                            if NPts < 3:
                                mola_logger.error(warning_message)
                                HasDegeneratedZones = True
                            else:
                                mola_logger.warning(warning_message)

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
            tRef = cgns.castNode(tRef)
            return tRef  #, 0, np.inf, np.inf, np.inf, np.inf

    NZones = tRef.numberOfZones()
    if NumberOfProcessors > NZones:
        if raise_error:
            raise MolaException((f'Requested number of procs ({NumberOfProcessors}) is higher than the final number of zones ({NZones}).\n'
                   'You may try the following:\n'
                   ' - set SplitBlocks=True to more grid components\n'
                   ' - Reduce the number of procs\n'
                   ' - increase the number of grid points'))
        else:
            return tRef  #, 0, np.inf, np.inf, np.inf, np.inf
    
    return tRef

def _distribute_with_cassiopee(tree, NumberOfProcessors:int, cores_per_node: int, raise_error=True):
    
    import Distributor2.PyTree as D2

    stats = dict()
    # NOTE see Cassiopee BUG #8244 -> need algorithm='fast'
    with redirect_streams_to_null():
        tree, stats = D2.distribute(tree, NumberOfProcessors, algorithm='fast', useCom='all')
        tree = cgns.castNode(tree)
    stats.update(stats)
   
    behavior = 'raise' if raise_error else 'silent'
    if hasAnyEmptyProc(tree, NumberOfProcessors, behavior=behavior):
        tree = cgns.castNode(tree)
        return tree  #, 0, np.inf, np.inf, np.inf, np.inf

    HighestLoad = getNbOfPointsOfHighestLoadedNode(tree, cores_per_node)
    HighestLoadProc = getNbOfPointsOfHighestLoadedProc(tree)

    # return tree, NZones, stats['varMax'], stats['meanPtsPerProc'], HighestLoad, HighestLoadProc
    return tree

def _splitAndDistributeUsingNPartsWithMaia(workflow):
    from mola.cfd.preprocess.mesh.tools import to_partitioned_if_distributed
    
    tree_is_distributed = bool(workflow.tree.get(':CGNS#Distribution'))
    if not tree_is_distributed: raise MolaException('unexpected behavior, expected dist_tree')
    
    from mpi4py import MPI
    size = MPI.COMM_WORLD.Get_size()
    rank = MPI.COMM_WORLD.Get_rank()
    nproc = workflow.RunManagement['NumberOfProcessors']
    if size != nproc:
        msg = f'Splitting with maia requires using same mpi number of ranks as requested splitting.\n'
        msg+= f'You are using {size} mpi ranks, and requested {nproc} procs, which are not equal.\n'
        msg+=  'Please adapt your MPI size or change your splitter in SplitAndDistribute options.'
        raise MolaException(msg)

    t = to_partitioned_if_distributed(workflow.tree)
    MPI.COMM_WORLD.barrier()
    return t

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

def hasAnyEmptyProc(t, NumberOfProcessors, behavior='raise'):
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
        if behavior == 'silent':
            pass
        elif behavior == 'print':
            mola_logger.error(MSG)
        else:
            raise MolaException(MSG)
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

    # unable to compute statistics on MPI partitioned tree
    # hint: use skeleton, see /stck/jcoulet/dev/dev-Tools/maia/Support/lbernard/skeleton.py
    if get_mpi_size() > 1: return 

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
    mola_logger.info(MSG, rank=0)

    for p in range(ResultingNProc):
        if p not in ProcDistributed:
            raise MolaException(f'Bad proc distribution! Rank {p} is empty')


def _getComponentsNamesBasedOnSplitPolicy(workflow):
    splitCompsUserData = workflow.SplittingAndDistribution["ComponentsToSplit"]
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

def get_mpi_size():
    from mpi4py import MPI
    return MPI.COMM_WORLD.Get_size()


def _assert_tree_has_good_distribution_assignment(workflow):
    expected_nb_rank = workflow.RunManagement['NumberOfProcessors']
    unasigned_ranks = list(range(expected_nb_rank))

    for zone in workflow.tree.zones():
        solver_param = zone.get(Name='.Solver#Param')
        if not solver_param:
            raise MolaException(f'zone {zone.path()} did not have .Solver#Param node')

        proc_node = solver_param.get(Name='proc')
        if not proc_node:
            raise MolaException(f'no node named "proc" under {solver_param.path()}')

        try:
            assigned_rank = int(proc_node.value())
        except BaseException as e:
            raise MolaException(f'could not retrieve proc value on {proc_node}') from e

        if assigned_rank > (expected_nb_rank - 1):
            raise MolaException(f'assigned rank "{assigned_rank}" to zone {zone.path()} is higher than NumberOfProcessors-1 ({expected_nb_rank - 1}) ')

        if assigned_rank in unasigned_ranks:
            unasigned_ranks.remove(assigned_rank)

    if unasigned_ranks:
        raise MolaException(f'distribution failed, since got unassigned ranks: {str(unasigned_ranks)}')
    
