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
from mola import misc, cgns

import Transform.PyTree as T
import Distributor2.PyTree as D2

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
    t = workflow.tree
    splitAndDistribUser = workflow.SplittingAndDistribution

    if splitAndDistribUser['Strategy'].lower() == 'atcomputation': return
    
    if splitAndDistribUser['NumberOfProcessors'] == 'auto':
        mode = 'auto'
    else:
        mode = 'imposed'
        NumberOfProcessors = splitAndDistribUser['NumberOfProcessors']
        
    print('splitting and distributing mesh...')
    
    cores_per_node = splitAndDistribUser['CoresPerNode']
    minimum_number_of_nodes = splitAndDistribUser['MinimumAllowedNodes']
    maximum_allowed_nodes = splitAndDistribUser['MaximumAllowedNodes']
    only_consider_full_node_nproc = splitAndDistribUser['DistributeExclusivelyOnFullNodes']
    maximum_number_of_points_per_node = splitAndDistribUser['MaximumNumberOfPointsPerNode']

    if mode == 'auto':

        TotalNPts = t.numberOfPoints()
        startNProc = cores_per_node*minimum_number_of_nodes+1
        if not only_consider_full_node_nproc: startNProc -= cores_per_node 
        endNProc = maximum_allowed_nodes*cores_per_node+1

        if NumberOfProcessors is not None and NumberOfProcessors > 0:
            print(misc.YELLOW+'User requested NumberOfProcessors=%d, switching to mode=="imposed"'%NumberOfProcessors+misc.ENDC)
            mode = 'imposed'

        elif minimum_number_of_nodes == maximum_allowed_nodes:
            if only_consider_full_node_nproc:
                NumberOfProcessors = minimum_number_of_nodes*cores_per_node
                print(misc.YELLOW+'User constrained to NumberOfProcessors=%d, switching to mode=="imposed"'%NumberOfProcessors+misc.ENDC)
                mode = 'imposed'

        elif minimum_number_of_nodes > maximum_allowed_nodes:
            raise ValueError(misc.RED+'minimum_number_of_nodes > maximum_allowed_nodes'+misc.ENDC)

        elif minimum_number_of_nodes < 1:
            raise ValueError(misc.RED+'minimum_number_of_nodes must be at least equal to 1'+misc.ENDC)

        if only_consider_full_node_nproc:
            NProcCandidates = np.array(list(range(startNProc-1,
                                                  (endNProc-1)+cores_per_node,
                                                  cores_per_node)))
        else:
            NProcCandidates = np.array(list(range(startNProc, endNProc)))

        EstimatedAverageNodeLoad = TotalNPts / (NProcCandidates / cores_per_node)
        NProcCandidates = NProcCandidates[EstimatedAverageNodeLoad < maximum_number_of_points_per_node]

        if len(NProcCandidates) < 1:
            raise ValueError(('maximum_number_of_points_per_node is too likely to be exceeded.\n'
                              'Try increasing maximum_allowed_nodes and/or maximum_number_of_points_per_node'))

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

            if AllNZones[i] == 0:
                start = misc.RED
                end = '  <== EXCEEDED nb. pts. per node with %d'%AllMaxPtsPerNode[i]+misc.ENDC
            else:
                start = end = ''
            Line = start + ColFmt.format(NumberOfProcessors)
            if AllNZones[i] == 0:
                Line += end
            else:
                Line += ColFmt.format(AllNZones[i])
                Line += ColFmt.format(AllMaxPtsPerProc[i])
                Line += ColFmt.format(AllMaxPtsPerNode[i])
                Line += ColFmt.format(AllVarMax[i] * 100)
                Line += ColFmt.format(AllAvgPts[i])
                Line += end

            print(Line)
            if cores_per_node>1 and (NumberOfProcessors%cores_per_node==0):
                print('-'*Ncol)
            

        BestOption = np.argmin( AllMaxPtsPerProc )

        for i, NumberOfProcessors in enumerate(NProcCandidates):
            if i == BestOption and AllNZones[i] > 0:
                Line = start = misc.GREEN + ColFmt.format(NumberOfProcessors)
                end = '  <== BEST'+misc.ENDC
                Line += ColFmt.format(AllNZones[i])
                Line += ColFmt.format(AllMaxPtsPerProc[i])
                Line += ColFmt.format(AllMaxPtsPerNode[i])
                Line += ColFmt.format(AllVarMax[i] * 100)
                Line += ColFmt.format(AllAvgPts[i])
                Line += end
                print(Line)
                break
        tRef = _splitAndDistributeUsingNProcs(workflow, NumberOfProcessors,
                                              raise_error=True)[0]

        tRef.setUniqueZoneNames()
        # tRef = connectMesh(tRef, InputMeshes) # TODO do it after split&dist

    elif mode == 'imposed':

        tRef = _splitAndDistributeUsingNProcs(workflow, NumberOfProcessors,
                                              raise_error=True)[0]

        tRef.setUniqueZoneNames()
        # tRef = connectMesh(tRef, InputMeshes) # TODO do it after split&dist

    showStatisticsAndCheckDistribution(tRef, CoresPerNode=cores_per_node)

    workflow.tree = tRef

    if 'NumberOfProcessors' not in workflow.RunManagement \
        or workflow.RunManagement['NumberOfProcessors'] is None:
        workflow.RunManagement['NumberOfProcessors'] = NumberOfProcessors

    return tRef

def _splitAndDistributeUsingNProcs(workflow, NumberOfProcessors, raise_error=False):

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
            MSG = 'WARNING: nb of zones after split (%d) is less than expected procs (%d)'%(NbOfZonesAfterSplit, remainingNProcs)
            print(misc.YELLOW+MSG+misc.ENDC)
            if splitter.lower() == 'cassiopee':
                MSG = 'attempting T.splitNParts()...'
                tSplit = T.splitNParts(tToSplit, remainingNProcs)
                tSplit = cgns.castNode(tSplit)
            else:
                raise ValueError(f'splitter {splitter} not implemented yet')

            splitZones = tSplit.zones()
            if len(splitZones) < remainingNProcs:
                MSG = ('could not split sufficiently. Try manually splitting '
                       'mesh and set SplittingAndDistribution["ComponentsToSplit"]=None')
                raise ValueError(misc.RED+MSG+misc.ENDC)
            
            for zone in splitZones:
                if zone.isStructured():
                    dims = zone.shape()
                    for NPts, dir in zip(dims, ['i', 'j', 'k']):
                        if NPts < 5:
                            if NPts < 3:
                                MSG = misc.RED+'ERROR: zone %s has %d pts in %s direction'%(zone[0],NPts,dir)+misc.ENDC
                                HasDegeneratedZones = True
                            else:
                                MSG = misc.YELLOW+'WARNING: zone %s has %d pts in %s direction'%(zone[0],NPts,dir)+misc.ENDC
                            print(MSG)

        if HasDegeneratedZones:
            raise ValueError(misc.RED+'grid has degenerated zones. See previous print error messages'+misc.ENDC)

        for splitbase in tSplit.bases():
            base = tRef.get(Name=splitbase.name(), Type='CGNSBase_t', Depth=1)
            if not base: raise ValueError(f'unexpected ! could not find base {splitbase.name()}')
            base.swap(splitbase)
        
        NZones = tRef.numberOfZones()
        if NumberOfProcessors > NZones:
            if raise_error:
                MSG = ('Requested number of procs ({}) is higher than the final number of zones ({}).\n'
                       'You may try the following:\n'
                       ' - Reduce the number of procs\n'
                       ' - increase the number of grid points').format( NumberOfProcessors, NZones)
                raise ValueError(misc.RED+MSG+misc.ENDC)
            return tRef, 0, np.inf, np.inf, np.inf, np.inf

    NZones = tRef.numberOfZones()
    if NumberOfProcessors > NZones:
        if raise_error:
            MSG = ('Requested number of procs ({}) is higher than the final number of zones ({}).\n'
                   'You may try the following:\n'
                   ' - set SplitBlocks=True to more grid components\n'
                   ' - Reduce the number of procs\n'
                   ' - increase the number of grid points').format( NumberOfProcessors, NZones)
            raise ValueError(misc.RED+MSG+misc.ENDC)
        else:
            return tRef, 0, np.inf, np.inf, np.inf, np.inf

    distributor = workflow.SplittingAndDistribution['Distributor']
    stats = dict()
    if distributor.lower() == 'cassiopee':
        # NOTE see Cassiopee BUG #8244 -> need algorithm='fast'
        silence = misc.OutputGrabber()
        with silence:
            tRef, stats = D2.distribute(tRef, NumberOfProcessors, algorithm='fast', useCom='all')
            tRef = cgns.castNode(tRef)
        stats.update(stats)
    else: 
        raise ValueError(f'distributor {distributor} not implemented yet')
   
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
            raise ValueError('exceeded maximum_number_of_points_per_node (%d>%d)'%(HighestLoad,
                                                maximum_number_of_points_per_node))
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
        MSG = misc.RED+'THERE ARE UNAFFECTED PROCS IN DISTRIBUTION!!\n'
        MSG+= 'Empty procs: %s'%str(UnaffectedProcs)+misc.ENDC
        if behavior == 'raise':
            raise ValueError(MSG)
        elif behavior == 'print':
            print(MSG)
        elif behavior != 'silent':
            raise ValueError('behavior %s not recognized'%behavior)
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

    MSG = '\nTotal number of processors is %d\n'%ResultingNProc
    MSG += 'Total number of zones is %d\n'%tNew.numberOfZones()
    MSG += 'Proc %d has lowest nb. of points with %d\n'%(ListOfProcs[ArgNPtsMin],
                                                    ListOfNPts[ArgNPtsMin])
    MSG += 'Proc %d has highest nb. of points with %d\n'%(ListOfProcs[ArgNPtsMax],
                                                    ListOfNPts[ArgNPtsMax])
    print(MSG)

    for node in NPtsPerNode:
        print('Node %d has %d points'%(node,NPtsPerNode[node]))

    print(misc.CYAN+'TOTAL NUMBER OF POINTS: '+'{:,}'.format(tNew.numberOfCells()).replace(',',' ')+'\n'+misc.ENDC)

    for p in range(ResultingNProc):
        if p not in ProcDistributed:
            raise ValueError('Bad proc distribution! rank %d is empty'%p)


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
            ValueError(misc.RED+msg+misc.ENDC)
    return basesToSplit, basesNotToSplit

def getProc(t):
    procs = []
    for zone in cgns.getZones(t):
        solverParam = zone.get(Name='.Solver#Param',Depth=1)
        procs += [ int(solverParam.get(Name='proc').value()) ]
    return np.array(procs, order='F', ndmin=1)