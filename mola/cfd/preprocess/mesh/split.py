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

import numpy as np
from mola import misc

import Transform.PyTree as T

def apply(workflow
                    #    t, InputMeshes, mode='auto', cores_per_node=48,
                    #    minimum_number_of_nodes=1,
                    #    maximum_allowed_nodes=20,
                    #    maximum_number_of_points_per_node=1e9,
                    #    only_consider_full_node_nproc=True,
                    #    NumberOfProcessors=None, SplitBlocks=True
                     ):
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

    if workflow.SplittingAndDistribution['Strategy'].lower() == 'atcomputation':
        return
    
    print('splitting and distributing mesh...')

    TotalNPts = t.numberOfPoints()

    cores_per_node = workflow.CoresPerNode
    minimum_number_of_nodes = workflow.MinimumAllowedNodes
    maximum_allowed_nodes = workflow.MaximumAllowedNodes
    only_consider_full_node_nproc = workflow.DistributeExclusivelyOnFullNodes
    maximum_number_of_points_per_node = workflow.MaximumNumberOfPointsPerNode

    if mode == 'auto':

        startNProc = cores_per_node*minimum_number_of_nodes+1
        if not only_consider_full_node_nproc: startNProc -= cores_per_node 
        endNProc = maximum_allowed_nodes*cores_per_node+1

        if NumberOfProcessors is not None and NumberOfProcessors > 0:
            print(misc.WARN+'User requested NumberOfProcessors=%d, switching to mode=="imposed"'%NumberOfProcessors+misc.ENDC)
            mode = 'imposed'

        elif minimum_number_of_nodes == maximum_allowed_nodes:
            if only_consider_full_node_nproc:
                NumberOfProcessors = minimum_number_of_nodes*cores_per_node
                print(misc.WARN+'User constrained to NumberOfProcessors=%d, switching to mode=="imposed"'%NumberOfProcessors+misc.ENDC)
                mode = 'imposed'

        elif minimum_number_of_nodes > maximum_allowed_nodes:
            raise ValueError(misc.FAIL+'minimum_number_of_nodes > maximum_allowed_nodes'+misc.ENDC)

        elif minimum_number_of_nodes < 1:
            raise ValueError(misc.FAIL+'minimum_number_of_nodes must be at least equal to 1'+misc.ENDC)

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
            _, NZones, varMax, meanPtsPerProc, MaxPtsPerNode, MaxPtsPerProc = _splitAndDistributeUsingNProcs(t,
                InputMeshes, NumberOfProcessors, cores_per_node, maximum_number_of_points_per_node,
                raise_error=False)
            AllNZones.append( NZones )
            AllVarMax.append( varMax )
            AllAvgPts.append( meanPtsPerProc )
            AllMaxPtsPerNode.append( MaxPtsPerNode )
            AllMaxPtsPerProc.append( MaxPtsPerProc )

            if AllNZones[i] == 0:
                start = misc.FAIL
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
        tRef = _splitAndDistributeUsingNProcs(t, InputMeshes, NProcCandidates[BestOption],
                cores_per_node, maximum_number_of_points_per_node, raise_error=True)[0]

        I._correctPyTree(tRef,level=3)
        tRef = connectMesh(tRef, InputMeshes)

    elif mode == 'imposed':

        tRef = _splitAndDistributeUsingNProcs(t, InputMeshes, NumberOfProcessors, cores_per_node,
                                 maximum_number_of_points_per_node, raise_error=True)[0]

        I._correctPyTree(tRef,level=3)
        tRef = connectMesh(tRef, InputMeshes)

    showStatisticsAndCheckDistribution(tRef, CoresPerNode=cores_per_node)

    return tRef

def _splitAndDistributeUsingNProcs(workflow, raise_error=False):

    tRef = I.copyRef(t)
    TotalNPts = C.getNPts(tRef)
    ProcPointsLoad = TotalNPts / NumberOfProcessors
    basesToSplit, basesBackground = getBasesBasedOnSplitPolicy(tRef, InputMeshes)

    remainingNProcs = NumberOfProcessors * 1
    baseName2NProc = dict()

    for base in basesBackground:
        baseNPts = C.getNPts(base)
        baseNProc = int( baseNPts / ProcPointsLoad )
        baseName2NProc[base[0]] = baseNProc
        remainingNProcs -= baseNProc


    if basesToSplit:

        tToSplit = I.merge([C.newPyTree([b[0],I.getZones(b)]) for b in basesToSplit])

        removeMatchAndNearMatch(tToSplit)
        tSplit = T.splitSize(tToSplit, 0, type=0, R=remainingNProcs,
                             minPtsPerDir=5)
        NbOfZonesAfterSplit = len(I.getZones(tSplit))
        HasDegeneratedZones = False
        if NbOfZonesAfterSplit < remainingNProcs:
            MSG = 'WARNING: nb of zones after split (%d) is less than expected procs (%d)'%(NbOfZonesAfterSplit, remainingNProcs)
            MSG += '\nattempting T.splitNParts()...'
            print(misc.WARN+MSG+misc.ENDC)
            tSplit = T.splitNParts(tToSplit, remainingNProcs)
            splitZones = I.getZones(tSplit)
            if len(splitZones) < remainingNProcs:
                MSG = ('could not split sufficiently. Try manually splitting '
                       'mesh and set SplitBlocks=False')
                raise ValueError(misc.FAIL+MSG+misc.ENDC)
            for zone in splitZones:
                zoneDims = I.getZoneDim(zone)
                if zoneDims[0] == 'Structured':
                    dims = zoneDims[1:-1]
                    for NPts, dir in zip(dims, ['i', 'j', 'k']):
                        if NPts < 5:
                            if NPts < 3:
                                MSG = misc.FAIL+'ERROR: zone %s has %d pts in %s direction'%(zone[0],NPts,dir)+misc.ENDC
                                HasDegeneratedZones = True
                            else:
                                MSG = misc.WARN+'WARNING: zone %s has %d pts in %s direction'%(zone[0],NPts,dir)+misc.ENDC
                            print(MSG)

        if HasDegeneratedZones:
            raise ValueError(misc.FAIL+'grid has degenerated zones. See previous print error messages'+misc.ENDC)

        for splitbase in I.getBases(tSplit):
            basename = splitbase[0]
            base = I.getNodeFromName2(tRef, basename)
            if not base: raise ValueError('unexpected !')
            I._rmNodesByType(base, 'Zone_t')
            base[2].extend( I.getZones(splitbase) )

        tRef = I.merge([tRef,tSplit])

        NZones = len( I.getZones( tRef ) )
        if NumberOfProcessors > NZones:
            if raise_error:
                MSG = ('Requested number of procs ({}) is higher than the final number of zones ({}).\n'
                       'You may try the following:\n'
                       ' - Reduce the number of procs\n'
                       ' - increase the number of grid points').format( NumberOfProcessors, NZones)
                raise ValueError(misc.FAIL+MSG+misc.ENDC)
            return tRef, 0, np.inf, np.inf, np.inf, np.inf

    NZones = len( I.getZones( tRef ) )
    if NumberOfProcessors > NZones:
        if raise_error:
            MSG = ('Requested number of procs ({}) is higher than the final number of zones ({}).\n'
                   'You may try the following:\n'
                   ' - set SplitBlocks=True to more grid components\n'
                   ' - Reduce the number of procs\n'
                   ' - increase the number of grid points').format( NumberOfProcessors, NZones)
            raise ValueError(misc.FAIL+MSG+misc.ENDC)
        else:
            return tRef, 0, np.inf, np.inf, np.inf, np.inf

    # NOTE see Cassiopee BUG #8244 -> need algorithm='fast'
    silence = misc.OutputGrabber()
    with silence:
        tRef, stats = D2.distribute(tRef, NumberOfProcessors, algorithm='fast', useCom='all')

    behavior = 'raise' if raise_error else 'silent'

    if hasAnyEmptyProc(tRef, NumberOfProcessors, behavior=behavior):
        return tRef, 0, np.inf, np.inf, np.inf, np.inf

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
    for zone in I.getZones(t):
        Proc, = getProc(zone)
        Node = Proc//cores_per_node
        try: NPtsPerNode[Node] += C.getNPts(zone)
        except KeyError: NPtsPerNode[Node] = C.getNPts(zone)

    nodes = list(NPtsPerNode)
    NodesLoad = np.zeros(max(nodes)+1, dtype=int)
    for node in NPtsPerNode: NodesLoad[node] = NPtsPerNode[node]
    HighestLoad = np.max(NodesLoad)

    return HighestLoad

def getNbOfPointsOfHighestLoadedProc(t):
    NPtsPerProc = {}
    for zone in I.getZones(t):
        Proc, = getProc(zone)
        try: NPtsPerProc[Proc] += C.getNPts(zone)
        except KeyError: NPtsPerProc[Proc] = C.getNPts(zone)

    procs = list(NPtsPerProc)
    ProcsLoad = np.zeros(max(procs)+1, dtype=int)
    for proc in NPtsPerProc: ProcsLoad[proc] = NPtsPerProc[proc]
    HighestLoad = np.max(ProcsLoad)

    return HighestLoad

def _isMaximumNbOfPtsPerNodeExceeded(t, maximum_number_of_points_per_node, cores_per_node):
    NPtsPerNode = {}
    for zone in I.getZones(t):
        Proc, = getProc(zone)
        Node = (Proc//cores_per_node)+1
        try: NPtsPerNode[Node] += C.getNPts(zone)
        except KeyError: NPtsPerNode[Node] = C.getNPts(zone)

    for node in NPtsPerNode:
        if NPtsPerNode[node] > maximum_number_of_points_per_node: return True
    return False

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

    for z in I.getZones(t):
        proc = int(D2.getProc(z))

        if proc < 0:
            if debug_filename: C.convertPyTree2File(t, debug_filename)
            raise ValueError('zone %s is not distributed'%z[0])

        if proc in Proc2Zones:
            Proc2Zones[proc].append( I.getName(z) )
        else:
            Proc2Zones[proc] = [ I.getName(z) ]

        try: UnaffectedProcs.remove( proc )
        except ValueError: pass


    if UnaffectedProcs:
        hasAnyEmptyProc = True
        if debug_filename: C.convertPyTree2File(t, debug_filename)
        MSG = misc.FAIL+'THERE ARE UNAFFECTED PROCS IN DISTRIBUTION!!\n'
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

def getBasesBasedOnSplitPolicy(t,InputMeshes):
    '''
    Returns two different lists, one with bases to split and other with bases
    not to split. The filter is done depending on the boolean value
    of ``SplitBlocks`` key provided by user for each component of **InputMeshes**.

    Parameters
    ----------

        t : PyTree
            assembled tree

        InputMeshes : :py:class:`list` of :py:class:`dict`
            user-provided preprocessing
            instructions as described in :py:func:`prepareMesh4ElsA` doc

    Returns
    -------

        basesToSplit : :py:class`list` of base
            bases that are to be split.

        basesNotToSplit : :py:class`list` of base
            bases that are NOT to be split.
    '''
    basesToSplit = []
    basesNotToSplit = []
    for meshInfo in InputMeshes:
        base = I.getNodeFromName1(t,meshInfo['baseName'])
        if meshInfo['SplitBlocks']:
            basesToSplit += [base]
        else:
            basesNotToSplit += [base]

    return basesToSplit, basesNotToSplit

def showStatisticsAndCheckDistribution(tNew, CoresPerNode=28):
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
    Distribution = D2.getProcDict(tNew)

    NPtsPerProc = {}
    for zone in I.getZones(tNew):
        Proc, = getProc(zone)
        try: NPtsPerProc[Proc] += C.getNPts(zone)
        except KeyError: NPtsPerProc[Proc] = C.getNPts(zone)

    NPtsPerNode = {}
    for zone in I.getZones(tNew):
        Proc, = getProc(zone)
        Node = (Proc//CoresPerNode)+1
        try: NPtsPerNode[Node] += C.getNPts(zone)
        except KeyError: NPtsPerNode[Node] = C.getNPts(zone)


    ListOfProcs = list(NPtsPerProc.keys())
    ListOfNPts = [NPtsPerProc[p] for p in ListOfProcs]
    ArgNPtsMin = np.argmin(ListOfNPts)
    ArgNPtsMax = np.argmax(ListOfNPts)

    MSG = '\nTotal number of processors is %d\n'%ResultingNProc
    MSG += 'Total number of zones is %d\n'%len(I.getZones(tNew))
    MSG += 'Proc %d has lowest nb. of points with %d\n'%(ListOfProcs[ArgNPtsMin],
                                                    ListOfNPts[ArgNPtsMin])
    MSG += 'Proc %d has highest nb. of points with %d\n'%(ListOfProcs[ArgNPtsMax],
                                                    ListOfNPts[ArgNPtsMax])
    print(MSG)

    for node in NPtsPerNode:
        print('Node %d has %d points'%(node,NPtsPerNode[node]))

    print(misc.CYAN+'TOTAL NUMBER OF POINTS: '+'{:,}'.format(C.getNPts(tNew)).replace(',',' ')+'\n'+misc.ENDC)

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


