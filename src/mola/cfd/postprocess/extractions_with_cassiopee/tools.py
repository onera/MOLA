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
from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()

import Converter.PyTree as C
import Converter.Internal as I

from treelab import cgns

def mergeContainers(t, FlowSolutionVertexName='FlowSolution',
        FlowSolutionCellCenterName='FlowSolution#Centers',
        BCDataSetFaceCenterName='BCDataSet',
        remove_suffix_if_single_container=False):
    '''
    Merge all *FlowSolution_t* containers into a single one (one at Vertex, another
    at CellCenter), adding a numerical tag suffix to flowfield name for easy
    identification (e.g. ``<FlowfieldName>.<NumericTag>``). 

    .. danger:: when adding the numeric tag, the number of characters of the 
        resulting CGNS node can be higher than 32. This is not a problem for 
        in-memory tree, but most CGNS writters/readers cannot support names longer
        than 32 characters, which may result in loose of data if the flowfields 
        names with tags are truncated when saving or reading a CGNS file

    Also, this function merges all *BCDataSet_t/BCData_t* containers into a single
    one named ``BCDataSet/NeumannData`` located at FaceCenter. 

    .. caution:: input ``BCDataSet_t/BCData_t`` must be all contained in **FaceCenter**

    New ``UserDefined_t`` CGNS nodes are added to ``Zone_t`` nodes and ``BC_t`` nodes,
    named ``multi_containers``, which contains all the required information for
    making the inverse operation (see :py:func:`recoverContainers`) for recovering
    the original structure of the tree

    Parameters
    ----------

        t : PyTree, Base, Zone or list of Zone
            Input containing zones where containers are to be merged

        FlowSolutionVertexName : str
            the name of the resulting Vertex FlowSolution CGNS node

        FlowSolutionCellCenterName : str
            the name of the resulting CellCenter FlowSolution CGNS node

        BCDataSetFaceCenterName : str
            the name of the resulting BCDataSet CGNS node

        remove_suffix_if_single_container : bool
            if the result of merging containers is a single container for a given
            grid location, then removes the suffix "0" since it is not much useful

    Returns
    -------

        tR : PyTree, Base, Zone or list of Zone
            copy as reference of **t**, but with merged containers 
    '''
    # HACK https://elsa.onera.fr/issues/11221
    # HACK https://elsa.onera.fr/issues/10641


    tR = I.copyRef(t)
    for zone in I.getZones(tR):
        _mergeFlowSolutions(zone, FlowSolutionVertexName, FlowSolutionCellCenterName)
        _mergeBCData(zone, BCDataSetFaceCenterName)
        if remove_suffix_if_single_container:
            _remove_suffix_if_single_container(zone)
    return cgns.castNode(tR)

def _mergeFlowSolutions(zone, FlowSolutionVertexName='FlowSolution',
        FlowSolutionCellCenterName='FlowSolution#Centers'):
    '''
    Merge all FlowSolution_t into one at Vertex and one at CellCenter
    
    Consider using higher-level function mergeContainers
    '''
    if zone[3] != 'Zone_t': return AttributeError('first argument must be a zone')
    monoFlowSolutionNames = dict(Vertex=FlowSolutionVertexName,
                                 CellCenter=FlowSolutionCellCenterName)
    fields_names = dict()
    containers_names = dict()
    nodes = dict()
    locations = dict()
    FlowSolutions = I.getNodesFromType1(zone,'FlowSolution_t')
    for fs in FlowSolutions:
        if not I.getNodesFromType1(fs, 'DataArray_t'):
            try:
                FlowSolutions.pop(fs)
            except:
                pass
    if not FlowSolutions: return
    sortNodesByName(FlowSolutions)
    for i, fs in enumerate(FlowSolutions):
        if not I.getNodesFromType1(fs, 'DataArray_t'): continue
        loc = _getFlowSolutionLocation(fs)
        tag = '%d'%i
        locations[tag] = loc
        containers_names[tag] = fs[0]
        fields = I.getNodesFromType1(fs,'DataArray_t')
        for f in fields:
            f[0] += tag
            if tag in fields_names:
                fields_names[tag] += [f[0]]
                nodes[tag] += [f]
            else:
                fields_names[tag]  = [f[0]]
                nodes[tag]  = [f]
    
    prev_zone = I.copyRef(zone)
    I._rmNodesByType1(zone,'FlowSolution_t')
    for tag, loc in locations.items():
        try:
            fields_nodes = nodes[tag]
        except KeyError as e:
            C.convertPyTree2File(prev_zone,f'debug_{zone[0]}.cgns')
            raise e

        if not fields_nodes: continue
        fs = I.getNodeFromName1(zone, monoFlowSolutionNames[loc])
        if not fs:
            fs = I.createUniqueChild(zone, monoFlowSolutionNames[loc],
                                        'FlowSolution_t', children=fields_nodes)
            I.createUniqueChild(fs,'GridLocation','GridLocation_t', value=loc,pos=0)
        else:
            fs[2] += fields_nodes

    set(zone, 'tags_containers', fields_names=fields_names,
                    containers_names=containers_names, locations=locations)

def _mergeBCData(zone, BCDataSetFaceCenterName='BCDataSet',
                       BCDataFaceCenterName='NeumannData'):
    '''
    Merge all BCData_t into one at Vertex and one at CellCenter
    
    Consider using higher-level function mergeContainers
    '''
    if zone[3] != 'Zone_t': return AttributeError('first argument must be a zone')
    monoBCDataSetNames = dict(FaceCenter=BCDataSetFaceCenterName)
    zbc = I.getNodeFromName1(zone,'ZoneBC')
    if not zbc: return
    BCs = I.getNodesFromType(zbc,'BC_t')
    if not BCs: return
    sortNodesByName(BCs)
    tags ='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for bc in BCs:
        nb = -1
        fields_names = dict()
        containers_names = dict()
        nodes = dict()
        locations = dict()
        BCDataSets = I.getNodesFromType1(bc,'BCDataSet_t')
        if not BCDataSets: 
            continue
        sortNodesByName(BCDataSets)
        for bcds in BCDataSets:
            loc = _getBCDataSetLocation(bcds)
            if loc != 'FaceCenter':
                path = '/'.join([zone[0],zbc[0],bc[0],bcds[0]])
                raise NotImplementedError(f'BCDataSet {path} must be located at FaceCenter, got {loc} instead')

            BCDatas = I.getNodesFromType1(bcds,'BCData_t')
            if not BCDatas: 
                continue
            sortNodesByName(BCDatas)
            for bcd in BCDatas:
                fields = I.getNodesFromType1(bcd,'DataArray_t')
                if len(fields) == 0:
                    # This BCData node contains no fields, it might be a extraction error.
                    continue
                nb += 1
                tag = tags[nb]
                locations[tag] = loc
                containers_names[tag] = bcds[0]+'/'+bcd[0]
                for f in fields:
                    f[0] += tag
                    if tag in fields_names:
                        fields_names[tag] += [f[0]]
                        nodes[tag] += [f]
                    else:
                        fields_names[tag]  = [f[0]]
                        nodes[tag]  = [f]

        I._rmNodesByType1(bc,'BCDataSet_t')
        for tag, loc in locations.items():
            fields_nodes = nodes[tag]
            if not fields_nodes: continue
            bcds = I.getNodeFromName1(bc, monoBCDataSetNames[loc])
            if not bcds:
                bcds = I.createUniqueChild(bc, monoBCDataSetNames[loc],
                                            'BCDataSet_t')
                I.createUniqueChild(bcds,'GridLocation','GridLocation_t', value=loc)
                I.createUniqueChild(bcds,BCDataFaceCenterName,'BCData_t',
                                         children=fields_nodes)
            else:
                bcd = I.getNodeFromName(bcds, BCDataFaceCenterName)
                bcd[2] += fields_nodes

        set(bc, 'tags_containers', fields_names=fields_names,
                   containers_names=containers_names, locations=locations)

def recoverContainers(t):
    tR = I.copyRef(t)
    for zone in I.getZones(tR):
        _recoverFlowSolutions(zone)
        _recoverBCData(zone)

    return tR

def _recoverFlowSolutions(zone):
    if zone[3] != 'Zone_t': return AttributeError('first argument must be a zone')
    had_multi_containers = I.getNodeFromName1(zone,'tags_containers')
    if not had_multi_containers: return
    fields_containers = get(zone, 'tags_containers')
    fields_names = fields_containers['fields_names']
    containers_names = fields_containers['containers_names']
    locations = fields_containers['locations']
    nodes = dict()

    MergedNodes = dict()
    for fs in I.getNodesFromType1(zone, 'FlowSolution_t'):
        merged_location = _getFlowSolutionLocation(fs)
        fields = I.getNodesFromType1(fs,'DataArray_t')
        if not fields: continue
        if merged_location in MergedNodes:
            MergedNodes[merged_location] += fields
        else:
            MergedNodes[merged_location] = fields

    for tag, loc in locations.items():
        for node_name in fields_names[tag].split():
            node = I.getNodeFromName(MergedNodes[loc], node_name)
            if not node:
                C.convertPyTree2File(zone,f'debug_zone_{rank}.cgns')
                raise ValueError(f'UNEXPECTED: could not find node {node_name}')
            if tag in nodes:
                nodes[tag] += [ node ]
            else:
                nodes[tag]  = [ node ]
    
    I._rmNodesByType1(zone, 'FlowSolution_t')
    for tag, fields in nodes.items():
        for f in fields: f[0] = f[0][:-len(tag)] # remove sufix
        fs = I.createUniqueChild(zone,containers_names[tag],'FlowSolution_t',
                                      children=fields)
        I.createUniqueChild(fs,'GridLocation','GridLocation_t',value=locations[tag], pos=0)
    I._rmNodesByName1(zone,'tags_containers')

def _recoverBCData(zone):
    if zone[3] != 'Zone_t': return AttributeError('first argument must be a zone')
    zbc = I.getNodeFromName1(zone,'ZoneBC')
    if not zbc: return
    for bc in I.getNodesFromType(zbc,'BC_t'):
        had_multi_containers = I.getNodeFromName1(bc,'tags_containers')
        if not had_multi_containers: return
        fields_containers = get(bc, 'tags_containers')
        fields_names = fields_containers['fields_names']
        containers_names = fields_containers['containers_names']
        locations = fields_containers['locations']
        nodes = dict()

        MergedNodes = dict()
        for bcds in I.getNodesFromType1(bc, 'BCDataSet_t'):
            merged_location = _getBCDataSetLocation(bcds)
            for bcd in I.getNodesFromType1(bcds, 'BCData_t'):
                fields = I.getNodesFromType1(bcd,'DataArray_t')
                if not fields: continue
                if merged_location in MergedNodes:
                    MergedNodes[merged_location] += fields
                else:
                    MergedNodes[merged_location] = fields

        for tag, loc in locations.items():
            for node_name in fields_names[tag].split():
                node = I.getNodeFromName(MergedNodes[loc], node_name)
                if not node:
                    C.convertPyTree2File(zone,'debug.cgns')
                    raise ValueError(f'UNEXPECTED: could not find node {node_name}')
                if tag in nodes:
                    nodes[tag] += [ node ]
                else:
                    nodes[tag]  = [ node ]
        
        I._rmNodesByType1(bc, 'BCDataSet_t')
        for tag, fields in nodes.items():
            for f in fields: f[0] = f[0][:-len(tag)] # remove sufix
            bcds_name, bcd_name = containers_names[tag].split('/')
            bcds = I.getNodeFromName1(bc, bcds_name)
            if not bcds:
                bcds = I.createUniqueChild(bc,bcds_name,'BCDataSet_t')
                I.createUniqueChild(bcds,'GridLocation','GridLocation_t',value=locations[tag])

            bcd = I.getNodeFromName1(bcds, bcd_name)
            if not bcd:
                I.createUniqueChild(bcds,bcd_name,'BCData_t', children=fields)
            else:
                bcd[2] += fields
        I._rmNodesByName1(bc, 'tags_containers')

def _getFlowSolutionLocation(FlowSolution_n):
    GridLocation_n = I.getNodeFromType1(FlowSolution_n,'GridLocation_t')
    if not GridLocation_n: return 'Vertex'
    return I.getValue(GridLocation_n)

def _getBCDataSetLocation(BCDataSet_n):
    GridLocationNodes = I.getNodesFromType(BCDataSet_n,'GridLocation_t')
    if not GridLocationNodes: return 'FaceCenter'
    if len(GridLocationNodes) == 1: return I.getValue(GridLocationNodes[0])
    first_name = GridLocationNodes[0][0]
    if not all([n[0]==first_name for n in GridLocationNodes[1:]]):
        print('WARNING: multiple grid locations found')
        return 'Multiple'
    return first_name

def mergeBCtagContainerWithFlowSolutionTagContainer(zone, surf_bc):
    if zone[3] != 'Zone_t': return AttributeError('1st argument must be a zone')    
    if surf_bc[3] != 'Zone_t': return AttributeError('2nd argument must be a zone')    

    zone_tags = I.getNodeFromName1(surf_bc, 'tags_containers')
    if not zone_tags: return
    bcname = surf_bc[0].split('\\')[-1]
    zbc = I.getNodeFromName1(zone,'ZoneBC')
    if not zbc: return
    bc_tags = None
    for bc in I.getNodesFromType1(zbc,'BC_t'):
        if bc[0] != bcname: continue
        bc_tags = I.getNodeFromName1(bc,'tags_containers')
        break
    if not bc_tags: return
    for n in zone_tags[2]:
        bc_n = I.getNodeFromName1(bc_tags, n[0])
        if not bc_n:
            C.convertPyTree2File(zone,'debug_zone.cgns')
            C.convertPyTree2File(surf_bc,'debug_bc.cgns')
            raise ValueError(f'could not find tags_container child {n[0]}')
        n[2] += bc_n[2]
    
    for n in I.getNodeFromName1(zone_tags, 'locations')[2]:
        if I.getValue(n) == 'FaceCenter':
            I.setValue(n,'CellCenter')

    for n in I.getNodeFromName1(zone_tags, 'containers_names')[2]:
        name = I.getValue(n)
        I.setValue(n, name.replace('/NeumannData',''))

def reshapeFieldsForStructuredGrid(t):
    for zone in I.getZones(t):
        topo, Ni, Nj, Nk, dim = I.getZoneDim(zone)
        if topo != 'Structured' or dim==1: continue
        for fs in I.getNodesFromType1(zone,'FlowSolution_t'):
            loc = _getFlowSolutionLocation(fs)
            for n in I.getNodesFromType1(fs,'DataArray_t'):
                if len(n[1].shape) != dim:
                    if dim == 2:
                        if loc == 'Vertex':
                            n[1] = n[1].reshape((Ni,Nj), order='F')
                        elif loc == 'CellCenter':
                            n[1] = n[1].reshape((Ni-1,Nj-1), order='F')
                        else:
                            raise NotImplementedError(f'loc must be "Vertex" or "CellCenter", but got: {loc}')
                    elif dim == 3:
                        if loc == 'Vertex':
                            n[1] = n[1].reshape((Ni,Nj,Nk), order='F')
                        elif loc == 'CellCenter':
                            n[1] = n[1].reshape((Ni-1,Nj-1,Nk-1), order='F')
                        else:
                            raise NotImplementedError(f'loc must be "Vertex" or "CellCenter", but got: {loc}')

def sortNodesByName(nodes):
    names = [n[0] for n in nodes]
    sorted_nodes = sortListsUsingSortOrderOfFirstList(names, nodes)[1]
    nodes[:] = sorted_nodes

def sortListsUsingSortOrderOfFirstList(*arraysOrLists):
    '''
    This function accepts an arbitrary number of lists (or arrays) as input.
    It sorts all input lists (or arrays) following the ordering of the first
    list after sorting.

    Returns all lists with new ordering.

    Parameters
    ----------

        arraysOrLists : comma-separated arrays or lists
            Arbitrary number of arrays or lists

    Returns
    -------

        NewArrays : list
            list containing the new sorted arrays or lists following the order
            of first the list or array (after sorting).

    Examples
    --------

    ::
        First = [5,1,6,4]
        Second = ['a','c','f','h']
        Third = np.array([10,20,30,40])

        NewFirst, NewSecond, NewThird = sortListsUsingSortOrderOfFirstList(First,Second,Third)
        print(NewFirst)
        print(NewSecond)
        print(NewThird)

    will produce

    ::

        [1, 4, 5, 6]
        ['c', 'h', 'a', 'f']
        [20, 40, 10, 30]

    '''
    SortInd = np.argsort(arraysOrLists[0])
    NewArrays = []
    for a in arraysOrLists:
        if type(a) == 'ndarray':
            NewArray = np.copy(a,order='K')
            for i in SortInd:
                NewArray[i] = a[i]

        else:
            NewArray = [a[i] for i in SortInd]

        NewArrays.append( NewArray )

    return NewArrays

def get(parent, childname):
    '''
    Recover the name and values of children of a node named *childname* inside a
    *parent* node. Such pair of name and values are
    recovered as a python dictionary:
    Dict[*nodename*] = **nodevalue**

    Parameters
    ----------
        parent : node
            the CGNS node where the child named *childname* is found
        childname : str
            a child node name contained in node *parent*, from which children
            are extracted. This operation is recursive.

    Returns
    -------
        pointers - (dict):
            A dictionary Dict[*nodename*] = **nodevalue**

    See Also
    --------
    set : set a CGNS node containing children
    '''

    child_n = I.getNodeFromName1(parent,childname)
    Dict = {}
    if child_n is not None:
        for n in child_n[2]:
            if n[1] is not None:
                if isinstance(n[1], float) or isinstance(n[1], int):
                    Dict[n[0]] = np.atleast_1d(n[1])
                elif n[1].dtype == '|S1':
                    Dict[n[0]] = I.getValue(n) # Cannot further modify
                else:
                    Dict[n[0]] = np.atleast_1d(n[1]) # Can further modify
            elif n[2]:
                Dict[n[0]] = get(child_n, n[0])
            else:
                Dict[n[0]] = None
    return Dict

def set(parent, childname, childType='UserDefinedData_t', **kwargs):
    '''
    Set (or add, if inexistent) a child node containing an arbitrary number
    of nodes.

    Return the pointers of the new CGNS nodes as a python dictionary get()

    Parameters
    ----------

        parent : node
            root node where children will be added

        childname : str
            name of the new child node.

        **kwargs
            Each pair *name* = **value** will be a node of
            type `DataArray_t`_ added as child to the node named *childname*.
            If **value** is a python dictionary, then their contents are added
            recursively following the same logic

    Returns
    -------

        pointers : dict
            literally, result of :py:func:`get` once all nodes have been
            added
    '''
    children = []
    SubChildren = []
    SubSet = []
    for v in kwargs:
        if isinstance(kwargs[v], dict):
            SubChildren += [[v,kwargs[v]]]
        elif isinstance(kwargs[v], list):
            if len(kwargs[v])>0:
                if isinstance(kwargs[v][0], str):
                    value = ' '.join(kwargs[v])
                elif isinstance(kwargs[v][0], dict):
                    p = I.createNode(v, childType)
                    for i in range(len(kwargs[v])):
                        set(p, 'set.%d'%i, **kwargs[v][i])
                    SubSet += [p]
                    continue
                else:
                    try:
                        value = np.atleast_1d(kwargs[v])
                        if value.dtype == 'O': 
                            # 'O' for 'Object'. It generally means that kwargs[v] contains mixed values
                            value = None
                    except ValueError:
                        value = None

            else:
                value = None
            children += [[v,value]]
        else:
            children += [[v,kwargs[v]]]
    _addSetOfNodes(parent,childname,children, type1=childType)
    NewNode = I.getNodeFromName1(parent,childname)
    NewNode[2].extend(SubSet)
    for sc in SubChildren: set(NewNode, sc[0], **sc[1])

    return get(parent, childname)

def _addSetOfNodes(parent, name, ListOfNodes, type1='UserDefinedData_t', type2='DataArray_t'):
    '''
    parent : Parent node
    name : name of the node
    ListOfNodes : First element is the node name,
    and the second element is its value...
    ... -> [[nodename1, value1],[nodename2, value2],etc...]
    '''

    children = []
    for e in ListOfNodes:
        typeOfNode = type2 if len(e) == 2 else e[2]
        children += [I.createNode(e[0],typeOfNode,value=e[1])]

    node = I.createUniqueChild(parent,name,type1, children=children)
    I._rmNodesByName1(parent, node[0])
    I.addChild(parent, node)


def _remove_suffix_if_single_container(zone):
    for container_per_location in _get_containers_per_location(zone):
        if len(container_per_location) == 1:
            _remove_suffix_of_fields(container_per_location[0])

def _get_containers_per_location(zone):
    containers_at_vertex = []
    containers_at_center = []
    for container in I.getNodesFromType1(zone, "FlowSolution_t"):
        grid_location = I.getNodeFromName1(container, 'GridLocation')
        
        if not grid_location:
            raise ValueError(f"expected GridLocation at node {zone[0]}/{container[0]}")
        
        grid_location_value = I.getValue(grid_location)
        if grid_location_value == 'Vertex':
            containers_at_vertex += [ container ]
        elif grid_location_value == 'CellCenter':
            containers_at_center += [ container ]
        else:
            raise TypeError(f"not supporting GridLocation {grid_location_value} at {zone[0]}/{container[0]}")
        
    return containers_at_vertex, containers_at_center

def _remove_suffix_of_fields(container):
    for field_node in I.getNodesFromType1(container,'DataArray_t'):
        field_node_name = I.getName(field_node)
        if field_node_name.endswith('0'):
            I.setName(field_node,field_node_name[:-1])