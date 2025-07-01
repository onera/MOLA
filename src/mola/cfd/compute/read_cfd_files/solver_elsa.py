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
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()

import maia

from treelab import cgns
import mola.naming_conventions as names
from mola.logging import MolaAssertionError

import Distributor2.PyTree as D2 

def apply_to_solver(workflow):
    
    splitter = workflow.SplittingAndDistribution['Splitter'].lower()

    if splitter == 'cassiopee':
        xdt_object = read_workflow_with_cassiopee(workflow)
    elif splitter == 'pypart': 
        xdt_object = read_workflow_with_pypart(workflow)
    elif splitter == 'maia': 
        # from packaging.version import Version
        # if Version(os.getenv('ELSAVERSION', '0.0.0')) < Version('5.3.03'):
        #     raise MolaAssertionError(
        #         'Maia splitter is not available for elsA simulations with MOLA ' 
        #         'for elsA versions less than v5.3.03'
        #         )
        xdt_object = read_workflow_with_maia(workflow)
    else:
        raise MolaAssertionError(f'Unknown Splitter "{splitter}". Must be either cassiopee, pypart or maia.')        

    return xdt_object

def read_workflow_with_pypart(workflow):
    import elsAxdt

    part_tree, skeleton, PyPartBase = read_and_split_with_pypart(names.FILE_INPUT_SOLVER)
    add_data_in_skeleton(skeleton, part_tree)

    workflow.tree = cgns.castNode(part_tree)
    workflow._Skeleton = cgns.castNode(skeleton)
    remove_workflow_attributes_from_bases(workflow, workflow.tree)
    remove_workflow_attributes_from_bases(workflow, workflow._Skeleton)
    workflow._PyPartBase = PyPartBase

    e = elsAxdt.XdtCGNS(tree=workflow.tree, links=[], paths=[])
    e.distribution = D2.getProcDict(workflow._Skeleton, prefixByBase=True)

    return e

def read_workflow_with_cassiopee(workflow):
    # For simulation with Chimera method (xdt will read the file)
    import Converter.Mpi as Cmpi
    import elsAxdt
    
    skeleton = Cmpi.convertFile2SkeletonTree(workflow.tree)
    workflow.read_tree('cassiopee_mpi')
    add_data_in_skeleton(skeleton, workflow.tree)
    workflow._Skeleton = cgns.castNode(skeleton)

    e = elsAxdt.XdtCGNS(names.FILE_INPUT_SOLVER)

    return e

# def read_workflow_with_cassiopee_bis(workflow):
#     import elsAxdt
#     import Converter.Mpi as Cmpi

#     skeleton = Cmpi.convertFile2SkeletonTree(workflow.tree)
#     workflow.read_tree('cassiopee_mpi')
#     add_data_in_skeleton(skeleton, workflow.tree)
#     workflow._Skeleton = cgns.castNode(skeleton)

#     Cmpi._convert2PartialTree(workflow.tree)
#     workflow.tree = cgns.castNode(workflow.tree)

#     # Cmpi.convertPyTree2File(workflow.tree, f'test.cgns')
#     import Converter.PyTree as C
#     C.convertPyTree2File(workflow.tree, f'test_{Cmpi.rank}.cgns')

#     e = elsAxdt.XdtCGNS(tree=workflow.tree, links=[], paths=[])
#     e.distribution = D2.getProcDict(workflow._Skeleton, prefixByBase=True)

#     return e

def read_workflow_with_maia(workflow):
    import elsAxdt
    import maia4elsA

    def add_FlowSolution_EoR(t):
        from mola.cfd.preprocess.extractions.solver_elsa import add_3d_extraction_to_zone
        for Extraction in workflow.Extractions:
            if Extraction['Type'] == 'Restart':
                break
        for zone in t.zones():
            add_3d_extraction_to_zone(zone, Extraction)

    is_to_split_with_maia = workflow.SplittingAndDistribution['Strategy'].lower() == 'atcomputation' \
        and workflow.SplittingAndDistribution['Splitter'].lower() == 'maia'
    
    was_already_split = workflow.SplittingAndDistribution['Strategy'].lower() == 'atpreprocess'  # whatever the splitter
    
    if is_to_split_with_maia:
        workflow.read_tree('maia')
        part_tree = maia.factory.partition_dist_tree(workflow.tree, comm, data_transfer='ALL')

    elif was_already_split:
        # distribution = D2.getProcDict(workflow.tree, prefixByBase=True)   
        # zone_to_parts = dict((zone_proc[0], [1.]) for zone_proc in distribution.items() if zone_proc[1]==rank)   
        if workflow.RunManagement['NumberOfProcessors'] == 1:
            workflow.read_tree('maia')
            part_tree = maia.factory.partition_dist_tree(workflow.tree, comm, data_transfer='ALL')
        else:
            part_tree = maia.io.file_to_part_tree(workflow.tree, comm) 

    else:
        raise MolaAssertionError('The splitting strategy is not taken into account.')
    
    part_tree = cgns.castNode(part_tree)
    add_FlowSolution_EoR(part_tree)  # HACK https://gitlab.onera.net/numerics/mesh/maia/-/issues/164 TODO solved for maia>1.5
    for zone in part_tree.zones():
        SolverParam = zone.get(Name='.Solver#Param')
        if not SolverParam:
            zone.setParameters('.Solver#Param', proc=rank)
        else:
            SolverParam.findAndRemoveNode(Name='proc', Depth=1)
            cgns.Node(Name='proc', Value=rank, Type='DataArray', Parent=SolverParam)

    maia4elsA.add_renumbering_data(part_tree)
    skeleton_tree = maia4elsA.get_skeleton_tree(part_tree, comm)
    add_data_in_skeleton(skeleton_tree, part_tree)
    distribution = maia4elsA.get_distribution(part_tree, comm)
    part_tree = maia.pytree.union(skeleton_tree, part_tree)  
    workflow.tree = cgns.castNode(part_tree)
    workflow._Skeleton = cgns.castNode(skeleton_tree)

    if workflow.tree.isHybrid():
        # see https://elsa.onera.fr/issues/11718
        maia4elsA.adapt_hybrid_join_pointlist(workflow.tree)

    e = elsAxdt.XdtCGNS(tree=workflow.tree, links=[], paths=[])
    e.distribution = distribution

    return e

def add_data_in_skeleton(Skeleton, PartTree):
    import Converter.Internal as I
    import Converter.Mpi as Cmpi

    I._rmNodesByName(Skeleton, 'FlowSolution*')

    # Needed nodes are read from PartTree
    def readNodesFromPaths(path):
        split_path = path.split('/')
        path_begining = '/'.join(split_path[:-1])
        name = split_path[-1]
        parent = I.getNodeFromPath(PartTree, path_begining)
        return I.getNodesFromName(parent, name)
        
    def replaceNodeByName(parent, parentPath, name):
        oldNode = I.getNodeFromName1(parent, name)
        path = '{}/{}'.format(parentPath, name)
        newNode = readNodesFromPaths(path)
        I._rmNode(parent, oldNode)
        I._addChild(parent, newNode)

    def replaceNodeValuesRecursively(node_skel, node_path):
        new_node = readNodesFromPaths(node_path)[0]
        node_skel[1] = new_node[1]
        for child in node_skel[2]:
            replaceNodeValuesRecursively(child, node_path+'/'+child[0])
        
    # containers2read = ['FlowSolution#Height',
    #                    ':CGNS#Ppart',
    #                    'FlowSolution#DataSourceTerm',
    #                    'FlowSolution#Average']

    containers2read = [':CGNS#Ppart', ':CGNS#Distribution', ':CGNS#GlobalNumbering']
    containers2read += ['FlowSolution#Height']
    if not I.getNodeFromName1(PartTree, 'FlowSolution#EndOfRun#Coords'):
        containers2read.append('GridCoordinates')
    
    for base in I.getBases(Skeleton):
        basename = I.getName(base)
        for zone in I.getNodesFromType1(base, 'Zone_t'):
            # Only for local zones on proc
            proc = I.getValue(I.getNodeFromName(zone, 'proc'))
            if proc != Cmpi.rank: 
                continue

            zonePath = '{}/{}'.format(basename, I.getName(zone))
            zoneInPartialTree = readNodesFromPaths(zonePath)[0]

            for nodeName2read in containers2read:
                if I.getNodeFromName1(zoneInPartialTree, nodeName2read):
                    replaceNodeByName(zone, zonePath, nodeName2read)

def read_and_split_with_pypart(src):
    import Converter.Internal as I
    import Converter.Mpi as Cmpi
    import etc.pypart.PyPart as PPA
    from mpi4py import MPI

    PyPartBase = PPA.PyPart(src,
                            lksearch=[names.DIRECTORY_OUTPUT, '.'],
                            loadoption='partial',
                            mpicomm=MPI.COMM_WORLD,
                            LoggingInFile=False,
                            LoggingFile=os.path.join(names.DIRECTORY_LOG, 'partTree'),
                            LoggingVerbose=40  # Filter: None=0, DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
                            )
    # reorder=[6, 2] is recommended, mostly for unstructured mesh.
    # It is also mandatory to use lussorscawf on unstructured mesh.
    PartTree = PyPartBase.runPyPart(method=2, partN=1, reorder=[6, 2])
    PyPartBase.finalise(PartTree, savePpart=True, method=1)
    Skeleton = PyPartBase.getPyPartSkeletonTree()

    # Put Distribution into the Skeleton
    Distribution = PyPartBase.getDistribution()
    for zone in I.getZones(Skeleton):
        zonePath = I.getPath(Skeleton, zone, pyCGNSLike=True)[1:]
        Cmpi._setProc(zone, Distribution[zonePath])

    PartTree = I.merge([Skeleton, PartTree])

    return PartTree, Skeleton, PyPartBase

def remove_workflow_attributes_from_bases(workflow, tree):
    # Pypart put WorkflowParameters node, and all its children, bellow the Base
    # This function removes them
    attributes_names = list(workflow.convert_to_dict())
    for base in tree.bases():
        base.findAndRemoveNode(Name=workflow._workflow_parameters_container_, Depth=1)
        for name in attributes_names:
            base.findAndRemoveNode(Name=name, Depth=1)
        

def _gc_name_pypart_to_maia(zone):
    import maia.pytree as PT

    name_to_gc = {}
    gc_nodes = PT.get_nodes_from_predicates(zone, 'ZoneGridConnectivity_t/GridConnectivity_t')
    for gc in gc_nodes:
        origin = PT.get_child_from_name(gc, 'OriginName')
        if origin is not None:
            try:
                name_to_gc[PT.get_value(origin)].append(gc[0])
            except KeyError:
                name_to_gc[PT.get_value(origin)] = [gc[0]]
    for gc in gc_nodes:
        origin_n = PT.get_child_from_name(gc, 'OriginName')
        if origin_n is not None:
            origin = PT.get_value(origin_n)
            pos = name_to_gc[origin].index(gc[0])
            PT.set_name(gc, f"{origin}.{pos}")

def pypart_to_maia_for_unstructured_mesh(pypart_tree, pypart_skel_tree=None, recover_jn_donor=True):
    """
    Reorganise a Partitioned CGNS Tree comming from PyPart such that its looks
    like to a Maia Partitioned tree.

    If a skeleton tree is provided, also recover the families and the opposite joins
    name (if recover_jn_donor == True)
    
    Ouput is a shallow copy of the input tree (data array are shared)
    """
    from mpi4py import MPI
    import maia.pytree as PT
    import maia.pytree.maia as MT

    tree = PT.shallow_copy(pypart_tree)

    # Get families from skeletton
    if pypart_skel_tree is not None:
        for base in PT.get_all_CGNSBase_t(tree):
            skel_base = PT.get_child_from_name(pypart_skel_tree, PT.get_name(base))
            for family in PT.get_children_from_label(skel_base, 'Family_t'):
                PT.add_child(base, family)

    # Compute jn donor name using Skeleton, who have all the data \o/
    skel_tree = None
    if pypart_skel_tree is not None and recover_jn_donor:
        from maia.algo.dist.matching_jns_tools import add_joins_donor_name
        skel_tree = PT.shallow_copy(pypart_skel_tree)
        for zone in PT.get_all_Zone_t(skel_tree):
            _gc_name_pypart_to_maia(zone)
        add_joins_donor_name(skel_tree, MPI.COMM_SELF)

    # Reorganise zone data
    for zone in PT.get_all_Zone_t(tree):
        PT.rm_children_from_name(zone, ":elsA#Hybrid")
        PT.print_tree(zone)
        vtx_lngn  = PT.get_node_from_path(zone, ':CGNS#Ppart/npVertexLNToGN')[1]
        face_lngn = PT.get_node_from_path(zone, ':CGNS#Ppart/npFaceLNToGN')[1]
        cell_lngn = PT.get_node_from_path(zone, ':CGNS#Ppart/npCellLNToGN')[1]

        MT.newGlobalNumbering({'Vertex': vtx_lngn, 'Cell': cell_lngn}, zone)

        ngon = PT.Zone.NGonNode(zone)
        MT.newGlobalNumbering({'Element': face_lngn}, ngon)
        nface = PT.Zone.NFaceNode(zone)
        MT.newGlobalNumbering({'Element': cell_lngn}, nface)

        bc_lngn_idx   = PT.get_node_from_path(zone, ':CGNS#Ppart/npFaceGroupIdx')[1]
        bc_lngn       = PT.get_node_from_path(zone, ':CGNS#Ppart/npFaceGroupLNToGN')[1]
        for i, bc in enumerate(PT.get_nodes_from_predicates(zone, 'ZoneBC_t/BC_t')):
            PT.set_name(bc, MT.conv.get_part_prefix(PT.get_name(bc)))
            _bc_lngn = bc_lngn[bc_lngn_idx[i]:bc_lngn_idx[i+1]].copy()
            MT.newGlobalNumbering({'Index': _bc_lngn}, bc)
        
        _gc_name_pypart_to_maia(zone)
        for gc in PT.iter_nodes_from_predicates(zone, 'ZoneGridConnectivity_t/GridConnectivity_t'):
            remove_me = lambda n: PT.get_name(n) in ['minCur', 'maxCur', 'minOpp', 'maxOpp', 'Ordinal']
            PT.rm_children_from_predicate(gc, remove_me)
            origin_n = PT.get_child_from_name(gc, 'OriginName')
            if origin_n is not None:
                lngn_node = PT.get_child_from_name(gc, 'LnToGn')
                MT.newGlobalNumbering({'Index': lngn_node[1]}, gc)
                PT.rm_child(gc, lngn_node)
                PT.rm_child(gc, origin_n)
                if skel_tree is not None:
                    skel_zone  = PT.get_node_from_name(skel_tree, PT.get_name(zone), depth=2)
                    skel_join  = PT.get_node_from_name(skel_zone, PT.get_name(gc), depth=2)
                    donor_name = PT.get_child_from_name(skel_join, f"GridConnectivityDonorName")
                    PT.add_child(gc, donor_name)

        nface_range  = PT.Element.Range(nface)
        pe_node = PT.get_child_from_name(ngon,"ParentElements")
        if pe_node:
          pe = PT.get_value(pe_node)
          pe_no_0 = pe[pe>0]
          min_pe = pe_no_0.min()
          max_pe = pe_no_0.max()
          if not (min_pe==nface_range[0] and max_pe==nface_range[1]):
            if min_pe!=1:
              raise RuntimeError("ParentElements values are not SIDS-compliant, and they do not start at 1")
            else:
              pe += (nface_range[0]-1)*(pe>0)
        
        PT.rm_children_from_name(zone, 'Ordinal')
        PT.rm_children_from_name(zone, ":CGNS#Ppart")

    return tree
