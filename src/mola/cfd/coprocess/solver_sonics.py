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
import glob
import shutil
import numpy as np

import maia
from mpi4py import MPI
from treelab import cgns

import mola.naming_conventions as names
from mola.logging import MolaException
from mola.cfd.preprocess.mesh.families import get_family_to_BCType
import mola.cfd.postprocess as POST
# no relative imports possible for the following line because the current file is called by
# call_solver_specific_function in manager.py
from mola.cfd.coprocess import rank, comm
from mola.cfd.coprocess.tools import (
    mpi_allgather_and_merge_trees, 
    update_signals_using, 
    get_bc_families_in_extraction, 
    write_extraction_log,
    extract_memory_usage,
    remove_not_needed_fields,
)
from mola.cfd.preprocess.solver_specific_tools.solver_sonics import translate_sonics_CGNS_field_names_to_MOLA


def perform_extractions(workflow, coprocess_manager):
    output_tree = get_output_tree(coprocess_manager)
    families_to_bctype = get_family_to_BCType(output_tree)

    for extraction in coprocess_manager.Extractions:
        if not extraction['IsToExtract']:
            continue

        coprocess_manager.mola_logger.debug(f'  update extraction of type {extraction["Type"]}', rank=0)
        
        if extraction['Type'] == 'Restart':
            update_restart_fields(workflow, coprocess_manager.output_tree)
            extraction['Data'] = workflow.tree
        
        elif extraction['Type'] == '3D':
            extraction['Data'] = extract_fields(output_tree, extraction)

        elif extraction['Type'] == 'BC':
            extraction['Data'] = extract_bc(output_tree, extraction, families_to_bctype)
        
        elif extraction['Type'] == 'IsoSurface':
            extraction['Data'] = extract_isosurface(output_tree, extraction)
            remove_not_needed_fields(extraction)
        
        elif extraction['Type'] == 'Integral':
            extract_integral(output_tree, extraction, families_to_bctype)  

        elif extraction['Type'] == 'Residuals':
            extract_residuals(extraction, output_tree)
        
        elif extraction['Type'] == 'MemoryUsage':
            extract_memory_usage(extraction, coprocess_manager.iteration) 

        elif extraction['Type'] == 'TimeMonitoring':
            extract_time_monitoring(extraction, coprocess_manager)
        
        else:
            coprocess_manager.mola_logger.warning(f"Type of extraction {extraction['Type']} is not available for SoNICS", rank=0)
            extraction['Data'] = cgns.Tree()

        write_extraction_log(extraction)

def get_output_tree(coprocess_manager):
    # output_tree is set in compute/solver_sonics.py
    output_tree = coprocess_manager.output_tree
    for gc in output_tree.group(Type='GridConnectivity'):
        if gc.get(Type='GridLocation').value() == 'Vertex':
            gc.remove()  # otherwise, error in the function centers_to_nodes below
    # partionning
    part_tree = maia.factory.partition_dist_tree(output_tree, MPI.COMM_WORLD)
    maia.transfer.dist_tree_to_part_tree_all(output_tree, part_tree, comm=MPI.COMM_WORLD)
    maia.algo.part.centers_to_nodes(part_tree, comm, ['Fields@Cell@End'])
    part_tree = cgns.castNode(part_tree)
    for zsr in part_tree.group(Type='ZoneSubRegion'):
        cgns.Node(Name='GridLocation', Type='GridLocation', Value='FaceCenter', Parent=zsr)
    for fs in part_tree.group(Name='Fields@Cell@End'):
        fs.setName(names.CONTAINER_OUTPUT_FIELDS_AT_CENTER)
    for fs in part_tree.group(Name='Fields@Cell@End#Vtx'):
        fs.setName(names.CONTAINER_OUTPUT_FIELDS_AT_VERTEX)
    
    return part_tree

def update_restart_fields(workflow, output_tree):
    for zone in output_tree.zones():
        zone.findAndRemoveNode(Name='Fields@Cell@Init')
        FS = zone.get(Name='Fields@Cell@End')
        if FS is not None: 
            FS.setName('Fields@Cell@Init')

    NodesToUpdate = output_tree.group(Name='Fields@Cell@Init*', Type='FlowSolution', Depth=3) # for initial field(s) (possible second order restart)
    # NodesToUpdate += output_tree.group(Name='FlowSolution#Average', Type='FlowSolution', Depth=3) 
    # NodesToUpdate += output_tree.group(Name='BCDataSet#Average') 

    for node in NodesToUpdate:
        path = node.path()
        node_to_update = workflow.tree.getAtPath(path)
        parent = node_to_update.Parent
        node_to_update.remove()
        parent.addChild(node)
    
    workflow.tree = cgns.castNode(workflow.tree)

def extract_fields(output_tree, extraction):
    t = output_tree.copy()
 
    t.findAndRemoveNodes(Name='GlobalConvergenceHistory', Depth=2)
    t.findAndRemoveNodes(Type='IntegralData', Depth=2)
    t.findAndRemoveNodes(Type='ZoneSubRegion', Depth=2)

    for gc in t.group(Type='GridConnectivity'):
        if gc.get(Type='GridLocation').value() == 'Vertex':
            gc.remove()

    POST.keep_only_requested_containers(t, extraction)
    POST.keep_only_requested_fields(t, extraction)

    for zone in t.zones():
        
        if not zone.get(Type='FlowSolution', Depth=1):
            # no more FlowSolution in the current zone
            # --> remove this zone
            zone.remove()
            continue
            
        # NOTE ZoneBC must be kept for to save tree with PyPart
        zone.findAndRemoveNodes(Type='BCDataSet')
    
    return t

def extract_bc(output_tree, extraction, DictBCNames2Type):

    SurfacesTree = cgns.Tree()

    families_to_extract = get_bc_families_in_extraction(extraction, DictBCNames2Type)

    for family in families_to_extract:
        data_tree = POST.extract_bc(output_tree, Family=family, BaseName=family, tool='maia_zsr')       
        data_tree = cgns.castNode(data_tree)
        SurfacesTree.merge(data_tree)
    
    if extraction['Name'] != 'ByFamily':
        POST.merge_bases_and_rename_unique_base(SurfacesTree, extraction['Name'])

    # HACK for now remove EdgeElements because otherwise Cassiopee Cmpi bugs when the file is saved
    SurfacesTree.findAndRemoveNodes(Name='EdgeElements', Type='Elements')

    rename_resulting_container_using_requested_name(SurfacesTree, extraction)
    POST.keep_only_requested_containers(SurfacesTree, extraction)
    POST.keep_only_requested_fields(SurfacesTree, extraction)

    return SurfacesTree

def rename_resulting_container_using_requested_name(tree : cgns.Tree, extraction : dict):
    requested_containers = extraction['ContainersToTransfer']
    
    if isinstance(requested_containers,list) and len(requested_containers) == 1:
        expected_container_name = requested_containers[0]
    elif isinstance(requested_containers,str) and requested_containers != 'all':
        expected_container_name = requested_containers
    else:
        return
        
    for zone in tree.zones():
        containers = zone.group(Type="FlowSolution_t", Depth=1)
        if len(containers) > 1:
            container_names = [n.name() for n in containers]
            raise NotImplementedError(f"obtained multiple containers at {zone.path()}: {container_names}")
        elif len(containers) == 0: 
            return
        container = containers[0]
        container.setName(expected_container_name)


def extract_isosurface(output_tree, extraction):
    if extraction['IsoSurfaceContainer'] == 'auto':
        extraction['IsoSurfaceContainer'] = deduce_container_for_slicing(extraction['IsoSurfaceField'])

    isosurface = POST.iso_surface(
        output_tree, 
        IsoSurfaceField = extraction['IsoSurfaceField'], 
        IsoSurfaceValue = extraction['IsoSurfaceValue'], 
        IsoSurfaceContainer = extraction['IsoSurfaceContainer'],
        Name = extraction['Name'],
        tool = 'maia',
        )
    
    POST.keep_only_requested_containers(isosurface, extraction)
    POST.keep_only_requested_fields(isosurface, extraction)

    return isosurface

def extract_integral(output_tree, extraction, DictBCNames2Type) -> None:

    families_to_extract = get_bc_families_in_extraction(extraction, DictBCNames2Type)
    
    t = cgns.Tree()
    base = cgns.Base(Name='Integral', Parent=t)

    for IntegralDataNode in output_tree.group(Type='IntegralData_t', Depth=2) \
        + output_tree.group(Name='*:VALVE', Type='ConvergenceHistory_t', Depth=2):

        IntegralDataNode = IntegralDataNode.copy(deep=True)
        IntegralDataNode_name = IntegralDataNode.name()

        if IntegralDataNode_name.endswith(':VALVE'): 
            # HACK The IntegralDataNode for ValveLawRadialEquilibrium does not contain the node Family
            # Fix that in SoNICS
            family = IntegralDataNode_name.split(':')[0]

        else:
            family = IntegralDataNode.get(Name='Family').value()        

        if family not in families_to_extract: 
            continue

        # IntegralDataNode.dettach()
        IntegralDataNode.setName('FlowSolution')
        IntegralDataNode.setType('FlowSolution_t')
        IntegralDataNode.setValue(None)
        for n in IntegralDataNode.children(): 
            n.setType('DataArray_t')
        translate_sonics_CGNS_field_names_to_MOLA(IntegralDataNode)
        renumber_iterations_for_mola(IntegralDataNode)

        # HACK for now, cgns_node_pattern has no effect on miles IntegralDataExtractor
        # we must split the IntegralDataNode by keeping only the variables required for this extraction
        remove_not_required_fields(extraction, IntegralDataNode)

        if IntegralDataNode_name.endswith(':VALVE'):
            zone = cgns.Zone(Name=IntegralDataNode_name, Parent=base, Children=[IntegralDataNode])
        else:
            zone = cgns.Zone(Name=extraction['Name'], Parent=base, Children=[IntegralDataNode])

        # multiply integrated data by the FluxCoef
        for node in zone.group(Type='DataArray'):
            if node.name() != 'Iteration':
                node.setValue(node.value() * extraction['FluxCoef'])
            

    current_iteration_signals = mpi_allgather_and_merge_trees(t)

    if 'Data' in extraction and extraction['Data'] is not None:
        previous_signals_to_be_updated = extraction['Data']
        update_signals_using(current_iteration_signals, previous_signals_to_be_updated)
    else: 
        extraction['Data'] = current_iteration_signals

def remove_not_required_fields(extraction, IntegralDataNode: cgns.Node):

    required_fields = extraction.get('Fields', [])
    if isinstance(required_fields, str):
        required_fields = [required_fields]

    required_fields.append('Iteration')
    if 'Force' in required_fields:
        required_fields += ['ForceX', 'ForceY', 'ForceZ']
    if 'Torque' in required_fields:
        required_fields += ['TorqueX', 'TorqueY', 'TorqueZ']

    for node in IntegralDataNode.group(Type='DataArray', Depth=1):
        if node.name() not in required_fields:
            node.remove()

def extract_residuals(extraction, output_tree):

    t = cgns.Tree()

    residuals = output_tree.base().get(Name='GlobalConvergenceHistory', Depth=2)
    if residuals: 
        residuals = residuals.copy()
        base = cgns.Base(Name='Residuals', Parent=t)
        # base/zone/FlowSolution structure required for allowing conversion to tecplot fmt
        residuals.setType('FlowSolution_t')
        residuals.setName('FlowSolution')
        residuals.setValue(None)
        renumber_iterations_for_mola(residuals)
        cgns.Zone(Name=base.name(), Parent=base, Children=[residuals])

    current_iteration_signals = mpi_allgather_and_merge_trees(t)

    if 'Data' in extraction and extraction['Data'] is not None:
        previous_signals_to_be_updated = extraction['Data']
        update_signals_using(current_iteration_signals, previous_signals_to_be_updated)
    else: 
        extraction['Data'] = current_iteration_signals

def deduce_container_for_slicing(IsoSurfaceField):
    if IsoSurfaceField in ['CoordinateX', 'CoordinateY', 'CoordinateZ']:
        return 'GridCoordinates'

    elif IsoSurfaceField in ['Radius', 'radius', 'CoordinateR', 'Slice']:
        return 'FlowSolution'

    elif IsoSurfaceField == 'ChannelHeight':
        return 'FlowSolution#Height'
    
    else:
        return 'Fields@Cell@End'
    
def move_log_files(w):
    if rank == 0:
        for filename in glob.glob('taskflow-residual-explicit-rank*-sync.dot') \
            + glob.glob('graph-post-rank*.dot')\
            + glob.glob('graph-full-rank*.*'):
            try:
                shutil.move(filename, os.path.join(names.DIRECTORY_LOG, filename))
            except FileNotFoundError:
                pass
    comm.barrier()

def renumber_iterations_for_mola(flowsolution_node: cgns.Node):
    node = flowsolution_node.get(Name='Iteration')
    iterations = node.value()
    # for now sonics starts at iteration 0. For consistency with other solvers, 
    # MOLA iteration is one more than sonics
    iterations[:] += 1 

def get_iteration(workflow):
    iter_sonics = workflow._iterators.initial + workflow._iterators.niter - 1
    # for now sonics starts at iteration 0. For consistency with other solvers, 
    # MOLA iteration is one more than sonics
    iter_mola = iter_sonics + 1
    return iter_mola

def get_status(workflow):
    return 'RUNNING_BEFORE_ITERATION' # TODO: implement this (using elsaXdt?)

def extract_time_monitoring(extraction, coprocess_manager):
    # At the end of stdout.log, the following line can be found when HookPbSizeTrigger is used: 
    #   + end computation[<iterations>]: time : (<execution_time>, <execution_time_for_all_ranks>, <time/cell/iteration>)

    t = cgns.Tree()
    if rank == 0:
        base = cgns.Base(Name='TimeMonitoring', Parent=t)
        InitialIteration = coprocess_manager.workflow.Numerics['IterationAtInitialState']
        zone = cgns.Zone(Name=f'From{InitialIteration}To{coprocess_manager.iteration}', Parent=base)
        fs = cgns.Node(Name='FlowSolution', Type='FlowSolution', Parent=zone)
        cgns.Node(Name='Iteration', Type='DataArray', Parent=fs, Value=np.array([coprocess_manager.iteration]))
        cgns.Node(Name='TotalRealTime', Type='DataArray', Parent=fs, Value=np.array([coprocess_manager.elapsed_time()]))
        
        TimePerCellPerIteration = None
        with open(names.FILE_STDOUT, 'r') as file:
            for line in file:
                if '+ end computation[' in line:
                    times = line.split('(')[-1].split(')')[0].split(',')
                    TimePerCellPerIteration = float(times[2].replace("'", ""))
                    break
        if TimePerCellPerIteration:
            cgns.Node(Name='TimePerCellPerIteration', Type='DataArray', Parent=fs, Value=TimePerCellPerIteration)

        if 'Data' in extraction and extraction['Data'] is not None:
            extraction['Data'].merge(t)
        else: 
            extraction['Data'] = t
    else:
        extraction['Data'] = t
