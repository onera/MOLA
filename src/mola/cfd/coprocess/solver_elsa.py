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

from treelab import cgns

from mola.logging import MolaException
import mola.naming_conventions as names
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
from mola.cfd.coprocess.probes import extract_probe
import mola.cfd.postprocess as POST
from mola.cfd.preprocess.mesh.tools import ravel_FlowSolution, remove_empty_BCDataSet, force_FamilyBC_as_FamilySpecified
from mola.cfd.preprocess.mesh.families import get_family_to_BCType
from mola.cfd.preprocess.solver_specific_tools.solver_elsa import translate_elsa_CGNS_field_names_to_MOLA

from mola.cfd.postprocess.signals.tree_manipulation import update_zones_shape_using_iteration_number


def perform_extractions(workflow, coprocess_manager):
    output_tree = get_elsa_output_tree(workflow._Skeleton)
    families_to_bctype = get_family_to_BCType(output_tree)
   
    for extraction in coprocess_manager.Extractions:
        if not extraction['IsToExtract']:
            continue

        coprocess_manager.mola_logger.debug(f'  update extraction of type {extraction["Type"]}', rank=0)
        
        if extraction['Type'] == 'Restart':
            update_restart_fields(workflow, output_tree)
            extraction['Data'] = workflow.tree
        
        elif extraction['Type'] == '3D':
            extraction['Data'] = extract_fields(output_tree, extraction)

        elif extraction['Type'] == 'BC':
            extraction['Data'] = extract_bc(output_tree, extraction, families_to_bctype)
        
        elif extraction['Type'] == 'IsoSurface':
            extraction['Data'] = extract_isosurface(output_tree, extraction)
            remove_not_needed_fields(extraction)

        elif extraction['Type'] == 'Residuals':
            extract_residuals(output_tree, extraction)
        
        elif extraction['Type'] == 'Integral':
            extract_integral(output_tree, extraction)            

        elif extraction['Type'] == 'Probe':
            extract_probe(output_tree, extraction, coprocess_manager)

        elif extraction['Type'] == 'MemoryUsage':
            extract_memory_usage(extraction, coprocess_manager.iteration) 

        elif extraction['Type'] == 'TimeMonitoring':
            extract_time_monitoring(extraction, coprocess_manager)

        else:
            coprocess_manager.mola_logger.warning(f"Type of extraction {extraction['Type']} is not available for elsA", rank=0)
            extraction['Data'] = cgns.Tree()

        if workflow.SplittingAndDistribution['Splitter'].lower() == 'pypart': 
            # Remove PyPart nodes for data that are not 3D (important to save them without PyPart)
            if extraction['Type'] not in ['Restart', '3D']:
                extraction['Data'].findAndRemoveNodes(Name=':CGNS#Ppart', Depth=3)

        write_extraction_log(extraction)

        comm.barrier()

def get_elsa_output_tree(skeleton):
    '''
    Extract the coupling CGNS PyTree from elsAxdt *OUTPUT_TREE* and make
    necessary adaptions, including migration of coordinates fields to
    GridCoordinates_t nodes, renaming of conventional fields names and
    adding the tree's Skeleton.

    Returns
    -------

        t : PyTree
            Coupling adapted PyTree

    '''
    import elsAxdt
    t = elsAxdt.get(elsAxdt.OUTPUT_TREE)
    t = cgns.castNode(t)   
    t.merge(skeleton)
    # ravel_FlowSolution(t) # don't ravel! this breaks multi-indexing!
    remove_empty_BCDataSet(t)
    # force_FamilyBC_as_FamilySpecified(t) # HACK https://elsa.onera.fr/issues/10928
    t.findAndRemoveNodes(Name='FlowSolution#Init*', Type='FlowSolution', Depth=3)
    # HACK Pypart puts WorkflowParameters under the base... need to remove it
    t.findAndRemoveNodes(Name=names.CONTAINER_WORKFLOW_PARAMETERS, Type='UserDefinedData', Depth=2) 
    rename_cellnf_fields(t)
    replace_relative_coordinates_with_absolute(t)
    return t

def rename_cellnf_fields(tree : cgns.Tree):
    
    for zone in tree.zones():
        for container in zone.group(Type='FlowSolution_t'):
            for field_node in container.children():
                name = field_node.name()
                if name == 'cellnf':
                    field_node.setName("cellN")


def replace_relative_coordinates_with_absolute(tree : cgns.Tree):
    
    for zone in tree.zones():

        absolute_coordinates = zone.get(Name='FlowSolution#EndOfRun#Coords',Depth=1)
        if not absolute_coordinates: continue

        zone.findAndRemoveNodes(Name='GridCoordinates', Type='GridCoordinates_t', Depth=1)

        absolute_coordinates.findAndRemoveNodes(Name='GridLocation',Depth=1)
        
        absolute_coordinates.setName('GridCoordinates')
        absolute_coordinates.setType('GridCoordinates_t')
        

def update_restart_fields(workflow, output_tree):
    output_tree = cgns.castNode(output_tree)
    for zone in output_tree.zones():
        zone.findAndRemoveNode(Name='FlowSolution#Init')
        FS = zone.get(Name='FlowSolution#EndOfRun')
        if FS is not None: 
            FS.setName('FlowSolution#Init')

    NodesToUpdate = output_tree.group(Name='FlowSolution#Init*', Type='FlowSolution', Depth=3) # for initial field(s) (possible second order restart)
    NodesToUpdate += output_tree.group(Name='FlowSolution#Average', Type='FlowSolution', Depth=3) 
    NodesToUpdate += output_tree.group(Name='BCDataSet#Average') 
    NodesToUpdate += output_tree.group(Name='ChoroData') 

    for node in NodesToUpdate:        
        parent = node.Parent
        parent_in_main_tree = workflow.tree.getAtPath(parent.path())
        node_to_update = parent_in_main_tree.get(Name=node.name(), Depth=1)
        # This node could not exist previously, for instance ChoroData, FlowSolution#Average or BCDataSet#Average
        if node_to_update is not None:
            node_to_update.remove()
        parent_in_main_tree.addChild(node)
    
    workflow.tree = cgns.castNode(workflow.tree)

def extract_fields(output_tree, extraction):

    t = output_tree.copy()
    t.findAndRemoveNodes(Name='GlobalConvergenceHistory', Depth=2)
    t.findAndRemoveNodes(Type='IntegralData', Depth=2)
    t.findAndRemoveNodes(Name='ELSA_TRIGGER')


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
    
        data_tree = POST.extract_bc(output_tree, Family=family, BaseName=family, tool='cassiopee')
        data_tree = cgns.castNode(data_tree)

        SurfacesTree.merge(data_tree)
    
    if extraction['Name'] != 'ByFamily' and len(SurfacesTree.bases()) > 0:
        POST.merge_bases_and_rename_unique_base(SurfacesTree, extraction['Name'])

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
    
    solver_output_name = extraction["_ElsaSolverOutputName"]
    bc_data_set_name = solver_output_name.replace(".Solver#Output","BCDataSet")
    
    for zone in tree.zones():
        container = zone.get(Name=bc_data_set_name, Type="FlowSolution_t", Depth=1)
        if container:
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
        tool = 'maia' if output_tree.isUnstructured() else 'cassiopee',
        )
    
    POST.keep_only_requested_containers(isosurface, extraction)
    POST.keep_only_requested_fields(isosurface, extraction)
    
    return isosurface

def extract_residuals(output_tree, extraction):
    
    t = cgns.Tree()
    residuals = output_tree.base().get(Name='GlobalConvergenceHistory', Depth=2)
    if residuals and residuals.get(Name='IterationNumber') is not None: 
        residuals = residuals.copy()
        residuals.findAndRemoveNode(Name='.Solver#Output')
        base = cgns.Base(Name='Residuals', Parent=t)
        # base/zone/FlowSolution structure required for allowing conversion to tecplot fmt
        residuals.setType('FlowSolution_t')
        residuals.setName('FlowSolution')
        residuals.setValue(None)
        residuals.get(Name='IterationNumber').setName('Iteration')
        cgns.Zone(Name=base.name(), Parent=base, Children=[residuals])

    current_iteration_signals = mpi_allgather_and_merge_trees(t)

    if 'Data' in extraction and extraction['Data'] is not None:
        previous_signals_to_be_updated = extraction['Data']
        update_signals_using(current_iteration_signals, previous_signals_to_be_updated)
    else: 
        extraction['Data'] = current_iteration_signals


def extract_integral(output_tree, extraction) -> None:

    def get_family_and_suffix(IntegralDataNode):
        # The name of IntergralData_t node is <Family>-<SUFFIX>: with <SUFFIX> is given from .Solver#Output<SUFFIX>
        name_without_double_points = IntegralDataNode.name()[:-1]  # remove final ':'
        full_name_parts = name_without_double_points.split('-')
        family = full_name_parts[0]
        try:
            suffix = full_name_parts[1]
        except: 
            suffix = ''
        return family, suffix

    IntegralDataTree = cgns.Tree()
    base = cgns.Base(Name='Integral', Parent=IntegralDataTree)

    for IntegralDataNode in output_tree.group(Type='IntegralData', Depth=2):
        family, _ = get_family_and_suffix(IntegralDataNode)
        if family == extraction['Source']: 
            data_node = IntegralDataNode.copy()
            data_node.setName('FlowSolution')
            data_node.setType('FlowSolution_t')
            for n in data_node.children(): 
                n.setType('DataArray_t')
            translate_elsa_CGNS_field_names_to_MOLA(data_node)

            zone = cgns.Zone(Name=extraction['Name'], Parent=base, Children=[data_node])

            # sort data: keep only data required in extraction
            POST.keep_only_requested_fields(base, extraction)

            # multiply integrated data by the FluxCoef
            for node in zone.group(Type='DataArray'):
                if node.name() != 'Iteration':
                    node.setValue(node.value() * extraction['FluxCoef'])

    current_iteration_signals = mpi_allgather_and_merge_trees(IntegralDataTree)

    if 'Data' in extraction and extraction['Data'] is not None:
        previous_signals_to_be_updated = extraction['Data']
        update_signals_using(current_iteration_signals, previous_signals_to_be_updated)
    else: 
        extraction['Data'] = current_iteration_signals
    
    remove_iteration_zero_from_integrals(extraction['Data'])
    update_zones_shape_using_iteration_number(extraction['Data'], Container="FlowSolution")


def remove_iteration_zero_from_integrals(extraction_tree : cgns):

    for zone in extraction_tree.zones():
        for container in zone.group(Type="FlowSolution_t", Depth=1):
            iteration = zone.fields(['Iteration'], container.name(), 'raise')

            if iteration[0] >= 0.9 or iteration.size == 1:
                continue

            for field_node in container.group(Type='DataArray_t', Depth=1):
                field = field_node.value()
                field_node.setValue(field[1:])


def extract_time_monitoring(extraction, coprocess_manager):
    # At the end of elsA_MPI* file, the following lines can be found: 
    # ---------------------------------------------
    # [ (CPU Time)/(Iteration*NbCell) ] (User) =      2.859961e+00 (µs/ite/nCel)
    #                                   (Sys)  =      2.465483e-01 (µs/ite/nCel)
    # [AdimCoef (Loc/Glob) ] =      4.930966e-06 / 4.930966e-06
    # ---------------------------------------------
    # and also the line:
    # Task (proc : 0) took 3.4062766e+01 seconds  (resolution = 1.0000000e-09 s)

    t = cgns.Tree()
    if rank == 0:

        base = cgns.Base(Name='TimeMonitoring', Parent=t)
        InitialIteration = coprocess_manager.workflow.Numerics['IterationAtInitialState']
        zone = cgns.Zone(Name=f'From{InitialIteration}To{coprocess_manager.iteration}', Parent=base)
        fs = cgns.Node(Name='FlowSolution', Type='FlowSolution', Parent=zone)
        cgns.Node(Name='Iteration', Type='DataArray', Parent=fs, Value=np.array([coprocess_manager.iteration]))
        cgns.Node(Name='TotalRealTime', Type='DataArray', Parent=fs, Value=np.array([coprocess_manager.elapsed_time()]))

        try:
            elsA_log_file = glob.glob('elsA_MPI*_N_0')[0]
        except IndexError:
            # Cannot find this elsA_MPI* file
            return
        
        user_time = None
        with open(elsA_log_file, 'r') as file:
            for line in file:
                if '[ (CPU Time)/(Iteration*NbCell) ] (User) = ' in line:
                    user_time_µs = float(line.split('=')[-1].split('(µs/ite/nCel)')[0])
                    user_time = user_time_µs / 1e6
                    break
        if user_time:
            cgns.Node(Name='TimePerCellPerIteration', Type='DataArray', Parent=fs, Value=np.array([user_time]))
    
        if 'Data' in extraction and extraction['Data'] is not None:
            extraction['Data'].merge(t)
        else: 
            extraction['Data'] = t
    else:
        extraction['Data'] = t

def update_elsa_input(new_tree):
    import elsAxdt
    elsAxdt.xdt(elsAxdt.PYTHON,(elsAxdt.RUNTIME_TREE, new_tree, 1))

def end_simulation(workflow):
    import elsAxdt
    elsAxdt.safeInterrupt()

def deduce_container_for_slicing(IsoSurfaceField):
    if IsoSurfaceField in ['CoordinateX', 'CoordinateY', 'CoordinateZ']:
        return 'GridCoordinates'

    elif IsoSurfaceField in ['Radius', 'radius', 'CoordinateR', 'Slice']:
        return 'FlowSolution'

    elif IsoSurfaceField == 'ChannelHeight':
        return 'FlowSolution#Height'
    
    else:
        return 'FlowSolution#EndOfRun'

def move_log_files(w):
    if rank == 0:
        for fn in glob.glob('elsA_MPI*'):
            shutil.move(fn, os.path.join(names.DIRECTORY_LOG, fn))

    comm.barrier()

def get_iteration(workflow):
    import elsAxdt
    status = get_status(workflow)
    if status == 'RUNNING_BEFORE_ITERATION':
        return elsAxdt.iteration() - 1
    elif status == 'RUNNING_AFTER_ITERATION':
        return elsAxdt.iteration()
    else: 
        raise Exception(f'unknown status: {status}')

def get_status(workflow):
    # BEWARE! state 16 => triggers *before* iteration
    return 'RUNNING_BEFORE_ITERATION' # TODO: implement this (using elsaXdt?)
