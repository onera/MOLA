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
import shutil
from fnmatch import fnmatch
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
from mola.cfd.coprocess.manager import mpi_allgather_and_merge_trees, update_signals_using, get_bc_families_in_extraction
from mola.cfd.preprocess.solver_specific_tools.solver_sonics import translate_sonics_CGNS_field_names_to_MOLA


def perform_extractions(workflow, coprocess_manager):
    output_tree = get_output_tree(coprocess_manager)
    families_to_bctype = get_family_to_BCType(output_tree)

    for extraction in coprocess_manager.Extractions:
        if extraction['IsToExtract'] == False:
            continue

        coprocess_manager.mola_logger.debug(f'  update extraction of type {extraction["Type"]}', rank=0)
        
        if extraction['Type'] == 'Restart':
            coprocess_manager.iteration = workflow.Numerics['NumberOfIterations']
            update_restart_fields(workflow, coprocess_manager.output_tree)
            extraction['Data'] = workflow.tree
        
        elif extraction['Type'] == '3D':
            extraction['Data'] = extract_fields(output_tree, extraction)

        elif extraction['Type'] == 'BC':
            extraction['Data'] = extract_bc(output_tree, extraction, families_to_bctype)
        
        elif extraction['Type'] == 'IsoSurface':
            extraction['Data'] = extract_isosurface(output_tree, extraction)
        
        elif extraction['Type'] == 'Integral':
            extract_integral(output_tree, extraction, families_to_bctype, NumberOfIterations=workflow.Numerics['NumberOfIterations'])  

        elif extraction['Type'] == 'Residuals':
            extract_residuals(extraction, 
                              coprocess_manager.workflow.Flow['Conservatives'],
                              coprocess_manager.workflow.Turbulence['Conservatives']
                              )
        
        else:
            coprocess_manager.mola_logger.warning(f"Type of extraction {extraction['Type']} is not available for SoNICS", rank=0)
            extraction['Data'] = cgns.Tree()

def get_output_tree(coprocess_manager):
    # output_tree is set in compute/solver_sonics.py
    output_tree = coprocess_manager.output_tree
    # partionning
    part_tree = maia.factory.partition_dist_tree(output_tree, MPI.COMM_WORLD)
    maia.transfer.dist_tree_to_part_tree_all(output_tree, part_tree, comm=MPI.COMM_WORLD)
    part_tree = cgns.castNode(part_tree)
    for zsr in part_tree.group(Type='ZoneSubRegion'):
        cgns.Node(Name='GridLocation', Type='GridLocation', Value='FaceCenter', Parent=zsr)
    
    return part_tree

def update_restart_fields(workflow, output_tree):
    for zone in output_tree.zones():
        zone.findAndRemoveNode(Name='FSolution#CellCenter#Init')
        FS = zone.get(Name='FSolution#CellCenter#EndOfRun')
        if FS is not None: 
            FS.setName('FSolution#CellCenter#Init')

    NodesToUpdate = output_tree.group(Name='FSolution#CellCenter#Init*', Type='FlowSolution', Depth=3) # for initial field(s) (possible second order restart)
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

    for zone in t.zones():
        # Remove FlowSolution nodes that are not the target
        for FS in zone.group(Type='FlowSolution', Depth=1):
            if FS.name() != extraction['Container']:
                FS.remove()
        
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

    return SurfacesTree

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
    
    return isosurface

def extract_integral(output_tree, extraction, DictBCNames2Type, NumberOfIterations) -> None:

    families_to_extract = get_bc_families_in_extraction(extraction, DictBCNames2Type)
    
    t = cgns.Tree()
    base = cgns.Base(Name='Integral', Parent=t)
    for IntegralDataNode in output_tree.group(Name='*:GCH', Type='ConvergenceHistory', Depth=2):
        family = IntegralDataNode.name().split(':')[0]

        if family not in families_to_extract: 
            continue

        IntegralDataNode.dettach()
        IntegralDataNode.setName('FlowSolution')
        IntegralDataNode.setType('FlowSolution_t')
        IntegralDataNode.setValue(None)
        for n in IntegralDataNode.children(): 
            n.setType('DataArray_t')
        translate_sonics_CGNS_field_names_to_MOLA(IntegralDataNode)
        cgns.Node(Name='IterationNumber', Type='DataArray', Value=np.arange(NumberOfIterations, dtype=float), Parent=IntegralDataNode)
        zone = cgns.Zone(Name=family, Parent=base, Children=[IntegralDataNode])
        zone.setParameters('MOLA:Extraction-Log',**extraction)

    current_iteration_signals = mpi_allgather_and_merge_trees(t)

    if 'Data' in extraction and extraction['Data'] is not None:
        and_previous_signals_to_be_updated = extraction['Data']
        update_signals_using(current_iteration_signals, and_previous_signals_to_be_updated)
    else: 
        extraction['Data'] = current_iteration_signals

def extract_residuals(extraction, Conservatives, TurbConservatives):

    t = cgns.Tree()

    if rank ==0:

        conservatives_residuals_filename = os.path.join(names.DIRECTORY_LOG, 'residual-normalize(norm_l2(ExplicitIncrement(mean_flow))).npy')
        turbulence_residuals_filename = os.path.join(names.DIRECTORY_LOG, 'residual-normalize(norm_l2(ExplicitIncrement(turbulence_closure))).npy')

        residuals = dict()
        if os.path.isfile(conservatives_residuals_filename):
            with open(conservatives_residuals_filename, 'rb') as f:
                residuals['IterationNumber'] = np.load(f, allow_pickle=True)
                data = np.load(f, allow_pickle=True)
                for i, name in enumerate(Conservatives):
                    residuals[name] = np.array([d[i] for d in data])

        if os.path.isfile(turbulence_residuals_filename):
            with open(turbulence_residuals_filename, 'rb') as f:
                residuals['IterationNumber'] = np.load(f, allow_pickle=True)
                data = np.load(f, allow_pickle=True)
                for i, name in enumerate(TurbConservatives):
                    residuals[name] = np.array([d[i] for d in data])

        if residuals: 
            # base/zone/FlowSolution structure required for allowing conversion to tecplot fmt
            base = cgns.Base(Name='Residuals', Parent=t)
            zone = cgns.utils.newZoneFromDict(base.name(), residuals)
            zone.attachTo(base)

    current_iteration_signals = mpi_allgather_and_merge_trees(t)

    if 'Data' in extraction and extraction['Data'] is not None:
        and_previous_signals_to_be_updated = extraction['Data']
        update_signals_using(current_iteration_signals, and_previous_signals_to_be_updated)
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
        return 'FSolution#CellCenter#Init'
    
def move_log_files(w):
    if rank == 0:
        filename = 'taskflow-residual-explicit-rank0-sync.dot'
        try:
            shutil.move(filename, os.path.join(names.DIRECTORY_LOG, filename))
        except FileNotFoundError:
            pass
    comm.barrier()

def get_iteration(workflow):
    return workflow.Numerics['NumberOfIterations']-1 # TODO

def get_status(workflow):
    return 'RUNNING_BEFORE_ITERATION' # TODO: implement this (using elsaXdt?)
