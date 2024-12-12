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
from fnmatch import fnmatch
import warnings
import numpy as np

import elsAxdt

from treelab import cgns

from mola.logging import MolaException
import mola.naming_conventions as names
# no relative imports possible for the following line because the current file is called by
# call_solver_specific_function in manager.py
from mola.cfd.coprocess import rank, comm
from mola.cfd.coprocess.manager import (
    mpi_allgather_and_merge_trees, 
    update_signals_using, 
    get_bc_families_in_extraction, 
    write_extraction_log
)
import mola.cfd.postprocess as POST
from mola.cfd.preprocess.mesh.tools import ravel_BCDataSet, remove_empty_BCDataSet, force_FamilyBC_as_FamilySpecified
from mola.cfd.preprocess.mesh.families import get_family_to_BCType
from mola.cfd.preprocess.solver_specific_tools.solver_elsa import translate_elsa_CGNS_field_names_to_MOLA

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

        elif extraction['Type'] == 'Residuals':
            extract_residuals(output_tree, extraction)
        
        elif extraction['Type'] == 'Integral':
            extract_integral(output_tree, extraction)            

        # elif extraction['Type'] == 'Probe':
        #     extraction['Data'] = extract_probe(output_tree)

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
    t = elsAxdt.get(elsAxdt.OUTPUT_TREE)
    t = cgns.castNode(t)
    t.merge(skeleton)
    ravel_BCDataSet(t) # HACK https://elsa.onera.fr/issues/11219
    remove_empty_BCDataSet(t)
    # force_FamilyBC_as_FamilySpecified(t) # HACK https://elsa.onera.fr/issues/10928
    t.findAndRemoveNodes(Name='FlowSolution#Init*', Type='FlowSolution', Depth=3)
    return t

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

    for node in NodesToUpdate:
        path = node.path()
        node_to_update = workflow.tree.getAtPath(path)
        parent = node_to_update.Parent
        node_to_update.remove()
        parent.addChild(node)
    
    workflow.tree = cgns.castNode(workflow.tree)

def extract_fields(output_tree, extraction):

    t = output_tree.copy()
    # HACK Pypart puts WorkflowParameters under the base... need to remove it
    t.findAndRemoveNodes(Name=names.CONTAINER_WORKLFOW_PARAMETERS, Type='UserDefinedData', Depth=2) 
    t.findAndRemoveNodes(Name='GlobalConvergenceHistory', Depth=2)
    t.findAndRemoveNodes(Type='IntegralData', Depth=2)
    t.findAndRemoveNodes(Name='ELSA_TRIGGER')

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
    
        data_tree = POST.extract_bc(output_tree, Family=family, BaseName=family)
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
        tool = 'maia' if output_tree.isUnstructured() else 'cassiopee',
        )
    
    return isosurface

def extract_residuals(output_tree, extraction):
    residuals = output_tree.base().get(Name='GlobalConvergenceHistory', Depth=2)
    if not residuals: return cgns.Tree()
    residuals = cgns.castNode(residuals)
    residuals.findAndRemoveNode(Name='.Solver#Output')
    t = cgns.Tree()
    base = cgns.Base(Name='Residuals', Parent=t)

    # base/zone/FlowSolution structure required for allowing conversion to tecplot fmt
    residuals.setType('FlowSolution_t')
    residuals.setName('FlowSolution')
    residuals.setValue(None)

    cgns.Zone(Name=base.name(), Parent=base, Children=[residuals])

    current_iteration_signals = mpi_allgather_and_merge_trees(t)

    if 'Data' in extraction and extraction['Data'] is not None:
        previous_signals_to_be_updated = extraction['Data']
        update_signals_using(current_iteration_signals, previous_signals_to_be_updated)
    else: 
        extraction['Data'] = current_iteration_signals


def extract_integral(output_tree, extraction) -> None:
    
    t = cgns.Tree()
    base = cgns.Base(Name='Integral', Parent=t)
    for IntegralDataNode in output_tree.group(Type='IntegralData', Depth=2):
        full_name_parts = IntegralDataNode.name().split('-')

        if len(full_name_parts) > 1 and full_name_parts[1].startswith('#'):
            IntegralName = full_name_parts[1][1:-1]

        else:
            IntegralName = full_name_parts[0]

        if IntegralName != extraction['Name']: continue

        IntegralDataNode.dettach()
        IntegralDataNode.setName('FlowSolution')
        IntegralDataNode.setType('FlowSolution_t')
        for n in IntegralDataNode.children(): 
            n.setType('DataArray_t')
        translate_elsa_CGNS_field_names_to_MOLA(IntegralDataNode)
        zone = cgns.Zone(Name=extraction['Name'], Parent=base, Children=[IntegralDataNode])
        break

    current_iteration_signals = mpi_allgather_and_merge_trees(t)

    if 'Data' in extraction and extraction['Data'] is not None:
        previous_signals_to_be_updated = extraction['Data']
        update_signals_using(current_iteration_signals, previous_signals_to_be_updated)
    else: 
        extraction['Data'] = current_iteration_signals


def extract_probe(output_tree):
    warnings.warning('skip extraction of type Probe (not implemented yet)')
    return cgns.Tree()

def update_elsa_input(new_tree):
    elsAxdt.xdt(elsAxdt.PYTHON,(elsAxdt.RUNTIME_TREE, new_tree, 1))

def end_simulation(workflow):
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
    return elsAxdt.iteration()

def get_status(workflow):
    return 'RUNNING_BEFORE_ITERATION' # TODO: implement this (using elsaXdt?)