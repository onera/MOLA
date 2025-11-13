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
from mola.logging import MolaException, MolaUserError, mola_logger, CYAN, ENDC
from . import default, autogrid, utils, reader, writer, unstructured

from treelab import cgns

def apply(workflow):
    read(workflow)
    unstructured.apply(workflow)

def read(workflow):
    
    if len(workflow.RawMeshComponents) > 1:
        mola_logger.info("  📖 reading and 🧩assembling meshes", rank=0)
    else:
        mola_logger.info("  📖 reading mesh", rank=0)
    
    meshes = []
    for component in workflow.RawMeshComponents:
        
        if 'Mesher' not in component or component['Mesher'] == 'default':
            mola_logger.info(f'   - read component {component["Name"]} with {CYAN}default reader{ENDC}', rank=0)
            base = default.reader(workflow, component)

        elif component['Mesher'].lower() == 'autogrid':
            mola_logger.info(f'   - read component {component["Name"]} with {CYAN}Autogrid reader{ENDC}', rank=0)
            base = autogrid.reader(workflow, component)

        else:
            raise MolaException(f"unknown Mesher: {component['Mesher']}")
        
        meshes += [base]
    
    workflow.tree = cgns.add(meshes)

    set_problem_dimension_based_on_grid(workflow)
    enforce_name_of_FamilyBC(workflow.tree)


def set_problem_dimension_based_on_grid(workflow):

    cell_multilayer_count = 0
    cell_monolayer_count = 0
    surface_count = 0
    for zone in workflow.tree.zones():
        if is_cell_multilayer(zone):
            cell_multilayer_count += 1
        
        if is_cell_monolayer(zone):
            cell_monolayer_count += 1
        
        if is_surface(zone):
            surface_count += 1

    topo_counts = (cell_multilayer_count, cell_monolayer_count, surface_count)
    
    set_homogeneous_dimension(workflow, topo_counts)
    

def is_cell_multilayer( zone : cgns.Zone ):
    cell_shape = zone.value()[:,1]
    if zone.isUnstructured(): return True # TODO how to check if unstructured is 2D?
    if len(cell_shape) < 3: return False
    for s in cell_shape:
        if s < 2:
            return False
    return True

def is_cell_monolayer( zone : cgns.Zone ):
    cell_shape = zone.value()[:,1]
    if len(cell_shape) < 3: return False
    for s in cell_shape:
        if s == 1:
            return True
    return False
    
def is_surface( zone : cgns.Zone ):
    cell_shape = zone.value()[:,1]
    if len(cell_shape) == 2: return True
    for s in cell_shape:
        if s == 0:
            return True
    return False


def set_homogeneous_dimension(workflow, topo_counts):
    _, counts = np.unique(topo_counts, return_counts=True)
    if counts[0] != 2:
        raise ValueError(f"grid dimensions are not homogeneous, had: {topo_counts}")
    if topo_counts[0] > 0:
        workflow.ProblemDimension = 3
    elif topo_counts[1] > 0:
        workflow.ProblemDimension = 2
    elif topo_counts[2] > 0:
        workflow.ProblemDimension = 2
    else:
        raise RuntimeError('fatal condition')




def enforce_name_of_FamilyBC(tree: cgns.Tree):
    # Mainly for PointWise, that use the name 'FamBC' for 'FamilyBC_t' nodes, 
    # whereas the CGNS norm is using the name 'FamilyBC'
    # See http://cgns.github.io/CGNS_docs_current/sids/misc.html#Family (point #2 in Notes) 
    for node in tree.group(Type='FamilyBC'):
        node.setName('FamilyBC')
