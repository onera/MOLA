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

import maia
from mpi4py import MPI
from treelab import cgns

from mola.logging import MolaException
# no relative imports possible for the following line because the current file is called by
# call_solver_specific_function in manager.py
from mola.cfd.coprocess import mola_logger, rank, comm


def perform_extractions(workflow, coprocess_manager):
    workflow.tree = cgns.castNode(workflow.tree)
    output_tree = workflow.tree

    for extraction in coprocess_manager.Extractions:
        if extraction['IsToExtract'] == False:
            continue

        mola_logger.debug(f'  update extraction of type {extraction["Type"]}', rank=0)
        
        if extraction['Type'] == 'Restart':
            coprocess_manager.iteration = workflow.Numerics['NumberOfIterations']
            update_restart_fields(workflow, output_tree)
            extraction['Data'] = workflow.tree
        
        # elif extraction['Type'] == '3D':
        #     extraction['Data'] = extract_fields(output_tree, extraction)

        # elif extraction['Type'] == 'BC':
        #     extraction['Data'] = extract_bc(output_tree, extraction, families_to_bctype)
        
        # elif extraction['Type'] == 'IsoSurface':
        #     extraction['Data'] = extract_isosurface(output_tree, extraction)

        # elif extraction['Type'] == 'Residuals':
        #     extraction['Data'] = extract_residuals(output_tree)

        else:
            mola_logger.warning(f"Type of extraction {extraction['Type']} is not available for elsA", rank=0)
            extraction['Data'] = cgns.Tree()

def update_restart_fields(workflow, output_tree):
    output_tree = cgns.castNode(output_tree)
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
    
