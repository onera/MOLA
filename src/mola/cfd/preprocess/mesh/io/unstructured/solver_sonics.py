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

from treelab import cgns
from mola.logging import mola_logger
from mola.cfd.preprocess.mesh.io.unstructured import convert_elements_to_ngon, merge_all_unstructured_zones_from_families
from mola.cfd.preprocess.mesh.tools import to_distributed, to_full_tree_at_rank_0

def apply_to_solver(workflow):
    if not workflow.tree.isUnstructured():
        mola_logger.user_warning('Make mesh fully unstructured for SoNICS')
        workflow.tree = make_mesh_unstructured(workflow.tree)
        
    any_not_ngon = any([elt_type not in ['NGON_n', 'NFACE_n'] for elt_type in workflow.tree.getElementsTypes()])
    tree_was_full = not(bool(workflow.tree.get(':CGNS#Distribution')) or bool(workflow.tree.get(':CGNS#GlobalNumbering')))

    if any_not_ngon:
        workflow.tree = convert_elements_to_ngon(workflow.tree)   

    workflow.tree = merge_all_unstructured_zones_from_families(workflow.tree)

    if any_not_ngon and tree_was_full:
        workflow.tree = to_full_tree_at_rank_0(workflow.tree)

def make_mesh_unstructured(t):
    import maia 
    from mpi4py import MPI
    from maia.io.fix_tree import fix_point_ranges
    
    t = to_distributed(t)
    fix_point_ranges(t)
    maia.algo.dist.convert_s_to_ngon(t, MPI.COMM_WORLD)
    maia.algo.pe_to_nface(t, MPI.COMM_WORLD)
    t = cgns.castNode(t)
    assert t.isUnstructured()

    return t
