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
from mola.cfd.preprocess.mesh.split import _assert_tree_has_good_distribution_assignment

def apply_to_solver(workflow):

    add_elsaHybrid_nodes_if_needed(workflow.tree)  # in elsA v5.3.01, it seems to be still mandatory for some hybrid meshes
    if hasattr(workflow, '_FULL_CGNS_MODE'):
        add_elsa_keys_to_cgns(workflow)
    if workflow.SplittingAndDistribution['Strategy'] == 'AtPreprocess':
        _assert_tree_has_good_distribution_assignment(workflow)

def add_elsaHybrid_nodes_if_needed(t):
    if not t.isStructured():
        import Converter.Internal as I
        I._createElsaHybrid(t, method=1)
        t = cgns.castNode(t)
    return t

def add_elsa_keys_to_cgns(workflow):
    '''
    Include node ``.Solver#Compute`` , where elsA keys are set in full CGNS mode.
    '''
    workflow.tree.findAndRemoveNodes(Name='.Solver#Compute', Depth=2)

    # Put all solver keys in a unique and flat dictionary
    AllElsAKeys = dict()
    for keySet in workflow.SolverParameters.values():
        AllElsAKeys.update(keySet)
      
    for base in workflow.tree.bases(): 
        base.setParameters('.Solver#Compute', **AllElsAKeys)