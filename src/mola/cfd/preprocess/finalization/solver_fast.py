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

from mola.logging import mola_logger, MolaException, MolaUserError

import Connector.PyTree as X
import Converter.PyTree as C
import Converter.Internal as I

def apply_to_solver(workflow):

    check_consistency(workflow) # TODO put in a more relevant place, such as at assembly
    add_reynolds_to_reference_state(workflow)
    add_Rok_to_reference_state(workflow) # CAVEAT, not even CGNS standard name
    add_RoOmega_to_reference_state(workflow) # CAVEAT, not even CGNS standard name
    add_ghost_cells(workflow)
    create_cell_center_tree(workflow)
    set_multibloc_transfer_data(workflow)


def add_ghost_cells(workflow):
    t = workflow.tree
    I._addGhostCells(t,t,2,adaptBCs=1,fillCorner=0)
    workflow.tree = t

def create_cell_center_tree(workflow):
    tc = C.node2Center(workflow.tree)
    # C._rmVars(tc, 'FlowSolution')
    # I._rmNodesFromName(tc,'GridCoordinates')
    workflow._treeCellCenter = tc

def set_multibloc_transfer_data(workflow):
    t = workflow.tree
    tc= workflow._treeCellCenter
    tc = X.setInterpData(t, tc, nature=1, loc='centers', storage='inverse', 
                        sameName=1, dim=workflow.ProblemDimension)
    workflow._treeCellCenter = cgns.castNode(tc) 
    workflow.tree = cgns.castNode(t)

def add_reynolds_to_reference_state(workflow):
    ρ = workflow.Flow['Density']
    U = workflow.Flow['VelocityForScalingAndTurbulence']
    μ = workflow.Flow['ViscosityMolecular']

    try:
        L = workflow.ApplicationContext['Length']
    except:
        L = 1.0
        mola_logger.warning(f"Undefined Length in application context. Using Length={L} for computing Reynolds.")

    Reynolds = ρ*U*L/μ

    ReynoldsNode = cgns.Node(Name="Reynolds",Type="DataArray_t",Value=Reynolds)
    for ref_state in workflow.tree.group(Type='ReferenceState_t',Depth=2):
        ref_state.addChild(ReynoldsNode.copy())

def add_Rok_to_reference_state(workflow):
    Rok = workflow.Turbulence['TurbulentEnergyKineticDensity']

    RokNode = cgns.Node(Name="Rok",Type="DataArray_t",Value=Rok)
    for ref_state in workflow.tree.group(Type='ReferenceState_t',Depth=2):
        ref_state.addChild(RokNode.copy())

def add_RoOmega_to_reference_state(workflow):
    RoOmega = workflow.Turbulence['TurbulentDissipationRateDensity']

    RoOmegaNode = cgns.Node(Name="RoOmega",Type="DataArray_t",Value=RoOmega)
    for ref_state in workflow.tree.group(Type='ReferenceState_t',Depth=2):
        ref_state.addChild(RoOmegaNode.copy())

def check_consistency(workflow):
    if not workflow.tree.isStructured():
        raise MolaUserError('mesh must be structured')