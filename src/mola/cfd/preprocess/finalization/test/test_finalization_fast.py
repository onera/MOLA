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

import pytest
pytestmark = pytest.mark.fast

import numpy as np
from treelab import cgns
from mola.logging import MolaException


from mola.cfd.preprocess.finalization import solver_fast

class FakeWorkflowMonoBlock():

    def __init__(self, NPts):
        self.tree = cgns.Tree()
        base = cgns.Base(Parent=self.tree)
        xyz = np.meshgrid( np.linspace(0,1,NPts),
                           np.linspace(0,1,NPts),
                           np.linspace(0,1,NPts), indexing='ij')
        mesh = cgns.newZoneFromArrays( 'block', ['x','y','z'], xyz)
        mesh.attachTo(base)
        
        self.ProblemDimension = 3


class FakeWorkflowTwoBlocks():

    def __init__(self, NPts):
        self.tree = cgns.Tree()
        self.ProblemDimension = 3
        base = cgns.Base(Parent=self.tree)
        xyz = np.meshgrid( np.linspace(0,1,NPts),
                           np.linspace(0,1,NPts),
                           np.linspace(0,1,NPts), indexing='ij')
        block1 = cgns.newZoneFromArrays( 'block.1', ['x','y','z'], xyz)
        block1.attachTo(base)

        xyz = np.meshgrid( np.linspace(0,1,NPts)+1,
                           np.linspace(0,1,NPts),
                           np.linspace(0,1,NPts), indexing='ij')
        block2 = cgns.newZoneFromArrays( 'block.2', ['x','y','z'], xyz)
        block2.attachTo(base)



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_add_ghost_cells():
    NbOfGhostCellsPerDirection = 2
    NbOfPts = 5
    workflow = FakeWorkflowMonoBlock(NbOfPts)
    solver_fast.add_ghost_cells(workflow)
    assert workflow.tree.numberOfPoints() == (NbOfPts+NbOfGhostCellsPerDirection*2)**3


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_create_cell_center_tree():
    NbOfPts = 5
    workflow = FakeWorkflowMonoBlock(NbOfPts)
    solver_fast.create_cell_center_tree(workflow)
    assert workflow._treeCellCenter.numberOfPoints() == (NbOfPts-1)**3


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_multibloc_transfer_data():
    
    import Connector.PyTree as X
    import Converter.PyTree as C
    import Converter.Internal as I

    NbOfPts = 5
    workflow = FakeWorkflowTwoBlocks(NbOfPts)
    t = workflow.tree
    
    t = X.connectMatch(t, tol=1e-8, dim=3)
    
    I._addGhostCells(t,t,2,adaptBCs=1,fillCorner=0)
    
    tc = C.node2Center(t)

    workflow.tree = cgns.castNode(t)
    workflow._treeCellCenter = cgns.castNode(tc)

    solver_fast.set_multibloc_transfer_data(workflow)

    tc = workflow._treeCellCenter

    block1 = tc.get(Name='block.1', Type='Zone_t', Depth=2)
    assert block1

    block2 = tc.get(Name='block.2', Type='Zone_t', Depth=2)
    assert block2

    ID_block2 = block1.get(Name='ID_block.2', Type='ZoneSubRegion_t', Depth=1)
    assert ID_block2
    assert ID_block2.value() == 'block.2'

    ID_block1 = block2.get(Name='ID_block.1', Type='ZoneSubRegion_t', Depth=1)
    assert ID_block1
    assert ID_block1.value() == 'block.1'


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_add_reynolds_to_reference_state():
    
    workflow = FakeWorkflowMonoBlock(5)
    
    base = workflow.tree.bases()[0]
    
    ref_state = cgns.Node(Name='ReferenceState', Type='ReferenceState_t', Parent=base)
    
    workflow.Flow = dict(Density=1.0,
                         VelocityForScalingAndTurbulence=1.0,
                         ViscosityMolecular=1.0)
    
    solver_fast.add_reynolds_to_reference_state(workflow)

    Re = ref_state.get('Reynolds')
    
    assert Re
    
    assert Re.value() == 1.0


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_add_Rok_to_reference_state():
    
    workflow = FakeWorkflowMonoBlock(5)
    
    base = workflow.tree.bases()[0]

    ref_state = cgns.Node(Name='ReferenceState', Type='ReferenceState_t', Parent=base)
    
    workflow.Turbulence = dict(TurbulentEnergyKineticDensity=1.0)
    
    solver_fast.add_Rok_to_reference_state(workflow)

    Rok = ref_state.get('Rok')

    assert Rok

    assert Rok.value() == 1.0


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_add_RoOmega_to_reference_state():
    
    workflow = FakeWorkflowMonoBlock(5)
    
    base = workflow.tree.bases()[0]

    ref_state = cgns.Node(Name='ReferenceState', Type='ReferenceState_t', Parent=base)
    
    workflow.Turbulence = dict(TurbulentDissipationRateDensity=1.0)
    
    solver_fast.add_RoOmega_to_reference_state(workflow)

    RoOmega = ref_state.get('RoOmega')

    assert RoOmega

    assert RoOmega.value() == 1.0
