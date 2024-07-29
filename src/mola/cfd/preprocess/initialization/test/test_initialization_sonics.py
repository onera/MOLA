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
from treelab import cgns
from mola.workflow.workflow import Workflow
from mola.cfd.preprocess.initialization import initialization, solver_sonics
from mola.cfd.preprocess.initialization.test.test_initialization import get_debug_mesh, apply_all_previous_stages

import pytest
pytestmark = pytest.mark.sonics

class FakeWorkflow():
    def __init__(self):
        # Build a base with two identical zones
        base = cgns.Node( Name='Base', Type='Base')
        z1 = cgns.Node( Name='Zone1', Type='Zone', Parent=base)
        z2 = cgns.Node( Name='Zone2', Type='Zone', Parent=base)
        for zone, shape in zip([z1, z2], [(1,2), (3,2)]):
            fs = cgns.Node( Name='FlowSolution#Init', Type='FlowSolution', Parent=zone )
            cgns.Node( Name='ChimeraCellType', Parent=fs )
            cgns.Node( Name='TurbulentDistance', Value=np.ones(shape, dtype=np.float64, order='F'), Parent=fs )
            cgns.Node( Name='OtherChild', Parent=fs )

        self.tree = base

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_apply_to_solver():

    workflow = FakeWorkflow()
    solver_sonics.apply_to_solver(workflow)

    assert workflow.tree.get(Name='FlowSolution#Init') is None
    assert len(workflow.tree.group(Name='FSolution#CellCenter#Init', Type='FlowSolution')) == 2

@pytest.mark.unit
@pytest.mark.cost_level_1
def test_initialization_uniform():
    mesh = get_debug_mesh()
    workflow = Workflow(
        RawMeshComponents = [dict(Name='cart', Source=mesh)],
        Flow = dict(Velocity=10.0),
        SplittingAndDistribution=dict(Strategy='AtComputation',Splitter='PyPart'),
        Turbulence = dict(Model='SA'),
    )
    apply_all_previous_stages(workflow)
    initialization.apply(workflow)

    FS = workflow.tree.get(Name='FSolution#CellCenter#Init', Type='FlowSolution')
    assert FS.get(Name='GridLocation', Type='GridLocation', Value='CellCenter') 
    assert np.allclose(FS.get(Name='Density', Type='DataArray').value(), 1.225)
    assert np.allclose(FS.get(Name='MomentumX', Type='DataArray').value(), 12.25)
    assert np.allclose(FS.get(Name='MomentumY', Type='DataArray').value(), 0.)
    assert np.allclose(FS.get(Name='MomentumZ', Type='DataArray').value(), 0.)
    assert np.allclose(FS.get(Name='EnergyStagnationDensity', Type='DataArray').value(), 253373.86097188)
    assert np.allclose(FS.get(Name='TurbulentSANuTildeDensity', Type='DataArray').value(), 4.41691234e-05)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_initialization_copy():    
    ref_fs = cgns.Node(['FSolution#CellCenter#Init', None, [
        ['GridLocation', 'CellCenter', [], 'GridLocation_t'], 
        ['Density', np.array([[[3.]],[[4.]]]), [], 'DataArray_t'], 
        ['MomentumX', np.array([[[50.]],[[-5.]]]), [], 'DataArray_t'], 
        ['MomentumY', np.array([[[0.1]],[[0.5]]]), [], 'DataArray_t'], 
        ['MomentumZ', np.array([[[0.]],[[1.]]]), [], 'DataArray_t'], 
        ['EnergyStagnationDensity', np.array([[[2e5]],[[3e5]]]), [], 'DataArray_t'], 
        ['TurbulentEnergyKineticDensity', np.array([[[0.2]],[[0.5]]]), [], 'DataArray_t'], 
        ['TurbulentDissipationRateDensity', np.array([[[125.]],[[12.]]]), [], 'DataArray_t']
    ], 'FlowSolution_t'])

    mesh = get_debug_mesh()
    source = mesh.copy(deep=True)
    zone = source.zones()[0]
    zone.addChild(ref_fs)    

    workflow = Workflow(
        RawMeshComponents = [dict(Name='cart', Source=mesh)],
        Flow = dict(Velocity=10.0),
        SplittingAndDistribution=dict(Strategy='AtComputation',Splitter='PyPart'),
        Turbulence = dict(Model='SA'),
        Initialization=dict(Method='copy', Source=source),
    )
    apply_all_previous_stages(workflow)
    initialization.apply(workflow)

    assert str(workflow.tree.get(Name='FSolution#CellCenter#Init')) == str(ref_fs)
