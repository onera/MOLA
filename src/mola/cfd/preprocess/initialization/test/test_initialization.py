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
import numpy as np
from treelab import cgns
from mola.workflow.workflow import Workflow
from mola.cfd.preprocess.initialization import initialization


def get_debug_mesh():
    import Converter.PyTree as C
    import Generator.PyTree as G

    zone = G.cart((0.,0.,0.), (0.1,0.1,0.1), (3,2,2))
    mesh = C.newPyTree(['cart', zone])
    mesh = cgns.castNode(mesh)

    return mesh

def apply_all_previous_stages(workflow):
    workflow.assemble()
    workflow.positioning()
    workflow.connect()
    workflow.define_families()
    workflow.split_and_distribute() 
    workflow.process_overset()
    workflow.compute_flow_and_turbulence()
    workflow.set_motion()


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_initialization_uniform():
    mesh = get_debug_mesh()
    workflow = Workflow(
        RawMeshComponents = [dict(Name='cart', Source=mesh)],
        SplittingAndDistribution = 'PyPart',
        Flow = dict(Velocity=10),
        Turbulence = dict(Model='SA'),
    )
    apply_all_previous_stages(workflow)
    initialization.apply(workflow)

    ref_fs = ['FlowSolution#Init', None, [
        ['GridLocation', np.array([b'C', b'e', b'l', b'l', b'C', b'e', b'n', b't', b'e', b'r'],dtype='|S1'), [], 'GridLocation_t'], 
        ['Density', np.array([[[1.225]],[[1.225]]]), [], 'DataArray_t'], 
        ['MomentumX', np.array([[[12.25]],[[12.25]]]), [], 'DataArray_t'], 
        ['MomentumY', np.array([[[0.]],[[0.]]]), [], 'DataArray_t'], 
        ['MomentumZ', np.array([[[0.]],[[0.]]]), [], 'DataArray_t'], 
        ['EnergyStagnationDensity', np.array([[[253373.86097188]],[[253373.86097188]]]), [], 'DataArray_t'], 
        ['TurbulentSANuTildeDensity', np.array([[[4.41691234e-05]],[[4.41691234e-05]]]), [], 'DataArray_t'], 
    ], 'FlowSolution_t']

    assert str(workflow.tree.get(Name='FlowSolution#Init')) == str(ref_fs)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_initialization_copy_not_existing_file():
    mesh = get_debug_mesh()
    workflow = Workflow(
        RawMeshComponents = [dict(Name='cart', Source=mesh)],
        SplittingAndDistribution = 'PyPart',
        Flow = dict(Velocity=10),
        Turbulence = dict(Model='SA'),
        Initialization=dict(method='copy', source='not_existing_file.cgns'),
    )
    apply_all_previous_stages(workflow)
    
    try:
        initialization.apply(workflow)
    except FileNotFoundError:
        return
    else:
        raise AssertionError('Should raise an exception when the source file for initialization does not exist.')
    


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_initialization_copy():    
    ref_fs = cgns.Node(['FlowSolution#Init', None, [
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
        SplittingAndDistribution = 'PyPart',
        Flow = dict(Velocity=10),
        Turbulence = dict(Model='SA'),
        Initialization=dict(method='copy', source=source),
    )
    apply_all_previous_stages(workflow)
    initialization.apply(workflow)

    assert str(workflow.tree.get(Name='FlowSolution#Init')) == str(ref_fs)

