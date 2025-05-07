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
import os
import numpy as np
from treelab import cgns
from mola.workflow.workflow import Workflow
from mola.cfd.preprocess.initialization import initialization
import mola.naming_conventions as names
from mola.logging import MolaException

@pytest.fixture
def debug_mesh(FlowSolution_ref):
    tree = cgns.Tree()
    base = cgns.Base(Name='cart', Parent=tree)
    # There must be at least 4 cells in mesh to be able to apply interpolation with maia with n_closest_pt=4
    shape = FlowSolution_ref.get(Name='Density').value().shape
    x, y, z = np.meshgrid( np.linspace(0,1,shape[0]+1),
                           np.linspace(0,1,shape[1]+1),
                           np.linspace(0,1,shape[2]+1), indexing='ij')
    zone = cgns.newZoneFromArrays( 'block', ['x','y','z'], [ x,  y,  z ])
    zone.addChild(FlowSolution_ref.copy(deep=True)) 
    base.addChild(zone)
    
    return tree

@pytest.fixture
def FlowSolution_ref():
    shape = (2,2,2)
    ref_fs = cgns.Node(Name='FlowSolution#Init', Type='FlowSolution_t')
    cgns.Node(Name='GridLocation', Value='CellCenter', Type='GridLocation_t', Parent=ref_fs)
    cgns.Node(Name='Density', Value=np.array(np.random.rand(*shape), order='F'), Type='DataArray_t', Parent=ref_fs)
    cgns.Node(Name='MomentumX', Value=np.array(np.random.rand(*shape), order='F'), Type='DataArray_t', Parent=ref_fs)
    cgns.Node(Name='MomentumY', Value=np.array(np.random.rand(*shape), order='F'), Type='DataArray_t', Parent=ref_fs)
    cgns.Node(Name='MomentumZ', Value=np.array(np.random.rand(*shape), order='F'), Type='DataArray_t', Parent=ref_fs)
    cgns.Node(Name='EnergyStagnationDensity', Value=np.array(np.random.rand(*shape), order='F'), Type='DataArray_t', Parent=ref_fs)
    cgns.Node(Name='TurbulentEnergyKineticDensity', Value=np.array(np.random.rand(*shape), order='F'), Type='DataArray_t', Parent=ref_fs)
    cgns.Node(Name='TurbulentDissipationRateDensity', Value=np.array(np.random.rand(*shape), order='F'), Type='DataArray_t', Parent=ref_fs)
    
    return ref_fs   


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
def test_initialization_copy_not_existing_file(debug_mesh):
    workflow = Workflow(
        RawMeshComponents = [dict(Name='cart', Source=debug_mesh)],
        Flow = dict(Velocity=10.0),
        SplittingAndDistribution=dict(Strategy='AtComputation',Splitter='PyPart'),
        Turbulence = dict(Model='SA'),
        Initialization=dict(Method='copy', Source='not_existing_file.cgns'),
    )
    apply_all_previous_stages(workflow)
    
    try:
        initialization.apply(workflow)
    except (FileNotFoundError, ValueError):
        # ValueError for maia.io.file_to_dist_tree if file does not exist
        return
    else:
        raise AssertionError('Should raise an exception when the source file for initialization does not exist.')
    
@pytest.mark.unit
@pytest.mark.cost_level_1
def test_compute_wall_distance_with_maia(debug_mesh):
    tree = initialization.compute_wall_distance_with_maia(debug_mesh)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_force_grid_location_as_first_sibling(debug_mesh):
    
    GridLocation = debug_mesh.get('GridLocation')
    FlowSolution = GridLocation.parent()
    GridLocation.dettach()
    GridLocation.attachTo(FlowSolution, position='last')

    initialization.force_grid_location_as_first_sibling(debug_mesh)

    FlowSolution = debug_mesh.get(Type='FlowSolution_t')
    assert FlowSolution.children()[0].name() == 'GridLocation'

def get_correct_FlowSolution_name_from_solver():
    solver = os.getenv('MOLA_SOLVER')
    if solver == 'sonics':
        fs_name = 'Fields@Cell@Init'
    elif solver == 'fast':
        fs_name = 'FlowSolution#Centers'
    elif solver == 'elsa':
        fs_name = 'FlowSolution#Init'
    else:
        raise Exception('unknown solver: {solver}. If needed, update this test function.')
    return fs_name

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_initialization_uniform(debug_mesh):

    fs_name = get_correct_FlowSolution_name_from_solver()

    workflow = Workflow(
        RawMeshComponents = [dict(Name='cart', Source=debug_mesh)],
        Flow = dict(Velocity=10.0),
        SplittingAndDistribution=dict(Strategy='AtComputation',Splitter='PyPart'),
        Turbulence = dict(Model='SA'),
    )
    apply_all_previous_stages(workflow)
    initialization.apply(workflow)

    FS = workflow.tree.get(Name=fs_name, Type='FlowSolution')
    assert FS.get(Name='GridLocation', Type='GridLocation', Value='CellCenter') 
    assert np.allclose(FS.get(Name='Density', Type='DataArray').value(), 1.225)
    assert np.allclose(FS.get(Name='MomentumX', Type='DataArray').value(), 12.25)
    assert np.allclose(FS.get(Name='MomentumY', Type='DataArray').value(), 0.)
    assert np.allclose(FS.get(Name='MomentumZ', Type='DataArray').value(), 0.)
    assert np.allclose(FS.get(Name='EnergyStagnationDensity', Type='DataArray').value(), 253373.86097188)
    assert np.allclose(FS.get(Name='TurbulentSANuTildeDensity', Type='DataArray').value(), 4.42211463e-05)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_initialization_copy(debug_mesh, FlowSolution_ref):    

    fs_name = get_correct_FlowSolution_name_from_solver()

    target = debug_mesh.copy(deep=True)
    target.findAndRemoveNodes(Type='FlowSolution')

    workflow = Workflow(
        RawMeshComponents = [dict(Name='cart', Source=target)],
        Flow = dict(Velocity=10.0),
        SplittingAndDistribution=dict(Strategy='AtComputation',Splitter='PyPart'),
        Turbulence = dict(Model='SST'),
        Initialization=dict(Method='copy', Source=debug_mesh),
    )
    apply_all_previous_stages(workflow)
    initialization.apply(workflow)

    fs = workflow.tree.get(Name=fs_name)
    FlowSolution_ref.setName(fs_name)

    import maia.pytree as PT
    assert PT.is_same_node(fs, FlowSolution_ref)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_initialization_copy_missing_variable(debug_mesh):  

    # remove one of the field
    debug_mesh.findAndRemoveNode(Name='Density')

    target = debug_mesh.copy(deep=True)
    target.findAndRemoveNodes(Type='FlowSolution')

    workflow = Workflow(
        RawMeshComponents = [dict(Name='cart', Source=target)],
        Flow = dict(Velocity=10.0),
        SplittingAndDistribution=dict(Strategy='AtComputation',Splitter='PyPart'),
        Turbulence = dict(Model='SST'),
        Initialization=dict(Method='copy', Source=debug_mesh),
    )  

    apply_all_previous_stages(workflow)
    with pytest.raises(MolaException):
        initialization.apply(workflow)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_initialization_interpolate(debug_mesh, FlowSolution_ref):    

    fs_name = get_correct_FlowSolution_name_from_solver()

    target = debug_mesh.copy(deep=True)
    target.findAndRemoveNodes(Type='FlowSolution')

    workflow = Workflow(
        RawMeshComponents = [dict(Name='cart', Source=target)],
        Flow = dict(Velocity=10.0),
        SplittingAndDistribution=dict(Strategy='AtComputation',Splitter='PyPart'),
        Turbulence = dict(Model='SST'),
        Initialization=dict(Method='interpolate', Source=debug_mesh),
    )
    apply_all_previous_stages(workflow)
    initialization.apply(workflow)

    fs = workflow.tree.get(Name=fs_name)
    FlowSolution_ref.setName(fs_name)

    import maia.pytree as PT
    assert PT.is_same_node(fs, FlowSolution_ref)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_initialization_interpolate_missing_variable(debug_mesh):    

    # remove one of the field
    debug_mesh.findAndRemoveNode(Name='Density')

    target = debug_mesh.copy(deep=True)
    target.findAndRemoveNodes(Type='FlowSolution')

    workflow = Workflow(
        RawMeshComponents = [dict(Name='cart', Source=target)],
        Flow = dict(Velocity=10.0),
        SplittingAndDistribution=dict(Strategy='AtComputation',Splitter='PyPart'),
        Turbulence = dict(Model='SST'),
        Initialization=dict(Method='interpolate', Source=debug_mesh),
    )  

    apply_all_previous_stages(workflow)
    with pytest.raises(MolaException):
        initialization.apply(workflow)
