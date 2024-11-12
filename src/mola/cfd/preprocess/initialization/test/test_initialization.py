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
import mola.naming_conventions as names

def get_debug_mesh():
    base = cgns.Base(Name='cart')
    x, y, z = np.meshgrid( np.linspace(0,1,3),
                           np.linspace(0,1,2),
                           np.linspace(0,1,2), indexing='ij')
    zone = cgns.newZoneFromArrays( 'block', ['x','y','z'], [ x,  y,  z ])
    base.addChild(zone)
    return base

def make_tree():
    tree = cgns.Tree()
    tree.addChild( get_debug_mesh())

    tree.useEquation("{field1}=1.0", Container=names.CONTAINER_INITIAL_FIELDS)
    tree.useEquation("{field2}=2.0", Container=names.CONTAINER_INITIAL_FIELDS)
    
    return tree

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
def test_initialization_copy_not_existing_file():
    mesh = get_debug_mesh()
    workflow = Workflow(
        RawMeshComponents = [dict(Name='cart', Source=mesh)],
        Flow = dict(Velocity=10.0),
        SplittingAndDistribution=dict(Strategy='AtComputation',Splitter='PyPart'),
        Turbulence = dict(Model='SA'),
        Initialization=dict(Method='copy', Source='not_existing_file.cgns'),
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
def test_compute_turbulent_distance_with_maia():
    tree = make_tree()
    tree = initialization.compute_turbulent_distance_with_maia(tree)
    test_compute_turbulent_distance_with_maia

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_force_grid_location_as_first_sibling():
    tree = make_tree()
    
    GridLocation = tree.get('GridLocation')
    FlowSolution = GridLocation.parent()
    GridLocation.dettach()
    GridLocation.attachTo(FlowSolution, position='last')

    initialization.force_grid_location_as_first_sibling(tree)

    FlowSolution = tree.get(Type='FlowSolution_t')
    assert FlowSolution.children()[0].name() == 'GridLocation'

if __name__ == '__main__':
    test_force_grid_location_as_first_sibling()