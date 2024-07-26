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
@pytest.mark.cost_level_1
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
    
