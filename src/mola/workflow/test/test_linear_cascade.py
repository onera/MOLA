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
from mola.workflow import WorkflowLinearCascade
from mola.logging import mola_logger, MolaAssertionError

def get_workflow():

    x, y, z = np.meshgrid( np.linspace(0,1,21),
                           np.linspace(0,1,21),
                           np.linspace(0,1,21), indexing='ij')
    mesh = cgns.newZoneFromArrays( 'block', ['x','y','z'], [ x,  y,  z ])

    w = WorkflowLinearCascade(

        RawMeshComponents=[
            dict(
                Name='cartesian',
                Source=mesh,
                Mesher='default',
                )
        ],

        Solver=os.environ.get('MOLA_SOLVER'),
        )
    return w

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_init():
    w = get_workflow()
    w.print_interface()
    assert w.Name == 'WorkflowLinearCascade'

@pytest.mark.unit
@pytest.mark.cost_level_3
def test_parametrize_with_height():
    w = get_workflow()
    w.assemble()
    try:
        w.parametrize_with_height('XY')
        assert w.tree.get(Name='FlowSolution#Height', Type='FlowSolution')
    except ImportError:
        mola_logger.warning('turbo module cannot be found!')
        pass

