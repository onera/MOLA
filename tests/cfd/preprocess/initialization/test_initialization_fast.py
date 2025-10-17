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
from mola.cfd.preprocess.initialization import solver_fast

import pytest
pytestmark = pytest.mark.fast

class FakeWorkflow():
    def __init__(self):
        self.ProblemDimension = 3
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
    solver_fast.apply_to_solver(workflow)
