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

from mola.cfd.preprocess.boundary_conditions import boundary_conditions as BC
from treelab import cgns

        
@pytest.mark.unit
@pytest.mark.cost_level_0
def test_Wall():
    class FakeWorkflow():
        def __init__(self):
            self.BoundaryConditions = [dict(Family='WING', type='BCWall')]

    workflow = FakeWorkflow()
    bc = workflow.BoundaryConditions[0]
    args, kwargs = BC.WallViscous(workflow, bc)

    assert args == [bc['Family']]
    assert kwargs['Motion'] == dict(
        RotationSpeed      = [0., 0., 0.],
        RotationAxisOrigin = [0., 0., 0.],
        TranslationSpeed   = [0., 0., 0.],
    )

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_Farfield():
    class FakeWorkflow():
        def __init__(self):
            self.BoundaryConditions = [dict(Family='UPSTREAM', type='BCFarfield')]

    workflow = FakeWorkflow()
    bc = workflow.BoundaryConditions[0]
    args, kwargs = BC.Farfield(workflow, bc)

    assert args == [bc['Family']]
    assert kwargs == dict()



