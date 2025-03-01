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
from mola.cfd.preprocess.motion import solver_fast
from mola.logging import MolaAssertionError

pytestmark = pytest.mark.fast


class FakeWorkflow():

    def __init__(self, Motion, TimeMarching):
        self.tree = cgns.Tree()
        base = cgns.Base(Parent=self.tree)
        cgns.Node(Name='Rotor', Type='Family', Parent=base)
        cgns.Node(Name='Stator', Type='Family', Parent=base)
        self.Motion = Motion
        self.Numerics = dict(TimeMarching=TimeMarching)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_apply_to_solver_steady_fix():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[0., 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[0., 0., 0.],
        )
    )
    workflow = FakeWorkflow(Motion, 'Steady')
    solver_fast.apply_to_solver(workflow)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_apply_to_solver_steady_mobile():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[500., 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[5., 0., 8.],
        )
    )
    workflow = FakeWorkflow(Motion, 'Steady')
    with pytest.raises(MolaAssertionError):
        solver_fast.apply_to_solver(workflow)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_apply_to_solver_unsteady():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[500., 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[5., 0., 8.],
        )
    )
    workflow = FakeWorkflow(Motion, 'Unsteady')
    solver_fast.apply_to_solver(workflow)
    