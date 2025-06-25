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
def test_get_rotation_parameter_1():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[10., 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[0., 0., 0.],
        )
    )

    fast_parameter = solver_fast.get_rotation_parameter(Motion)

    expected_parameter = [ 1.0, 0.0, 0.0, 3.0, 2.0, -1.0, 0.0, 0.0 ]
    
    assert np.allclose(fast_parameter, expected_parameter)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_rotation_parameter_2():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[0., 0., -5.],
            RotationAxisOrigin=[0., 0., 0.],
            TranslationSpeed=[0., 0., 0.],
        )
    )

    fast_parameter = solver_fast.get_rotation_parameter(Motion)

    expected_parameter = [ 0.0, 0.0,-1.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]
    
    assert np.allclose(fast_parameter, expected_parameter)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_rpm():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[-2*np.pi/60.0, 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[0., 0., 0.],
        ),
        AnotherRotor = dict(
            RotationSpeed=[10., 0., 0.],
            RotationAxisOrigin=[0., 0., 0.],
            TranslationSpeed=[0., 0., 0.],
        )
    )
    
    rpm = solver_fast.get_rpm(Motion)
    expected_rpm = 1
    
    assert np.allclose(rpm,expected_rpm)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_timestep_based_on_azimutal_step():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[-2*np.pi/60.0, 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[0., 0., 0.],
        ),
        AnotherRotor = dict(
            RotationSpeed=[10., 0., 0.],
            RotationAxisOrigin=[0., 0., 0.],
            TranslationSpeed=[0., 0., 0.],
        )
    )

    delta_psi = 6.0
    dt = solver_fast.get_timestep_based_on_azimutal_step(Motion, delta_psi)
    expected_dt = 1

    assert np.allclose(dt, expected_dt)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_must_be_unsteady_if_has_motion_when_steady_no_timestep():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[-2*np.pi/60.0, 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[0., 0., 0.],
        ),
        AnotherRotor = dict(
            RotationSpeed=[10., 0., 0.],
            RotationAxisOrigin=[0., 0., 0.],
            TranslationSpeed=[0., 0., 0.],
        )
    )
    Numerics = dict(TimeMarching='Steady')

    solver_fast._must_be_unsteady_if_has_motion(Motion, Numerics)
    dt = Numerics["TimeStep"]
    assert Numerics["TimeMarching"] == "Unsteady"

    excepted_dt = 1/12
    assert np.allclose(dt,excepted_dt)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_must_be_unsteady_if_has_motion_when_steady_with_timestep():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[-2*np.pi/60.0, 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[0., 0., 0.],
        ),
        AnotherRotor = dict(
            RotationSpeed=[10., 0., 0.],
            RotationAxisOrigin=[0., 0., 0.],
            TranslationSpeed=[0., 0., 0.],
        )
    )
    Numerics = dict(TimeMarching='Steady', TimeStep=1.0)

    solver_fast._must_be_unsteady_if_has_motion(Motion, Numerics)
    dt = Numerics["TimeStep"]
    assert Numerics["TimeMarching"] == "Unsteady"

    excepted_dt = 1
    assert np.allclose(dt,excepted_dt)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_check_unique_motion_when_it_is_not():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[-2*np.pi/60.0, 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[0., 0., 0.],
        ),
        AnotherRotor = dict(
            RotationSpeed=[10., 0., 0.],
            RotationAxisOrigin=[0., 0., 0.],
            TranslationSpeed=[0., 0., 0.],
        )
    )

    try:
        unique_motion = solver_fast.check_unique_motion(Motion)
    except AssertionError:
        return

    raise AssertionError("should have catched error, meaning motion is not unique")


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_check_unique_motion_when_it_is_true():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[-2*np.pi/60.0, 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[0., 0., 0.],
        ),
        AnotherRotor = dict(
            RotationSpeed=[-2*np.pi/60.0, 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[0., 0., 0.],
        )
    )

    unique_motion = solver_fast.check_unique_motion(Motion)
    assert unique_motion



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
    