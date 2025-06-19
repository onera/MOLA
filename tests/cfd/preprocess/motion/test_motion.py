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
import copy
import numpy as np
from mola.cfd.preprocess.motion import motion
from treelab import cgns

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_any_mobile_whith_no_family():
    Motion = dict()
    assert not motion.any_mobile(Motion)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_any_mobile_with_single_family_not_mobile():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[0., 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[0., 0., 0.],
        )
    )
    assert not motion.any_mobile(Motion)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_any_mobile_with_multiple_families():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[0., 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[0., 0., 0.],
        ),
        AnotherRotor = dict(
            RotationSpeed=[10., 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[0., 0., 0.],
        )
    )
    assert motion.any_mobile(Motion)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_first_found_rotation_axis_origin_vector_at_motion_1():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[-5., 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[0., 0., 0.],
        ),
        AnotherRotor = dict(
            RotationSpeed=[10., 0., 0.],
            RotationAxisOrigin=[0., 0., 0.],
            TranslationSpeed=[0., 0., 0.],
        )
    )
    rotation_axis_origin = motion.get_first_found_rotation_axis_origin_vector_at_motion(Motion)
    expected_rotation_axis_origin = [3,2,-1]
    assert np.allclose(rotation_axis_origin, expected_rotation_axis_origin)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_first_found_rotation_axis_origin_vector_at_motion_2():
    Motion = dict(
        AnotherRotor = dict(
            RotationSpeed=[10., 0., 0.],
            RotationAxisOrigin=[0., 0., 0.],
            TranslationSpeed=[0., 0., 0.],
        )
    )
    rotation_axis_origin = motion.get_first_found_rotation_axis_origin_vector_at_motion(Motion)
    expected_rotation_axis_origin = [0,0,0]
    assert np.allclose(rotation_axis_origin, expected_rotation_axis_origin)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_first_found_rotation_speed_vector_at_motion_1():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[-5., 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[0., 0., 0.],
        ),
        AnotherRotor = dict(
            RotationSpeed=[10., 0., 0.],
            RotationAxisOrigin=[0., 0., 0.],
            TranslationSpeed=[0., 0., 0.],
        )
    )
    rotation_speed = motion.get_first_found_rotation_speed_vector_at_motion(Motion)
    expected_rotation_speed = [-5,0,0]
    assert np.allclose(rotation_speed, expected_rotation_speed)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_first_found_rotation_speed_vector_at_motion_2():
    Motion = dict(
        AnotherRotor = dict(
            RotationSpeed=[10., 0., 0.],
            RotationAxisOrigin=[0., 0., 0.],
            TranslationSpeed=[0., 0., 0.],
        )
    )
    rotation_speed = motion.get_first_found_rotation_speed_vector_at_motion(Motion)
    expected_rotation_speed = [10,0,0]
    assert np.allclose(rotation_speed, expected_rotation_speed)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_rotation_axis_from_first_found_motion_1():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[-5., 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[0., 0., 0.],
        ),
        AnotherRotor = dict(
            RotationSpeed=[10., 0., 0.],
            RotationAxisOrigin=[0., 0., 0.],
            TranslationSpeed=[0., 0., 0.],
        )
    )
    rotation_axis = motion.get_rotation_axis_from_first_found_motion(Motion) 
    expected_rotation_axis = [-1,0,0]
    
    assert np.allclose(rotation_axis, expected_rotation_axis)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_rotation_axis_from_first_found_motion_2():
    Motion = dict(
        AnotherRotor = dict(
            RotationSpeed=[0., 0., 50.],
            RotationAxisOrigin=[0., 0., 0.],
            TranslationSpeed=[0., 0., 0.],
        )
    )
    rotation_axis = motion.get_rotation_axis_from_first_found_motion(Motion) 
    expected_rotation_axis = [0,0,1]
    
    assert np.allclose(rotation_axis, expected_rotation_axis)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_update_motion_with_defaults():
    Motion = dict()
    motion.update_motion_with_defaults(Motion)
    assert Motion == dict(
        RotationSpeed      = [0., 0., 0.],
        RotationAxisOrigin = [0., 0., 0.],
        TranslationSpeed   = [0., 0., 0.],
    )


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_update_motion_with_defaults2():
    Motion = dict(RotationSpeed=500)
    motion.update_motion_with_defaults(Motion)

    assert Motion == dict(
        RotationSpeed      = [500., 0., 0.],
        RotationAxisOrigin = [0., 0., 0.],
        TranslationSpeed   = [0., 0., 0.],
    )


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_update_motion_with_defaults3():
    Motion = dict(
        RotationSpeed=np.empty(3),
        RotationAxisOrigin=np.empty(3),
        TranslationSpeed=np.empty(3),
    )
    Motion_Ref = copy.copy(Motion)
    motion.update_motion_with_defaults(Motion)

    assert Motion == Motion_Ref


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_is_mobile1():
    Motion = dict()
    motion.update_motion_with_defaults(Motion)
    assert not motion.is_mobile(Motion)



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_is_mobile3():
    Motion = dict(
        RotationSpeed      = [500., 0., 0.],
    )
    motion.update_motion_with_defaults(Motion)
    assert motion.is_mobile(Motion)



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_is_mobile4():
    Motion = dict(
        TranslationSpeed   = [0., 1., 0.],
    )
    motion.update_motion_with_defaults(Motion)
    assert motion.is_mobile(Motion)



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_is_mobile5():
    Motion = dict(
        RotationAxisOrigin = [1., 0., 0.],
    )
    motion.update_motion_with_defaults(Motion)
    assert not motion.is_mobile(Motion)

class FakeWorkflow():
    def __init__(self):
        self.Motion = None
        self.tree = cgns.Tree()
        base = cgns.Base(Parent=self.tree)
        zone = cgns.Zone(Parent=base)
        udd = cgns.Node(Name='UDD', Type='UserDefinedData', Parent=zone)
        cgns.Node(Name='FamilyName', Type='FamilyName', Value='ThisIsATrap', Parent=udd)
        cgns.Node(Name='FamilyName', Type='FamilyName', Value='Fam1', Parent=zone)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_default_motion_on_families():
    workflow = FakeWorkflow()
    motion.set_default_motion_on_families(workflow)
    assert workflow.Motion == dict(
        Fam1 = dict(
            RotationSpeed = [0.0, 0.0, 0.0], 
            RotationAxisOrigin = [0.0, 0.0, 0.0], 
            TranslationSpeed = [0.0, 0.0, 0.0]
        )
    )
