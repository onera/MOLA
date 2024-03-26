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

import copy
import numpy as np
from mola.cfd.preprocess.motion import motion


def test_set_default_motion():
    Motion = dict()
    motion.set_default_motion(Motion)
    assert Motion == dict(
        RotationSpeed      = [0., 0., 0.],
        RotationAxisOrigin = [0., 0., 0.],
        TranslationSpeed   = [0., 0., 0.],
    )

def test_set_default_motion2():
    Motion = dict(RotationSpeed=500)
    motion.set_default_motion(Motion)

    assert Motion == dict(
        RotationSpeed      = [500., 0., 0.],
        RotationAxisOrigin = [0., 0., 0.],
        TranslationSpeed   = [0., 0., 0.],
    )

def test_set_default_motion3():
    Motion = dict(
        RotationSpeed=np.empty(3),
        RotationAxisOrigin=np.empty(3),
        TranslationSpeed=np.empty(3),
    )
    Motion_Ref = copy.copy(Motion)
    motion.set_default_motion(Motion)

    assert Motion == Motion_Ref

def test_set_default_motion_function():
    Motion = lambda x: x
    Motion_Ref = copy.copy(Motion)
    motion.set_default_motion(Motion)

    assert Motion == Motion_Ref

def test_is_mobile1():
    Motion = dict()
    motion.set_default_motion(Motion)
    assert not motion.is_mobile(Motion)

def test_is_mobile2():
    Motion = lambda x: x
    motion.set_default_motion(Motion)
    assert motion.is_mobile(Motion)

def test_is_mobile3():
    Motion = dict(
        RotationSpeed      = [500., 0., 0.],
    )
    motion.set_default_motion(Motion)
    assert motion.is_mobile(Motion)

def test_is_mobile4():
    Motion = dict(
        TranslationSpeed   = [0., 1., 0.],
    )
    motion.set_default_motion(Motion)
    assert motion.is_mobile(Motion)

def test_is_mobile5():
    Motion = dict(
        RotationAxisOrigin = [1., 0., 0.],
    )
    motion.set_default_motion(Motion)
    assert not motion.is_mobile(Motion)