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
from mola.cfd.preprocess.check import check

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_is_included_in_range():
    x = np.array([[1, 10], [1, 10]])
    y = np.array([[2, 7], [1, 10]])
    z = np.array([[2, 7], [1, 11]])

    assert check.is_included_in_range(y, x)
    assert not check.is_included_in_range(x, y)
    assert not check.is_included_in_range(z, x)
    assert not check.is_included_in_range(x, z)
    assert check.is_included_in_range(y, z)
    assert not check.is_included_in_range(z, y)

    a = np.array([[1,4], [1,1], [3,5]])
    b = np.array([[1,4], [5,5], [3,5]])
    c = np.array([[1,4], [1,1], [3, 1]])
    assert not check.is_included_in_range(a, b)
    assert not check.is_included_in_range(b, a)
    assert not check.is_included_in_range(a, c)
    assert not check.is_included_in_range(c, a)
    assert not check.is_included_in_range(b, c)
    assert not check.is_included_in_range(c, b)

