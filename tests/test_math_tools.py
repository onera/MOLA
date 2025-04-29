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

import mola.math_tools as mt

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_rotate_3d_vector_from_axis_and_angle_in_degrees():
    i = np.array([1,0,0])
    j = np.array([0,1,0])
    k = np.array([0,0,1])
    quarter = 90
    assert np.allclose(mt.rotate_3d_vector_from_axis_and_angle_in_degrees(2*i, i, quarter), 2*i)
    assert np.allclose(mt.rotate_3d_vector_from_axis_and_angle_in_degrees(3*i, k, quarter), 3*j)  
    assert np.allclose(mt.rotate_3d_vector_from_axis_and_angle_in_degrees(j, -k, quarter), i)  
    assert np.allclose(mt.rotate_3d_vector_from_axis_and_angle_in_degrees(j, i, quarter), k) 
    assert np.allclose(mt.rotate_3d_vector_from_axis_and_angle_in_degrees(2*i, j, 2*quarter), -2*i)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_normalize_numpy_vector():
    vector = np.array([0.5,1,0.75])
    expected_normalized = np.array([0.37139068, 0.74278135, 0.55708601])

    mt.normalize_numpy_vector(vector)
    assert np.allclose(vector, expected_normalized)