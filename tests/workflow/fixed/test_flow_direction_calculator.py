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

import mola.workflow.fixed.flow_direction_calculator as fdc

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_from_two_angles_and_aircraft_yaw_pitch_axis():
    alpha = 10
    beta = 5
    yaw_axis = [0,0,1]
    pitch_axis = [0,1,0]

    expected_directions = np.array([
        [ 0.98106026, -0.08583165,  0.17364818],
        [-0.08715574, -0.9961947,   0.        ],
        [-0.17298739,  0.01513444,  0.98480775]])

    directions = fdc.from_two_angles_and_aircraft_yaw_pitch_axis(alpha,beta,yaw_axis,pitch_axis)

    assert np.allclose(np.vstack(directions), expected_directions)

if __name__ == '__main__':
    test_from_two_angles_and_aircraft_yaw_pitch_axis()