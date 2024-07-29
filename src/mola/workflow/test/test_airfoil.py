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

from mola.workflow import WorkflowAirfoil

@pytest.mark.unit
@pytest.mark.elsa # because workflow airfoil not compatible with sonics yet (not working without cassiopee)
@pytest.mark.cost_level_0
def test_get_flow_directions():
    AngleOfAttackDeg = 15
    AngleOfSlipDeg   = 2
    YawAxis   = [1, 0, 0] 
    PitchAxis = [0, -1, 0]  
    drag_dir, side_dir, lift_dir = WorkflowAirfoil.get_flow_directions(AngleOfAttackDeg, AngleOfSlipDeg, YawAxis, PitchAxis) 
    assert np.allclose(drag_dir, [ 0.25881905, -0.03371033,  0.96533741]) 
    assert np.allclose(side_dir, [ 0.        , -0.99939083, -0.0348995 ]) 
    assert np.allclose(lift_dir, [ 0.96592583,  0.00903265, -0.25866138]) 