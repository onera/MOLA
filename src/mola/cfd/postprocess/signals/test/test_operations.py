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

from mola.cfd.postprocess import signals

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_slidding_average():
    array = np.arange(10)**2
    window = 5
    avg = signals.slidding_average(array, window)
    assert np.allclose(avg, np.array([1, 2, 6, 11, 18, 27, 38, 51, 62, 67]))

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_slidding_std():
    array = np.arange(10)**2
    window = 5
    std = signals.slidding_std(array, window)
    assert np.allclose(std, np.array([1.41421356,  3.87298335,  5.83095189,  
                                      8.60232527, 11.40175425, 14.2126704, 
                                      17.02938637, 19.84943324, 18.41195264, 
                                      15.93737745]))

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_slidding_rsd():
    array = np.arange(10)**2
    window = 5
    rsd = signals.slidding_rsd(array, window)
    assert np.allclose(rsd, np.array([1.41421356, 1.93649167, 0.97182532, 
                                      0.78202957, 0.63343079, 0.5263952,
                                      0.44814175, 0.38920457, 0.29696698, 
                                      0.23787131]))
