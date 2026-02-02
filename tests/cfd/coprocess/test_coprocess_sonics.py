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
from treelab import cgns
from mola.cfd.coprocess import solver_sonics
import mola.naming_conventions as names
from mola.workflow import Workflow

pytestmark = pytest.mark.sonics


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_sum_convective_and_diffusive_loads():
           
    b = cgns.Node(Name='IntegralDataNode')

    for component in ['X','Y','Z']:
        cgns.Node(Name=f'convForce{component}' , Value=1*np.arange(0, 10.5, 0.5), Type='DataArray_t', Parent=b)
        cgns.Node(Name=f'diffForce{component}' , Value=2*np.arange(0, 10.5, 0.5), Type='DataArray_t', Parent=b)
        cgns.Node(Name=f'convTorque{component}', Value=3*np.arange(0, 10.5, 0.5), Type='DataArray_t', Parent=b)
        cgns.Node(Name=f'diffTorque{component}', Value=4*np.arange(0, 10.5, 0.5), Type='DataArray_t', Parent=b)

    solver_sonics.sum_convective_and_diffusive_loads(b)

    for component in ['X','Y','Z']:
        assert np.allclose(b.get(Name=f'Force{component}').value()  , 3*np.arange(0, 10.5, 0.5))
        assert np.allclose(b.get(Name=f'Torque{component}').value() , 7*np.arange(0, 10.5, 0.5))
    


if __name__ == '__main__':
    test_sum_convective_and_diffusive_loads()