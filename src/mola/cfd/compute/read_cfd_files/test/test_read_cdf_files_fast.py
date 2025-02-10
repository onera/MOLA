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

from mola.cfd.compute.read_cfd_files.solver_fast import _split_global_and_local_parameters

pytestmark = pytest.mark.fast

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_split_global_and_local_parameters():

    Num2Zones = dict(key1='value1', key2='value2')
    Num2Zones['Local@Family'] = dict(
        local_key='local_value1', 
        local_value2='local_value2'
    )

    global_parameters, local_parameters = _split_global_and_local_parameters(Num2Zones)
    assert global_parameters == dict(key1='value1', key2='value2')
    assert local_parameters == dict(
        Family = dict(
            local_key='local_value1', 
            local_value2='local_value2'
        )
    )
