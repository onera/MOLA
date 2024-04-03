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
import os
from mola import __MOLA_PATH__
from mola.logging import check_error_message
from mola.cfd.preprocess.write_cfd_files import write_cfd_files
from mola import misc


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_default():
    Network = os.environ.get('MOLA_NETWORK')
    config_path = os.path.join(__MOLA_PATH__,'mola','env',Network,'config.py')
    config = misc.load_source('config', config_path)

    RunManagement = dict(
        Machine = 'auto',
        RunDirectory = '.',
        NumberOfProcessors = 3,
        )

    write_cfd_files.set_default(RunManagement)

    assert RunManagement['Machine'] in config.AvailableEnvironments
    # TODO Complete the assertion tests


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("NumberOfProcessors", [None, 10., 'number', [5, 6]])
def test_set_default_error_NumberOfProcessors(NumberOfProcessors):
    RunManagement = dict(NumberOfProcessors=NumberOfProcessors)

    expected_error_msg = f'The value {RunManagement["NumberOfProcessors"]} for NumberOfProcessors is not allowed. It must be an integer'
    check_error_message(expected_error_msg, write_cfd_files.set_default, RunManagement)
