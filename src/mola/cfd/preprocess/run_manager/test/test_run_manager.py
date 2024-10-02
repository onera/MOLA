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
from mola import __MOLA_PATH__
from mola.cfd.preprocess.run_manager import run_manager
from mola.server import remote

@pytest.mark.network_onera
@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_default():
    config = remote.get_network_config()

    RunManagement = dict(
        Machine = 'auto',
        RunDirectory = '.',
        NumberOfProcessors = 3,
        QuitMarginBeforeTimeOutInSeconds = 300,
        )

    run_manager.set_default(RunManagement)

    assert RunManagement['Machine'] in config.AvailableEnvironments
    # TODO Complete the assertion tests

@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize('in_out', [
    (50, 50),
    ('50', 50),
    ('10:03', 603),
    ('10:10:03', 36603),
    ('1-10:10:03', 3600*24 + 36603),
    ('1-10:10', 3600*24 + 36600),
    ('1-10', 3600*24 + 36000),
    ])
def test_convert_to_seconds_ss_int(in_out):
    input, output = in_out[0], in_out[1]
    assert run_manager.convert_to_seconds(input) == output


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_time_margin():
    RunManagement = dict(QuitMarginBeforeTimeOutInSeconds=600)
    scheduler_options = dict(time='00:30:00')
    run_manager.set_time_margin(RunManagement, scheduler_options)
    assert RunManagement['TimeOutInSeconds'] == 1200.

