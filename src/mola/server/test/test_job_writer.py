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
from mola.server import job_writer
from mola.server import server

onera_only = pytest.mark.skipif(server.get_network() != 'onera', reason="test on ONERA machines")

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
    assert job_writer.convert_to_seconds(input) == output

def test_time_margin():
    RunManagement = dict(SecondsMarginForQuitBeforeTimeOut=600)
    scheduler_options = dict(time='00:30:00')
    job_writer.set_time_margin(RunManagement, scheduler_options)
    assert RunManagement['TimeOutInSeconds'] == 1200.


@onera_only
def test_onera_get_job_text():
    RunManagement = dict(
        mola_target_path = __MOLA_PATH__,
        Machine = 'sator',
        JobName = 'mytest',
        AER = 'myAER',
    )
    job_text = job_writer.get_job_text(RunManagement, 'fake_solver')

    env = os.path.join(
        RunManagement['mola_target_path'],
        "mola",
        "env",
        "onera",
        RunManagement['Machine'],
        'fake_solver.sh')

    assert job_text == f'''#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --constraint=csl
#SBATCH --job-name=mytest
#SBATCH --output=output.%j.log
#SBATCH --error=error.%j.log
#SBATCH --comment=myAER

source {env}'''
