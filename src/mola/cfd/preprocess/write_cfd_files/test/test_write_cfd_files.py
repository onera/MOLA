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
import copy
from mola import __MOLA_PATH__
from mola.logging import check_error_message
from mola.cfd.preprocess.write_cfd_files import write_cfd_files


def test_convert_to_seconds_ss_int():
    assert write_cfd_files.convert_to_seconds(50) == 50

def test_convert_to_seconds_ss():
    assert write_cfd_files.convert_to_seconds('50') == 50

def test_convert_to_seconds_mm_ss():
    assert write_cfd_files.convert_to_seconds('10:03') == 603

def test_convert_to_seconds_hh_mm_ss():
    assert write_cfd_files.convert_to_seconds('10:10:03') == 36603

def test_convert_to_seconds_j_hh_mm_ss():
    assert write_cfd_files.convert_to_seconds('1-10:10:03') == 3600*24 + 36603

def test_convert_to_seconds_j_hh_mm():
    assert write_cfd_files.convert_to_seconds('1-10:10') == 3600*24 + 36600

def test_convert_to_seconds_j_hh():
    assert write_cfd_files.convert_to_seconds('1-10') == 3600*24 + 36000

RunManagement_default = dict(
        JobName='mola',
        RunDirectory='.',
        NumberOfProcessors=None,
        SubmitJob=False,
        LauncherCommand = 'auto', # or 'sbatch job.sh', './job.sh'...
        mola_target_path = __MOLA_PATH__,
        FilesAndDirectories=[],
        )

@pytest.mark.parametrize("NumberOfProcessors", [None, 10., 'number', [5, 6]])
def test_set_default_error_NumberOfProcessors(NumberOfProcessors):
    RunManagement = dict(NumberOfProcessors=NumberOfProcessors)

    expected_error_msg = f'The value {RunManagement["NumberOfProcessors"]} for NumberOfProcessors is not allowed. It must be an integer'
    check_error_message(expected_error_msg, write_cfd_files.set_default, RunManagement)

def test_set_default_default_spiro():
    RunManagement = dict(
        NumberOfProcessors = 5,
        Network = 'onera',
        Machine = 'spiro', 
    )
    write_cfd_files.set_default(RunManagement)
    
    from pprint import pprint
    pprint(RunManagement)

    RunManagement_default_with_context = copy.copy(RunManagement_default)
    RunManagement_default_with_context.update(
        dict(
            NumberOfProcessors = 5,
            Network = 'onera',
            Machine = 'spiro', 
            JobScheduler = 'SLURM',
            TimeLimit = '24:00:00',
            TimeOutInSeconds = 24*3600-180,
            SlurmConstraint = None,
            SlurmQualityOfService = 'c1_test_giga',
        )
    )

    for key, value in RunManagement.items():
        if key == 'JobSchedulerOptions':
            continue
        assert value == RunManagement_default_with_context[key]


def test_set_default_default_sator():
    RunManagement = dict(
        NumberOfProcessors = 5,
        Network = 'onera',
        Machine = 'sator',
        AER = 'FakeAER', 
    )
    write_cfd_files.set_default(RunManagement)
    
    from pprint import pprint
    pprint(RunManagement)

    RunManagement_default_with_context = copy.copy(RunManagement_default)
    RunManagement_default_with_context.update(
        dict(
            NumberOfProcessors = 5,
            Network = 'onera',
            Machine = 'sator', 
            JobScheduler = 'SLURM',
            TimeLimit = '15:00:00',
            SlurmConstraint = 'csl',
            TimeOutInSeconds = 15*3600-180,
            AER = 'FakeAER',
        )
    )

    for key, value in RunManagement.items():
        if key == 'JobSchedulerOptions':
            continue
        assert value == RunManagement_default_with_context[key]


def test_set_default_custom_sator():
    RunManagement = dict(
        JobName='customName',
        RunDirectory='/my_path/',
        NumberOfProcessors=5,
        SubmitJob=True,
        Network = 'onera',
        Machine = 'sator', 
        SlurmConstraint = 'csl | skl',
        TimeLimit = '0-10:00',
        SecondsMarginForQuitBeforeTimeOut = 10,
        LauncherCommand = 'auto',
        mola_target_path = '/mola/installation/custom',
        FilesAndDirectories=['file_to_copy', 'path/filename'],
        AER='000X111A',
    )
    RunManagement_ref = copy.copy(RunManagement)
    RunManagement_ref['TimeOutInSeconds'] = 10*3600-10
    RunManagement_ref['JobScheduler'] = 'SLURM'
    RunManagement_ref.pop('SecondsMarginForQuitBeforeTimeOut')

    write_cfd_files.set_default(RunManagement)

    for key, value in RunManagement.items():
        if key == 'JobSchedulerOptions':
            continue
        assert value == RunManagement_ref[key]

def test_get_job_text_sator():
    RunManagement = dict(
        NumberOfProcessors = 5,
        Network = 'onera',
        Machine = 'sator', 
        AER='000X111A',
    )
    write_cfd_files.set_default(RunManagement)
    job_text = write_cfd_files.get_job_text(RunManagement, 'my_solver')

    assert job_text == f'''#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --constraint=csl
#SBATCH --job-name=mola
#SBATCH --comment=000X111A
#SBATCH --ntasks=5
#SBATCH --output=output.%j.log
#SBATCH --error=error.%j.log

source {RunManagement["mola_target_path"]}/mola/env/onera/sator/my_solver.sh
'''

def test_get_job_text_spiro():
    RunManagement = dict(
        NumberOfProcessors = 5,
        Network = 'onera',
        Machine = 'spiro', 
        AER='000X111A',
    )
    write_cfd_files.set_default(RunManagement)
    job_text = write_cfd_files.get_job_text(RunManagement, 'my_solver')

    assert job_text == f'''#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --qos=c1_test_giga
#SBATCH --job-name=mola
#SBATCH --comment=000X111A
#SBATCH --ntasks=5
#SBATCH --output=output.%j.log
#SBATCH --error=error.%j.log

source {RunManagement["mola_target_path"]}/mola/env/onera/spiro/my_solver.sh
'''