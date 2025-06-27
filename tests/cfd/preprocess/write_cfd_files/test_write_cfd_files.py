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
from mola.cfd.preprocess.write_cfd_files import write_cfd_files

@pytest.mark.network_onera
@pytest.mark.unit
@pytest.mark.cost_level_0
def test_onera_sator_get_job_text():
    RunManagement = dict(
        mola_target_path = __MOLA_PATH__,
        Machine = 'sator',
        Scheduler = 'SLURM',
    )
    scheduler_options = {
        'time': '15:00:00',
        'job-name': 'mytest',
        'output': 'output.%j.log',
        'error': 'error.%j.log',
    }

    job_text = write_cfd_files.get_job_text('fake_solver', RunManagement, scheduler_options)

    env = os.path.join(
        RunManagement['mola_target_path'],
        "mola",
        "env",
        "onera",
        RunManagement['Machine'],
        'fake_solver.sh')
    
    assert job_text == f'''#!/bin/bash
#SBATCH --time=15:00:00
#SBATCH --job-name=mytest
#SBATCH --output=output.%j.log
#SBATCH --error=error.%j.log

source {env}
unset "${{!OMPI_@}}" "${{!MPI_@}}"'''