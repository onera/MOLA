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

import os
import numpy as np
from treelab import cgns
from mola import __MOLA_PATH__
from mola.cfd.preprocess.write_cfd_files import solver_fast

import pytest
pytestmark = pytest.mark.fast

import pathlib
module_directory = str(pathlib.Path(__file__).parent.resolve())


class FakeWorkflow():
    def __init__(self):
        self.RunManagement = dict(RunDirectory=module_directory,
                            Machine=os.getenv('MOLA_MACHINE'),
                            NumberOfProcessors=2,
                            NumberOfThreads=3,
                            Scheduler=None,
                            mola_target_path=__MOLA_PATH__,
                            SchedulerOptions = dict(opt1="a", opt2="b"),
                            )
        self.tree = cgns.Tree()
        self._treeCellCenter = cgns.Tree()

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_write_compute():

    RunManagement = dict(RunDirectory=module_directory, Machine=None)
    solver_fast.write_compute(RunManagement)
    os.unlink(os.path.join(module_directory,solver_fast.names.FILE_COMPUTE))


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_write_job_launcher():

    RunManagement = dict(RunDirectory=module_directory,
                         Machine=os.getenv('MOLA_MACHINE'),
                         NumberOfProcessors=2,
                         NumberOfThreads=3,
                         Scheduler=None,
                         mola_target_path=__MOLA_PATH__)
    
    scheduler_options = dict(opt1="a", opt2="b")

    solver_fast.write_job_launcher(RunManagement, scheduler_options)

    os.unlink(os.path.join(module_directory,solver_fast.names.FILE_JOB))


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_write_run_scripts():

    workflow = FakeWorkflow()

    solver_fast.write_run_scripts(workflow)

    os.unlink(os.path.join(module_directory,solver_fast.names.FILE_JOB))
    os.unlink(os.path.join(module_directory,solver_fast.names.FILE_COMPUTE))


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_write_data_files():
    
    workflow = FakeWorkflow()
    
    solver_fast.write_data_files(workflow)

    os.unlink(os.path.join(module_directory,solver_fast.names.FILE_INPUT_SOLVER))
    os.unlink(os.path.join(module_directory,"tc.cgns"))


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_apply_to_solver():
    
    workflow = FakeWorkflow()
    
    solver_fast.apply_to_solver(workflow)

    os.unlink(os.path.join(module_directory,solver_fast.names.FILE_JOB))
    os.unlink(os.path.join(module_directory,solver_fast.names.FILE_COMPUTE))
    os.unlink(os.path.join(module_directory,solver_fast.names.FILE_INPUT_SOLVER))
    os.unlink(os.path.join(module_directory,"tc.cgns"))


if __name__ == '__main__':
    # test_workflow_sphere_struct_local_dist()
    # test_prepare_workflow2()
    test_write_compute()