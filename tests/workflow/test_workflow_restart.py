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
import numpy as np

from treelab import cgns
from mola import solver
from mola import naming_conventions as names
from mola.workflow import Workflow, read_workflow
from .test_workflow import get_workflow_cart_monoproc


@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.cost_level_4
@pytest.mark.parametrize('TimeMarching', ['Steady', 'Unsteady'])
def test_restart(tmp_path, TimeMarching, niter_first_run=5, niter_second_run=7):

    # One whole run 
    single_run_dir = tmp_path / "single_run"
    w = get_workflow_cart_monoproc(single_run_dir)
    if TimeMarching == 'Unsteady':
        w.Numerics['TimeMarching'] = TimeMarching
        w.Numerics['TimeMarchingOrder'] = 2
        w.Numerics['TimeStep'] = 1e-6
    w.Numerics['NumberOfIterations'] = niter_first_run + niter_second_run
    w.RunManagement['Scheduler'] = 'local'
    w.set_workflow_parameters_in_tree()
    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {single_run_dir}; bash job.sh')
    w.assert_completed_without_errors()

    # Two runs
    two_runs_dir = tmp_path / "two_runs"
    w = get_workflow_cart_monoproc(two_runs_dir)
    if TimeMarching == 'Unsteady':
        w.Numerics['TimeMarching'] = TimeMarching
        w.Numerics['TimeMarchingOrder'] = 2
        w.Numerics['TimeStep'] = 1e-6
    w.Numerics['NumberOfIterations'] = niter_first_run
    w.RunManagement['Scheduler'] = 'local'
    w.set_workflow_parameters_in_tree()
    w.prepare()

    # First run
    w.write_cfd_files()
    w.submit(f'cd {two_runs_dir}; bash job.sh')
    w.assert_completed_without_errors()

    # update NumberOfIterations, it was 0 at the end of the first run
    os.system(f'cd {two_runs_dir}; mola_update --NumberOfIterations={niter_second_run}')

    # Second run: we must read the updated file main.cgns with workflow reader
    w = read_workflow(two_runs_dir / names.FILE_INPUT_SOLVER)
    w.submit(f'cd {two_runs_dir}; bash job.sh')
    w.assert_completed_without_errors()
    
    # Check that residuals in both directories are the same
    signals_without_restart = cgns.load(str(single_run_dir / names.DIRECTORY_OUTPUT / names.FILE_OUTPUT_1D))
    signals_with_restart = cgns.load(str(two_runs_dir / names.DIRECTORY_OUTPUT / names.FILE_OUTPUT_1D))
    residuals_zone = signals_without_restart.get(Name='Residuals', Type='Zone')
    all_residuals_names = [data.name() for data in residuals_zone.group('DataArray')]
    for residual_name in all_residuals_names:
        residual_without_restart = signals_without_restart.get(Name=residual_name).value()
        residual_with_restart = signals_with_restart.get(Name=residual_name).value()
        assert np.allclose(residual_without_restart, residual_with_restart, rtol=1e-12, atol=1e-12)


@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_4
@pytest.mark.parametrize('TimeMarching', ['Steady', 'Unsteady'])
def test_stop_and_go_for_timeout(tmp_path, TimeMarching, niter=500):
    
    w = get_workflow_cart_monoproc(tmp_path)

    if TimeMarching == 'Unsteady':
        w.Numerics['TimeMarching'] = TimeMarching
        w.Numerics['TimeMarchingOrder'] = 2
        w.Numerics['TimeStep'] = 1e-6

    w.Numerics['NumberOfIterations'] = niter
    w.RunManagement['Scheduler'] = 'local'

    w.prepare()
    # TimeOutInSeconds is modified to force several runs
    w.RunManagement['TimeOutInSeconds'] = 5  # after prepare() method to hack this value
    w.write_cfd_files()
    w.submit()
    w.assert_completed_without_errors()

