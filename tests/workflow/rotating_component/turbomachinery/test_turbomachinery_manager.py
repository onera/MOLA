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
from pathlib import Path
import mola.naming_conventions as names

from mola.workflow.rotating_component import turbomachinery
from .test_turbomachinery_workflow import get_compressor_example_rotor_only

@pytest.mark.integration
@pytest.mark.cost_level_1
def test_WorkflowManager_prepare(tmp_path):

    test_dir = str(tmp_path)
    w = get_compressor_example_rotor_only('not_used_directory')
    manager = turbomachinery.WorkflowManager(w, root_directory=test_dir,
                    manager_file_path=str(tmp_path/names.FILE_WORKFLOW_MANAGER))
    manager.add_isospeed_line(throttles=[1e5, 1.1e5, 1.2e5])
    manager.prepare()

    root_dirs = []
    files_list = []
    for root, dirs, files in os.walk(test_dir):
        root_dirs.append(root)
        files_list.append(files)

    assert set(Path(p) for p in root_dirs) == {
        tmp_path, 
        tmp_path/'isospeed_6000rpm',
        tmp_path/'isospeed_6000rpm'/'Pressure_100000.00',
        tmp_path/'isospeed_6000rpm'/'Pressure_110000.00',
        tmp_path/'isospeed_6000rpm'/'Pressure_120000.00',
        }
    