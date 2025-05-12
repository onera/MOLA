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
from pathlib import Path 

from mola import naming_conventions as names
from .test_workflow import  get_workflow_cart_monoproc

@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_4
def test_stop_and_go_for_timeout(tmp_path, niter=500):
    
    w = get_workflow_cart_monoproc(tmp_path)
    
    w.Numerics['NumberOfIterations'] = niter
    w.RunManagement['Scheduler'] = 'local'

    w.prepare()
    # TimeOutInSeconds is modified to force several runs
    w.RunManagement['TimeOutInSeconds'] = 5  # after prepare() method to hack this value
    w.write_cfd_files()
    w.submit()
    w.assert_completed_without_errors()

#     check_log_files(Path(w.RunManagement['RunDirectory']) / names.DIRECTORY_LOG)
    
# def check_log_files(log_directory : Path):
#     log_path = log_directory
#     coprocess_log_files = log_path.glob(names.FILE_COLOG.replace('.log', ''))
#     lines = []
#     for filename in coprocess_log_files:    
#         with open(filename, 'r') as f:
#             lines += f.readlines        
#             ...


if __name__ == '__main__':
    test_stop_and_go_for_timeout('test_stop_and_go_for_timeout')
