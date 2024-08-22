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
import shutil

from treelab import cgns
from mola.cfd.coprocess import comm, rank, NumberOfProcessors
from mola.cfd.compute import apply as compute_apply
from mola.cfd.coprocess.manager import CoprocessManager, MolaException, names
from mola.workflow.test.test_workflow import get_workflow_cart_monoproc

class FakeWorkflow():
    def __init__(self,RunDirectory=None):
        
        if RunDirectory is None: RunDirectory = '.' # trick
        
        self.Solver = os.environ.get('MOLA_SOLVER')

        self.Numerics = dict(IterationAtInitialState=1,
                                    NumberOfIterations=3,
                             MinimumNumberOfIterations=1,
                                    TimeAtInitialState=0.0,
                                        TimeMarching='Unsteady',
                                        TimeStep=0.1)
        
        self.Extractions = [
            dict(Type='Integral', Source='BCWall', Name="MyExtraction",
                 ExtractionPeriod=1, SavePeriod=1, Override=True,
                 ExtractAtEndOfRun=True, File=names.FILE_OUTPUT_1D),
        ]
        
        self.RunManagement = dict(RunDirectory=RunDirectory,
                                  TimeOutInSeconds=120)
        
        self.ConvergenceCriteria = []


def check_existance_of_coprocess_files_and_directories_by_removing_them(path):

    for file in [names.FILE_COLOG]:
        filepath = os.path.join(path,file)
        os.unlink(filepath)

    for directory in [names.DIRECTORY_OUTPUT, names.DIRECTORY_LOG]:
        dirpath = os.path.join(path,directory)
        shutil.rmtree(dirpath)

# - tests - #

# @pytest.mark.unit
# @pytest.mark.cost_level_0
# def test_run_iteration(tmp_path):
    
#     workflow = FakeWorkflow(tmp_path)
#     coprocess = CoprocessManager(workflow)

#     coprocess.run_iteration()

#     coprocess.status = 'COMPLETED'


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_update_iteration(tmp_path):

    workflow = FakeWorkflow(tmp_path)
    coprocess = CoprocessManager(workflow)

    coprocess.update_iteration()
    
    coprocess.status = 'COMPLETED'


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_status(tmp_path):
    
    workflow = FakeWorkflow(tmp_path)
    coprocess = CoprocessManager(workflow)

    assert coprocess.status == 'BEFORE_FIRST_ITERATION'

    coprocess.status = 'RUNNING'
    assert coprocess.status == 'RUNNING'

    try:
        coprocess.status = 'UNEXPECTED'
    except MolaException as e:
        pass

    coprocess.status = 'COMPLETED'
    assert coprocess.status == 'COMPLETED'

    check_existance_of_coprocess_files_and_directories_by_removing_them(tmp_path)

