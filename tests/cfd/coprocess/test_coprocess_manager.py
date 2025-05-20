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

import numpy as np

from treelab import cgns
from mola.cfd.coprocess import comm, rank, NumberOfProcessors
from mola.cfd.coprocess.manager import CoprocessManager, MolaException, names
from mola.cfd.coprocess.tools import update_signals_using, write_extraction_log, write_tagfile


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
                                  TimeLimit="0:2:00")
        
        self.ConvergenceCriteria = []

        self._status = 'RUNNING_BEFORE_ITERATION'
        self._iteration = 0

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
@pytest.mark.elsa
@pytest.mark.fast  # sonics needs the attribute coprocess._iterators
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

    coprocess.status = 'RUNNING_BEFORE_ITERATION'
    assert coprocess.status == 'RUNNING_BEFORE_ITERATION'

    try:
        coprocess.status = 'UNEXPECTED'
    except MolaException as e:
        pass

    coprocess.status = 'COMPLETED'
    assert coprocess.status == 'COMPLETED'

    check_existance_of_coprocess_files_and_directories_by_removing_them(tmp_path)


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize('arrays',[

    dict(previous_it    = [1],
         new_it         =    [2],
         expected_it    = [1, 2],
         expected_field = [1, 20]),

    dict(previous_it    = [1],
         new_it         = [1],
         expected_it    = [1],
         expected_field = [10]),

    dict(previous_it    = [2],
         new_it         = [1],
         expected_it    = [1],
         expected_field = [10]),

    dict(previous_it    = [0, 1],
         new_it         =       [2],
         expected_it    = [0, 1, 2],
         expected_field = [0, 1, 20]),

    dict(previous_it    = [0,1],
         new_it         =   [1],
         expected_it    = [0,1],
         expected_field = [0,10]),

    dict(previous_it    = [0,1,2],
         new_it         =           [4,5,6],
         expected_it    = [0,1,2,    4,5,6],
         expected_field = [0,1,2,   40,50,60]),

    dict(previous_it    =           [4,5,6],
         new_it         = [0,1,2],
         expected_it    = [0,1,2],
         expected_field = [0,10,20]),

    dict(previous_it    = [0,1,2,3],
         new_it         =       [3,4],
         expected_it    = [0,1,2,3,4],
         expected_field = [0,1,2,30,40]),

])
def test_update_signals(arrays):

    previous_it = np.array(arrays['previous_it'])
    previous_field = previous_it

    new_it = np.array(arrays['new_it'])
    new_field = new_it*10

    previous_tree = cgns.Tree(Integral=cgns.newZoneFromDict('ZoneName',
                        dict(Iteration=previous_it, field=previous_field)))
    current_tree = cgns.Tree(Integral=cgns.newZoneFromDict('ZoneName',
                        dict(Iteration=new_it, field=new_field)))

    update_signals_using(current_tree, previous_tree)

    updated_it = previous_tree.get('Iteration').value()
    updated_field = previous_tree.get('field').value()

    assert len(updated_it) == len(updated_field)

    assert len(updated_it.shape) == 1
    assert len(updated_field.shape) == 1

    assert np.allclose(arrays['expected_it'], updated_it )
    assert np.allclose(arrays['expected_field'], updated_field )

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_write_extraction_log():
    extraction = dict(
        Type = 'Integral',
        Source = 'BCWall*',
        toto = 1,
        nested = dict(test=3),
    )
    with pytest.raises(MolaException):
        write_extraction_log(extraction)

    extraction['Data'] = cgns.Tree()
    base = cgns.Base(Name='Base', Parent=extraction['Data'])
    cgns.Zone(Name='Zone', Parent=base)
    write_extraction_log(extraction)
    log = extraction['Data'].get(Type='Zone').getParameters(names.CGNS_NODE_EXTRACTION_LOG)
    for key, value in extraction.items():
        if key == 'Data': continue 
        assert log[key] == value

    extraction['Type'] = 'BC'
    extraction['Data'] = cgns.Tree()
    base = cgns.Base(Name='Base', Parent=extraction['Data'])
    write_extraction_log(extraction)
    log = extraction['Data'].get(Type='CGNSBase').getParameters(names.CGNS_NODE_EXTRACTION_LOG)
    for key, value in extraction.items():
        if key == 'Data': continue 
        assert log[key] == value

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_write_tagfile(tmp_path):
    workflow = FakeWorkflow(tmp_path)
    coprocess = CoprocessManager(workflow)

    write_tagfile('NEWJOB_REQUIRED', coprocess)
    os.unlink(os.path.join(tmp_path,'NEWJOB_REQUIRED'))
    coprocess.status = 'COMPLETED'

if __name__ == '__main__':
    test_update_signals(dict(previous_it    = [1],
                             new_it         =    [2],
                             expected_it    = [1, 2],
                             expected_field = [1, 20]))
