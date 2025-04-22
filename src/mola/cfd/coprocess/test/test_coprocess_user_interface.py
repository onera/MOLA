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

from mola.cfd.coprocess.manager import CoprocessManager, names, write_tagfile
from mola.cfd.coprocess import user_interface


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

# - tests - #

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_user_signal(tmp_path):

    workflow = FakeWorkflow(tmp_path)
    coprocess = CoprocessManager(workflow)

    write_tagfile('A_SIGNAL', coprocess)
    assert user_interface.get_user_signal(coprocess, 'A_SIGNAL')
    assert not os.path.isfile(os.path.join(tmp_path,'A_SIGNAL'))
    coprocess.status = 'COMPLETED'


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_save_extractions_from_types():
    Extractions = [
        dict(Type='BC', Name='extraction1'), 
        dict(Type='3D', Name='extraction2'),
        dict(Type='BC', Name='extraction3'), 
    ]
    user_interface.save_extractions_from_types(Extractions, ['BC'])
    assert Extractions[0] == dict(Type='BC', Name='extraction1', IsToExtract=True, IsToSave=True)
    assert Extractions[1] == dict(Type='3D', Name='extraction2')
    assert Extractions[2] == dict(Type='BC', Name='extraction3', IsToExtract=True, IsToSave=True)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_save_extractions_from_filename():
    Extractions = [
        dict(Type='BC', Name='extraction1', File='file1.cgns'), 
        dict(Type='3D', Name='extraction2', File='file2.cgns'), 
        dict(Type='BC', Name='extraction3', File='toto.cgns'), 
    ]
    user_interface.save_extractions_from_filename(Extractions, 'file*', )
    assert Extractions[0] == dict(Type='BC', Name='extraction1', File='file1.cgns', IsToExtract=True, IsToSave=True)
    assert Extractions[1] == dict(Type='3D', Name='extraction2', File='file2.cgns', IsToExtract=True, IsToSave=True)
    assert Extractions[2] == dict(Type='BC', Name='extraction3', File='toto.cgns')
