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
from mola.cfd.coprocess import stopping_criteria


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
def test_check_timeout(tmp_path):

    workflow = FakeWorkflow(tmp_path)
    coprocess = CoprocessManager(workflow)

    workflow.RunManagement['TimeOutInSeconds'] = -1 # forces time-out
    
    coprocess.status = 'RUNNING_BEFORE_ITERATION'
    stopping_criteria.check_timeout(coprocess)
    os.unlink(os.path.join(tmp_path,'NEWJOB_REQUIRED'))
    coprocess.status = 'COMPLETED'

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_has_reached_timeout(tmp_path):
    workflow = FakeWorkflow(tmp_path)
    coprocess = CoprocessManager(workflow)

    workflow.RunManagement['TimeOutInSeconds'] = -1 # forces time-out
    
    assert stopping_criteria.has_reached_timeout(coprocess.launch_time,
            coprocess.workflow.RunManagement['TimeOutInSeconds'])
    
    coprocess.status = 'COMPLETED'

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_check_max_iteration(tmp_path):

    workflow = FakeWorkflow(tmp_path)
    coprocess = CoprocessManager(workflow)

    coprocess.iteration = 1e9 # forces itmax
    
    coprocess.status = 'RUNNING_BEFORE_ITERATION'
    stopping_criteria.check_max_iteration(coprocess)
    coprocess.status = 'COMPLETED'


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_check_convergence_criteria_when_empty(tmp_path):

    workflow = FakeWorkflow(tmp_path)
    coprocess = CoprocessManager(workflow)

    workflow.ConvergenceCriteria = [] # note this is empty
    coprocess.iteration = 3 # able to evaluate convergence
    
    coprocess.status = 'RUNNING_BEFORE_ITERATION'
    stopping_criteria.check_convergence_criteria(coprocess)
    coprocess.status = 'COMPLETED'

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_check_convergence_criteria_verified(tmp_path):

    workflow = FakeWorkflow(tmp_path)

    workflow.Extractions += [ dict(Type='Integral', Name='FamA') ]

    workflow.ConvergenceCriteria = [
        dict(Variable='var1', ExtractionName='FamA',
            Threshold=1.1,
            Necessary=True,
            Sufficient=True),
        ]

    coprocess = CoprocessManager(workflow)
    coprocess.iteration = 3 # able to evaluate convergence
    coprocess.Extractions[-1]['Data'] = cgns.newZoneFromDict( 'FamA', 
             {'Iteration'    : np.array([1,2,3]),
                        'var1'     : np.array([1.2,1.1,1.0])} )

    coprocess.status = 'RUNNING_BEFORE_ITERATION'
    stopping_criteria.check_convergence_criteria(coprocess)
    assert coprocess.status == 'TO_STOP'
    coprocess.status = 'COMPLETED'
    

@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("params",[
    
    dict(Criteria=[
            dict( Threshold = 1.1, Necessary = False, Sufficient = True),],
        ConvergenceIsExpected = True),

    dict(Criteria=[
            dict( Threshold = 0.9, Necessary = False, Sufficient = True),],
        ConvergenceIsExpected = False),

    dict(Criteria=[
            dict( Threshold = 0.9, Necessary = True, Sufficient = False),
            dict( Threshold = 1.1, Necessary = False, Sufficient = True),],
        ConvergenceIsExpected = False),

    dict(Criteria=[
            dict( Threshold = 1.1, Necessary = True, Sufficient = False),
            dict( Threshold = 0.9, Necessary = False, Sufficient = True),],
        ConvergenceIsExpected = True),

    dict(Criteria=[
            dict( Threshold = 0.9, Necessary = False, Sufficient = True),
            dict( Threshold = 1.1, Necessary = True, Sufficient = False),
            dict( Threshold = 1.2, Necessary = True, Sufficient = False),],
        ConvergenceIsExpected = True),

    dict(Criteria=[
            dict( Threshold = 0.9, Necessary = False, Sufficient = True),
            dict( Threshold = 0.9, Necessary = True, Sufficient = False),
            dict( Threshold = 0.9, Necessary = True, Sufficient = False),],
        ConvergenceIsExpected = False),

    ]
    )
def test_is_converged(tmp_path, params):

    workflow = FakeWorkflow(tmp_path)


    for i, criterion in enumerate(params['Criteria']):
        workflow.Extractions += [ dict(Type='Integral', Name='Fam%d'%i) ]

        workflow.ConvergenceCriteria += [
            dict(Variable='var%d'%i, ExtractionName='Fam%d'%i,
                Threshold=criterion['Threshold'],
                Necessary=criterion['Necessary'],
                Sufficient=criterion['Sufficient']),
            ]

        criterion['Name'] = 'Fam%d'%i
        criterion['Data'] = cgns.newZoneFromDict( 'Fam%d'%i, 
             {'Iteration'    :np.array([1,2,3]),
                        'var%d'%i    :np.array([1.2,1.1,1.0])} )

    coprocess = CoprocessManager(workflow)

    def find_extraction_by_criterion_name(Extractions, criterion_name):
        for extraction in Extractions:
            if extraction['Name'] == criterion_name:
                return extraction
        raise ValueError('could not find extraction')

    for criterion in params['Criteria']:
        extraction = find_extraction_by_criterion_name(coprocess.Extractions, criterion['Name'])
        extraction['Data'] = criterion['Data']

    assert stopping_criteria.is_converged(coprocess) == params['ConvergenceIsExpected']

    coprocess.status = 'COMPLETED'


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_is_criterion_flux_lower_than_threshold():

    criterion = dict(Variable='var1', ExtractionName='FamA', Threshold=1)

    check_value = 0.3

    data1 = cgns.newZoneFromDict( 'zone', dict(Iteration=np.array([1,2,3]),
                                               var1=np.array([0.1,0.2,check_value])) )
    Extractions = [
        dict(Type='Integral', Name='FamA', Data=data1),
    ]

    is_lower = stopping_criteria.is_criterion_flux_lower_than_threshold(criterion, Extractions)
    
    assert is_lower

    assert criterion['CriterionVerified']



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_data_to_test_criterion():

    criterion = dict(Variable='var1', ExtractionName='FamA', Threshold=1)

    check_value = 0.3

    data1 = cgns.newZoneFromDict( 'zone', dict(Iteration=np.array([1,2,3]),
                                               var1=np.array([0.1,0.2,check_value])) )
    Extractions = [
        dict(Type='Integral', Name='FamA', Data=data1),
    ]

    Flux = stopping_criteria.get_data_to_test_criterion(criterion, Extractions)
    
    assert Flux[-1] == check_value





@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_convergence_message():
    
    ConvergenceCriteria = [
        dict(Variable='var1', FoundValue=0.1, Threshold=1, ExtractionName='FamilyA', Necessary=True, Sufficient=False, CriterionVerified=True),
        dict(Variable='var2', FoundValue=0.2, Threshold=0.9, ExtractionName='FamilyB', Necessary=True, Sufficient=False, CriterionVerified=True),
        dict(Variable='var3', FoundValue=0.1, Threshold=1, ExtractionName='FamilyB', Necessary=True, Sufficient=False, CriterionVerified=True),
        dict(Variable='var4', FoundValue=0.1, Threshold=1, ExtractionName='FamilyB', Necessary=False, Sufficient=True, CriterionVerified=False),
    ]

    txt = stopping_criteria.get_convergence_message(ConvergenceCriteria, 1)

    assert 'var1' in txt
    assert 'var2' in txt
    assert 'var3' in txt    
    assert 'var4' not in txt    

    print(txt)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_write_tagfile(tmp_path):
    workflow = FakeWorkflow(tmp_path)
    coprocess = CoprocessManager(workflow)

    stopping_criteria.write_tagfile('NEWJOB_REQUIRED', coprocess)
    os.unlink(os.path.join(tmp_path,'NEWJOB_REQUIRED'))
    coprocess.status = 'COMPLETED'



if __name__ == '__main__':
    # test_check_timeout(".")
    # test_has_reached_timeout(".")
    test_is_converged(".")