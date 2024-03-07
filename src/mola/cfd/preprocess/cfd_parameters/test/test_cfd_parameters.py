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
import copy
import numpy as np

from treelab import cgns
from mola import misc
from mola.logging import check_error_message
from mola.cfd.preprocess.cfd_parameters import cfd_parameters


class FakeWorkflow():
    def __init__(self, tree=None, Numerics=dict()):
        self.tree = tree
        self.Numerics = Numerics


@pytest.mark.parametrize("CellDimension", [1,2,3])
def test_set_problem_dimension(CellDimension):
    tree = cgns.Tree()
    base1 = cgns.Base(Name='Base1', Parent=tree)  
    base2 = cgns.Base(Name='Base2', Parent=tree) 
    base1.setCellDimension(CellDimension)
    base2.setCellDimension(CellDimension)
    workflow = FakeWorkflow(tree=tree)

    cfd_parameters.set_problem_dimension(workflow)

    assert workflow.ProblemDimension == CellDimension

def test_set_problem_dimension_2():
    tree = cgns.Tree()
    base1 = cgns.Base(Name='Base1', Parent=tree)  
    base2 = cgns.Base(Name='Base2', Parent=tree) 
    base2.setCellDimension(2)
    workflow = FakeWorkflow(tree=tree) 

    check_error_message('All bases have not the same physical dimension', cfd_parameters.set_problem_dimension, workflow)


default_numerical_parameters = dict(
    CFL = 1.0,
    IterationAtInitialState = 1,
    MinimumNumberOfIterations = 1000,
    NumberOfIterations = 10000,
    Scheme = 'Jameson',
    TimeAtInitialState = 0.0,
    TimeMarching = 'Steady',
)

def test_set_numerical_parameters_default():
    workflow = FakeWorkflow(
        Numerics = dict(
            CFL = 1
        )
    )
    cfd_parameters.set_numerical_parameters(workflow)
    assert workflow.Numerics == default_numerical_parameters


def test_set_numerical_parameters_custom():
    new_parameters = dict(
        CFL = 5.0,
        Scheme = 'Roe',
        OtherParam = 'new',
    )
    workflow = FakeWorkflow(Numerics=new_parameters)
    cfd_parameters.set_numerical_parameters(workflow)

    ref_numerical_parameters = copy.copy(default_numerical_parameters)
    ref_numerical_parameters.update(new_parameters)
    assert workflow.Numerics == ref_numerical_parameters

def test_set_numerical_parameters_unsteady_error():
    new_parameters = dict(
        CFL = 5.0,
        TimeMarching = 'gear',
    )
    workflow = FakeWorkflow(Numerics=new_parameters)

    expected_error_msg = f'TimeStep must be defined to perform a simulation with TimeMarching={workflow.Numerics["TimeMarching"]}'
    check_error_message(expected_error_msg, cfd_parameters.set_numerical_parameters, workflow)

def test_set_numerical_parameters_unsteady():
    new_parameters = dict(
        CFL = 5.0,
        TimeMarching = 'gear',
        TimeStep = 1e-3
    )
    workflow = FakeWorkflow(Numerics=new_parameters)
    
    cfd_parameters.set_numerical_parameters(workflow)
    ref_numerical_parameters = copy.copy(default_numerical_parameters)
    ref_numerical_parameters.update(new_parameters)
    ref_numerical_parameters['TimeMarchingOrder'] = 2
    assert workflow.Numerics == ref_numerical_parameters

def test_check_cfl_not_defined():
    workflow = FakeWorkflow(Numerics=dict())
    expected_error_msg = 'CFL is not defined. Please give a value or function in Workflow.Numerics'
    check_error_message(expected_error_msg, cfd_parameters.check_cfl, workflow)

def test_set_numerical_parameters_cfl_None():
    workflow = FakeWorkflow(Numerics=dict(
        CFL = None
    ))
    
    expected_error_msg = 'CFL must be a scalar or a dict'
    check_error_message(expected_error_msg, cfd_parameters.set_numerical_parameters, workflow)

@pytest.mark.parametrize("CFL", ['cfl', [1], np.empty(3)])
def test_set_numerical_parameters_invalid_cfl(CFL):
    workflow = FakeWorkflow(Numerics=dict(
        CFL = CFL
    ))

    expected_error_msg = 'CFL must be a scalar or a dict'
    check_error_message(expected_error_msg, cfd_parameters.set_numerical_parameters, workflow)

def test_set_numerical_parameters_cfl_dict():
    workflow = FakeWorkflow(Numerics=dict(
        IterationAtInitialState = 3,
        CFL = dict(
            EndIteration = 100,
            StartValue = 1,
            EndValue = 3,
        )
    ))
    cfd_parameters.set_numerical_parameters(workflow)
    assert workflow.Numerics['CFL'] == dict(
            StartIteration = 3,
            EndIteration = 100,
            StartValue = 1,
            EndValue = 3,
        )

@pytest.mark.parametrize("CFL", [dict(StartValue=1,EndValue=3), dict(EndIteration=100,EndValue=3), dict(EndIteration=100,StartValue=1)])
def test_set_numerical_parameters_cfl_dict_invalid(CFL):
    workflow = FakeWorkflow(Numerics=dict(CFL=CFL))

    expected_error_msg = f'If CFL is a dict, it must contains at least {", ".join(cfd_parameters.MANDATORY_KEYS_FOR_CFL_DICT)}. \
    You may also define StartIteration, otherwise it will be equal to IterationAtInitialState (1 by default).'

    check_error_message(expected_error_msg, cfd_parameters.set_numerical_parameters, workflow)
