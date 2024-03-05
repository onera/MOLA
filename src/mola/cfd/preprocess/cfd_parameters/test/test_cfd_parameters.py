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

    try:
        cfd_parameters.set_problem_dimension(workflow)
    except AssertionError as e:
        assert e.args[0] == 'All bases have not the same physical dimension'
    else:
        raise AssertionError('set_problem_dimension should raise an error if bases have different dimensions')
    

def test_set_physical_parameters_default():
    assert False, 'Not yet implemented'



default_numerical_parameters = dict(
    CFL = 1.0,
    IterationAtInitialState = 1,
    MinimumNumberOfIterations = 1000,
    NumberOfIterations = 10000,
    Scheme = 'Jameson',
    TimeAtInitialState = 0.0,
    TimeMarching = 'Steady',
    TimeStep = None
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
    
    try:
        cfd_parameters.set_numerical_parameters(workflow)
    except AssertionError as e:
        assert e.args[0] == misc.RED+f'TimeStep must be defined to perform a simulation with TimeMarching={workflow.Numerics["TimeMarching"]}'+misc.ENDC
    else:
        raise AssertionError('set_numerical_parameters should raise an AssertionError if TimeMarching is not Steady and TimeStep is not defined')

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

def test_set_numerical_parameters_cfl_None():
    workflow = FakeWorkflow(Numerics=dict(
        CFL = None
    ))
    
    try:
        cfd_parameters.set_numerical_parameters(workflow)
    except Exception as e:
        assert e.args[0] == misc.RED+'CFL is not defined. Please give a value or function in Workflow.Numerics'+misc.ENDC
    else:
        raise AssertionError('set_numerical_parameters should raise an Exception if CFL is None')


@pytest.mark.parametrize("CFL", ['cfl', [1], np.empty(3)])
def test_set_numerical_parameters_invalid_cfl(CFL):
    workflow = FakeWorkflow(Numerics=dict(
        CFL = CFL
    ))

    try:
        cfd_parameters.set_numerical_parameters(workflow)
    except Exception as e:
        assert e.args[0] == misc.RED+'CFL must be a scalar or a dict'+misc.ENDC
    else:
        raise AssertionError('set_numerical_parameters should raise an Exception if CFL is None')


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
    try:
        cfd_parameters.set_numerical_parameters(workflow)
    except AssertionError:
        return
    else:
        raise AssertionError('set_numerical_parameters should raise an AssertionError if CFL dict is invalid')
