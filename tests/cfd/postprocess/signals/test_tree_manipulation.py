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

import numpy as np

from treelab import cgns
from mola.cfd.postprocess.signals.tree_manipulation import (
    apply_operations_on_signal, 
    _build_operation_tree, 
    _get_complete_variable_name
)

@pytest.fixture
def test_node():
    node = cgns.Node(Name='Probe')
    cgns.Node(Name='Iteration', Type='DataArray', Parent=node,
              Value=np.arange(10))
    cgns.Node(Name='var', Type='DataArray', Parent=node,
              Value=2*np.arange(10))
    return node

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_apply_operations_on_signal_1(test_node):
    apply_operations_on_signal(test_node, 'rsd-avg-var', 5)
    print(test_node)
    assert test_node.getChildrenNames() == ['Iteration', 'var', 'avg-var', 'rsd-avg-var']
    assert test_node.get(Name='rsd-avg-var').value().size == 10

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_apply_operations_on_signal_2(test_node):
    apply_operations_on_signal(test_node, 'var', 5, ['rsd-avg'])
    assert test_node.getChildrenNames() == ['Iteration', 'var', 'avg-var', 'rsd-avg-var']
    assert test_node.get(Name='rsd-avg-var').value().size == 10

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_apply_operations_on_signal_3(test_node):
    # nothing is done with a window of size 1
    apply_operations_on_signal(test_node, 'var', 1, ['rsd-avg'])
    assert test_node.getChildrenNames() == ['Iteration', 'var']

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_build_operation_tree():
    operation_tree = _build_operation_tree('std-MassFlow', ['rsd-avg', 'std'])

# MassFlow DataArray_t
# └───std DataArray_t
#     ├───avg DataArray_t
#     │   └───rsd DataArray_t ""
#     └───std DataArray_t ""

    assert operation_tree.name() == 'MassFlow'
    std = operation_tree.get(Name='std', Depth=1)
    avg = std.get(Name='avg', Depth=1)
    std2 = std.get(Name='std', Depth=1) 
    assert std is not None
    rsd = avg.get(Name='rsd', Depth=1) 
    assert rsd is not None

    assert _get_complete_variable_name(rsd) == 'rsd-avg-std-MassFlow'
