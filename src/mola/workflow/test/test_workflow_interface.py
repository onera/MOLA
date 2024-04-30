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
from mola.workflow import WorkflowInterface

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_init():
    w = WorkflowInterface()


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_write_tree():
    w = WorkflowInterface()
    w.write_tree('test.cgns')
    os.unlink('test.cgns')

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_workflow_parameters_in_tree(filename=''):
    w = WorkflowInterface()
    w.set_workflow_parameters_in_tree()
    if filename: w.write_tree(filename)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_workflow_parameters_from_tree(filename=''):
    w = WorkflowInterface()
    w.set_workflow_parameters_in_tree()
    w.write_tree('test.cgns')
    w.tree = 'test.cgns'
    w.get_workflow_parameters_from_tree()
    os.unlink('test.cgns')
    if filename: w.write_tree(filename)


if __name__=='__main__':
    w = WorkflowInterface()
    w.set_workflow_parameters_in_tree()
    w.write_tree('test.cgns')
