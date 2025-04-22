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

from mola.cfd.preprocess.cfd_parameters import cfd_parameters


class FakeWorkflowTwoBlocks():

    def __init__(self, NPts):
        self.tree = cgns.Tree()
        self.ProblemDimension = 3
        base = cgns.Base(Parent=self.tree)
        xyz = np.meshgrid( np.linspace(0,1,NPts),
                           np.linspace(0,1,NPts),
                           np.linspace(0,1,NPts), indexing='ij')
        block1 = cgns.newZoneFromArrays( 'block.1', ['x','y','z'], xyz)
        block1.attachTo(base)

        xyz = np.meshgrid( np.linspace(0,1,NPts)+1,
                           np.linspace(0,1,NPts),
                           np.linspace(0,1,NPts), indexing='ij')
        block2 = cgns.newZoneFromArrays( 'block.2', ['x','y','z'], xyz)
        block2.attachTo(base)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_add_governing_equations():
    workflow = FakeWorkflowTwoBlocks(5)
    workflow.Turbulence = dict( Model = 'SA' )

    cfd_parameters.add_governing_equations(workflow)

    for base in workflow.tree.bases():
        assert base.get('GoverningEquations')
        assert base.get('EquationDimension')
