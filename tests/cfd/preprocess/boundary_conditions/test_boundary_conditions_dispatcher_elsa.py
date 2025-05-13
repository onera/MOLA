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

from mola.cfd.preprocess.boundary_conditions import _instantiate_bc_dispatcher

from mola import solver

class FakeWorkflow():
    def __init__(self) -> None:
        self.Solver = solver

@pytest.fixture
def bc_dispatcher():
    w = FakeWorkflow()
    return _instantiate_bc_dispatcher(w)


@pytest.mark.unit
@pytest.mark.elsa
@pytest.mark.cost_level_0
def test_to_generic_name(bc_dispatcher):
    assert bc_dispatcher.to_generic_name("inj1") == 'InflowStagnation'
    assert bc_dispatcher.to_generic_name("nref") == 'Farfield'
    assert bc_dispatcher.to_generic_name("outpres") == 'OutflowPressure'
        


@pytest.mark.unit
@pytest.mark.elsa
@pytest.mark.cost_level_0
def test_is_allowed_shell_pattern(bc_dispatcher):
    assert bc_dispatcher.is_allowed_shell_pattern("*inj1*")
    assert bc_dispatcher.is_allowed_shell_pattern("*stage_mxpl*")