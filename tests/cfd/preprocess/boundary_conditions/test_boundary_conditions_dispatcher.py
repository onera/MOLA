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
@pytest.mark.cost_level_0
def test_get_all_generic_names(bc_dispatcher):
    assert bc_dispatcher.get_all_generic_names()

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_all_specific_names(bc_dispatcher):
    assert bc_dispatcher.get_all_specific_names()

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_all_supported_names(bc_dispatcher):
    assert bc_dispatcher.get_all_supported_names()


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_name_used_by_solver(bc_dispatcher):
    for generic_name in bc_dispatcher.get_all_generic_names():
        assert bc_dispatcher.get_name_used_by_solver(generic_name)

    try:
        bc_dispatcher.get_name_used_by_solver("WrongNameBC")
    except AttributeError as e:
        if str(e).startswith('requested boundary-condition type'):
            pass


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_is_supported(bc_dispatcher):
    assert bc_dispatcher.is_supported("Wall")
    assert not bc_dispatcher.is_supported("WrongNameBC")


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_to_generic_name(bc_dispatcher):
    try:
        bc_dispatcher.to_generic_name("WrongNameBC")
    except ValueError as e:
        if not str(e).startswith('bc type "WrongNameBC" does not have a generic name match'):
            raise ValueError("unexpected behavior or error message in test") from e
        
    for specific_name in bc_dispatcher.get_all_specific_names():
        if specific_name in  bc_dispatcher._without_generic_name:
            continue
        assert bc_dispatcher.to_generic_name(specific_name)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_is_allowed_shell_pattern(bc_dispatcher):
    assert bc_dispatcher.is_allowed_shell_pattern("*Wall*")
    assert not bc_dispatcher.is_allowed_shell_pattern("*WrongNameBC*")
