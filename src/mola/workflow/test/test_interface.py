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
from mola.workflow import WorkflowInterface

class Fake():
    pass

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_init():
    w = WorkflowInterface(workflow=Fake())


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_print():
    w = WorkflowInterface(workflow=Fake())
    print(w)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_add_to_Extractions_Integral():
    w = WorkflowInterface(workflow=Fake())
    w.add_to_Extractions_Integral(Type="Integral", Source="MyFamily", Fields=['MassFlow'])
    assert w.Extractions[-1]['ExtractionPeriod'] == 1 # since by default we extract every iter


if __name__=='__main__':
    test_print()
