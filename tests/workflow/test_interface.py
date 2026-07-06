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
from mola.logging import MolaUserError, MolaUserAttributeError

class Fake():
    pass

@pytest.mark.unit
def test_init():
    w = WorkflowInterface(workflow=Fake())


@pytest.mark.unit
def test_print():
    w = WorkflowInterface(workflow=Fake())
    print(w)

@pytest.mark.unit
def test_add_to_Extractions_Integral():
    w = WorkflowInterface(workflow=Fake())
    w.add_to_Extractions_Integral(Type="Integral", Source="MyFamily", Fields=['MassFlow'])
    assert w.Extractions[-1]['ExtractionPeriod'] == 1 # since by default we extract every iter


@pytest.mark.unit
@pytest.mark.parametrize(
    ('params', 'expected_generator'), [
        (dict(Density=1., Temperature=288.15, Velocity=10.), 'External_rho_T_V'), 
        (dict(Mach=0.3, Temperature=288.15, Pressure=1e5), 'External_Mach_P_T'),
        (dict(PressureStagnation=1e5, TemperatureStagnation=288.15, Mach=0.5), 'Internal'),
        (dict(PressureStagnation=1e5, TemperatureStagnation=288.15, MassFlow=10.), 'Internal'),
    ]
)
def test_flow_generator_selection(params, expected_generator):
    w = WorkflowInterface(workflow=Fake())
    w.set_Flow(**params)
    assert w.Flow['Generator'] == expected_generator

@pytest.mark.unit
def test_flow_generator_selection_error():
    w = WorkflowInterface(workflow=Fake())
    with pytest.raises(MolaUserError):
        w.set_Flow(Density=1., Temperature=288.15, Fake=-1)

@pytest.mark.unit
def test_flow_generator_given():
    w = WorkflowInterface(workflow=Fake())
    with pytest.raises(MolaUserAttributeError):
        w.set_Flow(Generator='Internal', Density=1., Temperature=288.15, Velocity=10.)

