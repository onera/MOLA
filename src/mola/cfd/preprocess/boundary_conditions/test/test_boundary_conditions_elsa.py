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

from mola.cfd.preprocess.boundary_conditions import solver_elsa
from mola.cfd.preprocess.boundary_conditions.boundary_conditions import BoundaryConditionsNames
from mola.cfd.preprocess.boundary_conditions.test.test_boundary_conditions import get_workflow_prepared_to_test_bcs

from mola.workflow.rotating_component import turbomachinery
from mola.workflow.rotating_component.turbomachinery.test.test_turbomachinery_workflow import get_compressor_example_parameters

pytestmark = pytest.mark.elsa

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_functions_well_defined():
    BoundaryConditionsNamesInElsa = set(v['elsa'] for v in BoundaryConditionsNames.values() if 'elsa' in v)
    for fun_name in BoundaryConditionsNamesInElsa:
        assert getattr(solver_elsa, fun_name)


@pytest.mark.unit
@pytest.mark.cost_level_1
def test_bc():
    BoundaryConditions=[
            dict(Family='imin', Type='WallViscous'),
            dict(Family='imax', Type='Farfield'),
            dict(Family='jmin', Type='InflowStagnation'),
            dict(Family='jmax', Type='InflowMassFlow', MassFlow=1.),
            dict(Family='kmin', Type='OutflowPressure'),
            dict(Family='kmax', Type='OutflowMassFlow', MassFlow=1.),
        ]
    workflow = get_workflow_prepared_to_test_bcs(BoundaryConditions)
    workflow.set_boundary_conditions()

@pytest.mark.unit
@pytest.mark.cost_level_1
@pytest.mark.parametrize('interface_type', ['MixingPlane', 'UnsteadyRotorStatorInterface', 'ChorochronicInterface'])
def test_RotorStatorInterface(tmp_path, interface_type):

    params = get_compressor_example_parameters(tmp_path)
    params['BoundaryConditions'] = [
        dict(Family='Rotor_INFLOW', Type='InflowStagnation'),
        dict(Family='Stator_OUTFLOW', Type='OutflowPressure', Pressure=1e5),
        dict(Family='HUB', Type='WallInviscid'),
        dict(Family='SHROUD', Type='WallInviscid'),
        dict(Family='Rotor_stator_10_left', LinkedFamily='Rotor_stator_10_right', Type=interface_type)
    ]

    workflow = turbomachinery.Workflow(**params)

    workflow.assemble()
    workflow.positioning()
    workflow.define_families() 
    workflow.connect()
    workflow.split_and_distribute() 
    workflow.process_overset()
    workflow.compute_flow_and_turbulence()
    workflow.set_motion()

    workflow.set_boundary_conditions()
