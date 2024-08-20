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
