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

from mola.cfd.preprocess.boundary_conditions import solver_fast
from mola.cfd.preprocess.boundary_conditions.boundary_conditions_dispatcher_fast import BoundaryConditionsDispatcherFast
from .test_boundary_conditions import get_workflow_prepared_to_test_bcs

pytestmark = pytest.mark.fast

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_functions_well_defined():
    bc_dict = BoundaryConditionsDispatcherFast()
    for expected_function_name in bc_dict.get_all_specific_names():
        assert getattr(solver_fast, expected_function_name)

@pytest.mark.unit
@pytest.mark.cost_level_1
def test_bc_generic():
    BoundaryConditions=[
            dict(Family='imin', Type='Wall'),
            dict(Family='imax', Type='Farfield'),
            dict(Family='jmin', Type='SymmetryPlane'),
            dict(Family='jmax', Type='InflowStagnation'),
            dict(Family='kmin', Type='OutflowPressure'),
            dict(Family='kmax', Type='Farfield'),
        ]
    workflow = get_workflow_prepared_to_test_bcs(BoundaryConditions)
    workflow.set_boundary_conditions()


@pytest.mark.unit
@pytest.mark.cost_level_1
def test_bc_specific():
    BoundaryConditions=[
            dict(Family='imin', Type='BCWall'),
            dict(Family='imax', Type='BCFarfield'),
            dict(Family='jmin', Type='BCSymmetryPlane'),
            dict(Family='jmax', Type='BCInj1'),
            dict(Family='kmin', Type='BCOutpres'),
            dict(Family='kmax', Type='BCFarfield'),
        ]
    workflow = get_workflow_prepared_to_test_bcs(BoundaryConditions)
    workflow.set_boundary_conditions()