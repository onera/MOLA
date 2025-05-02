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

from mola.workflow.fixed.airplane.workflow import WorkflowAirplane

from ....cfd.postprocess.signals.test_airplane_coefficients_computer import (
    assert_coefficients_correctly_added_to_extraction_data,
    application_context,
    zone_with_loads)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_workflow_airplane_init():
    w = WorkflowAirplane()
    w.print_interface()
    assert w.Name == 'WorkflowAirplane'


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_default_flow_direction():
    w = WorkflowAirplane(ApplicationContext=dict(AngleOfAttackDeg=10.0))
    assert np.allclose(w.Flow['Direction'], [0.98480775, 0.0, 0.17364818])

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_compute_aerodynamic_coefficients(small_workflow):
    extraction = small_workflow.Extractions[0]
    small_workflow.compute_aerodynamic_coefficients(extraction)
    assert_coefficients_correctly_added_to_extraction_data(extraction,
                                            small_workflow.ApplicationContext)



# --------------------------------- fixtures --------------------------------- #
@pytest.fixture
def small_workflow(zone_with_loads,application_context):
    w = WorkflowAirplane(
        Extractions = [
                dict(
                    Type='Integral',
                    Source='BCWall',
                    Fields=['ForceX','ForceY','ForceZ','TorqueX','TorqueY','TorqueZ'],
                    ExtractAtEndOfRun=False,
                    PostprocessOperations=[
                        dict(Type="compute_aerodynamic_coefficients"),
                    ]
                )
        ],

    )
    w.Extractions[0]["Data"] = cgns.Tree(Base=zone_with_loads) # mimiks coprocess extraction
    w.ApplicationContext.update(application_context) # overrides computed app ctxt

    return w


