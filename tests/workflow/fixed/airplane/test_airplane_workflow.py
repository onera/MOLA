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

    assert w.SplittingAndDistribution["Strategy"] == 'AtComputation'
    assert w.SplittingAndDistribution["Splitter"] == 'PyPart'
    assert w.SplittingAndDistribution["Distributor"] == 'PyPart'


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_workflow_airplane_cart_init(workflow_cart_monoproc_params):
    w = WorkflowAirplane(**workflow_cart_monoproc_params)
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

@pytest.mark.integration
@pytest.mark.cost_level_1
def test_workflow_airplane_cart_full_pre1_comp1(tmp_path, workflow_cart_monoproc_params):
    w = WorkflowAirplane(**workflow_cart_monoproc_params)
    w.RunManagement['RunDirectory'] = str(tmp_path)

    w.prepare()
    w.write_cfd_files()
    w.submit()

    w.assert_completed_without_errors()


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

@pytest.fixture
def workflow_cart_monoproc_params():
    mesh = get_cart_block()

    params = dict(
        RawMeshComponents=[
            dict(
                Name='cart',
                Source=mesh,
                Families=[
                    dict(Name='Ground',
                         Location='kmin'),
                    dict(Name='Inlet',
                         Location='imin'),
                    dict(Name='Farfield',
                         Location='remaining'),
                ],
                )
        ],

        Flow=dict(
            Density = 0.2,
            Temperature = 100.,
            Velocity = 50.,
                 ),

        Turbulence = dict(
            Model = 'SA',
        ),

        Numerics = dict(
            NumberOfIterations=2,
            CFL=1.0,
        ),

        BoundaryConditions=[
            dict(Family='Ground',   Type='Wall'),
            dict(Family='Inlet',    Type='Farfield'),
            dict(Family='Farfield', Type='Farfield'),
        ],

        RunManagement=dict(
            NumberOfProcessors=1,
            RunDirectory='.',
            Scheduler = 'local',
            ),
        )

    return params

def get_cart_block(n_pts_dir = 14):
    assert n_pts_dir > 13 # otherwise RSD_L2_rh == 0 and elsa stops at it=1
    x, y, z = np.meshgrid( np.linspace(0,1,n_pts_dir),
                           np.linspace(0,1,n_pts_dir),
                           np.linspace(0,1,n_pts_dir), indexing='ij')
    block = cgns.newZoneFromArrays( 'block', ['x','y','z'], [ x,  y,  z ])
    return block