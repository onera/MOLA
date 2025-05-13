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
import os
import numpy as np

from mola.workflow.fixed.airfoil.workflow import WorkflowAirfoil


def get_workflow_micro_naca(tmp_path):
    w = WorkflowAirfoil(

        RawMeshComponents=[
            dict( Name='AIRFOIL',
                  Source='/stck/mola/data/open/mesh/micro_naca0012/mesh.cgns')
        ],

        Solver=os.environ.get('MOLA_SOLVER'),

        Flow = dict(
            Velocity = 100.0,
            Density = 1.225,
            Temperature = 288.15,
        ),

        ApplicationContext = dict(
            AngleOfAttackDeg = 5.0,
        ),

        Turbulence = dict(
            Level = 0.1 * 0.01,
            Viscosity_EddyMolecularRatio = 0.1,
            Model = 'SA',
        ),

        Numerics = dict(
            NumberOfIterations=2,
            CFL=1.,
        ),

        BoundaryConditions = [
            dict(Family='AIRFOIL', Type='Wall'),
            dict(Family='FARFIELD', Type='Farfield'),
        ],

        RunManagement = dict(
            NumberOfProcessors = 1,
            RunDirectory = tmp_path,
            ),
    )

    return w


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_workflow_airfoil_init():
    w = WorkflowAirfoil()
    w.print_interface()
    assert w.Name == 'WorkflowAirfoil'



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_default_flow_direction():
    w = WorkflowAirfoil(ApplicationContext=dict(AngleOfAttackDeg=10.0))
    assert np.allclose(w.Flow['Direction'], [0.98480775, 0.17364818, 0.0])


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_default_transition_zones():
    w = WorkflowAirfoil(Turbulence=dict(TransitionMode='zonal'))
    expected_keys = [
        "TopOrigin",
        "BottomOrigin",
        "TopLaminarImposedUpTo",
        "TopLaminarIfFailureUpTo",
        "TopTurbulentImposedFrom",
        "BottomLaminarImposedUpTo",
        "BottomLaminarIfFailureUpTo",
        "BottomTurbulentImposedFrom"]
    
    for key in expected_keys:
        assert key in w.Turbulence["TransitionZones"]

    kept_the_inherited_model = 'Model' in w.Turbulence
    assert kept_the_inherited_model


@pytest.mark.integration
@pytest.mark.elsa # TODO include fast and sonics
@pytest.mark.cost_level_3
def test_micro_naca(tmp_path):
    w = get_workflow_micro_naca(tmp_path)
    w.RunManagement['Scheduler'] = 'local'
    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()


if __name__ == '__main__':
    test_micro_naca('testing_micro_naca')