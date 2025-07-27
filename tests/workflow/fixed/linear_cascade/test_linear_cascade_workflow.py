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

from treelab import cgns
from mola.logging import mola_logger, MolaAssertionError
from mola.workflow.fixed.linear_cascade.workflow import WorkflowLinearCascade


def get_workflow_cube():

    x, y, z = np.meshgrid( np.linspace(0,1,21),
                           np.linspace(0,1,21),
                           np.linspace(0,1,21), indexing='ij')
    mesh = cgns.newZoneFromArrays( 'block', ['x','y','z'], [ x,  y,  z ])

    w = WorkflowLinearCascade(

        RawMeshComponents=[
            dict(
                Name='cartesian',
                Source=mesh,
                Mesher='default',
                Connection = [dict(Type='PeriodicMatch', Translation=[0,1,0], Tolerance=1e-8),]
                )
        ],

        Solver=os.environ.get('MOLA_SOLVER'),
        )
    return w

def get_workflow_spleen(tmp_path):

    w = WorkflowLinearCascade(

        RawMeshComponents=[
            dict(
                Name='SPLEEN_Base',
                Source='/stck/mola/data/open/mesh/spleen/SPLEEN.cgns',
                )
        ],

        Solver=os.environ.get('MOLA_SOLVER'),

        Flow = dict(
            Mach = 0.45,
            TemperatureStagnation = 285.,
            PressureStagnation = 8883.,
        ),

        ApplicationContext = dict(
            AngleOfAttackDeg = -37.3,
        ),

        Turbulence = dict(
            Level = 0.025,
            Viscosity_EddyMolecularRatio = 0.1,
            Model = 'SA',
        ),

        Numerics = dict(
            NumberOfIterations=2,
            CFL=1.,
        ),

        BoundaryConditions = [
            dict(Family='SPLEEN_INFLOW', Type='InflowStagnation'),
            dict(Family='SPLEEN_OUTFLOW', Type='OutflowPressure', Pressure=8883./1.6913),
            dict(Family='SPLEEN_BLADE', Type='WallViscous'),
            dict(Family='HUB', Type='WallInviscid'),
            dict(Family='SHROUD', Type='WallInviscid'),
        ],

        Extractions = [
            dict(Type='BC', Source='SPLEEN_BLADE', Fields=['Pressure'], ExtractAtEndOfRun=True),
            # dict(Type='IsoSurface', IsoSurfaceField='CoordinateZ', IsoSurfaceValue=0.001, ExtractAtEndOfRun=True), # midspan
            dict(Type='IsoSurface', IsoSurfaceField='ChannelHeight', IsoSurfaceValue=0.5, ExtractAtEndOfRun=True), # midspan
        ],

        RunManagement = dict(
            NumberOfProcessors = 4,
            RunDirectory = tmp_path,
            ),

    )
    return w

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_init():
    w = get_workflow_cube()
    w.print_interface()
    assert w.Name == 'WorkflowLinearCascade'

@pytest.mark.unit
@pytest.mark.elsa
@pytest.mark.fast  # no sonics because error to create the PeriodicMatch with Maia without families
@pytest.mark.cost_level_1
def test_get_periodic_direction():
    w = get_workflow_cube()
    w.assemble()
    w.positioning()
    w.connect()
    periodic_direction = w.get_periodic_direction()
    translation = np.array(w.RawMeshComponents[0]['Connection'][0]['Translation'])
    assert np.allclose(np.absolute(periodic_direction), np.absolute(translation))

# @pytest.mark.unit
# @pytest.mark.cost_level_3
# def test_parametrize_with_height_with_turbo():
#     w = get_workflow_cube()
#     w.assemble()
#     try:
#         w.parametrize_with_height_with_turbo('XY')
#         assert w.tree.get(Name='FlowSolution#Height', Type='FlowSolution')
#     except ImportError:
#         mola_logger.warning('turbo module cannot be found!')
#         pass

@pytest.mark.unit
@pytest.mark.cost_level_1
def test_parametrize_with_height(tmp_path):
    w = get_workflow_spleen(tmp_path)
    w.assemble()
    w.Initialization['ParametrizeWithHeight'] = 'maia'
    w.parametrize_with_height()
    assert w.tree.get(Name='FlowSolution#Height', Type='FlowSolution')

@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.sonics
@pytest.mark.cost_level_3
def test_spleen_cascade(tmp_path):
    w = get_workflow_spleen(tmp_path)
    w.RunManagement['Scheduler'] = 'local'
    try:
        w.prepare()
    except KeyError as e:
        w.tree.save('debug.cgns')
        raise KeyError('catched error') from e
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()
