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

import mola.naming_conventions as names
from mola.logging import MolaException
from mola.workflow import WorkflowTurbomachinery
from mola import server as SV

def get_workflow_rotor37():
    w = WorkflowTurbomachinery( 
        RawMeshComponents=[
            dict(
                Name='rotor37',
                Source = '/stck/mola/data/mesh/rotor37/rotor37.cgns',
                ) 
        ],

        ApplicationContext = dict(
            ShaftRotationSpeed = -1800., 
            Rows = dict(
                R37 = dict(
                    IsRotating = True,
                    NumberOfBlades = 36,
                )
            )
        ),

        Flow = dict(
            MassFlow              = 20.5114,  # for the 360 degrees section, even it is simulated entirely
            TemperatureStagnation = 288.15,
            PressureStagnation    = 101330.,
        ),

        Turbulence = dict(
            Level = 0.03,
            Viscosity_EddyMolecularRatio = 0.1,
            Model = 'smith',
        ),

        Numerics = dict(
            NumberOfIterations = 5,
            CFL = dict(EndIteration=300, StartValue=1., EndValue=30.)
        ),

        BoundaryConditions = [
            dict(Family='R37_INFLOW', Type='InflowStagnation'),
            dict(Family='R37_OUTFLOW', Type='OutflowPressure', Pressure=0.9936*1e5),
        ],

        Extractions = [
            dict(Type='IsoSurface', IsoSurfaceField='ChannelHeight', IsoSurfaceValue=0.9)
        ],

        RunManagement=dict(
            JobName='rotor37',
            RunDirectory=os.path.dirname(os.path.realpath(__file__)),
            NumberOfProcessors=4,
            ),

        )
    return w



@pytest.mark.user_case
@pytest.mark.cost_level_4
def test_rotor37_local():
    w = get_workflow_rotor37()
    w.prepare()
    w.write_cfd_files()
    w.submit()
    COMPLETED_PATH = os.path.join(w.RunManagement['RunDirectory'], names.FILE_JOB_COMPLETED)
    if not os.path.exists(COMPLETED_PATH):
        raise MolaException('simulation did not ended as expected')
    w.remove_cfd_files()


@pytest.mark.network_onera
@pytest.mark.user_case
@pytest.mark.cost_level_4
def test_rotor37_sator():
    w = get_workflow_rotor37()
    w.RunManagement['NumberOfProcessors'] = 12
    w.RunManagement['RunDirectory'] = f'/tmp_user/sator/$USER/.test/test_rotor37_sator/'
    scheduler_defaults = SV.get_scheduler_defaults('sator')
    w.RunManagement['AER'] = scheduler_defaults.AER_FOR_TEST
    w.RunManagement['TimeLimit'] = '00:30:00'

    SV.remove_path(w.RunManagement['RunDirectory'], machine='sator', file_only=False)

    w.prepare()
    w.write_cfd_files()
    w.submit()

    # NOTE: do not wait for job to end, since that approach would provoke
    # too important delays (waiting for resources of SLURM)
    # COMPLETED_PATH = os.path.join(w.RunManagement['RunDirectory'], names.FILE_JOB_COMPLETED)
    # SV.wait_until(SV.is_existing_path, path=COMPLETED_PATH, machine='sator', timeout=180)
    # SV.remove_path(w.RunManagement['RunDirectory'], machine='sator', file_only=False)


if __name__ == '__main__':
    # test_show_interface_1()
    test_rotor37_local()
    # test_wip()
