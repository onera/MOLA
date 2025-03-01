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

from mola.logging import mola_logger, MolaException, MolaAssertionError
from mola.workflow.rotating_component.turbomachinery.workflow import WorkflowTurbomachinery

def get_compressor_example_parameters(RunDirectory):
    params = dict( 
        RawMeshComponents=[
        dict(
            Name='CompressorStage',
            Source='/stck/mola/data/mesh/compressor_example/compressor_example.cgns',
            )
    ],

    ApplicationContext = dict(
        ShaftRotationSpeed = 6000 * np.pi / 30., 
        Rows = dict(
            Rotor = dict(IsRotating=True, NumberOfBlades=30), 
            Stator = dict(NumberOfBlades=40),
        )
    ),

    Flow = dict(
        Mach                  = 0.3,  
        TemperatureStagnation = 288.15,
        PressureStagnation    = 101325.,
    ),

    Turbulence = dict(
        Model='Wilcox2006',
    ),

    Numerics = dict(
        NumberOfIterations = 5,
        CFL = dict(EndIteration=300, StartValue=1., EndValue=30.),
    ),

    BoundaryConditions = [
        dict(Family='Rotor_INFLOW', Type='InflowStagnation'),
        dict(Family='Stator_OUTFLOW', Type='OutflowPressure', Pressure=110e3), #98500.),
        dict(Family='HUB', Type='WallInviscid'),
        dict(Family='SHROUD', Type='WallInviscid'),
        dict(Family='Rotor_stator_10_left', LinkedFamily='Rotor_stator_10_right', Type='MixingPlane')
    ],

    Extractions = [
        # dict(Type='3D', 
        #      Fields=['VelocityX', 'VelocityY', 'VelocityZ', 'Mach', 'Pressure', 'PressureStagnation', 'Entropy'], 
        #      ExtractionPeriod=500, SavePeriod=500),
    ],

    RunManagement=dict(
        JobName='CompressorStage',
        NumberOfProcessors=4,
        RunDirectory=RunDirectory,
        ),
    )
    return params

def get_compressor_example(RunDirectory):
    w = WorkflowTurbomachinery(**get_compressor_example_parameters(RunDirectory))
    return w

def get_compressor_example_rotor_only_parameters(RunDirectory):
    params = dict( 
        RawMeshComponents=[
        dict(
            Name='Base',
            Source='/stck/mola/data/mesh/compressor_example/compressor_example_rotor_only.cgns',
            )
    ],

    ApplicationContext = dict(
        ShaftRotationSpeed = 6000 * np.pi / 30., 
        Rows = dict(
            Rotor = dict(IsRotating=True, NumberOfBlades=30), 
        )
    ),

    Flow = dict(
        Mach                  = 0.3,  
        TemperatureStagnation = 288.15,
        PressureStagnation    = 101325.,
    ),

    Turbulence = dict(
        Model='SA',
    ),

    Numerics = dict(
        NumberOfIterations = 5,
        CFL = dict(EndIteration=300, StartValue=1., EndValue=30.),
    ),

    BoundaryConditions = [
        dict(Family='Rotor_INFLOW', Type='InflowStagnation'),
        dict(Family='Rotor_OUTFLOW', Type='OutflowPressure', Pressure=100e3), 
        dict(Family='HUB', Type='WallInviscid'),
        dict(Family='SHROUD', Type='WallInviscid'),
    ],

    Extractions = [
        dict(Type='3D', 
             Fields=['Mach', 'Pressure', 'PressureStagnation', 'Entropy'], 
             ExtractionPeriod=500, SavePeriod=500),
        dict(Type='BC', Source='Rotor_INFLOW', Fields=['PressureStagnation', 'TemperatureStagnation', 'VelocityX', 'VelocityY', 'VelocityZ']), 
        dict(Type='BC', Source='Rotor_OUTFLOW', Fields=['Pressure']), 
        dict(Type='BC', Source='Rotor_Blade', Fields=['VelocityX', 'VelocityY', 'VelocityZ']), 
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=-0.015, OtherOptions=dict(tag='InletPlane', ReferenceRow='Rotor')),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=0.06, OtherOptions=dict(tag='OutletPlane', ReferenceRow='Rotor')),
        dict(Type='IsoSurface', IsoSurfaceField='ChannelHeight', IsoSurfaceValue=0.5)
    ],

    RunManagement=dict(
        JobName='rotor',
        NumberOfProcessors=1,
        RunDirectory=RunDirectory,
        ),
    )
    return params

def get_compressor_example_rotor_only(RunDirectory):
    w = WorkflowTurbomachinery(**get_compressor_example_rotor_only_parameters(RunDirectory))
    return w

def get_workflow_rotor37(RunDirectory):
    w = WorkflowTurbomachinery( 
        RawMeshComponents=[
            dict(
                Name='rotor37',
                Source = '/stck/mola/data/mesh/rotor37/rotor37.cgns',
                ) 
        ],

        ApplicationContext = dict(
            ShaftRotationSpeed = -1800., 
            # Surface = 0.11062898087649121,
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
            RunDirectory=RunDirectory,
            NumberOfProcessors=4,
            ),

        )
    return w

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_init(tmp_path):
    w = get_compressor_example(tmp_path)
    w.print_interface()
    assert w.Name == 'WorkflowTurbomachinery'

@pytest.mark.integration
@pytest.mark.elsa # since not still functional using Fast nor Sonics
@pytest.mark.cost_level_4
def test_compressor_example_local(tmp_path):
    w = get_compressor_example(tmp_path)
    w.RunManagement['Scheduler'] = "local" # otherwise we will have sync problem at simulation_status
    w.prepare()
    w.write_cfd_files()
    w.submit()
    w.simulation_status()
    w.remove_cfd_files()

@pytest.mark.integration
@pytest.mark.cost_level_4
def test_compressor_example_local_rotor_only(tmp_path):
    w = get_compressor_example_rotor_only(tmp_path)
    w.RunManagement['Scheduler'] = "local" # otherwise we will have sync problem at simulation_status
    if w.Solver == 'fast':
        w.Numerics.update(dict(
            TimeMarching = 'Unsteady',
            TimeStep = 1e-6,
        ))
    w.prepare()
    w.write_cfd_files()
    w.submit()
    w.simulation_status()
    w.remove_cfd_files()

# @pytest.mark.network_onera
# @pytest.mark.user_case
# @pytest.mark.cost_level_4
# def test_rotor37_sator():
#     w = get_workflow_rotor37()
#     w.RunManagement['NumberOfProcessors'] = 12
#     w.RunManagement['RunDirectory'] = f'/tmp_user/sator/$USER/.test/test_rotor37_sator/'
#     scheduler_defaults = SV.get_scheduler_defaults('sator')
#     w.RunManagement['AER'] = scheduler_defaults.AER_FOR_TEST
#     w.RunManagement['TimeLimit'] = '00:30:00'

#     SV.remove_path(w.RunManagement['RunDirectory'], machine='sator', file_only=False)

#     w.prepare()
#     w.write_cfd_files()
#     w.submit()

#     # NOTE: do not wait for job to end, since that approach would provoke
#     # too important delays (waiting for resources of SLURM)
#     # COMPLETED_PATH = os.path.join(w.RunManagement['RunDirectory'], names.FILE_JOB_COMPLETED)
#     # SV.wait_until(SV.is_existing_path, path=COMPLETED_PATH, machine='sator', timeout=180)
#     # SV.remove_path(w.RunManagement['RunDirectory'], machine='sator', file_only=False)


if __name__ == '__main__':
    test_compressor_example_local()
