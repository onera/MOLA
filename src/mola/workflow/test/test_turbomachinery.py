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

import mola.naming_conventions as names
from mola.workflow import WorkflowTurbomachinery
from mola.logging import mola_logger, MolaException, MolaAssertionError
from mola import server as SV

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

@pytest.mark.unit
@pytest.mark.elsa
@pytest.mark.sonics
@pytest.mark.cost_level_2
def test_duplicate(tmp_path):
    params = get_compressor_example_parameters(tmp_path)
    params['ApplicationContext'] = dict(
        ShaftRotationSpeed = 6000 * np.pi / 30., 
        Rows = dict(
            Rotor = dict(IsRotating=True, NumberOfBlades=30, NumberOfBladesSimulated=2), 
            Stator = dict(NumberOfBlades=40),
        )
    )
    w = WorkflowTurbomachinery(**params)
    w.assemble()
    w.positioning()
    w.connect()
    rotor_zone_names = [zone.name() for zone in w.tree.zones() if zone.get(Type='FamilyName', Depth=1).value() == 'Rotor']
    w.define_families()
    if w.tree.isStructured():
        for name in rotor_zone_names:
            assert w.tree.get(Type='Zone', Name=f'{name}.D0') is not None
            assert w.tree.get(Type='Zone', Name=f'{name}.D1') is not None
    else:
        import maia
        from mpi4py import MPI
        if not w.tree.get(Name='NFaceElements'):
            maia.algo.pe_to_nface(w.tree, MPI.COMM_WORLD) # because for now, compute_azimuthal_extension_from_family use cassiopee and need NFaceElements
        assert np.isclose(w.compute_azimuthal_extension_from_family(w.tree, 'Rotor', [1,0,0]), np.radians(24), rtol=1e-3)


@pytest.mark.user_case
@pytest.mark.elsa # since not still functional using Fast
@pytest.mark.cost_level_4
def test_compressor_example_local(tmp_path):
    w = get_compressor_example(tmp_path)
    w.RunManagement['Scheduler'] = "local" # otherwise we will have sync problem at simulation_status
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
