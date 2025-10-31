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
from mola import naming_conventions as names
from mola.workflow.rotating_component import turbomachinery

def get_compressor_example_parameters(RunDirectory):
    params = dict( 
        RawMeshComponents=[
        dict(
            Name='CompressorStage',
            Source='/stck/mola/data/open/mesh/compressor_example/compressor_example.cgns',
            )
    ],

    ApplicationContext = dict(
        ShaftRotationSpeed = 6000., 
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
        Model='SST-V2003',
    ),

    Numerics = dict(
        NumberOfIterations = 5,
        CFL = dict(EndIteration=300, StartValue=1., EndValue=30.),
    ),

    BoundaryConditions = [
        dict(Family='Rotor_INFLOW', Type='InflowStagnation'),
        # dict(Family='Stator_OUTFLOW', Type='OutflowPressure', Pressure=110e3), #98500.),
        dict(Family='Stator_OUTFLOW', Type='OutflowRadialEquilibrium', 
             ValveLaw=dict(Type='Quadratic', ValveCoefficient=0.1)),
        dict(Family='HUB', Type='WallInviscid'),
        dict(Family='SHROUD', Type='WallInviscid'),
        dict(Family='Rotor_stator_10_left', LinkedFamily='Rotor_stator_10_right', Type='MixingPlane')
    ],

    Extractions = [
        # dict(Type='IsoSurface', IsoSurfaceField='ChannelHeight', IsoSurfaceValue=0.5, Fields=['Conservatives']),
    ],

    RunManagement=dict(
        JobName='CompressorStage',
        NumberOfProcessors=4,
        RunDirectory=RunDirectory,
        ),
    )
    return params

def get_compressor_example(RunDirectory):
    w = turbomachinery.Workflow(**get_compressor_example_parameters(RunDirectory))
    return w

def get_compressor_example_rotor_only_parameters(RunDirectory):
    params = dict( 
        RawMeshComponents=[
        dict(
            Name='Base',
            Source='/stck/mola/data/open/mesh/compressor_example/compressor_example_rotor_only.cgns',
            )
    ],

    ApplicationContext = dict(
        ShaftRotationSpeed = 6000., 
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
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=-0.015, Fields=['Conservatives'], OtherOptions=dict(tag='InletPlane', ReferenceRow='Rotor')),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=0.06, Fields=['Conservatives'], OtherOptions=dict(tag='OutletPlane', ReferenceRow='Rotor')),
        dict(Type='IsoSurface', IsoSurfaceField='ChannelHeight', Fields=['Conservatives'], IsoSurfaceValue=0.5)
    ],

    RunManagement=dict(
        JobName='rotor',
        NumberOfProcessors=1,
        RunDirectory=RunDirectory,
        ),
    )
    return params

def get_compressor_example_rotor_only(RunDirectory):
    w = turbomachinery.Workflow(**get_compressor_example_rotor_only_parameters(RunDirectory))
    return w

# @pytest.mark.unit
# @pytest.mark.elsa
# @pytest.mark.cost_level_3
# def test_initialize_with_turbo(tmp_path):
#     w = get_compressor_example(tmp_path)
#     w.Initialization['Method'] = 'turbo'
#     w.ApplicationContext['Rows']['Rotor']['FlowAngleAtTipDeg'] = 30.
#     w.ApplicationContext['Rows']['Rotor']['FlowAngleAtRootDeg'] = 30.

#     w.prepare_job()
#     w.assemble() 
#     w.positioning()
#     w.define_families() 
#     w.connect()
#     w.split_and_distribute() 
#     w.process_overset()
#     w.compute_flow_and_turbulence()
#     w.set_motion()
#     w.set_boundary_conditions()
#     w.set_cfd_parameters()  
    
#     if w.Solver != 'elsa':
#         with pytest.raises(MolaException):
#             w.initialize_flow() 
#     else:
#         w.initialize_flow()

#     w.write_cfd_files()

#     # no other FlowSolution nodes than Init nodes at this stage
#     expected_variables = list(w.Flow['Conservatives']) + list(w.Turbulence['Conservatives'])
#     for zone in w.tree.zones():
#         variables = zone.allFields(include_coordinates=False)
#         assert all([v in variables for v in expected_variables])

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_init(tmp_path):
    w = get_compressor_example(tmp_path)
    w.print_interface()
    assert w.Name == 'WorkflowTurbomachinery'

@pytest.mark.integration
@pytest.mark.cost_level_1
def test_extractions_definition_coherency(tmp_path):
    params = get_compressor_example_rotor_only_parameters(tmp_path)

    params["Extractions"] += [ dict(
            Type='Integral',
            Name='WALL_LOADS',
            Source='WallInviscid',
            Fields=['ForceX','ForceY','ForceZ','TorqueX','TorqueY','TorqueZ'],
            ExtractAtEndOfRun=True,
            PostprocessOperations = [dict(Type="TOTO_OPERATION")],
        ) ]

    w = turbomachinery.Workflow(**params)

    w.prepare()

    found_requested_extraction = False
    for e in w.Extractions:
        if "Name" in e and e["Name"]=="WALL_LOADS": 
            found_requested_extraction = True
    assert found_requested_extraction
    

@pytest.mark.integration
@pytest.mark.elsa  
@pytest.mark.sonics
@pytest.mark.cost_level_4
def test_compressor_example_local_stage(tmp_path):
    w = get_compressor_example(tmp_path)
    w.RunManagement['Scheduler'] = "local" # otherwise we will have sync problem at simulation_status
    w.prepare()
    w.write_cfd_files()
    w.submit()
    w.assert_completed_without_errors()

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
    w.assert_completed_without_errors()

    # Check outputs
    signals = cgns.load(str(tmp_path/names.DIRECTORY_OUTPUT/names.FILE_OUTPUT_1D), only_skeleton=True)
    extractions = cgns.load(str(tmp_path/names.DIRECTORY_OUTPUT/names.FILE_OUTPUT_2D), only_skeleton=True)
    assert signals.get(Name='Integral', Depth=1).get(Name='Rotor_INFLOW').get(Name='MassFlow') is not None
    assert signals.get(Name='Integral', Depth=1).get(Name='Rotor_OUTFLOW').get(Name='MassFlow') is not None
    # after turbo postprocessing
    if w.Solver == 'elsa':
        for var in ['MomentumX', 'ChannelHeight', 'PressureStagnationRel', 'VelocityRelRadius']:
            assert extractions.get(Name='Iso_H_0.5', Depth=1).get(Name=var) is not None
        for var in ['StagnationPressureRelDim', 'VelocityRadiusRel']:  # check these variables have been removed after turbo postprocess
            assert extractions.get(Name='Iso_H_0.5', Depth=1).get(Name=var) is None

if __name__ == '__main__':
    test_compressor_example_local_rotor_only("mytest_compressor_example_local_rotor_only")