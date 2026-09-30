import os
import pytest
import numpy as np
from treelab import cgns

from mola.workflow.rotating_component.propeller.workflow import WorkflowPropeller
from mola.misc import allclose_dict
from mola import solver
import mola.mesh.propeller_mesher as RW
from ..sector_mesh_tools_for_test import (workflow_sector_params,
                                          write_mesh_and_read_it_again)

@pytest.fixture
def workflow_propeller_sector_params(workflow_sector_params):
    workflow_sector_params["ApplicationContext"] = dict(
        ShaftRotationSpeed = 60.0,
        NumberOfBlades=1)
    
    workflow_sector_params["ConvergenceCriteria"] = [
            dict(
                ExtractionName = 'BLADE',
                Variable = "std-Thrust",
                Threshold = 1.0,
            )
        ]
    
    return workflow_sector_params


@pytest.mark.unit
def test_workflow_propeller_init():
    mandatory_params = dict(
        ApplicationContext=dict(
            ShaftRotationSpeed = 60.0,
            NumberOfBlades=3,
        )
    )
    w = WorkflowPropeller(**mandatory_params)
    w.print_interface()
    assert w.Name == 'WorkflowPropeller'

    
    from pprint import pformat as pretty
    print(pretty(w.ApplicationContext))

    expected_application_context = {
        'IsRotating': True,
        'Length': 1.0,
        'NumberOfBlades': 3,
        'NumberOfBladesInInitialMesh': 1,
        'NumberOfBladesSimulated': 1,
        'TurbulenceSetAtRelativeRadius': 0.75,
        'Rows': {'Propeller': {'IsRotating': True,
                                'NumberOfBlades': 3,
                                'NumberOfBladesInInitialMesh': 1,
                                'NumberOfBladesSimulated': 1}},
        'ShaftAxis': np.array([1., 0., 0.]),
        'ShaftRotationSpeed': 2*np.pi,
        'ShaftRotationSpeedUnit': 'rad/s',
        'Surface': 1.0}

    assert allclose_dict(w.ApplicationContext, expected_application_context)

    assert w.SplittingAndDistribution["Strategy"] == 'AtComputation'
    assert w.SplittingAndDistribution["Splitter"] == 'PyPart'
    assert w.SplittingAndDistribution["Distributor"] == 'PyPart'

@pytest.mark.unit
def test_compute_flow_and_turbulence(tmp_path, workflow_propeller_sector_params):

    workflow_propeller_sector_params['Flow']['Velocity'] = 0.0
    w = WorkflowPropeller(**workflow_propeller_sector_params)
    w.RunManagement['RunDirectory'] = str(tmp_path)

    Ω = w.ApplicationContext['ShaftRotationSpeed']
    rmax = w._blade_radius = 0.1 # trick to avoid process_mesh (accelerates test)
    r_rel = w.ApplicationContext["TurbulenceSetAtRelativeRadius"]
    V = Ω * rmax * r_rel
    Tu = w.Turbulence['Level']
    ρ = 1.0

    w.compute_flow_and_turbulence()

    ρk_expected = ρ*1.5*(Tu**2)*(V**2)
    ρk_computed = w.Turbulence['TurbulentEnergyKineticDensity']

    assert np.isclose(ρk_expected, ρk_computed)


@pytest.mark.unit
def test_set_boundary_conditions(tmp_path, workflow_propeller_sector_params):

    w = WorkflowPropeller(**workflow_propeller_sector_params)
    w.RunManagement['RunDirectory'] = str(tmp_path)

    w.process_mesh()
    w.set_boundary_conditions()

    farfield_node = w.tree.get(Name='FARFIELD', Depth=3)
    family_bc_node = farfield_node.get('FamilyBC')
    assert family_bc_node.value() == 'BCFarfield'



@pytest.mark.integration
@pytest.mark.skipif(solver=='sonics', reason='Requires fixing #324') # FIXME BUG
def test_blade_radius(tmp_path, workflow_propeller_sector_params):

    if solver == 'sonics':
        write_mesh_and_read_it_again(workflow_propeller_sector_params, tmp_path)

    w = WorkflowPropeller(**workflow_propeller_sector_params)
    w.RunManagement['RunDirectory'] = str(tmp_path)

    w.process_mesh()
    w.set_boundary_conditions()
    radius = w.blade_radius()

    assert np.isclose(radius, 0.1)


@pytest.mark.integration
def test_workflow_propeller_sector_pre1_comp1(tmp_path, workflow_propeller_sector_params):

    w = WorkflowPropeller(**workflow_propeller_sector_params)
    if solver == 'sonics': # FIXME BUG https://gitlab.onera.net/numerics/mola/-/work_items/324
        w.ApplicationContext['_BladeRadius'] = 1.0
    w.RunManagement['RunDirectory'] = str(tmp_path)

    w.prepare()
    w.write_cfd_files()
    w.submit()
    w.assert_completed_without_errors()


@pytest.mark.integration
def test_workflow_propeller_sector_pre1_comp2(tmp_path, workflow_propeller_sector_params):
    
    workflow_propeller_sector_params["RunManagement"]["NumberOfProcessors"] = 2
    w = WorkflowPropeller(**workflow_propeller_sector_params)
    if solver == 'sonics': # FIXME BUG https://gitlab.onera.net/numerics/mola/-/work_items/324
        w.ApplicationContext['_BladeRadius'] = 1.0
    w.RunManagement['RunDirectory'] = str(tmp_path)

    w.prepare()
    w.write_cfd_files()
    w.submit()
    w.assert_completed_without_errors()


@pytest.mark.integration
def test_wall_slip_extraction(tmp_path, workflow_propeller_sector_params):
    workflow_propeller_sector_params['Extractions'] = [
        dict(Type='BC', Source='WallInviscid', Fields=['Pressure'])
    ]
    w = WorkflowPropeller(**workflow_propeller_sector_params)
    if solver == 'sonics': # FIXME BUG https://gitlab.onera.net/numerics/mola/-/work_items/324
        w.ApplicationContext['_BladeRadius'] = 1.0
    w.RunManagement['RunDirectory'] = str(tmp_path)

    w.prepare()
    w.write_cfd_files()
    w.submit()

    w.assert_completed_without_errors()

@pytest.mark.integration
def test_propeller_ultra_light(tmp_path):

    mesh = RW.easy_mesher(dir_check=None)

    slice_fields = [
        'Density',
        'MomentumX',
        'MomentumY',
        'MomentumZ',
        'Viscosity_EddyMolecularRatio',
        'Mach']

    w = WorkflowPropeller(

            ApplicationContext=dict(
                NumberOfBlades=8,
                ShaftRotationSpeed = -2000.0,
            ),

            Flow=dict(
                Density = 1.0,
                Temperature = 300.0,
                Velocity = 100.0,
            ),

            Turbulence = dict(
                Model = 'SA',
            ),

            Numerics = dict(
                Scheme='Roe',
                NumberOfIterations=150,
                MinimumNumberOfIterations=3,
                # CFL=dict(EndIteration=1000, StartValue=1.0, EndValue=30.0),
            ),

            BoundaryConditions=[
                dict(Family='BLADE',   Type='WallViscous'),
                dict(Family='SPINNER',    Type='WallInviscid'),
                dict(Family='FARFIELD', Type='Farfield'),
            ],

            Extractions = [
                dict(Type="IsoSurface",
                     IsoSurfaceField="CoordinateX",
                     IsoSurfaceValue=0.5,
                     Fields=slice_fields),
                dict(Type="IsoSurface",
                     IsoSurfaceField="CoordinateZ",
                     IsoSurfaceValue=0.0,
                     Fields=slice_fields),
                dict(Type="IsoSurface",
                     IsoSurfaceField="Radius",
                     IsoSurfaceValue=0.75,
                     Fields=slice_fields),
                dict(Type="BC",
                     Source="BLADE",
                     Fields=[
                         "Pressure",
                         "SkinFriction"
                     ]),
            ],

            ConvergenceCriteria = [
                dict(
                    ExtractionName = 'BLADE',
                    Variable = "std-Thrust",
                    Threshold = 1.0,
                ),
                dict(
                    ExtractionName = 'BLADE',
                    Variable = "Thrust",
                    Threshold = -np.inf, # HINT just for showing Variable progress in coprocess.log
                ),
                dict(
                    ExtractionName = 'BLADE',
                    Variable = "Power",
                    Threshold = -np.inf, # HINT just for showing Variable progress in coprocess.log
                )
            ],

            RawMeshComponents=[
                dict(
                    Name='Base',
                    Source=mesh,
                )
            ],

            RunManagement=dict(
                NumberOfProcessors=1,
                NumberOfThreads=4,
                RunDirectory=str(tmp_path),
                Scheduler = 'local',
                ),
        )

    w.ApplicationContext['_BladeRadius'] = 1.0 # FIXME BUG https://gitlab.onera.net/numerics/mola/-/work_items/324

    w.prepare()
    w.write_cfd_files()
    w.submit()

    w.assert_completed_without_errors()
