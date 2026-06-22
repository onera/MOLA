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

from mola.cfd.preprocess.mesh.tools import compute_azimuthal_extension
from mola.cfd.preprocess.motion import motion
from mola.logging import mola_logger, MolaAssertionError, MolaUserError
from mola.workflow.rotating_component.workflow import WorkflowRotatingComponent

from .sector_mesh_tools_for_test import workflow_sector_params, CylinderMeshBuilder, MixedMeshBuilder

@pytest.fixture
def wf_params(workflow_sector_params) -> dict:
    workflow_sector_params['ApplicationContext'] = dict(
        ShaftRotationSpeed=60.0,
        Rows=dict(
            Propeller=dict(
                IsRotating = True,
                NumberOfBladesSimulated = 1,
                NumberOfBladesInInitialMesh = 1,
                NumberOfBlades = 1,
            )
        )
    )
    
    return workflow_sector_params


@pytest.fixture
def tree_with_perio_match() -> cgns.Tree:
    t = cgns.Tree()
    b = cgns.Base()
    z = cgns.Zone()
    z.attachTo(b)
    b.attachTo(t)

    zgc = cgns.Node(Name='ZoneGridConnectivity', Type='ZoneGridConnectivity_t', Parent=z)

    for i in range(2):
        gc = cgns.Node(Name=f'MyPerio{i+1}', Type='GridConnectivity_t', Value="Base/Zone", Parent=zgc)
        prop = cgns.Node(Name='GridConnectivityProperty', Type='GridConnectivityProperty_t', Parent=gc)
        perio = cgns.Node(Name='Periodic', Type='Periodic_t', Parent=prop)

        cgns.Node(Name='RotationAngle',Type='DataArray_t',Value=np.array([np.pi,0,0],dtype=float),Parent=perio)
        cgns.Node(Name='RotationCenter',Type='DataArray_t',Value=np.array([0,0,0],dtype=float),Parent=perio)

    return t

def get_workflow_annular_sector_parameters(RunDirectory):

    params = dict( 
        RawMeshComponents=[
        dict(
            Name='annularSector',
            Source='/stck/mola/data/open/mesh/annular_sector_45deg/annular_sector_45deg.cgns',
            Connection = [
                    dict(Type='PeriodicMatch', 
                         RotationAngle=np.array([45., 0., 0.]), 
                         Families=('PER1', 'PER2'),
                         Tolerance=1e-8
                         ),
                ],
            )
    ],

    ApplicationContext = dict(
        ShaftRotationSpeed = 0., 
        Rows = dict(
            Fluid = dict(NumberOfBlades=8), 
        )
    ),

    Flow = dict(
        Velocity = 10.,      
    ),

    # Turbulence = dict(
    #     Model='SA',
    # ),

    Numerics = dict(
        NumberOfIterations = 5,
        # CFL = dict(EndIteration=300, StartValue=1., EndValue=30.),
    ),

    BoundaryConditions = [
        dict(Family='Inflow', Type='InflowStagnation'),
        dict(Family='Outflow', Type='OutflowPressure'), 
    ],

    RunManagement=dict(
        JobName='annular_sector',
        NumberOfProcessors=4,
        RunDirectory=RunDirectory,
        ),
    )
    return params

def get_workflow_annular_sector(RunDirectory):
    return WorkflowRotatingComponent(**get_workflow_annular_sector_parameters(RunDirectory))

class FakeWorkflow(WorkflowRotatingComponent):

    def __init__(self):

        self.tree = cgns.Tree()
        cgns_version = self.tree.get("CGNSLibraryVersion",Depth=1)
        cgns_version.setValue(np.array([4.2],dtype=np.float32))

        base = cgns.Base(Name="Base", Value=np.array([1,3],dtype=np.int32,order='F'), Parent=self.tree)
        cgns.Node(Parent=base,Name="Shroud",Type='Family_t')
        cgns.Node(Parent=base,Name="test_Blade1",Type='Family_t')
        cgns.Node(Parent=base,Name="Hub_test",Type='Family_t')
        fs = cgns.Node(Parent=base,Name="fake_shroud",Type='Family_t')
        cgns.Node(Parent=fs,Name="FamilyBC",Type='FamilyBC_t',Value="BCFarfield")

        z = cgns.Zone(Name="Zone", Value=np.array([[2,1,0]],dtype=np.int32,order='F'), Parent=base)
        cgns.Node(Name="FamilyName",Value="Rotor",Type="FamilyName_t",Parent=z)
        zbc = cgns.Node(Name="ZoneBC",Type="ZoneBC_t",Parent=z)
        cgns.Node(Parent=zbc, Name="blade", Type="BC_t", Children=[
            cgns.Node(Name='FamilyName', Type='FamilyName_t', Value='test_Blade1')])
        cgns.Node(Parent=zbc, Name="hub", Type="BC_t", Children=[
            cgns.Node(Name='FamilyName', Type='FamilyName_t', Value='Hub_test')])        
        cgns.Node(Parent=zbc, Name="fake_shroud", Type="BC_t", Children=[
            cgns.Node(Name='FamilyName', Type='FamilyName_t', Value='fake_shroud')])

        self._hub_patterns = ['hub', 'moyeu', 'spinner']
        self._blade_patterns = ['blade', 'aube', 'propeller', 'rotor', 'stator']
        self._shroud_patterns = ['shroud', 'carter']


        self.BoundaryConditions = [
            dict(Family='fake_shroud', Type='Farfield')
            ]
        
        self.Motion = dict(
            Rotor = motion.update_motion_with_defaults(dict(RotationSpeed=100.)),
        )
        self.ApplicationContext = dict(
            ShaftAxis = [1., 0., 0.],
            Rows = dict(
                Rotor = dict(
                    IsRotating = True,
                    NumberOfBlades = 16,
                    NumberOfBladesSimulated = 1,
                )
            ),
        )


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_init(tmp_path):
    w = get_workflow_annular_sector(tmp_path)
    w.print_interface()
    assert w.Name == 'WorkflowRotatingComponent'

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_shroud_boundary_conditions():
    w = FakeWorkflow()
    w.set_shroud_boundary_conditions()
    
    # note the "assert not":
    # did not modified fake_shroud since bc was already defined in FakeWorkflow.
    # In my opinion (LB) this shall be marked CAVEAT or FIXME since appropriate
    # behavior should have been MOLA parameters overriding existing ones in mesh
    assert not dict(Family='fake_shroud', Type='Wall', Motion=dict(RotationSpeed=[0.0, 0.0, 0.0])) in w.BoundaryConditions



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_blade_boundary_conditions():
    w = FakeWorkflow()
    w.set_blade_boundary_conditions()
    assert dict(Family='test_Blade1', Type='Wall', Motion=w.Motion['Rotor']) in w.BoundaryConditions

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_hub_boundary_conditions_default():
    w = FakeWorkflow()
    w.set_hub_boundary_conditions()
    assert dict(Family='Hub_test', Type='Wall', Motion=w.Motion['Rotor']) in w.BoundaryConditions

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_hub_boundary_conditions_list():
    w = FakeWorkflow()
    w.ApplicationContext['HubRotationIntervals'] = [(1, 2)]
    w.set_hub_boundary_conditions()
    last_bc = w.BoundaryConditions[1]
    assert last_bc['Family'] == 'Hub_test'
    assert last_bc['Type'] == 'Wall'
    assert callable(last_bc['Motion']['RotationSpeed'])

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_hub_boundary_conditions_function():
    w = FakeWorkflow()
    w.ApplicationContext['HubRotationIntervals'] = lambda x: 2*x
    w.set_hub_boundary_conditions()
    last_bc = w.BoundaryConditions[1]
    assert last_bc['Family'] == 'Hub_test'
    assert last_bc['Type'] == 'Wall'
    assert callable(last_bc['Motion']['RotationSpeed'])

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_hub_boundary_conditions_error_axis():
    w = FakeWorkflow()
    w.ApplicationContext['HubRotationIntervals'] = [(1, 2)]
    w.ApplicationContext['ShaftAxis'] = [-1, 0, 0]
    try:
        w.set_hub_boundary_conditions()
        assert False
    except MolaAssertionError:
        return
    
@pytest.mark.unit
@pytest.mark.cost_level_0
def test_compute_fluxcoef_by_row():
    w = FakeWorkflow()
    w.compute_fluxcoef_by_row()
    assert w.ApplicationContext['NormalizationCoefficient'] == dict(
        fake_shroud = dict(FluxCoef=16.0)
        )

# @pytest.mark.unit
# @pytest.mark.cost_level_3
# def test_parametrize_with_height_with_turbo(tmp_path):
#     w = get_workflow_annular_sector(tmp_path)
#     w.assemble()
#     try:
#         w.parametrize_with_height_with_turbo()
#         assert w.tree.get(Name='FlowSolution#Height', Type='FlowSolution')
#     except ImportError:
#         mola_logger.warning('turbo module cannot be found!')
#         pass

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_parametrize_with_height(tmp_path):
    w = get_workflow_annular_sector(tmp_path)
    w.assemble()
    w.Initialization['ParametrizeWithHeight'] = 'maia'
    w.parametrize_with_height()
    assert w.tree.get(Name='FlowSolution#Height', Type='FlowSolution')

@pytest.mark.unit
@pytest.mark.elsa
@pytest.mark.sonics  # not available with fast due to the incompatibility between duplication and splitting with Cassiopee
@pytest.mark.cost_level_2
def test_duplicate(tmp_path):
    params = get_workflow_annular_sector_parameters(tmp_path)
    params['ApplicationContext']['Rows']['Fluid']['NumberOfBladesSimulated'] = 2
    w = WorkflowRotatingComponent(**params)
    w.SplittingAndDistribution['Strategy'] = 'AtComputation'
    w.process_mesh()

    alpha = np.degrees(compute_azimuthal_extension(w.tree, 'Fluid'))
    assert np.isclose(alpha, 90, rtol=1e-2), f'alpha = {alpha} degrees instead of 90'

@pytest.mark.integration
@pytest.mark.cost_level_1
def test_radius_slice_extraction(tmp_path, wf_params):
    wf_params['Extractions'] = [
        dict(Type="IsoSurface", IsoSurfaceField="Radius", IsoSurfaceValue=0.5,
             Fields=['MomentumX','MomentumY','MomentumZ'])]
    
    w = WorkflowRotatingComponent(**wf_params)
    w.RunManagement['RunDirectory'] = str(tmp_path)

    w.prepare()
    w.write_cfd_files()
    w.submit()

    w.assert_completed_without_errors()


@pytest.mark.integration
@pytest.mark.cost_level_4
@pytest.mark.elsa
def test_rna_hyb(tmp_path):

    rotor = CylinderMeshBuilder(
        zones_family_name="ROTOR",
        hub_family_name="ROTOR_HUB",
        blade_family_name="ROTOR_BLADE",
        interface_boundary = 'imax',
        ).get_tree()

    stator = CylinderMeshBuilder(
        zones_family_name="STATOR",
        hub_family_name="STATOR_HUB",
        blade_family_name="STATOR_BLADE",
        interface_boundary = 'imin',
        ).get_tree()
    
    for i, zone in enumerate(stator.zones()):
        x = zone.x()
        x += 1.0

    mesh = rotor
    mesh_base = rotor.bases()[0]
    stator_base = stator.bases()[0]
    rotor_base_child_names = mesh_base.getChildrenNames()
    for node in stator_base.children():
        if node.name() not in rotor_base_child_names:
            node.attachTo(mesh_base)
    
    # mesh.save(tmp_path/'mesh.cgns')

    w = WorkflowRotatingComponent(
        RawMeshComponents=[
            dict(
                Name='Base',
                Source=mesh,
            )
        ],

        ApplicationContext = dict(
            ShaftRotationSpeed = 60.0,
            ShaftRotationSpeedUnit = 'rpm',

            Rows = dict(ROTOR  = dict(IsRotating = True,  NumberOfBlades = 12, NumberOfBladesSimulated = 1, NumberOfBladesInInitialMesh = 1),
                        STATOR = dict(IsRotating = False, NumberOfBlades = 12, NumberOfBladesSimulated = 1, NumberOfBladesInInitialMesh = 1)
            )
        ),
        Flow=dict(
            Density = 1.0,
            Temperature = 300.0,
            Velocity = 10.0,
        ),

        Numerics = dict(
            NumberOfIterations=4,
            MinimumNumberOfIterations=3,
            TimeMarching="unsteady",
            TimeStep = 1/360,
        ),

        BoundaryConditions=[

            dict(Family='ROTOR_HUB',   Type='WallInviscid'),
            dict(Family='ROTOR_BLADE',    Type='WallViscous'),

            dict(Family='STATOR_HUB',   Type='WallInviscid'),
            dict(Family='STATOR_BLADE',    Type='WallViscous'),

            dict(Type='UnsteadyRotorStatorInterface', Family='ROTOR_INTERFACE',
                                                LinkedFamily='STATOR_INTERFACE'),

            dict(Family='FARFIELD', Type='Farfield'),
        ],

        RunManagement=dict(
            NumberOfProcessors=1,
            RunDirectory=tmp_path,
            Scheduler = 'local',
            ),
    )

    w.prepare()
    w.write_cfd_files()
    w.submit()

    w.assert_completed_without_errors()

@pytest.mark.integration
@pytest.mark.cost_level_4
@pytest.mark.elsa
def test_mixed(tmp_path):
    mesh = MixedMeshBuilder().get_tree()
    mesh.save(tmp_path/"mesh.cgns")
    ...


@pytest.mark.unit
@pytest.mark.cost_level_1
def test_check_if_user_is_providing_mesh_for_rotation_around_x_axis(tree_with_perio_match):
    assert isinstance(tree_with_perio_match, cgns.Tree)

    w = WorkflowRotatingComponent(

        RawMeshComponents=[
        dict(
            Name='Base',
            Source=tree_with_perio_match,
            Connection = [
                    dict(Type='PeriodicMatch', 
                         RotationAngle=np.array([45., 0., 0.]), 
                         Families=('MyPerio1', 'MyPerio2'),
                         Tolerance=1e-8
                         ),
                ],
            )
        ],


        ApplicationContext = dict(
            ShaftRotationSpeed=60.0,
            Rows=dict(
                Propeller=dict(
                    IsRotating = True,
                    NumberOfBladesSimulated = 1,
                    NumberOfBladesInInitialMesh = 1,
                    NumberOfBlades = 1,
                )
            )
            )
    )
    w.tree = tree_with_perio_match
    w.check_if_user_is_providing_mesh_for_rotation_around_x_axis()


@pytest.mark.unit
@pytest.mark.cost_level_1
def test_validate_rotation_axis_and_center_in_tree_correct(tree_with_perio_match):

    assert isinstance(tree_with_perio_match, cgns.Tree)

    w = WorkflowRotatingComponent(

            ApplicationContext = dict(
                ShaftRotationSpeed=60.0,
                Rows=dict(
                    Propeller=dict(
                        IsRotating = True,
                        NumberOfBladesSimulated = 1,
                        NumberOfBladesInInitialMesh = 1,
                        NumberOfBlades = 1,
                    )
                )
                )
    )
    w.tree = tree_with_perio_match
    w.validate_rotation_axis_and_center_in_tree()



@pytest.mark.unit
@pytest.mark.cost_level_1
def test_validate_rotation_axis_and_center_in_tree_incorrect_axis(tree_with_perio_match):

    angles = tree_with_perio_match.get("RotationAngle").value()
    angles[-1] = 1.0

    assert isinstance(tree_with_perio_match, cgns.Tree)

    w = WorkflowRotatingComponent(

            ApplicationContext = dict(
                ShaftRotationSpeed=60.0,
                Rows=dict(
                    Propeller=dict(
                        IsRotating = True,
                        NumberOfBladesSimulated = 1,
                        NumberOfBladesInInitialMesh = 1,
                        NumberOfBlades = 1,
                    )
                )
                )
    )
    w.tree = tree_with_perio_match

    with pytest.raises(MolaUserError):
        w.validate_rotation_axis_and_center_in_tree()

@pytest.mark.unit
@pytest.mark.cost_level_1
def test_validate_rotation_axis_and_center_in_tree_incorrect_center(tree_with_perio_match):

    center = tree_with_perio_match.get("RotationCenter").value()
    center[1] = 1.0

    assert isinstance(tree_with_perio_match, cgns.Tree)

    w = WorkflowRotatingComponent(

            ApplicationContext = dict(
                ShaftRotationSpeed=60.0,
                Rows=dict(
                    Propeller=dict(
                        IsRotating = True,
                        NumberOfBladesSimulated = 1,
                        NumberOfBladesInInitialMesh = 1,
                        NumberOfBlades = 1,
                    )
                )
                )
    )
    w.tree = tree_with_perio_match

    with pytest.raises(MolaUserError):
        w.validate_rotation_axis_and_center_in_tree()


@pytest.mark.unit
@pytest.mark.cost_level_1
def test_validate_rotation_axis_requested_by_user_correct():

    w = WorkflowRotatingComponent(

        RawMeshComponents=[
        dict(
            Name='Base',
            Source='dummy.cgns',
            Connection = [
                    dict(Type='PeriodicMatch', 
                         RotationAngle=np.array([45., 0., 0.]), 
                         Families=('MyPerio1', 'MyPerio2'),
                         Tolerance=1e-8
                         ),
                ],
            )
        ],

        ApplicationContext = dict(
            ShaftRotationSpeed=60.0,
            Rows=dict(
                Propeller=dict(
                    IsRotating = True,
                    NumberOfBladesSimulated = 1,
                    NumberOfBladesInInitialMesh = 1,
                    NumberOfBlades = 1,
                )
            )
            )
    )

    w.validate_rotation_axis_requested_by_user()


# HINT prefer using:
#
# pytest -k test_duplicate --basetemp=.
#
# than
#
# python test_rotating_component_workflow.py
#
# because they are equivalent but the first will allow relative path imports


# if __name__ == "__main__":
#     test_duplicate('.')

