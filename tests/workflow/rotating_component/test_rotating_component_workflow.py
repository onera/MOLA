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
from mola.logging import mola_logger, MolaAssertionError
from mola.workflow.rotating_component.workflow import WorkflowRotatingComponent
import maia.pytree as PT

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

        tree = PT.yaml.parse_yaml_cgns.to_cgns_tree('''
Base CGNSBase_t:
    Shroud Family_t:
    test_Blade1 Family_t:
    Hub_test Family_t: 
    fake_shroud Family_t:    
        FamilyBC FamilyBC_t "BCFarfield":                                            
    Zone Zone_t:
        FamilyName FamilyName_t "Rotor":
        ZoneBC ZoneBC_t:
            blade BC_t:
                FamilyName FamilyName_t "test_Blade1":  
            hub BC_t:
                FamilyName FamilyName_t "Hub_test":  
            fake_shroud BC_t:
                FamilyName FamilyName_t "fake_shroud":                                                                                               
''')
        self.tree = cgns.castNode(tree)
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

if __name__ == "__main__":
    test_duplicate('.')