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
from mola.workflow import WorkflowRotatingComponent
from mola.cfd.preprocess.motion import motion
from mola.logging import mola_logger, MolaAssertionError


import maia.pytree as PT

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
        # PT.print_tree(self.tree)

        self.BoundaryConditions = [
            dict(Family='fake_shroud', type='Farfield')
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
def test_extendListOfFamilies():
    families = ['hub']
    extended_families = WorkflowRotatingComponent._extendListOfFamilies(families)
    assert extended_families == ['hub', 'HUB', 'Hub']

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_shroud_boundary_conditions():
    w = FakeWorkflow()
    w.set_shroud_boundary_conditions()
    assert dict(Family='Shroud', type='Wall') in w.BoundaryConditions
    # Check fake_shroud has not been modified
    assert not dict(Family='fake_shroud', type='Wall') in w.BoundaryConditions
    assert dict(Family='fake_shroud', type='Farfield') in w.BoundaryConditions


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_blade_boundary_conditions():
    w = FakeWorkflow()
    w.set_blade_boundary_conditions()
    assert dict(Family='test_Blade1', type='Wall', Motion=w.Motion['Rotor']) in w.BoundaryConditions

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_hub_boundary_conditions_default():
    w = FakeWorkflow()
    w.set_hub_boundary_conditions()
    assert dict(Family='Hub_test', type='Wall', Motion=w.Motion['Rotor']) in w.BoundaryConditions

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_hub_boundary_conditions_list():
    w = FakeWorkflow()
    w.ApplicationContext['HubRotationSpeed'] = [(1, 2)]
    w.set_hub_boundary_conditions()
    last_bc = w.BoundaryConditions[1]
    assert last_bc['Family'] == 'Hub_test'
    assert last_bc['type'] == 'Wall'
    assert callable(last_bc['Motion']['RotationSpeed'])

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_hub_boundary_conditions_function():
    w = FakeWorkflow()
    w.ApplicationContext['HubRotationSpeed'] = lambda x: 2*x
    w.set_hub_boundary_conditions()
    last_bc = w.BoundaryConditions[1]
    assert last_bc['Family'] == 'Hub_test'
    assert last_bc['type'] == 'Wall'
    assert callable(last_bc['Motion']['RotationSpeed'])

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_hub_boundary_conditions_error_axis():
    w = FakeWorkflow()
    w.ApplicationContext['HubRotationSpeed'] = [(1, 2)]
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

