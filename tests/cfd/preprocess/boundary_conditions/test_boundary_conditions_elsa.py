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

import copy
import numpy as np
from treelab import cgns

from mola.cfd.preprocess.boundary_conditions import solver_elsa
from mola.cfd.preprocess.boundary_conditions.boundary_conditions_dispatcher_elsa import BoundaryConditionsDispatcherElsa
from .test_boundary_conditions import get_workflow_prepared_to_test_bcs

from mola.workflow.rotating_component import turbomachinery
from ....workflow.rotating_component.turbomachinery.test_turbomachinery_workflow import get_compressor_example_parameters

pytestmark = pytest.mark.elsa

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_functions_well_defined():
    bc_dict = BoundaryConditionsDispatcherElsa()
    for expected_function_name in bc_dict.get_all_specific_names():
        assert getattr(solver_elsa, expected_function_name)

@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize('inputs', [
    dict(
        bc_params=dict(PressureAtHub=0.9e5),
        expected_params=dict(valve_ref_pres=0.9e5, indpiv=1, dirorder=-1)        
        ),
    dict(
        bc_params=dict(PressureAtShroud=0.9e5),
        expected_params=dict(valve_ref_pres=0.9e5, indpiv=-1, dirorder=-1)        
        ),
    dict(
        bc_params=dict(MassFlow=2.),
        expected_params=dict(valve_type=2, valve_ref_pres=1e5, valve_ref_mflow=0.2, valve_relax=0.1, indpiv=1, dirorder=-1)        
        ),
    dict(
        bc_params=dict(ValveLaw=dict(Type='Linear', PressureRef=0.8e5, RelaxationCoefficient=0.05)),
        expected_params=dict(valve_type=1, valve_ref_pres=0.8e5, valve_ref_mflow=0.3, valve_relax=0.05, indpiv=1, dirorder=-1)        
        ),
    dict(
        bc_params=dict(ValveLaw=dict(Type='Quadratic', ValveCoefficient=0.1)),
        expected_params=dict(valve_type=4, valve_ref_pres=0.9e5, valve_ref_mflow=0.3, valve_relax=0.12e5, indpiv=1, dirorder=-1)        
        ),
]
)
def test_outradeq_interface(inputs):

    bc_params = inputs['bc_params']
    expected_params = inputs['expected_params']

    from mola.cfd.preprocess.boundary_conditions import boundary_conditions

    # Monkey patching to force fluxcoeff to be equal to 10
    def fake_fun(workflow, Family):
        return 10
    saved_fun = copy.deepcopy(boundary_conditions.get_fluxcoeff_on_bc)
    boundary_conditions.get_fluxcoeff_on_bc = fake_fun

    class FakeWorkflow():

        def __init__(self):
            self.Flow = dict(Pressure=1e5, PressureStagnation=1.2e5, MassFlow=3.)

    workflow = FakeWorkflow()
    Family = 'Outflow'
    boundary_conditions.OutflowRadialEquilibrium_interface(workflow, bc_params)
    params = solver_elsa.outradeq_interface(workflow, Family, **bc_params)
    for key, value in expected_params.items():
        assert params[key] == value      
    if not 'valve_type' in expected_params:
        assert not 'valve_type' in params

    boundary_conditions.get_fluxcoeff_on_bc = saved_fun  

@pytest.mark.unit
@pytest.mark.cost_level_1
def test_bc_generic():
    BoundaryConditions=[
            dict(Family='imin', Type='WallViscous'),
            dict(Family='imax', Type='Farfield'),
            dict(Family='jmin', Type='InflowStagnation'),
            dict(Family='jmax', Type='InflowMassFlow', MassFlow=1.),
            dict(Family='kmin', Type='OutflowPressure'),
            dict(Family='kmax', Type='OutflowMassFlow', MassFlow=1.),
        ]
    workflow = get_workflow_prepared_to_test_bcs(BoundaryConditions)
    workflow.set_boundary_conditions()

@pytest.mark.unit
@pytest.mark.cost_level_1
def test_bc_specific():
    BoundaryConditions=[
            dict(Family='imin', Type='walladia'),
            dict(Family='imax', Type='nref'),
            dict(Family='jmin', Type='inj1'),
            dict(Family='jmax', Type='injmfr1', MassFlow=1.),
            dict(Family='kmin', Type='outpres'),
            dict(Family='kmax', Type='outmfr2', MassFlow=1.),
        ]
    workflow = get_workflow_prepared_to_test_bcs(BoundaryConditions)
    workflow.set_boundary_conditions()

@pytest.mark.unit
@pytest.mark.cost_level_1
def test_bc_2D(tmp_path):

    def _write_outflow_2D_file(filename):
        x, y, z = np.meshgrid( [1],
                            np.linspace(0,1,5),
                            np.linspace(0,1,5), indexing='ij')
        imax_plane = cgns.newZoneFromArrays( 'block', ['x','y','z'], [ x,  y,  z ])
        # imax_plane.newFields(dict(Pressure=3), Container='FlowSolution#Centers', GridLocation='CellCenter')

        pressure = np.array(np.random.rand(1, 4, 4), order='F')
        FS = cgns.Node(Parent=imax_plane, Name='FlowSolution#Centers', Type='FlowSolution')
        cgns.Node(Parent=FS, Name='GridLocation', Type='GridLocation', Value='CellCenter')
        cgns.Node(Parent=FS, Name='Pressure', Type='DataArray', Value=pressure)
        base = cgns.Node(Name='Base', Type='CGNSBase')
        cgns.Node(Parent=base, Name='imax', Type='Family')
        base.addChild(imax_plane)
        cgns.Node(Parent=FS, Name='FamilyName', Type='FamilyName', Value='imax')
        base.save(str(tmp_path/filename))

    _write_outflow_2D_file('outflow_2D.cgns')

    BoundaryConditions = [
            dict(Family='imin', Type='InflowStagnation', PressureStagnation=lambda y: 1+2*y, variableForInterpolation='CoordinateY'),
            dict(Family='imax', Type='OutflowPressure', File=str(tmp_path/'outflow_2D.cgns')),
        ]
    workflow = get_workflow_prepared_to_test_bcs(BoundaryConditions)
    workflow.set_boundary_conditions()

    # bc_inflow = workflow.tree.get(Type='BC', Name='imin')  # after other tests, the name of the BC can be "imin.0", don't know why
    bc_inflow = [bc for bc in workflow.tree.group(Type='BC') if bc.get(Type='FamilyName').value() == 'imin'][0]
    assert bc_inflow.get(Name='PressureStagnation').value().shape == (4,4)
    assert bc_inflow.get(Name='EnthalpyStagnation').value().shape == (4,4)

    # bc_outflow = workflow.tree.get(Type='BC', Name='imax')
    bc_outflow = [bc for bc in workflow.tree.group(Type='BC') if bc.get(Type='FamilyName').value() == 'imax'][0]

    assert bc_outflow.get(Name='Pressure').value().shape == (4,4)

    
@pytest.mark.unit
@pytest.mark.cost_level_1
@pytest.mark.parametrize('interface_type', ['MixingPlane', 'UnsteadyRotorStatorInterface', 'ChorochronicInterface'])
def test_RotorStatorInterface(tmp_path, interface_type):

    params = get_compressor_example_parameters(tmp_path)
    params['BoundaryConditions'] = [
        dict(Family='Rotor_INFLOW', Type='InflowStagnation'),
        dict(Family='Stator_OUTFLOW', Type='OutflowRadialEquilibrium', PressureAtHub=1e5),
        dict(Family='HUB', Type='WallInviscid'),
        dict(Family='SHROUD', Type='WallInviscid'),
        dict(Family='Rotor_stator_10_left', LinkedFamily='Rotor_stator_10_right', Type=interface_type)
    ]

    workflow = turbomachinery.Workflow(**params)

    workflow.prepare_job()
    workflow.assemble()
    workflow.positioning()
    workflow.define_families() 
    workflow.connect()
    workflow.split_and_distribute() 
    workflow.process_overset()
    workflow.compute_flow_and_turbulence()
    workflow.set_motion()

    workflow.set_boundary_conditions()

@pytest.mark.unit
@pytest.mark.cost_level_1
def test_bc_giles(tmp_path):

    params = get_compressor_example_parameters(tmp_path)
    params['BoundaryConditions'] = [
        dict(Family='Rotor_INFLOW', Type='giles_inlet', NumberOfModes=3),
        dict(Family='Stator_OUTFLOW', Type='giles_outlet', PressureAtHub=1e5, NumberOfModes=3),
        dict(Family='HUB', Type='WallInviscid'),
        dict(Family='SHROUD', Type='WallInviscid'),
        dict(Family='Rotor_stator_10_left', LinkedFamily='Rotor_stator_10_right', Type='giles_stage_mxpl', NumberOfModes=3)
    ]

    workflow = turbomachinery.Workflow(**params)

    workflow.prepare_job()
    workflow.assemble()
    workflow.positioning()
    workflow.define_families() 
    workflow.connect()
    workflow.split_and_distribute() 
    workflow.process_overset()
    workflow.compute_flow_and_turbulence()
    workflow.set_motion()

    workflow.set_boundary_conditions()

if __name__ == "__main__":
    test_bc_generic()