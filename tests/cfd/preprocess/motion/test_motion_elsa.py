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
from mola.cfd.preprocess.motion import solver_elsa

pytestmark = pytest.mark.elsa


class FakeWorkflow():

    def __init__(self, Motion):
        self.tree = cgns.Tree()
        base = cgns.Base(Parent=self.tree)
        cgns.Node(Name='Rotor', Type='Family', Parent=base)
        cgns.Node(Name='Stator', Type='Family', Parent=base)
        self.Motion = Motion
        self.RawMeshComponents = []


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_apply_to_solver():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[500., 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[5., 0., 8.],
        )
    )

    workflow = FakeWorkflow(Motion)
    solver_elsa.apply_to_solver(workflow)

    ref_tree = ['Rotor', None, [
                ['.Solver#Motion', None, [
                    ['motion', np.array([b'm', b'o', b'b', b'i', b'l', b'e'], dtype='|S1'), [], 'DataArray_t'], 
                    ['omega', np.array([500.]), [], 'DataArray_t'], 
                    ['axis_pnt_x', np.array([3.]), [], 'DataArray_t'], 
                    ['axis_pnt_y', np.array([2.]), [], 'DataArray_t'], 
                    ['axis_pnt_z', np.array([-1.]), [], 'DataArray_t'], 
                    ['axis_vct_x', np.array([1.]), [], 'DataArray_t'], 
                    ['axis_vct_y', np.array([0.]), [], 'DataArray_t'], 
                    ['axis_vct_z', np.array([0.]), [], 'DataArray_t'],
                    ['transl_vct_x', np.array([0.52999894]), [], 'DataArray_t'], 
                    ['transl_vct_y', np.array([0.]), [], 'DataArray_t'], 
                    ['transl_vct_z', np.array([0.8479983]), [], 'DataArray_t'], 
                    ['transl_speed', np.array([9.43398113]), [], 'DataArray_t']
                ], 'UserDefinedData_t']], 'Family_t']
    
    assert str(workflow.tree.get(Name='Rotor')) == str(ref_tree)

    
@pytest.mark.unit
@pytest.mark.cost_level_0
def test_apply_to_solver_no_motion_for_all_families():
    Motion = dict(
        Stator = dict(
            RotationSpeed=[0., 0., 0.],
            RotationAxisOrigin=[0., 0., 0.],
            TranslationSpeed=[0.,0.,0.],
        )
    )

    workflow = FakeWorkflow(Motion)
    solver_elsa.apply_to_solver(workflow)
    assert str(workflow.tree.get(Name='Stator')) == str(['Stator', None, [], 'Family_t'])

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_apply_to_solver_no_motion_for_one_family():
    # NOTE The node .Solver#Motion must be defined even for fixed zones, 
    # if at least one zone is moving. Otherwise, elsA rises an error like in 
    # the issue https://elsa-e.onera.fr/issues/11050 :
    #   User Error : Block motion parameter must be defined consistently over all the blocks
    
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[100., 0., 0.],
            RotationAxisOrigin=[0., 0., 0.],
            TranslationSpeed=[0.,0.,0.],
        ),
        Stator = dict(
            RotationSpeed=[0., 0., 0.],
            RotationAxisOrigin=[0., 0., 0.],
            TranslationSpeed=[0.,0.,0.],
        ),
    )

    ref_Rotor = ['Rotor', None, [
                ['.Solver#Motion', None, [
                    ['motion', np.array([b'm', b'o', b'b', b'i', b'l', b'e'], dtype='|S1'), [], 'DataArray_t'], 
                    ['omega', np.array([100.]), [], 'DataArray_t'], 
                    ['axis_pnt_x', np.array([0.]), [], 'DataArray_t'], 
                    ['axis_pnt_y', np.array([0.]), [], 'DataArray_t'], 
                    ['axis_pnt_z', np.array([0.]), [], 'DataArray_t'], 
                    ['axis_vct_x', np.array([1.]), [], 'DataArray_t'], 
                    ['axis_vct_y', np.array([0.]), [], 'DataArray_t'], 
                    ['axis_vct_z', np.array([0.]), [], 'DataArray_t'],
                ], 'UserDefinedData_t']], 'Family_t']
    ref_Stator = ['Stator', None, [
                ['.Solver#Motion', None, [
                    ['motion', np.array([b'm', b'o', b'b', b'i', b'l', b'e'], dtype='|S1'), [], 'DataArray_t'], 
                    ['omega', np.array([0.]), [], 'DataArray_t'], 
                    ['axis_pnt_x', np.array([0.]), [], 'DataArray_t'], 
                    ['axis_pnt_y', np.array([0.]), [], 'DataArray_t'], 
                    ['axis_pnt_z', np.array([0.]), [], 'DataArray_t'], 
                    ['axis_vct_x', np.array([1.]), [], 'DataArray_t'], 
                    ['axis_vct_y', np.array([0.]), [], 'DataArray_t'], 
                    ['axis_vct_z', np.array([0.]), [], 'DataArray_t'],
                ], 'UserDefinedData_t']], 'Family_t']

    workflow = FakeWorkflow(Motion)
    solver_elsa.apply_to_solver(workflow)
    assert str(workflow.tree.get(Name='Rotor')) == str(ref_Rotor)
    assert str(workflow.tree.get(Name='Stator')) == str(ref_Stator)


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize("RotationSpeed", [[1.,3.,0.], [1.,0.,4.], [1.,1.,1.]])
def test_apply_to_solver_invalid_axis(RotationSpeed):
    Motion = dict(
        Rotor = dict(
            RotationSpeed=RotationSpeed,
            RotationAxisOrigin=np.empty(3),
            TranslationSpeed=np.empty(3),
        )
    )

    workflow = FakeWorkflow(Motion)

    try: 
        solver_elsa.apply_to_solver(workflow)
    except AssertionError as e:
        assert e.args[0] == 'For elsA, the rotation must be around one axis only'
    else:
        raise AssertionError('apply_to_solver must raise an AssertionError if the rotation is not around one cartesian axis')
