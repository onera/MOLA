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
from mola import cgns
from mola.cfd.preprocess.motion import solver_elsa

class FakeWorkflow():

    def __init__(self, Motion):
        self.tree = cgns.Tree()
        base = cgns.Base(Parent=self.tree)
        cgns.Node(Name='Rotor', Type='Family', Parent=base)
        self.Motion = Motion


def test_adapt_to_solver():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[500., 0., 0.],
            RotationAxisOrigin=[3., 2., -1.],
            TranslationSpeed=[5., 0., 8.],
        )
    )

    workflow = FakeWorkflow(Motion)
    solver_elsa.adapt_to_solver(workflow)

    ref_tree = ['Rotor', None, [[
                '.Solver#Motion', None, [
                    ['motion', np.array([b'm', b'o', b'b', b'i', b'l', b'e'], dtype='|S1'), [], 'DataArray_t'], 
                    ['omega', np.array([500.]), [], 'DataArray_t'], 
                    ['axis_pnt_x', np.array([3.]), [], 'DataArray_t'], 
                    ['axis_pnt_y', np.array([2.]), [], 'DataArray_t'], 
                    ['axis_pnt_z', np.array([-1.]), [], 'DataArray_t'], 
                    ['axis_vct_x', np.array([5.]), [], 'DataArray_t'], 
                    ['axis_vct_y', np.array([0.]), [], 'DataArray_t'], 
                    ['axis_vct_z', np.array([8.]), [], 'DataArray_t']
                ], 'UserDefinedData_t']], 'Family_t']

    assert str(workflow.tree.get(Type='Family')) == str(ref_tree)

    
def test_adapt_to_solver_no_motion():
    Motion = dict(
        Rotor = dict(
            RotationSpeed=[0., 0., 0.],
            RotationAxisOrigin=np.empty(3),
            TranslationSpeed=[0.,0.,0.],
        )
    )

    workflow = FakeWorkflow(Motion)
    solver_elsa.adapt_to_solver(workflow)

    print(workflow.tree.get(Type='Family'))

    assert str(workflow.tree.get(Type='Family')) == "['Rotor', None, [], 'Family_t']"


@pytest.mark.parametrize("RotationSpeed", [[1.,3.,0.], [1.,0.,4.], [1.,1.,1.]])
def test_adapt_to_solver_error(RotationSpeed):
    Motion = dict(
        Rotor = dict(
            RotationSpeed=RotationSpeed,
            RotationAxisOrigin=np.empty(3),
            TranslationSpeed=np.empty(3),
        )
    )

    workflow = FakeWorkflow(Motion)

    try: 
        solver_elsa.adapt_to_solver(workflow)
    except AssertionError as e:
        assert e.args[0] == 'For elsA, the rotation must be around one axis only'
    else:
        raise AssertionError('adapt_to_solver must raise an AssertionError if the rotation is not around one cartesian axis')
