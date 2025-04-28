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

import numpy as np
from treelab import cgns
from mola.cfd.preprocess.finalization import solver_elsa

import pytest
pytestmark = pytest.mark.elsa


class FakeWorkflow():
    def __init__(self, t):
        self.tree = t
        self.SolverParameters = dict(
            model = dict(
                param_model = 1
            ),
            numerics = dict(
                param_num = 5.,
                param_num2 = 'active',
            ),
        )

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_add_elsa_keys_to_cgns():
    t = cgns.Tree()
    base = cgns.Base(Name='Base1', Parent=t)
    sc = cgns.Node( Name='.Solver#Compute', Type='UserDefinedData', Parent=base)
    cgns.Node( Name='data', Value=10, Parent=sc)
    base = cgns.Base(Name='Base2', Parent=t)
    sc = cgns.Node( Name='.Solver#Compute', Type='UserDefinedData', Parent=base)
    cgns.Node( Name='data', Value=10, Parent=sc)

    workflow = FakeWorkflow(t)
    solver_elsa.add_elsa_keys_to_cgns(workflow)

    ref_sc = ['.Solver#Compute', None, [
            ['param_model', np.array([1], dtype=np.int32), [], 'DataArray_t'], 
            ['param_num', np.array([5.]), [], 'DataArray_t'], 
            ['param_num2', np.array([b'a', b'c', b't', b'i', b'v', b'e'], dtype='|S1'), [], 'DataArray_t']
        ], 'UserDefinedData_t']

    for base in workflow.tree.bases():
        sc = base.get(Name='.Solver#Compute', Type='UserDefinedData')
        assert sc is not None
        assert str(sc) == str(ref_sc)
    