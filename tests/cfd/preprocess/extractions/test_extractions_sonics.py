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
from mola.cfd.preprocess.extractions import solver_sonics
import mola.naming_conventions as names
from mola.workflow import Workflow

pytestmark = pytest.mark.sonics


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_add_reference_values_for_force_and_torque_extraction():


    class FakeWorkflow():
        def __init__(self):
            
            t = cgns.Tree()
            
            b = cgns.Base()
            b.attachTo(t)
            
            f = cgns.Node(Name='MyFamily', Value='BCWallViscous', Type='FamilyBC_t')
            f.attachTo(b)


            self.tree = t
            self.Flow = dict(Pressure=101325.0)

    w = FakeWorkflow()
    solver_sonics.add_reference_values_for_force_and_torque_extraction(w)
    
    assert w.tree.get(Name='ReferencePressure', Value=101325.0, Type='DataArray_t')


if __name__ == '__main__':
    test_add_reference_values_for_force_and_torque_extraction()