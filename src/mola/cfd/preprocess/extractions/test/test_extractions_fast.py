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
pytestmark = pytest.mark.fast


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_stress_example():
    import Converter.PyTree as C
    import Generator.PyTree as G
    import FastS.PyTree as FastS
    import FastC.PyTree as FastC
    import Initiator.PyTree as I

    ni = 5 ; dx = 100./(ni-1) ; dz = 1.
    a1 = G.cart((-50,-50,0.), (dx,dx,dz), (ni,ni,2))
    a1 = C.fillEmptyBCWith(a1, 'BLADE', 'FamilySpecified:BLADE', dim=2)
    a1 = I.initConst(a1, MInf=0.4, loc='centers')
    a1 = C.addState(a1, 'GoverningEquations', 'Euler')
    a1 = C.addState(a1, MInf=0.4)
    t = C.newPyTree(['Base', a1])
    C._tagWithFamily(t,'BLADE')
    C._addFamily2Base(t, 'BLADE', bndType='BCWall')

    # Numerics
    numb = { 'temporal_scheme':'implicit' }; numz = { 'scheme':'ausmpred' }
    FastC._setNum2Zones(t, numz); FastC._setNum2Base(t, numb)

    # Prim vars, solver tag, compact, metric
    (t, tc, metrics) = FastS.warmup(t, None)

    # Compute
    for nitrun in range(1,3):
        FastS._compute(t, metrics, nitrun)

    teff = FastS.createStressNodes(t, ['BLADE'])
    effort = FastS._computeStress(t, teff, metrics)

    assert len(effort) == 11


if __name__ == '__main__':
    test_stress_procedure()