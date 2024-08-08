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

from treelab import cgns
from mola.cfd.preprocess.extractions import solver_fast

def build_tree( nb_of_bases=2, nb_of_zones=2 ):
    
    import Converter.PyTree as C
    import Generator.PyTree as G

    npts = 3

    treelist = []
    for j in range(nb_of_bases):
        zones = []
        for i in range(nb_of_zones):
            zone = G.cart((npts*i,npts*j,0),(1,1,1),(npts,npts,npts))
            zone[0] = 'zone%d'%(i+nb_of_zones*j)
            zones += [ zone ]
        treelist += ['Base%d'%j, zones[:]]
    t = C.newPyTree(treelist)
    t = cgns.castNode(t)

    return t


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
    a1 = C.addState(a1, 'GoverningEquations', 'NSLaminar')
    a1 = C.addState(a1, MInf=0.4)
    t = C.newPyTree(['Base', a1])
    C._tagWithFamily(t,'BLADE')
    C._addFamily2Base(t, 'BLADE', bndType='BCWall')

    # Numerics
    numb = { 'temporal_scheme':'implicit' }; numz = { 'scheme':'ausmpred' }
    FastC._setNum2Zones(t, numz); FastC._setNum2Base(t, numb)

    # Prim vars, solver tag, compact, metric
    (t, tc, metrics) = FastS.warmup(t, None)

    # BUG compute provokes segfault when testing if selected>1 (and not isolated, so weird)
    # for nitrun in range(1): FastS._compute(t, metrics, nitrun)

    teff = FastS.createStressNodes(t, ['BLADE'])
    effort = FastS._computeStress(t, teff, metrics)

    assert len(effort) == 11

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_add_convergence_history():

    class FakeWorkflow():
        def __init__(self):
            self.tree = build_tree()

    workflow = FakeWorkflow()
    solver_fast.add_convergence_history(workflow)



if __name__ == '__main__':
    # test_stress_example()
    test_add_convergence_history()