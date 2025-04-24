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
    import Converter.Internal as I
    import Generator.PyTree as G
    import Fast.PyTree as Fast
    import Initiator.PyTree as Init

    npts = 9
    dx = 0.5
    z = G.cart((0.0,0.0,0.0), (dx,dx,dx), (npts,npts,npts))
    C._addBC2Zone(z, 'WALL', 'FamilySpecified:WALL', 'imin')
    C._fillEmptyBCWith(z, 'FARFIELD', 'FamilySpecified:FARFIELD', dim=3)
    C._addState(z, 'GoverningEquations', 'NSLaminar')
    Init._initConst(z, MInf=0.4, loc='centers')
    C._addState(z, MInf=0.4)
    t = C.newPyTree(['Base', z])
    C._tagWithFamily(t,'FARFIELD')
    C._tagWithFamily(t,'WALL')
    C._addFamily2Base(t, 'FARFIELD', bndType='BCFarfield')
    C._addFamily2Base(t, 'WALL', bndType='BCWall')
    I._addGhostCells(t,t,2,adaptBCs=1,fillCorner=0)

    numb = { 'temporal_scheme': 'implicit', 'ss_iteration':3, 'modulo_verif':1}
    numz = { 'scheme':'roe', 'slope':'minmod',
        'time_step':0.0007,'time_step_nature':'local', 'cfl':4}
    Fast._setNum2Zones(t, numz); Fast._setNum2Base(t, numb)


    import FastS.PyTree as FastS

    t, _, metrics = FastS.warmup(t, None)

    for nitrun in range(2): FastS._compute(t, metrics, nitrun)

    teff = FastS.createStressNodes(t, ['WALL'])
    effort = FastS._computeStress(t, teff, metrics)

    assert len(effort) == 11


if __name__ == '__main__':
    test_stress_example()
