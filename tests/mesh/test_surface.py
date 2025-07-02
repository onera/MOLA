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
import pytest
import os


@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_0
def test_getDistribution_and_copyDistribution():

    import mola.mesh.surface as GSD
    import mola.pytree.InternalShortcuts as J
    import Generator.PyTree as G
    import Geom.PyTree as D

    Ni, Nj = 5, 7
    xmax = 1.5
    ymax = 3.5
    bottom = D.line((0,0,0),(1,0,0),Ni)
    left = D.line((0,0,0),(0,1,0),Nj)
    top = D.line((0,1,0),(xmax,ymax,0),Ni)
    right = D.line((1,0,0),(xmax,ymax,0),Nj)

    surface = G.TFI([left, right, bottom, top])
    x, y, z = J.getxyz(surface)
    distribution = GSD.getDistribution(surface)
    new_identical_surface = GSD.copyDistribution(surface, distribution)
    # new_identical_surface = G.map(surface, distribution) # not working accurately
    xn, yn, zn = J.getxyz(new_identical_surface)

    Δ = np.sqrt((xn-x)**2+(yn-y)**2+(zn-z)**2)
    mismatch = Δ > 1e-8
    assert not mismatch.any()


if __name__ == '__main__':
    test_getDistribution_and_copyDistribution()