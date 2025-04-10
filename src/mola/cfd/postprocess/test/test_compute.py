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
from mola.cfd.postprocess.compute import compute_radius

@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize('axis', [None, 'x', 'y', 'z'])
def test_compute_radius(axis):
    N = 4 # number of cells in each direction
    a = 1
    x, y, z = np.meshgrid( np.linspace(0,a,N+1),
                           np.linspace(0,a,N+1),
                           np.linspace(0,a,N+1), 
                           indexing='ij')
    mesh = cgns.newZoneFromArrays( 'block', ['x','y','z'],
                                            [ x,  y,  z ])
    mesh = cgns.add(mesh)

    compute_radius(mesh, axis=axis, fieldname='r')
    mesh = cgns.castNode(mesh)

    r = mesh.get(Name='r').value()
    if axis is None:
        assert r[0,0,0] == 0
        assert r[N,0,0] == a
        assert r[0,N,0] == a
        assert r[0,0,N] == a
        assert r[N,N,0] == np.sqrt(2*a) 
        assert r[N,N,N] == np.sqrt(3*a) 
        assert r[N//2,N//2,N//2] == np.sqrt(3*a)/2 
    elif axis == 'x':
        assert r[0,0,0] == 0
        assert r[N,0,0] == 0
        assert r[0,N,0] == a
        assert r[0,0,N] == a
        assert r[N,N,0] == a 
        assert r[N,N,N] == np.sqrt(2*a) 
        assert r[N//2,N//2,N//2] == np.sqrt(2*a)/2 
    elif axis == 'y':
        assert r[0,0,0] == 0
        assert r[N,0,0] == a
        assert r[0,N,0] == 0
        assert r[0,0,N] == a
        assert r[N,N,0] == a 
        assert r[N,N,N] == np.sqrt(2*a) 
        assert r[N//2,N//2,N//2] == np.sqrt(2*a)/2 
    elif axis == 'z':
        assert r[0,0,0] == 0
        assert r[N,0,0] == a
        assert r[0,N,0] == a
        assert r[0,0,N] == 0
        assert r[N,N,0] == np.sqrt(2*a) 
        assert r[N,N,N] == np.sqrt(2*a) 
        assert r[N//2,N//2,N//2] == np.sqrt(2*a)/2 

