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

from mola.cfd.postprocess.signals import fields_manipulator as fields

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_project_load():
    shape = (10,)
    fx = np.full( shape, 1, dtype=np.float64 )
    fy = np.full( shape, 2, dtype=np.float64 )
    fz = np.full( shape, 3, dtype=np.float64 )

    vector = np.array([3, 2, 1],dtype=np.float64)
    vector /= np.linalg.norm(vector)

    result = fields.project_load(fx,fy,fz,vector)
    expected = np.full(shape, vector.dot(np.array([fx[0], fy[0], fz[0]])))

    assert np.allclose(result,expected)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_forces_from(zone_with_loads):
    arrays = fields.get_forces_from(zone_with_loads, 'FlowSolution')
    assert len(arrays) == 3
    
    number_of_grid_points = zone_with_loads.numberOfPoints()
    for a in arrays:
        assert a.size == number_of_grid_points

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_moments_from(zone_with_loads):
    arrays = fields.get_moments_from(zone_with_loads, 'FlowSolution')
    assert len(arrays) == 3
    
    number_of_grid_points = zone_with_loads.numberOfPoints()
    for a in arrays:
        assert a.size == number_of_grid_points


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_new_coefficients_from(zone_with_loads):
    coefs = fields.new_coefficients_from(zone_with_loads, 'FlowSolution')

    expected_keys = ['CL','CD','CS','CX','CY','CZ','CmL','CmD','CmS','CmX','CmY','CmZ']

    assert len(coefs) == len(expected_keys)

    number_of_grid_points = zone_with_loads.numberOfPoints()

    for key in expected_keys:
        assert coefs[key].size == number_of_grid_points

# --------------------------------- fixtures --------------------------------- #
@pytest.fixture
def zone_with_loads():
    x, y, z = np.meshgrid( np.linspace(0,1,5),
                           np.linspace(0,1,5),
                           np.linspace(0,1,5), 
                           indexing='ij')
    zone = cgns.newZoneFromArrays( 'block', ['x','y','z'],
                                            [ x,  y,  z ])
    field_names = ['ForceX', 'ForceY', 'ForceZ', 'TorqueX', 'TorqueY', 'TorqueZ']
    fx, fy, fz, tx, ty, tz = zone.fields(field_names, BehaviorIfNotFound='create')
    fx[:] = 3
    fy[:] = 2
    fz[:] = 1
    tx[:] = 1
    ty[:] = 2
    tz[:] = 3

    return zone
