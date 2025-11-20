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

import os
import pytest
import numpy as np
from treelab import cgns
import maia.pytree as PT
from mola.logging import MolaException
from mola.cfd.preprocess.check import check
from mola.cfd.preprocess.mesh.families import set_bc_family_from_location

# --------------------------------- fixtures --------------------------------- #
@pytest.fixture
def grid2D():
    x, y, z = np.meshgrid( np.linspace(0,1,5),
                           np.linspace(0,1,5),
                           np.linspace(0,1,2), 
                           indexing='ij')
    zone = cgns.newZoneFromArrays( 'block', ['x','y','z'],
                                            [ x,  y,  z ])
    return zone

@pytest.fixture
def tree2D_with_bc_defined(grid2D):
    t = cgns.Tree(Base=grid2D)

    for location in ['imin', 'imax', 'jmin', 'jmax']:
        set_bc_family_from_location(t.bases()[0], 'FARFIELD',location)

    return t

@pytest.fixture
def tree2D_with_bc_undefined(grid2D):
    t = cgns.Tree(Base=grid2D)

    for location in ['imin', 'imax']:
        set_bc_family_from_location(t.bases()[0], 'FARFIELD',location)

    return t

@pytest.fixture
def grid3D():
    x, y, z = np.meshgrid( np.linspace(0,1,5),
                           np.linspace(0,1,5),
                           np.linspace(0,1,5), 
                           indexing='ij')
    zone = cgns.newZoneFromArrays( 'block', ['x','y','z'],
                                            [ x,  y,  z ])
    return zone

@pytest.fixture
def tree3D_with_bc_defined(grid3D):
    t = cgns.Tree(Base=grid3D)

    for location in ['imin', 'imax', 'jmin', 'jmax', 'kmin', 'kmax']:
        set_bc_family_from_location(t.bases()[0], 'FARFIELD',location)

    return t

@pytest.fixture
def tree3D_with_bc_undefined(grid3D):
    t = cgns.Tree(Base=grid3D)

    for location in ['imin', 'imax', 'jmin', 'jmax']:
        set_bc_family_from_location(t.bases()[0], 'FARFIELD',location)

    return t

@pytest.fixture
def tree_dispatcher(request):
    return request.getfixturevalue(request.param)
all_trees = ["tree2D_with_bc_defined", "tree2D_with_bc_undefined", 
             "tree3D_with_bc_defined", "tree3D_with_bc_undefined"]
# ----------------------------- end of fixtures ----------------------------- #


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize('tree_dispatcher',all_trees, indirect=True)
def testassert_bc_and_connectivity_coherency(tree_dispatcher):
    check.assert_bc_and_connectivity_coherency(tree_dispatcher)


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize('tree_dispatcher',all_trees, indirect=True)
def test_ignore_undefined_periodic_boundaries_in_2D_structured_grids(tree_dispatcher):
    check._ignore_undefined_periodic_boundaries_in_2D_structured_grids(tree_dispatcher)


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.parametrize('tree_dispatcher',["tree2D_with_bc_undefined", "tree3D_with_bc_undefined"], indirect=True)
def test_raise_undefined_bc_error_saving_undefined_bc_surfaces(tree_dispatcher):

    try:
        check._raise_undefined_bc_error_saving_undefined_bc_surfaces(tree_dispatcher,0)

    except MolaException as e:

        if "UNDEFINED BC IN TREE" not in str(e):
            raise MolaException("unexpected error in test") from e
        
        os.unlink('debug_undefined_bc_0.cgns')
        


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_are_point_ranges_overlapping():
    x = np.array([[1, 10], [1, 10],[1,1]])
    y = np.array([[2, 7],  [1, 10],[1,1]])
    z = np.array([[2, 7],  [1, 11],[2,2]])

    assert check.are_point_ranges_overlapping(y, x)
    assert check.are_point_ranges_overlapping(x, y) 
    assert not check.are_point_ranges_overlapping(z, x)
    assert not check.are_point_ranges_overlapping(x, z)
    assert not check.are_point_ranges_overlapping(y, z)
    assert not check.are_point_ranges_overlapping(z, y)

    a = np.array([[1,4], [1,1], [3,5]])
    b = np.array([[1,4], [5,5], [3,5]])
    c = np.array([[1,4], [1,1], [3,1]])
    assert not check.are_point_ranges_overlapping(a, b)
    assert not check.are_point_ranges_overlapping(b, a)
    assert not check.are_point_ranges_overlapping(a, c)
    assert not check.are_point_ranges_overlapping(c, a)
    assert not check.are_point_ranges_overlapping(b, c)
    assert not check.are_point_ranges_overlapping(c, b)

    d = np.array([[3,6], [1,1], [2,7]])
    assert check.are_point_ranges_overlapping(a, d)

    e = np.array([[1,71], [1,1], [1, 2]])
    f = np.array([[1,71], [1,41], [1, 1]])
    assert not check.are_point_ranges_overlapping(e, f)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_check_no_empty_Family_of_BC():
    tree = PT.yaml.parse_yaml_cgns.to_cgns_tree('''
Base CGNSBase_t:
    Shroud Family_t:
        FamilyBC FamilyBC_t "BCWall": 
    Blade Family_t:
        FamilyBC FamilyBC_t "BCWall":                               
    Hub Family_t:                                          
    Zone Zone_t:
        FamilyName FamilyName_t "Rotor":
        ZoneBC ZoneBC_t:
            blade BC_t "FamilySpecified":
                FamilyName FamilyName_t "Blade":  
            hub BC_t "FamilySpecified":
                FamilyName FamilyName_t "Hub":  
            shroud BC_t "FamilySpecified":
                FamilyName FamilyName_t "Shroud":                                                                                               
''')
    tree = cgns.castNode(tree)
    with pytest.raises(MolaException):
        check.check_no_empty_Family_of_BC(tree)

    tree.findAndRemoveNode(Type='BC', Name='hub')
    check.check_no_empty_Family_of_BC(tree)


