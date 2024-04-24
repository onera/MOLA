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

from treelab import cgns
from mola.cfd.preprocess.cfd_parameters import solver_elsa
from mola.workflow import Workflow
from mola.logging import check_error_message, MolaException

pytestmark = pytest.mark.elsa
    
@pytest.fixture
def tree_struct():
    tree = cgns.Tree()
    base = cgns.Base(Parent=tree)
    zone = cgns.Zone(Parent=base)
    zone.get('ZoneType').setValue('Structured')
    return tree

@pytest.fixture
def tree_unstruct():
    tree = cgns.Tree()
    base = cgns.Base(Parent=tree)
    zone = cgns.Zone(Parent=base)
    zone.get('ZoneType').setValue('Unstructured')
    return tree

@pytest.fixture
def tree_hybrid():
    tree = cgns.Tree()
    base = cgns.Base(Parent=tree)
    zoneS = cgns.Zone(Name='ZoneS', Parent=base)
    zoneS.get('ZoneType').setValue('Structured')
    zoneU = cgns.Zone(Name='ZoneU', Parent=base)
    zoneU.get('ZoneType').setValue('Unstructured')
    return tree


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_wall_distance_setup_struct(tree_struct):
    assert solver_elsa.get_wall_distance_setup(tree_struct) == dict(walldistcompute='mininterf_ortho')

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_wall_distance_setup_unstruct(tree_unstruct):
    assert solver_elsa.get_wall_distance_setup(tree_unstruct) == dict(walldistcompute='mininterf')

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_wall_distance_setup_hybrid(tree_hybrid):
    assert solver_elsa.get_wall_distance_setup(tree_hybrid) == dict(walldistcompute='mininterf')

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_turbulence_cutoff_setup():
    # standard case
    Turbulence = dict(
        Conservatives = dict(fake1=1, fake2=2),
        TurbulenceCutOffRatio = 0.1,
    )
    TurbulenceCutOffSetup = solver_elsa.get_turbulence_cutoff_setup(Turbulence)
    assert TurbulenceCutOffSetup == dict(t_cutvar1=0.1, t_cutvar2=0.2)

    # RSM case with 7 variables
    Turbulence = dict(
        Conservatives = dict(fake1=1, fake2=2, fake3=3, fake4=4, fake5=5, fake6=6, fake7=7),
        TurbulenceCutOffRatio = 0.01,
    )
    TurbulenceCutOffSetup = solver_elsa.get_turbulence_cutoff_setup(Turbulence)
    assert TurbulenceCutOffSetup == dict(t_cutvar1=0.01, t_cutvar2=0.04, t_cutvar3=0.06, t_cutvar4=0.07)

    # case with more than 4 variables, but not 7 --> error
    Turbulence = dict(
        Conservatives = dict(fake1=1, fake2=2, fake3=3, fake4=4, fake5=5),
    )
    try:
        TurbulenceCutOffSetup = solver_elsa.get_turbulence_cutoff_setup(Turbulence)
        assert False
    except MolaException:
        return

if __name__ == '__main__':
    test_get_turbulence_cutoff_setup()