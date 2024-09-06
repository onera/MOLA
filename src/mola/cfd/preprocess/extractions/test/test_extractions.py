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
from mola.cfd.preprocess.extractions.extractions import get_familiesBC_nodes, get_bc_families_names_to_extract
from mola.workflow.test.test_workflow import get_workflow2
from mola.cfd.preprocess.mesh.io import read

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_familiesBC_nodes():

    workflow = get_workflow2()
    read(workflow)
    workflow.define_families()
    workflow.set_boundary_conditions()
    families = get_familiesBC_nodes(workflow.tree)
    assert {f.value() for f in families} == {'BCWallViscous', 'BCFarfield'}

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_bc_families_names_to_extract():

    workflow = get_workflow2()
    read(workflow)
    workflow.define_families()
    workflow.set_boundary_conditions()

    Extraction = dict(Type='BC', Fields=['Mach'], Source='Ground')
    fam_names = get_bc_families_names_to_extract(workflow.tree, Extraction)
    assert fam_names == ['Ground']

    Extraction = dict(Type='BC', Fields=['Mach'], Source='BCWall*')
    fam_names = get_bc_families_names_to_extract(workflow.tree, Extraction)
    assert fam_names == ['Ground']

    Extraction = dict(Type='BC', Fields=[], Source='Farfield')
    fam_names = get_bc_families_names_to_extract(workflow.tree, Extraction)
    assert fam_names == []

    Extraction = dict(Type='BC', Fields=['Mach'], Source='Fake') 
    fam_names = get_bc_families_names_to_extract(workflow.tree, Extraction)
    assert fam_names == []

