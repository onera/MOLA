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
from mola.cfd.preprocess.extractions.extractions import (       
    get_familiesBC_nodes, 
    get_bc_families_names_to_extract, 
    replace_shortcuts,
    update_extractions_from_convergence_criteria,
)
from ....workflow.test_workflow import  get_workflow2
from mola.cfd.preprocess.mesh.io import read
from mola.logging.exceptions import MolaUserError

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_familiesBC_nodes():

    workflow = get_workflow2()
    read(workflow)
    workflow.define_families()
    workflow.set_boundary_conditions()
    families = get_familiesBC_nodes(workflow.tree)
    assert len(families) == 2

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_bc_families_names_to_extract():

    workflow = get_workflow2()
    read(workflow)
    workflow.define_families()
    workflow.set_boundary_conditions()

    Extraction = dict(Type='BC', Fields=['Mach'], Source='Ground')
    fam_names = get_bc_families_names_to_extract(workflow, Extraction)
    assert fam_names == ['Ground']

    Extraction = dict(Type='BC', Fields=['Mach'], Source='Wall*')
    fam_names = get_bc_families_names_to_extract(workflow, Extraction)
    assert fam_names == ['Ground']

    Extraction = dict(Type='BC', Fields=[], Source='Farfield')
    try:
        fam_names = get_bc_families_names_to_extract(workflow, Extraction)
    except MolaUserError:
        pass

    Extraction = dict(Type='BC', Fields=['Mach'], Source='Fake')
    try:
        fam_names = get_bc_families_names_to_extract(workflow, Extraction)
    except MolaUserError:
        pass

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_replace_shortcuts():
    class FakeWorkflow:
        def __init__(self):
            self.Extractions = [dict(Fields=['var1', 'Conservatives', 'var2']), dict(), dict(Fields=['var0'])]
            self.Flow = dict(Conservatives = ['cons1', 'cons2'])
    
    workflow = FakeWorkflow()
    replace_shortcuts(workflow)

    assert workflow.Extractions == [dict(Fields=['var1', 'var2', 'cons1', 'cons2']), dict(), dict(Fields=['var0'])]

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_update_extractions_from_convergence_criteria_1():

    class FakeWorkflow:
        def __init__(self):
            self.Extractions = [
                dict(Type='Integral', Source='my_source', Fields=['Pressure']), 
                ]
            self.ConvergenceCriteria = [
                dict(ExtractionName='my_source', Variable='rsd-avg-MassFlow', Threshold=1e-3)
            ]
    
    workflow = FakeWorkflow()
    update_extractions_from_convergence_criteria(workflow)

    assert workflow.Extractions[0]['Fields'] == ['Pressure', 'MassFlow'] 
    assert workflow.Extractions[0]['PostprocessOperations'] == [
        dict(Type='avg', Variable='MassFlow', AtEndOfRunOnly=False),
        dict(Type='rsd', Variable='avg-MassFlow', AtEndOfRunOnly=False)
    ] 


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_update_extractions_from_convergence_criteria_2():

    class FakeWorkflow:
        def __init__(self):
            self.Extractions = [
                dict(Type='Integral', Source='my_source', Fields=['Pressure']), 
                ]
            self.ConvergenceCriteria = [
                dict(ExtractionName='my_source', Variable='avg-MassFlow', Threshold=1e-3),
                dict(ExtractionName='my_source', Variable='rsd-avg-MassFlow', Threshold=1e-3)
            ]
    
    workflow = FakeWorkflow()
    update_extractions_from_convergence_criteria(workflow)

    assert workflow.Extractions[0]['Fields'] == ['Pressure', 'MassFlow'] 
    assert workflow.Extractions[0]['PostprocessOperations'] == [
        dict(Type='avg', Variable='MassFlow', AtEndOfRunOnly=False),
        dict(Type='rsd', Variable='avg-MassFlow', AtEndOfRunOnly=False)
    ] 

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_update_extractions_from_convergence_criteria_3():
    # Test with a syntax error on the Variable name
    class FakeWorkflow:
        def __init__(self):
            self.Extractions = [
                dict(Type='Integral', Source='my_source', Fields=['Pressure']), 
                ]
            self.ConvergenceCriteria = [
                dict(ExtractionName='my_source', Variable='rsd-av-MassFlow', Threshold=1e-3)
            ]
    
    workflow = FakeWorkflow()
    with pytest.raises(MolaUserError):
        update_extractions_from_convergence_criteria(workflow)
