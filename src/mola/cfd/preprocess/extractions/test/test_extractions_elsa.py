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

import copy
import numpy as np
from mola import cgns
from mola.cfd.preprocess.extractions import solver_elsa

class FakeWorkflow():

    def __init__(self):

        self.Flow = dict(
            Conservatives = dict(Density=1.2,Momentum=10.,Energy=5.)
        )

        self.Extractions = [
            dict(type='fake'),
            dict(type='bc', BCType='BCWallViscous', fields=['Pressure']),
            dict(type='3D', fields=dict(Density=1.2,Momentum=10.,Energy=5.)),
        ]

        self.tree = cgns.Tree()
        base1 = cgns.Base(Name='Base1', Parent=self.tree)
        base2 = cgns.Base(Name='Base2', Parent=self.tree)
        zone = cgns.Zone(Parent=base1)

        self.overset_flag = False
    
    def has_overset_component(self):
        return self.overset_flag




def test_add_extractions_for_overset_components():
    workflow = FakeWorkflow()

    # Check that nothing happend if not overset components
    ref_Extractions = copy.copy(workflow.Extractions)
    solver_elsa.add_extractions_for_overset_components(workflow)
    assert workflow.Extractions == ref_Extractions

    # Check behavior with overset components
    workflow = FakeWorkflow()
    workflow.overset_flag = True
    ref_Extractions.append(
        dict(
            type      = '3D', 
            fields    = dict(Density=1.2,Momentum=10.,Energy=5.), 
            Container = 'FlowSolution#EndOfRun#Relative', 
            Frame     = 'relative'
        )
    )
    solver_elsa.add_extractions_for_overset_components(workflow)
    assert workflow.Extractions == ref_Extractions

    

def test_process_extractions_3d():
    assert False, 'Not implemented yet'

def test_process_extractions_2d():
    assert False, 'Not implemented yet'

def test_add_trigger():
    ref_trigger = ['ELSA_TRIGGER', None, [
        ['.Solver#Trigger', None, [
            ['next_state', np.array([16], dtype=np.int32), [], 'DataArray_t'], 
            ['next_iteration', np.array([1], dtype=np.int32), [], 'DataArray_t'], 
            ['file', np.array([b'c', b'o', b'p', b'r', b'o', b'c', b'e', b's', b's', b'.', b'p', b'y'], dtype='|S1'), [], 'DataArray_t']
        ], 'UserDefinedData_t']], 'Family_t']

    workflow = FakeWorkflow()
    solver_elsa.add_trigger(workflow.tree)

    for zone in workflow.tree.zones():
        assert zone.get(Name='ELSA_TRIGGER', Type='AdditionalFamilyName', Value='ELSA_TRIGGER')

    for base in workflow.tree.bases():
        trigger_fam_node = base.get(Name='ELSA_TRIGGER', Type='Family')
        assert str(trigger_fam_node) == str(ref_trigger)
    

def test_global_convergence_history():
    ref_node = ['GlobalConvergenceHistory', np.array([0], dtype=np.int32), [
            ['NormDefinitions', np.array([b'C', b'o', b'n', b'v', b'e', b'r', b'g', b'e', b'n', b'c', b'e', b'H', b'i', b's', b't', b'o', b'r', b'y'], dtype='|S1'), [], 'Descriptor_t']
            ], 'ConvergenceHistory_t']
    
    workflow = FakeWorkflow()
    solver_elsa.add_global_convergence_history(workflow)
    for base in workflow.tree.bases():
        conv_node = base.get(Name='GlobalConvergenceHistory')       
        assert str(conv_node) == str(ref_node)

    