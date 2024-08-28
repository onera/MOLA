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

from mola.cfd import apply_to_solver

def apply(workflow):

    add_residuals_extraction(workflow)
    process_extractions_2d(workflow)
    apply_to_solver(workflow)
    
def add_residuals_extraction(workflow):
    if not any([ext['Type'] == 'Residuals' for ext in workflow.Extractions]):
        workflow._interface.add_to_Extractions_Residuals()

def process_extractions_2d(workflow):
    for Extraction in workflow.Extractions:
        if Extraction['Type'] == 'BC':
            Extraction.setdefault('Fields', [])
            if isinstance(Extraction['Fields'], str):
                # NOTE Despite the check of the interface, Fields may be a str
                # when workflow.cgns is read directly, in the context of WorkflowManager
                Extraction['Fields'] = [Extraction['Fields']]

def get_familiesBC_nodes(workflow):

    families = workflow.tree.group(Type='Family', Depth=2)
    familiesBC = []
    for family in families:
        familyBC = family.get(Type='FamilyBC', Depth=1)
        if familyBC:
            familiesBC += [ familyBC ]

    return familiesBC
