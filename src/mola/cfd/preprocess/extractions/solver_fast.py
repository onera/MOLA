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

from treelab import cgns

def apply_to_solver(workflow):

    workflow._interface.add_to_Extractions_Restart(
        Container='FlowSolution#Centers'
        )

    for Extraction in workflow.Extractions: 
        if Extraction['Type'] == 'Residuals':
            add_convergence_history(workflow, Extraction['ExtractionPeriod'])
            Extraction['ExtractionPeriod'] = Extraction['SavePeriod']
        elif Extraction['Type'] == 'Integral':
            Extraction['ExtractionPeriod'] = Extraction['SavePeriod']



def add_convergence_history(worfklow, ExtactionPeriod=1):

    import FastS.PyTree as FastS

    FastS.createConvergenceHistory(worfklow.tree, ExtactionPeriod)
    cgns.castNode(worfklow.tree)

