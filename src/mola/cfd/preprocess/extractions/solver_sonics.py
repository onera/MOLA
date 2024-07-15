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
from mola.logging import mola_logger, MolaException

def apply_to_solver(workflow):

    mola_logger.warning('No custom extractions available with SoNICS for now.')
    # workflow._pytriggers = []
    add_extractions_for_restart(workflow)
    # process_extractions(workflow)

def add_extractions_for_restart(workflow):
    workflow._interface.add_to_Extractions_Restart(
        # Container='FlowSolution#EndOfRun', 
        Fields=['conservatives'],
        )

def process_extractions(workflow):
    import sonics.toolkit.triggers as triggers

    extractions_merged = []
    for extraction in workflow.Extractions:
        family = extraction.get('Family', '*')
        if family not in extractions_merged:
            extractions_merged[family] = extraction['Fields']
        else:
            extractions_merged[family] += extraction['Fields']

    trigger = triggers.ExtractTrigger(
        workflow.SolverParameters['configuration']['conf'], 
        extractions_merged, 
        workflow.SolverParameters['configuration']['hpc_conf']['hardware_target']
        ) 
    
    workflow._pytriggers += trigger
    