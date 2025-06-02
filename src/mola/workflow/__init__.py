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
from mola.logging import MolaAssertionError

from .workflow import Workflow 
from .interface import WorkflowInterface
from .manager import WorkflowManager

from .fixed import airplane
from .fixed import airfoil
from .fixed import linear_cascade
from .rotating_component import turbomachinery
from .rotating_component import propeller

# This list must be updated when a new workflow is added to mola
AVAILABLE_WORKFLOWS_CLASSES = [
    Workflow,
    airplane.Workflow,
    airfoil.Workflow,
    linear_cascade.Workflow,
    rotating_component.Workflow,
    propeller.Workflow,
    turbomachinery.Workflow,
]
AVAILABLE_WORKFLOWS = dict((w.__name__, w) for w in AVAILABLE_WORKFLOWS_CLASSES)

def read_workflow(source: str) -> Workflow:
    # Get the right class of Workflow
    try:
        workflow_name_node = cgns.load_from_path(source, 'WorkflowParameters/Name')
    except: 
        if not isinstance(source, str):
            raise MolaAssertionError(f'The argument of read_workflow must be a filename (a string).')
        else:
            raise MolaAssertionError(f'Unable to load the workflow: the file {source} does not contain the node WorkflowParameters/Name')
    workflow_name = workflow_name_node.value()
    PreviouslyUsedWorkflow = AVAILABLE_WORKFLOWS.get(workflow_name)

    workflow = PreviouslyUsedWorkflow(tree=source)

    return workflow
