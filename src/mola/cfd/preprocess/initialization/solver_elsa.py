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
    # Remove ChimeraCellType nodes
    groupOfNodes = workflow.tree.group(Name='FlowSolution#Init')
    for node in groupOfNodes:
        node.findAndRemoveNodes(Name='ChimeraCellType')     
    
    if workflow.tree.get(Name='TurbulentDistance', Type='DataArray'):
        import Converter.elsAProfile as elsAProfile
        import Converter.Internal as I
        previous_container = I.__FlowSolutionCenters__
        I.__FlowSolutionCenters__ = 'FlowSolution#Init'
        elsAProfile._addTurbulentDistanceIndex(workflow.tree)
        workflow.tree = cgns.castNode(workflow.tree)

        # absolutely required to restitute previous container name since this
        # may provoke unexpected boundary errors during pytests since this 
        # variable value may persist on different contexts and is dangerous
        I.__FlowSolutionCenters__ = previous_container 
