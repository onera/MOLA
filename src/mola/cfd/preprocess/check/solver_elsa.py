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
from mola.logging import MolaException

def apply_to_solver(workflow):
    pass
    # See evolution of issue https://gitlab.onera.net/numerics/mesh/maia/-/issues/164
    # check_FlowSolutionEoR(workflow.tree, list(workflow.Flow['ReferenceState']))

def check_FlowSolutionEoR(tree: cgns.Tree, conservatives: list):
    for zone in tree.zones():
        fs_EoR = zone.get(Name='FlowSolution#EndOfRun', Type='FlowSolution')
        if not fs_EoR:
            raise MolaException(f'FlowSolution#EndOfRun is missing in zone {zone.name()}')
        
        for name in conservatives:
            node = fs_EoR.get(Name=name, Type='DataArray')
            if not node or node.value() is not None:
                raise MolaException(f'FlowSolution#EndOfRun/{node.name()} is missing in zone {zone.name()}')
