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
    check_empty_str(workflow.tree)

def check_empty_str(tree):
    nodes = tree.group(Value='')
    if len(nodes) > 0:
        txt = '\n'.join([n.path() for n in nodes])
        raise MolaException(f'The value of following node(s) is an empty str, that is forbidden for elsA: \n{txt}')
    