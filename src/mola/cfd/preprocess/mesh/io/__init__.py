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

from mola.logging import MolaException, MolaUserError
from . import default, autogrid, utils, reader, writer, unstructured

from treelab import cgns

def apply(workflow):
    read(workflow)
    unstructured.apply(workflow)

def read(workflow):
    meshes = []
    for component in workflow.RawMeshComponents:
        
        if 'Mesher' not in component or component['Mesher'] == 'default':
            base = default.reader(workflow, component)

        elif component['Mesher'].lower() == 'autogrid':
            base = autogrid.reader(workflow, component)

        else:
            raise MolaException(f"unknown Mesher: {component['Mesher']}")
        
        meshes += [base]
    
    workflow.tree = cgns.add(meshes)

    dimOfBases = set(base.dim() for base in workflow.tree.bases())
    if len(dimOfBases) != 1:
        raise MolaUserError('All bases must have the same physical dimension')
    workflow.ProblemDimension = int(list(dimOfBases)[0])
