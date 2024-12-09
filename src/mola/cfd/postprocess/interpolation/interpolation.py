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

import maia
from mola.cfd.preprocess.mesh.tools import to_partitioned

def interpolate_closest(tree_source, tree_target, comm):

    containers_at_vertex = [
        fs.name() for fs in tree_source.group(Type='FlowSolution') 
        if not fs.get(Type='GridLocation') or fs.get(Type='GridLocation').value() == 'Vertex'
        ]
    containers_at_cellcenter = [
        fs.name() for fs in tree_source.group(Type='FlowSolution') 
        if fs.get(Type='GridLocation') and fs.get(Type='GridLocation').value() == 'CellCenter'
        ]
    
    tree_source = to_partitioned(tree_source) 
    tree_target = to_partitioned(tree_target) 

    maia.algo.part.interpolate(
        tree_source, 
        tree_target, 
        comm, 
        containers_name=containers_at_vertex, 
        location='Vertex',
        strategy='Closest',
        n_closest_pt=4,
        )
    maia.algo.part.interpolate(
        tree_source, 
        tree_target, 
        comm, 
        containers_name=containers_at_cellcenter, 
        location='CellCenter',
        strategy='Closest',
        n_closest_pt=4,
        )
    