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
    apply_to_solver(workflow)

def split_with_maia(tree):
    import maia
    import maia4elsA

    part_tree = maia.factory.partition_dist_tree(tree, comm)
    maia4elsA.add_renumbering_data(part_tree)
    skeleton_tree = maia4elsA.get_skeleton_tree(part_tree, comm)
    distribution = maia4elsA.get_distribution(part_tree, comm)
    part_tree = maia.pytree.union([skeleton_tree, part_tree])  

    return part_tree, skeleton_tree, distribution
