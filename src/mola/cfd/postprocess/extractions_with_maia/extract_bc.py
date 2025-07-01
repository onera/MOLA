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
import maia
import maia.pytree.maia.check_tree as check
from mola.logging import MolaException

def extract_bc_from_family(tree, Family, comm):
    tree_ref = maia.pytree.shallow_copy(tree)
    
    is_part = check.is_cgns_part_tree(tree_ref)
    is_dist = check.is_cgns_dist_tree(tree_ref)
    is_full = check.is_cgns_full_tree(tree_ref)
    
    if is_part:
        part_tree = tree_ref
    
    elif is_dist:
        part_tree = maia.factory.partition_dist_tree(tree_ref, comm)

    elif is_full:
        if comm.Get_size() > 1:
            raise MolaException('cannot execute maia using full tree in parallel MPI context')
        dist_tree = maia.factory.full_to_dist_tree(tree_ref, comm, owner=0)
        part_tree = maia.factory.partition_dist_tree(dist_tree, comm)
    
    else:
        raise MolaException("tree is not recognized as partitioned, distributed nor full. Cannot use maia.")

    surface = maia.algo.part.extract_part_from_family(part_tree, Family, comm,
        # CAUTION https://gitlab.onera.net/numerics/mesh/maia/-/issues/201
        containers_name=['BCDataSet'])
    return surface


def extract_bc_from_zsr(tree, Family, comm):
    zsr_names = []
    for zone in tree.zones():
        for zsr in zone.group(Type='ZoneSubRegion'):
            # a ZSR range is specified by one of PointRange, PointList, BCRegionName or GridConnectivityRegionName
            # see http://cgns.github.io/CGNS_docs_current/sids/gridflow.html#ZoneSubRegion
            BCRegionName = zsr.get(Name='BCRegionName')
            if BCRegionName:
                bc = zone.get(Type='BC', Name=BCRegionName.value())
                FamilyName_nodes = bc.group(Type='FamilyName') + bc.group(Type='AdditionalFamilyName')
                if any([node.value() == Family for node in FamilyName_nodes]):
                    zsr_names.append(zsr.name())

    # Gather zsr_names on all ranks and make a list with unique names
    all_zsr_names = comm.allgather(zsr_names)
    shared_zsr_names = list(set([item for sublist in all_zsr_names for item in sublist]))

    zones = []
    for zsr_name in shared_zsr_names:
        extracted_tree = maia.algo.part.extract_part_from_zsr(tree, zsr_name, comm, containers_name=[]) 
        extracted_tree = cgns.castNode(extracted_tree)
        # HACK
        extracted_tree.findAndRemoveNodes(Type='ZoneBC')
        zones.extend(extracted_tree.zones())

    return zones
