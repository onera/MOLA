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
from mola.logging import MolaException

def extract_bc_from_family(tree, Family, comm):
    tree_ref = maia.pytree.shallow_copy(tree)
    
    try:
        import maia.pytree.maia.check_tree as check
        is_part = check.is_cgns_part_tree(tree_ref)
        is_dist = check.is_cgns_dist_tree(tree_ref)
        is_full = check.is_cgns_full_tree(tree_ref)
    
    except ModuleNotFoundError:
        import mola.pytree.user.checker as check
        is_part = check.is_partitioned_for_use_in_maia(tree_ref)
        is_dist = check.is_distributed_for_use_in_maia(tree_ref)
        is_full = not is_part and not is_dist

    
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


def extract_bc_from_zsr(tree: cgns.Tree, Family, comm):
    
    rank = comm.Get_rank()
    extracted_zones = []

    all_zones_names = comm.gather([z.name() for z in tree.zones()])
    if rank == 0:
        all_zones_names = list(set([item for sublist in all_zones_names for item in sublist]))
    comm.barrier()
    all_zones_names = comm.bcast(all_zones_names, root=0)  # to be sure to have zones in the same order on all ranks

    # HACK extract_part_from_zsr works only for tree with one zone
    # see https://gitlab.onera.net/numerics/mesh/maia/-/issues/219
    for zone_name in all_zones_names:
        zone = tree.get(Type='Zone', Name=zone_name, Depth=2)
        zsr_names = []

        # make a shallow copy of tree with only the current zone
        tree_with_one_zone = tree.copy()
        for z in tree_with_one_zone.zones():
            if z.name() != zone_name:
                z.remove()

        if zone is not None:
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
        all_zsr_names = comm.gather(zsr_names)
        if rank == 0:
            shared_zsr_names = list(set([item for sublist in all_zsr_names for item in sublist]))
        else: 
            shared_zsr_names = None
        comm.barrier()
        shared_zsr_names = comm.bcast(shared_zsr_names, root=0)  # to be sure to have names in the same order on all ranks

        for zsr_name in shared_zsr_names:
            extracted_tree = maia.algo.part.extract_part_from_zsr(tree_with_one_zone, zsr_name, comm, containers_name=[]) 
            extracted_tree = cgns.castNode(extracted_tree)
            # HACK
            extracted_tree.findAndRemoveNodes(Type='ZoneBC')
            extracted_zones.extend(extracted_tree.zones())

    return extracted_zones
