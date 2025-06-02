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
from mola.pytree.user.checker import is_partitioned_for_use_in_maia

def extract_bc_from_family(tree, Family, comm):
    if not is_partitioned_for_use_in_maia(tree):
        raise MolaException('cannot extract a bc from a cgns tree that is not partitioned for use in maia')

    # CAVEAT cannot extract surface grid only, raises error if no BCDataSet found
    surface = maia.algo.part.extract_part_from_family(tree, Family, comm, containers_name=['BCDataSet'])
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
