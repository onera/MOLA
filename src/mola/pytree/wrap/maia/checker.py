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

try: import maia
except: pass

from mola.logging.exceptions import MolaException

def is_partitioned_for_use_in_maia(tree):
    return bool(maia.pytree.get_node_from_name(tree, ":CGNS#GlobalNumbering"))


def is_distributed_for_use_in_maia(tree):
    return bool(maia.pytree.get_node_from_name(tree, ":CGNS#Distribution"))


def assert_zones_have_zone_type_node(tree):
    path = ''
    for base in maia.pytree.get_nodes_from_label(tree, "CGNSBase_t", depth=1):
        for zone in maia.pytree.get_nodes_from_label(base, "Zone_t", depth=1):
            path = base[0]+"/"+zone[0]
            
            zone_type = maia.pytree.get_node_from_name(zone, "ZoneType", depth=1)
            
            if not zone_type:
                raise MolaException(f"zone {path} does not have ZoneType node")
            
            zone_type_value = maia.pytree.get_value(zone_type)
            if zone_type_value not in ['Structured', 'Unstructured']:
                raise MolaException(f"zone {path} has value {zone_type_value} which is not recognized")
