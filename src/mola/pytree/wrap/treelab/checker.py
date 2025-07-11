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

try: from treelab import cgns
except: pass

from mola.logging.exceptions import MolaException

def is_partitioned_for_use_in_maia(tree):
    t = cgns.castNode(tree)
    return bool(t.get(':CGNS#GlobalNumbering'))


def is_distributed_for_use_in_maia(tree):
    t = cgns.castNode(tree)
    return bool(t.get(':CGNS#Distribution'))

def assert_zones_have_zone_type_node(tree):
    t = cgns.castNode(tree)
    for zone in t.zones():
        zone_type = zone.get(Name="ZoneType")
        
        if not zone_type:
            raise MolaException(f"zone {zone.path()} does not have ZoneType node")
        
        zone_type_value = zone_type.value()
        if zone_type_value not in ['Structured', 'Unstructured']:
            raise MolaException(f"zone {zone.path()} has value {zone_type_value} which is not recognized")
