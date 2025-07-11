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

from mola.logging.exceptions import MolaException

try:
    import Converter.PyTree as C
    import Converter.Internal as I
except:
    pass


def is_partitioned_for_use_in_maia(tree):
    return bool(I.getNodeFromName(tree, ":CGNS#GlobalNumbering"))


def is_distributed_for_use_in_maia(tree):
    return bool(I.getNodeFromName(tree, ":CGNS#Distribution"))


def assert_zones_have_zone_type_node(tree):
    # assert I.getNodeFromName(tree,"ZoneType")
    path = ''
    for base in I.getBases(tree):
        for zone in I.getZones(base):
            path = base[0]+"/"+zone[0]
            zone_type = I.getNodeFromName1(zone,"ZoneType")
            
            if not zone_type:
                raise MolaException(f"zone {path} does not have ZoneType node")
            
            zone_type_value = I.getValue(zone_type)
            if zone_type_value not in ['Structured', 'Unstructured']:
                raise MolaException(f"zone {path} has value {zone_type_value} which is not recognized")


