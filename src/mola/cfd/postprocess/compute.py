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

from typing import Union
import numpy as np
from treelab import cgns

def compute_radius(
        tree: Union[cgns.Zone, cgns.Base, cgns.Tree], 
        axis: Union[None, str] = None, 
        fieldname: str = 'Radius',
        container: str = 'FlowSolution', 
        coordinate_system: str = 'cartesian'
        ):
    assert coordinate_system in ['cartesian']  # TODO extend to other systems ['cartesian', 'cylindrical', 'spherical']

    tree = cgns.Node(tree)

    for zone  in tree.zones():
        zone : cgns.Zone
        x, y, z = zone.xyz()

        if axis is None:
            R = np.sqrt(x*x + y*y + z*z) # spherical ?
        elif axis == 'x':
            R = np.sqrt(      y*y + z*z)
        elif axis == 'y':
            R = np.sqrt(x*x       + z*z)
        elif axis == 'z':
            R = np.sqrt(x*x + y*y      )
        else:
            raise NotImplementedError(f"axis={axis} not supported")
        
        fs = zone.get(Name=container, Type='FlowSolution_t', Depth=1)
        if not fs:
            fs = cgns.Node(Name=container, Type='FlowSolution_t', Parent=zone)
            cgns.Node(Name='GridLocation', Value='Vertex', Type='GridLocation_t', Parent=fs)
        
        cgns.Node(Name=fieldname, Value=R, Type='DataArray_t', Parent=fs, override_sibling_by_name=True)