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
    
    import Converter.PyTree as C
    import Converter.Internal as I

    __FlowSolutionNodes__BACKUP = I.__FlowSolutionNodes__
    I.__FlowSolutionNodes__ = container

    # if axis is None:
    #     tree.useEquation(f'{fieldname}=(CoordinateX**2, CoordinateY**2+CoordinateZ**2)**0.5', Container=container)
    # elif axis == 'x':
    #     tree.useEquation(f'{fieldname}=(CoordinateY**2+CoordinateZ**2)**0.5', Container=container)

    if axis is None:
        C._initVars(tree, f'{fieldname}=({{CoordinateX}}**2 + {{CoordinateY}}**2+{{CoordinateZ}}**2)**0.5')
    elif axis == 'x':
        C._initVars(tree, f'{fieldname}=({{CoordinateY}}**2+{{CoordinateZ}}**2)**0.5')
    elif axis == 'y':
        C._initVars(tree, f'{fieldname}=({{CoordinateX}}**2+{{CoordinateZ}}**2)**0.5')
    elif axis == 'z':
        C._initVars(tree, f'{fieldname}=({{CoordinateX}}**2+{{CoordinateY}}**2)**0.5')

    I.__FlowSolutionNodes__ = __FlowSolutionNodes__BACKUP

