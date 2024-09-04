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

def iso_surface(tree, IsoSurfaceField, IsoSurfaceValue, IsoSurfaceContainer, comm):

    containers_name = [fs.name() for fs in tree.group(Type='FlowSolution')]

    if IsoSurfaceContainer == 'GridCoordinates':
        # maia cannot do an iso_surface on GridCoordinates
        index_of_coord = dict(CoordinateX=0, CoordinateY=1, CoordinateZ=2)
        assert IsoSurfaceField in list(index_of_coord)
        plane_eq = [0, 0, 0, IsoSurfaceValue]  # caution, plane equation for maia is: ax+by+cz-d=0
        plane_eq[index_of_coord[IsoSurfaceField]] = 1
        surface = maia.algo.part.plane_slice(
                            tree, 
                            plane_eq, 
                            containers_name=containers_name, 
                            comm=comm,
                            )
    else:
        surface = maia.algo.part.iso_surface(
                            tree, 
                            f"{IsoSurfaceContainer}/{IsoSurfaceField}",
                            iso_val=IsoSurfaceValue,
                            containers_name=containers_name, 
                            comm=comm,
                            )
    return surface
