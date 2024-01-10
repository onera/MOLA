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

'''
Implements class **Point**, which inherits from :py:class:`Curve`

28/04/2022 - L. Bernardos - first creation
'''

from mola.misc import np, RED,GREEN,YELLOW,PINK,CYAN,ENDC
from . import Curve
from ...node import Node

class Point(Curve):
    """docstring for Point"""
    def __init__(self, Coordinates=[0,0,0], Name='Point'):
        super().__init__(Name=Name)

        GridCoordinates = Node(Parent=self,
               Name='GridCoordinates', Type='GridCoordinates_t',
               Children=[Node(Name='CoordinateX', Value=Coordinates[0], Type='DataArray_t'),
                         Node(Name='CoordinateY', Value=Coordinates[1], Type='DataArray_t'),
                         Node(Name='CoordinateZ', Value=Coordinates[2], Type='DataArray_t')])

        self.setValue(np.array([[1,1,0]],dtype=np.int32,order='F'))

