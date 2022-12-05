'''
Implements class **Point**, which inherits from :py:class:`Curve`

28/04/2022 - L. Bernardos - first creation
'''

from ...Core import np, RED,GREEN,WARN,PINK,CYAN,ENDC
from . import Curve
from ...Node import Node


class Point(Curve):
    """docstring for Point"""
    def __init__(self, Coordinates=[0,0,0], Name='Point'):
        super().__init__(Name=Name)

        GridCoordinates = Node(Parent=self,
               Name='GridCoordinates', Type='GridCoordinates_t',
               Children=[Node(Name='CoordinateX', Value=Coordinates[0], Type='DataArray_t'),
                         Node(Name='CoordinateY', Value=Coordinates[1], Type='DataArray_t'),
                         Node(Name='CoordinateZ', Value=Coordinates[2], Type='DataArray_t')])

        self.setValue(np.array([[1,1,0]],dtype=np.int,order='F'))

