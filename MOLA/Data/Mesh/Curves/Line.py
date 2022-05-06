'''
Implements class **Line**, which inherits from :py:class:`Curve`

28/04/2022 - L. Bernardos - first creation
'''

from ...Core import np, RED,GREEN,WARN,PINK,CYAN,ENDC,CGM
from . import Curve
from ...Node import Node


class Line(Curve):
    """docstring for Line"""
    def __init__(self, Start=[0,0,0], End=[1,0,0], N=2, Name='Line', **distribution):
        super().__init__(Name=Name)

        Start = np.array(Start, dtype=float)
        End = np.array(End, dtype=float)

        X = np.linspace(Start[0], End[0], N)
        Y = np.linspace(Start[1], End[1], N)
        Z = np.linspace(Start[2], End[2], N)
        GridCoordinates = Node(Parent=self,
               Name='GridCoordinates', Type='GridCoordinates_t',
               Children=[Node(Name='CoordinateX', Value=X, Type='DataArray_t'),
                         Node(Name='CoordinateY', Value=Y, Type='DataArray_t'),
                         Node(Name='CoordinateZ', Value=Z, Type='DataArray_t')])

        self.setValue(np.array([[N,N-1,0]],dtype=np.int,order='F'))

        if distribution: self.discretize(**distribution)
