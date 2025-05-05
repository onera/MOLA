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

from abc import ABC, abstractmethod

class BoundaryConditionsDispatcher(ABC):

    def __init__(self):

        types = [
            "Farfield",
            "InflowStagnation",
            "InflowMassFlow",
            "OutflowPressure",
            "OutflowSupersonic",
            "OutflowMassFlow",
            "OutflowRadialEquilibrium",
            "WallViscous",
            "WallViscousIsothermal",
            "WallInviscid",
            "SymmetryPlane",
            "MixingPlane",
            "UnsteadyRotorStatorInterface",
            "ChorochronicInterface"
        ]

    def assert_type_supported(self, requested_type : str) -> bool:
        if requested_type not in self.types:
            msg = (f'requested boundary-condition type "{requested_type}" not'
                   f' supported, must be in:\n{self.types}')
            raise AttributeError(msg)
    
    @abstractmethod
    def get_solver_type(self, requested_type : str):
        """
        Must be implemented by subclasses to map a generic BC name
        to a solver-specific BC string.
        """
        pass