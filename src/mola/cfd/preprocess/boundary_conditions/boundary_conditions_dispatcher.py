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
from fnmatch import fnmatch

class BoundaryConditionsDispatcher(ABC):

    def __init__(self):
        self._mapping = dict()
        self._without_generic_name = []
        self._cgns_to_generic = {
            # TODO double-check these names. TODO avoid use of this dict by redesign
            # strong CAVEAT : each solver put what they want as FamilyBC value...
            "BCFarfield"                    : "Farfield",
            "BCInflow"                      : "InflowStagnation",
            "BCInflowSubsonic"              : "InflowStagnation",
            "BCInflowMassFlow"              : "InflowMassFlow",
            "BCOutflow"                     : "OutflowPressure",
            "BCOutpres"                     : "OutflowPressure",
            "BCOutflowSubsonic"             : "OutflowPressure",
            "BCOutflowSupersonic"           : "OutflowSupersonic",
            "BCOutflowMassFlow"             : "OutflowMassFlow",
            "BCOutflowRadialEquilibrium"    : "OutflowRadialEquilibrium",
            "BCWall"                        : "Wall",
            "BCWallViscous"                 : "WallViscous",
            "BCWallViscousIsothermal"       : "WallViscousIsothermal",
            "BCWallInviscid"                : "WallInviscid",
            "BCSymmetryPlane"               : "SymmetryPlane",
            "BCMixingPlane"                 : "MixingPlane",
            "BCUnsteadyRotorStatorInterface": "UnsteadyRotorStatorInterface",
            "BCChorochronicInterface"       : "ChorochronicInterface",
        }
    
    def _remove_unsupported_bcs_from_mapping(self):
        for key in list(self._mapping):
            if self._mapping[key] is None:
                del self._mapping[key]

    def get_name_used_by_solver(self, requested_type: str) -> str:

        if requested_type in self.get_all_generic_names():
            return self._mapping[requested_type]
        
        elif requested_type in self.get_all_specific_names():
            return requested_type
        
        else:
            msg = (f'requested boundary-condition type "{requested_type}" not'
                   f' supported, must be in:\n{list(self._mapping)}\n'
                   f'or in:\n{self._without_generic_name}')
            raise AttributeError(msg)
    
    def get_all_generic_names(self) -> list:
        return list(self._mapping)
    
    def get_all_specific_names(self) -> list:
        return [self._mapping[k] for k in self._mapping] + self._without_generic_name
    
    def get_all_supported_names(self) -> list:
        return self.get_all_generic_names() + self.get_all_specific_names() + list(self._cgns_to_generic)

    def is_supported(self, bc_type : str) -> bool:
        return bc_type in self.get_all_supported_names()

    def to_generic_name(self, bc_type : str) -> str:
        
        translator = dict( (v,k) for k,v in self._mapping.items() )
        
        if bc_type in self.get_all_generic_names():
            return bc_type
        
        elif bc_type in self._cgns_to_generic:
            return self._cgns_to_generic[bc_type]
        
        elif bc_type in translator:
            return translator[bc_type]
        
        raise ValueError(f'bc type "{bc_type}" does not have a generic name match')

    def is_allowed_shell_pattern(self, type_with_pattern : str) -> bool:
        for name in self.get_all_supported_names():
            if fnmatch(name, type_with_pattern):
                return True
        return False
    