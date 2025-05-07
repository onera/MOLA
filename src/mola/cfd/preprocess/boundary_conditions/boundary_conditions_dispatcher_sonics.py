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

from mola.cfd.preprocess.boundary_conditions.boundary_conditions_dispatcher import BoundaryConditionsDispatcher

class BoundaryConditionsDispatcherSonics(BoundaryConditionsDispatcher):

    def __init__(self):
        super().__init__()

        self._mapping = {
            "Farfield" : "BCFarfield",
            "InflowStagnation" : "BCInflowSubsonicPressure",
            "InflowMassFlow" : "BCInflowSubsonicMassFlow",
            "OutflowPressure" : "BCOutflowSubsonic",
            "OutflowSupersonic" : None,
            "OutflowMassFlow" : None,
            "OutflowRadialEquilibrium" : "BCOutflowRadialEquilibrium",
            "Wall": "BCWallViscous",
            "WallViscous" : "BCWallViscous",
            "WallViscousIsothermal" : "BCWallViscousIsothermal",
            "WallInviscid" : "BCWallInviscid",
            "SymmetryPlane" : "BCSymmetryPlane",
            "MixingPlane" : "GCMixingPlane",
            "UnsteadyRotorStatorInterface" : None,
            "ChorochronicInterface" : None
        }

        self._without_generic_name = []

        self._remove_unsupported_bcs_from_mapping()

