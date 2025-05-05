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

from .boundary_conditions_dispatcher import BoundaryConditionsDispatcher

class BoundaryConditionsDispatcherElsa(BoundaryConditionsDispatcher):

    def __init__(self):
        super().__init__()

        self._mapping = {
            "Farfield" : "nref",
            "InflowStagnation" : "inj1",
            "InflowMassFlow" : "injmfr1",
            "OutflowPressure" : "outpres",
            "OutflowSupersonic" : "outsup",
            "OutflowMassFlow" : "outmfr2",
            "OutflowRadialEquilibrium" : "outradeqhyb",
            "WallViscous" : "walladia",
            # "WallViscousIsothermal" : None,
            "WallInviscid" : "wallslip",
            "SymmetryPlane" : "sym",
            "MixingPlane" : "stage_mxpl_hyb",
            "UnsteadyRotorStatorInterface" : "stage_red_hyb",
            "ChorochronicInterface" : "chorochronic"
        }

    def get_solver_type(self, requested_type: str) -> str:
        self.assert_type_supported(requested_type)
        try:
            solver_bc_type = self._mapping[requested_type]
        except KeyError:
            msg = f'boundary condition "{requested_type}" not implemented for elsa'
            raise AttributeError(msg)
        
        return solver_bc_type