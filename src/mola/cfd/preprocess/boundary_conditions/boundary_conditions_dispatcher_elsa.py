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

class BoundaryConditionsDispatcherElsa(BoundaryConditionsDispatcher):

    def __init__(self):
        super().__init__()

        # TODO add injrot, wallisoth and Giles conditions
        self._mapping = {
            "Farfield" : "nref",
            "InflowStagnation" : "inj1",
            "InflowMassFlow" : "injmfr1",
            "OutflowPressure" : "outpres",
            "OutflowSupersonic" : "outsup",
            "OutflowMassFlow" : "outmfr1",  #"outmfr2",
            "OutflowRadialEquilibrium" : "outradeqhyb",
            "Wall": "walladia",
            "WallViscous" : "walladia",
            "WallViscousIsothermal" : None, # not implemented
            "WallInviscid" : "wallslip",
            "SymmetryPlane" : "sym",
            "MixingPlane" : "stage_mxpl_hyb",
            "UnsteadyRotorStatorInterface" : "stage_red_hyb",
            "ChorochronicInterface" : "chorochronic"
        }

        self._without_generic_name = ['stage_mxpl', 'stage_red', 'outradeq', 'outmfr2',
                                      'giles_inlet', 'giles_outlet', 'giles_stage_mxpl']

        self._remove_unsupported_bcs_from_mapping()

