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

import numpy as np

from ... import Workflow
from .interface import WorkflowAirplaneInterface
from ..flow_direction_calculator import from_two_angles_and_aircraft_yaw_pitch_axis
from mola.cfd.postprocess.signals.airplane_coefficients_computer import add_aerodynamic_coefficients_to
from mola import solver

class WorkflowAirplane(Workflow):

    def __init__(self, **kwargs):
        self._interface = WorkflowAirplaneInterface(self, **kwargs)
        self.set_flow_directions()

    def set_flow_directions(self):
        flow_dirs = from_two_angles_and_aircraft_yaw_pitch_axis(
            self.ApplicationContext['AngleOfAttackDeg'], 
            self.ApplicationContext['AngleOfSlipDeg'], 
            self.ApplicationContext['YawAxis'], 
            self.ApplicationContext['PitchAxis']
            )
        
        self.ApplicationContext.update(
            dict(DragDirection=flow_dirs[0],
                 SideDirection=flow_dirs[1],
                 LiftDirection=flow_dirs[2]))

        self.Flow['Direction'] = self.ApplicationContext['DragDirection']

    def compute_flow_and_turbulence(self):
        super().compute_flow_and_turbulence()
        self.set_reference_values()

    def set_reference_values(self):
        self.ApplicationContext['FluxCoef'] = 1./ (self.Flow['PressureDynamic'] * self.ApplicationContext['Surface'])
        self.ApplicationContext['TorqueCoef'] = self.ApplicationContext['FluxCoef'] / self.ApplicationContext['Length']
        self.Flow['Reynolds'] = self.Flow['Density'] * self.Flow['VelocityForScalingAndTurbulence'] * self.ApplicationContext['Length'] / self.Flow['ViscosityMolecular']

    def compute_aerodynamic_coefficients(self, extraction : dict, **operation):
        if extraction["Type"] != "Integral": return
        add_aerodynamic_coefficients_to(extraction, self.ApplicationContext)
