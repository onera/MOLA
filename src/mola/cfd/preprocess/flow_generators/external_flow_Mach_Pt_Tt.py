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

from typing import Union

import numpy as np
from .external_flow import ExternalFlowGenerator 

class ExternalMPtTtFlowGenerator(ExternalFlowGenerator):

    name = 'External_Mach_Pt_Tt'

    def __init__(self, workflow):
        super().__init__(workflow)       

    def set_Flow_defaults(self,
            Direction              : Union [ list, tuple, np.ndarray ] = [1., 0., 0.],          
            VelocityForScalingAndTurbulence : float = None,
            *,
            Mach                             : float = None,
            PressureStagnation               : float = None,
            TemperatureStagnation            : float = None,
            ):
        # This function does nothing, but it is mandatory to be called by WorkflowInterface.
        # Its signature will be checked.
        return

    def set_flow_properties(self):

        Mach = self.Flow['Mach'] 
        TemperatureStagnation = self.Flow['TemperatureStagnation']
        PressureStagnation = self.Flow['PressureStagnation']

        Temperature = TemperatureStagnation/(1 + (self.Fluid['Gamma']-1)/2. * Mach**2)
        Pressure = PressureStagnation/(TemperatureStagnation/Temperature)**(self.Fluid['Gamma'] / (self.Fluid['Gamma']-1))
        Density = Pressure/(self.Fluid['IdealGasConstant'] * Temperature)  
        SoundSpeed = np.sqrt( self.Fluid['Gamma'] * self.Fluid['IdealGasConstant'] * Temperature )
        Velocity = self.Flow["Mach"]*SoundSpeed  

        self.Flow.update(dict(
            Temperature = Temperature,
            Pressure = Pressure,
            Density = Density,
            SoundSpeed = SoundSpeed,
            Velocity = Velocity,
        ))

        super().set_flow_properties()

