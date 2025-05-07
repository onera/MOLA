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
from mola.logging import MolaUserAttributeError, MolaUserError

class ExternalMPTFlowGenerator(ExternalFlowGenerator):

    name = 'External_Mach_P_T'

    def __init__(self, workflow):
        super().__init__(workflow)       

    def set_Flow_defaults(self,
            Direction              : Union [ list, tuple, np.ndarray ] = [1., 0., 0.],          
            VelocityForScalingAndTurbulence : float = None,
            *,
            Mach                   : float = None,
            Pressure               : float = None,
            Temperature            : float = None,
            ):
        # This function does nothing, but it is mandatory to be called by WorkflowInterface.
        # Its signature will be checked.
        return 

    def _check_inputs(self):
        if len(self.Flow['Direction']) != 3:
            raise MolaUserAttributeError('Direction argument must be a 3-float list, tuple or numpy')
        self.Flow['Direction'] = np.array(self.Flow['Direction'], dtype=float)

        SoundSpeed = np.sqrt( self.Fluid['Gamma'] * self.Fluid['IdealGasConstant'] * self.Flow['Temperature'] )
        Velocity = self.Flow['Mach']*SoundSpeed

        if not 'VelocityForScalingAndTurbulence' in self.Flow \
            or self.Flow['VelocityForScalingAndTurbulence'] is None:
            self.Flow['VelocityForScalingAndTurbulence'] = np.abs(Velocity)
            if self.Flow['VelocityForScalingAndTurbulence'] < 1e-5:
                raise MolaUserError('Velocity is very low. You must set a positive value for VelocityForScalingAndTurbulence')
        elif self.Flow['VelocityForScalingAndTurbulence'] <= 0:
            raise MolaUserError('You must provide positive value for VelocityForScalingAndTurbulence')

        return self.Flow

    def set_flow_properties(self):
    
        self._check_inputs()

        # TODO Put ViscosityMolecular in the Fluid attribute ?
        def SutherlandLaw(T, mus, Ts, S):
            return mus * (T/Ts)**1.5 * ((Ts + S)/(T + S))

        Mach = self.Flow['Mach'] 
        ViscosityMolecular = SutherlandLaw(self.Flow['Temperature'], self.Fluid['SutherlandViscosity'], self.Fluid['SutherlandTemperature'], self.Fluid['SutherlandConstant'])
        Density = self.Flow['Pressure']/(self.Fluid['IdealGasConstant'] * self.Flow['Temperature'])  
        SoundSpeed = np.sqrt( self.Fluid['Gamma'] * self.Fluid['IdealGasConstant'] * self.Flow['Temperature'] )
        Velocity = self.Flow["Mach"]*SoundSpeed      
        Pressure = Density * self.Fluid['IdealGasConstant'] * self.Flow['Temperature']        
        PressureDynamic = 0.5 * Density * self.Flow['VelocityForScalingAndTurbulence'] **2
        TemperatureStagnation = self.Flow['Temperature'] * (1 + (self.Fluid['Gamma']-1)/2. * Mach**2)
        PressureStagnation = Pressure * (TemperatureStagnation/self.Flow['Temperature']) ** (self.Fluid['Gamma'] / (self.Fluid['Gamma']-1)) 

        Momentum_vector = Density* Velocity * np.array(self.Flow['Direction'])
        MomentumX =  Momentum_vector[0]
        MomentumY =  Momentum_vector[1]
        MomentumZ =  Momentum_vector[2]
        EnergyStagnationDensity = Density * ( self.Fluid['cv'] * self.Flow['Temperature'] + 0.5 * Velocity **2)

        self.Flow['Conservatives'] = dict(
            Density = Density,
            MomentumX = float(MomentumX),
            MomentumY = float(MomentumY),
            MomentumZ = float(MomentumZ),
            EnergyStagnationDensity = float(EnergyStagnationDensity)
        )

        # Update ReferenceValues dictionary
        self.Flow.update(dict(
            Density                 = Density, 
            Velocity                = Velocity,
            Pressure                = Pressure,
            PressureDynamic         = PressureDynamic,
            PressureStagnation      = PressureStagnation,
            TemperatureStagnation   = TemperatureStagnation,
            ViscosityMolecular      = ViscosityMolecular,
            ViscosityEddy           = self.Turbulence['Viscosity_EddyMolecularRatio'] * ViscosityMolecular,
            MomentumX               = MomentumX,
            MomentumY               = MomentumY,
            MomentumZ               = MomentumZ,
            EnergyStagnationDensity = EnergyStagnationDensity,
            SoundSpeed              = SoundSpeed,
        ))

   