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
import scipy.optimize
from .external_flow import ExternalFlowGenerator 

class InternalFlowGenerator(ExternalFlowGenerator):

    def __init__(self, workflow):
        super().__init__(workflow)
        self.name = 'Internal'

        try:
            self.Surface = workflow.ApplicationContext['Surface']
        except KeyError:
            from..mesh.tools import get_surface_of_inflow
            self.Surface = get_surface_of_inflow(workflow)

    def set_flow_properties(self):
        self.compute_external_quantities_from_internal_quantities()        
        super().set_flow_properties()

    def compute_external_quantities_from_internal_quantities(self):
        assert not('MassFlow' in self.Flow and 'Mach' in self.Flow), 'MassFlow and Mach cannot be given together in Flow. Choose one'
        if 'MassFlow' in self.Flow:
            self.Flow['Mach'] = self.MachFromMassFlow(self.Flow['MassFlow'], 
                                                      self.Surface, 
                                                      self.Flow['PressureStagnation'], 
                                                      self.Flow['TemperatureStagnation'], 
                                                      self.Fluid['IdealGasConstant'], 
                                                      self.Fluid['Gamma']
                                                      )
        elif 'Mach' in self.Flow:
            self.Flow['MassFlow'] = self.MassFlowFromMach(self.Flow['Mach'], 
                                                          self.Surface, 
                                                          self.Flow['PressureStagnation'], 
                                                          self.Flow['TemperatureStagnation'], 
                                                          self.Fluid['IdealGasConstant'], 
                                                          self.Fluid['Gamma']
                                                          )
        else:
            raise Exception(f'Either MassFlow or Mach must be provided for the FlowGenerator {self.name}')

        Mach = self.Flow['Mach']
        Temperature  = self.Flow['TemperatureStagnation'] / (1. + 0.5*(self.Fluid['Gamma']-1.) * Mach**2)
        Pressure  = self.Flow['PressureStagnation'] / (1. + 0.5*(self.Fluid['Gamma']-1.) * Mach**2)**(self.Fluid['Gamma']/(self.Fluid['Gamma']-1))
        Density = Pressure / (Temperature * self.Fluid['IdealGasConstant'])
        SoundSpeed  = np.sqrt(self.Fluid['Gamma'] * self.Fluid['IdealGasConstant'] * Temperature)
        Velocity  = Mach * SoundSpeed

        self.Flow.update(dict(
            Temperature = Temperature,
            Pressure = Pressure,
            Density = Density,
            SoundSpeed = SoundSpeed,
            Velocity = Velocity,
        ))

    @staticmethod
    def MassFlowFromMach(Mx, S, Pt=101325.0, Tt=288.25, r=287.053, gamma=1.4):
        '''
        Compute the massflow rate through a section.

        Parameters
        ----------

            Mx : :py:class:`float`
                Mach number in the normal direction to the section.

            S : :py:class:`float`
                Surface of the section.

            Pt : :py:class:`float`
                Stagnation pressure of the flow.

            Tt : :py:class:`float`
                Stagnation temperature of the flow.

            r : :py:class:`float`
                Specific gas constant.

            gamma : :py:class:`float`
                Ratio of specific heats of the gas.


        Returns
        -------

            massflow : :py:class:`float`
                Value of massflow through the section.
        '''
        return S * Pt * (gamma/r/Tt)**0.5 * Mx / (1. + 0.5*(gamma-1.) * Mx**2) ** ((gamma+1) / 2 / (gamma-1))

    def MachFromMassFlow(self, massflow, S, Pt=101325.0, Tt=288.25, r=287.053, gamma=1.4):
        '''
        Compute the Mach number normal to a section from the massflow rate.

        Parameters
        ----------

            massflow : :py:class:`float`
                MassFlow rate through the section.

            S : :py:class:`float`
                Surface of the section.

            Pt : :py:class:`float`
                Stagnation pressure of the flow.

            Tt : :py:class:`float`
                Stagnation temperature of the flow.

            r : :py:class:`float`
                Specific gas constant.

            gamma : :py:class:`float`
                Ratio of specific heats of the gas.


        Returns
        -------

            Mx : :py:class:`float`
                Value of the Mach number in the normal direction to the section.
        '''
        if isinstance(massflow, (list, tuple, np.ndarray)):
            Mx = []
            for i, MF in enumerate(massflow):
                Mx.append(self.MassFlowFromMach(MF, S, Pt=Pt, Tt=Tt, r=r, gamma=gamma))
            if isinstance(massflow, np.ndarray):
                Mx = np.array(Mx)
            return Mx
        else:
            # Check that massflow is lower than the chocked massflow
            chocked_massflow = self.MassFlowFromMach(1., S, Pt=Pt, Tt=Tt, r=r, gamma=gamma)
            assert massflow < chocked_massflow, "MassFlow ({:6.3f}kg/s) is greater than the chocked massflow ({:6.3f}kg/s)".format(massflow, chocked_massflow)
            # MassFlow as a function of Mach number
            f = lambda Mx: self.MassFlowFromMach(Mx, S, Pt, Tt, r, gamma)
            # Objective function
            g = lambda Mx: f(Mx) - massflow
            # Search for the corresponding Mach Number between 0 and 1
            Mx = scipy.optimize.brentq(g, 0, 1)
            return Mx
