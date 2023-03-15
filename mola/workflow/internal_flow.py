#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.


class InternalFlow(object):

    def __init__(self, Fluid, Flow, Turbulence, Surface):
        self.Fluid = Fluid
        self.Flow = Flow
        self.Turbulence = Turbulence
        self.Surface = Surface


    def set_flow_properties(self):
         # Compute variables
        assert not(MassFlow and Mach), 'MassFlow and Mach cannot be given together in ReferenceValues. Choose one'
        if self.Flow['MassFlow']:
            self.machFromMassFlow()
        else:
            self.massflowFromMach()
        Temperature  = self.Flow['TemperatureStagnation'] / (1. + 0.5*(self.Fluid['Gamma']-1.) * Mach**2)
        Pressure  = self.Flow['PressureStagnation'] / (1. + 0.5*(self.Fluid['Gamma']-1.) * Mach**2)**(self.Fluid['Gamma']/(self.Fluid['Gamma']-1))
        Density = Pressure / (Temperature * self.Fluid['IdealGasConstant'])
        SoundSpeed  = np.sqrt(self.Fluid['Gamma'] * self.Fluid['IdealGasConstant'] * Temperature)
        Velocity  = Mach * SoundSpeed

        # REFERENCE VALUES COMPUTATION
        mus = self.Fluid['SutherlandViscosity']
        Ts  = self.Fluid['SutherlandTemperature']
        S   = self.Fluid['SutherlandConstant']
        ViscosityMolecular = mus * (Temperature/Ts)**1.5 * ((Ts + S)/(Temperature + S))

        # ReferenceValues = PRE.computeReferenceValues(Fluid,
        #     Density=Density,
        #     Velocity=Velocity,
        #     Temperature=Temperature,
        #     AngleOfAttackDeg=AngleOfAttackDeg,
        #     AngleOfSlipDeg = 0.0,
        #     YawAxis=YawAxis,
        #     PitchAxis=PitchAxis,
        #     TurbulenceLevel=TurbulenceLevel,
        #     Surface=Surface,
        #     Length=Length,
        #     TorqueOrigin=TorqueOrigin,
        #     TurbulenceModel=TurbulenceModel,
        #     Viscosity_EddyMolecularRatio=Viscosity_EddyMolecularRatio,
        #     TurbulenceCutoff=TurbulenceCutoff,
        #     TransitionMode=TransitionMode,
        #     )

        # addKeys = dict(
        #     PressureStagnation = PressureStagnation,
        #     TemperatureStagnation = TemperatureStagnation,
        #     MassFlow = MassFlow,
        #     )

        # ReferenceValues.update(addKeys)

        # return ReferenceValues

        self.Flow = dict()

    def set_turbulence_properties(self):
        self.Turbulence = dict()

    def massflowFromMach(self):
        '''
        Compute the massflow rate through a section.
        '''
        S = self.Surface
        Mx = self.Flow['Mach']
        Pt = self.Flow['PressureStagnation']
        Tt = self.Flow['TemperatureStagnation']
        r = self.Fluid['IdealGasConstant']
        gamma = self.Fluid['Gamma']

        self.MassFlow = S * Pt * (gamma/r/Tt)**0.5 * Mx / (1. + 0.5*(gamma-1.) * Mx**2) ** ((gamma+1) / 2 / (gamma-1))
        return self.MassFlow

    def machFromMassFlow(self):
        '''
        Compute the Mach number normal to a section from the massflow rate.
        '''

        if isinstance(massflow, (list, tuple, np.ndarray)):
            Mx = []
            for i, MF in enumerate(massflow):
                Mx.append(self.machFromMassFlow())
            if isinstance(massflow, np.ndarray):
                Mx = np.array(Mx)
            return Mx
        else:
            # Check that massflow is lower than the chocked massflow
            chocked_massflow = massflowFromMach(1., S, Pt=Pt, Tt=Tt, r=r, gamma=gamma)
            assert massflow < chocked_massflow, "MassFlow ({:6.3f}kg/s) is greater than the chocked massflow ({:6.3f}kg/s)".format(massflow, chocked_massflow)
            # MassFlow as a function of Mach number
            f = lambda Mx: massflowFromMach(Mx, S, Pt, Tt, r, gamma)
            # Objective function
            g = lambda Mx: f(Mx) - massflow
            # Search for the corresponding Mach Number between 0 and 1
            Mx = scipy.optimize.brentq(g, 0, 1)
            return Mx

