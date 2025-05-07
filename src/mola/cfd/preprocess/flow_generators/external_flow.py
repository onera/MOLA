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
from mola import math_tools
from mola.logging import MolaUserAttributeError, MolaUserError

K_OMEGA_TWO_EQN_MODELS = ['Wilcox2006-klim', 'Wilcox2006-klim-V',
            'Wilcox2006', 'Wilcox2006-V', 'SST-2003', 
            'SST-V2003', 'SST', 'SST-V',  'BSL', 'BSL-V']

K_OMEGA_MODELS = K_OMEGA_TWO_EQN_MODELS + [ 'SST-2003-LM2009',
                 'SST-V2003-LM2009', 'SSG/LRR-RSM-w2012']

AvailableTurbulenceModels = K_OMEGA_MODELS + ['smith', 'SA']

class ExternalFlowGenerator(object):

    name = 'External_rho_T_V'

    def __init__(self, workflow):

        # Set attributes
        self.Fluid = workflow.Fluid if workflow.Fluid is not None else dict()
        self.Flow = workflow.Flow if workflow.Flow is not None else dict()
        self.Turbulence = workflow.Turbulence if workflow.Turbulence is not None else dict()

    def set_Flow_defaults(self,
            Direction              : Union [ list, tuple, np.ndarray ] = [1., 0., 0.],
            VelocityForScalingAndTurbulence : float = None,
            *,
            Velocity               : float = 1.0,
            Density                : float = 1.225,
            Temperature            : float = 288.15,
            ):
        # This function does nothing, but it is mandatory to be called by WorkflowInterface.
        # Its signature will be checked.
        return
    
    def _check_inputs(self):
        if len(self.Flow['Direction']) != 3:
            raise MolaUserAttributeError('Direction argument must be a 3-float list, tuple or numpy')
        self.Flow['Direction'] = np.array(self.Flow['Direction'], dtype=float)
        
        if not 'VelocityForScalingAndTurbulence' in self.Flow \
            or self.Flow['VelocityForScalingAndTurbulence'] is None:
            self.Flow['VelocityForScalingAndTurbulence'] = np.abs(self.Flow['Velocity'])
            if self.Flow['VelocityForScalingAndTurbulence'] < 1e-5:
                raise MolaUserError('Velocity is very low. You must set a positive value for VelocityForScalingAndTurbulence')
        elif self.Flow['VelocityForScalingAndTurbulence'] <= 0:
            raise MolaUserError('You must provide positive value for VelocityForScalingAndTurbulence')

        return self.Flow


    def generate(self):
        self.set_fluid_properties()
        self.set_flow_properties()
        self.set_turbulence_properties()
        self.Flow['ReferenceState'] = dict(**self.Flow['Conservatives'], **self.Turbulence['Conservatives'])
    
    def set_fluid_properties(self):
        self.Fluid['cv'] = self.Fluid['IdealGasConstant'] / (self.Fluid['Gamma']-1.0)
        self.Fluid['cp'] = self.Fluid['Gamma'] * self.Fluid['cv']

    def set_flow_properties(self):

        self._check_inputs()

        # TODO Put ViscosityMolecular in the Fluid attribute ?
        def SutherlandLaw(T, mus, Ts, S):
            return mus * (T/Ts)**1.5 * ((Ts + S)/(T + S))
       
        ViscosityMolecular = SutherlandLaw(self.Flow['Temperature'], self.Fluid['SutherlandViscosity'], self.Fluid['SutherlandTemperature'], self.Fluid['SutherlandConstant'])

        SoundSpeed = np.sqrt( self.Fluid['Gamma'] * self.Fluid['IdealGasConstant'] * self.Flow['Temperature'] )
        Mach = self.Flow['VelocityForScalingAndTurbulence'] / SoundSpeed 
        Pressure = self.Flow['Density'] * self.Fluid['IdealGasConstant'] * self.Flow['Temperature']
        PressureDynamic = 0.5 * self.Flow['Density'] * self.Flow['VelocityForScalingAndTurbulence'] **2
        TemperatureStagnation = self.Flow['Temperature'] * (1 + (self.Fluid['Gamma']-1)/2. * Mach**2)
        PressureStagnation = Pressure * (TemperatureStagnation/self.Flow['Temperature']) ** (self.Fluid['Gamma'] / (self.Fluid['Gamma']-1)) 

        Momentum_vector = self.Flow['Density'] * self.Flow['Velocity'] * np.array(self.Flow['Direction'])
        MomentumX =  Momentum_vector[0]
        MomentumY =  Momentum_vector[1]
        MomentumZ =  Momentum_vector[2]
        EnergyStagnationDensity = self.Flow['Density'] * ( self.Fluid['cv'] * self.Flow['Temperature'] + 0.5 * self.Flow['Velocity'] **2)

        self.Flow['Conservatives'] = dict(
            Density = self.Flow['Density'],
            MomentumX = float(MomentumX),
            MomentumY = float(MomentumY),
            MomentumZ = float(MomentumZ),
            EnergyStagnationDensity = float(EnergyStagnationDensity)
        )

        # Update ReferenceValues dictionary
        self.Flow.update(dict(
            Mach                    = Mach,
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
            SoundSpeed              = SoundSpeed
        ))

    def set_turbulence_properties(self):

        if self.Turbulence['Model'] == 'Euler':
            self.Turbulence['Conservatives'] = dict()
            return

        # -> for SA model
        def computeTurbulentSANuTilde(Nu, Nut_Nu):
            def computeEddyViscosityFromNuTilde(Nu, NuTilde):
                    '''
                    Compute cinematic ViscosityEddy using Eqn. (A1) of DOI:10.2514/6.1992-439
                    '''
                    Cnu1 = 7.1
                    f_nu1 = (NuTilde/Nu)**3 / ((NuTilde/Nu)**3 + Cnu1**3)
                    CinematicViscosityEddy = NuTilde * f_nu1
                    return CinematicViscosityEddy

            def residualEddyViscosityRatioFromGivenNuTilde(NuTilde):
                return Nut_Nu - computeEddyViscosityFromNuTilde(Nu, NuTilde) / Nu

            sol = math_tools.secant(
                residualEddyViscosityRatioFromGivenNuTilde, x0=Nut_Nu*Nu, x1=1.5*Nut_Nu*Nu, 
                ftol=Nut_Nu*1e-5, bounds=(1e-14,1.e6), maxiter=1000)
            return float(sol['root'])

        TurbulentSANuTilde = computeTurbulentSANuTilde(
                                                    Nu=self.Flow['ViscosityMolecular']/self.Flow['Density'],
                                                    Nut_Nu=self.Turbulence['Viscosity_EddyMolecularRatio']
                                                    )

        # -> for k-omega models
        TurbulentEnergyKineticDensity   = self.Flow['Density']*1.5*(self.Turbulence['Level']**2)*(self.Flow['VelocityForScalingAndTurbulence']**2)
        TurbulentDissipationRateDensity = self.Flow['Density'] * TurbulentEnergyKineticDensity / (self.Turbulence['Viscosity_EddyMolecularRatio'] * self.Flow['ViscosityMolecular'])
        
        # -> for Smith k-l model
        k = TurbulentEnergyKineticDensity / self.Flow['Density']
        omega = TurbulentDissipationRateDensity / self.Flow['Density']
        TurbulentLengthScaleDensity = self.Flow['Density'] * k * 18.0**(1./3.) / (np.sqrt(2*k)*omega)
        
        # -> for k-kL model
        TurbulentEnergyKineticPLSDensity = TurbulentLengthScaleDensity*k

        # -> for Menter-Langtry assuming acceleration factor F(lambda_theta)=1
        IntermittencyDensity = self.Flow['Density'] * 1.0
        if self.Turbulence['Level']*100 <= 1.3:
            MomentumThicknessReynoldsDensity = self.Flow['Density'] * (1173.51 - 589.428*(self.Turbulence['Level']*100) + 0.2196*(self.Turbulence['Level']*100)**(-2.))
        else:
            MomentumThicknessReynoldsDensity = self.Flow['Density'] * ( 331.50*(self.Turbulence['Level']*100-0.5658)**(-0.671) )

        # -> for RSM models
        ReynoldsStressXX = ReynoldsStressYY = ReynoldsStressZZ = (2./3.) * TurbulentEnergyKineticDensity
        ReynoldsStressXY = ReynoldsStressXZ = ReynoldsStressYZ = 0.

        self.Turbulence.update(dict(
            TurbulentSANuTilde               = TurbulentSANuTilde,
            TurbulentEnergyKineticDensity    = TurbulentEnergyKineticDensity,
            TurbulentDissipationRateDensity  = TurbulentDissipationRateDensity,
            TurbulentLengthScaleDensity      = TurbulentLengthScaleDensity,
            TurbulentEnergyKineticPLSDensity = TurbulentEnergyKineticPLSDensity,
            IntermittencyDensity             = IntermittencyDensity,
            MomentumThicknessReynoldsDensity = MomentumThicknessReynoldsDensity,
            ReynoldsStressXX                 = ReynoldsStressXX,
            ReynoldsStressYY                 = ReynoldsStressYY,
            ReynoldsStressZZ                 = ReynoldsStressZZ,
            ReynoldsStressXY                 = ReynoldsStressXY,
            ReynoldsStressXZ                 = ReynoldsStressXZ,
            ReynoldsStressYZ                 = ReynoldsStressYZ,
            ReynoldsStressDissipationScale   = TurbulentDissipationRateDensity,
        ))

        self.set_turbulence_conservatives_depending_on_model()
    
    def set_turbulence_conservatives_depending_on_model(self):

        if self.Turbulence['Model'] == 'SA':
            self.Turbulence['Conservatives'] = dict(
                TurbulentSANuTildeDensity = self.Turbulence['TurbulentSANuTilde'] * self.Flow['Density']
            )

        elif self.Turbulence['Model'] in K_OMEGA_TWO_EQN_MODELS:
            self.Turbulence['Conservatives'] = dict(
                TurbulentEnergyKineticDensity   = self.Turbulence['TurbulentEnergyKineticDensity'],
                TurbulentDissipationRateDensity = self.Turbulence['TurbulentDissipationRateDensity'],
            )

        elif self.Turbulence['Model'] == 'smith':
            self.Turbulence['Conservatives'] = dict(
                TurbulentEnergyKineticDensity   = self.Turbulence['TurbulentEnergyKineticDensity'],
                TurbulentLengthScaleDensity = self.Turbulence['TurbulentLengthScaleDensity'],
            )

        elif 'LM2009' in self.Turbulence['Model']:
            self.Turbulence['Conservatives'] = dict(
                TurbulentEnergyKineticDensity    = self.Turbulence['TurbulentEnergyKineticDensity'],
                TurbulentDissipationRateDensity  = self.Turbulence['TurbulentDissipationRateDensity'],
                IntermittencyDensity             = self.Turbulence['IntermittencyDensity'],
                MomentumThicknessReynoldsDensity = self.Turbulence['MomentumThicknessReynoldsDensity'],
            )

        elif self.Turbulence['Model'] == 'SSG/LRR-RSM-w2012':
            self.Turbulence['Conservatives'] = dict(
                ReynoldsStressXX               = self.Turbulence['ReynoldsStressXX'],
                ReynoldsStressXY               = self.Turbulence['ReynoldsStressXY'],
                ReynoldsStressXZ               = self.Turbulence['ReynoldsStressXZ'],
                ReynoldsStressYY               = self.Turbulence['ReynoldsStressYY'],
                ReynoldsStressYZ               = self.Turbulence['ReynoldsStressYZ'],
                ReynoldsStressZZ               = self.Turbulence['ReynoldsStressZZ'],
                ReynoldsStressDissipationScale = self.Turbulence['ReynoldsStressDissipationScale'],
            )

        else:
            raise AttributeError(f'Turbulence model {self.Turbulence["Model"]} not implemented in workflow. Must be in: {AvailableTurbulenceModels}')
            
