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

import pytest
import numpy as np
import time

from mola import misc
from mola.cfd.preprocess.flow_generators.internal_flow import InternalFlowGenerator

class FakeWorkflow():

    def __init__(self):

        self.ApplicationContext = dict(Surface = 0.1)

        self.Fluid = dict(
            Gamma=1.4,
            IdealGasConstant=287.053,
            Prandtl=0.72,
            PrandtlTurbulent=0.9,
            SutherlandConstant=110.4,
            SutherlandViscosity=1.78938e-05,
            SutherlandTemperature=288.15
            )

        self.Flow = dict(
            MassFlow=10.,
            PressureStagnation = 1e5,
            TemperatureStagnation = 300.,
            Direction = [1, 0, 0], # needed here because we don't use the WorkflowInterface for tests
            )
        
        self.Turbulence = dict(
            Model='Wilcox2006-klim',
            Level=0.001,
            ReferenceVelocity='auto',
            Viscosity_EddyMolecularRatio=0.1,
            )

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_MassFlowFromMach():
    workflow = FakeWorkflow()
    FlowGen = InternalFlowGenerator(workflow)
    Massflow = FlowGen.MassFlowFromMach(0.5, 0.01)
    np.testing.assert_allclose(Massflow, 1.80018457)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_MachFromMassFlow():
    workflow = FakeWorkflow()
    FlowGen = InternalFlowGenerator(workflow)
    Mach = FlowGen.MachFromMassFlow(1.80018457, 0.01)
    np.testing.assert_allclose(Mach, 0.5)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_InternalFlowGenerator_Mach_and_MassFlow():
    workflow = FakeWorkflow()
    workflow.Flow['MassFlow'] = 1.
    workflow.Flow['Mach'] = 0.5
    FlowGen = InternalFlowGenerator(workflow)
    try:
        FlowGen.generate()
    except AssertionError as e:
        assert e.args[0] == 'MassFlow and Mach cannot be given together in Flow. Choose one'
    else:
        raise AssertionError('Giving a Massflow and a Mach, InternalFlowGenerator should raise an error, and it does not.')
    

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_InternalFlowGenerator_without_Mach_and_MassFlow():
    workflow = FakeWorkflow()
    workflow.Flow.pop('MassFlow')
    assert ('MassFlow' not in workflow.Flow) and ('Mach' not in workflow.Flow)

    FlowGen = InternalFlowGenerator(workflow)
    try:
        FlowGen.generate()
    except Exception as e:
        assert e.args[0] == f'Either MassFlow or Mach must be provided for the FlowGenerator Internal'
    else:
        raise AssertionError('Giving no Massflow nor Mach, InternalFlowGenerator should raise an error, and it does not.')



@pytest.mark.unit
@pytest.mark.cost_level_0
def test_InternalFlowGenerator():
    workflow = FakeWorkflow()
    FlowGen = InternalFlowGenerator(workflow)

    FlowGen.generate()

    RefFluid = dict(
        Gamma=1.4,
        IdealGasConstant=287.053,
        Prandtl=0.72,
        PrandtlTurbulent=0.9,
        SutherlandConstant=110.4,
        SutherlandViscosity=1.78938e-05,
        SutherlandTemperature=288.15,
        cp=1004.6855,
        cv=717.6325,
        )
    
    RefFlow = {'Conservatives': {'Density': 1.123444231728095,
                   'EnergyStagnationDensity': 243137.62755937327,
                   'MomentumX': 99.99999999999996,
                   'MomentumY': 0.0,
                   'MomentumZ': 0.0},
                'Density': 1.123444231728095,
                'Direction': [1, 0, 0],
                'EnergyStagnationDensity': 243137.62755937327,
                'Mach': 0.25805710351588457,
                'MassFlow': 10.0,
                'MomentumX': 99.99999999999996,
                'MomentumY': 0.0,
                'MomentumZ': 0.0,
                'Pressure': 95474.81134338387,
                'PressureDynamic': 4450.599200913549,
                'PressureStagnation': 100000.0,
                'ReferenceState': {'Density': 1.123444231728095,
                                    'EnergyStagnationDensity': 243137.62755937327,
                                    'MomentumX': 99.99999999999996,
                                    'MomentumY': 0.0,
                                    'MomentumZ': 0.0,
                                    'TurbulentDissipationRateDensity': 8208.910340124454,
                                    'TurbulentEnergyKineticDensity': 0.013351797602740648},
                'SoundSpeed': 344.93134583598834,
                'Temperature': 296.056908704829,
                'TemperatureStagnation': 300.0,
                'Velocity': 89.01198401827102,
                'VelocityForScalingAndTurbulence': 89.01198401827102,
                'ViscosityEddy': 1.8272827182289063e-06,
                'ViscosityMolecular': 1.8272827182289062e-05}


    RefTurbulence = {'Conservatives': {'TurbulentDissipationRateDensity': 8208.910340124454,
                    'TurbulentEnergyKineticDensity': 0.013351797602740648},
                    'IntermittencyDensity': 1.123444231728095,
                    'Level': 0.001,
                    'Model': 'Wilcox2006-klim',
                    'MomentumThicknessReynoldsDensity': 1276.8249270420831,
                    'ReferenceVelocity': 'auto',
                    'ReynoldsStressDissipationScale': 8208.910340124454,
                    'ReynoldsStressXX': 0.008901198401827098,
                    'ReynoldsStressXY': 0.0,
                    'ReynoldsStressXZ': 0.0,
                    'ReynoldsStressYY': 0.008901198401827098,
                    'ReynoldsStressYZ': 0.0,
                    'ReynoldsStressZZ': 0.008901198401827098,
                    'TurbulentDissipationRateDensity': 8208.910340124454,
                    'TurbulentEnergyKineticDensity': 0.013351797602740648,
                    'TurbulentEnergyKineticPLSDensity': 3.691552321470374e-07,
                    'TurbulentLengthScaleDensity': 3.106138428002434e-05,
                    'TurbulentSANuTilde': 4.019588539898499e-05,
                    'Viscosity_EddyMolecularRatio': 0.1}

    assert misc.allclose_dict(FlowGen.Fluid, RefFluid)
    assert misc.allclose_dict(FlowGen.Flow, RefFlow)
    assert misc.allclose_dict(FlowGen.Turbulence, RefTurbulence)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_InternalFlowGenerator_with_incidence():
    workflow = FakeWorkflow()
    α = np.radians(30)
    workflow.Flow['Direction'] = [np.cos(α), np.sin(α), 0]
    FlowGen = InternalFlowGenerator(workflow)
    FlowGen.generate()
    np.testing.assert_allclose(FlowGen.Flow['Mach'], 0.25805710351588457/np.cos(α))    


if __name__=='__main__':
    test_InternalFlowGenerator()