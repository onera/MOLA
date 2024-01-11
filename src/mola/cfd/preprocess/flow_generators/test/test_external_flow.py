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
from mola import misc
from mola.cfd.preprocess.flow_generators.external_flow import ExternalFlowGenerator

class FakeWorkflow():

    def __init__(self):
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
            Velocity=10
            )
        self.Turbulence = dict(
            Model='Wilcox2006-klim',
            Level=0.001,
            ReferenceVelocity='auto',
            Viscosity_EddyMolecularRatio=0.1,
            TurbulenceCutOffRatio=1e-8,
            TransitionMode=None
            )

def test_get_flow_directions():
    AngleOfAttackDeg = 15
    AngleOfSlipDeg   = 2
    YawAxis   = [1, 0, 0] 
    PitchAxis = [0, -1, 0]  
    workflow = FakeWorkflow()
    FlowGen = ExternalFlowGenerator(workflow)
    drag_dir, side_dir, lift_dir = FlowGen.get_flow_directions(AngleOfAttackDeg, AngleOfSlipDeg, YawAxis, PitchAxis) 
    assert np.allclose(drag_dir, [ 0.25881905, -0.03371033,  0.96533741]) 
    assert np.allclose(side_dir, [ 0.        , -0.99939083, -0.0348995 ]) 
    assert np.allclose(lift_dir, [ 0.96592583,  0.00903265, -0.25866138]) 

def test_ExternalFlowGenerator():
    workflow = FakeWorkflow()
    FlowGen = ExternalFlowGenerator(workflow)
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
    
    RefFlow = dict(
        AngleOfAttackDeg = 0.0,
        AngleOfSlipDeg = 0.0,
        Conservatives = dict(
            Density = 1.225,
            EnergyStagnationDensity = 253373.86097187505,
            MomentumX = -12.25,
            MomentumY = 0.0,
            MomentumZ = 0.0
        ),
        Density = 1.225,
        DragDirection = [-1.0, 0.0, 0.0],
        EnergyStagnationDensity = 253373.86097187505,
        LiftDirection = [0.0, 0.0, 1.0],
        Mach = 0.02938634853238396,
        MomentumX = -12.25,
        MomentumY = 0.0,
        MomentumZ = 0.0,
        PitchAxis = [0.0, -1.0, 0.0],
        Pressure = 101325.04438875,
        PressureDynamic = 61.25000000000001,
        ReferenceState = dict(
            Density = 1.225,
            EnergyStagnationDensity = 253373.86097187505,
            MomentumX = -12.25,
            MomentumY = 0.0,
            MomentumZ = 0.0,
            TurbulentDissipationRateDensity = 125.7942695235221,
            TurbulentEnergyKineticDensity = 0.00018375
        ),                   
        Reynolds = 684594.6640735897,
        SideDirection = [0.0, -1.0, 0.0],
        Temperature = 288.15,
        Velocity = 10,
        VelocityUsedForScalingAndTurbulence = 10,
        ViscosityEddy = 1.7893800000000003e-06,
        ViscosityMolecular = 1.78938e-05,
        YawAxis = [0.0, 0.0, 1.0]

    )

    RefTurbulence = {'Conservatives': {'TurbulentDissipationRateDensity': 125.7942695235221,
                   'TurbulentEnergyKineticDensity': 0.00018375},
                    'IntermittencyDensity': 1.225,
                    'Level': 0.001,
                    'Model': 'Wilcox2006-klim',
                    'MomentumThicknessReynoldsDensity': 1392.24582,
                    'ReferenceVelocity': 'auto',
                    'ReynoldsStressDissipationScale': 125.7942695235221,
                    'ReynoldsStressXX': 0.0001225,
                    'ReynoldsStressXY': 0.0,
                    'ReynoldsStressXZ': 0.0,
                    'ReynoldsStressYY': 0.0001225,
                    'ReynoldsStressYZ': 0.0,
                    'ReynoldsStressZZ': 0.0001225,
                    'TransitionMode': None,
                    'TurbulenceCutOffRatio': 1e-08,
                    'TurbulentDissipationRateDensity': 125.7942695235221,
                    'TurbulentEnergyKineticDensity': 0.00018375,
                    'TurbulentEnergyKineticPLSDensity': 4.061228067453527e-08,
                    'TurbulentLengthScaleDensity': 0.00027074853783023517,
                    'TurbulentSANuTilde': 3.605642728246728e-05,
                    'Viscosity_EddyMolecularRatio': 0.1}

    assert misc.allclose_dict(FlowGen.Fluid, RefFluid)
    assert misc.allclose_dict(FlowGen.Flow, RefFlow)
    assert misc.allclose_dict(FlowGen.Turbulence, RefTurbulence)

