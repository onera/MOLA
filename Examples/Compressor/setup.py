'''
setup.py file automatically generated in PREPROCESS
'''

PostParameters={'IsoSurfaces': {'ChannelHeight': [0.1, 0.5, 0.9],
                 'CoordinateX': [-0.2432, 0.0398]},
 'Variables': ['Pressure',
               'PressureStagnation',
               'TemperatureStagnation',
               'Entropy',
               'Radius',
               'rovr',
               'rovt',
               'rowt',
               'rowy',
               'rowz',
               'W',
               'V',
               'Vm',
               'a',
               'rvt',
               'Machr',
               'beta',
               'alpha',
               'phi']}

FluidProperties={'Gamma': 1.4,
 'Prandtl': 0.72,
 'PrandtlTurbulence': 0.9,
 'RealGas': 287.053,
 'SutherlandConstant': 110.4,
 'SutherlandTemperature': 288.15,
 'SutherlandViscosity': 1.78938e-05,
 'cp': 1004.6855000000002,
 'cv': 717.6325000000002}

elsAkeysModel={'cv': 717.6325000000002,
 'delta_compute': 'first_order_bl',
 'fluid': 'pg',
 'gamma': 1.4,
 'k_prod_compute': 'from_sij',
 'linearratiolim': 0.001,
 'phymod': 'nstur',
 'prandtl': 0.72,
 'prandtltb': 0.9,
 'pressratiolim': 0.001,
 'shearratiolim': 0.02,
 'suth_const': 110.4,
 'suth_muref': 1.78938e-05,
 'suth_tref': 288.15,
 'turbmod': 'smith',
 'visclaw': 'sutherland',
 'vortratiolim': 0.001,
 'walldistcompute': 'mininterf_ortho'}

elsAkeysNumerics={'.Solver#Function': {'function_type': 'linear',
                      'iterf': 1000,
                      'iteri': 1,
                      'name': 'f_cfl',
                      'valf': 10.0,
                      'vali': 1.0},
 'cfl_fct': 'f_cfl',
 'flux': 'roe',
 'freqcompres': 1,
 'global_timestep': 'inactive',
 'harten_type': 2,
 'implicit': 'lussorsca',
 'inititer': 1,
 'limiter': 'valbada',
 'misc_source_term': 'inactive',
 'multigrid': 'none',
 'muratiomax': 1e+20,
 'niter': 100000,
 'ode': 'backwardeuler',
 'residual_type': 'explimpl',
 'ssorcycle': 4,
 't_cutvar1': 4.95559791299361,
 't_cutvar2': 0.0007372722404233173,
 't_harten': 0.01,
 'time_algo': 'steady',
 'timestep_div': 'divided'}

elsAkeysCFD={'config': '3d', 'extract_filtering': 'inactive'}

ReferenceValues={'AngleOfAttackDeg': 0.0,
 'AngleOfSlipDeg': 0.0,
 'CoprocessOptions': {'AveragingIterations': 1000,
                      'ConvergenceCriterionFamilyName': 'row_1_Main_Blade',
                      'ItersMinEvenIfConverged': 1000,
                      'MaxConvergedCLStd': 1e-06,
                      'NewSurfacesFrequency': 500,
                      'SecondsMargin4QuitBeforeTimeOut': 900.0,
                      'TimeOutInSeconds': 53100.0,
                      'UpdateFieldsFrequency': 20,
                      'UpdateLoadsFrequency': 1,
                      'UpdateSurfacesFrequency': 10},
 'CoreNumberPerNode': 28,
 'Density': 1.067859061700879,
 'DragDirection': [1.0, 0.0, 0.0],
 'EnergyStagnationDensity': 225537.71262687253,
 'Fields': 'Density MomentumX MomentumY MomentumZ EnergyStagnationDensity TurbulentEnergyKineticDensity TurbulentLengthScaleDensity',
 'FieldsAdditionalExtractions': 'ViscosityMolecular Viscosity_EddyMolecularRatio Pressure Temperature PressureStagnation TemperatureStagnation Mach',
 'FieldsTurbulence': 'TurbulentEnergyKineticDensity TurbulentLengthScaleDensity',
 'FluxCoef': 0.00031585801138737615,
 'IntermittencyDensity': 1.067859061700879,
 'Length': 1.0,
 'LiftDirection': [0.0, 0.0, 1.0],
 'Mach': 0.5312701923070879,
 'Massflow': 36.0,
 'MomentumThicknessReynoldsDensity': 623.9517583263223,
 'MomentumX': 187.82758295018314,
 'MomentumY': 0.0,
 'MomentumZ': 0.0,
 'NProc': 8,
 'Pressure': 83607.6211667575,
 'PressureDynamic': 16518.659709978696,
 'PressureStagnation': 101325.0,
 'ReferenceState': [1.067859061700879,
                    187.82758295018314,
                    0.0,
                    0.0,
                    225537.71262687253,
                    4.95559791299361,
                    0.0007372722404233173],
 'ReferenceStateTurbulence': [4.95559791299361, 0.0007372722404233173],
 'Reynolds': 10957704.902357329,
 'ReynoldsStressDissipationScale': 6174.477679473449,
 'ReynoldsStressXX': 3.3037319419957396,
 'ReynoldsStressXY': 0.0,
 'ReynoldsStressXZ': 0.0,
 'ReynoldsStressYY': 3.3037319419957396,
 'ReynoldsStressYZ': 0.0,
 'ReynoldsStressZZ': 3.3037319419957396,
 'RotationSpeed': {'01_row_1': -1151.9173063162573},
 'SideDirection': [0.0, 1.0, 0.0],
 'Surface': 0.1916608,
 'Temperature': 272.75319055435654,
 'TemperatureStagnation': 288.15,
 'TorqueCoef': 0.00031585801138737615,
 'TorqueOrigin': [0.0, 0.0, 0.0],
 'TransitionMode': None,
 'TurbulenceCutoff': 1.0,
 'TurbulenceLevel': 0.01,
 'TurbulenceModel': 'smith',
 'TurbulentDissipationRateDensity': 6174.477679473449,
 'TurbulentEnergyKineticDensity': 4.95559791299361,
 'TurbulentEnergyKineticPLSDensity': 0.0034214484916487422,
 'TurbulentLengthScaleDensity': 0.0007372722404233173,
 'TurbulentSANuTilde': None,
 'Velocity': 175.8917348615393,
 'ViscosityEddy': 0.0008570571329666665,
 'ViscosityMolecular': 1.714114265933333e-05,
 'Viscosity_EddyMolecularRatio': 50.0}
