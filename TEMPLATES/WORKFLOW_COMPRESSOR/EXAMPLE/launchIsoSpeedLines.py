'''
Template for use with WORKFLOW COMPRESSOR
to launch multiple jobs

MOLA 1.11 - 26/10/2021 - T. Bontemps - creation
'''

import sys, os
import numpy as np

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.WorkflowCompressor as WF

FILE_MESH = 'mesh.cgns' # name of the input CGNS. It shall be mesh.cgns

TurboConfiguration = dict(
    # Shaft speed in rad/s
    # BEWARE: only for single shaft configuration
    ShaftRotationSpeed = -1800.,  # = 17188.7 rpm

    # Hub rotation speed
    # List of tuples. Each tuple (xmin, xmax) corresponds to a CoordinateX
    # interval where the speed at hub wall is ShaftRotationSpeed. It is zero
    # outsides these intervals.
    HubRotationSpeed = [(-999.0, 999.0)], # Here the whole hub is rotating (no stator part)

    # This dictionary has one entry for each row domain.
    # The key names must be the family names in the CGNS Tree.
    Rows = dict(
        R37 = dict(
            # For each row, set here the following parameters:
            # Rotation speed in rad/s (watch out for the sign)
            # Set 'auto' to automatically set ShaftRotationSpeed (for a rotor).
            RotationSpeed = 'auto',
            # The number of blades in the row
            NumberOfBlades = 36,
            # The number of blades in the computational domain
            # set to <NumberOfBlades> for a full 360 simulation
            # The default value is 1
            # If the value is >1, the mesh will be duplicated if it is not already
            # NumberOfBladesSimulated = 1,
            # The positions (in CoordinateX) of the inlet and outlet planes for
            # this row. These planes are used for post-processing and convergence
            # monitoring.
            InletPlane = -0.0419,
            OutletPlane = 0.08
            ),
        )
    )

ReferenceValues = dict(
    # Here we state the operating conditions and reference quantities.
    # They are used for the initialization and boundary conditions (depending on
    # user-provided parameters)
    # These variables are self-explanatory :
    MassFlow              = 20.5114,  # for the 360 degrees section, even it is simulated entirely
    TemperatureStagnation = 288.15,
    PressureStagnation    = 101330.,
    TurbulenceLevel       = 0.03,
    Viscosity_EddyMolecularRatio = 0.1,

    # This macro-key assures coherent turbulence modeling.
    # Most keys follow the NASA convention: https://turbmodels.larc.nasa.gov/
    # Possible values are :
    # 'SA', 'BSL','BSL-V','SST-2003','SST','SST-V','Wilcox2006-klim',
    # 'SST-2003-LM2009', 'SSG/LRR-RSM-w2012', 'smith'
    TurbulenceModel='smith',

    # Next dictionary is used for establishing the coprocessing options for the
    # simulation during the trigger call of coprocess.py script:
    CoprocessOptions=dict(
        # To monitor convergence, use the following list to define the
        # convergence criteria. Each element is a dictionary corresponding to
        # a criterion, with the following keys:
        #     * Family: states which Base in arrays.cgns is used for monitoring convergence
        #     * Variable: value in arrays.cgns to monitor to establish convergence
        #     * Threshold: establishes the threshold of convergence
        #     * Condition (optional): see documentation if needed
        ConvergenceCriteria = [
            dict(
                Family    = 'PERFOS_R37',
                Variable  = 'rsd-MassFlowIn',
                Threshold = 1e-4,
            ),
            dict(
                Family    = 'PERFOS_R37',
                Variable  = 'rsd-MassFlowOut',
                Threshold = 1e-4,
            ),
        ],

        # These keys are used to determine the save frequency of the files
        # arrays.cgns, surfaces.cgns and fields.cgns
        UpdateArraysFrequency     = 30,
        UpdateSurfacesFrequency   = 30,
        UpdateFieldsFrequency     = 1000,
        ),
    )

NumericalParams = dict(
    # Maximimum number of iterations. Normally, the simulation should end when
    # convergence criteria are satisfied or the time limit is reached.
    niter = 10000,
    # CFL ramp
    CFLparams=dict(vali=1.,valf=5.,iteri=1,iterf=1000,function_type='linear')
    )

Pt = ReferenceValues['PressureStagnation']
BoundaryConditions = [
    dict(type='InflowStagnation', option='uniform', FamilyName='R37_INFLOW'),
    dict(type='OutflowRadialEquilibrium', FamilyName='R37_OUTFLOW', valve_type=4, valve_ref_pres=0.75*Pt, valve_relax=0.1*Pt)
]

Initialization = dict(
    method = 'uniform',
    )

Extractions = [dict(type='IsoSurface', field='ChannelHeight', value=0.9)]

####################################################################################
PREFIX_JOB = 'run'
AER = '28771019F'
machine = 'sator'
DIRECTORY_WORK = '/tmp_user/sator/tbontemp/rafale_rotor37/'
ThrottleRange = [van*Pt for van in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]]
RotationSpeedRange = [-1800.]
NProc = 28

WF.launchIsoSpeedLines(PREFIX_JOB, AER, NProc, machine, DIRECTORY_WORK,
        ThrottleRange, RotationSpeedRange,
        mesh='mesh.cgns',
        ReferenceValuesParams=ReferenceValues,
        TurboConfiguration=TurboConfiguration,
        Extractions=Extractions,
        BoundaryConditions=BoundaryConditions,
        Initialization=Initialization)
