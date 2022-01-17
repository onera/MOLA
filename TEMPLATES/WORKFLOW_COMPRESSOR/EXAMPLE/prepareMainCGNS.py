'''
prepareMainCGNS.py template for use with WORKFLOW COMPRESSOR.

Produces the case main CGNS file (main.cgns)

MOLA 1.12 - 23/09/2021 - T. Bontemps - creation
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
                Threshold = 1e-2,
            ),
            dict(
                Family    = 'PERFOS_R37',
                Variable  = 'rsd-PressureStagnationRatio',
                Threshold = 1e-2,
            ),
            dict(
                Family    = 'PERFOS_R37',
                Variable  = 'rsd-EfficiencyIsentropic',
                Threshold = 1e-2,
                Condition = 'Sufficient',
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
    CFLparams=dict(vali=1.,valf=3.,iteri=1,iterf=1000,function_type='linear')
    )

BoundaryConditions = [
    dict(type='InflowStagnation', option='uniform', FamilyName='R37_INFLOW'),
    dict(type='OutflowPressure', FamilyName='R37_OUTFLOW', Pressure=0.5*ReferenceValues['PressureStagnation']),
    # dict(type='OutflowMassFlow', FamilyName='R37_OUTFLOW'),
    # dict(type='OutflowRadialEquilibrium', FamilyName='R37_OUTFLOW', valve_type=2)
]

Initialization = dict(
    method = 'uniform',
    )

Extractions = [dict(type='AllBCwall')]
for h in [0.1, 0.5, 0.9]:
    Extractions.append(dict(type='IsoSurface', field='ChannelHeight', value=h))

####################################################################################

WF.prepareMainCGNS4ElsA(mesh='mesh.cgns',
        ReferenceValuesParams=ReferenceValues,
        NumericalParams=NumericalParams,
        TurboConfiguration=TurboConfiguration,
        Extractions=Extractions,
        BoundaryConditions=BoundaryConditions,
        Initialization=Initialization
        )
