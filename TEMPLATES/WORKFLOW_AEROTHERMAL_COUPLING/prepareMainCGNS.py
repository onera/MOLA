'''
prepareMainCGNS.py template for use with WORKFLOW COMPRESSOR.

Produces the case main CGNS file (main.cgns)

MOLA 1.12 - 23/09/2021 - T. Bontemps - creation
'''
import sys, os
import numpy as np

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.WorkflowAerothermalCoupling as WAT

FILE_MESH = 'mesh.cgns' # name of the input CGNS. It shall be mesh.cgns

TurboConfiguration = dict(
    PeriodicTranslation = [0, 0.057074, 0],
    Rows = dict(
        T72 = dict()
        )
    )

ReferenceValues = dict(
    # Here we state the operating conditions and reference quantities.
    # They are used for the initialization and boundary conditions (depending on
    # user-provided parameters)
    # These variables are self-explanatory :
    MassFlow              = 0.14833125593441665,
    TemperatureStagnation = 348.6882213389165,
    PressureStagnation    = 102275.,
    TurbulenceLevel       = 0.01,
    Viscosity_EddyMolecularRatio = 0.1,
    AngleOfAttackDeg = 37.0,


    # This macro-key assures coherent turbulence modeling.
    # Most keys follow the NASA convention: https://turbmodels.larc.nasa.gov/
    # Possible values are :
    # 'SA', 'BSL','BSL-V','SST-2003','SST','SST-V','Wilcox2006-klim',
    # 'SST-2003-LM2009', 'SSG/LRR-RSM-w2012', 'smith'
    TurbulenceModel='smith', #'Wilcox2006-klim',

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
                Family    = 'PERFOS_T72',
                Variable  = 'rsd-MassFlowIn',
                Threshold = 1e-4,
            ),
        ],

        FirstIterationForAverage = 1000,

        # These keys are used to determine the save frequency of the files
        # arrays.cgns, surfaces.cgns and fields.cgns
        UpdateArraysFrequency     = 10,
        UpdateSurfacesFrequency   = 20,
        UpdateFieldsFrequency     = 500,

        # These keys are used to determine which surfaces are coupled with Zset
        # using CWIPI, and the frequency of the coupling with CWIPI
        CoupledSurfaces = ['T72_BLADE'],
        UpdateCWIPICouplingFrequency = 20,
        ),
    )

NumericalParams = dict(
    TimeMarching = 'UnsteadyFirstOrder',
    timestep = 1e-5,
    )

BoundaryConditions = [
    dict(type='InflowStagnation', option='uniform', FamilyName='T72_INFLOW'),
    dict(type='OutflowPressure', FamilyName='T72_OUTFLOW', Pressure=10316.),
    dict(type='WallViscousIsothermal', FamilyName='T72_BLADE', Temperature=300.)
]

Initialization = dict(
    method = 'uniform',
    # method = 'copy',
    # file = 'OUTPUT/tmp-fields.cgns'
    )

Extractions = [dict(type='AllBCwall')]

####################################################################################

WAT.prepareMainCGNS4ElsA(mesh='mesh.cgns',
        ReferenceValuesParams=ReferenceValues,
        NumericalParams=NumericalParams,
        TurboConfiguration=TurboConfiguration,
        Extractions=Extractions,
        BoundaryConditions=BoundaryConditions,
        Initialization=Initialization,
        )
