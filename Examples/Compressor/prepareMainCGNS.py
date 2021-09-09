'''
prepareRestart.py template for use with STANDARD WORKFLOW.

Produces the case main CGNS file (main.cgns)

MOLA 1.10 - 05/03/2021 - L. Bernardos - creation
'''


import sys, os
import numpy as np
from timeit import default_timer as tic

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.WorkflowCompressor as WF

toc = tic() # auxiliary variable used to log the script execution time

FILE_MESH = 'mesh.cgns' # name of the input CGNS. It shall be mesh.cgns

'''
It is NOT RECOMENDED to modify the order of the following function calls.

The first step is determining the reference values of the configuration
including the flight conditions, modeling parameters, and coprocess options.
All this is done in the call to the function PRE.computeReferenceValues
'''

TurboConfiguration = dict(
    # Shaft speed in rad/s
    # BEWARE: only for single shaft configuration
    ShaftRotationSpeed = -11000. * np.pi / 30.,

    # Hub rotation speed
    # List of tuples. Each tuple (xmin, xmax) corresponds to a CoordinateX
    # interval where the speed at hub wall is ShaftRotationSpeed. It is zero
    # outsides these intervals.
    HubRotationSpeed = [(-999.0, 0.0742685)],

    # This dictionary has one entry for each row domain.
    # The key names must be the family names in the CGNS Tree.
    Rows = dict(
        row_1 = dict(
            # For each row, set here the following parameters:
            # Rotation speed in rad/s (watch out for the sign)
            # Set 'auto' to automatically set ShaftRotationSpeed (for a rotor).
            RotationSpeed = 'auto',
            # The number of blades in the row
            NumberOfBlades = 16,
            # The number of blades in the computational domain
            # set to <NumberOfBlades> for a full 360 simulation
            NumberOfBladesSimulated = 1,
            # The positions (in CoordinateX) of the inlet and outlet planes for
            # this row. These planes are used for post-processing and convergence
            # monitoring.
            PlaneIn = -0.2432,
            PlaneOut = 0.0398
            )
        )
    )

ReferenceValues = dict(
    # Here we state the flight conditions and reference quantities.
    # These variables are self-explanatory :
    Massflow              = 36.0,
    TemperatureStagnation = 288.15,
    PressureStagnation    = 101325.,
    Surface               = 0.1916608,
    TurbulenceLevel       = 0.01,
    Viscosity_EddyMolecularRatio = 50.,

    # This macro-key assures coherent turbulence modeling.
    # Most keys follow the NASA convention: https://turbmodels.larc.nasa.gov/
    # Possible values are :
    # 'SA', 'BSL','BSL-V','SST-2003','SST','SST-V','Wilcox2006-klim',
    # 'SST-2003-LM2009', 'SSG/LRR-RSM-w2012'
    TurbulenceModel='smith',

    # Next dictionary is used for establishing the coprocessing options for the
    # simulation during the trigger call of coprocess.py script:
    CoprocessOptions=dict(

        # Following key states which BCWall Family Name is used for monitoring
        # convergence using standard deviation of Lift Coefficient.
        ConvergenceCriterionFamilyName='row_1_Main_Blade',

        # MaxConvergedCLStd establishes the threshold of convergence of
        # standard deviation statistic of Lift Coefficient.
        MaxConvergedCLStd   = 1e-6,

        # Following key establishes the number of iterations used for computing
        # the statistics of the loads
        AveragingIterations = 1000,

        # Following key states the minimum number of iterations to perform
        # even if the CONVERGED criterion is satisfied
        ItersMinEvenIfConverged= 1000,

        # These keys are used to determine the save frequency of the files
        # loads.cgns, surfaces.cgns and fields.cgns
        UpdateLoadsFrequency      =   1,
        UpdateSurfacesFrequency   = 10,
        UpdateFieldsFrequency     = 20,

        # Following key establishes the timeout of the simulation (in seconds)
        # and SecondsMargin4QuitBeforeTimeOut is the margin (in seconds) with
        # respect to the timeout.
        # elsA will safely stop if
        # TimeOutInSeconds+SecondsMargin4QuitBeforeTimeOut elapsed time is
        # reached, even if the total number of iterations is not completed
        # these proposed values are OK for 15h job (example of SATOR prod job)
        TimeOutInSeconds       = 53100.0, # 14.75 h * 3600 s/h = 53100 s
        SecondsMargin4QuitBeforeTimeOut = 900.,

        ),
    )

NumericalParams = dict(
    # following key is the numerical scheme. Choose one of: 'jameson' 'ausm+'
    NumericalScheme='roe',
    # following key is the time marching procedure. One of: 'steady' 'gear'
    TimeMarching='steady',
    # BEWARE: if "gear", then user shall indicate the timestep like this:
    timestep=0.01, # (only relevant if unsteady simulation)
    # following key states the initial iteration. Shall be 1 in general.
    inititer=1,
    # CFL ramp
    CFLparams=dict(vali=1.,valf=10.,iteri=1,iterf=1000,function_type='linear'),
    # following key states the maximum number of iterations of the iterations.
    # It is recommended to use a VERY HIGH value, as the simulation will stop
    # safely before timeout (see CoprocessOptions)
    Niter=100000)

PostParameters = dict(
    IsoSurfaces   = dict(
        CoordinateX   = [-0.2432, 0.0398],
        ChannelHeight = [0.1, 0.5, 0.9]
        ),
    # Variables     = ['Pressure', 'PressureStagnation', 'TemperatureStagnation', 'Entropy',
    #                  'Radius', 'rovr', 'rovt', 'rowt', 'rowy', 'rowz', 'W', 'V', 'Vm', 'a', 'rvt', 'Machr', 'beta', 'alpha', 'phi']
    )

WF.prepareMainCGNS4ElsA(FILE_MESH='mesh.cgns', ReferenceValuesParams=ReferenceValues,
        NumericalParams=NumericalParams, TurboConfiguration=TurboConfiguration,
        PostParameters=PostParameters, BodyForceInputData=[],
        writeOutputFields=True)
