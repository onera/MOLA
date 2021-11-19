'''
Template for use with COMPRESSOR WORKFLOW.

Produces the case main CGNS file (main.cgns)

MOLA 1.11 - 23/09/2021 - T. Bontemps - creation
'''

import sys, os
import numpy as np
from timeit import default_timer as tic

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.WorkflowCompressor as WF

toc = tic() # auxiliary variable used to log the script execution time

mesh = 'mesh.cgns' # name of the input CGNS. It shall be mesh.cgns

# This first dictionary contains parameters on the geometrical configuration.
TurboConfiguration = dict(
    # Shaft speed in rad/s
    # BEWARE: only for single shaft configuration
    ShaftRotationSpeed = 2211.11 * np.pi / 30.,

    # Hub rotation speed
    # List of tuples. Each tuple (xmin, xmax) corresponds to a CoordinateX
    # interval where the speed at hub wall is ShaftRotationSpeed. It is zero
    # outsides these intervals.
    HubRotationSpeed = [(-999.0, 0.45)],

    # This dictionary has one entry for each row domain.
    # The key names must be the family names in the CGNS Tree.
    Rows = dict(
        row_1 = dict(
            # For each row, set here the following parameters:
            # Rotation speed in rad/s (watch out for the sign)
            # Set 'auto' to automatically set ShaftRotationSpeed (for a rotor).
            RotationSpeed = 'auto',
            # The number of blades in the row
            NumberOfBlades = 12,
            # The number of blades in the computational domain
            # set to <NumberOfBlades> for a full 360 simulation
            NumberOfBladesSimulated = 1,
            # The positions (in CoordinateX) of the inlet and outlet planes for
            # this row. These planes are used for post-processing and convergence
            # monitoring.
            InletPlane = -0.17,
            OutletPlane = 0.45
            ),
        )
    )

fanRadius = 0.5 * 1.834

ReferenceValues = dict(
    # Here we state the flight conditions and reference quantities.
    # These variables are self-explanatory :
    MassFlow              = 151.95,
    TemperatureStagnation = 236.81,
    PressureStagnation    = 28536.14,
    Surface               = np.pi * fanRadius**2,
    TurbulenceLevel       = 0.001,
    Viscosity_EddyMolecularRatio = 0.1,

    # This macro-key assures coherent turbulence modeling.
    # Most keys follow the NASA convention: https://turbmodels.larc.nasa.gov/
    # Possible values are :
    # 'SA', 'BSL','BSL-V','SST-2003','SST','SST-V','Wilcox2006-klim',
    # 'SST-2003-LM2009', 'SSG/LRR-RSM-w2012', 'smith'
    TurbulenceModel='Wilcox2006-klim',


    # Next dictionary is used for establishing the coprocessing options for the
    # simulation during the trigger call of coprocess.py script:
    CoprocessOptions=dict(

        # Following key states which Base in arrays.cgns is used for monitoring
        # convergence
        # To monitor convergence, use the three following keys to define a
        # convergence criterion (or several if these parameters are lists):
        #     * ConvergenceCriterionFamilyName: states which Base in arrays.cgns
        #       is used for monitoring convergence
        #     * ConvergenceFluxName: value in arrays.cgns to monitor to establish
        #       convergence
        #     * MaxConvergedCriterionStd: establishes the threshold of
        #       convergence of ConvergenceFluxName
        ConvergenceCriterionFamilyName = ['PERFOS_row_1'],
        ConvergenceFluxName = ['std-MassFlowIn'],
        MaxConvergedCriterionStd = [1e-2],

        # Following key establishes the number of iterations used for computing
        # the statistics of the arrays
        AveragingIterations = 1000,

        # These keys are used to determine the save frequency of the files
        # arrays.cgns, surfaces.cgns and fields.cgns
        UpdateArraysFrequency      = 1e20,
        UpdateSurfacesFrequency   = 50,
        UpdateFieldsFrequency     = 1000,
        ),
    )

NumericalParams = dict(
    # following key is the numerical scheme. Choose one of: 'jameson' 'ausm+' 'roe'
    NumericalScheme='roe',
    # following key is the time marching procedure. One of: 'steady' 'gear'
    TimeMarching='steady',
    # BEWARE: if "gear", then user shall indicate the timestep like this:
    timestep=0.01, # (only relevant if unsteady simulation)
    # following key states the initial iteration. Shall be 1 in general.
    inititer=1,
    # CFL ramp
    CFLparams=dict(vali=1.,valf=10.,iteri=1,iterf=300,function_type='linear'),
    # following key states the maximum number of iterations of the iterations.
    # It is recommended to use a VERY HIGH value, as the simulation will stop
    # safely before timeout (see CoprocessOptions)
    niter=1e6)

'''
List of boundary conditions to set on the given mesh
Each element is a dictionary with the following keys:
    * type : elsA BC type
    * option (optional) : add a specification to type
    * other keys depending on type. They will be passed as an
        unpacked dictionary of arguments to the BC type-specific
        function.
'''
BoundaryConditions = [
    dict(type='inj1', option='uniform', FamilyName='row_1_INFLOW'),
    dict(type='outpres', FamilyName='row_1_OUTFLOW', pressure=0.8*ReferenceValues['PressureStagnation'])
]


# The list Extractions gathers all the extractions that will be done in co-processing.
# Each element is a dictionary that triggers one type of extraction.
Extractions = [
    #
    dict(type='AllBCwall'),
    dict(type='BCInflowSubsonic'),
    dict(type='row_1_OUTFLOW'),
    dict(type='Sphere', radius=0.5, name='Sphere_example'),
    dict(type='Plane', name='Plane_example1', point=(0, 0.5, 0), normal=(0.2, 0.8, 0))
]
# Add several 'IsoSurface' extractions on 'ChannelHeight'
for h in [0.1, 0.5, 0.9]:
    Extractions.append(dict(type='IsoSurface', field='ChannelHeight', value=h))

# Get the positions of inlet and outlet planes for each row
for row, rowParams in TurboConfiguration['Rows'].items():
    try:
       Extractions.append(dict(type='IsoSurface', field='CoordinateX', value=rowParams['InletPlane'], ReferenceRow=row, tag='InletPlane'))
       Extractions.append(dict(type='IsoSurface', field='CoordinateX', value=rowParams['OutletPlane'], ReferenceRow=row, tag='OutletPlane'))
    except:
        pass

####################################################################################
# Call the macro function that makes all the modifications needed on the mesh
# file to generate the input 'main.cgns' for elsA
# Modification of this function call is NOT recommended.
WF.prepareMainCGNS4ElsA(mesh=mesh,
        ReferenceValuesParams=ReferenceValues,
        NumericalParams=NumericalParams,
        TurboConfiguration=TurboConfiguration,
        Extractions=Extractions,
        BoundaryConditions=BoundaryConditions)

print('Elaped time: %g minutes'%((tic()-toc)/60.))
