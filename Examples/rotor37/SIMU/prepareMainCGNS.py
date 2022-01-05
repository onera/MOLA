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
    ShaftRotationSpeed = -1800.,  # = 17188.7 rpm

    # Hub rotation speed
    # List of tuples. Each tuple (xmin, xmax) corresponds to a CoordinateX
    # interval where the speed at hub wall is ShaftRotationSpeed. It is zero
    # outsides these intervals.
    HubRotationSpeed = [(-999.0, 999.0)],

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
            NumberOfBladesSimulated = 1,
            # The positions (in CoordinateX) of the inlet and outlet planes for
            # this row. These planes are used for post-processing and convergence
            # monitoring.
            InletPlane = -0.0419,
            OutletPlane = 0.08
            ),
        )
    )


ReferenceValues = dict(
    # Here we state the flight conditions and reference quantities.
    # These variables are self-explanatory :
    Massflow              = 20.5114,
    TemperatureStagnation = 288.15,
    PressureStagnation    = 101330.,
    Surface               = 0.110506,
    TurbulenceLevel       = 0.03,
    Viscosity_EddyMolecularRatio = 0.1,

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
        ConvergenceCriterionFamilyName='PERFOS_R37',

        # MaxConvergedCLStd establishes the threshold of convergence of
        # standard deviation statistic of Lift Coefficient.
        MaxConvergedCLStd   = 1e-4,

        # These keys are used to determine the save frequency of the files
        # arrays.cgns, surfaces.cgns and fields.cgns
        UpdateArraysFrequency      = 1e20,
        UpdateSurfacesFrequency   = 20,
        UpdateFieldsFrequency     = 1000,
        ),
    )

NumericalParams = dict(
    # following key is the numerical scheme. Choose one of: 'jameson' 'ausm+' 'roe'
    NumericalScheme='roe',
    # CFL ramp
    CFLparams=dict(vali=1.,valf=3.,iteri=1,iterf=1000,function_type='linear')
    )


Extractions = [
    #
    dict(type='AllBCwall'),
    dict(type='BCInflowSubsonic'),
    dict(type='row_1_OUTFLOW'),
    dict(type='Sphere', radius=0.5, name='Sphere_example'),
    dict(type='Plane', name='Plane_example1', point=(0, 0.5, 0), normal=(0.2, 0.8, 0))
]

for h in [0.1, 0.5, 0.9]:
    Extractions.append(dict(type='IsoSurface', field='ChannelHeight', value=h))

# Get the positions of inlet and outlet planes for each row
for row, rowParams in TurboConfiguration['Rows'].items():
    try:
       Extractions.append(dict(type='IsoSurface', field='CoordinateX', value=rowParams['InletPlane'], ReferenceRow=row, tag='InletPlane'))
       Extractions.append(dict(type='IsoSurface', field='CoordinateX', value=rowParams['OutletPlane'], ReferenceRow=row, tag='OutletPlane'))
    except:
        pass

pref = 0.5*ReferenceValues['PressureStagnation']
fluxcoeff = TurboConfiguration['Rows']['R37']['NumberOfBlades']/TurboConfiguration['Rows']['R37']['NumberOfBladesSimulated']
mref = ReferenceValues['Massflow'] / float(fluxcoeff)

BoundaryConditions = [
    dict(type='inj1', option='uniform', FamilyName='R37_INFLOW'),
    dict(type='outpres', FamilyName='R37_OUTFLOW', pressure=pref),
    #dict(type='outradeqhyb', FamilyName='R37_OUTFLOW', valve_type=2, valve_ref_pres=pref, valve_ref_mflow=mref, valve_relax=0.1)
]

####################################################################################

WF.prepareMainCGNS4ElsA(mesh='mesh.cgns', ReferenceValuesParams=ReferenceValues,
        NumericalParams=NumericalParams, TurboConfiguration=TurboConfiguration,
        Extractions=Extractions, BodyForceInputData=[], BoundaryConditions=BoundaryConditions,
        writeOutputFields=True, bladeFamilyNames=['_R37'])