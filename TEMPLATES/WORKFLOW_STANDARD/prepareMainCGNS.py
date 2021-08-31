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

import MOLA.Preprocess as PRE

toc = tic() # auxiliary variable used to log the script execution time

FILE_MESH = 'mesh.cgns' # name of the input CGNS. It shall be mesh.cgns

'''
It is NOT RECOMENDED to modify the order of the following function calls.

The first step is determining the reference values of the configuration
including the flight conditions, modeling parameters, and coprocess options.
All this is done in the call to the function PRE.computeReferenceValues
'''

Surface = 18.0 / 2.
MeanAeroChord = 1.499

FluidProperties = PRE.computeFluidProperties()

ReferenceValues = PRE.computeReferenceValues(FluidProperties,

    # Here we state the flight conditions and reference quantities.
    # These variables are self-explanatory :

    Density=0.904637,
    Temperature=268.338,
    Velocity=139.385 * (1.852/3.6), # CAS=120 => 139.385 kts
    AngleOfAttackDeg=0.0,
    Surface=Surface,
    Length=MeanAeroChord, 
    TorqueOrigin=[4.69554, 0, -0.365],

    # These two variables are used for computing the aerodynamic axis.
    # A positive rotation around pitch axis on a typical aircraft configuration
    # provokes a pitch-up movement, which increases Lift, and makes the airplane
    # go "up", as the pilot would pull the stick of the elevator control.
    # A positive rotation around YawAxis produces a rotation towards the left,
    # as the pilot would apply left rudder pedal.
    PitchAxis=[0.,1.,0.],
    YawAxis=[0.,0.,1.],
    
    # This macro-key assures coherent turbulence modeling.
    # Most keys follow the NASA convention: https://turbmodels.larc.nasa.gov/
    # Possible values are :
    # 'SA', 'BSL','BSL-V','SST-2003','SST','SST-V','Wilcox2006-klim',
    # 'SST-2003-LM2009', 'SSG/LRR-RSM-w2012'
    TurbulenceModel='Wilcox2006-klim',



    # Next dictionary is used for establishing the coprocessing options for the
    # simulation during the trigger call of coprocess.py script:
    CoprocessOptions=dict(

        # Following key states which BCWall Family Name is used for monitoring
        # convergence using standard deviation of Lift Coefficient.
        ConvergenceCriterionFamilyName='wallWING',

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
        UpdateLoadsFrequency      =   50,
        UpdateSurfacesFrequency   = 1000,
        UpdateFieldsFrequency     = 2000,

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

        # This key establishes the additional field extractions requested to
        # elsA (apart from the mandatory ones)
        FieldsAdditionalExtractions = 'Temperature',

                        )


'''
The following three function calls create the coherent set of elsA keys
of the CFD, Model and Numerics objects. They are stored in form of standard
Python dictionaries.
'''

elsAkeysCFD      = PRE.getElsAkeysCFD()
elsAkeysModel    = PRE.getElsAkeysModel(FluidProperties, ReferenceValues)
elsAkeysNumerics = PRE.getElsAkeysNumerics(ReferenceValues,
    # following key is the numerical scheme. Choose one of: 'jameson' 'ausm+'
    'jameson',
    # following key is the time marching procedure. One of: 'steady' 'gear'
    'steady',
    # BEWARE: if "gear", then user shall indicate the timestep like this:
    timestep=0.01, # (only relevant if unsteady simulation)

    # following key states the initial iteration. Shall be 1 in general.
    inititer=1,

    # following key states the maximum number of iterations of the iterations.
    # It is recommended to use a VERY HIGH value, as the simulation will stop
    # safely before timeout (see CoprocessOptions)
    Niter=30000)

t = C.convertFile2PyTree(FILE_MESH)    


AllSetupDics = dict(
FluidProperties=FluidProperties,
ReferenceValues=ReferenceValues, 
elsAkeysCFD=elsAkeysCFD,
elsAkeysModel=elsAkeysModel,
elsAkeysNumerics=elsAkeysNumerics)

'''
The following call is used for creating the main.cgns PyTree and setup.py file
Use initializeFlow=True for initialize the flows (recommended).
Use FULL_CGNS_MODE = True, if want to use elsA in full CGNS mode.
Full CGNS mode is NOT recommended if the computation is transitional or with
Overset meshes.
'''

t = PRE.newRestart(t, AllSetupDics, initializeFlow=True, FULL_CGNS_MODE=False)
to = PRE.newEndOfRunFromRestart(t)

'''
This last call is used for creating the OUTPUT/fields.cgns file.
Modification of this function call is NOT recommended.
'''
PRE.saveEndOfRunAndRestartUsingLinks(t, to, 'OUTPUT',
                               RestartFilename='main.cgns',
                               EndOfRunFilename='fields.cgns',
                               RestartFlowSolutionName='FlowSolution#Init',
                               EndOfRunFlowSolutionName='FlowSolution#Centers')

print('REMEMBER : configuration shall be run using %d procs'%ReferenceValues['NProc'])

print('Elaped time: %g minutes'%((tic()-toc)/60.))