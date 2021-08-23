'''
        ----------------------------------------
               BEMT DIRECT DESIGN EXAMPLE             
        ----------------------------------------

Through this script, the user will learn how to make direct
blade design with constraints using a direct approach (i.e.
massive calls to BEMT analysis function).

Thus, several steps are described in this script:

STEP 1 - Create Lifting Line and set design constraints

STEP 2 - Set-up the design variables and its bounds

STEP 3 - Set the objective and launch optimization 

STEP 4 - Plot the result blade geometrical laws

Previous knowledge required by the user:
-> BEMT_MinimalEx.py
-> BEMT_TrimEx.py
-> BEMT_MILdesignEx.py

File history
25/04/2020 - v1.2 - L. Bernardos - Adapted to MOLA v1.8
19/04/2020 - v1.1 - L. Bernardos - Minor enhancements
15/04/2020 - v1.0 - L. Bernardos - Adapted to MOLA v1.7
'''

# System modules
import sys
import numpy as np

# Cassiopee
import Converter.PyTree as C
import Converter.Internal as I

# MOLA
import MOLA.InternalShortcuts as J
import MOLA.PropellerAnalysis as PA
import MOLA.LiftingLine as LL

# --------------------------------------------------------- #
# STEP 1 - CREATE LIFTING LINE AND SET DESIGN CONSTRAINTS   #
# --------------------------------------------------------- #
'''
The first step consists in creating a LiftingLine object that
will be used in the optimization process. 

Currently, the important things to declare here are:
-> The radial distribution of polars (PolarsDict)
-> The relative thicknesses of the airfoils used in the polars
-> The minimum allowable absolute thickness radial distribution

Other variables shall be declared, such as Chord and Twist. 
However, these values will be bypassed in a further step by the
InitialGuess of the optimization process. Hence, we can set
random values, as they will be overridden. Just to highlight
this point, we will set absurd values
'''

# Polars distribution
PolarsDict = dict(
RelativeSpan     = [  0.20,    1.000],
PyZonePolarNames = ['OH310', 'OH309'],
InterpolationLaw = 'interp1d_linear',
)

# Relative thickness distribution
RelThkDict = dict(
RelativeSpan      = [  0.20,  1.000],
RelativeThickness = [  0.10,  0.090],
InterpolationLaw  = 'akima',
)

# VERY IMPORTANT -> Minimum allowable absolute thickness
MinThkDict = dict(
RelativeSpan     = [  0.200,        1.000],

#                   10mm at root   5mm at tip (example) 
MinimumThickness = [  0.010,        0.005],
InterpolationLaw = 'interp1d_linear',
)


# We set random (here absurd) values to Chord and Twist,
# as anyways these values will be overridden
ChordDict = dict(RelativeSpan=[0.,1.],Chord=[-999.,-999.], InterpolationLaw='interp1d_linear')
TwistDict = dict(RelativeSpan=[0.,1.],Twist=[-999.,-999.], InterpolationLaw='interp1d_linear')

# We create a blade discretization
Rmin = 0.15 # minimum radius of blade 
Rmax = 0.6  # maximum radius of blade
NPts =  50  # number of points discretizing the blade
RootSegmentLength = 0.0500 * Rmax
TipSegmentLength  = 0.0016 * Rmax
BladeDiscretization = dict(P1=(Rmin,0,0),P2=(Rmax,0,0),N=NPts, kind='tanhTwoSides',FirstCellHeight=RootSegmentLength,LastCellHeight=TipSegmentLength)

# and we build the Lifting Line
LiftingLine = LL.buildLiftingLine(
BladeDiscretization,
Polars=PolarsDict,
RelativeThickness=RelThkDict,
MinimumThickness=MinThkDict,
Chord =ChordDict,
Twist =TwistDict,
)

# Do not forget the interpolator functions
filenames = [
'HOST_Profil_OH310',
'HOST_Profil_OH309',
]
PyZonePolars = [LL.convertHOSTPolarFile2PyZonePolar(fn) for fn in filenames]
PolarsInterpFuns = LL.buildPolarsInterpolatorDict(PyZonePolars)


# --------------------------------------------------------- #
# STEP 2 - SET-UP THE DESIGN VARIABLES AND ITS BOUNDS       #
# --------------------------------------------------------- #
'''
Now, we are setting the design variables and its bounds.

Currently, in the pre-built direct-design function only
Chord and Twist poles are supported. This does not mean that
other design variables cannot be used, but rather that if other
variables designs are wished, then the user has to construct
the optimization problem by him/herself. In such case, the user
may be interested in looking at the implementation of function
PA.directDesignBEMT().

In this example, we are using 7 design variables (4 chord poles
and 3 twist poles).
'''

# The definition of the poles is simply done using 2xN arrays
# VERY IMPORTANT ! -> Initial guess must NOT violate the 
# optimization bounds/constraints. Hence, double-check that
# initial guess values are within the allowable range of values
ChordPoles = np.array([
[    0.060, 0.060, 0.060, 0.060], # Initial guess value 
[Rmin/Rmax, 0.400, 0.750, 1.000], # Relative span position
])

TwistPoles = np.array([
[    15.00, 2.000, -1.00], # Initial guess value
[Rmin/Rmax, 0.500,  1.00], # Relative span position
])


# Set the bounds of each design variable value as pairs of
# minimum and maximum
# VERY IMPORTANT ! -> bound values should not violate the 
# optimization constraints. Hence, double-check that
# bounds values do not exceed the constraints
bounds= [
# Chord
[0.05, 0.2], 
[0.04, 0.2], 
[0.04, 0.2], 
[0.01, 0.2], 

# Twist
[-10., 30.], 
[-10., 30.], 
[-10., 30.], 
    ]


# --------------------------------------------------------- #
# STEP 3 - SET THE OBJECTIVE AND LAUNCH OPTIMIZATION        #
# --------------------------------------------------------- #
'''
Now, we set the trim constraint and its value,
and the name of the objective variable:
'''
Constraint      = 'Thrust'
ConstraintValue =  800.
Objective       = 'PropulsiveEfficiency' 

'''
We launch the optimization
'''
Results = PA.directDesignBEMT(LiftingLine, PolarsInterpFuns,
    # Conditions
    NBlades=3, 
    Velocity=35.,       
    RPM=3200.,          
    Temperature = 288., 
    Density=1.225,

    # Optimization parameters
    Constraint=Constraint,
    ConstraintValue=ConstraintValue,
    ValueTol=1., # Tolerance for trim

    # AcceptableTol is a criterion for determining
    # if an estimate is acceptable based on the value
    # of its trim. Here 50 means that trimmed values
    # of ConstraintValue +/- 50 Newton will be 
    # acceptable. Make sure AcceptableTol > ValueTol
    AcceptableTol=10.,
    Objective=Objective,
    ChordPairs=ChordPoles,
    TwistPairs=TwistPoles,
    bounds=bounds,
    AttemptCommandGuess=[[10.,25.],[5.,30.],],
    TwistScale=0.5,

    # <method> is passed to scipy's minimize function
    method='L-BFGS-B', # 'L-BFGS-B', 'TNC', 'SLSQP'...
    # <OptimOptions> is the dictionary of options
    # passed to scipy's minimize function, whose keys
    # depend on the chosen method
    OptimOptions=dict(eps=0.05),

    # use True for NOT running optimization (only 
    # performs initial guess)
    makeOnlyInitialGuess=False,

    # Save the print output to a file,
    stdout='logDesign.txt',
    )

# Save the result
C.convertPyTree2File(LiftingLine,'DirectDesign.cgns')


# --------------------------------------------------------- #
# STEP 4 - PLOT THE RESULT BLADE GEOMETRICAL LAWS           #
# --------------------------------------------------------- #
'''
We use the pre-built function for this.
'''
LiftingLine[0] = "MyDirectDesign" # Used for plot legend
PA._plotChordAndTwist([LiftingLine], savefigname='GeomLaws.pdf')

