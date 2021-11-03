'''
             -------------------------------
             BEMT CODE USAGE MINIMAL EXAMPLE             
             -------------------------------

This script describes the procedure for performing BEMT 
computations using minimal input from user.

Nonetheless, in this script the geometry of the propeller
is produced and it is supposed that available airfoil polar
data exist in form of HOST-format file.

Thus, several steps are described in this script:

STEP 1 - Construction of the propeller's geometry

STEP 2 - Construction of the polar data objects

STEP 3 - BEMT computation

STEP 4 - Plotting results example

Previous knowledge required by the user:
-> None (this script is good start point) However,
   basic knowledge of Python, Numpy and Cassiopee is required.

File history
25/09/2020 - v1.3 - L. Bernardos - Adapted to MOLA v1.8
23/05/2020 - v1.2 - L. Bernardos - Adapted to MOLA v1.7
19/04/2020 - v1.1 - L. Bernardos - Minor enhancements
13/04/2020 - v1.0 - L. Bernardos - Adapted to MOLA v1.7
25/03/2020 - v0.2 - L. Bernardos - Functional
25/03/2020 - v0.1 - L. Bernardos - Creation
'''

# System modules
import numpy as np
import sys

# Cassiopee modules
import Converter.PyTree   as C
import Converter.Internal as I

# MOLA modules (recent version of Scipy is required)
import MOLA.InternalShortcuts  as J
import MOLA.PropellerAnalysis  as PA
import MOLA.LiftingLine  as LL

# -------------------------------------------------- #
#  STEP 1: CONSTRUCTION OF THE PROPELLER'S GEOMETRY  #
# -------------------------------------------------- #

'''
This step consists in producing the simplified geometry that
is required for executing the BEMT computation function.

Such geometry consists in a CGNS-structured object that we
call "LiftingLine". It may contain rich contextual-based
information, but keep in mind that "LiftingLine" object is
essentially a 1D (a curve) PyTree Zone.

The LiftingLine object is built via the following function:

PA.buildLiftingLine(arguments...) 

Essentially, user shall know that in order to build the Lifting
Line object, three Python dictionaries shall be provided which
indicate the spanwise distributions of polars, chord and twist.
'''

# Construction of the distribution of Chord Python dictionary,
ChordDict = dict(

# The 'RelativeSpan' argument states the control points where
# the chord dimensions are specified. A value of 0 stands for
# the rotation axis, and a value of 1 stands for the blade tip
RelativeSpan = [0.2,   0.45,  0.6,  1.0],

# The 'Chord' argument states the value of chord (in meters) 
# defined at each point of 'RelativeSpan'. THEREFORE, the 
# number of elements in 'Chord' must be equal to the number of
# elements in 'RelativeSpan'
Chord =        [0.08,  0.12, 0.12, 0.03],

# The 'InterpolationLaw' argument specifies how to interpolate
# the Chord values between the control points used to define the
# geometry. 
InterpolationLaw = 'akima',

# NOTE: Many interpolation laws are available. 
# As scipy's interpolators are used, the available functions
# depend on the installation of scipy library on your system.
# Some expected available functions are:
# --> 'interp1d_<kind>' : Makes use of the function
#      scipy.interpolate.interp1d(), where <kind> may be
#      for scipy v1.4.1:
#      (linear, nearest, zero, slinear, quadratic, cubic,
#      previous or next).
#
# --> 'pchip' : Makes use of the function
#      scipy.interpolate.PchipInterpolator()
#
# --> 'akima' : Makes use of the function
#      scipy.interpolate.Akima1DInterpolator()
#
# --> 'cubic' : Makes use of the function
#      scipy.interpolate.CubicSpline
#      In this case, an additional keyword is available for
#      specification of boundary conditions, for example:
#
# 'CubicSplineBoundaryConditions' = ('clamped', 'not_a_knot')
#
#      The first element of the tupple indicates the boundary
#      condition of the spline at root and the second one
#      indicates the boundary condition at blade's tip.
)


# Next Python dictionary indicates the Twist geometry.
# Same explanation as Chord's distribution applies here.
# Only difference, is to define keyword 'Twist' instead
# of 'Chord'.
# Please note that the number of elements of newly defined
# 'RelativeSpans' do not need be equal to number of elements
# of Chord distribution (neither the interpolation law)
TwistDict = dict(
RelativeSpan = [0.2,  0.6,  1.0],

# 'Twist' is defined in degrees. Positive values means 
# an increase of incidence with respect to the rotation plane
Twist        = [30.,  6.0, -7.0],
InterpolationLaw = 'akima',
)


# The last python dictionary is used to define polars' spanwise
# position. Again, same behavior as previous dictionaries 
# applies here. Only difference, is that we specify a key
# 'PyZonePolarNames' with the identifier of each polar.

PolarsDict = dict(
RelativeSpan     = [  0.20,    0.500,    1.000],
PyZonePolarNames = ['OH312',  'OH310',  'OH309'],
InterpolationLaw = 'interp1d_linear', # 'interp1d_linear' is recommended for polars
)


# We are almost done. We only need to propose a discretization
# of our lifting line. This can be done via a PyTree Curve,
# a Python list, or a 1D numpy array.
# Let us use the last option:

Rmin = 0.15 # minimum radius of blade 
Rmax = 0.6  # maximum radius of blade
NPts =  50  # number of points discretizing the blade

# Non-uniform adapted discretization is recommended
RootSegmentLength = 0.0500 * Rmax
TipSegmentLength  = 0.0016 * Rmax
BladeDiscretization = dict(P1=(Rmin,0,0),P2=(Rmax,0,0),N=NPts, kind='tanhTwoSides',FirstCellHeight=RootSegmentLength,LastCellHeight=TipSegmentLength)

# Now we have everything ready for construction of the 
# LiftingLine object:

LiftingLine = LL.buildLiftingLine(
BladeDiscretization,
Polars=PolarsDict,
Chord =ChordDict,
Twist =TwistDict,
)

# [ Beware that LiftingLine is a 1D PyTree Zone ! ] We can, for example,
# give it a name, like this:
LiftingLine[0] = "MyFirstBlade"

# We might want to set the twist values of the blade such
# that the r/Rmax=0.75 station yields exactly a value of zero
# (this is, blade section situated at r/Rmax=0.75 is aligned
# with the rotation plane). This is quite conventional choice.
# For this, we use the following function:

LL.resetPitch(LiftingLine, ZeroPitchRelativeSpan=0.75)

 
#              --------------
#              IMPORTANT NOTE
#              --------------
# Please note that we can save the LiftingLine like this:
C.convertPyTree2File(LiftingLine,'MyBladeLL.cgns')

# And we can read it again like this:
t = C.convertFile2PyTree('MyBladeLL.cgns')
LiftingLine = I.getZones(t)[0]

# This is useful for sharing geometries without need to
# re-generate the geometry from zero.


# -------------------------------------------------- #
#  STEP 2: Construction of the polar data objects    #
# -------------------------------------------------- #

'''
In this step, we are constructing the polar data objects
required by the BEMT computation function. Such objects are
dictionaries of polars-interpolation functions.

Polar data may come from many different sources (CFD, XFoil,
experimental data...) and may yield many different formats.

In this example, we suppose HOST-formatted polar files are
available for each identifier of the airfoil.

Hence, we are translating HOST-formatted polar files into
CGNS PyZonePolar objects, and then to polar interpolation
functions.

!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
VERY IMPORTANT NOTE: the first line of the HOST-formatted file
must be exactly identical to the identifier of the airfoil
used in LiftingLine object. This means, for example, that
polar file HOST_Profil_OH312, which corresponds to the airfoil
identifier OH312, must contain in its first line exactly the
characters OH312 without spaces. If the line reads for example

(line 1): " 72   Profil OH312" 

this is not valid !, it should read exactly:

(line 1): "OH312"

that is valid.

Furthermore, the aero coefficients must be labeled exactly
Cl, Cd and Cm. Thus, Cz, Cx are NOT valid.

PLEASE PREPARE YOUR HOST-FORMATTED FILES ACCORDINGLY BEFOREHAND
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
'''

# We list the HOST files in relative or absolute path:
filenames = [
'HOST_Profil_OH312', 
'HOST_Profil_OH310',
'HOST_Profil_OH309',
]

# We create a list of PyZonePolars
PyZonePolars = [LL.convertHOSTPolarFile2PyZonePolar(fn) for fn in filenames]

# Beware that PyZonePolars is Python list, where each element
# is a PyTree Zone. This is the CGNS version of the HOST polars
# and is required for practical use in BEMT.

# Please note that we can save the polars for re-use or sharing
# with friends (instead of using HOST-format file):
C.convertPyTree2File(PyZonePolars,'MyPolars.cgns')

# And we can re-use them by reading simply like this:
t = C.convertFile2PyTree('MyPolars.cgns')
PyZonePolars = I.getZones(t)

# We are almost done with polars. Next step consists in
# building the interpolators objects. This is simply done
# like this:
PolarsInterpFuns = LL.buildPolarsInterpolatorDict(PyZonePolars, InterpFields=['Cl', 'Cd'])

'''
------------------------------------------------------
FOR YOUR CUROISITY (information not needed for users)
------------------------------------------------------

PolarsInterpFuns is a Python dictionary.

The keys correspond to the airfoil's identifier, which 
coincides with the Zone names of PyZonePolars.

The values are callable functions which accepts three
attributes as inputs: angle-of-attack, Mach and Reynolds.

Each attribute must be a 1D numpy array containing at least
one element.

The result of such call is a list of fields defined with
the key "InterpFields" when the object was produced (in
the same order).

For example, if we wish to know the Cl and Cd of airfoil
OH312 at AoA=5., Mach=0.5, Reynolds=1.e6, then we do:

Cl, Cd = PolarsInterpFuns['OH312'](np.array([5.0]), np.array([0.5]), np.array([1.e6]))
'''



# -------------------------------------------------- #
#  STEP 3 - BEMT computation                         #
# -------------------------------------------------- #

'''
In this step we call the BEMT function, introducing the 
physical parameters and other arguments related to the
method.

The output of the PA.computeBEMT() function is a simple
Python Dictionary with the integral quantities of the 
result.

Detailed sectional quantities are introduced in the 
"LiftingLine" object (here, variable LiftingLine).

HENCE, LiftingLine OBJECT IS MODIFIED IN-PLACE
'''




print ('Launching BEMT computation...')
ResultsDict = PA.computeBEMT(
# Provide the Blade geometry (Lifting Line is modified)
LiftingLine,

# Provide the polars interpolator functions dictionary
PolarsInterpFuns, 

# Provide the number of blades
NBlades=3, 

# The next two values define the kind of computation. 
# Constraint me be:
# 'Pitch'  -> We impose the value of blade pitch
# 'Thrust' -> We make a trimmed computation based on Thrust
# 'Power'  -> We make a trimmed computation based on Power
# ConstraintValue -> desired value of Pitch, Thrust or Power.
Constraint='Pitch', 
ConstraintValue=17.,

# Operating conditions
Velocity=35.,       # Propeller's advance velocity (m/s)
RPM=3200.,           # Propellers angular speed (rev per min.)
Temperature = 288., # Temperature (Kelvin)
Density=1.225,      # Air density (kg/m3)

# The <kind> argument stands for the algorithm to be employed
# Possible values are: "Drela", "Adkins" or "Heene". We strongly
# recommend "Drela". Beware that "Adkins" and "Heene" do not have
# trim-function implemented yet. 
model='Adkins',       # BEMT kind (Drela, Adkins or Heene)

# TipLosses -> The way to model the losses due to blade-tip vortex
# We recommend using "Adkins"
TipLosses='Adkins', 

# If FailedAsNaN=True, then poorly converged computations 
# results are overridden by NaNs
FailedAsNaN=True,
)
print('BEMT computation COMPLETED')
print("Thrust: %g N,  Power: %g W,  Prop. Eff.: %g, | Pitch: %g deg"%(ResultsDict['Thrust'],ResultsDict['Power'],ResultsDict['PropulsiveEfficiency'],ResultsDict['Pitch']))

'''
# Please note that LiftingLine object has been modified after
# BEMT computation. It may be possible to save sectional
# arrays simply making this:
'''
C.convertPyTree2File(LiftingLine,'MyBladeLL.cgns')


# -------------------------------------------------- #
# STEP 4 - PLOTTING RESULTS EXAMPLE
# -------------------------------------------------- #

'''
In this step is shown an example of exploitation of the
results produced by the PA.computeBEMT() function.

Let us plot sectional Angle-of-Attack and lift-coefficient
versus the spanwise dimensionless coordinate r/R

In addition, let us add some text into the plot, using the
integral quantities stored in ResultsDict Python dictionary.
'''

# Import the plotting library
import matplotlib.pyplot as plt

# get the variables to be plotted
r, AoA, Cl = J.getVars(LiftingLine,['Span','AoA', 'Cl'])

fig, ax1 = plt.subplots()

ax1.plot(r/r.max(),AoA, 'o-', color='blue')
ax1.set_xlabel('$r/R$')
ax1.set_ylabel('$\\alpha$ (deg)', color='blue')

ax2 = ax1.twinx()
ax2.plot(r/r.max(),Cl, 's--', color='red')
ax2.set_ylabel('$c_L$ (deg)', color='red')

plt.title('Thrust %g N; Power %g W; $\\eta=%0.3f$'%(ResultsDict['Thrust'],ResultsDict['Power'],ResultsDict['PropulsiveEfficiency']))

plt.tight_layout()
plt.show()

'''
NOTA BENE: For convenience, a shortcut has been
implemented in PropellerAnalysis.py module.
You may simply do this:
PA._plotAoAAndCl(LiftingLine)
'''
