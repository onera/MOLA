'''
             -------------------------------
                    BEMT TRIM EXAMPLE             
             -------------------------------

This script describes the procedure for performing BEMT 
trimmed computations.

This script is intended to be applied AFTER successful 
execution of script BEMT_MinimalEx.py, because this example
also shows how to re-use Polar and LiftingLine data.

Thus, several steps are described in this script:

STEP 1 - Load Lifting Line and Polars CGNS objects

STEP 2 - Launch BEMT computation based on Trim

Previous knowledge required by the user:
-> BEMT_MinimalEx.py

File history
23/04/2020 - v1.2 - L. Bernardos - Adapted to MOLA v1.8
19/04/2020 - v1.1 - L. Bernardos - Minor enhancements
13/04/2020 - v1.0 - L. Bernardos - Adapted to MOLA v1.7
'''

# System modules
import numpy as np
import sys

# Cassiopee modules
import Converter.PyTree   as C
import Converter.Internal as I

# MOLA modules (recent version of Scipy is required)
import MOLA.PropellerAnalysis  as PA
import MOLA.LiftingLine  as LL

# -------------------------------------------------- #
#  STEP 1: LOAD LIFTING LINE AND POLARS CGNS OBJECTS
# -------------------------------------------------- #

'''
We recycle the data generated in BEMTminimalExample.py
'''

# Load the polars
PyZonePolars = C.convertFile2PyTree('MyPolars.cgns')
PyZonePolars = I.getZones(PyZonePolars)

# Build the Polars interpolator functions
PolarsInterpFuns = LL.buildPolarsInterpolatorDict(PyZonePolars)

# Load the Lifting Line object
LiftingLine = C.convertFile2PyTree('MyBladeLL.cgns')
LiftingLine = I.getZones(LiftingLine)[0]

# We make sure LiftingLine twist is conventional
LL.resetPitch(LiftingLine,ZeroPitchRelativeSpan=0.75)

# -------------------------------------------------- #
#  STEP 2: LAUNCH BEMT COMPUTATION BASED ON TRIM
# -------------------------------------------------- #
'''
Now, we prepare the BEMT computation using trim constraints.

Currently, only "Drela" algorithm supports this functionality.

Trim can be done aiming Thrust or Power, and the controlled
parameter may be either Pitch or RPM 
'''

Constraint      = 'Thrust' # 'Thrust' (or 'Power')
ConstraintValue =  800.    #  Newton  (or  Watt)
CommandType     = 'Pitch'  # 'Pitch' or 'RPM'

'''
An important parameter is the "initial guess" for the Pitch.
As we might not know a-priori the best initial guess for a given
condition, we can provide a list of "initial guesses", and the
algorithm will try them one by one until convergence. Each
element of the list specifies a minimum/maximum pair for each
attempt. The number of outter elements of the list (three
in this example) is the total number of attempts for trimming 
'''

AttemptCommandGuess=[
[10.,25.], # First attempt bounds of pitch (min/max)
[5.,30.],  # Second attempt bounds of pitch (min/max)
[0.,40.],  # Third attempt bounds of pitch (min/max)
]

'''
IMPORTANT NOTE -> The trim will be performed sequentially for
each item provided in the list AttremptCommandGuess. The number
of allowed attempts for finding the Trimmed condition is, hence,
the number of elements contained in AttremptCommandGuess list.

If the first element in the list is a very good estimate of the 
pitch interval, then only one attempt would be required.
This is optimal and will lead to fastest convergence.

On the other hand, if the provided guesses are far from the 
solution, several attempts would be required, requiring more CPU
time, and could ultimately produce one of the following 
undesired effects:

(a) - Computation do not find Trimmed condition for any attempt.

(b) - Computation find a Trimmed condition, but far from the
      desired operational point (it may be, for example, in
      stalled condition)

For this reason, it is recommended to use a moderate value of
command for the first elements of the list, and then increase
progressively the bounds alternating low and high values, as in
this example.
'''

# Let us launch the computation:
print ('Launching trimmed BEMT computation for %s: %g...'%(Constraint,ConstraintValue))
ResultsDict = PA.computeBEMT(
LiftingLine, PolarsInterpFuns, # Mandatory inputs
Constraint=Constraint,
ConstraintValue=ConstraintValue,

# ConstrintValue tolerance for trim termination
# (in Newton for Thrust, in Watt for Power)
ValueTol=0.1,

AttemptCommandGuess=AttemptCommandGuess,

# Return NaNs if fails to trim (resulting constraint is
# out of ValueTol range). If FailedAsNaN=False, then 
# the closest constraintvalue found is returned
FailedAsNaN=True, 
)
print ("Thrust: %g N,  Power: %g W,  Prop. Eff.: %g, | Pitch: %g deg"%(ResultsDict['Thrust'],ResultsDict['Power'],ResultsDict['PropulsiveEfficiency'],ResultsDict['Pitch']))

'''
And that's all !

Please note the following VERY IMPORTANT NOTE:

Did you observe that something is "missing" in the call of
PA.computeBEMT() function ?...

Indeed, we did NOT stated the operating flight conditions,
like Velocity, Temperature, Density, RPM, NBlades... and this
did not seem to bother the computation, as it run ok !

Actually, this information was ALREADY CONTAINED in the
Lifting Line object saved in CGNS format (this information
was provided in BEMTminimalExample.py). Hence, all the "missing"
attributes are contained in the ".Conditions" node of the 
Lifting Line.

But what if we wanted to use different conditions ?
That is easy. Let us say that we want to change Density, and that
for the sake of variety, we want to trim based on Power using RPM
instead of Pitch.
Then, we only have to do the following call:
'''

Constraint      = 'Power'
ConstraintValue = 40000.
NewDensity      = 1.1
CommandType     = 'RPM'

'''
Since Trim command control is perfomed using RPM, user must
impose a value for the Pitch. This is done using the following
variable. This variable is only relevant if CommandType == 'RPM'
'''
PitchIfTrimCommandIsRPM = 18. # degrees


AttemptCommandGuess = [  # Now this list means min/max RPM values
[2600.,3200.],
[2500.,3500.],
]

print ('Launching trimmed BEMT computation for %s: %g and Density=%g ...'%(Constraint,ConstraintValue,NewDensity))
ResultsDict = PA.computeBEMT(
LiftingLine, PolarsInterpFuns, # Mandatory inputs

# LOOK HERE ! We change Density (it overrides Density value 
# contained in .Conditions node)
Density = NewDensity,

Constraint=Constraint,
ConstraintValue=ConstraintValue,

PitchIfTrimCommandIsRPM = PitchIfTrimCommandIsRPM,
# ConstrintValue tolerance for trim termination
# (in Newton for Thrust, in Watt for Power)
ValueTol=0.1,

AttemptCommandGuess=AttemptCommandGuess,
CommandType='RPM',

FailedAsNaN=True, # Retrun NaNs if fails to trim

)
print ("Thrust: %g N,  Power: %g W,  Prop. Eff.: %g, | RPM: %g"%(ResultsDict['Thrust'],ResultsDict['Power'],ResultsDict['PropulsiveEfficiency'],ResultsDict['RPM']))

'''
And that is pretty much everything you have to know about
how to make Trimmed BEMT computations
'''
