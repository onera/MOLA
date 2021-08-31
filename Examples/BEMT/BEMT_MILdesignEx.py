'''
       ----------------------------------------
       BEMT MINIMUM INDUCED LOSS DESIGN EXAMPLE
       ----------------------------------------

In this script it is shown how to design a propeller using
the fixed-point minimum-induced loss (MIL) strategy proposed
by Adkins and Liebeck.

The following steps are covered in this example script:

 * STEP 0 - Define some design parameters

 * STEP 1 - Creating an auxiliar "initial guess" propeller

 * STEP 2 - Make MIL design

 * STEP 3 - BEMT trimmed analysis of the new design

 * STEP 4 - Plot new design proposals

Previous knowledge required by the user:
-> BEMT_MinimalEx.py
-> BEMT_TrimEx.py

File history:
21/09/2020 - v1.3 - L. Bernardos - Minor bug for Python 3 compatibility
23/05/2020 - v1.2 - L. Bernardos - Adapted to MOLA v1.8
19/04/2020 - v1.1 - L. Bernardos - Minor enhancements
14/04/2020 - v1.0 - L. Bernardos - Adapted to MOLA v1.7
25/03/2020 - v0.1 - L. Bernardos - Creation
'''

import sys
import numpy as np

import Converter.PyTree as C

import MOLA.InternalShortcuts as J
import MOLA.PropellerAnalysis as PA
import MOLA.LiftingLine as LL

# --------------------------------------------------------- #
# STEP 0 - DEFINE SOME DESIGN PARAMETERS                    #
# --------------------------------------------------------- #

'''
Hereafter, some design parameters are presented. These are 
choice of the designer, and may be used inside for-loops in 
order to make parametric analysis or design cartographies.

BEWARE: Airfoil distribution information is provided in Step 1
'''

# Some geometry inputs
Rmin    = 0.15 # Blade root (meters)
Rmax    = 0.6  # Blade tip  (meters)
NBlades = 3    # Number of blades

# Operating conditions
Velocity    = 120.*1.852/3.6  # Propeller's advance velocity (m/s)
Temperature = 273.+15. # Temperature (Kelvin)
Density     = 1.225    # Air density (kg / m3)
Mtip        = 0.6      # Approximate Mach at blade tip
RPM         = (30/np.pi)*Mtip*np.sqrt(1.4*287.*Temperature)/Rmax


# Design objective
Constraint      = 'Thrust' # 'Thrust' (or 'Power')
ConstraintValue =  1200.   #  Newton  (or   Watt )


# --------------------------------------------------------- #
# STEP 1 - Creating an auxiliar "initial guess" propeller   #
# --------------------------------------------------------- #

'''
In order to make the MIL design, an initial guess must be
provided by the user. This initial guess is simply a 
LiftingLine object. 

This LiftingLine object provides the following information:
-> Dimensions (like minimum and maximum radius) of the blade
-> Discretization to be employed (NPts, distribution...)
-> Spanwise distribution of the airfoils through identifiers
-> Interpolation laws to be employed
-> Initial guesses of Chord and Twist
-> and other less-relevant details...

We have two options: 
  (a) We build the initial-guess Lifting Line from scratch
  (b) We load a previously-saved Lifting Line object - making
      sure that its characteristics (dimensions, airfoils...)
      correspond well to the design objectives.

Option (b) is straightforward.
Hence, option (a) is shown here.
'''



# We make a dummy distribution for Chord and Twist, for instance
ChordAndTwistDict = {
'RelativeSpan' : [0.,1.],
'Chord'        : [1.,1.],
'Twist'        : [1.,1.],
'InterpolationLaw'  :'interp1d_linear',
}

# HOWEVER, the airfoil spanwise distribution is important, as the
# newly designed propeller will yield exactly this distribution.
# Extreme care has to be employed for choosing airfoils data.
# For this example, for simplicity, only 2 airfoils are employed
PolarsDict = {
'RelativeSpan'      : [Rmin/Rmax,    1.  ],
'PyZonePolarNames'  : [  'OH312', 'OH310'],
'InterpolationLaw'  :'interp1d_linear',
             }

# Invoke the "initial guess" auxiliar Lifting Line object
NPts = 50 # Number of points used to discretize the blade
RootSegmentLength = 0.0500 * Rmax
TipSegmentLength  = 0.0016 * Rmax
BladeDiscretization = dict(P1=(Rmin,0,0),P2=(Rmax,0,0),N=NPts, kind='tanhTwoSides',FirstCellHeight=RootSegmentLength,LastCellHeight=TipSegmentLength)
LiftingLine = LL.buildLiftingLine(BladeDiscretization,
Polars=PolarsDict,
Chord =ChordAndTwistDict,
Twist =ChordAndTwistDict)

# Do not forget we also need the PyZonePolars compatible with
# provided airfoils identifiers in PolarsDict:
filenames = [
'HOST_Profil_OH312', 
'HOST_Profil_OH310',
]
PyZonePolars = [LL.convertHOSTPolarFile2PyZonePolar(fn) for fn in filenames]
PolarsInterpFuns = LL.buildPolarsInterpolatorDict(PyZonePolars)

# --------------------------------------------------------- #
# STEP 2 - Make MIL design                                  #
# --------------------------------------------------------- #

'''
Now everything is ready for launching the design routine. 
There are many modes available for design.

Two different usages are shown in current step:

  (2.a) - Fully automated search of each airfoil's maximum 
          efficiency condition. This produces the best 
          propulsive efficiency design possible. However, due
          to poorly discretized polar data, this usually 
          produces noisy distributions. Next option avoids
          this problem.

  (2.b) - User-defined distribution of desired angle of 
          attack. This smooth distribution will lead to 
          smooth geometry results. The desired angle of attack
          must be close to the airfoil's maximum efficiency
          condition.
'''

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   #
# (2.a) - Fully automated search of max(Cl/Cd)
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   #
# The design call goes like this:
print ('Launching (2.a) design function...')
ResultsDict = PA.designPropellerAdkins(
LiftingLine, # Initial guess -> It will be modified !
PolarsInterpFuns,

# These design parameters were defined in Step 0
NBlades=NBlades,Velocity=Velocity,RPM=RPM,Temperature=Temperature,Density=Density,
Constraint=Constraint,ConstraintValue=ConstraintValue,

AirfoilAim='maxClCd', # This means automatically look for
                      # max(Cl/Cd) condition

# Number of iterations where AoA search is done 
# (few are usually enough)
itMaxAoAsearch=3

)
print('Launching (2.a) design function... COMPLETED\n')

eta_Step2a = ResultsDict['PropulsiveEfficiency']

# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   #
# (2.b) - Educated-guess imposition of angle-of-attack
# -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   #

# The resulting blade of (2.a) is noisy, as one can check:
r, AoA, Chord = J.getVars(LiftingLine,['Span','AoA','Chord'])
AoA_Step2a, Chord_Step2a = AoA*1., Chord*1.  # (produce copies) 

# The chord "Chord_Step2a" is noisy because the efficient 
# angle-of-attack AoA_Step2a is also noisy.

# One may want to impose a new smooth AoA distribution, yielding
# close values to the most efficient condition.
# For example, let us impose a linear distribution of AoA based
# on the previous result:
AoA_Root = np.mean(AoA[:int(len(AoA)/4)])
AoA_Tip  = np.mean(AoA[int(3*len(AoA)/4):])


# # Override AoA values:
AoA[:] = np.linspace(AoA_Root,AoA_Tip,len(AoA))

# Now, the design call goes like this:
print ('Launching (2.b) design function...')
ResultsDict = PA.designPropellerAdkins(
LiftingLine, # Initial guess -> It will be modified !
PolarsInterpFuns,

# These design parameters were defined in Step 0
NBlades=NBlades,Velocity=Velocity,RPM=RPM,Temperature=Temperature,Density=Density,
Constraint=Constraint,ConstraintValue=ConstraintValue,

AirfoilAim='AoA', # This means use the prescribed AoA contained
                  # in LiftingLine object
)


print ('Launching (2.b) design function... COMPLETED\n')

print ('Step (2.a) propulsive efficiency : %g'%eta_Step2a)
print ('Step (2.b) propulsive efficiency : %g'%ResultsDict['PropulsiveEfficiency'])
print ('Absolute eta difference (2.b)-(2.a) : %g\n'%abs(ResultsDict['PropulsiveEfficiency']-eta_Step2a))

# --------------------------------------------------------- #
# STEP 3 - BEMT analysis of the new design                  #
# --------------------------------------------------------- #
'''
It is important to point out that PA.designPropellerAdkins() is
a design routine, not an analysis routine. This means that some
simplifications are made in the design model, which may lead to
small differences of the prediction of the aerodynamic 
characteristics between the design routine and the analysis one.

Hence, it is convenient to perform a post-design analysis of
the design of the propeller. For this, we simply call
PA.computeBEMT() with Trim condition:
'''

# Post-design analysis of propeller
print(ResultsDict)
print('Computing BEMT...')
ResultsDictPost = PA.computeBEMT(
LiftingLine, PolarsInterpFuns,
model='Adkins',
Constraint=Constraint,
ConstraintValue=ConstraintValue,
ValueTol=0.01,
AttemptCommandGuess=[
[ResultsDict['Pitch'][0]-2.,ResultsDict['Pitch'][0]+2.],
[ResultsDict['Pitch'][0]-5.,ResultsDict['Pitch'][0]+5.],
[ResultsDict['Pitch'][0]-7.,ResultsDict['Pitch'][0]+7.]],
CommandType='Pitch',
FailedAsNaN=True, 
)


print ('Performance prediction of DESIGN routine:')
print ('Thrust = %g N; Power = %g W; eta = %g'%(ResultsDict['Thrust'], ResultsDict['Power'], ResultsDict['PropulsiveEfficiency']))
print ('Performance prediction of ANALYSIS routine:')
print ('Thrust = %g N; Power = %g W; eta = %g'%(ResultsDictPost['Thrust'], ResultsDictPost['Power'], ResultsDictPost['PropulsiveEfficiency']))
# sys.exit()
'''
Hence, one may observe that small differences do exist.
We can even observe that the constraint value imposed in design
is not strictly verified in post-analysis BEMT computation.
'''


# --------------------------------------------------------- #
# STEP 4 - Plot new design proposals                        #
# --------------------------------------------------------- #
'''
In this step, we plot the Chord and angle-of-attack distributions
of designed propellers following step (2.a) in dashed line and
step (2.b) in solid line.

User may observe that (2.b) propeller is much smoother than 
(2.a). Moreover, the loss of propulsive efficiency due to the
choice of a user-provided smooth AoA distribution do not lead
to a significant loss of propulsive efficiency.
'''

import matplotlib.pyplot as plt

r, Chord, AoA = J.getVars(LiftingLine,['Span','Chord','AoA'])

fig, (ax1, ax2) = plt.subplots(2,1,dpi=150, sharex=True)

ax1.plot(r/r.max(),Chord_Step2a, '.--', color='C0')
ax1.plot(r/r.max(),Chord, '.-', color='blue')
ax1.set_ylabel('Chord (m)')

ax2.plot(r/r.max(),AoA_Step2a, '--', color='C0', )
ax2.plot(r/r.max(),AoA, '-', color='blue')
ax2.set_xlabel('$r/R$')
ax2.set_ylabel('$\\alpha$ (deg)')

ax2.plot([],[],'--',color='black',label='Step (2.a)')
ax2.plot([],[],'-',color='black',label='Step (2.b)')

ax2.legend(loc='lower left')

plt.tight_layout()
plt.show()
