
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


# Construction of the distribution of Chord Python dictionary,
ChordDict = dict(

RelativeSpan = [0.2,   0.45,  0.6,  1.0],
Chord =        [0.08,  0.12, 0.12, 0.03],
InterpolationLaw = 'akima', 
)


TwistDict = dict(
RelativeSpan = [0.2,  0.6,  1.0],
Twist        = [30.,  6.0, -7.0],
InterpolationLaw = 'akima',
)

DihedralDict = dict(
RelativeSpan    = [0.2,  1.0],
Dihedral        = [0,    0.0],
InterpolationLaw = 'interp1d_linear',
)

SweepDict = dict(
RelativeSpan    = [0.2, 1.0],
Sweep           = [0,   0.0],
InterpolationLaw = 'interp1d_linear',
)


PolarsDict = dict(
RelativeSpan     = [  0.20,    0.500,    1.000],
PyZonePolarNames = ['OH312',  'OH310',  'OH309'],
InterpolationLaw = 'interp1d_linear', # 'interp1d_linear' is recommended for polars
)


Rmin = 0.15 # minimum radius of blade 
Rmax = 0.6  # maximum radius of blade
NPts =  50  # number of points discretizing the blade

# Non-uniform adapted discretization is recommended
RootSegmentLength = 0.0500 * Rmax
TipSegmentLength  = 0.0016 * Rmax
BladeDiscretization = dict(P1=(Rmin,0,0),P2=(Rmax,0,0),
                           N=NPts, kind='tanhTwoSides',
                           FirstCellHeight=RootSegmentLength,
                           LastCellHeight=TipSegmentLength)

# Now we have everything ready for construction of the 
# LiftingLine object:

LiftingLine = LL.buildLiftingLine(
BladeDiscretization,
Polars=PolarsDict,
Chord =ChordDict,
Twist =TwistDict,
Dihedral=DihedralDict,
Sweep=SweepDict
)

# [ Beware that LiftingLine is a 1D PyTree Zone ! ] We can, for example,
# give it a name, like this:
LiftingLine[0] = "MyFirstBlade"

LL.resetPitch(LiftingLine, ZeroPitchRelativeSpan=0.75)

 
#              --------------
#              IMPORTANT NOTE
#              --------------
# Please note that we can save the LiftingLine like this:
C.convertPyTree2File(LiftingLine,'MyBladeLL.cgns')

# And we can read it again like this:
t = C.convertFile2PyTree('MyBladeLL.cgns')
LiftingLine = I.getZones(t)[0]

# We list the HOST files in relative or absolute path:
filenames = [
'HOST_Profil_OH312', 
'HOST_Profil_OH310',
'HOST_Profil_OH309',
]

# We create a list of PyZonePolars
PyZonePolars = [LL.convertHOSTPolarFile2PyZonePolar(fn) for fn in filenames]
C.convertPyTree2File(PyZonePolars,'MyPolars.cgns')

# And we can re-use them by reading simply like this:
t = C.convertFile2PyTree('MyPolars.cgns')
PyZonePolars = I.getZones(t)

PolarsInterpFuns = LL.buildPolarsInterpolatorDict(PyZonePolars,
                                            InterpFields=['Cl', 'Cd', 'Cm'])

print ('Launching BEMT computation...')



ResultsDict = PA.computeBEMT(
LiftingLine,
PolarsInterpFuns, 

NBlades=3, 
Constraint='Pitch', 
ConstraintValue=18.0,

Velocity=30,       # Propeller's advance velocity (m/s)
RPM=4600.,           # Propellers angular speed (rev per min.)
Temperature = 288., # Temperature (Kelvin)
Density=1.225,      # Air density (kg/m3)
model='Heene',       # BEMT kind (Drela, Adkins or Heene)
TipLosses='Adkins', 

FailedAsNaN=True,
)
print("Thrust: %g N,  Power: %g W,  Prop. Eff.: %g, | Pitch: %g deg"%(ResultsDict['Thrust'],ResultsDict['Power'],ResultsDict['PropulsiveEfficiency'],ResultsDict['Pitch']))

print('BEMT computation COMPLETED')
C.convertPyTree2File(LiftingLine,'BEMT.cgns')

ResultsDict = PA.computeBEMTaxial3D(LiftingLine, PolarsInterpFuns,
NBlades=3, 
Constraint='Pitch', 
ConstraintValue=18.0,
AttemptCommandGuess=[[16.,20.],[15.,21.]],

Velocity=[0.,0.,-30.],  # Propeller's advance velocity (m/s)
RPM=4600.,              # Propellers angular speed (rev per min.)
Temperature = 288.,     # Temperature (Kelvin)
Density=1.225,          # Air density (kg/m3)
model='Heene',          # BEMT kind (Drela, Adkins or Heene)
TipLosses='Adkins', 

FailedAsNaN=True,
)

print("Thrust: %g N,  Power: %g W,  Prop. Eff.: %g, | Pitch: %g deg"%(ResultsDict['Thrust'],ResultsDict['Power'],ResultsDict['PropulsiveEfficiency'],ResultsDict['Pitch']))
print('3D BEMT computation COMPLETED')

C.convertPyTree2File(LiftingLine,'BEMT3D.cgns')
sys.exit()

# Import the plotting library
import matplotlib.pyplot as plt

# get the variables to be plotted
r, AoA, Cl, Cd, va = J.getVars(LiftingLine,['Span','AoA', 'Cl', 'Cd',
                                            'VelocityInducedAxial'])

fig, ax1 = plt.subplots()

ax1.plot(r/r.max(),va, 'o-', color='blue')
ax1.set_xlabel('$r/R$')
ax1.set_ylabel('Velocity Induced Axial (m/s)', color='blue')

ax2 = ax1.twinx()
ax2.plot(r/r.max(),Cl/Cd, 's--', color='red')
ax2.set_ylabel('$c_L / c_D$', color='red')

plt.title('Thrust %g N; Power %g W; $\\eta=%0.3f$'%(ResultsDict['Thrust'],ResultsDict['Power'],ResultsDict['PropulsiveEfficiency']))

print(ResultsDict)

plt.tight_layout()
plt.show()

'''
NOTA BENE: For convenience, a shortcut has been
implemented in PropellerAnalysis.py module.
You may simply do this:
PA._plotAoAAndCl(LiftingLine)
'''
