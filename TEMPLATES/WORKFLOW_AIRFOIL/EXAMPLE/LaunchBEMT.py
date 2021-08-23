import numpy as np
import Converter.PyTree   as C
import Converter.Internal as I
import MOLA.InternalShortcuts  as J
import MOLA.Wireframe  as W
import MOLA.PropellerAnalysis  as PA
import MOLA.LiftingLine  as LL


# Build a simple blade
ChordDict = dict(
RelativeSpan = [0.25,   0.45,  0.6,  1.0],
Chord =        [0.11,  0.12, 0.12, 0.03],
InterpolationLaw = 'akima', 
)


TwistDict = dict(
RelativeSpan = [0.25,  0.6,  1.0],
Twist        = [20.,  6.0, -1.0],
InterpolationLaw = 'akima',
)


PolarsDict = dict(RelativeSpan     = [  0.25,         1.000],
                  PyZonePolarNames = ['NACA4416','NACA4416'],
                  InterpolationLaw = 'interp1d_linear',)

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

LiftingLine = LL.buildLiftingLine(BladeDiscretization,
                                  Polars=PolarsDict,
                                  Chord =ChordDict,
                                  Twist =TwistDict,)
LL.resetPitch(LiftingLine, ZeroPitchRelativeSpan=0.75)

PyZonePolars = I.getZones( C.convertFile2PyTree('PolarCorrected.cgns') )
# PyZonePolars.extend(I.getZones( C.convertFile2PyTree('M14.cgns') ))


PolarsInterpFuns = LL.buildPolarsInterpolatorDict(PyZonePolars,
                                               InterpFields=['Cl', 'Cd','Cm'])


ResultsDict = PA.computeBEMTaxial3D(
LiftingLine,
PolarsInterpFuns, 

NBlades=3, 
Constraint='Pitch', 
ConstraintValue=10., # Pitch value (if Constraint is Pitch)

Velocity=[0,0,-15.0], # Propeller's advance velocity (m/s)
RPM=3000,             # Propellers angular speed (rev per min.)
Temperature = 288.,   # Temperature (Kelvin)
Density=1.225,        # Air density (kg/m3)
TipLosses='Adkins', 
FailedAsNaN=True,
)
print("Thrust: %g N,  Power: %g W,  Prop. Eff.: %g, | Pitch: %g deg"%(ResultsDict['Thrust'],ResultsDict['Power'],ResultsDict['PropulsiveEfficiency'],ResultsDict['Pitch']))

C.convertPyTree2File(LiftingLine,'LiftingLine.cgns')