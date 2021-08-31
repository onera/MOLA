import sys
import numpy as np

import PropellerAnalysis as PA
import Converter.PyTree as C
import Converter.Internal as I

import InternalShortcuts as J





# -------------------------------------------------------- #
#                        BLADE GEOMETRY

NBlades = 1
Nspan   = 51

GralSpan = np.array([2.83, 6.95, 7.69, 8.094, 8.192, 8.290, 8.388,8.430])

Rmax    = GralSpan.max()   # Blade  Tip radius at definition (meters)
Rmin    = GralSpan.min() # Blade Root radius at definition (meters)

GralRelSpan = GralSpan/Rmax


# PolarsDict = {
# 'RelativeSpan'      : [    0., 1.0],
# 'PyZonePolarNames'  : ['MyAnaliticPolar','MyAnaliticPolar'],
# 'InterpolationLaw'  :'interp1d_linear',
#              }

PolarsDict = {
'RelativeSpan'      : np.array([1.8,5.92,6.66,7.064,7.4])/7.4,
'PyZonePolarNames'  : ['OA312','OA312','OA309','OAxxx','OA407'],
'InterpolationLaw'  :'interp1d_linear',
             }


# Chord:
ChordDict = {
'RelativeSpan'      : GralRelSpan,
'Chord'             : [.420, .420, .420, .420, .39618, .32472, .20563, .140],
'InterpolationLaw'  :'interp1d_linear',
            }




# -------------------------------------------------------- #


# -------------------------------------------------------- #
#                         POLARS
filenames = [
'T3C309.OAA3FINE',
'T3C312.OAA3FINE',
'T3C407.OAA3FINE',
'T3Cxx2b.OAA3FINE',
]
PyZonePolars = [PA.convertHOSTPolarFile2PyZonePolar(fn) for fn in filenames]
C.convertPyTree2File(PyZonePolars,'Polars.cgns')
# -------------------------------------------------------- #

# -------------------------------------------------------- #
#                    FLIGHT CONDITIONS
Velocity    = 0.0
# MachTip     = 0.50
Pitch       = 0.
Density     = 1.058
Temperature = 279.65

# Gamma, Rgp = 1.4, 287.058
# SoundSpeed = np.sqrt(Gamma*Rgp*Temperature)
RPM = 279.177 #RPM
# -------------------------------------------------------- #


# -------------------------------------------------------- #

# DESIRED OBJECTIVE
DesiredCaxial_rR    = [0.725, 0.75, 0.775, 0.8, 0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975, 1.0]
DesiredCaxial_value = [0.3727272018682529, 0.3728515037357633, 0.37104763670212765, 0.365966642415575, 0.36060449800667643, 0.35143474185949036, 0.33912431679449845, 0.32098846402953407, 0.29135021790070864, 0.2436721826512079, 0.17994785368860333, 0.1360439557759109]

ModCaxial_value = [0.2850420009169359, 0.30365265280528736, 0.31198569804614806, 0.310696672995695, 0.3026217922021411, 0.28728590278719596, 0.26609303883182517, 0.23930133763478875, 0.20512340525959258, 0.16250898834609473, 0.1279101988064129, 0.1054763500779356]





# -------------------------------------------------------- #
#                    PREPARE BEMT

AnalyticalPolarsDict = {'MyAnaliticPolar':{'CL0':np.deg2rad(0.7)*2*np.pi}}

# Build new geometry
TwistDict = {
'RelativeSpan'      : GralRelSpan,
'Twist'             : np.array([12.45, 5.7689, 5.0139, 4.3887, 4.2079, 4.0272, 3.8464, 3.7689])-7.+3.93+0.1,
'InterpolationLaw'  :'interp1d_linear',
            }

LL = PA.buildLiftLine(
    np.linspace(Rmin,Rmax,Nspan),
            Polars=PolarsDict,
            Chord=ChordDict,
            Twist=TwistDict,
                )

Span,TwistGuess, = J.getVars(LL,['Span','Twist'])
OriginalTwist = TwistGuess.copy(order='F')
VariationTwistrR = 0.7


def costFunction(NewTwistVector,LL): 


    Span, Twist, = J.getVars(LL,['Span','Twist'])
    BoolSlice = Span/Span.max()>VariationTwistrR
    Twist[BoolSlice] = NewTwistVector[BoolSlice]
    Twist[np.logical_not(BoolSlice)] = OriginalTwist[np.logical_not(BoolSlice)]

    LL, res = PA.computeBladeElementMomentumTheoryAnalysisDrela(LL,NBlades=NBlades,Velocity=Velocity,RPM=RPM,Pitch=Pitch,
    PyZonePolars=PyZonePolars,
    # AnalyticalPolarsDict=AnalyticalPolarsDict,
    Temperature=Temperature, Density=Density, TipLosses='Adkins', DictOfEquations={})

    Span,Chord,Twist,AoA,Mach,Reynolds,Cl,Cd,phiRad = J.getVars(LL,['Span','Chord','Twist','AoA','Mach','Reynolds','Cl','Cd','phiRad'])

    Aim = np.interp(Span/Span.max(),DesiredCaxial_rR,DesiredCaxial_value)
    Actual = Cl*np.cos(phiRad)-Cd*np.sin(phiRad)

    residual = np.mean((Aim-Actual)**2) 

    print (Twist, residual)
    return Aim-Actual

import scipy.optimize

# # Using least squares
# OptRes = scipy.optimize.least_squares(costFunction, TwistGuess, max_nfev=50)
# print (OptRes)
# TwistSolution = OptRes.x

OptRes = scipy.optimize.root(costFunction, TwistGuess,args=(LL),options=dict(eps=0.001))
print(OptRes)
TwistSolution = OptRes.x

import matplotlib.pyplot as plt


# Original geometry
TwistDict = {
'RelativeSpan'      : Span/Span.max(),
'Twist'             : OriginalTwist,
'InterpolationLaw'  :'interp1d_linear',
            }

LL = PA.buildLiftLine(
    np.linspace(Rmin,Rmax,Nspan),
            Polars=PolarsDict,
            Chord=ChordDict,
            Twist=TwistDict,
                )



LL, res = PA.computeBladeElementMomentumTheoryAnalysisDrela(LL,NBlades=NBlades,Velocity=Velocity,RPM=RPM,Pitch=Pitch,
PyZonePolars=PyZonePolars,
# AnalyticalPolarsDict=AnalyticalPolarsDict,
 Temperature=Temperature, Density=Density, TipLosses='Adkins', DictOfEquations={})

Span,Chord,Twist,AoA,Mach,Reynolds,Cl,Cd,phiRad = J.getVars(LL,['Span','Chord','Twist','AoA','Mach','Reynolds','Cl','Cd','phiRad'])

print ('Actual Caxial')
Actual = Cl*np.cos(phiRad)-Cd*np.sin(phiRad)
print (Actual)

plt.plot(Span/Span.max(),Actual,label='Initial BEMT')

plt.plot(DesiredCaxial_rR,ModCaxial_value,label='Initial CFD')


# New geometry
TwistDict = {
'RelativeSpan'      : Span/Span.max(),
'Twist'             : TwistSolution,
'InterpolationLaw'  :'interp1d_linear',
            }

LL = PA.buildLiftLine(
    np.linspace(Rmin,Rmax,Nspan),
            Polars=PolarsDict,
            Chord=ChordDict,
            Twist=TwistDict,
                )
Span, Twist, = J.getVars(LL,['Span','Twist'])
BoolSlice = Span/Span.max()>VariationTwistrR
Twist[BoolSlice] = TwistSolution[BoolSlice]
Twist[np.logical_not(BoolSlice)] = OriginalTwist[np.logical_not(BoolSlice)]

LL, res = PA.computeBladeElementMomentumTheoryAnalysisDrela(LL,NBlades=NBlades,Velocity=Velocity,RPM=RPM,Pitch=Pitch,
PyZonePolars=PyZonePolars,
# AnalyticalPolarsDict=AnalyticalPolarsDict,
 Temperature=Temperature, Density=Density, TipLosses='Adkins', DictOfEquations={})
C.convertPyTree2File(LL,'LiftingLine.cgns')

Span,Chord,Twist,AoA,Mach,Reynolds,Cl,Cd,phiRad = J.getVars(LL,['Span','Chord','Twist','AoA','Mach','Reynolds','Cl','Cd','phiRad'])

print ('Actual Caxial')
Actual = Cl*np.cos(phiRad)-Cd*np.sin(phiRad)
print (Actual)

print ('\n\n')
print ("###########################")
print ("ABSOLUTE TWIST LAW PROPOSAL")
MyRelSpan = Span/Span.max()
print ('r/R (>=0.7):')
print (MyRelSpan[MyRelSpan>=0.7])
print ('Absolute Twist (deg) (r/R>=0.7):')
print (Twist[MyRelSpan>=0.7])



plt.plot(Span/Span.max(),Actual,label='New BEMT')
plt.plot(DesiredCaxial_rR,DesiredCaxial_value,'s',mfc='None',label='Desired')
plt.legend(loc='best')
plt.xlabel('r/R')
plt.ylabel('Caxial')
plt.show()

plt.clf()
plt.plot(Span/Span.max(),Twist, label='Result')
plt.plot(Span/Span.max(),OriginalTwist, label='Previous')
plt.legend(loc='best')
plt.xlabel('r/R')
plt.ylabel('Twist')
plt.show()

# -------------------------------------------------------- #
