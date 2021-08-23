'''
Example of scanWing() function usage.
'''

# System modules
import sys
import numpy as np
import scipy.optimize as so

# Cassiopee (>= v2.9)
import Converter.PyTree as C
import Converter.Internal as I
import Transform.PyTree as T

# MOLA (>= v1.3)
# MOdules pour des Logiciels en Aerodynamique
import GenerativeShapeDesign as GSD

# Get the blade surface
t = C.convertFile2PyTree('PALE_ISOLEE_03.cgns')
PALE = C.extractBCOfName(t, 'FamilySpecified:PALE')
PALEu = C.convertArray2Hexa(PALE)
PALEu = T.join(PALEu)

C.convertPyTree2File(PALEu,'PaleCityAirbus.cgns')
sys.exit()

# Input the Maximum Span of the blade:
MaxSpan = 1395.-5.7 # mm


# --------------------------------------------------- #
# USAGE 1 -> Uniformily scan the blade
# --------------------------------------------------- #
'''
Nsections = 51
RelativeSpans4Scan = np.linspace(0.005,0.995, Nsections)

MyScan = GSD.scanWing(PALEu, RelativeSpans4Scan, ChordRelRef=0.25, ImposeMaxSpan=MaxSpan, DeltaTwistPos=None, pos=1)

# Save all data:
C.convertPyTree2File(MyScan,'ScannedWing.cgns')

# Save only linear data, in separate file:
LL= I.getNodeFromName(MyScan,'LiftLine')
C.convertPyTree2File(LL,'LiftLine.plt')
'''

# --------------------------------------------------- #
# USAGE 2 -> Scan blade only at prescribed thickness
# --------------------------------------------------- #

MyListOfDesiredRelThickness = [0.302, 0.26, 0.20]


def getSpecificRelativeThicknessFoil(DesiredRelativeThickness=0.12):
    def residual(x):
        MyScan = GSD.scanWing(PALEu, np.array([x,0.995]), ChordRelRef=0.25, ImposeMaxSpan=MaxSpan, DeltaTwistPos=None, pos=1)
        LL= I.getNodeFromName(MyScan,'LiftLine')
        RelThck = I.getNodeFromName(LL,'RelativeThickness')[1][0]
        Span = I.getNodeFromName(LL,'Span')[1][0]
        print('Searching RelThick=%g : got %g at span=%g'%(DesiredRelativeThickness, RelThck, Span))
        return DesiredRelativeThickness-RelThck
    sol = so.root_scalar(residual,bracket=(0.16,0.995), method='bisect', x0=0.5,options={'xtol':0.0001})
    return sol



Abscissas = []
for DesiredThickness in MyListOfDesiredRelThickness:
    sol = getSpecificRelativeThicknessFoil(DesiredThickness)
    print (sol)
    Abscissas += [sol.root]
Abscissas += [0.995]

print("Abscissas found:")
print(Abscissas)
MyScan = GSD.scanWing(PALEu, Abscissas, ChordRelRef=0.25, ImposeMaxSpan=MaxSpan, DeltaTwistPos=None, pos=1)

# Save all data
C.convertPyTree2File(MyScan,'ScannedWingPrescribed.cgns')

# Save only linear data
LL= I.getNodeFromName(MyScan,'LiftLine')
C.convertPyTree2File(LL,'LiftLinePrescribed.plt')
