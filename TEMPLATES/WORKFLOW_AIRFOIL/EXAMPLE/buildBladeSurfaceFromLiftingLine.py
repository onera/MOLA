import numpy as np
import Converter.PyTree   as C
import Converter.Internal as I
import MOLA.InternalShortcuts  as J
import MOLA.LiftingLine  as LL


PyZonePolars = I.getZones( C.convertFile2PyTree('PolarCorrected.cgns') )
# PyZonePolars.extend(I.getZones( C.convertFile2PyTree('M14.cgns') ))


LiftingLine, = I.getZones( C.convertFile2PyTree('LiftingLine.cgns') )

surf = LL.postLiftingLine2Surface(LiftingLine, PyZonePolars, ChordRelRef=0.25,
                Variables=['Cp','delta','delta1','theta11','SkinFrictionX','s'])


Conditions = J.get(LiftingLine,'.Conditions')
RPM = Conditions['RPM']
Rho = Conditions['Density']
T   = Conditions['Temperature']

# Construct Pressure:
Span, Chord, U = J.getVars(LiftingLine,['Span','Chord','VelocityMagnitudeLocal'])
Pressure, = J.invokeFields(surf,['Pressure'])
Cp, = J.getVars(surf,['Cp'])
Omega = RPM*np.pi/30.
Pinf = Rho * 287.053 * T
for j in range(C.getNPts(LiftingLine)):
    Pressure[:,j] = 0.5 * Rho * U[j]**2 * Cp[:,j] + Pinf


# Adapt boundary-layer thicknesses scaling
BLvars = J.getVars(surf,['delta','delta1','theta11'])
for BLvar in BLvars:
    for j in range(C.getNPts(LiftingLine)):
        BLvar[:,j] *= Chord[j]


# modify name to avoid confusion with absolute axes frame
I._renameNode(surf, 'SkinFrictionX', 'SkinFrictionChordwise')

C.convertPyTree2File(surf,'surf.cgns')

