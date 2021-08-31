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
import Post.PyTree as P
import Geom.PyTree as D
import Intersector.PyTree as XOR

# MOLA (>= v1.3)
# MOdules pour des Logiciels en Aerodynamique
import GenerativeShapeDesign as GSD
import Wireframe as W
import InternalShortcuts as J

def reOrientateAndOpenAirfoil(zoneFoil,maxTrailingEdgeThickness=0.01):
    FoilName = zoneFoil[0]
    I._rmNodesByName(zoneFoil,'FlowSolution')

    # Temporarily split upper and lower sides:
    # Find Trailing Edge
    isClockwise = W.is2DCurveClockwiseOriented(zoneFoil)
    if not isClockwise: T._reorder(zoneFoil,(-1,2,3))

    AirfoilX, AirfoilY = J.getxy(zoneFoil)
    iLE = np.argmin(AirfoilX)

    # Split sides
    Side1 = T.subzone(zoneFoil,(iLE+1,1,1),(-1,-1,-1))
    Side2 = T.subzone(zoneFoil,(1,1,1),(iLE+1,1,1))
    Side1[0] = 'Side1'
    Side2[0] = 'Side2'
    S1x, S1y = J.getxy(Side1)
    S2x, S2y = J.getxy(Side2)
    if S1x[2]-S1x[0]<0: T._reorder(Side1,(-1,2,3))
    if S2x[2]-S2x[0]<0: T._reorder(Side2,(-1,2,3))

    # Determine if rotation around +X is required
    val1 = np.trapz(S1x,S1y)
    val2 = np.trapz(S2x,S2y)
    if val1>val2:
        T._rotate(zoneFoil,(0,0,0),(1,0,0),180.)

    # Determine if rotation around +Y is required
    CamberLine = W.getCamberOptim(zoneFoil)
    CLx, CLy = J.getxy(CamberLine)
    RelThicknes, = J.getVars(CamberLine,['RelativeThickness'])
    ThicknessIndex = np.argmax(RelThicknes)
    Thickness = RelThicknes[ThicknessIndex]
    MaxThicknessLocation = CLx[ThicknessIndex]
    if MaxThicknessLocation>=0.5:
        T._rotate(zoneFoil,(0,0,0),(0,1,0),180.)

    # Open the airfoil
    # split upper and lower sides:
    # Find Trailing Edge
    isClockwise = W.is2DCurveClockwiseOriented(zoneFoil)
    if not isClockwise: T._reorder(zoneFoil,(-1,2,3))

    AirfoilX, AirfoilY, AirfoilZ = J.getxyz(zoneFoil)
    iLE = np.argmin(AirfoilX)
    iTE = np.argmax(AirfoilX)

    # Put airfoil at (0,0)
    T._translate(zoneFoil,(-AirfoilX[iLE],-AirfoilY[iLE],-AirfoilZ[iLE]))


    Xmax = AirfoilX[iTE]
    Xmin = AirfoilX[iLE]
    Step = 1e-4
    CurrentPos = Xmax
    while CurrentPos > Xmin:
        CurrentPos -= Step
        Slice = P.isoLine(zoneFoil,'CoordinateX',CurrentPos)
        if C.getNPts(Slice) != 2:
            pass
        else:
            SliceX, SliceY = J.getxy(Slice)
            distance = ( (SliceX[1]-SliceX[0])**2 +
                         (SliceY[1]-SliceY[0])**2 ) **0.5 
            if distance >= maxTrailingEdgeThickness:
                break
    


    # Split Top and Bottom sides
    Side1 = T.subzone(zoneFoil,(iTE+1,1,1),(-1,-1,-1))
    Side2 = T.subzone(zoneFoil,(1,1,1),(iTE+1,1,1))
    Side1[0] = 'Side1'
    Side2[0] = 'Side2'
    S1x, S1y = J.getxy(Side1)
    S2x, S2y = J.getxy(Side2)
    if S1x[2]-S1x[0]<0: T._reorder(Side1,(-1,2,3))
    if S2x[2]-S2x[0]<0: T._reorder(Side2,(-1,2,3))
    S1x, S1y = J.getxy(Side1)
    S2x, S2y = J.getxy(Side2)

    iCutSide1 = np.where(S1x>SliceX[1])[0][0]
    iCutSide2 = np.where(S2x>SliceX[0])[0][0]

    Side1 = T.subzone(Side1,(1,1,1),(iCutSide1+1,1,1))
    Side2 = T.subzone(Side2,(1,1,1),(iCutSide2+1,1,1))

    S1x, S1y = J.getxy(Side1)
    S2x, S2y = J.getxy(Side2)

    S1x[-1] = SliceX[1]
    S1y[-1] = SliceY[1]

    S2x[-1] = SliceX[0]
    S2y[-1] = SliceY[0]

    NewFoil = T.join(Side2,Side1)

    NFx, NFy = J.getxy(NewFoil)
    if NFy[0] > NFy[-1]: T._reorder(NewFoil,(-1,2,3))
    

    NewFoil[0] = 'Or_%s'%FoilName

    return NewFoil







t = C.convertFile2PyTree('ScannedWingPrescribed.cgns')
Airfoils = I.getZones(I.getNodeFromName1(t,'Airfoils'))

NewFoils = [reOrientateAndOpenAirfoil(z) for z in Airfoils]

C.convertPyTree2File(NewFoils,'OrientatedFoils.cgns')

for f in NewFoils:
    C.convertPyTree2File(f,'%s.tp'%f[0])

