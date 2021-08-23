'''
               ---------------------------------
               EXAMPLES OF EXTRUSION OF AIRFOILS
               ---------------------------------

In this script, we apply the extrusion function to generate 2D meshes
around airfoil shapes. 
Both "C"-shaped and "O"-shaped meshing stategies are shown.

File history:
06/03/2019 - L. Bernardos - Creation
'''

# System modules
import sys
import numpy as np

# Original Cassiopee modules
import Converter.PyTree as C
import Converter.Internal as I
import Generator.PyTree as G
import Transform.PyTree as T
import Geom.PyTree as D
import Post.PyTree as P

import MOLA.InternalShortcuts as J
import MOLA.Wireframe as W
import MOLA.GenerativeVolumeDesign as GVD


'''
Example 1 : O-Mesh around airfoil.
'''

# Build an airfoil using W.airfoil()
f = W.airfoil('NACA0010',ClosedTolerance=1.1)

# Close the airfoil to make "O"-shaped mesh:
f = C.convertArray2Tetra(f)
G._close(f,tol=1.e-8)


# Build a orthogonal distribution using W.linelaw():
NPts              = 300
FirstCellHeight   = 1e-6
LastCellHeight    = 10.
ExtrusionDistance = 100. 
dist = W.linelaw(P2=(-ExtrusionDistance,0,0),N=NPts,
    Distribution=dict(kind='tanhTwoSides',FirstCellHeight=FirstCellHeight, LastCellHeight=LastCellHeight),
    )

# Set smoothing parameters
nf, gf, nit, git = J.invokeFields(dist,['normalfactor','growthfactor','normaliters','growthiters'])
gf[:]  = 0.1
git[:] = 100
nf[:]  = np.linspace(1e4,1e5,NPts)
nit[:] = 50

growthEquation='nodes:dH={nodes:dH}*maximum(1.+tanh(-{nodes:growthfactor}*{nodes:divs}),0.5)*maximum(1.+tanh({nodes:growthfactor}*0.01*mean({nodes:vol})/{nodes:vol}),0.5)'
tExtru = GVD.extrude(f,[dist],[],ExtrusionAuxiliarCellType='ORIGINAL',
    growthEquation=growthEquation,
    printIters=True, # print iters interactively in terminal
    plotIters=True,  # plot iters interactively using CPlot
    )

C.convertPyTree2File(tExtru,'AirfoilExrusions_Omesh.cgns')

'''
Example 2 : C-Mesh around airfoil.
'''

# Build an airfoil using W.airfoil()
f = W.airfoil('NACA0010',ClosedTolerance=1.1)

# Build two traling edge lines
x,y,z = J.getxyz(f)
TE_1 = W.linelaw(P2=(x[0],y[0],z[0]),
    P1=(x[0]+ExtrusionDistance,y[0],z[0]), 
    N=100, 
    Distribution=dict(kind='tanhTwoSides',
                      LastCellHeight=abs(x[1]-x[0]),
                      FirstCellHeight=LastCellHeight))
TE_1[0] = 'TE_1'

TE_2 = I.copyTree(TE_1)
T._reorder(TE_2,(-1,2,3))
TE_2[0] = 'TE_2'

f = W.concatenate([TE_1,f,TE_2]) # curve to extrude


# set smoothing values
gf[:]  = 0.2
git[:] = 100
nf[:]  = np.linspace(1e4,1e5,NPts)
nit[:] = 100


tExtru = GVD.extrude(f,[dist],[],ExtrusionAuxiliarCellType='ORIGINAL',
    growthEquation=growthEquation,
    printIters=True, # print iters interactively in terminal
    plotIters=True,  # plot iters interactively using CPlot
    )

C.convertPyTree2File(tExtru,'AirfoilExrusions_Cmesh.cgns')