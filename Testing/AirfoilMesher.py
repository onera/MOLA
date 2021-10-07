'''
Example on how to create a mesh around a 2D airfoil using MOLA/Cassiopee.

L. Bernardos - 09/02/2021
'''

import Converter.PyTree as C
import Converter.Internal as I

# export PYTHONPATH=$PYTHONPATH:/home/lbernard/MOLA/v1.9
import MOLA.Wireframe as W
import MOLA.GenerativeShapeDesign as GSD

# ============================ USER PARAMETERS ============================ #

Chord        = 1.0
ReynoldsMesh = 350000.0
DeltaYPlus   = 0.8

Dir = '/home/ffalissa/H2T/ETUDES/MOTUS/FLUX_2/POLAIRES/PROFILS/'

GeomPath  = Dir+'Airfoil_20.tp' # must be placed in XY plane and best if clockwise
                              # oriented starting from trailing edge or 
                              # selig / lednicer ASCI format
MeshFilename = 'mesh.cgns'

# Mesh generation parameters
Sizes = dict(
Height                  = 100.*Chord,
Wake                    = 100.*Chord,
BoundaryLayerMaxHeight  = 1.00*Chord,
TrailingEdgeTension     = 0.50*Chord,
)

Cells = dict(  
TrailingEdge = 0.005*Chord,
LeadingEdge  = 0.0005*Chord,
Farfield     = 5.0*Chord,
WakeFarfieldAspectRatio = 0.002,
LEFarfieldAspectRatio   = 1.0,
FarfieldAspectRatio     = 0.05,
ClosedWakeAbscissaCtrl  = 0.50,
ClosedWakeAbscissaRatio = 0.25,
)

Points = dict(
Extrusion               = 500, 
Bottom=[{'NPts':150,'BreakPoint(x)':None,'JoinCellLength':None}],
Top   =[{'NPts':200,'BreakPoint(x)':None,'JoinCellLength':None}],

Wake                    = 300,
WakeHeightMaxPoints     = 50,
BoundaryLayerGrowthRate = 1.05,
BoundaryLayerMaxPoints  = 400,
)

options = dict(
NProcs=28,
TEclosureTolerance= 1.e-6, 
)

# ======================= END OF USER PARAMETERS ============================ #



if GeomPath.endswith('.dat') or GeomPath.endswith('.txt'):
    airfoilCurve = W.airfoil(GeomPath,
                            ClosedTolerance=options['TEclosureTolerance'])
else:
    airfoilCurve = C.convertFile2PyTree(GeomPath)
    airfoilCurve, = I.getZones(airfoilCurve)

t, meshParams = GSD.extrudeAirfoil2D(airfoilCurve,ReynoldsMesh,DeltaYPlus,
                        Sizes=Sizes,Points=Points,Cells=Cells,options=options)

C.convertPyTree2File(t, MeshFilename)