'''
User-provided meshing parameters
'''


Chord = 1.0
# Mesh generation parameters
Sizes = dict(
Height                  = 100.*Chord,
Wake                    = 100.*Chord,
BoundaryLayerMaxHeight  = 0.10*Chord,
TrailingEdgeTension     = 0.10*Chord,
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
NProc=28,
TEclosureTolerance=1e-6,
)

# ======================= END OF USER PARAMETERS ============================ #

meshParams = {
    'References':{'DeltaYPlus':0.5},
    'Sizes':Sizes,
    'Cells':Cells,
    'Points':Points,
    'options':options,
}
