#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

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
NumberOfProcessors=48,
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
