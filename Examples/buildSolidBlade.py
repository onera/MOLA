import sys
import numpy as np

import Converter.PyTree as C
import Converter.Internal as I
import Generator.PyTree as G

import MOLA.InternalShortcuts as J
import MOLA.Wireframe as W
import MOLA.GenerativeShapeDesign as GSD

ChordDict = dict(RelativeSpan = [0.2,   0.45,  0.6,  1.0],
                 Chord =        [0.09,  0.12, 0.12, 0.03],
                 InterpolationLaw = 'akima',)

TwistDict = dict(RelativeSpan = [0.2,  0.6,  1.0],
                 Twist        = [30.,  6.0, -7.0],
                 InterpolationLaw = 'akima',)

DihedralDict = dict(RelativeSpan    = [0.2,  1.0],
                    Dihedral        = [0,    0.0],
                    InterpolationLaw = 'interp1d_linear',)

SweepDict = dict(RelativeSpan    = [0.2,  1.0],
                 Sweep           = [0,    0.0],
                 InterpolationLaw = 'interp1d_linear',)

NACA4412 = W.airfoil('NACA4412', ClosedTolerance=0)

AirfoilsDict = dict(RelativeSpan     = [  0.20,     1.000],
                    Airfoil          = [NACA4412,  NACA4412],
                    InterpolationLaw = 'interp1d_linear',)

BladeDiscretization = np.linspace(0.15,0.6,101)

NPtsTrailingEdge = 5 # MUST BE ODD !!!
IndexEdge = int((NPtsTrailingEdge+1)/2)
Sections, WingWall,_=GSD.wing(BladeDiscretization,ChordRelRef=0.25,
                              NPtsTrailingEdge=NPtsTrailingEdge,
                              Airfoil=AirfoilsDict,
                              Chord=ChordDict,
                              Twist=TwistDict,
                              Dihedral=DihedralDict,
                              Sweep=SweepDict,)

# Nj = I.getZoneDim(WingWall)[2]
# Sections = [GSD.getBoundary(WingWall,'jmin',layer=i) for i in range(Nj)]


ClosedSections = []
for s in Sections:
    ClosedSections.extend( GSD.closeAirfoil(s,Topology='ThickTE_simple',
        options=dict(NPtsUnion=NPtsTrailingEdge,
                     TFITriAbscissa=0.1,
                     TEdetectionAngleThreshold=30.)) )

WingSolidStructured = G.stack(ClosedSections,None) # Structured
_,Ni,Nj,Nk,_=I.getZoneDim(WingSolidStructured)

C.convertPyTree2File(WingSolidStructured, 'struct.cgns')

WingSolidStructured, = I.getZones(C.convertFile2PyTree('struct.cgns'))

C._addBC2Zone(WingSolidStructured,
              'Noeud_Encastrement',
              'FamilySpecified:Noeud_Encastrement',
              'kmin')


C._addBC2Zone(WingSolidStructured,
              'LeadingEdge',
              'FamilySpecified:LeadingEdge',
              wrange=[IndexEdge,IndexEdge,Nj,Nj,1,Nk])

C._addBC2Zone(WingSolidStructured,
              'TrailingEdge',
              'FamilySpecified:TrailingEdge',
              wrange=[IndexEdge,IndexEdge,1,1,1,Nk])

C._fillEmptyBCWith(WingSolidStructured,
                   'External_Forces',
                   'FamilySpecified:External_Forces')

t = C.newPyTree(['SOLID',WingSolidStructured])
C.convertPyTree2File(t,'solid.cgns','bin_adf')
