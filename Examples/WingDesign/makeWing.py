'''
makeWing.py example

Shows how to generate a wing surface using main wing geometrical
characteristics and airfoil characteristics.

03/12/2020 - L. Bernardos - First creation
'''


import numpy as np 

import Converter.PyTree as C

import MOLA.Wireframe as W
import MOLA.GenerativeShapeDesign as GSD


# Create a set of Python dictionaries with the geometrical laws of wing and
# airfoils. All airfoil attributes acceptable in function W.modifyAirfoil()
# can be used in GSD.wing()
Airfoil = dict(RelativeSpan    =[                  0.0,                   1.0 ],
               Airfoil         =[W.airfoil('NACA4412'), W.airfoil('NACA4412')],
               InterpolationLaw='interp1d_linear')

Chord = dict(RelativeSpan    = [0.0,  0.5,   1.0],
             Chord           = [0.8,  0.6,   0.2],
             InterpolationLaw='akima')

Sweep = dict(RelativeSpan    = [0., 1.0],
             Sweep           = [0., 0.5],
             InterpolationLaw='interp1d_linear')

MaxThickness = dict(RelativeSpan    = [0.,    1.0],
                    MaxThickness    = [0.09, 0.03],
                    InterpolationLaw='interp1d_linear')

MaxThicknessRelativeLocation = dict(RelativeSpan            = [0.,    1.0],
                            MaxThicknessRelativeLocation    = [0.30, 0.10],
                            InterpolationLaw='interp1d_linear')


Rmin, Rmax, Nspan = 0.5, 3.0, 20
StackRelativeChord = 0.25
SpanVector = np.linspace(Rmin, Rmax, Nspan)

Sections, Blade,_ = GSD.wing(SpanVector, StackRelativeChord,
             Airfoil=Airfoil, Chord=Chord, Sweep=Sweep,
             MaxThickness=MaxThickness,
             MaxThicknessRelativeLocation=MaxThicknessRelativeLocation)

C.convertPyTree2File(Blade, 'blade.cgns')

