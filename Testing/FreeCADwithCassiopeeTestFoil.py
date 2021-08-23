"""
This document tests Cassiopee and FreeCAD jointly.

A basic operation is performed: a coarse NACA 0012 is created
using Cassiopee's Geom module. Then, a CAD object is built
using FreeCAD from the previously generated NACA 0012 points.
This is done using FreeCAD's Draft's module makeBSpline() function.
Finally, a new fine curve is created by evaluation of CAD object.

Both coarse and fine discrete curves are saved in out.cgns file.

CAD object is saved in MyFoil.iges and MyFoil.FCStd files.

16/12/2019 - L. Bernardos
"""

import sys
import numpy as np

import Converter.PyTree as C
import Converter.Internal as I
import Geom.PyTree as D
import Transform.PyTree as T

import FreeCAD
vFreeCAD = int(''.join(FreeCAD.Version()[:2]))
import Part

# This trick is required for replacing bugged Draft
# module of FreeCAD v0.16 by a functional one
if vFreeCAD < 18: import MyDraft
else: import Draft as MyDraft

doc=FreeCAD.newDocument() 

Naca = D.naca(12.)
T._homothety(Naca,(0,0,0),1.e3) # work in mm
Naca[0] = 'Naca12FromGeom'


# Build a CAD spline from points of Naca curve
Vectors = []
x = I.getNodeFromName2(Naca,'CoordinateX')[1]
y = I.getNodeFromName2(Naca,'CoordinateY')[1]
z = I.getNodeFromName2(Naca,'CoordinateZ')[1]
lenX = len(x)
for i in xrange(lenX):
    delta = 1.e-6 if i == lenX -1 else 0
    Vectors += [ (x[i], y[i]+delta, z[i]) ]
Foil = MyDraft.makeBSpline(Vectors) # CAD object



# From CAD object, extract discrete points and 
# build a very smooth curve
Points = []
for s in np.linspace(0,1,4001):
    Vec = Foil.Shape.valueAt(s*Foil.Shape.LastParameter)
    Points += [(Vec.x, Vec.y, Vec.z)]
poly  = D.polyline(Points)
poly[0] = 'DiscreteCurveFromCADObject'

# Save discrete results
t = C.newPyTree(["Base",[poly,Naca]])
C.convertPyTree2File(t,'out.cgns')


# Save CAD data, both as IGS and FreeCAD Std document
doc.recompute()
Part.export([doc.getObject("BSpline")],"MyFoil.iges")
doc.saveAs('MyFoil.FCStd')
