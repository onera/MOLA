import sys
import numpy as np

import Converter.PyTree as C
import Converter.Internal as I
import Geom.PyTree as D

import FreeCAD
import Part
import Draft

doc=FreeCAD.newDocument() 

Naca = D.naca(12.)

Vectors = []
x = I.getNodeFromName2(Naca,'CoordinateX')[1]
y = I.getNodeFromName2(Naca,'CoordinateY')[1]
z = I.getNodeFromName2(Naca,'CoordinateZ')[1]
lenX = len(x)
for i in xrange(lenX):
    delta = 1.e-6 if i == lenX -1 else 0
    Vectors += [ (x[i], y[i]+delta, z[i]) ]
Foil = Draft.makeBSpline(Vectors)

# for i in dir(Foil.Shape): print i

print "LastParameter:",Foil.Shape.LastParameter
print "Length:",Foil.Shape.Length
Points = []
for s in np.linspace(0,1,2001):
    Vec = Foil.Shape.valueAt(s*Foil.Shape.LastParameter)
    Points += [(Vec.x, Vec.y, Vec.z)]

poly = D.polyline(Points)
print "Length by Geom is:",D.getLength(poly)
poly2 = D.polyline(Vectors)
t = C.newPyTree(["Base",[poly,Naca,poly2]])
C.convertPyTree2File(t,'out.cgns')

Draft.autogroup(Foil)
doc.recompute()
doc.saveAs('test.FCStd')