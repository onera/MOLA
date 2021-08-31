import sys
import timeit
import numpy as np

import Converter.PyTree as C
import Transform.PyTree as T
import Generator.PyTree as G
import Converter.Internal as I
import Geom.PyTree as D
import Intersector.PyTree as XOR

import Wireframe as W
import InternalShortcuts as J
import GenerativeShapeDesign as GSD

foil = W.airfoil('NACA9304', Closed=False)
# foil = D.naca('0010')
# foil = W.setTrailingEdge(foil)

foilY = J.gety(foil)
foilY *= 7

_, Top, Bottom = W.findLeadingEdgeAndSplit(foil,0.1)

    

tic = timeit.default_timer()
for i in xrange(1): CamberLine = W.getCamberOptim(foil,method='hybr',
    # options={'col_deriv':True},
    )
toc = timeit.default_timer()
print "getCamberOptim: %g"%(toc-tic)

NewFoil = W.buildAirfoilFromCamberLine(CamberLine); NewFoil[0]='NewFoil'
t = C.newPyTree(['Base',[foil,CamberLine],'BaseNewFoil',[NewFoil],'BaseTOPBOT',[Top,Bottom]])
C.convertPyTree2File(t,'out.cgns')