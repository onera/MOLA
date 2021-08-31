import sys
import numpy as np

import Converter.PyTree as C
import Converter.Internal as I
import Connector.PyTree as X
import Transform.PyTree as T
import Generator.PyTree as G
import Geom.PyTree as D
import Post.PyTree as P
import InternalShortcuts as J
import Wireframe as W
import GenerativeShapeDesign as GSD
import GenerativeVolumeDesign as GVD

WingTree = C.convertFile2PyTree('wing4Extruding2.cgns')

print "Building Hub...",
HubProfile = D.line((0,1,-0.25),(0,-1,-0.25),50)
Hub = D.axisym(HubProfile, (0.,0.,0.),(0,1,0), angle=360., Ntheta=360)
print "ok"

# Define the normal-extrusion distribution
NPtsNormal = 40
HghtNormal = 0.5
distNormal = W.linelaw(P2=(HghtNormal,0,0),N=NPtsNormal,
    Distribution={'kind':'ratio','growth':1.1,'FirstCellHeight':0.0001},
    )
nf, gf, nit, git = J.invokeFields(distNormal,['normalfactor','growthfactor','normaliters','growthiters'])
nf[:]  = 10.
nit[:] = np.linspace(3,80,NPtsNormal)
gf[:]  = np.linspace(1e3,2e3,NPtsNormal)
git[:] = np.linspace(20,100,NPtsNormal)


t = GVD.extrudeWingOnSupport(WingTree,Hub,[distNormal],extrapolate=False)

C.convertPyTree2File(t,'test.cgns')
sys.exit()
