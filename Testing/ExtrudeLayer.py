import sys
import numpy as np

import Converter.Internal as I
import Converter.PyTree as C
import Generator.PyTree as G
import Transform.PyTree as T
import Connector.PyTree as X
import Geom.PyTree as D
import Post.PyTree as P


import InternalShortcuts as J
import Wireframe as W
import GenerativeVolumeDesign as GVD
import GenerativeShapeDesign as GSD





# Let "s" be a surface that will be extruded
Lx, Ly = 1., 1.
Ni, Nj = 21, 41
s = G.cart((-Lx/2.,-Ly/2.,0),(Lx/(Ni-1.),Ly/(Nj-1.),0),(Ni,Nj,1))
x,y,z = J.getxyz(s)
Amplitude, SigmaX, SigmaY = -0.30, 0.10, 0.10 # Controls the gaussian
z[:] = Amplitude * np.exp(-( x**2/(2*SigmaX**2)+y**2/(2*SigmaY**2)))





DrivingLine = D.line((-Lx/2.,-Ly/2.,0),(Lx/2.,-Ly/2.,0),Ni)
DrivingLine[0] = 'DrivingLine'

# Split the surface and connect it
t = C.newPyTree(['Geometry',[s]])
t = T.splitNParts(t,6)


# Replace a struct zone by an unstr
zones = I.getNodesFromType2(t,'Zone_t')
zT = C.convertArray2Tetra(zones[0])
t = C.newPyTree(['Base',[zT]+zones[1:]])

# 
# DISTRIBUTIONS
NPtsDist = 30
ExtHeight= 0.50
line1 = W.linelaw(P1=(-0.5,+0.5,0),P2=(-0.5,+0.5,ExtHeight),N=NPtsDist,
    Distribution={'kind':'tanhOneSide','FirstCellHeight':0.001}); line1[0]='line1'
nf, gf, nit, git = J.invokeFields(line1,['normalfactor','growthfactor','normaliters','growthiters'])
nf[:]  = np.linspace(1,1e2,NPtsDist)
gf[:]  = np.hstack((np.linspace(0,0.01,6),np.linspace(0.01,8.0,NPtsDist-6)))
nit[:] = np.hstack((np.linspace(0,3,6),np.linspace(3,400,NPtsDist-6)))
git[:] = np.linspace(0,10,NPtsDist)

Distributions = [line1]

if False:
    line2 = W.linelaw(P1=(0.5,-0.5,0),P2=(0.5,-0.5,ExtHeight),N=10); line2[0]='line2'
    C._initVars(line2,'normalfactor',5.)
    C._initVars(line2,'normaliters',3)
    Distributions += [line2]

    line3 = W.linelaw(P1=(0.5,0.5,0),P2=(0.5,0.5,ExtHeight),N=10); line3[0]='line3'
    C._initVars(line3,'normalfactor',5.)
    C._initVars(line3,'normaliters',3)
    Distributions += [line3]

    line4 = W.linelaw(P1=(-0.5,-0.5,0),P2=(-0.5,-0.5,ExtHeight),N=10); line4[0]='line4'
    C._initVars(line4,'normalfactor',5.)
    C._initVars(line4,'normaliters',3)
    Distributions += [line4]


#
# CONSTRAINTS
Constraints = []

# Impose normals
# zExFace = P.exteriorFaces(s);zExFace[0]='ExtFace'
# C._initVars(zExFace,'sx={CoordinateX}*0')
# C._initVars(zExFace,'sy={CoordinateY}*0')
# C._initVars(zExFace,'sz',1.)
# C._normalize(zExFace,['sx','sy','sz'])
# Constraints += [dict(kind='Imposed',curve=zExFace,surface=None),]

# Projection Ortho
# GeneratorLine = D.line((-Lx,-Ly/2.,0),(Lx,-Ly/2.,0),2)
# SurfProj = D.axisym(GeneratorLine, (0,-1.5*Ly,0), (1,0,0), angle=180., Ntheta=60, rmod=None)
# SurfProj[0] = 'SurfProj'
# Constraints += [dict(kind='Projected',curve=DrivingLine,surface=SurfProj,ProjectionMode='ortho', ProjectionDir=(0,-1,0)),]


# Match
zExFace = P.exteriorFaces(s);zExFace[0]='ExtFace'
zExFace=C.convertBAR2Struct(zExFace)
SecUp = T.scale(zExFace,1.5);SecUp[0]='SecUp'
T._translate(SecUp,(0,0,ExtHeight))
ContourSurf, _ = GSD.multiSections([zExFace,SecUp],line1)
ContourSurf[0]='ContourSurf'
Constraints += [dict(kind='Match',curve=zExFace,surface=ContourSurf,MatchDir=None),]




tExtru = GVD.extrude(t,Distributions,Constraints,printIters=True,plotIters=True)

zExtru = I.getNodeFromName2(tExtru,'ExtrudeLayer')
I._rmNodesFromName(zExtru,'FlowSolution#Centers')
C.convertPyTree2File(tExtru,'test.cgns')

