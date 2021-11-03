import Converter.PyTree as C
import Converter.Internal as I
import Transform.PyTree as T

import MOLA.Postprocess as Post

import setup
PressureDynamic=setup.ReferenceValues['PressureDynamic']
PressureRef=setup.ReferenceValues['Pressure']

t = C.convertFile2PyTree('fields.cgns')

I._rmNodesByName(t,'FlowSolution#EndOfRun')
I._renameNode(t,'FlowSolution#Init','FlowSolution#Centers')

# Variables names must have always same order /!\ see #7672
FlowSolutionNodes = I.getNodesFromType(t,'FlowSolution_t')
for fs in FlowSolutionNodes: I._sortByName(fs, recursive=True)

Post.addPressureIfAbsent(t)

wall, = Post.getWalls(t)
surf = Post.buildAuxiliarWallNormalSurface(t, wall)


Lambda = '(-2./3. * {ViscosityMolecular})'
mu = '{ViscosityMolecular}'
divU = '{VelocityDivergence}'
Eqns=[]
# div(u)
Eqns += [('VelocityDivergence={gradxVelocityX}+'
                             '{gradyVelocityY}+'
                             '{gradzVelocityZ}')]

# Sij
Eqns += [('DeformationXX={gradxVelocityX}')]
Eqns += [('DeformationYY={gradyVelocityY}')]
Eqns += [('DeformationZZ={gradzVelocityZ}')]
Eqns += [('DeformationXY=0.5*({gradyVelocityX}+{gradxVelocityY})')]
Eqns += [('DeformationXZ=0.5*({gradzVelocityX}+{gradxVelocityZ})')]
Eqns += [('DeformationYZ=0.5*({gradzVelocityY}+{gradyVelocityZ})')]

# tau
Eqns += [('ShearStressXX={Lambda}*{divU}+2*{mu}*{SXX}').format(
    Lambda=Lambda,divU=divU,mu=mu,SXX='{DeformationXX}')]
Eqns += [('ShearStressYY={Lambda}*{divU}+2*{mu}*{SYY}').format(
    Lambda=Lambda,divU=divU,mu=mu,SYY='{DeformationYY}')]
Eqns += [('ShearStressZZ={Lambda}*{divU}+2*{mu}*{SZZ}').format(
    Lambda=Lambda,divU=divU,mu=mu,SZZ='{DeformationZZ}')]
Eqns += [('ShearStressXY=2*{mu}*{SXY}').format(
    mu=mu,SXY='{DeformationXY}')]
Eqns += [('ShearStressXZ=2*{mu}*{SXZ}').format(
    mu=mu,SXZ='{DeformationXZ}')]
Eqns += [('ShearStressYZ=2*{mu}*{SYZ}').format(
    mu=mu,SYZ='{DeformationYZ}')]

Eqns += [('SkinFrictionX={ShearStressXX}*{nx}+'
                        '{ShearStressXY}*{ny}+'
                        '{ShearStressXZ}*{nz}')]
Eqns += [('SkinFrictionY={ShearStressXY}*{nx}+'
                        '{ShearStressYY}*{ny}+'
                        '{ShearStressYZ}*{nz}')]
Eqns += [('SkinFrictionZ={ShearStressXZ}*{nx}+'
                        '{ShearStressYZ}*{ny}+'
                        '{ShearStressZZ}*{nz}')]
Eqns += [('SkinFriction=sqrt({SkinFrictionX}**2+'
                            '{SkinFrictionY}**2+'
                            '{SkinFrictionZ}**2)')]
Eqns += ['MaximumSkinFriction=0.0']
for Eqn in Eqns: C._initVars(surf, Eqn)
SkinFriction, MaxSkinFriction = Post.J.getVars(surf,['SkinFriction','MaximumSkinFriction'])
for i in range(SkinFriction.shape[0]):  MaxSkinFriction[i,:] = SkinFriction[i,:].max()

Post.postProcessBoundaryLayer(surf)
BoundaryLayer = Post.extractBoundaryLayer(surf,
                                        PressureDynamic=PressureDynamic,
                                        PressureRef=PressureRef)
C.convertPyTree2File(BoundaryLayer, 'BoundaryLayer.cgns')

zone = I.getNodeFromNameAndType(BoundaryLayer,'TopSide','Zone_t')
x = Post.J.getx(zone)
SkinFriction, MaxSkinFriction = Post.J.getVars(zone,['SkinFriction','MaximumSkinFriction'])
import matplotlib.pyplot as plt
plt.plot(x,SkinFriction,mfc='None',marker='o',ls='None',label='SkinFriction')
plt.plot(x,MaxSkinFriction,label='MaxSkinFriction')
plt.xlabel("x/c")
plt.legend(loc='best')
plt.savefig('friction.png')
plt.show()
