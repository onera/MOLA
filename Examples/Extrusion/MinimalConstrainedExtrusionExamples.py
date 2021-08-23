'''
               --------------------------------------
               MINIMAL CONSTRAINED EXTRUSION EXAMPLES
               --------------------------------------

In this script, several minimal constrained extrusion examples
are presented. These examples are "minimal" because they are not
applied to complex meshes. Rather, simple distributions and surfaces
are built on-the-fly. 

The purpose is to familiarize the user to different kind usages of the
constrained extrusion functions.

File history:
05/03/2019 - L. Bernardos - Creation
'''

# System modules
import sys
import numpy as np

# Original Cassiopee modules
import Converter.PyTree as C
import Converter.Internal as I
import Generator.PyTree as G
import Geom.PyTree as D
import Post.PyTree as P

# Additional Cassiopee-based modules
# To use them, add to your PYTHONPATH: 
# export PYTHONPATH=$PYTHONPATH:/home/lbernard/MOLA/v1.6
import MOLA.InternalShortcuts as J
import MOLA.Wireframe as W
import MOLA.GenerativeVolumeDesign as GVD



# ----------------------------------------------------------------- #
#                             EXAMPLE 1                             #
#  Simple constraint-free extrusion with given imposed distribution #
# ----------------------------------------------------------------- #

# Let "s" be a surface that will be extruded
Lx, Ly = 1., 1.
Ni, Nj = 21, 41
s = G.cart((-Lx/2.,-Ly/2.,0),(Lx/(Ni-1.),Ly/(Nj-1.),0),(Ni,Nj,1))
s[0] = 'Surface'
x,y,z = J.getxyz(s)
Amplitude, SigmaX, SigmaY = -0.20, 0.10, 0.10 # Controls the gaussian
z[:] = Amplitude * np.exp(-( x**2/(2*SigmaX**2)+y**2/(2*SigmaY**2)))

# Let "DistributionCurve" be the wall-normal distribution desired
# by the user. This curve will be used by the extrusion function 
# for determining the number of points and size of the wall-normal
# cells.
# For the moment, let us simply do a uniform distribution. For this,
# we draw a line using "Geom line" function:

MyDistribution = D.line((0,0,0),(0,0,0.25),20) 

# Rename the curve for easy identification
MyDistribution[0] = 'MyDistribution' 

# The extrusion function requires that the Curve distribution yield
# specific fields in "FlowSolution" container. These fields are to be
# used for smoothing purposes during the extrusion. Let us not bother
# about this for the moment. Let us simply declare them, like this:
J._invokeFields(MyDistribution,
    ['normalfactor','growthfactor',
     'normaliters','growthiters'])


# Now we are completely ready for performing our first extrusion:
print("Extruding example 1 (constraint-free, 1 distribution)")
ExtrusionTree = GVD.extrude(s,[MyDistribution])

'''
Let us point out several things:

The result of GVD.extrude() function is a PyTree with the following
structure:

 t
 |
 |____InitialSurface (Base):
 |     All surfaces that are extruded as input of GVD.extrude().
 |     In this example, there is only a single surface zone.
 |
 |____ExtrudedVolume (Base):
 |     This is actually the result of the extrusion that the user wants.
 |     It may be composed by several zones (volumes). But again,
 |     in this example, there is only a single volume zone.
 |
 |____ExtrudeLayerBase (Base):
 |     This is the "dummy-auxiliar" set of surfaces that are displaced
 |     during the extrusion process. At the end of the extrusion, this
 |     must correspond to the last layer. User should not bother about
 |     this.
 | 
 |____ConstraintWireframe (Base):
 |     These are the curves used for driving the constraints provided
 |     by the user. They do not apply to current example, so it is
 |     empty.
 |
 |____ConstraintSurfaces (Base):
 |     These are the surfaces used for driving the constraints provided
 |     by the user. They do not apply to current example, so it is
 |     empty.
 |
 |____DistributionsBase (Base):
       Here, the user-provided distributions are contained. In this
       example, a single distribution was employed (MyDistribution),
       but additional ones could be used for more complex extrusion.

We can observe that the result of the extrusion yields self-
intersecting cells. 

In the next example, we are setting "smooth options", in order to
correct this undesirable effect.
'''

# Save the result of Example 1
C.convertPyTree2File(ExtrusionTree,'MinimalConstrainedExtrusionExamples_Example1.cgns')




# ----------------------------------------------------------------- #
#                             EXAMPLE 2                             #
#  Simple constraint-free extrusion with given imposed distribution #
#  and smoothing options.                                           #
#  The purpose of Example 2 is avoiding the self-intersecting cells #
#  shown in Example 1, only by smoothing of the normals.            #
# ----------------------------------------------------------------- #


# Let "s" be a surface that will be extruded
Lx, Ly = 1., 1.
Ni, Nj = 21, 41
s = G.cart((-Lx/2.,-Ly/2.,0),(Lx/(Ni-1.),Ly/(Nj-1.),0),(Ni,Nj,1))
s[0] = 'Surface'
x,y,z = J.getxyz(s)
Amplitude, SigmaX, SigmaY = -0.20, 0.10, 0.10 # Controls the gaussian
z[:] = Amplitude * np.exp(-( x**2/(2*SigmaX**2)+y**2/(2*SigmaY**2)))


# Again, we build the distribution curve as follows:
MyDistribution = D.line((0,0,0),(0,0,0.25),20) 
MyDistribution[0] = 'MyDistribution' 
nf, gf, nit, git = J.invokeFields(MyDistribution,
    ['normalfactor','growthfactor',
     'normaliters','growthiters']) # all values initialized at 0

'''
Please note that variables "nf, gf, nit, git" are 1D numpy arrays.
These variables are employed for controlling the smoothing process.
Let us explain how they work:

* nf (for "normalfactor") and nit (for "normaliters") are used for
  smoothing the normals of each surface cell. 

  Increasing the value of "normalfactor" will increase the smoothness
  of the normals. The inconvenience is that this smooth deteriorates
  the orthogonality of the mesh. So, low values are required in regions
  where orthogonality is desired (i.e. boundary layers) while higher
  values may be adopted where orthogonality is not that important.

  Increasing the value of "normaliters" will increase the smoothness
  of the normals. The smoothing of the normals are done by using a 
  series of center2Node and node2Center operation on the normals. So,
  this parameter controls the amount of iterations.

* gf (for "growthfactor") and git (for "growthiters") are used for
  smoothing and controlling the current cell size iteration height
  [Nota Bene: this quantity is named "dH". Each cell will be extruded
  following the vector defined by: dH*(sx, sy, sz), where 
  sx, sy and sz are the unit-normals of the cell size].

  Increasing the value of "growthiters" will provoke a smoothing of
  the value of "dH" (in the same fashion as "normaliters" provoked
  the smoothing of the normals). 

  However, the value of "growthfactor" is dependent on the growth
  Equation that governs the extrusion, which is user-defined. So,
  its meaning depends on what the user desires. We will see this in
  another example.
'''

'''
Let us impose low normal smoothing near the wall (where orthogonal
cells are desired) and high normal smoothing further from the wall.
'''
nf[:]  = np.linspace(0,500,len(nf)) # Linear law from 0 at wall 
nit[:] = np.linspace(0,150,len(nf)) # Linear law from 0 at wall 

# Extrude
print("Extruding example 2 (constraint-free, 1 distribution) with normals smoothing")
ExtrusionTree = GVD.extrude(s,[MyDistribution])

'''
We can observe that the resulting extrusion do not yield 
self-intersecting cells. However, the result is not entirely 
satisfactory in the concave part of the mesh. Indeed, cells are 
skewed, and a singular point appears. 

This is because not only we need to smooth the normals, but also
we have to locally extrude with higher cell heights in concave regions.
The next example will illustrate this.
'''

# Save the result of Example 2
C.convertPyTree2File(ExtrusionTree,'MinimalConstrainedExtrusionExamples_Example2.cgns')



# ----------------------------------------------------------------- #
#                             EXAMPLE 3                             #
#  Simple constraint-free extrusion with given imposed distribution #
#  and smoothing options.                                           #
#  The purpose of Example 3 is avoiding the self-intersecting cells #
#  shown in Example 1, and also correcting the strong concavity     #
#  produced in Example 2.                                           #
# ----------------------------------------------------------------- #


# Let "s" be a surface that will be extruded
Lx, Ly = 1., 1.
Ni, Nj = 21, 41
s = G.cart((-Lx/2.,-Ly/2.,0),(Lx/(Ni-1.),Ly/(Nj-1.),0),(Ni,Nj,1))
s[0] = 'Surface'
x,y,z = J.getxyz(s)
Amplitude, SigmaX, SigmaY = -0.20, 0.10, 0.10 # Controls the gaussian
z[:] = Amplitude * np.exp(-( x**2/(2*SigmaX**2)+y**2/(2*SigmaY**2)))


# Again, we build the distribution curve as follows:
MyDistribution = D.line((0,0,0),(0,0,0.25),20) 
MyDistribution[0] = 'MyDistribution' 
nf, gf, nit, git = J.invokeFields(MyDistribution,
    ['normalfactor','growthfactor',
     'normaliters','growthiters']) # all values initialized at 0

'''
As previously done,
let us impose low normal smoothing near the wall (where orthogonal
cells are desired) and high normal smoothing further from the wall.
'''
nf[:]  = np.linspace(0,500,len(nf)) # Linear law from 0 at wall 
nit[:] = np.linspace(0,150,len(nf)) # Linear law from 0 at wall 

'''
As you may guess, this example consists in setting appropriate values
to the "growthfactor" and "growthiters" variables. In addition, a
growth equation has to be specified by the user. Let's do that:
'''

WeightCoeff = 'maximum(1.+tanh(-{nodes:growthfactor}*{nodes:divs}),0.2)'
growthEquation='nodes:dH={nodes:dH}*%s'%(WeightCoeff)


'''
            EXPLANATION OF THE USER-DEFINED EQUATION

As one may observe, the growth equation tells the algorithm to 
weight the local cell extrusion height {nodes:dH} with a coefficient.
Such user-defined weighting coefficient is the following one:

    maximum(1.+tanh(-{nodes:growthfactor}*{nodes:divs}),0.2)

The purpose of this weighting coefficient is to increase dH in concave
regions (with WeightCoeff>1)  and to decrease it in convex regions
(with WeightCoeff<1). This is accomplished through the use of the
quantity {nodes:divs}, which is the divergence of the normals:

    divs = d/dx (normalX) + d/dy (normalY) + d/dz (normalZ)

We note that divs<0 in concave regions and divs>0 in convex regions.
So it is an interesting measure of the concavity/convexity of a surface.
This can be exploited for weighting the growth rate of each cell.

The value of "growthfactor" is simply a secondary weighting coefficient
that controls the slope of the hyperbolic tangent at 1. This means that
"growthfactor" in this equation controls the stiffness of the change in
value of dH. This has somehow an opposite effect to the smoothing of 
dH via "growthiters".
'''

# Set the smoothing coefficients
gf[:]   = np.linspace(0.2,1,len(gf))
git[:]  = np.linspace(10,20,len(git))

# Extrude
print("Extruding example 3 (constraint-free, 1 distribution) with normals and differential cell heights smoothing")
ExtrusionTree = GVD.extrude(s,[MyDistribution])

# Save the result of Example 3
C.convertPyTree2File(ExtrusionTree,'MinimalConstrainedExtrusionExamples_Example3.cgns')
'''
We observe that in Example 3, the height of extrusion in the concave
region is greater. This compensates the concavity.
'''


# ----------------------------------------------------------------- #
#                             EXAMPLE 4                             #
#  Simple constraint-free extrusion with given several imposed      #
#  distributions.                                                   #
#                                                                   #
#  The purpose of this example is showing how the user can make a   #
#  local influence on the wall-normal spacing.                      #
# ----------------------------------------------------------------- #


# Let "s" be a surface that will be extruded
Lx, Ly = 1., 1.
Ni, Nj = 21, 41
s = G.cart((-Lx/2.,-Ly/2.,0),(Lx/(Ni-1.),Ly/(Nj-1.),0),(Ni,Nj,1))
s[0] = 'Surface'
x,y,z = J.getxyz(s)
Amplitude, SigmaX, SigmaY = -0.20, 0.10, 0.10 # Controls the gaussian
z[:] = Amplitude * np.exp(-( x**2/(2*SigmaX**2)+y**2/(2*SigmaY**2)))

'''
In this example, we are building two different distributions, which 
are placed in different places. This will produce a non-homogeneous
extrusion result, as will be seen.

'''

# First distribution of small length, at one corner
Distribution1 = D.line((0.5,-0.5,0),(0.5,-0.5,0.10),20) 
Distribution1[0] = 'Distribution1' 
nf, gf, nit, git = J.invokeFields(Distribution1,
    ['normalfactor','growthfactor',
     'normaliters','growthiters'])  # all values initialized at 0
nf[:]  = np.linspace(0,500,len(nf)) # Set a linear law from 0 at wall 
nit[:] = np.linspace(0,150,len(nf)) # Set a linear law from 0 at wall 
gf[:]   = np.linspace(0.1,0.2,len(gf))
git[:]  = np.linspace(10,20,len(git))


# Second distribution of bigger length, at opposite corner
Distribution2 = D.line((-0.5,0.5,0),(-0.5,0.5,0.40),20) 
Distribution2[0] = 'Distribution2' 
nf, gf, nit, git = J.invokeFields(Distribution2,
    ['normalfactor','growthfactor',
     'normaliters','growthiters'])  # all values initialized at 0
nf[:]  = np.linspace(0,500,len(nf)) # Set a linear law from 0 at wall 
nit[:] = np.linspace(0,150,len(nf)) # Set a linear law from 0 at wall 
gf[:]   = np.linspace(0.1,0.2,len(gf))
git[:]  = np.linspace(10,20,len(git))

# A third distribution, with big length, at center of domain
Distribution3 = D.line((0,0,Amplitude),(0,0,0.30),20) 
Distribution3[0] = 'Distribution3' 
nf, gf, nit, git = J.invokeFields(Distribution3,
    ['normalfactor','growthfactor',
     'normaliters','growthiters'])  # all values initialized at 0
nf[:]  = np.linspace(0,500,len(nf)) # Set a linear law from 0 at wall 
nit[:] = np.linspace(0,150,len(nf)) # Set a linear law from 0 at wall 
gf[:]   = np.linspace(0.1,0.2,len(gf))
git[:]  = np.linspace(10,20,len(git))

'''
NOTA BENE: As you can observe, it is possible to set different sets of 
smoothing values at each provided distribution. This may be useful.
However, in this example, we always impose the same set of smoothing
parameters for each distribution.
'''

# Extrude
ExtrusionTree = GVD.extrude(s,
    [Distribution1, Distribution2,Distribution3])

# Save the result of Example 4
print("Extruding example 4 - two distributions")
C.convertPyTree2File(ExtrusionTree,'MinimalConstrainedExtrusionExamples_Example4.cgns')



# ----------------------------------------------------------------- #
#                             EXAMPLE 5                             #
#  This example shows how to impose constraints on the extrusion.   #
#  In this example, a normal-imposed constraint is shown.           #
# ----------------------------------------------------------------- #

# Let "s" be a surface that will be extruded
Lx, Ly = 1., 1.
Ni, Nj = 21, 41
s = G.cart((-Lx/2.,-Ly/2.,0),(Lx/(Ni-1.),Ly/(Nj-1.),0),(Ni,Nj,1))
s[0] = 'Surface'
x,y,z = J.getxyz(s)
Amplitude, SigmaX, SigmaY = -0.20, 0.10, 0.10 # Controls the gaussian
z[:] = Amplitude * np.exp(-( x**2/(2*SigmaX**2)+y**2/(2*SigmaY**2)))


# Again, build a distribution 
Distribution1 = D.line((0.5,-0.5,0),(0.5,-0.5,0.50),20) 
Distribution1[0] = 'Distribution1' 
nf, gf, nit, git = J.invokeFields(Distribution1,
    ['normalfactor','growthfactor',
     'normaliters','growthiters'])  # all values initialized at 0
nf[:]  = np.linspace(0,500,len(nf)) # Set a linear law from 0 at wall 
nit[:] = np.linspace(0,150,len(nf)) # Set a linear law from 0 at wall 
gf[:]   = np.linspace(0.1,0.2,len(gf))
git[:]  = np.linspace(10,20,len(git))

'''
Now, we are going to build our "normal-imposed" constraint.
For this, first we define the cells where the surface is imposed.
We do this using a PyTree curve, for example, the contours:
'''
zExFace = P.exteriorFaces(s);zExFace[0]='ExtFace'

'''
Then, we set the values of the normal (x,y,z) components.
These are the fields named "sx", "sy" and "sz".

Now, we impose the values of the normals. Let us impose beautiful
radial-shaped normals:
'''
C._initVars(zExFace,'sx={CoordinateX}')
C._initVars(zExFace,'sy={CoordinateY}')
C._initVars(zExFace,'sz',0.5)

# (sx,sy,sz) shall be unit vectors, so:
C._normalize(zExFace,['sx','sy','sz']) 

'''
Next, we declare the "Constraint object". This is simply a Python
dictionary with context-dependent pairs of key/values.
In the case of imposed normal, we set:
'''
MyConstraint = dict(
kind='Imposed', # because we impose the value of normals 

curve=zExFace,  # This tells which mesh points are impacted by 
                # the constraint (the do not necessarily have to
                # lie on the boundaries, it can be interior points)
    )


'''
Everything is ready for extrusion... we just have to tell the function
a list containing all the constraints (only 1 in this case)
'''

# Extrude
print("Extruding example 5 - normal imposed constraint")
ExtrusionTree = GVD.extrude(s,
    [Distribution1], Constraints=[MyConstraint])

# Save the result of Example 5
C.convertPyTree2File(ExtrusionTree,'MinimalConstrainedExtrusionExamples_Example5.cgns')

'''
And we can observe that the constraint is well verified.

In next examples we will explore some more complex constraints and
combinations.
'''


# ----------------------------------------------------------------- #
#                             EXAMPLE 6                             #
#  This is the last example (and most complex one) of this script.  #
#  If you understood the principles of all previous examples, then  #
#  you are ready for a complex (while minimalist) example.          #
#                                                                   #
#  Here, we are going to perform a complex combination of           #
#  constraints of different types (normals imposed, projections,    #
#  coincidence...)                                                  #
# ----------------------------------------------------------------- #

# Let "s" be a surface that will be extruded
Lx, Ly = 1., 1.
Ni, Nj = 21, 41
s = G.cart((-Lx/2.,-Ly/2.,0),(Lx/(Ni-1.),Ly/(Nj-1.),0),(Ni,Nj,1))
s[0] = 'Surface'
x,y,z = J.getxyz(s)
Amplitude, SigmaX, SigmaY = -0.20, 0.10, 0.10 # Controls the gaussian
z[:] = Amplitude * np.exp(-( x**2/(2*SigmaX**2)+y**2/(2*SigmaY**2)))


# Again, build a distribution 
Distribution1 = D.line((0.5,-0.5,0),(0.5,-0.5,0.40),20) 
Distribution1[0] = 'Distribution1' 

# Set some smoothing parameters, as always:
nf, gf, nit, git = J.invokeFields(Distribution1,
    ['normalfactor','growthfactor',
     'normaliters','growthiters'])  # all values initialized at 0
nf[:]  = np.linspace(0,500,len(nf)) # Set a linear law from 0 at wall 
nit[:] = np.linspace(0,150,len(nf)) # Set a linear law from 0 at wall 
gf[:]   = np.linspace(0.1,0.2,len(gf))
git[:]  = np.linspace(10,20,len(git))

'''
Let us have some fun. We are going to impose a different kind of
constraint on each boundary. First, let us extract the boundaries:
'''
Bounds = Bound1, Bound2, Bound3, Bound4 = P.exteriorFacesStructured(s)
# C.convertPyTree2File(Bounds,'test.cgns'); sys.exit()

'''
First Constraint: Orthogonal projection on a surface.
'''

# Let us build our projection surface. For example, a cylinder
GenLine = D.line((-Lx/2.,-Ly,0),(-Lx/2.,+Ly,0),2)
ProjSurf = D.axisym(GenLine,(-Lx,0,0),(0,1,0),-60.,60)

# Let us deform a bit our projection surface. Otherwise it isn't fun ;-)
C._initVars(ProjSurf,
    'CoordinateX={CoordinateX} + {CoordinateY}*{CoordinateZ}')

# Set the constraint...
Constraint1 = dict(
kind='Projected',    # Tells that constraint is of projection kind

curve=Bound1,        # Tells which points of the extrude wall surface
                     # are concerned by current constraint

surface=ProjSurf,    # Tells which is the surface where points will be
                     # projected during the extrusion process

ProjectionMode='ortho',  # mode of projection 'ortho' or 'dir'

ProjectionDir=(-1,0,0), # directional projection
                        # (only relevant if ProjectionMode='dir')
)
'''
Second Constraint: exact coincidence following a surface.
For example, let us build a cylinder surface, and we tell the extrude
function to extrude by following Bound2 exactly on each point of
the surface of the cylinder.
'''

# Build the cylinder...
Cylinder = D.axisym(Bound2,(-Lx/2.,0,0),(0,1,0),-45.,C.getNPts(Distribution1))
Cylinder[0] = "MyCylinder"

# Set the constraint...
Constraint2 = dict(
kind='Match',    # Tells that constraint is of projection kind

curve=Bound2,        # Tells which points of the extrude wall surface
                     # are concerned by current constraint

surface=Cylinder,    # Tells which is the surface where points will be
                     # matched during the extrusion process

MatchDir=None, # automatically guess the structured-surface index to 
               # follow
)


# Extrude
print("Extruding example 6 - multiple constraints types and boundaries")
ExtrusionTree = GVD.extrude(s,
    [Distribution1], 

    # Do not forget to put here all constraints !...
    Constraints=[
    Constraint1,
    Constraint2])

# Save the result of Example 6
C.convertPyTree2File(ExtrusionTree,'MinimalConstrainedExtrusionExamples_Example6.cgns')

'''
And there we got our "complex extrusion" example.
'''
