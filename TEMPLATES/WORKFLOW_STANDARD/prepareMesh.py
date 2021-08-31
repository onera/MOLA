'''
prepareMesh.py template designed for STANDARD Workflow.

Produces mesh.cgns from another CGNS or raw meshes

MOLA 1.10 - 04/03/2020 - L. Bernardos - creation
'''

import sys, os
import numpy as np
from timeit import default_timer as tic

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.Preprocess as PRE

toc = tic() # auxiliary variable used to log the script execution time


# these variables are used for splitting purposes
NProcs = None # if None, then it is automatically determined
ProcPointsLoad = 250000

# in an Overset context, it is useful to define a global variable
# that indicates the smallest cell size, like this :
Snear = 2e-2 


'''
The following variable, InputMeshes, is the main source of information provided
by the user for the first stage of preprocessing.

It is a LIST OF DICTIONARIES !

Each dictionary corresponds to an OVERSET COMPONENT and will be translated
to a BASE in the new mesh.cgns output file.

If no OVERSET is present, InputMeshes will be composed of a single (1) 
dictionary.

Each dictionary may have, at most, the following keys:

- file                (MANDATORY)
- baseName            (MANDATORY)
- Connection          (RECOMMENDED)
- BoundaryConditions  (optional -> only if relevant)
- Transform           (optional -> only if relevant)
- OversetOptions      (optional -> only if relevant)
- SplitBlocks         (optional)

"optional" keys are to be set or adapted depending on the existing data
contained in input meshes files.

Each key is explained in detail in the following example.

Following example consists in 5 overset components (1 wing,
1 horizontal tail plane, and 3 refined regions for bodyforce propeller).

Hence, InputMeshes is a list of 5 dictionaries.

Each component HAS TO BE LOADED FROM INDIVIDUAL FILES.

The original files do not have any kind of BC, connectivity or family defined:
they are "raw meshes".
'''

InputMeshes = [ # this is a list !

# this is the first dictionary ! it corresponds to the first refinement region
# where the bodyforce propeller will be put
dict(

# key "file" : we specify the location of the first component mesh.
# it accepts cgns, plt... any file format supported by Cassiopee !
# we recommend using os.path.join() for appending directories

file=os.path.join('.','HLP_Root.cgns'),

# key "baseName" : this is the name you want to provide to the newly created
# CGNS Base. It must be unique.

baseName='HLPRoot',


# key 'Transform': applies a scale factor to the input mesh:

Transform=dict(scale=1.0),


# key "Connection" : this is a LIST OF DICTIONARIES. Each dictionary
# correspond to a connect Match operation. "type" is 'Match' or 'NearMatch'.
# Typically only one element is required. Two elements are required, at most,
# if NearMatch is necessary, for example :
# [dict(type='Match',
#       tolerance=1e-8),
#  dict(type='NearMatch',
#       ratio=2,
#       tolerance=1e-8)]

Connection=[dict(type='Match',
                 tolerance=5e-7),
           ],


# key "BoundaryConditions" : this is a LIST OF DICTIONARIES. Each dictionary
# corresponds to a set of instructions for setting a boundary condition.
# Each dictionary may have at most these keys : 
#
#   name -> the name you want to give to the BC (no specific naming rules)
#
#   type -> BCWall, BCOverlap, BCSymmetryPlane, etc... If you want to
#           put the BCWall in a family (recommended), start with
#           'FamilySpecified:NAME' and replace 'NAME' with the name you want
#           to give to the family.
#           !!! WARNING : DO NOT put BC Overlap on Family !!!!
#
#   familySpecifiedType (only relevant if type starts with 'FamilySpecified:')
#           -> must be BCWall, BCSymmetryPlane...
#
#   location -> one of: 'imin','imax','jmin','jmax','kmin','kmax','special'
#
#   specialLocation (only relevant if location='special') -> if location is set
#            to 'special', then this key specifies the location of the BC.
#            it can be one of :
#
#              'fillEmpty' -> will be applied to all undefined boundaries. Use 
#                         only fillEmpty at LAST POSITION of BoundaryConditions
#                         list.
#
#              'planeXY', 'planeXZ', 'planeYZ' -> BC will be set on boundaries
#                         contained in such plane locations.
#   WARNING: in future, specalLocation key will disappear and be merged with
#   'location' key. More special key options will be implemented.

BoundaryConditions= [
   dict(name='HLPRootBCOverlap',
        type='BCOverlap',
        location='special',
        specialLocation='fillEmpty'),
                     ],


# key "OversetOptions" : this is a Dictionary. The possible keys are the
# following:
# 
# OffsetDistanceOfOverlapMask -> distance to apply to the mask surface 
#   constructed from Overlap boundaries in the INWARDS direction.
#
# ForbiddenOverlapMaskingThisBase -> LIST OF STRINGS. Each element of the
#   list is the baseName from which overlap masks are not allowed to mask
#   this base. This is an optional protection for refined regions.
# 
# OnlyMaskedByWalls -> True or False. It is False by default. If True,
#   this base is only masked by masks constructed from walls of other bases.
#   This is a VERY STRONG optional protection for refined regions.
#
# CreateMaskFromOverlap -> True or False. It is True by default. If True,
#   create mask using overlap boundary and the corresponding offset distance.
#
# MaskOffsetNormalsSmoothIterations -> INTEGER. If greater than 0, applies a 
#   smoother of normals during the offset masking process. Increasing this
#   value may help in improving the mask geometry.

OversetOptions=dict(
   OffsetDistanceOfOverlapMask=np.sqrt(3)*Snear*4,
   OnlyMaskedByWalls = True,
   ),

# key "SplitBlocks" : True or False. It is False by default. If True, this
# base zones will be split in order to facilitate proper distribution of 
# among processors.
# WARNING : If base is a background cartesian mesh implementing NearMatch
# conditions, this key MUST be set to False !! Otherwise it can lead to 
# unexpected result.
SplitBlocks=False,
     ),






# ... and that is all ! here it is the second OVERSET component :
dict(file=os.path.join('.','HLP_Center.cgns'),
     baseName='HLPCenter',
     Connection=[dict(type='Match',
                      tolerance=5e-7),
                ],
     BoundaryConditions= [
        dict(name='HLPCenterBCOverlap',
             type='BCOverlap',
             location='special',
             specialLocation='fillEmpty'),
                          ],
     OversetOptions=dict(
        OffsetDistanceOfOverlapMask=np.sqrt(3)*Snear*4,
        OnlyMaskedByWalls = True, # REMEMBER: we use this in refined bodyforce
                                  # regions because we want it to take priority
                                  # with respect to the other components !
                        ),
     SplitBlocks=False,
     ),




# ... the third overset component :
dict(file=os.path.join('.','WTP.cgns'),
     baseName='WTP',
     Connection=[dict(type='Match',
                      tolerance=5e-7),
                ],
     BoundaryConditions= [
        dict(name='WTPBCOverlap',
             type='BCOverlap',
             location='special',
             specialLocation='fillEmpty'),
                          ],
     OversetOptions=dict(
        OffsetDistanceOfOverlapMask=np.sqrt(3)*Snear*4,
        OnlyMaskedByWalls = True,
                        ),
     SplitBlocks=False,
     ),




# This is the component of the wing : 
dict(file=os.path.join('..','WING','wing.cgns'),
     baseName='BaseWing',
     Connection=[dict(type='Match',
                      tolerance=1e-8),
                ],
     BoundaryConditions= [

        dict(name='WingBCWall',
             type='FamilySpecified:wallWING',
             familySpecifiedType='BCWall',
             location='kmin'),

        dict(name='WingBCOverlap',
             type='BCOverlap',
             location='kmax'),

        dict(name='WingSymmetry',
             type='FamilySpecified:symmetry',
             familySpecifiedType='BCSymmetryPlane',
             location='special',
             specialLocation='planeXZ'),

                          ],
     OversetOptions=dict(
        OffsetDistanceOfOverlapMask=np.sqrt(3)*Snear*4,
        CreateMaskFromWall = False, # we set it to false because we know in 
                                    # advance that wing wall does not intersect
                                    # any other component. This will accelerate
                                    # a bit the preprocessing with identical
                                    # result to True value.
        ),
     SplitBlocks=True,
     ),


# And this is the Horizontal Tail Plane component:
dict(file=os.path.join('..','HTP','HTPExtruded.cgns'),
     baseName='BaseHTP',
     Connection=[dict(type='Match',
                      tolerance=1e-8),
                ],
     BoundaryConditions= [
        dict(name='HTPBCWall',
             type='FamilySpecified:wallHTP',
             familySpecifiedType='BCWall',
             location='kmin'),
        dict(name='HTPBCOverlap',
             type='BCOverlap',
             location='kmax'),
        dict(name='HTPSymmetry',
             type='FamilySpecified:symmetry',
             familySpecifiedType='BCSymmetryPlane',
             location='special',
             specialLocation='planeXZ'),
                          ],     
     OversetOptions=dict(
        OffsetDistanceOfOverlapMask=np.sqrt(3)*Snear*3,
        CreateMaskFromWall = False,
        ),
     SplitBlocks=True,
     ),


# Our last component is the Background grid. Cartesian, in this case.
# Please note the 2 elements in Connection dictionary and
# the value False of SplitBlocks !

dict(file=os.path.join('.','cartOneSide_v3.cgns'),
     baseName='BaseCartesian',
     Connection=[dict(type='Match',
                      tolerance=1e-8),
                 dict(type='NearMatch', # yes, we need to specify NearMatch
                      ratio=2,
                      tolerance=1e-8)
                ],
     BoundaryConditions= [
        dict(name='CartesianSymmetry',
             type='FamilySpecified:symmetry',
             familySpecifiedType='BCSymmetryPlane',
             location='special',
             specialLocation='planeXZ'),
        dict(name='CartesianFarfield',
             type='FamilySpecified:farfield',
             familySpecifiedType='BCFarfield',
             location='special',
             specialLocation='fillEmpty'),
                          ],

     SplitBlocks=False, # of course, we must not split this mesh !

     ),
]


'''
Main inputs are provided by InputMeshes.

Now, it is time to apply the actual preprocessing functions.

The order amount of function calls are extremely important, and must be
adapted to each particular case.

In this example we show a "typical" flow of operations that can be used
as template or reference.
'''


# first, we assemble all the meshes. This step is MANDATORY.
t = PRE.getMeshesAssembled(InputMeshes)


# second, we make scaling of meshes. This step is OPTIONAL.
PRE.transform(t, InputMeshes)


# third, we connect the mesh and fourth, we apply boundary conditions.
# THESE STEPS ARE OPTIONAL (i.e. input meshes has all BC and connectivity
# already defined).
# BEWARE : even if Input Meshes have defined some BC conditions, but not ALL
# of them, PRE.setBoundaryConditions(t, InputMeshes) can be used to define the
# remaining BCs to produce a completely defined case.
PRE.connectMesh(t, InputMeshes)
PRE.setBoundaryConditions(t, InputMeshes)


# Fifth, we split and distribute the mesh. This step is MANDATORY, because
# distribution is mandatory. If you only want to distribute and NOT SPLIT,
# then set the key SplitBlocks=False to the dictionaries of InputMeshes.
t = PRE.splitAndDistribute(t, InputMeshes,
                           NProcs=NProcs,
                           ProcPointsLoad=ProcPointsLoad)



# --------- OPTIONAL :
# Let us imagine that our case has already defined some BCs that are NOT 
# set into families. Here, we would want to group those BCs into families,
# including possible changes in BC types of such families.
#
# In the following example, we would group all existing BCWall in a family
# named "wallWING", and all auxiliary "BCExtrapolate" in a family named
# 'wallHTP' that will be then converted into BCWall type :
#
# I._groupBCByBCType(t, btype='BCWall', name='wallWING')
# I._groupBCByBCType(t, btype='BCExtrapolate', name='wallHTP')
# I._renameNode(t, 'BCExtrapolate', 'BCWall')
#
# This step is useful for setting complex BCs using tkCassiopee Graphical
# User Interface, as it cannot currently add BCs with Families. For this reason
# we can use auxiliary BC types (such as BCExtrapolate) for creating new
# families.


# Sixth, add families to bases, tag zones, and set FamilyBCs. If the 
# case has families (which is recommended), then this step is MANDATORY.
PRE.addFamilies(t, InputMeshes)

# Seventh, if at least one zone has been split, reconnecting mesh has to be done
PRE.connectMesh(t, InputMeshes)

# Eighth, produce Overset data (ONLY static overset currently). This step
# is MANDATORY only if BCOverlap exist in the case.
t = PRE.addOversetData(t, InputMeshes, saveMaskBodiesTree=True)

# Ninth, Make adaptations inherent to elsA. This step is already contained
# in previous step. So if addOversetData is applied, then this line is 
# redundant
PRE.EP._convert2elsAxdt(t)


# Final check if all BCs are defined
PRE.J.checkEmptyBC(t)

# save the result
C.convertPyTree2File(t,'mesh.cgns')

# print the script elapsed time
print('Elaped time: %g minutes'%((tic()-toc)/60.))