'''
This example shows how to extract the camber line of an airfoil
(also known as skeleton line). Furthermore, it is also shown the
inverse operation: build an airfoil from a camber line.
'''

# Import system modules
import sys
import numpy as np

# Import Cassiopee modules
import Converter.PyTree as C
import Converter.Internal as I
import Geom.PyTree as D

# Import MOLA modules
import MOLA.Wireframe as W
import MOLA.InternalShortcuts as J


# ------------------------------------------------ #
# Step 1: Build an arbitrary airfoil
# ------------------------------------------------ #

MyFoil = W.airfoil('NACA4412')
# NOTE: MyFoil is a 1D-Structured PyTree Zone

I.setName(MyFoil, 'OriginalAirfoil')

# ------------------------------------------------ #
# Step 2: Get the camber line of the airfoil
# ------------------------------------------------ #
CamberLine = W.buildCamber(MyFoil)
# NOTE: CamberLine is ALSO a 1D-Structured PyTree Zone
# with FlowSolution field "RelativeThickness", which
# represents the  thickness of each point of the CamberLine

I.setName(CamberLine, 'CamberLine')

# ------------------------------------------------ #
# Step 3: Build an airfoil from the camber line
# ------------------------------------------------ #

NewFoil = W.buildAirfoilFromCamberLine(CamberLine)
# NOTE: NewFoil is a 1D-Structured PyTree Zone

I.setName(NewFoil, 'NewFoil')


# ------------------------------------------------ #
# Step 4: Save the result
# ------------------------------------------------ #
C.convertPyTree2File([MyFoil, CamberLine, NewFoil],
                     'buildAirfoilFromCamberLine.cgns')