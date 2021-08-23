'''
ModifyAirfoil.py example

Shows how to modify an existing airfoil through its main
geometrical characteristics.

03/12/2020 - L. Bernardos - First creation
'''

import sys
import numpy as np 

import matplotlib.pyplot as plt

import Converter.PyTree as C
import Converter.Internal as I

import MOLA.InternalShortcuts as J
import MOLA.Wireframe as W

# Let us construct an airfoil for this example
Airfoil = W.airfoil('NACA9510')

# Let us print some geometrical properties of this airfoil (for convenience)
AirfoilProperties, AirfoilCamber = W.getAirfoilPropertiesAndCamber(Airfoil)
for prop in AirfoilProperties:
    print('%s: %s'%(prop, str(AirfoilProperties[prop])))

'''
In next step, some of these airfoil properties will be modified.
Currently implemented modifiable parameters are:

'Chord'
'MaxThickness' or 'MaxRelativeThickness'
'MaxThicknessRelativeLocation'
'MaxCamber' or 'MaxRelativeCamber'
'MaxCamberRelativeLocation'
'MinCamber' or 'MinRelativeCamber'
'MinCamberRelativeLocation'
'''

# Let us now modify some of the aforementioned parameters
NewAirfoil = W.modifyAirfoil(Airfoil, Chord=None,
                  MaxThickness=None, MaxRelativeThickness=0.15,
                  MaxThicknessRelativeLocation=0.20,
                  MaxCamber=None, MaxRelativeCamber=None,
                  MaxCamberRelativeLocation=None,
                  MinCamber=None, MinRelativeCamber=None,
                  MinCamberRelativeLocation=None,
                  buildCamberOptions={},
                  InterpolationLaw='interp1d_cubic')

C.convertPyTree2File([Airfoil, NewAirfoil], 'ModifyAirfoil.cgns')

# Plot the result
Plots = [
dict(foil=Airfoil, foilCurveProps=dict(color='silver'),
                   camberCurveProps=dict(color='silver', linestyle=':')),

dict(foil=NewAirfoil, foilCurveProps=dict(color='k'),
                   camberCurveProps=dict(color='k', linestyle=':')),
]


fig, ax = plt.subplots(figsize=(5.,2.), dpi=200)
for Plot in Plots:
    _, camber = W.getAirfoilPropertiesAndCamber(Plot['foil'])
    FoilX, FoilY = J.getxy(Plot['foil'])
    CamberX, CamberY = J.getxy(camber)
    ax.plot(FoilX, FoilY, **Plot['foilCurveProps'])
    ax.plot(CamberX, CamberY, **Plot['camberCurveProps'])
ax.set_xlim(-0.05,1.1)
ax.set_ylim(-0.1,0.3)
ax.set_aspect('equal', 'box')
fig.tight_layout()
plt.axis('off')

fig.savefig('ModifyAirfoil.png')
plt.show()