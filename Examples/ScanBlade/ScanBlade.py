import sys
import os
import numpy as np

import Converter.PyTree as C 
import Converter.Internal as I

import MOLA.InternalShortcuts as J
import MOLA.GenerativeShapeDesign as GSD

BladeSurface = C.convertFile2PyTree('MainBladeGeometry.cgns')
RelativeSpanDistribution = np.linspace(0.25, 0.95, 41)
RotationCenter = np.array([0.5*(-0.078172-0.0228388),-0.0249745,0.0])
RotationAxis = np.array([-1.0, 0.0, 0.0])
BladeDirection = np.array([0.0, 0.0, 1.0])


scan = GSD.scanBlade(BladeSurface, RelativeSpanDistribution,
                     RotationCenter, RotationAxis, BladeDirection,
                     buildCamberOptions=dict(
                        SearchPortions=(+0.50,-0.50),
                                             ),
                     )

C.convertPyTree2File(scan, 'BladeScanner.cgns')