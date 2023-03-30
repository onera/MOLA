#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

import MOLA.WorkflowAirfoil as WA
import setup
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator

Field2Plot = 'Cp'

C, J = WA.C, WA.J
SurfsTree = C.convertFile2PyTree( 'surfaces.cgns' )
foil = WA.convertSurfaces2OrientedAirfoilCurveAtVertex(SurfsTree)
WA.addRelevantWallFieldsFromElsAFieldsAtVertex(foil,
                                   setup.ReferenceValues['PressureDynamic'],
                                   setup.ReferenceValues['Pressure'],)
WallFields = J.getVars2Dict(foil, C.getVarNames(foil,excludeXYZ=True)[0])
x,y,z = J.getxyz(foil)
WallFields.update( dict(CoordinateX=x, CoordinateY=y, CoordinateZ=z) )


CGNSname2LaTeXLabel = {'Cp':'$C_p$','Cf':'$C_f$','CoordinateX':'$x/c$',
    'CoordinateY':'$y/c$', 'ReynoldsTheta':'$Re_\\theta$','theta11':'$\\theta/c$',
    'delta1':'$\\delta^*/c$','hi':'$H$'}

fig, ax = plt.subplots(1,1,dpi=150)
ax.plot(WallFields['CoordinateX'],WallFields[Field2Plot], label='CFD')
if Field2Plot == 'Cp': ax.invert_yaxis()
try: ax.set_ylabel(CGNSname2LaTeXLabel[Field2Plot])
except KeyError: ax.set_ylabel(Field2Plot)
ax.set_xlabel(CGNSname2LaTeXLabel['CoordinateX'])
minLocX = AutoMinorLocator()
ax.xaxis.set_minor_locator(minLocX)
minLocY = AutoMinorLocator()
ax.yaxis.set_minor_locator(minLocY)
ax.xaxis.grid(True, which='major')
ax.xaxis.grid(True, which='minor',linestyle=':')
ax.yaxis.grid(True, which='major')
ax.yaxis.grid(True, which='minor',linestyle=':')
ax.legend(loc='best')
plt.tight_layout()
plt.savefig(Field2Plot+'.svg')
plt.show()
