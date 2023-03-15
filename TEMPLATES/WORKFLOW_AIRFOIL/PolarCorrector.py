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

t = WA.C.convertFile2PyTree('Polar.cgns')
WA.correctPolar(t, useBigRangeValuesIf_StdCLisHigherThan=0.0005,
                   Fields2Correct=['Cl','Cd','Cm'])
WA.C.convertPyTree2File(t, 'PolarCorrected.cgns')
