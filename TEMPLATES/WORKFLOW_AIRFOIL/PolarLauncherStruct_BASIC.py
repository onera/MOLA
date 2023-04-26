#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

import MOLA.WorkflowAirfoil as WF

AER = '28643013F'
FILE_GEOMETRY = 'psu94097.txt'

# To modify for each polar computation set:
PREFIX_JOB = 'a'
DIRECTORY_WORK = '/tmp_user/sator/lbernard/MYPOLAR/'

AoARange    = [0,1,2,3,4,-1,-2,-3,-4]
MachRange  =  [0.6,0.7,0.8]
ReynoldsOverMach = 1e6/0.8

WF.launchBasicStructuredPolars(PREFIX_JOB, FILE_GEOMETRY, AER, 'sator',
                          DIRECTORY_WORK, AoARange, MachRange, ReynoldsOverMach)
