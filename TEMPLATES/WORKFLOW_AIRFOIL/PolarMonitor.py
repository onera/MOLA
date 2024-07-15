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

import pprint
import MOLA.WorkflowAirfoil as WA

config = WA.JM.loadJobsConfiguration()

WA.printConfigurationStatus(config.DIRECTORY_WORK)

AoA = 6.0
Mach = 0.2
CASE_LABEL = WA.getCaseLabelFromAngleOfAttackAndMach(config, AoA, Mach)
Reynolds = WA.getReynoldsFromCaseLabel(config, CASE_LABEL)
IntegralLoads = WA.JM.getCaseArrays(config, CASE_LABEL)
print(pprint.pformat(IntegralLoads))
DistributedLoads = WA.getCaseDistributions(config, CASE_LABEL)
WA.getCaseFields(config, CASE_LABEL)
#
WA.compareAgainstXFoil(config.FILE_GEOMETRY, config, CASE_LABEL, DistributedLoads)
