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

from .external_flow import ExternalFlowGenerator
from .internal_flow import InternalFlowGenerator
from .external_flow_Mach_P_T import ExternalMPTFlowGenerator

AvailableFlowGenerators = dict()
for gen in [ExternalFlowGenerator, 
            InternalFlowGenerator,
            ExternalMPTFlowGenerator]:
    
    AvailableFlowGenerators[gen.name] = gen

def get_flow_generator(fg):
    if isinstance(fg, str):
        return AvailableFlowGenerators[fg]
    else:
        return fg
    