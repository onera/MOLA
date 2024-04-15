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

import numpy as np
from . import Workflow


class WorkflowLinearCascade(Workflow):

    def __init__(self, 
                 SplittingAndDistribution='PyPart',
                 FlowGenerator='Internal',
                 **kwargs
                 ):
    
        super().__init__(SplittingAndDistribution=SplittingAndDistribution, FlowGenerator=FlowGenerator, **kwargs)
        
        # channel height computation
        # postprocess on internal component, between two planes
        # automatic postprocess with turbo for cascade

        if self.tree is not None:
            for meshInfo in self.RawMeshComponents:
                meshInfo.setdefault('mesher', 'Autogrid')
            
            self.Extractions.extend([
                dict(type='bc', BCType='BCInflow*', fields=['MassFlow']),
                dict(type='bc', BCType='BCOutflow*', fields=['MassFlow']),
            ])
        


    # def get_periodic_direction(self):
    #     # Get periodic match connections
    #     perio_connections = [connec for connec in self.RawMeshComponents['Connection'] if connec['Type'] == 'PeriodicMatch']
    #     if len(perio_connections) == 1:
    #         periodic_direction = np.array(perio_connections[0]['Translation'])
    #         periodic_direction /= np.sqrt(np.sum(periodic_direction**2))
    #     elif len(perio_connections) == 0:
    #         # Check that Periodicity already given in the mesh and adapt it if necessary
    #         # For now raise an exception
    #         raise Exception('Not yet implemented')
    #     else:
    #         raise Exception('More than one PeriodicMatch')
        
    #     return periodic_direction

