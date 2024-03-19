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
from mola.workflow.workflow import Workflow, deep_update


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
            
            self.get_yaw_and_pitch_axes()

            self.Extractions.extend([
                dict(type='bc', BCType='BCInflow*', fields=['MassFlow']),
                dict(type='bc', BCType='BCOutflow*', fields=['MassFlow']),
            ])
    
    def set_yaw_and_pitch_axes(self):
        self.set_yaw_axis()
        self.set_pitch_axis()
        
    def set_yaw_axis(self):
        if 'YawAxis' not in self.Flow:
            # Get periodic match connections
            perio_connections = [connec for connec in self.RawMeshComponents['Connection'] if connec['Type'] == 'PeriodicMatch']
            if len(perio_connections) == 1:
                YawAxis = np.array(perio_connections[0]['Translation'])
                self.YawAxis = YawAxis / np.sqrt(np.sum(YawAxis**2))
            elif len(perio_connections) == 0:
                # Check that Periodicity already given in the mesh and adapt it if necessary
                # For now raise an exception
                raise Exception('Not yet implemented')
            else:
                raise Exception('More than one PeriodicMatch: Please give both YawAxis and PitchAxis')

    def set_pitch_axis(self):
        if 'SpanwiseDirection' in self.Flow:
            self.PitchAxis = self.Flow['SpanwiseDirection']
        else:
            RollAxis = np.array([1,0,0]) # Strong assumption here
            self.PitchAxis = np.cross(self.YawAxis, RollAxis)
