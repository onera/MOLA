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
from mola.workflow import WorkflowRotatingComponent

class WorkflowTurbomachinery(WorkflowRotatingComponent):

    def __init__(self, 
                 SplittingAndDistribution='PyPart',
                 FlowGenerator='Internal',
                 **kwargs
                 ):
        
        super().__init__(SplittingAndDistribution=SplittingAndDistribution, FlowGenerator=FlowGenerator, **kwargs)

        if self.tree is not None:
            for meshInfo in self.RawMeshComponents:
                meshInfo.setdefault('mesher', 'Autogrid')

            self.Extractions.append(
                dict(type='bc', BCType='BCWall*', fields=['Pressure', 'BoundaryLayer', 'yPlus']),
                dict(type='bc', BCType='BCInflow*', fields=['MassFlow']),
                dict(type='bc', BCType='BCOutflow*', fields=['MassFlow']),
            )
