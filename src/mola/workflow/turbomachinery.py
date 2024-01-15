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
                 Splitter='PyPart',
                 FlowGenerator='Internal',
                 **kwargs
                 ):
        
        super().__init__(Splitter=Splitter, FlowGenerator=FlowGenerator, **kwargs)

        self.name = 'Turbomachinery'

        if self.tree is not None:
            for meshInfo in self.RawMeshComponents:
                meshInfo.setdefault('mesher', 'Autogrid')

            self.TurboConfiguration = dict()

            self.Extractions.append(
                dict(type='bc', BCType='BCInflow*', fields=['MassFlow']),
                dict(type='bc', BCType='BCOutflow*', fields=['MassFlow']),
            )

    def set_turbo_configuration(self):
        for row, rowParams in self.TurboConfiguration['Rows'].items():
            for key, value in rowParams.items():
                if key == 'RotationSpeed' and value == 'auto':
                    rowParams[key] = self.TurboConfiguration['ShaftRotationSpeed']
            if hasattr(self, 'BodyForceInputData') and row in self.BodyForceInputData:
                # Replace the number of blades to be consistant with the body-force mesh
                deltaTheta = computeAzimuthalExtensionFromFamily(self.tree, row)
                rowParams['NumberOfBlades'] = int(2*np.pi / deltaTheta)
                rowParams['NumberOfBladesInInitialMesh'] = 1
                print(f'Number of blades for {row}: {rowParams["NumberOfBlades"]} (got from the body-force mesh)')
            if not 'NumberOfBladesSimulated' in rowParams:
                rowParams['NumberOfBladesSimulated'] = 1
            if not 'NumberOfBladesInInitialMesh' in rowParams:
                rowParams['NumberOfBladesInInitialMesh'] = getNumberOfBladesInMeshFromFamily(self.tree, row, rowParams['NumberOfBlades'])
            
