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
from typing import Union

from .rotating_component_interface import WorkflowRotatingComponentInterface


class WorkflowTurbomachineryInterface(WorkflowRotatingComponentInterface):

    def __init__(self, workflow, tree=None, **kwargs):
        super().__init__(workflow, tree, **kwargs)
        if tree is None:
            self.add_to_Extractions_BC(Source='BCWallViscous', Fields=['Pressure', 'BoundaryLayer', 'yPlus'])
            self.add_to_Extractions_Integral(Source='BCInflow*', Fields=['MassFlow'])
            self.add_to_Extractions_Integral(Source='BCOutflow*', Fields=['MassFlow'])

    def add_to_RawMeshComponents(self,
        Mesher        : str  = 'Autogrid',
        **kwargs):
        local_kwargs = self.get_default_values_from_local_signature()
        local_kwargs.update(kwargs)
        super().add_to_RawMeshComponents(**local_kwargs)

    def set_Flow(self,
                Generator : str = 'Internal',
                **kwargs):
        local_kwargs = self.get_default_values_from_local_signature()
        local_kwargs.update(kwargs)
        super().set_Flow(**local_kwargs)

    def set_SplittingAndDistribution(self, 
            Strategy                         : str = 'AtComputation',
            Splitter                         : str = 'PyPart',
            Distributor                      : str = 'PyPart',
            **kwargs):
        local_kwargs = self.get_default_values_from_local_signature()
        local_kwargs.update(kwargs)
        super().set_SplittingAndDistribution(**local_kwargs)

    def set_Numerics(self,
            Scheme : str   = 'Roe',
            **kwargs):
        local_kwargs = self.get_default_values_from_local_signature()
        local_kwargs.update(kwargs)
        super().set_Numerics(**local_kwargs)

    def add_to_Extractions_Integral(self,
            File : str = 'signals.cgns',
            Frame : str = 'relative',
            **kwargs):
        '''
        Summation over a given source of the mesh, providing a scalar integral value
        '''
        local_kwargs = self.get_default_values_from_local_signature()
        local_kwargs.update(kwargs)
        super().add_to_Extractions_Integral(**local_kwargs)
