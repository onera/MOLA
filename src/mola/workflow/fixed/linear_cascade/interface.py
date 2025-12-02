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

from typing import Union
from treelab import cgns
from mola import naming_conventions as names
from ... import WorkflowInterface


class WorkflowLinearCascadeInterface(WorkflowInterface):

    def __init__(self, workflow, tree=None, **kwargs):
        super().__init__(workflow, tree, **kwargs)
        if tree is None:
            self.add_to_Extractions_BC(Source='Wall*', Fields=['Pressure', 'BoundaryLayer', 'yPlus'])
            self.add_to_Extractions_Integral(Source='Inflow*', Fields=['MassFlow'])
            self.add_to_Extractions_Integral(Source='Outflow*', Fields=['MassFlow'])

    def set_ApplicationContext(self, 
            AngleOfAttackDeg : float = None,
        ):
        self.ApplicationContext = self._get_comp(self.set_ApplicationContext, self.get_default_values_from_local_signature())

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
        super().set_SplittingAndDistribution(**self.get_default_values_from_local_signature())
    
    def set_Initialization(self,
            Method    : str  = 'uniform',
            Source    : Union[     str,
                                  cgns.Tree,
                                  cgns.Base,
                                  cgns.Zone ]  = None,
            SourceContainer : str = None,
            ComputeWallDistanceAtPreprocess : bool = False,
            WallDistanceComputingTool : str = 'maia',
            ParametrizeWithHeight : str = None, # parameter specific to that workflow
            ):
        self.Initialization = self._get_comp(
            self.set_Initialization, self.get_default_values_from_local_signature())
        
    def add_to_Extractions_IsoSurface(self,
            ContainersToTransfer : Union[ str, # accepts "all"
                                                   list ] = [names.CONTAINER_OUTPUT_FIELDS_AT_VERTEX,
                                                             "FlowSolution#Height"],
            **kwargs):
        '''
        Summation over a given source of the mesh, providing a scalar integral value
        '''
        local_kwargs = self.get_default_values_from_local_signature()
        local_kwargs.update(kwargs)
        super().add_to_Extractions_IsoSurface(**local_kwargs)

        