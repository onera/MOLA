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

from . import WorkflowInterface


class WorkflowLinearCascadeInterface(WorkflowInterface):

    def __init__(self, workflow, tree=None, **kwargs):
        super().__init__(workflow, tree, **kwargs)
        if tree is None:
            self.add_to_Extractions_BC(Source='BCWall*', Fields=['Pressure', 'BoundaryLayer', 'yPlus'])
            self.add_to_Extractions_Integral(Source='BCInflow*', Fields=['MassFlow'])
            self.add_to_Extractions_Integral(Source='BCOutflow*', Fields=['MassFlow'])

    def set_ApplicationContext(self, 
            AngleOfAttackDeg : float = 0.,
        ):
        self.ApplicationContext = self._get_comp(self.set_ApplicationContext, self.get_default_values_from_local_signature())

    def add_to_RawMeshComponents(self,
        Mesher        : str  = 'Autogrid',
        **kwargs):
        local_kwargs = self.get_default_values_from_local_signature()
        local_kwargs.update(kwargs)
        return super().add_to_RawMeshComponents(**local_kwargs)

    def set_Flow(self,
                Generator : str = 'Internal',
                **kwargs):
        local_kwargs = self.get_default_values_from_local_signature()
        local_kwargs.update(kwargs)
        return super().set_Flow(**local_kwargs)

    def set_SplittingAndDistribution(self, 
            Strategy                         : str = 'AtComputation',
            Splitter                         : str = 'PyPart',
            Distributor                      : str = 'PyPart',
            **kwargs):
        return super().set_SplittingAndDistribution(**self.get_default_values_from_local_signature())
        