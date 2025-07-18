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
from ... import WorkflowInterface


class WorkflowAirplaneInterface(WorkflowInterface):

    def __init__(self, workflow, tree=None, **kwargs):
        super().__init__(workflow, tree, **kwargs)
        if tree is None:
            self.add_to_Extractions_BC(Source='Wall*', Fields=['Pressure', 'BoundaryLayer', 'yPlus'])
            self.add_to_Extractions_Integral(
                Source='Wall*',
                Fields=['Force', 'Torque'],
                PostprocessOperations=[
                    dict(Type="compute_aerodynamic_coefficients", AtEndOfRunOnly=False),
                    dict(Type='avg', Variable='CL'),
                    dict(Type='std', Variable='CL'),
                    dict(Type='avg', Variable='CD'),
                    dict(Type='std', Variable='CD'),
                ]
            )

    def set_ApplicationContext(self, 
        AngleOfAttackDeg : float = 0.0,
        AngleOfSlipDeg : float = 0.0,
        YawAxis : list = [0.0,0.0,1.0],
        PitchAxis : list = [0.0,1.0,0.0],
        Length : float = 1.0,
        Surface : float = 1.0):

        self.ApplicationContext = self._get_comp(self.set_ApplicationContext, self.get_default_values_from_local_signature())

    def set_SplittingAndDistribution(self,
            Strategy    : str = 'AtComputation',
            Splitter    : str = 'PyPart',
            Distributor : str = 'PyPart',
            **kwargs):

        super().set_SplittingAndDistribution(
            **self.get_default_values_from_local_signature(),
            **kwargs)
