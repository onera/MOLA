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
import numpy as np
from mola.workflow import WorkflowRotatingComponent

class WorkflowTurbomachinery(WorkflowRotatingComponent):

    def __init__(self, 
                 SplittingAndDistribution=None,
                 Flow=None,
                 **kwargs
                 ):
        
        super().__init__(SplittingAndDistribution=SplittingAndDistribution,
                         Flow=Flow,
                         **kwargs)

        if self.tree is None:
            for meshInfo in self.RawMeshComponents:
                meshInfo.setdefault('mesher', 'Autogrid')

            self.Extractions.extend([
                dict(Type='BC', Source='BCWall*', Fields=['Pressure', 'BoundaryLayer', 'yPlus']),
                dict(Type='BC', Source='BCInflow*', Fields=['MassFlow']),
                dict(Type='BC', Source='BCOutflow*', Fields=['MassFlow']),
            ])

    def set_Flow(self,
            Generator : str = 'Internal',
            Velocity  : float = 1.0,
            # Parameters relevant to InternalFlowGenerator
            MassFlow               : float = None,
            Mach                   : float = None,
            PressureStagnation     : float = None,
            TemperatureStagnation  : float = None,
            IdealGasConstant       : float = None,
            Gamma                  : float = None,
            ):
        return super().set_Flow(**self.repack_kwargs())

    def set_SplittingAndDistribution(self, 
            Strategy                         : str = 'AtComputation',
            Splitter                         : str = 'PyPart',
            Distributor                      : str = 'PyPart',
            ComponentsToSplit                : Union[ str,
                                                    None,
                                                    list ] = 'all',
            NumberOfProcessors               : Union[ str,
                                                    int]  = 'auto',
            MinimumAllowedNodes              : int = 1,
            MaximumAllowedNodes              : int = 20,
            MaximumNumberOfPointsPerNode     : int = int(1e9),
            CoresPerNode                     : int = 48,
            DistributeExclusivelyOnFullNodes : bool = True,
                       ):
        return super().set_SplittingAndDistribution(**self.repack_kwargs())
        

    def set_ApplicationContext(self,
            ShaftAxis : Union[list,
                             tuple,
                             np.ndarray] = [1,0,0],
            
            # TODO : redefine as Workflow's attributes with set_* and add_to_* ?
            Rows : dict = None,
            HubRotationSpeed : list = None,
            ShaftRotationSpeed : float = None,
            NormalizationCoefficient : dict = None):
        '''
        
        '''
        # shall make _get_comp accessible (staticmethod?)
        self.ApplicationContext = self._get_comp(self.set_ApplicationContext, self.repack_kwargs())

        self.ApplicationContext['ShaftAxis'] = np.array(self.ApplicationContext['ShaftAxis'],dtype=float)