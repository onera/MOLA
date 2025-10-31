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
import mola.naming_conventions as names
from ..interface import WorkflowRotatingComponentInterface

class WorkflowTurbomachineryInterface(WorkflowRotatingComponentInterface):

    def __init__(self, workflow, tree=None, **kwargs):
        super().__init__(workflow, tree, **kwargs)
        if tree is None:
            self.add_to_Extractions_BC(Source='WallViscous', Fields=['Pressure', 'BoundaryLayer', 'yPlus'])
            self.add_to_Extractions_Integral(Source='Inflow*', Fields=['MassFlow'])
            self.add_to_Extractions_Integral(Source='Outflow*', Fields=['MassFlow'])

    def add_to_RawMeshComponents(self,
        Mesher        : str  = 'Autogrid',
        **kwargs):
        local_kwargs = self.get_default_values_from_local_signature()
        local_kwargs.update(kwargs)
        super().add_to_RawMeshComponents(**local_kwargs)

    def set_ApplicationContext(self,
            ShaftAxis : Union[list,
                            tuple,
                            np.ndarray] = [1,0,0],
            ShaftRotationSpeedUnit : str = 'rpm', 
            HubRotationIntervals : list = None,
            Surface : float = None,
            NormalizationCoefficient : dict = None,
            RowType : str = 'Compressor',  # specific parameter to this Workflow
            *,
            ShaftRotationSpeed : Union[float, int] = None,
            Rows : dict = dict(),
            ):

        super().set_ApplicationContext(
            ShaftAxis=ShaftAxis,
            ShaftRotationSpeedUnit=ShaftRotationSpeedUnit,  
            HubRotationIntervals=HubRotationIntervals, 
            Surface=Surface,
            NormalizationCoefficient=NormalizationCoefficient,
            ShaftRotationSpeed=ShaftRotationSpeed,
            Rows=Rows
            )
        self.ApplicationContext['RowType'] = RowType

    def add_Row_to_ApplicationContext(self,
        IsRotating : bool = False,
        NumberOfBladesSimulated : int = 1,
        NumberOfBladesInInitialMesh : int = None, 
        FlowAngleAtRootDeg : float = None,
        FlowAngleAtTipDeg : float = None,
        *,
        _Key : str,  # auxilary parameter, removed at the end of this function
        NumberOfBlades : int,
        ):
        self.ApplicationContext['Rows'][_Key].update(
            self._get_comp(
                WorkflowRotatingComponentInterface.add_Row_to_ApplicationContext, 
                self.get_default_values_from_local_signature()
                )
            )
        self.ApplicationContext['Rows'][_Key].pop('_Key')

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

