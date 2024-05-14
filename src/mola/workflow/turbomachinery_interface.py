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

import os
import copy
import numpy as np
import copy
from treelab import cgns
from treelab.cgns.tree import Tree
from treelab.cgns.base import Base
from treelab.cgns.zone import Zone
import inspect
from typing import Union, Callable, Dict, get_type_hints
from mola.logging import (mola_logger,
                       MolaException,
                       MolaUserError,
                       MolaUserAttributeError,
                       redirect_streams_to_logger,
                       get_signature)
from mola.logging.formatters import BOLD, RED, CYAN, PINK, YELLOW, ENDC

from . import WorkflowInterface

class WorkflowTurbomachineryInterface(WorkflowInterface):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def add_to_RawMeshComponents(self,
        Mesher        : str  = 'Autogrid',
        **kwargs):
        local_kwargs = self.repack_kwargs()
        local_kwargs.update(kwargs)
        return super().add_to_RawMeshComponents(**local_kwargs)

    def set_Flow(self,
                Generator : str = 'Internal',
                **kwargs):
        local_kwargs = self.repack_kwargs()
        local_kwargs.update(kwargs)
        return super().set_Flow(**local_kwargs)

    def set_SplittingAndDistribution(self, 
            Strategy                         : str = 'AtComputation',
            Splitter                         : str = 'PyPart',
            Distributor                      : str = 'PyPart',
            **kwargs):
        return super().set_SplittingAndDistribution(**self.repack_kwargs())
        

    def set_ApplicationContext(self,
            ShaftAxis : Union[list,
                            tuple,
                            np.ndarray] = [1,0,0],
            
            Rows : dict = None,
            HubRotationSpeed : list = None,
            ShaftRotationSpeed : float = None,
            NormalizationCoefficient : dict = None):
        # shall make _get_comp accessible (staticmethod?)
        self.ApplicationContext = self._get_comp(self.set_ApplicationContext, self.repack_kwargs())

        self.ApplicationContext['ShaftAxis'] = np.array(self.ApplicationContext['ShaftAxis'],dtype=float)

    def add_to_Extractions_Integral(self,
            File : str = "turbo_signals.cgns",
            Frame : str = 'relative',
            **kwargs):
        '''
        Summation over a given source of the mesh, providing a scalar integral value
        '''
        local_kwargs = self.repack_kwargs()
        local_kwargs.update(kwargs)
        return super().add_to_Extractions_Integral(**local_kwargs)


class WorkflowCompressorInterface(WorkflowTurbomachineryInterface):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def set_Flow(self,
                Generator : str = 'InternalCompressor',
                **kwargs):
        local_kwargs = self.repack_kwargs()
        local_kwargs.update(kwargs)
        return super().set_Flow(**local_kwargs)

    def add_to_Extractions_Integral(self,
            File : str = "compressor_signals.cgns",
            **kwargs):
        '''
        Summation over a given source of the mesh, providing a scalar integral value
        '''
        local_kwargs = self.repack_kwargs()
        local_kwargs.update(kwargs)
        return super().add_to_Extractions_Integral(**local_kwargs)
    

    def add_to_RawMeshComponents(self,
            Mesher : str = "Autogrid2",
            **kwargs):
        local_kwargs = self.repack_kwargs()
        local_kwargs.update(kwargs)
        return super().add_to_RawMeshComponents(**local_kwargs)