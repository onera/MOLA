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

from treelab import cgns

from mola.logging import mola_logger, MolaException, MolaUserError
from .. import WorkflowInterface

class WorkflowRotatingComponentInterface(WorkflowInterface):

    def __init__(self, workflow, tree=None, **kwargs):
        super().__init__(workflow, tree, **kwargs)


    def set_ApplicationContext(self,
            ShaftAxis : Union[list,
                            tuple,
                            np.ndarray] = [1,0,0],
            ShaftRotationSpeedUnit : str = 'rpm', 
            HubRotationIntervals : list = None,
            Surface : float = None,
            NormalizationCoefficient : dict = None,
            *,
            ShaftRotationSpeed : Union[float, int] = None,
            Rows : dict = dict(),
            ):

        ShaftAxis = np.array(ShaftAxis,dtype=float)
        ShaftAxis /= np.linalg.norm(ShaftAxis)

        kwargs = self.get_default_values_from_local_signature()
        self.ApplicationContext = self._get_comp(WorkflowRotatingComponentInterface.set_ApplicationContext, kwargs)

        for key, row_parameters in Rows.items():
            self.add_Row_to_ApplicationContext(_Key=key, **row_parameters)
        

        self.apply_ShaftRotationSpeedUnit(default_ShaftRotationSpeedUnit=kwargs["ShaftRotationSpeedUnit"])
        if 'HubRotationIntervals' in self.ApplicationContext:
            self.set_HubRotationIntervals()
        
    def apply_ShaftRotationSpeedUnit(self, default_ShaftRotationSpeedUnit):
        if not self.ApplicationContext['ShaftRotationSpeedUnit'].lower() in ['rpm', 'rad/s']:
            raise MolaUserError(f'ShaftRotationSpeedUnit must be rpm or rad/s ({default_ShaftRotationSpeedUnit} by default)')
        if self.ApplicationContext['ShaftRotationSpeedUnit'].lower() == 'rpm':
            self.ApplicationContext['ShaftRotationSpeed'] *= np.pi / 30.
            self.ApplicationContext['ShaftRotationSpeedUnit'] = 'rad/s'
        self.ApplicationContext['ShaftRotationSpeed'] = float(self.ApplicationContext['ShaftRotationSpeed'])

    def set_HubRotationIntervals(self):
        if callable(self.ApplicationContext['HubRotationIntervals']):
            return
        
        HubRotationIntervals = []
        for interval in self.ApplicationContext['HubRotationIntervals']:
            raise_error = False
            if isinstance(interval, (tuple, list)):
                if not len(interval) == 2:
                    raise_error = True
                xmin, xmax = interval
                HubRotationIntervals.append(
                    dict(xmin=xmin, xmax=xmax)
                )
            elif isinstance(interval, dict):
                try:
                    xmin = interval.get('xmin', -1e20)
                    xmax = interval.get('xmax',  1e20)
                    HubRotationIntervals.append(
                        dict(xmin=xmin, xmax=xmax)
                    )
                except:
                    raise_error = True
            else:
                raise_error = True
                
            if raise_error:
                raise MolaUserError(
                    'Each element of HubRotationIntervals must be either a tuple or list '
                    'of 2 values (xmin, xmax), or a dict with 2 elements called xmin and xmax. '
                    f'Current value of HubRotationIntervals is {HubRotationIntervals}'
                    )
            
        self.ApplicationContext['HubRotationIntervals'] = HubRotationIntervals

    def add_Row_to_ApplicationContext(self,
            IsRotating : bool = False,
            NumberOfBladesSimulated : int = 1,
            NumberOfBladesInInitialMesh : int = None, 
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
