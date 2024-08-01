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

from mola.logging import mola_logger, MolaException, MolaUserError
from . import WorkflowInterface

class WorkflowRotatingComponentInterface(WorkflowInterface):

    def set_ApplicationContext(self,
            ShaftAxis : Union[list,
                            tuple,
                            np.ndarray] = [1,0,0],
            ShaftRotationSpeedUnit : str = 'rad/s', 
            HubRotationSpeed : list = None,
            NormalizationCoefficient : dict = None,
            *,
            ShaftRotationSpeed : float = None,
            Rows : dict = dict(),
            ):
        kwargs = self.get_default_values_from_local_signature()
        self.ApplicationContext = self._get_comp(self.set_ApplicationContext, kwargs)

        for key, row_parameters in Rows.items():
            self.add_Row_to_ApplicationContext(_Key=key, **row_parameters)
        
        self.ApplicationContext['ShaftAxis'] = np.array(self.ApplicationContext['ShaftAxis'],dtype=float)
        
        if not self.ApplicationContext['ShaftRotationSpeedUnit'].lower() in ['rpm', 'rad/s']:
            raise MolaUserError(f'ShaftRotationSpeedUnit must be rpm or rad/s ({kwargs["ShaftRotationSpeedUnit"]} by default)')
        if self.ApplicationContext['ShaftRotationSpeedUnit'].lower() == 'rpm':
            self.ApplicationContext['ShaftRotationSpeed'] *= np.pi / 30.
            self.ApplicationContext['ShaftRotationSpeedUnit'] = 'rad/s'

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
