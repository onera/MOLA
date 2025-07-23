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


class WorkflowAirfoilInterface(WorkflowInterface):

    def __init__(self, workflow, tree=None, **kwargs):
        super().__init__(workflow, tree, **kwargs)
        if tree is None:
            self.add_to_Extractions_BC(Source='BCWall*', Fields=['Pressure', 'BoundaryLayer', 'yPlus'])
            self.add_to_Extractions_Integral(Source='BCWall*', Fields=['Force', 'Torque'])

    def set_ApplicationContext(self, 
        AngleOfAttackDeg : float = 0.0,
        AngleOfSlipDeg : float = 0.0,
        YawAxis : list = [0.0,1.0,0.0],
        PitchAxis : list = [0.0,0.0,-1.0],
        Chord : float = 1.0,
        Surface : float = 1.0):

        self.ApplicationContext = self._get_comp(self.set_ApplicationContext, self.get_default_values_from_local_signature())

    def set_Turbulence(self,
            TransitionZones              :  dict = dict(
                TopOrigin                   = 0.002,
                BottomOrigin                = 0.010,
                TopLaminarImposedUpTo       = 0.001,
                TopLaminarIfFailureUpTo     = 0.2,
                TopTurbulentImposedFrom     = 0.995,
                BottomLaminarImposedUpTo    = 0.001,
                BottomLaminarIfFailureUpTo  = 0.2,
                BottomTurbulentImposedFrom  = 0.995),
            **kwargs):

        super().set_Turbulence(**kwargs)

        trans_mode = self.Turbulence.get("TransitionMode",None)
        if trans_mode is not None and trans_mode in ('LSTT', 'zonal'):
            self.Turbulence['TransitionZones'] = TransitionZones