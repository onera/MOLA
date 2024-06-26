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

from mola.logging import (mola_logger, MolaException)

from . import Workflow
from .linear_cascade_interface import WorkflowLinearCascadeInterface


class WorkflowLinearCascade(Workflow):

    def __init__(self, tree=None, **kwargs):
        
        self.Name = self.__class__.__name__
        self.tree = tree
        self._interface = WorkflowLinearCascadeInterface(workflow=self, **kwargs)
        if tree is not None:
            self.get_workflow_parameters_from_tree()
        else:
            self._interface.add_to_Extractions_BC(Source='BCWall*', Fields=['Pressure', 'BoundaryLayer', 'yPlus'])
            self._interface.add_to_Extractions_BC(Source='BCInflow*', Fields=['MassFlow'])
            self._interface.add_to_Extractions_BC(Source='BCOutflow*', Fields=['MassFlow'])

    def get_periodic_direction(self):
        # Get periodic match connections
        perio_connections = [connec for connec in self.RawMeshComponents['Connection'] if connec['Type'] == 'PeriodicMatch']
        if len(perio_connections) == 1:
            periodic_direction = np.array(perio_connections[0]['Translation'])
            periodic_direction /= np.sqrt(np.sum(periodic_direction**2))
        elif len(perio_connections) == 0:
            # Check that Periodicity already given in the mesh and adapt it if necessary
            # For now raise an exception
            raise MolaException('Not yet implemented')
        else:
            raise MolaException('More than one PeriodicMatch')
        
        return periodic_direction

