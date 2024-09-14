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

from treelab import cgns
from mola.logging import mola_logger, MolaException, redirect_streams_to_logger

from . import Workflow
from .linear_cascade_interface import WorkflowLinearCascadeInterface


class WorkflowLinearCascade(Workflow):

    def __init__(self, **kwargs):
        self._interface = WorkflowLinearCascadeInterface(self, **kwargs)

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
    
    def parametrize_with_height(self, lin_axis, method=2):
        '''
        Compute the variable *ChannelHeight* from a mesh PyTree **t**. This function
        relies on the turbo module.

        .. important::

            Dependency to *turbo* module. See file:///stck/jmarty/TOOLS/turbo/doc/html/index.html

        Parameters
        ----------

            lin_axis : str
                Axis for linear configuration.
                'XY' means that X-axis is the streamwise direction and Y-axis is the
                spanwise direction.(see turbo documentation)
            
            method : int
                Method used for ``turbo.height.generateHLinesAxial()``. Default value is 2.

        '''
        import os
        import Converter.Internal as I
        import turbo.height as TH

        def plot_hub_and_shroud_lines(t):
            # Get geometry
            hub     = I.getNodeFromName(t, 'Hub')
            xHub    = I.getValue(I.getNodeFromName(hub, 'CoordinateX'))
            yHub    = I.getValue(I.getNodeFromName(hub, 'CoordinateY'))
            shroud  = I.getNodeFromName(t, 'Shroud')
            xShroud = I.getValue(I.getNodeFromName(shroud, 'CoordinateX'))
            yShroud = I.getValue(I.getNodeFromName(shroud, 'CoordinateY'))
            # Import matplotlib
            import matplotlib.pyplot as plt
            # Plot
            plt.figure()
            plt.plot(xHub, yHub, '-', label='Hub')
            plt.plot(xShroud, yShroud, '-', label='Shroud')
            plt.axis('equal')
            plt.grid()
            plt.xlabel('x (m)')
            plt.ylabel('y (m)')
            # Save
            plt.savefig('shroud_hub_lines.png', dpi=150, bbox_inches='tight')
            return 0

        mola_logger.info('Add ChannelHeight in the mesh...')
        OLD_FlowSolutionNodes = I.__FlowSolutionNodes__
        I.__FlowSolutionNodes__ = 'FlowSolution#Height'

        with redirect_streams_to_logger(mola_logger, stdout_level='DEBUG', stderr_level='ERROR'):

            m = TH.generateMaskWithChannelHeightLinear(self.tree, lin_axis=lin_axis)
            TH._computeHeightFromMask(self.tree, m, writeMask='mask.cgns', lin_axis=lin_axis)
        
        I.__FlowSolutionNodes__ = OLD_FlowSolutionNodes

        self.tree = cgns.castNode(self.tree)
