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
from mola.math_tools import rotate_3d_vector_from_axis_and_angle_in_degrees
import mola.cfd.postprocess as POST
from . import Workflow
from .linear_cascade_interface import WorkflowLinearCascadeInterface

class WorkflowLinearCascade(Workflow):

    def __init__(self, **kwargs):
        self._interface = WorkflowLinearCascadeInterface(self, **kwargs)

    def compute_flow_and_turbulence(self):
        alpha = self.ApplicationContext.get('AngleOfAttackDeg')
        if alpha is not None:
            # Otherwise, Flow['Direction'] will be kept as given by user or default
            flow_direction = self.Flow['Direction'] # assume main axis is X
            periodic_direction = self.get_periodic_direction()
            if np.isclose(abs(np.dot(periodic_direction, np.array([0,1,0]))), 1):
                periodic_direction = np.array([0,1,0])
                self.lin_axis = 'XY'
            if np.isclose(abs(np.dot(periodic_direction, np.array([0,0,1]))), 1):
                periodic_direction = np.array([0,0,-1])
                self.lin_axis = 'XZ'
            else:
                self.lin_axis = None

            self.Flow['Direction'] = rotate_3d_vector_from_axis_and_angle_in_degrees(
                flow_direction, 
                np.cross(flow_direction, periodic_direction),
                self.ApplicationContext['AngleOfAttackDeg'], 
                )

        super().compute_flow_and_turbulence()

    def get_periodic_direction(self):
        periodic_node = self.tree.get(Type='Periodic')  # Periodic node in a GridConnectivity
        translation = periodic_node.get(Name='Translation').value()
        periodic_direction = translation / np.sqrt(np.sum(translation**2))

        # blade_family_node = self.tree.get(Type='Family', Name='*BLADE*')
        # if blade_family_node:
        #     blade_family = blade_family_node.name()
        #     try: 
        #         blade = POST.extract_bc(self.tree, Family=blade_family, tool='maia')
        #     except:
        #         blade = POST.extract_bc(self.tree, Family=blade_family, tool='cassiopee')

        #     x, y, z = np.array([]), np.array([]), np.array([])
        #     for zone in blade.zones():
        #         xi, yi, zi = zone.xyz(ravel=True)
        #         x = np.concatenate((x, xi))
        #         y = np.concatenate((y, yi))
        #         z = np.concatenate((z, zi))
        #     imin = np.argmin(x)
        #     imax = np.argmax(x)
        #     point_on_LE = np.array([x[imin], y[imin], z[imin]])
        #     point_on_TE = np.array([x[imax], y[imax], z[imax]])
        #     chord_vector = point_on_TE - point_on_LE

        #     # periodic_direction must points in the opposite direction the chord_vector, 
        #     # to be oriented from pressure side to suction side
        #     if np.dot(chord_vector, periodic_direction) > 0:
        #         periodic_direction *= -1
            
        # else:
        #     mola_logger.warning(f'Cannot extract blade family')
            
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
