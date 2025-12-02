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
import numpy as np

from treelab import cgns
import mola.naming_conventions as names
from mola.logging import mola_logger, MolaException, redirect_streams_to_logger
from mola.math_tools import rotate_3d_vector_from_axis_and_angle_in_degrees
from mola.cfd.preprocess.mesh.tools import parametrize_with_height
from mola.cfd.preprocess.mesh.families import get_bc_family_names_from_patterns
from mola.cfd.preprocess import initialization
import mola.cfd.postprocess as POST
from ... import Workflow
from .interface import WorkflowLinearCascadeInterface

class WorkflowLinearCascade(Workflow):

    def __init__(self, **kwargs):
        self._interface = WorkflowLinearCascadeInterface(self, **kwargs)

    def compute_flow_and_turbulence(self):
        periodic_direction = self.get_periodic_direction()
        if np.isclose(abs(np.dot(periodic_direction, np.array([0,1,0]))), 1):
            periodic_direction = np.array([0,1,0])
            self.ApplicationContext['lin_axis'] = 'XY'
        if np.isclose(abs(np.dot(periodic_direction, np.array([0,0,1]))), 1):
            periodic_direction = np.array([0,0,-1])
            self.ApplicationContext['lin_axis'] = 'XZ'
        else:
            self.ApplicationContext['lin_axis'] = None

        if self.ApplicationContext.get('AngleOfAttackDeg') is not None:
            # Otherwise, Flow['Direction'] will be kept as given by user or default
            flow_direction = self.Flow['Direction'] # assume main axis is X
            self.Flow['Direction'] = rotate_3d_vector_from_axis_and_angle_in_degrees(
                flow_direction, 
                np.cross(flow_direction, periodic_direction),
                self.ApplicationContext['AngleOfAttackDeg'], 
                )

        super().compute_flow_and_turbulence()

    def initialize_flow(self):
        if self.Initialization['Method'] == 'turbo':
            self.parametrize_with_height()
            super().initialize_flow()

        else:
            super().initialize_flow()
            # Do not recompute ChannelHeight if it was in source data
            if not self.tree.get(Name='ChannelHeight', Type='DataArray'):
                self.parametrize_with_height()

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
    
    def parametrize_with_height(self):
        self.Initialization.setdefault('ParametrizeWithHeight', 'maia')  # Force computation of this parameter, allowing postprocess
        # if self.Initialization['ParametrizeWithHeight'] is None \
        #     and any([ext['Type'] == 'IsoSurface' and ext['IsoSurfaceField'] == 'ChannelHeight' for ext in self.Extractions]):
        #     self.Initialization['ParametrizeWithHeight'] = 'maia'

        if self.Initialization['ParametrizeWithHeight'] == 'maia':
            self.parametrize_with_height_with_maia()
        elif self.Initialization['ParametrizeWithHeight'] == 'turbo':
            self.parametrize_with_height_with_turbo(self.ApplicationContext['lin_axis'])
    
    def parametrize_with_height_with_maia(self, hub_families=['hub', 'moyeu'], 
                                shroud_families=['shroud', 'carter'], GridLocation='Vertex'):
        self.tree = parametrize_with_height(
            self.tree, 
            hub_families=get_bc_family_names_from_patterns(self.tree, hub_families), 
            shroud_families=get_bc_family_names_from_patterns(self.tree, shroud_families), 
            GridLocation=GridLocation
            )
        
    def parametrize_with_height_with_turbo(self, lin_axis):
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
            
        '''
        import os
        import Converter.Internal as I
        import turbo.height as TH

        mola_logger.info('Add ChannelHeight in the mesh...')
        OLD_FlowSolutionNodes = I.__FlowSolutionNodes__
        I.__FlowSolutionNodes__ = 'FlowSolution#Height'

        with redirect_streams_to_logger(mola_logger, stdout_level='DEBUG', stderr_level='ERROR'):

            m = TH.generateMaskWithChannelHeightLinear(self.tree, lin_axis=lin_axis)
            mask_filename = os.path.join(self.RunManagement['RunDirectory'], 'mask.cgns')
            TH._computeHeightFromMask(self.tree, m, writeMask=mask_filename, lin_axis=lin_axis)
            os.remove(mask_filename) # remove this file for now, but it will be maybe necessary for other operations later
        
        I.__FlowSolutionNodes__ = OLD_FlowSolutionNodes

        self.tree = cgns.castNode(self.tree)

    def postprocess(
        self, 
        input_signals=os.path.join(names.DIRECTORY_OUTPUT, names.FILE_OUTPUT_1D),
        input_extractions=os.path.join(names.DIRECTORY_OUTPUT, names.FILE_OUTPUT_2D),
        output_signals=None, 
        output_extractions=None,
        **kwargs
        ):
        '''
        kwargs are parameters for postprocess_turbomachinery
        '''     
        import Converter.Mpi as Cmpi
        import Distributor2.PyTree as D2
        from mola.cfd.postprocess.tool_interface.turbo import postprocess_with_turbo
  
        if output_signals is None:
            output_signals = input_signals
        if output_extractions is None:
            output_extractions = input_extractions


        kwargs['config'] = 'linear' 
        kwargs['lin_axis'] = self.ApplicationContext['lin_axis']
        # ApplicationContext['Rows'] is mandatory for postprocess
        self.ApplicationContext['Rows'] = []
        for ext in self.Extractions:
            try:
                row = ext['OtherOptions']['ReferenceRow']
            except KeyError:
                continue
            else:
                self.ApplicationContext['Rows'].append(row)
                break

        signals = cgns.load(input_signals)
        # Read in parallel 
        surfaces = Cmpi.convertFile2SkeletonTree(input_extractions)
        D2._distribute(surfaces, Cmpi.size, useCom=0, algorithm='fast')
        Cmpi._readZones(surfaces, input_extractions, rank=Cmpi.rank)
        Cmpi._convert2PartialTree(surfaces)
        surfaces = cgns.castNode(surfaces)
        Cmpi.barrier()

        surfaces, signals = postprocess_with_turbo(self, surfaces, signals, **kwargs)
        Cmpi.barrier()
        Cmpi.convertPyTree2File(surfaces, output_extractions)
        if Cmpi.rank == 0: 
            signals.save(output_signals)
        Cmpi.barrier()

    def after_compute(self, logger):
        from mpi4py import MPI
        rank = MPI.COMM_WORLD.Get_rank()

        postprocess_possible = self.tree.get(Name='ChannelHeight') is not None \
            and isinstance(self.ApplicationContext['lin_axis'], str)
        if not postprocess_possible:
            return
        
        # check required variables are present
        for extraction in self.Extractions:
            if extraction['Type'] == 'IsoSurface':
                if not 'Fields' in extraction or \
                    not all([v in extraction['Fields'] for v in self.Flow['Conservatives']]):
                    logger.warning(
                        ('postprocess is available only if all conservative quantities '
                         'were extracted on each isosurface.'), 
                         rank=0)
                    return
        
        if self.Solver.lower() != 'elsa':
            logger.warning(f'For now, postprocess is available only with elsa solver.', rank=0)
            return

        logger.info('try to postprocess...', rank=0)
        try:
            self.postprocess()
        except Exception as err:
            logger.error(f'  > postprocess failed', rank=0)

            # TODO remove this redirect because it is not working since it does
            # not show the error on stderr.log file. Just let Python fail as usual
            # if rank == 0:
            #     # Add error message to file stderr.log 
            #     with open(names.FILE_STDERR, 'a') as f:
            #         f.write(str(err)+'\n')
            #     # Write file FAILED
            #     with open(names.FILE_JOB_FAILED, 'w') as f: 
            #         f.write(names.FILE_JOB_FAILED)
            # MPI.COMM_WORLD.Abort(1)

            raise ValueError('turbomachinery postprocess failed, see full traceback') from err

        else:
            logger.info(f'  > postprocess done.', rank=0)

        try:
            self.plot_radial_profiles()
        except Exception as err:
            logger.warning(f'Cannot plot radial profiles', rank=0)
