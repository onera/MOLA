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
from typing import Union
from treelab import cgns
import mola.naming_conventions as names
from mola.logging import MolaAssertionError

from ..workflow import WorkflowRotatingComponent
from .interface import WorkflowTurbomachineryInterface

class WorkflowTurbomachinery(WorkflowRotatingComponent):

    def __init__(self, **kwargs):
        super().__init__(_skip_interface=True) # used to recover the private attributes of WorkflowRotatingComponent
        self._interface = WorkflowTurbomachineryInterface(self, **kwargs)

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

        kwargs.setdefault('RowType', self.ApplicationContext['RowType'])

        signals = cgns.load(input_signals)
        # Read in parallel 
        surfaces = Cmpi.convertFile2SkeletonTree(input_extractions)
        D2._distribute(surfaces, Cmpi.size, useCom=0, algorithm='fast')
        Cmpi._readZones(surfaces, input_extractions, rank=Cmpi.rank)
        Cmpi._convert2PartialTree(surfaces)
        surfaces = cgns.castNode(surfaces)
        Cmpi.barrier()

        kwargs.setdefault('stages', self._get_default_stages_to_postprocess())

        surfaces, signals = postprocess_with_turbo(self, surfaces, signals, **kwargs)
        Cmpi.barrier()
        Cmpi.convertPyTree2File(surfaces, output_extractions)
        if Cmpi.rank == 0: 
            signals.save(output_signals)
        Cmpi.barrier()

    def after_compute(self, logger):

        postprocess_possible = self.tree.get(Name='ChannelHeight') is not None
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
    
    def _get_default_stages_to_postprocess(self):
        """
        Append all rows individually, plus the first and last rows together
        """

        def _append_stage_if_in_out_planes_defined(stages, raw_stage):
            try: 
                stage = (raw_stage, raw_stage) if isinstance(raw_stage, str) else raw_stage
                self._get_planes_names_for_perfo(stage)
                stages.append(raw_stage)
            except:
                pass

        def _search_first_row_by_tag(rows, tag):
            for row in rows:
                for extraction in self.Extractions:
                    if extraction['Type'] == 'IsoSurface':
                        try:
                            if (extraction['OtherOptions']['ReferenceRow'] == row
                                and extraction['OtherOptions']['tag'] == tag):
                                return row
                        except: 
                            pass
            return None

        stages = []
        rows = list(self.ApplicationContext['Rows'])
        for row in rows:
            _append_stage_if_in_out_planes_defined(stages, row)

        first_row = _search_first_row_by_tag(rows, tag='InletPlane')
        last_row  = _search_first_row_by_tag(rows[::-1], tag='OutletPlane')  # starting from the last row

        if first_row is not None and last_row is not None and first_row != last_row:
            stages.append((first_row, last_row))
        
        return stages
    
    def _get_planes_names_for_perfo(self, stage:Union[tuple, str]) -> tuple:
        # Get upstream and downstream rows from stage
        if isinstance(stage, (tuple, list)):
            assert len(stage) == 2
            row1, row2 = stage
        elif isinstance(stage, str):
            row1 = row2 = stage
        else:
            raise MolaAssertionError(f'stage must be either a str or a tuple, and the given type is {type(stage)}')
        
        # Check that rows are compatible with workflow
        for row in (row1, row2):
            if row not in self.ApplicationContext['Rows']:
                raise MolaAssertionError(f'The given row "{row}" is not in ApplicationContext["Rows"]')

        upstream_plane = None
        downstream_plane = None
        for extraction in self.Extractions:
            if extraction['Type'] == 'IsoSurface':
                try:
                    if (extraction['OtherOptions']['ReferenceRow'] == row1
                        and extraction['OtherOptions']['tag'] == 'InletPlane'):
                        upstream_plane = extraction['Name']
                except: 
                    pass
                try:
                    if (extraction['OtherOptions']['ReferenceRow'] == row2
                        and extraction['OtherOptions']['tag'] == 'OutletPlane'):
                        downstream_plane = extraction['Name']
                except: 
                    pass

        if not upstream_plane:
            raise MolaAssertionError(f'Cannot find upstream plane on {row1}')
        if not downstream_plane:
            raise MolaAssertionError(f'Cannot find downstream plane on {row2}')

        return upstream_plane, downstream_plane
    
