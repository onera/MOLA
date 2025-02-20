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
from treelab import cgns
import mola.naming_conventions as names
from mola.logging import MolaAssertionError

from ..workflow import WorkflowRotatingComponent
from .interface import WorkflowTurbomachineryInterface

class WorkflowTurbomachinery(WorkflowRotatingComponent):

    def __init__(self, **kwargs):
        self._interface = WorkflowTurbomachineryInterface(self, **kwargs)

    def postprocess(self, **kwargs):     
        import Converter.Mpi as Cmpi
        import Distributor2.PyTree as D2
        from mola.legacy.interface_turbo import postprocess_turbomachinery
  
        input_signals = kwargs.get('input_signals', os.path.join(names.DIRECTORY_OUTPUT, names.FILE_OUTPUT_1D))
        input_extractions = kwargs.get('input_extractions', os.path.join(names.DIRECTORY_OUTPUT, names.FILE_OUTPUT_2D))
        output_signals = kwargs.get('output_signals', input_signals)
        output_extractions = kwargs.get('output_extractions', input_extractions)
        stages = kwargs.get('stages', [])

        signals = cgns.load(input_signals)
        # Read in parallel 
        surfaces = Cmpi.convertFile2SkeletonTree(input_extractions)
        surfaces, stats = D2.distribute(surfaces, Cmpi.size, useCom=0, algorithm='fast')
        surfaces = Cmpi.readZones(surfaces, input_extractions, rank=Cmpi.rank)
        Cmpi._convert2PartialTree(surfaces)
        surfaces = cgns.castNode(surfaces)
        Cmpi.barrier()

        surfaces, signals = postprocess_turbomachinery(self, surfaces, signals, stages=stages)
        Cmpi.barrier()
        Cmpi.convertPyTree2File(surfaces, output_extractions)
        if Cmpi.rank == 0: 
            signals.save(output_signals)
        Cmpi.barrier()

    def after_compute(self):
        if self.Solver.lower() != 'elsa':
            raise MolaAssertionError('For now, postprocess is available only with elsa solver.')
        self.postprocess()
        self.plot_radial_profiles()
