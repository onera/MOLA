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

import pytest

import os
import shutil
import numpy as np

import treelab.cgns as cgns

import mola.naming_conventions as names
from mola.logging import mola_logger, MolaException, MolaUserError, mute_stdout
from mola import server as SV
from mola.cfd.preprocess.run_manager import run_manager
from mola.workflow import Workflow
from mola.misc import run_as_mpi_subprocess

pytestmark = pytest.mark.skip(reason="preprocess MPI not working yet") # FIXME

@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.parametrize("size", [2])
def test_prepare(tmp_path, size):
    test_dir = str(tmp_path)

    def actual_test(test_dir):
        import os
        from mpi4py.MPI import COMM_WORLD as comm
        from mola.workflow import Workflow

        w = Workflow(
        RawMeshComponents=[
            dict(
                Name='sphere',
                Source='/stck/mola/data/open/mesh/sphere/sphere_struct.cgns',
                Families=[
                    dict(Name='Wall', Location='kmin'),
                    dict(Name='Farfield', Location='kmax'),
                ],
                )
        ],

        SplittingAndDistribution=dict(
            Strategy='AtPreprocess', # "AtPreprocess" or "AtComputation"
            Splitter='maia', # or 'maia', 'PyPart' etc..
            Distributor='maia', 
            ComponentsToSplit='all', # 'all', or None or ['first', 'second'...]
            ),

        Flow=dict(
            Density = 0.2,
            Temperature = 100.,
            Velocity = 50.,
                 ),

        Turbulence = dict(
            Model = 'SA',
        ),

        Solver=os.environ.get('MOLA_SOLVER'),

        Numerics = dict(
            NumberOfIterations=2,
            CFL=1.,
        ),

        BoundaryConditions=[
            dict(Family='Wall', Type='Wall'),
            dict(Family='Farfield', Type='Farfield'),
        ],

        Extractions=[
            dict(Type='BC', Source='*', Name='ByFamily', Fields=['Pressure']),
            dict(Type='BC', Source='BCWall*', Name='ByFamily', Fields=['NormalVector', 'SkinFriction', 'BoundaryLayer']),
            dict(Type='IsoSurface', Name='MySurface', IsoSurfaceField='CoordinateZ', IsoSurfaceValue=1.e-6),
            dict(Type='3D', Fields=['Density','MomentumX','MomentumY','MomentumZ'],
                 GridLocation='CellCenter', GhostCells = False),
            ],

        RunManagement=dict(
            NumberOfProcessors=comm.Get_size(), 
            RunDirectory=test_dir,
            ),
        )
        w.RunManagement['Scheduler'] = 'local'
        w.prepare() # TODO WIP currently failing
        # w.write_cfd_files()
        # w.submit(f'cd {test_dir}; bash job.sh')
        # w.assert_completed_without_errors()
        
    run_as_mpi_subprocess(actual_test, size, None, test_dir)

if __name__ == '__main__':
    test_prepare(4)