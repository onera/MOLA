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

# ----------------------- IMPORT SYSTEM MODULES ----------------------- #
import os
from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
NumberOfProcessors = comm.Get_size()

import glob
import shutil

import mola.naming_conventions as names


def apply_to_solver(workflow):

    if rank==0:
        os.makedirs(names.DIRECTORY_LOG, exist_ok=True)

    launch_sonics_computation(workflow)
    moveLogFiles()


def launch_sonics_computation(workflow):

    import sonics
    import maia

    dist_tree = maia.io.file_to_dist_tree(names.FILE_INPUT_SOLVER, comm)

    sonics.solver.run(workflow.SolverParameters['configuration'], dist_tree, comm)

    maia.algo.pe_to_nface(dist_tree, comm)
    maia.io.dist_tree_to_file(dist_tree, f'solution.cgns', comm)
 
def moveLogFiles():
    if rank == 0:
        try: os.makedirs(names.DIRECTORY_LOG)
        except: pass

        for fn in glob.glob('*.log'):
            FilenameBase = fn[:-4]
            i = 1
            NewFilename = FilenameBase+'-%d'%i+'.log'
            while os.path.isfile(os.path.join(names.DIRECTORY_LOG, NewFilename)):
                i += 1
                NewFilename = FilenameBase+'-%d'%i+'.log'

            shutil.move(fn, os.path.join(names.DIRECTORY_LOG, NewFilename))

    comm.barrier()
