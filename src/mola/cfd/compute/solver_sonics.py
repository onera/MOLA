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

import mola.naming_conventions as names


def apply_to_solver(workflow):

    import sonics
    from sonics.toolkit.execute.run_graph import get_default_iterators
    from sonics.toolkit.execute.run_graph import get_default_pytriggers

    import maia

    if rank==0:
        os.makedirs(names.DIRECTORY_LOG, exist_ok=True)

    dist_tree = maia.io.file_to_dist_tree(names.FILE_INPUT_SOLVER, comm)

    # default_pytriggers = get_default_pytriggers(workflow.SolverParameters['configuration'], dist_tree, comm)
    # pytriggers += default_pytriggers
    # iterators = get_default_iterators(workflow.SolverParameters['configuration'], pytriggers, comm)

    sonics.solver.run(workflow.SolverParameters['configuration'], dist_tree, comm) #, iterators=iterators)

    maia.algo.pe_to_nface(dist_tree, comm)
    maia.io.dist_tree_to_file(dist_tree, 'solution.cgns', comm)
 

