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
from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
NumberOfProcessors = comm.Get_size()

from mola.cfd.compute.read_cfd_files import read_cfd_files

def apply_to_solver(workflow):

    import sonics

    workflow.tree, config = read_cfd_files.apply(workflow)

    from mola.cfd.coprocess.manager import CoprocessManager
    coprocess_manager = CoprocessManager(workflow)
    workflow._coprocess_manager = coprocess_manager

    sonics.solver.run(config, workflow.tree, comm) #, iterators=iterators)

    coprocess_manager.finalize()
    del workflow._coprocess_manager
 
