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

from mpi4py import MPI
rank = MPI.COMM_WORLD.Get_rank()

import mola.naming_conventions as names
from mola.cfd import apply_to_solver

def apply(workflow):
    apply_to_solver(workflow)

def check_stderr_and_create_COMPLETED():
    check_stderr()
    if rank==0:
        with open(names.FILE_JOB_COMPLETED,'w') as f: 
            f.write(names.FILE_JOB_COMPLETED)
    
def check_stderr():
    # TODO Simple check for now, but it should be different if this function is called in the coprocess script
    if rank==0:
        try:
            with open(names.FILE_STDERR,'r') as f:
                Error = f.read()
            raise Exception(Error)
        except FileNotFoundError:
            pass