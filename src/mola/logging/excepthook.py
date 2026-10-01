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

import sys
from ..mpi import get_mpi_elements

sys_excepthook = sys.excepthook

comm, rank, size, _ = get_mpi_elements()
is_sequential = size == 1

def mpi_excepthook(type, value, traceback):

    err_mssg = f"Your application aborted because of an uncaught exception on rank {rank}:\n\n"

    sys.stderr.write(err_mssg)
    sys_excepthook(type, value, traceback)
    sys.stderr.write('\n')
    sys.stdout.flush()
    sys.stderr.flush()

    comm.Abort(1)

def enable_mpi_excepthook():
    if is_sequential: return
    try:
        sys.excepthook = mpi_excepthook
    except:
        pass
    
def disable_mpi_excepthook():
    sys.excepthook = sys_excepthook
