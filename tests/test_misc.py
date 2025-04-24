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

from mola.misc import run_as_mpi_subprocess

_extra_env = {"I_MPI_DEBUG": "5"} # for better MPI debugging diagnostics

@pytest.mark.unit
@pytest.mark.parametrize("size", [1, 2])
def test_mpi(size):

    def actual_test():
        from mpi4py.MPI import COMM_WORLD as comm
        import os
        print(f"[Rank {comm.Get_rank()}] PYTHONPATH = {os.environ.get('MOLA_SOLVER')}")

    run_as_mpi_subprocess(actual_test, size, extra_env=_extra_env)
                                                        

@pytest.mark.unit
@pytest.mark.parametrize("size", [1, 2])
def test_mpi_with_error(size):

    def actual_test():
        from mpi4py.MPI import COMM_WORLD as comm
        raise ValueError("this is designed to raise an error")

    try:
        run_as_mpi_subprocess(actual_test, size, extra_env=_extra_env)
    except AssertionError:
        pass

if __name__ == '__main__':
    test_mpi(4)