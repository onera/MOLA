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

import numpy as np

import mola.cfd.preprocess.mesh.io as io
from mola.workflow import Workflow

from treelab import cgns

def build_zone():
    # create a grid
    x, y, z = np.meshgrid( np.linspace(0,1,11),
                           np.linspace(0,0.5,7),
                           np.linspace(0,0.3,4), indexing='ij')

    # create a field
    field = x*y

    # create the new zone using numpy arrays of coordinates and field
    zone = cgns.newZoneFromDict( 'block', dict(x=x, y=y, z=z, field=field) )

    return zone


def test_is_using_mpi():
    print(io.utils.is_using_mpi())

def test_file_reader():
    file_src = 'tmp_zone.cgns'
    import mpi4py.MPI as MPI
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        zone = build_zone()
        zone.save(file_src)
    MPI.COMM_WORLD.barrier()
    
    w = Workflow()
    io.reader.read(w,file_src)

    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.Get_rank() == 0:
        os.unlink(file_src)
    MPI.COMM_WORLD.barrier()

def test_file_writer():
    file_src = 'tmp_zone.cgns'
    import mpi4py.MPI as MPI
    
    if MPI.COMM_WORLD.Get_rank() == 0:
        zone = build_zone()
        zone.save(file_src)
    MPI.COMM_WORLD.barrier()
    
    w = Workflow()
    mesh = io.reader.read(w,file_src)

    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.Get_rank() == 0:
        os.unlink(file_src)
    MPI.COMM_WORLD.barrier()

    io.writer.write(w, mesh, file_src)

    MPI.COMM_WORLD.barrier()
    if MPI.COMM_WORLD.Get_rank() == 0:
        os.unlink(file_src)
    MPI.COMM_WORLD.barrier()

if __name__ == '__main__':
    # test_is_using_mpi()
    test_file_writer()