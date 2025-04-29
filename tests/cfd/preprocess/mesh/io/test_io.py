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

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_is_using_mpi():
    print(io.utils.is_using_mpi())

@pytest.mark.unit
@pytest.mark.cost_level_0
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

@pytest.mark.unit
@pytest.mark.cost_level_0
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

@pytest.fixture
def cube():
    x, y, z = np.meshgrid( np.linspace(0,1,5),
                           np.linspace(0,1,5),
                           np.linspace(0,1,5), indexing='ij')
    zone = cgns.newZoneFromArrays( 'cube', ['x','y','z'], [ x,  y,  z ])
    return zone

@pytest.fixture
def monolayer():
    x, y, z = np.meshgrid( np.linspace(0,1,5),
                           np.linspace(0,1,5),
                           np.linspace(0,1,2), indexing='ij')
    zone = cgns.newZoneFromArrays( 'monolayer', ['x','y','z'], [ x,  y,  z ])
    return zone

@pytest.fixture
def surface():
    x, y = np.meshgrid( np.linspace(0,1,5),
                           np.linspace(0,1,5), indexing='ij')
    zone = cgns.newZoneFromArrays( 'surface', ['x','y','z'], [ x,  y,  y*0 ])
    return zone

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_is_cell_multilayer(cube, monolayer, surface):
    assert io.is_cell_multilayer(cube)
    assert not io.is_cell_multilayer(monolayer)
    assert not io.is_cell_multilayer(surface)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_is_cell_monolayer(cube, monolayer, surface):
    assert not io.is_cell_monolayer(cube)
    assert io.is_cell_monolayer(monolayer)
    assert not io.is_cell_monolayer(surface)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_is_surface(cube, monolayer, surface):
    assert not io.is_surface(cube)
    assert not io.is_surface(monolayer)
    assert io.is_surface(surface)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_homogeneous_dimension():

    class FakeWorkflow():
        def __init__(self):                          
            self.ProblemDimension = None

    w = FakeWorkflow()
    
    io.set_homogeneous_dimension(w, [1,0,0])
    assert w.ProblemDimension == 3

    io.set_homogeneous_dimension(w, [0,1,0])
    assert w.ProblemDimension == 2

    io.set_homogeneous_dimension(w, [0,0,1])
    assert w.ProblemDimension == 2

    try:
        io.set_homogeneous_dimension(w, [0,2,1])
    except ValueError as e:
        msg = str(e)
        if not msg.startswith("grid dimensions are not homogeneous"):
            raise e


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_homogeneous_dimension_cube(cube):

    class FakeWorkflow():
        def __init__(self):                          
            self.tree = cgns.Tree(base=cube)

    w = FakeWorkflow()
    
    io.set_problem_dimension_based_on_grid(w)
    assert w.ProblemDimension == 3

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_homogeneous_dimension_monolayer(monolayer):

    class FakeWorkflow():
        def __init__(self):                          
            self.tree = cgns.Tree(base=monolayer)

    w = FakeWorkflow()
    
    io.set_problem_dimension_based_on_grid(w)
    assert w.ProblemDimension == 2


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_homogeneous_dimension_surface(surface):

    class FakeWorkflow():
        def __init__(self):                          
            self.tree = cgns.Tree(base=surface)

    w = FakeWorkflow()
    
    io.set_problem_dimension_based_on_grid(w)
    assert w.ProblemDimension == 2

if __name__ == '__main__':
    test_is_using_mpi()
