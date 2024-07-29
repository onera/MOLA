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
pytestmark = pytest.mark.fast

import numpy as np

from treelab import cgns
from mola.cfd.preprocess.cfd_parameters import solver_fast


class FakeWorkflowMonoBlock():

    def __init__(self, NPts):
        self.tree = cgns.Tree()
        base = cgns.Base(Parent=self.tree)
        xyz = np.meshgrid( np.linspace(0,1,NPts),
                           np.linspace(0,1,NPts),
                           np.linspace(0,1,NPts), indexing='ij')
        mesh = cgns.newZoneFromArrays( 'block', ['x','y','z'], xyz)
        mesh.attachTo(base)
        
        self.SolverParameters = {}
        self.ProblemDimension = 3


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_spatial_fluxes_jameson():
    Numerics = dict(Scheme='Jameson')
    SchemeSetup = solver_fast.get_spatial_fluxes(Numerics)
    assert Numerics['Scheme'] == "Roe"
    assert SchemeSetup['scheme'] == "roe_min"
    assert SchemeSetup['psiroe']


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_spatial_fluxes_roe():
    Numerics = dict(Scheme='Roe')
    SchemeSetup = solver_fast.get_spatial_fluxes(Numerics)
    assert Numerics['Scheme'] == "Roe"
    assert SchemeSetup['scheme'] == "roe_min"
    assert SchemeSetup['psiroe']


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_spatial_fluxes_ausm():
    Numerics = dict(Scheme='ausm+')
    SchemeSetup = solver_fast.get_spatial_fluxes(Numerics)
    assert SchemeSetup['scheme'] == "ausmpred"


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_time_marching_setup_steady():
    Numerics = dict(TimeMarching='Steady', CFL=1)
    SchemeSetup = solver_fast.get_time_marching_setup(Numerics)
    assert SchemeSetup["temporal_scheme"] == "implicit"


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_time_marching_setup_unsteady():
    Numerics = dict(TimeMarching='UnsteadyFirstOrder', TimeStep=0.1)
    SchemeSetup = solver_fast.get_time_marching_setup(Numerics)
    assert SchemeSetup["time_step"] == Numerics['TimeStep']


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_cfl_setup_float():
    cfl = solver_fast.get_cfl_setup(1.0)
    assert cfl['cfl'] == 1.0


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_cfl_setup_dict():
    cfl = solver_fast.get_cfl_setup(dict(EndValue=1.0))
    assert cfl['cfl'] == 1.0


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_put_numerics_in_tree():
    workflow = FakeWorkflowMonoBlock(5)
    fast_num = dict(
        temporal_scheme = "implicit_local",
        ss_iteration=5,
        ssdom_IJK=[10000,10000,10000],
        epsi_newton=0.01, 
        nb_relax=1, 
        modulo_verif=10,
        invalidkey="ThisWillNotBeStoredInTree",
        time_step = 0.1,
        time_step_nature = "local",
        cfl = 1.0
    )
    tree = workflow.tree
    solver_fast.put_numerics_in_tree(fast_num, workflow.tree)

    base = tree.bases()[0]
    zone = base.zones()[0]

    assert tree.get(".Solver#define",Depth=1)
    assert base.get(".Solver#define",Depth=1)
    assert zone.get(".Solver#define",Depth=1)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_numerics():
    workflow = FakeWorkflowMonoBlock(5)
    workflow.Numerics = dict( TimeMarching = "Steady", Scheme='ausm+', CFL=1 )
    solver_fast.set_numerics(workflow)

    tree = workflow.tree
    base = tree.bases()[0]
    zone = base.zones()[0]

    assert tree.get(".Solver#define",Depth=1)
    assert base.get(".Solver#define",Depth=1)
    assert zone.get(".Solver#define",Depth=1)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_apply_to_solver():
    workflow = FakeWorkflowMonoBlock(5)
    workflow.Numerics = dict( TimeMarching = "Steady", Scheme='ausm+', CFL=1 )
    solver_fast.apply_to_solver(workflow)

    tree = workflow.tree
    base = tree.bases()[0]
    zone = base.zones()[0]

    assert tree.get(".Solver#define",Depth=1)
    assert base.get(".Solver#define",Depth=1)
    assert zone.get(".Solver#define",Depth=1)
