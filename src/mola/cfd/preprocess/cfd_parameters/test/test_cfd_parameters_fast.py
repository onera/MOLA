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
    assert SchemeSetup['Num2Zones']['scheme'] == "roe_min"
    assert SchemeSetup['Num2Zones']['psiroe']


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_spatial_fluxes_roe():
    Numerics = dict(Scheme='Roe')
    SchemeSetup = solver_fast.get_spatial_fluxes(Numerics)
    assert Numerics['Scheme'] == "Roe"
    assert SchemeSetup['Num2Zones']['scheme'] == "roe_min"
    assert SchemeSetup['Num2Zones']['psiroe']


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_spatial_fluxes_ausm():
    Numerics = dict(Scheme='ausm+')
    SchemeSetup = solver_fast.get_spatial_fluxes(Numerics)
    assert SchemeSetup['Num2Zones']['scheme'] == "ausmpred"


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_time_marching_setup_steady():
    Numerics = dict(TimeMarching='Steady', CFL=1)
    SchemeSetup = solver_fast.get_time_marching_setup(Numerics)
    assert SchemeSetup['Num2Base']["temporal_scheme"] == "implicit"


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_time_marching_setup_unsteady():
    Numerics = dict(TimeMarching='UnsteadyFirstOrder', TimeStep=0.1)
    SchemeSetup = solver_fast.get_time_marching_setup(Numerics)
    assert SchemeSetup['Num2Zones']["time_step"] == Numerics['TimeStep']


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
def test_set_numerics():
    workflow = FakeWorkflowMonoBlock(5)
    workflow.Numerics = dict( TimeMarching = "Steady", Scheme='ausm+', CFL=1 )
    solver_fast.set_numerics(workflow)

    assert workflow.SolverParameters['Num2Base']["temporal_scheme"] == "implicit"
    assert workflow.SolverParameters['Num2Zones']["scheme"] == "ausmpred"
    assert workflow.SolverParameters['Num2Zones']["cfl"] == 1


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_apply_to_solver():
    workflow = FakeWorkflowMonoBlock(5)
    workflow.Numerics = dict( TimeMarching = "Steady", Scheme='ausm+', CFL=1 )
    solver_fast.apply_to_solver(workflow)

    assert workflow.SolverParameters['Num2Base']["temporal_scheme"] == "implicit"
    assert workflow.SolverParameters['Num2Zones']["scheme"] == "ausmpred"
    assert workflow.SolverParameters['Num2Zones']["cfl"] == 1
