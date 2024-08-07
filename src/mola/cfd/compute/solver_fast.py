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

from treelab import cgns
import mola.naming_conventions as names

def apply_to_solver(workflow):

    import FastS.PyTree as FastS

    from mola.cfd.coprocess.manager import CoprocessManager
    workflow._coprocess_manager = CoprocessManager(workflow)

    t, tc, metrics, graph = load_fast_objects(workflow)

    inititer, niter = get_range_of_iterations(workflow)

    # time-marching loop
    for it in range( inititer, inititer+niter ):
    
        FastS._compute(t, metrics, it, tc, graph)

        # FIXME when https://github.com/onera/Fast/issues/13 solved
        # if workflow.SolverParameters['Num2Base']['modulo_verif']%0:
        #     FastS.display_temporal_criteria(t, metrics, it, format='store')

        workflow._coprocess_manager.run_iteration()
                
    workflow._coprocess_manager.finalize()
    del workflow._coprocess_manager
    

def add_convergence_history(t, niter):

    import Converter.Internal as I
    import FastS.PyTree as FastS

    I._rmNodesByName(t, "ZoneConvergenceHistory")
    I._rmNodesByName(t, "GlobalConvergenceHistory")
    FastS.createConvergenceHistory(t, niter)


def set_numerics(workflow, t):

    import Fast.PyTree as Fast

    Fast._setNum2Base( t, workflow.SolverParameters['Num2Base'])
    Fast._setNum2Zones(t, workflow.SolverParameters['Num2Zones'])


def get_range_of_iterations(workflow):
    inititer = workflow.Numerics['IterationAtInitialState']
    niter = workflow.Numerics['NumberOfIterations']

    return inititer, niter

def load_fast_objects(workflow):

    import Fast.PyTree as Fast
    import FastS.PyTree as FastS

    inititer, niter = get_range_of_iterations(workflow)

    t, tc, ts, graph = Fast.load(names.FILE_INPUT_SOLVER, 'tc.cgns',
                                 restart=True if inititer>1 else False)

    set_numerics(workflow, t)

    t, tc, metrics = FastS.warmup(t, tc, graph)
    
    add_convergence_history(t, niter)
    
    t = cgns.castNode(t)
    tc = cgns.castNode(tc)
    
    workflow.tree = t
    workflow._treeAtCenters = tc 
    workflow._metrics = metrics

    return t, tc, metrics, graph