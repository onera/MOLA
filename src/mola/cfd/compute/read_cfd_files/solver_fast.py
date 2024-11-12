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

from treelab import cgns
import mola.naming_conventions as names
from mola.logging import MolaAssertionError
from mola.cfd.preprocess.extractions.solver_fast import add_convergence_history

def apply_to_solver(workflow):

    import Fast.PyTree as Fast
    import FastS.PyTree as FastS
    from mola.cfd.compute.solver_fast import get_range_of_iterations

    inititer, niter = get_range_of_iterations(workflow)

    t, tc, ts, graph = Fast.load(names.FILE_INPUT_SOLVER, 'tc.cgns',
                                 restart=True if inititer>1 else False)

    set_numerics(workflow, t)

    add_convergence_history(t, niter)

    t, tc, metrics = FastS.warmup(t, tc, graph)

    t = cgns.castNode(t)
    tc = cgns.castNode(tc)

    workflow.tree = t
    workflow._treeAtCenters = tc 
    workflow._fast_metrics = metrics
    workflow._fast_graph = graph


def set_numerics(workflow, t):

    import Fast.PyTree as Fast

    Fast._setNum2Base( t, workflow.SolverParameters['Num2Base'])
    Fast._setNum2Zones(t, workflow.SolverParameters['Num2Zones'])
