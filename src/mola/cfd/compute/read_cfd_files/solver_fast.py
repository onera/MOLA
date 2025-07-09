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
from mola.cfd.preprocess.motion.motion import any_mobile
from mola.cfd.compute.solver_fast import get_theta_and_omega

def apply_to_solver(workflow):

    import Fast.PyTree as Fast
    import FastS.PyTree as FastS
    from mola.cfd.compute.solver_fast import get_range_of_iterations

    inititer, niter = get_range_of_iterations(workflow)

    t, tc, ts, graph = Fast.load(names.FILE_INPUT_SOLVER, 'tc.cgns')

    set_numerics(workflow, t)

    add_convergence_history(t, niter)

    # For debugging & ticket creation
    # import Converter.PyTree as C
    # C.convertPyTree2File(t,'t.cgns')
    # C.convertPyTree2File(tc,'tc.cgns')
    # raise RuntimeError("STOP DUMPED FAST FILES")

    # The warmup function optimizes data storage in memory. 
    # Zones may be moved. After warmup, all operations on the trees must be in-place
    infos_ale = get_infos_ale(workflow)
    t, tc, metrics = FastS.warmup(t, tc, graph, infos_ale=infos_ale)

    workflow.tree = cgns.castNode(t)
    workflow._treeAtCenters = cgns.castNode(tc)
    workflow._Skeleton = cgns.load(names.FILE_INPUT_SOLVER, only_skeleton=True)
    workflow._fast_metrics = metrics
    workflow._fast_graph = graph


def set_numerics(workflow, t):

    import Fast.PyTree as Fast
    import Converter.PyTree as C
    import Converter.Internal as I

    global_parameters_on_bases, local_parameters_on_bases = _split_global_and_local_parameters(workflow.SolverParameters['Num2Base'])
    Fast._setNum2Base(t, global_parameters_on_bases)
    for base_name, base_parameters in local_parameters_on_bases.items():
        base = I.getNodeFromNameAndType(t, base_name, 'CGNSBase')
        Fast._setNum2Base(base, base_parameters)

    global_parameters_on_zones, local_parameters_on_zones = _split_global_and_local_parameters(workflow.SolverParameters['Num2Zones'])
    Fast._setNum2Zones(t, global_parameters_on_zones)
    for family_name, zone_parameters in local_parameters_on_zones.items():
        for zone in C.getFamilyZones(t, family_name):
            Fast._setNum2Zones(zone, zone_parameters)

def _split_global_and_local_parameters(parameters: dict, prefix_local: str = 'Local@') -> tuple:
        global_parameters = dict()
        local_parameters = dict()
        for key, value in parameters.items():
            if key.startswith(prefix_local):
                base_name = key[len(prefix_local):]
                local_parameters[base_name] = value  # value is a dict
            else:
                global_parameters[key] = value
        return global_parameters, local_parameters

def get_infos_ale(workflow):

    if any_mobile(workflow.Motion):
        theta, omega = get_theta_and_omega(workflow, workflow.Numerics['IterationAtInitialState'])
        return [theta, omega]
