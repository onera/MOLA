from mpi4py import MPI
comm = MPI.COMM_WORLD
import miles

import maia
# Log mute
from maia.utils.logging import turn_off
for feature in ["tasky", "sonics", "sonics_hpc", "sonics_debug"]: turn_off(feature)

import sonics
import sonics.toolkit.triggers as triggers
from sonics.toolkit.iterators import SteadyIterators

niter = 5
cfl = 1.0

dist_tree = maia.io.file_to_dist_tree('main.cgns', comm)
config = miles.Configuration()
config.from_cgns_base(dist_tree)


hardware_target = 'cpu'

execution_trigger = triggers.ExecutionTrigger(config, niter, nstep=2)
cfl_trigger = triggers.CflTrigger(config, lambda iteration: cfl)
time_record_trigger = triggers.HookPbSizeTrigger(config, dist_tree, execution_trigger)

pytriggers = [execution_trigger, cfl_trigger, time_record_trigger]

iterators = SteadyIterators(pytriggers, niter=niter, comm=comm)

sonics.run(dist_tree, comm, iterators = iterators, 
           additional_parameters = dict(
               output_folder = "LOGS",
               hpc_conf = dict(hardware_target=hardware_target),
               )
           )

maia.io.dist_tree_to_file(dist_tree, f'solution.cgns', comm)
