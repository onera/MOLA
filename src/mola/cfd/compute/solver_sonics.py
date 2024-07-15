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

import mola.naming_conventions as names


def apply_to_solver(workflow):

    import sonics
    from sonics.toolkit.execute.run_graph import get_default_iterators
    from sonics.toolkit.execute.run_graph import get_default_pytriggers

    import maia

    if rank==0:
        os.makedirs(names.DIRECTORY_OUTPUT, exist_ok=True)
        os.makedirs(names.DIRECTORY_LOG, exist_ok=True)

    workflow.tree = maia.io.file_to_dist_tree(names.FILE_INPUT_SOLVER, comm)

    # default_pytriggers = get_default_pytriggers(workflow.SolverParameters['configuration'], workflow.tree, comm)
    # pytriggers += default_pytriggers
    # iterators = get_default_iterators(workflow.SolverParameters['configuration'], pytriggers, comm)

    from mola.cfd.coprocess.manager import CoprocessManager
    coprocess_manager = CoprocessManager(workflow)
    workflow._coprocess_manager = coprocess_manager

    # NOTE Finally, sonics.solver.run will take only dist_tree (the configuration will be read inside the tree)
    workflow.SolverParameters['configuration'] = get_configuration_from_tree(workflow.tree, workflow)
    sonics.solver.run(workflow.SolverParameters['configuration'], workflow.tree, comm) #, iterators=iterators)

    coprocess_manager.finalize()
    del workflow._coprocess_manager
 
def get_configuration_from_tree(tree, workflow):
    import miles
    import copy

    configuration = copy.copy(workflow.SolverParameters['configuration'])

    my_config = miles.solver.config.Configuration(tree, pure_cgns_mode=False)
    my_config.add_template(configuration['conf'])
    my_config.set_numerics(CFL=workflow.Numerics['CFL'])
    
    conf = my_config.apply()
    configuration.update(conf)
    if rank==0:
        # print(my_config.spl_product)
        from pprint import pprint 
        pprint(configuration)
    
    return configuration
