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

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()

import mola.naming_conventions as names


def apply_to_solver(workflow):
    import maia
    # from sonics.toolkit.execute.run_graph import get_default_iterators
    # from sonics.toolkit.execute.run_graph import get_default_pytriggers

    workflow.tree = maia.io.file_to_dist_tree(names.FILE_INPUT_SOLVER, comm)

    # default_pytriggers = get_default_pytriggers(workflow.SolverParameters['configuration'], workflow.tree, comm)
    # pytriggers += default_pytriggers
    # iterators = get_default_iterators(workflow.SolverParameters['configuration'], pytriggers, comm)

    # NOTE Finally, sonics.solver.run will take only dist_tree (the configuration will be read inside the tree)
    workflow.SolverParameters['configuration'] = get_configuration_from_tree(workflow)

    return workflow.tree, workflow.SolverParameters['configuration']

def get_configuration_from_tree(workflow):
    import miles
    import copy

    configuration = copy.copy(workflow.SolverParameters['configuration'])
    configuration['conf'] = flatten_dict(configuration['conf'])
    param_list = []
    for key, value in configuration['conf'].items():
        if value in [True, False, None]:
            param_list.append(key)
        else:
            param_list.append(f'{key}:{value}')

    my_config = miles.solver.config.Configuration(workflow.tree)
    my_config.update(*param_list)
    my_config.set(CFL=workflow.Numerics['CFL'])
    
    conf = my_config.apply()
    configuration.update(conf)
    if rank==0:
        from pprint import pprint 
        pprint(configuration)
    
    return configuration

def flatten_dict(d, parent_key=''):
    items = {}
    for key, value in d.items():
        new_key = f"{parent_key}/{key}" if parent_key else key
        if isinstance(value, dict):
            items.update(flatten_dict(value, new_key))
        else:
            items[new_key] = value
    return items
