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

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = MPI.COMM_WORLD.Get_rank()

import mola.naming_conventions as names


def apply_to_solver(workflow):
    import maia

    workflow.tree = maia.io.file_to_dist_tree(names.FILE_INPUT_SOLVER, comm)

    # HACK Finally, sonics.solver.run will take only dist_tree (the configuration will be read inside the tree)
    # see https://gitlab.onera.net/numerics/solver/sonics/-/issues/102
    workflow.SolverParameters['configuration'] = get_configuration_from_tree(workflow)

    workflow.tree = cgns.castNode(workflow.tree)

    return workflow.tree, workflow.SolverParameters['configuration']

def get_configuration_from_tree(workflow):
    import miles
    import copy
    import mola.cfd.preprocess.cfd_parameters.solver_sonics as cfd_parameters

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

    _, fluid_parameters = cfd_parameters.get_fluid_template(workflow.Fluid)
    _, turb_parameters = cfd_parameters.get_turbulence_template(workflow.Turbulence)
    _, flux_parameters = cfd_parameters.get_spatial_fluxes_template(workflow.Numerics)
    _, time_parameters = cfd_parameters.get_time_marching_template(workflow.Numerics)
    my_config.set(
        **fluid_parameters,
        **turb_parameters, 
        **flux_parameters, 
        **time_parameters,
    )

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
