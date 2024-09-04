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
from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
NumberOfProcessors = comm.Get_size()

from treelab import cgns
import mola.naming_conventions as names
from mola.cfd.compute.read_cfd_files import read_cfd_files

def apply_to_solver(workflow):

    import sonics
    import maia

    # Log mute
    from maia.utils.logging import turn_on, turn_off
    turn_off("tasky")
    turn_off("sonics")
    turn_off("sonics_hpc")
    turn_off("sonics_debug")

    workflow.tree, configuration = read_cfd_files.apply(workflow)

    from mola.cfd.coprocess.manager import CoprocessManager
    coprocess_manager = CoprocessManager(workflow)
    workflow._coprocess_manager = coprocess_manager

    iterators = get_iterators(workflow, configuration)

    dist_tree = workflow.tree.copy()

    sonics.solver.run(configuration, dist_tree, comm, iterators=iterators)
    coprocess_manager.output_tree = cgns.castNode(dist_tree)
    if not coprocess_manager.output_tree.get(Name='NFaceElements'):
        maia.algo.pe_to_nface(dist_tree, comm)
        coprocess_manager.output_tree = cgns.castNode(dist_tree)

    maia.io.dist_tree_to_file(coprocess_manager.output_tree, f'{names.DIRECTORY_OUTPUT}/solution.cgns', comm)

    coprocess_manager.finalize()
    del workflow._coprocess_manager
 
def get_iterators(workflow, configuration):
    from pathlib import Path
    import sonics.toolkit.triggers as triggers
    from sonics.toolkit.iterators import SteadyIterators
    from mola.cfd.preprocess.extractions.solver_sonics import add_extractions_for_families

    transform_miles_config_in_sonics_config(configuration)

    pytriggers = []
    pytriggers.append(triggers.ExecutionTrigger(configuration["conf"], workflow.Numerics['NumberOfIterations']))
    pytriggers.append(triggers.CflTrigger(configuration["conf"], lambda iteration: workflow.Numerics['CFL']))
    pytriggers.append(triggers.ComputeAndExtractDataInGraphTrigger(configuration["conf"],
        add_extractions_for_families(workflow),
        configuration["hpc_conf"]["hardware_target"]))
    
    if any([ext['Type'] == 'Residuals' for ext in workflow.Extractions]):
        # pass  # FIXME for now, there is a bug in ResidualTrigger: 
        #       # KeyError: 'ERROR: cannot access element "IncrementMeanFlow" from namespace "sonics_1807767472669665571_-5341672739926498039"'

        pytriggers.append(triggers.ResidualTrigger(configuration["conf"], workflow.Numerics['NumberOfIterations'],
                                            output_folder=Path(names.DIRECTORY_LOG)))

    # pytriggers.append(triggers.MonitoringIntegralData(  # BUG in SoNICS
    #     configuration['conf'], 
    #     compute_integral_data_on_families(['INFLOW', 'OUTFLOW']), #compute_extracts_from_terms_monitor, 
    #     configuration['niter'], 
    #     configuration['hpc_conf']['hardware_target'], 
    #     period=1)
    # )

    # This Trigger write time at the end of run:
    #    + end computation[<iterations>]: time : (<execution_time>, <execution_time_for_all_ranks>, <time/cell/iteration>)
    execution_trigger = pytriggers[0]
    pytriggers.append(triggers.HookPbSizeTrigger(configuration['conf'], workflow.tree, execution_trigger))

    iterators = SteadyIterators(pytriggers, workflow.Numerics['NumberOfIterations'], comm)

    return iterators

def transform_miles_config_in_sonics_config(configuration):
    import sonics

    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            elif isinstance(v, list):
                d[k].extend(v)
            else:
                d[k] = v
        return d

    def nest_dict_from_string(s : str, leaf=None):
        keys = s.split('/')
        if len(keys) > 2:
            result = {keys[0]: nest_dict_from_string('/'.join(keys[1:]), leaf=leaf)}
        elif len(keys) == 2:
            if leaf is None:
                result = {keys[0]: keys[1]}
            else:
                result = {keys[0]: {keys[1]: leaf}}
        else:
            if leaf is None:
                raise Exception
                result = keys[0]
            else:
                result = {keys[0]: leaf}

        return result

    def convert_in_nest_dict(conf):
        result = dict(sonics=dict())
        subdicts = []
        for key, value in conf.items():
            if value in [True, False, None]:
                subdict = nest_dict_from_string(key)
            else:
                subdict = nest_dict_from_string(key, leaf=value)

            subdicts.append(subdict)
        for subdict in subdicts:
            deep_update(result, subdict)
        return result
    
    configuration['conf'] = sonics.configuration(convert_in_nest_dict(configuration['conf']))


