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

import numpy as np
from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
NumberOfProcessors = comm.Get_size()

from treelab import cgns
import mola.naming_conventions as names
from mola.cfd.compute.read_cfd_files import read_cfd_files
from mola.cfd.preprocess.extractions.solver_sonics import add_fields_and_bc_extractions, add_integral_extractions
from mola.cfd.preprocess.cfd_parameters.solver_sonics import get_cfl_function

def apply_to_solver(workflow):

    import sonics
    import maia

    # Log mute
    from maia.utils.logging import turn_on, turn_off
    turn_off("tasky")
    turn_off("sonics")
    turn_off("sonics_hpc")
    turn_off("sonics_debug")

    workflow.tree, config = read_cfd_files.apply(workflow)

    from mola.cfd.coprocess.manager import CoprocessManager
    coprocess_manager = CoprocessManager(workflow)
    workflow._coprocess_manager = coprocess_manager


    dist_tree = workflow.tree.copy()

    hardware_target = 'cpu'

    sonics.solver.run(dist_tree, comm, 
                      iterators = get_iterators(workflow, config, hardware_target), 
                      additional_parameters = dict(
                          output_folder = names.DIRECTORY_LOG,
                          hpc_conf = dict(hardware_target=hardware_target),
                          )
                      )

    coprocess_manager.output_tree = cgns.castNode(dist_tree)
    if not coprocess_manager.output_tree.get(Name='NFaceElements'):
        maia.algo.pe_to_nface(dist_tree, comm)
        coprocess_manager.output_tree = cgns.castNode(dist_tree)

    maia.io.dist_tree_to_file(coprocess_manager.output_tree, f'{names.DIRECTORY_OUTPUT}/solution.cgns', comm)

    coprocess_manager.finalize()
    del workflow._coprocess_manager
 
def get_iterators(workflow, config, hardware_target='cpu'): 
    from pathlib import Path
    import sonics.toolkit.triggers as triggers
    from sonics.toolkit.iterators import SteadyIterators
    
    execution_trigger = triggers.ExecutionTrigger(config, workflow.Numerics['NumberOfIterations'])
    cfl_trigger = triggers.CflTrigger(config, get_cfl_function(workflow.Numerics['CFL']))

    pytriggers = [
        execution_trigger,
        cfl_trigger,
    ]

    if any([ext['Type'] == 'Residuals' for ext in workflow.Extractions]):
        residuals_trigger = triggers.ResidualTrigger(
            config, 
            workflow.Numerics['NumberOfIterations'],
            output_folder=Path(names.DIRECTORY_LOG),
            check_convergence=triggers.convergence_per_subsystem({0:{0:1.e-14}}, normalize=False), 
            )
        pytriggers.append(residuals_trigger)

    if any([ext['Type'] in ['Restart', '3D', 'BC'] for ext in workflow.Extractions]):
        if any([ext['Type'] in ['3D', 'BC'] for ext in workflow.Extractions]):
            periods = [ext['ExtractionPeriod'] for ext in workflow.Extractions if ext['Type'] in ['3D', 'BC']]
        else:
            periods = [workflow.Numerics['NumberOfIterations']]
        
        fields_and_bc_extraction_trigger = triggers.ComputeAndExtractDataInGraphTrigger(
            config,
            add_fields_and_bc_extractions(workflow),
            hardware_target, 
            period=np.gcd.reduce(periods)
            )
        
        pytriggers.append(fields_and_bc_extraction_trigger)

    if any([ext['Type'] == 'Integral' for ext in workflow.Extractions]):
        periods = [ext['ExtractionPeriod'] for ext in workflow.Extractions if ext['Type'] == 'Integral']  # FIXME
        integral_extraction_trigger = triggers.MonitoringIntegralData( 
            config, 
            add_integral_extractions(workflow), 
            workflow.Numerics['NumberOfIterations'], 
            hardware_target, 
            period=10 #np.gcd.reduce(periods)
            ) 
        pytriggers.append(integral_extraction_trigger)

    if any([bc['Type'] == 'OutflowRadialEquilibrium' for bc in workflow.BoundaryConditions]):
        for bc in workflow.BoundaryConditions:
            try:
                valve_type = bc['valve_type']
            except:
                continue

            from mola.cfd.preprocess.boundary_conditions.solver_sonics import get_valve_law_trigger
            valve_law_trigger = get_valve_law_trigger(config, bc, period=10, hardware_target=hardware_target)
            pytriggers.append(valve_law_trigger)

    # This Trigger write time at the end of run:
    #    + end computation[<iterations>]: time : (<execution_time>, <execution_time_for_all_ranks>, <time/cell/iteration>)
    time_record_trigger = triggers.HookPbSizeTrigger(config, workflow.tree, execution_trigger)
    pytriggers.append(time_record_trigger)

    iterators = SteadyIterators(pytriggers, workflow.Numerics['NumberOfIterations'], comm)

    return iterators

