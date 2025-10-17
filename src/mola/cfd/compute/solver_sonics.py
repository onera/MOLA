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

import os
from packaging.version import Version
import numpy as np
from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
NumberOfProcessors = comm.Get_size()

from treelab import cgns
import mola.naming_conventions as names
from mola.cfd.compute.read_cfd_files import read_cfd_files
from mola.cfd.preprocess.extractions.solver_sonics import add_fields_and_bc_extractions, get_familiesBC_nodes, get_bc_families_names_to_extract
from mola.cfd.preprocess.cfd_parameters.solver_sonics import get_cfl_function
from mola.cfd.preprocess.solver_specific_tools.solver_sonics import translate_extraction_variables_to_sonics, translate_extraction_variables_to_sonics_function

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

    iterators = get_iterators(workflow, config, hardware_target)
    workflow._iterators = iterators

    sonics.run(dist_tree, comm, 
                iterators = iterators, 
                additional_parameters = dict(
                    output_folder = names.DIRECTORY_LOG,
                    hpc_conf = dict(hardware_target=hardware_target),
                    )
                )

    # Create NFaceElements for visualization
    coprocess_manager.output_tree = cgns.castNode(dist_tree)
    if not coprocess_manager.output_tree.get(Name='NFaceElements'):
        maia.algo.pe_to_nface(dist_tree, comm)
        coprocess_manager.output_tree = cgns.castNode(dist_tree)

    maia.io.dist_tree_to_file(coprocess_manager.output_tree, f'{names.DIRECTORY_OUTPUT}/solution.cgns', comm)

    coprocess_manager.finalize()
    del workflow._coprocess_manager
 
def get_iterators(workflow, config, hardware_target='cpu'): 
    import miles
    import sonics.toolkit.triggers as triggers
    from sonics.toolkit.iterators import SteadyIterators
    
    execution_trigger = triggers.ExecutionTrigger(config, workflow.Numerics['NumberOfIterations'], nstep=2)

    # CFL
    # cfl_trigger = triggers.CflTrigger(config, get_cfl_function(workflow.Numerics['CFL']))
    sched = miles.CFLScheduler(config, cfl=get_cfl_function(workflow.Numerics['CFL']))
    cfl_trigger = sched.apply()[0]

    pytriggers = [
        execution_trigger,
        cfl_trigger,
    ]

    if any([ext['Type'] == 'Residuals' for ext in workflow.Extractions]):    
        ext = miles.ResidualExtractor(
            config, 
            period=1, 
            # start_iter=workflow.Numerics['IterationAtInitialState'], 
            output_folder=names.DIRECTORY_LOG
            )
        # ext.add_matplotlib_callback(names.DIRECTORY_LOG+"/residuals_at_{it}.png",start_iter=1,period=1,
        #     separate_systems=True,legend=True,grid={"ls":":"},
        #     yscale="log",xlabel="Iterations",ylabel="Residual")
        residuals_trigger = ext.apply(niter=workflow.Numerics['NumberOfIterations'])

        pytriggers.extend(residuals_trigger)

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
        pytriggers += get_integral_triggers(workflow, config)

    if any([bc['Type'] == 'OutflowRadialEquilibrium' for bc in workflow.BoundaryConditions]):
        for bc in workflow.BoundaryConditions:
            if not is_a_bc_with_valve_law(bc):
                continue

            # TODO handle the fact that OUTFLOW family can be extracted twice: 
            # once with the default extraction of MassFlow, and once with the 
            # valve law trigger 

            from mola.cfd.preprocess.boundary_conditions.solver_sonics import get_valve_law_trigger
            valve_law_trigger = get_valve_law_trigger(
                workflow, 
                config, 
                bc, 
                hardware_target=hardware_target
                )
            pytriggers.append(valve_law_trigger)

    # This Trigger write time at the end of run:
    #    + end computation[<iterations>]: time : (<execution_time>, <execution_time_for_all_ranks>, <time/cell/iteration>)
    time_record_trigger = triggers.HookPbSizeTrigger(config, workflow.tree, execution_trigger)
    pytriggers.append(time_record_trigger)

    iterators = SteadyIterators(
        pytriggers, 
        niter=workflow.Numerics['NumberOfIterations'], 
        comm=comm,
        # initial=workflow.Numerics['IterationAtInitialState']
        )

    return iterators

def is_a_bc_with_valve_law(bc):
    if 'ValveLaw' in bc or 'MassFlow' in bc:
        return True
    else:
        False

def get_integral_triggers(workflow, config):
    from miles.trigger import IntegralDataExtractor
    import sonics.toolkit.triggers as triggers

    pytriggers = []

    # HACK for sonics >= 0.5.35
    # Different triggers must be defined for each family
    # see https://numerics.gitlab-pages.onera.net/coupling/miles/v0.0.4dev/known_issues/index.html#extracting-both-convective-diffusive-fluxes-in-the-same-trigger-deadlocks

    familiesBC = get_familiesBC_nodes(workflow.tree)
    for extraction in workflow.Extractions: 
        if extraction['Type'] != 'Integral':
            continue

        families = get_bc_families_names_to_extract(workflow, extraction, familiesBC)

        def get_extract_funtion_for_trigger(family, fields):

            def compute_extracts_from_terms_monitor_conv(conf, solver, topology):
                from sonics.toolkit.graph_utils import DataFactory
                from sonics.spl import guards

                extracts = []
                treg = solver.terms
                df = DataFactory(solver, topology)

                # THIS LINE IS MANDATORY, else NaN or deadlock
                elt_location = treg.cell if guards.cell_center in conf else treg.vertex
                extracts += df.create_zones(treg.dummy(treg.conservatives(treg.full)), elt_location)
                extracts += df.create_zones(treg.dummy(treg.grad(treg.primitives(treg.mean_flow))), elt_location)

                elt_location = treg.face if guards.cell_center in conf else treg.dual_facet
                sonics_fields = translate_extraction_variables_to_sonics(fields, solver)
                for field in sonics_fields:
                    extracts += df.create_families(
                        field, 
                        elt_location, 
                        family_type=treg.family_value, 
                        predicate=lambda n,v : v['name'] == family 
                    )
                return extracts
            
            return compute_extracts_from_terms_monitor_conv

        for family in families:

            integral_extraction_trigger = triggers.MonitoringIntegralData(
                config,
                get_extract_funtion_for_trigger(family, extraction['Fields']),
                niter=workflow.Numerics['NumberOfIterations'],
                )

            pytriggers.append(integral_extraction_trigger)

            # def _generate_setup_closure(self):
            #     import fnmatch
            #     from sonics.toolkit.graph_utils import DataFactory
            #     from sonics.spl import guards
            #     if not len(self._extractions):
            #         raise ParsingError(f"No extractions provided to IntegralDataExtractor.")
            #     def compute_extracts_from_terms_monitor(conf,solver,topology):
            #         treg = solver.terms
            #         df = DataFactory(solver, topology)
            #         elt_location = treg.cell if guards.cell_center in conf else treg.vertex
            #         extracts = []

            #         # THESE LINES ARE MANDATORY, else NaN or deadlock
            #         print("I have hacked IntegralDataExtractor!")
            #         elt_location_zone = treg.cell if guards.cell_center in conf else treg.vertex
            #         extracts += df.create_zones(treg.dummy(treg.conservatives(treg.full)), elt_location_zone)
                    
            #         for treg_lambda, family_patterns in self._extractions:
            #             terms = treg_lambda(treg)
            #             # TODO: checking typing better
            #             if isinstance(terms,(list,tuple,set)): pass
            #             else: terms = [terms]

            #             pred = lambda n,v : any(fnmatch.fnmatch(v['name'],pat) for pat in family_patterns)

            #             for term in terms:
            #                 extracts.extend(df.create_families(term,elt_location,predicate=pred))
            #         return extracts
            #     return compute_extracts_from_terms_monitor

            # IntegralDataExtractor._generate_setup_closure = _generate_setup_closure
            
            # extractor = IntegralDataExtractor(
            #     config, 
            #     # FIXME cgns_node_pattern does nothing for now (sonics 0.6.2)
            #     # cgns_node_pattern="{family_name}:"+extraction['Name'],  # {family_name} is mandatory in name, FIXME in miles 
            #     period=extraction['ExtractionPeriod'],
            #     # start_iter=workflow.Numerics['IterationAtInitialState']
            #     )
            # extractor.add_extraction(
            #     translate_extraction_variables_to_sonics_function(extraction['Fields']), 
            #     family=family
            #     )
            # # pattern_png = "{output_folder}/fig_{it}.png"
            # # pattern_csv = "{output_folder}/out.csv"
            # # extractor.add_csv_callback(pattern_csv,delimiter=";")
            # # extractor.add_matplotlib_callback(pattern_png,legend=True,grid={"ls":":"},
            # #     yscale="log",xlabel="Iterations",period=10,start_iter=100)
            # # extractor.add_print_callback(period=50)
            # integral_extraction_trigger = extractor.apply(niter=workflow.Numerics['NumberOfIterations'])[0]
            # pytriggers.append(integral_extraction_trigger)

    return pytriggers
