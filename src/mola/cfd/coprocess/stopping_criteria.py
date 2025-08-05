#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute self.iteration and/or modify
#    self.iteration under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that self.iteration will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

import datetime
from pprint import pformat as pretty

from mola.logging import MolaException, GREEN, ENDC
from . import rank, comm
from mola.cfd.coprocess.tools import write_tagfile
import mola.naming_conventions as names

def check_max_iteration(coprocess_manager):

    has_reached_max_iteration = False

    if coprocess_manager.status.startswith('RUNNING'):
        itinit = coprocess_manager.workflow.Numerics['IterationAtInitialState']
        itmax  = coprocess_manager.workflow.Numerics['NumberOfIterations']

        if coprocess_manager.iteration >= itinit + itmax:
            coprocess_manager.mola_logger.info(f'{GREEN}REACHED MAX ITERATION{ENDC}', rank=0)

            coprocess_manager.status = 'TO_STOP'
            has_reached_max_iteration = True

        comm.barrier()

    return has_reached_max_iteration

def check_timeout(coprocess_manager):

    is_to_stop = False
    
    launch_time = coprocess_manager.launch_time
    timeout = coprocess_manager.workflow.RunManagement['TimeOutInSeconds']

    if coprocess_manager.status.startswith('RUNNING'):
            
        if has_reached_timeout(coprocess_manager.elapsed_time(), timeout):
            date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            msg = f'REACHED MARGIN BEFORE TIMEOUT at {date} --> STOP SIMULATION'
            coprocess_manager.mola_logger.warning(msg, rank=0)

            write_tagfile(names.FILE_NEWJOB_REQUIRED, coprocess_manager)
            coprocess_manager.status = 'TO_STOP'
            is_to_stop = True

        comm.barrier()
    
    return is_to_stop

def has_reached_timeout(elapsed_time, TimeOutInSeconds):
    ReachedTimeOutMargin = False
    if rank == 0:
        ReachedTimeOutMargin = elapsed_time >= TimeOutInSeconds
            
    comm.Barrier()
    ReachedTimeOutMargin = comm.bcast(ReachedTimeOutMargin,root=0)

    return ReachedTimeOutMargin

def check_convergence_criteria(coprocess_manager):
    # Check if there is at least one criterion, otherwise return False
    if len(coprocess_manager.workflow.ConvergenceCriteria) == 0:
        return False

    has_reached_convergence_criteria = False

    it  = coprocess_manager.iteration + 1  # we are after the time advance of the solver
    itinit = coprocess_manager.workflow.Numerics['IterationAtInitialState']
    itmin = coprocess_manager.workflow.Numerics['MinimumNumberOfIterations']

    has_done_enough_iterations = (it - itinit) >= itmin 
    if has_done_enough_iterations and coprocess_manager.status.startswith('RUNNING'):
        if is_converged(coprocess_manager):
            coprocess_manager.status = 'TO_STOP'
            has_reached_convergence_criteria = True
        print_quantities_used_for_convergence(coprocess_manager)
        
    return has_reached_convergence_criteria


def is_converged(coprocess_manager):

    ConvergenceCriteria = coprocess_manager.workflow.ConvergenceCriteria
    if not ConvergenceCriteria: 
        return False

    all_necessary_criteria_are_verified = any([criterion['Necessary'] for criterion in ConvergenceCriteria])
    any_sufficient_criterion_is_verified = False
    CONVERGED = None
    if rank == 0:
       
        for criterion in ConvergenceCriteria:
            criterion_is_verified = is_criterion_flux_lower_than_threshold(criterion, coprocess_manager.Extractions)

            if criterion['Sufficient'] and criterion_is_verified:
                any_sufficient_criterion_is_verified = True
                break

            if criterion['Necessary'] and not criterion_is_verified:
                all_necessary_criteria_are_verified = False
                break

        CONVERGED = any_sufficient_criterion_is_verified or all_necessary_criteria_are_verified

        if CONVERGED:
            txt = get_convergence_message(ConvergenceCriteria, coprocess_manager.iteration)
            coprocess_manager.mola_logger.info(txt, rank=0)

    comm.barrier()
    CONVERGED = comm.bcast(CONVERGED, root=0)

    return CONVERGED


def is_criterion_flux_lower_than_threshold(criterion, Extractions) -> bool:

    Flux = get_data_to_test_criterion(criterion, Extractions)

    criterion['FoundValue'] = Flux[-1]
    criterion_verified = criterion['FoundValue'] < criterion['Threshold']
    criterion['CriterionVerified'] = True if criterion_verified else False

    return criterion_verified

def print_quantities_used_for_convergence(coprocess_manager):
    msg = ''
    for criterion in coprocess_manager.workflow.ConvergenceCriteria:
        if 'FoundValue' in criterion:
            value_str = "%g"%criterion['FoundValue']
            msg += criterion['Variable'] + '=' + value_str + ' '
    all_msg = comm.bcast(msg, root=0)
    single_line = ''.join(all_msg)
    coprocess_manager.mola_logger.info(single_line, rank=0)


def get_data_to_test_criterion(criterion, Extractions):
    for extraction in Extractions:
            
        if extraction['Type'] in ['Integral', 'Probe', 'Residuals'] \
            and extraction['Name'] == criterion['ExtractionName']:

            if 'Data' not in extraction:
                raise MolaException(f"data of extraction {extraction['Name']} is not available")

            try:
                return extraction['Data'].get(Name=criterion['Variable']).value()
            except:
                extraction['Data'].save('debug.cgns')
                extraction_no_data = dict((k, v) for k, v in extraction.items() if k != "Data")
                var_name = criterion['Variable']
                msg = (f'Cannot evaluate convergence for criterion {criterion} '
                       f'because the variable "{var_name}" is not found in '
                       f'extraction {pretty(extraction_no_data)})')
                raise MolaException(msg)


    raise MolaException((f"requested convergence variable {criterion['Variable']}"
                         f" not found in {criterion['ExtractionName']}"))


def get_convergence_message(ConvergenceCriteria, iteration) -> str:
    
    MSG = 'CONVERGED at iteration {} since:'.format(iteration)
    for criterion in ConvergenceCriteria:
        if 'CriterionVerified' in criterion and criterion['CriterionVerified']:
            
            MSG += '\n  {}={} < {} on {} (Sufficient={}, Necessary={})'.format(criterion['Variable'],
                                                criterion['FoundValue'],
                                                criterion['Threshold'],
                                                criterion['ExtractionName'],
                                                criterion['Sufficient'],
                                                criterion['Necessary'])
    stars = 43*'*'
    txt = f"{GREEN}{stars}\n{MSG}\n{stars}{ENDC}"

    return txt


