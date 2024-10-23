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

import os
import timeit
import datetime

from mola.logging import MolaException, GREEN, ENDC
from . import rank, comm
from mola.cfd.coprocess.user_interface import write_tagfile
import mola.naming_conventions as names

def check_timeout(coprocess_manager):
    
    launch_time = coprocess_manager.launch_time
    timeout = coprocess_manager.workflow.RunManagement['TimeOutInSeconds']
    logger = coprocess_manager.mola_logger 

    if coprocess_manager.status.startswith('RUNNING'):
            
        if has_reached_timeout( launch_time, timeout, logger):
            write_tagfile(names.FILE_NEWJOB_REQUIRED, coprocess_manager)
            coprocess_manager.status = 'TO_STOP'

        comm.barrier()



def check_max_iteration(coprocess_manager):

    if coprocess_manager.status.startswith('RUNNING'):
        itinit = coprocess_manager.workflow.Numerics['IterationAtInitialState']
        itmax  = coprocess_manager.workflow.Numerics['NumberOfIterations']

        if coprocess_manager.iteration >= itinit + itmax:
            coprocess_manager.mola_logger.info(f'{GREEN}REACHED itmax{ENDC}', rank=0)
            write_tagfile(names.FILE_JOB_COMPLETED, coprocess_manager)

            coprocess_manager.status = 'TO_STOP'

        comm.barrier()


def check_convergence_criteria(coprocess_manager):

    it  = coprocess_manager.iteration
    itinit = coprocess_manager.workflow.Numerics['IterationAtInitialState']
    itmin = coprocess_manager.workflow.Numerics['MinimumNumberOfIterations']

    has_done_enough_iterations = (it - itinit) > itmin 
    if has_done_enough_iterations and coprocess_manager.status.startswith('RUNNING'):
        if is_converged(coprocess_manager):
            coprocess_manager.status = 'TO_STOP'


def is_converged(coprocess_manager):

    ConvergenceCriteria = coprocess_manager.workflow.ConvergenceCriteria
    if not ConvergenceCriteria: return False

    CONVERGED = False
    all_necessary_criteria_are_verified = True
    a_sufficient_criterion_is_verified = False
    if rank == 0:
       
        for criterion in ConvergenceCriteria:
            criterion_is_verified = is_criterion_flux_lower_than_threshold(criterion, coprocess_manager.Extractions)

            if not criterion_is_verified and criterion['Necessary']:
                all_necessary_criteria_are_verified = False
                break

            if criterion_is_verified and criterion['Sufficient']:
                a_sufficient_criterion_is_verified = True
                break

        if a_sufficient_criterion_is_verified or all_necessary_criteria_are_verified:
            CONVERGED = True
        else:
            return False
        
        txt = get_convergence_message(ConvergenceCriteria, coprocess_manager.iteration)
        coprocess_manager.mola_logger.info(txt, rank=0)

    comm.barrier()
    CONVERGED = comm.bcast(CONVERGED, root=0)

    return CONVERGED


def is_criterion_flux_lower_than_threshold(criterion, Extractions) -> bool:

    Flux = get_data_to_test_criterion(criterion, Extractions)

    if Flux is None:
        raise MolaException(
           f"requested convergence variable {criterion['Variable']} not found in {criterion['ExtractionName']}")

    criterion['FoundValue'] = Flux[-1]
    criterion_verified = criterion['FoundValue'] < criterion['Threshold']
    criterion['CriterionVerified'] = True if criterion_verified else False

    return criterion_verified


def get_data_to_test_criterion(criterion, Extractions):
    for extraction in Extractions:
        if extraction['Type'] in ['Integral', 'Probe'] \
            and extraction['Name'] == criterion['ExtractionName']:

            if 'Data' not in extraction:
                raise MolaException(f"data of extraction {extraction['Name']} is not available")

            try:
                return extraction['Data'].get(Name=criterion['Variable']).value()
            except:
                extraction['Data'].save('debug.cgns')
                raise MolaException(f'Cannot evaluate convergence for criterion {criterion}, because the variable is not found in extracted data.')


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


def has_reached_timeout(LaunchTime, TimeOutInSeconds, mola_logger=None):

    ReachedTimeOutMargin = False
    if rank == 0:
        ElapsedTime = timeit.default_timer() - LaunchTime
        ReachedTimeOutMargin = ElapsedTime >= TimeOutInSeconds
        if ReachedTimeOutMargin:
            date = datetime.datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            msg = f'REACHED MARGIN BEFORE TIMEOUT at {date} --> STOP SIMULATION'
            if mola_logger:
                mola_logger.warning(msg, rank=0)
            else:
                print(msg)
    comm.Barrier()
    ReachedTimeOutMargin = comm.bcast(ReachedTimeOutMargin,root=0)

    return ReachedTimeOutMargin


