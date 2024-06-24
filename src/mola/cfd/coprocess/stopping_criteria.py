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

import timeit
import datetime

from mola.logging import GREEN, ENDC
from . import mola_logger, rank, comm

def check_timeout(coprocess_manager):
    if coprocess_manager.status == 'RUNNING':
        ReachedTimeOutMargin = _has_reached_timeout(coprocess_manager.launch_time, coprocess_manager.workflow.RunManagement['TimeOutInSeconds'])
        if ReachedTimeOutMargin :
            if rank == 0:
                with open('NEWJOB_REQUIRED', 'w') as f: 
                    f.write('NEWJOB_REQUIRED')

            coprocess_manager.status = 'TO_STOP'
            coprocess_manager.operations_stack.clear()

        comm.barrier()

def check_max_iteration(coprocess_manager):
    if coprocess_manager.status == 'RUNNING':
        Numerics = coprocess_manager.workflow.Numerics
        if coprocess_manager.iteration >= Numerics['IterationAtInitialState'] + Numerics['NumberOfIterations']:
            mola_logger.warning(f'{GREEN}REACHED itmax{ENDC}', rank=0)
            if rank == 0:
                with open('COMPLETED', 'w') as f: 
                    f.write('COMPLETED')

            coprocess_manager.status = 'TO_STOP'
            coprocess_manager.operations_stack.clear()

        comm.barrier()

def check_convergence_criteria(coprocess_manager):
    has_done_enough_iterations = (coprocess_manager.iteration - coprocess_manager.workflow.Numerics['IterationAtInitialState']) > coprocess_manager.workflow.Numerics['MinimumNumberOfIterations'] 
    if has_done_enough_iterations and coprocess_manager.status == 'RUNNING':
        if _is_converged(coprocess_manager.workflow.ConvergenceCriteria, coprocess_manager.Extractions, coprocess_manager.iteration):
            coprocess_manager.status = 'TO_STOP'

def _is_converged(ConvergenceCriteria, Extractions, iteration):
    '''
    This method is used to determine if the current simulation is converged by
    looking at user-provided convergence criteria.
    If converged, the signal returns :py:obj:`True` to all ranks and writes a
    message to ``coprocess.log`` file.

    Parameters
    ----------

        ConvergenceCriteria : :py:class:`list` of :py:class:`dict`
            Each :py:class:`dict` corresponds to a criterion. Its has the
            following keys:

            * ``ExtractionName``: Name of the zone to monitor (shall exist in
            ``arrays.cgns``)

            * ``Variable``: Name of the variable to monitor on ``ExtractionName``

            * ``Threshold``: Value of the threshold to consider. The current
            criterion is satisfied if the value of the last element of
            ``Variable`` on ``ExtractionName`` in ``arrays.cgns`` is lower than
            ``Threshold``.

            * ``Condition`` (optinal, 'Necessary' by default): logical
            requirement for the current criterion. To verify convergence,
            criteria tagged 'Necessary' must all be satisfied simultaneously
            and at least one criterion tagged 'Sufficient' must be satisfied.
            For instance, if CN1 and CN2 are 'Necessary' and CS1 and CS2 are
            'Sufficient', convergence is reached when:
            (CN1 AND CN2) AND (CS1 OR CS2)

    Returns
    -------

        CONVERGED : bool
            :py:obj:`True` if the convergence criteria are satisfied
    '''
    if not ConvergenceCriteria:
        return False
    
    def get_data_to_test_criterion(criterion, Extractions):
        for extraction in Extractions:
            if extraction['Type'] in ['Integral', 'Probe'] \
                and extraction['Name'] == criterion['ExtractionName']:
                try:
                    return extraction['Data'].get(Name=criterion['Variable']).value()
                except:
                    pass
                    # mola_logger.warning(f'Cannot evaluate convergence for criterion {criterion}, because the variable is not found in extracted data.')
            
        return 

    CONVERGED = False
    if rank == 0:
        AllNecessaryCriteria = True
        OneSufficientCriterion = True
        # Default value of Condition = 'Necessary'
        for criterion in ConvergenceCriteria:
            if 'Condition' not in criterion:
                criterion['Condition'] = 'Necessary'
            elif criterion['Condition'] == 'Sufficient':
                OneSufficientCriterion = False
        try:
            for criterion in ConvergenceCriteria:
                if OneSufficientCriterion and criterion['Condition'] == 'Sufficient':
                    continue

                Flux = get_data_to_test_criterion(criterion, Extractions)
                if Flux is None and criterion['Condition'] == 'Necessary':
                    mola_logger.warning(f"requested convergence variable {criterion['Variable']} not found in {criterion['ExtractionName']}", rank=0)
                    AllNecessaryCriteria = False
                    continue
                criterion['FoundValue'] = Flux[-1]
                IsSatisfied = criterion['FoundValue'] < criterion['Threshold']
                if criterion['Condition'] == 'Necessary' and not IsSatisfied:
                    AllNecessaryCriteria = False
                    break
                elif criterion['Condition'] == 'Sufficient' and IsSatisfied:
                    OneSufficientCriterion = criterion['Variable']

            CONVERGED = OneSufficientCriterion and AllNecessaryCriteria
            if CONVERGED:
                MSG = 'CONVERGED at iteration {} since:'.format(iteration - 1)
                for criterion in ConvergenceCriteria:
                    if criterion['Condition'] == 'Necessary' \
                        or criterion['Variable'] == OneSufficientCriterion:
                        MSG += '\n  {}={} < {} on {} ({})'.format(criterion['Variable'],
                                                            criterion['FoundValue'],
                                                            criterion['Threshold'],
                                                            criterion['ExtractionName'],
                                                            criterion['Condition'])
                txt = f'''{GREEN}*******************************************
{MSG} 
*******************************************{ENDC}'''
                mola_logger.info(txt, rank=0)
        except BaseException as e:
            mola_logger.error(f'_is_converged failed: {e}', rank=0)

    comm.barrier()
    CONVERGED = comm.bcast(CONVERGED, root=0)

    return CONVERGED

def _has_reached_timeout(LaunchTime, TimeOutInSeconds):

    ReachedTimeOutMargin = False
    if rank == 0:
        ElapsedTime = timeit.default_timer() - LaunchTime
        ReachedTimeOutMargin = ElapsedTime >= TimeOutInSeconds
        if ReachedTimeOutMargin:
            date = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            mola_logger.warning(f'REACHED MARGIN BEFORE TIMEOUT at {date} --> STOP SIMULATION', rank=0)
    comm.Barrier()
    ReachedTimeOutMargin = comm.bcast(ReachedTimeOutMargin,root=0)

    return ReachedTimeOutMargin

