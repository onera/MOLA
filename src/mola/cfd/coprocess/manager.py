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
from fnmatch import fnmatch
import glob
import shutil
import timeit
import copy
import numpy as np

from treelab import cgns
from mola.logging import (MolaException, MolaAssertionError, MolaUserError,
                          MolaLogger, CYAN, ENDC, GREEN)
import mola.naming_conventions as names
import mola.server as SV
from mola.cfd import call_solver_specific_function
from mola.cfd.preprocess.mesh.io.writer import write

from . import rank, comm
from .tools import move_log_files, check_stderr, write_tagfile
from .stopping_criteria import check_timeout, check_max_iteration, check_convergence_criteria
from .user_interface import check_and_execute_user_signal
from .probes import has_probes, search_zone_and_index_for_probes

AVAILABLE_SIMULATION_STATUS = [
    'BEFORE_FIRST_ITERATION',
    'RUNNING_BEFORE_ITERATION',
    'RUNNING_AFTER_ITERATION', 
    'TO_STOP', 
    'TO_FINALIZE',
    'COMPLETED', 
]


class CoprocessManager():

    def __init__(self, workflow):
        self.workflow = workflow

        self.make_directories_and_log()

        self.iteration = self.workflow.Numerics['IterationAtInitialState'] - 1
        self.time = self.workflow.Numerics['TimeAtInitialState']

        self.launch_time = timeit.default_timer()
        if self.workflow.Numerics['NumberOfIterations'] == 0:
            err_msg = 'NumberOfIterations=0 => simulation cannot begin. Please change this value and submit again.'
            self.mola_logger.error(err_msg, rank=0)
            raise MolaUserError(err_msg)

        self.status = 'BEFORE_FIRST_ITERATION'

        # NOTE It is important to have a copy of Extractions
        # because several keys will be added for each extraction: 
        #   IsToExtract (bool), IsToSave (bool), Data (PyTree or other kind of volumic data)
        # and these elements must not be saved when saving the workflow.
        self.Extractions = copy.deepcopy(workflow.Extractions)
        self.initialize_extraction_data_from_last_run()

        if has_probes(workflow):
            search_zone_and_index_for_probes(self, comm)

    def __del__(self):
        if self.status != 'COMPLETED':
            self.mola_logger.warning(f'CoprocessManager is deleted but simulation status is {self.status} instead of COMPLETED.', rank=0)

    def run_iteration(self):
        self.update_iteration()
        has_reached_max_iteration = check_max_iteration(self)
        if not has_reached_max_iteration:
            has_reached_timeout = check_timeout(self)
            if not has_reached_timeout:
                check_and_execute_user_signal(self)
                self.apply_operations()
                check_convergence_criteria(self)

        if self.status == 'TO_STOP':
            self.end_simulation()


    def update_iteration(self):
        self.status = call_solver_specific_function(self.workflow, 'get_status', 3)
        for extraction in self.Extractions:
            extraction['IsToExtract'] = False
            extraction['IsToSave'] = False

        self.iteration = call_solver_specific_function(self.workflow, 'get_iteration', 3)
        self.mola_logger.info(f'iteration {self.iteration:d}', rank=0)

        if self.workflow.Numerics['TimeMarching'] != 'Steady':
            self.time += self.workflow.Numerics['TimeStep']

        self.update_extractions_to_perform()
        # TODO add body-force in the operations_stack if needed

    
    def update_extractions_to_perform(self):
        # FIXME Fast starts at iteration 0, so all extractions are extracted and saved
        # at iteration 0 (because of modulo). 
        # Change iteration number in MOLA (n in MOLA <--> n-1 in Fast) ?
        # But residuals have iterations coming directly from Fast...
        if self.iteration == self.workflow.Numerics['IterationAtInitialState'] - 1:
            # e.g. exclude the iteration 0 for a simulation with elsa
            for extraction in self.Extractions:                
                extraction['IsToExtract'] = False
                extraction['IsToSave'] = False
            return
        
        for extraction in self.Extractions:                
            if self.iteration % extraction['ExtractionPeriod'] == 0:
                extraction['IsToExtract'] = True
            if self.iteration % extraction['SavePeriod'] == 0:
                extraction['IsToSave'] = True
                    
    def apply_operations(self):
        if any([extraction['IsToExtract'] for extraction in self.Extractions]):
            self.mola_logger.debug(f'Performing extractions..', rank=0)
            self.perform_extractions()
            # FIXME postprocess is after the update of signals, so variables like avg-MomentumX are not updated
            self.postprocess_extractions()
            self._update_workflow_parameters_for_restart_if_needed()
            
        comm.barrier()

        if any([extraction['IsToSave'] for extraction in self.Extractions]):
            self.mola_logger.debug(f'Saving data...', rank=0)
            self.save_data()
    
    def initialize_extraction_data_from_last_run(self):

        # dictonary that indicates Base name for each extraction Type
        type_to_base_name = dict((k,k) for k in ['Integral','Residuals', 'TimeMonitoring', 'MemoryUsage'])
        type_to_base_name['Probe'] = 'Probes'

        for extraction in self.Extractions:
            if extraction['Type'] not in list(type_to_base_name): continue

            if 'File' not in extraction: continue

            try:
                previous_tree = cgns.load(os.path.join(names.DIRECTORY_OUTPUT,extraction['File']))
            except FileNotFoundError:
                continue

            for base in previous_tree.bases():
                if base.name() != type_to_base_name[extraction['Type']]:
                    base.dettach()
                
                if extraction["Type"] in ['Integral', 'Probe']:
                    for zone in base.zones():
                        if zone.name() != extraction["Name"]:
                            zone.dettach()

            if previous_tree.numberOfBases() > 0:
                extraction['Data'] = previous_tree
            else:
                # None because the previous file was empty
                # It is important that the stored value is None for tests during coprocess
                extraction['Data'] = None

    def end_simulation(self):
        if self.status == 'TO_STOP':
            self.status = 'TO_FINALIZE'
            call_solver_specific_function(self.workflow, 'end_simulation', 3)

    def perform_extractions(self):
        call_solver_specific_function(self.workflow, 'perform_extractions', 3, self)

    def save_data(self):

        def sort_extractions_to_save_by_file():
            files_to_save = dict()
            for extraction in self.Extractions:
                if 'Data' not in extraction: continue

                if extraction['IsToSave'] and extraction['Data'] is not None:
                    if extraction['Type'] == 'Restart':
                        filename = extraction['File']
                    else:
                        filename = os.path.join(names.DIRECTORY_OUTPUT, extraction['File'])
                    override = extraction.get('Override', True)
                    if not override:
                        # Add a suffix _AfterIter<Iteration>
                        f2cSplit = filename.split('.')
                        name = '.'.join(f2cSplit[:-1])
                        fmt = f2cSplit[-1]
                        filename = f'{name}_AfterIter{self.iteration}.{fmt}'
                    files_to_save.setdefault(filename, [])
                    files_to_save[filename].append(extraction['Data'])
                
            return files_to_save

        files_to_save = sort_extractions_to_save_by_file()
        for filename, data_pytrees in files_to_save.items():
            tree_to_save = cgns.merge(data_pytrees)
            self.save(tree_to_save, filename)

    def save(self, data, filename):
        self.mola_logger.info(f'{CYAN}saving {filename}...{ENDC}', rank=0)
        if self.workflow.SplittingAndDistribution['Splitter'].lower() in ['cassiopee', 'pypart']:
            io_tool = 'cassiopee_mpi'
        else:
            io_tool = None
        write(self.workflow, data, filename, io_tool=io_tool)
        self.mola_logger.info(f'{GREEN}saving {filename}... OK{ENDC}', rank=0)
         
    def finalize(self):
        self.iteration = call_solver_specific_function(self.workflow, 'get_iteration', 3)
        self.mola_logger.info(f'>> finalize after iteration {self.iteration}', rank=0)
        self.status = 'TO_FINALIZE'
        for extraction in self.Extractions:
            if extraction['ExtractAtEndOfRun']:
                extraction['IsToExtract'] = True
                extraction['IsToSave'] = True
        self.update_extractions_to_perform()
        self.apply_operations()

        self.after_compute()

        self.status = 'COMPLETED'
        move_log_files()
        try:
            call_solver_specific_function(self.workflow, 'move_log_files', 3)
        except MolaException:
            pass
        
        check_stderr()
        if not SV.is_file(names.FILE_NEWJOB_REQUIRED):
            write_tagfile(names.FILE_JOB_COMPLETED, self)

    def after_compute(self):
        if hasattr(self.workflow, 'after_compute'):
            self.mola_logger.info('try to postprocess...', rank=0)
            try:
                self.workflow.after_compute()
                self.mola_logger.info(f'  {CYAN}> postprocess done.{ENDC}', rank=0)
            except Exception as err:
                if rank == 0:
                    with open('stderr-post.log', 'w') as f:
                        f.write(str(err)+'\n')
                self.mola_logger.warning(f'  > postprocess failed. See stderr-post.log', rank=0)

    def _update_workflow_parameters_for_restart_if_needed(self):
        found_restart_tree = False
        for extraction in self.Extractions:
            if extraction['Type'] == 'Restart' and extraction['IsToExtract']:
                # restart tree is in extraction['Data']
                assert 'Data' in extraction and extraction['Data'] is not None
                found_restart_tree = True
                break
        if not found_restart_tree: 
            return

        self.workflow.Numerics['NumberOfIterations'] -= self.iteration - self.workflow.Numerics['IterationAtInitialState'] + 1
        self.workflow.Numerics['IterationAtInitialState'] = self.iteration + 1 
        if 'TimeStep' in self.workflow.Numerics:
            self.workflow.Numerics['TimeAtInitialState'] = self.iteration * self.workflow.Numerics['TimeStep']

        # Update only Numerics node in tree
        WorkflowParameters = extraction['Data'].get(Name=self.workflow._workflow_parameters_container_, Depth=1)
        if WorkflowParameters:
            WorkflowParameters.setParameters('Numerics', **self.workflow.Numerics)

    def make_directories_and_log(self):

        run_dir = self.workflow.RunManagement.get('RunDirectory','.')
        
        if run_dir == "." or run_dir == os.path.basename(os.getcwd()):
            output_dir = names.DIRECTORY_OUTPUT
            log_dir = names.DIRECTORY_LOG
            colog_file_path = names.FILE_COLOG

        else: 
            output_dir = os.path.join(run_dir,names.DIRECTORY_OUTPUT)
            log_dir = os.path.join(run_dir,names.DIRECTORY_LOG)
            colog_file_path = os.path.join(run_dir, names.FILE_COLOG)
                
            
        if rank==0:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(log_dir, exist_ok=True)

        self.mola_logger = MolaLogger(stream=False, filename=colog_file_path, level='DEBUG')

    @property
    def status(self):
        return self._status
    
    @status.setter
    def status(self, value):
        if value in AVAILABLE_SIMULATION_STATUS:
            self._status = value
        else:
            raise MolaException(f"The value {value} is not among the AVAILABLE_SIMULATION_STATUS ({', '.join(AVAILABLE_SIMULATION_STATUS)})")

    def __del__(self):
        if self.status != 'COMPLETED':
            self.mola_logger.warning(f'CoprocessHandler is deleted but simulation status is {self.status} instead of COMPLETED.', rank=0)

    def elapsed_time(self):
        '''Return the elapsed time from run starting in seconds'''
        elapsed_time = timeit.default_timer() - self.launch_time
        return elapsed_time

    def postprocess_extractions(self):
        from mola.cfd.postprocess.signals import apply_operations_on_signal, AVAILABLE_OPERATIONS_ON_SIGNALS

        for extraction in self.Extractions:
            PostprocessOperations = extraction.get('PostprocessOperations', [])

            for operation in PostprocessOperations:
                AtEndOfRunOnly = operation.get('AtEndOfRunOnly', True)
                is_to_postprocess = self.status=='TO_FINALIZE' or not AtEndOfRunOnly
                if not is_to_postprocess:
                    continue

                if operation['Type'] in AVAILABLE_OPERATIONS_ON_SIGNALS:
                    # ex: PostprocessOperations = [dict(Type='avg', Variable='MassFlow')]
                    self.mola_logger.debug(f"  compute {operation['Type']}-{operation['Variable']} on {extraction['Name']}", rank=0)
                    apply_operations_on_signal(extraction['Data'], operation['Variable'], 
                                               extraction['TimeAveragingIterations'], 
                                               operations=[operation['Type']])

