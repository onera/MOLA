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
import glob
import shutil
import timeit
import copy

from treelab import cgns
from mola.logging import MolaException, MolaAssertionError, MolaUserError
import mola.naming_conventions as names
from mola.cfd import call_solver_specific_function

from . import mola_logger, rank, comm
from .stopping_criteria import check_timeout, check_max_iteration, check_convergence_criteria
from .user_interface import update_operations_from_user_signal


AVAILABLE_SIMULATION_STATUS = [
    'BEFORE_FIRST_ITERATION',
    'RUNNING', 
    'TO_STOP', 
    'TO_FINALIZE',
    'COMPLETED', 
]

SORTED_AVAILABLE_OPERATIONS = [
    'PERFORM_EXTRACTIONS', 
    'COMPUTE_BODYFORCE', 
    'SAVE_BODYFORCE', 
    'SAVE_SIGNALS', 
    'SAVE_EXTRACTIONS', 
    'SAVE_FIELDS',
    'SAVE_RESTART',
]

class OperationsStack(list):

    def check(self, value):
        if value not in SORTED_AVAILABLE_OPERATIONS:
            raise MolaAssertionError(f"Undefined operation '{value}' (must be among {', '.join(SORTED_AVAILABLE_OPERATIONS)})")
    
    def sort(self):
        super().sort(key=lambda x: SORTED_AVAILABLE_OPERATIONS.index(x))

    def __setitem__(self, index, value):
        self.check(value)
        if value not in self:
            super()[index] = value
            self.sort()

    def insert(self, index, value):
        self.check(value)
        if value not in self:
            super().insert(index, value)
            self.sort()

    def append(self, value):
        self.check(value)
        if value not in self:
            super().append(value)
            self.sort()


class CoprocessManager():

    def __init__(self, workflow):
        self.workflow = workflow

        self.iteration = self.workflow.Numerics['IterationAtInitialState'] - 1
        self.launch_time = timeit.default_timer()
        if self.workflow.Numerics['NumberOfIterations'] == 0:
            err_msg = 'NumberOfIterations=0 => simulation cannot begin. Please change this value and submit again.'
            mola_logger.error(err_msg, rank=0)
            raise MolaUserError(err_msg)

        # self.operations_stack = OperationsStack()
        # self.extractions_to_perform = []
        self._status = 'BEFORE_FIRST_ITERATION'

        self.skeleton = None

        # NOTE It is important to have a copy of Extractions
        # because several keys will be added for each extraction: 
        #   IsToExtract (bool), IsToSave (bool), Data (PyTree or other kind of volumic data)
        # and these elements must not be saved when saving the workflow.
        self.Extractions = copy.deepcopy(workflow.Extractions)
        

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

        # for extraction in self.workflow.Extractions:
        #     for key in ['IsToExtract', 'IsToSave', 'Data']:
        #         if key in extraction:
        #             del extraction[key]

        if self.status != 'COMPLETED':
            mola_logger.warning(f'CoprocessHandler is deleted but simulation status is {self.status} instead of COMPLETED.', rank=0)

                    
    def run_iteration(self):
        self.update_iteration()
        update_operations_from_user_signal(self)
        check_timeout(self)
        self.apply_operations()
        check_max_iteration(self)
        check_convergence_criteria(self)

        if self.status == 'TO_STOP':
            self.end_simulation()

    def update_iteration(self):
        self.status = 'RUNNING'
        # self.operations_stack.clear()
        # self.extractions_to_perform.clear()
        for extraction in self.Extractions:
            extraction['IsToExtract'] = False
            extraction['IsToSave'] = False

        self.iteration += 1
        mola_logger.info(f'iteration {self.iteration:d}', rank=0)

        self.update_extractions_to_perform()

        # TODO add body-force in the operations_stack if needed
    
    def update_extractions_to_perform(self, force_extractions=False):
        # for extraction in self.Extractions:
        #     on_extraction_period = self.iteration % extraction['ExtractionPeriod'] == 0
        #     ask_save_fields = 'SAVE_FIELDS' in self.operations_stack and extraction['Type'] in ['Restart', '3D']
        #     ask_save_extractions = 'SAVE_EXTRACTIONS' in self.operations_stack and extraction['Type'] in ['BC', 'IsoSurface']
        #     ask_save_signals = 'SAVE_SIGNALS' in self.operations_stack and extraction['Type'] in ['Integral', 'Probe']

        #     if on_extraction_period or ask_save_fields or ask_save_extractions or ask_save_signals:
        #         self.operations_stack.append('PERFORM_EXTRACTIONS')
        #         self.extractions_to_perform.append(extraction)

        #     if self.iteration % extraction['SavePeriod'] == 0:
        #         if extraction['Type'] in ['Restart', '3D']:
        #             self.operations_stack.append('SAVE_FIELDS')
        #         elif extraction['Type'] in ['BC', 'IsoSurface']:
        #             self.operations_stack.append('SAVE_EXTRACTIONS')
        #         elif extraction['Type'] in ['Integral', 'Probe']:
        #             self.operations_stack.append('SAVE_SIGNALS')
        #         else:
        #             mola_logger.warning(f"Unknown extraction type: {extraction['Type']}")

        for extraction in self.Extractions:
            if self.iteration % extraction['ExtractionPeriod'] == 0 or force_extractions:
                extraction['IsToExtract'] = True
            if self.iteration % extraction['SavePeriod'] == 0 or force_extractions:
                extraction['IsToSave'] = True
                    
    def apply_operations(self):
        # for operation in self.operations_stack:
        #     mola_logger.debug(f'next operation if applicable: {operation}', rank=0)
        #     method = getattr(self, operation.lower())
        #     method()

        if any([extraction['IsToExtract'] for extraction in self.Extractions]):
            mola_logger.debug(f'Performing extractions..', rank=0)
            self.perform_extractions()

            if any([extraction['Type'] == 'Restart' and extraction['IsToExtract']  for extraction in self.Extractions]):
                self._update_workflow_parameters_for_restart()
            
        comm.barrier()

        if any([extraction['IsToSave'] for extraction in self.Extractions]):
            mola_logger.debug(f'Saving data...', rank=0)
            self.save_data()
    
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
                if extraction['IsToSave'] and extraction['Data'] is not None:
                    if extraction['Type'] == 'Restart':
                        filename = extraction['File']
                    else:
                        filename = os.path.join(names.DIRECTORY_OUTPUT, extraction['File'])
                    files_to_save.setdefault(filename, [])
                    files_to_save[filename].append(extraction['Data'])
            return files_to_save
        
        def merge_data(data_pytrees):
            import Converter.Internal as I
            return cgns.castNode(I.merge(data_pytrees))
            # if len(data_pytrees) > 1:
            #     tree_to_save = data_pytrees[0].merge(data_pytrees[1:])
            # else:
            #     tree_to_save = data_pytrees[0]
            # return tree_to_save
    
        files_to_save = sort_extractions_to_save_by_file()
        for filename, data_pytrees in files_to_save.items():
            tree_to_save = merge_data(data_pytrees)
            self.save(tree_to_save, filename)

    def save(self, data, filename, tag_with_iteration=False):
        from mola.cfd.preprocess.mesh.io.writer import write
        from mola.logging import CYAN, ENDC, GREEN

        if data is not None:
            if tag_with_iteration:
                f2cSplit = filename.split('.')
                name = '.'.join(f2cSplit[:-1])
                fmt = f2cSplit[-1]
                filename = f'{name}_AfterIter{self.iteration}.{fmt}'

            mola_logger.info(f'{CYAN}saving {filename}...{ENDC}', rank=0)
            write(self.workflow, data, filename)
            mola_logger.info(f'{GREEN}saving {filename}... OK{ENDC}', rank=0)

            
    def finalize(self):
        # mola_logger.info(f'>> finalize', rank=0)
        # self.operations_stack.clear()
        # self.operations_stack.extend(['SAVE_RESTART', 'SAVE_FIELDS', 'SAVE_EXTRACTIONS', 'SAVE_SIGNALS'])
        # self.extractions_to_perform.clear()
        # self.update_extractions_to_perform()

        # self.apply_operations()

        # self.update_and_save_workflow_for_restart()

        # self.status = 'COMPLETED'
        # moveLogFiles()
        # check_stderr_and_create_COMPLETED()

        mola_logger.info(f'>> finalize', rank=0)
        self.update_extractions_to_perform(force_extractions=True)

        self.apply_operations()

        # self.update_and_save_workflow_for_restart()

        self.status = 'COMPLETED'
        moveLogFiles()
        check_stderr_and_create_COMPLETED()

    def _update_workflow_parameters_for_restart(self):
        if rank == 0:
            self.workflow.Numerics['NumberOfIterations'] -= self.iteration - self.workflow.Numerics['IterationAtInitialState'] + 1
            self.workflow.Numerics['IterationAtInitialState'] = self.iteration + 1
            if 'TimeStep' in self.workflow.Numerics:
                self.workflow.Numerics['TimeAtInitialState'] = self.iteration * self.workflow.Numerics['TimeStep']
            
            from mola.cfd.preprocess.cfd_parameters import apply
            apply(self.workflow)

            self.workflow.set_workflow_parameters_in_tree()
    

def moveLogFiles():
    if rank == 0:
        try: os.makedirs(names.DIRECTORY_LOG)
        except: pass

        for fn in glob.glob('*.log'):
            FilenameBase = fn[:-4]
            i = 1
            NewFilename = FilenameBase+'-%d'%i+'.log'
            while os.path.isfile(os.path.join(names.DIRECTORY_LOG, NewFilename)):
                i += 1
                NewFilename = FilenameBase+'-%d'%i+'.log'

            shutil.move(fn, os.path.join(names.DIRECTORY_LOG, NewFilename))

        for fn in glob.glob('elsA_MPI*'):
            shutil.move(fn, os.path.join(names.DIRECTORY_LOG, fn))

    comm.barrier()

def check_stderr_and_create_COMPLETED():
    check_stderr()
    if rank == 0:
        with open(names.FILE_JOB_COMPLETED,'w') as f: 
            f.write(names.FILE_JOB_COMPLETED)
    
def check_stderr():
    # TODO Simple check for now, but it should be different if this function is called in the coprocess script
    if rank == 0:
        try:
            with open(names.FILE_STDERR,'r') as f:
                Error = f.read()
            raise Exception(Error)
        except FileNotFoundError:
            pass


