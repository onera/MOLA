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

class CoprocessManager():

    def __init__(self, workflow):
        self.workflow = workflow

        self.iteration = self.workflow.Numerics['IterationAtInitialState'] - 1
        self.launch_time = timeit.default_timer()
        if self.workflow.Numerics['NumberOfIterations'] == 0:
            err_msg = 'NumberOfIterations=0 => simulation cannot begin. Please change this value and submit again.'
            mola_logger.error(err_msg, rank=0)
            raise MolaUserError(err_msg)

        self._status = 'BEFORE_FIRST_ITERATION'

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
        for extraction in self.Extractions:
            extraction['IsToExtract'] = False
            extraction['IsToSave'] = False

        self.iteration += 1
        mola_logger.info(f'iteration {self.iteration:d}', rank=0)

        self.update_extractions_to_perform()

        # TODO add body-force in the operations_stack if needed
    
    def update_extractions_to_perform(self):
        for extraction in self.Extractions:
            if self.iteration % extraction['ExtractionPeriod'] == 0:
                extraction['IsToExtract'] = True
            if self.iteration % extraction['SavePeriod'] == 0:
                extraction['IsToSave'] = True
                    
    def apply_operations(self):
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
            if self.workflow.SplittingAndDistribution['Splitter'].lower() in ['cassiopee', 'pypart']:
                io_tool = 'cassiopee_mpi'
            else:
                io_tool = None
            write(self.workflow, data, filename, io_tool=io_tool)
            mola_logger.info(f'{GREEN}saving {filename}... OK{ENDC}', rank=0)
         
    def finalize(self):
        mola_logger.info(f'>> finalize', rank=0)
        for extraction in self.Extractions:
            if extraction['ExtractAtEndOfRun']:
                extraction['IsToExtract'] = True
                extraction['IsToSave'] = True
        self.update_extractions_to_perform()
        self.apply_operations()

        self.status = 'COMPLETED'
        moveLogFiles()
        check_simulation_end_and_create_COMPLETED(self.Extractions)

    def _update_workflow_parameters_for_restart(self):
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

def check_simulation_end_and_create_COMPLETED(Extractions):
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

