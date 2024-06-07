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

from treelab import cgns
from mola.logging import MolaException, MolaAssertionError, MolaUserError
import mola.naming_conventions as names
from mola.cfd import apply_to_solver, call_solver_specific_function

from . import mola_logger, rank, comm
from .stopping_criteria import check_timeout, check_max_iteration, check_convergence_criteria
from .user_interface import update_operations_from_user_signal
from .tools import save, load_skeleton


AVAILABLE_SIMULATION_STATUS = [
    'BEFORE_FIST_ITERATION',
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
            raise MolaUserError('NumberOfIterations=0 => simulation cannot begin. Please change this value and submit again.')

        self.signals = None
        self.extractions = None
        self.fields = None
        self.restart_fields = None

        self.operations_stack = OperationsStack()
        self.extractions_to_perform = []
        self._status = 'BEFORE_FIST_ITERATION'

        self.skeleton = None
        

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
        self.operations_stack.clear()
        self.extractions_to_perform.clear()
        
        self.iteration += 1
        mola_logger.info(f'iteration {self.iteration:d}', rank=0)

        self.update_extractions_to_perform()

        # TODO add body-force in the operations_stack if needed
    
    def update_extractions_to_perform(self):
        for extraction in self.workflow.Extractions:
            on_extraction_period = self.iteration % extraction['ExtractionPeriod'] == 0
            ask_save_fields = 'SAVE_FIELDS' in self.operations_stack and extraction['Type'] in ['Restart', '3D']
            ask_save_extractions = 'SAVE_EXTRACTIONS' in self.operations_stack and extraction['Type'] in ['BC', 'IsoSurface']
            ask_save_signals = 'SAVE_SIGNALS' in self.operations_stack and extraction['Type'] in ['Integral', 'Probe']

            if on_extraction_period or ask_save_fields or ask_save_extractions or ask_save_signals:
                self.operations_stack.append('PERFORM_EXTRACTIONS')
                self.extractions_to_perform.append(extraction)

            if self.iteration % extraction['SavePeriod'] == 0:
                if extraction['Type'] in ['Restart', '3D']:
                    self.operations_stack.append('SAVE_FIELDS')
                elif extraction['Type'] in ['BC', 'IsoSurface']:
                    self.operations_stack.append('SAVE_EXTRACTIONS')
                elif extraction['Type'] in ['Integral', 'Probe']:
                    self.operations_stack.append('SAVE_SIGNALS')
                else:
                    mola_logger.warning(f"Unknown extraction type: {extraction['Type']}")
                    
    def apply_operations(self):
        for operation in self.operations_stack:
            mola_logger.debug(f'next operation if applicable: {operation}', rank=0)
            method = getattr(self, operation.lower())
            method()
    
    def end_simulation(self):
        if self.status == 'TO_STOP':
            self.status = 'TO_FINALIZE'
            call_solver_specific_function(self.workflow, 'end_simulation', 3)

    def perform_extractions(self):
        call_solver_specific_function(self.workflow, 'perform_extractions', 3, self)
    
    def compute_bodyforce(self):
        ...

    def save_bodyforce(self):
        ...

    def save_signals(self):
        if self.signals is not None:
            save(self.signals, os.path.join(names.DIRECTORY_OUTPUT, names.FILE_OUTPUT_1D), coprocess_manager=self)

    def save_extractions(self):
        if self.extractions is not None:
            save(self.extractions, os.path.join(names.DIRECTORY_OUTPUT, names.FILE_OUTPUT_2D), coprocess_manager=self)

    def save_fields(self):
        if self.fields is not None:
            save(self.fields, os.path.join(names.DIRECTORY_OUTPUT, names.FILE_OUTPUT_3D), coprocess_manager=self)
    
    def save_restart(self):
        if self.restart_fields is not None:
            save(self.restart_fields, os.path.join(names.DIRECTORY_OUTPUT, names.FILE_OUTPUT_RESTART), coprocess_manager=self)

    def finalize(self):
        mola_logger.info(f'>> finalize', rank=0)
        self.operations_stack.clear()
        self.operations_stack.extend(['SAVE_RESTART', 'SAVE_FIELDS', 'SAVE_EXTRACTIONS', 'SAVE_SIGNALS'])
        self.extractions_to_perform.clear()
        self.update_extractions_to_perform()

        self.apply_operations()

        self.update_and_save_workflow_for_restart()

        self.status = 'COMPLETED'
        moveLogFiles()
        check_stderr_and_create_COMPLETED()

    def update_and_save_workflow_for_restart(self):
        self._update_workflow_parameters_for_restart()
        self._update_workflow_tree_for_restart()
        if self.workflow.SplittingAndDistribution['Splitter'].lower() == 'pypart':
            if rank == 0:
                self.workflow.tree.save(names.FILE_INPUT_SOLVER)
        else:
            save(self.workflow.tree, names.FILE_INPUT_SOLVER, coprocess_manager=self)
        if rank == 0:
            try:
                os.remove(os.path.join(names.DIRECTORY_OUTPUT, names.FILE_OUTPUT_RESTART))
            except:
                pass

    def _update_workflow_parameters_for_restart(self):
        self.workflow.Numerics['NumberOfIterations'] -= self.iteration - self.workflow.Numerics['IterationAtInitialState'] + 1
        self.workflow.Numerics['IterationAtInitialState'] = self.iteration + 1
        if 'TimeStep' in self.workflow.Numerics:
            self.workflow.Numerics['TimeAtInitialState'] = self.iteration * self.workflow.Numerics['TimeStep']
        self.workflow.set_workflow_parameters_in_tree()
    
    def _update_workflow_tree_for_restart(self):
        # TODO For now there is an IO that should be avoided, but it requires a bit a work with PyPart...
        if rank == 0:
        
            self.restart_fields = cgns.load(os.path.join(names.DIRECTORY_OUTPUT, names.FILE_OUTPUT_RESTART))
            NodesToUpdate = self.restart_fields.group(Name='FlowSolution#Init*', Type='FlowSolution', Depth=3) # for initial field(s) (possible second order restart)
            NodesToUpdate += self.restart_fields.group(Name='FlowSolution#Average', Type='FlowSolution', Depth=3) 
            NodesToUpdate += self.restart_fields.group(Name='BCDataSet#Average') 

            for node in NodesToUpdate:
                path = node.path()
                node_to_update = self.workflow.tree.getAtPath(path)
                parent = node_to_update.Parent
                node_to_update.remove()
                parent.addChild(node)


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


