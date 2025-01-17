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
from mola.cfd import call_solver_specific_function
from mola.cfd.preprocess.mesh.io.writer import write

from . import rank, comm
from .stopping_criteria import check_timeout, check_max_iteration, check_convergence_criteria
from .user_interface import get_user_signal, write_tagfile


AVAILABLE_SIMULATION_STATUS = [
    'BEFORE_FIRST_ITERATION',
    'RUNNING_BEFORE_ITERATION',
    'RUNNING_AFTER_ITERATION', 
    'TO_STOP', 
    'TO_FINALIZE',
    'COMPLETED', 
]

# Control Flags for interactive control using command 'touch <flag>'
AVAILABLE_SIGNALS = [
    'CONVERGED',
    'SAVE_ALL',
    'COMPUTE_BODYFORCE',
    'SAVE_BODYFORCE',
    'SAVE_RESTART',
    'SAVE_FIELDS',
    'SAVE_EXTRACTIONS'
    'SAVE_SIGNALS',
    'QUIT',
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

    def __del__(self):
        if self.status != 'COMPLETED':
            self.mola_logger.warning(f'CoprocessManager is deleted but simulation status is {self.status} instead of COMPLETED.', rank=0)

    def run_iteration(self):
        self.update_iteration()
        check_timeout(self)
        self.apply_operations()
        check_max_iteration(self)
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
        for extraction in self.Extractions:
            if self.iteration % extraction['ExtractionPeriod'] == 0:
                extraction['IsToExtract'] = True
            if self.iteration % extraction['SavePeriod'] == 0:
                extraction['IsToSave'] = True
                    
    def apply_operations(self):
        if any([extraction['IsToExtract'] for extraction in self.Extractions]):
            self.mola_logger.debug(f'Performing extractions..', rank=0)
            self.perform_extractions()

            if any([extraction['Type'] == 'Restart' and extraction['IsToExtract']  for extraction in self.Extractions]):
                self._update_workflow_parameters_for_restart()
            
        comm.barrier()

        if any([extraction['IsToSave'] for extraction in self.Extractions]):
            self.mola_logger.debug(f'Saving data...', rank=0)
            self.save_data()
    
    def initialize_extraction_data_from_last_run(self):

        for extraction in self.Extractions:
            if extraction['Type'] not in ['Integral','Residuals','Probe']: continue

            if 'File' not in extraction: continue

            try:
                previous_tree = cgns.load(os.path.join(names.DIRECTORY_OUTPUT,extraction['File']))
            except FileNotFoundError:
                continue

            for base in previous_tree.bases():
                if base.name() != extraction["Type"]:
                    base.dettach()
                
                if extraction["Type"] == 'Integral':
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
        self.normalize_data_from_extractions()
    
    def normalize_data_from_extractions(self):
        for extraction in self.Extractions:
            if not extraction['IsToExtract']:
                continue
            
            if extraction['Type'] in ['BC', 'Integral']:
                try:
                    # Do that only if the workflow has a method normalize_data_from_extraction
                    self.workflow.normalize_data_from_extraction(extraction['Source'], extraction['Data'])
                except AttributeError:
                    pass

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
        self.mola_logger.info(f'>> finalize', rank=0)
        for extraction in self.Extractions:
            if extraction['ExtractAtEndOfRun']:
                extraction['IsToExtract'] = True
                extraction['IsToSave'] = True
        self.update_extractions_to_perform()
        self.apply_operations()

        self.status = 'COMPLETED'
        move_log_files()
        try:
            call_solver_specific_function(self.workflow, 'move_log_files', 3)
        except MolaException:
            pass
        
        check_stderr()
        write_tagfile(names.FILE_JOB_COMPLETED, self)

    def _update_workflow_parameters_for_restart(self):
        self.workflow.Numerics['NumberOfIterations'] -= self.iteration - self.workflow.Numerics['IterationAtInitialState'] + 1
        self.workflow.Numerics['IterationAtInitialState'] = self.iteration + 1
        if 'TimeStep' in self.workflow.Numerics:
            self.workflow.Numerics['TimeAtInitialState'] = self.iteration * self.workflow.Numerics['TimeStep']

        # Update only Numerics node in tree
        WorkflowParameters = self.workflow.tree.get(Name=self.workflow._workflow_parameters_container_, Depth=1)
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


def move_log_files():
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

    comm.barrier()

    
def check_stderr():
    # TODO Simple check for now, but it should be different if this function is called in the coprocess script
    if rank == 0:
        try:
            with open(names.FILE_STDERR,'r') as f:
                Error = f.read()
            raise Exception(Error)
        except FileNotFoundError:
            pass

def mpi_allgather_and_merge_trees(local_tree : cgns.Tree, comm=comm ) -> cgns.Tree:

    comm.barrier()
    trees = comm.allgather(local_tree)
    merged_tree = cgns.merge(trees)
    comm.barrier() 

    return merged_tree


def update_signals_using( current_iteration_signals : cgns.Tree,
                          previous_signals_to_be_updated : cgns.Tree ) -> None:
    
    previous_tree = previous_signals_to_be_updated
    current_tree = current_iteration_signals

    for current_base in current_tree.bases():
        previous_base = previous_tree.get(Name=current_base.name(), Type='CGNSBase_t', Depth=1)
        if not previous_base:
            current_base.attachTo(previous_tree)
            continue

        for current_zone in current_base.zones():
            previous_zone = previous_base.get(Name=current_zone.name(), Type='Zone_t', Depth=1)
            if not previous_zone:
                current_zone.attachTo(previous_base)
                continue
        
            _update_signals_zones(current_zone, previous_zone)


def _update_signals_zones(current_zone : cgns.Zone, previous_zone : cgns.Zone) -> None:

    previous_flow_sol = previous_zone.get(Name='FlowSolution',Depth=1) # this is modified in-place
    current_flow_sol  =  current_zone.get(Name='FlowSolution',Depth=1)


    PreviousIterationsNode = previous_flow_sol.get(Name='IterationNumber',Type='DataArray_t',Depth=1)
    if not PreviousIterationsNode: return
    PreviousIterations = PreviousIterationsNode.value()
    CurrentIterationsNode = current_flow_sol.get(Name='IterationNumber',Type='DataArray_t',Depth=1)
    if not CurrentIterationsNode: return
    CurrentIterations = CurrentIterationsNode.value()


    override_all = True if CurrentIterations[0] <= PreviousIterations[0] else False
    stack_all = True if CurrentIterations[0] > PreviousIterations[-1] else False
    
    if override_all:
        _update_signals_container_overriding_all(previous_flow_sol, current_flow_sol)

    elif stack_all:
        _update_signals_container_stacking_all(previous_flow_sol, current_flow_sol)

    elif not override_all and not stack_all:
        _update_signals_container_stacking_partially(previous_flow_sol, current_flow_sol)
    
    else:
        raise MolaException(f"unexpected case override_all={override_all} stack_all={stack_all}")
    
    current_zone.updateShape()


def _update_signals_container_overriding_all(previous_flow_sol, current_flow_sol):

    for current_data in current_flow_sol.children():
        if current_data.type() != 'DataArray_t': continue

        previous_data = previous_flow_sol.get(current_data.name(),Type='DataArray_t',Depth=1)
        if not previous_data:
            current_data.attachTo(previous_flow_sol)
            continue

        previous_data.setValue(current_data.value())


def _update_signals_container_stacking_all(previous_flow_sol, current_flow_sol):

    for current_data in current_flow_sol.children():
        if current_data.type() != 'DataArray_t': continue
        
        previous_data = previous_flow_sol.get(current_data.name(),Type='DataArray_t',Depth=1)
        if not previous_data:
            previous_it = previous_flow_sol.get('IterationNumber',Type='DataArray_t',Depth=1).value()
            previous_value = np.empty_like(previous_it)
            previous_value[:] = np.nan
        else:
            previous_value = previous_data.value()
        
        current_value = current_data.value()
        updated_value = np.hstack((previous_value, current_value))

        previous_data.setValue(updated_value)



def _update_signals_container_stacking_partially(previous_flow_sol, current_flow_sol):

    PreviousIterations = previous_flow_sol.get(Name='IterationNumber',Type='DataArray_t',Depth=1).value()
    CurrentIterations = current_flow_sol.get(Name='IterationNumber',Type='DataArray_t',Depth=1).value()

    ε = 1e-12
    UpdatePortion = PreviousIterations > (CurrentIterations[0] - ε)
    if all(np.logical_not(UpdatePortion)) and len(UpdatePortion) == 1:
        FirstPreviousIndex2Update = len(PreviousIterations) - 1 
    else:
        try:
            FirstPreviousIndex2Update = np.where(UpdatePortion)[0][0]
        except IndexError:
            msg = "FATAL: add case to test_update_signals:\n"
            msg+= f'PreviousIterations:\n{PreviousIterations}\n'
            msg+=f'CurrentIterations:\n{CurrentIterations}\n'
            msg+=f'UpdatePortion={UpdatePortion}\n'
            msg+=f'np.where(UpdatePortion)={np.where(UpdatePortion)}'
            raise IndexError(msg)

    for current_data in current_flow_sol.children():
        if current_data.type() != 'DataArray_t': continue

        previous_data = previous_flow_sol.get(current_data.name(),Type='DataArray_t',Depth=1)
        if not previous_data:
            previous_it = previous_flow_sol.get('IterationNumber',Type='DataArray_t',Depth=1).value()
            previous_value = np.empty_like(previous_it)
            previous_value[:] = np.nan
        else:
            previous_value = previous_data.value()
        
        current_value = current_data.value()

        updated_value = np.hstack((previous_value[:FirstPreviousIndex2Update],
                                   current_value))

        previous_data.setValue(updated_value)

def get_bc_families_in_extraction(extraction, DictBCNames2Type):
    families = []
    for BCFamilyName in DictBCNames2Type:
        BCType = DictBCNames2Type[BCFamilyName]
        if fnmatch(BCType, extraction['Source']):
            # Case of source matching one or several names of BC: 'BCWall', 'BCInflow*', '*', etc.
            source = BCType
            family = BCFamilyName
        elif fnmatch(BCFamilyName, extraction['Source']):
            # Case of source matching a family name
            source = BCFamilyName
            family = BCFamilyName
        else:
            continue
        families.append(family)

    return families
    
def write_extraction_log(extraction):
    def _check_data(extraction):
        try: 
            assert isinstance(extraction['Data'], cgns.Tree)
        except KeyError:
            raise MolaException(f"No 'Data' in extraction {extraction}")
        except AssertionError:
            raise MolaException(f"extraction['Data'] must be a cgns.Tree (now its type is {type(extraction['Data'])})")

    _check_data(extraction)
    extraction_log = dict((k,v) for k,v in extraction.items() if k not in ['Data', 'IsToExtract', 'IsToSave'])

    if extraction['Type'] in ['BC', 'IsoSurface']:
        for base in extraction['Data'].bases():
            base.setParameters(names.CGNS_NODE_EXTRACTION_LOG, **extraction_log)

    elif extraction['Type'] in ['Residuals', 'Integral', 'Probe']:
        for zone in extraction['Data'].zones():
            zone.setParameters(names.CGNS_NODE_EXTRACTION_LOG, **extraction_log)
    