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
from typing import List
import copy
from fnmatch import fnmatch
import shutil

from mola import __MOLA_PATH__
from mola.logging import mola_logger, MolaAssertionError, MolaException, CYAN, ENDC
from mola.workflow import Workflow
from mola.cfd.preprocess.write_cfd_files import write_cfd_files

def build_loop_on_cases(sequence_of_paths, solver):

    if solver == 'elsa':
        launch_command = 'mpirun -np $SLURM_NTASKS elsA.x -C xdt-runtime-tree -- compute.py 1>stdout.log 2>stderr.log'
    else:
        raise MolaException('Not yet implemented')
    
    loop_on_cases = f'''
SECONDS=0
SEQUENCE_OF_PATHS={sequence_of_paths}

for case in $SEQUENCE_OF_PATHS; do

    echo "entering $case"
    cd $case

    while [ ! -f "COMPLETED" ] && [ ! -f "FAILED" ]; do

        mola_prepare workflow.cgns

        echo "compute case $case at $SECONDS s"
        # {launch_command}
        # # python postprocess.py

        # if [ -f "NEWJOB_REQUIRED" ]; then
        #     rm NEWJOB_REQUIRED
        #     echo "LAUNCHING THIS JOB AGAIN"
        #     cd ..
        #     sbatch job_sequence.sh --dependency=singleton
        #     exit 0
        # fi
    done

    cd ..

done
'''
    return loop_on_cases

def set_value_on_leaf(node, path, value):
    current_path = path[0]

    if isinstance(node, Workflow):
        assert len(path) > 1
        attr = getattr(node, current_path)
        set_value_on_leaf(attr, path[1:], value)

    elif isinstance(node, dict):
        if len(path) == 1:
            node[current_path] = value
        else:
            set_value_on_leaf(node[current_path], path[1:], value)
    
    elif isinstance(node, (list, tuple)):
        assert len(path) > 1
        assert current_path.count('=') == 1
        # search the good element 
        name, search_value = current_path.split('=')
        found = False
        for dico in node:
            if name in dico and str(dico[name]) == search_value:
                found = True
                break
        assert found
        set_value_on_leaf(dico, path[1:], value)
    
    else:
        raise MolaException(f'Unknown type: {type(node)}')
          
def get_value_on_leaf(node, path):
    try:
        current_path = path[0]
    except IndexError:
        return node
    
    if isinstance(node, Workflow):
        return get_value_on_leaf(getattr(node, current_path), path[1:])

    elif isinstance(node, dict):
        return get_value_on_leaf(node[current_path], path[1:])
    
    elif isinstance(node, (list, tuple)):
        assert current_path.count('=') == 1
        # search the good element 
        name, search_value = current_path.split('=')
        found = False
        for dico in node:
            if name in dico and str(dico[name]) == search_value:
                found = True
                break
        assert found
        return get_value_on_leaf(dico, path[1:])
    
    else:
        raise MolaException(f'Unknown type: {type(node)}')

def get_values_in_collection_from_key(collec, patterns, elements=[]):
    '''
    In the nested collection **collec** (may be a dictionary or a list),
    find all the values corresponding to the key **searchKey**.

    Parameters
    ----------

        collec : :py:class:`dict` or :py:class:`list`
            Nested dictionary or list where **searchKey** is searched

        searchKey : str
            Key to find in **collec**

        elements : list
            accumulated list of found values. Works as an accumulator in the
            recursive function

    Returns
    -------

        elements : list
            list of the found values correspondingto **searchKey**
    '''
    if not isinstance(patterns, list): 
        patterns = [patterns]
    
    if isinstance(collec, Workflow):
        collec = collec.convert_to_dict()
    
    if isinstance(collec, dict):
        for key, value in collec.items():
            if isinstance(value, (dict, list)):
                get_values_in_collection_from_key(value, patterns, elements=elements)
            elif any([match_pattern(key, pattern) for pattern in patterns]):
                elements.append(value)

    elif isinstance(collec, list):
        for elem in collec:
            get_values_in_collection_from_key(elem, patterns, elements=elements)

    return elements

def get_values_in_collection_from_pattern(collec, patterns, elements=[]):
    if not isinstance(patterns, list): 
        patterns = [patterns]
    
    # if isinstance(collec, Workflow):
    #     collec = collec.convert_to_dict()
    if isinstance(collec, Workflow):
        for attr_name in collec.__dict__:
            attr = getattr(collec, attr_name)
            if isinstance(attr, (dict, list)):
                get_values_in_collection_from_pattern(attr, patterns, elements=elements)

    elif isinstance(collec, dict):
        for key, value in collec.items():
            if isinstance(value, (dict, list)):
                get_values_in_collection_from_pattern(value, patterns, elements=elements)
            elif any([match_pattern(value, pattern) for pattern in patterns]):
                elements.append(value)

    elif isinstance(collec, list):
        for elem in collec:
            get_values_in_collection_from_pattern(elem, patterns, elements=elements)

    return elements

def match_pattern(value, pattern):
    if isinstance(pattern, str):
        return fnmatch(str(value), pattern) 
    elif isinstance(pattern, (int, float)):
        return value == pattern
    else:
        raise TypeError


class WorkflowDispatcher():

    def __init__(self, workflow):
        self.base_workflow = workflow
        self.table_of_workflows = []
        self.workflows_in_current_job = None
        self.root_directories = []
    
    def new_job(self, directory):
        self.workflows_in_current_job = []
        self.table_of_workflows.append(self.workflows_in_current_job)
        self.root_directories.append(directory)
    
    def add_variations(self, variations, initialize_from_previous=True):
        self._check_new_job_is_declared()
        new_workflow = copy.deepcopy(self.base_workflow)

        if initialize_from_previous:
            self._add_variations_to_initialize_from_previous(variations)

        for request, value in variations:
            path_in_workflow = self.request_to_paths(request)
            set_value_on_leaf(new_workflow, path_in_workflow, value)

        self.workflows_in_current_job.append(new_workflow)

    # def reorder(self, request, reverse=False):
    #     path_in_workflow = self.request_to_paths(request)
    #     leaves = [get_value_on_leaf(workflow, path_in_workflow) for workflow in self.workflows]
    #     # Sort workflows accordind leaves
    #     self.workflows = [w for _, w in sorted(zip(leaves, self.workflows))]
    #     if reverse:
    #         self.workflows = self.workflows[::-1]
    
    def _check_new_job_is_declared(self):
        if self.workflows_in_current_job is None:
            raise MolaAssertionError('Before calling `add_variations`, `new_job` must be called first to declare directory.')
    
    def _add_variations_to_initialize_from_previous(self, variations):
        try:
            previous_workflow = self.workflows_in_current_job[-1]
            previous_case_path = previous_workflow.RunManagement['RunDirectory']
            init_variations = [
                ('Initialization|method', 'copy'),
                ('Initialization|filename', f'../{previous_case_path}/OUTPUT/fields.cgns'),
            ]
            variations += init_variations
        except IndexError:
            mola_logger.warning(f'Cannot initialize the first case of a sequence ({self.root_directories[-1]}) from a previous case')

    @staticmethod
    def request_to_paths(request):
        if isinstance(request, str):
            request = request.split('|')
        return request

    def get_directories_in_current_job(self, worflows_sequence=None):
        if worflows_sequence is None:
            worflows_sequence = self.workflows_in_current_job
        return [workflow.RunManagement['RunDirectory'] for workflow in worflows_sequence]

    def get_local_directories(self):
        directories = []
        for worflows_sequence in self.table_of_workflows:
            job_directories = self.get_directories_in_current_job(worflows_sequence)
            directories.append(job_directories)
        return directories

    def get_directories(self):
        directories = []
        for root, worflows_sequence in zip(self.root_directories, self.table_of_workflows):
            job_directories = self.get_directories_in_current_job(worflows_sequence)
            directories.extend([os.path.join(root, d) for d in job_directories])
        return directories


class WorkflowParallelScheduler():

    def __init__(self, dispatcher, root_directory='.', data_directory='SHARED_DATA'):

        # table_of_workflows = [[A1, B1, ...], [A2, B2, ...], ...]
        #   several jobs in parallel: 
        #     sequence of [A1, B1, ...]
        #     sequence of [A2, B2, ...]
        #     ...
        #
        # Warning: if A1 == A2 (same Python object, without copy), then it will bug without raising an error

        self.root_directory = root_directory
        self.data_directory = data_directory
        self.sequences_directories = dispatcher.root_directories
        self._set_table_of_workflows(dispatcher.table_of_workflows)
        self._update_filenames()

    def _set_table_of_workflows(self, table_of_workflows):
        assert isinstance(table_of_workflows, list)
        self.table_of_workflows = []
        for i, sequence in enumerate(table_of_workflows):
            root_directory = os.path.join(self.root_directory, self.sequences_directories[i])
            sequential_scheduler = WorkflowSequentialScheduler(sequence, root_directory)
            self.table_of_workflows.append(sequential_scheduler)
        
    def _update_filenames(self):
        # need to grab all workflow parameters that are filenames, and update their paths 
        # following the attributes of the Scheduler
        workflows_list_flatten = [w for scheduler in self.table_of_workflows for w in scheduler.workflows] 
        for workflow in workflows_list_flatten:
            # files2copy = get_values_in_collection_from_pattern(workflow, ['*.cgns'], [])

            # TODO for now, the solution is working but it paths in the workflows are hardcoded.
            # It would be better to have a function to do:
            # for path in paths:
            #     leaf = get_leaf(workflow, path)
            #     adapt(leaf)
            #
            # path = ['RawMeshComponents', 'Source']
            # leaf = get_value_on_leaf(workflow, path)
            # if leaf.endswith('.cgns'):
            #     self.copy_file_to_data_directory(leaf)
            #     set_value_on_leaf(workflow, path, self.get_adapted_path(leaf))

            for Component in workflow.RawMeshComponents:
                try:
                    if Component['Source'].endswith('.cgns'):
                        self.copy_file_to_data_directory(Component['Source'])
                        Component['Source'] = self.get_adapted_path(Component['Source'])
                except AttributeError:
                    pass

    def copy_file_to_data_directory(self, path):
        filename = path.split(os.path.sep)[-1]
        data_dir = os.path.join(self.root_directory, self.data_directory)
        new_filename = os.path.join(data_dir, filename)
        if not os.path.exists(new_filename):
            os.makedirs(data_dir, exist_ok=True)
            mola_logger.info(f'copy {path} to {data_dir}')
            shutil.copy2(path, new_filename)
    
    def get_adapted_path(self, path):
        filename = path.split(os.path.sep)[-1]
        return os.path.join('..', '..', self.data_directory, filename)

    def prepare(self):
        os.makedirs(self.root_directory, exist_ok=True)
        for sequence_of_workflows in self.table_of_workflows:
            sequence_of_workflows.prepare()

    def submit(self):
        for sequence_of_workflows in self.table_of_workflows:
            sequence_of_workflows.submit()

    
class WorkflowSequentialScheduler():

    def __init__(self, workflows: List[Workflow], root_directory):

        self.root_directory = root_directory
        self.workflows = workflows
        self._check_structure_of_workflows()
        self._check_and_set_local_paths()
        self._check_all_workflows_have_different_working_directories()

    def _check_structure_of_workflows(self):
        assert isinstance(self.workflows, list)
        for workflow in self.workflows:
            assert isinstance(workflow, Workflow)

    def _check_all_workflows_have_different_working_directories(self):
        assert len(set(self.cases_local_paths)) == len(self.workflows), 'Several workflows have the same working directory'

    def _check_and_set_local_paths(self):
        self.cases_local_paths = []
        for workflow in self.workflows:
            path = workflow.RunManagement['RunDirectory']
            if not path.startswith(os.path.sep):
                # local path
                self.cases_local_paths.append(path)
                workflow.RunManagement['RunDirectory'] = os.path.join(self.root_directory, path)
            else:
                # absolute path
                if not path.startswith(self.root_directory):
                    raise MolaAssertionError(
                        f'RunDirectory {path} is not consistent with the root path {self.root_directory}.'
                        )
                self.cases_local_paths.append(path.replace(self.root_directory, ''))

    def prepare(self):
        os.makedirs(self.root_directory, exist_ok=True)
        for workflow in self.workflows:
            mola_logger.info(f"\n{CYAN}  > preparing {workflow.RunManagement['RunDirectory']}...{ENDC}")
            os.makedirs(workflow.RunManagement['RunDirectory'], exist_ok=True)
            self.write_workflow_without_prepare(workflow)

        self.write_sequence_job()
    
    @staticmethod
    def prepare_workflow(workflow):
        workflow.prepare()
        workflow.write_cfd_files()

    @staticmethod
    def write_workflow_without_prepare(workflow):
        RunDirectory = copy.deepcopy(workflow.RunManagement['RunDirectory'])
        workflow.RunManagement['RunDirectory'] = '.'
        workflow.set_workflow_parameters_in_tree()
        workflow.write_tree(filename=os.path.join(RunDirectory, 'workflow.cgns'))
    
    def write_sequence_job(self):
        first_workflow = self.workflows[0]      
        write_cfd_files.set_default(first_workflow.RunManagement)
        job_text = write_cfd_files.get_job_text(first_workflow.RunManagement, first_workflow.Solver)

        paths_in_bash = '"{}"'.format(' '.join(self.cases_local_paths))
        loop_on_cases = build_loop_on_cases(paths_in_bash, first_workflow.Solver)
        job_text += loop_on_cases
        
        write_cfd_files.save_file('job_sequence.sh', job_text, self.root_directory)

    def submit(self):
        print(f'{CYAN}  > fake submission of job sequence in {self.root_directory}')
        # os.system(f'bash {self.root_directory}/job_sequence.sh')

