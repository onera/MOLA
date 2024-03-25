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

        echo "preprocess case $case at $SECONDS s"
        python preprocess.py $SECONDS

        if [ -f "FAILED" ]; then
            echo "WARNING: case $case cannot be launched"
            break
        fi

        echo "compute case $case at $SECONDS s"
        {launch_command}
        python postprocess.py

        if [ -f "NEWJOB_REQUIRED" ]; then
            rm NEWJOB_REQUIRED
            echo "LAUNCHING THIS JOB AGAIN"
            cd ..
            sbatch job_sequence.sh --dependency=singleton
            exit 0
        fi
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
  

class WorkflowDispatcher():

    def __init__(self, workflow):

        self.base_workflow = workflow
        self.workflows = [self.base_workflow]
    
    def add_variations(self, variations):
        new_workflow = copy.deepcopy(self.base_workflow)

        for request, value in variations:
            path_in_workflow = self.request_to_paths(request)
            set_value_on_leaf(new_workflow, path_in_workflow, value)

        self.workflows.append(new_workflow)

    def reorder(self, request, reverse=False):
        path_in_workflow = self.request_to_paths(request)
        leaves = [get_value_on_leaf(workflow, path_in_workflow) for workflow in self.workflows]
        # Sort workflows accordind leaves
        self.workflows = [w for _, w in sorted(zip(leaves, self.workflows))]
        if reverse:
            self.workflows = self.workflows[::-1]
    
    @staticmethod
    def request_to_paths(request):
        if isinstance(request, str):
            request = request.split('|')
        return request

    def get_directories(self):
        return [workflow.RunManagement['RunDirectory'] for workflow in self.workflows]


class WorkflowParallelScheduler():

    def __init__(self, table_of_workflows, root_directory, sequences_directories):

        # table_of_workflows = [[A1, B1, ...], [A2, B2, ...], ...]
        #   several jobs in parallel: 
        #     sequence of [A1, B1, ...]
        #     sequence of [A2, B2, ...]
        #     ...
        #
        # Warning: if A1 == A2 (same Python object, without copy), then it will bug without raising an error

        self.root_directory = root_directory
        self.sequences_directories = sequences_directories
        self._set_table_of_workflows(table_of_workflows)

    def _set_table_of_workflows(self, table_of_workflows):
        assert isinstance(table_of_workflows, list)
        self.table_of_workflows = []
        for i, sequence in enumerate(table_of_workflows):
            root_directory = os.path.join(self.root_directory, self.sequences_directories[i])
            sequential_scheduler = WorkflowSequentialScheduler(sequence, root_directory)
            self.table_of_workflows.append(sequential_scheduler)

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
            # with redirect_streams_to_logger(mola_logger, stdout_level='WARNING'):
            workflow.prepare()
            workflow.write_cfd_files()
        self.write_sequence_job()
    
    def write_sequence_job(self):
        first_workflow = self.workflows[0]      
        job_text = write_cfd_files.get_job_text(first_workflow.RunManagement, first_workflow.Solver)

        paths_in_bash = '"{}"'.format(' '.join(self.cases_local_paths))
        loop_on_cases = build_loop_on_cases(paths_in_bash, first_workflow.Solver)
        job_text += loop_on_cases
        
        write_cfd_files.save_file('job_sequence.sh', job_text, self.root_directory)

    def submit(self):
        print(f'{CYAN}  > fake submission of job sequence in {self.root_directory}')

