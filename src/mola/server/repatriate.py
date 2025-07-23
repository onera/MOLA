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

import mola.naming_conventions as names
from mola.logging import MolaException
from .files_operations import rsync

def extract_variables_from_file(file_path):
    """
    Reads a file and extracts variables as key-value pairs.
    
    :param file_path: Path to the file to be read
    :return: Dictionary containing the extracted variables
    """
    variables = {}
    
    try:
        with open(file_path, 'r') as file:
            for line in file:
                # Ignore empty lines and those starting with '#'
                if line.strip() and not line.startswith('#'):
                    # Split the line into key and value
                    key_value = line.split('=')
                    if len(key_value) == 2:
                        key, value = [item.strip() for item in key_value]
                        if value == 'None': 
                            value = None
                        variables[key] = value

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
    
    return variables

def get_remote_info():
    remote_info = extract_variables_from_file(names.FILE_LOG_REMOTE_DIRECTORY)

    run_directory = remote_info['RunDirectory']
    machine = remote_info['Machine']
    user = remote_info['User']

    # check that run_directory is not empty (else all the root /* will be copy !)
    assert run_directory != ''
    # It is important that run_directory ends with '/', 
    # otherwise rsync won't work as intended
    if not run_directory.endswith(os.path.sep):
        run_directory += os.path.sep

    return run_directory, machine, user

def get_workflow_manager_info():
    from mola.workflow import WorkflowManager

    manager = WorkflowManager(names.FILE_WORKFLOW_MANAGER)

    run_directories_tmp = manager.run_directories
    root_directory = manager.root_directory

    machine = manager.machine
    try:    
        user = manager.user
    except:
        user = None

    run_directories = []
    for run_directory in run_directories_tmp:
        # check that run_directory is not empty (else all the root /* will be copy !)
        assert run_directory != ''
        # It is important that run_directory ends with '/', 
        # otherwise rsync won't work as intended
        if not run_directory.endswith(os.path.sep):
            run_directory += os.path.sep
        run_directories.append(run_directory)

    return root_directory, run_directories, machine, user
    
    


def get_one_case(included_files: set=set(), excluded_files: set=set()):
    assert isinstance(included_files, set)
    assert isinstance(excluded_files, set)

    run_directory, machine, user = get_remote_info()

    rsync(
        source_path=run_directory, source_machine=machine, source_user=user, 
        destination_path='./', destination_machine='localhost',
        included_files=included_files, excluded_files=excluded_files
        )
    
def get_all_cases_from_workflow_manager(included_files: set=set(), excluded_files: set=set()):
    assert isinstance(included_files, set)
    assert isinstance(excluded_files, set)

    root_directory, run_directories, machine, user = get_workflow_manager_info()

    for run_directory in run_directories:
        local_dir = os.path.relpath(run_directory, root_directory)
        os.makedirs(local_dir, exist_ok=True)
        rsync(
            source_path=run_directory, source_machine=machine, source_user=user, 
            destination_path=local_dir, destination_machine='localhost',
            included_files=included_files, excluded_files=excluded_files
            )
