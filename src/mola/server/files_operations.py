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

import sys
import os
import subprocess

from mola.logging import mola_logger, MolaException
from . import server as SV

def save_file(filename, text, directory='.'):
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, filename)
    with open(filename, 'w') as f:
        f.write(text)
    os.chmod(filename, 0o777)

def save_file_maybe_remote(filename, txt, directory='.', machine=None):
    if not directory.endswith(os.path.sep):
        directory += os.path.sep

    if SV.run_on_localhost(machine, directory):
        save_file(filename, txt, directory)
        
    else:
        save_file(filename, txt, '.')
        copy_remote(
            source_path=filename, 
            destination_path=directory, 
            destination_machine=machine,
            )

def is_existing_path(path, machine=None, user=None, file_only=False):
    '''
    Check is the given path exists. If the machine (and optionally the user) is provided, 
    then check it on the given remote machine.

    Parameters
    ----------
    path : str
        path to check (file or directory)
    machine : str, optional
    user : str, optional
    file_only : bool
        If :py:obj:`True`, the function returns :py:obj:`False` if **path** targets a directory.

    Returns
    -------
    bool
    '''
    ssh_host = SV.get_ssh_host_command(machine, user, path)

    if file_only:
        option = '-f'
    else:
        option = '-e'

    # mola_target_path = RunManagement['mola_target_path']
    # network = SV.get_network()
    # env = os.path.join(mola_target_path, 'mola', 'env', network, 'env.sh')
    # python_command = f"{sys.executable} -c 'import os; os.path.exits({path})'"
    # source_env = f"source {env}"
    
    try:
        subprocess.run([f'{ssh_host} test {option} {path} || exit 1'], shell=True, check=True)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # precision_if_needed = f' on {machine}' if machine is not None else ''
        # raise MolaException(f'File or directory {path} does not exist{precision_if_needed}.')
        return False

def is_file(path, machine=None, user=None):
    return is_existing_path(path, machine, user, file_only=True)
   
def is_directory(path, machine=None, user=None):
    return is_existing_path(path, machine, user, file_only=False) and not is_existing_path(path, machine, user, file_only=True)

def remove_path(path, machine=None, user=None, file_only=True):

    ssh_host = SV.get_ssh_host_command(machine, user, path)

    if file_only:
        recursive_option = ''
    else:
        recursive_option = 'r'

    try:
        subprocess.run([f'{ssh_host} rm -f{recursive_option} {path} || exit 1'], shell=True, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        precision_if_needed = f' on {machine}' if machine is not None else ''
        raise MolaException(f'Cannot remove {path}{precision_if_needed}.')
    
def makedirs_remote(path, machine=None, user=None):
    ssh_host = SV.get_ssh_host_command(machine, user, path)
    subprocess.run([f'{ssh_host} mkdir -p {path}'], shell=True)

def scp(source_path, destination_path, source_machine=None, destination_machine=None, source_user=None, destination_user=None, force_copy=False, timeout=60):

    if not is_existing_path(source_path, source_machine, source_user) :
        precision_if_needed = f' on {source_machine}' if source_machine is not None else ''
        raise MolaException(f'The source path {source_path} does not exist{precision_if_needed}.')
    
    if (source_machine == destination_machine) and (os.path.realpath(source_path) == os.path.realpath(destination_path)):
        raise MolaException(f'The source path and the destination path are the same ({source_path}).')
    
    raise_error = (
        not force_copy 
        and not destination_path.endswith(os.path.sep)
        and is_existing_path(destination_path, destination_machine, destination_user) 
    )
    if raise_error:
        precision_if_needed = f' on {destination_machine}' if destination_machine is not None else ''
        raise MolaException(
            f'The destination path {destination_path} already exists{precision_if_needed}.'
            ' To force copy and erase previous path, use force_copy=True.'
            )
    
    def get_path_with_machine(path, machine=None, user=None):
        if not SV.run_on_localhost(machine, path):
            if user is None:
                return f'{machine}:{path}'
            else:
                return f'{user}@{machine}:{path}'
        else:
            return path

    source = get_path_with_machine(source_path, source_machine, source_user)
    destination = get_path_with_machine(destination_path, destination_machine, destination_user)

    try:
        subprocess.run([f'scp -r {source} {destination}'], shell=True, check=True, capture_output=True, timeout=timeout)
    except:
        if destination_path.endswith(os.path.sep):
            destination_dir = destination_path
        else:
            destination_dir = os.path.sep.join(destination_path.split(os.path.sep)[:-1])
        makedirs_remote(destination_dir, machine=destination_machine, user=destination_user)
        subprocess.run([f'scp -r {source} {destination}'], shell=True, check=True, capture_output=True, timeout=timeout)

def copy_remote(source_path, destination_path, source_machine=None, destination_machine=None, source_user=None, destination_user=None, force_copy=False):
    '''
    Repatriate a file or directory towards a destination location.

    Parameters
    ----------
    source_path : str
        Path string of the source to be copied.
        May correspond to a directory or a file.
    destination_path : str
        Path string of the destination where the source
        will be copied. If it makes reference to an inexistent directory,
        then all required paths are automatically created in order to
        satisfy the destination path (if permissions allow for it).
    machine : str, optional
        Remote machine corresponding to **source_path**. 
        If not given, try to guess it with :py:func:`guess_machine_from_path`
    user : str, optional
        Useful only if the username is not the same on the remote **machine** that on the local host. 
    force_copy : bool, optional
        If :py:obj:`True`, force the copy and erase the previous **destination_path**.
        By default False.
    '''
    def normalize_path_by_removing_current_dir_at_beginning(path):
        cwd = f'.{os.path.sep}'
        if path.startswith(cwd): 
            path = path[len(cwd):]
        return path

    source_path = normalize_path_by_removing_current_dir_at_beginning(source_path)
    destination_path = normalize_path_by_removing_current_dir_at_beginning(destination_path)

    if source_machine is None:
        try:
            source_machine = SV.guess_machine_from_path(source_path) 
        except:
            pass
    
    if destination_machine is None:
        try:
            destination_machine = SV.guess_machine_from_path(destination_path) 
        except:
            pass

    scp(source_path, destination_path, 
               source_machine, destination_machine, 
               source_user, destination_user, 
               force_copy=force_copy
               )
