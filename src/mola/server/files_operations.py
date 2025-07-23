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

import shutil
import os
import subprocess
from pathlib import Path

import mola.naming_conventions as names
from mola.logging import mola_logger, MolaException
from . import remote

def read_text_file_from_errors(filepath, machine=None, user=None, max_lines=1000,
        start_keywords=['error','traceback','abort','signal: segmentation fault'],
        skip_keywords=['userwarning','warnings.warn']):

    separator_line = 'SCANNED_ERRORS\n'
    if remote.run_on_localhost(machine=machine, run_directory=filepath):
        # implementation of the function
        found_info = False
        result_lines = [separator_line]
        with open(filepath, 'r') as f:
            for line in f:
                if found_info or any([word in line.lower() for word in start_keywords]):
                    if any([word in line.lower() for word in skip_keywords]):
                        continue
                    found_info = True
                    result_lines += [line]
                    if len(result_lines) > max_lines: break
        if found_info: return ''.join(result_lines)

    else:
        # remote call to the function
        pycode = [f"import mola.server.files_operations as FOP"]
        pycode+= [f"print(FOP.read_text_file_from_errors('{filepath}'))"]
        pycode = ';'.join(pycode)
        out = remote.submit_command(f'python3 -c "{pycode}"', machine, user=user,
                                    use_mola_env=True)
        
        if separator_line in out: return separator_line+out.split(separator_line)[-1]
        return ''
    
    
def read_last_run_error_file_in_log_directory(run_directory, 
                                          **read_text_file_from_errors_params):

    def get_stderr_filename(i):
        fname = names.FILE_STDERR.replace('.log','-%d.log'%i)
        return os.path.join(run_directory, names.DIRECTORY_LOG, fname)

    i=1
    any_file = False
    filepath = get_stderr_filename(i)
    while True:
        if is_file( filepath ):
            any_file = True
        i += 1
        next_filepath = get_stderr_filename(i)
        if is_file( next_filepath ):            
            filepath = next_filepath
        else:
            break

    if any_file: 
        return read_text_file_from_errors(filepath, **read_text_file_from_errors_params)

    else:
        return ''


def save_file(filename, text, directory='.'):
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, filename)
    with open(filename, 'w') as f:
        f.write(text)
    os.chmod(filename, 0o777)

def save_file_maybe_remote(filename, txt, directory='.', machine=None, force_copy=False):
    if not directory.endswith(os.path.sep):
        directory += os.path.sep

    if remote.run_on_localhost(machine, directory):
        save_file(filename, txt, directory)
        
    else:
        save_file(filename, txt, '.')
        move_remote(
            source_path=filename, 
            source_machine='localhost',
            destination_path=os.path.join(directory, filename), 
            destination_machine=machine,
            force_copy=force_copy
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
    ssh_host = remote.get_ssh_host_command(machine, user, path)

    if file_only:
        option = '-f'
    else:
        option = '-e'

    # mola_target_path = RunManagement['mola_target_path']
    # network = remote.get_network()
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

    ssh_host = remote.get_ssh_host_command(machine, user, path)

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
    ssh_host = remote.get_ssh_host_command(machine, user, path)
    subprocess.run([f'{ssh_host} mkdir -p {path}'], shell=True)

def get_path_with_machine(path, machine=None, user=None):
    if not remote.run_on_localhost(machine, path):
        if user is None:
            return f'{machine}:{path}'
        else:
            return f'{user}@{machine}:{path}'
    else:
        return path

def scp(source_path, destination_path, source_machine=None, destination_machine=None, source_user=None, destination_user=None, force_copy=False, timeout=60):

    # Force convertion to str in case paths are Path objects from pathlib
    source_path = str(source_path)
    destination_path = str(destination_path)

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

    source = get_path_with_machine(source_path, source_machine, source_user)
    destination = get_path_with_machine(destination_path, destination_machine, destination_user)

    if is_local_copy(source_path, source_machine, destination_path, destination_machine):
        safe_local_copy(source, destination, force_copy)

    else:
        try:
            subprocess.run(['scp', '-r', source,destination], check=True, capture_output=True, timeout=timeout)

        except:
            if destination_path.endswith(os.path.sep):
                destination_dir = destination_path
            else:
                destination_dir = os.path.sep.join(destination_path.split(os.path.sep)[:-1])
            makedirs_remote(destination_dir, machine=destination_machine, user=destination_user)
            subprocess.run(['scp', '-r', source,destination], check=True, capture_output=True, timeout=timeout)

def rsync(source_path, destination_path, source_machine=None, destination_machine=None, 
          source_user=None, destination_user=None, included_files=None, excluded_files=None):
    '''
    Synchronize local and remote directories using rsync

    Parameters
    ----------
    included_files : list, optional
        If not None, the list of files to include
    excluded_files : list, optional
       If not None, the list of files to exclude. 
       By default, if an **included_file** is given, all other files are by default not included.
    '''
    source = get_path_with_machine(source_path, source_machine, source_user)
    destination = get_path_with_machine(destination_path, destination_machine, destination_user)

    if included_files is None:
        included_files = []
    else:
        if excluded_files is None:
            excluded_files = ['*']
    if excluded_files is None:
        excluded_files = []

    rsync_cmd = ['rsync', '-avz']  # Options: archive mode, verbose, compression

    # Adding --include and --exclude options
    for pattern in included_files:
        rsync_cmd.append("--include=" + pattern)
    for pattern in excluded_files:
        rsync_cmd.append("--exclude=" + pattern)

    # Adding source and destination
    rsync_cmd.append(source)
    rsync_cmd.append(destination)

    # Executing the rsync command with subprocess
    mola_logger.debug(f"Executing command: {' '.join(rsync_cmd)}")
    subprocess.run(rsync_cmd, check=True)


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
        path = str(path)  # if path is a pathlib.Path
        cwd = f'.{os.path.sep}'
        if path.startswith(cwd): 
            path = path[len(cwd):]
        return path

    source_path = normalize_path_by_removing_current_dir_at_beginning(source_path)
    destination_path = normalize_path_by_removing_current_dir_at_beginning(destination_path)

    if source_machine is None:
        try:
            source_machine = remote.guess_machine_from_path(source_path) 
        except:
            pass
    
    if destination_machine is None:
        try:
            destination_machine = remote.guess_machine_from_path(destination_path) 
        except:
            pass

    scp(source_path, destination_path, 
               source_machine, destination_machine, 
               source_user, destination_user, 
               force_copy=force_copy
               )

def move_remote(source_path, destination_path, source_machine=None, destination_machine=None, source_user=None, destination_user=None, force_copy=False, file_only=True):
    copy_remote(source_path, destination_path, source_machine, destination_machine, source_user, destination_user, force_copy)
    if is_existing_path(destination_path, machine=destination_machine, user=destination_user, file_only=file_only):
        remove_path(source_path, machine=source_machine, user=source_user, file_only=file_only)
    else:
        raise MolaException(f"Copy {source_path} to {destination_path} failed")

def is_local_copy(source_path, source_machine,
                  destination_path, destination_machine):
    is_source_local = remote.run_on_localhost(source_machine, source_path)
    is_destination_local = remote.run_on_localhost(destination_machine, destination_path)
    return is_source_local and is_destination_local
    
def safe_local_copy(source_path, destination_path, force_copy=False):

    src = Path(source_path)
    dst = Path(destination_path)

    if not src.exists():
        raise FileNotFoundError(f"Source path {src} does not exist.")

    if src.is_dir():
        if dst.exists():
            if dst.is_file():
                raise ValueError(f"Cannot copy directory {src} to existing file {dst}.")
            if force_copy:
                shutil.rmtree(dst)
            else:
                raise FileExistsError(f"Destination {dst} already exists. Use force_copy=True to overwrite.")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src, dst)
    else:
        # If dst is an existing dir, copy inside it like scp
        if dst.exists() and dst.is_dir():
            dst = dst / src.name
        elif str(destination_path).endswith(os.path.sep):  # e.g. 'dir/'
            dst.mkdir(parents=True, exist_ok=True)
            dst = dst / src.name
        else:
            dst.parent.mkdir(parents=True, exist_ok=True)

        if dst.exists():
            if force_copy:
                dst.unlink()
            else:
                raise FileExistsError(f"Destination {dst} already exists. Use force_copy=True to overwrite.")

        shutil.copy2(src, dst)


def get_module_name_from(module_path : str):
    return module_path.split(os.sep)[-1]

def trim(module_path : str, levels_from_end : int = 3):
    if levels_from_end < 0: raise AttributeError("levels_from_end cannot be <0")
    path_split = module_path.split(os.sep)
    path_to_keep = path_split[:-levels_from_end]
    trimmed_path = os.sep.join(path_to_keep)
    return trimmed_path

def remove_mola_path_from(module_path : str):
    mola_path = os.environ["MOLA"]
    if module_path.startswith(mola_path):
        return module_path.replace(mola_path,'')
    return module_path