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

'''
MOLA - _cpmv_.py

AUXILIARY COPY/MOVE MODULE

Python wrapper of convenient copy and move operations for
files and directories, including between localhost and a remote server.

May be used in MODULE MODE or in TERMINAL MODE.


Example: Move an entire directory from sator to spiro.

-------------- Example of usage in MODULE MODE --------------
python3
>>> import MOLA._cpmv_ as cpmv
>>> cpmv.cpmvWrap4MultiServer('mv',
'/tmp_user/sator/username/sandbox/',
'/scratchm/username/sandbox/')


------------- Example of usage in TERMINAL MODE -------------
# REMEMBER: environment variables must be properly set
alias cpmv='python3 $MOLA/_cpmv_.py'

# Then in terminal one may tape this:
cpmv mv /tmp_user/sator/username/sandbox/ /scratchm/username/sandbox/

-------------------------- IMPORTANT --------------------------
This module must import standard python3 libraries only !!!
Otherwise, calling this module in TERMINAL MODE will produce
an error. Remember that even a usage as a MODULE will lead to
a usage in TERMINAL MODE because of the function cpmvWrap4MultiServer
---------------------------------------------------------------

First creation:
28/07/2020 - L. Bernardos - creation
'''

import sys
import os
import shutil
import getpass
import subprocess
from distutils.dir_util import copy_tree
import time
import timeit

from mola.logging import mola_logger, MolaException
from mola import __MOLA_PATH__


# def move(In, Out):
#     if In == Out or not os.path.exists(In): 
#         return

#     if not os.path.isdir(Out):
#         os.makedirs(os.path.dirname(Out), exist_ok=True)
#         shutil.move(In,Out)

# def copy(In, Out):
#     if In == Out or not os.path.exists(In): 
#         return
    
#     if os.path.isdir(In):
#         copy_tree(In,Out)
#     else:
#         os.makedirs(os.path.dirname(Out), exist_ok=True)
#         shutil.copy2(In, Out, follow_symlinks=True)

# def remove(file_or_directory):
#     if not os.path.exists(file_or_directory): 
#         return
    
#     if os.path.isdir(file_or_directory):
#         for PathName in os.listdir(file_or_directory):
#             file_path = os.path.join(file_or_directory, PathName)
#             try:
#                 if os.path.isfile(file_path) or os.path.islink(file_path):
#                     os.unlink(file_path)
#                 elif os.path.isdir(file_path):
#                     shutil.rmtree(file_path)
#             except Exception as e:
#                 print('Failed to delete file %s. Reason: %s' % (file_path, e))

#         try:
#             shutil.rmtree(file_or_directory)
#         except Exception as e:
#             print('Failed to delete directory %s. Reason: %s' % (file_or_directory, e))

#     elif os.path.isfile(file_or_directory) or os.path.islink(file_or_directory):
#         try:
#             os.unlink(file_or_directory)
#         except Exception as e:
#             print('FAILED in deleting %s. Error: %s'%(file_or_directory, e))

# def move_remote(In, Out, server, user=None):
#     if In == Out: 
#         return
#     if user is None:
#         user = getpass.getuser()
#     Host = f'{user}@{server}'
#     CMD = f'python3 $MOLA/mola/server/_cpmv_.py mv {In} {Out}'
#     _launchSubprocess(Host, CMD)

# def copy_remote(In, Out, server, user=None):
#     if In == Out: 
#         return
#     if user is None:
#         user = getpass.getuser()
#     Host = f'{user}@{server}'
#     CMD = f'python3 $MOLA/mola/server/_cpmv_.py cp {In} {Out}'
#     _launchSubprocess(Host, CMD)

# def remove_remote(In, server, user=None):
#     if user is None:
#         user = getpass.getuser()
#     Host = f'{user}@{server}'
#     CMD = f'python3 $MOLA/mola/server/_cpmv_.py rm {In}'
#     _launchSubprocess(Host, CMD)

# def _launchSubprocess(Host, CMD):
#     '''
#     Wrapper for launching a subprocess in a remote server.

#     INPUTS

#     Host - (string) - hostname where subprocess will be submitted. For example,
#         'username@sator' or 'spiro-daaa'

#     CMD - (string) - command to submit to server. For multiple lines, use the
#         following syntax: '"command1; command2; command3"'
#     '''
#     ssh = subprocess.Popen(
#         'ssh %s %s'%(Host,CMD),
#         shell=True,
#         stdout=subprocess.PIPE,
#         stderr=subprocess.PIPE,
#         env=os.environ.copy(),
#         )
#     ssh.wait()
#     Error = readStderr(ssh)
#     Output = readStdout(ssh)
#     if len(Output)>0:
#         mola_logger.info('\n'.join(Output))
#     if len(Error)>0:
#         for e in Error:
#             WillRaise = False
#             if 'warning:' in e:
#                 if not 'bind:' in e:
#                     mola_logger.warning(str(e))
#             else:
#                 WillRaise = True
#                 mola_logger.error(str(e))

#         if WillRaise:
#             raise MolaException(f'Host: {Host}\nCMD={CMD}\nerror message:\n' + '\n'.join(Error))

# def readStderr(ssh):
#     '''
#     Read the standard error from the object **ssh* obtained from
#     subprocess.Popen

#     Parameters
#     ----------

#         ssh : object
#             returned by subprocess.Popen

#     Returns
#     -------

#         Error : :py:class:`list` of :py:class:`str`
#             error lines

#     See also
#     --------
#     readStdout
#     '''
#     Error = ssh.stderr.readlines()
#     for i, e in enumerate(Error):
#         if isinstance(e, bytes):
#             Error[i] = e.decode('utf-8')
#     return Error

# def readStdout(ssh):
#     '''
#     Read the standard output from the object **ssh* obtained from
#     subprocess.Popen

#     Parameters
#     ----------

#         ssh : object
#             returned by subprocess.Popen

#     Returns
#     -------

#         Output : :py:class:`list` of :py:class:`str`
#             output lines

#     See also
#     --------
#     readStderr
#     '''
#     Output = ssh.stdout.readlines()
#     for i, o in enumerate(Output):
#         if isinstance(o, bytes):
#             Output[i] = o.decode('utf-8')
#     return Output

# def wait_for_server(filename, request_interval=0.5, timeout=60.):
#     '''
#     This function is employed to determine if a file exist on a given path.
#     The algorithm will check for file existence every <request_interval> seconds,
#     up to a limit of <timeout> seconds.
#     As soon as file is detected, the function returns True. Otherwise, if the
#     timeout is reached, the function returns False and raises a warning.

#     INPUTS

#     filename - (string) - full path of the element to check existance

#     request_interval - (float) - seconds to wait between two consecutive checks

#     timeout - (float) - maximum total waiting time, after which the function
#         will return False.

#     OUTPUTS

#     pathfound - (boolean) - True if the path is found before timeout. False
#         otherwise
#     '''
#     tic = timeit.default_timer()
#     ElapsedTime = 0.
#     while ElapsedTime < timeout:
#         time.sleep(request_interval)
#         ElapsedTime = timeit.default_timer() - tic
#         if os.path.exists(filename): return True
#     if ElapsedTime >= timeout:
#         mola_logger.warning('timeout reached')
#         return False


def is_path_exists(path, machine=None, user=None, file_only=False):
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

    if machine is None:
        ssh_host = ''
    else:
        if user is None:
            ssh_host = f'ssh {machine}'
        else:
            ssh_host = f'ssh {user}@{machine}'

    if file_only:
        option = '-f'
    else:
        option = '-e'
    
    try:
        subprocess.run([f'{ssh_host} test {option} {path} || exit 1'], shell=True, check=True)
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
        # precision_if_needed = f' on {machine}' if machine is not None else ''
        # raise MolaException(f'File or directory {path} does not exist{precision_if_needed}.')
        return False

def is_file(path, machine=None, user=None):
    return is_path_exists(path, machine, user, file_only=True)
   
def is_directory(path, machine=None, user=None):
    return is_path_exists(path, machine, user, file_only=False) and not is_path_exists(path, machine, user, file_only=True)
    
    
def makedirs_remote(path, machine=None, user=None):
    if machine is not None:
        if user is not None:
            machine = f'{user}@{machine}'
        print(f'ssh {machine} mkdir -p {path}')
        subprocess.run([f'ssh {machine} mkdir -p {path}'], shell=True)
    else:
        print(f'mkdir -p {path}')
        subprocess.run([f'mkdir -p {path}'], shell=True)

def scp(source_path, destination_path, source_machine=None, destination_machine=None, source_user=None, destination_user=None, force_copy=False, timeout=60):

    if not is_path_exists(source_path, source_machine, source_user) :
        precision_if_needed = f' on {source_machine}' if source_machine is not None else ''
        raise MolaException(f'The source path {source_path} does not exist{precision_if_needed}.')
    
    raise_error = (
        not force_copy 
        and not destination_path.endswith('/')
        and is_path_exists(destination_path, destination_machine, destination_user) 
    )
    if raise_error:
        precision_if_needed = f' on {destination_machine}' if destination_machine is not None else ''
        raise MolaException(
            f'The destination path {destination_path} already exists{precision_if_needed}.'
            'To force copy and erase previous path, use force_copy=True.'
            )
    
    def get_path_with_machine(path, machine=None, user=None):
        if machine is not None:
            if destination_user is None:
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
        makedirs_remote(destination_path, machine=destination_machine, user=destination_user)
        subprocess.run([f'scp -r {source} {destination}'], shell=True, check=True, capture_output=True, timeout=timeout)


# if __name__ == '__main__':
    # mode = sys.argv[1] # 'cp', 'mv', 'cp_forced', 'mv_forced'
    # In   = sys.argv[2]
    # Out  = sys.argv[3]
    # if mode == 'cp':
    #     copy(In, Out)
    # elif mode == 'mv':
    #     move(In, Out)
    # elif mode == 'rm':
    #     remove(In)
    # else:
    #     raise MolaException(f'unknown argument {mode} (must be cp, mv or rm)')
