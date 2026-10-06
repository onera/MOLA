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
import shlex
from fnmatch import fnmatch
import subprocess
import time

from mola.dependency_injector.retriever import load_source
from mola.logging.exceptions import MolaException, MolaAssertionError
from mola import __MOLA_PATH__, __owner__

def submit_command(command, machine, input=None, user=None, use_mola_env=False,
        remote_solver=os.environ.get('MOLA_SOLVER'),

        # due to abusive redirection to stderr of slurm commands:
        false_errors_contain=['warning', 'machar = _get_machar(dtype)'],  # machar: warning of compatibility between numpy and socle
        false_errors_start_with='sbatch:',
        true_errors_start_with='sbatch: error:',
        true_errors_contain=['error']):

    ssh_host = get_ssh_host_command(machine=machine, user=user)
    env = os.environ.copy()
    if ssh_host != '':
        assert input is None
        input = '\n'.join(command.split(';'))
        command = f'{ssh_host}'
        if use_mola_env:
            input = add_mola_env(machine, remote_solver) + input
            env = {}
    elif use_mola_env:
        # For local execution, also source the correct solver environment
        # This ensures that job.sh is written with the correct MPI launcher
        mola_env_cmd = add_mola_env(machine, remote_solver)
        command = f'{mola_env_cmd} {command}'

    if ssh_host == '' and input is not None:
        # A local command receiving stdin does not need a shell.  Avoiding one
        # makes this path work on Windows, while remote commands continue to
        # be sent through the POSIX shell on the SSH host above.
        command = shlex.split(command, posix=(os.name != 'nt'))
        shell = False
    else:
        shell = True

    output = subprocess.run(command, input=input,
                shell=shell, check=False, capture_output=True, env=env,
                encoding='UTF-8')
    errlines = [] 
    for line in output.stderr.split('\n')[:-1]:

        if line.startswith(true_errors_start_with) or \
                any([true_error in line.lower() for true_error in true_errors_contain]):
        
            if line.startswith(false_errors_start_with) or \
                    any([false_error in line.lower() for false_error in false_errors_contain]):
                continue
        
            errlines += [line]


    if errlines:
        msg = f'got error using command: {command} with input:\n{input}, error is:\n'
        raise MolaException(msg+'\n'.join(errlines))

    return output.stdout

def get_network():
    network = os.getenv('MOLA_NETWORK')
    if network is None:
        raise MolaAssertionError(
            'The environment variable MOLA_NETWORK is undefined. '
            'It is probably because it is not defined in the environment that was sourced. '
            'See documentation to know how to install properly MOLA.'
            )
    return network

def get_network_config():
    network = get_network()
    return load_source('config', os.path.join(__MOLA_PATH__, 'mola', 'env', network, 'network.py'))

def get_scheduler_defaults(machine, mola_target_path=__MOLA_PATH__):
    network = get_network()
    for path in [
        os.path.join(mola_target_path, 'mola', 'env', network, machine, 'scheduler_defaults.py'),
        os.path.join(__MOLA_PATH__, 'mola', 'env', network, machine, 'scheduler_defaults.py')
        ]:
        try:
            return load_source('scheduler_defaults', path)
        except FileNotFoundError:
            pass

    return None

def add_mola_env(machine, solver=os.environ.get('MOLA_SOLVER'), target_user=None, mola_path=None):
    if machine == 'localhost':
        machine = guess_localhost()
    if mola_path is None:
        mola_path = get_mola_installation_path(machine)
    network = os.environ.get('MOLA_NETWORK', 'onera')
    if target_user is not None:
        mola_path = mola_path.replace(os.environ.get('USER', ''), target_user)
    mola_env = os.path.join(mola_path, 'mola', 'env', network, 'env.sh')
    return  f"MOLA={mola_path} MOLA_MACHINE={machine} source {mola_env} {solver} || {{ echo 'ERROR: Failed to source {mola_env} with solver {solver}' >&2; exit 1; }} && "


def guess_localhost():
    try:
        network_config = get_network_config()
        return network_config.guess_localhost()
    except BaseException as e:
        raise MolaException(f'Cannot guess local host') from e

def guess_machine_from_path(path):

    path = os.path.abspath(path)

    try:
        network_config = get_network_config()
        for pattern, machine in network_config.PathsToEnvironments.items():
            if fnmatch(path, pattern):
                return machine
        raise
    except:
        raise MolaException(f'Cannot guess machine from path {path}')

def guess_machine(path=None):
    try:
        machine = guess_machine_from_path(path)
    except:
        # assume machine is localhost
        machine = guess_localhost()
    return machine

def run_on_localhost(machine=None, run_directory='.'): 
    '''
    Parameters
    ----------
    machine : str or None
        Name of a machine, that will be test to check if that is the localhost or not.
        If :py:obj:`None` (default value), then try to guess the machine with :py:fun:`guess_machine`.
    run_directory : str
        Path that can be used to guess the machine, if **machine** is None.
    
    Returns
    -------
    bool
        True if the machine is given or can be guessed, and that is compared to localhost with success.
        False if not or if :py:fun:`guess_localhost` return an error.
    '''  
    if machine == 'localhost':
        return True
    
    if machine is None:
        try:
            machine = guess_machine(path=run_directory)
        except:
            return True
        
    try:
        localhost = guess_localhost()
        return (localhost == machine)
    except:
        return True
    
def get_ssh_host_command(machine=None, user=None, path='.'):
    
    if not run_on_localhost(machine, path):

        if not machine:
            machine = guess_machine(path)
            assert machine, f"FATAL: could not guess remote machine"

        if user is None:
            ssh_host = f'ssh -q -T -o BatchMode=yes {machine}'
        else:
            ssh_host = f'ssh -q -T -o BatchMode=yes {user}@{machine}'
    else:
        ssh_host = ''
    return ssh_host
    
def get_mola_installation_path(machine):
    scheduler_defaults = get_scheduler_defaults(machine)

    if not scheduler_defaults or not hasattr(scheduler_defaults,'MOLA_PATH'):
        return __MOLA_PATH__
    
    input_mola_install_path = scheduler_defaults.MOLA_PATH
    adapted_mola_install_path = adapt_username_of_mola_installation_path(input_mola_install_path)
    return adapted_mola_install_path

def adapt_username_of_mola_installation_path(mola_install_path : str):
    '''
    mola_path is a str that may contain "$USER". In such case, will replace it
    with the user name of the mola distribution being used.

    This is in response to https://gitlab.onera.net/numerics/mola/-/issues/286#note_81604
    '''
    return mola_install_path.replace('$USER',__owner__)

def wait_until(predicate, timeout=30., period=1.0, *args, **kwargs):
    must_end = time.time() + timeout
    while time.time() < must_end:
        if predicate(*args, **kwargs): 
            return 
        time.sleep(period)
    raise MolaException('Reach TimeOut')

def job_is_submitted_or_running(job_path, machine, user=None):
    user_option = f'-u {user}' if user else '--me'
    command = f'squeue {user_option} -o "%Z" | grep -qx "{job_path}" && echo true || echo false'
    output = submit_command(command, machine, user=user)
    last_line = output.split('\n')[-2]
    if last_line == 'true':
        return True
    else:
        return False
