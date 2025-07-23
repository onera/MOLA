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
from fnmatch import fnmatch
import subprocess
import time

from mola.dependency_injector.retriever import load_source
from mola.logging import mola_logger, MolaException, MolaAssertionError
from mola import __MOLA_PATH__

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

    if input is None:
        mola_logger.debug(f'run command: {command}')
    else:
        mola_logger.debug(f'run command: {command} with input:\n{input}')

    output = subprocess.run([command], input=input, shell=True, check=False,
                capture_output=True, env=env, encoding='UTF-8')
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

    mola_logger.debug(output.stdout)

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

def add_mola_env(machine, solver=os.environ.get('MOLA_SOLVER')):
    mola_path = get_mola_installation_path(machine)
    network = get_network()
    mola_env = os.path.join(mola_path, 'mola', 'env', network, 'env.sh')
    return  f"source {mola_env} {solver} &>/dev/null || {{ echo 'Error: Cannot source this environment!' >&2; exit 1; }} && "


def guess_localhost():
    try:
        network_config = get_network_config()
        return network_config.guess_localhost()
    except:
        raise MolaException(f'Cannot guess local host')

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
        if user is None:
            ssh_host = f'ssh -T {machine}'
        else:
            ssh_host = f'ssh -T {user}@{machine}'
    else:
        ssh_host = ''
    return ssh_host
    
def get_mola_installation_path(machine):
    try:
        scheduler_defaults = get_scheduler_defaults(machine)
        return scheduler_defaults.MOLA_PATH
    except:
        # By default, return the current installation path, assuming it will be accessible from the specified machine
        return __MOLA_PATH__

def wait_until(predicate, timeout=30., period=1.0, *args, **kwargs):
    must_end = time.time() + timeout
    while time.time() < must_end:
        if predicate(*args, **kwargs): 
            return 
        time.sleep(period)
    raise MolaException('Reach TimeOut')

def job_is_submitted_or_running(job_name, machine, user=None):
    user_option = f'-u {user}' if user else '--me'
    command = f'squeue {user_option} -h -n {job_name} | grep --quiet . && echo "true" || echo "false"'
    output = submit_command(command, machine, user=user)
    last_line = output.split('\n')[-2]
    if last_line == 'true':
        return True
    else:
        return False
