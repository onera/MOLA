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
import socket
from fnmatch import fnmatch
import subprocess
import time

from mola import misc
from mola.logging import mola_logger, MolaException
from mola import __MOLA_PATH__

def submit_command(command, machine, input=None, user=None, envfile=None):

    if not run_on_localhost(machine):
        if user is not None:
            ssh_host = f"ssh {user}@{machine}"
        else:
            ssh_host = f"ssh {machine}"

        if envfile is not None:
            command = f'{ssh_host} "source {envfile}; {command}"'
        else:
            command = f'{ssh_host} "{command}"'

    mola_logger.debug(command)
    subprocess.run([command], input=input, shell=True, check=True, env=os.environ.copy(), encoding='UTF-8')

def get_network():
    return os.getenv('MOLA_NETWORK')

def get_network_config():
    network = get_network()
    return misc.load_source('config', os.path.join(__MOLA_PATH__, 'mola', 'env', network, 'config.py'))

def guess_localhost():
    HostName = socket.gethostname()
    try:
        network_config = get_network_config()
        for pattern, env in network_config.PatternsToEnvironments.items():
            if fnmatch(HostName, pattern):
                return env
    except:
        raise MolaException(f'Host name {HostName} is unknown')

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
    if machine is None:
        machine = guess_machine(path=run_directory)
        
    try:
        localhost = guess_localhost()
        return (localhost == machine)
    except:
        return True
    
def get_mola_installation_path(machine):
    try:
        network = get_network()
        path = os.path.join(__MOLA_PATH__, 'mola', 'env', network, machine, 'scheduler_defaults.py')
        scheduler_defaults = misc.load_source('scheduler_defaults', path)
        try:
            return scheduler_defaults.MOLA_PATH
        except AttributeError:
            raise
            
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
