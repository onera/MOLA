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

from mola import misc
from mola.logging import mola_logger, MolaException
from mola import __MOLA_PATH__

def submit_command(command, machine, user=None, envfile=None):

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
    subprocess.run([command], shell=True, check=True, env=os.environ.copy())

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

    cwd = os.getcwd()
    if path[0] != os.path.sep: 
        path = os.path.join(cwd, path)

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
    try:
        localhost = guess_localhost()
        return (localhost == machine)
    except:
        if run_directory == '.':
            return True
        else:
            return False
    
