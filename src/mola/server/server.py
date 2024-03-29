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


from mola import misc
from mola.logging import mola_logger, MolaException
from mola import __MOLA_PATH__
from mola.server import _cpmv_

def get_network():
    return os.getenv('MOLA_NETWORK')

def get_network_config():
    network = get_network()
    return misc.load_source('config', os.path.join(__MOLA_PATH__, 'mola', 'env', network, 'config.py'))

def guess_host():
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
    except:
        raise MolaException(f'Cannot guess machine from path {path}')


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

    if source_path.startswith('./'): 
        source_path = source_path[2:]
    if destination_path.startswith('./'): 
        destination_path = destination_path[2:]

    if source_path == destination_path:
        # nothing to do 
        return

    if source_machine is None:
        try:
            source_machine = guess_machine_from_path(source_path) 
        except:
            pass
    
    if destination_machine is None:
        try:
            destination_machine = guess_machine_from_path(destination_path) 
        except:
            pass

    _cpmv_.scp(source_path, destination_path, 
               source_machine, destination_machine, 
               source_user, destination_user, 
               force_copy=force_copy
               )

