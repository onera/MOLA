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
import getpass

def guess_host(Network):
    '''
    Returns the host name. For example:   'sator', 'spiro' or 'ld'
    '''
    HostName = socket.gethostname()
    if Network == 'onera':
        PossibleMachineNamesInHostName = ('sator','spiro','visung','ld')
        for name in PossibleMachineNamesInHostName:
            if name in HostName: 
                if name == 'visung': 
                    return 'ld'
                else:
                    return name
            if HostName.startswith('n'): 
                return 'sator'
        UserName = getpass.getuser()
        if not os.path.exists(os.path.join(os.path.sep,'stck',UserName)):
            HostName = 'StckInvisible'
        return HostName

    else:
        raise Exception(f'Unknown Network: {Network}')
