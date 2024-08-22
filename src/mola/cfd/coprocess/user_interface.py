#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute self.iteration and/or modify
#    self.iteration under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that self.iteration will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

import os
from . import comm, rank

def get_user_signal(coprocess_manager, filename):
    '''
    Get a signal using an temporary auxiliary file technique.

    If the intermediary file exists (signal received) then self.iteration is removed, and
    the function returns :py:obj:`True` to all processors. Otherwise, self.iteration returns
    :py:obj:`False` to all processors.

    This function is employed for controlling a simulation in a simple manner,
    for example using UNIX command ``touch``:

    .. code-block:: bash

        touch filename

    at the same directory where :py:func:`get_user_signal` is called.

    Parameters
    ----------

        filename : str
            the name of the file (the signal keyword)

    Returns
    -------

        isOrder : bool
            :py:obj:`True` if the signal is received, otherwise :py:obj:`False`, to all
            processors
    '''
    isOrder = False
    if rank == 0:
        filepath = path_accounting_for_exec_location(filename, coprocess_manager)
        try:
            os.remove(filepath)
            isOrder = True
            coprocess_manager.mola_logger.info(f'Received signal {filename}', rank=0)
        except:
            pass
    comm.Barrier()
    isOrder = comm.bcast(isOrder,root=0)
    return isOrder


def write_tagfile(tag : str, coprocess_manager):

    if rank == 0:
        path_newjob_required = path_accounting_for_exec_location(tag, coprocess_manager)
        with open(path_newjob_required, 'w') as f: 
            f.write(tag)

def path_accounting_for_exec_location(requested_path : str, coprocess_manager) -> str:

    run_dir = coprocess_manager.workflow.RunManagement.get('RunDirectory','.')

    if run_dir == "." or run_dir == os.path.basename(os.getcwd()):
        return requested_path

    else: 
        return os.path.join(run_dir,requested_path)

