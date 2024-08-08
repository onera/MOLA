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

# TODO make this a proper method of coproces_manager
def get_user_signal(coprocess_manager, filename):
    '''
    Get a signal using an temporary auxiliar file technique.

    If the intermediary file exists (signal received) then self.iteration is removed, and
    the function returns :py:obj:`True` to all processors. Otherwise, self.iteration returns
    :py:obj:`False` to all processors.

    This function is employed for controling a simulation in a simple manner,
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
        try:
            os.remove(filename)
            isOrder = True
            coprocess_manager.mola_logger.info(f'Received signal {filename}', rank=0)
        except:
            pass
    comm.Barrier()
    isOrder = comm.bcast(isOrder,root=0)
    return isOrder

# TODO make this a proper method of coproces_manager
def update_operations_from_user_signal(coprocess_manager):

    # Control Flags for interactive control using command 'touch <flag>'

    if get_user_signal(coprocess_manager,'QUIT'): 
        os._exit(0)
    
    if get_user_signal(coprocess_manager,'CONVERGED'):
        coprocess_manager.status = 'TO_STOP'
        return

    if get_user_signal(coprocess_manager,'COMPUTE_BODYFORCE'):
        coprocess_manager.operations_stack.append('COMPUTE_BODYFORCE')
    if get_user_signal(coprocess_manager,'SAVE_BODYFORCE'):
        coprocess_manager.operations_stack.append('SAVE_BODYFORCE')
    
    if get_user_signal(coprocess_manager,'SAVE_RESTART'):
        coprocess_manager.operations_stack.append('SAVE_RESTART')
    if get_user_signal(coprocess_manager,'SAVE_FIELDS'):
        coprocess_manager.operations_stack.append('SAVE_FIELDS')
    if get_user_signal(coprocess_manager,'SAVE_EXTRACTIONS'):
        coprocess_manager.operations_stack.append('SAVE_EXTRACTIONS')
    if get_user_signal(coprocess_manager,'SAVE_SIGNALS'):
        coprocess_manager.operations_stack.append('SAVE_SIGNALS')
    if get_user_signal(coprocess_manager,'SAVE_ALL'):
        coprocess_manager.operations_stack.extend(['SAVE_RESTART', 'SAVE_FIELDS', 'SAVE_EXTRACTIONS', 'SAVE_SIGNALS'])
    
    # TODO Signal RELOAD_SETUP not plugged yet


