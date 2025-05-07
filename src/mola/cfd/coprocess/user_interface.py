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

from pathlib import Path
from fnmatch import fnmatch
from . import comm, rank
from mola.logging import MolaException, CYAN, ENDC

# Control Flags for interactive control using command 'touch <flag>'
AVAILABLE_SIGNALS = [
    'STOP',
    'SAVE_*',
    # 'COMPUTE_BODYFORCE',
    'QUIT',
]

# TODO rename this "signal" by "command"
def check_and_execute_user_signal(coprocess_manager):
    for signal_pattern in AVAILABLE_SIGNALS:
        signal_received = get_user_signal(coprocess_manager, signal_pattern)
        
        if signal_received:
            if signal_received == 'STOP':
                coprocess_manager.status = 'TO_STOP'

            elif signal_received.startswith('SAVE_'):
                filename = signal_received[5:]
                save_extractions(coprocess_manager.Extractions, filename, coprocess_manager.mola_logger)

            elif signal_received == 'QUIT':
                raise MolaException(f'Aborted simulation following QUIT signal.')
            
            else:
                raise MolaException(f'Unknown user signal: {signal_received}')

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
    signal = False
    if rank == 0:
        try:
            run_dir = Path(coprocess_manager.workflow.RunManagement['RunDirectory'])
            # return from glob method is a generator. Acces to an element is done with next(...).
            # If the generator is empty, it raises a StopIteration exception
            run_dir_absolute = Path(run_dir).resolve() # added twice sometimes
            if not run_dir_absolute.is_dir(): # HACK
                run_dir_absolute = run_dir_absolute.parent

            filename = next(run_dir_absolute.glob(filename))  
            filename.unlink()
            signal = filename.name
            coprocess_manager.mola_logger.info(f'{CYAN}Received signal {signal}{ENDC}', rank=0)
        except StopIteration:
            pass

    comm.barrier()
    signal = comm.bcast(signal, root=0)
    return signal

def save_extractions(Extractions, arg, mola_logger):

    SIGNALS = ['Residuals', 'Integral', 'Probe']
    SURFACES = ['BC', 'IsoSurface']
    FIELDS = ['Interpolation', '3D']
    EXTRACTION_TYPES = SIGNALS + SURFACES + FIELDS

    if arg in ['ALL', '*']:
        # Extract and save all extractions
        save_extractions_from_types(Extractions, EXTRACTION_TYPES)
    
    elif arg == 'SIGNALS':
        save_extractions_from_types(Extractions, SIGNALS)

    elif arg == 'SURFACES':
        save_extractions_from_types(Extractions, SURFACES)

    elif arg == 'FIELDS':
        save_extractions_from_types(Extractions, FIELDS)

    elif arg in EXTRACTION_TYPES:
        # Extract and save all extractions with this type
        save_extractions_from_types(Extractions, [arg])

    else:
        # Extract and save extractions with a filename foll
        save_extractions_from_filename(Extractions, arg, mola_logger)

def save_extractions_from_types(Extractions, types):
    for extraction in Extractions:
        if extraction['Type'] in types:
            extraction['IsToExtract'] = True
            extraction['IsToSave'] = True


def save_extractions_from_filename(Extractions, filename, mola_logger=None):
    founded = False
    output_files = set()
    for extraction in Extractions:
        # a str must be added like that to a set, otherwise it 
        # is the individual characters of the string that will be added
        output_files.update([extraction['File']])  
        if fnmatch(extraction['File'], filename):
            extraction['IsToExtract'] = True
            extraction['IsToSave'] = True
            founded = True
    
    if not founded and mola_logger:
        mola_logger.warning((
            f'No extraction matches whith the filename {filename}. '
            f'Output files for current simualtions are: {output_files}.'
        ), rank=0)


