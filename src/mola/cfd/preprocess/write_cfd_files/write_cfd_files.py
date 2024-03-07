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
from mola import misc
from mola import __MOLA_PATH__
from mola.logging import mola_logger, MolaException
from mola.server.__cpmv__ import guess_host

def apply(workflow):

    set_default(workflow.RunManagement)
    misc.apply_to_solver(workflow)

def set_default(RunManagement):

    # Set default parameters
    RunManagementDefault = dict(
        JobName='MOLAjob',
        RunDirectory='.',
        NumberOfProcessors=None,
        SubmitJob=False,
        Network = 'onera',
        Machine = 'auto', 
        TimeLimit = 'auto',
        SecondsMarginForQuitBeforeTimeOut = 180,
        LauncherCommand = 'auto', # or 'sbatch job.sh', './job.sh'...
        mola_target_path = __MOLA_PATH__,
        FilesAndDirectories=[],
        AER='not_given',
        )
    for key, default_value in RunManagementDefault.items():
        RunManagement.setdefault(key, default_value)

    # NumberOfProcessors must be set before this stage
    # It may have been set during an automatic splitting operation
    if not isinstance(RunManagement['NumberOfProcessors'], int):
        raise MolaException(f'The value {RunManagement["NumberOfProcessors"]} for NumberOfProcessors is not allowed. It must be an integer')
    
    if RunManagement['Machine'] == 'auto':
        RunManagement['Machine'] = guess_host(Network=RunManagement['Network'])
        mola_logger.info(f"The detected Machine on Network {RunManagement['Network']} is {RunManagement['Machine']}")

    if RunManagement['TimeLimit'] == 'auto':
        # To update depending on the cluster
        if RunManagement['Machine'] in ['sator', 'spiro']:
            RunManagement['TimeLimit'] = '0-15:00'
        else:
            RunManagement['TimeLimit'] = '0-24:00'

    if 'SlurmConstraint' not in RunManagement:
        if RunManagement['Machine'] == 'sator':
            # TODO Remove this constraint if it is not useful anymore
            RunManagement['SlurmConstraint'] = 'csl'
        else:
            RunManagement['SlurmConstraint'] = None

    if RunManagement['Machine'] == 'spiro':
        RunManagement.setdefault('SlurmQualityOfService', 'c1_test_giga')
        
    if RunManagement['AER'] == '':
        # if an empty string is written in the tree, elsA is bugging with the following error message:
        #   File "/stck/elsa/Public/v5.1.03/Dist/lib/py/elsA/Parse/loadCGNSPython.py", line 143, in loadOne
        #     if not isinstance(data[0], np.string_) and not isinstance(data[0], np.str_) and data.dtype not in [np.float32,np.float64,np.int32,np.int64,'|S1']:
        #   IndexError: index 0 is out of bounds for axis 0 with size 0
        RunManagement['AER'] == 'not_given' 

    # Time margin
    RunManagement['TimeOutInSeconds'] = convert_to_seconds(RunManagement['TimeLimit']) - convert_to_seconds(RunManagement['SecondsMarginForQuitBeforeTimeOut'])
    RunManagement.pop('SecondsMarginForQuitBeforeTimeOut')

def convert_to_seconds(time_value):
    '''
    Convert a time in seconds.

    Parameters
    ----------
    time_value : str or int
        Could be seconds, as an int or str, or a str with one of the following formats :
        'mm:ss', 'hh:mm:ss', 'j-hh:mm:ss', 'j-hh:mm', 'j-hh'.

    Returns
    -------
    int
        number of seconds in **time_value**
    '''
    time_value = str(time_value)
    if '-' in time_value:
        # The number of days is given
        days, daytime_value = time_value.split('-')
        number_of_columns = daytime_value.count(':')
        if number_of_columns == 0:
            daytime_value += ':00:00'
        elif number_of_columns == 1:
            daytime_value += ':00'
        else:
            assert number_of_columns == 2
    else:
        # No day is given
        days = 0
        daytime_value = time_value

    l = list(map(int, daytime_value.split(':')))
    return int(days)*3600*24 + sum(n * sec for n, sec in zip(l[::-1], (1, 60, 3600)))


def get_job_text(RunManagement, Solver):

    job_text = f'''#!/bin/bash
#SBATCH -J {RunManagement['JobName']}
#SBATCH --comment {RunManagement['AER']}
#SBATCH -o output.%j.log
#SBATCH -e error.%j.log
#SBATCH -t {RunManagement['TimeLimit']}
#SBATCH -n {RunManagement['NumberOfProcessors']}
'''
    if RunManagement['SlurmConstraint'] is not None:
        job_text += f"#SBATCH --constraint={RunManagement['SlurmConstraint']}\n"
    
    if 'SlurmQualityOfService' in RunManagement and RunManagement['SlurmQualityOfService'] is not None:
        job_text += f"#SBATCH --qos={RunManagement['SlurmQualityOfService']}\n"
    
    job_text += f'\nsource {RunManagement["mola_target_path"]}/mola/env/{RunManagement["Network"]}/{RunManagement["Machine"]}/{Solver}.sh\n'

    return job_text

def save_file(filename, text, RunManagement):
    os.makedirs(RunManagement['RunDirectory'], exist_ok=True)
    filename = os.path.join(RunManagement['RunDirectory'], filename)
    with open(filename, 'w') as f:
        f.write(text)
    os.chmod(filename, 0o777)

