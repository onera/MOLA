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
from mola.cfd import apply_to_solver
from mola import __MOLA_PATH__
from mola.logging import mola_logger, MolaException
from mola.server import server
from mola import misc

def apply(workflow):

    set_default(workflow.RunManagement)
    apply_to_solver(workflow)

def set_default(RunManagement):

    # Set default parameters
    RunManagementDefault = dict(
        RunDirectory='.',
        NumberOfProcessors=None,
        SubmitJob=False,
        SecondsMarginForQuitBeforeTimeOut = 180,
        LauncherCommand = 'auto', # or 'sbatch job.sh', './job.sh'...
        mola_target_path = __MOLA_PATH__,
        FilesAndDirectories=[],
        )
    for key, default_value in RunManagementDefault.items():
        RunManagement.setdefault(key, default_value)

    set_machine(RunManagement)
    set_job_scheduler_options(RunManagement)

    if RunManagement['Machine'] == 'sator' and not RunManagement['JobSchedulerOptions']['comment']:
        raise MolaException('AER is needed to run a job on sator') 

    # NumberOfProcessors must be set before this stage
    # It may have been set during an automatic splitting operation
    if not isinstance(RunManagement['NumberOfProcessors'], int):
        raise MolaException(f'The value {RunManagement["NumberOfProcessors"]} for NumberOfProcessors is not allowed. It must be an integer')
    
    # Time margin
    RunManagement.setdefault('TimeLimit', '24:00:00')
    RunManagement['TimeOutInSeconds'] = convert_to_seconds(RunManagement['TimeLimit']) - convert_to_seconds(RunManagement['SecondsMarginForQuitBeforeTimeOut'])
    RunManagement.pop('SecondsMarginForQuitBeforeTimeOut')

def set_machine(RunManagement):
    if ('Machine' not in RunManagement) or (RunManagement['Machine'] == 'auto'):
        RunManagement['Machine'] = server.guess_machine(RunManagement['RunDirectory'])

def set_job_scheduler_options(RunManagement):
    scheduler, scheduler_options = server.get_scheduler_and_default_options(RunManagement['Machine'])

    if scheduler == 'SLURM':
        set_slurm_options_and_update_RunManagement(RunManagement, scheduler_options)

    RunManagement['JobScheduler'] = scheduler
    RunManagement['JobSchedulerOptions'] = scheduler_options

# TODO move this function in server ?
def set_slurm_options_and_update_RunManagement(RunManagement, scheduler_options):
    MolaToSlurm = dict(
            JobName = 'job-name',
            Comment = 'comment',
            AER = 'comment',
            NumberOfProcessors = 'ntasks',
            TimeLimit = 'time',
        )
    scheduler_options.setdefault('job-name', 'mola')

    for key, option in MolaToSlurm.items():
        if key in RunManagement:
            scheduler_options[option] = RunManagement[key]
        elif option in scheduler_options:
            RunManagement[key] = scheduler_options[option]
    
    scheduler_options['output'] = 'output.%j.log'
    scheduler_options['error'] = 'error.%j.log'
    
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


def build_job_scheduler_header(job_scheduler, job_scheduler_options):
    header = ''
    if job_scheduler == 'SLURM':
        for option, value in job_scheduler_options.items():
            header += f"#SBATCH --{option}={value}\n"
    return header

def get_job_text(RunManagement, Solver):

    network = server.get_network()

    job_scheduler = RunManagement['JobScheduler']
    job_scheduler_options = RunManagement['JobSchedulerOptions']

    header = build_job_scheduler_header(job_scheduler, job_scheduler_options)
    env = os.path.join(
        RunManagement["mola_target_path"],
        "mola",
        "env",
        network,
        RunManagement["Machine"],
        Solver+'.sh')

    job_text = ('#!/bin/bash\n'
               f'{header}\n'
               f'source {env}')

    return job_text

def save_file(filename, text, directory='.'):
    os.makedirs(directory, exist_ok=True)
    filename = os.path.join(directory, filename)
    with open(filename, 'w') as f:
        f.write(text)
    os.chmod(filename, 0o777)

