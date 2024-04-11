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

from . import server as SV 
from mola import misc

MolaToScheduler = dict(
    SLURM = dict(
        JobName = 'job-name',
        Comment = 'comment',
        NumberOfProcessors = 'ntasks',
        TimeLimit = 'time',
    ),
)

SchedulerDefaults = dict(
    SLURM = {
        'job-name' : 'mola',
        'output' : 'output.%j.log',
        'error' : 'error.%j.log',
    },
)


def get_job_text(RunManagement, solver):

    network = SV.get_network()

    header = build_job_scheduler_header(RunManagement)

    env = os.path.join(
        RunManagement['mola_target_path'],
        "mola",
        "env",
        network,
        RunManagement['Machine'],
        solver+'.sh')

    job_text = ('#!/bin/bash\n'
               f'{header}\n'
               f'source {env}')

    return job_text


def build_job_scheduler_header(RunManagement):
    header = ''
    scheduler, scheduler_options = get_scheduler_and_options(RunManagement)
    set_time_margin(RunManagement, scheduler_options)
    set_launcher_command(RunManagement)
    if scheduler == 'SLURM':
        for option, value in scheduler_options.items():
            header += f"#SBATCH --{option}={value}\n"
    return header


def get_scheduler_and_options(RunManagement):
    # Get default options from the machine scheduler_defaults.py
    scheduler_defaults = SV.get_scheduler_defaults(RunManagement['Machine'], mola_target_path=RunManagement['mola_target_path'])
    if scheduler_defaults is None:
        scheduler = None
        scheduler_options = dict()
    else:
        try:
            scheduler = scheduler_defaults.JOB_SCHEDULER
        except AttributeError:
            scheduler = None
            
        try:
            scheduler_options = scheduler_defaults.JOB_SCHEDULER_OPTIONS
        except AttributeError:
            scheduler_options = dict()
        
        try:
            MolaToScheduler[scheduler].update(scheduler_defaults.MOLA_TO_SCHEDULER)
        except AttributeError:
            pass

    # update with default options from the scheduler, regardless the machine
    try:
        for key, default_value in SchedulerDefaults[scheduler].items():
            scheduler_options.setdefault(key, default_value)
    except KeyError:
        pass

    # update with options from the user
    try:
        for key, option in MolaToScheduler[scheduler].items():
            if key in RunManagement:
                scheduler_options[option] = RunManagement[key]
    except KeyError:
        pass

    return scheduler, scheduler_options


def set_time_margin(RunManagement, scheduler_options):
    time_limit = scheduler_options.get('time', '24:00:00') 
    try: 
        margin = RunManagement.pop('SecondsMarginForQuitBeforeTimeOut')
    except KeyError:
        margin = 600
    RunManagement['TimeOutInSeconds'] = convert_to_seconds(time_limit) - convert_to_seconds(margin)

def set_launcher_command(RunManagement):
    if 'LauncherCommand' not in RunManagement \
        or RunManagement['LauncherCommand'] == 'auto':
        scheduler, scheduler_options = get_scheduler_and_options(RunManagement)
        job_path = os.path.join(RunManagement['RunDirectory'], 'job.sh')
        if scheduler == 'SLURM':
            # RunManagement['LauncherCommand'] = f'sbatch {job_path}'
            RunManagement['LauncherCommand'] = f"cd {RunManagement['RunDirectory']}; sbatch job.sh"
        else:
            RunManagement['LauncherCommand'] = f"cd {RunManagement['RunDirectory']}; ./job.sh"

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
