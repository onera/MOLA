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

import mola.naming_conventions as names
from mola.logging import mola_logger, MolaException, MolaUserError
from mola import server as SV

MolaToScheduler = dict(
    SLURM = dict(
        JobName = 'job-name',
        Comment = 'comment',
        NumberOfProcessors = 'ntasks', # $SLURM_NTASKS
        NumberOfThreads = 'cpus-per-task', # $SLURM_CPUS_PER_TASK
        TimeLimit = 'time',
    ),
)

SchedulerDefaults = dict(
    SLURM = {
        'job-name' : 'mola',
        'output' : 'output.%j.log',
        'error' : 'error.%j.log',
        'exclusive': None,
    },
)

ALLOWED_SCHEDULERS = ['local','SLURM']

def apply(workflow):

    workflow.RunManagement['SchedulerOptions'] = set_default(workflow.RunManagement)

def set_default(RunManagement, check_run_dir=True):
    # CAVEAT: cannot use other contextual information contained in Workflow if 
    # only provides RunManagement in function
    set_default_machine(RunManagement)
    
    RunManagement.setdefault('mola_target_path', SV.get_mola_installation_path(RunManagement['Machine']))

    for key, value in RunManagement.items():
        if isinstance(value, str) and len(value) == 0:
            raise MolaUserError(f'The value of parameter RunManagement["{key}"] cannot be an empty string.')
        
    if not SV.run_on_localhost(RunManagement['Machine'], RunManagement['RunDirectory']):
        path = RunManagement['RunDirectory']
        machine = RunManagement['Machine']
        user = RunManagement.get('User')
        mola_target_path = RunManagement['mola_target_path']
        mola_logger.info((
            f"> Run on a remote machine ({machine}):\n"
            f"    on path {path}\n"
            f"    sourcing {mola_target_path}"
            ), rank=0
        )
        
        if not SV.is_existing_path(mola_target_path, machine, user):
            raise MolaException(f"Cannot access to {mola_target_path}")
        
        if RunManagement['RemovePreviousRunDirectory']:
            mola_logger.user_warning('Remove previous run directory')
            SV.remove_path(path, machine, user, file_only=False)
        elif check_run_dir and SV.is_existing_path(f'{path}/{names.FILE_INPUT_SOLVER}', machine, user):
            raise MolaException(f"Run Directory {path} already exists")
    
    scheduler, scheduler_options = get_scheduler_and_options(RunManagement)
    set_time_margin(RunManagement, scheduler_options)
    set_launcher_command(RunManagement)

    return scheduler_options

def set_default_machine(RunManagement):
    if ('Machine' not in RunManagement) or (RunManagement['Machine'] == 'auto'):
        RunManagement['Machine'] = SV.guess_machine(RunManagement['RunDirectory'])


def get_scheduler_and_options(RunManagement):

    default_scheduler = 'local'  # None

    # Get default options from the machine scheduler_defaults.py
    scheduler_defaults = SV.get_scheduler_defaults(RunManagement['Machine'],
                            mola_target_path=RunManagement['mola_target_path'])
    
    if scheduler_defaults is None:
        scheduler = default_scheduler
        scheduler_options = dict()
    
    else:
        try:
            scheduler = scheduler_defaults.JOB_SCHEDULER
        except AttributeError:
            scheduler = default_scheduler
            
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

    # Parameters given by user directly for the job scheduler
    if 'SchedulerOptions' in RunManagement:
        for key, value in RunManagement['SchedulerOptions'].items():
            scheduler_options[key] = value

    # possibly want to run locally (e.g. within same slurm node) without
    # submitting new sbatch jobs (and having to wait for them), which is
    # required by test_WorkflowManager_sphere_local when running
    # in juno. Otherwise, jobs would be launched, test will continue and raise
    # and exception because the tests cannot be completed
    if 'Scheduler' not in RunManagement:
        RunManagement['Scheduler'] = scheduler

    elif RunManagement['Scheduler'] not in ALLOWED_SCHEDULERS:
        requested_scheduler = RunManagement['Scheduler']
        raise MolaUserError(f'You requested RunManagement Scheduler "{requested_scheduler}" but must be one of: {ALLOWED_SCHEDULERS}')

    return scheduler, scheduler_options


def set_time_margin(RunManagement, scheduler_options):
    time_limit = scheduler_options.get('time', '24:00:00')
    margin = RunManagement['QuitMarginBeforeTimeOutInSeconds']
    RunManagement['TimeOutInSeconds'] = convert_to_seconds(time_limit) - convert_to_seconds(margin)

def set_launcher_command(RunManagement):
    if 'LauncherCommand' not in RunManagement \
        or RunManagement['LauncherCommand'] == 'auto':
        scheduler, scheduler_options = get_scheduler_and_options(RunManagement)
        if RunManagement['Scheduler'] == 'SLURM':
            RunManagement['LauncherCommand'] = f"cd {RunManagement['RunDirectory']}; sbatch {names.FILE_JOB}"
        else:
            RunManagement['LauncherCommand'] = f"cd {RunManagement['RunDirectory']}; bash {names.FILE_JOB}"

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
