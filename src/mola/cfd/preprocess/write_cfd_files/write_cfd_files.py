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
from mola import server as SV
from mola import naming_conventions as names

def apply(workflow, items : str ='all'):
    '''
        items : str
            Options:
                * `'all'`: will write all scripts, data files and job
                * `'only_job'`: will only write job
                * `'except_existing_job'`: will write all except the job if already exists
    '''

    apply_to_solver(workflow, items=items)

def write_compute(RunManagement):

    compute_file_path = os.path.join(os.path.dirname(__file__), 'compute.py')
    with open(compute_file_path, 'r') as f:
        txt = f.read()

    SV.save_file_maybe_remote(names.FILE_COMPUTE, txt, RunManagement['RunDirectory'], machine=RunManagement['Machine'])


def write_info_for_data_retrieval(RunManagement):
    run_directory = str(RunManagement['RunDirectory'])
    if not run_directory.endswith(os.path.sep):
        run_directory += os.path.sep

    file_text = f"""# MOLA log file
# The following lines can be used for retrieving the results of a MOLA run on a remote machine 
RunDirectory={run_directory}
Machine={RunManagement['Machine']}
User={str(RunManagement.get('User'))}
"""
    with open(names.FILE_LOG_REMOTE_DIRECTORY, 'w') as log_file:
        log_file.write(file_text)

def get_job_text(solver, RunManagement, scheduler_options):

    network = SV.get_network()

    header = build_job_scheduler_header(RunManagement['Scheduler'], scheduler_options)

    env = os.path.join(
        RunManagement['mola_target_path'],
        "mola",
        "env",
        network,
        RunManagement['Machine'],
        solver+'.sh')

    job_text = ('#!/bin/bash\n'
               f'{header}\n'
               f'source {env}\n'
                'unset "${!OMPI_@}" "${!MPI_@}"\n' # https://stackoverflow.com/questions/76672866/running-an-independent-slurm-job-with-mpirun-inside-a-python-script-recursive
                'set -e\n' # aborts on first failure encountered
               f'{get_prepare_command()}' # TODO displace at solver-level for conditionally prepending MPI launcher
                )

    return job_text

def build_job_scheduler_header(Scheduler, scheduler_options):
    header = ''
    if Scheduler == 'SLURM':
        for option, value in scheduler_options.items():
            header += f"#SBATCH --{option}"
            if value is not None or (isinstance(value,str) and value != ''):
                header += f"={value}"
            header += "\n"
    
    return header

def get_prepare_command():
    # NOTE: it has no effect if names.FILE_INPUT_WORKFLOW is absent
    # NOTE: the pipe ("||") is used to catch error
    cmd = (f'python3 -m mola.cfd.preprocess.prepare 1>stdout_pre.log 2>stderr_pre.log || '+'{\n'
           f'touch {names.FILE_ERROR_PREPARING_WORKFLOW}\n'
           f'rm {names.FILE_PREPROCESSING}\n'
            'exit 1\n'
            '}\n')
    return cmd

def get_lines_to_submit_job_again(RunManagement):
    command = RunManagement['LauncherCommand'] 
    # remove change of directory in the command, because we are already in the right directory now
    if command.startswith('cd'):
        separators_to_test =  [';', '&&', '&']
        for separator in separators_to_test:
            commands = command.split(separator)
            if len(commands) > 1:
                command = separator.join(commands[1:]) 
                break

    if RunManagement['Scheduler'] == 'SLURM':
        command += ' --dependency=singleton'

    signal = names.FILE_RELAUNCH_JOB

    text = f"""
if [ -f "{signal}" ]; then
    rm -f {signal}
    echo "LAUNCHING JOB AGAIN"
    {command}
    exit 0
fi
"""
    return text
