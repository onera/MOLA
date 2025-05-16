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

def apply(workflow):

    apply_to_solver(workflow)

    run_on_localhost = SV.run_on_localhost(workflow.RunManagement['Machine'], workflow.RunManagement['RunDirectory'])
    if not run_on_localhost:
        write_info_for_data_retrieval(workflow.RunManagement)

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
                'unset "${!OMPI_@}" "${!MPI_@}"' # https://stackoverflow.com/questions/76672866/running-an-independent-slurm-job-with-mpirun-inside-a-python-script-recursive
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

def get_lines_to_submit_job_again(RunManagement):
    command = RunManagement['LauncherCommand'] 
    if RunManagement['Scheduler'] == 'SLURM':
        command += ' --dependency=singleton'

    text = f"""
if [ -f "{names.FILE_NEWJOB_REQUIRED}" ]; then
    echo "LAUNCHING JOB AGAIN"
    {command}
    rm -f {names.FILE_NEWJOB_REQUIRED}
    exit 0
fi
"""
    return text
