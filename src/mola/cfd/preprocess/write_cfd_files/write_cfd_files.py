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

def apply(workflow):

    apply_to_solver(workflow)

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
            header += f"#SBATCH --{option}={value}\n"
    return header
