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

import mola.cfd.preprocess.mesh.io as io
import mola.naming_conventions as names
from mola.logging import mola_logger, MolaException, MolaUserError, redirect_streams_to_logger
from mola import server as SV
from mola.cfd.preprocess.write_cfd_files.write_cfd_files import get_job_text


def apply_to_solver(workflow):

    write_run_scripts(workflow)
    write_data_files(workflow)

def write_run_scripts(workflow):

    write_compute(workflow.RunManagement)

    write_job_launcher(workflow.RunManagement, workflow.RunManagement['SchedulerOptions'])
    

def write_data_files(workflow):

    run_on_localhost = SV.run_on_localhost(workflow.RunManagement['Machine'], workflow.RunManagement['RunDirectory'])

    t = workflow.tree
    tc = workflow._treeCellCenter

    # Save FILE_INPUT_SOLVER with the 3D fields
    for filename, tree in zip([names.FILE_INPUT_SOLVER,"tc.cgns"],[t,tc]):
        with redirect_streams_to_logger(mola_logger):
            if run_on_localhost:
                dst = os.path.join(workflow.RunManagement['RunDirectory'], filename)
            else:
                dst = filename
            io.writer.write(workflow, tree, dst)
        
        if not run_on_localhost:
            SV.copy_remote(
                source_path=filename, 
                destination_path=os.path.join(workflow.RunManagement['RunDirectory'], filename), 
                destination_machine=workflow.RunManagement['Machine'],
                )
            SV.remove_path(filename, machine='localhost')


def write_compute(RunManagement):
    txt = 'from mola.workflow import read_workflow\n'
    txt+= 'import mola.naming_conventions as names\n'
    txt+= 'workflow = read_workflow(names.FILE_INPUT_SOLVER)\n'
    txt+= 'workflow.compute()'

    SV.save_file_maybe_remote(names.FILE_COMPUTE, txt, RunManagement['RunDirectory'], machine=RunManagement['Machine'])

def write_job_launcher(RunManagement, scheduler_options):
    if 'cpus-per-task' not in scheduler_options:
        scheduler_options['cpus-per-task'] = 1

    nranks = RunManagement["NumberOfProcessors"]

    if "NumberOfThreads" not in RunManagement or \
        RunManagement["NumberOfThreads"] is None:
        if RunManagement['Scheduler'] == 'SLURM':
            nthreads = '$SLURM_CPUS_PER_TASK'
        elif RunManagement['Scheduler'] == 'local':
            import multiprocessing
            nthreads = multiprocessing.cpu_count()
        elif RunManagement['Scheduler'] is None:
            raise MolaUserError(f"for solver fast you must provide RunManagement['Scheduler'] value")
        else:
            raise MolaUserError(f"Scheduler {RunManagement['Scheduler']} not supported")

    else:
        nthreads = RunManagement["NumberOfThreads"]


    job_text = get_job_text('fast', RunManagement, scheduler_options)+'\n\n'
    job_text += 'export KMP_WARNINGS=FALSE\n'
    if nranks == 1:
        job_text += f'export OMP_NUM_THREADS={nthreads}\n'
        job_text += f'python3 {names.FILE_COMPUTE} 1>{names.FILE_STDOUT} 2>{names.FILE_STDERR}'
    else:
        job_text += 'export OMP_PLACES=cores\n'
        argcmd = '-a $OPENMPIOVERSUBSCRIBE' if bool(os.environ.get("OPENMPIOVERSUBSCRIBE")) else ''
        job_text += f'kpython -n {nranks} -t {nthreads} {argcmd} {names.FILE_COMPUTE} 1>{names.FILE_STDOUT} 2>{names.FILE_STDERR}'

    job_text += '\nmola_plot --no-show\n'
    
    SV.save_file_maybe_remote(names.FILE_JOB, job_text, RunManagement['RunDirectory'], machine=RunManagement['Machine'])


