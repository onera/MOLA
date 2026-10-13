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
from mola.cfd.preprocess.write_cfd_files.write_cfd_files import (
    get_job_text,
    get_lines_to_submit_job_again,
    write_compute)


def apply_to_solver(workflow, items : str ='all'):

    adjust_run_management(workflow)

    if items == 'all':
        write_compute(workflow.RunManagement)
        write_data_files(workflow)
        write_job_launcher(workflow.RunManagement, workflow.RunManagement['SchedulerOptions'])

    elif items == 'only_job':
        write_job_launcher(workflow.RunManagement, workflow.RunManagement['SchedulerOptions'])

    elif items == 'except_existing_job':
        write_compute(workflow.RunManagement)
        write_data_files(workflow)
        path_of_job = os.path.join(workflow.RunManagement['RunDirectory'], names.FILE_JOB)
        machine = workflow.RunManagement['Machine']
        if not SV.files_operations.is_file(path_of_job, machine):
            write_job_launcher(workflow.RunManagement, workflow.RunManagement['SchedulerOptions'])
        
    else:
        raise NotImplementedError(f'items={items} not implemented, shall be one of: "all","only_job","except_existing_job"')

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


def write_job_launcher(RunManagement, scheduler_options):

    nthreads = RunManagement["NumberOfThreads"]
    nranks = RunManagement["NumberOfProcessors"]

    job_text = get_job_text('fast', RunManagement, scheduler_options)+'\n\n'
    job_text += 'export KMP_WARNINGS=FALSE\n'
    if nranks == 1:
        job_text += f'export OMP_NUM_THREADS={nthreads}\n'
        job_text += f'python3 {names.FILE_COMPUTE} 1>{names.FILE_STDOUT} 2>{names.FILE_STDERR}'
    else:
        job_text += 'export OMP_PLACES=cores\n'
        argcmd = '-a $OPENMPIOVERSUBSCRIBE' if bool(os.environ.get("OPENMPIOVERSUBSCRIBE")) else ''
        job_text += f'kpython -n {nranks} -t {nthreads} {argcmd} {names.FILE_COMPUTE} 1>{names.FILE_STDOUT} 2>{names.FILE_STDERR}'

    job_text += get_lines_to_submit_job_again(RunManagement)
    job_text += '\nmola_plot --no-show\n'
    
    SV.save_file_maybe_remote(names.FILE_JOB, job_text, RunManagement['RunDirectory'], machine=RunManagement['Machine'])


def adjust_run_management(workflow):
    workflow.RunManagement['SchedulerOptions']['cpus-per-task'] = 1

    if "NumberOfThreads" not in workflow.RunManagement or \
        workflow.RunManagement["NumberOfThreads"] is None:
        if workflow.RunManagement['Scheduler'] == 'SLURM':
            nthreads = '$SLURM_CPUS_PER_TASK'
        elif workflow.RunManagement['Scheduler'] == 'local':
            import multiprocessing
            nthreads = multiprocessing.cpu_count()
        elif workflow.RunManagement['Scheduler'] is None:
            raise MolaUserError(f"for solver fast you must provide RunManagement['Scheduler'] value")
        else:
            raise MolaUserError(f"Scheduler {workflow.RunManagement['Scheduler']} not supported")

    else:
        nthreads = workflow.RunManagement["NumberOfThreads"]

    workflow.RunManagement["NumberOfThreads"] = nthreads

    workflow.set_workflow_parameters_in_tree()
