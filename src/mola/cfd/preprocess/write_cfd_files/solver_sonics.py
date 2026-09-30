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
from treelab import cgns
import mola.cfd.preprocess.mesh.io as io
import mola.naming_conventions as names
from mola.logging import mola_logger, MolaException, redirect_streams_to_logger
from mola import server as SV
from mola.cfd.preprocess.write_cfd_files.write_cfd_files import (
    get_job_text,
    write_compute)
from mola.mpi import get_mpi_launcher


def apply_to_solver(workflow, items : str ='all'):

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

    # HACK required in order to avoid AssertionError at line 771 in
    # etc/pypart/PpartCGNS/LayoutsS.pxi, Layouts.splitBCDataSet 
    for node in t.group(Name='BCDataSet#Average', Type='BCDataSet'):
        node.setType('UserDefinedData')

    # Save FILE_OUTPUT_3D with the 3D fields
    with redirect_streams_to_logger(mola_logger):
        if run_on_localhost:
            dst = os.path.join(workflow.RunManagement['RunDirectory'], names.FILE_INPUT_SOLVER)
        else:
            dst = names.FILE_INPUT_SOLVER
        io.writer.write(workflow, t, dst, io_tool='maia')
    
    if not run_on_localhost:
        SV.copy_remote(
            source_path=names.FILE_INPUT_SOLVER, 
            destination_path=os.path.join(workflow.RunManagement['RunDirectory'], names.FILE_INPUT_SOLVER), 
            destination_machine=workflow.RunManagement['Machine'],
            )
        SV.remove_path(names.FILE_INPUT_SOLVER, machine='localhost')


def write_job_launcher(RunManagement, scheduler_options):

    job_text = get_job_text('sonics', RunManagement, scheduler_options)+'\n\n'

    launcher = get_mpi_launcher(RunManagement)
    
    job_text += f'{launcher} python3 {names.FILE_COMPUTE} 1>{names.FILE_STDOUT} 2>{names.FILE_STDERR}\n'

    job_text += f'mola_plot --no-show\n'
    SV.save_file_maybe_remote(names.FILE_JOB, job_text, RunManagement['RunDirectory'], machine=RunManagement['Machine'])
