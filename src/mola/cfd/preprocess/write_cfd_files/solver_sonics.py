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
from mola.cfd.preprocess.write_cfd_files.write_cfd_files import get_job_text

def apply_to_solver(workflow):

    write_run_scripts(workflow)
    write_data_files(workflow)

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
        
def write_run_scripts(workflow):
    write_compute(workflow.RunManagement)
    write_job_launcher(workflow.RunManagement, workflow.RunManagement['SchedulerOptions'])

def write_compute(RunManagement):
    txt = f'''
from mola.workflow import read_workflow
import mola.naming_conventions as names

workflow = read_workflow(names.FILE_INPUT_SOLVER)
workflow.compute()
'''
    SV.save_file_maybe_remote(names.FILE_COMPUTE, txt, RunManagement['RunDirectory'], machine=RunManagement['Machine'])

def write_job_launcher(RunManagement, scheduler_options):

    job_text = get_job_text('sonics', RunManagement, scheduler_options)+'\n\n'
    job_text += f'mpirun $OPENMPIOVERSUBSCRIBE -np {RunManagement["NumberOfProcessors"]} python3 {names.FILE_COMPUTE} 1>{names.FILE_STDOUT} 2>{names.FILE_STDERR}\n'
    job_text += 'mola_plot --no-show\n'
    SV.save_file_maybe_remote(names.FILE_JOB, job_text, RunManagement['RunDirectory'], machine=RunManagement['Machine'])
