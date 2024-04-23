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
from mola.logging import mola_logger, MolaException, redirect_streams_to_logger
from mola import server as SV

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

    # Save fields.cgns with the 3D fields
    with redirect_streams_to_logger(mola_logger):
        if run_on_localhost:
            os.makedirs(os.path.join(workflow.RunManagement['RunDirectory'], 'OUTPUT'), exist_ok=True)
            t.save(os.path.join(workflow.RunManagement['RunDirectory'], 'OUTPUT', 'fields.cgns'))
        else:
            os.makedirs('OUTPUT', exist_ok=True)
            t.save(os.path.join('OUTPUT', 'fields.cgns'))

    # Save main.cgns with links to OUTPUT/fields.cgns for 
    NodesToLink = t.group(Name='FSolution#*#Init*', Type='FlowSolution', Depth=3) # for initial field(s) (possible second order restart)
    NodesToLink += t.group(Name='FlowSolution#Average', Type='FlowSolution', Depth=3) 
    NodesToLink += t.group(Name='BCDataSet#Average') 
    
    for FlowSolutionInit in NodesToLink:
        path = FlowSolutionInit.path()
        FlowSolutionInit.remove()
        t.addLink(path=path, target_file='OUTPUT/fields.cgns', target_path=path)
        
    with redirect_streams_to_logger(mola_logger):
        if run_on_localhost:
            t.save(os.path.join(workflow.RunManagement['RunDirectory'], 'main.cgns'))
        else:
            t.save('main.cgns')
    
    if not run_on_localhost:
        SV.copy_remote(
            source_path='main.cgns', 
            destination_path=os.path.join(workflow.RunManagement['RunDirectory'], 'main.cgns'), 
            destination_machine=workflow.RunManagement['Machine'],
            )
        SV.copy_remote(
            source_path=os.path.join('OUTPUT', 'fields.cgns'), 
            destination_path=os.path.join(workflow.RunManagement['RunDirectory'], 'OUTPUT', 'fields.cgns'), 
            destination_machine=workflow.RunManagement['Machine'],
            )
        SV.remove_path('main.cgns', machine='localhost')
        SV.remove_path('OUTPUT', machine='localhost', file_only=False)

def write_run_scripts(workflow):
    write_compute(workflow.RunManagement)
    write_job_launcher(workflow.RunManagement)

def write_compute(RunManagement):
    # FIXME Here it is not necessarily Workflow that should be imported
    # but the Workflow* that serves to preprocess the case.
    # See how it is done in bin/mola_prepare

    txt = '''
from mola.workflow import Workflow

workflow = Workflow('main.cgns')
workflow.print()
workflow.compute()
'''
    SV.save_file_maybe_remote('compute.py', txt, RunManagement['RunDirectory'], machine=RunManagement['Machine'])

def write_job_launcher(RunManagement):

    job_text = SV.get_job_text(RunManagement, 'elsa')+'\n\n'
    job_text += f'mpirun $OPENMPIOVERSUBSCRIBE -np {RunManagement["NumberOfProcessors"]} python3 compute.py 1>stdout.log 2>stderr.log\n'
    SV.save_file_maybe_remote('job.sh', job_text, RunManagement['RunDirectory'], machine=RunManagement['Machine'])
