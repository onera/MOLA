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
import shutil 
from treelab import cgns
from mola import misc
from mola.logging import mola_logger, MolaException, redirect_streams_to_logger
from mola import __MOLA_PATH__
from mola import server as SV

def apply_to_solver(workflow):

    add_reference_state(workflow)
    add_governing_equations(workflow)
    if hasattr(workflow, '_FULL_CGNS_MODE'):
        add_elsa_keys_to_cgns(workflow)

    write_run_scripts(workflow)
    write_data_files(workflow)

def add_reference_state(workflow):
    '''
    Add ``ReferenceState`` node to CGNS using user-provided conditions
    '''

    ReferenceState = dict(**workflow.Flow['ReferenceState'])

    for var in ['Reynolds','Mach','Pressure','Temperature']:
        ReferenceState[var] = workflow.Flow[var]
 
    namesForCassiopee = dict(
        cv                    = 'Cv',
        Gamma                 = 'Gamma',
        SutherlandViscosity   = 'Mus',
        SutherlandConstant    = 'Cs',
        SutherlandTemperature = 'Ts',
        Prandtl               = 'Pr',
    )
    for var in ['cv','Gamma','SutherlandViscosity','SutherlandConstant','SutherlandTemperature','Prandtl']:
        ReferenceState[namesForCassiopee[var]] = workflow.Fluid[var]

    for base in workflow.tree.bases():
        base.setParameters('ReferenceState', ContainerType='ReferenceState', **ReferenceState)

def add_governing_equations(workflow):
    '''
    Add the nodes corresponding to `FlowEquationSet_t`
    '''
    FlowEquationSet = cgns.Node(Name='FlowEquationSet', Type='FlowEquationSet')
    cgns.Node(Parent=FlowEquationSet, Name='GoverningEquations', Type='GoverningEquations', Value='NSTurbulent')
    cgns.Node(Parent=FlowEquationSet, Name='EquationDimension', Type='EquationDimension', Value=workflow.ProblemDimension)

    workflow.tree.findAndRemoveNodes(Type='FlowEquationSet', Depth=2)
    for base in workflow.tree.bases():
        base.addChild(FlowEquationSet)

def add_elsa_keys_to_cgns(workflow):
    '''
    Include node ``.Solver#Compute`` , where elsA keys are set in full CGNS mode.
    '''
    workflow.tree.findAndRemoveNodes(Name='.Solver#Compute', Depth=2)

    # Put all solver keys in a unique and flat dictionary
    AllElsAKeys = dict()
    for keySet in workflow.SolverParameters.values():
        AllElsAKeys.update(keySet)
      
    for base in workflow.tree.bases(): 
        base.setParameters('.Solver#Compute', **AllElsAKeys)

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
    NodesToLink = t.group(Name='FlowSolution#Init*', Type='FlowSolution', Depth=3) # for initial field(s) (possible second order restart)
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

def write_run_scripts(workflow):
    write_compute(workflow.RunManagement)
    write_coprocess(workflow.RunManagement)
    write_job_launcher(workflow.RunManagement)

def write_compute(RunManagement):

    txt = '''
from mola.workflow.workflow import Workflow

workflow = Workflow('main.cgns')
workflow.print()
workflow.compute()
'''
    save_file_maybe_remote(RunManagement, 'compute.py', txt)

def write_coprocess(RunManagement):
    save_file_maybe_remote(RunManagement, 'coprocess.py', '# do nothing')

def write_job_launcher(RunManagement):

    # shutil.copy2(f'{__MOLA_PATH__}/TEMPLATES/job_template.sh', 'job.sh')
    job_text = SV.get_job_text(RunManagement, 'elsa')+'\n\n'
    job_text += f'mpirun $OPENMPIOVERSUBSCRIBE -np {RunManagement["NumberOfProcessors"]} elsA.x -C xdt-runtime-tree compute.py 1>stdout.log 2>stderr.log\n'
    # Write job file
    save_file_maybe_remote(RunManagement, 'job.sh', job_text)

def save_file_maybe_remote(RunManagement, filename, txt):
    if SV.run_on_localhost(RunManagement['Machine'], RunManagement['RunDirectory']):
        SV.save_file(filename, txt, RunManagement['RunDirectory'])
    else:
        SV.save_file(filename, txt, '.')
        SV.copy_remote(
            source_path=filename, 
            destination_path=RunManagement['RunDirectory'], 
            destination_machine=RunManagement['Machine'],
            )