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
from mola import (cgns, misc)
from mola import __MOLA_PATH__

def adapt_to_solver(workflow):

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

def write_run_scripts(workflow):
    write_compute(workflow)
    write_coprocess(workflow)
    write_job_launcher(workflow)

def write_data_files(workflow):

    t = workflow.tree

    # Save fields.cgns with the 3D fields
    os.makedirs(os.path.join(workflow.RunManagement['RunDirectory'], 'OUTPUT'), exist_ok=True)
    t.save(os.path.join(workflow.RunManagement['RunDirectory'], 'OUTPUT', 'fields.cgns'))

    # Save main.cgns with links to fields.cgns for FlowSolution#Init nodes
    # --> Replace all FlowSolution#Init nodes with paths to OUTPUT/fields.cgns
    for FlowSolutionInit in t.group(Name='FlowSolution#Init', Type='FlowSolution', Depth=3):
        path = FlowSolutionInit.path()
        FlowSolutionInit.remove()
        t.addLink(path=path, target_file='OUTPUT/fields.cgns', target_path=path)
    t.save(os.path.join(workflow.RunManagement['RunDirectory'], 'main.cgns'))

def write_compute(workflow):

    txt = '''
from mola.workflow.workflow import Workflow

workflow = Workflow('main.cgns')
workflow.print()
workflow.compute()
'''
    compute_filename = os.path.join(workflow.RunManagement['RunDirectory'], 'compute.py')
    with open(compute_filename, 'w') as File:
        File.write(txt)
    os.chmod(compute_filename, 0o777)

def write_coprocess(workflow):
    with open(os.path.join(workflow.RunManagement['RunDirectory'], 'coprocess.py'), 'w') as File:
        File.write('# do nothing')

def write_job_launcher(workflow, jobFile='job.sh'):

    # shutil.copy2(f'{__MOLA_PATH__}/TEMPLATES/job_template.sh', 'job.sh')

    # with open(f'{__MOLA_PATH__}/TEMPLATES/job_template.sh', 'r') as f:
    #     JobText = f.read()

    # JobText = JobText.replace('<JobName>', workflow.RunManagement['JobName'])
    # JobText = JobText.replace('<AERnumber>', str(workflow.RunManagement['AER']))
    # JobText = JobText.replace('<TimeLimit>', str(workflow.RunManagement["TimeLimit"]))
    # JobText = JobText.replace('<NumberOfProcessors>', str(workflow.RunManagement['NumberOfProcessors']))
    # JobText = JobText.replace('$NPROCMPI', str(workflow.RunManagement['NumberOfProcessors']))

    # if workflow.RunManagement['SlurmConstraint'] is None:
    #     JobText = JobText.replace('#SBATCH --constraint=<SlurmConstraint>', '')
    # else:
    #     JobText = JobText.replace('<SlurmConstraint>', workflow.RunManagement['SlurmConstraint']) 

    # if workflow.RunManagement['SlurmQualityOfService'] is None:
    #     JobText = JobText.replace('#SBATCH --qos=<SlurmQualityOfService>', '')
    # else:
    #     JobText = JobText.replace('<SlurmQualityOfService>', workflow.RunManagement['SlurmQualityOfService']) 


    JobText = f'''#!/bin/bash
#SBATCH -J {workflow.RunManagement['JobName']}
#SBATCH --comment {workflow.RunManagement['AER']}
#SBATCH -o output.%j.log
#SBATCH -e error.%j.log
#SBATCH -t {workflow.RunManagement['TimeLimit']}
#SBATCH -n {workflow.RunManagement['NumberOfProcessors']}
'''
    if workflow.RunManagement['SlurmConstraint'] is not None:
        JobText += f"#SBATCH --constraint={workflow.RunManagement['SlurmConstraint']}\n"
    
    if workflow.RunManagement['SlurmQualityOfService'] is not None:
        JobText += f"#SBATCH --qos={workflow.RunManagement['SlurmQualityOfService']}\n\n"

    JobText += f'source {__MOLA_PATH__}/mola/env/{workflow.RunManagement["Network"]}/{workflow.RunManagement["Machine"]}/{workflow.Solver}.sh\n\n'


    JobText += 'mpirun $OPENMPIOVERSUBSCRIBE -np $NPROCMPI elsA.x -C xdt-runtime-tree compute.py 1>stdout.log 2>stderr.log\n'

    # Write job file
    job_filename = os.path.join(workflow.RunManagement['RunDirectory'], 'jobFile.py')
    with open(job_filename, 'w') as f:
        f.write(JobText)
    os.chmod(job_filename, 0o777)
