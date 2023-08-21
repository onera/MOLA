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
from mola.cfd.preprocess.write_cfd_files.solver_elsa import compute

workflow = Workflow('main.cgns')
workflow.print()
compute(workflow)
'''

    with open(os.path.join(workflow.RunManagement['RunDirectory'], 'compute.py'), 'w') as File:
        File.write(txt)

    # shutil.copy2(f'{__MOLA_PATH__}/TEMPLATES/WORKFLOW_STANDARD/.sh', 'job.sh')

def write_coprocess(workflow):
    with open(os.path.join(workflow.RunManagement['RunDirectory'], 'coprocess.py'), 'w') as File:
        File.write('# do nothing')

def write_job_launcher(workflow, jobFile='job.sh'):

    # shutil.copy2(f'{__MOLA_PATH__}/TEMPLATES/job_template.sh', 'job.sh')

    with open(f'{__MOLA_PATH__}/TEMPLATES/job_template.sh', 'r') as f:
        JobText = f.read()

    JobText = JobText.replace('<JobName>', workflow.RunManagement['JobName'])
    JobText = JobText.replace('<AERnumber>', str(workflow.RunManagement['AER']))
    JobText = JobText.replace('<TimeLimit>', str(workflow.RunManagement["TimeLimit"]))
    JobText = JobText.replace('<NumberOfProcessors>', str(workflow.RunManagement['NumberOfProcessors']))
    JobText = JobText.replace('$NPROCMPI', str(workflow.RunManagement['NumberOfProcessors']))

    if workflow.RunManagement['SlurmConstraint'] is None:
        JobText = JobText.replace('#SBATCH --constraint=<SlurmConstraint>', '')
    else:
        JobText = JobText.replace('<SlurmConstraint>', workflow.RunManagement['SlurmConstraint']) 

    if workflow.RunManagement['SlurmQualityOfService'] is None:
        JobText = JobText.replace('#SBATCH --qos=<SlurmQualityOfService>', '')
    else:
        JobText = JobText.replace('<SlurmQualityOfService>', workflow.RunManagement['SlurmQualityOfService']) 

    with open(jobFile, 'w') as f:
        f.write(JobText)
    os.chmod(jobFile, 0o777)

def launch_elsa_computation(workflow, FILE_CGNS):
    import elsA_user
    if not hasattr(workflow, '_FULL_CGNS_MODE'):

        Cfdpb = elsA_user.cfdpb(name='cfd')
        Mod   = elsA_user.model(name='Mod')
        Num   = elsA_user.numerics(name='Num')

        CfdDict  = workflow.SolverParameters['cfdpb']
        ModDict  = workflow.SolverParameters['model']
        NumDict  = workflow.SolverParameters['numerics']

        elsAobjs = [Cfdpb,   Mod,     Num]
        elsAdics = [CfdDict, ModDict, NumDict]

        for obj, dic in zip(elsAobjs, elsAdics):
            [obj.set(v,dic[v]) for v in dic if not isinstance(dic[v], dict)]

        for k in NumDict:
            if '.Solver#Function' in k:
                funDict = NumDict[k]
                funName = funDict['name']
                if funName == 'f_cfl':
                    f_cfl=elsA_user.function(funDict['function_type'],name=funName)
                    for v in funDict:
                        if v in ('iterf','iteri','valf','vali'):
                            f_cfl.set(v,  funDict[v])
                    Num.attach('cfl', function=f_cfl)

    import elsAxdt
    elsAxdt.trace(0)

    if workflow.SplittingAndDistribution['Strategy'].lower() == 'atcomputation':
        if workflow.SplittingAndDistribution['Splitter'].lower() == 'pypart':
            from ..mesh.split import splitWithPyPart
            t, Skeleton, PyPartBase, Distribution = splitWithPyPart()
        else:
            raise Exception(f"Unkwown Splitter: {workflow.SplittingAndDistribution['Splitter']}")
        e = elsAxdt.XdtCGNS(tree=t, links=[], paths=[])
        e.distribution = Distribution
    else:
        e = elsAxdt.XdtCGNS(FILE_CGNS)

    e.action=elsAxdt.COMPUTE
    e.mode=elsAxdt.READ_ALL
    e.compute()
    e.save('solution.cgns')

def compute(workflow):
    
    # ----------------------- IMPORT SYSTEM MODULES ----------------------- #
    import os
    from mpi4py import MPI
    comm   = MPI.COMM_WORLD
    rank   = comm.Get_rank()
    NumberOfProcessors = comm.Get_size()

    # ------------------------- IMPORT  CASSIOPEE ------------------------- #
    import Converter.PyTree as C
    import Converter.Internal as I
    import Converter.Filter as Filter
    import Converter.Mpi as Cmpi

    # ------------------------------ SETTINGS ------------------------------ #
    # TODO: List all MOLA keywords in mola.__init__.py ? 
    FULL_CGNS_MODE   = False
    FILE_CGNS        = 'main.cgns'
    FILE_SURFACES    = 'surfaces.cgns'
    FILE_ARRAYS      = 'arrays.cgns'
    FILE_FIELDS      = 'tmp-fields.cgns' # BEWARE of tmp- suffix
    FILE_COLOG       = 'coprocess.log'
    DIRECTORY_OUTPUT = 'OUTPUT'
    DIRECTORY_LOGS   = 'LOGS'

    if rank==0:
        os.makedirs(DIRECTORY_OUTPUT, exist_ok=True)
        os.makedirs(DIRECTORY_LOGS, exist_ok=True)

    # --------------------------- END OF IMPORTS --------------------------- #

    # ----------------- DECLARE ADDITIONAL GLOBAL VARIABLES ----------------- #
    # CO.invokeCoprocessLogFile()
    # arrays = CO.invokeArrays()

    # if workflow.Numerics['NumberOfIterations'] == 0:
    #     CO.printCo('WARNING: niter = 0 -> will only make extractions', proc=0, color=J.YELLOW)
    # inititer = setup.elsAkeysNumerics['inititer']
    # itmax    = inititer+niter-2 # BEWARE last iteration accessible trigger-state-16

    # Skeleton = CO.loadSkeleton()

    # ========================== LAUNCH ELSA ========================== #

    launch_elsa_computation(workflow, FILE_CGNS)

def compute_test(workflow):
    
    # ----------------------- IMPORT SYSTEM MODULES ----------------------- #
    import sys
    import os
    import numpy as np
    import shutil
    import timeit
    LaunchTime = timeit.default_timer()
    from mpi4py import MPI
    comm   = MPI.COMM_WORLD
    rank   = comm.Get_rank()
    NumberOfProcessors = comm.Get_size()

    # ------------------------- IMPORT  CASSIOPEE ------------------------- #
    import Converter.PyTree as C
    import Converter.Internal as I
    import Converter.Filter as Filter
    import Converter.Mpi as Cmpi

    # ------------------------------ SETTINGS ------------------------------ #
    FULL_CGNS_MODE   = False
    FILE_CGNS        = 'main.cgns'
    FILE_SURFACES    = 'surfaces.cgns'
    FILE_ARRAYS      = 'arrays.cgns'
    FILE_FIELDS      = 'tmp-fields.cgns' # BEWARE of tmp- suffix
    FILE_COLOG       = 'coprocess.log'
    DIRECTORY_OUTPUT = 'OUTPUT'
    DIRECTORY_LOGS   = 'LOGS'

    if rank==0:
        try: os.makedirs(DIRECTORY_OUTPUT)
        except: pass
        try: os.makedirs(DIRECTORY_LOGS)
        except: pass

    # --------------------------- END OF IMPORTS --------------------------- #

    # ========================== LAUNCH ELSA ========================== #

    import elsA_user
    if not FULL_CGNS_MODE:

        Cfdpb = elsA_user.cfdpb(name='cfd')
        Mod   = elsA_user.model(name='Mod')
        Num   = elsA_user.numerics(name='Num')

        CfdDict  = workflow.SolverParameters['cfdpb']
        ModDict  = workflow.SolverParameters['model']
        NumDict  = workflow.SolverParameters['numerics']

        elsAobjs = [Cfdpb,   Mod,     Num]
        elsAdics = [CfdDict, ModDict, NumDict]

        for obj, dic in zip(elsAobjs, elsAdics):
            [obj.set(v,dic[v]) for v in dic if not isinstance(dic[v], dict)]

        for k in NumDict:
            if '.Solver#Function' in k:
                funDict = NumDict[k]
                funName = funDict['name']
                if funName == 'f_cfl':
                    f_cfl=elsA_user.function(funDict['function_type'],name=funName)
                    for v in funDict:
                        if v in ('iterf','iteri','valf','vali'):
                            f_cfl.set(v,  funDict[v])
                    Num.attach('cfl', function=f_cfl)

    import elsAxdt
    elsAxdt.trace(0)
    # CO.elsAxdt = elsAxdt

    if workflow.SplittingAndDistribution['Strategy'] == 'AtComputation':
        if workflow.SplittingAndDistribution['Splitter'] == 'PyPart':
            e = elsAxdt.XdtCGNS(tree=t, links=[], paths=[])
            e.distribution = Distribution
        elif workflow.SplittingAndDistribution['Splitter'] == 'maia':
            import maia
            dist_tree  = maia.io.file_to_dist_tree(FILE_CGNS, comm)
            zone_to_parts = maia.factory.partitioning.compute_balanced_weights(dist_tree, comm)
            part_tree = maia.factory.partition_dist_tree(dist_tree, comm, zone_to_parts=zone_to_parts)
            C.convertPyTree2File(part_tree, f'part_tree_{rank}.cgns')
            comm.barrier()
            e = elsAxdt.XdtCGNS(tree=part_tree, links=[], paths=[])
            # e.distribution = Distribution
        else:
            raise Exception(f"Unknown splitter: {workflow.SplittingAndDistribution['Splitter']}")
    else:
        e=elsAxdt.XdtCGNS(FILE_CGNS)

    e.action=elsAxdt.COMPUTE
    e.mode=elsAxdt.READ_ALL
    e.compute()
    e.save('solution.cgns')
