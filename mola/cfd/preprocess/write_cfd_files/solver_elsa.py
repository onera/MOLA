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
    os.makedirs(os.path.join(workflow.RunManagement['RunDirectory'], 'OUTPUT'), exist_ok=True)
    workflow.tree.save(os.path.join(workflow.RunManagement['RunDirectory'], 'OUTPUT', 'fields.cgns'))
    mainCGNS = workflow.tree.copy()

    # Replace all FlowSolution#Init nodes with paths to OUTPUT/fields.cgns
    for FlowSolutionInit in mainCGNS.group(Name='FlowSolution#Init', Type='FlowSolution', Depth=3):
        path = FlowSolutionInit.path()
        FlowSolutionInit.remove()
        mainCGNS.addLink(path=path, target_file='OUTPUT/fields.cgns', target_path=path)

    workflow.tree.save(os.path.join(workflow.RunManagement['RunDirectory'], 'main.cgns'))

def write_compute(workflow):

    txt = '''
from mola.workflow import Workflow
from mola.cfd.preprocess.write_cfd_files.elsa import compute

workflow = Workflow('main.cgns')
compute(workflow)
'''

    with open(os.path.join(workflow.RunManagement['RunDirectory'], 'compute.py'), 'w') as File:
        File.write(txt)

    # shutil.copy2(f'{__MOLA_PATH__}/TEMPLATES/WORKFLOW_STANDARD/.sh', 'job.sh')

def write_coprocess(workflow):
    with open(os.path.join(workflow.RunManagement['RunDirectory'], 'coprocess.py'), 'w') as File:
        File.write('# do nothing')

def write_job_launcher(workflow):

    shutil.copy2(f'{__MOLA_PATH__}/TEMPLATES/job_template.sh', 'job.sh')

    # with open(os.path.join(workflow.RunManagement['RunDirectory'], 'job.sh'), 'w') as File:
    #     File.write('# do nothing')
    
def compute(workflow):
    
    # ----------------------- IMPORT SYSTEM MODULES ----------------------- #
    import sys
    import os
    import numpy as np
    # np.seterr(all='raise')
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

    # ----------------- DECLARE ADDITIONAL GLOBAL VARIABLES ----------------- #
    # CO.invokeCoprocessLogFile()
    # arrays = CO.invokeArrays()

    # if workflow.Numerics['NumberOfIterations'] == 0:
    #     CO.printCo('WARNING: niter = 0 -> will only make extractions', proc=0, color=J.YELLOW)
    # inititer = setup.elsAkeysNumerics['inititer']
    # itmax    = inititer+niter-2 # BEWARE last iteration accessible trigger-state-16

    Skeleton = CO.loadSkeleton()

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

    e=elsAxdt.XdtCGNS(FILE_CGNS)

    # TODO : remove conditioning on UNSTEADY_OVERSET once elsA bug 
    # https://elsa.onera.fr/issues/10824 is fixed
    # UNSTEADY_OVERSET = hasattr(setup,'OversetMotion') and setup.OversetMotion

    # CO.loadMotionForElsA(elsA_user, Skeleton)

    e.mode = elsAxdt.READ_MESH
    e.mode |= elsAxdt.READ_CONNECT
    e.mode |= elsAxdt.READ_BC
    e.mode |= elsAxdt.READ_BC_INIT
    e.mode |= elsAxdt.READ_INIT
    e.mode |= elsAxdt.READ_FLOW
    e.mode |= elsAxdt.READ_COMPUTATION
    e.mode |= elsAxdt.READ_OUTPUT
    e.mode |= elsAxdt.READ_TRACE
    e.mode |= elsAxdt.SKIP_GHOSTMASK # NOTE https://elsa.onera.fr/issues/3480
    e.action=elsAxdt.TRANSLATE

    e.compute()
    # CO.readStaticMasksForElsA(e, elsA_user, Skeleton)
    # CO.loadUnsteadyMasksForElsA(e, elsA_user, Skeleton)

    Cmpi.barrier()
    # CO.printCo('launch compute',proc=0)
    Cfdpb.compute()
    Cfdpb.extract()
    Cmpi.barrier()
    e.save('solution.cgns')
