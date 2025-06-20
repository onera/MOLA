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

# ----------------------- IMPORT SYSTEM MODULES ----------------------- #
import os
from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
NumberOfProcessors = comm.Get_size()

import mola.naming_conventions as names
from mola.cfd.compute.read_cfd_files import read_cfd_files

def apply_to_solver(workflow):

    import elsAxdt
    elsAxdt.trace(0)

    if not hasattr(workflow, '_FULL_CGNS_MODE'):  # FIXME Full CGNS mode cannot work well because niter et al. must be taken in workflow.Numerics
        set_parameters_in_elsa_objects(workflow.SolverParameters, workflow.Numerics)

    elsa_parser = read_cfd_files.apply(workflow)

    from mola.cfd.coprocess.manager import CoprocessManager
    coprocess_manager = CoprocessManager(workflow)
    workflow._coprocess_manager = coprocess_manager

    elsa_parser.action = elsAxdt.COMPUTE
    elsa_parser.mode = elsAxdt.READ_ALL
    elsa_parser.compute()

    if rank==0:
        table = elsa_parser.symboltable()
        with open(os.path.join(names.DIRECTORY_LOG, f'symbol_table.log'), 'w') as f:
            import pprint
            f.write(pprint.pformat(table))

    coprocess_manager.finalize()
    del workflow._coprocess_manager
    

def set_parameters_in_elsa_objects(SolverParameters, Numerics):
    import elsA_user

    Cfdpb = elsA_user.cfdpb(name='cfd')
    Mod   = elsA_user.model(name='Mod')
    Num   = elsA_user.numerics(name='Num')

    CfdDict  = SolverParameters['cfdpb']
    ModDict  = SolverParameters['model']
    NumDict  = SolverParameters['numerics']

    elsAobjs = [Cfdpb,   Mod,     Num]
    elsAdics = [CfdDict, ModDict, NumDict]

    for obj, dic in zip(elsAobjs, elsAdics):
        [obj.set(v,dic[v]) for v in dic if not isinstance(dic[v], dict)]

    Num.set('niter', Numerics['NumberOfIterations'])
    Num.set('inititer', Numerics['IterationAtInitialState'])
    Num.set('itime', Numerics['TimeAtInitialState'])


    funDict = get_cfl_function(NumDict)
    if funDict:
        set_cfl_function(elsA_user, Num, funDict)
    
    if SolverParameters['cfdpb']['config'] == "2d":
        Cfdpb.set_ghostcell(2,2,2,2,0,1)

def get_cfl_function(NumDict):
    for k in NumDict:
        if '.Solver#Function' in k:
            funDict = NumDict[k]
            if funDict['name'] == NumDict['cfl_fct']:
                return funDict
    
    return None

def set_cfl_function(elsA_user, Num, funDict):
    f_cfl = elsA_user.function(funDict['function_type'], name=funDict['name'])
    for v in ('iterf','iteri','valf','vali'):
        f_cfl.set(v,  funDict[v])
    Num.attach('cfl', function=f_cfl)
