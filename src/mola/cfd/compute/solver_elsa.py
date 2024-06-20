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

    if rank==0:
        os.makedirs(names.DIRECTORY_OUTPUT, exist_ok=True)
        os.makedirs(names.DIRECTORY_LOG, exist_ok=True)

    import elsAxdt
    elsAxdt.trace(0)

    from mola.cfd.coprocess.manager import CoprocessManager
    coprocess_manager = CoprocessManager(workflow)
    workflow._coprocess_manager = coprocess_manager

    if not hasattr(workflow, '_FULL_CGNS_MODE'):
        set_parameters_in_elsa_objects(workflow.SolverParameters)

    e = read_cfd_files.apply(workflow)

    e.action = elsAxdt.COMPUTE
    e.mode = elsAxdt.READ_ALL
    e.compute()

    coprocess_manager.finalize()
    del workflow._coprocess_manager
    

def set_parameters_in_elsa_objects(SolverParameters):
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

    funDict = get_cfl_function(NumDict)
    if funDict:
        set_cfl_function(elsA_user, Num, funDict)

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
