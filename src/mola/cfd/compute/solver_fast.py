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
from mola.cfd.coprocess.manager import CoprocessManager

def apply_to_solver(workflow):

    import Fast.PyTree as Fast
    import FastS.PyTree as FastS


    if rank==0:
        os.makedirs(names.DIRECTORY_OUTPUT, exist_ok=True)
        os.makedirs(names.DIRECTORY_LOG, exist_ok=True)

    workflow._coprocess_manager = CoprocessManager(workflow)

    inititer = workflow.Numerics['IterationAtInitialState']
    niter = workflow.Numerics['NumberOfIterations']

    t,tc,ts,graph = Fast.load(names.FILE_INPUT_SOLVER, 'tc.cgns', restart=False)

    Fast._setNum2Base( t, workflow.SolverParameters['Num2Base'])
    Fast._setNum2Zones(t, workflow.SolverParameters['Num2Zones'])

    (t, tc, metrics) = FastS.warmup(t, tc, graph)
    
    for it in range( inititer, inititer+niter ):
    
        print("it=%d"%it)
        FastS._compute(t, metrics, it, tc, graph)
                
    workflow._coprocess_manager.finalize()
    del workflow._coprocess_manager
    
