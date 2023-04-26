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

'''
elsA compute.py script for WORKFLOW STANDARD

MOLA Dev

File History:
23/12/2020 - L. Bernardos
'''

# ----------------------- IMPORT SYSTEM MODULES ----------------------- #
import sys
import os
import numpy as np
# np.seterr(all='raise')
import shutil
import timeit
from MOLA.Coprocess import printCo
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

# ---------------------------- IMPORT MOLA ---------------------------- #
import MOLA.Coprocess as CO
import MOLA.LiftingLine as LL
import MOLA.InternalShortcuts as J

# ------------------------------ SETTINGS ------------------------------ #
FULL_CGNS_MODE   = False
FILE_SETUP       = 'setup.py'
FILE_CGNS        = 'main.cgns'
FILE_SURFACES    = 'surfaces.cgns'
FILE_ARRAYS      = 'arrays.cgns'
FILE_FIELDS      = 'tmp-fields.cgns' # BEWARE of tmp- suffix
FILE_COLOG       = 'coprocess.log'
FILE_BODYFORCESRC= 'bodyforce.cgns'
DIRECTORY_OUTPUT = 'OUTPUT'
DIRECTORY_LOGS   = 'LOGS'

# will be overriden by coprocess.py : 
RequestedStatistics = [] 
TagSurfacesWithIteration = False
# ------------------ IMPORT AND SET CURRENT SETUP DATA ------------------ #
setup = J.load_source('setup',FILE_SETUP)

# Load and appropriately set variables of coprocess module
CO.FULL_CGNS_MODE   = FULL_CGNS_MODE
CO.FILE_SETUP       = FILE_SETUP
CO.FILE_CGNS        = FILE_CGNS
CO.FILE_SURFACES    = FILE_SURFACES
CO.FILE_ARRAYS      = FILE_ARRAYS
CO.FILE_FIELDS      = FILE_FIELDS
CO.FILE_COLOG       = FILE_COLOG
CO.FILE_BODYFORCESRC= FILE_BODYFORCESRC
CO.DIRECTORY_OUTPUT = DIRECTORY_OUTPUT
CO.setup            = setup

if rank==0:
    try: os.makedirs(DIRECTORY_OUTPUT)
    except: pass
    try: os.makedirs(DIRECTORY_LOGS)
    except: pass

# --------------------------- END OF IMPORTS --------------------------- #

# ----------------- DECLARE ADDITIONAL GLOBAL VARIABLES ----------------- #
try: BodyForceInputData = setup.BodyForceInputData
except: BodyForceInputData = None
CO.invokeCoprocessLogFile()
arrays = CO.invokeArrays()

niter    = setup.elsAkeysNumerics['niter']
if niter == 0:
    CO.printCo('WARNING: niter = 0 -> will only make extractions', proc=0, color=J.WARN)
inititer = setup.elsAkeysNumerics['inititer']
itmax    = inititer+niter-2 # BEWARE last iteration accessible trigger-state-16

Skeleton = CO.loadSkeleton()

# ========================== LAUNCH ELSA ========================== #

import elsA_user
if not FULL_CGNS_MODE:

    Cfdpb = elsA_user.cfdpb(name='cfd')
    Mod   = elsA_user.model(name='Mod')
    Num   = elsA_user.numerics(name='Num')

    CfdDict  = setup.elsAkeysCFD
    ModDict  = setup.elsAkeysModel
    NumDict  = setup.elsAkeysNumerics

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
CO.elsAxdt = elsAxdt

e=elsAxdt.XdtCGNS(FILE_CGNS)


# ------------------------------- BODYFORCE ------------------------------- #
toWithSourceTerms = []
BodyForceDisks = []
BODYFORCE_INITIATED = False
if BodyForceInputData:
    LocalBodyForceInputData = LL.getLocalBodyForceInputData(BodyForceInputData)
    LL.invokeAndAppendLocalObjectsForBodyForce(LocalBodyForceInputData)
    NumberOfSerialRuns = LL.getNumberOfSerialRuns(BodyForceInputData, NumberOfProcessors)
# ------------------------------------------------------------------------- #

# TODO : remove conditioning on UNSTEADY_OVERSET once elsA bug 
# https://elsa.onera.fr/issues/10824 is fixed
UNSTEADY_OVERSET = hasattr(setup,'OversetMotion') and setup.OversetMotion

CO.loadMotionForElsA(elsA_user, Skeleton)

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
CO.readStaticMasksForElsA(e, elsA_user, Skeleton)
CO.loadUnsteadyMasksForElsA(e, elsA_user, Skeleton)

Cmpi.barrier()
CO.printCo('launch compute',proc=0)
Cfdpb.compute()
Cfdpb.extract()
Cmpi.barrier()


CO.CurrentIteration += 1
CO.printCo('iteration %d -> end of run'%CO.CurrentIteration, proc=0, color=J.MAGE)
CO.updateAndWriteSetup(setup)
t = CO.extractFields(Skeleton)

# save arrays
arraysTree = CO.extractArrays(t, arrays, RequestedStatistics=RequestedStatistics,
          Extractions=setup.Extractions, addMemoryUsage=True)
CO.save(arraysTree, os.path.join(DIRECTORY_OUTPUT,FILE_ARRAYS))

# save surfaces
surfs = CO.extractSurfaces(t, setup.Extractions)
CO.save(surfs, os.path.join(DIRECTORY_OUTPUT,FILE_SURFACES),
               tagWithIteration=TagSurfacesWithIteration)

# save bodyforce disks
CO.save(BodyForceDisks,os.path.join(DIRECTORY_OUTPUT,FILE_BODYFORCESRC))

# save fields
tR = J.moveFields(t)
CO.save(tR, os.path.join(DIRECTORY_OUTPUT,FILE_FIELDS))

elsAxdt.free("xdt-runtime-tree")
elsAxdt.free("xdt-output-tree")

CO.moveTemporaryFile(os.path.join(DIRECTORY_OUTPUT,FILE_FIELDS))

CO.printCo('END OF compute.py',0)
CO.moveLogFiles()
