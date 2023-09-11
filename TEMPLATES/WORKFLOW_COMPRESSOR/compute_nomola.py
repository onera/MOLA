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
elsA compute.py script for general configuration

File History:
23/12/2020 - L. Bernardos
'''

# ----------------------- IMPORT SYSTEM MODULES ----------------------- #
import sys
import os
import numpy as np
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

import setup
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

if rank==0:
    try: os.makedirs(DIRECTORY_OUTPUT)
    except: pass
    try: os.makedirs(DIRECTORY_LOGS)
    except: pass

# --------------------------- END OF IMPORTS --------------------------- #

# ----------------- DECLARE ADDITIONAL GLOBAL VARIABLES ----------------- #
try: Splitter = setup.Splitter
except: Splitter = None
try: BodyForceInputData = setup.BodyForceInputData
except: BodyForceInputData = None

niter    = setup.elsAkeysNumerics['niter']
inititer = setup.elsAkeysNumerics['inititer']
itmax    = inititer+niter-2 # BEWARE last iteration accessible trigger-state-16

# SPLIT WITH PYPART #
import etc.pypart.PyPart     as PPA

PyPartBase = PPA.PyPart(FILE_CGNS,
                        lksearch=[DIRECTORY_OUTPUT, '.'],
                        loadoption='partial',
                        mpicomm=comm,
                        LoggingInFile=False, 
                        LoggingFile='{}/PYPART_partTree'.format(DIRECTORY_LOGS),
                        LoggingVerbose=40  # Filter: None=0, DEBUG=10, INFO=20, WARNING=30, ERROR=40, CRITICAL=50
                        )
# reorder=[6, 2] is recommended by CLEF, mostly for unstructured mesh
# with modernized elsA. 
# Mandatory arguments to use lussorscawf: reorder=[6,2], nCellPerCache!=0
# See http://elsa.onera.fr/restricted/MU_MT_tuto/latest/MU-98057/Textes/Attribute/numerics.html#numerics.implicit
PartTree = PyPartBase.runPyPart(method=2, partN=1, reorder=[6, 2], nCellPerCache=1024)
PyPartBase.finalise(PartTree, savePpart=True, method=1)
Skeleton = PyPartBase.getPyPartSkeletonTree()
I._rmNodesByName(Skeleton,'ZoneBCGT') # https://elsa.onera.fr/issues/11149
Distribution = PyPartBase.getDistribution()

# Put Distribution into the Skeleton
for zone in I.getZones(Skeleton):
    zonePath = I.getPath(Skeleton, zone, pyCGNSLike=True)[1:]
    Cmpi._setProc(zone, Distribution[zonePath])

t = I.merge([Skeleton, PartTree])

# ========================== LAUNCH ELSA ========================== #

if not FULL_CGNS_MODE:
    import elsA_user

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

if Splitter == 'PyPart':
    e = elsAxdt.XdtCGNS(tree=t, links=[], paths=[])
    e.distribution = Distribution
else:
    e=elsAxdt.XdtCGNS(FILE_CGNS)

e.action=elsAxdt.COMPUTE
e.mode=elsAxdt.READ_ALL
e.compute()

Cmpi._convert2PartialTree(t)
I._rmNodesByName(t, '.Solver#Param')
I._rmNodesByType(t, 'IntegralData_t')
Cmpi.barrier()
PyPartBase.mergeAndSave(t, 'PyPart_fields')

