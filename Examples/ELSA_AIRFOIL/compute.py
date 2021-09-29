'''
elsA compute.py script for WORKFLOW AIRFOIL

MOLA Dev

06/05/2020 - L. Bernardos
'''

# ----------------------- IMPORT SYSTEM MODULES ----------------------- #
import sys
import os
import imp
import numpy as np
import psutil
import timeit
import shutil
LaunchTime = timeit.default_timer()
from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
NProcs = comm.Get_size()

# ------------------------- IMPORT  CASSIOPEE ------------------------- #
import Converter.PyTree as C
import Converter.Internal as I
import Converter.Mpi as Cmpi

# ---------------------------- IMPORT MOLA ---------------------------- #
import MOLA.Coprocess as CO
# import MOLA.LiftingLine as LL
import MOLA.InternalShortcuts as J

# ------------------------------ SETTINGS ------------------------------ #
FULL_CGNS_MODE   = False
FILE_SETUP       = 'setup.py'
FILE_CGNS        = 'main.cgns'
FILE_SURFACES    = 'surfaces.cgns'
FILE_LOADS       = 'loads.cgns'
FILE_FIELDS      = 'tmp-fields.cgns' # BEWARE of tmp- suffix
FILE_COLOG       = 'coprocess.log'
FILE_BODYFORCESRC= 'BodyForceSources.cgns'
DIRECTORY_OUTPUT = 'OUTPUT'
DIRECTORY_LOGS   = 'LOGS'

# ------------------ IMPORT AND SET CURRENT SETUP DATA ------------------ #
try: os.remove('setup.pyc')
except: pass
setup = imp.load_source('setup', FILE_SETUP)

# Load and appropriately set variables of coprocess module
CO.FULL_CGNS_MODE   = FULL_CGNS_MODE
CO.FILE_SETUP       = FILE_SETUP
CO.FILE_CGNS        = FILE_CGNS
CO.FILE_SURFACES    = FILE_SURFACES
CO.FILE_LOADS       = FILE_LOADS
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
loads = CO.invokeLoads()

niter    = setup.elsAkeysNumerics['niter']
inititer = setup.elsAkeysNumerics['inititer']
itmax    = inititer+niter-1 # BEWARE last iteration accessible trigger-state-16

Skeleton = Cmpi.convertFile2SkeletonTree(FILE_CGNS)
I._rmNodesByName(Skeleton, 'FlowSolution*')

# ---- functions ---- #
from datetime import datetime

if rank == 0:
    DATEANDTIME=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open(FILE_COLOG, 'w') as f:
        f.write('COPROCESS LOG FILE STARTED AT %s\n'%DATEANDTIME)
Cmpi.barrier()


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
CO.elsAxdt = elsAxdt

e=elsAxdt.XdtCGNS(FILE_CGNS)


e.action=elsAxdt.READ_ALL
e.compute()

to = elsAxdt.get(elsAxdt.OUTPUT_TREE)
CO.adaptEndOfRun(to)
toWithSkeleton = I.merge([Skeleton, to])

# save loads
CO.extractIntegralData(to, loads, Extractions=[],
                        DesiredStatistics=DesiredStatistics)
CO.addMemoryUsage2Loads(loads)
loadsTree = CO.loadsDict2PyTree(loads)
CO.save(loadsTree, os.path.join(DIRECTORY_OUTPUT,FILE_LOADS))

# save surfaces
surfs = CO.extractSurfaces(toWithSkeleton, setup.Extractions)
CO.save(surfs,os.path.join(DIRECTORY_OUTPUT,FILE_SURFACES))

# save fields
tmp_fields = os.path.join(DIRECTORY_OUTPUT,FILE_FIELDS)
final_fields = tmp_fields.replace('tmp-','')
CO.save(toWithSkeleton,final_fields)
elsAxdt.free("xdt-runtime-tree")
elsAxdt.free("xdt-output-tree")
