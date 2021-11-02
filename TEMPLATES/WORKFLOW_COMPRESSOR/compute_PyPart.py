'''
elsA compute.py script for general configuration

File History:
23/12/2020 - L. Bernardos
'''

# ----------------------- IMPORT SYSTEM MODULES ----------------------- #
import sys
import os
import imp
import numpy as np
import timeit
LaunchTime = timeit.default_timer()
from mpi4py import MPI
comm   = MPI.COMM_WORLD
rank   = comm.Get_rank()
NProcs = comm.Get_size()

# ------------------------- IMPORT  CASSIOPEE ------------------------- #
import Converter.PyTree as C
import Converter.Internal as I
import Converter.Filter as Filter
import Converter.Mpi as Cmpi

# ------------------------- IMPORT  PyPart    ------------------------- #
import etc.pypart.PyPart     as PPA

# ---------------------------- IMPORT MOLA ---------------------------- #
import MOLA.Coprocess as CO
import MOLA.Postprocess as POST
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


# ========================== LAUNCH PYPART ========================== #
PyPartBase = PPA.PyPart(FILE_CGNS,
                        lksearch=['OUTPUT/', '.'],
                        loadoption='partial',
                        mpicomm=MPI.COMM_WORLD,
                        LoggingInFile=True,
                        LoggingFile='LOGS/partTree',
                        LoggingVerbose=0
                        )
PartTree = PyPartBase.runPyPart(method=2, partN=1, reorder=[4, 3])
PyPartBase.finalise(PartTree, savePpart=True, method=1)
Distribution = PyPartBase.getDistribution()
Skeleton = PyPartBase.getPyPartSkeletonTree()
CO.PyPartBase = PyPartBase
# Put Distribution into the Skeleton
for zone in I.getZones(Skeleton):
    zonePath = I.getPath(Skeleton, zone, pyCGNSLike=True)[1:]
    Cmpi._setProc(zone, Distribution[zonePath])

t = I.merge([Skeleton, PartTree])

for zone in I.getZones(PartTree):
    path = I.getPath(PartTree, zone)
    # Add GridCoordinates
    coords = I.getNodeFromName(zone, 'GridCoordinates')
    Skeleton = I.append(Skeleton, coords, path)
    # Add Height parametrization for turbomachinery
    ch = I.getNodeFromName(zone, 'FlowSolution#Height')
    if ch:
        Skeleton = I.append(Skeleton, ch, path)
    # Add PyPart special node for the mergeAndSave latter
    for node in I.getChildren(I.getNodeFromName(zone, ':CGNS#Ppart')):
        nodePath = I.getPath(PartTree, node)
        nodeInSkel = I.getNodeFromPath(Skeleton, nodePath)
        if not nodeInSkel:
            Skeleton = I.append(Skeleton, node, path+'/:CGNS#Ppart')

# Add empty Coordinates for skeleton zones
# Needed to make Cmpi.convert2PartialTree work
for zone in I.getZones(Skeleton):
    GC = I.getNodeFromType(zone, 'GridCoordinates_t')
    if not GC:
        J.set(zone, 'GridCoordinates', childType='GridCoordinates_t',
            CoordinateX=None, CoordinateY=None, CoordinateZ=None)

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
CO.elsAxdt = elsAxdt

e = elsAxdt.XdtCGNS(tree=t, links=[], paths=[])
e.distribution = Distribution

# ------------------------------- BODYFORCE ------------------------------- #
toWithSourceTerms = []
BodyForceDisks = []
BODYFORCE_INITIATED = False
if BodyForceInputData:
    LocalBodyForceInputData = LL.getLocalBodyForceInputData(BodyForceInputData)
    LL.invokeAndAppendLocalObjectsForBodyForce(LocalBodyForceInputData)
    NumberOfSerialRuns = LL.getNumberOfSerialRuns(BodyForceInputData, NProcs)
# ------------------------------------------------------------------------- #

e.action = elsAxdt.READ_ALL
e.compute()

to = elsAxdt.get(elsAxdt.OUTPUT_TREE)
CO.adaptEndOfRun(to)
toWithSkeleton = I.merge([Skeleton, to])

# save loads
CO.extractIntegralData(to, loads)
CO.addMemoryUsage2Loads(loads)
loadsTree = CO.loadsDict2PyTree(loads)
CO.save(loadsTree, os.path.join(DIRECTORY_OUTPUT,FILE_LOADS))

# save surfaces
surfs = CO.extractSurfaces(toWithSkeleton, setup.Extractions)
CO.monitorTurboPerformance(surfs, loads, DesiredStatistics)
# surfs = POST.absolute2Relative(surfs, loc='nodes')
CO.save(surfs,os.path.join(DIRECTORY_OUTPUT,FILE_SURFACES))

# save fields
tmp_fields = os.path.join(DIRECTORY_OUTPUT,FILE_FIELDS)
CO.save(toWithSkeleton, tmp_fields)

elsAxdt.free("xdt-runtime-tree")
elsAxdt.free("xdt-output-tree")

CO.moveTemporaryFile(tmp_fields)

CO.printCo('END OF compute.py',0)
