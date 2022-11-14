'''
elsA compute.py script for aerothermal coupled simulations.

File History:
07/04/2022 - T. Bontemps
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

# ---------------------------- IMPORT MOLA ---------------------------- #
import MOLA.Coprocess as CO
import MOLA.InternalShortcuts as J
import MOLA.WorkflowAerothermalCoupling as WAT

# ------------------------------ SETTINGS ------------------------------ #
FULL_CGNS_MODE   = True
FILE_SETUP       = 'setup.py'
FILE_CGNS        = 'main.cgns'
FILE_SURFACES    = 'surfaces.cgns'
FILE_ARRAYS      = 'arrays.cgns'
FILE_FIELDS      = 'tmp-fields.cgns' # BEWARE of tmp- suffix
FILE_COLOG       = 'coprocess.log'
FILE_BODYFORCESRC= 'bodyforce.cgns'
DIRECTORY_OUTPUT = 'OUTPUT'
DIRECTORY_LOGS   = 'LOGS'

# ------------------ IMPORT AND SET CURRENT SETUP DATA ------------------ #
setup = J.load_source('setup', FILE_SETUP)

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
CO.DIRECTORY_LOGS   = DIRECTORY_LOGS
CO.setup            = setup

if rank==0:
    try: os.makedirs(DIRECTORY_OUTPUT)
    except: pass
    try: os.makedirs(DIRECTORY_LOGS)
    except: pass

# ---------------------------- IMPORT elsA ---------------------------- #
import elsAxdt
from elsA.CGNS import core

# --------------------------- END OF IMPORTS --------------------------- #

# ----------------- DECLARE ADDITIONAL GLOBAL VARIABLES ----------------- #
try: Splitter = setup.Splitter
except: Splitter = None

################################################################################
# GET GLOBAL AND LOCAL MPI COMMUNICATORS
################################################################################

NBPROC_main_Zebulon = int(os.getenv('NBPROC_main_Zebulon', None))
newGroup = comm.group.Incl(range(NumberOfProcessors-NBPROC_main_Zebulon))
CommElsA = comm.Create_group(newGroup)
LocalRank = CommElsA.Get_rank()
LocalNProcs = CommElsA.Get_size()

CO.comm = CommElsA
CO.elsAxdt = elsAxdt
CO.Cmpi.setCommunicator(CO.comm)
CO.invokeCoprocessLogFile()
arrays = CO.invokeArrays()

niter    = setup.elsAkeysNumerics['niter']
if niter == 0:
    CO.printCo('niter = 0: Please update this value and run the simulation again', proc=0, color=J.WARN)
    exit()
inititer = setup.elsAkeysNumerics['inititer']
itmax    = inititer+niter-2 # BEWARE last iteration accessible trigger-state-16

################################################################################
# READ THE MESH
################################################################################
if Splitter == 'PyPart':
    t, Skeleton, PyPartBase, Distribution = CO.splitWithPyPart()
    CO.PyPartBase = PyPartBase

    # Pypart loses AdditionalFamilyName_t nodes in BCs, and also changes
    # .elsA#Hybrid nodes
    ####################################################
    # WARNING : pyC2 manages only quad unstructured mesh
    ####################################################
    I._renameNode(t, 'InternalElts', 'InternalQuads')
    I._renameNode(t, 'ExternalElts', 'ExternalQuads')

else:
    t = Cmpi.convertFile2PyTree('main.cgns')
    Skeleton = CO.loadSkeleton()

################################################################################
# PREPARE VARIABLES TO LOCATE INTERFACES
################################################################################

CouplingSurfaces = WAT.locateCouplingSurfaces(t)
fwk, pyC2Connections = WAT.initializeCWIPIConnections(t, Distribution, CouplingSurfaces)

################################################################################
# ELSA LAUNCHER
################################################################################

# elsA interface initialized with Cwipicommunicator
__e = core.elsA()
__e.initialize(sys.argv, CommElsA)
__e.distribution = Distribution # distribution on local communicator
__e.parse([t, [], []])
__e.compute()


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
CO.monitorTurboPerformance(surfs, arrays, RequestedStatistics)
CO.save(surfs,os.path.join(DIRECTORY_OUTPUT,FILE_SURFACES))

# save fields
CO.save(t, os.path.join(DIRECTORY_OUTPUT,FILE_FIELDS))

elsAxdt.free("xdt-runtime-tree")
elsAxdt.free("xdt-output-tree")
# fwk.trace('leave elsA')
# del fwk

CO.moveTemporaryFile(os.path.join(DIRECTORY_OUTPUT,FILE_FIELDS))

CO.printCo('END OF compute.py',0)
CO.moveLogFiles()
