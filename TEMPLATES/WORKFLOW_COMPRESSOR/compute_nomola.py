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
import glob
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
                        LoggingVerbose=40  
                        )
PartTree = PyPartBase.runPyPart(method=2, partN=1, reorder=[6, 2], nCellPerCache=1024)
PyPartBase.finalise(PartTree, savePpart=True, method=1)
Skeleton = PyPartBase.getPyPartSkeletonTree()
#I._rmNodesByName(Skeleton,'ZoneBCGT') # https://elsa.onera.fr/issues/11149
Distribution = PyPartBase.getDistribution()

# Put Distribution into the Skeleton
for zone in I.getZones(Skeleton):
    zonePath = I.getPath(Skeleton, zone, pyCGNSLike=True)[1:]
    Cmpi._setProc(zone, Distribution[zonePath])

t = I.merge([Skeleton, PartTree])

# loadSkeleton
addCoordinates = True
if not Skeleton: Skeleton = Cmpi.convertFile2SkeletonTree(FILE_CGNS)

FScoords = I.getNodeFromName1(Skeleton, 'FlowSolution#EndOfRun#Coords')
if FScoords: addCoordinates = False

I._rmNodesByName(Skeleton, 'FlowSolution#EndOfRun*')
I._rmNodesByName(Skeleton, 'ID_*')

if PartTree:
    # Needed nodes are read from PartTree
    def readNodesFromPaths(path):
        split_path = path.split('/')
        path_begining = '/'.join(split_path[:-1])
        name = split_path[-1]
        parent = I.getNodeFromPath(PartTree, path_begining)
        return I.getNodesFromName(parent, name)

    FScoords = I.getNodeFromName1(PartTree, 'FlowSolution#EndOfRun#Coords')
    if FScoords: addCoordinates = False
else:
    # Needed nodes are read from FILE_CGNS with Converter.Filter
    def readNodesFromPaths(path):
        return Filter.readNodesFromPaths(FILE_CGNS, [path])


def replaceNodeByName(parent, parentPath, name):
    oldNode = I.getNodeFromName1(parent, name)
    path = '{}/{}'.format(parentPath, name)
    newNode = readNodesFromPaths(path)
    I._rmNode(parent, oldNode)
    I._addChild(parent, newNode)

def replaceNodeValuesRecursively(node_skel, node_path):
    new_node = readNodesFromPaths(node_path)[0]
    node_skel[1] = new_node[1]
    for child in node_skel[2]:
        replaceNodeValuesRecursively(child, node_path+'/'+child[0])

containers2read = ['FlowSolution#Height',
                    ':CGNS#Ppart',
                    'FlowSolution#DataSourceTerm',
                    'FlowSolution#Average']

for base in I.getBases(Skeleton):
    basename = I.getName(base)
    for zone in I.getNodesFromType1(base, 'Zone_t'):
        # Only for local zones on proc
        proc = I.getValue(I.getNodeFromName(zone, 'proc'))
        if proc != rank: continue

        zonePath = '{}/{}'.format(basename, I.getName(zone))
        zoneInPartialTree = readNodesFromPaths(zonePath)[0]

        # Coordinates
        if addCoordinates: replaceNodeByName(zone, zonePath, 'GridCoordinates')

        for nodeName2read in containers2read:
            if I.getNodeFromName1(zoneInPartialTree, nodeName2read):
                replaceNodeByName(zone, zonePath, nodeName2read)

        # For unstructured mesh
        if I.getZoneType(zone) == 2: # unstructured zone
            replaceNodeByName(zone, zonePath, ':elsA#Hybrid')
            # TODO: Add other types of Elements_t nodes if needed
            replaceNodeByName(zone, zonePath, 'NGonElements')
            replaceNodeByName(zone, zonePath, 'NFaceElements')
            # PointList in BCs and GridConnectivities
            for BC in I.getNodesFromType2(zone, 'BC_t'):
                BCpath = '{}/ZoneBC/{}'.format(zonePath, I.getName(BC))
                replaceNodeByName(BC, BCpath, 'PointList')
            for GC in I.getNodesFromType2(zone, 'GridConnectivity_t'):
                GCpath = '{}/ZoneGridConnectivity/{}'.format(zonePath, I.getName(GC))
                replaceNodeByName(GC, GCpath, 'PointList')



    # always require to fully read Mask nodes 
    masks = I.getNodeFromName1(base, '.MOLA#Masks')
    if masks:
        replaceNodeValuesRecursively(masks, '/'.join([basename, masks[0]]))

# Add empty Coordinates for skeleton zones
# Needed to make Cmpi.convert2PartialTree work
for zone in I.getZones(Skeleton):
    GC = I.getNodeFromType1(zone, 'GridCoordinates_t')
    if not GC:
        GC = I.createUniqueChild(zone, 'GridCoordinates', 'GridCoordinates_t')
        I.createUniqueChild(GC, 'CoordinateX', 'DataArray_t')
        I.createUniqueChild(GC, 'CoordinateY', 'DataArray_t')
        I.createUniqueChild(GC, 'CoordinateZ', 'DataArray_t')
    elif I.getZoneType(zone) == 2:
        # HACK For unstructured zone, correct the node NFaceElements/ElementConnectivity
        # Problem with PyPart: see issue https://elsa-e.onera.fr/issues/9002
        # C._convertArray2NGon(zone)
        NFaceElements = I.getNodeFromName(zone, 'NFaceElements')
        if NFaceElements:
            node = I.getNodeFromName(NFaceElements, 'ElementConnectivity')
            I.setValue(node, np.abs(I.getValue(node)))


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


# extractFields
t = elsAxdt.get(elsAxdt.OUTPUT_TREE)
I._renameNode(t, 'FlowSolution#EndOfRun', 'FlowSolution#Init')
t = I.merge([Skeleton, t])
Cmpi.barrier()

# save
tpt = I.copyRef(t)
Cmpi._convert2PartialTree(tpt)
I._rmNodesByName(tpt, '.Solver#Param')
I._rmNodesByType(tpt, 'IntegralData_t')
Cmpi.barrier()
PyPartBase.mergeAndSave(tpt, 'PyPart_fields')
Cmpi.barrier()
