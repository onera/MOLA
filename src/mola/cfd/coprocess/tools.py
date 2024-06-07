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
import glob

import Converter.PyTree as C
import Converter.Internal as I
import Converter.Mpi as Cmpi
import Converter.Filter as Filter

import mola.naming_conventions as names
from mola.logging import GREEN, CYAN, ENDC, MolaException
from . import mola_logger, rank, NumberOfProcessors, comm

def save(t, filename, coprocess_manager=None, tagWithIteration=False):
    '''
    Generic function to save a PyTree **t** in parallel. Works whatever the
    dimension of the PyTree. Use it to save ``'fields.cgns'``,
    ``'surfaces.cgns'`` or ``'arrays.cgns'``.

    .. important::
        If the mesh was split with PyPart and if the function is called to save
        *FILE_FIELDS*, the tree is automatically merged and saved using PyPart.
        In that case, the variable **PyPartBase** should be defined (normally,
        in ``compute.py``)

    Parameters
    ----------

        t : PyTree
            tree to save

        filename : str
            Name of the file

        tagWithIteration : bool
            if :py:obj:`True`, adds a suffix ``_AfterIter<iteration>``
            to the saved filename (creates a copy)
    '''
    if coprocess_manager is not None:
        if coprocess_manager.workflow.SplittingAndDistribution['Splitter'].lower() == 'pypart' \
            and (filename.endswith(names.FILE_OUTPUT_3D) or filename.endswith(names.FILE_OUTPUT_RESTART)):
            saveWithPyPart(t, filename, coprocess_manager, tagWithIteration=tagWithIteration)
            return

    # t = I.copyRef(t) if I.isTopTree(t) else C.newPyTree(['Base', J.getZones(t)])
    t = I.copyRef(t) if I.isTopTree(t) else C.newPyTree(['Base', I.getZones(t)])
    removeNonLocalZones(t) # HACK https://elsa.onera.fr/issues/11397
    I._adaptZoneNamesForSlash(t)
    for z in I.getZones(t):
        SolverParam = I.getNodeFromName(z,'.Solver#Param')
        if not SolverParam or not I.getNodeFromName(SolverParam,'proc'):
            Cmpi._setProc(z, Cmpi.rank)
    I._rmNodesByName(t,'ID_*')
    I._rmNodesByType(t,'IntegralData_t')

    Skel = getStructure(t)

    UseMerge = False
    try:
        trees = comm.allgather( Skel )        
        trees.insert( 0, t )
        tWithSkel = I.merge( trees )
        renameTooLongZones(tWithSkel)
        for l in 2,3: I._correctPyTree(tWithSkel,l) # unique base and zone names
    except SystemError:
        UseMerge = True
        mola_logger.warning(f'Cmpi.KCOMM.gather FAILED. Using merge=True', rank=0)
        UseMerge = comm.bcast(UseMerge,root=Cmpi.rank)
        tWithSkel = t


    Cmpi.barrier()
    if Cmpi.rank==0:
        try:
            if os.path.islink(filename):
                os.unlink(filename)
            else:
                os.remove(filename)
        except:
            pass
    Cmpi.barrier()

    mola_logger.info(f'{CYAN}saving {filename}...{ENDC}', rank=0) #, end='')
    Cmpi.convertPyTree2File(tWithSkel, filename, merge=UseMerge) # BUG https://elsa.onera.fr/issues/11398
    mola_logger.info(f'{GREEN} OK{ENDC}', rank=0)
    Cmpi.barrier()

    if coprocess_manager is not None:
        if tagWithIteration and Cmpi.rank == 0: 
            copyOutputFiles(coprocess_manager.iteration, filename)

def getStructure(t):
    '''
    Get a PyTree's base structure (children of base nodes are empty)

    Parameters
    ----------

        t : PyTree
            tree from which structure is to be extracted

    Returns
    -------
        Structure : PyTree
            reference copy of **t**, with empty bases
    '''
    tR = I.copyRef(t)
    for n in I.getZones(tR):
        n[2] = []
    return tR

def copyOutputFiles(iteration, *files2copy):
    '''
    Copy the files provided as input *(comma-separated variables)* by addding to
    their name ``'_AfterIter<X>'`` where ``<X>`` will be replaced with the
    corresponding interation

    Parameters
    ----------

        iteration : int
            current iteration

        file2copy : comma-separated :py:class:`str`
            file(s) name(s) to copy at ``OUTPUT`` directory.

    Examples
    --------

    ::

        copyOutputFiles('surfaces.cgns','arrays.cgns')

    '''
    for file2copy in files2copy:
        f2cSplit = file2copy.split('.')
        name = '.'.join(f2cSplit[:-1])
        fmt = f2cSplit[-1]
        newFileName = f'{name}_AfterIter{iteration}.{fmt}'
        try:
            shutil.copy2(file2copy, newFileName)
        except:
            pass

def removeNonLocalZones(t):
    # HACK https://elsa.onera.fr/issues/11397
    for base in I.getBases(t):
        children_to_keep = []
        for child in base[2]:
            if child[3] != 'Zone_t':
                children_to_keep += [ child ]
            elif zoneHasData(child):
                children_to_keep += [ child ]
        base[2] = children_to_keep

def renameTooLongZones(to, n=25):
    '''
    .. warning:: this is a private function, employed by :py:func:`saveSurfaces`

    This function rename zones in a PyTree **to** if their names are too long
    to be save in a CGNS file (maximum length = 32 characters).

    The new name of a zone follows this format:
    ``<NewName>`` = ``<First <n> characters of old name>_<ID>``
    with ``<n>`` an integer and ``<ID>`` the lowest integer (starting form 0) such as
    ``<NewName>`` does not already exist in the PyTree.

    Parameters
    ----------

        to : PyTree
            PyTree to check. Zones with a too long name will be renamed.

            .. note:: tree **to** is modified

        n : int
            Number of characters to keep in the old zone name.
    '''
    for zone in I.getZones(to):
        zoneName = I.getName(zone)
        if len(zoneName) > 32:
            CurrentZoneNames = [I.getName(z) for z in I.getZones(to)]
            c = 0
            newName = '{}_{}'.format(zoneName[:n+1], c)
            while newName in CurrentZoneNames and c < 1000:
                c += 1
                newName = '{}_{}'.format(zoneName[:n+1], c)
            if c == 1000:
                ERRMSG = 'Zone {} has not been renamed by renameTooLongZones() but its length ({}) is greater than maximum authorized length (32)'.format(zoneName, len(zoneName))
                raise MolaException(ERRMSG)
            I.setName(zone, newName)

def load_skeleton(Skeleton=None, PartTree=None):
    '''
    Load the skeleton tree (if not given) and add nodes that are required for
    coprocessing.

    Parameters
    ----------

        Skeleton : PyTree or :py:obj:`None`
            Skeleton tree, got from ``Cmpi.convertFile2SkeletonTree`` with
            Cassiopee or ``PyPartBase.getPyPartSkeletonTree`` with PyPart.
            If :py:obj:`None`, load the Skeleton tree with
            ``Cmpi.convertFile2SkeletonTree(FILE_CGNS)``.

        PartTree : PyTree or :py:obj:`None`
            Partial tree, got from ``Cmpi.convertFile2PyTree(..., proc=rank)``
            with Cassiopee or ``PyPartBase.runPyPart`` with PyPart.
            If :py:obj:`None`, only the needed nodes are read with
            ``Converter.Filter``.

    Returns
    -------

        Skeleton : PyTree
            Skeleton Tree with additional information, required for Coprocess
            functions.
    '''
    addCoordinates = True
    if not Skeleton: Skeleton = Cmpi.convertFile2SkeletonTree(names.FILE_INPUT_SOLVER)

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
            return Filter.readNodesFromPaths(names.FILE_INPUT_SOLVER, [path])


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

            # put BCDataSet#Average in Skeleton
            if not PartTree: # from file
                for zonebc in I.getNodesFromType1(zone,'ZoneBC_t'):
                    for bc in I.getNodesFromType1(zonebc,'BC_t'):
                        bcds_avg = I.getNodeFromName1(bc,'BCDataSet#Average')
                        if bcds_avg is None: continue
                        for bcdata in I.getNodesFromType1(bcds_avg,'BCData_t'):
                            bcdatapath = '/'.join([basename,
                                                   zone[0],
                                                   zonebc[0],
                                                   bc[0],
                                                   bcds_avg[0],
                                                   bcdata[0]])
                            for data in I.getNodesFromType1(bcdata,'DataArray_t'):
                                replaceNodeByName(bcdata, bcdatapath, data[0])
            else: # from PartTree
                for zonebc in I.getNodesFromType1(zone,'ZoneBC_t'):
                    for bc in I.getNodesFromType1(zonebc,'BC_t'):
                        bcds_avg = I.getNodeFromName1(bc,'BCDataSet#Average')
                        if bcds_avg is None: continue
                        bcpath = '/'.join([basename, zone[0], zonebc[0], bc[0]])
                        replaceNodeByName(bc, bcpath, 'BCDataSet#Average')

        # always require to fully read Mask nodes 
        masks = I.getNodeFromName1(base, '.MOLA#Masks')
        if masks:
            replaceNodeValuesRecursively(masks, '/'.join([basename, masks[0]]))

    return Skeleton

def zoneHasData(zone):
    if zone[3] != 'Zone_t': raise AttributeError('argument must be a zone')
    gcs = I.getNodesFromType1(zone, 'GridCoordinates_t')
    fss = I.getNodesFromType1(zone, 'FlowSolution_t')
    containers = gcs + fss
    if not containers: return False
    for container in containers:
        for data in I.getNodesFromType1(container,'DataArray_t'):
            if data[1] is not None: 
                return True
            
def ravelBCDataSet(t):
    # HACK https://elsa.onera.fr/issues/11219
    # HACK https://elsa-e.onera.fr/issues/10750
    for zone in I.getZones(t):
        for zbc in I.getNodesFromType1(zone,'ZoneBC_t'):
            for bc in I.getNodesFromType1(zbc,'BC_t'):
                for bcds in I.getNodesFromType1(bc,'BCDataSet_t'):
                    for bcd in I.getNodesFromType1(bcds,'BCData_t'):
                        for da in I.getNodesFromType1(bcd,'DataArray_t'):
                            if da[1] is not None:
                                da[1] = da[1].ravel(order='K')

def forceFamilyBCasFamilySpecified(t):
    # https://elsa.onera.fr/issues/10928
    for base in I.getBases(t):
        for zone in I.getZones(base):
            for ZoneBC in I.getNodesFromType1(zone,'ZoneBC_t'):
                for BC in I.getNodesFromType1(ZoneBC,'BC_t'):
                    FamilyNameNode = I.getNodeFromType1(BC,'FamilyName_t')
                    if FamilyNameNode is not None:
                        I.setValue(BC,'FamilySpecified')
                        FamilyName = I.getValue(FamilyNameNode)
                        if not I.getNodeFromName1(base,FamilyName):
                            FamilyAtBase = I.createNode(FamilyName,'Family_t',parent=base)
                            I.createNode('FamilyBC','FamilyBC_t',value='UserDefined',parent=FamilyAtBase)
                        continue

####################################################################################
# PYPART FUNCTIONS
####################################################################################
def saveWithPyPart(t, filename, coprocess_manager, tagWithIteration=False):
    '''
    Function to save a PyTree **t** with PyPart. The PyTree must have been
    splitted with PyPart in ``compute.py``. An important point is the presence
    in every zone of **t** of the special node ``:CGNS#Ppart``.

    Use this function to save ``'fields.cgns'``.

    .. note:: For more details on PyPart, see the dedicated pages on elsA
        support:
        `PyPart alone <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/PreprocessTutorials/etc_pypart_alone.html>`_
        and
        `PyPart with elsA <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/PreprocessTutorials/etc_pypart_elsa.html>`_

    Parameters
    ----------

        t : PyTree
            tree to save

        filename : str
            Name of the file

        tagWithIteration : bool
            if :py:obj:`True`, adds a suffix ``_AfterIter<iteration>``
            to the saved filename (creates a copy)
    '''
    # raise MolaException('Function saveWithPyPart is not implemented yet')

    PyPartBase = coprocess_manager.PyPartBase

    tpt = I.copyRef(t)
    removeNonLocalZones(tpt)
    I._rmNodesByName(tpt, '.Solver#Param')
    I._rmNodesByType(tpt, 'IntegralData_t')
    Cmpi.barrier()
    mola_logger.info(f'{CYAN}saving {filename}...{ENDC}', rank=0) #, end='')
    Cmpi.barrier()
    PyPartBase.mergeAndSave(tpt, 'PyPart_fields')
    Cmpi.barrier()
    if rank == 0:
        t_merged = C.convertFile2PyTree('PyPart_fields_all.hdf')
        # addLostFieldsExtractors(t_merged)
        migrateSolverOutputOfFlowSolutions(t, t_merged)
        I._rmNodesByName(t_merged, 'FlowSolution#EndOfRun*')
        ravelBCDataSet(t_merged)
        forceFamilyBCasFamilySpecified(t_merged) 
        C.convertPyTree2File(t_merged, filename)
        for fn in glob.glob('PyPart_fields_*.hdf'):
            try:
                os.remove(fn)
            except:
                pass

    mola_logger.info(f'{GREEN} OK{ENDC}', rank=0)
    Cmpi.barrier()
    if tagWithIteration and rank == 0:
        copyOutputFiles(filename)

def migrateSolverOutputOfFlowSolutions(t_dnr, t_rcv):
    # Required because of https://elsa.onera.fr/issues/11137
    zones = I.getZones(t_dnr)
    for zm in I.getZones(t_rcv):
        for z in zones:
            if z[0].startswith(zm[0]):
                all_fs  = I.getNodesFromType(z, 'FlowSolution_t')
                all_fsm = I.getNodesFromType(zm, 'FlowSolution_t')
                for fsm in all_fsm:
                    for fs in all_fs:
                        if fs[0] == fsm[0]:
                            SolverOutput = I.getNodeFromName(fs, '.Solver#Output')
                            if SolverOutput:
                                fsm[2].append( SolverOutput )
                                continue

####################################################################################
# END OF PYPART FUNCTIONS
####################################################################################

