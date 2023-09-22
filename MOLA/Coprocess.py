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
MOLA Coprocess module - designed to be used in coupling (trigger) elsA context

Recommended syntax for use:

::

    import MOLA.Coprocess as CO

23/12/2020 - L. Bernardos - creation by recycling
'''

import MOLA
from . import InternalShortcuts as J
from . import Preprocess as PRE
from . import JobManager as JM
from . import BodyForceTurbomachinery as BF
from . import Postprocess as POST

if not MOLA.__ONLY_DOC__:
    import os
    from datetime import datetime
    import numpy as np
    from scipy.ndimage.filters import uniform_filter1d
    import shutil
    import psutil
    import glob
    import copy
    import traceback
    from mpi4py import MPI
    comm   = MPI.COMM_WORLD
    rank   = comm.Get_rank()
    NumberOfProcessors = comm.Get_size()
    nbOfDigitsOfNProcs = int(np.ceil(np.log10(NumberOfProcessors+1)))

    import Converter.PyTree as C
    import Converter.Internal as I
    import Converter.Filter as Filter
    import Converter.Mpi as Cmpi
    import Transform.PyTree as T
    import Post.PyTree as P
    import Geom.PyTree as D



# ------------------------------------------------------------------ #
# Following variables should be overridden using compute.py and coprocess.py
# scripts
FULL_CGNS_MODE   = False
FILE_SETUP       = 'setup.py'
FILE_CGNS        = 'main.cgns'
FILE_SURFACES    = 'surfaces.cgns'
FILE_ARRAYS      = 'arrays.cgns'
FILE_FIELDS      = 'fields.cgns'
FILE_COLOG       = 'coprocess.log'
FILE_BODYFORCESRC= 'bodyforce.cgns'
DIRECTORY_OUTPUT = 'OUTPUT'
DIRECTORY_LOGS   = 'LOGS'
setup            = None
CurrentIteration = 0
elsAxdt          = None
PyPartBase       = None
EndOfRun         = True
# ------------------------------------------------------------------- #
FAIL  = '\033[91m'
GREEN = '\033[92m'
WARN  = '\033[93m'
MAGE  = '\033[95m'
CYAN  = '\033[96m'
ENDC  = '\033[0m'


def invokeCoprocessLogFile():
    '''
    This function creates the ``coprocess.log`` file used for monitoring the
    progress of the simulation.
    '''
    if rank > 0: return
    DATEANDTIME=datetime.now().strftime("%d/%m/%Y %H:%M:%S")
    with open(FILE_COLOG, 'w') as f:
        f.write('COPROCESS LOG FILE STARTED AT %s\n'%DATEANDTIME)

def printCo(message, proc=None, color=None):
    '''
    This function is used for easily writing messages in ``coproces.log`` file.
    It is designed to be used in a MPI context (coprocessing).

    Parameters
    ----------

        message : str
            Message to be written in ``coprocess.log`` file

        proc : int or None
            if provided, only local MPI rank will write the message.
            If :py:obj:`None`, all procs will write the message.

        color : str
            endscape code for terminal colored output.
            For example, red output is obtained like this: ``color='\\033[91m'``
    '''
    if proc is not None and rank != proc: return
    preffix = ('[{:0%d}]: '%nbOfDigitsOfNProcs).format(rank)
    if color:
        message = color+message+ENDC
    with open(FILE_COLOG, 'a') as f:
        f.write(preffix+message+'\n')

def extractFields(Skeleton):
    '''
    Extract the coupling CGNS PyTree from elsAxdt *OUTPUT_TREE* and make
    necessary adaptions, including migration of coordinates fields to
    GridCoordinates_t nodes, renaming of conventional fields names and
    adding the tree's Skeleton.

    Parameters
    ----------

        Skeleton : PyTree
            Skeleton tree as obtained from :py:func:`Converter.Mpi.convertFile2SkeletonTree`

    Returns
    -------

        t : PyTree
            Coupling adapted PyTree

    '''
    t = elsAxdt.get(elsAxdt.OUTPUT_TREE)
    adaptEndOfRun(t)
    resumeFieldsAveraging(Skeleton, t)
    t = I.merge([Skeleton, t])
    removeEmptyBCDataSet(t)
    PRE.forceFamilyBCasFamilySpecified(t) # HACK https://elsa.onera.fr/issues/10928
    ravelBCDataSet(t) # HACK https://elsa.onera.fr/issues/11219

    return t


def extractArrays(t, arrays, RequestedStatistics=[], Extractions=[],
                  addMemoryUsage=True, addResiduals=True):
    '''
    Extract the arrays (1D data) as a PyTree, including additional
    optional information requested by optional arguments of the function.

    Parameters
    ----------

        t : PyTree
            Output PyTree as obtained from :py:func:`extractFields`

        arrays : dict
            Dictionary of arrays, as created using :py:func:`invokeArrays`

        RequestedStatistics : :py:class:`list` of :py:class:`str`
            same argument as :py:func:`extractIntegralData`

        Extractions : :py:class:`list` of :py:class:`dict`
            .. note:: to be implemented

        addMemoryUsage : bool
            if :py:obj:`True`, register and add the current memory usage

        addResiduals : bool
            if :py:obj:`True`, add the residuals information

    Returns
    -------

        arraysTree : PyTree
            PyTree equivalent of the input :py:class:`dict` **arrays**
    '''
    if addResiduals: extractResiduals(t, arrays)
    if addMemoryUsage: addMemoryUsage2Arrays(arrays)
    extractIntegralData(t, arrays, RequestedStatistics=RequestedStatistics,
                         Extractions=Extractions)
    _scatterArraysFromRootToLocal(arrays)
    arraysTree = arraysDict2PyTree(arrays)

    return arraysTree


def extractSurfaces(t, Extractions, arrays=None):
    '''
    Extracts flowfield data as surfacic (or curvilinear) zones from input
    fields **t** and requested extraction information provided in **Extractions**.

    Parameters
    ----------

        t : PyTree
            Output PyTree as obtained from :py:func:`extractFields`

        Extractions : :py:class:`list` of :py:class:`dict`
            Each element of this list is a dictionary that specifies the kind of
            requested extraction. Possible keys of the dictionary are:

            * ``type`` : mandatory
                Can be one of :

                * ``AllBC<type>``
                    Extract all BC windows corresponding to a given *<type>*,
                    and stores them in separated CGNSBases using their families'
                    names.

                * ``BC<type>``
                    Extract all BC windows corresponding to a given *<type>*,
                    not necessarily defined by means of a family, and stores
                    them in a single CGNSBase.

                * ``FamilySpecified:<FamilyName>``
                    Extract BC windows defined by a family named *<FamilyName>*,
                    and stores them in a single CGNSBase.

                * ``IsoSurface``
                    Slice the input flowfields following a given field name and
                    value

                * ``Sphere``
                    Slice the input flowfields using a sphere defined using a
                    center and a radius.

                * ``Plane``
                    Slice the input flowfields using a plane defined using a
                    point and a normal direction.
                
                * ``Probe``
                    A probe to extract, defining a name, a location (x,y,z) and variables to extract. 
                    
                    Example: 

                    .. code-block:: python

                            dict(name='probeTest', location=(0.012, 0.24, -0.007), variables=['Temperature', 'Pressure'])

                    .. note:: If not provided, defaut name will be 'Probe_X_Y_Z'
              

            * ``name`` : :py:class:`str` (optional)
                If provided, this name replaces the default name of the CGNSBase
                container of the surfaces

                .. note:: not relevant if ``type`` starts with  ``AllBC``

            * ``field`` : :py:class:`str` (contextual)
                Name of the field employed for slicing if ``type`` = ``IsoSurface``

            * ``value`` : :py:class:`str` (contextual)
                Value of the field employed for slicing if ``type`` = ``IsoSurface``

            * ``center`` : :py:class:`list` of 3 :py:class:`float` (optional, contextual)
                Coordinates of the sphere center employed for slicing if
                ``type`` = ``Sphere``

            * ``radius`` : :py:class:`float` (contextual)
                Radius of the sphere employed for slicing if ``type`` = ``Sphere``

            * ``point`` : :py:class:`list` of 3 :py:class:`float` (contextual)
                Coordinates of the point employed for slicing if
                ``type`` = ``Plane``

            * ``normal`` : :py:class:`list` of 3 :py:class:`float` (contextual)
                Normal vector employed for slicing if ``type`` = ``Plane``

            * ``family`` : :py:class:`list` of 3 :py:class:`float` (contextual)
                Name of a zone Family. The extraction is performed only in zones
                with this ``FamilyName``.

            * ``AllowedFields`` : :py:class:`str`, :py:class:`list` of :py:class:`str` or :py:class:`bool`
                Specifies the allowed field names to be stored in the surfaces.
                If *AllowedFields* is ``'all'`` or :py:obj:`True`, then all fields
                are kept in surfaces (default behavior).
                If *AllowedFields* is a :py:class:`list` of :py:class:`str`, then 
                only fields with corresponding name are kept in surfaces, if 
                available.

                .. hint::
                    if *AllowedFields* is :py:obj:`False` or :py:obj:`None` or 
                    an empty list ``[]`` then only coordinates are kept in final
                    surfaces. This is optimum for plotting q-criterion iso-surfaces
                    without coloring.

                .. danger::
                    if you are making an Overset type of simulation, do not 
                    forget to include the field ``'cellN'`` to *AllowedFields*
                    if you wish to keep the blanking information

        arrays : dict
            if provided, some worfklow-specific operations may add quantities 
            to arrays using surfacic postprocessing. This is the case for 
            example for plane-to-plane turbomachine postprocessing.

    Returns
    -------

        SurfacesTree : PyTree
            Tree containing all requested surfaces as a set of zones stored
            in possibly different CGNSBases

    '''

    cellDimOutputTree = I.getZoneDim(I.getZones(t)[0])[-1]

    def addBase2SurfacesTree(basename):
        if not zones: return
        # Rename zones like the base
        for i, zone in enumerate(zones):
            # The name of the parent zone is kept in a temporary node .parentZone, 
            # that will be removed before saving
            # There might be a \ in zone name if it is a result of C.ExtractBCOfType
            zoneName = I.getName(zone).split('/')[0]
            I.createNode('.parentZone', 'UserDefinedData_t', value=zoneName, parent=zone)
            I.setName(zone, f'{basename}_R{rank}N{i}')
        base = I.newCGNSBase(basename, cellDim=cellDimOutputTree-1, physDim=3,
            parent=SurfacesTree)
        I._addChild(base, zones)
        J.set(base, '.ExtractionInfo', **ExtractionInfo)
        return base

    def keepOnlyAllowedFields(zones, AllowedFields):
        if not AllowedFields:
            I._rmNodesByType(zones,'FlowSolution_t')
        elif isinstance(AllowedFields, list):
            for zone in zones:
                for fs in I.getNodesFromType1(zone, 'FlowSolution_t'):
                    fields2remove = []
                    for field in I.getChildren(fs):
                        fieldName = I.getName(field)
                        fieldType = I.getType(field)
                        if fieldType == 'DataArray_t' and fieldName not in AllowedFields:
                            fields2remove += [ field ]
                    for field in fields2remove: I.rmNode(fs,field)

    I._rmNodesByName(t, 'FlowSolution#EndOfRun*')
    reshapeBCDatasetNodes(t)
    I._rmNodesByName(t, 'BCDataSet#Init') # see MOLA #75 and Cassiopee #10641
    DictBCNames2Type = C.getFamilyBCNamesDict(t)
    SurfacesTree = I.newCGNSTree()
    PartialTree = Cmpi.convert2PartialTree(t)

    # Prepare a base for Probes
    # FIXME Paraview cannot read probes with this implementation
    ProbesBase = I.newCGNSBase('Probes', cellDim=0, physDim=3)
    
    for Extraction in Extractions:
        TypeOfExtraction = Extraction['type']
        ExtractionInfo = copy.deepcopy(Extraction)

        if 'family' in Extraction:
            Tree4Extraction = I.copyTree(PartialTree)
            for base in I.getBases(Tree4Extraction):
                I._rmNodesByType1(base, 'Zone_t')
                basePartialTree = I.getNodeFromName1(PartialTree, I.getName(base))
                zones2keep = C.getFamilyZones(basePartialTree, Extraction['family'])
                for zone in zones2keep:
                    I._addChild(base, zone)
        else:
            Tree4Extraction = PartialTree

        if TypeOfExtraction.startswith('AllBC'):
            BCFilterName = TypeOfExtraction.replace('AllBC','')
            for BCFamilyName in DictBCNames2Type:
                BCType = DictBCNames2Type[BCFamilyName]
                if BCFilterName.lower() in BCType.lower():
                    zones = POST.extractBC(Tree4Extraction, Name='FamilySpecified:'+BCFamilyName)
                    ExtractionInfo['type'] = 'BC'
                    ExtractionInfo['BCType'] = BCType
                    addBase2SurfacesTree(BCFamilyName)

        elif TypeOfExtraction.startswith('BC'):
            zones = POST.extractBC(Tree4Extraction, Type=TypeOfExtraction)
            try: basename = Extraction['name']
            except KeyError: basename = TypeOfExtraction
            ExtractionInfo['type'] = 'BC'
            ExtractionInfo['BCType'] = TypeOfExtraction
            addBase2SurfacesTree(basename)

        elif TypeOfExtraction.startswith('FamilySpecified:'):
            zones = POST.extractBC(Tree4Extraction, Name=TypeOfExtraction)
            try: basename = Extraction['name']
            except KeyError: basename = TypeOfExtraction.replace('FamilySpecified:','')
            ExtractionInfo['type'] = 'BC'
            ExtractionInfo['BCType'] = TypeOfExtraction
            addBase2SurfacesTree(basename)

        elif TypeOfExtraction == 'IsoSurface':
            if Extraction['field'] in ['Radius', 'radius', 'CoordinateR']:
                C._initVars(Tree4Extraction, '{}=({{CoordinateY}}**2+{{CoordinateZ}}**2)**0.5'.format(Extraction['field']))
            container = deduceContainerForSlicing(Extraction)
            zones = POST.isoSurface(Tree4Extraction,
                                    fieldname=Extraction['field'],
                                    value=Extraction['value'],
                                    container=container)
            try: basename = Extraction['name']
            except KeyError:
                FieldName = Extraction['field'].replace('Coordinate','').replace('Radius', 'R').replace('ChannelHeight', 'H')
                basename = 'Iso_%s_%g'%(FieldName,Extraction['value'])
            addBase2SurfacesTree(basename)

        elif TypeOfExtraction == 'Sphere':
            try: center = Extraction['center']
            except KeyError: center = 0,0,0
            Eqn = '({x}-{x0})**2+({y}-{y0})**2+({z}-{z0})**2-{r}**2'.format(
                x='{CoordinateX}',y='{CoordinateY}',z='{CoordinateZ}',
                r=Extraction['radius'], x0=center[0],
                y0=center[1], z0=center[2])
            C._initVars(Tree4Extraction,'Slice=%s'%Eqn)
            zones = POST.isoSurface(Tree4Extraction,
                                    fieldname='Slice',
                                    value=0.0,
                                    container='FlowSolution')
            try: basename = Extraction['name']
            except KeyError: basename = 'Sphere_%g'%Extraction['radius']
            addBase2SurfacesTree(basename)

        elif TypeOfExtraction == 'Plane':
            n = np.array(Extraction['normal'])
            Pt = np.array(Extraction['point'])
            PlaneCoefs = n[0],n[1],n[2],-n.dot(Pt)
            C._initVars(Tree4Extraction,'Slice=%0.12g*{CoordinateX}+%0.12g*{CoordinateY}+%0.12g*{CoordinateZ}+%0.12g'%PlaneCoefs)
            zones = POST.isoSurface(Tree4Extraction,
                                    fieldname='Slice',
                                    value=0.0,
                                    container='FlowSolution')
            try: basename = Extraction['name']
            except KeyError: basename = 'Plane'
            addBase2SurfacesTree(basename)

        elif TypeOfExtraction == 'Probe':
            zone = D.point(Extraction['location'])
            I.setName(zone, Extraction['name'])
            J.set(zone, '.ExtractionInfo', **Extraction)
            I._addChild(ProbesBase, zone)
    
    Cmpi._convert2PartialTree(SurfacesTree)
    J.forceZoneDimensionsCoherency(SurfacesTree)
    Cmpi.barrier()
    restoreFamilies(SurfacesTree, t)
    I._rmNodesFromName(SurfacesTree, '.parentZone')
    I._rmNodesFromName(SurfacesTree, ':CGNS#Ppart')
    Cmpi.barrier()

    # Workflow specific postprocessings
    SurfacesTree = _extendSurfacesWithWorkflowQuantities(SurfacesTree, arrays=arrays)

    # Keep only allowed fields in surfaces
    for base in I.getBases(SurfacesTree):
        try: 
            ExtractionInfo = I.getNodeFromName1(base, '.ExtractionInfo')
            AllowedFieldsNode = I.getNodeFromName1(ExtractionInfo, 'AllowedFields')
            AllowedFields = I.getValue(AllowedFieldsNode).split()
        except:
            AllowedFields = True

        if isinstance(AllowedFields,str) and AllowedFields.lower() != 'all':
            AllowedFields = [ AllowedFields ]

        keepOnlyAllowedFields(I.getZones(base), AllowedFields)

    # Add probes if there are any
    if len(I.getZones(ProbesBase)) > 0:
        I._addChild(SurfacesTree, ProbesBase)



    return SurfacesTree

def deduceContainerForSlicing(Extraction):
    if 'field_container' in Extraction:
        return Extraction['field_container']

    elif Extraction['field'] in ['CoordinateX', 'CoordinateY', 'CoordinateZ']:
        return 'GridCoordinates'

    elif Extraction['field'] in ['Radius', 'radius', 'CoordinateR', 'Slice']:
        return 'FlowSolution#Height'

    elif Extraction['field'] == 'ChannelHeight':
        return 'FlowSolution#Height'
    
    else:
        return 'FlowSolution#Init'


def extractIntegralData(to, arrays, Extractions=[],
                        RequestedStatistics=['std-CL', 'std-CD']):
    '''
    Extract integral data from coupling tree **to**, and update **arrays** Python
    dictionary adding statistics requested by the user.

    Parameters
    ----------

        to : PyTree
            Coupling tree as obtained from :py:func:`adaptEndOfRun`

        arrays :dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array

            .. note:: **arrays** is modified in-place

        RequestedStatistics : :py:class:`list` of :py:class:`str`
            Here, the user requests the additional statistics to be computed.
            The syntax of each quantity must be as follows:

            ::

                '<preffix>-<integral_quantity_name>'

            `<preffix>` can be ``'avg'`` (for cumulative average) or ``'std'``
            (for standard deviation). ``<integral_quantity_name>`` can be any
            quantity contained in arrays, including other statistics.

            .. hint:: chaining preffixes is perfectly accepted, like
                ``'std-std-CL'`` which would compute the cumulative standard
                deviation of the cumulative standard deviation of the
                lift coefficient (:math:`\sigma(\sigma(C_L))`)

    '''
    IntegralDataNodes = I.getNodesFromType2(to, 'IntegralData_t')
    for IntegralDataNode in IntegralDataNodes:
        IntegralDataName = getIntegralDataName(IntegralDataNode)
        _appendIntegralDataNode2Arrays(arrays, IntegralDataNode)
        _extendArraysWithWorkflowQuantities(arrays, IntegralDataName)
        _extendArraysWithProjectedLoads(arrays, IntegralDataName) # TODO replace in _extendArraysWithWorkflowQuantities
        _normalizeMassFlowInArrays(arrays, IntegralDataName) # TODO replace in _extendArraysWithWorkflowQuantities
    for IntegralDataName in arrays:
        _extendArraysWithStatistics(arrays, IntegralDataName, RequestedStatistics)
    Cmpi.barrier()

def extractResiduals(to, arrays):
    '''
    Extract residuals from coupling tree **to**, and update **arrays** Python
    dictionary.

    Parameters
    ----------

        to : PyTree
            Coupling tree as obtained from :py:func:`adaptEndOfRun`

        arrays : dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array

            .. note:: **arrays** is modified in-place

    '''
    ConvergenceHistoryNodes = I.getNodesByType(to, 'ConvergenceHistory_t')
    ConvergenceDict = dict()
    for ConvergenceHistory in ConvergenceHistoryNodes:
        for DataArrayNode in I.getNodesFromType(ConvergenceHistory, 'DataArray_t'):
            DataArrayValue = I.getValue(DataArrayNode)
            if isinstance(DataArrayValue, int) or isinstance(DataArrayValue, float):
                DataArrayValue = np.array([DataArrayValue],dtype=type(DataArrayValue))
            if len(DataArrayValue.shape) == 1:
                ConvergenceDict[I.getName(DataArrayNode)] = DataArrayValue
        appendDict2Arrays(arrays, ConvergenceDict, I.getName(ConvergenceHistory))
        break # we ignore possibly multiple ConvergenceHistory_t nodes

def save(t, filename, tagWithIteration=False):
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
    if PyPartBase and filename.endswith(FILE_FIELDS):
        saveWithPyPart(t, filename, tagWithIteration=tagWithIteration)
        return

    t = I.copyRef(t) if I.isTopTree(t) else C.newPyTree(['Base', J.getZones(t)])
    Cmpi._convert2PartialTree(t)
    I._adaptZoneNamesForSlash(t)
    for z in I.getZones(t):
        SolverParam = I.getNodeFromName(z,'.Solver#Param')
        if not SolverParam or not I.getNodeFromName(SolverParam,'proc'):
            Cmpi._setProc(z, Cmpi.rank)
    I._rmNodesByName(t,'ID_*')
    I._rmNodesByType(t,'IntegralData_t')

    Skel = J.getStructure(t)

    UseMerge = False
    try:
        trees = comm.allgather( Skel )
        trees.insert( 0, t )
        tWithSkel = I.merge( trees )
        renameTooLongZones(tWithSkel)
        for l in 2,3: I._correctPyTree(tWithSkel,l) # unique base and zone names
    except SystemError:
        UseMerge = True
        printCo('Cmpi.KCOMM.gather FAILED. Using merge=True', color=J.WARN)
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

    printCo('will save %s ...'%filename,0, color=J.CYAN)
    Cmpi.convertPyTree2File(tWithSkel, filename, merge=UseMerge)
    printCo('... saved %s'%filename,0, color=J.CYAN)
    Cmpi.barrier()
    if tagWithIteration and Cmpi.rank == 0: copyOutputFiles(filename)

def saveWithPyPart(t, filename, tagWithIteration=False):
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

    tpt = I.copyRef(t)
    Cmpi._convert2PartialTree(tpt)
    I._rmNodesByName(tpt, '.Solver#Param')
    I._rmNodesByType(tpt, 'IntegralData_t')
    Cmpi.barrier()
    printCo('will save %s ...' % filename, 0, color=J.CYAN)
    Cmpi.barrier()
    PyPartBase.mergeAndSave(tpt, 'PyPart_fields')
    Cmpi.barrier()
    if rank == 0:
        t_merged = C.convertFile2PyTree('PyPart_fields_all.hdf')
        migrateSolverOutputOfFlowSolutions(tpt, t_merged)
        removeEmptyBCDataSet(t_merged)
        PRE.forceFamilyBCasFamilySpecified(t_merged) # https://elsa.onera.fr/issues/10928
        ravelBCDataSet(t_merged) # HACK https://elsa.onera.fr/issues/11219
        I._rmNodesByName(t_merged, 'FlowSolution#EndOfRun*')
        C.convertPyTree2File(t_merged, filename)
        for fn in glob.glob('PyPart_fields_*.hdf'):
            try:
                os.remove(fn)
            except:
                pass
    printCo('... saved %s' % filename, 0, color=J.CYAN)
    Cmpi.barrier()
    if tagWithIteration and rank == 0:
        copyOutputFiles(filename)

def saveWithPyPart_NEW(t, filename, tagWithIteration=False):
    '''
    .. danger::

        This function is still in development, and MPI deadlocks may happen depending of the
        parallelisation of the case. It should replace :py:func:`saveWithPyPart` once it is debuged.
        Should answer to issue #79.

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
    import Distributor2.PyTree as D2
    
    # Write PyPart files
    t = I.copyRef(t)
    Cmpi._convert2PartialTree(t)
    I._rmNodesByName(t, '.Solver#Param')
    I._rmNodesByType(t,'IntegralData_t')
    Cmpi.barrier()
    printCo('will save %s ...'%filename,0, color=J.CYAN)
    PyPartBase.mergeAndSave(t, 'PyPart_fields')
    Cmpi.barrier()
    # Read PyPart files in parallel 
    t = Cmpi.convertFile2SkeletonTree('PyPart_fields_all.hdf')
    t, stats = D2.distribute(t, NumberOfProcessors, useCom=0, algorithm='fast')
    t = Cmpi.readZones(t, 'PyPart_fields_all.hdf', rank=rank)
    Cmpi.barrier()
    # Remove PyPart files
    for fn in glob.glob('PyPart_fields_*.hdf'):
        try: os.remove(fn)
        except: pass
    # Write a unique file
    Cmpi._convert2PartialTree(t)
    Cmpi.barrier()
    Cmpi.convertPyTree2File(t, os.path.join(DIRECTORY_OUTPUT, FILE_FIELDS))
    printCo('... saved %s'%filename,0, color=J.CYAN)
    Cmpi.barrier()
    if tagWithIteration and rank == 0: copyOutputFiles(filename)

def moveTemporaryFile(temporary_file):
    '''
    Removes ``tmp-`` characters of given **temporary_file** using a *move*
    operation.

    Parameters
    ----------

        temporary_file : str
            path of the file whose name contains characters ``tmp-``

    '''
    final_file = temporary_file.replace('tmp-','')
    if final_file == temporary_file: return
    if Cmpi.rank == 0:
        printCo('deleting %s ...'%final_file)
        try:
            os.remove(final_file)
            WillReplace = True
        except:
            WillReplace = False

        if WillReplace:
            printCo('deleting %s ... OK'%final_file,color=GREEN)
            printCo('moving %s to %s ...'%(temporary_file,final_file))
            try:
                shutil.move(temporary_file,final_file)
                printCo('moving %s to %s ... OK'%(temporary_file,final_file),
                        color=GREEN)
            except:
                printCo('moving %s to %s ... FAILED'%(temporary_file,final_file),
                        color=FAIL)
        else:
            printCo('deleting %s ... FAILED'%final_file,color=FAIL)
    Cmpi.barrier()


def restoreFamilies(surfaces, skeleton):
    '''
    Restore families in the PyTree **surfaces** (e.g read from
    ``'surfaces.cgns'``) based on information in **skeleton** (e.g read from
    ``'main.cgns'``). Also add the ReferenceState to be able to use function
    computeVariables from Cassiopee Post module.

    .. tip:: **skeleton** may be a skeleton tree.

    Parameters
    ----------

        surfaces : PyTree
            tree where zone names are the same as in **skeleton** (or with a
            suffix in '\\<bcname>'), but without information on families and
            ReferenceState.

        skeleton : PyTree
            tree of the full 3D domain with zones, families and ReferenceState.
            No data is needed so **skeleton** may be a skeleton tree.
    '''
    I._adaptZoneNamesForSlash(surfaces)

    ReferenceState = I.getNodeFromType2(skeleton, 'ReferenceState_t') # To compute variables

    FamilyNodes = I.getNodesFromType2(skeleton, 'Family_t')

    for base in I.getNodesFromType1(surfaces, 'CGNSBase_t'):
        I.addChild(base, ReferenceState)
        familiesInBase = []
        for zone in I.getZones(base):
            zoneName = I.getValue(I.getNodeFromName1(zone, '.parentZone'))
            zoneInFullTree = I.getNodeFromNameAndType(skeleton, zoneName, 'Zone_t')
            
            # surface comes from extractBC => already contains all families
            if not zoneInFullTree: continue 
            
            fam = I.getNodeFromType1(zoneInFullTree, 'FamilyName_t')
            I.addChild(zone, fam)
            familiesInBase.append(I.getValue(fam))
        for family in FamilyNodes:
            if I.getName(family) in familiesInBase:
                I.addChild(base, family)

def monitorTurboPerformance(surfaces, arrays, RequestedStatistics=[]):
    '''
    Monitor performance (massflow in/out, total pressure ratio, total
    temperature ratio, isentropic efficiency) for each row in a compressor
    simulation. This processing is triggered if at least two bases in the PyTree
    **surfaces** fill the following requirements:

    #. there is a node ``'.ExtractionInfo'`` of type ``'UserDefinedData_t'``

    #. it contains a node ``'ReferenceRow'``, whose value is a :py:class:`str`
       corresponding to a row Family in ``'main.cgns'``.

    #. it contains a node ``'tag'``, whose value is a :py:class:`str` equal
       to ``'InletPlane'`` or ``'OutletPlane'``.

    .. note:: For one ``'ReferenceRow'``, the monitor is processed only if both
        ``'InletPlane'`` and ``'OutletPlane'`` are found.

    .. note:: These bases must contain variables ``'PressureStagnation'`` and
        ``'TemperatureStagnation'``

    .. important:: This function is adapted only to **Workflow Compressor** cases.

    Parameters
    ----------

        surfaces : PyTree
            as produced by :py:func:`extractSurfaces`

        arrays :dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array

        RequestedStatistics : :py:class:`list` of :py:class:`str`
            Here, the user requests the additional statistics to be computed.
            See documentation of function :py:func:`extractIntegralData` for
            more details.

    Returns
    -------

        arraysTree : PyTree
            PyTree equivalent of the :py:class:`dict` ``arrays``

    '''
    # FIXME: Segmentation fault bug when this function is used after
    #        POST.absolute2Relative (in co -proccessing only)
    def massflowWeightedIntegral(t, var):
        t = C.initVars(t, 'rou_var={MomentumX}*{%s}'%(var))
        C._initVars(t, 'rov_var={MomentumY}*{%s}'%(var))
        C._initVars(t, 'row_var={MomentumZ}*{%s}'%(var))
        integ  = abs(P.integNorm(t, 'rou_var')[0][0]) \
               + abs(P.integNorm(t, 'rov_var')[0][1]) \
               + abs(P.integNorm(t, 'row_var')[0][2])
        return integ

    def surfaceWeightedIntegral(t, var):
        integ  = P.integ(t, var)[0]
        return integ

    for row, rowParams in setup.TurboConfiguration['Rows'].items():

        planeUpstream   = I.newCGNSTree()
        planeDownstream = I.newCGNSTree()
        if 'PeriodicTranslation' in setup.TurboConfiguration:
            IsRotor = False  # Linear cascade
        elif not 'RotationSpeed' in rowParams:
            continue
        elif rowParams['RotationSpeed'] != 0:
            IsRotor = True
        else:
            IsRotor = False

        for base in I.getNodesFromType(surfaces, 'CGNSBase_t'):
            if I.getName(base) == 'RadialProfiles': continue

            try:
                ExtractionInfo = I.getNodeFromNameAndType(base, '.ExtractionInfo', 'UserDefinedData_t')
                ReferenceRow = I.getValue(I.getNodeFromName(ExtractionInfo, 'ReferenceRow'))
                tag = I.getValue(I.getNodeFromName(ExtractionInfo, 'tag'))
                if ReferenceRow == row and tag == 'InletPlane':
                    planeUpstream = base
                elif ReferenceRow == row and tag == 'OutletPlane':
                    planeDownstream = base
            except:
                pass

        if IsRotor:
            VarAndMeanList = [
                (['PressureStagnation', 'TemperatureStagnation'], massflowWeightedIntegral)
            ]
        else:
            VarAndMeanList = [
                (['PressureStagnation'], massflowWeightedIntegral),
                (['Pressure'], surfaceWeightedIntegral)
            ]

        dataUpstream   = integrateVariablesOnPlane(  planeUpstream, VarAndMeanList)
        dataDownstream = integrateVariablesOnPlane(planeDownstream, VarAndMeanList)

        if not dataUpstream or not dataDownstream:
            continue

        perfos = dict()
        if rank == 0:
            if 'PeriodicTranslation' in setup.TurboConfiguration:
                fluxcoeff = 1.
            else:
                fluxcoeff = rowParams['NumberOfBlades'] / float(rowParams['NumberOfBladesSimulated'])
            if IsRotor:
                perfos = computePerfoRotor(dataUpstream, dataDownstream, fluxcoeff=fluxcoeff)
            else:
                perfos = computePerfoStator(dataUpstream, dataDownstream, fluxcoeff=fluxcoeff)
        appendDict2Arrays(arrays, perfos, 'PERFOS_{}'.format(row))
        if perfos:
            _extendArraysWithStatistics(arrays, 'PERFOS_{}'.format(row), RequestedStatistics)

    arraysTree = arraysDict2PyTree(arrays)

    return arraysTree

def computePerfoRotor(dataUpstream, dataDownstream, fluxcoeff=1., fluxcoeffOut=None):

    if not fluxcoeffOut:
        fluxcoeffOut = fluxcoeff

    gamma = setup.FluidProperties['Gamma']

    # Compute total quantities ratio between in/out planes
    meanPtIn  =      dataUpstream['PressureStagnation'] /   dataUpstream['MassFlow']
    meanPtOut =    dataDownstream['PressureStagnation'] / dataDownstream['MassFlow']
    meanTtIn  =   dataUpstream['TemperatureStagnation'] /   dataUpstream['MassFlow']
    meanTtOut = dataDownstream['TemperatureStagnation'] / dataDownstream['MassFlow']
    PtRatio = meanPtOut / meanPtIn
    TtRatio = meanTtOut / meanTtIn
    # Compute Isentropic Efficiency
    etaIs = (PtRatio**((gamma-1.)/gamma) - 1.) / (TtRatio - 1.)

    perfos = dict(
        IterationNumber            = CurrentIteration-1,  # Because extraction before current iteration (next_state=16)
        MassFlowIn                 = dataUpstream['MassFlow']*fluxcoeff,
        MassFlowOut                = dataDownstream['MassFlow']*fluxcoeffOut,
        PressureStagnationRatio    = PtRatio,
        TemperatureStagnationRatio = TtRatio,
        EfficiencyIsentropic       = etaIs
    )

    return perfos

def computePerfoStator(dataUpstream, dataDownstream, fluxcoeff=1., fluxcoeffOut=None):

    if not fluxcoeffOut:
        fluxcoeffOut = fluxcoeff

    meanPtIn  =   dataUpstream['PressureStagnation'] /   dataUpstream['MassFlow']
    meanPtOut = dataDownstream['PressureStagnation'] / dataDownstream['MassFlow']
    meanPsIn  =             dataUpstream['Pressure'] /       dataUpstream['Area']

    perfos = dict(
        IterationNumber         = CurrentIteration-1,  # Because extraction before current iteration (next_state=16)
        MassFlowIn              = dataUpstream['MassFlow']*fluxcoeff,
        MassFlowOut             = dataDownstream['MassFlow']*fluxcoeffOut,
        PressureStagnationRatio = meanPtOut / meanPtIn,
        PressureStagnationLossCoeff = (meanPtIn - meanPtOut) / (meanPtIn - meanPsIn)
    )

    return perfos

def integrateVariablesOnPlane(surface, VarAndMeanList):
    '''
    Integrate variables on a surface.

    Parameters
    ----------

        surface : PyTree
            Surface for integration. Required variables must be present already
            in **surface**.

        VarAndMeanList : :py:class:`list` of :py:class:`tuple`
            List of 2-tuples. Each tuple associates:

                * a list of variables, that must be found in **surface**

                * a function to perform the weighted integration wished for
                  variables

            For example:

            ::

                VarAndMeanList = [([var1, var2], meanFunction1),
                                  ([var3], meanFunction2), ...]

            Example of function for a massflow weighted integration:

            ::

                import Converter.PyTree as C
                import Post.PyTree      as P
                def massflowWeightedIntegral(t, var):
                    t = C.initVars(t, 'rou_var={MomentumX}*{%s}'%(var))
                    integ  = abs(P.integNorm(t, 'rou_var')[0][0])
                    return integ

    Returns
    -------

        data : :py:class:`dict` or :py:obj:`None`
            dictionary that contains integrated values of variables. If
            **surface** is empty, does not contain a ``FlowSolution`` node or
            does not contains required variables, the function returns
            :py:obj:`None`.

    '''

    try:
        container_at_vertex = setup.PostprocessOptions['container_at_vertex']
        if isinstance(container_at_vertex, list):
            container_at_vertex = container_at_vertex[0]
    except (TypeError, AttributeError, KeyError):
        container_at_vertex = 'FlowSolution#InitV'

    surface = I.copyRef(surface)
    for zone in I.getZones(surface):
        fsToRemove = [n[0] for n in I.getNodesFromType1(zone,'FlowSolution_t') \
                      if n[0] != container_at_vertex]
        for fsname in fsToRemove:
            I._rmNodesByName(zone,fsname)

    # Convert to Tetra arrays for integration # TODO identify bug and notify
    surface = POST.convertToTetra(surface)
    
    previous_container = I.__FlowSolutionNodes__
    I.__FlowSolutionNodes__ = container_at_vertex

    check =  True
    data = dict()

    if I.getNodesFromType(surface, 'FlowSolution_t') == []:
        for var in ['MassFlow', 'Area']:
            data[var] = 0
        for varList, meanFunction in VarAndMeanList:
            for var in varList:
                data[var] = 0
    else:
        C._initVars(surface, 'ones=1')
        data['Area']     = abs(P.integNorm(surface, var='ones')[0][0])
        data['MassFlow'] = abs(P.integNorm(surface, var='MomentumX')[0][0]) \
                         + abs(P.integNorm(surface, var='MomentumY')[0][1]) \
                         + abs(P.integNorm(surface, var='MomentumZ')[0][2])
        try:
            for varList, meanFunction in VarAndMeanList:
                for var in varList:
                    data[var] = meanFunction(surface, var)
        except:
            # Variables cannot be found
            check = False

    # Check if the needed variables were extracted
    Cmpi.barrier()
    check = comm.allreduce(check, op=MPI.LAND) #LAND = Logical AND
    data['MassFlow'] = comm.allreduce(data['MassFlow'], op=MPI.SUM)
    data['Area'] = comm.allreduce(data['Area'], op=MPI.SUM)
    Cmpi.barrier()
    if not check or data['Area']==0:
        I.__FlowSolutionNodes__ = previous_container
        return None

    # MPI Reduction to sum quantities on proc 0
    Cmpi.barrier()
    for varList, meanFunction in VarAndMeanList:
        for var in varList:
            data[var] = comm.reduce(data[var], op=MPI.SUM, root=0)
    Cmpi.barrier()
    I.__FlowSolutionNodes__ = previous_container
    return data

def integrateVariablesOnVolume(t, VarAndMeanList, container='FlowSolution#Init', localComm=comm):
    '''
    Integrate variables on a volume.

    Parameters
    ----------

        t : PyTree or list of zones
            Tree for integration. Required variables must be present already
            in **t**.

        VarAndMeanList : :py:class:`list` of :py:class:`tuple`
            List of 2-tuples. Each tuple associates:

                * a list of variables, that must be found in **t**

                * a function to perform the weighted integration wished for
                  variables

            For example:

            ::

                VarAndMeanList = [([var1, var2], meanFunction1),
                                  ([var3], meanFunction2), ...]

            Example of function for a massflow weighted integration:

            ::

                import Converter.PyTree as C
                import Post.PyTree      as P
                def massflowWeightedIntegral(t, var):
                    t = C.initVars(t, 'rou_var={MomentumX}*{%s}'%(var))
                    integ  = abs(P.integNorm(t, 'rou_var')[0][0])
                    return integ
        
        container : str
            Name of the `FlowSolution_t` node with a ``GridLocation='CellCenter'`` on which perform the average.  
        
        localComm : MPI communicator
            local communicator for processors involved in this function (t may be a filtered tree, a zone or a list of zones)

    Returns
    -------

        data : :py:class:`dict` or :py:obj:`None`
            dictionary that contains integrated values of variables. If
            **t** is empty, does not contain a ``FlowSolution`` node or
            does not contains required variables, the function returns
            :py:obj:`None`.

    '''
    t = I.copyRef(t)
    previous_container = I.__FlowSolutionCenters__
    I.__FlowSolutionCenters__ = container

    check =  True
    data = dict()
    if I.getNodesFromType(t, 'FlowSolution_t') == []:
        data['Volume'] = 0
        for varList, meanFunction in VarAndMeanList:
            for var in varList:
                data[var] = 0
    else:
        C._initVars(t, 'centers:ones=1')
        data['Volume'] = P.integ(t, var='centers:ones')[0]
        try:
            for varList, meanFunction in VarAndMeanList:
                for var in varList:
                    data[var] = meanFunction(t, var)
        except:
            # Variables cannot be found
            check = False

    # Check if volume > 0
    localComm.barrier()
    data['Volume'] = localComm.allreduce(data['Volume'], op=MPI.SUM)
    localComm.barrier()
    if not check or data['Volume'] == 0:
        I.__FlowSolutionCenters__ = previous_container
        return None

    # MPI Reduction to sum quantities
    localComm.barrier()
    for varList, meanFunction in VarAndMeanList:
        for var in varList:
            data[var] = localComm.allreduce(data[var], op=MPI.SUM) 
    localComm.barrier()
    I.__FlowSolutionCenters__ = previous_container
    return data

def volumicAverage(t, container='FlowSolution#Init', localComm=comm):
    '''
    Perform a volumic average of quantities in the given **container**.

    Parameters
    ----------

        t : PyTree
            Input tree

        container : str
            Name of the `FlowSolution_t` node with a ``GridLocation='CellCenter'`` on which perform the average.

        localComm : MPI communicator
            local communicator for processors involved in this function (t may be a filtered tree, a zone or a list of zones) 

    Returns
    -------

        dict

            Dictionary with averaged quantities, plus the volume ('Volume') of the domain.
    '''
    def volumicIntegral(t, var):
        integ  = P.integ(t, var)[0]
        return integ

    # Get the list of variables in the first zone with the given container
    varList = []
    for zone in I.getZones(t):
        if I.getNodeFromName1(zone, container):
            varList = J.getVars2Dict(zone, Container=container)
            break
    # Rename (cell-centered) variables for Cassiopee
    varList = [f'centers:{v}' for v in varList]

    # integratedData = integrateVariablesOnVolume(t, [(varList, volumicIntegral)], container=container, localComm=localComm)
    integratedData = dict((var, 0.) for var in varList)
    integratedData['Volume'] = 1.
    # localComm.barrier()

    averagedData = dict()
    for k, v in integratedData.items():
        k = k.replace('centers:', '')
        if k == 'Volume':
            averagedData[k] = v
        else:
            averagedData[k] = v / integratedData['Volume']

    return averagedData

def updateAndWriteSetup(setup):
    '''
    This function is used for adapting ``setup.py`` information for a new run.

    Parameters
    ---------

        setup : module
            Python module object as obtained from command

            >>> import setup
    '''
    if rank == 0:
        printCo('updating setup.py ...', proc=0, color=GREEN)
        if CurrentIteration > 1:
            setup.elsAkeysNumerics['niter'] -= CurrentIteration - setup.elsAkeysNumerics['inititer'] + 1
            setup.elsAkeysNumerics['inititer'] = CurrentIteration + 1 
            if 'itime' in setup.elsAkeysNumerics:
                setup.elsAkeysNumerics['itime'] = CurrentIteration * setup.elsAkeysNumerics['timestep']
        PRE.writeSetupFromModuleObject(setup)
        printCo('updating setup.py ... OK', proc=0, color=GREEN)
    comm.Barrier()

def invokeArrays():
    '''
    Create **arrays** Python dictionary by reading any pre-existing data
    contained in ``OUTPUT/arrays.cgns``

    .. note:: an empty dictionary is returned if no ``OUTPUT/arrays.cgns`` file
        is found

    Returns
    -------

        arrays :dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array
    '''
    Cmpi.barrier()
    try: os.remove('COMPLETED')
    except: pass
    arrays = dict()
    FullPathArraysFile = os.path.join(DIRECTORY_OUTPUT, FILE_ARRAYS)
    ExistingArraysFile = os.path.exists(FullPathArraysFile)
    Cmpi.barrier()
    inititer = setup.elsAkeysNumerics['inititer']
    if ExistingArraysFile and inititer>1:
        if rank==0:
            t = C.convertFile2PyTree(FullPathArraysFile)

            for zone in I.getZones(t):
                ZoneName = I.getName(zone)
                VarNames, = C.getVarNames(zone, excludeXYZ=True)
                FlowSol_n = I.getNodeFromName1(zone, 'FlowSolution')
                arrays[ZoneName] = dict()
                arraysSubset = arrays[ZoneName]
                if FlowSol_n:
                    for VarName in VarNames:
                        Var_n = I.getNodeFromName1(FlowSol_n, VarName)
                        if Var_n:
                            arraysSubset[VarName] = Var_n[1]

                try:
                    iters = np.copy(arraysSubset['IterationNumber'])
                    for VarName in arraysSubset:
                        arraysSubset[VarName] = arraysSubset[VarName][iters<inititer]
                except KeyError:
                    pass

    Cmpi.barrier()

    return arrays

def addMemoryUsage2Arrays(arrays):
    '''
    This function adds or updates a component in **arrays** for monitoring the
    employed memory. Only nodes are monitored (not every single proc, as this
    would produce redundant information). The number of cores contained in each
    computational node is retreived from the environment variable
    ``SLURM_CPUS_ON_NODE``.

    If this information does not exist, a value of ``28`` is taken by default.

    Parameters
    ----------

        arrays :dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array

            parameter **arrays** is modified
    '''
    CoreNumberPerNode = int(os.getenv('SLURM_CPUS_ON_NODE', 28))

    if rank%CoreNumberPerNode == 0:
        ZoneName = 'MemoryUsageOfProc%d'%rank
        UsedMemory = psutil.virtual_memory().used
        UsedMemoryPctg = psutil.virtual_memory().percent
        try:
            ArraysItem = arrays[ZoneName]
        except KeyError:
            arrays[ZoneName] = dict(IterationNumber=np.array([],dtype=int),
                                   UsedMemoryInPercent=np.array([],dtype=float),
                                   UsedMemory=np.array([],dtype=float),)
            ArraysItem = arrays[ZoneName]

        try:
            ArraysItem['IterationNumber'] = np.hstack((
                ArraysItem['IterationNumber'].ravel(order='F'),
                int(CurrentIteration))).ravel(order='F')
            ArraysItem['UsedMemoryInPercent'] = np.hstack((
                ArraysItem['UsedMemoryInPercent'].ravel(order='F'),
                float(UsedMemoryPctg))).ravel(order='F')
            ArraysItem['UsedMemory'] = np.hstack((
                ArraysItem['UsedMemory'].ravel(order='F'),
                float(UsedMemory))).ravel(order='F')
        except KeyError:
            del arrays[ZoneName]
    Cmpi.barrier()

def arraysDict2PyTree(arrays):
    '''
    This function converts the **arrays** Python dictionary to a PyTree (CGNS)
    structure **t**.

    Parameters
    ----------

        arrays :dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array

    Returns
    -------

        t : PyTree
            same information as input, but structured in a PyTree CGNS form

    .. warning:: after calling the function, **arrays** and **t** do *NOT*
        share memory, which means that modifications on **arrays** will not
        affect **t** and vice-versa
    '''
    zones = []
    for ZoneName in arrays:
        arraysSubset = arrays[ZoneName]
        Arrays, Vars = [], []
        OrderedVars = [var for var in arraysSubset]

        OrderedVars.sort()
        for var in OrderedVars:
            Vars.append(var)
            Arrays.append(arraysSubset[var])

        zone = J.createZone(ZoneName, Arrays, Vars)
        if zone: zones.append(zone)

    if zones:
        Cmpi._setProc(zones, rank)
        t = C.newPyTree(['Base', zones])
        for base in I.getBases(t): I._sortByName(base)

        for zone in zones:
            zone_name = I.getName(zone)
            for e in setup.Extractions:
                if e['type'] == 'Probe' and e['name'] == zone_name:
                    J.set(zone, '.ExtractionInfo', **e)
                    break
            
    else:
        t = C.newPyTree(['Base'])
    Cmpi.barrier()

    return t

def appendDict2Arrays(arrays, dictToAppend, basename):
    '''
    This function add data defined in **dictToAppend** in the base **basename**
    of **arrays**.

    Parameters
    ----------

        arrays :dict
            Contains integral data in the following form:

            >>> arrays[basename]['VariableName'] = np.array

        dictToAppend : dict
            Contains data to append in **arrays**. For each element:
                * key is the variable name
                * value is the associated value

        basename : str
            Name of the base in which values will be appended.

    '''

    if not dictToAppend or arrays is None: return

    if not basename in arrays: arrays[basename] = dict()

    for var, value in dictToAppend.items():
        np_value = np.array([value],ndmin=1).ravel(order='F')
        if var in arrays[basename]:
            arrays[basename][var] = np.hstack(
                                        (arrays[basename][var].ravel(order='F'),
                                         np_value )
                                              ).ravel(order='F')
        else:
            arrays[basename][var] = np_value


def _scatterArraysFromRootToLocal(arrays):
    local_keys = [k for k in arrays]

    # for debug
    # for r in range(NumberOfProcessors):
    #     Cmpi.barrier()
    #     if r==rank: printCo('local_keys= %s'%str(local_keys),color=J.WARN)
    #     Cmpi.barrier()

    all_local_keys = comm.gather(local_keys, 0)
    all_local_keys = comm.bcast(all_local_keys, 0)
    comm.barrier()
    i = -1
    for rank_recv, local_keys in enumerate(all_local_keys):
        for lk in local_keys:
            i +=1
            if rank==0:
                if rank_recv == 0:
                    root_item = arrays[lk]
                else:
                    obj = arrays[lk] if lk in arrays else None
                    comm.send(obj, rank_recv, tag=i)
                    if obj: del arrays[lk]
            
            if rank != 0:
                if rank == rank_recv:
                    root_item = comm.recv(source=0, tag=i)
                    if not root_item: continue # nothing to append

                else:
                    continue # nothing to append

            if lk not in arrays: continue 
            
            rootHasItNb = True if 'IterationNumber' in root_item else False

            try:
                localHasItNb = True if 'IterationNumber' in arrays[lk] else False
            except:
                printCo(traceback.format_exc(),color=J.FAIL)
                printCo('FATAL: failed processing signal %s'%lk)
                printCo('arrays keys are: %s'%','.join([k for k in arrays]),color=J.FAIL)
                os._exit(0)

            override_all = False
            if rootHasItNb and localHasItNb:
                RegisteredIterations = root_item['IterationNumber'].ravel(order='F')
                IterationNumber = arrays[lk]['IterationNumber'].ravel(order='F')

                try:
                    if IterationNumber[0] <= RegisteredIterations[0]:
                        override_all = True

                    if not override_all:
                        eps = 1e-12
                        UpdatePortion = IterationNumber > (RegisteredIterations[-1] + eps)
                        FirstIndex2Update = np.where(UpdatePortion)[0][0]

                except:
                    printCo(traceback.format_exc(),color=J.FAIL)
                    printCo('processing signal %s'%lk,color=J.FAIL)
                    printCo('RegisteredIterations',color=J.FAIL)
                    printCo(str(RegisteredIterations),color=J.FAIL)
                    printCo(str(type(RegisteredIterations)),color=J.FAIL)
                    printCo('IterationNumber',color=J.FAIL)
                    printCo(str(IterationNumber),color=J.FAIL)
                    os._exit(0)
            else:
                FirstIndex2Update = 0
            
            for var, value in root_item.items():
                if var in arrays[lk]:
                    if override_all:
                        arrays[lk][var] = np.array([value],ndmin=1).ravel(order='F')
                    else:
                        arrays[lk][var] = np.hstack((value, arrays[lk][var][FirstIndex2Update:])).ravel(order='F')
                else:
                    arrays[lk][var] = np.array([value],ndmin=1).ravel(order='F')


def _appendIntegralDataNode2Arrays(arrays, IntegralDataNode):
    '''
    Beware: this is a private function, employed by :py:func:`extractIntegralData`

    This function converts the CGNS IntegralDataNode (as provided by elsA)
    into the Python dictionary structure of arrays dictionary, and append it
    to the latter.

    Parameters
    ----------

        arrays :dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array

        IntegralDataNode : node
            Contains integral data as provided by elsA
    '''
    IntegralData = dict()
    for DataArrayNode in I.getChildren(IntegralDataNode):
        IntegralData[I.getName(DataArrayNode)] = I.getValue(DataArrayNode)
    IntegralDataName = getIntegralDataName(IntegralDataNode)
    IterationNumber = IntegralData['IterationNumber']

    try:
        arraysSubset = arrays[IntegralDataName]
    except KeyError:
        arrays[IntegralDataName] = dict()
        arraysSubset = arrays[IntegralDataName]


    try: RegisteredIterations = arraysSubset['IterationNumber']
    except KeyError: RegisteredIterations = np.array([])
    if len(RegisteredIterations) > 0:
        PreviousRegisteredArrays = True
        eps = 1e-12
        UpdatePortion = IterationNumber > (RegisteredIterations[-1] + eps)
        try: FirstIndex2Update = np.where(UpdatePortion)[0][0]
        except IndexError: return
    else:
        PreviousRegisteredArrays = False

    for integralKey in IntegralData:
        if PreviousRegisteredArrays:
            PreviousArray = arraysSubset[integralKey]
            AppendArray = IntegralData[integralKey][FirstIndex2Update:]
            arraysSubset[integralKey] = np.hstack((PreviousArray, AppendArray))
        else:
            arraysSubset[integralKey] = np.array(IntegralData[integralKey],
                                               order='F', ndmin=1)

def _normalizeMassFlowInArrays(arrays, IntegralDataName):
    '''
    Beware: this is a private function, employed by :py:func:`extractIntegralData`

    If:

        * the variable 'convflux_ro' is in **arrays[IntegralDataName]** (massflow
          extracted by elsA on the Family **IntegralDataName**),

        * and it exists a **FluxCoef** associted to **IntegralDataName** in
          ** ReferenceValues**, written in:

          >>> setup.ReferenceValues['NormalizationCoefficient'][IntegralDataName]['FluxCoef']

    Then the variable 'MassFlow' is added in **arrays[IntegralDataName]** by
    multiplying 'convflux_ro' by this **FluxCoef**.

    Parameters
    ----------

        arrays : dict
            Contains integral data in the following form:

            >>> np.array = arrays['FamilyBCNameOrElementName']['VariableName']

        IntegralDataName : str
            Name of the IntegralDataNode (CGNS) provided by elsA. It is used as
            key for arrays dictionary.
    '''
    arraysSubset = arrays[IntegralDataName]
    try:
        FluxCoef = setup.ReferenceValues['NormalizationCoefficient'][IntegralDataName]['FluxCoef']
        arraysSubset['MassFlow'] = arraysSubset['convflux_ro'] * FluxCoef
    except:
        return

def _extendArraysWithProjectedLoads(arrays, IntegralDataName):
    '''
    Beware: this is a private function, employed by :py:func:`extractIntegralData`

    This function is employed for adding aerodynamic-relevant coefficients to
    the arrays dictionary. The new quantites are the following :

        elsA Extractions  ---->   New projections (aero coefficients)
        -------------------------------------------------------------
        MomentumXFlux               CL   (force coef following LiftDirection)
        MomentumYFlux               CD   (force coef following DragDirection)
        MomentumZFlux               CY   (force coef following SideDirection)
        TorqueX                     Cn   (torque coef around LiftDirection)
        TorqueY                     Cl   (torque coef around DragDirection)
        TorqueZ                     Cm   (torque coef around SideDirection)


    The aerodynamic coefficients are dimensionless. They are computed using
    the scales and origins 'FluxCoef', 'TorqueCoef' and 'TorqueOrigin' contained
    in the setup.py dictionary,
                ReferenceValues['NormalizationCoefficient'][<IntegralDataName>]

    where <IntegralDataName> is the name of the component.
    If they are not provided, then global scaling factors are taken from
    ReferenceValues of setup.py,

                        ReferenceValues['FluxCoef']
                        ReferenceValues['TorqueCoef']
                        ReferenceValues['TorqueOrigin']


    ***************************************************************************
    * VERY IMPORTANT NOTE : in order to obtain meaningful Cn, Cl and Cm       *
    * coefficients, elsA integral torques (TorqueX, TorqueY and TorqueZ) must *
    * be applied at absolute zero coordinates (0,0,0), which means that       *
    * special elsA CGNS keys xtorque, ytorque and ztorque must all be 0.      *
    * Also, all Momentum and Torque fluxes shall be dimensional, which means  *
    * that special elsA CGNS keys fluxcoeff and torquecoeff must be exactly 1.*
    * Indeed, proper scaling and normalization is done using the setup.py     *
    * ReferenceValues, as indicated previously.                               *
    ***************************************************************************

    Parameters
    ----------

        arrays : dict
            Contains integral data in the following form:

            >>> np.array = arrays['FamilyBCNameOrElementName']['VariableName']

        IntegralDataName : str
            Name of the IntegralDataNode (CGNS) provided by elsA. It is used as
            key for arrays dictionary.
    '''


    DragDirection=np.array(setup.ReferenceValues['DragDirection'],dtype=np.float64)
    SideDirection=np.array(setup.ReferenceValues['SideDirection'],dtype=np.float64)
    LiftDirection=np.array(setup.ReferenceValues['LiftDirection'],dtype=np.float64)

    arraysSubset = arrays[IntegralDataName]

    try:
        FluxCoef = setup.ReferenceValues['NormalizationCoefficient'][IntegralDataName]['FluxCoef']
        TorqueCoef = setup.ReferenceValues['NormalizationCoefficient'][IntegralDataName]['TorqueCoef']
        TorqueOrigin = setup.ReferenceValues['NormalizationCoefficient'][IntegralDataName]['TorqueOrigin']
    except:
        FluxCoef = setup.ReferenceValues['FluxCoef']
        TorqueCoef = setup.ReferenceValues['TorqueCoef']
        TorqueOrigin = setup.ReferenceValues['TorqueOrigin']

    try:
        FX = arraysSubset['MomentumXFlux']
        FY = arraysSubset['MomentumYFlux']
        FZ = arraysSubset['MomentumZFlux']
        MX = arraysSubset['TorqueX']
        MY = arraysSubset['TorqueY']
        MZ = arraysSubset['TorqueZ']
    except KeyError:
        return # no required fields for computing external aero coefficients

    # Pole change
    # TODO make ticket for elsA concerning CGNS parsing of xtorque ytorque ztorque
    TX = MX-(TorqueOrigin[1]*FZ - TorqueOrigin[2]*FY)
    TY = MY-(TorqueOrigin[2]*FX - TorqueOrigin[0]*FZ)
    TZ = MZ-(TorqueOrigin[0]*FY - TorqueOrigin[1]*FX)

    arraysSubset['CL']=FX*LiftDirection[0]+FY*LiftDirection[1]+FZ*LiftDirection[2]
    arraysSubset['CD']=FX*DragDirection[0]+FY*DragDirection[1]+FZ*DragDirection[2]
    arraysSubset['CY']=FX*SideDirection[0]+FY*SideDirection[1]+FZ*SideDirection[2]
    arraysSubset['Cn']=TX*LiftDirection[0]+TY*LiftDirection[1]+TZ*LiftDirection[2]
    arraysSubset['Cl']=TX*DragDirection[0]+TY*DragDirection[1]+TZ*DragDirection[2]
    arraysSubset['Cm']=TX*SideDirection[0]+TY*SideDirection[1]+TZ*SideDirection[2]

    # Normalize forces and moments
    for Force in ('CL','CD','CY'):  arraysSubset[Force]  *= FluxCoef
    for Torque in ('Cn','Cl','Cm'): arraysSubset[Torque] *= TorqueCoef

def _extendArraysWithWorkflowQuantities(arrays, IntegralDataName):
    try: Workflow = setup.Workflow
    except AttributeError: return

    if Workflow == 'Propeller':
        from . import WorkflowPropeller as WP
        WP._extendArraysWithPropellerQuantities(arrays, IntegralDataName, setup)



def _extendArraysWithStatistics(arrays, IntegralDataName, RequestedStatistics):
    '''
    Beware: this is a private function, employed by :py:func:`extractIntegralData`

    Add to arrays dictionary the relevant statistics requested by the user
    through the RequestedStatistics list of special named strings.

    Parameters
    ----------

        arrays : dict
            Contains integral data in the following form:
            ::
                >>> np.array = arrays['FamilyBCNameOrElementName']['VariableName']

        IntegralDataName : str
            Name of the IntegralDataNode (CGNS) provided by elsA. It is used as
            key for loads dictionary.

        RequestedStatistics : :py:class:`list` of :py:class:`str`
            Desired statistics to infer from loads dictionary. For more
            information see documentation of function
            :py:func:`extractIntegralData`
    '''

    AvgIt = setup.ReferenceValues["CoprocessOptions"]["AveragingIterations"]

    arraysSubset = arrays[IntegralDataName]
    try:
        IterationNumber = arraysSubset['IterationNumber']
    except BaseException as e:
        return # this is the case for GlobalConvergenceHistory at present
    IterationWindow = len(IterationNumber[IterationNumber>(IterationNumber[-1]-AvgIt)])
    if IterationWindow < 2: return

    for StatKeyword in RequestedStatistics:
        KeywordsSplit = StatKeyword.split('-')
        StatType = KeywordsSplit[0]
        VarName = '-'.join(KeywordsSplit[1:])
        if VarName not in arraysSubset: continue

        try:
            InstantaneousArray = arraysSubset[VarName]
            InvalidValues = np.logical_not(np.isfinite(InstantaneousArray))
            InstantaneousArray[InvalidValues] = 0.
        except:
            continue

        if StatType.lower() == 'avg':
            StatisticArray = sliddingAverage(InstantaneousArray, IterationWindow)

        elif StatType.lower() == 'std':
            avg = sliddingAverage(InstantaneousArray, IterationWindow)
            arraysSubset['avg-'+VarName] = avg
            StatisticArray = sliddingSTD(InstantaneousArray, IterationWindow, avg=avg)

        elif StatType.lower() == 'rsd':
            avg = sliddingAverage(InstantaneousArray, IterationWindow)
            arraysSubset['avg-'+VarName] = avg
            std = sliddingSTD(InstantaneousArray, IterationWindow, avg=avg)
            arraysSubset['std-'+VarName] = std
            StatisticArray = sliddingRSD(InstantaneousArray, IterationWindow,
                                        avg=avg, std=std)

        arraysSubset[StatKeyword] = StatisticArray

def sliddingAverage(array, window):
    '''
    Compute the slidding average of the signal

    Parameters
    ----------

        array : numpy.ndarray
            input signal

        window : int
            length of the slidding window

    Returns
    -------

        average : numpy.ndarray
            sliding average
    '''
    average = uniform_filter1d(array, size=window)
    InvalidValues = np.logical_not(np.isfinite(average))
    average[InvalidValues] = 0.
    return average

def sliddingSTD(array, window, avg=None):
    '''
    Compute the slidding standard deviation of the signal

    Parameters
    ----------

        array : numpy.ndarray
            input signal

        window : int
            length of the slidding window

        avg : numpy.ndarray or :py:obj:`None`
            slidding average of **array** on the same **window**. If
            :py:obj:`None`, it is computed

    Returns
    -------

        std : numpy.ndarray
            sliding standard deviation
    '''
    if avg is None:
        avg = sliddingAverage(array, window)

    AvgSqrd = uniform_filter1d(array**2, size=window)

    InvalidValues = np.logical_not(np.isfinite(AvgSqrd))
    AvgSqrd[InvalidValues] = 0.
    AvgSqrd[AvgSqrd<0] = 0.

    std = np.sqrt(np.abs(AvgSqrd - avg**2))

    return std

def sliddingRSD(array, window, avg=None, std=None):
    '''
    Compute the relative slidding standard deviation of the signal

    .. math::

        rsd = std / avg

    Parameters
    ----------

        array : numpy.ndarray
            input signal

        window : int
            length of the slidding window

        average : numpy.ndarray or :py:obj:`None`
            slidding average of **array** on the same **window**. If
            :py:obj:`None`, it is computed

        std : numpy.ndarray or :py:obj:`None`
            slidding standard deviation of **array** on the same **window**. If
            :py:obj:`None`, it is computed

    Returns
    -------

        rsd : numpy.ndarray
            sliding relative standard deviation
    '''
    if avg is None:
        avg = sliddingAverage(array, window)
    if std is None:
        std = sliddingSTD(array, window, avg)

    rsd = std / np.abs(avg)

    InvalidValues = np.logical_not(np.isfinite(rsd))
    rsd[InvalidValues] = 0.

    return rsd

def getIntegralDataName(IntegralDataNode):
    '''
    Transforms the elsA provided **IntegralDataNode** name into a suitable name
    for further storing it at **arrays** dictionary.

    Parameters
    ----------

        IntegralDataNode : node
            Contains integral data as provided by elsA

    Returns
    -------

        IntegralDataName : str
            name of the integral quantity
    '''
    return I.getName(IntegralDataNode).split('-')[0]


def isConverged(ConvergenceCriteria):
    '''
    This method is used to determine if the current simulation is converged by
    looking at user-provided convergence criteria.
    If converged, the signal returns :py:obj:`True` to all ranks and writes a
    message to ``coprocess.log`` file.

    Parameters
    ----------

        ConvergenceCriteria : :py:class:`list` of :py:class:`dict`
            Each :py:class:`dict` corresponds to a criterion. Its has the
            following keys:

            * ``Family``: Name of the zone to monitor (shall exist in
              ``arrays.cgns``)

            * ``Variable``: Name of the variable to monitor on ``Family``

            * ``Threshold``: Value of the threshold to consider. The current
              criterion is satisfied if the value of the last element of
              ``Variable`` on ``Family`` in ``arrays.cgns`` is lower than
              ``Threshold``.

            * ``Condition`` (optinal, 'Necessary' by default): logical
              requirement for the current criterion. To verify convergence,
              criteria tagged 'Necessary' must all be satisfied simultaneously
              and at least one criterion tagged 'Sufficient' must be satisfied.
              For instance, if CN1 and CN2 are 'Necessary' and CS1 and CS2 are
              'Sufficient', convergence is reached when:
              (CN1 AND CN2) AND (CS1 OR CS2)

    Returns
    -------

        CONVERGED : bool
            :py:obj:`True` if the convergence criteria are satisfied
    '''
    if not ConvergenceCriteria:
        return False

    CONVERGED = False
    if rank == 0:
        AllNecessaryCriteria = True
        OneSufficientCriterion = True
        # Default value of Condition = 'Necessary'
        for criterion in ConvergenceCriteria:
            if 'Condition' not in criterion:
                criterion['Condition'] = 'Necessary'
            elif criterion['Condition'] == 'Sufficient':
                OneSufficientCriterion = False
        try:
            arraysTree = C.convertFile2PyTree(os.path.join(DIRECTORY_OUTPUT, FILE_ARRAYS))
            arraysZones = I.getZones(arraysTree)
            for criterion in ConvergenceCriteria:
                if OneSufficientCriterion and criterion['Condition'] == 'Sufficient':
                    continue
                zone, = [z for z in arraysZones if z[0] == criterion['Family']]
                Flux, = J.getVars(zone, [criterion['Variable']])
                if Flux is None and criterion['Condition'] == 'Necessary':
                    printCo('WARNING: requested convergence variable %s not found in %s'%(criterion['Variable'],criterion['Family']),color=WARN)
                    AllNecessaryCriteria = False
                    continue
                criterion['FoundValue'] = Flux[-1]
                IsSatisfied = criterion['FoundValue'] < criterion['Threshold']
                if criterion['Condition'] == 'Necessary' and not IsSatisfied:
                    AllNecessaryCriteria = False
                    break
                elif criterion['Condition'] == 'Sufficient' and IsSatisfied:
                    OneSufficientCriterion = criterion['Variable']

            CONVERGED = OneSufficientCriterion and AllNecessaryCriteria
            if CONVERGED:
                MSG = 'CONVERGED at iteration {} since:'.format(CurrentIteration-1)
                for criterion in ConvergenceCriteria:
                    if criterion['Condition'] == 'Necessary' \
                        or criterion['Variable'] == OneSufficientCriterion:
                        MSG += '\n  {}={} < {} on {} ({})'.format(criterion['Variable'],
                                                               criterion['FoundValue'],
                                                               criterion['Threshold'],
                                                               criterion['Family'],
                                                               criterion['Condition'])
                printCo('*******************************************',color=GREEN)
                printCo(MSG, color=GREEN)
                printCo('*******************************************',color=GREEN)

        except BaseException as e:
            printCo("isConverged failed: {}".format(e),color=FAIL)


    Cmpi.barrier()
    CONVERGED = comm.bcast(CONVERGED,root=0)

    return CONVERGED


def hasReachedTimeOutMargin(ElapsedTime, TimeOut, MarginBeforeTimeOut):
    '''
    This function returns :py:obj:`True` to all processors if the margin before
    time-out is satisfied. Otherwise, it returns :py:obj:`False` to all processors.

    Parameters
    ----------

        ElapsedTime : float
            total elapsed time in seconds since the launch of ``compute.py``
            elsA script.

        TimeOut : float
            total time-out in seconds. It shall correspond to the
            remaining time of a slurm job before forced termination.

        MarginBeforeTimeOut : float
            Margin in seconds before time out is reached. Shall account for safe
            postprocessing operations before the job triggers forced termination

    Returns
    -------

        ReachedTimeOutMargin : bool
            :py:obj:`True` if

            ::

                ElapsedTime >= (TimeOut - MarginBeforeTimeOut)

            Otherwise, returns :py:obj:`False`.

            .. note:: the same value (:py:obj:`True` or :py:obj:`False`) is sent to *all*
                processors.
    '''
    ReachedTimeOutMargin = False
    if rank == 0:
        ReachedTimeOutMargin = ElapsedTime >= (TimeOut - MarginBeforeTimeOut)
        if ReachedTimeOutMargin:
            printCo('REACHED MARGIN BEFORE TIMEOUT at %s'%datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                            proc=0, color=WARN)
    comm.Barrier()
    ReachedTimeOutMargin = comm.bcast(ReachedTimeOutMargin,root=0)

    return ReachedTimeOutMargin


def copyOutputFiles(*files2copy):
    '''
    Copy the files provided as input *(comma-separated variables)* by addding to
    their name ``'_AfterIter<X>'`` where ``<X>`` will be replaced with the
    corresponding interation

    Parameters
    ----------

        file2copy : comma-separated :py:class:`str`
            file(s) name(s) to copy at ``OUTPUT`` directory.

    Examples
    --------

    ::

        copyOutputFiles('surfaces.cgns','arrays.cgns')

    '''
    for file2copy in files2copy:
        f2cSplit = file2copy.split('.')
        newFileName = '{name}_AfterIter{it}.{fmt}'.format(
                        name='.'.join(f2cSplit[:-1]),
                        it=CurrentIteration,
                        fmt=f2cSplit[-1])
        try:
            shutil.copy2(file2copy, newFileName)
        except:
            pass


def computeTransitionOnsets(to):
    '''
    Extracts the airfoil's top and bottom side transition onset locations.

    .. important:: This function is adapted only to **Workflow Airfoil** cases.

    Parameters
    ----------

        to : PyTree
            Coupling tree as obtained from `adaptEndOfRun`

    Returns
    -------

        XtrTop : :py:class:`float`
            X-coordinate location of the transition onset location
            at the top side of the airfoil

        XtrBottom : :py:class:`float`
            X-coordinate location of the transition onset location
            at the bottom side of the airfoil
    '''
    XtrTop, XtrBottom = np.nan, np.nan
    surf   = boundaryConditions2Surfaces(to, onlyWalls=True)
    Cmpi._convert2PartialTree(surf)
    surf   = T.merge(surf)
    Slices = P.isoSurfMC(surf,'intermittency',value=0.5)

    # Really required ?
    TransitionPoints = T.splitConnexity(Slices) if len(Slices) > 0 else None

    # Each proc send to rank=0 the transition points found
    comm.Barrier()
    if rank == 0:
        TransitionPoints = comm.gather(TransitionPoints,0)
    else:
        comm.gather(TransitionPoints,0)

    # Post process the transition onset (rank=0)
    if rank == 0:
        TransPts2postProcess = []
        for p in TransitionPoints:
            if p is not None:
                isStdNode = I.isStdNode(p)
                if isStdNode == 0:
                    TransPts2postProcess += p
                elif isStdNode == -1:
                    TransPts2postProcess += [p]
        QtyOfTransitionOnsetPoints = len(TransPts2postProcess)
        if QtyOfTransitionOnsetPoints != 2:
            print(J.WARN+'WARNING: unexpected number of Transition onset points (%d, expected 2). Replacing by NaN'%QtyOfTransitionOnsetPoints+J.ENDC)
            TopPointXY = BottomPointXY = [np.array([np.nan]), np.array([np.nan])]
        else:
            TopPointXY    = J.getxy(TransPts2postProcess[0])
            BottomPointXY = J.getxy(TransPts2postProcess[1])

            if TopPointXY[1][0] < BottomPointXY[1][0]:
                TopPointXY, BottomPointXY = BottomPointXY, TopPointXY

        XtrTop    = TopPointXY[0][0]
        XtrBottom = BottomPointXY[0][0]
    comm.Barrier()

    return XtrTop, XtrBottom


def addArrays(arrays, ZoneName, ListOfArraysNames, NumpyArrays):
    '''
    This function is an interface for adding new user-defined arrays into the
    **arrays** Python dictionary.

    Parameters
    ----------

        arrays :dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array

            parameter **arrays** is modified

        ZoneName : str
            Name of the existing or new component where new arrays are going
            to be added. (FamilyBC or Component name)

        ListOfArraysNames : :py:class:`list` of :py:class:`str`
            Each element of this list is a name of the new arrays to be added.
            For example:

            ::

                ['MyFirstLoad', 'AnotherLoad']

        NumpyArrays : :py:class:`list` of numpy 1d arrays
            Values to be added to **arrays**

            .. attention::
                All arrays provided to **NumpyArrays** *(which belongs to the
                same component)* must have exactly the same number of elements
    '''
    try:
        arraysSubset = arrays[ZoneName]
    except KeyError:
        arrays[ZoneName] = {}
        arraysSubset = arrays[ZoneName]
        for array, name in zip(NumpyArrays, ListOfArraysNames):
            arraysSubset[name] = array.flatten()
        return

    for array, name in zip(NumpyArrays, ListOfArraysNames):
        try:
            ExistingArray = arraysSubset[name]
        except KeyError:
            arraysSubset[name] = array.flatten()
            continue

        arraysSubset[name] = np.hstack((ExistingArray.flatten(), array.flatten()))


def addBodyForcePropeller2Arrays(arrays, BodyForceDisks):
    '''
    This function is an interface adapted to body-force computations.
    It transfers the integral information of each body-force disk into
    the **arrays** dictionary.

    The fields that are appended to **arrays** are:

    ::

        ['Thrust', 'RPM', 'Power', 'Pitch']

    hence, these values must exist in ``.Info`` CGNS node of each CGNS zone
    contained in **BodyForceDisks**

    .. note:: The new component of load dictionary has the same name as its
        corresponding body-force disk.

    Parameters
    ----------

        arrays :dict
            Contains integral data in the following form:

            >>> arrays['FamilyBCNameOrElementName']['VariableName'] = np.array

            parameter **arrays** is modified

        BodyForceDisks : :py:class:`list` of zone
            Current bodyforce disks as obtained from
            :py:func:`MOLA.LiftingLine.computePropellerBodyForce`
    '''
    Cmpi.barrier()
    for BodyForceDisk in BodyForceDisks:
        BodyForceDiskName = I.getName( BodyForceDisk )
        Loads = J.get(BodyForceDisk, '.AzimutalAveragedLoads')
        if Loads:
            LoadsNames = [k for k in Loads]
            LoadsArrays = [np.array([Loads[k]]) for k in Loads]
            Commands = J.get(BodyForceDisk, '.Commands')
            if Commands:
                CommandsNames = [k for k in Commands]
                CommandsArrays = [np.array([Commands[k]]) for k in Commands]
                LoadsNames.extend(CommandsNames)
                LoadsArrays.extend(CommandsArrays)

            LoadsNames.append('IterationNumber')
            LoadsArrays.append(np.array([CurrentIteration]))
            addArrays(arrays, BodyForceDiskName, LoadsNames, LoadsArrays)
    Cmpi.barrier()


def getSignal(filename):
    '''
    Get a signal using an temporary auxiliar file technique.

    If the intermediary file exists (signal received) then it is removed, and
    the function returns :py:obj:`True` to all processors. Otherwise, it returns
    :py:obj:`False` to all processors.

    This function is employed for controling a simulation in a simple manner,
    for example using UNIX command ``touch``:

    .. code-block:: bash

        touch filename

    at the same directory where :py:func:`getSignal` is called.

    Parameters
    ----------

        filename : str
            the name of the file (the signal keyword)

    Returns
    -------

        isOrder : bool
            :py:obj:`True` if the signal is received, otherwise :py:obj:`False`, to all
            processors
    '''
    isOrder = False
    if rank == 0:
        try:
            os.remove(filename)
            isOrder = True
            printCo(CYAN+"Received signal %s"%filename+ENDC, proc=0)
        except:
            pass
    comm.Barrier()
    isOrder = comm.bcast(isOrder,root=0)
    return isOrder


def adaptEndOfRun(to):
    '''
    This function is used to make adaptations of the coupling trigger tree
    provided by elsA. The following operations are performed:

    * ``GridCoordinates`` node is created from ``FlowSolution#EndOfRun#Coords``
    * adapt name of masking field (``cellnf`` is renamed as ``cellN``)
    * rename ``FlowSolution#EndOfRun`` as ``FlowSolution#Init``

    Parameters
    ----------

         to : PyTree
            Coupling tree as obtained from function

            >>> elsAxdt.get(elsAxdt.OUTPUT_TREE)

            .. note:: tree **to** is modified
    '''
    moveCoordsFromEndOfRunToGridCoords(to)
    I._renameNode(to, 'cellnf', 'cellN')
    I._renameNode(to, 'FlowSolution#EndOfRun', 'FlowSolution#Init')

def moveCoordsFromEndOfRunToGridCoords(to):
    '''
    This function is used to make adaptations of the coupling trigger tree
    provided by elsA. The following operations are performed:

    * ``GridCoordinates`` node is created from ``FlowSolution#EndOfRun#Coords``


    Parameters
    ----------

         to : PyTree
            Coupling tree as obtained from function

            >>> elsAxdt.get(elsAxdt.OUTPUT_TREE)

            .. note:: tree **to** is modified
    '''
    FScoords = I.getNodeFromName(to, 'FlowSolution#EndOfRun#Coords')
    if FScoords:
        I._renameNode(to,'FlowSolution#EndOfRun#Coords','GridCoordinates')
        for GridCoordsNode in I.getNodesFromName3(to, 'GridCoordinates'):
            GridLocationNode = I.getNodeFromType1(GridCoordsNode, 'GridLocation_t')
            if I.getValue(GridLocationNode) != 'Vertex':
                zone = I.getParentOfNode(to, GridCoordsNode)
                ERRMSG = ('Extracted coordinates of zone '
                          '%s must be located in Vertex')%I.getName(zone)
                raise ValueError(FAIL+ERRMSG+ENDC)
            I.rmNode(to, GridLocationNode)
            I.setType(GridCoordsNode, 'GridCoordinates_t')
    Cmpi.barrier()

def boundaryConditions2Surfaces(to, onlyWalls=True):
    '''
    Extract the BC data contained in the coupling tree as a PyTree.

    Parameters
    ----------

        to : PyTree
            Coupling tree as obtained from :py:func:`adaptEndOfRun`

        onlyWalls : bool
            if :py:obj:`True`, only BC with keyword ``'wall'`` contained in
            their type are extracted. Otherwise, all BC are extracted,
            regardless of their type.

    Returns
    -------

        BCs : PyTree
            PyTree with one base by BC Family. Include fields stored in
            FlowSolution containers
    '''
    Cmpi.barrier()
    tR = I.renameNode(to, 'FlowSolution#Init', 'FlowSolution#Centers')
    I.__FlowSolutionCenters__ = 'FlowSolution#Centers'
    # TODO make it multi-container of FlowSolution
    DictBCNames2Type = C.getFamilyBCNamesDict(tR)

    BCs = I.newCGNSTree()
    for FamilyName in DictBCNames2Type:
        BCType = DictBCNames2Type[FamilyName]
        BC = None
        if onlyWalls:
            if 'wall' in BCType.lower():
                BC = C.extractBCOfName(tR,'FamilySpecified:'+FamilyName)
        else:
            BC = C.extractBCOfName(tR,'FamilySpecified:'+FamilyName)
        if BC:
            base = I.newCGNSBase(FamilyName, 2, 3, parent=BCs)
            for bc in BC:
                I.addChild(base, bc)

    Cmpi.barrier()

    return BCs

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
                raise ValueError(FAIL+ERRMSG+ENDC)
            I.setName(zone, newName)

def reshapeBCDatasetNodes(to):
    '''
    ..warning:: this is a private function, employed by :py:func:`saveSurfaces`

    This function checks the shape of DataArray in all ``BCData_t`` nodes in a
    PyTree **to**.
    For some unknown reason, the extraction of BCData throught a BCDataSet is
    done sometimes in unstructured 1D shape, so it is not consistent with BC
    PointRange. If so, this function reshape the DataArray to the BC shape.
    Link to ``Anomaly #6186`` (see <https://elsa-e.onera.fr/issues/6186>`_)

    Parameters
    ----------

        to : PyTree
            PyTree to check.

            .. note:: tree **to** is modified
    '''
    def _getBCShape(bc):
        PointRange_n = I.getNodeFromName(bc, 'PointRange')
        if not PointRange_n: return
        PointRange = I.getValue(PointRange_n)
        imin = PointRange[0, 0]
        imax = PointRange[0, 1]
        jmin = PointRange[1, 0]
        jmax = PointRange[1, 1]
        kmin = PointRange[2, 0]
        kmax = PointRange[2, 1]
        if imin == imax:
            bc_shape = (jmax-jmin, kmax-kmin)
        elif jmin == jmax:
            bc_shape = (imax-imin, kmax-kmin)
        else:
            bc_shape = (imax-imin, jmax-jmin)
        return bc_shape

    for bc in I.getNodesFromType(to, 'BC_t'):
        bc_shape = _getBCShape(bc)
        if not bc_shape: continue
        for BCData in I.getNodesFromType(bc, 'BCData_t'):
            for node in I.getNodesFromType(BCData, 'DataArray_t'):
                value = I.getValue(node)
                if isinstance(value, np.ndarray):
                    if value.shape != bc_shape:
                        I.setValue(node, value.reshape(bc_shape, order='F'))


def getOption(OptionName, default=None):
    '''
    This function is an interface for easily accessing the values of dictionary
    ``ReferenceValues['CoprocessOptions']`` contained in ``setup.py``.

    Parameters
    ----------

        OptionName : str
            Name of the key to return.

        default
            value to return if **OptionName** key is not present in
            ``ReferenceValues['CoprocessOptions']`` dictionary

    Returns
    -------

        value
            the value contained in
            ``ReferenceValues['CoprocessOptions'][<OptionName>]`` or the default
            value if key is not found
    '''
    try:
        value = setup.ReferenceValues['CoprocessOptions'][OptionName]
    except KeyError:
        value = default

    return value

def write4Debug(MSG):
    '''
    This function allows for writing into separated files (one per processor)
    named ``LOGS/rank<P>.log`` where ``<P>`` is replaced with the proc number.
    It is useful for debugging.

    Parameters
    ----------

        MSG : str
            Message to append into appropriate file following the local
            proc number.
    '''
    with open('LOGS/rank%d.log'%rank,'a') as f: f.write('%s\n'%MSG)

def loadSkeleton(Skeleton=None, PartTree=None):
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

    return Skeleton

def splitWithPyPart():
    '''
    Use PyPart to split the mesh in ``main.cgns``. This function should be use
    in ``compute.py`` to prepare the mesh before calling ``elsAxdt.XdtCGNS()``.

    .. note:: For more details on PyPart, see the dedicated pages on elsA
        support:
        `PyPart alone <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/PreprocessTutorials/etc_pypart_alone.html>`_
        and
        `PyPart with elsA <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/PreprocessTutorials/etc_pypart_elsa.html>`_

    .. important:: Dependence to ETC module

    Returns
    -------

        t : PyTree
            Split tree, merged with the skeleton. It will be the **tree**
            argument of ``elsAxdt.XdtCGNS()`` in ``compute.py``

        Skeleton : PyTree
            Skeleton tree to use in ``coprocess.py``

        PyPartBase : PyPart object
            PyPart objet that is mandatory to use its method mergeAndSave latter

        Distribution : dict
            Correspondence between zones and processors.

        PartTree : PyTree
            Required for adequately saving tree using PyPart (see https://elsa.onera.fr/issues/11149)

    '''

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
    is_unsteady = setup.elsAkeysNumerics['time_algo'] != 'steady'
    try:
        avg_requested = setup.ReferenceValues['CoprocessOptions']['FirstIterationForFieldsAveraging'] is not None
    except:
        avg_requested = False

    if is_unsteady and avg_requested:
        printCo('WARNING: removing "ZoneBCGT", but this may cause deadlock at fields.cgns save: https://elsa.onera.fr/issues/11149#note-11',0,J.WARN)
        I._rmNodesByName(Skeleton,'ZoneBCGT') # https://elsa.onera.fr/issues/11149
    Distribution = PyPartBase.getDistribution()

    # Put Distribution into the Skeleton
    for zone in I.getZones(Skeleton):
        zonePath = I.getPath(Skeleton, zone, pyCGNSLike=True)[1:]
        Cmpi._setProc(zone, Distribution[zonePath])

    t = I.merge([Skeleton, PartTree])

    Skeleton = loadSkeleton(Skeleton, PartTree)
    # Add empty Coordinates for skeleton zones
    # Needed to make Cmpi.convert2PartialTree work
    for zone in I.getZones(Skeleton):
        GC = I.getNodeFromType1(zone, 'GridCoordinates_t')
        if not GC:
            J.set(zone, 'GridCoordinates', childType='GridCoordinates_t',
                CoordinateX=None, CoordinateY=None, CoordinateZ=None)
        elif I.getZoneType(zone) == 2:
            # HACK For unstructured zone, correct the node NFaceElements/ElementConnectivity
            # Problem with PyPart: see issue https://elsa-e.onera.fr/issues/9002
            # C._convertArray2NGon(zone)
            NFaceElements = I.getNodeFromName(zone, 'NFaceElements')
            if NFaceElements:
                node = I.getNodeFromName(NFaceElements, 'ElementConnectivity')
                I.setValue(node, np.abs(I.getValue(node)))

    if 'CoupledSurfaces' in setup.ReferenceValues['CoprocessOptions']:
        # This part is linked to the WorkflowAerothermalCoupling
        # For unstructured zones, AdditionnalFamilyName nodes are lost
        # See Anomaly #10494 on elsA support # TODO: Fixed since elsA v5.2.01
        # We need to restore them
        for i, famBCTrigger in enumerate(setup.ReferenceValues['CoprocessOptions']['CoupledSurfaces']):
            surfaceName = 'ExchangeSurface{}'.format(i)
            for zone in I.getZones(t):
                if I.getZoneType(zone) == 2:
                    for BC in C.getFamilyBCs(t, famBCTrigger):
                        I.createChild(BC, 'SurfaceName', 'AdditionalFamilyName_t', value=surfaceName)

    return t, Skeleton, PyPartBase, Distribution

def moveLogFiles():
    if Cmpi.rank == 0:
        try: os.makedirs(DIRECTORY_LOGS)
        except: pass

        for fn in glob.glob('*.log'):
            FilenameBase = fn[:-4]
            i = 1
            NewFilename = FilenameBase+'-%d'%i+'.log'
            while JM.fileExists('LOGS', NewFilename):
                i += 1
                NewFilename = FilenameBase+'-%d'%i+'.log'

            shutil.move(fn, os.path.join('LOGS', NewFilename))
        for fn in glob.glob('elsA_MPI*'):
            shutil.move(fn, os.path.join('LOGS', fn))

    Cmpi.barrier()

def createSymbolicLink(src, dst):
    if Cmpi.rank == 0:
        J.createSymbolicLink(src, dst)


def updateBodyForce(t, previousTreeWithSourceTerms=[]):
    '''
    In a turbomachinery context, update the source terms for body-force modelling.
    The user-defined parameter **BodyForceInputData** (in `setup.py`) controls the 
    behavior of this function.

    The optional parameter **BodyForceInitialIteration** (=1 by default) may be modified in
    `CoprocessOptions`.

    For each row modelled with body force, the following parameters are optional:

    * **relax** (=0.5 by default): Relaxation coefficient for the source terms. 
       Should be less than 1 (the new source terms are equal to the previous ones).

    * **rampIterations** (=50 by default): Number of iterations (starting from **BodyForceInitialIteration**)
       to activate body force progressively, with a coefficient ramping from 0 to 1.

    Parameters
    ----------
    
        t : PyTree
            Output PyTree as obtained from :py:func:`extractFields`

        previousTreeWithSourceTerms : PyTree
            Previous output of this function. It is used for relaxing the source terms, depending 
            on the value of the **relax** argument. 
            The first time this function is called, this parameter may be initialized with an empty list.

    Returns
    -------
        newTreeWithSourceTerms: PyTree
            _description_
    '''

    def getBodyForceZones(t, BodyForceFamily):
        '''
        Return zones in BodyForceFamily and processors that manage them, and create also 
        a sub-commnunicator with these processors
        '''
        zones = []
        for zone in C.getFamilyZones(t, BodyForceFamily):
            DataSourceTermNode = I.getNodeByName1(zone, 'FlowSolution#DataSourceTerm')
            if DataSourceTermNode:
                zones.append(zone)

        # Create a subcommunicator for processors managing t
        procDict = Cmpi.getProcDict(zones)
        procList = list(set(p for p in procDict.values()))
        # printCo(f'{procList}', rank, J.MAGE)
        newGroup = comm.group.Incl(procList)
        subComm = comm.Create_group(newGroup)

        return zones, subComm, procList
    
    # def splitCommunicatorByBodyForceFamily(t, BodyForceFamilies):
    #     zones = []
    #     color = -1
    #     for famColor, BodyForceFamily in enumerate(BodyForceFamilies):
    #         for zone in C.getFamilyZones(t, BodyForceFamily):
    #             DataSourceTermNode = I.getNodeByName1(zone, 'FlowSolution#DataSourceTerm')
    #             if DataSourceTermNode:
    #                 zones.append(zone)
    #                 if color == -1: 
    #                     color = famColor
    #                 else:
    #                     assert color == famColor

    #     commBF = comm.Split(color=color, key=0)
    #     return zones, commBF

    def compute_optimal_relaxation(NewSourceTerms, previousSourceTerms, relax):
        # Optimal relaxation coefficient
        NormOfNewSourceTerms = sum([x**2 for x in NewSourceTerms.values()])**0.5
        NormOfPreviousSourceTerms = sum([x**2 for x in previousSourceTerms.values()])**0.5
        num = np.amax( np.absolute(NormOfNewSourceTerms - NormOfPreviousSourceTerms) )
        den = np.amax( np.absolute(NormOfNewSourceTerms + NormOfPreviousSourceTerms) )
        if den != 0:
            relax_optim  =  num / den 
            relax = min(max(relax_optim, relax), 0.999) # must be between relax and 0.999
        else:
            relax = 0.999
        printCo(f'  relax = {relax}', 0, J.MAGE)
        return relax

    printCo('Update body force...', 0, color=J.CYAN)

    newTreeWithSourceTerms = I.copyRef(t)
    
    BodyForceInitialIteration = setup.ReferenceValues['CoprocessOptions'].get('BodyForceInitialIteration', 1)

    for BodyForceComponent in setup.BodyForceInputData:

        BFtype = BodyForceComponent.get('type', 'AnalyticalByFamily')
        assert BFtype =='AnalyticalByFamily', 'Body-force "type" must be "AnalyticalByFamily" for now'

        BodyForceFamily = BodyForceComponent['Family']
        BodyForceParameters = copy.deepcopy(BodyForceComponent['BodyForceParameters'])
        CouplingOptions = copy.deepcopy(BodyForceComponent.get('CouplingOptions', dict()))

        relax = CouplingOptions.get('relax', 0.5)
        BodyForceFinalIteration = BodyForceInitialIteration + CouplingOptions.get('rampIterations', 50.)
        coeff_eff = J.rampFunction(BodyForceInitialIteration, BodyForceFinalIteration, 0., 1.)

        # Create a local communicator active only for processors that manage zones in BodyForceFamily
        # BFzones, subComm, procList = getBodyForceZones(newTreeWithSourceTerms, BodyForceFamily)
        # BodyForceParameters['communicator'] = subComm
        # if rank not in procList:
        #     # Only processors involved in the computation of source terms for BodyForceFamily
        #     # read lines that follow in the loop
        #     continue

        BFzones = []
        for zone in I.getZones(newTreeWithSourceTerms):
            DataSourceTermNode = I.getNodeByName1(zone, 'FlowSolution#DataSourceTerm')
            if DataSourceTermNode is not None:
                BFzones.append(zone)

        # Add FluidProperties, ReferenceValues and TurboConfiguration in BodyForceParameters 
        # This latter could be a dict or a list of dict
        if isinstance(BodyForceParameters, list):
            for BFParams in BodyForceParameters:
                BFParams['FluidProperties'] = setup.FluidProperties
                BFParams['ReferenceValues'] = setup.ReferenceValues
                BFParams['TurboConfiguration'] = setup.TurboConfiguration
        else:
            BodyForceParameters['FluidProperties'] = setup.FluidProperties
            BodyForceParameters['ReferenceValues'] = setup.ReferenceValues
            BodyForceParameters['TurboConfiguration'] = setup.TurboConfiguration

        NewSourceTermsGlobal = BF.computeBodyForce(BFzones, BodyForceParameters)

        for zone in BFzones:

            DataSourceTermNode = I.getNodeByName1(zone, 'FlowSolution#DataSourceTerm')
            NewSourceTerms = NewSourceTermsGlobal[I.getName(zone)]

            for key, value in NewSourceTerms.items():
                NewSourceTerms[key] = coeff_eff(CurrentIteration) * value

            FSSourceTerm = I.newFlowSolution('FlowSolution#SourceTerm', gridLocation='CellCenter', parent=zone)
            SourceTermPath = I.getPath(newTreeWithSourceTerms, FSSourceTerm)

            # Get previous source terms
            previousSourceTerms = dict()
            if CurrentIteration > BodyForceInitialIteration and CurrentIteration > setup.elsAkeysNumerics['inititer']: 
                previousFSSourceTerm = I.getNodeFromPath(previousTreeWithSourceTerms, SourceTermPath)
                for name in NewSourceTerms:
                    previousSourceTerms[name] = I.getValue(I.getNodeFromName(previousFSSourceTerm, name))
            else:
                for name in NewSourceTerms:
                    previousSourceTerms[name] = 0.
            
            # # Optimal relaxation coefficient
            # relax = compute_optimal_relaxation(NewSourceTerms, previousSourceTerms, relax)

            ActiveSourceTermNode = I.getNodeFromName1(DataSourceTermNode, 'ActiveSourceTerm')
            if ActiveSourceTermNode:
                ActiveSourceTerm = I.getValue(ActiveSourceTermNode)
            else:
                ActiveSourceTerm = 1
                
            for name in NewSourceTerms:
                newSourceTerm = (1-relax) * NewSourceTerms[name] + relax * previousSourceTerms[name]
                newSourceTerm *= ActiveSourceTerm
                I.newDataArray(name=name, value=newSourceTerm, parent=FSSourceTerm)
            
            # subComm.Free()

    I._rmNodesByName(newTreeWithSourceTerms, 'FlowSolution#Init')
    I._rmNodesByName(newTreeWithSourceTerms, 'FlowSolution#DataSourceTerm')
    I._rmNodesByName(newTreeWithSourceTerms, 'FlowSolution#tmpMOLAFlow')
    Cmpi.barrier()
    return newTreeWithSourceTerms


#_______________________________________________________________________________
# PROBES MANAGEMENT
#_______________________________________________________________________________
def hasProbes():
    for Extraction in setup.Extractions:
        if Extraction['type'] == 'Probe':
            return True
    return False

def appendProbes2Arrays_extractMesh(t, arrays, Probes, order=2):
    '''
    Parameter
    ---------

        t : PyTree

        arrays : dict

        Probes :
            :py:class:`dict` of the form:

            >>> Probes = dict( probeName1=(x1,y1,z1), ... )

        order : int
            order of interpolation
    '''
    import Post.Mpi as Pmpi

    t = Cmpi.convert2PartialTree(t)
    I._renameNode(t, 'FlowSolution#Init', 'FlowSolution#Centers')
    I._rmNodesByName(t, I.__FlowSolutionNodes__)

    probesTree = I.newCGNSTree()
    probesBase = I.newCGNSBase('PROBES', parent=probesTree, cellDim=0)
    for probeName, location in Probes.items():
        probe = D.point(location)
        I.setName(probe, probeName)
        I._addChild(probesBase, probe)

    P._extractMesh(t, probesTree, mode='accurate', order=order, constraint=0, extrapOrder=0)  # use a hook ? or Pmpi ?

    # Delete empty probes
    for zone in I.getZones(probesTree):
        Density = I.getNodeFromName(zone, 'Density')
        if not Density or I.getValue(Density) == 0:
            # Probe is not in this zone
            I._rmNode(probesTree, zone)
    Cmpi._setProc(probesTree, rank)

    I._rmNodesByType(probesTree, 'Elements_t')
    I._renameNode(probesTree, 'FlowSolution#Centers', 'FlowSolution')

    probesTree = Cmpi.allgatherTree(probesTree)

    ProbesDict = dict()
    for probeZone in I.getZones(probesTree):
        ProbesDict = dict( IterationNumber = CurrentIteration-1 )
        GC = I.getNodeByName1(probeZone, 'GridCoordinates')
        FS = I.getNodeByName1(probeZone, I.__FlowSolutionCenters__)
        if not FS: continue
        for data in I.getNodesByType(GC, 'DataArray_t') + I.getNodesByType(FS, 'DataArray_t'):
            ProbesDict[I.getName(data)] = I.getValue(data)
        appendDict2Arrays(arrays, ProbesDict, I.getName(probeZone))

    return probesTree


def appendProbes2Arrays(t, arrays):
    '''
    Append probes with picked data in **arrays**.

    Parameter
    ---------

        t : PyTree

        arrays : dict

    '''
    for Probe in setup.Extractions:
        if Probe['type'] != 'Probe':
            continue
        if Probe['rank'] != rank:
            continue
        ProbesDict = dict( IterationNumber = CurrentIteration-1 )
        if setup.elsAkeysNumerics['time_algo'] != 'steady': 
            ProbesDict['Time'] = ProbesDict['IterationNumber'] * setup.elsAkeysNumerics['timestep']

        variables = Probe['variables']
        if isinstance(variables, str):
            variables = [variables]
        zone = I.getNodeFromName2(t, Probe['zone'])
        variablesDict = J.getVars2Dict(zone, VariablesName=variables, Container='FlowSolution#Init')
        for var, value in variablesDict.items():
            ProbesDict[var] = value.ravel('F')[Probe['element']]

        appendDict2Arrays(arrays, ProbesDict, Probe['name'])

def searchZoneAndIndexForProbes(t, method='getNearestPointIndex', tol=1e-2):
    
    '''
    Search for the nearest vertex from each probe in **setup.Extractions** in a PyTree.

    Parameters
    ----------
    t : PyTree
        Input PyTree.

    method : str
        One of 'getNearestPointIndex' (from Cassiopee Geom module) or 'nearestNodes' (from Converter module).

    tol : float, optional
        The tolerance for minimum distance. Default is 1e-2.

    Notes
    -----
        - The function modifies the probe dictionaries by adding information about the zone, element, distance to the nearest vertex, and processor rank.
        - Probes that are too far from the nearest vertex are removed from the list.
    '''
    # Put data at cell center, including coordinates
    # IMPORTANT: In this function, the mesh will be now the dual mesh, with nodes corresponding cell centers of the input mesh
    t = C.node2Center(t)

    for Probe in setup.Extractions:
        if Probe['type'] != 'Probe':
            continue

        # Search the nearest points in all zones
        nearestElement = None
        minDistance = 1e20
        for zone in I.getZones(t):
            x = J.getx(zone)
            if x is None:
                # This zone is a skeleton zone, so the current processor is not in charge of this zone
                continue

            if method == 'getNearestPointIndex':
                element, squaredDistance = D.getNearestPointIndex(zone, Probe['location'])
                distance = np.sqrt(squaredDistance)

            elif method == 'nearestNodes':
                # Get the nearest node of the dual mesh 
                # Prefer this function C.nearestNodes to D.getNearestPointIndex for performance
                # (see https://elsa.onera.fr/issues/8236)
                hook = C.createGlobalHook(zone, function='nodes')
                nodes, distances = C.nearestNodes(hook, D.point(Probe['location']))
                element, distance = nodes[0], distances[0]
            
            else:
                raise Exception('method must be getNearestPointIndex or nearestNodes')

            if distance < minDistance:
                minDistance = distance
                nearestElement = element
                probeZone = zone
        
        Probe['rank'] = -1
        Cmpi.barrier()
        minDistanceForAllProcessors = comm.allreduce(minDistance, op=MPI.MIN)
        if minDistance == minDistanceForAllProcessors:
            # Probe on this proc
            Probe['rank'] = rank
            Probe['zone'] = I.getName(probeZone)
            Probe['element'] = nearestElement
            Probe['distanceToNearestCellCenter'] = minDistance     
            x, y, z = J.getxyz(probeZone)
            Probe['location'] = x.ravel(order='F')[nearestElement], y.ravel(order='F')[nearestElement], z.ravel(order='F')[nearestElement]
            if 'name' not in Probe:
                Probe['name'] = 'Probe_{:.3g}_{:.3g}_{:.3g}'.format(Probe['location'][0], Probe['location'][1], Probe['location'][2])
        Cmpi.barrier()
        rankForComm = comm.allreduce(Probe['rank'], op=MPI.MAX)
        Cmpi.barrier()
        UpdatedProbe = comm.bcast(Probe, root=rankForComm)
        Cmpi.barrier()
        Probe.update(UpdatedProbe)

        if minDistanceForAllProcessors > tol:
            printCo(f'The probe {Probe["name"]} is too far from the nearest vertex ({minDistanceForAllProcessors} m). It is removed.', 0, J.WARN)
            setup.Extractions.remove(Probe)
    

def loadUnsteadyMasksForElsA(e, elsA_user, Skeleton):
    
    Cmpi.barrier()
    AllMaskedZones = dict()
    for base in I.getBases(Skeleton):
        elsA_masks = []
        masks = I.getNodeFromName1(base, '.MOLA#Masks')
        if not masks: continue

        for mask in masks[2]:
            mask_name = I.getValue(mask).replace('.','_').replace('-','_')
            WndNames, ZonePaths, PtWnds = [], [], []
            for i, patch in enumerate(I.getNodesFromName(mask,'patch*')):
                zone_name = I.getValue(I.getNodeFromName1(patch,'Zone'))
                base_name = J._getBaseWithZoneName(Skeleton, zone_name)[0]
                wnd_node = I.getNodesFromName(patch,'Window*')[0]
                w = I.getValue( wnd_node )
                if w is None:
                    msg = 'value of node %s not loaded'%os.path.join(base[0],
                            masks[0], mask[0], patch[0], wnd_node[0])
                    raise ValueError(msg)
                wnd_name = 'wnd_'+mask_name+'%d'%i
                wnd_name = wnd_name.replace('-','_').replace('.','_')
                WndNames += [ wnd_name ]
                ZonePaths += [ base_name + '/' +  zone_name ]
                PtWnds += [ [ int(w[0,0]), int(w[0,1]),
                              int(w[1,0]), int(w[1,1]),
                              int(w[2,0]), int(w[2,1])] ]

            elsA_windows = []
            for name, path, wnd in zip(WndNames, ZonePaths, PtWnds):
                elsA_windows += [elsA_user.window(
                                   e.e_getBlockInternalName(path), name=name)]
                elsA_windows[-1].set('wnd',wnd)
                elsA_windows[-1].show()

            printCo('setting unsteady mask '+mask_name,proc=0)
            elsA_masks += [ elsA_user.mask( ' '.join(WndNames), name=mask_name ) ] 
            
            Parameters = J.get(mask,'Parameters')
            for p in Parameters:
                value = Parameters[p]
                if isinstance(value, np.ndarray):
                    dtype = str(value.dtype)
                    if dtype.startswith('int'):
                        value = int(value)
                    elif dtype.startswith('float'):
                        value = float(value)
                    else: 
                        raise TypeError('FATAL: numpy dtype %s not supported'%dtype)
                    Parameters[p] = value
                

            elsA_masks[-1].setDict(Parameters)

            Neighbours = I.getValue(I.getNodeFromName(mask,'MaskedZones')).split(' ')
            for n in Neighbours:
                elsA_masks[-1].attach(e.e_getBlockInternalName(n))
                if n not in AllMaskedZones:
                    AllMaskedZones[n] = ZonePaths
                else:
                    for zp in ZonePaths:
                        if zp not in AllMaskedZones[n]:
                            AllMaskedZones[n] += [ zp ]
            elsA_masks[-1].show()
    Cmpi.barrier()

    # create ghost masks
    for ghost_zonename in AllMaskedZones:
        ghost_name = 'maskG_'+ghost_zonename
        ghost_name = ghost_name.replace('.','_').replace('/','_').replace('-','_')
        ghost_mask = elsA_user.mask( e.blockwindow(ghost_zonename), name=ghost_name)
        ghost_mask.set('type','ghost')
        for interpolant_name in AllMaskedZones[ghost_zonename]:
            ghost_mask.attach( e.blockname(interpolant_name) )

    Cmpi.barrier()




def readStaticMasksForElsA(e, elsA_user, Skeleton):
    
    Cmpi.barrier()
    bases = I.getBases(Skeleton)
    for base in bases:

        meshInfo = J.get(base,'.MOLA#InputMesh')
        if 'Motion' in meshInfo: continue
        
        for zone in I.getZones(base):
            mask_file = os.path.join(PRE.DIRECTORY_OVERSET,
                                     'hole_%s_%s.v3d'%(base[0],zone[0]))
            
            if os.path.isfile(mask_file):
                baseAndZoneName = base[0]+'/'+zone[0]
                blockname = e.blockname( baseAndZoneName )
                mask_name = 'staticMask_%s_%s'%(base[0],zone[0])
                mask_name = mask_name.replace('.','_')
                printCo('setting static mask %s at base %s'%(mask_file,base[0]), proc=0)
                mask = elsA_user.mask(e.blockwindow(baseAndZoneName), name=mask_name)
                mask.set('type', 'file')
                mask.set('file', mask_file)
                mask.set('format', 'bin_v3d')
                mask.attach(blockname)
    Cmpi.barrier()


def loadMotionForElsA(elsA_user, Skeleton):
    Cmpi.barrier()
    
    AllMotions = []
    bases = I.getBases(Skeleton)
    for base in bases:

        motion = I.getNodeFromName2(base, '.Solver#Motion')
        if not motion: continue

        function_name = I.getNodeFromName1(motion, 'function_name')
        if not function_name: continue
        function_name = I.getValue(function_name)

        MOLA_motion = I.getNodeFromName2(base, '.MOLA#Motion')
        Parameters = dict()
        for p in MOLA_motion[2]:
            parameter_name = p[0]
            value = I.getValue(p)
            if isinstance(value, np.ndarray): value = value.tolist()
            Parameters[parameter_name] = value
        
        printCo('setting elsA motion function %s at base %s'%(function_name,base[0]), proc=0)
        AllMotions.append(elsA_user.function(Parameters['type'],name=function_name))
        AllMotions[-1].setDict(Parameters)
        AllMotions[-1].show()
    Cmpi.barrier()


def _extendSurfacesWithWorkflowQuantities(surfaces, arrays=None):
    '''
    Perform post-process specific to the workflow.

    Parameters
    ----------
        surfaces : PyTree
            Tree as given by :py:func:`extractSurfaces`

    Returns
    -------
        PyTree
            Same as the input **surfaces** with eventual post-processed data.
    '''

    try:
        Workflow = setup.Workflow
    except AttributeError:
        return surfaces
    
    try:
        PostprocessOptions = setup.PostprocessOptions
    except AttributeError:
        return surfaces

    if Workflow == 'Compressor' and PostprocessOptions is not None:
        import MOLA.WorkflowCompressor as WC
        class ChannelHeightError(Exception):
            pass

        if EndOfRun or setup.elsAkeysNumerics['time_algo'] != 'steady':

            if not EndOfRun and setup.elsAkeysNumerics['time_algo'] != 'steady':
                computeRadialProfiles = False
            else: 
                computeRadialProfiles = True

            if NumberOfProcessors > 1:
                # Share the skeleton on all procs
                Cmpi._setProc(surfaces, rank)
                Skeleton = J.getStructure(surfaces)
                trees = comm.allgather(Skeleton)
                trees.insert(0, surfaces)
                surfaces = I.merge(trees)
                Cmpi._convert2PartialTree(surfaces)
                # Ensure that bases are in the same order on all procs. 
                # It is MANDATORY for next post-processings
                J._reorderBases(surfaces)
            Cmpi.barrier()

            try:
                LocalChannelHeight = bool(I.getNodeFromName(surfaces, 'ChannelHeight'))
                GlobalChannelHeight = any(comm.allgather(LocalChannelHeight))
                if not GlobalChannelHeight:
                    printCo('Postprocess cannot be done because ChannelHeight is missing', color=J.WARN)
                    raise ChannelHeightError


                printCo('making postprocess_turbomachinery...', proc=0, color=J.MAGE)
                WC.postprocess_turbomachinery(surfaces,
                    computeRadialProfiles=computeRadialProfiles, **PostprocessOptions)
                printCo('making postprocess_turbomachinery... done', proc=0, color=J.MAGE)

                if rank == 0:
                    # Move 0D averages to arrays
                    averagesDict = dict()
                    Averages0D = [I.getZones(b) for b in I.getBases(surfaces) \
                                  if b[0].startswith('Averages0D')]
                    for zones in Averages0D: 
                        for zone in zones:
                            for FS in I.getNodesFromType1(zone, 'FlowSolution_t'):
                                FSname = I.getName(FS)

                                if FSname.startswith('Comparison'):
                                    zoneName = I.getName(zone) \
                                               + FSname.replace('Comparison','')
                                else:
                                    zoneName = I.getName(zone) \
                                               + FSname.replace('FlowSolution','')

                                for node in I.getNodesFromType1(FS, 'DataArray_t'):
                                    averagesDict[I.getName(node)] = I.getValue(node)
                                averagesDict['IterationNumber'] = CurrentIteration-1
                                appendDict2Arrays(arrays, averagesDict, zoneName)
            
                else:
                    # Remove RadialProfiles for all proc except one, because only proc 0 is up-to-date
                    I._rmNodesFromName1(surfaces, 'RadialProfiles')

                # Remove 0D averages from surfaces tree for all procs
                I._rmNodesFromName(surfaces, 'Averages0D*')

            except ImportError: # https://gitlab.onera.net/numerics/analysis/turbo/-/issues/1
                printCo('Postprocess cannot be done (ImportError)', proc=0, color=J.WARN)
                pass
            except ChannelHeightError:
                pass
    return surfaces

def checkAndUpdateMainCGNSforChoroRestart():
    '''
    Check the main.cgns and update it with links to ChoroData nodes located in fields.cgns if necessary.
    '''    
    if rank == 0:
        mainSkel = Filter.convertFile2SkeletonTree(FILE_CGNS)
        ChoroNodesMain = I.getNodeFromName(mainSkel, 'ChoroData')
        if I.getNodeFromName(mainSkel, 'choro_file'): 
            # Chrochronic simulation
            printCo('Chorochronic simulation detected. Checking if main.cgns should be updated for restart', proc=0, color=J.CYAN)
            if I.getNodeFromName(mainSkel, 'ChoroData'):
                printCo('main.cgns file already up-to-date for chorochronic computation.', proc=0, color=J.GREEN)
                return
            else:
                printCo('ChoroData nodes detected in fields.cgns. Gathering links between main.cgns and fields.', proc=0, color=J.CYAN)
                AllCGNSLinks = []
                main = C.convertFile2PyTree(FILE_CGNS, links=AllCGNSLinks)
                t = Filter.convertFile2SkeletonTree(os.path.join(DIRECTORY_OUTPUT, 'fields.cgns'))
                ChoroNodes = I.getNodesFromName(t, 'ChoroData')
                for node in ChoroNodes:
                    ChoroPath = I.getPath(t,node)
                    AllCGNSLinks.append(['.', os.path.join(DIRECTORY_OUTPUT, 'fields.cgns'), ChoroPath, ChoroPath],)
                
                C.convertPyTree2File(main, FILE_CGNS, links=AllCGNSLinks)
                printCo('main.cgns updated with links to fields.cgns ChoroData nodes for restart.', proc=0, color=J.GREEN)

def resumeFieldsAveraging(Skeleton, t, container_name='FlowSolution#Average'):
    '''
    use any pre-existing average fields contained in ``FlowSolution#Average``
    nodes in order to resume the fields averaging process
    '''
    inititer = setup.elsAkeysNumerics['inititer']
    firstiter = setup.ReferenceValues['CoprocessOptions']['FirstIterationForFieldsAveraging']
    if firstiter is None: return
    firstiter -= 1
    setup.ReferenceValues['CoprocessOptions']['FirstIterationForFieldsAveraging']
    cit = CurrentIteration

    # adapt 3D fields:
    old = _getDictofNodesFieldsPerZone(Skeleton, container_name)
    tot = _getDictofNodesFieldsPerZone(t, container_name)
    if cit == firstiter:
        ini = _getDictofNodesFieldsPerZone(t, 'FlowSolution#Init')
    for zone_name in tot:
        for field_name in tot[zone_name]:
            avg_old = old[zone_name][field_name] # BEWARE this is a CGNS node
            avg_tot = tot[zone_name][field_name] # BEWARE this is a CGNS node
            
            if cit == firstiter:
                avg_old[1] = np.copy(avg_tot[1], order='F')
                avg_tot[1] = np.copy(ini[zone_name][field_name][1], order='F')
                continue

            if cit < firstiter: 
                avg_old[1] = None
                avg_new    = None
            
            else:
                if avg_old[1] is None or avg_tot[1] is None: continue
                if inititer < firstiter:
                    avg_new =  (avg_tot[1]*(cit-inititer+1) \
                            -avg_old[1]*(firstiter-inititer+1))/(cit-firstiter)
                
                else:
                    avg_new =  (avg_old[1]*(inititer-(firstiter+1)) \
                            +avg_tot[1]*(cit-inititer+1))/(cit-firstiter)

            avg_tot[1] = avg_new # update of OUTPUT_TREE

    # adapt BC fields:
    old = _getDictofNodesBCFieldsPerZone(Skeleton, '.Solver#Output#Average')
    tot = _getDictofNodesBCFieldsPerZone(t, '.Solver#Output#Average')
    if cit == firstiter:
        ini = _getDictofNodesBCFieldsPerZone(t, '.Solver#Output#Output')
    for zone_name in tot:
        for bcfamily_name in tot[zone_name]:
            for field_name in tot[zone_name][bcfamily_name]:
                avg_old = old[zone_name][bcfamily_name][field_name] # BEWARE this is a CGNS node
                avg_tot = tot[zone_name][bcfamily_name][field_name] # BEWARE this is a CGNS node
                
                if cit == firstiter:
                    avg_old[1] = np.copy(avg_tot[1], order='F')
                    avg_tot[1] = np.copy(ini[zone_name][bcfamily_name][field_name][1], order='F')
                    continue

                if cit < firstiter: 
                    avg_old[1] = None
                    avg_new    = None
                
                else:
                    if avg_old[1] is None or avg_tot[1] is None: continue
                    if inititer < firstiter:
                        avg_new =  (avg_tot[1]*(cit-inititer+1) \
                                -avg_old[1]*(firstiter-inititer+1))/(cit-firstiter)
                    
                    else:
                        avg_new =  (avg_old[1]*(inititer-(firstiter+1)) \
                                +avg_tot[1]*(cit-inititer+1))/(cit-firstiter)

                avg_tot[1] = avg_new # update of OUTPUT_TREE




def _getDictofNodesFieldsPerZone(t, Container):
    fields = dict()
    for base in I.getNodesByType1(t, 'CGNSBase_t'):
        for zone in I.getNodesByType1(base, 'Zone_t'):
            zone_name = zone[0]
            fields[zone_name] = dict()
            fs = I.getNodeFromName1(zone, Container)
            if not fs:
                del fields[zone_name]
                continue
            for f in fs[2]:
                if f[3] != 'DataArray_t': continue
                fields[zone_name][f[0]] = f
    return fields

def _getDictofNodesBCFieldsPerZone(t, Container):
    fields = dict()
    for base in I.getNodesByType1(t, 'CGNSBase_t'):
        for zone in I.getNodesByType1(base, 'Zone_t'):
            zone_name = zone[0]
            fields[zone_name] = dict()
            for bc in I.getNodesFromType(zone,'BC_t'):
                bcfamily_name = bc[0]
                bcds = I.getNodeFromName1(bc, Container)
                if bcds:
                    data = I.getNodeFromName1(bcds,'NeumannData')
                    for f in data[2]:
                        if f[3] != 'DataArray_t': continue
                        fields[zone_name][bcfamily_name][f[0]] = f
    return fields

def removeEmptyBCDataSet(t):
    for z in I.getZones(t):
        for zbc in I.getNodesFromType1(z,'ZoneBC_t'):
            for bc in I.getNodesFromType1(zbc,'BC_t'):
                for n in bc[2]:
                    if n[0].startswith('BCDataSet') and not n[2]:
                        I._rmNode(t, n)

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


def ravelBCDataSet(t):
    # HACK https://elsa.onera.fr/issues/11219
    # HACK https://elsa-e.onera.fr/issues/10750
    for zone in I.getZones(t):
        for zbc in I.getNodesFromType1(zone,'ZoneBC_t'):
            for bc in I.getNodesFromType1(zbc,'BC_t'):
                for bcds in I.getNodesFromType1(bc,'BCDataSet_t'):
                    for bcd in I.getNodesFromType1(bcds,'BCData_t'):
                        for da in I.getNodesFromType1(bcd,'DataArray_t'):
                            da[1] = da[1].ravel(order='K')
