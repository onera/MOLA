'''
MOLA - WorkflowCompressor.py

WORKFLOW COMPRESSOR

Collection of functions designed for Workflow Compressor

BEWARE:
There is no equivalent of Preprocess ``prepareMesh4ElsA``.
``prepareMainCGNS4ElsA`` takes as an input a CGNS file assuming that the following
elements are already set:
    * connectivities
    * splitting and distribution
    * families
    * (optional) parametrization with channel height in a ``FlowSolution#Height`` node

File history:
31/08/2021 - T. Bontemps - Creation
'''

import sys
import os
import numpy as np
import pprint
import scipy.optimize

# BEWARE: in Python v >= 3.4 rather use: importlib.reload(setup)
import imp


import Converter.PyTree    as C
import Converter.Internal  as I
import Distributor2.PyTree as D2
import Post.PyTree         as P
import Generator.PyTree    as G
import Transform.PyTree    as T

from . import InternalShortcuts as J
from . import Preprocess        as PRE


def prepareMesh4ElsA(filename, NProcs=None, ProcPointsLoad=250000,
                        duplicationInfos={}, blocksToRename={}):
    '''
    This is a macro-function used to prepare the mesh for an elsA computation
    from a CGNS file provided by Autogrid 5.

    The sequence of operations performed are the following:

    #. load and clean the mesh from Autogrid 5
    #. apply transformations
    #. apply connectivity
    #. split the mesh
    #. distribute the mesh
    #. make final elsA-specific adaptations of CGNS data

    .. warning:: The following assumptions on the input mesh are made:
        * it does not need any scaling
        * the shaft axis is the Z-axis, pointing downstream (convention in
        Autgrid 5). The mesh will be rotated to follow the elsA convention, thus
        the shaft axis will be the X-axis, pointing downstream.

    Parameters
    ----------

        filename : str
            Name of the CGNS mesh file from Autogrid 5

        NProcs : int
            If a positive integer is provided, then the
            distribution of the tree (and eventually the splitting) will be done in
            order to satisfy a total number of processors provided by this value.
            If not provided (:py:obj:`None`) then the number of procs is automatically
            determined using as information **ProcPointsLoad** variable.

        ProcPointsLoad : int
            this is the desired number of grid points
            attributed to each processor. If **SplitBlocks** = :py:obj:`True`, then it is used to
            split zones that have more points than **ProcPointsLoad**. If
            **NProcs** = :py:obj:`None` , then **ProcPointsLoad** is used to determine
            the **NProcs** to be used.

        duplicationInfos : dict
            User-provided data related to domain duplication.
            Each key corresponds to a row FamilyName.
            The associated element is a dictionary with the following parameters:

                * NumberOfBlades: number of blades in the row (in reality)

                * NumberOfDuplications: number of duplications to make of the
                    input row domain.

                * MergeBlocks: boolean, if True the duplicated blocks are merged.

        blocksToRename : dict
            Each key corresponds to the name of a zone to modify, and the associated
            value is the new name to give.

    Returns
    -------

        t : PyTree
            the pre-processed mesh tree (usually saved as ``mesh.cgns``)

            .. important:: This tree is **NOT** ready for elsA computation yet !
                The user shall employ function :py:func:`prepareMainCGNS4ElsA`
                as next step
    '''
    t = C.convertFile2PyTree(filename)

    BladeNumberList = [I.getValue(bn) for bn in I.getNodesFromName(t, 'BladeNumber')]
    angles = list(set([360./float(bn) for bn in BladeNumberList]))

    InputMeshes = [dict(
                    baseName=I.getName(I.getNodeByType(t, 'CGNSBase_t')),
                    Transform=dict(
                        scale=1.0,
                        rotate=[((0,0,0), (0,1,0), 90),((0,0,0), (1,0,0), 90)]
                        ),
                    Connection=[dict(type='Match', tolerance=1e-8),
                                dict(type='PeriodicMatch', tolerance=1e-8, angles=angles)
                                ],
                    SplitBlocks=True,
                    )]

    t = cleanMeshFromAutogrid(t, basename=InputMeshes[0]['baseName'], blocksToRename=blocksToRename)
    PRE.transform(t, InputMeshes)
    for row, rowParams in duplicationInfos.items():
        try: MergeBlocks = rowParams['MergeBlocks']
        except: MergeBlocks = True
        duplicate(t, row, rowParams['NumberOfBlades'],
                nDupli=rowParams['NumberOfDuplications'], merge=MergeBlocks)
    if not InputMeshes[0]['SplitBlocks']:
        t = PRE.connectMesh(t, InputMeshes)
    t = PRE.splitAndDistribute(t, InputMeshes, NProcs=NProcs,
                                ProcPointsLoad=ProcPointsLoad)
    PRE.adapt2elsA(t, InputMeshes)
    J.checkEmptyBC(t)

    return t

def prepareMainCGNS4ElsA(FILE_MESH='mesh.cgns', ReferenceValuesParams={},
        NumericalParams={}, TurboConfiguration={}, Extractions={},
        BodyForceInputData=[], writeOutputFields=True):
    '''
    This is mainly a function similar to Preprocess :py:func:`prepareMainCGNS4ElsA`
    but adapted to compressor computations. Its purpose is adapting the CGNS to
    elsA. Wall boundary conditions are set in this function. Other types of BCs
    (inflow, outflow, Rotor/stator interfaces) must be set separately.

    Parameters
    ----------

        t : PyTree
            A grid that already contains:
                * connectivities
                * splitting and distribution
                * families
                * (optional) parametrization with channel height in a ``FlowSolution#Height`` node

    Returns
    -------

        files : None
            A number of files are written:

            * ``main.cgns``
                main CGNS file to be read directly by elsA

            * ``OUTPUT/fields.cgns``
                file containing the initial fields (if ``writeOutputFields=True``)

            * ``setup.py``
                ultra-light file containing all relevant info of the simulation
    '''

    def addFieldExtraction(fieldname):
        try:
            FieldsExtr = ReferenceValuesParams['FieldsAdditionalExtractions']
            if fieldname not in FieldsExtr.split():
                FieldsExtr += ' '+fieldname
        except:
            ReferenceValuesParams['FieldsAdditionalExtractions'] = fieldname


    t = C.convertFile2PyTree(FILE_MESH)

    hasBCOverlap = True if C.extractBCOfType(t, 'BCOverlap') else False


    if hasBCOverlap: PRE.addFieldExtraction('ChimeraCellType')
    if BodyForceInputData: PRE.addFieldExtraction('Temperature')

    FluidProperties = PRE.computeFluidProperties()
    ReferenceValues = computeReferenceValues(FluidProperties,
                                             **ReferenceValuesParams)

    NProc = max([I.getNodeFromName(z,'proc')[1][0][0] for z in I.getZones(t)])+1
    ReferenceValues['NProc'] = int(NProc)
    ReferenceValuesParams['NProc'] = int(NProc)
    elsAkeysCFD      = PRE.getElsAkeysCFD()
    elsAkeysModel    = PRE.getElsAkeysModel(FluidProperties, ReferenceValues)
    if BodyForceInputData: NumericalParams['useBodyForce'] = True
    elsAkeysNumerics = PRE.getElsAkeysNumerics(ReferenceValues, **NumericalParams)
    TurboConfiguration = setTurboConfiguration(**TurboConfiguration)

    AllSetupDics = dict(FluidProperties=FluidProperties,
                        ReferenceValues=ReferenceValues,
                        elsAkeysCFD=elsAkeysCFD,
                        elsAkeysModel=elsAkeysModel,
                        elsAkeysNumerics=elsAkeysNumerics,
                        TurboConfiguration=TurboConfiguration,
                        Extractions=Extractions)
    if BodyForceInputData: AllSetupDics['BodyForceInputData'] = BodyForceInputData

    t = newCGNSfromSetup(t, AllSetupDics, initializeFlow=True, FULL_CGNS_MODE=False)
    to = PRE.newRestartFieldsFromCGNS(t)
    setBC_Walls(t, TurboConfiguration)
    PRE.saveMainCGNSwithLinkToOutputFields(t,to,writeOutputFields=writeOutputFields)

    print('REMEMBER : configuration shall be run using %s%d%s procs'%(J.CYAN,
                                               ReferenceValues['NProc'],J.ENDC))

def cleanMeshFromAutogrid(t, basename='Base#1', blocksToRename={}):
    '''
    Clean a CGNS mesh from Autogrid 5.
    The sequence of operations performed are the following:

    #. remove useless nodes specific to AG5
    #. rename base
    #. rename zones
    #. clean Joins & Periodic Joins
    #. clean Rotor/Stator interfaces
    #. join HUB and SHROUD families

    Parameters
    ----------

        t : PyTree
            CGNS mesh from Autogrid 5

        basename: str
            Name of the base. Will replace the default AG5 name.

        blocksToRename : dict
            Each key corresponds to the name of a zone to modify, and the associated
            value is the new name to give.

    Returns
    -------

        t : PyTree
            modified mesh tree

    '''

    I._rmNodesByName(t, 'Numeca*')
    I._rmNodesByName(t, 'blockName')
    I._rmNodesByName(t, 'meridional_base')
    I._rmNodesByName(t, 'tools_base')

    # Clean Names
    # - Recover BladeNumber and Clean Families
    for fam in I.getNodesFromType(t, 'Family_t'):
        I._rmNodesByName(fam, 'RotatingCoordinates')
        I._rmNodesByName(fam, 'Periodicity')
        I._rmNodesByName(fam, 'DynamicData')
    I._rmNodesByName(t, 'FamilyProperty')

    # - Rename base
    base = I.getNodeFromType(t, 'CGNSBase_t')
    I.setName(base, basename)

    # - Rename Zones
    for zone in I.getNodesFromType(t, 'Zone_t'):
        name = I.getName(zone)
        for block in blocksToRename:
            if block in name:
                newName = name.replace(block, blocksToRename[block])
                print("Zone {} is renamed: {}".format(name,newName))
                I._renameNode(tree, name, newName)
        # Delete some usual patterns in AG5
        new_name = name
        for pattern in ['_flux_1', '_flux_2', '_flux_3', '_Main_Blade']:
            new_name = new_name.replace(pattern, '')
        I._renameNode(t, name, new_name)

    # Clean Joins & Periodic Joins
    I._rmNodesByType(t, 'ZoneGridConnectivity_t')
    periodicFamilyNames = [I.getName(fam) for fam in I.getNodesFromType(t, "Family_t")
        if 'PER' in I.getName(fam)]
    for fname in periodicFamilyNames:
        # print('|- delete PeriodicBC family of name {}'.format(name))
        C._rmBCOfType(t, 'FamilySpecified:%s'%fname)
        fbc = I.getNodeFromName2(t, fname)
        I.rmNode(t, fbc)

    # Clean RS interfaces
    I._rmNodesByType(t,'InterfaceType')
    I._rmNodesByType(t,'DonorFamily')

    # Join HUB and SHROUD families
    joinFamilies(t, 'HUB')
    joinFamilies(t, 'SHROUD')
    return t

def joinFamilies(t, pattern):
    '''
    In the CGNS tree t, gather all the Families <ROW_I>_<PATTERN>_<SUFFIXE> into
    Families <ROW_I>_<PATTERN>, so as many as rows.
    Useful to join all the row_i_HUB* or (row_i_SHROUD*) together

    Parameters
    ----------

        t : PyTree
            A PyTree read by Cassiopee

        pattern : str
            The pattern used to gather CGNS families. Should be for example 'HUB' or 'SHROUD'
    '''
    fam2remove = set()
    fam2keep = set()
    # Loop on the BCs in the tree
    for bc in I.getNodesFromType(t, 'BC_t'):
        # Get BC family name
        famBC_node = I.getNodeFromType(bc, 'FamilyName_t')
        famBC = I.getValue(famBC_node)
        # Check if the pattern is present in FamilyBC name
        if pattern not in famBC:
            continue
        # Split to get the short name based on pattern
        split_fanBC = famBC.split(pattern)
        assert len(split_fanBC) == 2, 'The pattern {} is present more than once in the FamilyBC {}. It must be more selective.'.format(pattern, famBC)
        preffix, suffix = split_fanBC
        # Add the short name to the set fam2keep
        short_name = '{}{}'.format(preffix, pattern)
        fam2keep |= {short_name}
        if suffix != '':
            # Change the family name
            I.setValue(famBC_node, '{}'.format(short_name))
            fam2remove |= {famBC}

    # Remove families
    for fam in fam2remove:
        print('Remove family {}'.format(fam))
        I._rmNodesByNameAndType(t, fam, 'Family_t')

    # Check that families to keep still exist
    base = I.getNodeFromType(t,'CGNSBase_t')
    for fam in fam2keep:
        fam_node = I.getNodeFromNameAndType(t, fam, 'Family_t')
        if fam_node is None:
            print('Add family {}'.format(fam))
            I._newFamily(fam, parent=base)

def duplicate(tree, rowFamily, nBlades, nDupli=None, merge=False, axis=(1,0,0)):
    '''
    Duplicate **nDupli** times the domain attached to the family **rowFamily**
    around the axis of rotation.

    .. warning:: This function works only for empty meshes. It can be used on a
        PyTree with FlowSolution containers, but the vectors will not be
        rotated !

    Parameters
    ----------

        tree : PyTree
            tree to modify

        rowFamily : str
            Name of the CGNS family attached to the row domain to Duplicate

        nBlades : int
            Number of blades in the row. Used to compute the azimuthal length of
            a blade sector.

        nDupli : int
            Number of duplications to make

            .. warning:: This is the number of duplication of the input mesh
                domain, not the wished number of simulated blades. Keep this
                point in mind if there is already more than one blade in the
                input mesh.

        merge : bool
            if :py:obj:`True`, merge all the blocks resulting from the
            duplication.

            .. tip:: This option is useful is the mesh is to split and if a
                globborder will be defined on a BC of the duplicated domain. It
                allows the splitting procedure to provide a 'matricial' ordering
                (see `elsA Tutorial about globborder <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/globborder.html>`_)

        axis : tuple
            axis of rotation given as a 3-tuple of integers or floats
    '''
    # TODO: rotate vectors in FlowSolution nodes. It will allows to use this function
    # in other situations that currently

    if nDupli is None:
        nDupli = nBlades # for a 360 configuration
    if nDupli == nBlades:
        print('Duplicate {} over 360 degrees ({} blades in row)'.format(rowFamily, nBlades))
    else:
        print('Duplicate {} on {} blades ({} blades in row)'.format(rowFamily, nDupli, nBlades))
    base = I.getNodeFromType(tree, 'CGNSBase_t')
    check = False
    for zone in I.getNodesFromType(tree, 'Zone_t'):
        zone_name = I.getName(zone)
        zone_family = I.getValue(I.getNodeFromName(zone, 'FamilyName'))
        if zone_family == rowFamily:
            print('  > zone {}'.format(zone_name))
            check = True
            zones2merge = [zone]
            for n in range(nDupli-1):
                ang = 360./nBlades*(n+1)
                rot = T.rotate(I.copyNode(zone),(0.,0.,0.), axis, ang)
                I.setName(rot, "{}_{}".format(zone_name, n+2))
                I._addChild(base, rot)
                zones2merge.append(rot)
            if merge:
                for node in zones2merge:
                    I.rmNode(base, node)
                tree_dist = T.merge(zones2merge, tol=1e-8)
                for i, node in enumerate(I.getZones(tree_dist)):
                    I._addChild(base, node)
                    disk_block = I.getNodeFromName(base, I.getName(node))
                    disk_block[0] = '{}_{:02d}'.format(zone_name, i)
                    I.createChild(disk_block, 'FamilyName', 'FamilyName_t', value=rowFamily)
                autoMergeBCs(tree)

    assert check, 'None of the zones was duplicated. Check the name of row family'
    return tree

def autoMergeBCs(t, familyNames=[]):
    '''
    Merge BCs that are contiguous, belong to the same family and are of the same
    type, for all zones of a PyTree

    Parameters
    ----------

        t : PyTree
            input tree

        familyNames : :py:class:`list` of :py:class:`str`
            restrict the merge operation to the listed family name(s).
    '''
    treeFamilies = [I.getName(fam) for fam in I.getNodesFromType(t, 'Family_t')]

    for family in familyNames:
        if family not in treeFamilies:
            raise AttributeError('Family '+family+' given by user does not appear in the pyTree')

    def getBCInfo(bc):
        pt  = I.getNodeFromName(bc, 'PointRange')
        fam = I.getNodeFromName(bc, 'FamilyName')
        if not fam:
            fam = I.createNode('FamilyName', 'FamilyName_t', Value='Unknown')
        return I.getName(bc), I.getValue(fam), pt

    for block in I.getNodesFromType(t, 'Zone_t'):
        if I.getNodeFromType(block, 'ZoneBC_t'):
            somethingWasMerged = True
            while somethingWasMerged : # recursively attempts to merge bcs until nothing possible is left
                somethingWasMerged = False
                bcs = I.getNodesFromType(block, 'BC_t')
                zoneBcsOut = I.copyNode(I.getNodeFromType(block, 'ZoneBC_t')) # a duplication of all BCs of current block
                mergedBcs = []
                for bc1 in bcs:
                    bcName1, famName1, pt1 = getBCInfo(bc1)
                    for bc2 in [b for b in bcs if b is not bc1]:
                        bcName2, famName2, pt2 = getBCInfo(bc2)
                        # check if bc1 and bc2 can be merged
                        mDim = areContiguous(pt1, pt2)
                        if bc1 not in mergedBcs and bc2 not in mergedBcs \
                            and mDim>=0 \
                            and famName1 == famName2 \
                            and (len(familyNames) == 0 or famName1 in familyNames) :
                            # does not check inward normal index, necessarily the same if subzones are contiguous
                            newPt = np.zeros(np.shape(pt1[1]),dtype=np.int32,order='F')
                            for dim in range(np.shape(pt1[1])[0]):
                                if dim != mDim :
                                    newPt[dim,0] = pt1[1][dim, 0]
                                    newPt[dim,1] = pt1[1][dim, 1]
                                else :
                                    newPt[dim,0] = min(pt1[1][dim, 0], pt2[1][dim, 0])
                                    newPt[dim,1] = max(pt1[1][dim, 1], pt2[1][dim, 1])
                            # new BC inheritates from the name of first BC
                            bc = I.createNode(bcName1, 'BC_t', bc1[1])
                            I.createChild(bc, pt1[0], 'IndexRange_t', value=newPt)
                            I.createChild(bc, 'FamilyName', 'FamilyName_t', value=famName1)
                            # TODO : include case with flow solution

                            I._rmNodesByName(zoneBcsOut, bcName1)
                            I._rmNodesByName(zoneBcsOut, bcName2)
                            I.addChild(zoneBcsOut, bc)
                            mergedBcs.append(bc1)
                            mergedBcs.append(bc2)
                            somethingWasMerged = True
                            # print('BCs {} and {} were merged'.format(bcName1, bcName2))

                block = I.rmNodesByType(block,'ZoneBC_t')
                I.addChild(block,zoneBcsOut)
                del(zoneBcsOut)

    return t

def areContiguous(PointRange1, PointRange2):
    '''
    Check if subZone of the same block defined by PointRange1 and PointRange2
    are contiguous.

    Parameters
    ----------

        PointRange1 : PyTree
            PointRange (PyTree of type ``IndexRange_t``) of a ``BC_t`` node

        PointRange2 : PyTree
            Same as PointRange2

    Returns
    -------
        dimension : int
            an integer of value -1 if subZones are not contiguous, and of value
            equal to the direction along which subzone are contiguous else.

    '''
    assert I.getType(PointRange1) == 'IndexRange_t' \
        and I.getType(PointRange2) == 'IndexRange_t', \
        'Arguments are not IndexRange_t'

    pt1 = I.getValue(PointRange1)
    pt2 = I.getValue(PointRange2)
    if pt1.shape != pt2.shape:
        return -1
    spaceDim = pt1.shape[0]
    indSpace = 0
    MatchingDims = []
    for dim in range(spaceDim):
        if pt1[dim, 0] == pt2[dim, 0] and pt1[dim, 1] == pt2[dim, 1]:
            indSpace += 1
            MatchingDims.append(dim)
    if indSpace != spaceDim-1 :
        # matching dimensions should form a hyperspace of original space
        return -1

    for dim in [d for d in range(spaceDim) if d not in MatchingDims]:
        if pt1[dim][0] == pt2[dim][1] or pt2[dim][0] == pt1[dim][1]:
            return dim

    return -1

def computeReferenceValues(FluidProperties, Massflow, PressureStagnation,
        TemperatureStagnation, Surface, TurbulenceLevel=0.001,
        Viscosity_EddyMolecularRatio=0.1, TurbulenceModel='Wilcox2006-klim',
        TurbulenceCutoff=1.0, TransitionMode=None, CoprocessOptions={},
        Length=1.0, TorqueOrigin=[0., 0., 0.],
        FieldsAdditionalExtractions=['ViscosityMolecular', 'Viscosity_EddyMolecularRatio', 'Pressure', 'Temperature', 'PressureStagnation', 'TemperatureStagnation', 'Mach']):
    '''
    This function is the Compressor's equivalent of :py:func:`PRE.computeReferenceValues()`.
    The main difference is that in this case reference values are set through
    ``Massflow``, total Pressure ``PressureStagnation``, total Temperature
    ``TemperatureStagnation`` and ``Surface``.

    Please, refer to :py:func:`PRE.computeReferenceValues()` doc for more details.
    '''
    # Fluid properties local shortcuts
    Gamma   = FluidProperties['Gamma']
    RealGas = FluidProperties['RealGas']
    cv      = FluidProperties['cv']
    cp      = FluidProperties['cp']

    # Compute variables
    Mach  = machFromMassflow(Massflow, Surface, Pt=PressureStagnation,
                            Tt=TemperatureStagnation)
    Temperature  = TemperatureStagnation / (1. + 0.5*(Gamma-1.) * Mach**2)
    Pressure  = PressureStagnation / (1. + 0.5*(Gamma-1.) * Mach**2)**(Gamma/(Gamma-1))
    Density = Pressure / (Temperature * RealGas)
    SoundSpeed  = np.sqrt(Gamma * RealGas * Temperature)
    Velocity  = Mach * SoundSpeed

    # REFERENCE VALUES COMPUTATION
    mus = FluidProperties['SutherlandViscosity']
    Ts  = FluidProperties['SutherlandTemperature']
    S   = FluidProperties['SutherlandConstant']
    ViscosityMolecular = mus * (Temperature/Ts)**1.5 * ((Ts + S)/(Temperature + S))

    ReferenceValues = PRE.computeReferenceValues(FluidProperties,
        Density=Density,
        Velocity=Velocity,
        Temperature=Temperature,
        AngleOfAttackDeg = 0.0,
        AngleOfSlipDeg = 0.0,
        YawAxis = [0.,0.,1.],
        PitchAxis = [0.,1.,0.],
        TurbulenceLevel=TurbulenceLevel,
        Surface=Surface,
        Length=Length,
        TorqueOrigin=TorqueOrigin,
        TurbulenceModel=TurbulenceModel,
        Viscosity_EddyMolecularRatio=Viscosity_EddyMolecularRatio,
        TurbulenceCutoff=TurbulenceCutoff,
        TransitionMode=TransitionMode,
        CoprocessOptions=CoprocessOptions,
        FieldsAdditionalExtractions=FieldsAdditionalExtractions)

    addKeys = dict(
        PressureStagnation = PressureStagnation,
        TemperatureStagnation = TemperatureStagnation,
        Massflow = Massflow,
        )

    ReferenceValues.update(addKeys)

    return ReferenceValues

def setTurboConfiguration(ShaftRotationSpeed=0., HubRotationSpeed=[], Rows={}):
    '''
    Construct a dictionary concerning the compressor properties.

    Parameters
    ----------

        ShaftRotationSpeed : py:class:`float`
            Shaft speed in rad/s

            .. attention:: only for single shaft configuration

            .. attention:: Pay attention to the sign of **ShaftRotationSpeed**

        HubRotationSpeed : :py:class:list of :py:class:tuple
            Hub rotation speed. Each tuple (``xmin``, ``xmax``) corresponds to a
            ``CoordinateX`` interval where the speed at hub wall is
            ``ShaftRotationSpeed``. It is zero outside these intervals.

        Rows : py:class:`dict`
            This dictionary has one entry for each row domain. The key names
            must be the family names in the CGNS Tree.
            For each family name, the following entries are expected:

                * RotationSpeed : py:class:`float` or py:class:`str`
                    Rotation speed in rad/s. Set ``'auto'`` to automatically
                    set ``ShaftRotationSpeed``.

                    .. attention:: Use **RotationSpeed**=``'auto'`` for rotors
                    only.

                    .. attention:: Pay attention to the sign of
                    **RotationSpeed**

                * NumberOfBlades : py:class:`int`
                    The number of blades in the row

                * NumberOfBladesSimulated : py:class:`int`
                    The number of blades in the computational domain. Set to
                    ``<NumberOfBlades>`` for a full 360 simulation.

                * InletPlane : py:class:`float`, optional
                    Position (in ``CoordinateX``) of the inlet plane for this
                    row. This plane is used for post-processing and convergence
                    monitoring.

                * OutletPlane : py:class:`float`, optional
                    Position of the outlet plane for this row.

    Returns
    -------

        TurboConfiguration : :py:class:`dict`
            set of compressor properties
    '''

    TurboConfiguration = dict(
        ShaftRotationSpeed = ShaftRotationSpeed,
        HubRotationSpeed   = HubRotationSpeed,
        Rows               = Rows
        )
    for row, rowParams in TurboConfiguration['Rows'].items():
        for key, value in rowParams.items():
            if key == 'RotationSpeed' and value == 'auto':
                rowParams[key] = ShaftRotationSpeed

    return TurboConfiguration

def getRotationSpeedOfRows(t):
    '''
    Get the rotationnal speed of each row in the PyTree ``<t>``

    Parameters
    ----------

        t : PyTree
            PyTree with declared families (Family_t) for each row with a
            ``.Solver#Motion`` node.

    Returns
    -------

        omegaDict : :py:class:`dict`
            dictionary with the rotation speed associated to each row family
            name.
    '''
    omegaDict = dict()
    for node in I.getNodesFromName(t, '.Solver#Motion'):
        rowNode, pos = I.getParentOfNode(t, node)
        if I.getType(rowNode) != 'Family_t':
            continue
        omega = I.getValue(I.getNodeFromName(node, 'omega'))
        rowName = I.getName(rowNode)
        omegaDict[rowName] = omega

    return omegaDict

def newCGNSfromSetup(t, AllSetupDictionaries, initializeFlow=True,
                     FULL_CGNS_MODE=False, dim=3):
    '''
    This is mainly a function similar to Preprocess :py:func:`newCGNSfromSetup`
    but adapted to compressor computations. Its purpose is creating the main
    CGNS tree and writes the ``setup.py`` file.

    The only differences with Preprocess newCGNSfromSetup are:
    #. addSolverBC is not applied, in order not to disturb the special
       turbomachinery BCs
    #. extraction of coordinates in ``FlowSolution#EndOfRun#Coords`` is
        desactivated
    '''
    t = I.copyRef(t)

    PRE.addTrigger(t)
    PRE.addExtractions(t, AllSetupDictionaries['ReferenceValues'],
                      AllSetupDictionaries['elsAkeysModel'], extractCoords=False)
    PRE.addReferenceState(t, AllSetupDictionaries['FluidProperties'],
                         AllSetupDictionaries['ReferenceValues'])
    PRE.addGoverningEquations(t, dim=dim) # TODO replace dim input by elsAkeysCFD['config'] info
    if initializeFlow:
        PRE.newFlowSolutionInit(t, AllSetupDictionaries['ReferenceValues'])
    if FULL_CGNS_MODE:
        PRE.addElsAKeys2CGNS(t, [AllSetupDictionaries['elsAkeysCFD'],
                             AllSetupDictionaries['elsAkeysModel'],
                             AllSetupDictionaries['elsAkeysNumerics']])

    AllSetupDictionaries['ReferenceValues']['NProc'] = int(max(D2.getProc(t))+1)
    AllSetupDictionaries['ReferenceValues']['CoreNumberPerNode'] = 28

    PRE.writeSetup(AllSetupDictionaries)

    return t

def massflowFromMach(Mx, S, Pt=101325.0, Tt=288.25, r=287.053, gamma=1.4):
    '''
    Compute the massflow rate through a section.

    Parameters
    ----------

        Mx : :py:class:`float`
            Mach number in the normal direction to the section.

        S : :py:class:`float`
            Surface of the section.

        Pt : :py:class:`float`
            Stagnation pressure of the flow.

        Tt : :py:class:`float`
            Stagnation temperature of the flow.

        r : :py:class:`float`
            Specific gas constant.

        gamma : :py:class:`float`
            Ratio of specific heats of the gas.


    Returns
    -------

        massflow : :py:class:`float`
            Value of massflow through the section.
    '''
    return S * Pt * (gamma/r/Tt)**0.5 * Mx / (1. + 0.5*(gamma-1.) * Mx**2) ** ((gamma+1) / 2 / (gamma-1))

def machFromMassflow(massflow, S, Pt=101325.0, Tt=288.25, r=287.053, gamma=1.4):
    '''
    Compute the Mach number normal to a section from the massflow rate.

    Parameters
    ----------

        massflow : :py:class:`float`
            Massflow rate through the section.

        S : :py:class:`float`
            Surface of the section.

        Pt : :py:class:`float`
            Stagnation pressure of the flow.

        Tt : :py:class:`float`
            Stagnation temperature of the flow.

        r : :py:class:`float`
            Specific gas constant.

        gamma : :py:class:`float`
            Ratio of specific heats of the gas.


    Returns
    -------

        Mx : :py:class:`float`
            Value of the Mach number in the normal direction to the section.
    '''
    if isinstance(massflow, (list, tuple, np.ndarray)):
        Mx = []
        for i, MF in enumerate(massflow):
            Mx.append(machFromMassflow(MF, S, Pt=Pt, Tt=Tt, r=r, gamma=gamma))
        if isinstance(massflow, np.ndarray):
            Mx = np.array(Mx)
        return Mx
    else:
        # Check that massflow is lower than the chocked massflow
        chocked_massflow = massflowFromMach(1., S, Pt=Pt, Tt=Tt, r=r, gamma=gamma)
        assert massflow < chocked_massflow, "Massflow ({:6.3f}kg/s) is greater than the chocked massflow ({:6.3f}kg/s)".format(massflow, chocked_massflow)
        # Massflow as a function of Mach number
        f = lambda Mx: massflowFromMach(Mx, S, Pt, Tt, r, gamma)
        # Objective function
        g = lambda Mx: f(Mx) - massflow
        # Search for the corresponding Mach Number between 0 and 1
        Mx = scipy.optimize.brentq(g, 0, 1)
        return Mx


################################################################################
################# Boundary Conditions Settings  ################################
################################################################################

def setBC_Walls(t, TurboConfiguration, bladeFamilyNames=['Blade']):
    '''
    Set all the wall boundary conditions in a turbomachinery context, by making
    the following operations:
        * set the rotation speed for all families related to row domains. It is
            defined in:

            >>> TurboConfiguration['Rows'][rowName]['RotationSpeed'] = float

        * set BCs related to each blade.
        * set BCs related to hub. The intervals where the rotation speed is the
            shaft speed (for rotor platforms) are set in the following form:

            >>> TurboConfiguration['HubRotationSpeed'] = [(xmin1, xmax1), ..., (xminN, xmaxN)]

        * set BCs related to shroud. Rotation speed is set to zero.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        TurboConfiguration : dict
            as produced :py:func:`setTurboConfiguration`

        bladeFamilyNames : :py:class:`list` of :py:class:`str`
            list of patterns to find families related to blades.

    '''

    def omegaHubAtX(x):
        omega = np.zeros(x.shape, dtype=float)
        for (x1, x2) in TurboConfiguration['HubRotationSpeed']:
            omega[(x1<=x) & (x<=x2)] = TurboConfiguration['ShaftRotationSpeed']
        omega = np.asfortranarray(omega)
        return omega

    # Add info on row movement (.Solver#Motion)
    for row, rowParams in TurboConfiguration['Rows'].items():
        famNode = I.getNodeFromNameAndType(t, row, 'Family_t')
        J.set(famNode, '.Solver#Motion',
                motion='mobile',
                omega=rowParams['RotationSpeed'],
                axis_pnt_x=0., axis_pnt_y=0., axis_pnt_z=0.,
                axis_vct_x=1., axis_vct_y=0., axis_vct_z=0.)

    # BLADES
    for blade_family in bladeFamilyNames:
        for famNode in I.getNodesFromNameAndType(t, '*{}*'.format(blade_family), 'Family_t'):
            famName = I.getName(famNode)
            row_omega = None
            for row, rowParams in TurboConfiguration['Rows'].items():
                if row in famName:
                    row_omega = rowParams['RotationSpeed']
                    break
            assert row_omega is not None, 'Cannot determine associated row for family {}. '.format(famName)

            I.newFamilyBC(value='BCWallViscous', parent=famNode)
            J.set(famNode, '.Solver#BC',
                    type='walladia',
                    data_frame='user',
                    omega=row_omega,
                    axis_pnt_x=0., axis_pnt_y=0., axis_pnt_z=0.,
                    axis_vct_x=1., axis_vct_y=0., axis_vct_z=0.)

    # HUB
    for famNode in I.getNodesFromNameAndType(t, '*HUB*', 'Family_t'):
        famName = I.getName(famNode)
        I.newFamilyBC(value='BCWallViscous', parent=famNode)
        J.set(famNode, '.Solver#BC',
                type='walladia',
                data_frame='user',
                omega=0.,
                axis_pnt_x=0., axis_pnt_y=0., axis_pnt_z=0.,
                axis_vct_x=1., axis_vct_y=0., axis_vct_z=0.)

        wallHubBC = C.extractBCOfName(t, 'FamilySpecified:{0}'.format(famName))
        wallHubBC = C.node2Center(wallHubBC)
        for w in wallHubBC:
            xw = I.getValue(I.getNodeFromName(w,'CoordinateX'))
            zname, wname = I.getName(w).split(os.sep)
            znode = I.getNodeFromNameAndType(t,zname,'Zone_t')
            wnode = I.getNodeFromNameAndType(znode,wname,'BC_t')
            BCDataSet = I.newBCDataSet(name='BCDataSet#Init', value='Null',
                gridLocation='FaceCenter', parent=wnode)
            J.set(BCDataSet, 'NeumannData',
                    omega=omegaHubAtX(xw))

    # SHROUD
    for famNode in I.getNodesFromNameAndType(t, '*SHROUD*', 'Family_t'):
        I.newFamilyBC(value='BCWallViscous', parent=famNode)
        J.set(famNode, '.Solver#BC',
                type='walladia',
                data_frame='user',
                omega=0.,
                axis_pnt_x=0., axis_pnt_y=0., axis_pnt_z=0.,
                axis_vct_x=1., axis_vct_y=0., axis_vct_z=0.)

def setBC_inj1(t, FamilyName, ImposedVariables, bc=None):
    '''
    Generic function to impose a Boundary Condition ``inj1``. The following
    functions are more specific:
        * :py:func:`setBC_inj1_uniform`
        * :py:func:`setBC_inj1_interpFromFile`

    .. note:: see `elsA Tutorial about inj1 condition <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#inj1/>`_

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        ImposedVariables : dict
            Dictionary of variables to imposed on the boudary condition. Keys
            are variable names, and values must be:
                * either scalars: in that case they are imposed once for the
                    family **FamilyName** in the corresponding ``Family_t`` node.
                * or numpy arrays: in that case they are imposed for the ``BC_t``
                    node **bc**.

        bc : PyTree
            ``BC_t`` node on which the boundary condition will be imposed. Must
                be :py:obj:`None` if the condition must be imposed once in the
                ``Family_t`` node.

    See also
    --------

        setBC_inj1_uniform, setBC_inj1_interpFromFile

    '''
    inlet = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    I._rmNodesByType(inlet, 'FamilyBC_t')
    I.newFamilyBC(value='BCInflowSubsonic', parent=inlet)

    if all([np.ndim(v)==0 for v in ImposedVariables.values()]):
        J.set(inlet, '.Solver#BC', type='inj1', **ImposedVariables)

    else:
        J.set(inlet, '.Solver#BC', type='inj1')

        assert bc is not None

        PointRange = I.getValue(I.getNodeFromType(bc, 'IndexRange_t'))
        bc_shape = PointRange[:, 1] - PointRange[:, 0]
        if bc_shape[0] == 0:
            bc_shape = (bc_shape[1], bc_shape[2])
        elif bc_shape[1] == 0:
            bc_shape = (bc_shape[0], bc_shape[2])
        elif bc_shape[2] == 0:
            bc_shape = (bc_shape[0], bc_shape[1])
        else:
            raise ValueError('Wrong BC shape {} in {}'.format(bc_shape, I.getPath(t, bc)))

        for var, value in ImposedVariables.items():
            assert value.shape == bc_shape, \
                'Wrong shape for variable {}: {} (shape {} for {})'.format(var, value.shape, bc_shape, I.getPath(t, bc))

        BCDataSet = I.newBCDataSet(name='BCDataSet#Init', value='Null',
            gridLocation='FaceCenter', parent=bc)
        J.set(BCDataSet, 'DirichletData', **ImposedVariables)

def setBC_inj1_uniform(t, FamilyName, FluidProperties, ReferenceValues):
    '''
    Set a Boundary Condition ``inj1`` with uniform inflow values. These values
    are them in **ReferenceValues**.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        FluidProperties : dict
            as obtained from :py:func:`computeFluidProperties`

        ReferenceValues : dict
            as obtained from :py:func:`computeReferenceValues`

    See also
    --------

        setBC_inj1, setBC_inj1_interpFromFile

    '''

    # Get turbulent variables names and values
    turbVars = ReferenceValues['FieldsTurbulence']
    turbVars = [var.replace('Density', '') for var in turbVars]
    turbValues = [val/ReferenceValues['Density'] for val in ReferenceValues['ReferenceStateTurbulence']]
    turbDict = dict(zip(turbVars, turbValues))

    # Convert names to inj_tur1 and (if needed) inj_tur2
    if 'TurbulentSANuTilde' in turbDict:
        turbDict = dict(inj_tur1=turbDict['TurbulentSANuTilde'])
    else:
        turbDict['inj_tur1'] = turbDict['TurbulentEnergyKinetic']
        turbDict.pop('TurbulentEnergyKinetic')
        inj_tur2 = [var for var in turbDict if var != 'inj_tur1']
        assert len(inj_tur2) == 1, \
            'Turbulent models with more than 2 equations are not supported yet'
        inj_tur2 = inj_tur2[0]
        turbDict['inj_tur2'] = turbDict[inj_tur2]
        turbDict.pop(inj_tur2)

    ImposedVariables = dict(
        stagnation_pressure = ReferenceValues['PressureStagnation'],
        stagnation_enthalpy = FluidProperties['cp'] * ReferenceValues['TemperatureStagnation'],
        txv                 = 1.0,
        tyv                 = 0.0,
        tzv                 = 0.0,
        **turbDict
        )

    setBC_inj1(t, FamilyName, ImposedVariables)

def setBC_inj1_interpFromFile(t, FamilyName, ReferenceValues, filename, fileformat=None):
    '''
    Set a Boundary Condition ``inj1`` using the field map in the file
    **filename**. It is expected to be a surface with the following variables
    defined at cell centers:

        * the coordinates

        * the stagnation pressure ``'PressureStagnation'``

        * the stagnation enthalpy ``'EnthalpyStagnation'``

        * the three components of the unit vector for the velocity direction:
            ``'VelocityUnitVectorX'``, ``'VelocityUnitVectorY'``, ``'VelocityUnitVectorZ'``

        * the primitive turbulent variables (so not multiplied by density)
            comptuted from ``ReferenceValues['FieldsTurbulence']`` and
            depending on the turbulence model.
            For example: ``'TurbulentEnergyKinetic'`` and
            ``'TurbulentDissipationRate'`` for a k-omega model.

    Field variables will be extrapolated on the BCs attached to the family
    **FamilyName**, except if:

        * the file can be converted in a PyTree

        * with zone names like: ``<ZONE>\<BC>``, as obtained from function
            C.extractBCOfName

        * and all zone names and BC names are consistent with the current tree
            **t**

    In that case, field variables are just read in **filename** and written in
    BCs of **t**.

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        ReferenceValues : dict
            as obtained from :py:func:`computeReferenceValues`

        filename : str
            name of the input filename

        fileformat : optional, str
            format of the input file to be passed to Converter.convertFile2PyTree
            Cassiopee function.

            .. note:: see `available file formats <http://elsa.onera.fr/Cassiopee/Converter.html?highlight=initvars#fileformats>`_

    See also
    --------

        setBC_inj1, setBC_inj1_uniform

    '''

    var2interp = ['PressureStagnation', 'EnthalpyStagnation',
        'VelocityUnitVectorX', 'VelocityUnitVectorY', 'VelocityUnitVectorZ']
    turbVars = ReferenceValues['FieldsTurbulence']
    turbVars = [var.replace('Density', '') for var in turbVars]
    var2interp += turbVars

    donor_tree = C.convertFile2PyTree(filename, format=fileformat)
    FSCenters_nodes = I.getNodesFromType(donor_tree, I.__FlowSolutionCenters__)
    if FSCenters_nodes == []:
        donor_tree = C.node2Center(donor_tree, I.__FlowSolutionNodes__)
        I._rmNodesByNameAndType(donor_tree, I.__FlowSolutionNodes__, 'FlowSolution_t')
        for FS in I.getNodesFromType(donor_tree, 'FlowSolution_t'):
            I.setName(FS, I.__FlowSolutionCenters__)

    inlet_BC_nodes = C.extractBCOfName(t, 'FamilySpecified:{0}'.format(FamilyName))
    I._adaptZoneNamesForSlash(inlet_BC_nodes)
    # hook = None # TODO: Add a hook 
    for w in inlet_BC_nodes:
        bcLongName = I.getName(w)  # from C.extractBCOfName: <zone>\<bc>
        zname, wname = bcLongName.split('\\')
        znode = I.getNodeFromNameAndType(t, zname, 'Zone_t')
        bcnode = I.getNodeFromNameAndType(znode, wname, 'BC_t')

        donor_BC = I.getNodeFromName(donor_tree, bcLongName)
        if not donor_BC:
            print('Interpolate Inflow condition on BC {}...'.format(bcLongName))
            I._rmNodesByType(w, 'FlowSolution_t')
            # if not hook:
            #     hook = C.createHook(donor_tree, function='extractMesh')
            donor_BC = P.extractMesh(donor_tree, w) #, hook=[hook])
            donor_BC = C.node2Center(donor_BC, I.__FlowSolutionNodes__)
            I._rmNodesByName(donor_BC, I.__FlowSolutionNodes__)

        ImposedVariables = dict()
        for var in var2interp:
            ImposedVariables[var] = I.getValue(I.getNodeFromName(donor_BC, var))

        setBC_inj1(t, FamilyName, ImposedVariables, bc=bcnode)
    # C.freeHook(hook)

def setBC_outpres(t, FamilyName, pressure):
    '''
    Set an outflow boundary condition of type ``outpres``.

    .. note:: see `elsA Tutorial about outpres condition <http://elsa.onera.fr/restricted/MU_MT_tuto/latest/Tutos/BCsTutorials/tutorial-BC.html#outpres/>`_

    Parameters
    ----------

        t : PyTree
            Tree to modify

        FamilyName : str
            Name of the family on which the boundary condition will be imposed

        pressure : float
            Value of the static pressure to impose

    '''
    outlet = I.getNodeFromNameAndType(t, FamilyName, 'Family_t')
    I._rmNodesByType(outlet, 'FamilyBC_t')
    I.newFamilyBC(value='BCOutflowSubsonic', parent=outlet)
    J.set(outlet, '.Solver#BC', type='outpres', pressure=pressure)
