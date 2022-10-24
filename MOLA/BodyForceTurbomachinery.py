'''
MOLA - BodyForceTurbomachinery.py

There are 2 kinds of functions in this file :

* Functions for preprocessing, that help to create a mesh adapted for body-force
  and to extract the geometrical parameters needed for the model.

* Functions for coprocessing, that implement body-force models to update source terms 
  during the simulation.

File history:
8/09/2022 - T. Bontemps - Creation
'''

import os
import glob
import numpy as np

import Converter.PyTree as C
import Converter.Internal as I
import Geom.PyTree as D
import Generator.PyTree as G
import Connector.PyTree as X
import Transform.PyTree as T
import Post.PyTree as P

import MOLA.Preprocess as PRE
import MOLA.InternalShortcuts as J
import MOLA.Wireframe as W

##################################################################################
# Functions for preprocessing
# The following functions help to create a mesh adapted for body-force
# and to extract the geometrical parameters needed for the model
##################################################################################


def replaceRowWithBodyForceMesh(t, BodyForceRows):
    '''
    In the mesh **t**, replace the row domain corresponding to the family 
    **row** by a mesh adapted to body-force.

    Parameters
    ----------
        t : PyTree
            input mesh, from Autogrid for instance.

        BodyForceRows : :py:class:`dict` of :py:class:`dict`
            Parameters to generate body-force meshes for the specified rows.
            The keys of the dictionary corresponds to the names of row 
            families that will be replaced by meshes adapted to body-force.
            For each row, the associated value is a dictionary (may be empty)
            corresponding to the arguments to give to function 
            :py:func:`buildBodyForceMeshForOneRow`. See the documentation of 
            this function.

            .. note:: 
                If the parameter **meshType** is not given, it is 
                automatically set depending on the initial mesh.
    
    Returns
    -------

        newRowMeshes : :py:class:`list` of PyTree

            New mesh, identical to the input mesh **t** except that the domain 
            corresponding to the family **row** has been replaced by a mesh adapted
            to body-force.

    '''
    def printCells(n):
        if n > 1e6:
            return f'{n/1e6:.1f}M'
        elif n > 1e3:
            return f'{n/1e3:.0f}k'
        else:
            return f'{n}'

    def restoreBCFamilies(zones, newRowMesh):
        FamilyBCNames = []
        for zone in zones:
            for BC in I.getNodesFromType2(zone, 'BC_t'):
                try:
                    FamilyBC = I.getValue(I.getNodeFromName1(BC, 'FamilyName'))
                    if FamilyBC not in FamilyBCNames:
                        FamilyBCNames.append(FamilyBC)
                except:
                    pass

        for FamilyBC in FamilyBCNames:
            FamilyBCLower = FamilyBC.lower()
            if any([name in FamilyBCLower for name in ['inflow', 'inlet', 'right']]):
                I._renameNode(newRowMesh, 'Inflow', FamilyBC)
            elif any([name in FamilyBCLower for name in ['outflow', 'outlet', 'left']]):
                I._renameNode(newRowMesh, 'Outflow', FamilyBC)
            elif any([name in FamilyBCLower for name in ['hub', 'moyeu']]):
                I._renameNode(newRowMesh, 'Hub', FamilyBC)
            elif any([name in FamilyBCLower for name in ['shroud', 'carter']]):
                I._renameNode(newRowMesh, 'Shroud', FamilyBC)

    newRowMeshes = []

    # Join HUB and SHROUD families
    J.joinFamilies(t, 'HUB')
    J.joinFamilies(t, 'SHROUD')

    for row, BodyForceParams in BodyForceRows.items():

        RowFamilyNode = I.getNodeFromNameAndType(t, row, 'Family_t')
        BladeNumber = I.getValue(I.getNodeFromName(RowFamilyNode, 'BladeNumber'))

        zones = C.getFamilyZones(t, row)
        NumberOfCellsInitial = C.getNCells(zones)
        if not 'meshType' in BodyForceParams:
            if PRE.hasAnyUnstructuredZones(zones):
                BodyForceParams['meshType'] = 'unstructured'
            else:
                BodyForceParams['meshType'] = 'structured'

        # Get the meridional info from rowTree
        meridionalMesh = extractRowGeometricalData(t, row)

        newRowMesh = buildBodyForceMeshForOneRow(meridionalMesh,
                                                 NumberOfBlades=BladeNumber,
                                                 **BodyForceParams
                                                 )

        restoreBCFamilies(zones, newRowMesh)
        I._renameNode(newRowMesh, 'Fluid', row)
        I._renameNode(newRowMesh, 'upstream', f'{row}_upstream')
        I._renameNode(newRowMesh, 'downstream', f'{row}_downstream')
        I._renameNode(newRowMesh, 'bodyForce', f'{row}_bodyForce')

        newRowMeshes.append(newRowMesh)

        for zone in zones:
            I.rmNode(t, zone)

        I._rmNodesByNameAndType(t, f'{row}_*Blade*', 'Family_t')
        I._rmNodesByNameAndType(t, f'{row}_*BLADE*', 'Family_t')
        I._rmNodesByNameAndType(t, f'{row}_*Aube*', 'Family_t')
        I._rmNodesByNameAndType(t, f'{row}_*AUBE*', 'Family_t')

        NumberOfCellsBFM = C.getNCells(newRowMesh)

        print(f'Number of cells for the family {row} in the initial mesh : {printCells(NumberOfCellsInitial)}')
        print(f'Number of cells for the family {row} in the BFM mesh     : {printCells(NumberOfCellsBFM)}')
        print(f'    --> Reduction with a factor {NumberOfCellsInitial/NumberOfCellsBFM:.1f}')

    return t, newRowMeshes


def buildBodyForceMeshForOneRow(t, NumberOfBlades, 
                                NumberOfRadialPoints=None, NumberOfAxialPointsBeforeLE=None,
                                NumberOfAxialPointsBetweenLEAndTE=None, NumberOfAxialPointsAfterTE=None,
                                NumberOfAzimutalPoints=5, AzimutalAngleDeg=2.,
                                RadialDistribution=None, AxialDistributionBeforeLE=None,
                                AxialDistributionBetweenLEAndTE=None, AxialDistributionAfterTE=None,
                                meshType='structured',
                                model='hall',
                                ):
    '''
    Build a mesh suitable for body-force simulations from a PyTree with lines corresponding
    to hub, shroud, inlet, outlet, leading edge and trailing edge, for a single row.

    Parameters
    ----------
        t : PyTree
            input tree. Must contains the following 1D zones (lines), named: Hub, Shroud, 
            Inlet, Outlet, LeadingEdge, TrailingEdge.

        NumberOfBlades : int
            Number of blades (for the 360deg row). Used to compute the blockage coefficient.

        NumberOfRadialPoints : :py:class:`int` or :py:obj:`None`
           Number of points in the radial direction. If :py:obj:`None`, then **RadialDistribution**
           must be a curve whose distribution is to be copied.

        NumberOfAxialPointsBeforeLE : :py:class:`int` or :py:obj:`None`
            Number of points in the axial direction between the inlet and the blade leading edge.
            If :py:obj:`None`, then **AxialDistributionBeforeLE** must be a curve whose distribution is to be copied.

        NumberOfAxialPointsBetweenLEAndTE : :py:class:`int` or :py:obj:`None`
            Number of points in the axial direction between the blade leading edge and trailing edge.
            If :py:obj:`None`, then **AxialDistributionBetweenLEAndTE** must be a curve whose distribution is to be copied.

        NumberOfAxialPointsAfterTE : :py:class:`int` or :py:obj:`None`
            Number of points in the axial direction between the blade trailing edge and the outlet.
            If :py:obj:`None`, then **AxialDistributionAfterTE** must be a curve whose distribution is to be copied.

        NumberOfAzimutalPoints : int, optional
            Number of points in the azimutal direction, by default 5.

        AzimutalAngleDeg : _type_, optional
            Azimutal extension in degrees of the output computational domain, by default 10.

        RadialDistribution : :py:class:`dict` or zone
            Points distribution in the radial direction. For details, see the documentation of
            :func:`MOLA.Wireframe.discretize`. Could be for instance: 
            
            >>> dict(kind='tanhTwoSides',FirstCellHeight=CellWidthAtWall,LastCellHeight=CellWidthAtWall)

            If not given, then the distribution in the initial mesh is used.

        AxialDistributionBeforeLE : :py:class:`dict` or zone
            Points distribution in the axial direction between the inlet and the blade leading edge.
            If not given, then the distribution in the initial mesh is used.

        AxialDistributionBetweenLEAndTE : :py:class:`dict` or zone
            Points distribution in the axial direction between the blade leading edge and trailing edge.
            If not given, then the distribution in the initial mesh is used.

        AxialDistributionAfterTE : :py:class:`dict` or zone
            Points distribution in the axial direction between the blade trailing edge and the outlet.
            If not given, then the distribution in the initial mesh is used.

        meshType : str, optional
            Type of the mesh. Should be 'structured' or 'unstructured'. By default 'structured'

    Returns
    -------
        PyTree
            3D mesh with connectivities and periodic connectivities, suitable for a body-force simulation,
            ready to be used in :func:`MOLA.WorkflowCompressor.prepareMainCGNS4ElsA`.

    '''
    def rediscretize_LE_and_TE(LeadingEdgeTmp, TrailingEdgeTmp):
        # Compute cell width on hub near LE and TE intersections
        HubTmp = D.getCurvilinearAbscissa(HubBetweenLEAndTE)
        s = I.getValue(I.getNodeFromName(HubTmp, 's')) * D.getLength(HubTmp)
        ds = s[1:]-s[:-1]
        CellWidthAtHubLE = ds[0]
        CellWidthAtHubTE = ds[-1]
        # Compute cell width on shroud near LE and TE intersections
        ShroudTmp = D.getCurvilinearAbscissa(ShroudBetweenLEAndTE)
        s = I.getValue(I.getNodeFromName(ShroudTmp, 's')) * \
            D.getLength(ShroudTmp)
        ds = s[1:]-s[:-1]
        CellWidthAtShroudLE = ds[0]
        CellWidthAtShroudTE = ds[-1]

        # New distributions for LE and TE. The end cells near hub and shroud has a the same width than
        # cells on hub and shroud. Thus, the intersection will be well computed because cells have the same width.
        tmpDistributionLE = dict(
            kind='tanhTwoSides', FirstCellHeight=CellWidthAtHubLE, LastCellHeight=CellWidthAtShroudLE)
        tmpDistributionTE = dict(
            kind='tanhTwoSides', FirstCellHeight=CellWidthAtHubTE, LastCellHeight=CellWidthAtShroudTE)

        LE = W.discretize(LeadingEdgeTmp, Distribution=tmpDistributionLE)
        TE = W.discretize(TrailingEdgeTmp, Distribution=tmpDistributionTE)
        return LE, TE
    
    # Get the lines defining the geometry 
    Hub = I.getNodeFromName2(t, 'Hub')
    Shroud = I.getNodeFromName2(t, 'Shroud')
    LeadingEdge = I.getNodeFromName2(t, 'LeadingEdge')
    TrailingEdge = I.getNodeFromName2(t, 'TrailingEdge')
    Inlet = I.getNodeFromName2(t, 'Inlet')
    Outlet = I.getNodeFromName2(t, 'Outlet')

    # Extrapolate LE and TE lines to be sure that they intersect hub and shroud lines
    ExtrapDistance = 0.05 * W.getLength(LeadingEdge)
    LeadingEdgeTmp = W.extrapolate(LeadingEdge, ExtrapDistance, opposedExtremum=False)
    LeadingEdgeTmp = W.extrapolate(LeadingEdgeTmp, ExtrapDistance, opposedExtremum=True)
    ExtrapDistance = 0.05 * W.getLength(TrailingEdge)
    TrailingEdgeTmp = W.extrapolate(TrailingEdge, ExtrapDistance, opposedExtremum=False)
    TrailingEdgeTmp = W.extrapolate(TrailingEdgeTmp, ExtrapDistance, opposedExtremum=True)

    # Compute curves intersections and split curves
    # Split Hub
    cut_point_1 = W.getNearestIntersectingPoint(Hub, LeadingEdgeTmp)
    cut_point_2 = W.getNearestIntersectingPoint(Hub, TrailingEdgeTmp)
    HubBetweenLEAndTE = W.trimCurveAlongDirection(Hub, np.array([1., 0, 0]), cut_point_1, cut_point_2)
    HubBetweenInletAndLE = W.trimCurveAlongDirection(Hub, np.array([1., 0, 0]), np.array([-999, 0, 0]), cut_point_1)
    HubBetweenTEAndOutlet = W.trimCurveAlongDirection(Hub, np.array([1., 0, 0]), cut_point_2, np.array([999, 0, 0]))
    # Split Shroud
    cut_point_1 = W.getNearestIntersectingPoint(Shroud, LeadingEdgeTmp)
    cut_point_2 = W.getNearestIntersectingPoint(Shroud, TrailingEdgeTmp)
    ShroudBetweenLEAndTE = W.trimCurveAlongDirection(Shroud, np.array([1., 0, 0]), cut_point_1, cut_point_2)
    ShroudBetweenInletAndLE = W.trimCurveAlongDirection(Shroud, np.array([1., 0, 0]), np.array([-999, 0, 0]), cut_point_1)
    ShroudBetweenTEAndOutlet = W.trimCurveAlongDirection(Shroud, np.array([1., 0, 0]), cut_point_2, np.array([999, 0, 0]))
    # Rediscretize LE and TE
    # This operation is needed, otherwise the intersection may be badly defined, 
    # leading to negative volume cells 
    LeadingEdgeTmp, TrailingEdgeTmp = rediscretize_LE_and_TE(LeadingEdgeTmp, TrailingEdgeTmp)
    # Split LE
    cut_point_1 = W.getNearestIntersectingPoint(LeadingEdgeTmp, Hub)
    cut_point_2 = W.getNearestIntersectingPoint(LeadingEdgeTmp, Shroud)
    LeadingEdgeTmp = W.trimCurveAlongDirection(LeadingEdgeTmp, np.array([0, 1, 0]), cut_point_1, cut_point_2)
    # Split TE
    cut_point_1 = W.getNearestIntersectingPoint(TrailingEdgeTmp, Hub)
    cut_point_2 = W.getNearestIntersectingPoint(TrailingEdgeTmp, Shroud)
    TrailingEdgeTmp = W.trimCurveAlongDirection(TrailingEdgeTmp, np.array([0, 1, 0]), cut_point_1, cut_point_2)

    # Modify ends of LE line to collapse them on end points hub and shroud lines.
    # Else there will be an error during TFI (TypeError: TFI: input arrays must be C0)
    # For the LE
    inter1 = W.point(HubBetweenLEAndTE, index=0)
    inter2 = W.point(ShroudBetweenLEAndTE, index=0)
    x, y = J.getxy(LeadingEdgeTmp)
    x[0], y[0] = inter1[0], inter1[1]  # Modify the first point of the line
    x[-1], y[-1] = inter2[0], inter2[1]  # Modify the last point of the line
    # For the TE
    inter1 = W.point(HubBetweenLEAndTE, index=-1)
    inter2 = W.point(ShroudBetweenLEAndTE, index=-1)
    x, y = J.getxy(TrailingEdgeTmp)
    x[0], y[0] = inter1[0], inter1[1]  # Modify the first point of the line
    x[-1], y[-1] = inter2[0], inter2[1]  # Modify the last point of the line

    # Rename lines
    I.setName(HubBetweenLEAndTE, 'HubBetweenLEAndTE')
    I.setName(HubBetweenInletAndLE, 'HubBetweenInletAndLE')
    I.setName(HubBetweenTEAndOutlet, 'HubBetweenTEAndOutlet')
    I.setName(ShroudBetweenLEAndTE, 'ShroudBetweenLEAndTE')
    I.setName(ShroudBetweenInletAndLE, 'ShroudBetweenInletAndLE')
    I.setName(ShroudBetweenTEAndOutlet, 'ShroudBetweenTEAndOutlet')
    I.setName(LeadingEdge, 'LeadingEdge')
    I.setName(TrailingEdge, 'TrailingEdge')

    # If points distribution are not given, they are the same than in the initial mesh
    if not RadialDistribution:
        RadialDistribution = LeadingEdge
    if not AxialDistributionBeforeLE:
        AxialDistributionBeforeLE = HubBetweenInletAndLE
    if not AxialDistributionBetweenLEAndTE:
        AxialDistributionBetweenLEAndTE = HubBetweenLEAndTE
    if not AxialDistributionAfterTE:
        AxialDistributionAfterTE = HubBetweenTEAndOutlet

    # Apply the points distribution for each line
    Inlet = W.discretize(Inlet, N=NumberOfRadialPoints, Distribution=RadialDistribution)
    Outlet = W.discretize(Outlet, N=NumberOfRadialPoints, Distribution=RadialDistribution)
    LeadingEdge = W.discretize(LeadingEdgeTmp, N=NumberOfRadialPoints, Distribution=RadialDistribution)
    TrailingEdge = W.discretize(TrailingEdgeTmp, N=NumberOfRadialPoints, Distribution=RadialDistribution)

    HubBetweenInletAndLE = W.discretize(HubBetweenInletAndLE, N=NumberOfAxialPointsBeforeLE, Distribution=AxialDistributionBeforeLE)
    ShroudBetweenInletAndLE = W.discretize(ShroudBetweenInletAndLE, N=NumberOfAxialPointsBeforeLE, Distribution=AxialDistributionBeforeLE)

    HubBetweenLEAndTE = W.discretize(HubBetweenLEAndTE, N=NumberOfAxialPointsBetweenLEAndTE, Distribution=AxialDistributionBetweenLEAndTE)
    ShroudBetweenLEAndTE = W.discretize(ShroudBetweenLEAndTE, N=NumberOfAxialPointsBetweenLEAndTE, Distribution=AxialDistributionBetweenLEAndTE)

    HubBetweenTEAndOutlet = W.discretize(HubBetweenTEAndOutlet, N=NumberOfAxialPointsAfterTE, Distribution=AxialDistributionAfterTE)
    ShroudBetweenTEAndOutlet = W.discretize(ShroudBetweenTEAndOutlet, N=NumberOfAxialPointsAfterTE, Distribution=AxialDistributionAfterTE)


    # Use TFI to create the 2D mesh
    upstream = G.TFI([Inlet, LeadingEdge, HubBetweenInletAndLE, ShroudBetweenInletAndLE])
    bodyForce = G.TFI([LeadingEdge, TrailingEdge, HubBetweenLEAndTE, ShroudBetweenLEAndTE])
    downstream = G.TFI([TrailingEdge, Outlet, HubBetweenTEAndOutlet, ShroudBetweenTEAndOutlet])
    I.setName(upstream, 'upstream')
    I.setName(bodyForce, 'bodyForce')
    I.setName(downstream, 'downstream')

    # bodyForceCoarse = G.TFI([
    #     W.discretize(LeadingEdge, N=10),
    #     W.discretize(TrailingEdge, N=10),
    #     W.discretize(HubBetweenLEAndTE, N=10),
    #     W.discretize(ShroudBetweenLEAndTE, N=10),
    #     ])

    if model == 'hall':
        # Interpolate skeleton data
        Skeleton = I.getNodeFromName2(t, 'Skeleton')
        if Skeleton:
            P._extractMesh(Skeleton, bodyForce, order=2, extrapOrder=0) #, constraint=0.)
            # P._extractMesh(Skeleton, bodyForceCoarse, order=2, extrapOrder=0)
            bodyForce = computeBlockage(bodyForce, NumberOfBlades)
            bodyForce = C.node2Center(bodyForce, I.__FlowSolutionNodes__)
            I._rmNodesByName1(bodyForce, I.__FlowSolutionNodes__)

    mesh2d = C.newPyTree(['Base', [upstream, bodyForce, downstream]])

    # Make a partial revolution to build the 3D mesh
    mesh3d = D.axisym(mesh2d, (0, 0, 0), (1, 0, 0), angle=AzimutalAngleDeg, Ntheta=NumberOfAzimutalPoints)

    bodyForce = I.getNodeFromName2(mesh3d, 'bodyForce')
    # C._initVars(bodyForce, '{centers:isf}=1')
    if model == 'hall':
        # Move coordinates to cell center and compute radius and azimuthal angle
        for coord in ['CoordinateX', 'CoordinateY', 'CoordinateZ']:
            C._node2Center__(bodyForce, coord)
        C._initVars(bodyForce, '{centers:radius}=sqrt({centers:CoordinateY}**2+{centers:CoordinateZ}**2)')
        C._initVars(bodyForce, '{centers:theta}=arctan2({centers:CoordinateZ},{centers:CoordinateY})')
        # Rename x,y,z at cell centers, otherwise Cassiopee and elsA have a problem reading the tree...
        # Notice that renaming CoordinateX to x does not solve the problem because Cassiopee know x
        # as an alias for CoordinateX, so it will fail to read the mesh anyway.
        FS = I.getNodeFromType1(bodyForce, 'FlowSolution_t')
        I._renameNode(FS, 'CoordinateX', 'xCell')
        I._renameNode(FS, 'CoordinateY', 'yCell')
        I._renameNode(FS, 'CoordinateZ', 'zCell')

    elif model == 'Tspread':
        G._getVolumeMap(bodyForce)
        volumeCell = I.getValue(I.getNodeFromName(bodyForce, 'vol'))
        # C._initVars(bodyForce,
        #             '{{centers:volumicFraction}}={{centers:vol}}/{totalVolume}'.format(totalVolume=np.sum(volumeCell)))
        C._initVars(bodyForce,'{{centers:totalVolume}}={totalVolume}'.format(totalVolume=np.sum(volumeCell)))
        I._rmNodesByNameAndType(bodyForce, 'vol', 'DataArray_t')
        

    base = I.getBases(mesh3d)[0]
    upstream = I.getNodeFromNameAndType(base, 'upstream', 'Zone_t')
    bodyForce = I.getNodeFromNameAndType(base, 'bodyForce', 'Zone_t')
    downstream = I.getNodeFromNameAndType(base, 'downstream', 'Zone_t')

    # Add BC and Families
    C._addFamily2Base(base, 'Fluid')
    for zone in I.getZones(base):
        C._tagWithFamily(zone, 'Fluid')
    # C._addFamily2Base(base, 'BFM')
    # C._tagWithFamily(bodyForce, 'BFM', add=True)

    FamilyNode = I.newFamily(name='Inflow', parent=base)
    I.newFamilyBC(value='BCInflow', parent=FamilyNode)
    FamilyNode = I.newFamily(name='Outflow', parent=base)
    I.newFamilyBC(value='BCOutflow', parent=FamilyNode)
    FamilyNode = I.newFamily(name='Hub', parent=base)
    I.newFamilyBC(value='BCWallViscous', parent=FamilyNode)
    FamilyNode = I.newFamily(name='Shroud', parent=base)
    I.newFamilyBC(value='BCWallViscous', parent=FamilyNode)

    C._addBC2Zone(upstream, 'inflow',
                  'FamilySpecified:Inflow',   wrange='imin')
    C._addBC2Zone(downstream, 'outflow',
                  'FamilySpecified:Outflow', wrange='imax')
    for zone in I.getZones(base):
        C._addBC2Zone(zone, 'hub', 'FamilySpecified:Hub',    wrange='jmin')
        C._addBC2Zone(zone, 'shroud', 'FamilySpecified:Shroud', wrange='jmax')

    if meshType == 'structured':
        # Periodic conditions
        X.connectMatch(mesh3d, tol=1e-8)
        mesh3d = X.connectMatchPeriodic(
            mesh3d, rotationAngle=[AzimutalAngleDeg, 0., 0.], tol=1e-8)
    elif meshType == 'unstructured':
        # Periodic conditions
        mesh3d = C.convertArray2NGon(mesh3d, recoverBC=1)
        X.connectMatch(mesh3d, tol=1e-8)
        mesh3d = X.connectMatchPeriodic(
            mesh3d, rotationAngle=[AzimutalAngleDeg, 0., 0.], tol=1e-8)
        I._createElsaHybrid(mesh3d, method=1)
    else:
        raise ValueError(
            "The argument meshType must be 'structured' or 'unstructured'")

    I.checkPyTree(mesh3d)
    I._correctPyTree(mesh3d)
    I._renameNode(mesh3d, I.__FlowSolutionCenters__, 'FlowSolution#DataSourceTerm')

    return mesh3d


def extractRowGeometricalData(mesh, row, save=False):
    '''
    Extract geometry data from the input **mesh** for the family **row**.
    The output tree may be passed to function :py:func:`buildBodyForceMeshForOneRow`.

    .. important:: Dependency to Ersatz

    Parameters
    ----------
    mesh : PyTree
        input mesh
    row : str
        Name of the family of the blade of interest
    save : bool, optional
        If :p:obj:`True`, save the output tree with the name 'BodyForceData_{row}.cgns'. 

    Returns
    -------
    PyTree
        The ouput tree has the following 1D zones (lines): 
        Hub, Shroud, LeadingEdge, TrailingEdge, Inlet, Outlet.
        The last zone 'Skeleton' is a 2D zone, located between the LE and the TE, 
        with geometrical data on the blade:
        'nx', 'nr', 'nt', 'thickness', 'AbscissaFromLE'
    '''

    def profilesExtractionAndAnalysis(tree, row, 
                                        zones_for_meridional_lines_extraction=None,
                                        zones_for_blade_profiles_extraction=None,
                                        directory_meridional_lines='meridional_lines',
                                        directory_profiles='profiles'
                                        ):

        import Ersatz as EZ

        zones = C.getFamilyZones(tree, row)
        if zones_for_meridional_lines_extraction is None:
            zones_for_meridional_lines_extraction = []
            # Attention! Ordre de l'amont a l'aval

            for zone in zones:
                zoneName = I.getName(zone)
                if any([zoneName.endswith(suffix) for suffix in ['_upS', '_upStream', '_upstream']]):
                    zones_for_meridional_lines_extraction.append(zoneName)
            for zone in zones:
                zoneName = I.getName(zone)
                if zoneName.endswith('_up'):
                    zones_for_meridional_lines_extraction.append(zoneName)
            for zone in zones:
                zoneName = I.getName(zone)
                if any([zoneName.endswith(suffix) for suffix in ['_downS', '_downStream', '_downstream']]):
                    zones_for_meridional_lines_extraction.append(zoneName)

        if zones_for_blade_profiles_extraction is None:
            zones_for_blade_profiles_extraction = []
            
            FamilyBCList = list(C.getFamilyBCNamesDict(tree))
            BladeFamilies = []
            for fam in FamilyBCList:
                if any([fam.endswith(suffix) for suffix in ['BLADE', 'Blade', 'blade', 'AUBE', 'Aube', 'aube']]):
                    BladeFamilies.append(fam)

            for zone in zones:
                zoneName = I.getName(zone)
                for BladeFamily in BladeFamilies:
                    BCs = C.getFamilyBCs(zone, BladeFamily)
                    if BCs and not ('gap' in zoneName or 'tip' in zoneName):
                        zones_for_blade_profiles_extraction.append(zoneName)
                        break

        print(f'Extraction of hub and shroud lines from zones: {", ".join(zones_for_meridional_lines_extraction)}')
        print(f'Extraction of profiles from zones: {" ".join(zones_for_blade_profiles_extraction)}')
        assert len(zones_for_meridional_lines_extraction) > 0 and len(zones_for_blade_profiles_extraction) > 0
         
        # if directory doesnt exist create it
        if (not(os.path.isdir(directory_meridional_lines))):
            os.mkdir(directory_meridional_lines)

        # if directory doesnt exist create it
        if (not(os.path.isdir(directory_profiles))):
            os.mkdir(directory_profiles)

        # recuperer le nbre de lignes meridiennes
        for zone in I.getNodesFromType(tree, 'Zone_t'):
            if I.getName(zone) in zones_for_meridional_lines_extraction[0]:
                meridional_lines_number = I.getZoneDim(zone)[2]

        for mi in range(meridional_lines_number):
            # extract meridional lines
            tree_t = C.newPyTree(['Base', 3])
            for zm in zones_for_meridional_lines_extraction:
                for zone in I.getNodesFromType(tree, 'Zone_t'):
                    if I.getName(zone) == zm:
                        zone_i_dim = I.getZoneDim(zone)[1]
                        zone_j_dim = I.getZoneDim(zone)[2]
                        zone_k_dim = I.getZoneDim(zone)[3]

                        zone_t = T.subzone(zone, (zone_i_dim, mi+1, 1), (zone_i_dim, mi+1, zone_k_dim))
                        zone_t = T.reorder(zone_t, (-1, 2, 3))
                        zone_t = C.initVars(zone_t, '{myX}={CoordinateZ}')
                        zone_t = C.initVars(zone_t, '{r}=({CoordinateX}**2+{CoordinateY}**2)**0.5')
                        tree_t[2][1][2].append(zone_t)
            tree_t = C.extractVars(tree_t, ['myX', 'r'])
            C.convertPyTree2File(tree_t, directory_meridional_lines+'/merid%03d.dat' % (mi+1))
            os.system('sed -i {} -e s/"myX"/"x"/'.format(directory_meridional_lines+'/merid%03d.dat' % (mi+1)))

        # extract blade profiles
        for zone in I.getNodesFromType(tree, 'Zone_t'):
            if I.getName(zone) in zones_for_blade_profiles_extraction:
                _, _, zone_j_dim, zone_k_dim, _ = I.getZoneDim(zone)
                for mi in range(zone_j_dim):
                    tree_t = C.newPyTree(['Base', 3])
                    zone_t = T.subzone(zone, (1, mi+1, 1), (1, mi+1, zone_k_dim))
                    tree_t[2][1][2].append(zone_t)
                    tree_t = C.initVars(tree_t, '{myX}={CoordinateZ}')
                    tree_t = C.initVars(tree_t, '{myY}={CoordinateX}')
                    tree_t = C.initVars(tree_t, '{myZ}={CoordinateY}')
                    tree_t = C.extractVars(tree_t, ['myX', 'myY', 'myZ'])
                    C.convertPyTree2File(tree_t, directory_profiles+'/profile%03d.dat' % (mi+1))
                    os.system('sed -i {} -e s/"myX"/"x"/'.format(directory_profiles+'/profile%03d.dat' % (mi+1)))
                    os.system('sed -i {} -e s/"myY"/"y"/'.format(directory_profiles+'/profile%03d.dat' % (mi+1)))
                    os.system('sed -i {} -e s/"myZ"/"z"/'.format(directory_profiles+'/profile%03d.dat' % (mi+1)))

        nprofil = len(os.listdir(directory_profiles))
        Npro = nprofil
        step = (nprofil-1) / float(Npro-1)
        ind_range = [int(np.ceil(step*n)) for n in range(Npro)]

        ezpb = EZ.Ersatz()
        # hub and shroud information (needed for the "height" variable calculation)
        ezpb.set('hub', '%s/merid001.dat' % directory_meridional_lines)
        ezpb.set('shroud', '%s/merid%03d.dat' % (directory_meridional_lines, nprofil))

        profile = []
        for n in ind_range:
            profile.append(EZ.Profile(ezpb))
            profile[-1].set('file', '%s/profile%03d.dat' %(directory_profiles, n+1))
            # associate meridional line
            profile[-1].set('meridline', '%s/merid%03d.dat' %(directory_meridional_lines, n+1))
            # LE index
            # Si trop de points Autogrid buggue (Tangent break error)
            profile[-1].set('ns', 101)
            # profile[-1].set('nsk', 100)

        # geometric analysis
        ezpb.set('extract_skeleton', 1)
        ezpb.set('extract_thickness', 1)
        ezpb.set('extract_chord', 1)
        ezpb.set('extract_TE', 1)
        ezpb.set('extract_LE', 1)

        ezpb.submit()
        ezpb.compute()

        os.system('rm -f EZMesh*.plt')


    def readEndWallLine(filenane):
        tree = C.convertFile2PyTree(filenane)
        x_zones = []
        r_zones = []
        for zone in I.getZones(tree):
            x_zones.append(I.getValue(I.getNodeFromName(zone, 'CoordinateX')))
            r_zones.append(I.getValue(I.getNodeFromName(zone, 'r')))
        # Sort by x
        ind_xmin = sorted(enumerate(x_zones), key=lambda x: np.amin(x[1]))

        xhub = []
        rhub = []
        for i, xp in ind_xmin:
            xhub += list(xp)
            rhub += list(r_zones[i])
        return np.array(xhub), np.array(rhub)

    def readLEorTE(filename):
        tree = C.convertFile2PyTree(filename)
        x, y, z = J.getxyz(I.getNodeFromType(tree, 'Zone_t'))
        r = np.sqrt(y**2 + z**2)
        return x, r

    def curve(name, x, y, z=None):
        if z is None: z = np.zeros(x.size)
        curve = J.createZone(name, [x, y, z], ['CoordinateX', 'CoordinateY', 'CoordinateZ'])
        return curve


    directory_meridional_lines = 'meridional_lines_{}'.format(row)
    directory_profiles = 'profiles_{}'.format(row)
    profilesExtractionAndAnalysis(mesh, row,
                                directory_meridional_lines=directory_meridional_lines,
                                directory_profiles=directory_profiles)

    chordTree = C.convertFile2PyTree('chord.dat')
    C._extractVars(chordTree, ['chord', 'chordx'])

    thicknessTree = C.convertFile2PyTree('thickness.dat')
    C._extractVars(thicknessTree, ['xsc', 'thickness', 'mnorm'])

    skeletonTree = C.convertFile2PyTree('skeleton.dat')
    G._getNormalMap(skeletonTree)
    skeletonTree = C.center2Node(skeletonTree, 'centers:sx')
    skeletonTree = C.center2Node(skeletonTree, 'centers:sy')
    skeletonTree = C.center2Node(skeletonTree, 'centers:sz')
    I._rmNodesByName(skeletonTree, 'FlowSolution#Centers')
    C._initVars(skeletonTree, '{theta}=arctan2({CoordinateZ}, {CoordinateY})')
    C._initVars(skeletonTree, '{CoordinateY}=({CoordinateY}**2+{CoordinateZ}**2)**0.5')
    C._initVars(skeletonTree, '{CoordinateZ}=0.')
    C._initVars(skeletonTree, '{sr}= cos({theta})*{sy} + sin({theta})*{sz}') 
    C._initVars(skeletonTree, '{st}=-sin({theta})*{sy} + cos({theta})*{sz}')
    C._initVars(skeletonTree, '{nx}={sx}/({sx}**2+{sy}**2+{sz}**2)**0.5')
    C._initVars(skeletonTree, '{nr}={sr}/({sx}**2+{sy}**2+{sz}**2)**0.5')
    C._initVars(skeletonTree, '{nt}={st}/({sx}**2+{sy}**2+{sz}**2)**0.5')
    C._extractVars(skeletonTree, ['CoordinateX', 'CoordinateY', 'CoordinateZ', 'nx', 'nr', 'nt'])

    skeletonZone = I.getZones(skeletonTree)[0]

    I.setName(skeletonZone, 'Skeleton')
    FlowSolution = I.getNodeFromType1(skeletonZone, 'FlowSolution_t')

    for node in I.getNodesFromType(thicknessTree, 'DataArray_t'):
        I.addChild(FlowSolution, node)

    shape = I.getValue(I.getNodeFromType(FlowSolution, 'DataArray_t')).shape
    for node in I.getNodesFromType(chordTree, 'DataArray_t'):
        value = I.getValue(node)
        reshapedValue = np.zeros(shape)
        for i in range(shape[0]):
            reshapedValue[i, :] = value
        I.newDataArray(name=I.getName(node), value=reshapedValue, parent=FlowSolution)

    C._initVars(skeletonTree, '{AbscissaFromLE}={mnorm}*{chord}')
    C._rmVars(skeletonTree, ['xsc', 'mnorm', 'chord', 'chordx'])

    xhub, rhub = readEndWallLine(f'{directory_meridional_lines}/merid001.dat')
    merid_files = glob.glob(f'{directory_meridional_lines}/merid*.dat')
    shroud_file = sorted(merid_files, key=lambda name: int(name.split('.dat')[0].split('/merid')[-1]))[-1]
    print(shroud_file)
    xshroud, rshroud = readEndWallLine(shroud_file)
    xLE, rLE = readLEorTE('LE.dat')
    xTE, rTE = readLEorTE('TE.dat')

    Hub          = curve('Hub', xhub, rhub)
    Shroud       = curve('Shroud', xshroud, rshroud)
    LeadingEdge  = curve('LeadingEdge', xLE, rLE)
    TrailingEdge = curve('TrailingEdge', xTE, rTE)
    Inlet        = curve('Inlet', np.array([xhub[0], xshroud[0]]), np.array([rhub[0], rshroud[0]]))
    Outlet       = curve('Outlet', np.array([xhub[-1], xshroud[-1]]), np.array([rhub[-1], rshroud[-1]]))

    t = C.newPyTree(['Base', [Hub, Shroud, LeadingEdge,
                    TrailingEdge, Inlet, Outlet, skeletonZone]])

    os.system(f'rm -rf {directory_meridional_lines}')
    os.system(f'rm -rf {directory_profiles}')
    os.system('rm -f LE.dat TE.dat chord.dat thickness.dat skeleton.dat')

    if save:
        C.convertPyTree2File(t, f'BodyForceData_{row}.cgns')

    return t


def computeBlockage(t, Nblades, eps=1e-8):
    '''
    Compute the blockage.

    Parameters
    ----------
    t : PyTree
        input tree. Must contains the nodes 'thickness' and 'nt' (azimuthal component 
        of the unit vector normal to the blade).
    Nblades : int
        Number of blades in the row (used to compute the pitch)
    eps : float, optional
        numerical parameter added to 'nt' square, to prevent division by zero. By default 1e-8

    Returns
    -------
    PyTree 
        Updated tree, with the new nodes 'blockage', 'gradxb' and 'gradrb'.
    '''
    C._initVars(t, '{b}=-{thickness}/({nt}**2+%.15f)**0.5/%.15f/{CoordinateY}' % (eps, 2*np.pi/Nblades))
    t = P.computeGrad(t, 'b')
    # C._initVars(bodyForceCoarse, '{b}=-{thickness}/({nt}**2+%.15f)**0.5/%.15f/{CoordinateY}' % (eps, 2*np.pi/Nblades))
    # bodyForceCoarse = P.computeGrad(bodyForceCoarse, 'b')
    # C._extractVars(bodyForceCoarse, ['CoordinateX', 'CoordinateY', 'CoordinateZ', 'centers:gradxb', 'centers:gradyb'])
    # P._extractMesh(bodyForceCoarse, t, order=2, extrapOrder=0, constraint=40.)

    C._initVars(t, '{blockage}=1.+{b}')
    I._rmNodesByName(t, 'b')
    I._rmNodesByName(t, 'gradzb')
    I._renameNode(t, 'gradyb', 'gradrb')

    C._initVars(t, '{radius}=sqrt({CoordinateY}**2+{CoordinateZ}**2)')
    C._initVars(t, '{{blade2BladeDistance}}=2*{}*{{radius}}/{}*{{nt}}*{{blockage}}'.format(np.pi, Nblades))

    return t


def filterDataSourceTermsRadial(t):
    ''' NOT USED '''
    val_filtered        = ['nx','nr','nt','xc','chord','chordx','b','gradxb','gradrb','delta0']
    cg = 0.0005  # parametre de lissage [-]
    # Coefficient de lissage des termes source sur les derniers pc radial de veine
    coef_env_c = 15.*cg
    # Coefficient de lissage des termes source sur les premiers pc radial de veine
    coef_env_m = 40.*cg
    for zone in I.getNodesFromType(t, 'Zone_t'):
        zone_dim = I.getZoneDim(zone)
        Ni, Nj = zone_dim[1], zone_dim[2]
        dist_inlet_hub = T.subzone(zone, (1, 1, 1), (1, Nj, 1))
        h_in = D.getLength(dist_inlet_hub)
        jprot_hub, jprot_shroud = 0, Nj-1
        for j in range(Nj):
            dist_inlet_tp = T.subzone(zone, (1, 1, 1), (1, j+1, 1))
            h_tp = D.getLength(dist_inlet_tp)
            if h_tp < h_in*coef_env_m:
                jprot_hub = j
            if h_tp <= h_in*(1.-coef_env_c):
                jprot_shroud = j
        for flowSol in I.getNodesFromType(zone, 'FlowSolution_t'):
            for data in I.getNodesFromType(flowSol, 'DataArray_t'):
                data_name = I.getName(data)
                if data_name in val_filtered:
                    data_t = I.getValue(data)
                    data_2d_t = np.zeros((Ni, Nj))
                    for j in range(0, jprot_hub):
                        data_2d_t[:, j] = data_t[:, jprot_hub+1]
                    for j in range(jprot_hub, jprot_shroud):
                        data_2d_t[:, j] = data_t[:, j]
                    for j in range(jprot_shroud, Nj):
                        data_2d_t[:, j] = data_t[:, jprot_shroud]
                    I.setValue(data, np.asfortranarray(data_2d_t))

def filterDataSourceTermsAxial(t):
    ''' NOT USED '''
    val_filtered = ['nx', 'nr', 'nt', 'xc', 'chord',
                    'chordx', 'b', 'gradxb', 'gradrb', 'delta0']
    cg          = 0.0005 # parametre de lissage [-]
    coef_env_BA = 5.*cg     #Coefficient de lissage des termes source sur les derniers pc axial de veine
    coef_env_BF = 5.*cg     #Coefficient de lissage des termes source sur les premiers pc axial de veine

    for zone in I.getNodesFromType(t, 'Zone_t'):
        zone_dim = I.getZoneDim(zone)
        Ni, Nj             = zone_dim[1],zone_dim[2]
        dist_inlet_hub    = T.subzone(zone, (1,1,1),(Ni,1,1))
        xBABF_hub = D.getLength(dist_inlet_hub)
        iprot_BA, iprot_BF = 0,Ni-1
        for i in range(Ni):
            dist_outlet_tp = T.subzone(zone, (1,1,1),(i+1,1,1))
            xBAtp =  D.getLength(dist_outlet_tp)
            if xBAtp < xBABF_hub * coef_env_BA:
                iprot_BA = i
            if xBAtp<=xBABF_hub*(1.-coef_env_BF): 
                iprot_BF = i
        for flowSol in I.getNodesFromType(zone, 'FlowSolution_t'):
            for data in I.getNodesFromType(flowSol, 'DataArray_t'):
                data_name = I.getName(data)
                if data_name in val_filtered:
                    data_t = I.getValue(data)
                    data_2d_t = np.zeros((Ni, Nj))
                    for i in range(0,iprot_BA):
                        data_2d_t[i,:] = data_t[iprot_BA,:]
                    for i in range(iprot_BA,iprot_BF): 
                        data_2d_t[i,:] = data_t[i,:]
                    for i in range(iprot_BF,Ni):
                        data_2d_t[i,:] = data_t[iprot_BF,:]
                    I.setValue(data, np.asfortranarray(data_2d_t))


##################################################################################
# Functions for coprocessing
# The following functions implement body-force models to update source terms 
# during the simulation.
# They are all called by th main function computeBodyForce. This latter one
# is called by Coprocess.updateBodyForce
##################################################################################


def computeBodyForce(zone, BodyForceParams, FluidProperties, TurboConfiguration):
    '''
    Compute Body force source terms in **zone**.

    Parameters
    ----------
        zone : PyTree
            Zone in which the source terms will be compute

        BodyForceParams : dict
            Body force parameters for the current family. Correspond to a value 
            in **BodyForceInputData**, as read from `setup.py`. 

        FluidProperties : dict
            as read from `setup.py`

        TurboConfiguration : dict
            as read from `setup.py`

    Returns
    -------
        dict
            New source terms to apply. Should be for example : 

            >>> TotalSourceTerms = dict(Density=ndarray, MomentumX=ndarray, ...)

    '''
    # Get the list of source terms to compute
    if isinstance(BodyForceParams['model'], list):
        models = BodyForceParams['model']
    else:
        if BodyForceParams['model'] == 'hall':
            models = ['blockage', 'hall_without_blockage', 'EndWallsProtection']
        else:
            models = [BodyForceParams['model']]

    # Compute and gather all the required source terms
    TotalSourceTerms = dict()
    for model in models:
        # Default tolerance
        tol = BodyForceParams.get('tol', 1e-5)

        # Compute the correct source terms
        if model == 'blockage':
            NewSourceTerms = computeBodyForce_Blockage(zone, tol)
        elif model == 'EndWallsProtection':
            NewSourceTerms = computeBodyForce_EndWallsProtection(
                zone, TurboConfiguration, BodyForceParams.get('ProtectedHeight', 0.05))
        elif model == 'constant':
            NewSourceTerms = computeBodyForce_constant(
                zone, BodyForceParams['SourceTerms'])
        elif model == 'ThrustSpread':
            NewSourceTerms = computeBodyForce_ThrustSpread(
                zone, BodyForceParams['Thrust'], tol)
        elif model == 'hall_without_blockage':
            NewSourceTerms = computeBodyForce_Hall(
                zone, FluidProperties, TurboConfiguration, tol)
        else:
            raise Exception(f"The body-force model {model} is not implemented")
            
        # Add the computed source terms to the total source terms
        for key, value in NewSourceTerms.items():
            if key in TotalSourceTerms:
                TotalSourceTerms[key] += value
            else:
                TotalSourceTerms[key] = value

    return TotalSourceTerms


def computeBodyForce_Blockage(zone, tol=1e-5):
    '''
    Compute actualized source terms corresponding to blockage.

    Parameters
    ----------

        zone : PyTree
            current zone

        tol : float
            minimum value for quantities used as a denominator.

    Returns
    -------

        NewSourceTerms : dict
            Computed source terms. The keys are Density, MomentumX, MomentumY,
            MomentumZ and EnergyStagnation.
    '''

    FlowSolution = J.getVars2Dict(zone, Container='FlowSolution#Init')
    DataSourceTerms=J.getVars2Dict(zone, ['theta', 'blockage', 'gradxb', 'gradrb', 'nt'], Container='FlowSolution#DataSourceTerm')

    Blockage = DataSourceTerms['blockage']
    Density = np.maximum(FlowSolution['Density'], tol)
    EnthalpyStagnation = (FlowSolution['EnergyStagnationDensity'] + FlowSolution['Pressure']) / Density
    Vx, Vy, Vz = FlowSolution['MomentumX']/Density, FlowSolution['MomentumY'] / Density, FlowSolution['MomentumZ']/Density
    Vr = Vy * np.cos(DataSourceTerms['theta']) + Vz * np.sin(DataSourceTerms['theta'])

    Sb = -(Vx * DataSourceTerms['gradxb'] + Vr * DataSourceTerms['gradrb']) / Blockage

    NewSourceTerms = dict(
        Density                 = Sb,
        MomentumX               = Sb * Vx,
        MomentumY               = Sb * Vy,
        MomentumZ               = Sb * Vz,
        EnergyStagnationDensity = Sb * EnthalpyStagnation
    )

    return NewSourceTerms

def computeBodyForce_Hall(zone, FluidProperties, TurboConfiguration, tol=1e-5):
    '''
    Compute actualized source terms corresponding to the Hall model.

    Parameters
    ----------

        zone : PyTree
            current zone
        
        FluidProperties : dict
            as read in `setup.py`

        TurboConfiguration : dict
            as read in `setup.py`

        tol : float
            minimum value for quantities used as a denominator.

    Returns
    -------

        NewSourceTerms : dict
            Computed source terms. The keys are Density, MomentumX, MomentumY, 
            MomentumZ and EnergyStagnation. Blockage effect is included.
    '''

    rowName = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
    NumberOfBlades = TurboConfiguration['Rows'][rowName]['NumberOfBlades']
    RotationSpeed = TurboConfiguration['Rows'][rowName]['RotationSpeed']

    FlowSolution    = J.getVars2Dict(zone, Container='FlowSolution#Init')
    DataSourceTerms = J.getVars2Dict(zone, Container='FlowSolution#DataSourceTerm')
    # Variables needed in DataSourceTerms: 'radius', 'theta', 'blockage', 'nx', 'nr', 'nt', 'AbscissaFromLE', 'blade2BladeDistance'
    # Optional variables in DataSourceTerms: 'delta0'

    # Coordinates
    cosTheta = np.cos(DataSourceTerms['theta'])
    sinTheta = np.sin(DataSourceTerms['theta'])

    # Flow data
    Density = np.maximum(FlowSolution['Density'], tol)
    Vx, Vy, Vz = FlowSolution['MomentumX']/Density, FlowSolution['MomentumY']/Density, FlowSolution['MomentumZ']/Density
    Wx, Wr, Wt = Vx, Vy*cosTheta+Vz*sinTheta, -Vy*sinTheta + Vz*cosTheta - DataSourceTerms['radius'] * RotationSpeed
    Vmag = (Vx**2 + Vy**2 + Vz**2)**0.5
    Wmag = np.maximum(tol, (Wx**2 + Wr**2 + Wt**2)**0.5)
    Temperature = np.maximum(tol, (FlowSolution['EnergyStagnationDensity']/Density-0.5*Vmag**2.)/FluidProperties['cp'])
    Mrel = Wmag/(FluidProperties['Gamma']*FluidProperties['IdealGasConstant']*Temperature)**0.5

    # Velocity normal and parallel to the skeleton
    # See Cyril Dosnes bibliography synthesis for the local frame of reference
    Wn = np.absolute(Wx*DataSourceTerms['nx'] + Wr*DataSourceTerms['nr'] + Wt*DataSourceTerms['nt']) # Velocity component normal to the blade surface
    Wnx = Wn * DataSourceTerms['nx'] * np.sign(Wx*DataSourceTerms['nx'])
    Wnr = Wn * DataSourceTerms['nr'] * np.sign(Wr*DataSourceTerms['nr'])
    Wnt = Wn * DataSourceTerms['nt'] * np.sign(Wt*DataSourceTerms['nt'])
    Wpx, Wpr, Wpt = Wx-Wnx,   Wr-Wnr,   Wt-Wnt # Velocity component in the plane tangent to the blade surface
    Wp = np.maximum(tol, (Wpx**2+Wpr**2+Wpt**2)**0.5)

    # Deviation of the flow with respect to the blade surface
    # Careful to the sign
    incidence = np.arcsin(Wn/Wmag) 
    incidence*= np.sign(Wx*DataSourceTerms['nx'] + Wr*DataSourceTerms['nr'] + Wt*DataSourceTerms['nt'])
    
    # Unit vector normal the velocity. Direction of application of the normal force
    unitVectorN_x = np.cos(incidence) * DataSourceTerms['nx'] - np.sin(incidence)*Wpx/Wp
    unitVectorN_r = np.cos(incidence) * DataSourceTerms['nr'] - np.sin(incidence)*Wpr/Wp
    unitVectorN_t = np.cos(incidence) * DataSourceTerms['nt'] - np.sin(incidence)*Wpt/Wp

    # Unit vector parallel to the velocity. Direction of application of the parallel force
    unitVectorP_x = - Wx / Wmag
    unitVectorP_r = - Wr / Wmag
    unitVectorP_t = - Wt / Wmag

    # Compressibility correction 
    CompressibilityCorrection = 3. * np.ones(Density.shape)
    subsonic_bf, supersonic_bf = np.less_equal(Mrel,0.99), np.greater_equal(Mrel,1.01)
    CompressibilityCorrection[subsonic_bf]  = np.clip(1.0/(1-Mrel[subsonic_bf]**2)**0.5, 0.0, 3.0)
    CompressibilityCorrection[supersonic_bf]= np.clip(4.0/(2*np.pi)/(Mrel[supersonic_bf]**2-1)**0.5, 0.0, 3.0)

    # Friction on blade
    Viscosity = FluidProperties['SutherlandViscosity']*np.sqrt(Temperature/FluidProperties['SutherlandTemperature'])*(1+FluidProperties['SutherlandConstant'])/(1+FluidProperties['SutherlandConstant']*FluidProperties['SutherlandTemperature']/Temperature)
    Re_x = Density*DataSourceTerms['AbscissaFromLE'] * Wmag / Viscosity
    cf = 0.0592*Re_x**(-0.2)

    # Force normal to the chord
    fn = -0.5*Wmag**2. * CompressibilityCorrection * 2*np.pi*incidence / DataSourceTerms['blade2BladeDistance']

    # Force parallel to the chord
    delta0 = DataSourceTerms.get('delta0', 0.)
    fp = 0.5*Wmag**2. * (2*cf + 2*np.pi*(incidence - delta0)**2.) / DataSourceTerms['blade2BladeDistance']

    # Force in the cylindrical frame of reference
    fx = fn * unitVectorN_x + fp * unitVectorP_x
    fr = fn * unitVectorN_r + fp * unitVectorP_r
    ft = fn * unitVectorN_t + fp * unitVectorP_t

    # Force in the cartesian frame of reference
    fy = -sinTheta * ft + cosTheta * fr
    fz =  cosTheta * ft + sinTheta * fr

    NewSourceTerms = dict(
        Density                 = np.zeros(Density.shape),
        MomentumX               = Density * fx,
        MomentumY               = Density * fy,
        MomentumZ               = Density * fz,
        EnergyStagnationDensity = Density * DataSourceTerms['radius'] * RotationSpeed * ft
    )

    return NewSourceTerms


def computeBodyForce_EndWallsProtection(zone, TurboConfiguration, ProtectedHeight=0.05):
    ''' 
    Protection of the boudary layer ofr body-force modelling, as explain in the appendix D of 
    W. Thollet PdD manuscrit.

    .. danger:: Available only for structured mesh. Furthermore, there must be a unique BF zone.

    Parameters
    ----------

        zone : PyTree
            current zone

        TurboConfiguration : dict
            as read in `setup.py`

        ProtectedHeight : float
            Height of the channel flow corresponding to the boundary layer. 
            By default, 0.05.

    Returns
    -------

        BLProtectionSourceTerms : dict
            Source terms to protect the boundary layer. The keys are Density (=0), MomentumX, 
            MomentumY, MomentumZ and EnergyStagnation (=0).
    
    '''

    rowName = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
    RotationSpeed = TurboConfiguration['Rows'][rowName]['RotationSpeed']

    FlowSolution = J.getVars2Dict(zone, Container='FlowSolution#Init')
    DataSourceTerms = J.getVars2Dict(zone, Container='FlowSolution#DataSourceTerm')

    # Extract Boundary layers edges, based on ProtectedHeightPercentage
    zone = I.renameNode(zone, 'FlowSolution#Init', 'FlowSolution#Centers')
    I._renameNode(zone, 'FlowSolution#Height', 'FlowSolution')
    BoundarayLayerEdgeAtHub    = P.isoSurfMC(zone, 'ChannelHeight', ProtectedHeight)
    BoundarayLayerEdgeAtShroud = P.isoSurfMC(zone, 'ChannelHeight', 1-ProtectedHeight)
    zone = C.node2Center(zone, 'ChannelHeight')
    h, = J.getVars(zone, VariablesName=['ChannelHeight'], Container='FlowSolution#Centers')

    # Coordinates
    radius, theta = DataSourceTerms['radius'], DataSourceTerms['theta']
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)
    # Flow data
    Vx = FlowSolution['MomentumX'] / FlowSolution['Density']
    Vy = FlowSolution['MomentumY'] / FlowSolution['Density']
    Vz = FlowSolution['MomentumZ'] / FlowSolution['Density']
    Wr =  Vy*cosTheta + Vz*sinTheta 
    Wt = -Vy*sinTheta + Vz*cosTheta - radius*RotationSpeed
    Wmag = (Vx**2+Wr**2+Wt**2)**0.5

    def get_mean_W_and_gradP(z):
        if not z: 
            return 1, 0, 0

        C._initVars(z, 'radius=({CoordinateY}**2+{CoordinateZ}**2)**0.5')
        C._initVars(z, 'theta=arctan2({CoordinateZ}, {CoordinateY})')
        C._initVars(z, 'Wx={MomentumX}/{Density}')
        C._initVars(z, 'Vy={MomentumY}/{Density}')
        C._initVars(z, 'Vz={MomentumZ}/{Density}')
        C._initVars(z, 'Wr={Vy}*cos({theta})+{Vz}*sin({theta})')
        C._initVars(z, 'Wt=-{{Vy}}*sin({{theta}})+{{Vz}}*cos({{theta}})-{{radius}}*{}'.format(RotationSpeed))
        C._initVars(z, 'Wmag=({Wx}**2+{Wr}**2+{Wt}**2)**0.5')
        C._initVars(z, 'DpDr=cos({theta})*{DpDy}+sin({theta})*{DpDz}')

        meanWmag = C.getMeanValue(z, 'Wmag')
        meanDpDx = C.getMeanValue(z, 'DpDx')
        meanDpDr = C.getMeanValue(z, 'DpDr')
        return meanWmag, meanDpDx, meanDpDr

    W_HubEdge, DpDx_HubEdge, DpDr_HubEdge = get_mean_W_and_gradP(BoundarayLayerEdgeAtHub)
    W_ShroudEdge, DpDx_ShroudEdge, DpDr_ShroudEdge = get_mean_W_and_gradP(BoundarayLayerEdgeAtShroud)

    # Source terms
    S_BL_Hub_x    = (1. - (Wmag /    W_HubEdge)**2) * DpDx_HubEdge
    S_BL_Hub_r    = (1. - (Wmag /    W_HubEdge)**2) * DpDr_HubEdge
    S_BL_Shroud_x = (1. - (Wmag / W_ShroudEdge)**2) * DpDx_ShroudEdge
    S_BL_Shroud_r = (1. - (Wmag / W_ShroudEdge)**2) * DpDr_ShroudEdge

    S_BL_x = S_BL_Hub_x * (h<ProtectedHeight) + S_BL_Shroud_x * (h>1-ProtectedHeight)
    S_BL_r = S_BL_Hub_r * (h<ProtectedHeight) + S_BL_Shroud_r * (h>1-ProtectedHeight)

    BLProtectionSourceTerms = dict(
        Density                 = np.zeros(Wmag.shape),
        MomentumX               = S_BL_x,
        MomentumY               = cosTheta * S_BL_r,
        MomentumZ               = sinTheta * S_BL_r,
        EnergyStagnationDensity = np.zeros(Wmag.shape)
    )
    return BLProtectionSourceTerms

def computeProtectionSourceTerms_OLD(zone, TurboConfiguration, ProtectedHeightPercentage=5., tol=1e-5):
    ''' 
    Protection of the boudary layer ofr body-force modelling, as explain in the appendix D of 
    W. Thollet PdD manuscrit.

    .. danger:: Available only for structured mesh. Furthermore, there must be a unique BF zone.

    Parameters
    ----------

        zone : PyTree
            current zone

        TurboConfiguration : dict
            as read in `setup.py`

        ProtectedHeightPercentage : float
            Percentage of the total height of the channel flow corresponding to the boundary layer. 
            By default, 5.

        tol : float
            minimum value for quantities used as a denominator.

    Returns
    -------

        BLProtectionSourceTerms : dict
            Source terms to protect the boundary layer. The keys are Density (=0), MomentumX, 
            MomentumY, MomentumZ and EnergyStagnation (=0).
    
    '''

    rowName = I.getValue(I.getNodeFromType1(zone, 'FamilyName_t'))
    RotationSpeed = TurboConfiguration['Rows'][rowName]['RotationSpeed']

    FlowSolution = J.getVars2Dict(zone, Container='FlowSolution#Init')
    DataSourceTerms = J.getVars2Dict(zone, Container='FlowSolution#DataSourceTerm')

    # Coordinates
    x, radius, theta = DataSourceTerms['xCell'], DataSourceTerms['radius'], DataSourceTerms['theta']
    cosTheta = np.cos(theta)
    sinTheta = np.sin(theta)

    # Flow data
    Density = np.maximum(FlowSolution['Density'], tol)
    Vx, Vy, Vz = FlowSolution['MomentumX']/Density, FlowSolution['MomentumY'] / Density, FlowSolution['MomentumZ']/Density
    Wx, Wr, Wt = Vx, Vy*cosTheta+Vz*sinTheta, -Vy*sinTheta+Vz*cosTheta-radius*RotationSpeed
    Wmag = np.maximum(tol, (Wx**2+Wr**2+Wt**2)**0.5)

    # Find radial direction
    dr_i = np.absolute(radius[-1, 0, 0] - radius[0, 0, 0])
    dr_j = np.absolute(radius[ 0,-1, 0] - radius[0, 0, 0])
    dr_k = np.absolute(radius[ 0, 0,-1] - radius[0, 0, 0])
    dr_array = np.array([dr_i, dr_j, dr_k])
    radialIndex = np.argmin(dr_array) # 0, 1 or 2, for i, j or k

    Npoints_bottomHalf = int(np.floor(radius.shape[radialIndex]/2.))
    Npoints_topHalf = radius.shape[radialIndex] - Npoints_bottomHalf

    if radialIndex == 0:
        radialMeshLine = ((x[0, 0, 0]-x[:, 0, 0])**2 + (radius[0, 0, 0]-radius[:, 0, 0])**2)**0.5
    elif radialIndex == 1:
        radialMeshLine = ((x[0, 0, 0]-x[0, :, 0])**2 + (radius[0, 0, 0]-radius[0, :, 0])**2)**0.5
    else:
        radialMeshLine = ((x[0, 0, 0]-x[0, 0, :])**2 + (radius[0, 0, 0]-radius[0, 0, :])**2)**0.5

    # These three lines do not depend on the radial direction index
    radialMeshLineLength = np.absolute(radialMeshLine[-1] - radialMeshLine[0])
    hubEdgeIndex = np.argmin(np.absolute(radialMeshLine - radialMeshLineLength * ProtectedHeightPercentage/100.))
    shroudEdgeIndex = np.argmin(np.absolute(radialMeshLine - radialMeshLineLength * ProtectedHeightPercentage/100.))

    def broadcastBLEdgeValues(vector):
        '''
        From the radial distribution **vector**, locate values at the boudary layer edges near the hub and the 
        shroud and broadcast them on the whole span of the flow channel.

        Parameters
        ----------
            vector : numpy.array
                3D-array corresponding of a field of interest 

        Returns
        -------
            vectorEdges : numpy.array
                3D-array uniform in the radial direction on each half of the flow channel, 
                built by repeatition of the surfaces at the edge of the boudary layers at hub and shroud.
        '''
        if radialIndex == 0:
            vectorAtHubBL            = vector[hubEdgeIndex, :, :]
            vectorAtShroudBL         = vector[shroudEdgeIndex, :, :]
            vectorAtHubBLRepeated    = np.repeat(vectorAtHubBL[np.newaxis, :, :], Npoints_bottomHalf, axis=2)
            vectorAtShroudBLRepeated = np.repeat(vectorAtShroudBL[np.newaxis, :, :], Npoints_topHalf, axis=2)
        elif radialIndex == 1:
            vectorAtHubBL            = vector[:, hubEdgeIndex, :]
            vectorAtShroudBL         = vector[:, shroudEdgeIndex, :]
            vectorAtHubBLRepeated    = np.repeat(vectorAtHubBL[:, np.newaxis, :], Npoints_bottomHalf, axis=2)
            vectorAtShroudBLRepeated = np.repeat(vectorAtShroudBL[:, np.newaxis, :], Npoints_topHalf, axis=2)
        else:
            vectorAtHubBL            = vector[:, :, hubEdgeIndex]
            vectorAtShroudBL         = vector[:, :, shroudEdgeIndex]
            vectorAtHubBLRepeated    = np.repeat(vectorAtHubBL[:, :, np.newaxis], Npoints_bottomHalf, axis=2)
            vectorAtShroudBLRepeated = np.repeat(vectorAtShroudBL[:, :, np.newaxis], Npoints_topHalf, axis=2)

        vectorEdges = np.concatenate((vectorAtHubBLRepeated, vectorAtShroudBLRepeated), axis=2)
        return vectorEdges

    Wedges = broadcastBLEdgeValues(Wmag)
    DpDxEdges = broadcastBLEdgeValues(FlowSolution['DpDx'])
    DpDyEdges = broadcastBLEdgeValues(FlowSolution['DpDy'])
    DpDzEdges = broadcastBLEdgeValues(FlowSolution['DpDz'])
    DpDrEdges = cosTheta*DpDyEdges + sinTheta*DpDzEdges

    # Source terms
    S_corr_BL_x = (1. - np.minimum((Wmag/Wedges)**2.,1.) ) * DpDxEdges * 10.
    S_corr_BL_r = (1. - np.minimum((Wmag/Wedges)**2.,1.) ) * DpDrEdges * 10.
    S_corr_BL_y = cosTheta*S_corr_BL_r
    S_corr_BL_z = sinTheta*S_corr_BL_r

    BLProtectionSourceTerms = dict(
        Density                 = np.zeros(Wmag.shape),
        MomentumX               = S_corr_BL_x,
        MomentumY               = S_corr_BL_y,
        MomentumZ               = S_corr_BL_z,
        EnergyStagnationDensity = np.zeros(Wmag.shape)
    )
    return BLProtectionSourceTerms

def computeBodyForce_ThrustSpread(zone, Thrust, tol=1e-5):
    '''
    Compute actualized source terms corresponding to the Tspread model.

    Parameters
    ----------

        zone : PyTree
            current zone
        
        Thrust : float
            Value of thrust (in Newtons) to apply on the whole volume of body-force zones

        tol : float
            minimum value for quantities used as a denominator.

    Returns
    -------

        NewSourceTerms : dict
            Computed source terms. The keys are Density, MomentumX, MomentumY, 
            MomentumZ and EnergyStagnation. 
    '''

    FlowSolution    = J.getVars2Dict(zone, Container='FlowSolution#Init')
    DataSourceTerms = J.getVars2Dict(zone, Container='FlowSolution#DataSourceTerm')

    # Flow data
    Density = np.maximum(FlowSolution['Density'], tol)
    Vx, Vy, Vz = FlowSolution['MomentumX']/Density, FlowSolution['MomentumY']/Density, FlowSolution['MomentumZ']/Density
    Vmag = np.maximum(tol, (Vx**2+Vy**2+Vz**2)**0.5)

    f = Thrust / DataSourceTerms['totalVolume']

    NewSourceTerms = dict(
        Density                 = np.zeros(Vmag.shape),
        MomentumX               = Density * f * Vx/Vmag,
        MomentumY               = Density * f * Vy/Vmag,
        MomentumZ               = Density * f * Vz/Vmag,
        EnergyStagnationDensity = Density * f * Vmag,
    )

    return NewSourceTerms

def computeBodyForce_constant(zone, SourceTerms):
    '''
    Compute constant source terms (reshape given source terms if they are uniform).

    Parameters
    ----------

        zone : PyTree
            current zone
        
        SourceTerms : dict
            Source terms to apply. For each value, the given value may be a 
            float or a numpy array (the shape must correspond to the zone shape).

    Returns
    -------

        NewSourceTerms : dict
            Computed source terms. 
    '''

    FlowSolution    = J.getVars2Dict(zone, Container='FlowSolution#Init')
    ones = np.ones(FlowSolution['Density'].shape)

    NewSourceTerms = dict()
    for key, value in SourceTerms.items():
        NewSourceTerms[key] = value * ones

    return NewSourceTerms

def computeBodyForce_ShockWaveLoss(zone, FluidProperties):
    '''
    Compute the volumic force parallel to the flow (and in the opposite direction) corresponding 
    to shock wave loss. 

    .. note::
        
        See the following reference for details on equations:

        Pazireh and Defoe, 
        A New Loss Generation Body Force Model for Fan/Compressor Blade Rows: Application to Uniform 
        and Non-Uniform Inflow in Rotor 67,
        Journal of Turbomachinery (2022)

    Parameters
    ----------

        zone : PyTree
            current zone
        
        FluidProperties : dict
            as read in `setup.py`

        TurboConfiguration : dict
            as read in `setup.py`

        tol : float
            minimum value for quantities used as a denominator.

    Returns
    -------

        NewSourceTerms : dict
            Computed source terms. The keys are Density, MomentumX, MomentumY, 
            MomentumZ and EnergyStagnation. Blockage effect is included.
    '''
    cv = FluidProperties['cv']
    R = FluidProperties['IdealGasConstant']
    gamma = FluidProperties['Gamma']
    PressureStagnationLossRatio = cv/R * 2*gamma*(gamma-1) / 3. / (gamma+1)**2 * (Mrel**2-1)**3
    fp = Pt_rel * PressureStagnationLossRatio / DataSourceTerm['blade2BladeDistance']
    
    return fp