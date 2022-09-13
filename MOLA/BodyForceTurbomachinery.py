'''
MOLA - BodyForceTurbomachinery.py

File history:
8/09/2022 - T. Bontemps - Creation
'''

import os
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


def replaceRowWithBodyForceMesh(t, rows):
    '''
    In the mesh **t**, replace the row domain corresponding to the family 
    **row** by a mesh adapted to body-force.

    Parameters
    ----------
        t : PyTree
            input mesh, from Autogrid for instance.

        rows : :py:class:`str` or :py:class:`list` of :py:class:`str`
            Name of the row families that will be replaced by a mesh adapted
            to body-force.
    
    Returns
    -------

        newRowMeshes : :py:class:`list` of PyTree

            New mesh, identical to the input mesh **t** except that the domain 
            corresponding to the family **row** has been replaced by a mesh adapted
            to body-force.

    '''
    if isinstance(rows, str):
        rows = [rows]

    newRowMeshes = []

    for row in rows:

        RowFamilyNode = I.getNodeFromNameAndType(t, row, 'Family_t')
        BladeNumber = I.getValue(I.getNodeFromName(RowFamilyNode, 'BladeNumber'))

        zones = C.getFamilyZones(t, row)
        meshType = 'unstructured' if PRE.hasAnyUnstructuredZones(zones) else 'structured'

        # Get the meridional info from rowTree
        meridionalMesh = extractRowGeometricalData(t, row)

        CellWidthAtWall = 0.005 
        CellWidthAtLE = 0.01
        CellWidthAtInlet = 0.05
        RadialDistribution = dict( kind='tanhTwoSides', FirstCellHeight=CellWidthAtWall, LastCellHeight=CellWidthAtWall)
        AxialDistributionBeforeLE = dict(kind='tanhTwoSides', FirstCellHeight=CellWidthAtInlet, LastCellHeight=CellWidthAtLE)
        AxialDistributionBetweenLEAndTE = dict(kind='tanhTwoSides', FirstCellHeight=CellWidthAtLE, LastCellHeight=CellWidthAtLE)
        AxialDistributionAfterTE = dict(kind='tanhTwoSides', FirstCellHeight=CellWidthAtLE, LastCellHeight=CellWidthAtInlet)

        newRowMesh = buildBodyForceMeshForOneRow(meridionalMesh,
                            NumberOfBlades=BladeNumber,
                            NumberOfRadialPoints=51,
                            NumberOfAxialPointsBeforeLE=21,
                            NumberOfAxialPointsBetweenLEAndTE=21,
                            NumberOfAxialPointsAfterTE=21,
                            RadialDistribution=RadialDistribution,
                            AxialDistributionBeforeLE=AxialDistributionBeforeLE,
                            AxialDistributionBetweenLEAndTE=AxialDistributionBetweenLEAndTE,
                            AxialDistributionAfterTE=AxialDistributionAfterTE,
                            meshType=meshType
                            )

        I._renameNode(newRowMesh, 'upstream', f'{row}_upstream')
        I._renameNode(newRowMesh, 'downstream', f'{row}_downstream')
        I._renameNode(newRowMesh, 'bodyForce', f'{row}_bodyForce')
        I._renameNode(newRowMesh, 'Inflow', f'{row}_Inflow')
        I._renameNode(newRowMesh, 'Outflow', f'{row}_Outflow')
        I._renameNode(newRowMesh, 'Hub', f'{row}_Hub')
        I._renameNode(newRowMesh, 'Shroud', f'{row}_Shroud')

        I._renameNode(newRowMesh, 'Fluid', row)

        newRowMeshes.append(newRowMesh)

        for zone in zones:
            I.rmNode(t, zone)

    return t, newRowMeshes


def buildBodyForceMeshForOneRow(t, NumberOfBlades, NumberOfRadialPoints, NumberOfAxialPointsBeforeLE,
                    NumberOfAxialPointsBetweenLEAndTE, NumberOfAxialPointsAfterTE,
                    NumberOfAzimutalPoints=3, AzimutalAngleDeg=10.,
                    RadialDistribution=None, AxialDistributionBeforeLE=None,
                    AxialDistributionBetweenLEAndTE=None, AxialDistributionAfterTE=None,
                    meshType='structured'
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

        NumberOfRadialPoints : int
           Number of points in the radial direction.

        NumberOfAxialPointsBeforeLE : int
            Number of points in the axial direction between the inlet and the blade leading edge.

        NumberOfAxialPointsBetweenLEAndTE : int
            Number of points in the axial direction between the blade leading edge and trailing edge.

        NumberOfAxialPointsAfterTE : int
            Number of points in the axial direction between the blade trailing edge and the outlet.

        NumberOfAzimutalPoints : int, optional
            Number of points in the azimutal direction, by default 3

        AzimutalAngleDeg : _type_, optional
            Azimutal extension in degrees of the output computational domain, by default 10.

        RadialDistribution : dict, optional
            Points distribution in the radial direction. For details, see the documentation of
            :func:`MOLA.Wireframe.discretize`. Could be for instance: 
            
            >>> dict(kind='tanhTwoSides',FirstCellHeight=CellWidthAtWall,LastCellHeight=CellWidthAtWall)

        AxialDistributionBeforeLE : dict, optional
            Points distribution in the axial direction between the inlet and the blade leading edge.

        AxialDistributionBetweenLEAndTE : dict, optional
            Points distribution in the axial direction between the blade leading edge and trailing edge.

        AxialDistributionAfterTE : dict, optional
            Points distribution in the axial direction between the blade trailing edge and the outlet.

        meshType : str, optional
            Type of the mesh. Should be 'structured' or 'unstructured'. By default 'structured'

    Returns
    -------
        PyTree
            3D mesh with connectivities and periodic connectivities, suitable for a body-force simulation,
            ready to be used in :func:`MOLA.WorkflowCompressor.prepareMainCGNS4ElsA`.

    '''
    # Get the lines defining the geometry 
    Hub = I.getNodeFromName2(t, 'Hub')
    Shroud = I.getNodeFromName2(t, 'Shroud')
    LeadingEdge = I.getNodeFromName2(t, 'LeadingEdge')
    TrailingEdge = I.getNodeFromName2(t, 'TrailingEdge')
    Inlet = I.getNodeFromName2(t, 'Inlet')
    Outlet = I.getNodeFromName2(t, 'Outlet')

    # Extrapolate LE and TE lines to be sure that they intersect hub and shroud lines
    ExtrapDistance = 0.05 * W.getLength(LeadingEdge)
    LeadingEdge = W.extrapolate(LeadingEdge, ExtrapDistance, opposedExtremum=False)
    LeadingEdge = W.extrapolate(LeadingEdge, ExtrapDistance, opposedExtremum=True)
    ExtrapDistance = 0.05 * W.getLength(TrailingEdge)
    TrailingEdge = W.extrapolate(TrailingEdge, ExtrapDistance, opposedExtremum=False)
    TrailingEdge = W.extrapolate(TrailingEdge, ExtrapDistance, opposedExtremum=True)

    # Compute curves intersections and split curves
    # Split Hub
    cut_point_1 = W.getNearestIntersectingPoint(Hub, LeadingEdge)
    cut_point_2 = W.getNearestIntersectingPoint(Hub, TrailingEdge)
    HubBetweenLEAndTE = W.trimCurveAlongDirection(Hub, np.array([1., 0, 0]), cut_point_1, cut_point_2)
    HubBetweenInletAndLE = W.trimCurveAlongDirection(Hub, np.array([1., 0, 0]), np.array([-999, 0, 0]), cut_point_1)
    HubBetweenTEAndOutlet = W.trimCurveAlongDirection(Hub, np.array([1., 0, 0]), cut_point_2, np.array([999, 0, 0]))
    # Split Shroud
    cut_point_1 = W.getNearestIntersectingPoint(Shroud, LeadingEdge)
    cut_point_2 = W.getNearestIntersectingPoint(Shroud, TrailingEdge)
    ShroudBetweenLEAndTE = W.trimCurveAlongDirection(Shroud, np.array([1., 0, 0]), cut_point_1, cut_point_2)
    ShroudBetweenInletAndLE = W.trimCurveAlongDirection(Shroud, np.array([1., 0, 0]), np.array([-999, 0, 0]), cut_point_1)
    ShroudBetweenTEAndOutlet = W.trimCurveAlongDirection(Shroud, np.array([1., 0, 0]), cut_point_2, np.array([999, 0, 0]))
    # Split LE
    cut_point_1 = W.getNearestIntersectingPoint(LeadingEdge, Hub)
    cut_point_2 = W.getNearestIntersectingPoint(LeadingEdge, Shroud)
    LeadingEdge = W.trimCurveAlongDirection(LeadingEdge, np.array([0, 1, 0]), cut_point_1, cut_point_2)
    # Split TE
    cut_point_1 = W.getNearestIntersectingPoint(TrailingEdge, Hub)
    cut_point_2 = W.getNearestIntersectingPoint(TrailingEdge, Shroud)
    TrailingEdge = W.trimCurveAlongDirection(TrailingEdge, np.array([0, 1, 0]), cut_point_1, cut_point_2)

    # Modify ends of LE line to collapse them on end points hub and shroud lines.
    # Else there will be an error during TFI (TypeError: TFI: input arrays must be C0)
    # For the LE
    inter1 = W.point(HubBetweenLEAndTE, index=0)
    inter2 = W.point(ShroudBetweenLEAndTE, index=0)
    x, y, z = J.getxyz(LeadingEdge)
    x[0], y[0] = inter1[0], inter1[1]  # Modify the first point of the line
    x[-1], y[-1] = inter2[0], inter2[1]  # Modify the last point of the line
    I.setValue(I.getNodeFromName(LeadingEdge, 'CoordinateX'), x)
    I.setValue(I.getNodeFromName(LeadingEdge, 'CoordinateY'), y)
    I.setValue(I.getNodeFromName(LeadingEdge, 'CoordinateZ'), z)
    # For the TE
    inter1 = W.point(HubBetweenLEAndTE, index=-1)
    inter2 = W.point(ShroudBetweenLEAndTE, index=-1)
    x, y, z = J.getxyz(TrailingEdge)
    x[0], y[0] = inter1[0], inter1[1]  # Modify the first point of the line
    x[-1], y[-1] = inter2[0], inter2[1]  # Modify the last point of the line
    I.setValue(I.getNodeFromName(TrailingEdge, 'CoordinateX'), x)
    I.setValue(I.getNodeFromName(TrailingEdge, 'CoordinateY'), y)
    I.setValue(I.getNodeFromName(TrailingEdge, 'CoordinateZ'), z)

    # Rename lines
    I.setName(HubBetweenLEAndTE, 'HubBetweenLEAndTE')
    I.setName(HubBetweenInletAndLE, 'HubBetweenInletAndLE')
    I.setName(HubBetweenTEAndOutlet, 'HubBetweenTEAndOutlet')
    I.setName(ShroudBetweenLEAndTE, 'ShroudBetweenLEAndTE')
    I.setName(ShroudBetweenInletAndLE, 'ShroudBetweenInletAndLE')
    I.setName(ShroudBetweenTEAndOutlet, 'ShroudBetweenTEAndOutlet')
    I.setName(LeadingEdge, 'LeadingEdge')
    I.setName(TrailingEdge, 'TrailingEdge')

    # Apply the points distribution for each line
    Inlet = W.discretize(Inlet, N=NumberOfRadialPoints, Distribution=RadialDistribution)
    Outlet = W.discretize(Outlet, N=NumberOfRadialPoints, Distribution=RadialDistribution)
    LeadingEdge = W.discretize(LeadingEdge, N=NumberOfRadialPoints, Distribution=RadialDistribution)
    TrailingEdge = W.discretize(TrailingEdge, N=NumberOfRadialPoints, Distribution=RadialDistribution)

    HubBetweenInletAndLE = W.discretize(HubBetweenInletAndLE, N=NumberOfAxialPointsBeforeLE, Distribution=AxialDistributionBeforeLE)
    ShroudBetweenInletAndLE = W.discretize(ShroudBetweenInletAndLE, N=NumberOfAxialPointsBeforeLE, Distribution=AxialDistributionBeforeLE)

    HubBetweenLEAndTE = W.discretize(HubBetweenLEAndTE, N=NumberOfAxialPointsBetweenLEAndTE, Distribution=AxialDistributionBetweenLEAndTE)
    ShroudBetweenLEAndTE = W.discretize(ShroudBetweenLEAndTE, N=NumberOfAxialPointsBetweenLEAndTE, Distribution=AxialDistributionBetweenLEAndTE)

    HubBetweenTEAndOutlet = W.discretize(HubBetweenTEAndOutlet, N=NumberOfAxialPointsAfterTE, Distribution=AxialDistributionAfterTE)
    ShroudBetweenTEAndOutlet = W.discretize(ShroudBetweenTEAndOutlet, N=NumberOfAxialPointsAfterTE, Distribution=AxialDistributionAfterTE)


    # Use TFI to create the 2D mesh
    upstream = G.TFI([Inlet, LeadingEdge, HubBetweenInletAndLE, ShroudBetweenInletAndLE])
    bodyForce = G.TFI([LeadingEdge, TrailingEdge,HubBetweenLEAndTE, ShroudBetweenLEAndTE])
    downstream = G.TFI([TrailingEdge, Outlet, HubBetweenTEAndOutlet, ShroudBetweenTEAndOutlet])
    I.setName(upstream, 'upstream')
    I.setName(bodyForce, 'bodyForce')
    I.setName(downstream, 'downstream')

    # Interpolate skeleton data
    Skeleton = I.getNodeFromName2(t, 'Skeleton')
    if Skeleton:
        P._extractMesh(Skeleton, bodyForce, order=2, extrapOrder=0, constraint=0.)
        bodyForce = computeBlockage(bodyForce, NumberOfBlades)
        bodyForce = C.node2Center(bodyForce, I.__FlowSolutionNodes__)
        C._initVars(bodyForce, '{centers:isf}=1')
        I._rmNodesByName1(bodyForce, I.__FlowSolutionNodes__)

    mesh2d = C.newPyTree(['Base', [upstream, bodyForce, downstream]])

    # Make a partial revolution to build the 3D mesh
    mesh3d = D.axisym(mesh2d, (0, 0, 0), (1, 0, 0), angle=AzimutalAngleDeg, Ntheta=NumberOfAzimutalPoints)

    # Move coordinates to cell center and compute radius and azimuthal angle
    bodyForce = I.getNodeFromName2(mesh3d, 'bodyForce')
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
        X.connectMatch(mesh3d)
        mesh3d = X.connectMatchPeriodic(
            mesh3d, rotationAngle=[AzimutalAngleDeg, 0., 0.])
    elif meshType == 'unstructured':
        # Periodic conditions
        mesh3d = C.convertArray2NGon(mesh3d, recoverBC=1)
        X.connectMatch(mesh3d)
        mesh3d = X.connectMatchPeriodic(
            mesh3d, rotationAngle=[AzimutalAngleDeg, 0., 0.])
        I._createElsaHybrid(mesh3d, method=1)
    else:
        raise ValueError(
            "The argument meshType must be 'structured' or 'unstructured'")

    I.checkPyTree(mesh3d)
    I._correctPyTree(mesh3d)
    I._renameNode(mesh3d, I.__FlowSolutionCenters__, 'FlowSolution#DataSourceTerm')

    return mesh3d


def extractRowGeometricalData(mesh, row, save=False):

    def profilesExtractionAndAnalysis(tree, row, blade='Main_Blade',
                                        zones_for_meridional_lines_extraction=None,
                                        zones_for_blade_profiles_extraction=None,
                                        directory_meridional_lines='meridional_lines',
                                        directory_profiles='profiles'):

        import Ersatz as EZ

        if zones_for_meridional_lines_extraction is None:
            # Attention! Ordre de l'amont a l'aval
            zones_for_meridional_lines_extraction = [row+'_flux_1_'+blade+'_upS',
                                                    row+'_flux_1_'+blade+'_up',
                                                    row+'_flux_1_'+blade+'_downS']
        if zones_for_blade_profiles_extraction is None:
            zones_for_blade_profiles_extraction = [row+'_flux_1_'+blade+'_skin']

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

                        zone_t = T.subzone(
                            zone, (zone_i_dim, mi+1, 1), (zone_i_dim, mi+1, zone_k_dim))
                        zone_t = T.reorder(zone_t, (-1, 2, 3))
                        zone_t = C.initVars(zone_t, '{myX}={CoordinateZ}')
                        # zone_t = P.renameVars(zone_t,['myX'],['x'])
                        zone_t = C.initVars(zone_t, '{r}=({CoordinateX}**2+{CoordinateY}**2)**0.5')
                        # zone_t = C.extractVars(zone_t, ['x','r'])
                        tree_t[2][1][2].append(zone_t)
            tree_t = C.extractVars(tree_t, ['myX', 'r'])
            C.convertPyTree2File(tree_t, directory_meridional_lines+'/merid%03d.dat' % (mi+1))
            os.system('sed -i {} -e s/"myX"/"x"/'.format(directory_meridional_lines+'/merid%03d.dat' % (mi+1)))

        # extract blade profiles
        for zone in I.getNodesFromType(tree, 'Zone_t'):
            if I.getName(zone) in zones_for_blade_profiles_extraction:
                zone_i_dim = (I.getZoneDim(zone))[1]
                zone_j_dim = (I.getZoneDim(zone))[2]
                zone_k_dim = (I.getZoneDim(zone))[3]
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
    C._extractVars(thicknessTree, ['xsc', 'thickness'])

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

    xhub, rhub = readEndWallLine(f'{directory_meridional_lines}/merid001.dat')
    xshroud, rshroud = readEndWallLine(f'{directory_meridional_lines}/merid141.dat')
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


def computeBlockage(t, Nblades, eps=1e-6):
    C._initVars(t, '{b}=-{thickness}/({nt}**2+%.15f)**0.5/%.15f/{CoordinateY}' % (eps, 2*np.pi/Nblades))
    t = P.computeGrad(t, 'b')
    C._initVars(t, '{blockage}=1.+{b}')
    I._rmNodesByName(t, 'b')
    I._rmNodesByName(t, 'gradzb')
    I._renameNode(t, 'gradyb', 'gradrb')
    return t


def filterDataSourceTermsRadial(t):
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
                    for j in range(
                            jprot_hub, jprot_shroud):
                        data_2d_t[:, j] = data_t[:, j]
                    for j in range(jprot_shroud, Nj):
                        data_2d_t[:, j] = data_t[:, jprot_shroud]
                    I.setValue(data, np.asfortranarray(data_2d_t))


def filterDataSourceTermsAxial(t):
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
                    data_2d_t     = np.zeros((Ni, Nj))
                    for i in range(0,iprot_BA):
                        data_2d_t[i,:] = data_t[iprot_BA,:]
                    for i in range(iprot_BA,iprot_BF): 
                        data_2d_t[i,:] = data_t[i,:]
                    for i in range(iprot_BF,Ni):
                        data_2d_t[i,:] = data_t[iprot_BF,:]
                    I.setValue(data, np.asfortranarray(data_2d_t))

def computeBlockageSourceTerms(zone, tol=1e-5):
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

    Sb = -(Vx * DataSourceTerms['gradxb'] + Vr *DataSourceTerms['gradrb']) / Blockage

    NewSourceTerms = dict(
        Density          = Sb,
        MomentumX        = Sb * Vx,
        MomentumY        = Sb * Vy,
        MomentumZ        = Sb * Vz,
        EnergyStagnation = Sb * EnthalpyStagnation
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

    # Coordinates
    cosTheta = np.cos(DataSourceTerms['theta'])
    sinTheta = np.sin(DataSourceTerms['theta'])

    # Flow data
    Density = np.maximum(FlowSolution['Density'], tol)
    Vx, Vy, Vz = FlowSolution['MomentumX']/Density, FlowSolution['MomentumY']/Density, FlowSolution['MomentumZ']/Density
    Wx, Wr, Wt = Vx, Vy*cosTheta+Vz*sinTheta, -Vy*sinTheta + Vz*cosTheta-DataSourceTerms['radius']*RotationSpeed
    Vmag, Wmag = (Vx**2+Vy**2+Vz**2)**0.5, np.maximum(tol, (Wx**2+Wr**2+Wt**2)**0.5)
    Temperature = np.maximum(tol, (FlowSolution['EnergyStagnationDensity']/Density-0.5*Vmag**2.)/FluidProperties['cp'])
    Mrel = Wmag/(FluidProperties['Gamma']*FluidProperties['IdealGasConstant']*Temperature)**0.5

    # Vitesses normales et parallele au squelette
    Wpn = Wx*DataSourceTerms['nx'] + Wr*DataSourceTerms['nr'] + Wt*DataSourceTerms['nt']
    Wnx = Wpn * DataSourceTerms['nx']
    Wnr = Wpn * DataSourceTerms['nr']
    Wnt = Wpn * DataSourceTerms['nt']
    Wpx, Wpr, Wpt = Wx-Wnx,   Wr-Wnr,   Wt-Wnt
    Wp = np.maximum(tol, (Wpx**2+Wpr**2+Wpt**2)**0.5)

    # Compressibility correction 
    CompressibilityCorrection = 3. * np.ones(Density.shape)
    subsonic_bf, supersonic_bf = np.less_equal(Mrel,0.99), np.greater_equal(Mrel,1.01)
    CompressibilityCorrection[subsonic_bf]  = np.clip(1.0/(1-Mrel[subsonic_bf]**2)**0.5, 0.0, 3.0)
    CompressibilityCorrection[supersonic_bf]= np.clip(4.0/(2*np.pi)/(Mrel[supersonic_bf]**2-1)**0.5, 0.0, 3.0)

    # Force normal to the chord
    blade2BladeDistance = 2*np.pi*DataSourceTerms['radius'] / NumberOfBlades * \
        np.absolute(DataSourceTerms['nt']) * DataSourceTerms['blockage']
    incidence = np.arcsin(Wpn/Wmag)
    fn = -0.5*Wmag**2. * CompressibilityCorrection * 2*np.pi*incidence / blade2BladeDistance * DataSourceTerms['isf']

    # Friction on blade
    Viscosity = FluidProperties['SutherlandViscosity']*np.sqrt(Temperature/FluidProperties['SutherlandTemperature'])*(1+FluidProperties['SutherlandConstant'])/(1+FluidProperties['SutherlandConstant']*FluidProperties['SutherlandTemperature']/Temperature)
    Re_x = Density*DataSourceTerms['xsc']*DataSourceTerms['chordx'] * Wmag / Viscosity
    cf = 0.0592*Re_x**(-0.2)

    # Force parallel to the chord
    delta0 = DataSourceTerms.get('delta0', 0.)
    fp = -0.5*Wmag**2. * (2*cf + 2*np.pi*(incidence - delta0)**2.) / blade2BladeDistance * DataSourceTerms['isf']

    fx = fn*(np.cos(incidence)*DataSourceTerms['nx']-np.sin(incidence)*Wpx/Wp) + fp*Wx/Wmag
    fr = fn*(np.cos(incidence)*DataSourceTerms['nr']-np.sin(incidence)*Wpr/Wp) + fp*Wr/Wmag
    ft = fn*(np.cos(incidence)*DataSourceTerms['nt']-np.sin(incidence)*Wpt/Wp) + fp*Wt/Wmag
    fy = -sinTheta * ft + cosTheta * fr
    fz =  cosTheta * ft + sinTheta * fr

    NewSourceTerms = dict(
        Density          = 0.,
        MomentumX        = Density * fx,
        MomentumY        = Density * fy,
        MomentumZ        = Density * fz,
        EnergyStagnation=Density * DataSourceTerms['radius'] * RotationSpeed * ft
    )

    # Add blockage terms
    BlockageSourceTerms = computeBlockageSourceTerms(zone, tol=tol)
    for key, value in BlockageSourceTerms.items():
        NewSourceTerms[key] += value

    return NewSourceTerms


def computeProtectionSourceTerms(zone, TurboConfiguration, ProtectedHeightPercentage=5., tol=1e-5):
    ''' 
    Protection of the boudary layer ofr body-force modelling, as explain in the appendix D of 
    W. Thollet PdD manuscrit.

    .. danger:: Available only for structured mesh.

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
        Density          = 0.,
        MomentumX        = S_corr_BL_x,
        MomentumY        = S_corr_BL_y,
        MomentumZ        = S_corr_BL_z,
        EnergyStagnation = 0.
    )
    return BLProtectionSourceTerms