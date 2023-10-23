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
MOLA - BodyForceTurbomachinery.py

There are 2 kinds of functions in this file :

* Functions for preprocessing, that help to create a mesh adapted for body-force
  and to extract the geometrical parameters needed for the model.

* Functions for coprocessing, that implement body-force models to update source terms 
  during the simulation.

File history:
8/09/2022 - T. Bontemps - Creation
'''

import MOLA

if not MOLA.__ONLY_DOC__:
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


def replaceRowWithBodyForceMesh(t, BodyForceRows, saveGeometricalDataForBodyForce=False):
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

            .. note:: 
                You may include a dict **ErsatzParameters** in for a row in **BodyForceRows**.
                If so, parameters are used as it for ertsaz profile parametrization (profile[-1].set(param, value))

        saveGeometricalDataForBodyForce : bool
            If :py:obj:`True`, save the intermediate files 'BodyForceData_{row}.cgns' for each row.
            These files contain a CGNS tree with :
                #. 4 lines (1D zones) corresponding to Hub, Shroud, Leading edge and Trailing Edge.
                #. The zone'Skeleton' with geometrical data on blade profile (used for interpolation later). 
    
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

        if 'locateLE' in BodyForceParams:
            locateLE = BodyForceParams.pop('locateLE')
        else:
            locateLE = 'auto'
        if 'ErsatzParameters' in BodyForceParams:
            ErsatzParameters = BodyForceParams.pop('ErsatzParameters')
        else:
            ErsatzParameters = dict()

        # Get the meridional info from rowTree
        if os.path.isfile(f'BodyForceData_{row}.cgns'):
            print(J.CYAN + f'Find body-force input BodyForceData_{row}.cgns' + J.ENDC)
            meridionalMesh = C.convertFile2PyTree(f'BodyForceData_{row}.cgns')
        else:
            meridionalMesh = extractRowGeometricalData(t, row, save=saveGeometricalDataForBodyForce, locateLE=locateLE, **ErsatzParameters)

        newRowMesh = buildBodyForceMeshForOneRow(meridionalMesh,
                                                 NumberOfBlades=BladeNumber,
                                                 RowFamily=row,
                                                 **BodyForceParams
                                                 )
        restoreBCFamilies(zones, newRowMesh)
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

def generateThroughFlowGeometry(Hub, Shroud, Blade):
    '''
    Generate the input geometry for :py:func:`buildBodyForceMeshForOneRow`.

    Parameters
    ----------
    Hub : PyTree
        Hub line, oriented such as :math:`(x,y,z) = (x_{hub},r_{hub},0)`
    Shroud : PyTree
        Shroud line, oriented such as :math:`(x,y,z) = (x_{shroud},r_{shroud},0)`
    Blade : PyTree
        Sections given by :py:func:`GenerativeShapeDesign.wing`
    
    Returns
    -------
        PyTree
            2D mesh ready to be used in :py:func:`buildBodyForceMeshForOneRow`.
    '''

    def curve(name, x, y, z=None):
        if z is None: z = np.zeros(x.size)
        curve = J.createZone(name, [x, y, z], ['CoordinateX', 'CoordinateY', 'CoordinateZ'])
        return curve

    assert len(I.getZones(Hub)) == 1
    assert len(I.getZones(Shroud)) == 1
    Hub = I.getZones(Hub)[0] 
    I.setName(Hub, 'Hub')
    Shroud = I.getZones(Shroud)[0] 
    I.setName(Shroud, 'Shroud')

    xhub, rhub = J.getxy(Hub)
    xshroud, rshroud = J.getxy(Shroud)
    Inlet        = curve('Inlet', np.array([xhub[0], xshroud[0]]), np.array([rhub[0], rshroud[0]]))
    Outlet       = curve('Outlet', np.array([xhub[-1], xshroud[-1]]), np.array([rhub[-1], rshroud[-1]]))

    if I.isStdNode(Blade) == 0:
       # List of PyTree: Blade is a list of Sections, as returned by GSD.wing
       Sections = Blade
    elif I.isStdNode(Blade) == -1:
        # PyTree: Blade is a Wing PyTree, as returned by GSD.wing
        # We must decompose the blade into sections  
        raise Exception('Not yet implemented. Blade must be a list of PyTree (Sections)')
    else:
        raise Exception('Blade must be a PyTree (Wing) or a list of PyTree (Sections)')

    SkeletonLines = [] 

    W.T._rotate(Sections, (0,0,0), (1, 0, 0), 90., vectors=[['TangentX', 'TangentY', 'TangentZ']])

    xLE = [] 
    rLE = [] 
    xTE = [] 
    rTE = []
    i = 1
    for section in Sections:
        print(i); i+=1
        AirfoilProperties, Camber = W.getAirfoilPropertiesAndCamber(section)

       # Update LE and TE curves 
        LE = AirfoilProperties['LeadingEdge']
        TE = AirfoilProperties['TrailingEdge']
        xLE.append(LE[0])
        xTE.append(TE[0])
        rLE.append((LE[1]**2+LE[2]**2)**0.5)
        rTE.append((TE[1]**2+TE[2]**2)**0.5)

       # Get the skeleton line by flattening the camber line 
        FlatCamber = I.copyTree(Camber)
        x, y, z = J.getxyz(FlatCamber)
        z[:] = z[0]

        C._initVars(FlatCamber, '{nx}=-{TangentZ}')
        C._initVars(FlatCamber, '{nr}=0')
        C._initVars(FlatCamber, '{nt}={TangentX}')
        C._initVars(FlatCamber, f'{{thickness}}= {{RelativeThickness}}*{AirfoilProperties["Chord"]}') 
        C._initVars(FlatCamber, f'{{AbscissaFromLE}}={{s}}*{W.getLength(Camber)}')
        C._extractVars(FlatCamber, ['CoordinateX', 'CoordinateY', 'CoordinateZ', 'nx', 'nr', 'nt', 'thickness', 'AbscissaFromLE'])

        SkeletonLines.append(FlatCamber)

    skeletonZone = G.stack(SkeletonLines)
    I.setName(skeletonZone, 'Skeleton')

    xLE = np.array(xLE)
    rLE = np.array(rLE) 
    xTE = np.array(xTE) 
    rTE = np.array(rTE)

    LeadingEdge  = curve('LeadingEdge', xLE, rLE)
    TrailingEdge = curve('TrailingEdge', xTE, rTE)
    
    t = C.newPyTree(['Base', [Hub, Shroud, LeadingEdge,
                    TrailingEdge, Inlet, Outlet, skeletonZone]])

    return t

def buildBodyForceMeshForOneRow(t, NumberOfBlades, RowFamily='ROW',
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

        RowFamily : str
            Name of the zone family for the output tree. Default is 'ROW'.

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
    I.setName(upstream, f'{RowFamily}_upstream')
    I.setName(bodyForce, f'{RowFamily}_bodyForce')
    I.setName(downstream, f'{RowFamily}_downstream')

    if model == 'hall':
        # Interpolate skeleton data
        Skeleton = I.getNodeFromName2(t, 'Skeleton')
        if Skeleton:
            # Post._extractMesh is not robust for 2D to 2D interpolation
            # Solution: make the source tree Skeleton 'thick' to make a 3D to 2D interpolation
            T._addkplane(Skeleton) # Add a plane in k direction at z0+1
            T._translate(Skeleton, (0,0,-0.5))
            P._extractMesh(Skeleton, bodyForce, order=2, extrapOrder=0) 
            bodyForce = computeBlockage(bodyForce, NumberOfBlades)
            bodyForce = P.computeDiff(bodyForce, 'CoordinateX')
            I._renameNode(bodyForce, 'diffCoordinateX', 'dx')
            # addMetalAngle(bodyForce)
            bodyForce = C.node2Center(bodyForce, I.__FlowSolutionNodes__)
            I._rmNodesByName1(bodyForce, I.__FlowSolutionNodes__)

    mesh2d = C.newPyTree(['Base', [upstream, bodyForce, downstream]])

    # Make a partial revolution to build the 3D mesh
    mesh3d = D.axisym(mesh2d, (0, 0, 0), (1, 0, 0), angle=AzimutalAngleDeg, Ntheta=NumberOfAzimutalPoints)

    bodyForce = I.getNodeFromName2(mesh3d, f'{RowFamily}_bodyForce')
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
        C._initVars(bodyForce,'{{centers:totalVolume}}={totalVolume}'.format(totalVolume=np.sum(volumeCell)))
        I._rmNodesByNameAndType(bodyForce, 'vol', 'DataArray_t')
        

    base = I.getBases(mesh3d)[0]
    upstream = I.getNodeFromNameAndType(base, f'{RowFamily}_upstream', 'Zone_t')
    bodyForce = I.getNodeFromNameAndType(base, f'{RowFamily}_bodyForce', 'Zone_t')
    downstream = I.getNodeFromNameAndType(base, f'{RowFamily}_downstream', 'Zone_t')

    # Add BC and Families
    C._addFamily2Base(base, RowFamily)
    for zone in I.getZones(base):
        C._tagWithFamily(zone, RowFamily)

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

def interpolateBodyForceMeshesForOneRow(meshes, AzimuthalPositions, NumberOfAzimutalPoints=100, kind='pchip'):
    '''
    Generate a 360 mesh with non-uniform data source terms for body-force.

    Parameters
    ----------
    meshes : :py:class:`list` of PyTree
        Meshes generated by :py:func:`buildBodyForceMeshForOneRow`. Only the first K index will be used, 
        so it may be as thin as possible.
    
    AzimuthalPositions : :py:class:`list` of :py:class:`float`
        Positions (in degrees) of each mesh to perform the interpolation.
    
    kind : str
        Interpolation method

    Returns
    -------
    PyTree
        _description_
    '''
    import scipy

    I.__FlowSolutionCenters__ = 'FlowSolution#DataSourceTerm'

    def extract_mesh_k0(mesh3D):
        mesh2D = I.newCGNSTree()
        base = I.newCGNSBase('Base', cellDim=2, physDim=3, parent=mesh2D)
        for zone in I.getZones(mesh3D):
            zone2D = T.subzone(zone, (1,1,1), (-1,-1,1))
            name = I.getName(zone2D)
            I.setName(zone2D, '.'.join(name.split('.')[:-1]))
            I._addChild(base, zone2D)
        return mesh2D
    
    def interpolate(theta, data, axis=1):
        if kind.startswith('interp1d'):
            ScipyLaw = kind.split('_')[1]
            return scipy.interpolate.interp1d(theta, data, axis=axis, kind=ScipyLaw, bounds_error=False, fill_value='extrapolate')
        if kind == 'pchip':
            return scipy.interpolate.PchipInterpolator(theta, data, axis=axis, extrapolate=True)
        if kind == 'akima':
            return scipy.interpolate.Akima1DInterpolator(theta, data, axis=axis)
        if kind == 'cubic':
            return scipy.interpolate.CubicSpline(theta, data, axis=axis, extrapolate='periodic')

    def prepareMeshList(meshes, Nazim):
        # Cell to node for all meshes
        newMeshes = []
        for n, mesh in enumerate(meshes):
            newMesh = I.copyRef(mesh)
            for zone in I.getZones(newMesh):
                FlowSolution = I.getNodeFromName1(zone, 'FlowSolution#DataSourceTerm')
                if FlowSolution:
                    VariableNames = [I.getName(node) for node in I.getNodesFromType1(FlowSolution, 'DataArray_t')]
                    for var in ['centers:'+name for name in VariableNames]:
                        C._center2Node__(zone, var, cellNType=0)
            
            I._rmNodesByName(zone, 'FlowSolution#DataSourceTerm')

            # Extract 2D mesh
            newMesh = extract_mesh_k0(newMesh)

            # Rotate meshes
            T._rotate(newMesh, (0,0,0), (1,0,0), 360*n/Nazim, vectors=[['nx', 'ny', 'nz']])

            newMeshes.append(newMesh)

        return newMeshes, VariableNames

    # AzimuthalPositions = np.array(AzimuthalPositions)
    Nazim = AzimuthalPositions.size
            
    # Cell to node, remove unnecessary FlowSolution, extract 2D meshes, and rotate meshes
    meshes, VariableNames = prepareMeshList(meshes, Nazim)

    # Add a last mesh to buckle the loop
    AzimuthalPositions.append(360.)
    meshes.append(meshes[0])
    Nazim += 1

    # Make a revolution to build the 3D mesh
    t = D.axisym(meshes[0], (0, 0, 0), (1, 0, 0), angle=360., Ntheta=NumberOfAzimutalPoints)

    for zone in I.getZones(t):
        zoneName = I.getName(zone)
        zoneType, ni, nj, nk, celldim = I.getZoneDim(zone)
        assert zoneType == 'Structured'

        x, y, z = J.getxyz(zone)
        interpAzimPositions = np.arctan2(y[0, :, :], z[0, :, :]) * 180./np.pi
        InterpDataSourceTerms = dict()
        if zoneName.endswith('bodyforce'):
            for var in VariableNames:
                InterpDataSourceTerms[var] = np.zeros((ni, nj, nk),dtype=np.float64,order='F')
        
        for j in range(nj): # loop on slice at constant height

            # Initialize data arrays
            InterpXmatrix = np.zeros((ni, Nazim),dtype=np.float64,order='F')
            InterpYmatrix = np.zeros((ni, Nazim),dtype=np.float64,order='F')
            InterpZmatrix = np.zeros((ni, Nazim),dtype=np.float64,order='F')
            DataSourceTerms = dict()
            for name in InterpDataSourceTerms:
                DataSourceTerms[name] = np.zeros((ni, Nazim),dtype=np.float64,order='F')
            
            # Fill data arrays for interpolation
            for n, mesh in enumerate(meshes):
                zoneMesh = I.getNodeFromName2(mesh, zoneName)

                # TODO : Interpolate only x and r

                InterpXmatrix[:,n] = J.getx(zoneMesh)[:,j]
                InterpYmatrix[:,n] = J.gety(zoneMesh)[:,j]  
                InterpZmatrix[:,n] = J.getz(zoneMesh)[:,j]     
                if len(InterpDataSourceTerms) != 0:
                    FS = I.getNodeFromName(zoneMesh, 'FlowSolution') 
                    for node in I.getNodesFromType(FS, 'DataArray_t'):
                        DataSourceTerms[I.getName(node)][:,n] = I.getValue(node)[:,j] 
            
            # Build interpolators
            interpX = interpolate(AzimuthalPositions, InterpXmatrix)
            interpY = interpolate(AzimuthalPositions, InterpYmatrix)
            interpZ = interpolate(AzimuthalPositions, InterpZmatrix)
            interpData = dict()
            for name in DataSourceTerms:
                interpData[name] = interpolate(AzimuthalPositions, DataSourceTerms[name])
            
            # Evaluate interpolators
            x[:, j, :] = interpX(interpAzimPositions[j, :])
            y[:, j, :] = interpY(interpAzimPositions[j, :])
            z[:, j, :] = interpZ(interpAzimPositions[j, :])
            for name in DataSourceTerms:
                InterpDataSourceTerms[name][:, j, :] = interpData[name](interpAzimPositions[j, :])
        
        if len(InterpDataSourceTerms) != 0:
            J.set(zone, 'FlowSolution', childType='FlowSolution_t', **InterpDataSourceTerms)
            for var in VariableNames:
                C._node2Center__(zone, var)
    
    I._rmNodesByName(t, 'FlowSolution')
    I.__FlowSolutionCenters__ = 'FlowSolution#Centers'

    base = I.getBases(t)[0]
    upstream = I.getNodeFromNameAndType(base, '*_upstream', 'Zone_t')
    downstream = I.getNodeFromNameAndType(base, '*_downstream', 'Zone_t')

    for family in ['Inflow', 'Outflow', 'Hub', 'Shroud']:
        I._rmNodesFromName1(base, family)

    FamilyNode = I.newFamily(name='Inflow', parent=base)
    I.newFamilyBC(value='BCInflow', parent=FamilyNode)
    FamilyNode = I.newFamily(name='Outflow', parent=base)
    I.newFamilyBC(value='BCOutflow', parent=FamilyNode)
    FamilyNode = I.newFamily(name='Hub', parent=base)
    I.newFamilyBC(value='BCWallViscous', parent=FamilyNode)
    FamilyNode = I.newFamily(name='Shroud', parent=base)
    I.newFamilyBC(value='BCWallViscous', parent=FamilyNode)

    C._addBC2Zone(upstream, 'inflow', 'FamilySpecified:Inflow', wrange='imin')
    C._addBC2Zone(downstream, 'outflow', 'FamilySpecified:Outflow', wrange='imax')
    for zone in I.getZones(base):
        C._addBC2Zone(zone, 'hub', 'FamilySpecified:Hub',    wrange='jmin')
        C._addBC2Zone(zone, 'shroud', 'FamilySpecified:Shroud', wrange='jmax')

    X.connectMatch(t, tol=1e-8)
    I.checkPyTree(t)
    I._correctPyTree(t)

    return t

def addMetalAngle(t):
    '''
    Add the reference flow incidence ``delta0`` for the loss part of the Hall-Thollet model.
    It is the metal angle, between the blade skeleton and the axial direction.

    :math:`\delta_0 = sign(t_x) * arccos(t_x / t)`
    :math:`\delta_0 = sign(n_t) * arccos(n_t)` with :math:`t_x=n_t` and :math:`t=n=1`

    .. danger::
        
        WORK IN PROGRESS, DO NOT USE THIS FUNCTION ! TO TEST

    Parameters
    ----------
    t : PyTree
        input mesh tree. Must contains the node 'nt' (azimuthal component of the unit vector normal to the blade).
    '''
    C._initVars(t, '{delta0}=sign({nt})*arccos({nt})')


def addReferenceFlowIncidence(t, tsource, omega):
    '''
    Add the reference flow incidence ``delta0`` for the loss part of the Hall-Thollet model.

    .. danger::
        
        WORK IN PROGRESS, DO NOT USE THIS FUNCTION ! TO TEST

    Parameters
    ----------
    t : PyTree
        input mesh tree
    tsource : PyTree
        Source tree with flow solution. It must be a previous body-force simulation converged on the maximum
        efficiency operating point.
    omega : float
        rotationnal speed in rad/s
    '''

    I._renameNode(tsource, 'FlowSolution#Init', I.__FlowSolutionCenters__)

    # Move coordinates to cell center and compute radius and azimuthal angle
    for coord in ['CoordinateX', 'CoordinateY', 'CoordinateZ']:
        C._node2Center__(tsource, coord)
        C._initVars(tsource, '{centers:radius}=sqrt({centers:CoordinateY}**2+{centers:CoordinateZ}**2)')
        C._initVars(tsource, '{centers:theta}=arctan2({centers:CoordinateZ},{centers:CoordinateY})')
    C._initVars(tsource,'{centers:rovr}= {centers:MomentumY}*cos({centers:theta})+{centers:MomentumZ}*sin({centers:theta})')
    C._initVars(tsource,'{centers:rovt}=-{centers:MomentumY}*sin({centers:theta})+{centers:MomentumZ}*cos({centers:theta})')
    C._initVars(tsource, '{{centers:rowt}}={{centers:rovt}}-{{centers:Density}}*{{centers:radius}}*{omega}'.format(omega))

    C._initVars(
        tsource, '{centers:delta0}=arcsin(({centers:rovx*nx+rovr}*{centers:nr}+{centers:rowt}*{centers:nt})\
                                  /(({centers:rovx}**2+{centers:rovr}**2+{centers:rowt}**2)**0.5+0.0001))')

    # Interpolate delta0 on the mesh
    C._extractVars(tsource, ['centers:delta0'])
    P._extractMesh(tsource, t, order=2, extrapOrder=0)


def extractRowGeometricalData(mesh, row, save=False, locateLE='auto', **ErsatzParameters):
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
        If :py:obj:`True`, save the output tree with the name 'BodyForceData_{row}.cgns'. 
    locateLE : str
        Must be 'auto' (default value) or 'from_index'. If 'from_index', the index of leading edge
        is set to :math:`N/2 + 1`, where :math:`N` is the number of points on the whole profile (suction
        and pressure sides).

    **ErsatzParameters : 
        Parameters used as it for ertsaz profile parametrization (profile[-1].set(param, value))

    Returns
    -------
    PyTree
        The ouput tree has the following 1D zones (lines): 
        Hub, Shroud, LeadingEdge, TrailingEdge, Inlet, Outlet.
        The last zone 'Skeleton' is a 2D zone, located between the LE and the TE, 
        with geometrical data on the blade:
        'nx', 'nr', 'nt', 'thickness', 'AbscissaFromLE'
    '''

    def profilesExtractionAndAnalysis(tree, row, directory_profiles='profiles', locateLE='auto', **ErsatzParameters):

        import Ersatz as EZ
        import etc.geom.xr_features as XR

        # if directory doesnt exist create it
        if (not(os.path.isdir(directory_profiles))):
            os.mkdir(directory_profiles)

        familyNames = [] 
        for pattern in ['HUB', 'hub', 'Hub']: 
            families = I.getNodesFromNameAndType(tree, f'{row}_*{pattern}*', 'Family_t')
            familyNames += [I.getName(fam) for fam in families]
        x, r = XR.extract_xr_family(tree, familyNames)
        curve = J.createZone('hub', [x, r], ['x', 'r'])
        C.convertPyTree2File(curve, 'hub.dat')

        familyNames = [] 
        for pattern in ['SHROUD', 'shroud', 'Shroud']: 
            families = I.getNodesFromNameAndType(tree, f'{row}_*{pattern}*', 'Family_t')
            familyNames += [I.getName(fam) for fam in families]
        x, r = XR.extract_xr_family(tree, familyNames)
        curve = J.createZone('shroud', [x, r], ['x', 'r'])
        C.convertPyTree2File(curve, 'shroud.dat')

        # Get blade profiles, assuming that there is only one zone around the blade skin (O-block)
        for zone in C.getFamilyZones(tree, row):
            # Search a Blade BC
            FOUND_BLADE = False
            for bc in I.getNodesFromType2(zone, 'BC_t'):
                FamilyNameNode = I.getNodeFromType1(bc, 'FamilyName_t')
                if not FamilyNameNode: continue
                FamilyName = I.getValue(FamilyNameNode)
                if any([pattern in FamilyName for pattern in ['BLADE', 'blade', 'Blade']]):
                    FOUND_BLADE = True
                    break
            if not FOUND_BLADE: 
                continue

            # Get dimension
            _, _, zone_j_dim, zone_k_dim, _ = I.getZoneDim(zone)
            NptsOnProfile = []
            for j in range(zone_j_dim):
                zone_t = T.subzone(zone, (1, j+1, 1), (1, j+1, zone_k_dim))
                x, y, z = J.getxyz(zone_t)
                curve = J.createZone('profile', [x, y, z], ['x', 'y', 'z'])
                C.convertPyTree2File(curve, f'{directory_profiles}/profile{j+1:03d}.dat')
                NptsOnProfile.append(x.size)
            
            # Because of the assumption that there is only one zone around the blade skin,
            # the loop can be broken
            Nprofiles = zone_j_dim
            break

        ezpb = EZ.Ersatz()
        # hub and shroud information (needed for the "height" variable calculation)
        ezpb.set('hub', 'hub.dat')
        ezpb.set('shroud', 'shroud.dat')

        profile = []
        for n in range(Nprofiles):
            profile.append(EZ.Profile(ezpb))
            profile[-1].set('file', f'{directory_profiles}/profile{n+1:03d}.dat')
            for param, value in ErsatzParameters.items():
                profile[-1].set(param, value)
            if locateLE == 'from_index':
                profile[-1].set('LEindex', NptsOnProfile[n]/2+1)
            else:
                assert locateLE =='auto'

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


    directory_profiles = 'profiles_{}'.format(row)
    profilesExtractionAndAnalysis(mesh, row, directory_profiles=directory_profiles, locateLE=locateLE, **ErsatzParameters)

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

    xhub, rhub = readEndWallLine('hub.dat')
    xshroud, rshroud = readEndWallLine('shroud.dat')
    xLE, rLE = readLEorTE('LE.dat')
    xTE, rTE = readLEorTE('TE.dat')
    assert rTE.size > 5, J.FAIL+f'Building of body-force mesh in MOLA assumes that the Autogrid mesh contains more than 5 points in the radial direction ({rTE.size} points detected in the current AG5 mesh)'+J.ENDC
    
    Hub          = curve('Hub', xhub, rhub)
    Shroud       = curve('Shroud', xshroud, rshroud)
    LeadingEdge  = curve('LeadingEdge', xLE, rLE)
    TrailingEdge = curve('TrailingEdge', xTE, rTE)
    Inlet        = curve('Inlet', np.array([xhub[0], xshroud[0]]), np.array([rhub[0], rshroud[0]]))
    Outlet       = curve('Outlet', np.array([xhub[-1], xshroud[-1]]), np.array([rhub[-1], rshroud[-1]]))

    t = C.newPyTree(['Base', [Hub, Shroud, LeadingEdge,
                    TrailingEdge, Inlet, Outlet, skeletonZone]])

    os.system(f'rm -f hub.dat shroud.dat')
    os.system(f'rm -rf {directory_profiles}')
    os.system('rm -f LE.dat TE.dat chord.dat thickness.dat skeleton.dat')

    if save:
        C.convertPyTree2File(t, f'BodyForceData_{row}.cgns')

    return t


def computeBlockage(t, Nblades, eps=1e-6):
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
        numerical parameter added to 'nt' square, to prevent division by zero. By default 1e-6

    Returns
    -------
    PyTree 
        Updated tree, with the new nodes 'blockage', 'gradxb' and 'gradrb'.
    '''


    C._initVars(t, f'{{b}}=maximum(-1+{eps}, -{{thickness}} / ({{nt}}**2+{eps})**0.5 / ({2*np.pi/Nblades}*{{CoordinateY}}))')

    # Force b = 0 on the edges on I indices (LE and TE)
    # Force a constant grad(b) on the edges on J indicies (hub and shroud)
    # It helps to have consistant grandients at the edge of the BF domain
    bNode = I.getNodeFromName(t, 'b')
    b = I.getValue(bNode)
    b[ 0, :] = 0 # LE
    b[-1, :] = 0 # TE
    b[ :, 0] = b[ :, 1] # Hub
    b[ :,-1] = b[ :,-2] # Shroud
    I.setValue(bNode, b)

    # Compute gradient of b and correct the definition of b
    t = P.computeGrad(t, 'b')
    C._initVars(t, '{blockage}=1.+{b}')
    I._rmNodesByName(t, 'b')
    I._rmNodesByName(t, 'gradzb')
    I._renameNode(t, 'gradyb', 'gradrb')

    # Compute de distance between 2 blades in the direction norm
    C._initVars(t, '{radius}=sqrt({CoordinateY}**2+{CoordinateZ}**2)')
    C._initVars(t, f'{{blade2BladeDistance}}=2*{np.pi}*{{radius}}/{Nblades}*{{nt}}*{{blockage}}')

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

import MOLA.BodyForceModels.BodyForceModels as BFM

AvailableBodyForceModels = dict(
    blockage = BFM.BodyForceModel_blockage,
    blockage_correction = BFM.BodyForceModel_blockage_correction,
    hall_without_blockage = BFM.BodyForceModel_hall_without_blockage,
    HallThollet = BFM.BodyForceModel_HallThollet,
    ThrustSpread = BFM.BodyForceModel_ThrustSpread,
    constant = BFM.BodyForceModel_constant,
    ShockWaveLoss = BFM.BodyForceModel_ShockWaveLoss,
    EndWallsProtection = BFM.BodyForceModel_EndWallsProtection,
    spreadPressureLossAlongChord = BFM.spreadPressureLossAlongChord,
)

def computeBodyForce(t, BodyForceParameters):
    '''
    Compute Body force source terms.

    Parameters
    ----------
        t : PyTree

            Tree in which the source terms will be compute
        
        BodyForceParameters : dict
            Body force parameters for the current family.

    Returns
    -------
        dict
            New source terms to apply. Should be for example : 

            >>> TotalSourceTermsGloblal['zoneName'] = dict(Density=ndarray, MomentumX=ndarray, ...)

    '''
    # Get the list of source terms to compute
    if not isinstance(BodyForceParameters, list):
        BodyForceParameters = [BodyForceParameters]

    # Compute and gather all the required source terms
    TotalSourceTermsGlobal = dict()
    for modelParameters in BodyForceParameters:
        model = modelParameters.pop('model')
        NewSourceTermsGlobal = AvailableBodyForceModels[model](t, modelParameters)
        # Add the computed source terms to the total source terms
        addDictionaries(TotalSourceTermsGlobal, NewSourceTermsGlobal)

    return TotalSourceTermsGlobal


def addDictionaries(d1, d2):
    '''
    Update **d1** by adding values of **d2** to values in **d1**

    .. important:: 

        Dictionaries must have two levels like that:

        >>> d1['zoneName']['Density'] = np.ndarray(...)

    Parameters
    ----------

        d1 : dict
            Dictionary that will be updated
        
        d2 : dict
            Dictionary that will be added to **d1**

    '''
    for zone in d2:
        if not zone in d1:
            d1[zone] = d2[zone]
        else:
            for key, value in d2[zone].items():
                if key in d1[zone]:
                    d1[zone][key] += value
                else:
                    d1[zone][key] = value

def getAdditionalFields(zone, FluidProperties, RotationSpeed, tol=1e-5):
    '''
    Compute additional flow quantities used in body-force models, and store them into a
    temporary node in **zone** to have access to them later if the function is called more 
    than once.

    Parameters
    ----------

        zone : PyTree
            Current zone
        
        FluidProperties : dict
            as read in `setup.py`

        RotationSpeed : float
            Rotation speed of the current zone

        tol : float
            minimum value for quantities used as a denominator.

    Returns
    -------
    
        dict

            Newly computed quantities
    '''
    tmpMOLAFlowNode = I.getNodeFromName(zone, 'FlowSolution#tmpMOLAFlow')
    
    if tmpMOLAFlowNode:
        return J.getVars2Dict(zone, Container='FlowSolution#tmpMOLAFlow')

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

    tmpMOLAFlow = dict(
        theta = DataSourceTerms['theta'],

        Vx = Vx,
        Vy = Vy, 
        Vz = Vz,
        Vmag = Vmag,

        Wx = Wx,
        Wr = Wr,
        Wt = Wt,
        Wmag = Wmag,
        Wn = Wn,
        Wnx = Wnx,
        Wnr = Wnr,
        Wnt = Wnt,
        Wp = Wp,
        Wpx = Wpx,
        Wpr = Wpr,
        Wpt = Wpt,

        Temperature = Temperature,
        Mrel = Mrel,

        incidence = incidence,

        # Unit vector normal the velocity. Direction of application of the normal force
        unitVectorNormalX = np.cos(incidence) * DataSourceTerms['nx'] - np.sin(incidence)*Wpx/Wp,
        unitVectorNormalR = np.cos(incidence) * DataSourceTerms['nr'] - np.sin(incidence)*Wpr/Wp,
        unitVectorNormalT = np.cos(incidence) * DataSourceTerms['nt'] - np.sin(incidence)*Wpt/Wp,

        # Unit vector parallel to the velocity. Direction of application of the parallel force
        unitVectorParallelX = - Wx / Wmag,
        unitVectorParallelR = - Wr / Wmag,
        unitVectorParallelT = - Wt / Wmag,

    )

    J.set(zone, 'FlowSolution#tmpMOLAFlow', childType='FlowSolution_t', **tmpMOLAFlow)
    tmpMOLAFlowNode = I.getNodeFromName(zone, 'FlowSolution#tmpMOLAFlow')
    I.createChild(tmpMOLAFlowNode, 'GridLocation', 'GridLocation_t', value='CellCenter')

    return tmpMOLAFlow 

def getForceComponents(fn, fp, tmpMOLAFlow):
    '''
    Compute cartesian and cylindrical components of the force with its components in the blade local frame.

    Parameters
    ----------

        fn : numpy.ndarray
            Force component in the direction normal to the chord
        
        fp : numpy.ndarray
            Force component in the direction parallel to the chord (oriented upstream)

        tmpMOLAFlow : dict
            temporary container of flow quantities, as got by :py:func:`getAdditionalFields`

    Returns
    -------
    
        :py:class:`tuple` of :py:class:`numpy.ndarray`

        Force components in x, y, z, r and theta. 
    '''

    # Force in the cylindrical frame of reference
    fx = fn * tmpMOLAFlow['unitVectorNormalX'] + fp * tmpMOLAFlow['unitVectorParallelX']
    fr = fn * tmpMOLAFlow['unitVectorNormalR'] + fp * tmpMOLAFlow['unitVectorParallelR']
    ft = fn * tmpMOLAFlow['unitVectorNormalT'] + fp * tmpMOLAFlow['unitVectorParallelT']

    # Force in the cartesian frame of reference
    fy = -np.sin(tmpMOLAFlow['theta']) * ft + np.cos(tmpMOLAFlow['theta']) * fr
    fz =  np.cos(tmpMOLAFlow['theta']) * ft + np.sin(tmpMOLAFlow['theta']) * fr

    return fx, fy, fz, fr, ft 

def getFieldsAtLeadingEdge(t):
    ...
