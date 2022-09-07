'''
MOLA - Postprocess.py

BETA Module

12/05/2021 - L. Bernardos - Creation from recycling
'''

# System modules
import sys
import os
import time
import timeit
import shutil
import imp
import copy
import numpy as np
from itertools import product

# Cassiopee
import Converter.PyTree as C
import Converter.Internal as I
import Transform.PyTree as T
import Generator.PyTree as G
import Geom.PyTree as D
import Post.PyTree as P
import Converter.elsAProfile as eP

# MOLA
from . import InternalShortcuts as J
from . import Wireframe as W
from . import GenerativeShapeDesign as GSD
from . import GenerativeVolumeDesign as GVD

def extractBoundaryLayer(t, rotation_scale_edge_threshold=0.003,
        boundary_layer_maximum_height=0.5, auxiliar_grid_maximum_nb_points=300,
        auxiliar_grid_first_cell_height=1e-6, auxiliar_grid_growth_rate=1.05):
    '''
    Produce a set of surfaces including relevant boundary-layer quantities from
    a 3D fields tree using conservative fields (for example, a result of
    ``OUTPUT/fields.cgns``).

    This function produces a set of surfaces (similar to ``OUTPUT/surfaces.cgns``)

    Parameters
    ----------

        t : PyTree
            tree containing at least conservative flowfields and *BCWall*
            conditions. For exemple, it can be the tree contained in
            ``OUTPUT/fields.cgns``.

            .. important::
                conservative flowfields must be contained at containers named
                ``FlowSolution#Centers`` or ``FlowSolution#Init`` located at
                *CellCenter*. The following fields must be present: ``Density``,
                ``MomentumX``, ``MomentumY``, ``MomentumZ``, ``ViscosityMolecular``.
                Field ``Pressure`` shall also be present. Otherwise, an attempt
                is done of computing this quantity if required conservative fields
                and ReferenceState exist in **t**

        rotation_scale_edge_threshold : float
            value used to find the boundary-layer edge location based on the
            rotation scale. Higher values lead to thinner boundary-layer height,
            and lower values lead to thicker boundary-layer height.

        boundary_layer_maximum_height : float
            maximum height of boundary-layer thickness (in mesh length units).

        auxiliar_grid_maximum_nb_points : int
            maximum number of points in the wall-normal direction used to
            build the auxiliar grid employed for the boundary-layer postprocess

        auxiliar_grid_first_cell_height : float
            height of the wall-adjacent cell used in the auxiliar grid employed
            for the boundary-layer postprocess. Should be lower than the actual
            wall cell height of **t**. Dimension is mesh length units.

        auxiliar_grid_growth_rate : float
            growth factor of the cell heights in the wall-normal direction used
            for the construction of the auxiliar grid employed for the
            boundary-layer postprocess


    Returns
    -------

        surfaces : PyTree
            tree containing surfaces with fields contained in ``FlowSolution#Centers``
            located at CellCenter (interfaces). Most relevant added fields are:
            ``DistanceToWall``, ``delta``, ``delta1``, ``theta11``, ``runit``
            ``VelocityTangential``, ``VelocityTransversal``,
            ``VelocityEdgeX``, ``VelocityEdgeY``, ``VelocityEdgeZ``,
            ``SkinFrictionX``, ``SkinFrictionY``, ``SkinFrictionZ``, ``Pressure``
    '''

    aux_grid = _buildAuxiliarWallExtrusion(t,
                    MaximumBoundaryLayerDistance=boundary_layer_maximum_height,
                    MaximumBoundaryLayerPoints=auxiliar_grid_maximum_nb_points,
                    BoundaryLayerGrowthRate=auxiliar_grid_growth_rate,
                    FirstCellHeight=auxiliar_grid_first_cell_height)

    # TODO optionnally keep existing quantities by renaming strategy
    C._rmVars(aux_grid, ['delta','delta1','theta11','runit',
                         'SkinFrictionX','SkinFrictionY','SkinFrictionZ'])

    _computeBoundaryLayerQuantities(aux_grid,
                                    VORTRATIOLIM=rotation_scale_edge_threshold)
    _computeSkinFriction(aux_grid)

    if 'Pressure' not in C.getVarNames(aux_grid, excludeXYZ=True)[0]:
        P._computeVariables(aux_grid, ['Pressure'] )

    _subzoneAuxiliarWallExtrusion(aux_grid)

    for z in I.getZones(aux_grid):
        AllFieldsNames, = C.getVarNames(z, excludeXYZ=True)
        for fn in AllFieldsNames:
            C.node2Center__(z, fn)
    I._rmNodesByName(aux_grid, 'FlowSolution')
    C._normalize(aux_grid,['centers:nx','centers:ny','centers:nz'])

    return aux_grid


def mergeWallsAndSplitAirfoilSides(t):
    '''
    Given a 1-cell-depth surface defining the wall of an airfoil, produce two
    curves corresponding to the Top and Bottom sides of the airfoil.

    Parameters
    ----------

        t : PyTree, base, zone, list of zones
            CGNS tree containing a 1-cell depth wall surface. It may contain
            flow solutions fields

    Returns
    -------

        t : PyTree
            Airfoil split in top and bottom sides including original fields.
    '''

    tRef = I.copyRef(t)
    tRef = T.merge(tRef)
    foil, = C.node2Center(tRef)

    TopSide, BottomSide = W.splitAirfoil(foil,
                                         FirstEdgeSearchPortion = 0.9,
                                         SecondEdgeSearchPortion = -0.9,
                                         RelativeRadiusTolerance = 1e-2,
                                         MergePointsTolerance = 1e-10)


    tOut =C.newPyTree(['Airfoil',[TopSide, BottomSide]])
    J.migrateFields(foil, tOut)

    return tOut


def getBoundaryLayerEdges(t):
    '''
    Produce new zones representing the boundary-layer edges if **t**
    contains the normal fields ``nx``, ``ny``, ``nz`` and at least one of:
    ``delta``, ``delta1``, ``delta11``

    Parameters
    ----------

        t : PyTree, base, zone, list of zones
            surface containing the fields
            ``nx`` ``ny`` ``nz`` and at least one of: ``delta``, ``delta1``, ``theta11``

            .. note:: the input **t** is modified : a new base including the
                zones of the boundary-layer thicknesses is appended to the tree

    Returns
    -------

        t : PyTree
            Tree containing the new boundary-layer characteristic edges
    '''


    t2 = C.normalize(t, ['nx','ny','nz'])

    Trees = []
    for Thickness2Plot in ["{delta}", "{delta1}", "{theta11}"]:
        Thickness2PlotNoBrackets = Thickness2Plot[1:-1]
        if C.isNamePresent(t2, Thickness2PlotNoBrackets) != 1: continue

        t3 = I.copyTree(t2)

        C._initVars(t3,'dx=%s*{nx}'%Thickness2Plot)
        C._initVars(t3,'dy=%s*{ny}'%Thickness2Plot)
        C._initVars(t3,'dz=%s*{nz}'%Thickness2Plot)
        T._deform(t3, ['dx','dy','dz'])

        I._rmNodesByType(t3,'FlowSolution_t')

        for z in I.getZones(t3):
            z[0] += '.'+Thickness2Plot[1:-1]

        Trees.append(t3)

    # TODO separate characteristic boundary per bases
    tM = I.merge(Trees)
    NewZones = I.getZones(tM)
    tNew = C.newPyTree(['BoundaryLayerEdges', NewZones])

    return tNew


def _computeBoundaryLayerQuantities(t, VORTRATIOLIM=1e-3):
    '''
    TODO doc
    '''
    NewFields = ['DistanceToWall', 'delta', 'delta1', 'theta11', 'runit',
                 'VelocityTangential', 'VelocityTransversal',
                 'VelocityEdgeX', 'VelocityEdgeY', 'VelocityEdgeZ']

    for zone in I.getZones(t):
        J._invokeFields(zone,NewFields)
        AllFieldsNames, = C.getVarNames(zone, excludeXYZ=True)
        v = J.getVars2Dict(zone, AllFieldsNames)
        x, y, z = J.getxyz(zone)

        Ni, Nj, NumberOfBoundaryLayerPoints = I.getZoneDim(zone)[1:4]
        for i,j in product(range(Ni), range(Nj)):

            eta = v['DistanceToWall'][i,j,:]
            VelocityTangential = v['VelocityTangential'][i,j,:]
            VelocityTransversal = v['VelocityTransversal'][i,j,:]
            VelocityX = v['VelocityX'][i,j,:]
            VelocityY = v['VelocityY'][i,j,:]
            VelocityZ = v['VelocityZ'][i,j,:]
            RotationScale = v['RotationScale'][i,j,:]
            RotationScaleMax = RotationScale.max()

            BoundaryLayerRegion, = np.where(RotationScale>VORTRATIOLIM*RotationScaleMax)

            if len(BoundaryLayerRegion) == 0:
                BoundaryLayerEdgeIndex = NumberOfBoundaryLayerPoints-1
            else:
                BoundaryLayerEdgeIndex = BoundaryLayerRegion[-1]

            # zero-th order boundary layer edge search
            v['VelocityEdgeX'][i,j,:] = VelocityX[BoundaryLayerEdgeIndex]
            v['VelocityEdgeY'][i,j,:] = VelocityY[BoundaryLayerEdgeIndex]
            v['VelocityEdgeZ'][i,j,:] = VelocityZ[BoundaryLayerEdgeIndex]

            VelocityEdgeVector = np.array([v['VelocityEdgeX'][i,j,0],
                                           v['VelocityEdgeY'][i,j,0],
                                           v['VelocityEdgeZ'][i,j,0]])

            NormalVector = np.array([v['nx'][i,j,0],
                                     v['ny'][i,j,0],
                                     v['nz'][i,j,0]])
            NormalVector/= np.sqrt(NormalVector.dot(NormalVector))

            BinormalVector = np.cross(VelocityEdgeVector, NormalVector)
            BinormalVector/= np.sqrt(BinormalVector.dot(BinormalVector))

            TangentVector = np.cross(NormalVector, BinormalVector)
            TangentVector/= np.sqrt(TangentVector.dot(TangentVector))


            VelocityTangential[:] =(VelocityX*TangentVector[0] +
                                    VelocityY*TangentVector[1] +
                                    VelocityZ*TangentVector[2])

            VelocityTransversal[:] =(VelocityX*BinormalVector[0] +
                                     VelocityY*BinormalVector[1] +
                                     VelocityZ*BinormalVector[2])

            eta[:] =((x[i,j,:]-x[i,j,0])*NormalVector[0] +
                     (y[i,j,:]-y[i,j,0])*NormalVector[1] +
                     (z[i,j,:]-z[i,j,0])*NormalVector[2])

            v['delta'][i,j,:] = eta[BoundaryLayerEdgeIndex]

            Ue = VelocityTangential[BoundaryLayerEdgeIndex]
            Ut = VelocityTangential[:BoundaryLayerEdgeIndex]

            IntegrandDelta1 = np.maximum(0, 1. - (Ut/Ue) )
            IntegrandTheta = np.maximum(0, (Ut/Ue)*(1. - (Ut/Ue)) )

            v['delta1'][i,j,:] = np.trapz(IntegrandDelta1,
                                              eta[:BoundaryLayerEdgeIndex])

            v['theta11'][i,j,:] = np.trapz(IntegrandTheta,
                                               eta[:BoundaryLayerEdgeIndex])

            v['runit'][i,j,:] = (v['Density'][i,j,BoundaryLayerEdgeIndex]*Ue/
                                v['ViscosityMolecular'][i,j,BoundaryLayerEdgeIndex])

def _computeSkinFriction(t):
    '''
    TODO doc
    '''

    Lambda = '(-2./3. * {ViscosityMolecular})'
    mu = '{ViscosityMolecular}'
    divU = '{VelocityDivergence}'

    Eqns = []

    # TODO implement this
    '''
    # REYNOLDSSTRESS COMPUTATION
    # mu_t
    Eqns += [('centers:ViscosityEddy={centers:Viscosity_EddyMolecularRatio}*{centers:ViscosityMolecular}')]

    # div(u)
    Eqns += [('centers:VelocityDivergence={centers:gradxVelocityX}+'
                                         '{centers:gradyVelocityY}+'
                                         '{centers:gradzVelocityZ}')]

    # Sij
    Eqns += [('centers:BarDeformationXX={centers:gradxVelocityX}-{centers:VelocityDivergence}/3.0')]
    Eqns += [('centers:BarDeformationYY={centers:gradyVelocityY}-{centers:VelocityDivergence}/3.0')]
    Eqns += [('centers:BarDeformationZZ={centers:gradzVelocityZ}-{centers:VelocityDivergence}/3.0')]
    Eqns += [('centers:BarDeformationXY=0.5*({centers:gradyVelocityX}+{centers:gradxVelocityY})')]
    Eqns += [('centers:BarDeformationXZ=0.5*({centers:gradzVelocityX}+{centers:gradxVelocityZ})')]
    Eqns += [('centers:BarDeformationYZ=0.5*({centers:gradzVelocityY}+{centers:gradyVelocityZ})')]

    # tau
    Eqns += [('centers:ReynoldsStressXX=2*{mut}*{SXX}-(2./3.)*{rok}').format(
        divU=divU,mut=mut,SXX='{centers:BarDeformationXX}',rok='{centers:TurbulentEnergyKineticDensity}')]
    Eqns += [('centers:ReynoldsStressYY=2*{mut}*{SYY}-(2./3.)*{rok}').format(
        divU=divU,mut=mut,SYY='{centers:BarDeformationYY}',rok='{centers:TurbulentEnergyKineticDensity}')]
    Eqns += [('centers:ReynoldsStressZZ=2*{mut}*{SZZ}-(2./3.)*{rok}').format(
        divU=divU,mut=mut,SZZ='{centers:BarDeformationZZ}',rok='{centers:TurbulentEnergyKineticDensity}')]
    Eqns += [('centers:ReynoldsStressXY=2*{mut}*{SXY}').format(
        mut=mut,SXY='{centers:BarDeformationXY}')]
    Eqns += [('centers:ReynoldsStressXZ=2*{mut}*{SXZ}').format(
        mut=mut,SXZ='{centers:BarDeformationXZ}')]
    Eqns += [('centers:ReynoldsStressYZ=2*{mut}*{SYZ}').format(
        mut=mut,SYZ='{centers:BarDeformationYZ}')]
    '''



    # div(u)
    Eqns += [('VelocityDivergence={gradxVelocityX}+'
                                 '{gradyVelocityY}+'
                                 '{gradzVelocityZ}')]

    # Sij
    Eqns += [('DeformationXX={gradxVelocityX}')]
    Eqns += [('DeformationYY={gradyVelocityY}')]
    Eqns += [('DeformationZZ={gradzVelocityZ}')]
    Eqns += [('DeformationXY=0.5*({gradyVelocityX}+{gradxVelocityY})')]
    Eqns += [('DeformationXZ=0.5*({gradzVelocityX}+{gradxVelocityZ})')]
    Eqns += [('DeformationYZ=0.5*({gradzVelocityY}+{gradyVelocityZ})')]

    # tau
    Eqns += [('ShearStressXX={Lambda}*{divU}+2*{mu}*{SXX}').format(
        Lambda=Lambda,divU=divU,mu=mu,SXX='{DeformationXX}')]
    Eqns += [('ShearStressYY={Lambda}*{divU}+2*{mu}*{SYY}').format(
        Lambda=Lambda,divU=divU,mu=mu,SYY='{DeformationYY}')]
    Eqns += [('ShearStressZZ={Lambda}*{divU}+2*{mu}*{SZZ}').format(
        Lambda=Lambda,divU=divU,mu=mu,SZZ='{DeformationZZ}')]
    Eqns += [('ShearStressXY=2*{mu}*{SXY}').format(
        mu=mu,SXY='{DeformationXY}')]
    Eqns += [('ShearStressXZ=2*{mu}*{SXZ}').format(
        mu=mu,SXZ='{DeformationXZ}')]
    Eqns += [('ShearStressYZ=2*{mu}*{SYZ}').format(
        mu=mu,SYZ='{DeformationYZ}')]

    Eqns += [('SkinFrictionX={ShearStressXX}*{nx}+'
                            '{ShearStressXY}*{ny}+'
                            '{ShearStressXZ}*{nz}')]
    Eqns += [('SkinFrictionY={ShearStressXY}*{nx}+'
                            '{ShearStressYY}*{ny}+'
                            '{ShearStressYZ}*{nz}')]
    Eqns += [('SkinFrictionZ={ShearStressXZ}*{nx}+'
                            '{ShearStressYZ}*{ny}+'
                            '{ShearStressZZ}*{nz}')]

    for Eqn in Eqns: C._initVars(t, Eqn)


def _subzoneAuxiliarWallExtrusion(t):
    '''
    TODO doc
    '''
    for zone in I.getZones(t):
        wall = GSD.getBoundary(zone,'kmin')
        zone[1] = wall[1]
        I._rmNodesByType(zone,'GridCoordinates_t')
        I._rmNodesByType(zone,'FlowSolution_t')
        NodesToMigrate = I.getNodesFromType(wall,'GridCoordinates_t')
        NodesToMigrate.extend(I.getNodesFromType(wall,'FlowSolution_t'))
        zone[2].extend(NodesToMigrate)


def _extractWalls(t):
    '''
    (WIP: PROOF OF CONCEPT)
    TODO refactorize with Coprocess extractSurfaces
    '''
    t = I.copyRef(t)
    cellDimOutputTree = I.getZoneDim(I.getZones(t)[0])[-1]

    from .Coprocess import reshapeBCDatasetNodes, restoreFamilies


    def addBase2SurfacesTree(basename):
        if not zones: return
        base = I.newCGNSBase(basename, cellDim=cellDimOutputTree-1, physDim=3,
            parent=SurfacesTree)
        I._addChild(base, zones)
        J.set(base, '.ExtractionInfo', **ExtractionInfo)
        return base

    t = I.renameNode(t, 'FlowSolution#Init', 'FlowSolution#Centers') # or merge?
    I._renameNode(t, 'FlowSolution#Height', 'FlowSolution')
    I._rmNodesByName(t, 'FlowSolution#EndOfRun*')
    reshapeBCDatasetNodes(t)
    DictBCNames2Type = C.getFamilyBCNamesDict(t)
    SurfacesTree = I.newCGNSTree()

    # See Anomaly 8784 https://elsa.onera.fr/issues/8784
    for BCDataSetNode in I.getNodesFromType(t, 'BCDataSet_t'):
        for node in I.getNodesFromType(BCDataSetNode, 'DataArray_t'):
            if I.getValue(node) is None:
                I.rmNode(BCDataSetNode, node)

    Extraction=dict(type='AllBCWall')
    TypeOfExtraction = Extraction['type']
    ExtractionInfo = copy.deepcopy(Extraction)
    BCFilterName = TypeOfExtraction.replace('AllBC','')
    for BCFamilyName in DictBCNames2Type:
        BCType = DictBCNames2Type[BCFamilyName]
        if BCFilterName.lower() in BCType.lower():
            zones = C.extractBCOfName(t,'FamilySpecified:'+BCFamilyName)
            ExtractionInfo['type'] = 'BC'
            ExtractionInfo['BCType'] = BCType
            addBase2SurfacesTree(BCFamilyName)

    J.forceZoneDimensionsCoherency(SurfacesTree)
    restoreFamilies(SurfacesTree, t)

    return SurfacesTree


def _buildAuxiliarWallExtrusion(t, MaximumBoundaryLayerDistance=0.5,
        MaximumBoundaryLayerPoints=300, BoundaryLayerGrowthRate=1.05,
        FirstCellHeight=1e-6, container='FlowSolution#Init'):
    '''
    TODO doc
    '''

    tR = I.copyRef(t)

    BoundaryLayerDistribution = GVD.newExtrusionDistribution(
                             MaximumBoundaryLayerDistance,
                             maximum_number_of_points=MaximumBoundaryLayerPoints,
                             distribution_law='ratio',
                             first_cell_height=FirstCellHeight,
                             ratio_growth=BoundaryLayerGrowthRate,
                             smoothing_normals_iterations=0,
                             smoothing_growth_iterations=0,
                             smoothing_expansion_factor=0.)

    # grid needs to be located at Vertex, see: Cassiopee #10404

    if container != 'FlowSolution':
        I._rmNodesByName(tR, 'FlowSolution')
        zone = I.getNodeFromType3(tR, 'Zone_t')
        FlowSolution_n = I.getNodeFromName1(zone, container)
        GridLocation_n = I.getNodeFromName(FlowSolution_n, 'GridLocation')
        if not GridLocation_n:
            field = FlowSolution_n[2][0][1]
            x = J.getx(zone)
            if field.shape[0] != x.shape[0]:
                ContainerIsCellCentered = True
            else:
                ContainerIsCellCentered = False
        else:
            GridLocation = I.getValue(GridLocation_n)
            if GridLocation != 'Vertex':
                ContainerIsCellCentered = True
            else:
                ContainerIsCellCentered = False


    if ContainerIsCellCentered:
        I.__FlowSolutionCenters__ = container

        for zone in I.getZones(tR):
            AllFieldsNames, = C.getVarNames(zone, excludeXYZ=True)
            CentersFieldsNames = [fn for fn in AllFieldsNames if 'centers:' in fn]
            x = J.getx(zone)

            for cfn in CentersFieldsNames:
                # TODO - Caution! do not produce dim 3 ! Cassiopee bug #8131
                C.center2Node__(zone, cfn, cellNType=0)

                # therefore, workaround is required:
                FlowSol_node = I.getNodeFromName(zone, I.__FlowSolutionNodes__)
                field_node = I.getNodeFromName(FlowSol_node, cfn.replace('centers:',''))
                if not np.all(field_node[1].shape == x.shape ):
                    array = np.broadcast_to( field_node[1], x.shape )
                    field_node[1] = np.asfortranarray( array )


    else:
        I.renameNode(tR, container, 'FlowSolution')


    AllFlowSolutionNodes = I.getNodesFromType(tR, 'FlowSolution_t')
    for FlowSolutionNode in AllFlowSolutionNodes:
        if FlowSolutionNode[0] != 'FlowSolution':
            I.rmNode(tR, FlowSolutionNode)




    walls_tree = _extractWalls(t)
    I._rmNodesByType( walls_tree , 'FlowSolution_t')

    # workaround: see Cassiopee #10404
    C._initVars(walls_tree, 'MomentumX', 0.)
    C._initVars(walls_tree, 'MomentumY', 0.)
    C._initVars(walls_tree, 'MomentumZ', 0.)
    _fitFields(walls_tree, tR, ['Momentum'+i for i in ['X','Y','Z']])


    aux_grids = []

    for wall in I.getZones(walls_tree):

        G._getNormalMap(wall)

        extrusion = GVD.extrude(wall, [BoundaryLayerDistribution])
        aux_grid, = I.getZones(I.getNodeFromName2(extrusion,'ExtrudedVolume'))
        aux_grid[0] = wall[0]
        P._renameVars(wall, ['centers:sx','centers:sy','centers:sz'],
                            ['centers:nx','centers:ny','centers:nz'])

        for n in ['nx','ny','nz']:
            x = J.getx(wall)
            # TODO - Caution! do not produce dim 2 ! Cassiopee bug #8131
            C.center2Node__(wall, 'centers:'+n, cellNType=0)
            # therefore, workaround is required:
            FlowSol_node = I.getNodeFromName(wall, I.__FlowSolutionNodes__)
            field_node = I.getNodeFromName(FlowSol_node, n.replace('centers:',''))
            if not np.all(field_node[1].shape == x.shape ):
                array = np.broadcast_to( field_node[1], x.shape )
                field_node[1] = np.asfortranarray( array )


        I._rmNodesByName(wall,I.__FlowSolutionCenters__)
        C._normalize(wall, ['nx','ny','nz'])
        J.migrateFields(wall, aux_grid,
                         keepMigrationDataForReuse=False, # TODO why cannot write if False ?
                         )

        P._extractMesh(tR, aux_grid, order=2, extrapOrder=1,
                       mode='accurate', constraint=40., tol=1.e-10)

        I.__FlowSolutionCenters__ = 'FlowSolution#Centers'

        # workaround: see Cassiopee #10404
        C._initVars(wall, 'MomentumX', 0.)
        C._initVars(wall, 'MomentumY', 0.)
        C._initVars(wall, 'MomentumZ', 0.)
        _fitFields(wall, aux_grid, ['Momentum'+i for i in ['X','Y','Z']])


        for v in ('X','Y','Z'):
            C._initVars(aux_grid, 'Velocity%s={Momentum%s}/{Density}'%(v,v))

        for v in ('X','Y','Z'): aux_grid = P.computeGrad(aux_grid,'Velocity%s'%v)

        AllFieldsNames, = C.getVarNames(aux_grid, excludeXYZ=True)
        CentersFieldsNames = [fn for fn in AllFieldsNames if 'centers:' in fn]

        x = J.getx(aux_grid)
        for cfn in CentersFieldsNames:
            # TODO BEWARE do not produce dim 3 ! ticket Cassiopee #8131
            C.center2Node__(aux_grid,cfn,cellNType=0)

            # workaround:
            FlowSol_node = I.getNodeFromName(aux_grid,I.__FlowSolutionNodes__)
            field_node = I.getNodeFromName(FlowSol_node, cfn.replace('centers:',''))
            if not np.all(field_node[1].shape == x.shape ):
                array = np.broadcast_to( field_node[1], x.shape )
                field_node[1] = np.asfortranarray( array )

        I._rmNodesByName(aux_grid,I.__FlowSolutionCenters__)


        GradientsNames = dict()
        for v, V in product(['x','y','z'],['X','Y','Z']):
            KeyName =  'vag' + v              + V.lower()
            Value   = 'grad' + v + 'Velocity' + V
            GradientsNames[KeyName] = '{'+Value+'}'

        # sqrt( 2 * Omega_ij * Omega_ij )
        Eqn = (
        '  sqrt(  ( {vagzy} - {vagyz} )**2 '
        '       + ( {vagxz} - {vagzx} )**2 '
        '       + ( {vagyx} - {vagxy} )**2 )'
        ).format(**GradientsNames)
        C._initVars(aux_grid,'RotationScale='+Eqn)

        '''
        # BEWARE! RotationScale seems to be a more appropriate scalar for
        # determining boundary-layer edge !
        # sqrt( 2 * barS_ij * barS_ij )
        Eqn = (
        '  sqrt( 2 * ( {vagxx}**2 + {vagyy}**2 + {vagzz}**2 )  '
        '          + ( {vagxy} + {vagyx})**2 '
        '          + ( {vagxz} + {vagzx})**2 '
        '          + ( {vagyz} + {vagzy})**2 '
        '          - (2./3.) * ({vagxx} + {vagyy} + {vagzz})**2'
        '       )'
        ).format(**GradientsNames)
        C._initVars(aux_grid,'DeformationScale='+Eqn)
        '''


        wall[1] = aux_grid[1]
        I._rmNodesByType(wall,'GridCoordinates_t')
        I._rmNodesByType(wall,'FlowSolution_t')
        NodesToMigrate = I.getNodesFromType(aux_grid,'GridCoordinates_t')
        NodesToMigrate.extend(I.getNodesFromType(aux_grid,'FlowSolution_t'))
        wall[2].extend(NodesToMigrate)

        aux_grids.append(aux_grid)

    return walls_tree

def getMeshDimensionFromTree(t):
    '''
    Return the dimensions of a PyTree (0, 1, 2, 3)

    Parameters
    ----------

        t : PyTree
            tree with zones that have all the same dimension

    Returns
    -------

        Dimension : int
            can be 0, 1, 2 or 3
    '''
    Dims = np.array([I.getZoneDim(z)[4] for z in I.getZones(t)])
    if not np.all(Dims == Dims[0]):
        ERRMSG = 'Input tree must yield all equal dimensions (2D or 3D)'
        raise ValueError(ERRMSG)
    Dimension = Dims[0]
    return Dimension

def putNormalsPointingOutwards(zone):
    '''
    Force the normals of a airfoil wall surface ``nx`` ``ny`` ``nz`` to point outwards

    Parameters
    ----------

        zone : zone
            surface containing ``nx`` ``ny`` ``nz``. It is modified in-place.
    '''
    x = J.getx(zone)
    nx, ny, nz = J.getVars(zone, ['nx', 'ny', 'nz'])
    if nx[np.argmin(x)] > 0:
        nx *= -1
        ny *= -1
        nz *= -1

def addViscosityMolecularIfAbsent(t):
    '''
    Add field ``{centers:ViscosityMolecular}`` using *ReferenceState* and primitive
    fields.

    Parameters
    ----------

        t : PyTree
            PyTree with required fields. It is modified.
    '''
    if C.isNamePresent(t,'centers:ViscosityMolecular') == -1:
        try:
            P._computeVariables(t, ['centers:ViscosityMolecular'] )
        except:
            pass

    if C.isNamePresent(t,'centers:ViscosityMolecular') == -1:
        if C.isNamePresent(t,'Temperature') == -1:
            try:
                C._initVars(t,'Temperature={Pressure}/(287.053*{Density})')
            except:
                Eqn = ('centers:Temperature={centers:Pressure}'
                                            '/(287.053*{centers:Density})')
                C._initVars(t,Eqn)

        S   = 110.4
        mus = 1.78938e-05
        Ts  = 288.15
        try:
            Eqn = ('ViscosityMolecular={mus}*({Temp}/{Ts})**1.5'
                   '*(({Ts} + {S})/({Temp} + {S}))').format(
                   mus=mus, Ts=Ts, Temp='{Temperature}',S=S)
            C._initVars(t,Eqn)
        except:
            Eqn = ('centers:ViscosityMolecular={mus}*({Temp}/{Ts})**1.5'
                   '*(({Ts} + {S})/({Temp} + {S}))').format(
                   mus=mus, Ts=Ts, Temp='{centers:Temperature}',S=S)
            C._initVars(t,Eqn)

def addPressureIfAbsent(t):
    '''
    Add field ``{centers:Pressure}`` using *ReferenceState* and primitive fields.

    Parameters
    ----------

        t : PyTree
            PyTree with required fields. It is modified.
    '''

    if C.isNamePresent(t,'centers:Pressure') == -1:
        try:
            P._computeVariables(t, ['centers:Pressure'] )
        except:
            pass

    if C.isNamePresent(t,'Pressure') == -1:
        try:
            P._computeVariables(t, ['Pressure'] )
        except:
            pass

#======================= turbomachinery =======================================#


def absolute2Relative(t, loc='centers', container=None, containerRelative=None):
    '''
    In a turbomachinery context, change the frame of reference of variables in
    the FlowSolution container **container** from absolute to relative.
    Families with ``'.Solver#Motion'`` node must be found to use the rotation
    speed for the change of FoR.

    Parameters
    ----------

        t : PyTree
            tree to modify

        loc : str
            location of variables, must be 'centers' or 'nodes'. Used to restore
            variables with Post.computeVariables from Cassiopee after the change
            of FoR.

        container : str
            Name of the FlowSolution container to change of frame of reference.
            By default, this is the default container at cell centers or nodes
            (depending on **loc**) of Cassiopee

        containerRelative : str
            Name of the new FlowSolution container with variables in the
            relative frame of reference. By default, this is **container** with
            the suffix ``'#Relative'``

    Returns
    -------

        t : PyTree
            modified tree

    '''
    import etc.transform as trf

    assert loc in ['centers', 'nodes'], 'loc must be centers or nodes'
    if not container:
        if loc == 'centers': container = I.__FlowSolutionCenters__
        else: container = I.__FlowSolutionNodes__

    conservatives = ['Density', 'MomentumX', 'MomentumY', 'MomentumZ',
        'EnergyStagnationDensity']
    primTurbVars = ['TurbulentEnergyKinetic', 'TurbulentDissipationRate',
        'TurbulentDissipation', 'TurbulentSANuTilde', 'TurbulentLengthScale']
    consTurbVars = [var+'Density' for var in primTurbVars]
    turbVars = primTurbVars + consTurbVars
    noFoRDependentVariables = ['Pressure', 'Temperature', 'Entropy',
        'ViscosityMolecular', 'Viscosity_EddyMolecularRatio', 'ChannelHeight']
    var2keep = conservatives + turbVars + noFoRDependentVariables

    cassiopeeVariables = [
        'VelocityX', 'VelocityY', 'VelocityZ', 'VelocityMagnitude', 'Pressure',
        'Temperature', 'Enthalpy', 'Entropy', 'Mach', 'ViscosityMolecular',
        'PressureStagnation', 'TemperatureStagnation', 'PressureDynamic']

    I._adaptZoneNamesForSlash(t)

    if not containerRelative:
        containerRelative = container+'#Relative'

    var2compute = []

    for FS in I.getNodesFromNameAndType(t, container, 'FlowSolution_t'):
        copyOfFS = I.copyTree(FS)
        I.setName(copyOfFS, containerRelative)
        varInFS = []
        # Keep some variables
        for node in I.getNodesFromType(copyOfFS, 'DataArray_t'):
            var = I.getName(node)
            if var not in var2keep:
                if var not in var2compute and var in cassiopeeVariables:
                    var2compute.append(var)
                I.rmNode(copyOfFS, node)
            else:
                varInFS.append(var)
        if not all([(var in varInFS) for var in conservatives]):
            continue
        parent, pos = I.getParentOfNode(t, FS)
        I.addChild(parent, copyOfFS, pos=pos+1)

    for base in I.getNodesFromType(t, 'CGNSBase_t'):
        # Base per base, else trf.absolute2Relative does not found omega
        base = trf.absolute2Relative(base, containerRelative, frame='cartesian')

    # restore variables
    if loc == 'centers':
        oldContainer = I.__FlowSolutionCenters__
        I.__FlowSolutionCenters__ = containerRelative
        P._computeVariables(t, [loc+':'+var for var in var2compute])
        I.__FlowSolutionCenters__ = oldContainer
    else:
        oldContainer = I.__FlowSolutionNodes__
        I.__FlowSolutionNodes__ = containerRelative
        P._computeVariables(t, [loc+':'+var for var in var2compute])
        I.__FlowSolutionNodes__ = oldContainer

    return t


def _fitFields(donor, receiver, fields_names_to_fit=[], tol=1e-6):
    '''
    TODO doc
    '''
    donor_zones = I.getZones( donor )
    receiver_zones = I.getZones( receiver )
    number_of_donor_zones = len( donor_zones )

    for receiver_zone in receiver_zones:
        fields_receiver = J.getVars( receiver_zone, fields_names_to_fit )
        fields_receiver = [f.ravel(order='F') for f in fields_receiver]

        hook = C.createHook( receiver_zone, function='nodes')
        res = C.nearestNodes( hook, donor_zones )

        if number_of_donor_zones == 1: res = [res]

        for di, donor_zone in enumerate( donor_zones ):
            nodes, dist = res[di]
            close_enough_points = dist<tol
            nodes = nodes[dist<tol]
            if len(nodes) == 0: continue
            receiver_indexes = nodes - 1
            fields_donor = J.getVars( donor_zone, fields_names_to_fit )
            fields_donor = [f.ravel(order='F') for f in fields_donor]
            for fr, fd in zip(fields_receiver, fields_donor):
                fr[ receiver_indexes ] = fd[ close_enough_points ]

def computeIntegralLoads(t, torque_center=[0,0,0]):
    '''
    Compute the total integral forces and torques of a set of surfaces with 
    Pressure and SkinFriction fields

    Parameters
    ----------

        t : PyTree, Base, Zone or :py:class:`list` of zone
            surfaces contributing to the total forces and torques being computed

            .. note::
                surfaces contained in **t** must contain the following fields
                contained at centers: ``Pressure``, ``SkinFrictionX``,
                ``SkinFrictionY``, ``SkinFrictionZ``. It may also contain 
                normals ``nx``, ``ny``, ``nz``. Otherwise they are computed.

        torque_center : 3-float :py:class:`list` or :py:class:`tuple` or :py:class:`numpy`
            center for computation the torque contributions

    Returns
    -------

        loads : dict
            dictionary including ``ForceX``, ``ForceY``, ``ForceZ``,
            ``TorqueX``, ``TorqueY`` and ``TorqueZ`` keys with its associated 
            values (:py:class:`float`)
    '''

    tR = I.copyRef(t)
    _addNormalsIfAbsent(tR)

    # surfacic forces
    C._initVars(tR, 'centers:fx=-{centers:Pressure}*{centers:nx}+{centers:SkinFrictionX}')
    C._initVars(tR, 'centers:fy=-{centers:Pressure}*{centers:ny}+{centers:SkinFrictionY}')
    C._initVars(tR, 'centers:fz=-{centers:Pressure}*{centers:nz}+{centers:SkinFrictionZ}')

    # computation of total forces
    ForceX = -P.integ(tR,'centers:fx')[0]
    ForceY = -P.integ(tR,'centers:fy')[0]
    ForceZ = -P.integ(tR,'centers:fz')[0]

    # computation of total torques
    TorqueX, TorqueY, TorqueZ = P.integMoment(tR, center=torque_center,
                                vector=['centers:fx','centers:fy','centers:fz'])
    
    TorqueX *= -1
    TorqueY *= -1
    TorqueZ *= -1


    loads = dict(ForceX=ForceX,ForceY=ForceY,ForceZ=ForceZ,
                 TorqueX=TorqueX,TorqueY=TorqueY,TorqueZ=TorqueZ)
    
    return loads

def computeSectionalLoads(t, start_point, end_point, distribution,
        torque_center=[0,0,0]):
    '''
    Compute the sectional loads (spanwise distributions) along a direction 
    from a set of surfaces

    Parameters
    ----------
    
        t : PyTree, Base, Zone or :py:class:`list` of Zone
            surfaces from which sectional loads are to be computed

            .. note::
                surfaces contained in **t** must contain the following fields
                 (preferrably at centers): ``Pressure``, ``SkinFrictionX``,
                ``SkinFrictionY``, ``SkinFrictionZ``. It may also contain 
                normals ``nx``, ``ny``, ``nz``. Otherwise they are computed.

        start_point : 3-float :py:class:`list` or :py:class:`tuple` or :py:class:`numpy`
            :math:`(x,y,z)` coordinates of the starting point from which 
            sectional loads are to be computed

        end_point : 3-float :py:class:`list` or :py:class:`tuple` or :py:class:`numpy`
            :math:`(x,y,z)` coordinates of the end point up to which 
            sectional loads are to be computed

        distribution : 1D :py:class:`float` list or :py:class:`numpy`
            dimensionless coordinate (from *start_point* to *end_point*) used 
            for discretizing the sectional loads. This must be :math:`\in [0,1]`.

            .. hint:: for example 

                >>> distribution = np.linspace(0,1,200)

        torque_center : 3-float :py:class:`list` or :py:class:`tuple` or :py:class:`numpy`
            center for computation the torque contributions

    Returns
    -------

        SectionalLoads : dict
            dictionary including ``SectionalForceX``, ``SectionalForceY``, ``SectionalForceZ``,
            ``SectionalTorqueX``, ``SectionalTorqueY``, ``SectionalTorqueZ``
            and ``SectionalSpan`` keys with its associated 1D array values (numpy of :py:class:`float`)

    '''

    tR = I.copyRef(t)

    start_point = np.array(start_point)
    end_point = np.array(end_point)
    span_direction = end_point - start_point
    total_scan_span = np.linalg.norm(span_direction)
    span_direction /= total_scan_span
    Eqn = W.computePlaneEquation(start_point, span_direction)
    C._initVars(tR, 'Span='+Eqn)
    
    SectionalForceX  = []
    SectionalForceY  = []
    SectionalForceZ  = []
    SectionalTorqueX = []
    SectionalTorqueY = []
    SectionalTorqueZ = []
    SectionalSpan    = []

    for d in distribution:
        span = d*total_scan_span
        section = P.isoSurfMC(tR, 'Span', span)
        if not section: continue
        C._normalize(section,['nx','ny','nz'])
        C._initVars(section, 'fx=-{Pressure}*{nx}+{SkinFrictionX}')
        C._initVars(section, 'fy=-{Pressure}*{ny}+{SkinFrictionY}')
        C._initVars(section, 'fz=-{Pressure}*{nz}+{SkinFrictionZ}')

        # computation of sectional forces
        SectionalForceX += [ -P.integ(section,'fx')[0] ]
        SectionalForceY += [ -P.integ(section,'fy')[0] ]
        SectionalForceZ += [ -P.integ(section,'fz')[0] ]

        # computation of sectional torques
        STorqueX, STorqueY, STorqueZ = P.integMoment(section, center=torque_center,
                                    vector=['fx','fy','fz'])
        SectionalTorqueX += [ -STorqueX ]
        SectionalTorqueY += [ -STorqueY ]
        SectionalTorqueZ += [ -STorqueZ ]
        SectionalSpan    += [ span ]

    SectionalLoads = dict(SectionalForceX=SectionalForceX,
                          SectionalForceY=SectionalForceY,
                          SectionalForceZ=SectionalForceZ,
                          SectionalTorqueX=SectionalTorqueX,
                          SectionalTorqueY=SectionalTorqueY,
                          SectionalTorqueZ=SectionalTorqueZ,
                          SectionalSpan=SectionalSpan)

    return SectionalLoads


def _addNormalsIfAbsent(t):
    for z in I.getZones(t):
        CentersFields = C.getVarNames(z, excludeXYZ=True, loc='centers')[0]
        if not 'centers:nx' in CentersFields:
            G._getNormalMap(z)
            FlowSol = I.getNodeFromName1(z, I.__FlowSolutionCenters__)
            for i in 'xyz':
                I.rmNode(FlowSol,'n'+i)
                I.renameNode(FlowSol,'s'+i,'n'+i)
        C._normalize(z,['centers:nx', 'centers:ny', 'centers:nz'])    
