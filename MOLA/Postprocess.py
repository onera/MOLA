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
MOLA - Postprocess.py

Postprocess routines

12/05/2021 - L. Bernardos - Creation from recycling
'''

import MOLA

if not MOLA.__ONLY_DOC__:
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

    from mpi4py import MPI
    comm   = MPI.COMM_WORLD
    rank   = comm.Get_rank()


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
    I._rmNodesByName(aux_grid, I.__FlowSolutionNodes__)
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
    from .Coprocess import extractSurfaces
    return extractSurfaces(t, [dict(type='AllBCWall')])


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

    if container != I.__FlowSolutionNodes__:
        I._rmNodesByName(tR, I.__FlowSolutionNodes__)
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
        I.renameNode(tR, container, I.__FlowSolutionNodes__)


    AllFlowSolutionNodes = I.getNodesFromType(tR, 'FlowSolution_t')
    for FlowSolutionNode in AllFlowSolutionNodes:
        if FlowSolutionNode[0] != I.__FlowSolutionNodes__:
            I.rmNode(tR, FlowSolutionNode)


    from .Coprocess import extractSurfaces

    walls_tree = extractSurfaces(t, [dict(type='AllBCWall')])
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


def computeAndAddSpanToSurface(t, start_point, end_point):
    '''
    Computes the span of a surface based on 2 points and adds it as variable in the provided tree.

    Parameters
    ----------

        t : PyTree, Base, Zone or :py:class:`list` of zone
        
        start_point : 3-float :py:class:`list` or :py:class:`tuple` or :py:class:`numpy`
            :math:`(x,y,z)` coordinates of the starting point from which 
            sectional loads are to be computed. Must be provided for SpanBased slicing.

        end_point : 3-float :py:class:`list` or :py:class:`tuple` or :py:class:`numpy`
            :math:`(x,y,z)` coordinates of the end point up to which the span must be computed

    Returns
    -------

        -

    '''

    start_point = np.array(start_point)
    end_point = np.array(end_point)
    span_direction = end_point - start_point
    total_scan_span = np.linalg.norm(span_direction)
    span_direction /= total_scan_span
    print('Scan span',total_scan_span)
    Eqn = W.computePlaneEquation(start_point, span_direction)
    C._initVars(t, 'Span='+Eqn)


def computeIntegralLoads(t, torque_center=[0,0,0],reference_pressure=0.):
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
        
        reference_pressure : float
            Reference pressure. Put ambiant pressure as a reference for integration over a surface that is not closed (such as blades).

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
    C._initVars(tR, 'centers:fx=-({centers:Pressure}-%s)*{centers:nx}+{centers:SkinFrictionX}'%(reference_pressure))
    C._initVars(tR, 'centers:fy=-({centers:Pressure}-%s)*{centers:ny}+{centers:SkinFrictionY}'%(reference_pressure))
    C._initVars(tR, 'centers:fz=-({centers:Pressure}-%s)*{centers:nz}+{centers:SkinFrictionZ}'%(reference_pressure))

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

def computeSectionalLoads(surface, distribution = np.linspace(0,1,100), slicing_options=dict(slicing_method='SpanBased',custom_variable=None), geometrical_parameters=dict(start_point=None,end_point=None, axis_direction=None),
        torque_center=[0,0,0], reference_pressure=0.):
    '''
    Compute the sectional loads (spanwise distributions) along a direction 
    from a set of surfaces

    Parameters
    ----------
    
        surface : PyTree, Base, Zone or :py:class:`list` of Zone
            surfaces from which sectional loads are to be computed

            .. note::
                surfaces contained in **t** must contain the following fields
                 (preferrably at centers): ``Pressure``, ``SkinFrictionX``,
                ``SkinFrictionY``, ``SkinFrictionZ``. It may also contain 
                normals ``nx``, ``ny``, ``nz``. Otherwise they are computed.

        slicing_options: dict with two keys
            slicing_method : str
                Options:
                - SpanBased: Computes the span based on 2 points (see below) provided by the user. Each section corresponds to an isoSurface of the Span variable.
                - AbscissaBased: computes the abscissa based on the distance d to the axis provided by the user. Each section corresponds to an isoSurface of the Abscissa variable.
                  Abscissa = (d-dmin)/(dmax-dmin)
                - Custom: uses the 'custom_variable' parameter provided by the user as the reference variable to perform the isoSurface for each section.
            
            custom_variable : str 
                Name of the variable used to perform slices.
                Examples: 'CoordinateX', 'CoordinateY', 'CoordinateZ','Radius'
                **Must be provided for Custom slicing.**

        geometrical_parameters: dict with three keys
            start_point : 3-float :py:class:`list` or :py:class:`tuple` or :py:class:`numpy`
                :math:`(x,y,z)` coordinates of the starting point from which 
                sectional loads are to be computed. Must be provided for SpanBased slicing.
                    **Must be provided for SpanBased slicing.**
                    **Must be provided for AbscissaBased slicing.**

            end_point : 3-float :py:class:`list` or :py:class:`tuple` or :py:class:`numpy`
                :math:`(x,y,z)` coordinates of the end point up to which 
                sectional loads are to be computed. 
                    **Must be provided for SpanBased slicing.**
            
            axis_direction : 3-float :py:class:`list` or :py:class:`tuple` or :py:class:`numpy`
                :math:`(x,y,z)` direction of the reference axis along which 
                sectional loads are to be computed.
                    **Must be provided for AbscissaBased slicing.**
        
        
        distribution : 1D :py:class:`float` list or :py:class:`numpy`
            dimensionless coordinate (from *start_point* to *end_point*) used 
            for discretizing the sectional loads. This must be :math:`\in [0,1]`.

            .. hint:: for example 

                >>> distribution = np.linspace(0,1,200)

            .. note:: for slicing_method = 'custom', this function automatically recomputes the span :math:`\in [0,1]` 
            to perform the slices. The span is based on the 'custom_variable' andi s computed as follows : 
            (var-varMin)/(varMax-varMin)

        torque_center : 3-float :py:class:`list` or :py:class:`tuple` or :py:class:`numpy`
            center for computation the torque contributions

        ReferencePressure : float
            Reference pressure. Put ambiant pressure as a reference for integration over a surface that is not closed (such as blades).

    Returns
    -------

        loads : Zone_t containing a FlowSolution node with the following variables: 
        ``SectionalForceX``, ``SectionalForceY``, ``SectionalForceZ``,
            ``SectionalTorqueX``, ``SectionalTorqueY``, ``SectionalTorqueZ``
            and ``SectionalSpan``.


    '''

    def Abscissa(d): return (d-dmin)/(dmax-dmin)
  
    
    if slicing_options['slicing_method'] == 'SpanBased':
        if geometrical_parameters['start_point'] == None or geometrical_parameters['end_point'] == None:
            ERRMSG = 'Span based sectional load computation requires both start_point and end_point as input parameters'
            raise ValueError(ERRMSG)
        else:
            computeAndAddSpanToSurface(surface, np.array(geometrical_parameters['start_point']), np.array(geometrical_parameters['end_point']))
            dmin = C.getMinValue(surface, 'Span')
            dmax = C.getMaxValue(surface, 'Span')            
            surface = C.initVars(surface,'Span2', Abscissa, ['Span'])
            slicing_var = 'Span2'

    elif slicing_options['slicing_method'] == 'AbscissaBased':
        if geometrical_parameters['start_point'] == None or geometrical_parameters['axis_direction']== None:
            ERRMSG = 'Abscissa based sectional load computation requires both start_point and axis_direction as input parameters'
            raise ValueError(ERRMSG)
        else:
            W.addDistanceRespectToLine(surface, np.array(geometrical_parameters['start_point']), np.array(geometrical_parameters['axis_direction']), FieldNameToAdd='Distance2Axis')
            
            dmin = C.getMinValue(surface, 'Distance2Axis')
            dmax = C.getMaxValue(surface, 'Distance2Axis')
            surface = C.initVars(surface,'Abscissa', Abscissa, ['Distance2Axis']) 
            slicing_var = 'Abscissa'

    
    elif slicing_options['slicing_method'] == 'Custom':
            if slicing_options['custom_variable'] == None:
                ERRMSG = 'The user needs to provide a custom_variable value when performing custom variable based sectional load computation.'
                raise ValueError(ERRMSG)
            else:
            
                dmin = C.getMinValue(surface, slicing_options['custom_variable'])
                dmax = C.getMaxValue(surface, slicing_options['custom_variable'])
                surface = C.initVars(surface,'Span', Abscissa, [slicing_options['custom_variable']]) 
                slicing_var = 'Span'
               
        
    SectionalForceX      = []
    SectionalForceY      = []
    SectionalForceZ      = []
    SectionalTorqueX     = []
    SectionalTorqueY     = []
    SectionalTorqueZ     = []
    SectionalSpan        = []

    sectionalLoads = I.newCGNSBase('SectionalLoads', cellDim=1, physDim=3, parent=None)
    for d in distribution:

        section = P.isoSurfMC(surface, slicing_var, d)
        
        C._normalize(section,['nx','ny','nz'])
        C._initVars(section, 'fx=-({Pressure}-%s)*{nx}+{SkinFrictionX}'%(reference_pressure))
        C._initVars(section, 'fy=-({Pressure}-%s)*{ny}+{SkinFrictionY}'%(reference_pressure))
        C._initVars(section, 'fz=-({Pressure}-%s)*{nz}+{SkinFrictionZ}'%(reference_pressure))

        # computation of sectional forces
        SectionalForceX += [ -P.integ(section,'fx')[0] ]
        SectionalForceY += [ -P.integ(section,'fy')[0] ]
        SectionalForceZ += [ -P.integ(section,'fz')[0] ]

        # computation of sectional torques
        STorqueX, STorqueY, STorqueZ = P.integMoment(section, center=torque_center,
                                    vector=['fx','fy','fz'])

        SectionalTorqueX     += [ -STorqueX ]
        SectionalTorqueY     += [ -STorqueY ]
        SectionalTorqueZ     += [ -STorqueZ ]

        # if slicing_options['slicing_method'] != 'Custom':
        SectionalSpan        += [ d ]
        # else:
        #     SectionalSpan        += [ value ]
    
    sloads = dict(SectionalForceX=np.array(SectionalForceX),SectionalForceY=np.array(SectionalForceY),SectionalForceZ=np.array(SectionalForceZ),
                 SectionalTorqueX=np.array(SectionalTorqueX),SectionalTorqueY=np.array(SectionalTorqueY),SectionalTorqueZ=np.array(SectionalTorqueZ),SectionalSpan=np.array(SectionalSpan))
     
 
    varValues = []
    varNames = []

    for key in sloads.keys():
        varNames.append(key)
        varValues.append(sloads[key])
        
    sectionalLoads = J.createZone('SectionalLoads',Arrays=varValues,Vars=varNames) 

    return sectionalLoads


def _addNormalsIfAbsent(t):
    for z in I.getZones(t):
        CentersFields = C.getVarNames(z, excludeXYZ=True, loc='centers')[0]
        if not 'centers:nx' in CentersFields:
            G._getNormalMap(z)
            FlowSol = I.getNodeFromName1(z, I.__FlowSolutionCenters__)
            for i in 'xyz':
                I.rmNode(FlowSol,'n'+i)
                I._renameNode(FlowSol,'s'+i,'n'+i)
        C._normalize(z,['centers:nx', 'centers:ny', 'centers:nz'])    


def getSubIterations():

    import glob

    setup = J.load_source('setup', 'setup.py')
   
    time_algo = setup.elsAkeysNumerics['time_algo']
    if time_algo not in ['gear', 'dts']:
        print(J.WARN + f'Cannot monitor sub-iterations for time_algo={time_algo}' + J.ENDC)
        print(J.WARN + f'  (time_algo must be gear or dts)' + J.ENDC)
        return None, None, None

    cwd = os.getcwd()

    # Find all elsA_MPI_* log files
    elsA_log_files = glob.glob('elsA_MPI_*')
    if len(elsA_log_files) == 0:
        elsA_log_files = glob.glob('LOGS/elsA_MPI_*')

    # Find the last modified file
    elsA_log_file = None
    maxtime = 0.
    for filename in elsA_log_files:
        full_path = os.path.join(cwd, filename)
        mtime = os.stat(full_path).st_mtime
        if mtime > maxtime:
            maxtime = mtime
            elsA_log_file = filename
    print(f'Last modified elsA log file: {elsA_log_file}')

    # Read this file
    with open(elsA_log_file, 'r') as fi:
        lines = fi.readlines()

    iterationList     = []
    residualList      = []
    dualIterationList = []

    for line in lines:

        if 'iteration no' in line:
            iteration = int(line.split('iteration no')[-1])
            iterationList.append(iteration)

            residualList.append([])
            dualIterationList.append([])
        
        if 'DualIter' in line:
            # print(line.split('residual :')[-1].split('DualIter :')[0])
            residual = float(line.split('residual :')[-1].split('DualIter :')[0])
            dualIteration = int(line.split('DualIter :')[-1].split('(Outer Iter:')[0])

            residualList[-1].append(residual)
            dualIterationList[-1].append(dualIteration)

    iterationArray = np.array(iterationList)
    dualIterationArray = np.array(dualIterationList)
    residualArray  = np.array(residualList)

    return iterationArray, dualIterationArray, residualArray

def extractBC(t, Family=None, Name=None, Type=None):
    '''
    This is a multi-container wrapper of Cassiopee C.extractBC* functions, 
    as requested in https://elsa.onera.fr/issues/10641. 

    Parameters
    ----------

        t : PyTree
            input tree where surfaces will be extracted

        Family : str
            (optional) family name of the BC to be extracted

        Name : str
            (optional) name of the BC to be extracted

        Type : str
            (optional) type of the BC to be extracted

    Returns
    -------

        surfaces : :py:class:`list` of Zone_t
            list of surfaces (zones) with multi-containers (including *BCData_t* 
            transformed into *FlowSolution_t* nodes)    
    '''
    # HACK https://elsa.onera.fr/issues/10641
    args = [Family, Name, Type]
    if args.count(None) != len(args)-1:
        raise AttributeError('must provide only one of: Name, Type or Family')

    if Family is not None:
        if Family.startswith('FamilySpecified:'):
            Type = Family
        else:
            Type = 'FamilySpecified:'+Family
        Name = None

    elif Name is not None and Name.startswith('FamilySpecified:'):
        Type = Name
        Name = None


    if Name:
        extractBCarg = Name
        extractBCfun = C.extractBCOfName
    else:
        extractBCarg = Type
        extractBCfun = C.extractBCOfType
    
    t = mergeContainers(t, FlowSolutionVertexName=I.__FlowSolutionNodes__,
                           FlowSolutionCellCenterName=I.__FlowSolutionCenters__)

    bases_children_except_zones = []
    for base in I.getBases(t):
        for n in base[2]:
            if n[3] != 'Zone_t': 
                bases_children_except_zones.append( n )

    bcs = []
    for zone in I.getZones(t):
        extracted_bcs = I.getZones( extractBCfun(zone, extractBCarg))
        if not extracted_bcs: continue
        I._adaptZoneNamesForSlash(extracted_bcs)
        for surf in extracted_bcs:
            _mergeBCtagContainerWithFlowSolutionTagContainer(zone, surf)
            bc_multi_container = recoverContainers(surf)
            bcs += [ bc_multi_container ]
        reshapeFieldsForStructuredGrid(bcs)

    t_merged = C.newPyTree(['Base',bcs])
    base = I.getBases(t_merged)[0]
    base[2].extend( bases_children_except_zones )
    zones = I.getZones(t_merged)

    return zones

def reshapeFieldsForStructuredGrid(t):
    for zone in I.getZones(t):
        topo, Ni, Nj, Nk, dim = I.getZoneDim(zone)
        if topo != 'Structured' or dim==1: continue
        for fs in I.getNodesFromType1(zone,'FlowSolution_t'):
            loc = _getFlowSolutionLocation(fs)
            for n in I.getNodesFromType1(fs,'DataArray_t'):
                if len(n[1].shape) != dim:
                    if dim == 2:
                        if loc == 'Vertex':
                            n[1] = n[1].reshape((Ni,Nj))
                        elif loc == 'CellCenter':
                            n[1] = n[1].reshape((Ni-1,Nj-1))
                        else:
                            raise NotImplementedError(f'loc must be "Vertex" or "CellCenter", but got: {loc}')
                    elif dim == 3:
                        if loc == 'Vertex':
                            n[1] = n[1].reshape((Ni,Nj,Nk))
                        elif loc == 'CellCenter':
                            n[1] = n[1].reshape((Ni-1,Nj-1,Nk-1))
                        else:
                            raise NotImplementedError(f'loc must be "Vertex" or "CellCenter", but got: {loc}')


def extractBC_old(t, Name=None, Type=None):
    if Name is not None and Type is not None:
        raise AttributeError('must provide either Name or Type, but not both.')
    
    t = I.copyRef(t)

    bases_children_except_zones = []
    for base in I.getBases(t):
        for n in base[2]:
            if n[3] != 'Zone_t': 
                bases_children_except_zones.append( n )

    # information must be in BCDataSet_t containers
    I._rmNodesByType(t, 'FlowSolution_t')
    BCDataSetNames = set()
    for zone in I.getZones(t):
        zbc = I.getNodeFromType1(zone, 'ZoneBC_t')
        for bc in zbc[2]:
            for bcds in I.getNodesFromType1(bc,'BCDataSet_t'):
                BCDataSetNames.add( bcds[0] )
 
    if len(BCDataSetNames) == 0:
        # No fields extracted, only the surface geometry
        if Name:
            zones_merged = I.getZones( C.extractBCOfName(t, Name, extrapFlow=False) )
        else:
            zones_merged = I.getZones( C.extractBCOfType(t, Type, extrapFlow=False) )
        I._adaptZoneNamesForSlash(zones_merged)

    else:
        zones_merged = []
        for BCDataSetName in BCDataSetNames:
            tR = I.copyRef(t)
            for zone in I.getZones(tR):
                zbc = I.getNodeFromType1(zone, 'ZoneBC_t')
                LocalBCDataSetNames = set()
                for bc in zbc[2]:
                    for bcds in I.getNodesFromType1(bc,'BCDataSet_t'):
                        LocalBCDataSetNames.add( bcds[0] )
                for bc in zbc[2]:
                    for LocalBCDataSetName in LocalBCDataSetNames:
                        if LocalBCDataSetName != BCDataSetName:
                            I._rmNodesByName1(bc, BCDataSetName)
            if Name:
                zones = I.getZones( C.extractBCOfName(tR, Name, extrapFlow=False) )
            else:
                zones = I.getZones( C.extractBCOfType(tR, Type, extrapFlow=False) )
            
            I._adaptZoneNamesForSlash(zones)

            for z in zones:
                fs = I.getNodeFromType1(z, 'FlowSolution_t')
                if fs: 
                    fs[0] = BCDataSetName
                    
                if zones_merged:
                    if not fs: continue
                    for zm in zones_merged:
                        containers_names = [f[0] for f in I.getNodesFromType1(zm, 'FlowSolution_t')]
                        if z[0] == zm[0]:                        
                            if fs[0] not in containers_names: 
                                zm[2] += [ fs ]
                            break
                else:
                    zones_merged = zones

                fs_nodes = I.getNodesFromType1(z, 'FlowSolution_t')
                if len(fs_nodes) > 1:
                    I.printTree(z, 'debug_extractBC_z.txt')
                    raise ValueError('unexpected number of fs_nodes. Check debug_extractBC_z.txt')
    
    t_merged = C.newPyTree(['Base',zones_merged])
    base = I.getBases(t_merged)[0]
    base[2].extend( bases_children_except_zones )
    zones = I.getZones(t_merged)
        
    return zones


def isoSurface(t, fieldname=None, value=None, container='FlowSolution#Init'):
    '''
    This is a multi-container wrapper of Cassiopee Post.isoSurfMC function, as
    requested in https://elsa.onera.fr/issues/11221.

    .. attention::
        all zones contained in **t** must have the same containers. If this is
        not your case, you may want to first select the zones with same containers
        before using this function (see :py:func:`MOLA.InternalShortcuts.selectZones`)

    Parameters
    ----------

        t : PyTree
            input tree where iso-surface will be performed

        fieldname : str
            name of the field used for making the iso-surface. It can be the
            coordinates names such as ``'CoordinateX'``, ``'CoordinateY'`` or 
            ``'CoordinateZ'`` (in such cases, parameter **container** is ignored)

        value : float
            value used for computing the iso-surface of field **fieldname**

        container : str
            name of the *FlowSolution_t* CGNS container where the field
            **fieldname** is contained. This parameter is ignored if **fieldname**
            is a coordinate.

    Returns
    -------

        surfaces : :py:class:`list` of Zone_t
            list of zones with fields arranged at multiple containers following
            the original data structure.

    '''
    # HACK https://elsa.onera.fr/issues/11221
    bases_children_except_zones = []
    for base in I.getBases(t):
        for n in base[2]:
            if n[3] != 'Zone_t': 
                bases_children_except_zones.append( n )
    if not I.getNodeFromType3(t,'Zone_t'): return
    tPrev = I.copyRef(t)
    t = mergeContainers(t, FlowSolutionVertexName=I.__FlowSolutionNodes__,
                           FlowSolutionCellCenterName=I.__FlowSolutionCenters__)

    isosurfs = []
    for zone in I.getZones(t):

        # NOTE slicing will provoque all containers to be located at Vertex
        tags_containers = I.getNodeFromName1(zone, 'tags_containers')
        tags_containers_dict = J.get(zone, 'tags_containers')

        containers_names = I.getNodeFromName1(tags_containers, 'containers_names')
        if fieldname not in ['CoordinateX', 'CoordinateY', 'CoordinateZ']:
            fieldnameWithTag = None
            for cn in containers_names[2]:
                container_name = I.getValue(cn)
                tag = cn[0]
                if container_name == container:
                    fieldnameWithTag = fieldname + tag
                    break
            if fieldnameWithTag is None:
                C.convertPyTree2File(tPrev,f'debug_tPrev_{rank}.cgns')
                C.convertPyTree2File(zone,f'debug_zone_{rank}.cgns')
                raise ValueError(f'could not find tag <-> container "{container}" correspondance')
        else:
            fieldnameWithTag = fieldname

        for n in containers_names[2]:
            tag = n[0]
            loc = tags_containers_dict['locations'][tag]
            if loc == 'CellCenter':
                cont_name = I.getValue(n)
                I.setValue(n,cont_name+'V') # https://gitlab.onera.net/numerics/mola/-/issues/146#note_20639

        for n in I.getNodeFromName1(tags_containers, 'locations')[2]:
            if I.getValue(n) == 'CellCenter':
                I.setValue(n,'Vertex')

        # HACK https://elsa.onera.fr/issues/11255
        if I.getZoneType(zone) == 2: # unstructured zone
            if I.getNodeFromName1(zone,I.__FlowSolutionCenters__):
                fieldnames = C.getVarNames(zone, excludeXYZ=True, loc='centers')[0]
                for f in fieldnames:
                    C._center2Node__(zone,f,0)
                I._rmNodesByName1(zone,I.__FlowSolutionCenters__)

            # HACK https://gitlab.onera.net/numerics/mola/-/issues/111
            # HACK https://elsa.onera.fr/issues/10997#note-6
            zone = T.breakElements(zone)

        surfs = P.isoSurfMC(zone, fieldnameWithTag, value)
        for surf in I.getZones(surfs):
            surf[2] += [ tags_containers ]
            isosurfs += [ recoverContainers(surf) ]

    t_merged = C.newPyTree(['Base', isosurfs])
    base = I.getBases(t_merged)[0]
    base[2].extend( bases_children_except_zones )
    surfs = I.getZones(t_merged)
    
    return surfs

def convertToTetra(t):
    tR = I.copyRef(t)
    fs_per_zone = []
    for z in I.getZones(tR):
        fs_per_zone += [ I.getNodesFromType3(t, 'FlowSolution_t') ]
        I._rmNodesFromType1(z, 'FlowSolution_t')
    tetra = C.convertArray2Tetra(tR)

    for z, fs_at_zone in zip( I.getZones(tetra), fs_per_zone ):
        z[2].extend( fs_at_zone )

    return tetra

def mergeContainers(t, FlowSolutionVertexName='FlowSolution',
        FlowSolutionCellCenterName='FlowSolution#Centers',
        BCDataSetFaceCenterName='BCDataSet'):
    '''
    Merge all *FlowSolution_t* containers into a single one (one at Vertex, another
    at CellCenter), adding a numerical tag suffix to flowfield name for easy
    identification (e.g. ``<FlowfieldName>.<NumericTag>``). 

    .. danger:: when adding the numeric tag, the number of characters of the 
        resulting CGNS node can be higher than 32. This is not a problem for 
        in-memory tree, but most CGNS writters/readers cannot support names longer
        than 32 characters, which may result in loose of data if the flowfields 
        names with tags are truncated when saving or reading a CGNS file

    Also, this function merges all *BCDataSet_t/BCData_t* containers into a single
    one named ``BCDataSet/NeumannData`` located at FaceCenter. 

    .. caution:: input ``BCDataSet_t/BCData_t`` must be all contained in **FaceCenter**

    New ``UserDefined_t`` CGNS nodes are added to ``Zone_t`` nodes and ``BC_t`` nodes,
    named ``multi_containers``, which contains all the required information for
    making the inverse operation (see :py:func:`recoverContainers`) for recovering
    the original structure of the tree

    Parameters
    ----------

        t : PyTree, Base, Zone or list of Zone
            Input containing zones where containers are to be merged

        FlowSolutionVertexName : str
            the name of the resulting Vertex FlowSolution CGNS node

        FlowSolutionCellCenterName : str
            the name of the resulting CellCenter FlowSolution CGNS node

        BCDataSetFaceCenterName : str
            the name of the resulting BCDataSet CGNS node

    Returns
    -------

        tR : PyTree, Base, Zone or list of Zone
            copy as reference of **t**, but with merged containers 
    '''
    # HACK https://elsa.onera.fr/issues/11221
    # HACK https://elsa.onera.fr/issues/10641


    tR = I.copyRef(t)
    for zone in I.getZones(tR):
        _mergeFlowSolutions(zone, FlowSolutionVertexName, FlowSolutionCellCenterName)
        _mergeBCData(zone, BCDataSetFaceCenterName)
    return tR

def _mergeFlowSolutions(zone, FlowSolutionVertexName='FlowSolution',
        FlowSolutionCellCenterName='FlowSolution#Centers'):
    '''
    Merge all FlowSolution_t into one at Vertex and one at CellCenter
    
    Consider using higher-level function mergeContainers
    '''
    if zone[3] != 'Zone_t': return AttributeError('first argument must be a zone')
    monoFlowSolutionNames = dict(Vertex=FlowSolutionVertexName,
                                 CellCenter=FlowSolutionCellCenterName)
    fields_names = dict()
    containers_names = dict()
    nodes = dict()
    locations = dict()
    FlowSolutions = I.getNodesFromType1(zone,'FlowSolution_t')
    for fs in FlowSolutions:
        if not I.getNodesFromType1(fs, 'DataArray_t'):
            try:
                FlowSolutions.pop(fs)
            except:
                pass
    if not FlowSolutions: return
    J.sortNodesByName(FlowSolutions)
    for i, fs in enumerate(FlowSolutions):
        if not I.getNodesFromType1(fs, 'DataArray_t'): continue
        loc = _getFlowSolutionLocation(fs)
        tag = '%d'%i
        locations[tag] = loc
        containers_names[tag] = fs[0]
        fields = I.getNodesFromType1(fs,'DataArray_t')
        for f in fields:
            f[0] += tag
            if tag in fields_names:
                fields_names[tag] += [f[0]]
                nodes[tag] += [f]
            else:
                fields_names[tag]  = [f[0]]
                nodes[tag]  = [f]
    
    prev_zone = I.copyRef(zone)
    I._rmNodesByType1(zone,'FlowSolution_t')
    for tag, loc in locations.items():
        try:
            fields_nodes = nodes[tag]
        except KeyError as e:
            C.convertPyTree2File(prev_zone,f'debug_{zone[0]}.cgns')
            raise e

        if not fields_nodes: continue
        fs = I.getNodeFromName1(zone, monoFlowSolutionNames[loc])
        if not fs:
            fs = I.createUniqueChild(zone, monoFlowSolutionNames[loc],
                                        'FlowSolution_t', children=fields_nodes)
            I.createUniqueChild(fs,'GridLocation','GridLocation_t', value=loc,pos=0)
        else:
            fs[2] += fields_nodes

    J.set(zone, 'tags_containers', fields_names=fields_names,
                    containers_names=containers_names, locations=locations)

def _mergeBCData(zone, BCDataSetFaceCenterName='BCDataSet',
                       BCDataFaceCenterName='NeumannData'):
    '''
    Merge all BCData_t into one at Vertex and one at CellCenter
    
    Consider using higher-level function mergeContainers
    '''
    if zone[3] != 'Zone_t': return AttributeError('first argument must be a zone')
    monoBCDataSetNames = dict(FaceCenter=BCDataSetFaceCenterName)
    zbc = I.getNodeFromName1(zone,'ZoneBC')
    if not zbc: return
    BCs = I.getNodesFromType(zbc,'BC_t')
    if not BCs: return
    J.sortNodesByName(BCs)
    tags ='ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    for bc in BCs:
        nb = -1
        fields_names = dict()
        containers_names = dict()
        nodes = dict()
        locations = dict()
        BCDataSets = I.getNodesFromType1(bc,'BCDataSet_t')
        if not BCDataSets: continue
        J.sortNodesByName(BCDataSets)
        for bcds in BCDataSets:
            loc = _getBCDataSetLocation(bcds)
            if loc != 'FaceCenter':
                path = '/'.join([zone[0],zbc[0],bc[0],bcds[0]])
                raise NotImplementedError(f'BCDataSet {path} must be located at FaceCenter, got {loc} instead')

            BCDatas = I.getNodesFromType1(bcds,'BCData_t')
            if not BCDatas: continue
            J.sortNodesByName(BCDatas)
            for bcd in BCDatas:
                nb += 1
                tag = tags[nb]
                locations[tag] = loc
                containers_names[tag] = bcds[0]+'/'+bcd[0]
                fields = I.getNodesFromType1(bcd,'DataArray_t')
                for f in fields:
                    f[0] += tag
                    if tag in fields_names:
                        fields_names[tag] += [f[0]]
                        nodes[tag] += [f]
                    else:
                        fields_names[tag]  = [f[0]]
                        nodes[tag]  = [f]

        I._rmNodesByType1(bc,'BCDataSet_t')
        for tag, loc in locations.items():
            fields_nodes = nodes[tag]
            if not fields_nodes: continue
            bcds = I.getNodeFromName1(bc, monoBCDataSetNames[loc])
            if not bcds:
                bcds = I.createUniqueChild(bc, monoBCDataSetNames[loc],
                                            'BCDataSet_t')
                I.createUniqueChild(bcds,'GridLocation','GridLocation_t', value=loc)
                I.createUniqueChild(bcds,BCDataFaceCenterName,'BCData_t',
                                         children=fields_nodes)
            else:
                bcd = I.getNodeFromName(bcds, BCDataFaceCenterName)
                bcd[2] += fields_nodes

        J.set(bc, 'tags_containers', fields_names=fields_names,
                   containers_names=containers_names, locations=locations)


def recoverContainers(t):
    tR = I.copyRef(t)
    for zone in I.getZones(tR):
        _recoverFlowSolutions(zone)
        _recoverBCData(zone)

    return tR


def _getFlowSolutionLocation(FlowSolution_n):
    GridLocation_n = I.getNodeFromType1(FlowSolution_n,'GridLocation_t')
    if not GridLocation_n: return 'Vertex'
    return I.getValue(GridLocation_n)

def _getBCDataSetLocation(BCDataSet_n):
    GridLocationNodes = I.getNodesFromType(BCDataSet_n,'GridLocation_t')
    if not GridLocationNodes: return 'FaceCenter'
    if len(GridLocationNodes) == 1: return I.getValue(GridLocationNodes[0])
    first_name = GridLocationNodes[0][0]
    if not all([n[0]==first_name for n in GridLocationNodes[1:]]):
        print('WARNING: multiple grid locations found')
        return 'Multiple'
    return first_name

def _recoverFlowSolutions(zone):
    if zone[3] != 'Zone_t': return AttributeError('first argument must be a zone')
    had_multi_containers = I.getNodeFromName1(zone,'tags_containers')
    if not had_multi_containers: return
    fields_containers = J.get(zone, 'tags_containers')
    fields_names = fields_containers['fields_names']
    containers_names = fields_containers['containers_names']
    locations = fields_containers['locations']
    nodes = dict()

    MergedNodes = dict()
    for fs in I.getNodesFromType1(zone, 'FlowSolution_t'):
        merged_location = _getFlowSolutionLocation(fs)
        fields = I.getNodesFromType1(fs,'DataArray_t')
        if not fields: continue
        if merged_location in MergedNodes:
            MergedNodes[merged_location] += fields
        else:
            MergedNodes[merged_location] = fields

    for tag, loc in locations.items():
        for node_name in fields_names[tag].split():
            node = I.getNodeFromName(MergedNodes[loc], node_name)
            if not node:
                J.save(zone,f'debug_zone_{rank}.cgns')
                raise ValueError(f'UNEXPECTED: could not find node {node_name}')
            if tag in nodes:
                nodes[tag] += [ node ]
            else:
                nodes[tag]  = [ node ]
    
    I._rmNodesByType1(zone, 'FlowSolution_t')
    for tag, fields in nodes.items():
        for f in fields: f[0] = f[0][:-len(tag)] # remove sufix
        fs = I.createUniqueChild(zone,containers_names[tag],'FlowSolution_t',
                                      children=fields)
        I.createUniqueChild(fs,'GridLocation','GridLocation_t',value=locations[tag], pos=0)
    I._rmNodesByName1(zone,'tags_containers')


def _recoverBCData(zone):
    if zone[3] != 'Zone_t': return AttributeError('first argument must be a zone')
    zbc = I.getNodeFromName1(zone,'ZoneBC')
    if not zbc: return
    for bc in I.getNodesFromType(zbc,'BC_t'):
        had_multi_containers = I.getNodeFromName1(bc,'tags_containers')
        if not had_multi_containers: return
        fields_containers = J.get(bc, 'tags_containers')
        fields_names = fields_containers['fields_names']
        containers_names = fields_containers['containers_names']
        locations = fields_containers['locations']
        nodes = dict()

        MergedNodes = dict()
        for bcds in I.getNodesFromType1(bc, 'BCDataSet_t'):
            merged_location = _getBCDataSetLocation(bcds)
            for bcd in I.getNodesFromType1(bcds, 'BCData_t'):
                fields = I.getNodesFromType1(bcd,'DataArray_t')
                if not fields: continue
                if merged_location in MergedNodes:
                    MergedNodes[merged_location] += fields
                else:
                    MergedNodes[merged_location] = fields

        for tag, loc in locations.items():
            for node_name in fields_names[tag].split():
                node = I.getNodeFromName(MergedNodes[loc], node_name)
                if not node:
                    J.save(zone,'debug.cgns')
                    raise ValueError(f'UNEXPECTED: could not find node {node_name}')
                if tag in nodes:
                    nodes[tag] += [ node ]
                else:
                    nodes[tag]  = [ node ]
        
        I._rmNodesByType1(bc, 'BCDataSet_t')
        for tag, fields in nodes.items():
            for f in fields: f[0] = f[0][:-len(tag)] # remove sufix
            bcds_name, bcd_name = containers_names[tag].split('/')
            bcds = I.getNodeFromName1(bc, bcds_name)
            if not bcds:
                bcds = I.createUniqueChild(bc,bcds_name,'BCDataSet_t')
                I.createUniqueChild(bcds,'GridLocation','GridLocation_t',value=locations[tag])

            bcd = I.getNodeFromName1(bcds, bcd_name)
            if not bcd:
                I.createUniqueChild(bcds,bcd_name,'BCData_t', children=fields)
            else:
                bcd[2] += fields
        I._rmNodesByName1(bc, 'tags_containers')

def _mergeBCtagContainerWithFlowSolutionTagContainer(zone, surf_bc):
    if zone[3] != 'Zone_t': return AttributeError('1st argument must be a zone')    
    if surf_bc[3] != 'Zone_t': return AttributeError('2nd argument must be a zone')    

    zone_tags = I.getNodeFromName1(surf_bc, 'tags_containers')
    if not zone_tags: return
    bcname = surf_bc[0].split('\\')[-1]
    zbc = I.getNodeFromName1(zone,'ZoneBC')
    if not zbc: return
    bc_tags = None
    for bc in I.getNodesFromType1(zbc,'BC_t'):
        if bc[0] != bcname: continue
        bc_tags = I.getNodeFromName1(bc,'tags_containers')
        break
    if not bc_tags: return
    for n in zone_tags[2]:
        bc_n = I.getNodeFromName1(bc_tags, n[0])
        if not bc_n:
            J.save(zone,'debug_zone.cgns')
            J.save(surf_bc,'debug_bc.cgns')
            raise ValueError(f'could not find tags_container child {n[0]}')
        n[2] += bc_n[2]
    
    for n in I.getNodeFromName1(zone_tags, 'locations')[2]:
        if I.getValue(n) == 'FaceCenter':
            I.setValue(n,'CellCenter')

    for n in I.getNodeFromName1(zone_tags, 'containers_names')[2]:
        name = I.getValue(n)
        I.setValue(n, name.replace('/NeumannData',''))

