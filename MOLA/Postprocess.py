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

def extractBoundaryLayer(surf, PressureDynamic=1.0, PressureRef=101325.):
    '''
    Given a surface (of 1-cell-depth in k-index, with j-increasing index being
    normal to the wall --typical of airfoil type--) make the required
    postprocessing for computing relevant wall quantities as well as
    boundary-layer edges.

    Parameters
    ----------

        surf : zone
            surface containing flowfields around the airfoil, as result
            of function :py:func:`buildAuxiliarWallNormalSurface`

        PressureDynamic : float
            Dynamic Pressure in [Pa] used for computation
            of skin-friction and pressure coefficients.

        PressureRef : float
            Reference Pressure in [Pa] used for computation
            of pressure coefficient (typically, freestream static pressure).

    Returns
    -------

        WallCurve : PyTree
            identical to output of function :py:func:`postProcessWallsFromElsaExtraction`

    '''

    _, Ni, Nj, Nk, dim = I.getZoneDim(surf)

    if dim == 2:
        WallCurve = T.subzone(surf,(1,1,1),(Ni,1,1))
    else:
        WallCurve = T.subzone(surf,(1,1,1),(Ni,Nj,1))

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

    for Eqn in Eqns: C._initVars(WallCurve, Eqn)

    WallCurve = postProcessWallsFromElsaExtraction(WallCurve,
                                                   PressureDynamic=PressureDynamic,
                                                   PressureRef=PressureRef)

    return WallCurve


def postProcessWallsFromElsaExtraction(t, PressureDynamic=1., PressureRef=1.):
    '''
    Create a PyTree with TopSide and BottomSide airfoil zones and
    boundary-layer edges zones.

    TopSide and BottomSide zones yield additional post-processed variables.


    Parameters
    ----------

        t : PyTree, base, zone, list of zones
            CGNS tree containing a 1-cell
            depth wall surface.

            .. attention:: It **must** contain the required fields by function
                :py:func:`addNewWallVariablesFromElsaQuantities`

        PressureDynamic : float
            Dynamic Pressure in [Pa] used for computation
            of skin-friction and pressure coefficients.

        PressureRef : float
            Reference Pressure in [Pa] used for computation
            of pressure coefficient (typically, freestream static pressure)

    Returns
    -------

        t : PyTree
            Airfoil split in top and bottom sides including postprocessed
            fields. It also includes zones corresponding to boundary-layer edges
    '''

    t = mergeWallsAndSplitAirfoilSides(t)

    addNewWallVariablesFromElsaQuantities(t,
                                          PressureDynamic=PressureDynamic,
                                          PressureRef=PressureRef)

    addBoundaryLayerEdges(t)

    return t

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

def addNewWallVariablesFromElsaQuantities(t, PressureDynamic=1., PressureRef=1.):
    '''
    Given a wall in pytree **t** and some reference values, produce additional
    variables:  `cf`, `Cp` and `ReynoldsTheta` of an AIRFOIL.

    As the sign of ``cf`` depends on the side of the airfoil, if special keyword
    ``'bottomside'`` is contained in a zone name of **t**, the sign is considered as
    *negative* in i-increasing direction, otherwise it is considered *positive*.

    Parameters
    ----------

        t : PyTree
            CGNS tree containing the two sides of the airfoil, as
            produced by :py:func:`mergeWallsAndSplitAirfoilSides`.
            New fields are added to the tree.

            .. note:: tree **t** is modified

        PressureDynamic : float
            Dynamic Pressure in [Pa] used for computation
            of skin-friction and pressure coefficients.

        PressureRef : float
            Reference Pressure in [Pa] used for computation
            of pressure coefficient (typically, freestream static pressure).
    '''

    CompulsoryFields = ('nx', 'ny', 'nz', 'Pressure', 'runit', 'theta11',
                        'SkinFrictionX', 'SkinFrictionY')
    for CompulsoryField in CompulsoryFields:
        if C.isNamePresent(t, CompulsoryField) != 1:
            I.printTree(t, color=True)
            ERRMSG = (
                'Field {} is missing on input data.\n'
                'Please check elsA extractions provided the following '
                'mandatory fields:\n{}'
                ).format(CompulsoryField, str(CompulsoryFields))
            raise AttributeError(FAIL+ERRMSG+ENDC)

    NewFields = ['ReynoldsTheta', 'Cp', 'cf']

    for zone in I.getZones(t):
        J._invokeFields(zone,NewFields)
        AllFieldsNames, = C.getVarNames(zone, excludeXYZ=True)
        v = J.getVars2Dict(zone, AllFieldsNames)

        nx, ny, nz = v['nx'], v['ny'], v['nz']
        NormalsNorm = np.sqrt( nx**2 + ny**2 + nz**2 )

        v['Cp'][:] = ( v['Pressure'] - PressureRef ) / PressureDynamic

        v['ReynoldsTheta'][:] = v['runit'] * v['theta11']

        v['cf'][:]  = ny*v['SkinFrictionX'] - nx*v['SkinFrictionY']
        v['cf'][:] /= (NormalsNorm*PressureDynamic)
        if 'bottom' in I.getName(zone).lower(): v['cf'] *= -1

def addBoundaryLayerEdges(t):
    '''
    Add to **t** new zones representing the boundary-layer edges if **t**
    contains the normal fields ``nx``, ``ny``, ``nz`` and at least one of:
    ``delta``, ``delta1``, ``delta11``

    Parameters
    ----------

        t : PyTree, base, zone, list of zones
            surface containing the fields
            ``nx`` ``ny`` ``nz`` and at least one of: ``delta``, ``delta1``, ``theta11``

            .. note:: the input **t** is modified : a new base including the
                zones of the boundary-layer thicknesses is appended to the tree
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

    if Trees:
        tM = I.merge(Trees)
        NewZones = I.getZones(tM)
        tNew = C.newPyTree(['BoundaryLayerEdges', NewZones])
        NewBase, = I.getBases(tNew)
        I.addChild(t,NewBase)

def postProcessBoundaryLayer(surf, VORTRATIOLIM=1e-3):
    '''
    Compute the boundary-layer of a structured 2D flowfield (typically around an
    airfoil) where the i-minimum window corresponds to the wall and j-increasing
    corresponds to wall-normal direction.


    New added fields include:

    ``TurbulentDistance``, ``delta``, ``delta1``, ``theta11``, ``runit``
     ``VelocityTangential``, ``VelocityTransversal``
     ``VelocityEdgeX``, ``VelocityEdgeY``, ``VelocityEdgeZ``

    Parameters
    ----------

        surf : zone
            as result of function :py:func:`buildAuxiliarWallNormalSurface`.

            .. note:: **surf** is modified *(new fields are added)*

        VORTRATIOLIM : float
            threshold for determining the boundary-layer edge
            with respect to the maximum value of field ``RotationScale`` starting
            from the wall.

    '''
    NewFields = ['TurbulentDistance', 'delta', 'delta1', 'theta11', 'runit',
                 'VelocityTangential', 'VelocityTransversal',
                 'VelocityEdgeX', 'VelocityEdgeY', 'VelocityEdgeZ']
    J._invokeFields(surf,NewFields)
    AllFieldsNames, = C.getVarNames(surf, excludeXYZ=True)
    v = J.getVars2Dict(surf, AllFieldsNames)
    x, y, z = J.getxyz(surf)

    NumberOfStations, NumberOfBoundaryLayerPoints = I.getZoneDim(surf)[1:3]
    for station in range(NumberOfStations):

        eta = v['TurbulentDistance'][station,:]
        VelocityTangential = v['VelocityTangential'][station,:]
        VelocityTransversal = v['VelocityTransversal'][station,:]
        VelocityX = v['VelocityX'][station,:]
        VelocityY = v['VelocityY'][station,:]
        VelocityZ = v['VelocityZ'][station,:]
        RotationScale = v['RotationScale'][station,:]
        RotationScaleMax = RotationScale.max()

        BoundaryLayerRegion, = np.where(RotationScale>VORTRATIOLIM*RotationScaleMax)

        if len(BoundaryLayerRegion) == 0:
            BoundaryLayerEdgeIndex = NumberOfBoundaryLayerPoints-1
        else:
            BoundaryLayerEdgeIndex = BoundaryLayerRegion[-1]

        # zero-th order boundary layer edge search
        v['VelocityEdgeX'][station,:] = VelocityX[BoundaryLayerEdgeIndex]
        v['VelocityEdgeY'][station,:] = VelocityY[BoundaryLayerEdgeIndex]
        v['VelocityEdgeZ'][station,:] = VelocityZ[BoundaryLayerEdgeIndex]

        VelocityEdgeVector = np.array([v['VelocityEdgeX'][station,0],
                                       v['VelocityEdgeY'][station,0],
                                       v['VelocityEdgeZ'][station,0]])

        NormalVector = np.array([v['nx'][station,0],
                                 v['ny'][station,0],
                                 v['nz'][station,0]])
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

        eta[:] =((x[station,:]-x[station,0])*NormalVector[0] +
                 (y[station,:]-y[station,0])*NormalVector[1] +
                 (z[station,:]-z[station,0])*NormalVector[2])

        v['delta'][station,:] = eta[BoundaryLayerEdgeIndex]

        Ue = VelocityTangential[BoundaryLayerEdgeIndex]
        Ut = VelocityTangential[:BoundaryLayerEdgeIndex]

        IntegrandDelta1 = np.maximum(0, 1. - (Ut/Ue) )
        IntegrandTheta = np.maximum(0, (Ut/Ue)*(1. - (Ut/Ue)) )

        v['delta1'][station,:] = np.trapz(IntegrandDelta1,
                                          eta[:BoundaryLayerEdgeIndex])

        v['theta11'][station,:] = np.trapz(IntegrandTheta,
                                           eta[:BoundaryLayerEdgeIndex])


        v['runit'][station,:] = (v['Density'][station,BoundaryLayerEdgeIndex]*Ue/
                            v['ViscosityMolecular'][station,BoundaryLayerEdgeIndex])


def getWalls(t):
    '''
    Get the walls from PyTree as a list of zones (surfaces)

    Parameters
    ----------

        t : PyTree
            tree containing *BCWall* type

    Returns
    -------

        OutputWalls : :py:class:`list` of zone
            list of surfaces corresponding to the walls
    '''
    walls = C.extractBCOfType(t,'BCWall',reorder=True)
    I._rmNodesByType(walls,'FlowSolution_t')
    walls = T.merge(walls)

    if getMeshDimensionFromTree(t) == 2:
        C._initVars(walls,'CoordinateZ', 0.)
        T._addkplane(walls, N=1)

        C._initVars(t, 'CoordinateZ', 0.)
        T._addkplane(t, N=1)
        T._translate(t, (0,0,-0.5))

    G._getNormalMap(walls)
    P._renameVars(walls,['centers:sx','centers:sy','centers:sz'],
                        ['centers:nx','centers:ny','centers:nz'])

    OutputWalls = []
    for wall in walls:
        wall = mergeWallsAndSplitAirfoilSides(wall)
        for cfn in ['nx','ny','nz']: C.center2Node__(wall,'centers:'+cfn,0)
        I._rmNodesByName(wall, I.__FlowSolutionCenters__)
        C._normalize(wall,['nx','ny','nz'])

        wall,= T.merge(wall)
        putNormalsPointingOutwards(wall)
        if W.is2DCurveClockwiseOriented(wall): T._reorder(wall,(-1,2,3))

        OutputWalls += [wall]

    OutputWalls = I.correctPyTree(OutputWalls, level=2)
    OutputWalls = I.correctPyTree(OutputWalls, level=3)

    return OutputWalls

def buildAuxiliarWallNormalSurface(t, wall, MaximumBoundaryLayerDistance=0.5,
        MaximumBoundaryLayerPoints=100, BoundaryLayerGrowthRate=1.05,
        FirstCellHeight=1e-6):
    '''
    Build an auxiliar merged surface from a PyTree of a 2D computation of the
    flow around an Airfoil from elsA.

    .. note:: this auxiliar surface is required for performing post-processing
        operations.

    Parameters
    ----------

        t : PyTree
            saved tree of an elsA simulation of the 2D flow around an airfoil

        wall : PyTree, base, zone, list of zones
            wall surfaces as got from :func:`getWalls`

    Returns
    -------

        surf : zone
            auxiliar surface including scales based on velocity
            gradients, required for further boundary-layer type of post-processing
    '''

    xMax = C.getMaxValue(wall,'CoordinateX')
    xMin = C.getMinValue(wall,'CoordinateX')
    Chord = xMax - xMin

    BoundaryLayerDistribution = W.linelaw(
        P1=(0,0,0),
        P2=(MaximumBoundaryLayerDistance*Chord,0,0),
        N=MaximumBoundaryLayerPoints,
        Distribution=dict(
            kind='ratio',
            growth=BoundaryLayerGrowthRate,
            FirstCellHeight=FirstCellHeight*Chord
            ))

    surf = G.addNormalLayers(wall,BoundaryLayerDistribution,check=1,niter=0)
    surf[0] = 'BoundaryLayerSurface'

    P._extractMesh(t, surf, order=2, extrapOrder=1,
                   mode='accurate', constraint=40., tol=1.e-10)

    AllFieldsNames, = C.getVarNames(surf, excludeXYZ=True)
    CentersFieldsNames = [fn for fn in AllFieldsNames if 'centers:' in fn]
    for cfn in CentersFieldsNames: C.center2Node__(surf,cfn,cellNType=0)

    MoX, MoY, MoZ = J.getVars(surf, ['MomentumX', 'MomentumY', 'MomentumZ'])
    if len(MoX.shape) == 2:
        MoX[:,0] = 0.
        MoY[:,0] = 0.
        MoZ[:,0] = 0.
    else:
        MoX[:,:,0] = 0.
        MoY[:,:,0] = 0.
        MoZ[:,:,0] = 0.

    for v in ('X','Y','Z'):
        C._initVars(surf, 'Velocity%s={Momentum%s}/{Density}'%(v,v))
        surf = P.computeGrad(surf,'Velocity%s'%v)



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
    C._initVars(surf,'centers:RotationScale='+Eqn)

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
    C._initVars(surf,'centers:DeformationScale='+Eqn)
    '''

    AllFieldsNames, = C.getVarNames(surf, excludeXYZ=True)
    CentersFieldsNames = [fn for fn in AllFieldsNames if 'centers:' in fn]
    for cfn in CentersFieldsNames: C.center2Node__(surf,cfn,cellNType=0)
    I._rmNodesByName(surf,I.__FlowSolutionCenters__)

    return surf

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
