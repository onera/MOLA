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
MOLA - GenerativeVolumeDesign.py

This module proposes functionalities for volume meshing, useful
for parametric mesh generation.

This module makes use of Cassiopee modules.

First creation:
01/03/2019 - L. Bernardos - Creation by recycling.
'''

import MOLA
from . import InternalShortcuts as J
from . import Wireframe as W
from . import GenerativeShapeDesign as GSD

if not MOLA.__ONLY_DOC__:

    # System modules
    import sys, os, time, timeit
    from copy import deepcopy as cdeep
    import numpy as np
    import numpy.linalg as la
    import scipy.interpolate
    from timeit import default_timer as tic

    # Cassiopee
    import Converter.PyTree as C
    import Converter.Internal as I
    import Geom.PyTree as D
    import Post.PyTree as P
    import Generator.PyTree as G
    import Transform.PyTree as T
    import Connector.PyTree as X
    import Intersector.PyTree as XOR
    try: import CPlot.PyTree as CPlot
    except: CPlot = None

    initVarsCostInSeconds = np.array([0.0])
    computeExtrusionIndicatorsTime = np.array([0.0])
    constrainedSmoothing = np.array([0.0])
    GrowthTime = np.array([0.0])
    smoothField = np.array([0.0])
    keepConstrainedHeight = np.array([0.0])
    ApplyConstraintsTime = np.array([0.0])
    TransferDataTime = np.array([0.0])
    StackLayersTime = np.array([0.0])
    ExtrusionPreprocess = np.array([0.0])
    Weighting = np.array([0.0])
    Gathering = np.array([0.0])
    Transformation = np.array([0.0])
    DisplaceLayerTime = np.array([0.0])
    newBaseLayerTime = np.array([0.0])

def extrudeWingOnSupport(Wing, Support, Distributions, Constraints=[],
        SupportMode='intersection', extrapolatePoints=30, InterMinSep=0.1,
        CollarRelativeRootDistance=0.1,
        extrudeOptions=dict(ExtrusionAuxiliarCellType='TRI',
                            printIters=True,
                            plotIters=False)):
    '''
    Extrude a wing-like (or rotor/blade) surface to generate a
    volume mesh. The Root of the wing is supported on the
    user-provided **Support** surface.

    The extrusion is done using the user-provided **Distributions**
    and **Constraints**, which are compatible with :py:func:`extrude` function,
    and hence **Distributions** include smoothing parameters fields.

    .. tip:: please read carefully documentation of function :py:func:`extrude`


    Parameters
    ----------

        Wing : PyTree
            PyTree with at least one zone which is the main wing
            structured surface.

            .. important:: **Wing** surface must be
                i-oriented following airfoil wise contour, and j-oriented
                following the root-to-tip wingspan.

        Support : PyTree
            contains the surface at root where wing will be supported

        Distributions : list
            equivalent input as for :py:func:`extrude`

        Constraints : dict
            equivalent input as for :py:func:`extrude`

        SupportMode : str
            indicate the provided support configuration with
            respect to the wing. Three possibilities exist:

            * ``'extrapolate'``
                the wing does not intersect the support *(an
                extrapolation is required)*

            * ``'intersection'``
                the wing intersects the support *(a trim is required)*

            * ``'supported'``
                the wing is already perfectly supported on support surface
                (a line-to-surface extrusion projected on the support is
                performed and then it is used as constraint for wing extrusion)

        extrapolatePoints : int
            Number of points for the discretization of the extrapolation zone

        InterMinSep : float
            Minimum distance that shall verify the trimming of
            the wing when configuration is such that **SupportMode** =
            ``'intersection'``

        CollarRelativeRootDistance : float
            Relative distance from wing's Root (``0``) and Tip (``1``) from
            which to perform the collar grid.

        extrudeOptions : dict
            Specifies additional options to :py:func:`extrude`

    Returns
    -------

        t : PyTree
            tree containing the extruded wing volume grid.
    '''

    # TODO: replace SupportMode input by a function that automatically computes
    # the type of configuration using geometrical operations.

    if SupportMode == 'extrapolate':
        # Total wing geometry up to support is not provided.
        # Hence, perform a constraint-free extrusion, then
        # project the root onto the hub and make multisection.


        # Extrude the wing
        tExtru = extrude(Wing,Distributions,Constraints,**extrudeOptions)

        ExtrudedVolume = I.getNodeFromName1(tExtru,'ExtrudedVolume')

        # Find the main wing VOLUME by proximity to support's
        # barycenter
        BarycenterSupport = tuple(G.barycenter(Support))
        MainWing,MainWingNoz = J.getNearestZone(ExtrudedVolume, BarycenterSupport)

        # Extract the extrapolation face, which shall be placed
        # in 'jmin' if wing was ordered consistently
        ExtrapFace = GSD.getBoundary(MainWing,'jmin')

        # Calculate the mean cell span length
        JMinPlus1  = GSD.getBoundary(MainWing,'jmin',1)
        x0,y0,z0   = J.getxyz(ExtrapFace)
        x1,y1,z1   = J.getxyz(JMinPlus1)
        MeanDist   = np.mean(np.sqrt( (x1-x0)**2 + (y1-y0)**2 + (z1-z0)**2))


        # Calculate the mean WallCellHeight
        WallCellHeight = []
        for d in Distributions:
            xd, yd, zd = J.getxyz(d)
            WallCellHeight += [((xd[1]-xd[0])**2+
                                (yd[1]-yd[0])**2+
                                (zd[1]-zd[0])**2)**0.5]
        WallCellHeight = np.mean(WallCellHeight)

        # Extrapolate the wing-extruded volume up to Support
        SupportFace = T.projectOrthoSmooth(ExtrapFace,Support,niter=100)
        Extrapolation, _= multiSections([SupportFace,ExtrapFace],[{'N':extrapolatePoints,'BreakPoint':1.,'kind':'tanhTwoSides','FirstCellHeight':WallCellHeight,'LastCellHeight':MeanDist}], InterpolationData={'InterpolationLaw':'interp1d_linear'},)

        # Join Extrapolation zone and MainWing
        MainWing = T.join(Extrapolation,MainWing)

        # Prapare output
        ExtrudedVolumeZones = I.getNodesFromType1(ExtrudedVolume,'Zone_t')
        ResultZones = []
        for noz in range(len(ExtrudedVolumeZones)):
            if noz != MainWingNoz:
                ResultZones += [ExtrudedVolumeZones[noz]]
            else:
                ResultZones += [MainWing]

        t = C.newPyTree(['WingExtrudedOnSupport',ResultZones])

        return t

    elif SupportMode == 'supported':
        # Total wing geometry up to support is provided.
        # perform partial-wing extrusion and then construct
        # collar grid.

        # Find the main wing SURFACE by proximity to support's
        # barycenter
        BarycenterSupport = tuple(G.barycenter(Support))
        MainWing,MainWingNoz = J.getNearestZone(Wing, BarycenterSupport)

        # Find the Root (shall be 'jmin')
        Root = GSD.getBoundary(MainWing,'jmin')
        xRoot, yRoot, zRoot = G.barycenter(Root)


        # Find the Tip (shall be 'jmax')
        Tip = GSD.getBoundary(MainWing,'jmax')
        xTip, yTip, zTip = G.barycenter(Tip)

        # Find the split point
        spt = CollarRelativeRootDistance
        xSplit = xTip*spt + xRoot*(1-spt)
        ySplit = yTip*spt + yRoot*(1-spt)
        zSplit = zTip*spt + zRoot*(1-spt)
        SplitPoint = (xSplit, ySplit, zSplit)

        # Split the wing span at split point
        NearestIndex, _ = D.getNearestPointIndex(MainWing,SplitPoint)
        _,Ni,Nj,Nk,_ = I.getZoneDim(MainWing)
        _,jsplit,_ = np.unravel_index(NearestIndex,(Ni,Nj,Nk),order='F')
        WingCollarWall = T.subzone(MainWing,(1,1,1),(Ni,jsplit+1,1))
        WingExtrusionWall = T.subzone(MainWing,(1,jsplit+1,1),(Ni,Nj,1))



        # Perform the Supported surface extrusion of the Root
        # Close the Root in order to make an O-mesh
        Root = C.convertArray2Tetra(Root)
        G._close(Root,tol=1.e-12)
        CurveConst = I.copyTree(Root); CurveConst[0]='ProjectionConstraintCurve'
        print("Root surface extrusion...")
        SuppExt = extrude(Root,Distributions,[],
            ExtrusionAuxiliarCellType='ORIGINAL',printIters=True,plotIters=False
            )

        # Get the Root projected surface
        RootSurf = I.getNodeFromName1(SuppExt,'ExtrudedVolume')[2][0]; RootSurf[0] = 'RootSurf'
        T._projectOrtho(RootSurf,Support)

        # Perform the wing extrusion
        zonesWing = I.getNodesFromType2(Wing,'Zone_t')
        zonesWing.remove(MainWing)
        zonesWing += [WingExtrusionWall]
        WingExtrude = C.newPyTree(['Base',zonesWing])
        print("Making wing extrusion...")
        tExtru = extrude(WingExtrude,Distributions,Constraints,**extrudeOptions)
        ExtrudedVolume = I.getNodeFromName1(tExtru,'ExtrudedVolume')

        # Find the main wing VOLUME by proximity to support's
        # barycenter
        MainWing,MainWingNoz = J.getNearestZone(ExtrudedVolume, BarycenterSupport)

        # Build the Side2-and-Side1/Outter Collar boundary
        _,Ni,Nj,Nk,_ = I.getZoneDim(MainWing)
        Side1 = RootSurf
        Side2 = GSD.getBoundary(MainWing,'jmin')
        Side2Curve = T.subzone(MainWing,(1,1,Nk),(Ni,1,Nk))
        Side1Curve = GSD.getBoundary(RootSurf,'jmax')

        # BUILD THE OUTTER SURFACE
        # Get the distribution
        _,Ni,Nj,Nk,_ = I.getZoneDim(WingCollarWall)
        JonctDist = T.subzone(WingCollarWall,(1,1,1),(1,Nj,1))
        Outter,_ = GSD.multiSections([Side1Curve,Side2Curve],JonctDist)


        # BUILD THE COLLAR GRID
        CollarZones = [Outter, WingCollarWall, Side1, Side2]
        Collar = fillCollar(*CollarZones)


        # Join Extrapolation zone and MainWing
        MainWing = T.join(Collar,MainWing)
        T._reorder(MainWing,(2,1,3))

        # Prapare output
        ExtrudedVolumeZones = I.getNodesFromType1(ExtrudedVolume,'Zone_t')
        ResultZones = []
        for noz in range(len(ExtrudedVolumeZones)):
            if noz != MainWingNoz:
                ResultZones += [ExtrudedVolumeZones[noz]]
            else:
                ResultZones += [MainWing]

        t = C.newPyTree(['WingExtrudedOnSupport',ResultZones])

        return t

    elif SupportMode == 'intersection':
        # BEWARE: Current intersection algorithm is approximate
        # The provided wing intersects the support.

        # Find the main wing SURFACE by proximity to support's
        # barycenter
        BarycenterSupport = tuple(G.barycenter(Support))
        MainWing,MainWingNoz = J.getNearestZone(Wing, BarycenterSupport)

        # Find the Root (shall be 'jmin')
        Root = GSD.getBoundary(MainWing,'jmin')
        xRoot, yRoot, zRoot = G.barycenter(Root)


        # Find the Tip (shall be 'jmax')
        Tip = GSD.getBoundary(MainWing,'jmax')
        xTip, yTip, zTip = G.barycenter(Tip)

        # Find the split point
        spt = CollarRelativeRootDistance
        xSplit = xTip*spt + xRoot*(1-spt)
        ySplit = yTip*spt + yRoot*(1-spt)
        zSplit = zTip*spt + zRoot*(1-spt)
        SplitPoint = (xSplit, ySplit, zSplit)

        # Split the wing span at split point
        NearestIndex, _ = D.getNearestPointIndex(MainWing,SplitPoint)
        _,Ni,Nj,Nk,_ = I.getZoneDim(MainWing)
        _,jsplit,_ = np.unravel_index(NearestIndex,(Ni,Nj,Nk),order='F')
        WingCollarWall = T.subzone(MainWing,(1,1,1),(Ni,jsplit+1,1)) # This still intersects the support
        WingExtrusionWall = T.subzone(MainWing,(1,jsplit+1,1),(Ni,Nj,1))


        # =============== APPROXIMATE METHOD =============== #
        # Performs an approximate intersection by averaging
        # directed projections for all sections between the
        # split point and the root of the wing.

        _,Ni,Nj,Nk,_ = I.getZoneDim(WingCollarWall)

        RootSection = T.subzone(MainWing,(1,1,1),(Ni,1,1))
        Projections, Distances = [], []
        for j in range(1,Nj):
            SectionM1= T.subzone(WingCollarWall,(1,j,1),(Ni,j,1))
            RSx, RSy, RSz = J.getxyz(RootSection)
            Section = T.subzone(WingCollarWall,(1,j+1,1),(Ni,j+1,1))
            dx, dy, dz = J.invokeFields(Section,['dx','dy','dz'])
            Sx, Sy, Sz = J.getxyz(Section)
            dx[:] = Sx-RSx
            dy[:] = Sy-RSy
            dz[:] = Sz-RSz

            Projected = T.projectAllDirs(Section,Support,vect=['dx','dy','dz'],oriented=0)
            Projections += [Projected]
            Distances += [W.distances(Projected,Section)[1]]
            if j>1:
                DeltaDist = Distances[-1]-Distances[-2]
                if Distances[-1] >= InterMinSep and DeltaDist>0:
                    break
        jroot = j

        if jroot >= jsplit:
            raise ValueError('The tip collar section (j=%d) is closer than the root collar section (j=%d).\nTry increasing CollarRelativeRootDistance and/or decrasing InterMinSep.'%(jsplit,jroot))

        # Perform weighted average of the projection
        Weights = 1./np.array(Distances)
        TotalWeights = np.sum(Weights)

        MeanProjection = I.copyTree(Projections[0])
        MeanProjection[0] = 'MeanProjection'
        MPx, MPy, MPz = J.getxyz(MeanProjection)
        MPx[:] *= Weights[0] / TotalWeights
        MPy[:] *= Weights[0] / TotalWeights
        MPz[:] *= Weights[0] / TotalWeights

        for i in range(1,len(Projections)):
            Proj = Projections[i]
            Px,Py,Pz = J.getxyz(Proj)

            MPx += Weights[i] * Px / TotalWeights
            MPy += Weights[i] * Py / TotalWeights
            MPz += Weights[i] * Pz / TotalWeights

        # Guarantee projection is supported
        T._projectOrtho(MeanProjection,Support)


        # Retrieve all collar sections
        Secs = [T.subzone(WingCollarWall,(1,j+1,1),(Ni,j+1,1)) for j in range(jroot-1,jsplit+1)]


        Secs = [MeanProjection] + Secs

        # Calculate the mean WallCellHeight
        WallCellHeight = []
        for d in Distributions:
            xd, yd, zd = J.getxyz(d)
            WallCellHeight += [((xd[1]-xd[0])**2+
                                (yd[1]-yd[0])**2+
                                (zd[1]-zd[0])**2)**0.5]
        WallCellHeight = np.mean(WallCellHeight)


        # The join cell width is the distance between the two last sections
        JoinCellWidth = W.distances(Secs[-1],Secs[-2])[1]

        Extrapolation, _= GSD.multiSections(Secs,[{'N':extrapolatePoints,'BreakPoint':1.,'kind':'tanhTwoSides','FirstCellHeight':WallCellHeight,'LastCellHeight':JoinCellWidth}], InterpolationData={'InterpolationLaw':'interp1d_linear'},)


        # Perform the Supported surface extrusion of the Root
        # Close the Root in order to make an O-mesh
        Root = C.convertArray2Tetra(MeanProjection)
        G._close(Root,tol=1.e-12)
        CurveConst = I.copyTree(Root); CurveConst[0]='ProjectionConstraintCurve'
        print("Root surface extrusion...")
        Const = dict(kind='Projected',curve=CurveConst,surface=Support,ProjectionMode='ortho')

        SuppExt = extrude(Root,Distributions,[],
            **extrudeOptions
            # ExtrusionAuxiliarCellType='ORIGINAL',printIters=True,plotIters=False
            )

        # Get the Root projected surface
        RootSurf = I.getNodeFromName1(SuppExt,'ExtrudedVolume')[2][0]; RootSurf[0] = 'RootSurf'
        T._projectOrthoSmooth(RootSurf,Support,niter=10)

        # Perform the wing extrusion
        zonesWing = I.getNodesFromType2(Wing,'Zone_t')
        zonesWing.remove(MainWing)
        zonesWing += [WingExtrusionWall]
        WingExtrude = C.newPyTree(['Base',zonesWing])
        print("Making wing extrusion...")
        tExtru = extrude(WingExtrude,Distributions,Constraints,**extrudeOptions)
        ExtrudedVolume = I.getNodeFromName1(tExtru,'ExtrudedVolume')

        # =============> RECYCLED FROM HERE

        # Find the main wing VOLUME by proximity to support's
        # barycenter
        MainWing,MainWingNoz = J.getNearestZone(ExtrudedVolume, BarycenterSupport)

        # Build the Side2-and-Side1/Outter Collar boundary
        _,Ni,Nj,Nk,_ = I.getZoneDim(MainWing)
        Side1 = RootSurf
        Side2 = GSD.getBoundary(MainWing,'jmin')
        Side2Curve = T.subzone(MainWing,(1,1,Nk),(Ni,1,Nk))
        Side1Curve = GSD.getBoundary(RootSurf,'jmax')

        # BUILD THE OUTTER SURFACE
        # Get the distribution
        _,Ni,Nj,Nk,_ = I.getZoneDim(Extrapolation)
        JonctDist = T.subzone(Extrapolation,(1,1,1),(1,Nj,1))
        Outter,_ = GSD.multiSections([Side1Curve,Side2Curve],JonctDist)


        # BUILD THE COLLAR GRID
        CollarZones = [Outter, Extrapolation, Side1, Side2]
        Collar = fillCollar(*CollarZones)


        # Join Extrapolation zone and MainWing
        MainWing = T.join(Collar,MainWing)
        T._reorder(MainWing,(2,1,3))

        # Prapare output
        ExtrudedVolumeZones = I.getNodesFromType1(ExtrudedVolume,'Zone_t')
        ResultZones = []
        for noz in range(len(ExtrudedVolumeZones)):
            if noz != MainWingNoz:
                ResultZones += [ExtrudedVolumeZones[noz]]
            else:
                ResultZones += [MainWing]

        t = C.newPyTree(['WingExtrudedOnSupport',ResultZones])

        return t



def extrude(t, Distributions, Constraints=[], extractMesh=None,
            ExtrusionAuxiliarCellType='QUAD', modeSmooth='smoothField',
            growthEquation='default',
            closeExtrusionLayer=False,
            printIters=False, plotIters=False,
            extractMeshHardSmoothOptions={'niter':0,
                                          'HardSmoothLayerProtection':100,
                                          'FactordH':2.,
                                          'window':'jmin'},
            HardSmoothPoints=[], HardSmoothRescaling=False,
            starting_message=''):
    '''
    Make a hyperbolic extrusion of a surface following its normal direction.
    **Constraints** to be satisfied through extrusion can be provided by the user.

    Please refer to the tutorial ``"MinimalConstrainedExtrusionExamples.py"`` for
    proper understanding of the basic capabilities of this function.

    Parameters
    ----------

        t : PyTree, base, zone or list of zones
            CGNS object where the surfaces
            to be extruded are located (exclusively). Beware that the extrusion
            direction is the normal direction of the surfaces.

        Distributions : list
            List of zones of 1D structured curves used to
            guide the local cell heights through the extrusion process. Smoothing
            parameters are contained in these curves via FlowSolution fields
            named ``'normalfactor'``, ``'growthfactor'``, ``'normaliters'`` and
            ``'growthiters'``.

            .. note:: aforementioned fields **MUST BE** present in the
                ``FlowSolution`` container of each curve zone.

            .. note:: at least one distribution guide is required for the
                extrusion.

        Constraints : list
            Each element of this list is
            a dictionary used to set the constraints that shall be satisfied
            during the extrusion process. For information on relevant accepted
            values to provide, please refer to :py:func:`addExtrusionConstraint`
            doc.

            .. note:: the dictionary is entirely passed to :py:func:`addExtrusionConstraint`
                via ``**kwargs``)

        extractMesh : zone
            .. warning :: this is an EXPERIMENTAL feature and may
                significantly change or even be deleted in future.*

            If user assigns a
            zone to this argument, then any flowfield contained in the provided zone
            will override the fields of the same name of the extrusion front. This
            includes, for example: ``sx``, ``sy``, ``sz`` (direction of extrusion). These
            will hence be employed for extrusion instead of being calculated.
            The extrusion front surface grid coordinates are directly smoothed.
            If **extractMesh** is not :py:obj:`None`, then **extractMeshHardSmoothOptions**
            dictionary is meaningful. This technique may be useful for allowing for
            smoothing of the extrusion layer front while veryfing imposed
            extrusion directions (it can be seen as a soft overall constraint).

        ExtrusionAuxiliarCellType : str
            Sets the cell type employed
            by the extrusion front surface. May be one of: 'TRI','QUAD','ORIGINAL'.

            'TRI' is Recommended.

        modeSmooth : str
            Sets the normals and cell height smoothing
            technique. Currently, only two options are available:

            * ``'smoothField'``
                Note that distribution's fields
                'expansionfactor', 'normaliters' and 'growthiters' are employed in
                this technique

            * ``'dualing'``
                Note that distribution's fields
                'normalfactor', 'normaliters' and 'growthiters' are employed in
                this technique

            * ``'ignore'``
                do not smooth

        growthEquation : str
            This is a ``Converter.initVars``-compatible
            string used for tuning the local extrusion cell height of the front.
            Extrusion quality and robustness strongly depends on this equation.
            The equation may involve any ``FlowSolution`` field defined on the extrusion
            front surface (which is a CGNS zone).
            Some examples of quantities typically employed in the equation are:

            * ``{nodes:dH}``
                the local cell height of extrusion.

            * ``{nodes:growthfactor}``
                parameter inferred from Distributions list.

            * ``{nodes:convexity}``
                a measure of the concavity/convexity. It corresponds
                to the divergence of the normals.

            * ``{nodes:vol}``
                the volume (or more precisely, the area) of each cell
                of the extrusion front.

            * ``{nodes:stretching}``
                The stretching of the cells (the laplacian of the volume).

        closeExtrusionLayer : bool
            if :py:obj:`True`, force the extrusion front surface
            to be closed. This is necessary in order to perform "O-shaped" extrusion
            around watertight surfaces.

        printIters : bool
            if :py:obj:`True`, prints to standard output messages
            during the extrusion process.

        plotIters : bool
            if :py:obj:`True`, interactively open a CPlot window and
            show the extrusion front during the extrusion process. This is a
            valuable help for setting the smoothing parameters and growth equation

        extractMeshHardSmoothOptions : dict
            here, a set of
            parameters are sent to the extractMesh-constraining technique.
            Relevant parameters are:

            * ``niter`` : int
                number of iterations to pass to function :py:func:`T.smooth()`
                that is applied on the extrusion front surface

            * ``HardSmoothLayerProtection`` : int
                this parameter behaves as a
                protection in order to avoid smoothing the grid coordinates of the
                extrusion front at layers lower than this value. It is useful to
                avoid smoothing layers that are likely conained in boundary-layer

            * ``window`` : str
                must be a window keyword compatible with
                :py:func:`MOLA.GenerativeShapeDesign.getBoundary`. It indicates
                from which window a slice of the
                provided zone in extractMesh attribute is built.

            * ``FactordH`` : float
                This indicates the local scaling of the window
                slice in order to create a proper 1-cell height volume mesh for
                accurate :py:func:`P.extractMesh` usage.

        HardSmoothPoints : list
            each element of the list is a zone
            point (as obtained from function :py:func:`D.point` ). During extrusion process,
            all points of this list are projected towards the extrusion surface
            front. Each point defines a region where a hard :py:func:`T.smooth` operation
            will be performed at each extrusion step. The parameters for the call
            of the :py:func:`T.smooth` function are extracted from ``FlowSolution`` of each
            point. Hence, the ``FlowSolution`` of each point may only exclusively
            be composed of the accepted attributes of :py:func:`T.smooth` function.

        HardSmoothRescaling : bool
            if :py:obj:`True`, performs an auxiliary
            normalization of the extrusion front for the application of the
            HardSmoothPoints. This may be necessary for big grids, as :py:func:`T.smooth`
            depends on the dimensions of the grid.

    Returns
    -------

        tExtru : PyTree
            tree where each base contains different kind of
            information (e.g. the resulting extruded volume, the constraints,
            the curves employed as distributions, etc)
    '''
    toc = tic()
    tExtru = invokeExtrusionPyTree(t)

    # Create auxiliary surface that will be used for extrusion
    _addExtrusionLayerSurface(tExtru, ExtrusionAuxiliarCellType, closeExtrusionLayer)
    ExtrudeLayer = I.getNodeFromName2(tExtru,'ExtrudeLayer')

    NLayers = []
    for d in Distributions:
        NLayers += [C.getNPts(d)]
        _addExtrusionDistribution(tExtru,d)
    NLayers = min(NLayers)

    for c in Constraints: addExtrusionConstraint(tExtru, **c)

    _addHardSmoothPoints(tExtru, HardSmoothPoints)

    _transferDistributionData(tExtru,layer=0)

    AllLayersBases = [newBaseLayer(tExtru,layer=0)]

    ExtrusionPreprocess[0] = tic() - toc

    OptionsIntegerFields = ('niter','type')
    toc = tic()
    nbOfDigits = int(np.ceil(np.log10(NLayers+1)))
    LastLayer = ('{:0%d}'%nbOfDigits).format(NLayers)
    for l in range(1,NLayers):
        currentLayer = ('{:0%d}'%nbOfDigits).format(l+1)
        Message = starting_message+' %s/%s | cost: %0.5f s'%(currentLayer,LastLayer,tic()-toc)
        toc = tic()
        if printIters:
            print('{:s}\r'.format(''), end='', flush=True)
            print(Message, end='')

        tok = tic()
        _transferDistributionData(tExtru,layer=l)
        _transferExtractMeshData(ExtrudeLayer,extractMesh,extractMeshHardSmoothOptions,layer=l)
        TransferDataTime[0] += tic() - tok


        if HardSmoothRescaling:
            BoundingBoxOfExtrusionSurface = G.BB(ExtrudeLayer, method='OBB')
            BoundingBoxOfExtrusionSurface[0] = 'BoundingBoxOfExtrusionSurface'
            xBB, yBB, zBB = J.getxyz(BoundingBoxOfExtrusionSurface)
            xBB = xBB.ravel(order='F')
            yBB = yBB.ravel(order='F')
            zBB = zBB.ravel(order='F')
            DiameterLine = D.line((xBB[0], yBB[0], zBB[0]),
                                  (xBB[-1], yBB[-1], zBB[-1]), 2)
            Diameter = D.getLength(DiameterLine)
            T._homothety(ExtrudeLayer, (0,0,0), 1./Diameter)
            T._homothety(HardSmoothPoints, (0,0,0), 1./Diameter)
            wrapInitVars(HardSmoothPoints,'radius={radius}*%g'%(1./Diameter))

        for HardSmoothPt in HardSmoothPoints:
            T._projectOrtho(HardSmoothPt, ExtrudeLayer)
            OptionsContainedInPoint, = C.getVarNames(HardSmoothPt,
                                                     excludeXYZ=True)
            PointFieldsDict = J.getVars2Dict(HardSmoothPt,OptionsContainedInPoint)
            options = dict()
            for opt in OptionsContainedInPoint:
                if opt in OptionsIntegerFields:
                    options[opt] = int(PointFieldsDict[opt])
                else:
                    options[opt] = PointFieldsDict[opt]
            options['point'] = J.getxyz(HardSmoothPt)
            options['fixedConstraints'] = [P.exteriorFaces(ExtrudeLayer)]

            T._smooth(ExtrudeLayer,**options)

        if HardSmoothRescaling:
            T._homothety(ExtrudeLayer, (0,0,0), Diameter)
            T._homothety(HardSmoothPoints, (0,0,0), Diameter)
            wrapInitVars(HardSmoothPoints,'radius={radius}*%g'%(Diameter))

        if modeSmooth != 'ignore':
            _computeExtrusionIndicators(tExtru)
        else:
            _extrusionApplyConstraints(tExtru)


        _constrainedSmoothing(tExtru, mode=modeSmooth, growthEquation=growthEquation)

        _displaceLayer(tExtru)

        AllLayersBases += [newBaseLayer(tExtru,layer=l)]

        if plotIters and CPlot:
            CPlot.display(ExtrudeLayer,
                 # mode='Scalar',
                 # scalarStyle=0,bgColor=0,
                 # scalarField='dH',
                 displayIsoLegend=1,
                 colormap=8,
                    # win=(800,600),export='Frame%05d.png'%k
                    )
            Message = 'Showing layer %d of %d'%(l+1,NLayers)
            CPlot.setState(message=Message)
            # time.sleep(0.1)
    if printIters:
        print('\nstacking layers... ',end='')
    _stackLayers(tExtru, AllLayersBases) # Stack layers
    if printIters:
        print(J.GREEN+'ok'+J.ENDC)

    return tExtru

def _constrainedSmoothing(tExtru, mode='dualing',
                          growthEquation='default'):
    '''
    This is a private function employed by extrude().

    Its purpose is to apply the constrained smoothing following the technique
    provided by argument <mode>, and the growth law provided by argument
    <growthEquation>. All operations are done in-place in PyTree <tExtru>.

    This is an interface for smoothing of cell height {dH} and normals
    {sx}, {sy}, {sz}

    INPUTS

    tExtru - (PyTree) - the PyTree of extrusion. It is modified in-place.

    mode - (string) - can be 'dualing' or 'ignore'

    growthEquation - (string) - the user-provided growth equation employing
        variables contained in the extrusion front surface
    '''

    toc = tic()
    ExtLayBase = I.getNodeFromName1(tExtru,'ExtrudeLayerBase')
    z = I.getNodeFromName1(ExtLayBase,'ExtrudeLayer')

    if mode == 'dualing':

        # Perform smoothing of cell height (to deal with concavities)
        # This shall be done in order to keep the mean
        # user-prescribed extrusion height
        MeanDH = C.getMeanValue(z,'nodes:dH')
        if growthEquation: wrapInitVars(z,growthEquation)
        NewMeanDH = C.getMeanValue(z,'nodes:dH')
        RatioDH = MeanDH/NewMeanDH
        wrapInitVars(z,'nodes:dH={nodes:dH}*%g'%RatioDH) # TODO Really required?

        niter = int(C.getMeanValue(z,'growthiters'))
        for i in range(niter):
            C.node2Center__(z, 'nodes:dH')
            C.center2Node__(z, 'centers:dH', cellNType=0)
            _keepConstrainedHeight(tExtru)

        # Perform smoothing of normals
        niter = int(C.getMeanValue(z,'normaliters'))
        for i in range(niter):

            # TODO: replace with smoothing fuction and/or T._deformNormals()
            sx, sy, sz, dH = J.getVars(z, ['sx','sy','sz','dH'], 'FlowSolution')
            NormalsDiffuseFactor = C.getMeanValue(z,'normalfactor')*dH/NewMeanDH
            BoolRegion = dH > NewMeanDH
            NormalsDiffuseFactor = NormalsDiffuseFactor[BoolRegion]
            sx = sx[BoolRegion]
            sy = sy[BoolRegion]
            sz = sz[BoolRegion]
            nsubiter = 50
            for j in range(nsubiter):
                C.node2Center__(z,'nodes:sx')
                C.node2Center__(z,'nodes:sy')
                C.node2Center__(z,'nodes:sz')
                C.center2Node__(z,'centers:sx',cellNType=0)
                C.center2Node__(z,'centers:sy',cellNType=0)
                C.center2Node__(z,'centers:sz',cellNType=0)

                sx *= NormalsDiffuseFactor
                sy *= NormalsDiffuseFactor
                sz *= NormalsDiffuseFactor

                # For debugging:
                '''
                if j==0:
                    # CPlot.display(z,
                    #     mode='Scalar',
                    #     posCam=(6.33443, -5.77483, 0.432796),
                    #     posEye=(5.23491, -6.22685, 0.182558),
                    #     dirCam=(-0.170465, -0.0660772, 0.983146),
                    #     scalarField='nodes:dH',
                    #     displayIsoLegend=1,
                    #     colormap=8,
                    #     )

                    CPlot.display(z,
                        mode='Vector',
                        posCam=(6.10043, -5.90142, 0.236647),
                        posEye=(5.20552, -6.2679, 0.115938),
                        dirCam=(-0.216225, 0.233212, 0.948082),
                        vectorField1='nodes:sx',
                        vectorField2='nodes:sy',
                        vectorField3='nodes:sz',
                        vectorDensity=500.,
                        vectorNormalize=True,
                        vectorStyle=1,
                        vectorShape=1,
                        vectorShowSurface=1,
                        scalarField='nodes:dH',
                        displayIsoLegend=1,
                        colormap=8,
                        )

                    Message = 'smooth subiteration %d of %d'%(j+1,nsubiter)
                    CPlot.setState(message=Message)
                else:
                    CPlot.display(z)
                    CPlot.setState(message='smooth subiteration %d of %d'%(j+1,nsubiter))
                time.sleep(0.1)
                '''
            normalizeVector(z, ['sx', 'sy', 'sz'], container='FlowSolution')
            _extrusionApplyConstraints(tExtru)

        I._rmNodesByName(z,'FlowSolution#Centers')
        normalizeVector(z, ['sx', 'sy', 'sz'], container='FlowSolution')

    elif mode == 'smoothField':
        _extrusionApplyConstraints(tExtru)

        I._rmNodesByName(z,'FlowSolution#Centers')
        sx, sy, sz, dH, convexity = J.getVars(z, ['sx','sy','sz','dH','convexity'], 'FlowSolution')
        type = 0
        niter = int(C.getMeanValue(z,'growthiters'))
        try:
            subniter = int(C.getMeanValue(z,'growthsubiters'))
        except:
            subniter = 100

        gf, stretch, convexity = J.getVars(z,['growthfactor','stretching','convexity'])

        if subniter:
            for i in range(niter):

                toki = tic()

                MeanDH = np.median(dH)
                if growthEquation:
                    if growthEquation == 'default':
                        v = stretch - convexity
                        dH*=(1+gf*np.tanh((v*(v<0))/(np.abs(np.min(v))/3.0)+(v*(v>0))/(np.max(v)/3.0)))
                    else:
                        wrapInitVars(z,growthEquation)

                wrapSmoothField(z, 0.9, subniter, type, ['dH'])
                NewMeanDH = np.mean(dH)
                dH *= MeanDH/NewMeanDH
                GrowthTime[0] += tic() - toki

                _keepConstrainedHeight(tExtru)

        # Perform smoothing of normals
        niter = int(C.getMeanValue(z,'normaliters'))
        try:
            subniter = int(C.getMeanValue(z,'normalsubiters'))
        except:
            subniter = 100

        if subniter:
            for i in range(niter):
                wrapSmoothField(z, 0.9, subniter, type, ['sx','sy','sz'])
                normalizeVector(z, ['sx', 'sy', 'sz'], container='FlowSolution')
                _extrusionApplyConstraints(tExtru)

        try:
            ef = I.getNodeFromName(z,'expansionfactor')[1]
        except:
            ef = 10.
        normalizeVector(z, ['sx', 'sy', 'sz'], container='FlowSolution')
        # wrapSmoothField(z, 0.9, 200, type, ['dsx','dsy','dsz'])
        sx, sy, sz, dsx, dsy, dsz, dH = J.getVars(z,[ 'sx',  'sy',  'sz',
                                                     'dsx', 'dsy', 'dsz','dH'])

        sx += ef*dsx/dH
        sy += ef*dsy/dH
        sz += ef*dsz/dH
        normalizeVector(z, ['sx', 'sy', 'sz'], container='FlowSolution')
        _extrusionApplyConstraints(tExtru)


    constrainedSmoothing[0] += tic() - toc

    return tExtru



def normalizeVector(t, vectorNames=['sx','sy','sz'], container='FlowSolution'):
    '''
    This function is a convenient alternative of :py:func:`C._normalize`. It deals with
    specific container of the aimed vector.

    Parameters
    ----------

        t : PyTree, base, zone or list of zones
            input where CGNS zones with the aimed fields to normalize exist

            .. note:: **t** is modified

        vectorNames : :py:class:`list` of 3 :py:class:`str`
            specify the three 3D components of the flow field to normalize

        container : str
            specify the container where the vector fields are located
    '''

    for z in I.getZones(t):
        vx, vy, vz = J.getVars(z, vectorNames, container)
        Magnitude = np.maximum(np.sqrt(vx*vx + vy*vy + vz*vz), 1e-12)
        vx /= Magnitude
        vy /= Magnitude
        vz /= Magnitude



def _extrusionApplyConstraints(tExtru):
    '''
    This is a private function employed by user function extrude() through
    private function _constrainedSmoothing()

    This function applies the user-provided constraints (list of python
    dictionaries provided to extrude(Constraints) argument)

    INPUTS

    tExtru - (PyTree) - the extrusion PyTree
    '''

    toc = tic()
    ExtLayBase = I.getNodeFromName1(tExtru,'ExtrudeLayerBase')
    zE = I.getNodeFromName1(ExtLayBase,'ExtrudeLayer')
    x,y,z = J.getxyz(zE)
    FlowSolExtLay = I.getNodeFromName1(zE,'FlowSolution')
    sx = I.getNodeFromName1(FlowSolExtLay,'sx')[1]
    sy = I.getNodeFromName1(FlowSolExtLay,'sy')[1]
    sz = I.getNodeFromName1(FlowSolExtLay,'sz')[1]
    dH = I.getNodeFromName1(FlowSolExtLay,'dH')[1]
    ConstraintWireframe = I.getNodeFromName1(tExtru,'ConstraintWireframe')
    ConstraintSurfaces = I.getNodeFromName1(tExtru,'ConstraintSurfaces')
    for constraint in ConstraintWireframe[2]:
        NPts = C.getNPts(constraint)
        silence = J.OutputGrabber()
        cdH, = J.invokeFields(constraint,['dH']) # not optional: data MUST be translated from ExtrudeLayer to constraint
        cdH = cdH.ravel(order='F')
        with silence: cdH, csx, csy, csz, cdx, cdy, cdz = J.getVars(constraint,['dH','sx','sy','sz','dx','dy','dz'])
        cx, cy, cz = J.getxyz(constraint)
        ExtrusionDataNode = I.getNodeFromName1(constraint,'.ExtrusionData')
        PointListReceiver = I.getNodeFromName1(ExtrusionDataNode,'PointListReceiver')[1]

        kind = I.getValue(I.getNodeFromName1(ExtrusionDataNode,'kind'))

        if kind == 'Imposed':
            try: sx[PointListReceiver] = csx.ravel(order='F')
            except: pass
            try: sy[PointListReceiver] = csy.ravel(order='F')
            except: pass
            try: sz[PointListReceiver] = csz.ravel(order='F')
            except: pass

            # required for displacing the constraint
            cdH[:] = dH[PointListReceiver]

        elif kind == 'Initial':
            try: csx[:] = sx[PointListReceiver]
            except: pass
            try: csy[:] = sy[PointListReceiver]
            except: pass
            try: csz[:] = sz[PointListReceiver]
            except: pass
            cdH[:] = dH[PointListReceiver]

            I.createUniqueChild(ExtrusionDataNode,'kind','DataArray_t',value='Imposed')

        elif kind == 'Copy':

            cdH, csx, csy, csz = J.invokeFields(constraint,['dH','sx','sy','sz'])
            pointsToCopyZoneName = I.getValue(I.getNodeFromName(ExtrusionDataNode,'pointsToCopyZoneName'))
            pointsToCopy = I.getNodeFromName1(ConstraintWireframe,pointsToCopyZoneName)
            PointListDonor = I.getNodeFromName(ExtrusionDataNode,'PointListDonor')[1]
            PointListReceiver = I.getNodeFromName(ExtrusionDataNode,'PointListReceiver')[1]

            rsx, rsy, rsz = J.getVars(zE,['sx','sy','sz'])
            sx[PointListReceiver] = csx[:] = rsx[PointListDonor]
            sy[PointListReceiver] = csy[:] = rsy[PointListDonor]
            sz[PointListReceiver] = csz[:] = rsz[PointListDonor]
            dH[PointListReceiver] = cdH[:] = dH[PointListDonor]

        elif kind == 'CopyAndRotate':

            cdH, csx, csy, csz = J.invokeFields(constraint,['dH','sx','sy','sz'])
            pointsToCopyZoneName = I.getValue(I.getNodeFromName(ExtrusionDataNode,'pointsToCopyZoneName'))
            pointsToCopy = I.getNodeFromName1(ConstraintWireframe,pointsToCopyZoneName)
            PointListDonor = I.getNodeFromName(ExtrusionDataNode,'PointListDonor')[1]
            PointListReceiver = I.getNodeFromName(ExtrusionDataNode,'PointListReceiver')[1]
            RotationCenter = I.getNodeFromName(ExtrusionDataNode,'RotationCenter')[1]
            RotationAxis = I.getNodeFromName(ExtrusionDataNode,'RotationAxis')[1]
            RotationAngle = I.getNodeFromName(ExtrusionDataNode,'RotationAngle')[1]

            # Validated approach :
            ExtrudeLayerRotated = T.rotate(zE, RotationCenter,
                                           RotationAxis,float(RotationAngle),
                                           vectors=[['sx','sy','sz']])
            rsx, rsy, rsz = J.getVars(ExtrudeLayerRotated,['sx','sy','sz'])
            sx[PointListReceiver] = csx[:] = rsx[PointListDonor]
            sy[PointListReceiver] = csy[:] = rsy[PointListDonor]
            sz[PointListReceiver] = csz[:] = rsz[PointListDonor]
            dH[PointListReceiver] = cdH[:] = dH[PointListDonor]

        elif kind == 'Projected':

            try:
                ProjSurfaceName = I.getValue(I.getNodeFromName(ExtrusionDataNode,'ProjectionSurfaceName'))
                ProjSurface = I.getNodeFromName1(ConstraintSurfaces,ProjSurfaceName)
            except:
                pass

            ProjMode = I.getValue(I.getNodeFromName(ExtrusionDataNode,'ProjectionMode'))

            cdH, csx, csy, csz, cdx, cdy, cdz = J.invokeFields(constraint,['dH','sx','sy','sz','dx','dy','dz'])

            if ProjMode == 'ortho':

                # Import existing values from ExtrudeLayer
                csx[:]=sx[PointListReceiver]
                csy[:]=sy[PointListReceiver]
                csz[:]=sz[PointListReceiver]
                cdH[:]=dH[PointListReceiver]

                # New displaced auxiliar Curve from constraint
                cdx[:] = csx*cdH
                cdy[:] = csy*cdH
                cdz[:] = csz*cdH
                AuxCurve = T.deform(constraint,vector=['dx','dy','dz'])
                AuxCurve[0]='AuxCurve'

                # Projects curve into surface
                T._projectOrtho(AuxCurve,ProjSurface)

                # Update normals with new distances
                acx, acy, acz  = J.getxyz(AuxCurve)

                # Converter normalization:
                cdH[:] = np.sqrt((acx - cx)**2 + (acy - cy)**2 + (acz - cz)**2)
                csx[:] = acx - cx
                csy[:] = acy - cy
                csz[:] = acz - cz
                C._normalize(constraint,['sx','sy','sz'])

                # Export the values to ExtrudeLayer
                sx[PointListReceiver] = csx
                sy[PointListReceiver] = csy
                sz[PointListReceiver] = csz
                # dH[PointListReceiver] = cdH # TODO: further investigate wrapping consequences


            elif ProjMode == 'dir':
                ProjDir = I.getNodeFromName(ExtrusionDataNode,'ProjectionDir')[1]

                # Import existing values from ExtrudeLayer
                csx[:]=sx[PointListReceiver]
                csy[:]=sy[PointListReceiver]
                csz[:]=sz[PointListReceiver]
                cdH[:]=dH[PointListReceiver]

                # New displaced auxiliar Curve from constraint
                cdx[:] = csx*cdH
                cdy[:] = csy*cdH
                cdz[:] = csz*cdH
                AuxCurve = T.deform(constraint,vector=['dx','dy','dz']); AuxCurve[0]='AuxCurve'

                # Projects curve into surface
                T._projectDir(AuxCurve,ProjSurface,(ProjDir[0],ProjDir[1],ProjDir[2]),smooth=0,oriented=0)

                # Update normals with new distances
                acx, acy, acz  = J.getxyz(AuxCurve)

                # Converter normalization:
                csx[:] = acx - cx
                csy[:] = acy - cy
                csz[:] = acz - cz
                C._normalize(constraint,['sx','sy','sz'])

                # Export the normal values to ExtrudeLayer
                sx[PointListReceiver] = csx
                sy[PointListReceiver] = csy
                sz[PointListReceiver] = csz

            elif ProjMode == 'CylinderRadial':
                a = I.getValue(I.getNodeFromName(ExtrusionDataNode,'ProjectionAxis'))
                c = I.getValue(I.getNodeFromName(ExtrusionDataNode,'ProjectionCenter'))

                csx = csx.ravel(order='F')
                csy = csy.ravel(order='F')
                csz = csz.ravel(order='F')

                # Import existing values from ExtrudeLayer
                csx[:]=sx[PointListReceiver]
                csy[:]=sy[PointListReceiver]
                csz[:]=sz[PointListReceiver]
                cdH[:]=dH[PointListReceiver] # this will allow constraint curve to adapt

                T._alignVectorFieldWithRadialCylindricProjection(constraint, c, a,
                                                           ['sx','sy','sz'])

                sx[PointListReceiver] = csx
                sy[PointListReceiver] = csy
                sz[PointListReceiver] = csz

            else:
                raise ValueError('Extrusion constraints: Projection mode %s not recognized'%ProjMode)


        elif kind == 'Match':

            cdH, csx, csy, csz, cdx, cdy, cdz = J.invokeFields(constraint,['dH','sx','sy','sz','dx','dy','dz'])

            cx, cy, cz = J.getxyz(constraint)

            # Get the Matching surface
            MatchSurfaceName = I.getValue(I.getNodeFromName(ExtrusionDataNode,'MatchSurfaceName'))
            MatchSurface = I.getNodeFromName1(ConstraintSurfaces,MatchSurfaceName)
            mx, my, mz = J.getxyz(MatchSurface)
            mShape = mx.shape
            lenmShape = len(mShape)


            # DETERMINE THE CURRENT AND AIM SLICES
            Pts = [(cx[i],cy[i],cz[i]) for i in range(NPts)]
            Res = D.getNearestPointIndex(MatchSurface,Pts)
            indicesCurrSurf = [r[0] for r in Res]
            ijkCurrSurf = np.unravel_index(indicesCurrSurf,mShape,order='F')
            ijkVStack   = np.vstack(ijkCurrSurf).T

            MatchDirNode = I.getNodeFromName(ExtrusionDataNode,'MatchDir')
            if MatchDirNode is None:
                # TODO: STRESS THIS ROBUSTNESS
                # Try to infer the MatchDir
                # MatchDir is 1, 2 or 3 for constant i, j, k
                if lenmShape == 1:
                    # Constraint element is a curve
                    MatchDir = 1
                elif lenmShape == 2:
                    # Constraint element is a surface
                    # so the surface evolves in i and j
                    isIslice = not np.any(np.diff(ijkVStack[:,0]))
                    MatchDir = 1 if isIslice else 2
                elif lenmShape == 3:
                    # Constraint element is a degenerated
                    # surface or a volume mesh.
                    IsliceCandidate = not np.any(np.diff(ijkVStack[:,0,0]))
                    JsliceCandidate = not np.any(np.diff(ijkVStack[0,:,0]))
                    KsliceCandidate = not np.any(np.diff(ijkVStack[0,0,:]))

                    if IsliceCandidate and JsliceCandidate:
                        # The slice is done following i and j.
                        # So the MatchDir (look direction) is following k
                        MatchDir = 3
                    elif IsliceCandidate and KsliceCandidate:
                        MatchDir = 2
                    else:
                        MatchDir = 1


                # Changes the sign (look orientation) of MatchDir
                # if the first element belonging to the slice
                # is not the first slice of the surface

                if ijkVStack[0][MatchDir-1] > 0: MatchDir *= -1

                # Store MatchDir
                I.createUniqueChild(ExtrusionDataNode,'MatchDir','DataArray_t',value=MatchDir)

            else:
                MatchDir = MatchDirNode[1]

            # CurrentLayer is the current layer number
            CurrentLayer = ijkVStack[0][abs(MatchDir)-1]

            AimIndexVec = np.zeros(lenmShape,dtype=int,order='F')
            absMatchDir = abs(MatchDir)
            AimIndexVec[absMatchDir-1] = np.sign(MatchDir)

            mxr = mx.ravel(order='F')
            myr = my.ravel(order='F')
            mzr = mz.ravel(order='F')

            multi_index = ijkVStack+AimIndexVec
            try:
                ravel_index = np.array([np.ravel_multi_index(mi,mShape,order='F') for mi in multi_index])
            except:
                print(mShape)
                print(ijkVStack)
                print('MatchDir')
                print(MatchDir)
                print(AimIndexVec)
                C.convertPyTree2File([constraint,MatchSurface],'match.cgns')
                sys.exit()

            Dx = mxr[ravel_index] - cx[:]
            Dy = myr[ravel_index] - cy[:]
            Dz = mzr[ravel_index] - cz[:]

            cdH[:] = normScal = np.sqrt(Dx**2+Dy**2+Dz**2)
            csx[:] = Dx / normScal
            csy[:] = Dy / normScal
            csz[:] = Dz / normScal

            # Export the values to ExtrudeLayer
            sx[PointListReceiver] = csx
            sy[PointListReceiver] = csy
            sz[PointListReceiver] = csz
            dH[PointListReceiver] = cdH

        cx[:] = x[PointListReceiver]
        cy[:] = y[PointListReceiver]
        cz[:] = z[PointListReceiver]

    ApplyConstraintsTime[0] += tic() - toc

def _keepConstrainedHeight(tExtru):
    '''
    This is a private function employed by user-function extrude() through
    private function _constrainedSmoothing()

    Its purpose is to migrate extrusion cell height {dH} from constraint zones
    towards corresponding points of extrusion surface front.

    INPUTS

    tExtru - (PyTree) - the extrusion PyTree
    '''
    toc = tic()
    ExtLayBase = I.getNodeFromName1(tExtru,'ExtrudeLayerBase')
    zE = I.getNodeFromName1(ExtLayBase,'ExtrudeLayer')
    FlowSolExtLay = I.getNodeFromName1(zE,'FlowSolution')
    dH = I.getNodeFromName1(FlowSolExtLay,'dH')[1]
    ConstraintWireframe = I.getNodeFromName1(tExtru,'ConstraintWireframe')
    for constraint in ConstraintWireframe[2]:
        ExtrusionDataNode = I.getNodeFromName1(constraint,'.ExtrusionData')
        PointListReceiver = I.getNodeFromName1(ExtrusionDataNode,'PointListReceiver')[1]
        kind = I.getValue(I.getNodeFromName1(ExtrusionDataNode,'kind'))
        if kind != 'match': continue
        cdH, = J.getVars(constraint,['dH'])
        dH[PointListReceiver] = cdH
    keepConstrainedHeight[0] += tic() - toc


def _transferDistributionData(tExtru,layer=0):
    '''
    This is a private function employed by the user-function extrude()

    The purpose of this function is to migrate the flow-fields contained in
    list Distributions provided by extrude() towards the extrusion front
    surface at the layer number provided by argument <layer>. In this manner,
    not only smoothing parameters are migrated from distributions to the
    extrusion front, but also any other auxiliary variable that user may want
    to employ or call during the extrusion process through the growthEquation
    law.

    The process of migrating 1D distributions towards the extrusion surface
    front is performed by a distance weighting technique.

    INPUTS

    tExtru - (PyTree) - the extrusion tree. It is modified in-place.

    layer - (integer) - the layer at which the flow fields are extracted.
    '''

    DistributionsBase = I.getNodeFromName1(tExtru,'DistributionsBase')
    ExtLayBase = I.getNodeFromName1(tExtru,'ExtrudeLayerBase')
    ExtrudeLayer = I.getNodeFromName1(ExtLayBase,'ExtrudeLayer')

    toc = tic()
    FieldsAndCoords = {'Point':[]}
    for dist in DistributionsBase[2]:

        # Compute Cell Height
        dH, = J.invokeFields(dist,['dH'])
        x, y, z = J.getxyz(dist)
        for i in range(len(x)-1):
            dH[i] = ((x[i+1]-x[i])**2+(y[i+1]-y[i])**2+(z[i+1]-z[i])**2)**0.5
        dH[-1] = dH[-2]

        FieldsNodes = I.getNodeFromName1(dist,'FlowSolution')[2]
        FieldsAndCoords['Point'] += [(x[layer], y[layer], z[layer])]
        for fn in FieldsNodes:
            try: FieldsAndCoords[fn[0]] += [fn[1][layer]]
            except KeyError: FieldsAndCoords[fn[0]] = [fn[1][layer]]
    Gathering[0] += tic() - toc


    toc = tic()
    FieldsNames = [fn for fn in FieldsAndCoords]
    FieldsNames.remove('Point')

    # Tranform Distribution field lists into numpy arrays
    for fn in FieldsNames:
        FieldsAndCoords[fn] = np.array(FieldsAndCoords[fn],order='F')

    WeightedFields = J.invokeFields(ExtrudeLayer,FieldsNames,'nodes:')
    NFlds     = len(FieldsNames)
    NDistribs = len(DistributionsBase[2])
    Transformation[0] += tic() - toc

    x,y,z = J.getxyz(ExtrudeLayer)

    # TODO: Vectorize from here:
    toc = tic()
    # for i in range(len(x)):
    #     Weights = [(x[i]-p[0])**2 + (y[i]-p[1])**2+(z[i]-p[2])**2 for p in FieldsAndCoords['Point']]
    #     Weights = 1./np.array(Weights)
    #     TotalWeights = np.sum(Weights)
    #
    #     # Weighted average of each distribution field
    #     for j in range(NFlds):
    #         WeightedFields[j][i] = (Weights).dot(FieldsAndCoords[FieldsNames[j]]) / TotalWeights
    ###################
    xd,yd,zd = [], [], []
    for p in FieldsAndCoords['Point']:
        xd.append(p[0])
        yd.append(p[1])
        zd.append(p[2])
    xd = np.array(xd)
    yd = np.array(yd)
    zd = np.array(zd)

    if len(xd) > 1:
        for j in range(NFlds):
            WeightedFields[j][:] = scipy.interpolate.Rbf(xd,yd,zd,FieldsAndCoords[FieldsNames[j]],kernel='multiquadric')(x,y,z)
    else:
        for j in range(NFlds):
            WeightedFields[j][:] = FieldsAndCoords[FieldsNames[j]]

    Weighting[0] += tic() - toc


def _transferExtractMeshData(ExtrudeLayer, extractMesh,
                             extractMeshHardSmoothOptions, layer):
    '''
    This is a private function employed by the user-function extrude()

    The purpose of this function is applying the extractMesh+T.smooth
    technique.

    INPUTS

    ExtrudeLayer - (zone) - the extrusion layer front (in-place modification)

    extractMesh - (zone) - the user-provided extractMesh zone from which
        flowfields are imposed. Provided to extrude() function

    extractMeshHardSmoothOptions - (Python dictionary) - the set of options
        provided to extrude() function.

    layer - (integer) - the layer number of the current extrusion process
    '''
    if extractMesh is not None:

        if layer>extractMeshHardSmoothOptions['HardSmoothLayerProtection']:
            EF = P.exteriorFaces(ExtrudeLayer)
            options=dict(fixedConstraints=[EF],
                        # projConstraints=[ExtrudeLayer], # This is too costly
                        )
            if 'eps' in extractMeshHardSmoothOptions:
                options['eps']=extractMeshHardSmoothOptions['eps']
            if 'niter' in extractMeshHardSmoothOptions:
                options['niter']=extractMeshHardSmoothOptions['niter']
            if 'type' in extractMeshHardSmoothOptions:
                options['type']=extractMeshHardSmoothOptions['type']

            ExtrudeLayerBeforeSmooth = I.copyTree(ExtrudeLayer)
            T._smooth(ExtrudeLayer,**options)
            T._projectOrtho(ExtrudeLayer, ExtrudeLayerBeforeSmooth)

        try: window=extractMeshHardSmoothOptions['window']
        except KeyError: window='jmin'

        extractMeshLayer = GSD.getBoundary(extractMesh, window=window,
                                           layer=layer)

        FactordH = extractMeshHardSmoothOptions['FactordH']
        wrapInitVars(extractMeshLayer,'dx=%g*{nodes:sx}*{nodes:dH}'%FactordH)
        wrapInitVars(extractMeshLayer,'dy=%g*{nodes:sy}*{nodes:dH}'%FactordH)
        wrapInitVars(extractMeshLayer,'dz=%g*{nodes:sz}*{nodes:dH}'%FactordH)
        UpExtract = T.deform(extractMeshLayer,vector=['dx','dy','dz'])
        wrapInitVars(extractMeshLayer,'dx=-%g*{nodes:dx}'%FactordH)
        wrapInitVars(extractMeshLayer,'dy=-%g*{nodes:dy}'%FactordH)
        wrapInitVars(extractMeshLayer,'dz=-%g*{nodes:dz}'%FactordH)
        DwExtract = T.deform(extractMeshLayer,vector=['dx','dy','dz'])
        extractMeshLayer = G.stack(UpExtract,DwExtract)

        ExtractedZone = P.extractMesh(extractMeshLayer, ExtrudeLayer,
                               extrapOrder=0, mode='accurate', constraint=40. )

        VarNames = C.getVarNames(extractMeshLayer, excludeXYZ=True, loc='nodes')[0]

        NumpyArraysExtraction = J.getVars(ExtractedZone,VarNames)
        NumpyArraysOriginal = J.invokeFields(ExtrudeLayer,VarNames)

        # Override original numpy arrays of ExtrudeLayer
        for i in range(len(VarNames)):
            NumpyArraysOriginal[i][:] = NumpyArraysExtraction[i]


def newBaseLayer(tExtru,layer=0):
    '''

    .. note:: this is a private function called by user-function :py:func:`extrude`

    The purpose of this function is to create an intermediary base employed
    to store the surface of the extrusion front surface for a given layer.
    This information will then be employed by ``_stackLayers`` for stacking zones
    and creating the final volume mesh.

    Parameters
    ----------

        tExtru : PyTree
            the extrusion tree

        layer : int
            the current layer number at which extrusion surface front
            will be stored

    Returns
    -------

        NewLayerBase : base
            new base containing the stored surface front
    '''

    toc = tic()

    InitSurfBase = I.getNodeFromName1(tExtru,'InitialSurface')
    ExtLayBase   = I.getNodeFromName1(tExtru,'ExtrudeLayerBase')
    ExtrudeLayer = I.getNodeFromName1(ExtLayBase,'ExtrudeLayer')

    xE = I.getNodeFromName2(ExtrudeLayer,'CoordinateX')[1].ravel(order='K')
    yE = I.getNodeFromName2(ExtrudeLayer,'CoordinateY')[1].ravel(order='K')
    zE = I.getNodeFromName2(ExtrudeLayer,'CoordinateZ')[1].ravel(order='K')



    NewLayerBase = I.copyTree(InitSurfBase)
    zones = I.getNodesFromType1(NewLayerBase,'Zone_t')
    for zone in zones:
        x = I.getNodeFromName2(zone,'CoordinateX')[1].ravel(order='K')
        y = I.getNodeFromName2(zone,'CoordinateY')[1].ravel(order='K')
        z = I.getNodeFromName2(zone,'CoordinateZ')[1].ravel(order='K')
        PointListDonor = I.getNodeFromName2(zone,'PointListDonor')[1]
        x[:] = xE[PointListDonor]
        y[:] = yE[PointListDonor]
        z[:] = zE[PointListDonor]
        zone[0] += '_layer%d'%layer
    NewLayerBase[0] += '_layer%d'%layer

    newBaseLayerTime[0] += tic() - toc

    return NewLayerBase



def _displaceLayer(tExtru):
    '''
    This is a private function employed by user-function extrude().

    Its purpose is to displace the extrusion surface front a distance of {dH}
    following the (smoothed) normals direction {sx}, {sy} and {sz}.

    INPUTS

    tExtru - (PyTree) - the extrusion pytree. It is modified in-place
    '''

    toc = tic()

    ExtLayBase   = I.getNodeFromName1(tExtru,'ExtrudeLayerBase')
    ExtrudeLayer = I.getNodeFromName1(ExtLayBase,'ExtrudeLayer')

    normalizeVector(ExtrudeLayer, ['sx', 'sy', 'sz'], container='FlowSolution')

    sx,sy,sz,dH = J.getVars(ExtrudeLayer,['sx','sy','sz','dH'])
    x,y,z = J.getxyz(ExtrudeLayer)
    x += sx*dH
    y += sy*dH
    z += sz*dH

    ConstWireBase   = I.getNodeFromName1(tExtru,'ConstraintWireframe')
    constraints = I.getNodesFromType1(ConstWireBase,'Zone_t')
    for constraint in constraints:

        sx,sy,sz,dH = J.getVars(constraint,['sx','sy','sz','dH'])
        x,y,z = J.getxyz(constraint)
        x += sx*dH
        y += sy*dH
        z += sz*dH

    DisplaceLayerTime[0] += tic() - toc

def _distanceBetweenSurfaces__(zone1, zone2):
    '''
    This is a private function, employed following the flow :
    User-level -> extrude()
        private -> _stackLayers()
            user-level -> stackUnstructured()

    Its purpose is to measure the point-by-point vector between two
    surfaces, and store the result as {PointDistanceXYZ} field in zone1.

    BEWARE that zone1 and zone2 must be point-by-point bijective

    INPUTS

    zone1 - (zone) - first surface (can be structured or not).
        Modified in place (a new field is added)

    zone2 - (zone) - second surface (can be structured or not) and must be
        bijective with respect to zone1 (as obtained from a deformation)
    '''
    x1 = I.getNodeFromName2(zone1,'CoordinateX')[1].ravel(order='K')
    y1 = I.getNodeFromName2(zone1,'CoordinateY')[1].ravel(order='K')
    z1 = I.getNodeFromName2(zone1,'CoordinateZ')[1].ravel(order='K')
    x2 = I.getNodeFromName2(zone2,'CoordinateX')[1].ravel(order='K')
    y2 = I.getNodeFromName2(zone2,'CoordinateY')[1].ravel(order='K')
    z2 = I.getNodeFromName2(zone2,'CoordinateZ')[1].ravel(order='K')
    PointDistanceX, PointDistanceY, PointDistanceZ = J.invokeFields(zone1, ['PointDistanceX', 'PointDistanceY', 'PointDistanceZ'])
    def _computeAndStoreDistance__(i):
        PointDistanceX[i] = x2[i]-x1[i]
        PointDistanceY[i] = y2[i]-y1[i]
        PointDistanceZ[i] = z2[i]-z1[i]

    # map(lambda i: _computeAndStoreDistance__(i), range(len(x1)))
    [_computeAndStoreDistance__(i) for i in range(len(x1))]



def stackUnstructured(ListOfZones2Stack):
    '''
    Stacks a list of unstructured zones.
    This shall be replaced by yet-to-be-implemented evolution of :py:func:`G.stack`.

    .. warning:: This does not work with NGON nor BAR element types

    Parameters
    ----------

        ListOfZones2Stack : :py:func:`list` of zone
            unstructured surfaces of the same element type.

    Returns
    -------

        StackedZone : zone
            resulting volume mesh after stacking surfaces
    '''

    Zones2Join = []
    isNGON = not not I.getNodeFromName1(ListOfZones2Stack[0],'NGonElements')
    if isNGON:
        raise ValueError('NGON stack to be implemented.')
    else:
        for iz in range(len(ListOfZones2Stack)-1):
            zone1 = ListOfZones2Stack[iz]
            zone2 = ListOfZones2Stack[iz+1]
            _distanceBetweenSurfaces__(zone1,zone2)
            Zones2Join += [G.grow(zone1,['PointDistanceX', 'PointDistanceY', 'PointDistanceZ'])]
            I._rmNodesByName(Zones2Join,'FlowSolution*')

    StackedZone = T.join(Zones2Join)

    return StackedZone



def _computeExtrusionIndicators(tExtru):
    toc = tic()
    ExtLayBase   = I.getNodeFromName1(tExtru,'ExtrudeLayerBase')
    ExtrudeLayer = I.getNodeFromName1(ExtLayBase,'ExtrudeLayer')

    G._getVolumeMap(ExtrudeLayer)
    _,_,_,_,DimExtrudeLayer = I.getZoneDim(ExtrudeLayer)
    if DimExtrudeLayer == 2:
        G._getNormalMap(ExtrudeLayer)
        G._getMaxLength(ExtrudeLayer)
    elif DimExtrudeLayer == 1:
        W.getCurveNormalMap(ExtrudeLayer)
        C._initVars(ExtrudeLayer,'centers:MaxLength={centers:vol}')
    else:
        raise TypeError('ExtrudeLayer shall be a surface or a curve, not a volume mesh.')

    for fieldname in ['vol','sx','sy','sz','MaxLength']:
        C.center2Node__(ExtrudeLayer,'centers:'+fieldname,cellNType=0)
    I._rmNodesFromName(ExtrudeLayer,I.__FlowSolutionCenters__)
    normalizeVector(ExtrudeLayer,['sx','sy','sz'],container='FlowSolution')

    # TODO why here ?
    # dH = I.getNodeFromName2(ExtrudeLayer,'dH')[1]
    _extrusionApplyConstraints(tExtru)

    tempExtLayer = I.copyRef(ExtrudeLayer)

    # add convexity/concavity indicator "convexity"
    tempExtLayer = P.computeGrad(tempExtLayer,'sx')
    tempExtLayer = P.computeGrad(tempExtLayer,'sy')
    tempExtLayer = P.computeGrad(tempExtLayer,'sz')
    wrapInitVars(tempExtLayer,'centers:convexity={centers:gradxsx}+{centers:gradysy}+{centers:gradzsz}')
    C.center2Node__(tempExtLayer,'centers:convexity',cellNType=0)
    FlowSolTemp = I.getNodeFromName1(tempExtLayer,I.__FlowSolutionNodes__)
    convexity = I.getNodeFromName1(FlowSolTemp,'convexity')[1]
    FlowSol = I.getNodeFromName1(ExtrudeLayer,I.__FlowSolutionNodes__)
    I.createUniqueChild(FlowSol, 'convexity', 'DataArray_t', value=convexity)

    # add stretching indicator
    tempExtLayer = P.computeGrad(ExtrudeLayer,'vol')
    tempExtLayer = P.computeGrad(tempExtLayer,'centers:gradxvol')
    tempExtLayer = P.computeGrad(tempExtLayer,'centers:gradyvol')
    tempExtLayer = P.computeGrad(tempExtLayer,'centers:gradzvol')
    C.center2Node__(tempExtLayer,'centers:gradxvol',cellNType=0)
    C.center2Node__(tempExtLayer,'centers:gradyvol',cellNType=0)
    C.center2Node__(tempExtLayer,'centers:gradzvol',cellNType=0)
    C.center2Node__(tempExtLayer,'centers:gradxgradxvol',cellNType=0)
    C.center2Node__(tempExtLayer,'centers:gradygradyvol',cellNType=0)
    C.center2Node__(tempExtLayer,'centers:gradzgradzvol',cellNType=0)
    FlowSolTemp = I.getNodeFromName1(tempExtLayer,'FlowSolution')
    MaxLength = I.getNodeFromName1(FlowSolTemp,'MaxLength')[1]
    gradxvol = I.getNodeFromName1(FlowSolTemp,'gradxvol')[1]
    gradyvol = I.getNodeFromName1(FlowSolTemp,'gradyvol')[1]
    gradzvol = I.getNodeFromName1(FlowSolTemp,'gradzvol')[1]
    gradxgradxvol = I.getNodeFromName1(FlowSolTemp,'gradxgradxvol')[1]
    gradygradyvol = I.getNodeFromName1(FlowSolTemp,'gradygradyvol')[1]
    gradzgradzvol = I.getNodeFromName1(FlowSolTemp,'gradzgradzvol')[1]
    stretching = gradxgradxvol+gradygradyvol+gradzgradzvol
    I.createUniqueChild(FlowSol, 'stretching', 'DataArray_t', value=stretching)
    dsx = gradxvol*MaxLength
    dsy = gradyvol*MaxLength
    dsz = gradzvol*MaxLength
    I.createUniqueChild(FlowSol, 'dsx', 'DataArray_t', value=dsx)
    I.createUniqueChild(FlowSol, 'dsy', 'DataArray_t', value=dsy)
    I.createUniqueChild(FlowSol, 'dsz', 'DataArray_t', value=dsz)



    computeExtrusionIndicatorsTime[0] += tic() - toc

def _stackLayers(tExtru, AllLayersBases):
    '''
    This is a private function employed by user-level function extrude()

    Its purpose is to interfacing the multi-block and multi-topology stacking of
    surfaces stored at each layer, in order to construct the final extruded
    volume mesh, and add it to tExtru

    INPUTS

    tExtru - (PyTree) - extrusion PyTree. ExtrudedVolume base is added.

    AllLayersBases - (list of bases) - list of bases as appended from
        newBaseLayer() function
    '''


    toc = tic()
    tB = C.newPyTree(AllLayersBases)
    AllBases = I.getNodesFromType1(tB,'CGNSBase_t')

    # Compute number of Stacks
    baseDummy = I.getNodeFromType1(tB,'CGNSBase_t')
    NStacks = len(I.getNodesFromType1(baseDummy,'Zone_t'))

    StackedZones = []
    for nz in range(NStacks):
        # ListOfZones2Stack = map(lambda l: I.getNodesFromType1(l,'Zone_t')[0], AllBases)
        ListOfZones2Stack = [I.getNodesFromType1(l,'Zone_t')[0] for l in AllBases]

        TypeZone,Ni,Nj,Nk, DimZone = I.getZoneDim(ListOfZones2Stack[0])

        if TypeZone == 'Structured':
            if Nj == Nk == 1:
                stacked = GSD.stackSections(ListOfZones2Stack)
            else:
                stacked = G.stack(ListOfZones2Stack,None)

        else:
            if DimZone==2:
                stacked = stackUnstructured(ListOfZones2Stack)
            else:
                ListOfZones2Stack = [C.convertBAR2Struct(z2stk) for z2stk in  ListOfZones2Stack]
                stacked = G.stack(ListOfZones2Stack,None)
        stacked[0] = ListOfZones2Stack[0][0]
        StackedZones += [stacked]

        # release memory
        [I.rmNode(tB, I.getNodesFromType1(l,'Zone_t')[0]) for l in AllBases]

    ExtrudedVolumeBase = I.getNodeFromName1(tExtru,'ExtrudedVolume')
    for s in StackedZones: I.addChild(ExtrudedVolumeBase,s)
    StackLayersTime[0] = tic() - toc


def extrudeSurfaceFollowingCurve(surface, curve):
    '''
    Extrude a **surface** following the direction defined by a 3D structured
    **curve**.

    Parameters
    ----------

        surface : zone
            surface to be extruded

        curve : zone
            curve used for extrusion

    Returns
    -------

        t : PyTree
            the new PyTree containing the new volume zone
    '''

    xCurve, yCurve, zCurve = J.getxyz(curve)
    dxV = xCurve[1:]-xCurve[0]
    dyV = yCurve[1:]-yCurve[0]
    dzV = zCurve[1:]-zCurve[0]

    VolumeComponents = []
    for zone in I.getZones(surface):
        Layers = [T.translate(zone,(dx,dy,dz)) for dx, dy, dz in zip(dxV, dyV, dzV)]
        Stacked = G.stack(Layers)
        Stacked[0] = zone[0]
        VolumeComponents.append( Stacked )

    t = C.newPyTree(['extruded',VolumeComponents])

    return t

def invokeExtrusionPyTree(tSurf):
    '''
    This is a private function called by user-level function :py:func:`extrude`

    This function invokes the extrusion PyTree **tExtru**.

    Parameters
    ----------

        tSurf : PyTree, base, zone or list of zones
            the surfaces to extrude

    Returns
    -------

        tExtru : PyTree
            the extrusion tree employed during the extrusion process
    '''
    tSurf = I.rmNodesFromType(tSurf, 'FlowSolution_t')
    tExtru = C.newPyTree(['InitialSurface',I.getZones(tSurf),
                     'ExtrudedVolume',     [],
                     'ExtrudeLayerBase',   [],
                     'ConstraintWireframe',[],
                     'ConstraintSurfaces', [],
                     'DistributionsBase',[],
                     'HardSmoothPoints',[]])

    return tExtru



def _addExtrusionLayerSurface(tExtru, mode='TRI', closeExtrusionLayer=False):
    '''
    This is a private function called by user-level function extrude()

    Its purpose is to transform the user-provided surfaces into a single
    auxiliar surface suitable employed as extrusion front.

    INPUTS

    tExtru - (PyTree) - the extrusion tree where the extrusion front will be
        stored

    mode - (string) - cell type of the auxiliar extrusion font.

    closeExtrusionLayer - (boolean) - if True, attempt to close the new auxiliar
        surface.
    '''
    InitialSurfaceBase = I.getNodeFromName1(tExtru,'InitialSurface')
    ExtrudeLayerBase = I.getNodeFromName1(tExtru,'ExtrudeLayerBase')

    if mode == 'NGON':
        raise ValueError('Cannot use "NGON". This will be implemented in future.')
        # tNG = C.convertArray2NGon(InitialSurfaceBase)
    elif mode == 'TRI':
        tNG = C.convertArray2Tetra(InitialSurfaceBase)
    elif mode == 'QUAD':
        tNG = C.convertArray2Hexa(InitialSurfaceBase)
    elif mode == 'ORIGINAL':
        tNG = I.copyTree(InitialSurfaceBase)
    else:
        raise ValueError('Extrusion auxiliar cell type "%s" not implemented.'%mode)

    zones = I.getNodesFromType1(tNG,'Zone_t')
    tNG = T.join(zones)
    if closeExtrusionLayer: G._close(tNG)
    ExtrudeLayer = I.getNodeFromType2(tNG,'Zone_t')
    ExtrudeLayer[0] = 'ExtrudeLayer'
    I.addChild(ExtrudeLayerBase,ExtrudeLayer)

    # Connect ExtrudeLayer with InitialLayer
    InitialSurfaceZones = I.getNodesFromType1(InitialSurfaceBase,'Zone_t')
    for initialZone in InitialSurfaceZones:
        # Invoke .ExtrusionData Node where attributes are stored
        ExtrusionData = I.createUniqueChild(initialZone, '.ExtrusionData', 'UserDefinedData_t',value=None,children=None)
        I.createUniqueChild(ExtrusionData,'kind','DataArray_t',value='Match')
        xC = I.getNodeFromName2(initialZone,'CoordinateX')[1].ravel(order='K')
        yC = I.getNodeFromName2(initialZone,'CoordinateY')[1].ravel(order='K')
        zC = I.getNodeFromName2(initialZone,'CoordinateZ')[1].ravel(order='K')
        ZoneNPts = len(xC)

        _,_,_,_,DimZone = I.getZoneDim(initialZone)

        if DimZone == 1:
            # TODO: Improve this strategy!
            PointListDonor = np.arange(ZoneNPts,dtype=np.int64)
        else:
            # TupplesPointsList = map(lambda i: (xC[i],yC[i],zC[i]) ,range(ZoneNPts))
            TupplesPointsList = [(xC[i],yC[i],zC[i]) for i in range(ZoneNPts)]
            Res = D.getNearestPointIndex(ExtrudeLayer,TupplesPointsList)
            # Indices = map(lambda r: r[0], Res)
            Indices = [r[0] for r in Res]
            PointListDonor = np.array(Indices, dtype=np.int64,order='F')

        # anyRepeatedPoint = np.any(map(lambda plr: plr-PointListDonor == 0,PointListDonor))
        # anyRepeatedPoint = np.any([plr-PointListDonor == 0 for plr in PointListDonor])
        # if anyRepeatedPoint:
        #     print('WARNING: addExtrusionConstraint(): Multiply defined match for initialZone zone "%s"'%initialZone[0])
        I.createUniqueChild(ExtrusionData,'PointListDonor','DataArray_t',value=PointListDonor)



def addExtrusionConstraint(tExtru, kind='Imposed', curve=None, surface=None,
        ProjectionMode='ortho', ProjectionDir=None, ProjectionCenter=None,
        ProjectionAxis=None, MatchDir=None,
        pointsToCopy=None, RotationAxis=None, RotationAngle=None, RotationCenter=None):
    '''
    This is a private function called by user-level function :py:func:`extrude`

    Its purpose is to add the constraints surfaces and curves as well as
    characteristics, as CGNS data into **tExtru** pytree.

    Parameters
    ----------

        kind : str
            kind of constraint to be processed into **tExtru**. Possible values:

            * ``'Imposed'``
                impose the extrusion direction following the fields
                ``{sx}``, ``{sy}`` and ``{sz}`` defined in user-provided zone through
                attribute **curve**

            * ``'Initial'``
                like ``'Imposed'``, except that instead of using the ``{sx}``,
                ``{sy}`` and ``{sz}`` fields of the user-provided zone **curve**, it uses
                the local normals of the extrusion front, and avoid to smooth them

            * ``'Projected'``
                constraint the closest points of the extrusion front
                with respect to the provided curve towards a user-provided
                surface.

            * ``'Copy'``
                Copy the fields at points near to **curve** using fields defined
                at **copyCurve**

            * ``'CopyAndRotate'``
                Copy the fields at points near to **curve** using fields defined
                at **copyCurve** and apply a rotation of the normals using
                **RotationAxis**, **RotationCenter** and **RotationAngle**

            * ``'Match'``
                Exactly follow a neighbor point set.

        curve : zone
            curve that will serve to define the points of the
            extrusion front where constraint is applied.

        surface : zone
            auxiliar surface employed by different kinds of
            constraints (e.g. ``'Projected'`` or ``'Match'``)

        ProjectionMode : str
            Accepted values:

            * ``'ortho'``
                projection mode is orthogonal

            * ``'dir'``
                projection mode is directional

            * ``'CylinderRadial'``
                projection mode is radial using cylinder, hence
                **ProjectionCenter** and **ProjectionAxis** must be
                specified

            .. note:: only relevant if **kind** == ``'Projected'``

        ProjectionDir : :py:class:`tuple` containing 3 :py:class:`float`
            if **ProjectionMode** == ``'dir'``, then this
            attribute indicates the direction of projection, that will remain
            constant throughout the entire extrusion process.

        ProjectionCenter : :py:class:`tuple` containing 3 :py:class:`float`
            if **ProjectionMode** == ``'CylinderRadial'``, then this
            attribute indicates the coordinates of the passing point of the
            cylindrical projection axis

        ProjectionAxis : :py:class:`tuple` containing 3 :py:class:`float`
            if **ProjectionMode** == ``'CylinderRadial'``, then this
            attribute indicates the unitary vector pointing towards the
            cylindrical projection axis

        MatchDir : int
            If procided, it specifies the front-advance index of matching
            (1=i, 2=j, 3=k) of the **surface**. If not provided (:py:obj:`None`),
            then the matching direction is automatically computed.

            .. note:: only relevant if **kind** == ``'Match'``

        pointsToCopy : zone
            set the points used to extract fields from the extrusion front (normals and
            height) and used to be replicated elsewhere using constraint
            **kind** == ``'Copy'`` or **kind** == ``'CopyAndRotate'``

        RotationAxis : :py:class:`list` or :py:class:`tuple` or numpy array of 3 :py:class:`float`
            rotation axis used for the transformation of the extrusion normals

            .. note:: only relevant if **kind** == ``'CopyAndRotate'``

        RotationCenter : :py:class:`list` or :py:class:`tuple` or numpy array of 3 :py:class:`float`
            center of rotation used for the transformation of the extrusion normals

            .. note:: only relevant if **kind** == ``'CopyAndRotate'``

        RotationAngle : :py:class:`list` or :py:class:`tuple` or numpy array of 3 :py:class:`float`
            angle of rotation used for the transformation of the extrusion normals

            .. note:: only relevant if **kind** == ``'CopyAndRotate'``

    '''


    ExtrudeLayer = I.getNodeFromNameAndType(tExtru,'ExtrudeLayer','Zone_t')

    if curve is not None: curve = I.copyTree(curve)
    if pointsToCopy is not None: pointsToCopy = I.copyTree(pointsToCopy)

    if kind in ('Imposed', 'Initial'):
        if not curve:
            raise AttributeError('Extrusion kind=%s: curve was not provided.'%kind)
        PossibleFields = ['sx','sy','sz']
        silence = J.OutputGrabber()
        with silence: flds = J.getVars(curve,PossibleFields)
        if any([f is None for f in flds]):
            print(J.WARN+'warning: "%s" constraint shall be triggered invoking at least one of "sx","sy","sz" fields in provided constraint curve'%kind+J.ENDC)

        # Add curve to Extrusion PyTree:
        ConstraintWireframeBase = I.getNodeFromNameAndType(tExtru,'ConstraintWireframe','CGNSBase_t')
        ExistingZonesNames = [z[0] for z in ConstraintWireframeBase[2]]
        if curve[0] in ExistingZonesNames:
            item = 0
            NewZoneName = '%s.%d'%(curve[0],item)
            while NewZoneName in ExistingZonesNames:
                item+=1
                NewZoneName = '%s.%d'%(curve[0],item)
            curve[0] = NewZoneName

        I.addChild(ConstraintWireframeBase,curve)

        # Invoke .ExtrusionData Node where attributes are stored
        ExtrusionData = I.createUniqueChild(curve, '.ExtrusionData', 'UserDefinedData_t',value=None,children=None)
        I.createUniqueChild(ExtrusionData,'kind','DataArray_t',value=kind)

    elif kind == 'Projected':
        if not curve:
            raise AttributeError('Extrusion kind=%s: curve was not provided.'%kind)
        if not surface and ProjectionMode in ('dir', 'ortho'):
            raise AttributeError('ProjectionMode=%s : surface was not provided.'%ProjectionMode)
        # Add data to Extrusion PyTree:
        ConstraintWireframeBase = I.getNodeFromNameAndType(tExtru,'ConstraintWireframe','CGNSBase_t')
        ExistingZonesNames = [z[0] for z in ConstraintWireframeBase[2]]
        if curve[0] in ExistingZonesNames:
            item = 0
            NewZoneName = '%s.%d'%(curve[0],item)
            while NewZoneName in ExistingZonesNames:
                item+=1
                NewZoneName = '%s.%d'%(curve[0],item)
            curve[0] = NewZoneName

        I.addChild(ConstraintWireframeBase,curve)



        ConstraintSurfacesBase = I.getNodeFromName1(tExtru,'ConstraintSurfaces')
        if surface:
            ConstraintSurfacesBase[2].append(surface)
            I._correctPyTree(ConstraintSurfacesBase,level=3)

        # Invoke .ExtrusionData Node where attributes are stored
        ExtrusionData = dict(kind=kind, ProjectionMode=ProjectionMode)
        if ProjectionMode in ('dir','ortho'):
            ExtrusionData['ProjectionSurfaceName'] = surface[0]
        elif ProjectionMode == 'CylinderRadial':
            if ProjectionCenter is None:
                raise AttributeError('CylinderRadial projection requires "ProjectionCenter" argument')
            ProjectionCenter = np.array(ProjectionCenter,dtype=np.float64)
            ExtrusionData['ProjectionCenter'] = ProjectionCenter
            if ProjectionAxis is None:
                raise AttributeError('CylinderRadial projection requires "ProjectionAxis" argument')
            ProjectionAxis = np.array(ProjectionAxis,dtype=np.float64)
            ProjectionAxis/= np.sqrt(ProjectionAxis.dot(ProjectionAxis))
            ExtrusionData['ProjectionAxis'] = ProjectionAxis
        J.set(curve,'.ExtrusionData',**ExtrusionData)

    elif kind == 'Copy':
        if not curve:
            raise AttributeError('constraint kind=%s: curve was not provided.'%kind)
        if not pointsToCopy:
            raise AttributeError('constraint kind=%s: pointsToCopy was not provided.'%kind)

        ConstraintWireframeBase = I.getNodeFromName1(tExtru,'ConstraintWireframe')
        ExistingZonesNames = [z[0] for z in ConstraintWireframeBase[2]]

        NewZoneName = curve[0]
        # print('curve proposal : %s'%NewZoneName)
        # print('curve existing : %s'%str(ExistingZonesNames))
        if NewZoneName in ExistingZonesNames:
            item = 0
            NewZoneName = '%s.%d'%(curve[0],item)
            while NewZoneName in ExistingZonesNames:
                item+=1
                NewZoneName = '%s.%d'%(curve[0],item)
        curve[0] = NewZoneName
        ExistingZonesNames.append(NewZoneName)
        # print('curve corrected proposal : %s'%NewZoneName)
        I.addChild(ConstraintWireframeBase,curve)

        # curve is the DESTINATION, pointsToCopy is the SOURCE
        xC, yC, zC = J.getxyz(pointsToCopy)
        xC = np.ravel(xC, order='F')
        yC = np.ravel(yC, order='F')
        zC = np.ravel(zC, order='F')
        CurveNPts = len(xC)
        Indices = [J.getNearestPointIndex(ExtrudeLayer,(xC[i],yC[i],zC[i]))[0] for i in range(CurveNPts)]
        PointListDonor = np.array(Indices, dtype=np.int64,order='F')
        if  any(np.diff(PointListDonor)==0):
            print('WARNING: addExtrusionConstraint(): Multiply defined constraint for single curve "%s"'%pointsToCopy[0])


        NewZoneName = pointsToCopy[0]
        # if NewZoneName.endswith('.pt')
        # print('point proposal : %s'%NewZoneName)
        # print('point existing : %s'%str(ExistingZonesNames))
        # if NewZoneName in ExistingZonesNames:
        #     item = 0
        #     NewZoneName = '%s.%d'%(pointsToCopy[0],item)
        #     while NewZoneName in ExistingZonesNames:
        #         item+=1
        #         NewZoneName = '%s.%d'%(pointsToCopy[0],item)
        # pointsToCopy[0] = NewZoneName
        # ExistingZonesNames.append(NewZoneName)
        # print('point corrected proposal : %s'%NewZoneName)

        J.set(curve,'.ExtrusionData', kind=kind, pointsToCopyZoneName=pointsToCopy[0],
                                      PointListDonor=PointListDonor)

        J.set(pointsToCopy,'.ExtrusionData', kind='None', PointListReceiver=PointListDonor)

    elif kind == 'CopyAndRotate':
        if not curve:
            raise AttributeError('Extrusion kind=%s: curve was not provided.'%kind)
        if not pointsToCopy:
            raise AttributeError('Extrusion kind=%s: pointsToCopy was not provided.'%kind)
        if not RotationCenter:
            raise AttributeError('constraint kind=%s: RotationCenter was not provided.'%kind)
        if not RotationAxis:
            raise AttributeError('constraint kind=%s: RotationAxis was not provided.'%kind)
        if not RotationAngle:
            raise AttributeError('constraint kind=%s: RotationAngle was not provided.'%kind)

        ConstraintWireframeBase = I.getNodeFromName1(tExtru,'ConstraintWireframe')
        ExistingZonesNames = [z[0] for z in ConstraintWireframeBase[2]]

        NewZoneName = curve[0]
        if NewZoneName in ExistingZonesNames:
            item = 0
            NewZoneName = '%s.%d'%(curve[0],item)
            while NewZoneName in ExistingZonesNames:
                item+=1
                NewZoneName = '%s.%d'%(curve[0],item)
        curve[0] = NewZoneName
        ExistingZonesNames.append(NewZoneName)
        I.addChild(ConstraintWireframeBase,curve)

        # curve is the DESTINATION, pointsToCopy is the SOURCE
        xC, yC, zC = J.getxyz(pointsToCopy)
        xC = np.ravel(xC, order='F')
        yC = np.ravel(yC, order='F')
        zC = np.ravel(zC, order='F')
        CurveNPts = len(xC)
        Indices = [J.getNearestPointIndex(ExtrudeLayer,(xC[i],yC[i],zC[i]))[0] for i in range(CurveNPts)]
        PointListDonor = np.array(Indices, dtype=np.int64,order='F')
        if  any(np.diff(PointListDonor)==0):
            print('WARNING: addExtrusionConstraint(): Multiply defined constraint for single curve "%s"'%pointsToCopy[0])


        NewZoneName = pointsToCopy[0]
        if NewZoneName in ExistingZonesNames:
            item = 0
            NewZoneName = '%s.%d'%(pointsToCopy[0],item)
            while NewZoneName in ExistingZonesNames:
                item+=1
                NewZoneName = '%s.%d'%(pointsToCopy[0],item)
        pointsToCopy[0] = NewZoneName
        ExistingZonesNames.append(NewZoneName)

        J.set(curve,'.ExtrusionData', kind=kind, pointsToCopyZoneName=pointsToCopy[0],
                    RotationCenter=RotationCenter, RotationAxis=RotationAxis,
                    RotationAngle=RotationAngle, PointListDonor=PointListDonor)

        J.set(pointsToCopy,'.ExtrusionData', kind='None', PointListReceiver=PointListDonor)


    elif kind == 'Match':
        if not curve:
            raise AttributeError('Extrusion kind=%s: curve was not provided.'%kind)
        if not surface:
            raise AttributeError('Extrusion kind=%s: surface was not provided.'%kind)
        # Add data to Extrusion PyTree:
        ConstraintWireframeBase = I.getNodeFromName1(tExtru,'ConstraintWireframe')
        # ExistingZonesNames = map(lambda z: z[0],ConstraintWireframeBase[2])
        ExistingZonesNames = [z[0] for z in ConstraintWireframeBase[2]]
        if curve[0] in ExistingZonesNames:
            item = 0
            NewZoneName = '%s.%d'%(curve[0],item)
            while NewZoneName in ExistingZonesNames:
                item+=1
                NewZoneName = '%s.%d'%(curve[0],item)
            curve[0] = NewZoneName

        I.addChild(ConstraintWireframeBase,curve)



        ConstraintSurfacesBase = I.getNodeFromName1(tExtru,'ConstraintSurfaces')
        # ExistingZonesNames = map(lambda z: z[0],ConstraintSurfacesBase[2])
        ExistingZonesNames = [z[0] for z in ConstraintSurfacesBase[2]]
        if surface[0] in ExistingZonesNames:
            item = 0
            NewZoneName = '%s.%d'%(surface[0],item)
            while NewZoneName in ExistingZonesNames:
                item+=1
                NewZoneName = '%s.%d'%(surface[0],item)
            surface[0] = NewZoneName

        I.addChild(ConstraintSurfacesBase,surface)

        # Invoke .ExtrusionData Node where attributes are stored
        ExtrusionData = I.createUniqueChild(curve, '.ExtrusionData', 'UserDefinedData_t',value=None,children=None)
        I.createUniqueChild(ExtrusionData,'kind','DataArray_t',value=kind)
        I.createUniqueChild(ExtrusionData,'MatchSurfaceName','DataArray_t',value=surface[0])
        if MatchDir is not None: I.createUniqueChild(ExtrusionData,'MatchDir','DataArray_t',value=MatchDir)

    # Set connection of curve with ExtrudeLayer
    xC, yC, zC = J.getxyz(curve)
    xC = np.ravel(xC, order='F')
    yC = np.ravel(yC, order='F')
    zC = np.ravel(zC, order='F')
    CurveNPts = len(xC)
    Indices = [J.getNearestPointIndex(ExtrudeLayer,(xC[i],yC[i],zC[i]))[0] for i in range(CurveNPts)]
    PointListReceiver = np.array(Indices, dtype=np.int64,order='F')
    if  any(np.diff(PointListReceiver)==0):
        print('WARNING: addExtrusionConstraint(): Multiply defined constraint for single curve "%s"'%curve[0])

    ExtrusionDataNode = I.getNodeFromName1(curve,'.ExtrusionData')
    I.createUniqueChild(ExtrusionDataNode,'PointListReceiver','DataArray_t',value=PointListReceiver)



def _addHardSmoothPoints(tExtru, HardSmoothPoints):
    '''
    Private function called by user-level function extrude()

    Its function consist in migrating the user-provided hard smoothing points
    into the main <tExtru> extrusion tree

    INPUTS

    tExtru - (PyTree) - extrusion PyTree. It is modified.

    HardSmoothPoints - (list of zones) - Points where hard smoothing (T.smooth)
        of the extrusion front will be applied. Each point may have FlowSolution
        fields that will be parsed and sent to T.smooth(**options) as **kwargs.
    '''
    base = I.getNodeFromNameAndType(tExtru,'HardSmoothPoints','CGNSBase_t')
    base[2] += HardSmoothPoints



def _addExtrusionDistribution(tExtru, DistributionZone):
    '''
    Private function called by user-level function extrude()

    Its purpose consist in adding a distribution zone into the extrusion tree
    <tExtru>

    INPUTS

    tExtru - (PyTree) - extrusion tree where the distrbution zone will be added

    DistributionZone - (PyTree) - Distribution zone as provided to extrude()
    '''
    DistributionsBase = I.getNodeFromNameAndType(tExtru,'DistributionsBase','CGNSBase_t')

    # Check if all data is present in Distribution
    RequiredFields = ['normaliters','normalfactor','growthfactor','growthiters']
    RequiredFields = []
    try:
        flds = J.getVars(DistributionZone,RequiredFields)
    except:
        raise AttributeError('_addExtrusionDistribution(): DistributionZone named %s must have a FlowSolution containing: %s'%(DistributionZone[0],str(RequiredFields)))
    for i in range(len(RequiredFields)):
        if flds[i] is None:
            raise AttributeError('_addExtrusionDistribution(): DistributionZone named %s must have a FlowSolution named %s.'%(DistributionZone[0],RequiredFields[i]))

    # Add Distribution to Extrusion PyTree:
    # ExistingZonesNames = map(lambda z: z[0],DistributionsBase[2])
    ExistingZonesNames = [z[0] for z in DistributionsBase[2]]
    if DistributionZone[0] in ExistingZonesNames:
        item = 0
        NewZoneName = '%s.%d'%(DistributionZone[0],item)
        while NewZoneName in ExistingZonesNames:
            item+=1
            NewZoneName = '%s.%d'%(DistributionZone[0],item)
        DistributionZone[0] = NewZoneName

    I.addChild(DistributionsBase,DistributionZone)



def stackSections(Sections):
    """
    .. warning:: this function must be replaced by ``G.stack()``
    """
    Nk = len(Sections)
    Ni, Nj = J.getx(Sections[0]).shape
    Volume = G.cart((0,0,0),(1,1,1),(Ni,Nj,Nk))
    SurfX, SurfY, SurfZ = J.getxyz(Volume)
    for k in range(Nk):
        SecX, SecY, SecZ = J.getxyz(Sections[k])
        SurfX[:,:,k] = SecX
        SurfY[:,:,k] = SecY
        SurfZ[:,:,k] = SecZ
    return Volume



def multiSections(ProvidedSections, SpineDiscretization,InterpolationData={'InterpolationLaw':'interp1d_linear'}):
    '''
    This function makes a sweep across a list of provided sections (surfaces)
    that are exactly placed in 3D space (passing points).

    Parameters
    ----------

        ProvidedSections : :py:class:`list` of zone
            each zone must be a 1D structured surface.
            All provided sections must have the same number of points. Also, they
            shall have the same index ordering, in order to avoid self-intersecting
            resulting surface. Each one of the provided sections must be exactly
            placed in 3D space at the passing points where the new surface will be
            pass across. Bad results can be expected if some sections are coplanar.

        SpineDiscretization : numpy 1d array or zone
            This is a polymorphic
            argument that provides information on how to discretize the spine of
            the surface to build. If it is a numpy 1D, it shall be a monotonically
            increasing vector between 0 and 1. If it is a zone, it must be a 1D
            structured curve, and the algorithm will extract its distribution for
            use it as SpineDiscretization.

        InterpolationData : dict
            This is a dictionary that contains
            options for the interpolation process. Relevant options:

            * InterpolationLaw : str
                Indicates the interpolation law to be
                employed when constructing the surface. Interpolation is performed
                index-by-index of each point of the provided surface coordinates.
                The interpolation abscissa is the **SpineDiscretization**,
                whereas the
                interpolated quantities are the grid coordinates.

    Returns
    -------

        Volume : zone
            3D structured grid of the volume that passes across
            all the user-provided sections.

        SpineCurve : zone
            1D structured curve corresponding to the spine of the volume.
    '''
    AllowedInterpolationLaws = ('interp1d_<KindOfInterpolation>', 'pchip', 'akima', 'cubic')

    # Construct Spine
    # Barycenters = map(lambda s: G.barycenter(s), ProvidedSections)
    Barycenters = [G.barycenter(s) for s in ProvidedSections]
    SpineCurve  = D.polyline(Barycenters)
    RelPositions= W.gets(SpineCurve)
    # Verify SpineDiscretization argument
    typeSpan=type(SpineDiscretization)
    if I.isStdNode(SpineDiscretization) == 0: # It is a node
        try:
            s=J.gets(SpineDiscretization)
            Ns = len(s)
        except:
            ErrMsg = "multiSections(): SpineDiscretization argument was a PyTree node (named %s), but I could not obtain the CurvilinearAbscissa.\nPerhaps you forgot GridCoordinates nodes?"%SpineDiscretization[0]
            raise AttributeError(ErrMsg)

    elif typeSpan is np.ndarray: # It is a numpy array
        s  = SpineDiscretization
        if len(s.shape)>1:
            ErrMsg = "multiSections(): SpineDiscretization argument was detected as a numpy array of dimension %g!\nSpan MUST be a monotonically increasing VECTOR (1D numpy array) and between [0,1] interval."%len(s.shape)
            raise AttributeError(ErrMsg)
        Ns = s.shape[0]
        if any( np.diff(s)<0):
            ErrMsg = "multiSections(): SpineDiscretization argument was detected as a numpy array.\nHowever, it was NOT monotonically increasing. SpineDiscretization MUST be monotonically increasing and between [0,1] interval. Check that, please."
            raise AttributeError(ErrMsg)
        if any(s>1) or any(s<0):
            ErrMsg = "multiSections(): SpineDiscretization argument was detected as a numpy array.\nHowever, it was NOT between [0,1] interval. Check that, please."
            raise AttributeError(ErrMsg)
    elif isinstance(SpineDiscretization, list): # It is a list
        if isinstance(SpineDiscretization[0], dict):
            SpineCurve  = W.polyDiscretize(SpineCurve, SpineDiscretization)
            s  = W.gets(SpineCurve)
            Ns = len(s)
        else:
            try:
                s = np.array(SpineDiscretization,dtype=np.float64)
            except:
                ErrMsg = 'multiSections(): Could not transform SpineDiscretization argument into a numpy array.\nCheck your SpineDiscretization argument.\n'
                raise AttributeError(ErrMsg)
            if len(s.shape)>1:
                ErrMsg = "multiSections(): SpineDiscretization argument was detected as a numpy array of dimension %g!\nSpan MUST be a monotonically increasing VECTOR (1D numpy array) and between [0,1] interval."%len(s.shape)
                raise AttributeError(ErrMsg)
            Ns = s.shape[0]
            if any( np.diff(s)<0):
                ErrMsg = "multiSections(): SpineDiscretization argument was detected as a numpy array.\nHowever, it was NOT monotonically increasing. SpineDiscretization MUST be monotonically increasing and between [0,1] interval. Check that, please."
                raise AttributeError(ErrMsg)
            if any(s>1) or any(s<0):
                ErrMsg = "multiSections(): SpineDiscretization argument was detected as a numpy array.\nHowever, it was NOT between [0,1] interval. Check that, please."
                raise AttributeError(ErrMsg)
    else:
        raise AttributeError('multiSections(): Type of SpineDiscretization argument not recognized. Check your input.')




    NPtsI,NPtsJ = J.getx(ProvidedSections[0]).shape

    Sections = [G.cart((0,0,0),(1,1,1),(NPtsI,NPtsJ,1)) for k in range(Ns)] # Invoke all sections

    NinterFoils = len(ProvidedSections)
    InterpXmatrix = np.zeros((NinterFoils,NPtsI,NPtsJ),dtype=np.float64,order='F')
    InterpYmatrix = np.zeros((NinterFoils,NPtsI,NPtsJ),dtype=np.float64,order='F')
    InterpZmatrix = np.zeros((NinterFoils,NPtsI,NPtsJ),dtype=np.float64,order='F')
    for k in range(NinterFoils):
        try:
            InterpXmatrix[k,:,:] = J.getx(ProvidedSections[k])
        except BaseException as e:
            J.save(ProvidedSections,'debug.cgns')
            raise e

        InterpYmatrix[k,:,:] = J.gety(ProvidedSections[k])
        InterpZmatrix[k,:,:] = J.getz(ProvidedSections[k])
    if 'interp1d' in InterpolationData['InterpolationLaw'].lower():
        ScipyLaw = InterpolationData['InterpolationLaw'].split('_')[1]
        interpX = scipy.interpolate.interp1d( RelPositions, InterpXmatrix, axis=0, kind=ScipyLaw, bounds_error=False, fill_value='extrapolate')
        interpY = scipy.interpolate.interp1d( RelPositions, InterpYmatrix, axis=0, kind=ScipyLaw, bounds_error=False, fill_value='extrapolate')
        interpZ = scipy.interpolate.interp1d( RelPositions, InterpZmatrix, axis=0, kind=ScipyLaw, bounds_error=False, fill_value='extrapolate')
        for k in range(Ns):
            Section = Sections[k]
            SecX,SecY,SecZ = J.getxyz(Section)
            SecX[:] = interpX(s[k])
            SecY[:] = interpY(s[k])
            SecZ[:] = interpZ(s[k])
    elif 'pchip' == InterpolationData['InterpolationLaw'].lower():
        interpX = scipy.interpolate.PchipInterpolator( RelPositions, InterpXmatrix, axis=0, extrapolate=True)
        interpY = scipy.interpolate.PchipInterpolator( RelPositions, InterpYmatrix, axis=0, extrapolate=True)
        interpZ = scipy.interpolate.PchipInterpolator( RelPositions, InterpZmatrix, axis=0, extrapolate=True)
        for k in range(Ns):
            Section = Sections[k]
            SecX,SecY,SecZ = J.getxyz(Section)
            SecX[:] = interpX(s[k])
            SecY[:] = interpY(s[k])
            SecZ[:] = interpZ(s[k])
    elif 'akima' == InterpolationData['InterpolationLaw'].lower():
        interpX = scipy.interpolate.Akima1DInterpolator( RelPositions, InterpXmatrix, axis=0)
        interpY = scipy.interpolate.Akima1DInterpolator( RelPositions, InterpYmatrix, axis=0)
        interpZ = scipy.interpolate.Akima1DInterpolator( RelPositions, InterpZmatrix, axis=0)
        for k in range(Ns):
            Section = Sections[k]
            SecX,SecY,SecZ = J.getxyz(Section)
            SecX[:] = interpX(s[k],extrapolate=True)
            SecY[:] = interpY(s[k],extrapolate=True)
            SecZ[:] = interpZ(s[k],extrapolate=True)
    elif 'cubic' == InterpolationData['InterpolationLaw'].lower():
        try: bc_type = InterpolationData['CubicSplineBoundaryConditions']
        except KeyError: bc_type = 'not_a_knot'
        interpX = scipy.interpolate.CubicSpline( RelPositions, InterpXmatrix, axis=0,bc_type=bc_type, extrapolate=True)
        interpY = scipy.interpolate.CubicSpline( RelPositions, InterpYmatrix, axis=0,bc_type=bc_type, extrapolate=True)
        interpZ = scipy.interpolate.CubicSpline( RelPositions, InterpZmatrix, axis=0,bc_type=bc_type, extrapolate=True)
        for k in range(Ns):
            Section = Sections[k]
            SecX,SecY,SecZ = J.getxyz(Section)
            SecX[:] = interpX(s[k],extrapolate=True)
            SecY[:] = interpY(s[k],extrapolate=True)
            SecZ[:] = interpZ(s[k],extrapolate=True)
    else:
        raise AttributeError('multiSections(): InterpolationLaw %s not recognized.\nAllowed values are: %s.'%(InterpolationData['InterpolationLaw'],str(AllowedInterpolationLaws)))



    Volume = stackSections(Sections)


    return Volume, SpineCurve


def stackSurfacesWithFields(FirstSurface, LastSurface, Distribution):
    '''
    This function performs a two-section extrusion following a given
    Distribution while keeping the original FlowSolution, if existing.
    Surfaces MUST be structured zones with equal :math:`N_i \\times N_j`

    Parameters
    ----------

        FirstSurface : zone
            surface containing possibly fields at nodes of type `FlowSolution_t`

        LastSurface : zone
            last surface of the extrusion

        Distribution : multiple types
            distribution as accepted by the function
            :py:func:`MOLA.InternalShortcuts.getDistributionFromHeterogeneousInput__`

    Returns
    -------

        VolumeMesh : zone
            resulting volume grid containing possibly fields
    '''

    xF, yF, zF = J.getxyz(FirstSurface)
    xL, yL, zL = J.getxyz(LastSurface)

    Spine = J.getDistributionFromHeterogeneousInput__(Distribution)[0]
    RelativeDistribution = Spine/(Spine.max()-Spine.min())

    Layers = []
    for CurrentRelativeAbscissa in RelativeDistribution:
        NewLayer = I.copyTree(FirstSurface)
        xN, yN, zN = J.getxyz(NewLayer)

        xN[:] = (1-CurrentRelativeAbscissa) * xF + CurrentRelativeAbscissa * xL
        yN[:] = (1-CurrentRelativeAbscissa) * yF + CurrentRelativeAbscissa * yL
        zN[:] = (1-CurrentRelativeAbscissa) * zF + CurrentRelativeAbscissa * zL

        Layers.append(NewLayer)


    VolumeMesh = G.stack(Layers)

    return VolumeMesh



def computeVolumeAndHullAreaOfSurface(t,method='NGON'):
    '''
    Compute the solid-volume of a surface defined by argument **t**.
    The input surface shall be connex, but not necessarily watertight.
    The algorithm will attempt to close the surface in order to make it
    watertight. If it fails to do this, a warning is prompted.

    Parameters
    ----------

        t : PyTree, base, zone or list of zones
            connex surfaces.

        method : str
            Two techniques are currently available:
            ``'NGON'`` or ``'tetraMesher'``

    Returns
    -------

        Volume : float
            Volume contained in the watertight surface.

        Surface : float
            Total area of closed Surface (hull).
    '''

    # Make a single unstructured surface mesh
    t = C.convertArray2Tetra(t)
    zones = I.getNodesFromType(t,'Zone_t')
    t = T.join(zones)

    # Fix any existing gaps
    t = G.gapsmanager(t)
    zones = I.getNodesFromType(t,'Zone_t')
    t = T.join(zones)

    # Check if surface is watertight:
    try:
        P.exteriorFaces(t)
        # If success, surface is not watertight
        print("WARNING computeWingVolume(): Could not make watertight surface.")
        # TODO: Cassiopee shall change exteriorFaces output for watertight surfaces
    except:
        pass


    # Method 1: Compute solid and sum over the cells volume.
    if method == 'tetraMesher':
        Solid = G.tetraMesher(t,algo=1)
        Volume = None
        if C.getNPts(Solid) > 1:
            G._getVolumeMap(Solid)
            Volume,  = J.getVars(Solid,['vol'])
            Volume   = np.sum(Volume)

    elif method == 'NGON':
        t = C.convertArray2NGon(t)
        t = XOR.convertNGON2DToNGON3D(t)
        res = XOR.statsSize(t)
        Volume = res[3]
        # res[0] : le span de ton objet (plus grande dimension de la boite englobante AABB)
        # res[1] : s min
        # res[2] : s max
        # res[3] : v min
        # res[4] : v max

    return Volume



def fillCollar(Outter, Inner, Side1, Side2):
    '''
    From the 4 boundaries defining a collar grid, perform the
    required operations in order to fill by transfinite-interpolation (TFI)
    the volume contained within the user-provided boundary surfaces.

    .. attention:: For **ALL** the zones, the airfoil-wise points must
        be ordered in i-direction and must be consistent.


    Parameters
    ----------

        Outter : zone
            structured surface defining the external (outter) surface.

        Inner  : zone
            structured surface defining the wing-wall surface.

        Side1  : zone
            structured surface defining one of the sides.

        Side2  : zone
            structured surface defining the ramaining side that closes
            Collar volume.

    Returns
    -------

        Zone : zone
            Collar grid.
    '''

    # Perform some verifications upon Outter and Inner surfaces
    TypeOut,NiOut,NjOut,_,_ = I.getZoneDim(Outter)
    TypeInn,NiInn,NjInn,_,_ = I.getZoneDim(Inner)
    if TypeOut != 'Structured':
        raise ValueError('Outter surface (%s) must be structured.'%Outter[0])
    if TypeInn != 'Structured':
        raise ValueError('Inner surface (%s) must be structured.'%Inner[0])

    if NiInn != NiOut:
        raise ValueError('Inner (%s) and Outter (%s) surfaces must have same number of i-points.'%(Inner[0],Outter[0]))
    if NjInn != NjOut:
        raise ValueError('Inner (%s) and Outter (%s) surfaces must have same number of j-points.'%(Inner[0],Outter[0]))

    # Perform some verifications upon Side1 and Side2 surfaces
    TypeS1,NiS1,NjS1,_,_ = I.getZoneDim(Side1)
    TypeS2,NiS2,NjS2,_,_ = I.getZoneDim(Side2)
    if TypeS1 != 'Structured':
        raise ValueError('Side1 surface (%s) must be structured.'%Side1[0])
    if TypeS2 != 'Structured':
        raise ValueError('Side2 surface (%s) must be structured.'%Side2[0])

    if NiS1 != NiS2:
        raise ValueError('Side1 (%s) and Side2 (%s) surfaces must have same number of i-points.'%(Side1[0],Side2[0]))
    if NjS1 != NjS2:
        raise ValueError('Inner (%s) and Outter (%s) surfaces must have same number of j-points.'%(Side1[0],Side2[0]))

    if NiS1 != NiInn:
        raise ValueError('Ni pts of Inner/Outter surfaces shall be equal to Ni pts of Side1/Side2')

    # Define global number of points
    Ni = NiS1  # airfoil's countour wise
    Nj = NjOut # spanwise
    Nk = NjS1  # wall-normal wise.

    # --------------------- BUILD 1st TFI --------------------- #
    # Build IMin surf
    IMinBoun1 = T.subzone(Inner, (1,1,1),(1,NjInn,1))
    IMinBoun2 = T.subzone(Outter,(1,1,1),(1,NjOut,1))
    IMinBoun3 = T.subzone(Side1, (1,1,1),(1,NjS1,1))
    IMinBoun4 = T.subzone(Side2, (1,1,1),(1,NjS2,1))
    IMinSurf  = G.TFI([IMinBoun1,IMinBoun2,IMinBoun3,IMinBoun4])

    # Build IMax surf
    IMaxBoun1 = T.subzone(Inner, (int(Ni/2),1,1),(int(Ni/2),NjInn,1))
    IMaxBoun2 = T.subzone(Outter,(int(Ni/2),1,1),(int(Ni/2),NjOut,1))
    IMaxBoun3 = T.subzone(Side1, (int(Ni/2),1,1),(int(Ni/2),NjS1,1))
    IMaxBoun4 = T.subzone(Side2, (int(Ni/2),1,1),(int(Ni/2),NjS2,1))
    IMaxSurf  = G.TFI([IMaxBoun1,IMaxBoun2,IMaxBoun3,IMaxBoun4])

    # Split Boundaries
    OutterFst = T.subzone(Outter,(1,1,1),(int(Ni/2),NjOut,1))
    InnerFst  = T.subzone(Inner, (1,1,1),(int(Ni/2),NjInn,1))
    Side1Fst  = T.subzone(Side1, (1,1,1),(int(Ni/2),NjS1,1))
    Side2Fst  = T.subzone(Side2, (1,1,1),(int(Ni/2),NjS2,1))

    # Generate the first TFI:
    FirstTFI = G.TFI([
                      IMinSurf, IMaxSurf,   # airfoil-wise (i)
                      Side1Fst, Side2Fst,   # spanwise     (j)
                      InnerFst, OutterFst,  # wall-normal  (k)
                      ])
    T._reorder(FirstTFI,(1,3,-2))


    # --------------------- BUILD 2nd TFI --------------------- #
    # Build IMin surf
    IMinBoun1 = T.subzone(Inner, (int(Ni/2),1,1),(int(Ni/2),NjInn,1))
    IMinBoun2 = T.subzone(Outter,(int(Ni/2),1,1),(int(Ni/2),NjOut,1))
    IMinBoun3 = T.subzone(Side1, (int(Ni/2),1,1),(int(Ni/2),NjS1,1))
    IMinBoun4 = T.subzone(Side2, (int(Ni/2),1,1),(int(Ni/2),NjS2,1))
    IMinSurf  = G.TFI([IMinBoun1,IMinBoun2,IMinBoun3,IMinBoun4])

    # Build IMax surf
    IMaxBoun1 = T.subzone(Inner, (Ni,1,1),(Ni,NjInn,1))
    IMaxBoun2 = T.subzone(Outter,(Ni,1,1),(Ni,NjOut,1))
    IMaxBoun3 = T.subzone(Side1, (Ni,1,1),(Ni,NjS1,1))
    IMaxBoun4 = T.subzone(Side2, (Ni,1,1),(Ni,NjS2,1))
    IMaxSurf  = G.TFI([IMaxBoun1,IMaxBoun2,IMaxBoun3,IMaxBoun4])

    # Split Boundaries
    OutterSnd = T.subzone(Outter,(int(Ni/2),1,1),(Ni,NjOut,1))
    InnerSnd  = T.subzone(Inner, (int(Ni/2),1,1),(Ni,NjInn,1))
    Side1Snd  = T.subzone(Side1, (int(Ni/2),1,1),(Ni,NjS1,1))
    Side2Snd  = T.subzone(Side2, (int(Ni/2),1,1),(Ni,NjS2,1))

    # Generate the second TFI:
    SecondTFI = G.TFI([
                      IMinSurf, IMaxSurf,   # airfoil-wise (i)
                      Side1Snd, Side2Snd,   # spanwise     (j)
                      InnerSnd, OutterSnd,  # wall-normal  (k)
                      ])
    T._reorder(SecondTFI,(1,3,-2))

    # Close the collar grid
    Collar = T.join(FirstTFI,SecondTFI)

    return Collar



def buildBodyForceRotorMesh(NCellsWidth=30, Width=0.25,
                            NCellBuffer=10, BufferWidth=0.30,
                            ExternalCellSize=0.05, propellerDiscParams={}):
    '''
    This function allows for easily building a refined structured volume disc
    suitable for the CFD-Bodyforce technique in an Overset grid context.

    Parameters
    ----------

        NCellsWidth : int
            number of cells that discretizes the interior of the disc.

        Width : float
            width length of the interior of the disc

        NCellBuffer : int
            number of cells discretizing each side around the interior of the
            disc (buffer, where overlap will be found)

        BufferWidth : float
            Dimensions of the buffer zone

        ExternalCellSize : float
            Dimensions of the buffer's external cell size.
            It must be lower that BufferWidth and as close as possible as the
            background grid, in an Overset context.

        propellerDiscParams : dict
            parameters to be passed to function
            :py:func:`MOLA.GenerativeShapeDesign.buildPropellerDisc`, for the
            construction of the 2D disc.

    Returns
    -------

        t : PyTree
            tree containing the resulting mesh
    '''

    ThicknessCellSize = Width/float(NCellsWidth)
    WidthWithBuffer = 2*BufferWidth+Width

    tDisc = GSD.buildPropellerDisc(**propellerDiscParams)
    T._translate(tDisc,(0,0,0.5*WidthWithBuffer))

    BufferStart = W.linelaw(P1=(0,0,0), P2=(0,0,-BufferWidth), N=NCellBuffer+1,
                            Distribution=dict(kind='tanhTwoSides',
                                              FirstCellHeight=ExternalCellSize,
                                              LastCellHeight=ThicknessCellSize))

    CoreCurve = D.line((0,0,-BufferWidth),(0,0,-BufferWidth-Width),NCellsWidth)

    BufferEnd = W.linelaw(P1=(0,0,-BufferWidth-Width),
                          P2=(0,0,-2*BufferWidth-Width), N=NCellBuffer+1,
                            Distribution=dict(kind='tanhTwoSides',
                                              LastCellHeight=ExternalCellSize,
                                              FirstCellHeight=ThicknessCellSize))

    DrivingCurve = T.join([BufferStart, CoreCurve, BufferEnd])

    t = extrudeSurfaceFollowingCurve(tDisc, DrivingCurve)

    return t


def wrapInitVars(t, eqn):
    toc = tic()
    C._initVars(t,eqn)
    initVarsCostInSeconds[0] += tic()-toc

def wrapSmoothField(z, eps, subniter, type, listvar):
    toc = tic()
    T._smoothField(z, eps, subniter, type, listvar)
    smoothField[0] += tic() - toc

def newExtrusionDistribution(maximum_length,
                             maximum_number_of_points=200,
                             distribution_law='ratio',
                             first_cell_height=1e-6,
                             last_cell_height=1e-3,
                             ratio_growth=1.05,
                             smoothing_start_at_layer=80,
                             smoothing_normals_iterations=1,
                             smoothing_normals_subiterations=[5,200,'distance'],
                             smoothing_growth_iterations=2,
                             smoothing_growth_subiterations=120,
                             smoothing_growth_coefficient=[0,0.03,'distance'],
                             smoothing_expansion_factor=0.1,
                             start_point=(0,0,0), direction=(1,0,0)):
    '''
    Create a distribution curve suitable for use as input of
    :py:func:`extrude`.

    Parameters
    ----------

        maximum_length : float
            used to set the maximum authorized length of the extrusion

        maximum_number_of_points : float
            used to set the maximum authorized number of points in normal
            direction of the extrusion

        distribution_law : str
            same as **kind** parameter of :py:func:`~MOLA.Wireframe.linelaw`

        first_cell_height : float
            same as **FirstCellHeight** parameter of :py:func:`~MOLA.Wireframe.linelaw`

        last_cell_height : float
            same as **LastCellHeight** parameter of :py:func:`~MOLA.Wireframe.linelaw`

        ratio_growth : float
            same as **growth** parameter of :py:func:`~MOLA.Wireframe.linelaw`

        smoothing_start_at_layer : int
            starting layer from which smoothing is applyied. Use values higher
            than zero in order to avoid smoothing on initial layers, where
            boundary-layer is likely to be found

        smoothing_normals_iterations : :py:class:`int` or 3-element :py:class:`list` [:py:class:`int`, :py:class:`int`, :py:class:`str`]
            outer number of iterations employed for smoothing the normals.
            Extrusion constraints are applyied between outer iteration calls.

            .. note:: high number of outer iterations produces more robust results
              and can propagate constraint information towards the neighbor
              regions, but it significantly increases CPU cost.

            .. hint:: you can establish varying values of this parameter by
              using a 3-element list: [**start_value**, **end_value**, **law**].
              Where **law** is a :py:class:`str` and can be:

              * ``'distance'``
                the parameter goes from **start_value** to **end_value** as linear
                function of the extrusion distance

              * ``'index'``
                the parameter goes from **start_value** to **end_value** as linear
                function of the extrusion index (or layer) number

        smoothing_normals_subiterations : :py:class:`int` or 3-element :py:class:`list` [:py:class:`int`, :py:class:`int`, :py:class:`str`]
            inner number of iterations employed for smoothing the normals.
            Extrusion constraints are **not** applied between inner iteration calls.
            Note that this parameter also accepts varying values.

            .. note:: high number of inner iterations produces more robust results
              with very little increase in CPU cost, but constaint information
              remains unchanged. Beware that using high normals iteration values
              can significantly degrade the orthogonal quality near the walls.

            .. hint:: this parameter also accepts varying values:
              [**start_value**, **end_value**, ``'distance'`` or ``'index'``]

        smoothing_growth_iterations : :py:class:`int` or 3-element :py:class:`list` [:py:class:`int`, :py:class:`int`, :py:class:`str`]
            outer number of iterations employed for smoothing the extrusion
            height. Extrusion constraints are applyied between outer iteration calls.
            Note that this parameter also accepts varying values.

            .. note:: high number of outer iterations produces more stable results
              but reduces the ability to handle concavities, as less cell-height
              difference exist. This may produce self-intersecting cells.
              Too-low values can produce instabilities during the extrusion
              process, leading to extrusion geometrical crash.
              High values significantly increases CPU cost.

        smoothing_growth_subiterations : :py:class:`int` or 3-element :py:class:`list` [:py:class:`int`, :py:class:`int`, :py:class:`str`]
            inner number of iterations employed for smoothing the extrusion
            height. Extrusion constraints are **not** applyied between inner iteration calls.
            Note that this parameter also accepts varying values.

            .. note:: high number of inner iterations produces more stable results
              but reduces the ability to handle concavities, as less cell-height
              difference exist. This may produce self-intersecting cells.
              Too-low values can produce instabilities during the extrusion
              process, leading to extrusion geometrical crash.
              Increasing this value has little effect on CPU cost.

        smoothing_growth_coefficient : :py:class:`float` or 3-element :py:class:`list` [:py:class:`float`, :py:class:`float`, :py:class:`str`]
            relative cell extrusion height difference that would exist between
            the biggest extrusion height and the mean one, and the lowest extrusion
            height and the mean one, if no smoothing of cell height is done.
            Note that this parameter also accepts varying values.

            .. note:: use slightly positive values in order to handle
              concavities and to improve cell regularity. This parameter has no
              impact on CPU cost

            .. danger:: high values of this parameter can lead to instabilities
              and geometrical crash

            .. tip:: start setting this parameter to **0**, and then progressively
              increase it while checking the stability and quality of the mesh

            .. important:: increasing the values of **smoothing_growth_iterations**
              or **smoothing_growth_subiterations** usually requires to increase
              the value of **smoothing_growth_coefficient**, since the smoothing
              iterations has the side-effect of reducing the effective growth
              coefficient value

        smoothing_expansion_factor : :py:class:`float` or 3-element :py:class:`list` [:py:class:`float`, :py:class:`float`, :py:class:`str`]
            expansion factor coefficient used to deviate the normals from the
            refined regions towards the coarse regions of the mesh, enhancing
            in this way the regularity.
            Note that this parameter also accepts varying values.

            .. note:: use slightly positive values in order to improve the
              cell regularity. This parameter has no impact on CPU cost

            .. danger:: high values of this parameter can lead to instabilities
              and geometrical crash

            .. tip:: start setting this parameter to **0**, and then progressively
              increase it while checking the stability and quality of the mesh

        start_point : 3-:py:class:`float` :py:class:`list` or numpy array
            coordinates :math:`(x,y,z)` of the point where the distribution
            starts

            .. note:: only relevant if more than one distribution is used
              during the extrusion

        direction : 3-:py:class:`float` :py:class:`list` or numpy array
            unitary vector pointing towards the local extrusion direction

            .. note:: only relevant if more than one distribution is used
              during the extrusion

    Returns
    -------
        distribution : zone
            structured curve ready to be employed in :py:func:`extrude`

    '''


    x,y,z=start_point
    nx,ny,nz = direction
    L = maximum_length

    distribution = W.linelaw(P1=start_point,
                             P2=(x+L*nx, y+L*ny, z+L*nz),
                             N=maximum_number_of_points,
                             Distribution=dict(
                                 kind=distribution_law,
                                 growth=ratio_growth,
                                 FirstCellHeight=first_cell_height,
                                 LastCellHeight=last_cell_height))

    _setExtrusionSmoothingParameters(distribution,
                                    smoothing_start_at_layer,
                                    smoothing_normals_iterations,
                                    smoothing_normals_subiterations,
                                    smoothing_growth_iterations,
                                    smoothing_growth_subiterations,
                                    smoothing_growth_coefficient,
                                    smoothing_expansion_factor)

    return distribution

def _setExtrusionSmoothingParameters(curve,
        smoothing_start_at_layer=80,
        smoothing_normals_iterations=1,
        smoothing_normals_subiterations=[5,200,'distance'],
        smoothing_growth_iterations=2,
        smoothing_growth_subiterations=120,
        smoothing_growth_coefficient=[0,0.03,'distance'],
        smoothing_expansion_factor=0.1):

    smoothing_parameters = ['normaliters', 'normalsubiters', 'growthiters',
                            'growthsubiters', 'growthfactor','expansionfactor']
    smoothing_inputs = [smoothing_normals_iterations, smoothing_normals_subiterations,
        smoothing_growth_iterations, smoothing_growth_subiterations,
        smoothing_growth_coefficient, smoothing_expansion_factor]

    params = J.invokeFields(curve,smoothing_parameters)
    s = W.gets(curve)
    bl = smoothing_start_at_layer

    for p, i in zip(params, smoothing_inputs):
        if isinstance(i,list):
            mode = i[2].lower()
            if mode == 'distance':
                p[bl:] = np.interp(s, np.linspace(0,      1,len(s)),
                                      np.linspace(i[0],i[1],len(s)))[bl:]
            elif mode == 'index':
                p[bl:] = np.linspace(i[0],i[1],len(s[bl:]))
            else:
                raise ValueError('unrecognized ditribution mode "%s"'%mode)
        else:
            p[bl:] = i

def trimCartesianGridAtOrigin(t, trim_plane='XZ', reverse=False,
                              impose_points_at_plane=True):
    '''
    Trim a cartesian grid set generated with Cassiopee's function
    :py:func:`Apps.Mesh.Cart.generateCartMesh` following a cartesian plane
    passing at origin :math:`(0, 0, 0)`.

    Parameters
    ----------

        t : PyTree
            the mesh to be trimmed, as generated by :py:func:`Apps.Mesh.Cart.generateCartMesh`

            ..note::
                the grid **t** must be trimmable on requested **trim_plane**,
                otherwise it can lead to unexpected results.

        trim_plane : str
            Cartesian plane used for trimming. Must be one of `'XZ'`, `'XY'` or
            `'YZ'`.

        reverse : bool
            if :py:obj:`True` then the result corresponds to the negative values
            of coordinates with respect to the requested **trim_plane** normal
            direction

        impose_points_at_plane : bool
            if :py:obj:`True` then all grid points where trim is done are
            put exactly at the trimming plane

    Returns
    -------

        zones : :py:class:`list` of zone
            the trimmed zones

            ..attention:: existing connectivity and BC are lost
    '''
    trim_direction = dict(XZ='CoordinateY',
                          XY='CoordinateZ',
                          YZ='CoordinateX')

    # destroy BC and connectivity, will be required to rebuild it.
    # This is necessary, since after the trimming operation there will be 
    # missing some BCMatch, and the trimming plane should be identified as 
    # a new undefined BC, and not as a connectivity
    I._rmNodesByType(t, 'ZoneGridConnectivity_t')
    I._rmNodesByType(t, 'ZoneBC_t')

    zones = I.getZones(t)
    zones2Keep = []
    for zone in zones:
        coord = I.getNodeFromName2(zone, trim_direction[trim_plane])
        if coord is None:
            raise ValueError('coordinate %s not found'%trim_direction[trim_plane])

        hasNegative = np.any(coord[1]<-1e-6)
        hasPositive = np.any(coord[1]>+1e-6)

        if reverse:

            if hasNegative and not hasPositive:
                zones2Keep.append( zone )
                continue

            elif hasPositive and not hasNegative:
                continue

        else:

            if hasPositive and not hasNegative:
                zones2Keep.append( zone )
                continue

            elif hasNegative and not hasPositive:
                continue

        Ni, Nj, Nk = coordShape = coord[1].shape
        NearestGlobalIndex,_ = D.getNearestPointIndex(zone, (0,0,0)) # CAVEAT: costly
        iCutIndex, jCutIndex, kCutIndex = np.unravel_index(NearestGlobalIndex,
                                           coordShape,
                                           order='F')

        if trim_plane == 'XZ':
            if reverse:
                start_index = (1,1,1)
                end_index = (Ni,jCutIndex+1,Nk)
            else:
                start_index = (1,jCutIndex+1,1)
                end_index = (Ni,Nj,Nk)

        elif trim_plane == 'XY':
            if reverse:
                start_index = (1,1,1)
                end_index = (Ni,Nj,kCutIndex+1)
            else:
                start_index = (1,1,kCutIndex+1)
                end_index = (Ni,Nj,Nk)

        else:
            if reverse:
                start_index = (1,1,1)
                end_index = (iCutIndex+1,Nj,Nk)
            else:
                start_index = (iCutIndex+1,1,1)
                end_index = (Ni,Nj,Nk)

        NewZone = T.subzone(zone, start_index, end_index)
        coord = I.getNodeFromName2(NewZone, trim_direction[trim_plane])

        if impose_points_at_plane and len(coord[1].shape) == 3:

            if trim_plane == 'XZ':
                if reverse: coord[1][:,-1,:] = 0.0
                else: coord[1][:,0,:] = 0.0
            elif trim_plane == 'XY':
                if reverse: coord[1][:,:,-1] = 0.0
                else: coord[1][:,:,0] = 0.0
            else:
                if reverse: coord[1][-1,:,:] = 0.0
                else: coord[1][-1,:,:] = 0.0


        zones2Keep += [NewZone]

    I._correctPyTree(zones2Keep, level=3)

    return zones2Keep


def buildCartesianBackground(t, InputMeshes):
    '''
    Make automatic generation of cartesian background grid using overlap and 
    motion user-data. It is activated by a  special item of the list
    **InputMeshes** (as documented in :py:func:`MOLA.Preprocess.prepareMesh4ElsA`),
    which contains a :py:class:`dict` including a value ``'GENERATE'`` for the
    keyword ``file``.

    .. note::
        this function is automatically called by :py:func:`MOLA.Preprocess.prepareMesh4ElsA`,
        and more specifically by :py:func:`MOLA.Preprocess.getMeshesAssembled`.

    .. hint::
        by default, snear size around existing *BCOverlap* is taking by calculating 
        the median cell height normal to the masks. This should be optimum for 
        overset technique. If user want to tune this value, this can be done 
        through OversetOptions :py:class:`dict` using a keyword ``'Snear'`` 
        and a desired :py:class:`float` value.

    Parameters
    ----------

        t : PyTree
            Assembled PyTree as obtained from :py:func:`MOLA.Preprocess.getMeshesAssembled`

        InputMeshes : :py:class:`list` of :py:class:`dict`
            preprocessing information provided by user as defined in
            :py:func:`prepareMesh4ElsA` doc. On of the items of **InputMeshes**
            must be a :py:class:`dict` including a value ``'GENERATE'`` 
            associated to keyword ``file``:

            >>> [ dict(file='GENERATE', ...)]

            In this case, all the pairs of keyword/values documented in 
            :py:func:`MOLA.Preprocess.prepareMesh4ElsA` are still valid, and 
            some additional options are proposed for controlling the grid 
            generation:

            * extension : :py:class:`float`
                the maximum extension (farfield) of the cartesian grid, in 
                meters. If not provided, then next keyword ``extension_factor`` 
                is employed.

            * extension_factor : :py:class:`float`
                If **extension** is not provided, then the extension is calculated
                using the formula ``extension_factor * diag`` where ``diag`` is 
                the cartesian diagonal of the bounding-box covering all overlaps
                of the provided grid.

            * MinimumNumberOfPointsPerDirection : :py:class:`int`
                Minimum number of points per :math:`(x, y, z)` direction. 
                Default value: ``5`` 

            * MaximumNumberOfPoints : :py:class:`int`
                Maximum number of points that can contain a zone block. 
                Default value: ``90000`` 

            * AnalyticalRegions : :py:class:`list` of :py:class:`dict`
                Each item of this :py:class:`list` is a :py:class:`dict` that 
                defines a region (analytically) where refinement is requested.
                Pair of keyword/arguments of the :py:class:`dict` are:

                * ``geometry`` : :py:class:`str`
                    Indicates the geometry of the analytical refinement component.
                    Can be one of:

                    * ``'cylinder'``
                        the refinement region is a cylinder with the following 
                        relevant parameters: ``snear``, ``diameter``, ``base_center``,
                        ``height``, ``direction`` and ``azimutal_resolution``.
                        
                    * ``'box'``
                        the refinement region is a box with the following 
                        relevant parameters: ``snear``, ``length``,
                        ``base_center``,``width``,``height``,
                        ``width_direction``,``length_direction``,
                        ``height_direction``.

                * ``snear`` : :py:class:`float`
                    cartesian cell size desired around this region. If not 
                    provided, it will use the average median cell height around 
                    overlap masks in the provided grid.

                * ``diameter`` : :py:class:`float`
                    diameter of the cylinder.
                    Only relevant if ``geometry='cylinder'``.

                * ``base_center`` : :py:class:`list` of 3 :py:class:`float`
                    coordinates of the center of the base of the cylinder or 
                    of the box.
                    Only relevant if ``geometry='cylinder' or 'box'``.

                * ``height`` : :py:class:`list` of 3 :py:class:`float`
                    height of the cylinder or length of one of the 3 sides of 
                    a box.
                    Only relevant if ``geometry='cylinder' or 'box'``.

                * ``direction`` : :py:class:`list` of 3 :py:class:`float`
                    vector direction of the cylinder, from base to base.
                    Only relevant if ``geometry='cylinder'``.

                * ``azimutal_resolution`` : :py:class:`float`
                    Angle in degrees of the resolution taken for the cylinder.
                    A Value of 20 deg is usually good enough.
                    Only relevant if ``geometry='cylinder'``.

                * ``width`` : :py:class:`float`
                    length of one of the 3 sides of a box.
                    Only relevant if ``geometry='box'``.

                * ``length`` : :py:class:`float`
                    length of one of the 3 sides of a box.
                    Only relevant if ``geometry='box'``.

                * ``width_direction`` : :py:class:`list` of 3 :py:class:`float`
                    vector direction of the width side of the box.
                    Only relevant if ``geometry='box'``.

                * ``length_direction`` : :py:class:`list` of 3 :py:class:`float`
                    vector direction of the length side of the box.
                    Only relevant if ``geometry='box'``.

                * ``height_direction`` : :py:class:`list` of 3 :py:class:`float`
                    vector direction of the height side of the box.
                    Only relevant if ``geometry='box'``.
    
            * UserDefinedRegions : :py:class:`list` of :py:class:`dict`
                Each item of this :py:class:`list` is a :py:class:`dict` that 
                defines a region (existing file) where refinement is requested.
                Pair of keyword/arguments of the :py:class:`dict` are:

                * ``file`` : :py:class:`str`
                    full or relative path of the file containing the geometry

                * ``snear`` : :py:class:`float`
                    cartesian cell size desired around this region. If not 
                    provided, it will use the average median cell height around 
                    overlap masks in the provided grid.

            * trim_options : :py:class:`dict`
                Allows for trimming the final grid (for example, for making a
                simulation with a symmetry plane). It is a :py:class:`dict` 
                whose pair of keyword/arguments are literally the options of
                :py:func:`trimCartesianGridAtOrigin`

            * generateCartMesh_kwargs : :py:class:`dict`
                pair of keyword/value arguments of optional parameters of 
                Cassiopee function ``Apps.Mesh.Cart.generateCartMesh()``. 
                Defaults to ``ext=0``, ``dimPb=3`` and ``expand=0``

    Returns
    -------

        t_cart : PyTree
            resulting cartesian grid
    '''
    GenerationInfo = False
    for meshInfo in InputMeshes:
        if meshInfo['file'] == 'GENERATE':
            GenerationInfo = meshInfo
            GenerationInfo['SplitBlocks'] = False
    if not GenerationInfo: return

    
    from . import ExtractSurfacesProcessor as ESP
    from .Preprocess import getMeshInfoFromBaseName, setBoundaryConditions, connectMesh
    from .UnsteadyOverset import _getRotorMotionParameters, _addAzimuthalGhostComponent
    import Converter.elsAProfile as EP
    import Apps.Mesh.Cart as CART

    print('extracting offset surfaces...')
    baseName2body = dict()
    for base in I.getBases(t):
        meshInfo = getMeshInfoFromBaseName(base[0], InputMeshes)
        try: NCellsOffset = meshInfo['OversetOptions']['NCellsOffset']
        except: NCellsOffset = None

        tR = C.newPyTree([])
        tR[2].append(I.copyRef(base))
        if NCellsOffset:
            connectMesh(tR, [meshInfo])
            setBoundaryConditions(tR, [meshInfo])
            EP._overlapGC2BC(tR)
            FamilyName = 'F_OV_'+base[0]
            body = ESP.extractSurfacesByOffsetCellsFromBCFamilyName(tR, FamilyName,
                                                                NCellsOffset)
        else:
            body = C.extractBCOfType(tR, 'BCOverlap', reorder=True)
        I._rmNodesByType(body,'GridConnectivity1to1_t') # HACK https://elsa.onera.fr/issues/11338
        baseName2body[base[0]] = body

    print('generating refinement regions...')
    bodies = []
    for meshInfo in InputMeshes:
        if meshInfo['file'] == 'GENERATE': continue
        baseName = meshInfo['baseName']

        if 'Motion' in meshInfo:
            rot_ctr, rot_axis, scale, Dpsi = _getRotorMotionParameters(meshInfo)
            new_bodies = _addAzimuthalGhostComponent(baseName2body[baseName],
                                       rot_ctr, rot_axis, scale, Dpsi)
        else:
            try: new_bodies = baseName2body[baseName]
            except KeyError: new_bodies = []

        try: imposed_snear = meshInfo['OversetOptions']['Snear']
        except: imposed_snear = None
        
        _setSnearAtBodies(new_bodies, imposed_snear)

        bodies.extend(new_bodies)

    all_snear = []
    for b in bodies:
        solver_define = I.getNodeFromName1(b, '.Solver#define')
        if not solver_define: continue
        try: all_snear+=[I.getValue(I.getNodeFromName1(solver_define,'snear'))]
        except: continue
    mean_snear = np.mean(all_snear) if all_snear else 0
    
    bodies.extend(_generateAnalyticalRegions(GenerationInfo, mean_snear))
    bodies.extend(_addUserDefinedRegions(GenerationInfo, mean_snear))
    I._correctPyTree(bodies,level=3)
    _setCartesianExtension(GenerationInfo, bodies)

    from .Preprocess import DIRECTORY_OVERSET
    try: os.makedirs(DIRECTORY_OVERSET)
    except: pass

    C.convertPyTree2File(bodies, os.path.join(DIRECTORY_OVERSET,
                                    'refinement_bodies.cgns'))

    t_cart = _generateCartesianFromProcessedBodies(GenerationInfo, bodies)
    try: trim_options = GenerationInfo['trim_options']
    except: trim_options = None
    if trim_options: t_cart = trimCartesianGridAtOrigin(t_cart, **trim_options)

    return t_cart




def _getFrameFromInitialAxis(direction):
    norm = np.linalg.norm

    def misalignmentInDegrees(a, b):
        return np.abs(np.rad2deg( np.arccos( a.dot(b) / (norm(a)*norm(b)) ) ))

    InitialAzimutDirection = np.array([1.0,0.0,0.0])
    angle = misalignmentInDegrees(InitialAzimutDirection, direction)
    while angle < 5 or angle > 175:
        InitialAzimutDirection[0] += 0.01
        InitialAzimutDirection[1] += 0.02
        InitialAzimutDirection[2] += 0.03
        InitialAzimutDirection/= norm(InitialAzimutDirection)
        angle = misalignmentInDegrees(InitialAzimutDirection, direction)

    # Force initial azimut direction to be on the Rotation plane
    CoplanarBinormalVector = np.cross(InitialAzimutDirection, direction)
    InitialAzimutDirection = np.cross(direction, CoplanarBinormalVector)
    InitialAzimutDirection /= norm(InitialAzimutDirection)
    sweepwise = np.cross(direction,InitialAzimutDirection)   

    PropFrame = (tuple(InitialAzimutDirection),     # Blade-wise
                tuple(sweepwise), # sweep-wise
                tuple(direction))   # Rotation axis

    return PropFrame

def _generateAnalyticalRegions(GenerationInfo, mean_snear):
    
    try: AnalyticalRegions = GenerationInfo['AnalyticalRegions']
    except: return []
    
    import Apps.Mesh.Cart as CART

    all_bodies = []
    for region in AnalyticalRegions:
        geom = region['geometry']
        
        if geom == 'cylinder':
            diameter = region['diameter']

            try: base_center = np.array(region['base_center'],dtype=float)
            except: base_center = np.array([0,0,0],dtype=float)

            try: height = region['height']
            except: height = diameter

            try: direction = np.array(region['direction'],dtype=float)
            except: direction = np.array([1,0,0],dtype=float)
            direction /= np.linalg.norm(direction)

            try: azimutal_resolution = region['azimutal_resolution']
            except: azimutal_resolution = 20.0

            if 'snear' in region:
                snear = region['snear']
            else:
                if mean_snear == 0:
                    raise ValueError('must define a snear value')
                snear = mean_snear


            nb_parts = int(90.0/azimutal_resolution)-1
            disc = D.disc((0,0,0), 0.5*diameter, N=nb_parts)
            T._reorder(disc,(1,-2,3))
            spacing_factor_snear = 5
            spacing = spacing_factor_snear * snear
            nb_disc = np.maximum(int(np.round(height/spacing)),2)
            bodies = [T.translate(disc,(0,0,spacing*i)) for i in range(nb_disc)]
            flat_bodies = []
            for body_set in bodies:
                flat_bodies.extend(body_set)
            bodies = flat_bodies
            I._correctPyTree(bodies, level=3)
            NewFrame = _getFrameFromInitialAxis(direction)
            T._rotate(bodies, (0,0,0), ((1,0,0),(0,1,0),(0,0,1)), NewFrame)
            T._translate(bodies, base_center)

        elif geom == 'box':

            length = region['length']

            if 'snear' in region:
                snear = region['snear']
            else:
                if mean_snear == 0:
                    raise ValueError('must define a snear value')
                snear = mean_snear

            try: base_center = np.array(region['base_center'],dtype=float)
            except: base_center = np.array([0,0,0],dtype=float)

            try: width = region['width']
            except: width = length

            try: height = region['height']
            except: height = length

            # width direction
            try:
                width_direction = np.array(region['width_direction'],dtype=float)
            except:
                try:
                    ld = np.array(region['length_direction'],dtype=float)
                    ld /= np.linalg.norm(ld)
                    hd = np.array(region['height_direction'],dtype=float)
                    hd /= np.linalg.norm(hd)
                    width_direction = np.cross(hd,ld)
                except:
                    width_direction = np.array([1,0,0],dtype=float)
            width_direction /= np.linalg.norm(width_direction)
            region['width_direction'] = width_direction

            # height direction
            try:
                height_direction = np.array(region['height_direction'],dtype=float)
            except:
                try:
                    ld = np.array(region['length_direction'],dtype=float)
                    ld /= np.linalg.norm(ld)
                    wd = np.array(region['width_direction'],dtype=float)
                    wd /= np.linalg.norm(wd)
                    height_direction = np.cross(ld,wd)
                except:
                    height_direction = np.array([0,1,0],dtype=float)
            height_direction /= np.linalg.norm(height_direction)
            region['height_direction'] = height_direction

            # length direction
            wd = np.array(region['width_direction'],dtype=float)
            hd = np.array(region['height_direction'],dtype=float)
            length_direction = np.cross(wd,hd)
            region['length_direction'] = length_direction

            cart = G.cart((-width/2,-height/2,0),(width, height,1),(2,2,1))
            nb_bodies = int(np.round(length/snear))
            bodies = [T.translate(cart,(0,0,snear*i)) for i in range(nb_bodies)]
            I._correctPyTree(bodies, level=3)
            T._rotate(bodies, (0,0,0), ((1,0,0),(0,1,0),(0,0,1)),
                     (tuple(region['width_direction']),
                      tuple(region['height_direction']),
                      tuple(region['length_direction'])))
            T._translate(bodies, base_center)
            
        else:
            raise ValueError('geometry "%s" not implemented'%geom)

        CART._setSnear(bodies, snear)
        all_bodies.extend(bodies)

    return all_bodies


def _addUserDefinedRegions(GenerationInfo, mean_snear):
    
    try: UserDefinedRegions = GenerationInfo['UserDefinedRegions']
    except: return []

    import Apps.Mesh.Cart as CART

    bodies = []
    for region in UserDefinedRegions:
        t = C.convertFile2PyTree(region['file'])
        try: snear = region['snear']
        except: snear = mean_snear
        new_bodies = []
        for zone in I.getZones(t):
            topo, Ni,Nj,Nk, dim = I.getZoneDim(zone)

            if dim == 2:
                new_bodies.append( zone )
            elif dim ==3:
                if topo == 'Structured':
                    slices = []
                    for i in range(1,Ni+1):
                        slices += [ T.subzone(zone,(i,1,1),(i,Nj,Nk)) ]
                    new_bodies.extend(slices)
                else:
                    new_bodies.extend(I.getZones(P.exteriorFaces(zone)))
            else:
                raise ValueError('refinement regions must be at least 2D')
    
        CART._setSnear(new_bodies, snear)
        bodies.extend(new_bodies)
    
    I._correctPyTree(bodies, level=3)

    return bodies



def _setSnearAtBodies(bodies, imposed_snear=None):
    import Apps.Mesh.Cart as CART
    for body in I.getZones(bodies):
        if imposed_snear:
            CART._setSnear(body, imposed_snear)
            continue

        offset_info = I.getNodeFromName1(body,'.MOLA#Offset')
        if offset_info:
            heights_info = I.getNodesFromName(offset_info,'MedianHeight*')
            if not heights_info:
                C.convertPyTree2File(body,'debug.cgns')
                raise ValueError('no MedianHeight at body %s'%body[0])
        else:
            heights_info = []
        
        if heights_info:
            snear = np.mean([I.getValue(n) for n in heights_info])
        else:
            tri = C.convertArray2Tetra(body)
            G._getCircumCircleMap(tri)
            r = J.getVars(body, ['ccradius'], Container=I.__FlowSolutionCenters__)
            snear = np.median(r)
        CART._setSnear(body, snear)

def _setCartesianExtension(GenerationInfo, bodies):
    import Apps.Mesh.Cart as CART

    try:
        extension = GenerationInfo['extension']
    except:
        try: extension_factor = GenerationInfo['extension_factor']
        except: extension_factor = 5
        diag = np.linalg.norm(np.diff(np.array(G.bbox(bodies)).reshape((2,3)).T,axis=1))
        extension = extension_factor * diag
    CART._setDfar(bodies, extension)


def _generateCartesianFromProcessedBodies(GenerationInfo, bodies):
    # TODO make this call MPI parallel
    import Apps.Mesh.Cart as CART

    try: vmin = GenerationInfo['MinimumNumberOfPointsPerDirection']
    except: vmin = 5

    try: sizeMax = GenerationInfo['MaximumNumberOfPoints']
    except: sizeMax = 90000

    try: generateCartMesh_kwargs = GenerationInfo['generateCartMesh_kwargs']
    except: generateCartMesh_kwargs = dict(vmin=vmin,sizeMax=sizeMax,
                                           ext=0, dimPb=3, expand=0)
    
    if 'baseName' not in GenerationInfo:
        GenerationInfo['baseName'] = 'BACKGROUND'

    FineRegion = C.newPyTree(['RefinementRegions', bodies])
    I._correctPyTree(FineRegion,level=3)
    t = CART.generateCartMesh(FineRegion, **generateCartMesh_kwargs)
    base = I.getBases(t)[0]
    base[0] = GenerationInfo['baseName']
    I._rmNodesByName(t,'.Solver#Param')
    J.set(base,'.MOLA#InputMesh',**GenerationInfo)

    return t

def checkNegativeVolumeCells(t, volume_threshold=0):
    tR = G.getVolumeMap(t)
    C._initVars(tR,'centers:tag={centers:vol}<%g'%volume_threshold)
    tR = P.selectCells2(tR, 'centers:tag')
    negative_volume_cells = []
    for zone in I.getZones(tR):
        if C.getNCells(zone) > 0:
            negative_volume_cells += [zone]
    if negative_volume_cells:
        print(J.WARN+f'\nWARNING: mesh has {C.getNCells(negative_volume_cells)} negative cells'+J.ENDC)
    return negative_volume_cells
