#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

'''
MOLA - GenerativeShapeDesign.py

This module proposes functionalities for shape generation, useful
for parametric mesh generation. The functions of this module
deal mainly with surfaces.

This module makes use of Cassiopee modules.

First creation:
01/03/2019 - L. Bernardos - Creation by recycling.
'''

import MOLA
from . import InternalShortcuts as J
from . import Wireframe as W

if not MOLA.__ONLY_DOC__:
    # System modules
    import sys
    from copy import deepcopy as cdeep
    import numpy as np
    import pprint
    from timeit import default_timer as Tok


    # Cassiopee
    import Converter.PyTree as C
    import Converter.Internal as I
    import Geom.PyTree as D
    import Post.PyTree as P
    import Generator.PyTree as G
    import Transform.PyTree as T
    import Connector.PyTree as X
    import Distributor2.PyTree as D2
    import Intersector.PyTree as XOR


    verbose = False
    MAX = np.maximum
    MIN = np.minimum

def sweepSections(sections=[], SpanPositions=None,
                  rotation=[0.], rotationLaw='linear',
                  NormalDirection=(1,0,0),
                  spine=None, sectionShapeLaw='linear'):
    '''
    This function builds a sweep surface from a given profile, or a set of
    profiles, throughout a spine. The result is a structured surface.

    Parameters
    ----------

        sections : :py:class:`list` of zone
            List of sections that will be used to sweep around the **spine**.
            They are structured curves.
            Beware that the total number of points of each section must be the
            same. The orientation of the section must be the same too.

            .. attention:: **sections** are supposed to be at **XY** plane

        SpanPositions : numpy 1D array between 0 and 1
            It must have the same number of elements as the number of sections.
            It is used to place each section at the desired position along the
            **spine**. If not provided, sections are distributed uniformely.

        rotation : numpy 1D array
            Vector of twist angles imposed to the profiles along the spine.
            You may use a single value, and the imposed rotation will be the
            same throughout the spine. You may use only two values, and the
            rotation will be distributed linearly from start to end. You may use
            the same number of values as the number of sections, then the twist
            angle is imposed following the law defined by user **rotationLaw**
            at each section's position.

        rotationLaw : str
            It controls the law imposed to interpolate the rotation
            between two sections. Possible values:
            ``'linear'``, ``'nearest'``, ``'zero'``, ``'slinear'``,
            ``'quadratic'`` or ``'cubic'``.

        NormalDirection : 3-element :py:class:`tuple` of :py:class:`float`
            Vector describing the extrusion orientation used to find the local
            reference frame. As the sections are supposed to be
            placed in the XY plane, this vector will typically be ``(0,0,1)``.

        spine : zone
            structured curve where the sections will be swept along.
            The spine-wise discretization is exactly the same as the node
            distribution of the spine.

            .. hint:: if you wish to control de discretization of the spine,
                you may want to use :py:func:`MOLA.Wireframe.discretize`

        sectionShapeLaw : str
            It controls the law imposed to interpolate the airfoils
            between two sections. Possible values:
            ``'linear'``, ``'nearest'``, ``'zero'``, ``'slinear'``,
            ``'quadratic'`` or ``'cubic'``.

    .. important:: We suppose that the input sections are positioned in the
        XY plane. The reference point from which the rotation
        angles are applied is the origin (0,0,0). If you wish to re-position
        this reference, you may apply a translation to the sections beforehand.
    '''
    NormalDirection = np.array(NormalDirection,dtype=np.float64)

    import scipy.interpolate

    # Checks the Number of sections and number of spine points
    Ns = len(sections)
    if Ns == 1:
        sections.append(sections)
        Ns+=1
    Ntot = len(J.getx(spine))
    spine_x, spine_y, spine_z = J.getxyz(spine)

    if not SpanPositions: SpanPositions = np.linspace(0,1,Ns)
    elif np.max(SpanPositions) > 1 or np.min(SpanPositions) < 0:
            print('Aero sweepSections warning: SpanPositions badly imposed, \
                   it shall be defined between [0,1]. Switching to uniform spacing.')
            SpanPositions = np.linspace(0,1,Ns)

    if not rotation:
        rotation = np.zeros(Ns)
    elif len(rotation) == 1:
        rotation = rotation[0]*np.ones(Ns)
    elif len(rotation) == 2:
        rotation = np.linspace(rotation[0],rotation[1],Ns)
    elif len(rotation) != Ns:
        raise ValueError("sweepSections: did not understand the imposed rotation: \
            number of sections: %d, number of rotation planes: %d"%(Ns,len(rotation)))


    # Invokes the Surface
    Surf = T.addkplane(sections[0],Ntot-1)
    Surf_x, Surf_y, Surf_z = J.getxyz(Surf)

    N_pts_section = len(J.getx(sections[0]))
    # Makes the interpolation matrices based upon the provided sections
    InterpXmatrix = np.zeros((Ns,N_pts_section),dtype=np.float64,order='F')
    InterpYmatrix = np.zeros((Ns,N_pts_section),dtype=np.float64,order='F')
    for s in range(Ns):
        InterpXmatrix[s,:] = J.getx(sections[s])
        InterpYmatrix[s,:] = J.gety(sections[s])

    # Searches the elementary spanwise coordinate
    UnitCurvAbscissa = W.gets(spine)

    # Searches the tangent unitary vector of the spine
    Tang = D.getTangent(spine)
    Tang_xyz = np.vstack(J.getxyz(Tang))
    # Makes the interpolation functions
    interpX = scipy.interpolate.interp1d( SpanPositions, InterpXmatrix, axis=0,
        kind=sectionShapeLaw, bounds_error=False, fill_value=InterpXmatrix[0,:])
    interpY = scipy.interpolate.interp1d( SpanPositions, InterpYmatrix, axis=0,
        kind=sectionShapeLaw, bounds_error=False, fill_value=InterpYmatrix[0,:])
    interpR = scipy.interpolate.interp1d( SpanPositions, rotation, axis=0,
        kind=rotationLaw, bounds_error=False, fill_value=rotation[0])

    # Constructs the surface plane by plane
    for k in range(Ntot):
        InterpolatedProfile = cdeep(sections[0]) # Initialization
        IP_x, IP_y, IP_z = J.getxyz(InterpolatedProfile)

        # X and Y coords are based upon the interpolating functions
        IP_x[:] = interpX(UnitCurvAbscissa[k])
        IP_y[:] = interpY(UnitCurvAbscissa[k])

        # Z coords results of a imposed twist + 3D rotation + translation
        # imposed twist
        T._rotate(InterpolatedProfile,(0,0,0), (0.,0.,interpR(UnitCurvAbscissa[k])))
        # 3D rotation: We look for the local reference frame e1 e2 e3
        e3  = Tang_xyz[:,k].flatten()
        e2  = np.cross(e3,NormalDirection)
        e2 /= np.linalg.norm(e2) # normalization
        e1  = np.cross(e2,e3)
        T._rotate(InterpolatedProfile,(0.,0.,0.), ((1,0,0),(0,1,0),(0,0,1)),(e1, e2, e3) )

        # translation
        T._translate(InterpolatedProfile, (spine_x[k],spine_y[k],spine_z[k]))
        # The interpolated profile at this point is done, we save it in our mesh
        # and then we migrate the data
        IP_x, IP_y, IP_z = J.getxyz(InterpolatedProfile)

        Surf_x[:,k] = IP_x
        Surf_y[:,k] = IP_y
        Surf_z[:,k] = IP_z

    return Surf

def stackSections(Sections):
    """
    .. warning:: this function must be replaced by ``G.stack()``
    """
    Nj = len(Sections)
    Ni = C.getNPts(Sections[0])
    Surface = G.cart((0,0,0),(1,1,1),(Ni,Nj,1))
    SurfX, SurfY, SurfZ = J.getxyz(Surface)
    for j in range(Nj):
        SecX, SecY, SecZ = J.getxyz(Sections[j])
        SurfX[:,j] = SecX
        SurfY[:,j] = SecY
        SurfZ[:,j] = SecZ
    return Surface

def wing(Span, ChordRelRef=0.25, NPtsTrailingEdge=5,
         AvoidAirfoilModification=True,
         splitAirfoilOptions=dict(FirstEdgeSearchPortion=0.99,
                                  SecondEdgeSearchPortion=-0.99,
                                  RelativeRadiusTolerance = 1e-1), 
         AirfoilDiscretization=None,
         **kwargs):
    '''
    This function is used to build a wing-like surface (also suitable for
    tail planes, propeller or rotor blades). It employs geometrical laws defined
    through Python dictionaries as well as airfoil geometries provided by user.

    The new wing is positioned such that spanwise direction is in (-Z) direction,
    trailing edge points roughly in (-X) direction, and top side of airfoils
    point towards roughly (+Y direction).

    Parameters
    ----------

        Span : multiple types accepted
            This polymorphic input is used to infer the spanwise
            dimensions and discretization that new wing surface will use.
            It shall start at wing's root (minimum span) and end at wing's tip
            (maximum span).

            For detailed information on possible inputs of Span, please refer to
            :py:func:`MOLA.InternalShortcuts.getDistributionFromHeterogeneousInput__` doc.

            For example, a wing starting at :math:`\mathrm{root} =1 \,\mathrm{m}` and ending at
            :math:`\mathrm{tip} =10 \,\mathrm{m}`, with uniform discretization using 101 points would
            be obtained using:

            >>> Span = np.linspace(1, 10, 101)

        ChordRelRef : float
            This is the stacking point for the sections. Twist law is applied
            with respect to this reference, as well as sweep and dihedral laws.
            Typical practice is using the quart of chord line (:math:`c/4`), which
            is obtained setting:

            >>> ChordRelRef = 0.25

        NPtsTrailingEdge : int
            If provided sections are open at trailing edge, this parameter is
            used for discretizing the gap distance at trailing edge (closing).

        AvoidAirfoilModification : bool
            If :py:obj:`True`, the user-provided airfoils are interpolated, scaled and
            rotated, in order to position each section at its corresponding
            place. This requires that the user-provided airfoils must be placed
            on **XY** plane, with leading edge situated at ``(0,0)`` and
            trailing edge at ``(1,0)``, with top side at greater Y-coordinates
            and oriented **clockwise** starting from trailing edge. Moreover,
            all airfoils shall yield the same number of points (beware that the
            final wing distribution will use this discretization).

            If :py:obj:`False`, then the aforementioned constraints are not
            compulsory, and spanwise airfoil-modification laws can be provided.
            However, surface construction will take longer, as automatic
            detection of top and bottom sides (and other characteristics) of
            each section is performed.

            .. hint:: using **AvoidAirfoilModification** = :py:obj:`True` is faster and
                more robust, but one cannot modify airfoils following its
                properties (like thickness, camber, etc...)

        splitAirfoilOptions : dict
            Parameters to be passed to :py:func:`MOLA.Wireframe.splitAirfoil`
            required for airfoil modifications.
            This is only relevant if **AvoidAirfoilModification** = :py:obj:`False`.
            Their values may determine the accuracy of the automatic computation
            of the leading and trailing edges of arbitrary airfoil-like geometry.

        AirfoilDiscretization : dict
            Parameters to be passed to :py:func:`MOLA.Wireframe.discretizeAirfoil`
            required for airfoil re-discretization.

        kwargs : additional parameters
            These are used for providing the spanwise geometrical laws and
            airfoil sections. Some possible argument names are:

            ``Airfoil=, Chord=, Twist=, Sweep=, Dihedral=``

            Additional argument names are acceptable if the parameter
            **AvoidAirfoilModification** = :py:obj:`False` and they are supported in
            :py:func:`MOLA.Wireframe.modifyAirfoil` function, like for example:

            ::

                MaxThickness MaxRelativeThickness MaxThicknessRelativeLocation
                MaxCamber MaxRelativeCamber MaxCamberRelativeLocation MinCamber
                MinRelativeCamber MinCamberRelativeLocation

            All *geometrical laws* employ the same input interface based in
            standard Python dictionaries. For example:

            ::

                Twist = dict(RelativeSpan = [0.2,  0.6,  1.0],
                                    Twist = [30.,  6.0, -7.0],
                             InterpolationLaw = 'akima')

            Note that arguments name (here, **Twist**) is repeated in both the
            function's argument name and its dictionary element.

            **InterpolationLaw** may be any supported by
            :py:func:`MOLA.InternalShortcuts.interpolate__` function.

            For the special case of **Airfoil** input, each provided element
            must be a 1D zone. For example:

            ::

                NACA4412 = W.airfoil('NACA4412', ClosedTolerance=0)
                AirfoilsDict = dict(RelativeSpan     = [  0.20,     1.000],
                                    Airfoil          = [NACA4412,  NACA4412],
                                    InterpolationLaw = 'rectbivariatespline_1')

            And in this case, **InterpolationLaw** must be
            ``rectbivariatespline_X`` replacing ``X`` with the desired order of
            interpolation (1, 2, 3...).

            .. note:: Airfoil's geometrical interpolation functions are **not**
                provided by :py:func:`MOLA.InternalShortcuts.interpolate__`
                accepted laws.

    Returns
    -------

        Sections : :py:class:`list` of zone
            list of spanwise sections defining the wing

        Wing : zone
            structured surface defining the wing

        DistributionResult : dict
            dictionary containing the resulting geometrical laws as 1D numpy
            arrays with same dimensions
    '''
    import scipy.interpolate
    # ------------ PERFORM SOME VERIFICATIONS ------------ #
    AllowedInterpolationLaws = ('interp1d_<KindOfInterpolation>', 'pchip', 'akima', 'cubic')
    GeometricalParameters    = kwargs.keys()

    WingSpan,Abscissa,_ = J.getDistributionFromHeterogeneousInput__(Span)
    RelWingSpan = WingSpan / WingSpan.max()
    Ns = len(WingSpan)

    # Verify the Geometrical variables arguments
    if 'Airfoil' not in GeometricalParameters: raise AttributeError('wing(): Requires at least one Airfoil')
    AirfoilList = kwargs['Airfoil']['Airfoil']

    anyOpen = False
    anyClosed = False
    for airfoil in AirfoilList:
        if W.isCurveClosed(airfoil): anyClosed = True
        else: anyOpen = True
        if anyOpen == anyClosed == True: 
            raise AttributeError('provided airfoils must be all closed or all open')

    for GeomParam in kwargs:
        for MustKey in ['RelativeSpan','InterpolationLaw',GeomParam]:
            if MustKey not in kwargs[GeomParam]:
                raise AttributeError('wing(): %s dictionnary MUST contain "%s" key.'%(GeomParam,MustKey))
        if len(kwargs[GeomParam]['RelativeSpan']) != len(kwargs[GeomParam][GeomParam]): raise AttributeError('wing(): There MUST be the SAME amount of elements in "RelativeSpan" and "%s" lists'%GeomParam)
        if len(kwargs[GeomParam]['RelativeSpan'])==1:
            kwargs[GeomParam]['RelativeSpan'] = [0,kwargs[GeomParam]['RelativeSpan'][0]]
            kwargs[GeomParam][GeomParam]     += [kwargs[GeomParam][GeomParam][0]]
            if 'InterpolationLaw' not in kwargs[GeomParam]:
                kwargs[GeomParam]['InterpolationLaw'] = 'interp1d_linear'
        if any( np.diff(kwargs[GeomParam]['RelativeSpan'] )<0 ): raise AttributeError("wing(): 'RelativeSpan' values shall be monotonically increasing.")

    if AirfoilDiscretization is None:
        # Verify if all airfoils have the same number of points:
        ListOfNPts = np.array([C.getNPts(a) for a in AirfoilList])
        NPts = ListOfNPts[0]
        if not all(NPts == ListOfNPts):
            msg = 'WARNING: GSD.wing(): all Airfoils MUST have the SAME number of points.\n'
            msg+= 'Adjust using AirfoilDiscretization if required.\n'
            msg+= 'Will impose same number of points using first airfoil distribution.'
            print(J.WARN+msg+J.ENDC)
            AirfoilList = W.useEqualNumberOfPointsOrSameDiscretization(AirfoilList)
            NPts = C.getNPts(AirfoilList[0])
    # ----------------- END OF VERIFICATIONS ----------------- #



    # -------------- CONSTRUCTION OF THE WING -------------- #
    # This operation is performed following these steps:
    # 0. Re-descretize airfoils if required using AirfoilDiscretization
    # 1. Interpolate each airfoil
    # 2. Scale each airfoil based upon the chord distribution.
    # 3. Scale each airfoil based upon the thickness distribution.
    # 5. Apply the Twist using a Rotation.
    # 5. Apply the Sweep using a Translation along tangent.
    # 6. Apply the Dihedral using a Translation along tangent.
    # 7. Put sections along span.

    DistributionResult = {}

    # STEP 0: Re-descretize airfoils if required
    if AirfoilDiscretization is not None:
        # Start with a re-discretization of airfoils
        for i in range(len(AirfoilList)):
            AirfoilList[i] = W.discretizeAirfoil(AirfoilList[i], **AirfoilDiscretization)

        # Verify if all airfoils have the same number of points:
        ListOfNPts = np.array([C.getNPts(a) for a in AirfoilList])
        NPts = ListOfNPts[0]
        if not all(NPts == ListOfNPts):
            raise ValueError(J.FAIL+'FATAL: airfoils after rediscretization do not have same nb of pts. Please notify bug'+J.ENDC)


    Spine = D.line((0,0,0),(1,0,0),Ns)
    SpineX, SpineY, SpineZ = J.getxyz(Spine)

    # STEP 1: Interpolate each airfoil: this step is mandatory.
    order = J._inferOrderFromInterpLawName(kwargs['Airfoil']['InterpolationLaw'])
    Airfoils = kwargs['Airfoil']['Airfoil']
    Positions = kwargs['Airfoil']['RelativeSpan']

    Sections = W.interpolateAirfoils(Airfoils, Positions, RelWingSpan, order=order)

    # STEP 2: Modify each airfoil based upon the distributions
    # contained in AirfoilParameters tuple. Optional step.
    AirfoilParameters = ('Chord',
                         'MaxThickness',
                         'MaxRelativeThickness',
                         'MaxThicknessRelativeLocation',
                         'MaxCamber',
                         'MaxRelativeCamber',
                         'MaxCamberRelativeLocation',
                         'MinCamber',
                         'MinRelativeCamber',
                         'MinCamberRelativeLocation')


    for GeomParam in AirfoilParameters:
        if GeomParam in kwargs:
            DistributionResult[GeomParam] = J.interpolate__(RelWingSpan,
                kwargs[GeomParam]['RelativeSpan'],
                kwargs[GeomParam][GeomParam],
                Law=kwargs[GeomParam]['InterpolationLaw'])


    for j in range(Ns):
        Params = {}
        CurrentSection = Sections[j]
        PreviousTEgap = W.distance(getBoundary(CurrentSection,'imax'),
                                   getBoundary(CurrentSection,'imin'))
        for GeomParam in AirfoilParameters:
            if GeomParam in kwargs:
                Params[GeomParam]=DistributionResult[GeomParam][j]
        if verbose: print('processing airfoil %d of %d'%(j+1,Ns))

        Params['ScalingRelativeChord'] = ChordRelRef

        if AvoidAirfoilModification:
            ModSection = I.copyTree(CurrentSection)
            T._translate(ModSection,(-ChordRelRef,0,0))
            T._homothety(ModSection,(0,0,0), Params['Chord'])
            AirfoilProperties = dict(Chord=Params['Chord'],
                                     ScalingCenter = np.array([0.,0.,0.]))

        else:
            Params['splitAirfoilOptions'] = splitAirfoilOptions

            ModSection, ModCamber = W.modifyAirfoil(Sections[j],**Params)
            AirfoilProperties = J.get(ModSection, '.AirfoilProperties')

            NewSection = I.copyRef(CurrentSection)
            T._homothety(NewSection, AirfoilProperties['ScalingCenter'],
                                     AirfoilProperties['Chord'])

            # modification produced closing of open airfoil
            AfterTEgap = W.distance(getBoundary(ModSection,'imax'),
                                    getBoundary(ModSection,'imin'))
            if AfterTEgap == 0 and PreviousTEgap > 0:
                ModSection = T.subzone(ModSection, (2,1,1),(-2,-1,-1))


        if not W.isCurveClosed(ModSection) and NPtsTrailingEdge>0:
            ModSection = W.closeCurve(ModSection,NPtsTrailingEdge)

        NewSection = ModSection


        J.set(NewSection, '.AirfoilProperties', **AirfoilProperties)
        Sections[j] = NewSection


    # STEP 4: Apply the Twist using a Rotation. Optional step.
    if 'Twist' in kwargs:
        GeomParam = 'Twist'
        DistributionResult[GeomParam] = J.interpolate__(RelWingSpan,
                                            kwargs[GeomParam]['RelativeSpan'],
                                            kwargs[GeomParam][GeomParam],
                                            Law=kwargs[GeomParam]['InterpolationLaw'])

        for j in range(Ns):
            Section = Sections[j]

            # Need to compute the rotation point:
            AirfoilProperties = J.get(Section, '.AirfoilProperties')
            TwistCenter = tuple(AirfoilProperties['ScalingCenter'])

            T._rotate(Section,TwistCenter,(0,0,-1),DistributionResult[GeomParam][j])

    # STEP 5: Apply the Sweep using a Translation along tangent.
    #         This is an optional step.
    GeomParam = 'Sweep'
    if not GeomParam in kwargs:
        kwargs[GeomParam] = {}
        kwargs[GeomParam]['RelativeSpan'] = [0,1]
        kwargs[GeomParam][GeomParam] = [0,0]
        kwargs[GeomParam]['InterpolationLaw'] = 'interp1d_linear'

    DistributionResult[GeomParam] = J.interpolate__(RelWingSpan,
                                        kwargs[GeomParam]['RelativeSpan'],
                                        kwargs[GeomParam][GeomParam],
                                        Law=kwargs[GeomParam]['InterpolationLaw'])

    for j in range(Ns):
        Section = Sections[j]
        # In absolute value:
        SweepDispl= DistributionResult['Sweep'][j]
        # In degrees:
        # SweepDispl= s[j] * np.tan( DistributionResult['Sweep'][j] * np.pi / 180.)
        T._translate(Section,(SweepDispl,0,0))
        SpineX[j] = SweepDispl

    # STEP 6: Apply the Dihedral using a Translation along
    #         tangent. This is an optional step.
    GeomParam = 'Dihedral'
    if not GeomParam in kwargs:
        kwargs[GeomParam] = {}
        kwargs[GeomParam]['RelativeSpan'] = [0,1]
        kwargs[GeomParam][GeomParam] = [0,0]
        kwargs[GeomParam]['InterpolationLaw'] = 'interp1d_linear'


    DistributionResult[GeomParam] = J.interpolate__(RelWingSpan,
                                        kwargs[GeomParam]['RelativeSpan'],
                                        kwargs[GeomParam][GeomParam],
                                        Law=kwargs[GeomParam]['InterpolationLaw'])

    for j in range(Ns):
        Section       = Sections[j]
        # In absolute value:
        DihedralDispl = DistributionResult['Dihedral'][j]
        # In Degrees:
        # DihedralDispl = s[j] * np.tan( DistributionResult['Dihedral'][j] * np.pi / 180.)
        T._translate(Section,(0,DihedralDispl,0))
        SpineY[j] = DihedralDispl


    # STEP 7: Put sections along span. This is a mandatory step.
    for j in range(Ns):
        Section = Sections[j]
        SecZ = J.getz(Section)
        SecZ[:] = -WingSpan[j]
        SpineZ[j] = -WingSpan[j]

    DistributionResult['SpineX'] = SpineX
    DistributionResult['SpineY'] = SpineY
    DistributionResult['SpineZ'] = SpineZ

    Wing = stackSections(Sections) # TODO replace with G.stack
    Wing[0] = 'wing'

    AirfoilProperties = J.get(Sections[0], '.AirfoilProperties')
    TwistCenter = AirfoilProperties['ScalingCenter']
    T._translate(Wing, -TwistCenter)
    T._translate(Sections, -TwistCenter)

    return Sections, Wing, DistributionResult


def extendBlockByTFI2D(cart, distrib, reverse=False, angleSplit=20.,
                       boundaryKind='Arc'):
    '''
    This function is used to extend a 2D zone in the XY plane
    (2D function) given a desired distribution. The resulting
    mesh is of kind H.

    Parameters
    ----------

        cart : zone
            Zone 2D structured

        distrib : zone
            Zone Structured curve 1D. Shall be an horizontal
            line starting at (0,0,0), as obtained from
            :py:func:`D.getDistribution()`

        reverse : bool
            Used to change the orientation of boundary

        angleSplit : float
            Angle (degrees) of threshold for the definition of the contour of
            cart.

        boundaryKind : str
            Shall be ``'Line'`` or ``'Arc'``. Defines the
            geometry of the external boundary of the resulting mesh.

    Returns
    -------

        t : PyTree
            with a base and several zones of the resulting mesh.

        ExternalFaces : :py:class:`list` of zones
            The contour of the external boundary.
    '''
    ef = P.exteriorFaces(cart)
    ef = C.convertBAR2Struct(ef)
    if reverse: T._reorder(ef,(-1,2,3))
    ef = T.splitCurvatureAngle(ef, angleSplit)
    Nef = len(ef)
    # Tangents = map(lambda f: D.getTangent(f), ef)
    Tangents = [D.getTangent(f) for f in ef]

    # Make generative lines
    GenLines, Angles = [], []
    for i in range(Nef):
        fx, fy = J.getxy(ef[i])
        # Normal direction of starting tangent of current face
        Tan1_x, Tan1_y = J.getxy(Tangents[i])
        Dir1 = Tan1_y[0], -Tan1_x[0]

        # Normal direction of ending tangent of previous face
        Tan0_x, Tan0_y = J.getxy(Tangents[i-1])
        Dir0 = Tan0_y[-1], -Tan0_x[-1]

        # Mean normal direction of start point of current face
        DirN = 0.5*(Dir0[0]+Dir1[0]), 0.5*(Dir0[1]+Dir1[1])
        AngleStart= np.arctan2(DirN[1],DirN[0]) * 180 / np.pi
        Angles   += [AngleStart]
        StartLine = T.rotate(distrib,(0,0,0),(0,0,1),AngleStart)
        StartLine = T.translate(StartLine,(fx[0],fy[0],0))
        StartLine[0] += '_face%d_start'%i
        GenLines += [StartLine]

    # Produce TFI
    TFI, ExternalFaces = [], []
    for i in range(Nef):
        # left boundary
        left = GenLines[i]

        # right boundary
        right = GenLines[(i+1)%Nef]

        # bottom boundary
        bottom = ef[i]

        # top boundary
        left_x,left_y,left_z = J.getxyz(left)
        right_x,right_y,right_z = J.getxyz(right)
        if boundaryKind == 'Line':
            top = D.line((left_x[-1],left_y[-1],left_z[-1]),
                     (right_x[-1],right_y[-1],right_z[-1]),
                     C.getNPts(bottom))
        elif boundaryKind == 'Arc':
            leftAux  = T.scale(left,3.)
            rightAux = T.scale(right,3.)
            Inter = XOR.intersection(leftAux,rightAux)
            tetas = Angles[i]
            tetae = Angles[(i+1)%Nef]
            if tetae-tetas<0:
                tetas, tetae = 360+tetae, tetas
            lx,ly = J.getxy(left)
            AuxPt = D.point((lx[-1],ly[-1],0))
            R     = W.distance(AuxPt,Inter)

            top = D.circle(J.getxyz(Inter),R,tetas,tetae,C.getNPts(bottom))
        else:
            raise ValueError("extendBlockByTFI: boundaryKind shall be in ('Line','Arc'). Provided value:%s"%boundaryKind)

        ExternalFaces += [top]

        CurrentTFI = G.TFI([left,right,bottom,top])
        TFI += [CurrentTFI]

    # Join all TFIs
    joinedTFI = TFI[0]
    for i in range(1,Nef):
        joinedTFI = T.join(joinedTFI,TFI[i])
    joinedTFI[0] = 'joinedTFI'
    T._reorder(joinedTFI,(2,1,3))

    # Arrange the new TFI in a new, connected PyTree
    t = C.newPyTree(['BaseExtendedBlockTFI',[cart,joinedTFI]])
    t = X.connectMatch(t,dim=2)

    return t, ExternalFaces

def getSuitableSetOfPointsForTFITri(N1, N2, N3,
        choosePriority=['N1','N2','N3','best'], QtySearch=3,
        tellMeWhatYouDo=False):
    '''
    The function returns a new set of suitable number of points that
    best matches the user-provided ones, such that :py:func:`Generator.TFITri`
    can be performed.

    Parameters
    ----------

        N1 : int
            Initial guess of number of points discritizing the first
            curve (boundary) of the TFITri contour.

        N2 : int
            Initial guess of number of points discritizing the second
            curve (boundary) of the TFITri contour.

        N3 : int
            Initial guess of number of points discritizing the third
            curve (boundary) of the TFITri contour.

        choosePriority : :py:class:`list` of :py:class:`str`
            list of priority (descending order) for
            proposing a new set of suitable :py:func:`G.TFITri` points.
            Possible choices:

            * ``'N1'`` : Attempt to respect provided **N1** by adapting **N2** and **N3**

            * ``'N2'`` : Attempt to respect provided **N2** by adapting **N1** and **N3**

            * ``'N3'`` : Attempt to respect provided **N3** by adapting **N1** and **N2**

            * ``'best'`` : Adapt **N1**, **N2** and **N3**, as closely as possible from provided ones

        QtySearch : int
            Initial amount of points (plus and minus) allowed for
            adaptation of new set of number of points. The higher the value, the
            more costly the search procedure is, but best chances exist of finding
            a suitable set of number of points. If no suitable set is found within
            the initial **QtySearch**, the algorithm recursively calls itself adding 1
            at each call, in order to have best chances of finding a suitable set

        tellMeWhatYouDo : bool
            if :py:obj:`True`, the function prints relevant information about the
            adaptation of points

    Returns
    -------

        N1new : :py:class:`int`
            New suitable proposal of number of points discritizing
            the first curve (boundary) of the TFITri contour.

        N2new : :py:class:`int`
            New suitable proposal of number of points discritizing
            the second curve (boundary) of the TFITri contour.

        N3new : :py:class:`int`
            New suitable proposal of number of points discritizing
            the third curve (boundary) of the TFITri contour.
    '''

    Check0 = (N3-N2+N1)%2 == 1
    Check1 = (N3-N2+N1+1)/2 >= 2
    Check2 = (N3-N2+N1+1)/2 <= N1-1
    Check3 = (N3-N2+N1+1)/2 <= N2-1
    CompCond = N2+N3-1 > N1 > MAX(N3-N2+1,N2-N3+1)

    N2min = int(np.maximum(N3-N1+3, np.ceil((N3+N1)/3. +1)))
    N1max = N2+N3-1
    N1min = np.maximum(N3-N2+1, N2-N3+1)

    if tellMeWhatYouDo:
        print ("Provided number of points (N1, N2, N3) =",(N1, N2, N3))
        print ('Check 0:                                N3-N2+N1 is odd: %s'%Check0)
        print ('Check 1:                        (N3-N2+N1+1)/2 >= 2    : %s'%Check1)
        print ('Check 2:                        (N3-N2+N1+1)/2 <= N1-1 : %s'%Check2)
        print ('Check 3:                        (N3-N2+N1+1)/2 <= N2-1 : %s'%Check3)
        print ('\nComposite Condition N2+N3-1 > N1 > MAX(N3-N2+1,N2-N3+1): %s'%CompCond)
        print ('NOTE: Only "Check0" and Composite Condition are required.')

        print ('\nN2min = %g        : %s'%(N2min,N2>=N2min))

        print ('N1 is in (%g, %g) : %s\n'%(N1min,N1max,N1min<N1<N1max))

    if not Check0 or not CompCond:
        if tellMeWhatYouDo: print ("Cannot use TFITri with this set of points.\nComputing close alternatives within an interval of (N+-%g)..."%QtySearch)
        Candidates = []
        for N3new in range(MAX(N3-QtySearch,1),N3+QtySearch):
            for N2new in range(MAX(N2-QtySearch,N2min),MAX(N2+QtySearch,N2min+QtySearch)):
                N1newMax = N2new+N3new-1
                N1newMin = np.maximum(N3new-N2new+1, N2new-N3new+1)

                for N1new in range(N1newMin+1,N1newMax):
                    Check0 = (N3new-N2new+N1new)%2 == 1
                    CompCond = N2new+N3new-1 > N1new > MAX(N3new-N2new+1,N2new-N3new+1)
                    if Check0 and CompCond:
                        Candidate   = [N1new, N2new, N3new]
                        if Candidate not in Candidates: Candidates += [Candidate]

        DeltaWRToriginal = []
        for c in Candidates:
            DeltaWRToriginal += [np.sqrt((c[0]-N1)**2+(c[1]-N2)**2+(c[2]-N3)**2)]

        SortInd = np.argsort(DeltaWRToriginal)
        Candidates= np.array([Candidates[i] for i in SortInd])

        nCand = len(Candidates)
        if nCand == 0:
            print ('No alternative candidates found. Trying again...')

            return getSuitableSetOfPointsForTFITri(N1,N2,N3,choosePriority, QtySearch+1)
        else:
            if tellMeWhatYouDo:
                print ("%g alternatives found.\nShowing the closest ones by decreasing order (best set is first line):"%nCand)
                print ('    N1      N2      N3')
                print ('  ------  ------  ------')
                for i in range(np.minimum(10,nCand)):
                    c = Candidates[i]
                    print ('%6g  %6g  %6g'%(c[0],c[1],c[2])    )

            Results = {'best':Candidates[0]}
            ClosestN1 = Candidates[Candidates[:,0]==N1,:]
            if len(ClosestN1)>0:
                TheClosestN1=ClosestN1[0]
                Results['N1']=TheClosestN1
                if tellMeWhatYouDo: print ("Closest set for N1=%g: N1=%g, N2=%g, N3=%g"%(N1,TheClosestN1[0],TheClosestN1[1],TheClosestN1[2]))
                for cN1 in ClosestN1:
                    if cN1[1]==cN1[2]:
                        Results['N1,N2=N3']=cN1
                        if tellMeWhatYouDo: print ("Closest set for N1,N2=N3: N1=%g, N2=%g, N3=%g"%(cN1[0],cN1[1],cN1[2]))
                        break

            else:
                if tellMeWhatYouDo: print ("Closest set for N1=%g: Not Found."%N1)


            ClosestN2 = Candidates[Candidates[:,1]==N2,:]
            if len(ClosestN2)>0:
                ClosestN2=ClosestN2[0]
                Results['N2']=ClosestN2
                if tellMeWhatYouDo: print ("Closest set for N2=%g: N1=%g, N2=%g, N3=%g"%(N2,ClosestN2[0],ClosestN2[1],ClosestN2[2]))
            else:
                if tellMeWhatYouDo: print ("Closest set for N2=%g: Not Found."%N2)

            ClosestN3 = Candidates[Candidates[:,2]==N3,:]
            if len(ClosestN3)>0:
                ClosestN3=ClosestN3[0]
                Results['N3']=ClosestN3
                if tellMeWhatYouDo: print ("Closest set for N3=%g: N1=%g, N2=%g, N3=%g"%(N3,ClosestN3[0],ClosestN3[1],ClosestN3[2]))
            else:
                if tellMeWhatYouDo: print ("Closest set for N3=%g: Not Found."%N3)

            for prior in choosePriority:
                try:
                    N1new, N2new, N3new = Results[prior]
                    if tellMeWhatYouDo: print ("New set of points: (%g,%g,%g) (Mode=%s)"%(N1new, N2new, N3new,prior))
                    return N1new, N2new, N3new
                except KeyError: continue

            if tellMeWhatYouDo: print ("\nWARNING: Could not provide the desired %s point alternative.\nYou may try again increasing the value of QtySearch.\nNow, I will automatically provide 'best' candidate."%str(choosePriority))
            prior = 'best'
            N1new, N2new, N3new = Results[prior]
            if tellMeWhatYouDo: print ("New set of points: (%g,%g,%g) (Mode=%s)"%(N1new, N2new, N3new,prior))
            return N1new, N2new, N3new


    else:

        return N1, N2, N3

def closeWingTipAndRoot(wingsurface, tip_window='jmax', close_root=True,
        airfoil_top2bottom_NPts=21, sharp_TrailingEdge_TFI_chord_ref=0.01,
        thick_TrailingEdge_detection_angle=30.,
        sharp_TrailingEdge_post_smoothing=dict(eps=0.9, niter=500, type=2)):
    '''
    closes the wing tip (and possibly root) of a wing-like surface

    Parameters
    ----------

        wingsurface : zone
            a structured wing-like surface. Trailing-edge can be open or closed.

        tip_window : str
            indicates the window where the tip boundary is found.
            Can be one of: ``'imin'``, ``'imax'``, ``'jmin'`` or ``'jmax'``.

        close_root : bool
            if :py:obj:`True`, then the root section is closed too

        airfoil_top2bottom_NPts : int
            number of points used to close the boundaries, in top-to-bottom
            direction. It must be *odd*.

        sharp_TrailingEdge_TFI_chord_ref : float
            relative chord reference (starting from trailing up to leading-edge)
            used for placing the TFI diamond if a sharp topology is required

        thick_TrailingEdge_detection_angle : float
            threshold angle (in degrees) used as detection technique for
            identifying the trailing-edge, when a thick-trailing edge topology
            is required

        sharp_TrailingEdge_post_smoothing : dict
            dictionary of additional parameters to be passed to :py:func:`Transform.smooth`
            for smoothing when sharp trailing edge topology is required

    Returns
    -------

        wing : list of zone
            list of surfaces composing the wing and its tip (and eventually root)
    '''

    wing, = I.getZones(wingsurface)

    root_window = tip_window[0]+'max' if tip_window.endswith('min') else tip_window[0]+'min'

    tipContour = getBoundary(wing, tip_window)
    rootContour = getBoundary(wing, root_window)

    AirfoilIsOpen = not W.isCurveClosed(tipContour)
    AirfoilHasSeveralSharpEdges = len(T.splitCurvatureAngle(tipContour,
                                      thick_TrailingEdge_detection_angle))>1

    if not AirfoilIsOpen:
        first = T.subzone(tipContour,(1,1,1),(5,1,1))
        second = T.subzone(tipContour,(-5,-1,-1),(-1,-1,-1))
        TE_neighborhood = T.join(first,second)
        AirfoilHasRoundTrailingEdge = not len(T.splitCurvatureAngle(TE_neighborhood,
                                          thick_TrailingEdge_detection_angle))>1
    else:
        AirfoilHasRoundTrailingEdge = False

    if AirfoilIsOpen or AirfoilHasSeveralSharpEdges or AirfoilHasRoundTrailingEdge:
        AirfoilHasThickTrailingEdgeTopology = True
    else:
        AirfoilHasThickTrailingEdgeTopology = False

    if AirfoilHasThickTrailingEdgeTopology:
        if AirfoilIsOpen or AirfoilHasRoundTrailingEdge:
            TEdetectionAngleThreshold = None
        else:
            TEdetectionAngleThreshold = thick_TrailingEdge_detection_angle

        tip = closeAirfoil(tipContour, Topology='ThickTE_simple',
                            options=dict(NPtsUnion=airfoil_top2bottom_NPts,
                            TEdetectionAngleThreshold=TEdetectionAngleThreshold))
        tip, = I.getZones(tip)
        zones = [wing, tip]

        if close_root:
            root = closeAirfoil(rootContour, Topology='ThickTE_simple',
                                options=dict(NPtsUnion=airfoil_top2bottom_NPts,
                                TEdetectionAngleThreshold=TEdetectionAngleThreshold))
            root = I.getZones(root)[0]
            T._reorder(root,(1,-2,3))
            zones.append(root)

        if AirfoilIsOpen:
            TrailingEdge_NPts = np.minimum( C.getNPts(getBoundary(tip,'imin')),
                                            C.getNPts(getBoundary(tip,'jmin')) )
            x, y, z = J.getxyz(tipContour)
            TE_JoinLine1 = D.line((x[0],y[0],z[0]), (x[-1],y[-1],z[-1]), TrailingEdge_NPts)
            x, y, z = J.getxyz(rootContour)
            TE_JoinLine2 = D.line((x[0],y[0],z[0]), (x[-1],y[-1],z[-1]), TrailingEdge_NPts)
            TEwndw = 'i' if tip_window.startswith('j') else 'j'
            Wing_JoinLine1 = getBoundary(wing,TEwndw+'min')
            Wing_JoinLine2 = getBoundary(wing,TEwndw+'max')
            TFIwires = [TE_JoinLine1, TE_JoinLine2, Wing_JoinLine1, Wing_JoinLine2]
            TrailingEdge = G.TFI([TE_JoinLine1, TE_JoinLine2, Wing_JoinLine1, Wing_JoinLine2])
            TrailingEdge[0] = 'TrailingEdge'
            zones.append(TrailingEdge)

    else:
        tip = closeAirfoil(tipContour,Topology='SharpTE_TFITri',
                options=dict(NPtsUnion=airfoil_top2bottom_NPts,
                             TFITriAbscissa=sharp_TrailingEdge_TFI_chord_ref))
        zones = [wing] + tip
        if close_root:
            root = closeAirfoil(rootContour, Topology='SharpTE_TFITri',
                                options=dict(NPtsUnion=airfoil_top2bottom_NPts,
                                TFITriAbscissa=sharp_TrailingEdge_TFI_chord_ref))
            T._reorder(root,(1,-2,3))
            zones.extend(root)

    for i,t in enumerate(I.getZones(tip)): t[0]='tip.%d'%i
    if close_root:
        for i,r in enumerate(I.getZones(root)): r[0]='root.%d'%i

    if not AirfoilHasThickTrailingEdgeTopology:

        prepareGlue(tip, [tipContour])
        T._smooth(tip, fixedConstraints=[tipContour],**sharp_TrailingEdge_post_smoothing)
        applyGlue(tip, [tipContour])
        I._rmNodesByName(tip,'.glueData')

        if close_root:
            prepareGlue(root, [rootContour])
            T._smooth(root, fixedConstraints=[rootContour],**sharp_TrailingEdge_post_smoothing)
            applyGlue(root, [rootContour])
            I._rmNodesByName(root,'.glueData')

    silence = J.OutputGrabber()
    with silence: T._reorderAll(zones)

    return zones


def closeAirfoil(Airfoil, Topology='ThickTE_simple', options=dict(NPtsUnion=21,
                            TFITriAbscissa=0.1,TEdetectionAngleThreshold=None)):
    '''
    Given an airfoil-like curve, this function meshes the area inside the
    contour following a user-requested topology kind.

    This function is useful for closing the tip of wings or blades, or even for
    meshing the interior of a wing or blade (for thermal or structural analysis)

    Parameters
    ----------

        Airfoil : zone
            1D PyTree structured zone.

        Topology : str
            Choose the kind of topology for meshing the interior
            of the airfoil:

            * ``'SharpTE_TFITri'``:
                Suitable for sharp Trailing Edge. Relevant **options**:

                * NPtsUnion : :py:class:`int`
                    Number of points across the airfoil.

                * TFITriAbscissa : :py:class:`float`
                    Curvilinear-abscissa reference (between
                    0 and 1) up to where the TRI TFI will be performed from the
                    ``i=0`` point (usually, the trailing edge).

            * ``'ThickTE_simple'``:
                    Suitable for thick Trailing Edge (open airfoil).
                    Build a single TFI based on the 4 boundaries defined using the
                    Trailing Edge as one boundary. Relevant **options**:

                    * TEdetectionAngleThreshold : :py:class:`float`
                        If provided, use
                        this value as the threshold angle (degrees) used for split
                        Trailing Edge curve from the rest of the airfoil. If not
                        provided (None) then uses **NPtsUnion** as reference (see next)

                    * NPtsUnion : :py:class:`int`
                        Only relevant if value of None is given
                        to **TEdetectionAngleThreshold**. It uses this amount of points
                        for splitting top/bottom sides of the airfoil and building
                        Leading Edge and Trailing Edge curves.

        options : dict
            Contextual arguments (see **Topology** for relevant options)

    Returns
    -------

        ClosedZones : :py:class:`list` of zone
            list of surfaces representing the closed region inside the airfoil
    '''
    NPts = C.getNPts(Airfoil)
    s = W.gets(Airfoil)

    if Topology == 'SharpTE_TFITri':
        NPtsUnion      = options['NPtsUnion']
        TFITriAbscissa = options['TFITriAbscissa']

        if NPts%2==0:
            raise NotImplementedError('Airfoil shall have an ODD number of points. Provided was NPts=%g'%NPts)
        elif not W.isCurveClosed(Airfoil):
            raise ValueError('Airfoil shall be closed. Your airfoil is open.')

        iTFITriBot = np.where(s>=TFITriAbscissa)[0][0]
        iTFITriTop = np.where(s<=1-TFITriAbscissa)[0][-1]

        N1=NPtsUnion
        NTop = NPts-iTFITriTop
        NBot = iTFITriBot


        N1, NTop, NBot = getSuitableSetOfPointsForTFITri(N1,NTop,NBot,
                            choosePriority=['N1','N2','N3','best'], QtySearch=3,
                            tellMeWhatYouDo=False)


        TopTFIline   = T.subzone(Airfoil,(NPts-NTop+1,1,1),(-1,-1,-1))
        FrontTFIline = T.subzone(Airfoil,(NBot,1,1),(NPts-NTop+1,1,1))
        BotTFIline   = T.subzone(Airfoil,(1,1,1),(NBot,1,1))


        lTx, lTy, lTz = J.getxyz(TopTFIline)
        lBx, lBy, lBz = J.getxyz(BotTFIline)

        UnionLine = D.line((lBx[-1], lBy[-1], lBz[-1]),
                           (lTx[0], lTy[0], lTz[0]),N1)



        TFITri = G.TFITri(UnionLine,TopTFIline,BotTFIline)
        TFITri = I.getNodesFromType(TFITri,'Zone_t')
        FirstTFIelement2beJoined, ind1st = J.getNearestZone(TFITri,(lTx[0], lTy[0], lTz[0]))
        SecondTFIelement2beJoined, ind2nd = J.getNearestZone(TFITri,(lBx[-1], lBy[-1], lBz[-1]))
        TFIRearJoin = T.join(FirstTFIelement2beJoined,SecondTFIelement2beJoined)
        TFIRearSingleElmntIndex = [i for i in range(3) if i not in (ind1st,ind2nd)][0]
        TFIRearSingleElmnt = TFITri[TFIRearSingleElmntIndex]


        N2 = FrontNPts = C.getNPts(FrontTFIline)

        TFIMonoPossible  =(N1-N2)%2==0
        TFIHalfOPossible = (N1%2)==1 and (N2%2)==1

        if TFIMonoPossible:
            TFIFront = G.TFIMono(UnionLine,FrontTFIline)
            TFIFrontJoin = T.join(TFIFront,TFIRearJoin)
            ClosedZones = [TFIFrontJoin, TFIRearSingleElmnt]
            _reorderAll(ClosedZones,1)
        else:
            raise AttributeError('Could not close airfoil. Change the number of points of the airfoil.')
    elif Topology == 'ThickTE_simple':
        try:
            TEdetectionAngleThreshold = options['TEdetectionAngleThreshold']
        except KeyError:
            TEdetectionAngleThreshold = None

        if TEdetectionAngleThreshold is None:
            NPtsUnion = options['NPtsUnion']

            I._rmNodesByType(Airfoil,'FlowSolution_t')
            Side1, Side2 = SplitCurves = T.splitNParts(Airfoil,2)

            if not W.isCurveClosed(Airfoil):
                x, y, z = J.getxyz(Airfoil)
                JoinLine1 = D.line( (x[0], y[0], z[0]),
                                   ((x[0]+x[-1])/2, (y[0]+y[-1])/2, (z[0]+z[-1])/2),
                                   int((NPtsUnion-1)/2))
                JoinLine2 = D.line( (x[-1], y[-1], z[-1]),
                                   ((x[0]+x[-1])/2, (y[0]+y[-1])/2, (z[0]+z[-1])/2),
                                   int((NPtsUnion-1)/2))
                Side1 = T.join(JoinLine1, Side1)
                Side2 = T.join(Side2, JoinLine2)

            Side1[0] = 'Side1'
            Side2[0] = 'Side2'

            SemiPts = int(NPtsUnion/2)
            TEcurve1 = T.subzone(Side1,(1,1,1),(SemiPts,1,1))
            TEcurve1[0] = 'TEcurve1'
            LEcurve1 = T.subzone(Side1,(C.getNPts(Side1)-SemiPts+1,1,1),(-1,-1,-1))
            LEcurve1[0] = 'LEcurve1'
            MainCurve1  = T.subzone(Side1,(SemiPts,1,1),(C.getNPts(Side1)-SemiPts+1,1,1))
            MainCurve1[0] = 'MainCurve1'

            LEcurve2 = T.subzone(Side2,(1,1,1),(SemiPts,1,1))
            LEcurve2[0] = 'LEcurve2'
            TEcurve2 = T.subzone(Side2,(C.getNPts(Side2)-SemiPts+1,1,1),(-1,-1,-1))
            TEcurve2[0] = 'TEcurve2'
            MainCurve2  = T.subzone(Side2,(SemiPts,1,1),(C.getNPts(Side2)-SemiPts+1,1,1))
            MainCurve2[0] = 'MainCurve2'

            TopSide, BotSide = MainCurve2, MainCurve1
            LEcurve = T.join(LEcurve1,LEcurve2)
            TEcurve = T.join(TEcurve1,TEcurve2)


        else:
            # Split the airfoil in order to extract the TE curve
            AirfoilSplit = T.splitCurvatureAngle(Airfoil,TEdetectionAngleThreshold)

            # Extract the TE curve using the number of points
            NPtsOfSplit = np.array([C.getNPts(s) for s in AirfoilSplit])
            indTEcurve  = np.argmin(NPtsOfSplit)
            TEcurve     = AirfoilSplit.pop(indTEcurve)
            AirfoilWoTE = T.join(AirfoilSplit)


            NPtsLE = C.getNPts(TEcurve)
            Nside  = int((NPts+3)/2) - NPtsLE
            TopSide = T.subzone(AirfoilWoTE,(1,1,1),(Nside,1,1))
            LEcurve = T.subzone(AirfoilWoTE,(Nside,1,1),(Nside+NPtsLE-1,1,1))
            BotSide = T.subzone(AirfoilWoTE,(Nside+NPtsLE-1,1,1),(-1,-1,-1))

        if C.getNPts(LEcurve) != C.getNPts(TEcurve) or C.getNPts(TopSide) != C.getNPts(BotSide):
            raise ValueError("Number of points of bounds are not suitable for TFI. Modify your discretization.")
        TFIboundaries = [LEcurve,TEcurve,TopSide,BotSide,]
        Closed = G.TFI([TopSide,BotSide,LEcurve,TEcurve])
        T._reorder(Closed,(-1,2,3))
        Boundaries = TFIboundaries
        ClosedZones = [Closed]

    elif Topology == 'ThickTE_Hshape':
        raise ValueError('Topology="ThickTE_Hshape" not implemented yet.')
    else:
        raise ValueError('Topology="%s" not recognized.'%Topology)


    return ClosedZones

def multiSections(ProvidedSections, SpineDiscretization,
                    InterpolationData={'InterpolationLaw':'interp1d_linear'}):
    '''
    This function makes a sweep across a list of provided sections (curves)
    that are exactly placed in 3D space (passing points).

    Parameters
    ----------

        ProvidedSections : :py:class:`list` of zones
            each zone must be a 1D structured curve
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
            use it as **SpineDiscretization**.

        InterpolationData : dict
            This is a dictionary that contains
            options for the interpolation process. Relevant options:

            * InterpolationLaw : :py:class:`str`
                Indicates the interpolation law to be
                employed when constructing the surface. Interpolation is performed
                index-by-index of each point of the provided surface coordinates.
                The interpolation abscissa is the SpineDiscretization, whereas the
                interpolated quantities are the grid coordinates.

    Returns
    -------

        Surface : zone
            2D structured surface of the surface that passes across
            all the user-provided sections.

        SpineCurve : zone
            1D structured curve corresponding to the spine of the surface.
    '''
    AllowedInterpolationLaws = ('interp1d_<KindOfInterpolation>', 'pchip',
                                'akima', 'cubic')

    # Construct Spine
    # Barycenters = map(lambda s: G.barycenter(s), ProvidedSections)
    Barycenters = [G.barycenter(s) for s in ProvidedSections]
    SpineCurve  = D.polyline(Barycenters)
    RelPositions= W.gets(SpineCurve)
    # Verify SpineDiscretization argument
    typeSpan=type(SpineDiscretization)
    if I.isStdNode(SpineDiscretization) == -1: # It is a node
        try:
            s=W.gets(SpineDiscretization)
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
            try:
                SpineCurve  = W.polyDiscretize(SpineCurve, SpineDiscretization)
                s  = W.gets(SpineCurve)
                Ns = len(s)
            except:
                ErrMsg = 'multiSections(): SpineDiscretization argument is a list of dictionnaries.\nI thought each element was a Discretization Dictionnary compatible with W.polyDiscretize(), but it was not.\nCheck your SpineDiscretization argument.\n'
                raise AttributeError(ErrMsg)
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

    import scipy.interpolate

    NPts = C.getNPts(ProvidedSections[0])

    # Invoke all sections
    Sections = [D.line((0,0,0),(1,0,0),NPts) for i in range(Ns)]

    NinterFoils = len(ProvidedSections)
    InterpXmatrix = np.zeros((NinterFoils,NPts),dtype=np.float64,order='F')
    InterpYmatrix = np.zeros((NinterFoils,NPts),dtype=np.float64,order='F')
    InterpZmatrix = np.zeros((NinterFoils,NPts),dtype=np.float64,order='F')
    for j in range(NinterFoils):
        InterpXmatrix[j,:] = J.getx(ProvidedSections[j])
        InterpYmatrix[j,:] = J.gety(ProvidedSections[j])
        InterpZmatrix[j,:] = J.getz(ProvidedSections[j])
    if 'interp1d' in InterpolationData['InterpolationLaw'].lower():
        ScipyLaw = InterpolationData['InterpolationLaw'].split('_')[1]
        interpX = scipy.interpolate.interp1d( RelPositions, InterpXmatrix, axis=0, kind=ScipyLaw, bounds_error=False, fill_value='extrapolate')
        interpY = scipy.interpolate.interp1d( RelPositions, InterpYmatrix, axis=0, kind=ScipyLaw, bounds_error=False, fill_value='extrapolate')
        interpZ = scipy.interpolate.interp1d( RelPositions, InterpZmatrix, axis=0, kind=ScipyLaw, bounds_error=False, fill_value='extrapolate')
        for j in range(Ns):
            Section = Sections[j]
            SecX,SecY,SecZ = J.getxyz(Section)
            SecX[:] = interpX(s[j])
            SecY[:] = interpY(s[j])
            SecZ[:] = interpZ(s[j])
    elif 'pchip' == InterpolationData['InterpolationLaw'].lower():
        interpX = scipy.interpolate.PchipInterpolator( RelPositions, InterpXmatrix, axis=0, extrapolate=True)
        interpY = scipy.interpolate.PchipInterpolator( RelPositions, InterpYmatrix, axis=0, extrapolate=True)
        interpZ = scipy.interpolate.PchipInterpolator( RelPositions, InterpZmatrix, axis=0, extrapolate=True)
        for j in range(Ns):
            Section = Sections[j]
            SecX,SecY,SecZ = J.getxyz(Section)
            SecX[:] = interpX(s[j])
            SecY[:] = interpY(s[j])
            SecZ[:] = interpZ(s[j])
    elif 'akima' == InterpolationData['InterpolationLaw'].lower():
        interpX = scipy.interpolate.Akima1DInterpolator( RelPositions, InterpXmatrix, axis=0)
        interpY = scipy.interpolate.Akima1DInterpolator( RelPositions, InterpYmatrix, axis=0)
        interpZ = scipy.interpolate.Akima1DInterpolator( RelPositions, InterpZmatrix, axis=0)
        for j in range(Ns):
            Section = Sections[j]
            SecX,SecY,SecZ = J.getxyz(Section)
            SecX[:] = interpX(s[j],extrapolate=True)
            SecY[:] = interpY(s[j],extrapolate=True)
            SecZ[:] = interpZ(s[j],extrapolate=True)
    elif 'cubic' == InterpolationData['InterpolationLaw'].lower():
        try: bc_type = InterpolationData['CubicSplineBoundaryConditions']
        except KeyError: bc_type = 'not-a-knot'
        interpX = scipy.interpolate.CubicSpline( RelPositions, InterpXmatrix, axis=0,bc_type=bc_type, extrapolate=True)
        interpY = scipy.interpolate.CubicSpline( RelPositions, InterpYmatrix, axis=0,bc_type=bc_type, extrapolate=True)
        interpZ = scipy.interpolate.CubicSpline( RelPositions, InterpZmatrix, axis=0,bc_type=bc_type, extrapolate=True)
        for j in range(Ns):
            Section = Sections[j]
            SecX,SecY,SecZ = J.getxyz(Section)
            SecX[:] = interpX(s[j],extrapolate=True)
            SecY[:] = interpY(s[j],extrapolate=True)
            SecZ[:] = interpZ(s[j],extrapolate=True)
    else:
        raise AttributeError('multiSections(): InterpolationLaw %s not recognized.\nAllowed values are: %s.'%(InterpolationData['InterpolationLaw'],str(AllowedInterpolationLaws)))



    Surface = stackSections(Sections) # TODO replace with G.stack

    return Surface, SpineCurve

def scanBlade(BladeSurface, SpanwiseInput, RotationCenter,
          RotationAxis, BladeDirection, RelativeChordReference=0.25,
          PitchAxis=None, PitchRelativeCenter=[0,0,0],
          buildCamberOptions={}, splitAirfoilOptions={}):
    '''
    Make a scanner of a blade surface in order to construct the sections,
    infer the airfoils and its camber lines, as well as getting the blade's
    geometrical laws.

    Parameters
    ----------

        BladeSurface : PyTree
            Tree containing the blade surface. It can be
            mono- or multi-block, unstructured or structured, or both.

        SpanwiseInput : multiple types
            Specifies the spanwise scanning distribution. Can be one of the
            following types:
            
            * 1D :py:class:`numpy.ndarray` or :py:class:`list` of :py:class:`float`
                Specifies the relative scan points along **BladeDirection**. Hence,
                it must be an array between 0 and 1.

            * :py:class:`list` of 3-:py:class:`float` list
                specifies the :math:`(x,y,z)` coordinates of each point of the
                scan. The scan is done in the normal direction of each point.
            
            * zone
                A user-defined 3D curve given as a structured curve. The scan is
                done in the normal direction of each point.

        RotationCenter : 3-:py:class:`float` array
            coordinates of the blade rotation center

        RotationAxis : 3-:py:class:`float` array
            unit vector specifying the rotation
            direction of the blade using right-hand rule convention

        BladeDirection : 3-:py:class:`float` array
            unit vector specifying the orientation of
            the blade. Scanner is made perpendicular to this direction

        RelativeChordReference : float
            stacking reference of section's chord. Typical value is 0.25.
            Inferred geometrical laws are meaningful considering the stacking
            **RelativeChordReference**.

        PitchAxis : :py:obj:`None` or 3-:py:class:`float` array
            If provided (not :py:obj:`None`), then it specifies the direction
            of the pitch axis. If not provided (:py:obj:`None`) it takes the 
            value of **BladeDirection**

        PitchRelativeCenter : 3-:py:class:`float` array
            Relative direction of the passing point of the **PitchAxis** relative
            to the position of the **RotationCenter**. By default,
            **PitchRelativeCenter** = :math:`(0,0,0)`, which means that the 
            pitch point coincides with **RotationCenter**

        buildCamberOptions : dict
            Optional argument passed to :py:func:`MOLA.Wireframe.buildCamber`
            function. Parameters may influence precision of the geometrical laws

        splitAirfoilOptions : dict
            Optional argument passed to :py:func:`MOLA.Wireframe.splitAirfoil`
            function. Parameters may influence precision of the resulting
            geometrical laws

    Returns
    -------

        ScannerPyTree : PyTree
            contains the *BladeSurface*, the *BladeLine*,
            the sections and its cambers; and normalized airfoils and cambers
    '''

    def getUnitVector(vector):
        v = np.array(vector, dtype=np.float64)
        v /= np.sqrt(v.dot(v))
        return v

    Blade = I.copyRef(BladeSurface)
    I._rmNodesByType(Blade,'FlowSolution_t')

    RotationCenter = np.array(RotationCenter, dtype=np.float64)

    RotationAxis = getUnitVector(RotationAxis)

    BladeDirection = getUnitVector(BladeDirection)

    


    AdvanceDirection = np.cross(RotationAxis, BladeDirection)
    AdvanceDirection /= np.linalg.norm(AdvanceDirection)

    TopDirection = -AdvanceDirection

    splitAirfoilOptions.setdefault('FieldCriterion','TopDirection')
    buildCamberOptions.setdefault('splitAirfoilOptions',splitAirfoilOptions)
    buildCamberOptions.setdefault('FinalDistribution',dict(N=201,
                                  kind='trigonometric', parameter=1))

    if splitAirfoilOptions['FieldCriterion']=='TopDirection':
        Eqn = W.computePlaneEquation(RotationCenter, TopDirection)
        C._initVars(Blade, 'TopDirection='+Eqn)

    Eqn = W.computePlaneEquation(RotationCenter, BladeDirection)
    C._initVars(Blade, 'Span='+Eqn)
    MaximumSpan = C.getMaxValue(Blade,'Span')
    MinimumSpan = C.getMinValue(Blade,'Span')

    # build auxiliar scanning SpanwiseCurve
    if W.isStructuredCurve(SpanwiseInput):
        SpanwiseCurve = SpanwiseInput
    else:
        try:
            SpanwiseArray = np.array(SpanwiseInput)
        except BaseException as e:
            raise TypeError('could not transform SpanwiseInput to a numpy array') from e
        if len(SpanwiseArray.shape) == 1:
            if not np.all(np.diff(SpanwiseArray)>0):
                msg = 'SpanwiseInput is not monotonically increasing (detected as a flat array)'
                raise AttributeError(J.FAIL+msg+J.ENDC)
            elif SpanwiseArray[0] < 0 or SpanwiseArray[-1] > 1:
                msg = 'SpanwiseInput is not bounded between 0 and 1 (detected as a flat array)'
                raise AttributeError(J.FAIL+msg+J.ENDC)
            SpanwiseArray = SpanwiseArray*(MaximumSpan-MinimumSpan)+MinimumSpan
            SpanwiseArrayT = SpanwiseArray.reshape((len(SpanwiseArray),1))
            coords = RotationCenter + BladeDirection*SpanwiseArrayT
            SpanwiseCurve = D.polyline([tuple(c) for c in coords])
        elif len(SpanwiseArray.shape) == 2:
            if not (SpanwiseArray.shape[1] == 3 and SpanwiseArray.shape[0]>1):
                msg = 'SpanwiseInput (detected as a matrix) is not a Nx3 matrix with N>=2'
                raise AttributeError(J.FAIL+msg+J.ENDC)
            SpanwiseCurve = D.polyline([tuple(c) for c in coords])
        else:
            raise TypeError(J.FAIL+'unsupported SpanwiseInput, see doc of scanBlade'+J.ENDC)
    SpanwiseCurve[0] = 'SpanwiseCurve'
    SpanwiseAbscissa = W.gets(SpanwiseCurve)
    SpanwiseCurveX, SpanwiseCurveY, SpanwiseCurveZ = J.getxyz(SpanwiseCurve)
    ScanPlaneX, ScanPlaneY, ScanPlaneZ = J.getxyz(D.getTangent(SpanwiseCurve))

    Fields2StoreInLine = ['Span', 'Chord', 'Twist',
        'MaxRelativeThickness',
        'MaxThickness', 'MaxThicknessRelativeLocation',
        'MaxCamber','MaxRelativeCamber','MaxCamberRelativeLocation',
        'MinCamber','MinRelativeCamber','MinCamberRelativeLocation',
        'ChordwiseX','ChordwiseY','ChordwiseZ',
         'SpanwiseX', 'SpanwiseY', 'SpanwiseZ',
        'ThickwiseX','ThickwiseY','ThickwiseZ',
        'PitchRelativeCenterX','PitchRelativeCenterY','PitchRelativeCenterZ',
        'PitchAxisX','PitchAxisY','PitchAxisZ']

    NumberOfSections = C.getNPts(SpanwiseCurve)
    BladeLine = D.line((0,0,0),(0,0,0),NumberOfSections)
    I.setName(BladeLine,'BladeLine')
    BladeLineFields = J.invokeFieldsDict(BladeLine, Fields2StoreInLine)
    BladeLineX, BladeLineY, BladeLineZ = J.getxyz(BladeLine)


    Sections = []
    Cambers  = []
    NormalizedAirfoils = []
    NormalizedCambers  = []
    RightHandRuleRotation = None
    for i in range(NumberOfSections):
        Span = np.interp(SpanwiseAbscissa[i],[0,1],[MinimumSpan, MaximumSpan])
        print('scanning section %d of %d (s=%g)...'%(i+1,NumberOfSections,SpanwiseAbscissa[i]))

        SpanwiseVector = n = np.array([ScanPlaneX[i], ScanPlaneY[i], ScanPlaneZ[i]])
        Pt = np.array([SpanwiseCurveX[i], SpanwiseCurveY[i], SpanwiseCurveZ[i]])
        PlaneCoefs = n[0],n[1],n[2],-n.dot(Pt)
        C._initVars(Blade,'Slice=%0.12g*{CoordinateX}+%0.12g*{CoordinateY}+%0.12g*{CoordinateZ}+%0.12g'%PlaneCoefs)
        SliceResult = P.isoSurfMC(Blade,'Slice',value=0) # NOTE most expensive part of algorithm
        SliceResult = T.join(SliceResult)
        SliceResult = T.splitConnexity(SliceResult)
        SliceResult = J.getNearestZone(SliceResult, Pt)[0]
        if not SliceResult:
            print('No blade found, skipping section.')
            continue
        Span = C.getMeanValue(SliceResult,'Span')
        print('Span = %g'%Span)


        Structs = []
        for s in SliceResult:
            s = C.convertBAR2Struct(s)
            Structs.append(T.oneovern(s,(2,1,1)))
        Slice = T.join(Structs)
        AirfoilCurve = C.convertBAR2Struct(Slice)
        W.removeMultiplePoints(AirfoilCurve)
        I.setName(AirfoilCurve, 'Section-s%0.3f'%SpanwiseAbscissa[i])
        Sections += [AirfoilCurve]
        try:
            AirfoilProperties, CamberLine = W.getAirfoilPropertiesAndCamber(
                                        AirfoilCurve,
                                        buildCamberOptions=buildCamberOptions,
                                        splitAirfoilOptions=splitAirfoilOptions)
        except BaseException as e:
            J.save(AirfoilCurve,'debug.cgns')
            raise ValueError('could not extract airfoil properties, check debug.cgns') from e
        I.setName(CamberLine, 'Camber-s%0.3f'%SpanwiseAbscissa[i])
        Cambers += [CamberLine]

        LeadingEdge        = AirfoilProperties['LeadingEdge']
        ChordDirection     = AirfoilProperties['ChordDirection']
        ThickwiseDirection = AirfoilProperties['NormalDirection']
        Chord              = AirfoilProperties['Chord']


        if RightHandRuleRotation is None: # first section
            if PitchAxis is None: PitchAxis = SpanwiseVector 
            Dir=np.sign(BladeDirection.dot(np.cross(ThickwiseDirection,ChordDirection)))
            if Dir > 0:
                RightHandRuleRotation = True
                Dir = 1
                print(J.CYAN+'Twist will be computed according to detected RightHandRuleRotation'+J.ENDC)
            else:
                RightHandRuleRotation = False
                Dir = -1
                print(J.CYAN+'Twist will be computed according to detected LeftHandRuleRotation'+J.ENDC)

        # project vector ChordDirection into plane defined by SpanwiseVector
        ChordwiseProjected = ChordDirection - ChordDirection.dot(SpanwiseVector)*SpanwiseVector
        if Dir < 0:
            a = -ChordwiseProjected
            b = AdvanceDirection
        else:
            a = AdvanceDirection
            b = -ChordwiseProjected
        Twist = np.arctan2(np.cross(a,b).dot(PitchAxis),a.dot(b))
        if Twist > np.pi: angle -= 2 * np.pi
        elif Twist <= -np.pi: angle += 2 * np.pi

        Twist = np.rad2deg(Twist)
        print(f'Twist = {Twist}')

        ControlPointAirfoil = (LeadingEdge +
                               RelativeChordReference*ChordDirection*Chord)
        BladeLineX[i] = ControlPointAirfoil[0]
        BladeLineY[i] = ControlPointAirfoil[1]
        BladeLineZ[i] = ControlPointAirfoil[2]

        NormalizedAirfoil = I.copyTree(AirfoilCurve)
        I.setName(NormalizedAirfoil, 'NormAirfoil-s%0.3f'%SpanwiseAbscissa[i])
        W.normalizeFromAirfoilProperties(NormalizedAirfoil, AirfoilProperties)
        W.putAirfoilClockwiseOrientedAndStartingFromTrailingEdge(NormalizedAirfoil)
        NormalizedAirfoils += [NormalizedAirfoil]

        NormalizedCamber = I.copyTree(CamberLine)
        I.setName(NormalizedCamber, 'NormCamber-s%0.3f'%SpanwiseAbscissa[i])
        W.normalizeFromAirfoilProperties(NormalizedCamber, AirfoilProperties)
        NormalizedCambers += [NormalizedCamber]

        BladeLineFields['Span'][i] = Span
        BladeLineFields['Twist'][i] = Twist
        BladeLineFields['ThickwiseX'][i] = AirfoilProperties['NormalDirection'][0]
        BladeLineFields['ThickwiseY'][i] = AirfoilProperties['NormalDirection'][1]
        BladeLineFields['ThickwiseZ'][i] = AirfoilProperties['NormalDirection'][2]
        BladeLineFields['ChordwiseX'][i] = AirfoilProperties['ChordDirection'][0]
        BladeLineFields['ChordwiseY'][i] = AirfoilProperties['ChordDirection'][1]
        BladeLineFields['ChordwiseZ'][i] = AirfoilProperties['ChordDirection'][2]
        BladeLineFields['SpanwiseX'][i] = SpanwiseVector[0]
        BladeLineFields['SpanwiseY'][i] = SpanwiseVector[1]
        BladeLineFields['SpanwiseZ'][i] = SpanwiseVector[2]
        BladeLineFields['PitchRelativeCenterX'][i] = PitchRelativeCenter[0]
        BladeLineFields['PitchRelativeCenterY'][i] = PitchRelativeCenter[1]
        BladeLineFields['PitchRelativeCenterZ'][i] = PitchRelativeCenter[2]
        BladeLineFields['PitchAxisX'][i] = PitchAxis[0]
        BladeLineFields['PitchAxisY'][i] = PitchAxis[1]
        BladeLineFields['PitchAxisZ'][i] = PitchAxis[2]

        for prop in AirfoilProperties:
            if prop in BladeLineFields:
                BladeLineFields[prop][i] = AirfoilProperties[prop]

    ThrustAxis = Dir * RotationAxis
    print(f"detected ThrustAxis = {ThrustAxis}, with Dir={Dir}")
    J.set(BladeLine,'ScanInformation', RotationCenter=RotationCenter,
            RightHandRuleRotation=RightHandRuleRotation,
            PitchAxis=PitchAxis,
            RotationAxis= RotationAxis)

    CamberSurface = G.stack(Cambers)
    CamberSurface[0] = 'CamberSurface'

    # Save the result as a PyTree
    # HACK need to prepend "Base" to name of the base https://elsa.onera.fr/issues/11458
    t = C.newPyTree([
        'BaseBladeSurface',2,I.getZones(Blade),
        'BaseBladeLine', 1,[BladeLine],
        'BaseSections',1,Sections,
        'BaseCambers',1,Cambers,
        'BaseCamberSurface',2,CamberSurface,
        'BaseNormalizedAirfoils',1,NormalizedAirfoils,
        'BaseNormalizedCambers',1, NormalizedCambers,
        ])
    t = I.correctPyTree(t, level=3)

    return t


def getBoundary(zone,window='imin',layer=0):
    '''
    Given a structured zone, extract the window corresponding to
    ``'imin'``, ``'imax'``, ``'jmin'``, ``'jmax'``, ``'kmin'`` or ``'kmax'``.
    The optional argument **layer** is used to extract the layer
    corresponding to ``(min+layer)`` or ``(max-layer)``, if **layer** is
    an :py:class:`int`. If **layer** is a :py:class:`tuple`, the function attempts
    to make a full slice (volume).

    Parameters
    ----------

        zone : zone
            PyTree Structured zone (1D, 2D or 3D).

        window : str
            from which window to extract (``'imin'``, ``'imax'``, ``'jmin'``,
            ``'jmax'``, ``'kmin'`` or ``'kmax'``)

        layer : :py:class:`int` or (:py:class:`int`, :py:class:`int`)
            Possibilities:

            * :py:class:`int`
                extracts ``(min+layer)`` or ``(max-layer)``
            * (:py:class:`int`, :py:class:`int`)
                extracts ``(min+layer[0] to min+layer[0])`` or
                ``(max-layer[0] to max-layer[1])``

            .. attention:: **layer** values shall always be **positive**
                *(negative values may produce unexpected results)*

    Returns
    -------

        window : zone
            The extracted window surface or volume


    Examples
    --------

    ::

        import Converter.PyTree as C
        import Generator.PyTree as G
        import MOLA.GenerativeShapeDesign as GSD

        zone = G.cart((0,0,0),(1,1,1),(20,20,20))

        # Extract the window kmin:
        winK = GSD.getBoundary(zone,'kmin')
        C.convertPyTree2File(winK, 'surface.cgns')

        # Extract the first 10 layers of i-window:
        winI = GSD.getBoundary(zone,'imin',(0,10))
        C.convertPyTree2File(winI, 'volume.cgns')



    '''
    TypeZone,Ni,Nj,Nk,Dim= I.getZoneDim(zone)

    if TypeZone != 'Structured':
        raise AttributeError('Provided zone "%s" is not structured.'%(zone[0]))

    if isinstance(layer, list) or isinstance(layer, tuple):
        l1 = layer[0]
        l2 = layer[1]
    else:
        l1 = l2 = layer

    window = window.lower()

    if   window == 'imin':
        Extraction = T.subzone(zone,(1+l1,1,1),(1+l2,Nj,Nk))
    elif window == 'imax':
        Extraction = T.subzone(zone,(Ni+l2,1,1),(Ni+l1,Nj,Nk))
    elif window == 'jmin':
        Extraction = T.subzone(zone,(1,1+l1,1),(Ni,1+l2,Nk))
    elif window == 'jmax':
        Extraction = T.subzone(zone,(1,Nj+l2,1),(Ni,Nj+l1,Nk))
    elif window == 'kmin':
        Extraction = T.subzone(zone,(1,1,1+l1),(Ni,Nj,1+l2))
    elif window == 'kmax':
        Extraction = T.subzone(zone,(1,1,Nk+l2),(Ni,Nj,Nk+l1))
    else:
        raise AttributeError('Window %s not recognized.'%window)

    Extraction[0] += '.'+window
    return Extraction

def magnetize(zones, magneticzones, tol=1e-10):
    '''
    Glues the points of **zones** to the closest points of **magneticzones**
    given a tolerance.

    Parameters
    ----------

        zones : :py:class:`list` of zone
            The zones whose points are to be glued.

            ..note:: the **zones** are modified.

        magneticzones : :py:class:`list` of zone
            zones defining the final position where
            the points of "zones" are to be glued.

    '''
    for zone in zones:
        x,y,z = J.getxyz(zone)
        x = x.ravel(order='K')
        y = y.ravel(order='K')
        z = z.ravel(order='K')

        for magzone in magneticzones:
            mx,my,mz = J.getxyz(magzone)
            mx = mx.ravel(order='K')
            my = my.ravel(order='K')
            mz = mz.ravel(order='K')

            NPtsMagZone = len(mx)
            PointsArray = np.arange(NPtsMagZone)
            Points = [(mx[i],my[i],mz[i]) for i in range(NPtsMagZone)]


            Ind_Dist = D.getNearestPointIndex(zone,Points)

            IndCand  = np.array([i[0] for i in Ind_Dist])
            DistCand = np.sqrt([i[1] for i in Ind_Dist])

            MangeticPoints = DistCand<tol
            IndicesZone    =     IndCand[MangeticPoints]
            IndicesMagZone = PointsArray[MangeticPoints]

            x[IndicesZone] = mx[IndicesMagZone]
            y[IndicesZone] = my[IndicesMagZone]
            z[IndicesZone] = mz[IndicesMagZone]


def prepareGlue(zones,gluezones,tol=1e-10):
    '''
    Add the .glueData nodes that specifies the ``PointList`` to glue
    between the provided **zones** and **gluezones**. This function
    is intended to be used jointly with :py:func:`applyGlue`, e.g.:

    Add glue information:

    >>> prepareGlue(zones,gluezones)

    User operations that may brake the mesh connection:
    *Projections, extrusions, etc...*

    Apply glue information to reconnect the mesh

    >>> applyGlue(zones,gluezones)

    Parameters
    ----------

        zones : :py:class:`list` of zone
            where Glue information will be added

            .. note:: **zones** are modified *(* ``.glueData`` *is added)*

        gluezones : :py:class:`list` of zone
            specify the location where the zones will be glued.

    '''
    for zone in I.getZones(zones):
        glueElements = -1
        for magzone in I.getZones(gluezones):
            mx,my,mz = J.getxyz(magzone)
            mx = mx.ravel(order='K')
            my = my.ravel(order='K')
            mz = mz.ravel(order='K')

            NPtsMagZone = len(mx)
            PointsArray = np.arange(NPtsMagZone)
            Points = [(mx[i],my[i],mz[i]) for i in range(NPtsMagZone)]

            Ind_Dist = D.getNearestPointIndex(zone,Points)
            IndCand  = np.array([i[0] for i in Ind_Dist],order='F')
            DistCand = np.sqrt(np.array([i[1] for i in Ind_Dist],order='F'))

            MangeticPoints = DistCand<tol
            PointList      =     IndCand[MangeticPoints]
            PointListDonor = PointsArray[MangeticPoints]

            if len(PointList)>0:
                glueElements += 1
                # Prepare node .gluePoints
                gluePoints = I.createNode('.gluePoints_%d'%glueElements,'UserDefinedData_t',value=magzone[0])
                I.createChild(gluePoints,'PointList','DataArray_t', value=PointList)
                I.createChild(gluePoints,'PointListDonor','DataArray_t', value=PointListDonor)


                if glueElements == 0:
                    glueData = I.createUniqueChild(zone, '.glueData', 'UserDefinedData_t',value=None,children=[])

                I.addChild(glueData,gluePoints)



def applyGlue(zones, gluezones):
    """
    Use the information contained in ``.glueData`` nodes in order
    to apply glue to points of **zone** towards **gluezones** locations.
    For usage see documentation of :py:func:`prepareGlue`.

    Parameters
    ----------

        zones : :py:class:`list` of zone
            where Glue information will be added

            .. note:: **zones** are modified *(points are displaced)*

        gluezones : :py:class:`list` of zone
            specify the location where the zones will be glued.
    """

    for zone in I.getZones(zones):
        x,y,z = J.getxyz(zone)
        x = x.ravel(order='K')
        y = y.ravel(order='K')
        z = z.ravel(order='K')

        glueDataNode = I.getNodeFromName1(zone,'.glueData')
        if not glueDataNode: continue
        for gluePoints in glueDataNode[2]:
            gluezoneName = I.getValue(gluePoints)
            gluezone = [gz for gz in I.getZones(gluezones) if gz[0]==gluezoneName][0]
            mx,my,mz = J.getxyz(gluezone)
            mx = mx.ravel(order='K')
            my = my.ravel(order='K')
            mz = mz.ravel(order='K')

            PointList = I.getNodeFromName1(gluePoints,'PointList')[1]
            PointListDonor = I.getNodeFromName1(gluePoints,'PointListDonor')[1]

            try: x[PointList]
            except IndexError as ie:
                print ("ERROR:",ie)
                print ("x[PointList] FAILED")
                print ("x.shape=",x.shape)
                print ("zone %s and gluezone %s"%(zone[0],gluezoneName))
                print ("PointList")
                print (PointList)
                sys.exit()

            try: mx[PointListDonor]
            except IndexError as ie:
                print ("ERROR:",ie)
                print ("mx[PointListDonor] FAILED")
                print ("mx.shape=",mx.shape                )
                print ("zone %s and gluezone %s"%(zone[0],gluezoneName))
                print ("PointListDonor")
                print (PointListDonor)
                sys.exit()

            x[PointList] = mx[PointListDonor]
            y[PointList] = my[PointListDonor]
            z[PointList] = mz[PointListDonor]



def surfacesIntersection(surface1, surface2):
    '''
    Compute the intersection between two surfaces, resulting
    in a BAR curve.

    Parameters
    ----------

        surface1 : zone or :py:class:`list` of zones
            PyTree mono or multi-block surface, not necessarily closed nor
            unstructured.

        surface2 : zone or :py:class:`list` of zones
            PyTree mono or multi-block surface, not necessarily closed nor
            unstructured.

    Returns
    -------

        theIntersection : zone
            unstructured curve BAR of the intersection
    '''

    t = C.newPyTree(['Base',[surface1, surface2]])

    I._rmNodesByType(t,'FlowSolution_t')

    Surf1TRI = C.convertArray2Tetra(surface1)
    Surf1TRI = T.join(I.getZones(Surf1TRI))
    Surf2TRI = C.convertArray2Tetra(surface2)
    Surf2TRI = T.join(I.getZones(Surf2TRI))

    conformed = XOR.conformUnstr(Surf1TRI, Surf2TRI, left_or_right=2, itermax=1)
    Manifold  = T.splitManifold(conformed)
    zonesManifold = I.getNodesFromType(Manifold,'Zone_t')
    intersection = None
    for Zi in zonesManifold:
        inter = P.exteriorFaces(Zi)
        inter = T.splitConnexity(inter)
        zs = I.getZones(inter)
        if len(zs) == 1:
            intersection = zs

    return intersection

def extrapolateSurface(Surface, Boundary, SpineDiscretization, mode='tangent',
                       direction=(1,0,0)):
    '''
    Extrapolate a **Surface** from a **Boundary** following a given
    dimensional extrapolation **direction**.

    Parameters
    ----------

        Surface : zone
            Structured surface.

        Boundary : str
            One of: ``'imin','imax','jmin','jmax'``

        SpineDiscretization : 1D numpy array or zone
            used to define the position and distance of the new
            extrapolation sections (array or curve)

        mode : str
            Can be one of:

            * ``'tangent'``
                extrapolates the surface tangentially at the selected
                boundary

            * ``'directional'``
                extrapolates the surface following a user-defined direction

        direction : 3-:py:class:`float` :py:class:`list`, :py:class:`tuple` or array
            unitary vector pointing towards the desired extrapolation direction.

            .. note:: only relevant if **mode** = ``'directional'``

    Returns
    -------

        surface : zone
            New surface including the extrapolation region
    '''

    # Verify SpineDiscretization argument
    typeSpan=type(SpineDiscretization)
    if I.isStdNode(SpineDiscretization) == -1: # It is a node
        try:
            s=W.gets(SpineDiscretization)
            Ns = len(s)
        except:
            ErrMsg = "extrapolateSurface(): SpineDiscretization argument was a PyTree node (named %s), but I could not obtain the CurvilinearAbscissa.\nPerhaps you forgot GridCoordinates nodes?"%SpineDiscretization[0]
            raise AttributeError(ErrMsg)

    elif typeSpan is np.ndarray: # It is a numpy array
        s  = SpineDiscretization
        if len(s.shape)>1:
            ErrMsg = "extrapolateSurface(): SpineDiscretization argument was detected as a numpy array of dimension %g!\nSpan MUST be a monotonically increasing VECTOR (1D numpy array)."%len(s.shape)
            raise AttributeError(ErrMsg)
        Ns = s.shape[0]
        if any( np.diff(s)<0):
            ErrMsg = "extrapolateSurface(): SpineDiscretization argument was detected as a numpy array.\nHowever, it was NOT monotonically increasing. SpineDiscretization MUST be monotonically increasing. Check that, please."
            raise AttributeError(ErrMsg)
    elif isinstance(SpineDiscretization, list): # It is a list
        if isinstance(SpineDiscretization[0], dict):
            try:
                SpineCurve  = W.polyDiscretize(SpineCurve, SpineDiscretization)
                s  = W.gets(SpineCurve)
                Ns = len(s)
            except:
                ErrMsg = 'extrapolateSurface(): SpineDiscretization argument is a list of dictionnaries.\nI thought each element was a Discretization Dictionnary compatible with W.polyDiscretize(), but it was not.\nCheck your SpineDiscretization argument.\n'
                raise AttributeError(ErrMsg)
        else:
            try:
                s = np.array(SpineDiscretization,dtype=np.float64)
            except:
                ErrMsg = 'extrapolateSurface(): Could not transform SpineDiscretization argument into a numpy array.\nCheck your SpineDiscretization argument.\n'
                raise AttributeError(ErrMsg)
            if len(s.shape)>1:
                ErrMsg = "extrapolateSurface(): SpineDiscretization argument was detected as a numpy array of dimension %g!\nSpan MUST be a monotonically increasing VECTOR (1D numpy array)."%len(s.shape)
                raise AttributeError(ErrMsg)
            Ns = s.shape[0]
            if any( np.diff(s)<0):
                ErrMsg = "extrapolateSurface(): SpineDiscretization argument was detected as a numpy array.\nHowever, it was NOT monotonically increasing. SpineDiscretization MUST be monotonically increasing. Check that, please."
                raise AttributeError(ErrMsg)
    else:
        raise AttributeError('extrapolateSurface(): Type of SpineDiscretization argument not recognized. Check your input.')

    if mode == 'tangent':
        # Get information about the tangents
        BoundaryContour = getBoundary(Surface,Boundary,layer=0)
        PreviousContour = getBoundary(Surface,Boundary,layer=1)

        BCx, BCy, BCz = J.getxyz(BoundaryContour)
        PCx, PCy, PCz = J.getxyz(PreviousContour)

        tx, ty, tz = J.invokeFields(BoundaryContour,['tx','ty','tz'])
        Norms = np.sqrt((BCx-PCx)**2+(BCy-PCy)**2+(BCz-PCz)**2)
        tx[:] = (BCx - PCx) / Norms
        ty[:] = (BCy - PCy) / Norms
        tz[:] = (BCz - PCz) / Norms

        # Create the new sections based on the provided distrib.
        NewSection = I.copyTree(BoundaryContour)
        Sections = [NewSection]
        for i in range(1,Ns):
            dx, dy, dz = J.invokeFields(NewSection,['dx','dy','dz'])
            dH = s[i] - s[i-1] # Displacement distance

            dx[:] = tx * dH    # | Displacement
            dy[:] = ty * dH    # | direction
            dz[:] = tz * dH    # |

            NewSection = T.deform(NewSection,vector=['dx','dy','dz'])

            Sections += [NewSection]

        StackedSections = G.stack(Sections,None)
        I._rmNodesFromName(StackedSections,'FlowSolution*')

        NewSurface = T.join(Surface,StackedSections)
    elif mode == 'directional':
        BoundaryContour = getBoundary(Surface,Boundary,layer=0)
        Sections = []
        for i in range(0,Ns):
            TranslationArray = np.array(direction)*s[i]
            TranslationTuple = (TranslationArray[0],
                                TranslationArray[1],
                                TranslationArray[2])
            Sections += [T.translate(BoundaryContour,TranslationTuple)]

        StackedSections = G.stack(Sections,None)
        NewSurface = T.join(Surface,StackedSections)
    else:
        raise ValueError('Mode %s not recognized'%mode)
    NewSurface = T.reorder(NewSurface,(-2,1,3))

    return NewSurface

def extrudeAirfoil2D(airfoilCurve,References={},Sizes={},
                                  Points={},Cells={},options={}):
    '''
    Build a 2D mesh around a given airfoil geometry.

    Parameters
    ----------

        airfoilCurve : zone
            An airfoil positioned in the XY plane. The point (0,0) must be the
            Leading Edge and point (1,0) must be the Trailing Edge.
            Index-ordering must be **clockwise** (first index starts from
            trailing edge of bottom side).
            Airfoil might be either open or closed.

        References : dict
            Relevant keys are:

            * Reynolds : :py:class:`float`
                Reference Reynolds number used for computation of
                wall-adjacent cell size following:

                *Frank M. White's Fluid Mechanics 5th Ed., page 467*

            * DeltaYPlus : :py:class:`float`
                Reference :math:`\Delta y^+` number used for computation of
                wall-adjacent cell size following:

                *Frank M. White's Fluid Mechanics 5th Ed., page 467*

        Sizes : dict
            Dictionary describing the sizing of the computational domain.
            If not provided (or partially provided),
            then missing information is got from following default values:

            ::

                Sizes = dict(
                Height                  = 50.*Chord, # Domain height

                Wake                    = 50.*Chord, # Domain width

                BoundaryLayerMaxHeight  = 0.1*Chord, # Maximum allowable orthogonal
                                                     # extrusion for Boundary-layer

                TrailingEdgeTension     = 0.5*Chord # controls the wake direction
                                                     # coming out of Trailing Edge
                                                     # (fraction of Wake)
                )

        Points : dict
            Dictionary describing the
            sampling of the grid. If not provided (or partially provided),
            then missing information is got from following default values:

            ::

                Points = dict(
                Extrusion               = 300, # Nb of pts in extrusion direction

                # The following two arguments are used to discretize the airfoil's
                # Bottom and Top side, using lists of parameters (clockwise
                # direction). Useful for refinement in shock-wave or bubbles.
                # NPts (integer) Number of points in interval
                # BreakPoint(x) (float) Reference breakpoint in x-coordinate where
                #     JoinCellLength will be imposed
                # JoinCellLength (float) Cell length to be imposed at BreakPoint

                Bottom=[{'NPts': 70,'BreakPoint(x)':None,'JoinCellLength':None}],
                Top   =[{'NPts':100,'BreakPoint(x)':None,'JoinCellLength':None}],

                Wake                    = 200, # Nb of pts in wake's direction

                WakeHeightMaxPoints     = 50,  # Maximum allowable Nb of pts in height
                                               # direction of wake (only relevant if
                                               # airfoil is open at Trailing Edge)

                BoundaryLayerGrowthRate = 1.05,# Geometrical growth rate for boundary
                                               # layer extrusion

                BoundaryLayerMaxPoints  = 100, # Maximum allowable Nb of pts for
                                               # boundary-layer extrusion
                )

        Cells : dict
            Dictionary describing the
            cell sizes of the grid. If not provided (or partially provided),
            then missing information is got from following default values:

            ::

                Cells = dict(
                TrailingEdge = 0.005*Chord,  # foilwise size of Trailing Edge cell

                LeadingEdge  = 0.0005*Chord, # foilwise size of Leading Edge cell

                Farfield     = 2.0*Chord,    # normalwise size of Farfield cells

                WakeFarfieldAspectRatio = 0.02, # Aspect ratio of farfield Wake cell

                LEFarfieldAspectRatio   = 1.0,  # Aspect ratio of farfield cell
                                                # propagated from Leading Edge

                FarfieldAspectRatio     = 0.05, # Aspect ratio of farfield cells
                                                # propagating from Leading Edge and
                                                # Trailing Edge

                # The following two parameters are only relevant if airfoil is
                # closed (closed Trailing Edge). They are used to control the
                # parabolic cell height augmentation from Trailing Edge up to
                # Wake farfield
                ClosedWakeAbscissaCtrl  = 0.50, # Wake-wise abscissa control point
                                                # Must be in (0,1)

                ClosedWakeAbscissaRatio = 0.25, # Cell Height fraction (of Wake
                                                  farfield height) at the ctrl point
                                                  Must be in (0,1)

                )

        options : dict
            Dictionary describing the
            additional grid options . If not provided (or partially provided),
            then missing information is got from following default values:

            ::

                options = dict(
                NumberOfProcessors=28,                        # Number of blocs for split

                DenseSamplingNPts = 5000,         # Number of points for sampling
                                                  # geometrical entities during
                                                  # auxiliary operations

                LEsearchAbscissas = [0.35, 0.65], # foilwise curvilinear abscissa
                                                  # range of search for Leading Edge
                                                  # point. Should be around 0.5.

                LEsearchEpsilon   = 1.e-8,        # Small tolerance criterion for
                                                  # determining the Leading Edge point

                MappingLaw        = 'cubic',      # Mapping law of the airfoil. For
                                                  # allowable values see doc of:
                                                  # W.discretize() function

                UseOShapeIfTrailingEdgeAngleIsBiggerThan=80., # if the trailing edge
                                                              # angle is bigger than
                                                              # this value (in deg)
                                                              # then use "O" topology

                TEclosureTolerance= 3.e-5,        # Euclidean distance (in chord
                                                  # units) used to determine if
                                                  # provided airfoils is open or
                                                  # closed, which will serve to
                                                  # determine if additional wake
                                                  # zone has to be built or not
                    )


    Returns
    -------

        grid : PyTree
            Including zones, splitting, connectivity, etc

        meshParamsDict : :py:class:`dict`
            Includes size,pts,cells,opts dictionaries
    '''

    xmax = C.getMaxValue(airfoilCurve,'CoordinateX')
    xmin = C.getMinValue(airfoilCurve,'CoordinateX')
    Chord = xmax - xmin

    import Intersector.PyTree as XOR

    size = dict(            # Default values
    Height                  = 50.*Chord,
    Wake                    = 50.*Chord,
    BoundaryLayerMaxHeight  = 0.1*Chord,
    TrailingEdgeTension     = 0.5*Chord,
    )
    size.update(Sizes) # User-provided values

    pts = dict(             # Default values
    Extrusion               = 300,
    Bottom=[{'NPts': 70,'BreakPoint(x)':None,'JoinCellLength':None}],
    Top   =[{'NPts':100,'BreakPoint(x)':None,'JoinCellLength':None}],
    Wake                    = 200,
    WakeHeightMaxPoints     = 50,
    BoundaryLayerGrowthRate = 1.05,
    BoundaryLayerMaxPoints  = 100,
    )
    pts.update(Points) # User-provided values


    cells = dict(  # Default values
    TrailingEdge = 0.005*Chord,
    LeadingEdge  = 0.0005*Chord,
    Farfield     = 2.0*Chord,
    WakeFarfieldAspectRatio = 0.02,
    LEFarfieldAspectRatio   = 1.0,
    FarfieldAspectRatio     = 0.05,
    ClosedWakeAbscissaCtrl  = 0.50,
    ClosedWakeAbscissaRatio = 0.25,
    )
    cells.update(Cells) # User-provided values

    opts = dict(  # Default values
    NumberOfProcessors=28,
    DenseSamplingNPts = 5000,
    LEsearchAbscissas = [0.35, 0.65],
    LEsearchEpsilon   = 1.e-8,
    MappingLaw        = 'cubic', # requires recent version of scipy
    TEclosureTolerance= 3.e-5,
    UseOShapeIfTrailingEdgeAngleIsBiggerThan=80.,
    FarfieldFamilyName= 'FARFIELD',
    AirfoilFamilyName = 'AIRFOIL',
    )
    opts.update(options) # User-provided values

    '''
    Compute the Trailing Edge cell height following:
    Frank M. White's Fluid Mechanics 5th Ed., page 467.
    Hence, if rho=mu=1, then ReL=U*L, so that:
    '''


    try:
        Reynolds = References['Reynolds']
    except KeyError:
        raise KeyError('Reynolds not found in meshParams')

    try:
        DeltaYPlus = References['DeltaYPlus']
    except KeyError:
        raise KeyError('DeltaYPlus not found in meshParams')

    ReL = Reynolds / Chord
    WallCellHeight = DeltaYPlus/(((0.026/ReL**(1./7.))*(ReL)**2/2.)**0.5)

    wires, surfs = [], []

    # Build dense airfoil
    NPts = opts['DenseSamplingNPts']


    foilDense = C.initVars(airfoilCurve,'CoordinateZ',0.)
    foilDense = W.discretize(foilDense, NPts, Distribution=None, MappingLaw=opts['MappingLaw'])
    sDense = W.gets(foilDense)


    # Look for Leading Edge
    LookInterval = (sDense>opts['LEsearchAbscissas'][0]) * (sDense < opts['LEsearchAbscissas'][1])
    D._getCurvatureRadius(foilDense)
    rad, = J.getVars(foilDense,['radius'])
    rad = rad[LookInterval]
    AveragingRadius  = opts['LEsearchEpsilon']*(rad.max()-rad.min())
    LEcandidates = rad<(rad.min()+AveragingRadius)
    sLEmean = 0.5*(sDense[LookInterval][LEcandidates].min()+sDense[LookInterval][LEcandidates].max())

    # Remap the input airfoil
    Distributions = []
    PreviousJoinCell = cells['TrailingEdge']
    for distr in pts['Bottom']:
        try: joincell = distr['JoinCellLength']
        except: joincell = None
        Distr = dict(kind='tanhTwoSides',N=distr['NPts'],
                     FirstCellHeight=PreviousJoinCell,
                     LastCellHeight=joincell,
                     )
        try: bkpt = distr['BreakPoint(x)']
        except: bkpt = None
        if bkpt is not None:
            Distr['BreakPoint'] = min(W.getAbscissaAtStation(foilDense,bkpt, coordinate='x'))
        PreviousJoinCell = Distr['LastCellHeight']
        Distributions += [Distr]
    Distributions[-1]['LastCellHeight'] = cells['LeadingEdge']
    Distributions[-1]['BreakPoint']     = sLEmean

    PreviousJoinCell = cells['LeadingEdge']
    for distr in pts['Top']:
        try: joincell = distr['JoinCellLength']
        except: joincell = None
        Distr = dict(kind='tanhTwoSides',N=distr['NPts'],
                     FirstCellHeight=PreviousJoinCell,
                     LastCellHeight=joincell,
                     )
        try: bkpt = distr['BreakPoint(x)']
        except: bkpt = None
        if bkpt is not None:
            Distr['BreakPoint'] = max(W.getAbscissaAtStation(foilDense, bkpt, coordinate='x'))
        PreviousJoinCell = Distr['LastCellHeight']
        Distributions += [Distr]
    Distributions[-1]['LastCellHeight'] = cells['TrailingEdge']
    Distributions[-1]['BreakPoint']     = 1.0


    foil = W.polyDiscretize(foilDense, Distributions,MappingLaw=opts['MappingLaw'])
    foil[0] = 'foil'


    wires += [foil]

    s = W.gets(foil)
    iLE = np.argmin(np.abs(s-sLEmean))
    NPtsBottom = iLE+1
    NPtsTop = len(s)-NPtsBottom


    # Build wake curves
    fX, fY, fZ = J.getxyz(foil)
    fZ *= 0.
    TEdistance = ( (fX[0]-fX[-1])**2 + (fY[0]-fY[-1])**2 )**0.5

    # Find Trailing-Edge direction
    TEdir = 0.5*( np.array([fX[-1]-fX[-2],fY[-1]-fY[-2]]) +
                  np.array([fX[0]-fX[1],fY[0]-fY[1]]) )
    TEdir /= np.sqrt(TEdir.dot(TEdir))

    isFoilClosed = True if TEdistance < opts['TEclosureTolerance'] else False

    TE_top = np.array([fX[-2]-fX[-1],fY[-2]-fY[-1],fZ[-2]-fZ[-1]])
    TE_bottom = np.array([fX[1]-fX[0],fY[1]-fY[0],fZ[1]-fZ[0]])
    TE_top_norm = np.sqrt(TE_top.dot(TE_top))
    TE_bottom_norm = np.sqrt(TE_bottom.dot(TE_bottom))
    TE_angle = np.rad2deg(np.arccos( TE_top.dot(TE_bottom) / (TE_top_norm*TE_bottom_norm) ))

    Topology = 'O' if TE_angle > opts['UseOShapeIfTrailingEdgeAngleIsBiggerThan'] else 'C'

    if Topology == 'C':

        if isFoilClosed:
            fX[0] = fX[-1] = 0.5*(fX[0]+fX[-1])
            fY[0] = fY[-1] = 0.5*(fY[0]+fY[-1])

            # Wake guide
            WakeGuide = D.polyline([
                (fX[0],fY[0],0.),
                (fX[0]+TEdir[0]*size['TrailingEdgeTension']*size['Wake'],fY[0]+TEdir[1]*size['TrailingEdgeTension']*size['Wake'],0.),
                (fX[0]+size['Wake'],fY[0]+TEdir[1]*size['TrailingEdgeTension']*size['Wake'],0.),
                ],)
            WakeGuide[0] = 'WakeGuide'
            wires += [WakeGuide]
            WakeDense = D.bezier(WakeGuide, N=opts['DenseSamplingNPts'])
            WakeDense[0] = 'WakeDense'

            # Discretize wake curves
            WakeTop = W.discretize(WakeDense, N=pts['Wake'],
                Distribution=dict(
                    kind='tanhTwoSides',
                    FirstCellHeight=cells['TrailingEdge'],
                    LastCellHeight=cells['Farfield']),
                MappingLaw=opts['MappingLaw'],
            )
            WakeTop[0] = 'WakeTop'
            WakeBottom = I.copyTree(WakeTop)
            WakeBottom[0] = 'WakeBottom'
            T._reorder(WakeBottom, (-1, 2, 3))
            wires += [WakeTop,WakeBottom]


            T._reorder(WakeBottom,(-1,2,3))
            CellHeightFieldTop,    = J.invokeFields(WakeTop,   ['WallCellHeight'])
            CellHeightFieldBottom, = J.invokeFields(WakeBottom,['WallCellHeight'])
            CellHeightFieldFoil,   = J.invokeFields(foil,      ['WallCellHeight'])
            CellHeightFieldFoil[:] = WallCellHeight
            sWake = W.gets(WakeTop)


            WakeFarfieldCellHeight = cells['WakeFarfieldAspectRatio']*cells['Farfield']
            for i in range(1,pts['Wake']-1):
                Hratio = cells['ClosedWakeAbscissaRatio']
                CellHeight = J.interpolate__(sWake[i],
                    [0.,cells['ClosedWakeAbscissaCtrl'] , 1.],
                    [WallCellHeight,
                     Hratio*WakeFarfieldCellHeight+(1-Hratio)*WallCellHeight,
                     WakeFarfieldCellHeight],
                    Law='interp1d_quadratic')
                CellHeightFieldTop[i]    = CellHeight
                CellHeightFieldBottom[i] = CellHeight
            T._reorder(WakeBottom,(-1,2,3))

        else:
            # Build Two separate trailing edge wake curves

            # Compute trailing edge wake discretization
            UniformNPtsWake = np.maximum(int(TEdistance/WallCellHeight), 6)
            NPtsWakeHeight = np.minimum(pts['WakeHeightMaxPoints'],UniformNPtsWake)
            WakeLeft = W.linelaw(P1=(fX[0], fY[0], 0), P2=(fX[-1], fY[-1], 0), N=NPtsWakeHeight, Distribution=dict(
                kind='tanhTwoSides',
                FirstCellHeight=WallCellHeight,
                LastCellHeight=WallCellHeight))
            WakeLeft[0] = 'WakeLeft'

            WakeEndPointTop = np.array([fX[-1]+size['Wake'],fY[-1]+TEdir[1]*size['TrailingEdgeTension']*size['Wake']+0.5*NPtsWakeHeight*cells['WakeFarfieldAspectRatio']*cells['Farfield'],0.])

            WakeEndPointBottom = np.array([fX[0]+size['Wake'],fY[0]+TEdir[1]*size['TrailingEdgeTension']*size['Wake']-0.5*NPtsWakeHeight*cells['WakeFarfieldAspectRatio']*cells['Farfield'],0.])

            WakeRight = D.line(WakeEndPointBottom,WakeEndPointTop,NPtsWakeHeight)
            WakeRight[0] = 'WakeRight'
            wires += [WakeRight]

            # Wake guide (top side)
            WakeGuideTop = D.polyline([
                (fX[-1],fY[-1],0.),
                (fX[-1]+TEdir[0]*size['TrailingEdgeTension']*size['Wake'],fY[-1]+TEdir[1]*size['TrailingEdgeTension']*size['Wake'],0.),
                tuple(WakeEndPointTop), # TODO: ask for evolution to allow np.array
                ],)
            WakeGuideTop[0] = 'WakeGuideTop'
            wires += [WakeGuideTop]
            WakeDenseTop = D.bezier(WakeGuideTop, N=opts['DenseSamplingNPts'])
            WakeDenseTop[0] = 'WakeDenseTop'

            # Wake guide (bottom side)
            WakeGuideBottom = D.polyline([
                (fX[0],fY[0],0.),
                (fX[0]+TEdir[0]*size['TrailingEdgeTension']*size['Wake'],fY[0]+TEdir[1]*size['TrailingEdgeTension']*size['Wake'],0.),
                tuple(WakeEndPointBottom), # TODO: ask for evolution to allow np.array
                ])
            WakeGuideBottom[0] = 'WakeGuideBottom'
            wires += [WakeGuideBottom]
            WakeDenseBottom = D.bezier(WakeGuideBottom, N=opts['DenseSamplingNPts'])
            WakeDenseBottom[0] = 'WakeDenseBottom'

            # Discretize wake curves
            WakeTop = W.discretize(WakeDenseTop, N=pts['Wake'],
                Distribution=dict(
                    kind='tanhTwoSides',
                    FirstCellHeight=cells['TrailingEdge'],
                    LastCellHeight=cells['Farfield']),
                MappingLaw=opts['MappingLaw'],
            )
            WakeTop[0] = 'WakeTop'
            WakeBottom = W.discretize(WakeDenseBottom, N=pts['Wake'],
                Distribution=dict(
                    kind='tanhTwoSides',
                    FirstCellHeight=cells['TrailingEdge'],
                    LastCellHeight=cells['Farfield']),
                MappingLaw=opts['MappingLaw'],
            )
            WakeBottom[0] = 'WakeBottom'
            T._reorder(WakeBottom, (-1, 2, 3))
            wires += [WakeTop,WakeBottom]


            # # This fails (negative cells)
            # # TODO: Create Cassiopee ticket ?
            # WakeTFI = G.TFI([WakeLeft,WakeRight,WakeBottom,WakeTop])
            # WakeTFI[0] = 'WakeTFI'
            # surfs += [WakeTFI]


            # Compute FirstCellHeights of extrusion contour
            T._reorder(WakeBottom,(-1,2,3))

            CellHeightFieldTop,    = J.invokeFields(WakeTop,   ['WallCellHeight'])
            CellHeightFieldBottom, = J.invokeFields(WakeBottom,['WallCellHeight'])
            CellHeightFieldFoil,   = J.invokeFields(foil,      ['WallCellHeight'])


            LengthWakeLeft    = D.getLength(WakeLeft)
            LengthWakeRight   = D.getLength(WakeRight)

            StartCellHeight,_ = W.getFirstAndLastCellLengths(WakeLeft)
            LastCellHeight,_  = W.getFirstAndLastCellLengths(WakeRight)


            CellHeightFieldTop[0]     = WallCellHeight
            CellHeightFieldBottom[0]  = WallCellHeight
            CellHeightFieldTop[-1]    = LastCellHeight
            CellHeightFieldBottom[-1] = LastCellHeight
            CellHeightFieldFoil[:] = WallCellHeight

            xTop, yTop, zTop = J.getxyz(WakeTop)
            xBot, yBot, zBot = J.getxyz(WakeBottom)

            for i in range(1,pts['Wake']-1):
                CurrentLength = ((xTop[i]-xBot[i])**2+
                                 (yTop[i]-yBot[i])**2+
                                 (zTop[i]-zBot[i])**2)**0.5

                CellHeight = np.interp(CurrentLength,
                    [LengthWakeLeft,LengthWakeRight],
                    [StartCellHeight,LastCellHeight])


                CellHeightFieldTop[i] = CellHeightFieldBottom[i] = CellHeight

        # Build Extrusion curve (boundary-layer)
        C._rmVars([WakeBottom,foil,WakeTop],['s'])
        ExtrusionCurve    = T.join([WakeBottom,foil,WakeTop])
        ExtrusionCurve[0] = 'ExtrusionCurve'
        sEC = W.gets(ExtrusionCurve)
        WallCellHeightDist,   = J.getVars(ExtrusionCurve, ['WallCellHeight'])


    elif Topology == 'O':

        ExtrusionCurve    = foil
        ExtrusionCurve[0] = 'ExtrusionCurve'
        WallCellHeightDist,   = J.invokeFields(ExtrusionCurve, ['WallCellHeight'])
        sEC = W.gets(ExtrusionCurve)
        WallCellHeightDist[:] = WallCellHeight

    else:
        raise ValueError('FATAL')

    wires += [ExtrusionCurve]

    # Build distribution law (1D curve, boundary layer)
    BLdistrib = W.linelaw(P1=(0,0,0),P2=(0,size['BoundaryLayerMaxHeight'],0),
            N=pts['BoundaryLayerMaxPoints'], Distribution=dict(
                kind='ratio',
                FirstCellHeight=WallCellHeight,
                growth=pts['BoundaryLayerGrowthRate']))
    BLdistribY = J.gety(BLdistrib)
    NPtsBL = len(BLdistribY)

    if Topology == 'O':
        FarTop = W.linelaw(
            P1=(0,BLdistribY[-1],0),
            P2=(0,BLdistribY[-1]+size['Height'],0),
            N=pts['Extrusion']-NPtsBL+2,
            Distribution=dict(
                kind='tanhTwoSides',
                FirstCellHeight=abs(BLdistribY[-1]-BLdistribY[-2]),
                LastCellHeight=cells['Farfield'],))
        FarTop[0] = 'FarTop'
        BLdistrib = T.join(BLdistrib, FarTop)
        BLdistribY = J.gety(BLdistrib)
        NPtsBL = len(BLdistribY)

    # From field 'WallCellHeight', build distribution for extrusion
    Ni = C.getNPts(ExtrusionCurve)
    Lcurve = D.getLength(ExtrusionCurve)
    Distrib4Extrusion = G.cart((0,0,0),(1./(Ni-1),1./(NPtsBL-1),1),(Ni,NPtsBL,1))
    x, y = J.getxy(Distrib4Extrusion)

    if Topology == 'C':

        # Build FarWakeExtrude distribution law (1D curve)
        dH = cells['Farfield']*cells['WakeFarfieldAspectRatio']
        FarWakeDistrib = D.line(
            (0,0,0),
            (0,dH*(NPtsBL-1),0),
            NPtsBL)
        FarWakeDistribY = J.gety(FarWakeDistrib)
        MaxStretch = dH/WallCellHeight
        MinStretch = 1.0
        ExtrusionDistributions = []
        for i in range(Ni):
            x[i,:] = sEC[i]

            for j in range(NPtsBL):
                StretchFactor = np.maximum(WallCellHeightDist[i]/WallCellHeight,1.0)

                NewY = np.interp(StretchFactor,
                    [MinStretch, MaxStretch],
                    [BLdistribY[j], FarWakeDistribY[j]])

                y[i,j] = NewY
        y[0,:] = y[-1,:] = FarWakeDistribY

    elif Topology == 'O':
        for i in range(Ni):
            x[i,:] = sEC[i]
            y[i,:] = BLdistribY

    I._rmNodesByType([ExtrusionCurve,Distrib4Extrusion],'FlowSolution_t')

    ExtrudedMesh = G.hyper2D(ExtrusionCurve, Distrib4Extrusion,Topology)
    ExtrudedMesh[0] = 'ExtrudedMesh'


    if Topology == 'C':
        # Correct hyper2D mismatch (see Ticket #7517)
        xCurve, yCurve, zCurve = J.getxyz(ExtrusionCurve)
        xExtruded, yExtruded, zExtruded = J.getxyz(ExtrudedMesh)
        xExtruded[0,0] = xExtruded[-1,0] = 0.5*(xExtruded[0,1] + xExtruded[-1,1])
        yExtruded[0,0] = yExtruded[-1,0] = 0.5*(yExtruded[0,1] + yExtruded[-1,1])
        zExtruded[0,0] = zExtruded[-1,0] = 0.5*(zExtruded[0,1] + zExtruded[-1,1])
        # xExtruded[1:-1,0] = xCurve[1:-1]
        # yExtruded[1:-1,0] = yCurve[1:-1]
        # zExtruded[1:-1,0] = zCurve[1:-1]
        xExtruded[:,0] = xCurve[:]
        yExtruded[:,0] = yCurve[:]
        zExtruded[:,0] = zCurve[:]


        # Compute Cell Heights (j-direction)
        CellHeight, = J.invokeFields(ExtrudedMesh, ['CellHeight'])
        x,y,z = J.getxyz(ExtrudedMesh)
        for j in range(1,NPtsBL):
            CellHeight[:,j] =np.sqrt((x[:,j]-x[:,j-1])**2 +
                                     (y[:,j]-y[:,j-1])**2 +
                                     (z[:,j]-z[:,j-1])**2)
        CellHeight[0,:] = CellHeight[1,:]
        NPtsFarfield = pts['Extrusion']-NPtsBL+2

        if not isFoilClosed:
            # Build Wake zone AFTER Extrusion
            # because hyper2D produced mismatch (see Ticket #7517)
            ContourExtrusion = T.subzone(ExtrudedMesh,(1,1,1),(Ni,1,1))
            SideTop     = T.subzone(ContourExtrusion,(Ni-pts['Wake']+1,1,1),(Ni,1,1))
            SideBottom  = T.subzone(ContourExtrusion,(1,1,1),(pts['Wake'],1,1))
            xTop, yTop = J.getxy(SideTop)
            xBot, yBot = J.getxy(SideBottom)
            if xTop[-1]-xTop[0] < 0: T._reorder(SideTop,(-1,2,3))
            if xBot[-1]-xBot[0] < 0: T._reorder(SideBottom,(-1,2,3))
            xTop, yTop = J.getxy(SideTop)
            xBot, yBot = J.getxy(SideBottom)
            if yTop[0] < yBot[0]: SideTop, SideBottom = SideBottom, SideTop
            xTop, yTop = J.getxy(SideTop)
            xBot, yBot = J.getxy(SideBottom)

            SideBottom[0] = 'SideBottom'
            SideTop[0]    = 'SideTop'
            wires += [SideBottom, SideTop]

            VerticalLines = []
            for i in range(pts['Wake']):
                VerticalLine = W.linelaw(
                    P1=(xBot[i],yBot[i],0),
                    P2=(xTop[i],yTop[i],0),
                    N=NPtsWakeHeight, Distribution=dict(
                        kind='tanhTwoSides',
                        FirstCellHeight=CellHeightFieldBottom[i],
                        LastCellHeight=CellHeightFieldTop[i]))
                VerticalLines += [VerticalLine]
            WakeZone = G.stack(VerticalLines,None)
            WakeZone[0] = 'WakeZone'
            surfs += [WakeZone]



        FarBottom = W.linelaw(
            P1=(size['Wake'],-size['Height'],0),
            P2=(1,-size['Height'],0),
            N=pts['Wake'],
            Distribution=dict(
                kind='tanhTwoSides',
                FirstCellHeight=cells['Farfield'],
                LastCellHeight=cells['Farfield']*cells['FarfieldAspectRatio'],))
        FarBottom[0] = 'FarBottom'

        FarTop = W.linelaw(
            P1=(1,+size['Height'],0),
            P2=(size['Wake'],+size['Height'],0),
            N=pts['Wake'],
            Distribution=dict(
                kind='tanhTwoSides',
                FirstCellHeight=cells['Farfield']*cells['FarfieldAspectRatio'],
                LastCellHeight=cells['Farfield'],))
        FarTop[0] = 'FarTop'

        wires += [FarBottom, FarTop]

        BLedge = T.subzone(ExtrudedMesh,(1,NPtsBL,1),(Ni,NPtsBL,1))
        BLedge[0] = 'BLedge'
        BLedgeY = J.gety(BLedge)
        if BLedgeY[-1] < BLedgeY[0]: T._reorder(BLedge,(-1,2,3))




        BLedgeBottom = T.subzone(BLedge, (1,1,1),(pts['Wake'],1,1))
        BLedgeTop    = T.subzone(BLedge, (Ni-pts['Wake']+1,1,1),(Ni,1,1))
        BLedgeFoilBot= T.subzone(BLedge, (pts['Wake'],1,1),(pts['Wake']+NPtsBottom-1,1,1))
        BLedgeFoilTop= T.subzone(BLedge, (pts['Wake']+NPtsBottom-1,1,1),(Ni-pts['Wake']+1,1,1))
        BLedgeBottom[0] = 'BLedgeBottom'
        BLedgeTop[0] = 'BLedgeTop'
        BLedgeFoilBot[0] = 'BLedgeFoilBot'
        BLedgeFoilTop[0] = 'BLedgeFoilTop'

        BLortoImin   = T.subzone(ExtrudedMesh,  (1,1,1),  (1,NPtsBL,1))
        BLortoImax   = T.subzone(ExtrudedMesh, (Ni,1,1), (Ni,NPtsBL,1))
        BLortoBottom = T.subzone(ExtrudedMesh, (pts['Wake'],1,1),(pts['Wake'],NPtsBL,1))
        BLortoLE     = T.subzone(ExtrudedMesh, (pts['Wake']+NPtsBottom-1,1,1),(pts['Wake']+NPtsBottom-1,NPtsBL,1))
        BLortoTop    = T.subzone(ExtrudedMesh, (Ni-pts['Wake']+1,1,1),(Ni-pts['Wake']+1,NPtsBL,1))

        BLortoImin[0]   = 'BLortoImin'
        BLortoImax[0]   = 'BLortoImax'
        BLortoBottom[0] = 'BLortoBottom'
        BLortoLE[0]     = 'BLortoLE'
        BLortoTop[0]    = 'BLortoTop'

        wires += [BLedgeBottom,BLedgeFoilBot,BLedgeFoilTop,BLedgeTop,
            BLortoBottom,BLortoLE,BLortoTop, BLortoImin, BLortoImax]

        Arc    = D.circle((1,0,0), size['Height'], tetas=90., tetae=270., N=180*4+1)
        Arc    = C.convertArray2Tetra(Arc)
        Arc[0] = 'Arc'
        # wires += [Arc]

        # Project LE on Arc - makes use of intersector
        x,y,z            = J.getxyz(BLortoLE)
        StartPoint       = np.array([x[-1],y[-1],z[-1]])
        BeforeStartPoint = np.array([x[-2],y[-2],z[-2]])
        lineLEdir        = StartPoint-BeforeStartPoint
        lineLEdir       /= np.sqrt(lineLEdir.dot(lineLEdir))
        EndPoint         = StartPoint + lineLEdir * 2.* size['Height']
        lineLE           = C.convertArray2Tetra(D.line(StartPoint,EndPoint,2))
        lineLE[0]        = 'lineLE'

        # COMPUTE ARCS
        # Reorder split arcs so that they are clockwise and name
        # them appropriately
        Arcs = W.splitCurves(Arc, lineLE, select=1)
        barycenters = [G.barycenter(arc) for arc in Arcs]

        if barycenters[1][1] > barycenters[0][1]:
            ArcBottom, ArcTop = Arcs
        else:
            ArcTop, ArcBottom = Arcs
        x,y = J.getxy(ArcBottom)
        if x[0] < x[-1]: T._reorder(ArcBottom,(-1,2,3))
        x,y = J.getxy(ArcTop)
        if x[0] > x[-1]: T._reorder(ArcTop,(-1,2,3))
        ArcTop[0] = 'ArcTop'
        ArcBottom[0] = 'ArcBottom'
        CellJoinHeight = cells['Farfield']*cells['FarfieldAspectRatio']
        LECell         = cells['Farfield']*cells['LEFarfieldAspectRatio']
        ArcBottom = W.discretize(ArcBottom, N=C.getNPts(BLedgeFoilBot),
            Distribution=dict(
                kind='tanhTwoSides',
                FirstCellHeight=CellJoinHeight,
                LastCellHeight=LECell))
        ArcTop = W.discretize(ArcTop, N=C.getNPts(BLedgeFoilTop),
            Distribution=dict(
                kind='tanhTwoSides',
                FirstCellHeight=LECell,
                LastCellHeight=CellJoinHeight))
        wires += [ArcBottom, ArcTop]

        # Add Leading-edge farfield propagation line
        curve1 = BLedgeFoilBot
        curve2 = ArcBottom
        xC1, yC1 = J.getxy(curve1)
        CellHeight, = J.getVars(curve1, ['CellHeight'])
        xC2, yC2 = J.getxy(curve2)
        LEfarLine = W.linelaw(
            P1=(xC1[-1],yC1[-1],0),
            P2=(xC2[-1],yC2[-1],0),
            N=NPtsFarfield,
            Distribution=dict(
                kind='tanhTwoSides',
                FirstCellHeight=CellHeight[-1],
                LastCellHeight=cells['Farfield']))
        LEfarLine[0] = 'LEfarLine'
        wires += [LEfarLine]

        # BUILD BOTTOM-WAKE DOMAIN
        curve1 = BLedgeBottom
        curve2 = FarBottom
        xC1, yC1 = J.getxy(curve1)
        CellHeight, = J.getVars(curve1, ['CellHeight'])
        xC2, yC2 = J.getxy(curve2)
        VerticalLines = []
        for i in range(pts['Wake']):
            VerticalLine = W.linelaw(
                P1=(xC1[i],yC1[i],0),
                P2=(xC2[i],yC2[i],0),
                N=NPtsFarfield, Distribution=dict(
                    kind='tanhTwoSides',
                    FirstCellHeight=CellHeight[i],
                    LastCellHeight=cells['Farfield']))
            VerticalLines += [VerticalLine]

        BottomWake = G.stack(VerticalLines,None)
        BottomWake[0] = 'BottomWake'
        surfs += [BottomWake]

        LowerRightLine = VerticalLines[0]
        LowerRightLine[0] = 'LowerRightLine'
        wires += [LowerRightLine]

        MiddleDownLine = VerticalLines[-1]
        MiddleDownLine[0] = 'MiddleDownLine'
        wires += [MiddleDownLine]



        # BUILD TOP-WAKE DOMAIN
        curve1 = BLedgeTop
        curve2 = FarTop
        xC1, yC1 = J.getxy(curve1)
        CellHeight, = J.getVars(curve1, ['CellHeight'])
        xC2, yC2 = J.getxy(curve2)
        VerticalLines = []
        for i in range(pts['Wake']):
            VerticalLine = W.linelaw(
                P1=(xC1[i],yC1[i],0),
                P2=(xC2[i],yC2[i],0),
                N=NPtsFarfield, Distribution=dict(
                    kind='tanhTwoSides',
                    FirstCellHeight=CellHeight[i],
                    LastCellHeight=cells['Farfield']))
            VerticalLines += [VerticalLine]
        TopWake = G.stack(VerticalLines,None)
        TopWake[0] = 'TopWake'
        surfs += [TopWake]

        MiddleTopLine = VerticalLines[0]
        MiddleTopLine[0] = 'MiddleTopLine'
        wires += [MiddleTopLine]

        UpperRightLine = VerticalLines[-1]
        UpperRightLine[0] = 'UpperRightLine'
        wires += [UpperRightLine]


        # BUILD UPSTREAM-BOTTOM DOMAIN
        UpstreamBottom = G.TFI([MiddleDownLine,LEfarLine,BLedgeFoilBot,ArcBottom])
        UpstreamBottom[0] = 'UpstreamBottom'
        surfs += [UpstreamBottom]

        # BUILD UPSTREAM-TOP DOMAIN
        UpstreamTop = G.TFI([LEfarLine,MiddleTopLine,BLedgeFoilTop,ArcTop])
        UpstreamTop[0] = 'UpstreamTop'
        surfs += [UpstreamTop]

        # --- Mesh is built. Join subparts --- #

        # Split ExtrudedMesh (boundary-layer) in 4 subparts
        ExtrudedMeshBottomRear = T.subzone(ExtrudedMesh,(1,1,1),(pts['Wake'],NPtsBL,1))
        ExtrudedMeshBottomRear[0] = 'ExtrudedMeshBottomRear'
        surfs += [ExtrudedMeshBottomRear]

        ExtrudedMeshBottomFront = T.subzone(ExtrudedMesh,(pts['Wake'],1,1),(pts['Wake']+NPtsBottom-1,NPtsBL,1))
        ExtrudedMeshBottomFront[0] = 'ExtrudedMeshBottomFront'
        surfs += [ExtrudedMeshBottomFront]

        ExtrudedMeshTopFront = T.subzone(ExtrudedMesh,(pts['Wake']+NPtsBottom-1,1,1),(pts['Wake']+NPtsBottom-1+NPtsTop,NPtsBL,1))
        ExtrudedMeshTopFront[0] = 'ExtrudedMeshTopFront'
        surfs += [ExtrudedMeshTopFront]

        ExtrudedMeshTopRear = T.subzone(ExtrudedMesh,(pts['Wake']+NPtsBottom-1+NPtsTop,1,1),(-1,-1,-1))
        ExtrudedMeshTopRear[0] = 'ExtrudedMeshTopRear'
        surfs += [ExtrudedMeshTopRear]


        I._rmNodesByType(surfs,'FlowSolution_t')
        if isFoilClosed:
            BottomRear = T.join([ExtrudedMeshBottomRear,BottomWake])
            BottomRear[0] = 'BottomRear'
            surfs += [BottomRear]

            BottomFront = T.join([ExtrudedMeshBottomFront,UpstreamBottom])
            BottomFront[0] = 'BottomFront'
            surfs += [BottomFront]

            TopFront = T.join([ExtrudedMeshTopFront,UpstreamTop])
            TopFront[0] = 'TopFront'
            surfs += [TopFront]

            TopRear = T.join([ExtrudedMeshTopRear,TopWake])
            TopRear[0] = 'TopRear'
            surfs += [TopRear]

            grid = T.join([BottomRear, BottomFront, TopFront, TopRear])
            grid[0] = 'grid'
            surfs += [grid]

            zones = [grid]
        else:
            Rear = T.join([BottomWake,ExtrudedMeshBottomRear,WakeZone,ExtrudedMeshTopRear,TopWake])
            Rear[0] = 'Rear'
            T._reorder(Rear,(2,1,3))

            BottomFront = T.join([ExtrudedMeshBottomFront,UpstreamBottom])
            BottomFront[0] = 'BottomFront'
            surfs += [BottomFront]

            TopFront = T.join([ExtrudedMeshTopFront,UpstreamTop])
            TopFront[0] = 'TopFront'
            surfs += [TopFront]

            Front = T.join([BottomFront, TopFront])
            Front[0] = 'Front'

            surfs += [Rear,Front]

            zones = [Rear,Front]
    elif Topology == 'O':
        zones = [ExtrudedMesh]

    t = C.newPyTree(['Base',zones])

    # boundary conditions
    X.connectMatch(t, tol=1.e-10, dim=2)
    base, = I.getBases(t)

    if Topology == 'C':
        if isFoilClosed:
            C._addBC2Zone(grid,'BC_imin','FamilySpecified:%s'%opts['FarfieldFamilyName'],'imin')
            C._addBC2Zone(grid,'BC_imax','FamilySpecified:%s'%opts['FarfieldFamilyName'],'imax')
            C._addBC2Zone(grid,'BC_jmax','FamilySpecified:%s'%opts['FarfieldFamilyName'],'jmax')
        else:
            C._addBC2Zone(Rear,'BC_jmin','FamilySpecified:%s'%opts['FarfieldFamilyName'],'jmin')
            C._addBC2Zone(Rear,'BC_jmax','FamilySpecified:%s'%opts['FarfieldFamilyName'],'jmax')
            C._addBC2Zone(Rear,'BC_imax','FamilySpecified:%s'%opts['FarfieldFamilyName'],'imax')
            C._addBC2Zone(Front,'BC_jmax','FamilySpecified:%s'%opts['FarfieldFamilyName'],'jmax')

    elif Topology == 'O':
        C._addBC2Zone(zones[0],'BC_jmax','FamilySpecified:%s'%opts['FarfieldFamilyName'],'jmax')

    C._fillEmptyBCWith(t,'airfoil','FamilySpecified:%s'%opts['AirfoilFamilyName'],dim=2)

    C._addFamily2Base(base, opts['FarfieldFamilyName'], bndType='BCFarfield')
    C._addFamily2Base(base, opts['AirfoilFamilyName'],  bndType='BCWallViscous')


    # Splitting and distribution
    if opts['NumberOfProcessors'] > 1:
        I._rmNodesByType(t,'ZoneGridConnectivity_t')

        # Perform Splitting and distribution
        t = T.splitNParts(t, opts['NumberOfProcessors'], multigrid=0, dirs=[1,2], recoverBC=True)

        # Re-Connect the resulting blocks
        t = X.connectMatch(t, dim=2, tol=1e-10)

        # Force check multiply-defined zone names
        t = I.correctPyTree(t,level=3)
        silence = J.OutputGrabber()
        with silence: t,stats=D2.distribute(t, opts['NumberOfProcessors'], useCom=0)

        # Check if all procs have at least one block assigned
        zones = I.getZones(t)
        ProcDistributed = [I.getValue(I.getNodeFromName(z,'proc')) for z in zones]

        for p in range(max(ProcDistributed)):
            if p not in ProcDistributed:
                raise ValueError('Bad proc distribution! rank %d is empty'%p)

    # Move Family nodes at end of base (conventional)
    base, = I.getBases(t)
    familyNodes = I.getNodesFromType1(base,'Family_t')
    I._rmNodesByType1(base,'Family_t') # detach
    base[2] += familyNodes             # attach

    T._addkplane(t,1) # control depth ?
    T._translate(t,(0.0,0.0,-0.5))

    meshParamsDict = dict(Sizes=size,Points=pts,Cells=cells,options=opts,
        References=dict(Reynolds=Reynolds,
                        DeltaYPlus=DeltaYPlus,
                        Chord=Chord,
                        Depth=1.0, # CAVEAT: measure this
                        TorqueOrigin=[xmin+0.25*Chord,0,0],
                        ))

    return t, meshParamsDict


def buildWatertightBodyFromSurfaces(walls):
    '''
    Given a set of surfaces, construct a closed watertight surface.
    Resulting surface is unstructured of triangles elements (TRI)

    Parameters
    ----------

        walls : :py:class:`list` of zone
            multiple blocks defining the surface patches of
            where the body will be supported

    Returns
    -------

        body : zone
            watertight body surface unstructured (TRI)
    '''
    tri = convertSurfaces2SingleTriangular(walls)
    body = G.gapsmanager(tri)
    body = T.join(body)
    G._close(body)
    body, = I.getZones(body)

    return body


def convertSurfaces2SingleTriangular(t):
    '''
    Given a set of surfaces, transform them into a single unstructured TRI
    surface (merging).

    Parameters
    ----------

        t : PyTree, base, zone, or list of zones
            contains the surfaces to merge

    Returns
    -------

        tri : zone
            a single surface (TRI) connecting all provided surfaces
    '''
    tri = C.convertArray2Tetra(t)
    I._rmNodesByType(tri, 'FlowSolution_t') # required by join
    tri = T.join(tri)
    tri,= I.getZones(tri)

    return tri


def buildPropellerDisc(Rmin=0.1, Rmax=0.6, NCellRadial=30, NCellQuarter=20,
                       buildHub=True, RadialBufferFractionOfRmax=0.4,
                       NCellRadialBuffer=11, RadialCellSizeBoundary=0.06,
                       ghostHubCartesianFractionFromRmin=0.80):
    '''
    Build a 2D flat propeller disc on XY plane.

    Parameters
    ----------

        Rmin : float
            minimum radius of propeller

        Rmax : float
            maximum radius of propeller

        NCellRadial : int
            number of radial cells between **Rmin** and **Rmax**

        NCellQuarter : int
            number of azimutal cells in 90 degrees

        ghostHubCartesianFractionFromRmin : float
            fraction of **Rmin** from which
            middle hub zone (diamond) is constructed and number of connecting
            cells is computed (recommended value is ``0.8``)

    Returns
    -------

        t : PyTree
            2D Propeller disc PyTree
    '''
    def newQuarter(zone, i):
        return T.rotate(zone, (0,0,0),(0,0,1),90.*(i+1))

    RmaxWithBuffer = Rmax * ( 1 + RadialBufferFractionOfRmax )
    RadialCellSize = (Rmax-Rmin)/float(NCellRadial)

    # Wireframe
    arcMin = D.circle( (0,0,0), Rmin,
                       tetas=0., tetae=90., N=NCellQuarter+1 )
    arcMax = D.circle( (0,0,0), RmaxWithBuffer,
                       tetas=0., tetae=90., N=NCellQuarter+1 )

    RadialConnectorStart = D.line( (Rmin,0,0), (Rmax,0,0), NCellRadial+1 )
    RadialConnectorStartBuffer = W.linelaw(P1=(Rmax,0,0),
        P2=(RmaxWithBuffer,0,0), N=NCellRadialBuffer+1,
        Distribution=dict(kind='tanhTwoSides',
                          FirstCellHeight=RadialCellSize,
                          LastCellHeight=RadialCellSizeBoundary))
    RadialConnectorStart = T.join(RadialConnectorStart,
                                  RadialConnectorStartBuffer)

    RadialConnectorEnd   = D.line( (0,Rmin,0), (0,Rmax,0), NCellRadial+1 )
    RadialConnectorEndBuffer = W.linelaw(P1=(0,Rmax,0),
        P2=(0,RmaxWithBuffer,0), N=NCellRadialBuffer+1,
        Distribution=dict(kind='tanhTwoSides',
                          FirstCellHeight=RadialCellSize,
                          LastCellHeight=RadialCellSizeBoundary))
    RadialConnectorEnd = T.join(RadialConnectorEnd,
                                RadialConnectorEndBuffer)

    SingleQuarterCircleSurface = G.TFI([arcMin, arcMax,
                                      RadialConnectorStart, RadialConnectorEnd])

    CylinderSurfaces=[newQuarter(SingleQuarterCircleSurface, i) for i in range(3)]
    CylinderSurfaces.append(SingleQuarterCircleSurface)
    CylinderSurface = T.join(CylinderSurfaces)
    CylinderSurface[0] = 'PropDisc'

    zones = [CylinderSurface]

    if buildHub:
        discFrac = ghostHubCartesianFractionFromRmin
        cartLine = D.line((discFrac*Rmin,0,0),(0,discFrac*Rmin,0),NCellQuarter+1)
        Gap2Cylinder = (1-discFrac)*Rmin
        NCellJoin = int(2*Gap2Cylinder/RadialCellSize + 1)
        cart2cyl1 = D.line((discFrac*Rmin,0,0), (Rmin,0,0), NCellJoin+1)
        cart2cyl2 = newQuarter(cart2cyl1, 0)
        SingleGhostHubSurf = G.TFI([cartLine, arcMin, cart2cyl1, cart2cyl2])

        GhostHubSurfaces=[newQuarter(SingleGhostHubSurf, i) for i in range(3)]
        GhostHubSurfaces.append(SingleGhostHubSurf)
        CartSurfBnds = [newQuarter(cartLine, i) for i in range(3)]
        CartSurfBnds.append(cartLine)
        CartSurf = G.TFI(CartSurfBnds)
        CartSurf[0] = 'HubCenter'
        GhostHubSurface = T.join(GhostHubSurfaces)
        GhostHubSurface[0] = 'HubTrans'


        # Smooth of GhostHubSurface
        fixedBoundaries = [newQuarter(arcMin, i) for i in range(3)]
        fixedBoundaries.append(arcMin)
        T._smooth([GhostHubSurface,CartSurf], type=0, niter=100,
                  fixedConstraints=fixedBoundaries)
        zones.extend([GhostHubSurface, CartSurf])

    t = C.newPyTree(['Base', zones])
    t = T.reorderAll(t)

    return t


def isClosed(surface, tol=1e-8):
    '''
    Determines if a surface is closed or not.

    .. hint:: see ticket `7867 <https://elsa.onera.fr/issues/7867>`_

    Parameters
    ----------

        surface : PyTree, base, zone, or list of zones
            input surface (or set of surfaces) where the closed topology will be
            evaluated.

        tol : float
            geometrical tolerance to merge points when closing the
            auxiliary surface used for determining if input is closed.

    Returns
    -------

        closed : bool
            if :py:obj:`True`, surface topology is closed, :py:obj:`False` otherwise.
    '''
    # See ticket #7867
    surfTRI = C.convertArray2Tetra(surface)
    surfTRI = T.join(surfTRI)
    G._close(surfTRI, tol=tol)
    try:
        ExteriorFaces = P.exteriorFaces(surfTRI)
        return False
    except ValueError:
        return True


def getAreas(t):
    '''
    Compute the total area of each surface zone provided by the user.

    Parameters
    ----------

        t : PyTree, base, zone or list of zones
            object containing exclusively the surfaces whose area will be
            computed.

    Returns
    -------

        areas : :py:class:`list` of :py:class:`float`
            the areas of each one of the user-provided surfaces
    '''
    areas = []
    for zone in I.getZones(t):
        surf = I.rmNodesByType(zone, 'FlowSolution_t')
        C._initVars(surf,'f',1.)
        areas += P.integ(surf, 'f')
    return areas


def filterSurfacesByArea(surfaces, ratio=0.5):
    '''
    Distribute a user-provided list of surfaces into two lists (large surfaces
    and small surfaces). Large surfaces are those whose area is bigger than
    the median. If the selection criterion is requested to be different, then
    user may change the parameter ratio. If ``ratio=0.5``, then the selection
    criterion corresponds to the median.

    Parameters
    ----------

        surfaces : PyTree, base or list of zones
            inputs surfaces to filter.

    Returns
    -------

        LargeSurfaces : :py:class:`list` of zone
            surfaces whose area is greater or equal
            than ``ratio*(max(areas)+min(areas))``

        SmallSurfaces : :py:class:`list` of zone
            surfaces whose area is less than ``ratio*(max(areas)+min(areas))``
    '''
    surfzones = I.getZones(surfaces)
    areas = getAreas(surfzones)
    median = ratio*(max(areas)+min(areas))

    LargeSurfaces, SmallSurfaces = [], []
    for zone, area in zip(surfzones, areas):
        if area >= median:
            LargeSurfaces.append(zone)
        else:
            SmallSurfaces.append(zone)

    return LargeSurfaces, SmallSurfaces

def _reorderAll(*args):
    if verbose: return T._reorderAll(*args)
    silence = J.OutputGrabber()
    with silence: T._reorderAll(*args)


def makeH(boundaries, inner_contour, inner_cell_size=0.1,
          outter_cell_size=0.1, number_of_points_union=21,
          inner_normal_tension=0.5, outter_normal_tension=0.5,
          projection_support=None, global_projection_relaxation=0.,
          local_projection_relaxation_length=0.,forced_split_index=None):
    '''
    Build an H-mesh surface from four structured curves as exterior boundary and
    an single structure curve as interior contour.

    Parameters
    ----------

        boundaries : :py:class:`list` of zone
            Four (4) structured curves. They are the external boundaries that
            the H-grid must verify.

        inner_contour : zone
            A single (1) structured curve. This is the inside contour that the
            H-grid must verify.

            .. warning::
                The total number of segments of **boundaries** must be equal
                to the total number of segments of **inner_contour**

        inner_cell_size : float
            Size of the **inner_contour** -adjacent cell

        outter_cell_size : float
            Size of the **boundaries** -adjacent cell

        number_of_points_union : int
            Number of points in the boundary-normal direction

        inner_normal_tension : float
            Specific tension (relative to the distance between **inner_contour**
            and **boundaries**) of the **inner_contour** -adjacent cell normal
            direction.

            .. note::
                higher values may improve the orthogonality of the mesh
                in the neighborhood of the inside contour, but this can
                deteriorate the mesh quality further away.

        outter_normal_tension : float
            Specific tension (relative to the distance between **inner_contour**
            and **boundaries**) of the **boundaries** -adjacent cell normal
            direction.

            .. note::
                higher values may improve the orthogonality of the mesh
                in the neighborhood of the boundaries contour, but this can
                deteriorate the mesh quality further away.

        projection_support : zone or :py:obj:`None`
            if provided (not :py:obj:`None`), this is a surface of projection
            of the support for the H-grid construction. For example, this may
            be the surface of a hub, spinner or fuselage.

            .. warning::
                if **projection_support** is provided, the user must guarantee
                that both **boundaries** and **inner_contour** are already
                projected onto **projection_support** before calling this
                function

    Returns
    -------

        surfaces : :py:class:`list` of zone
            list of four (4) structured surfaces that constitute the H-grid.
    '''
    outter_boundaries, inner_boundaries,_ = W.splitInnerContourFromOutterBoundariesTopology(
                                boundaries, inner_contour,forced_split_index)

    if len(outter_boundaries) != len(inner_boundaries):
        raise ValueError(J.FAIL+'number of curves in outter_boundaries must be the same as inner_boundaries'+J.ENDC)

    if allHaveNormals( inner_contour ):
        J.migrateFields(inner_contour, inner_boundaries)
    else:
        W.addNormals(inner_boundaries, projection_support)

    if allHaveNormals( boundaries ):
        J.migrateFields(boundaries, outter_boundaries)
    else:
        W.computeBarycenterDirectionalField(outter_boundaries, projection_support)


    def applyRelaxation(original, projected):

        if global_projection_relaxation == 1:
            return I.copyRef(original)
        lr = local_projection_relaxation_length
        relaxed = I.copyTree( original )
        s = W.gets( relaxed )
        L = D.getLength( relaxed )
        x, y, z = J.getxyz( relaxed )
        xo, yo, zo = J.getxyz( original )
        xp, yp, zp = J.getxyz( projected )
        OPx = xp-xo
        OPy = yp-yo
        OPz = zp-zo

        d = np.sqrt( OPx**2 + OPy**2 + OPz**2 )
        not_matching = d > 1e-10
        v = np.vstack((OPx, OPy, OPz))
        v[:,not_matching] /= d[not_matching]
        local_relax = np.interp(s,[0,lr/L],[0,1])
        # local_relax[local_relax<1] = np.sqrt(local_relax[local_relax<1])
        relaxation = d*local_relax*(1-global_projection_relaxation)
        x[:] = xo + v[0,:] * relaxation
        y[:] = yo + v[1,:] * relaxation
        z[:] = zo + v[2,:] * relaxation

        I._rmNodesByType(relaxed,'FlowSolution_t')

        return relaxed

    MinDistances = []
    for out_bnd, in_bnd in zip(outter_boundaries, inner_boundaries):
        MinDistances += [W.pointwiseDistances(in_bnd, out_bnd)[0]]
    MinDistance = np.min( MinDistances )


    TFI_surfs = []
    for out_bnd, in_bnd in zip(outter_boundaries, inner_boundaries):
        sx, sy, sz = J.getVars(in_bnd, ['sx', 'sy', 'sz'])
        sx_o, sy_o, sz_o = J.getVars(out_bnd, ['sx', 'sy', 'sz'])
        union_curves = []
        for i in range(len(sx)):
            start_pt = W.point(in_bnd,i)
            start_normal = np.array([sx[i],sy[i],sz[i]])
            end_pt = W.point(out_bnd,i)
            end_normal = np.array([sx_o[i],sy_o[i],sz_o[i]])
            start_to_end = W.distance(start_pt, end_pt)
            bezier_start_tension_pt = start_pt+inner_normal_tension*MinDistance*start_normal
            bezier_end_tension_pt = end_pt+outter_normal_tension*start_to_end*end_normal
            bezier_pts = D.polyline([tuple(start_pt),
                                     tuple(bezier_start_tension_pt),
                                     tuple(bezier_end_tension_pt),
                                     tuple(end_pt)])

            if projection_support:
                bezier_pts_proj = T.projectOrtho(bezier_pts, projection_support)

            fine_bezier = D.bezier(bezier_pts,N=500)

            if projection_support:
                fine_bezier_proj = T.projectOrtho(fine_bezier, projection_support)
                fine_bezier = applyRelaxation(fine_bezier, fine_bezier_proj)

            union_curve = W.discretize(fine_bezier, N=number_of_points_union,
                                Distribution={'kind':'tanhTwoSides',
                                              'FirstCellHeight':inner_cell_size,
                                              'LastCellHeight':outter_cell_size})

            if projection_support:
                union_curve_proj = T.projectOrtho(union_curve, projection_support)
                union_curve = applyRelaxation(union_curve, union_curve_proj)
                xu, yu, zu = J.getxyz( union_curve )
                xu[0] = start_pt[0]
                yu[0] = start_pt[1]
                zu[0] = start_pt[2]
                xu[-1] = end_pt[0]
                yu[-1] = end_pt[1]
                zu[-1] = end_pt[2]

            union_curves += [ union_curve ]

        TFI_surf = G.stack(union_curves)
        TFI_surf[0] = in_bnd[0].replace('.inner','') + '.TFI'
        TFI_surfs += [ TFI_surf ]

    return TFI_surfs


def allHaveNormals(t):
    has_normals = True
    for z in I.getZones(t):
        FlowSol = I.getNodeFromName(z, I.__FlowSolutionNodes__)
        if FlowSol is None: return False
        sx = I.getNodeFromName1(FlowSol, 'sx')
        sy = I.getNodeFromName1(FlowSol, 'sy')
        sz = I.getNodeFromName1(FlowSol, 'sz')
        has_normals = True
        for i in sx, sy, sz:
            if i is None:
                return False
    return has_normals

def _alignNormalsWithRadialCylindricProjection(t, rotation_center, rotation_axis):
    alignmentTol=1.0-1.e-10
    c = np.array(rotation_center,dtype=np.float64)
    a = np.array(rotation_axis,dtype=np.float64)
    q = c + a
    qc = c - q
    for zone in I.getZones(t):
        x,y,z = J.getxyz(zone)
        sx,sy,sz = J.getVars(zone,['sx','sy','sz'])
        x = x.ravel(order='F')
        y = y.ravel(order='F')
        z = z.ravel(order='F')
        sx = sx.ravel(order='F')
        sy = sy.ravel(order='F')
        sz = sz.ravel(order='F')
        for i in range( len(x) ):
            p = np.array([x[i],y[i],z[i]],dtype=float)
            v = np.array([sx[i],sy[i],sz[i]],dtype=float)
            v_norm = np.sqrt(v.dot(v))
            qp = p - q
            qp /= np.sqrt(qp.dot(qp))
            alignment = np.abs(qp.dot(a))
            if alignment >= alignmentTol: continue
            n = np.cross(qc,qp)
            b = np.cross(n,v)
            t = np.cross(b,n)
            t /= np.sqrt(t.dot(t))
            t *= v_norm
            sx[i] = t[0]
            sy[i] = t[1]
            sz[i] = t[2]


def _hasMatchingFace(contour_struct, faces):
    for f in faces:
        for c in contour_struct:
            if C.getNPts(c) != C.getNPts(f): continue
            if W.isSubzone(f, c): return True
            if W.isSubzone(c, f): return True
    return False

def selectConnectingSurface(surfaces, candidates, mode='first'):
    contours = [P.exteriorFacesStructured(s) for s in surfaces]

    returned_surfaces = []
    for candidate in candidates:
        exterior_faces = P.exteriorFacesStructured( candidate )
        for contour in contours:
            match = True
            if not _hasMatchingFace(contour, exterior_faces):
                match = False
                break
        if match:
            if mode == 'first':
                return candidate
            elif mode == 'all':
                returned_surfaces += [ candidate ]
            else:
                raise ValueError('mode %s not recognized'%mode)

    if not returned_surfaces:
        t = C.newPyTree(['SURFACES',surfaces,
                         'CANDIDATES',candidates])
        C.convertPyTree2File(t,'debug.cgns')
        raise ValueError('no matching surfaces. Check debug.cgns.')

    return returned_surfaces

def writeGeomTurbo(Rows, Hub, Shroud, Zmin=None, Zmax=None, filename='geometry.geomTurbo', unit='Meters'):
    '''
    Write a geomTurbo file for Autogrid5 given a geometry for rows, hub and shroud built with MOLA.

    Parameters
    ----------
    Rows : dict 
        Dictionary of rows to write in the geomTurbo file. 
        For instance:

        .. code-block:: python

            Rows = dict( 
                Rotor = dict(
                    Periodicity = 30,
                    ShroudGap = 1e-4, # optional
                    # Blades are given with as a PyTree or a filename 
                    # It refers to the blades in the current row
                    # Most of the time, there is a single blade defined per row.
                    Blades = RotorBlades  # or getSectionsForAutogrid(RotorBlades)
                ), 
                Stator = dict(
                    Periodicity = 40,
                    # HubGap = 2e-4, # optional
                    Blades = StatorBlades
                ), 
            )
        
        .. important:: 
            The axis of the machine is Z. Please apply a rotation if necessary before using this function.

    Hub : PyTree Zone
        Zone corresponding to hub line. Axial direction must be Z and radial direction must be X.
    Shroud : PyTree Zone
        Zone corresponding to shroud line. Axial direction must be Z and radial direction must be X.
    Zmin : float, optional
        Bottom threshold in the axial direction (threshold on hub and shroud lines).
        If not given, it is max(min(zHub, zShroud)).
    Zmax : float, optional
        Top threshold in the axial direction (threshold on hub and shroud lines).
        If not given, it is min(max(zHub, zShroud)).
    filename : str, optional
        Name of the written file, by default 'geometry.geomTurbo'
    unit : str, optional
        Unit in the geomTurbo file, by default 'Meters'
    '''
    # WATCH OUT: There must not be any empty line between the lines
    # defining each section for SS and PS. 
    # If there is, the geomTurbo file cannot be read correctly by Autogrid v>9

    if not filename.endswith('geomTurbo'): 
        filename += '.geomTurbo'
    print(J.CYAN+f'Start to build the geometry file {filename} for Autogrid5...'+J.ENDC)
    
    floatFormat = '.15g' # Above that, Autogrid will not save additional figures

    def writeZRCurve(fout, name, z, r, ZminForCurve=-1e20, ZmaxForCurve=1e20):
        txt = f"""
  NI_BEGIN basic_curve
    NAME     {name}
    NI_BEGIN zrcurve
    ZR
"""
        indexMin = np.amin(np.argwhere(z>ZminForCurve))
        indexMax = np.amax(np.argwhere(z<ZmaxForCurve))

        txt += f'    {indexMax-indexMin+1}\n'
        for item in range(indexMin, indexMax+1):
            txt += f'    {z[item]:{floatFormat}} {r[item]:{floatFormat}}\n'
        txt += '    NI_END zrcurve\n    NI_END   basic_curve'
        fout.write(txt)
    
    def writeChannelCurve(fout, name, curveNames):
        '''
        Write the channel curve called **name**, written as the concatenation of zero, one or several curves
        already defined. CurveNames is the list of their names.
        '''
        txt = f"""
  NI_BEGIN channel_curve {name}    
    NAME     {name}
"""     
        # First point of each curve
        for curveName in curveNames:
            txt += f'    VERTEX   CURVE_P {curveName} 0\n'
        # Last point of the last curve
        if len(curveNames) != 0:
            txt += f'    VERTEX   CURVE_P {curveName} 1\n'
        txt += f'    NI_END   channel_curve {name}'
        fout.write(txt)


    # Write the header of the file
    fout = open(filename,'w')
    fout.write(f"""GEOMETRY TURBO
VERSION    5.6
TOLERANCE  1e-06
UNITS        {unit}
UNITS-FACTOR 1

NI_BEGIN CHANNEL""")
    
    # Axial direction must be Z and radial direction must be X
    xHub, _, zHub = J.getxyz(Hub)
    xShroud, _, zShroud = J.getxyz(Shroud)

    # Detection of bulb geometry if hub radius reaches zero (upstream or downstream)
    if xHub[0] < 1e-12:
        print('  --> Bulb geometry detected upstream')
        xHub[0] = 0
        BulbGeometryUpstream = True
    else:
        BulbGeometryUpstream = False
    if xHub[-1] < 1e-12:
        print('  --> Bulb geometry detected downstream')
        xHub[-1] = 0
        BulbGeometryDownstream = True
    else:
        BulbGeometryDownstream = False

    if Zmin is None: 
        if BulbGeometryUpstream:
            Zmin = zShroud[0]
        else:
            Zmin = max(zHub[0], zShroud[0])
    if Zmax is None: 
        if BulbGeometryDownstream:
            Zmax = zShroud[-1]
        else:
            Zmax = min(zHub[-1], zShroud[-1])
    print(f'  Zmin={Zmin}, Zmax={Zmax}')
    
    hubCurves = []
    # Eventual extension of hub upstream (with a direct line, not a spline)
    if zHub[0] > Zmin and not BulbGeometryUpstream:
        print('  Extend the hub upstream until Zmin')
        writeZRCurve(fout, 'hub_upstream', np.array([Zmin, zHub[0]]), np.array([xHub[0], xHub[0]]))
        hubCurves.append('hub_upstream')
    # Main curve defining hub
    writeZRCurve(fout, 'hub_main', zHub, xHub, Zmin, Zmax) 
    hubCurves.append('hub_main')
    # Eventual extension of hub downstream (with a direct line, not a spline)
    if zHub[-1] < Zmax and not BulbGeometryDownstream:
        print('  Extend the hub downstream until Zmax')
        writeZRCurve(fout, 'hub_downstream', np.array([zHub[-1], Zmax]), np.array([xHub[-1], xHub[-1]]))
        hubCurves.append('hub_downstream')

    shroudCurves = []
    # Eventual extension of shroud upstream (with a direct line, not a spline)
    if zShroud[0] > Zmin:
        print('  Extend the shroud upstream until Zmin')
        writeZRCurve(fout, 'shroud_upstream', np.array([Zmin, zShroud[0]]), np.array([xShroud[0], xShroud[0]]))
        shroudCurves.append('shroud_upstream')
    # Main curve defining shroud
    writeZRCurve(fout, 'shroud_main', zShroud, xShroud, Zmin, Zmax)
    shroudCurves.append('shroud_main')
    # Eventual extension of shroud downstream (with a direct line, not a spline)
    if zShroud[-1] < Zmax:
        print('  Extend the shroud downstream until Zmax')
        writeZRCurve(fout, 'shroud_downstream', np.array([zShroud[-1], Zmax]), np.array([xShroud[-1], xShroud[-1]]))
        shroudCurves.append('shroud_downstream')

    # Write the block with Channel curves. There must be curves named hub and shroud, and eventually a nozzle (for bypass configuration)
    writeChannelCurve(fout, 'hub', hubCurves)
    writeChannelCurve(fout, 'shroud', shroudCurves)
    # writeChannelCurve(fout, 'nozzle', [])

    fout.write('\n  NI_END   CHANNEL\n')


               
    for row, rowParams in Rows.items():
        print(f'  Build ROW {row} (Periodicity={rowParams["Periodicity"]})')
        rowParams.setdefault('RotationSpeed', 500)      
        fout.write(f"""
NI_BEGIN nirow
  NAME      {row}
  TYPE      normal
  PERIODICITY     {rowParams['Periodicity']}
  ROTATION_SPEED  {rowParams['RotationSpeed']}""")
        
        if not isinstance(rowParams['Blades'], dict):
            rowParams['Blades'] = getSectionsForAutogrid(rowParams['Blades'])
        
        for blade, bladeParams in rowParams['Blades'].items():
            print(f'    Build BLADE {blade} ({len(bladeParams["suction"])} sections)')
            bladeParams.setdefault('ExpandTip', 0.05)
            bladeParams.setdefault('ExpandRoot', 0.05)

            fout.write(f"""
  NI_BEGIN NIBlade
    NAME      {blade}""")
            
            if 'HubGap' in bladeParams:
                print(f'      with hub gap = {bladeParams["HubGap"]}')
                fout.write(f"""
    NI_BEGIN NIRootGap
      WIDTH_AT_LEADING_EDGE  {bladeParams['HubGap']}
      WIDTH_AT_TRAILING_EDGE {bladeParams['HubGap']}
      NI_END   NIRootGap""")
            
            if 'ShroudGap' in bladeParams:
                print(f'      with shroud gap = {bladeParams["HubGap"]}')
                fout.write(f"""
    NI_BEGIN NITipGap
      WIDTH_AT_LEADING_EDGE  {bladeParams['ShroudGap']}
      WIDTH_AT_TRAILING_EDGE {bladeParams['ShroudGap']}
      NI_END   NITipGap""")
                
            fout.write(f"""
    NI_BEGIN nibladegeometry
      TYPE  GEOMTURBO
      blade_expansion_factor_hub    {bladeParams['ExpandRoot']}
      blade_expansion_factor_shroud   {bladeParams['ExpandTip']}""")
            
            for oneside in ['suction','pressure']:
                sections = bladeParams[oneside]
                print(f'      {sections[0][0].size} points found on {oneside} side (for section 1)')
                fout.write(f"""
      {oneside}
        SECTIONAL
        {len(sections)}
""")
                for item in range(len(sections)):
                    NbOfPoints = sections[item][0].size
                    fout.write(f"""        # section {item+1}
        XYZ
        {NbOfPoints}
""")
                    for i in range(NbOfPoints):
                        fout.write(f'        {sections[item][0][i]:{floatFormat}} {sections[item][1][i]:{floatFormat}} {sections[item][2][i]:{floatFormat}}\n')
            
            fout.write("""
      NI_END   nibladegeometry
    NI_END   NIBlade""")

        fout.write("""
  NI_END   nirow
""")
    
    fout.write("""
NI_END GEOMTURBO
NI_BEGIN GEOMETRY
NI_END   GEOMETRY
""")

    fout.close()
    print(J.GREEN+f'> File {filename} written'+J.ENDC)

def getSectionsForAutogrid(mesh, indexLE=None):
    '''
    From a given **mesh**, get the sections and organize them as expected for a .geomTurbo file. 
    For each section, suction side et pressure side are separated and points are sorted from 
    leading edge to trailing edge. 

    Parameters
    ----------
    mesh : str or PyTree
        Row mesh, gievn as a filename or a PyTree
         
    indexLE : int, optional
        Index of the leading edge on the section.
        If not given, it is taken at N//2, where N is the number of points on the section.

    Returns
    -------
    dict
        dictionary of sections to give to :py:func:`WriteGeomTurbo`

    '''

    if isinstance(mesh,str):
        t = C.convertFile2PyTree(mesh)
    elif I.isStdNode(mesh) != -2:
        # Node or list of nodes
        t = mesh
    else:
        raise ValueError('parameter mesh must be either a filename or a PyTree')

    Blades = dict()
    for blade in I.getZones(t):
        bladeName = I.getName(blade)
        Blades[bladeName] = dict(
            suction = [],
            pressure = []
        )
        
        x, y, z = J.getxyz(blade)
        if indexLE is None:
            # By default, SS and PS are assumed to be discretized with the same number of points
            indexLE = x.shape[0] // 2 
                
        for j in range(x.shape[1]):
            coordsPS = np.array([x[:indexLE+1, j], y[:indexLE+1, j], z[:indexLE+1, j]])
            coordsSS = np.array([x[indexLE:, j], y[indexLE:, j], z[indexLE:, j]])

            # CAUTION: In Autogrid5, points for SS and PS must be ordered from LE to TE !!!
            # Flip points in order to respect this requirement
            coordsPS = np.fliplr(coordsPS)
            
            Blades[bladeName]['suction'].append(coordsSS)
            Blades[bladeName]['pressure'].append(coordsPS)
    
    return Blades


