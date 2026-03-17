import numpy as np

from .Global import *
from .Kinematics import *
from .AirfoilPolarsInterpolators import *
from .LiftingLine import (_applyPolarOnLiftingLine,
                          getAirfoilsNodeOfLiftingLine,
                          buildLiftingLine,
                          mirrorBlade,
                          remapLiftingLine)

try:

    from mola.pytree import InternalShortcuts as J
    import mola.mesh.curve as W
    import mola.mesh.surface as GSD

    import Geom.PyTree as D

except ModuleNotFoundError as e:

    msg = "not all modules requiring Cassiopee are available. Usage of LiftingLine AdvancedOps is limited."
    raise ModuleNotFoundError(msg) from e


def buildLiftingLineFromScan(t, SpanwiseRediscretization=None, resetPitchRelativeSpan=0.75,
        GeometricalLawsInterpolations={}, OverridingKinematics={}):
    '''
    This function takes as input the result of :py:func:`MOLA.GenerativeShapeDesign.scanBlade`
    and builds a LiftingLine in its canonical position. The resulting LiftingLine
    can be used directly (polars file is required) as element for BFM or VPM
    modeling, for instance. 

    Parameters
    ----------

        t : PyTree
            Rigorously, the result of :py:func:`MOLA.GenerativeShapeDesign.scanBlade`

        SpanwiseRediscretization : multiple
            **doc this**

        resetPitchRelativeSpan : float
            This is the relative span (or :math:`r/R`) used to relocate the blade
            such that the section at **resetPitchRelativeSpan** yields zero twist.
            This is important, since the actual sweep and dihedral directions
            depend on the pitch value of the blade. In order to circumvent this
            ambiguity, sweep and dihedral positions are defined using as reference
            a relocation of the blade such that the section located at
            **resetPitchRelativeSpan** has zero twist.


        GeometricalLawsInterpolations : dict
            Each keyword correspond to a GeometricalLaw name that can be passed to
            :py:func:`buildLiftingLine`. Also, such keyword must correspond to 
            a field named contained in **t** zone *BladeLine*. The value 
            associated to each keyword is a :py:class:`str` specifying the 
            type of interpolation law. By default, if not specified all 
            geometrical laws are set using the interpolation law ``'interp1d_linear'``

        OverridingKinematics : dict
            By default (if **OverridingKinematics** is an empty 
            :py:class:`dict`) function :py:func:`setKinematicsUsingConstantRotationAndTranslation`
            will be applied using the scanned RotationCenter, axis and direction.
            If **OverridingKinematics** is provided by user, then such options 
            will override the scanned ones.

    Returns
    -------

        LiftingLine : zone
            The new lifting-line located at canonical position. 

        PyZonePolars : :py:class:`list` of zones
            The airfoil polars (only coordinates, no fields), which can be used
            to make a 3D reconstruction of the blade using :py:func:`postLiftingLine2Surface`
        
        scanPitch : float
            The pitch of the section at **resetPitchRelativeSpan**
    '''
    t = I.copyRef(t)

    BaseBladeLine = I.getNodeFromName1(t,'BaseBladeLine')
    if not BaseBladeLine:
        msg = 'could not find base "BaseBladeLine"\n'
        msg += 'Please make sure the first argument is the output of GSD.scanBlade'
        raise IOError(J.FAIL+msg+J.ENDC)

    BladeLine = I.getNodeFromName1(BaseBladeLine,'BladeLine')
    if not BaseBladeLine:
        msg = 'could not find zone "BladeLine"\n'
        msg += 'Please make sure the first argument is the output of GSD.scanBlade'
        raise IOError(J.FAIL+msg+J.ENDC)

    scan_info = J.get(BladeLine,'ScanInformation')
    
    # PUT BLADELINE IN CANONICAL POSITION ROTATING RELEVANT FIELDS
    # put blade line into rotation center (0,0,0)
    T._translate(BladeLine, -scan_info['RotationCenter'])

    RightHandRuleRotation=bool(scan_info['RightHandRuleRotation'])
    Dir = 1 if RightHandRuleRotation else -1

    # put blade line into canonical orientation
    FinalFrame = [(1,0,0),(0,1,0),(0,0,1)]
    InitialFrame = [tuple(scan_info['PitchAxis']),
        tuple(np.cross(Dir*scan_info['RotationAxis'],scan_info['PitchAxis'])),
        tuple(Dir*scan_info['RotationAxis'])]
    T._rotate(BladeLine, (0,0,0), InitialFrame, FinalFrame,
        vectors=NamesOfChordSpanThickwiseFrameNoTangential)


    # determine the pitch of the blade-line
    Abscissa = W.gets(BladeLine)
    Twist, SpanScan = J.getVars(BladeLine,['Twist','Span'])
    RelativeSpanScan = SpanScan/SpanScan.max()
    pitch = np.interp(resetPitchRelativeSpan, RelativeSpanScan, Twist)
    print(f'scanned pitch at r/R={resetPitchRelativeSpan} is {pitch} deg')

    # substract pitch to put blade line in canonical position
    PitchCtr = J.getVars(BladeLine,
        ['PitchRelativeCenterX','PitchRelativeCenterY','PitchRelativeCenterZ'])
    PitchAxis = J.getVars(BladeLine, ['PitchAxisX','PitchAxisY','PitchAxisZ'])
    for i in range(3): PitchAxis[i][:] *= Dir
    PitchCtr_pt = (PitchCtr[0][0]*1.0, PitchCtr[1][0]*1.0, PitchCtr[2][0]*1.0)
    PitchAxis_vec = (PitchAxis[0][0], PitchAxis[1][0], PitchAxis[2][0])
    T._rotate(BladeLine, PitchCtr_pt, PitchAxis_vec, -pitch, 
            vectors=NamesOfChordSpanThickwiseFrameNoTangential)
    
    
    # INFER SWEEP AND DIHEDRAL FROM COORDINATES
    x, y, z = J.getxyz(BladeLine)
    Span, Sweep, Dihedral = J.invokeFields(BladeLine, ['Span', 'Sweep','Dihedral'])
    Span[:]=x
    Sweep[:] = -y
    Dihedral[:] = z
    
    
    # BUILD GEOMETRICALLAWS 
    airfoils = I.getZones(I.getNodeFromName1(t,'BaseNormalizedAirfoils'))
    GeometricalLawsNamesList = ['Chord', 'Twist', 'Sweep','Dihedral', 'Airfoils',
        'ChordwiseX','ChordwiseY','ChordwiseZ',
         'SpanwiseX', 'SpanwiseY', 'SpanwiseZ',
        'ThickwiseX','ThickwiseY','ThickwiseZ']
    
    GeometricalLawsDict = {}
    for geom in GeometricalLawsNamesList:
        # GeometricalLawsDict[geom] = {'RelativeSpan' : RelativeSpanScan}
        GeometricalLawsDict[geom] = {'Abscissa' : Abscissa} # better this (monotonically increasing)
        if geom == 'Airfoils':
            GeometricalLawsDict[geom]['PyZonePolarNames'] = [a[0] for a in airfoils]
        else:
            GeometricalLawsDict[geom][geom] = J.getVars(BladeLine,[geom])[0]
        try:
            GeometricalLawsDict[geom]['InterpolationLaw'] = GeometricalLawsInterpolations[geom]
        except KeyError:
            GeometricalLawsDict[geom]['InterpolationLaw'] = 'interp1d_linear'       

    # Create the lifting line
    if not RightHandRuleRotation: mirrorBlade(BladeLine)
    PitchCtr = J.getVars(BladeLine,
        ['PitchRelativeCenterX','PitchRelativeCenterY','PitchRelativeCenterZ'])
    PitchAxis = J.getVars(BladeLine, ['PitchAxisX','PitchAxisY','PitchAxisZ'])
    PitchCtr_pt = (PitchCtr[0][0]*1.0, PitchCtr[1][0]*1.0, PitchCtr[2][0]*1.0)
    PitchAxis_vec = (PitchAxis[0][0]*1.0, PitchAxis[1][0]*1.0, PitchAxis[2][0]*1.0)

    # build auxiliar curve from user-defined SpanwiseRediscretization
    if isinstance(SpanwiseRediscretization, int):
        SpanwiseCurve = W.discretize(BladeLine, SpanwiseRediscretization)
    elif W.isStructuredCurve(SpanwiseRediscretization):
        SpanwiseCurve = G.map(BladeLine, D.getDistribution(SpanwiseRediscretization))
    elif SpanwiseRediscretization is None:
        SpanwiseCurve = BladeLine
    else:
        dist = J.getDistributionFromHeterogeneousInput__(SpanwiseRediscretization)[-1]
        SpanwiseCurve = G.map(BladeLine,dist)

    LiftingLine = buildLiftingLine(SpanwiseCurve, RightHandRuleRotation=RightHandRuleRotation,
        PitchRelativeCenter=PitchCtr_pt, PitchAxis=PitchAxis_vec,
        RotationCenter=[0,0,0], **GeometricalLawsDict)

    # propagate kinematics information, but keep canonical position
    scan_info['RotationAxis'] *= Dir # since in kinematics it is actually ThrustAxis
    scan_info.update(OverridingKinematics)
    del scan_info['PitchAxis']
    setKinematicsUsingConstantRotationAndTranslation(LiftingLine, **scan_info)

    # CONSTRUCT COORDINATE-ONLY PYZONE POLARS
    PyZonePolarsGeometryOnly = []
    for zone in I.getZones(I.getNodeFromName1(t,'BaseNormalizedAirfoils')):
        airfoil = I.copyRef(zone)
        node = I.getNodeFromName1(airfoil, 'GridCoordinates')
        node[0] = '.Polar#FoilGeometry'
        node[3] = 'UserDefinedData_t'
        I._rmNodesByName1(node,'CoordinateZ')
        I._rmNodesByType1(airfoil,'FlowSolution_t')
        PyZonePolarsGeometryOnly.append(airfoil)
    

    return LiftingLine, PyZonePolarsGeometryOnly, pitch

def makeBladeSurfaceFromLiftingLineAndAirfoilsPolars(LiftingLine, AirfoilsPolars,
        blade_radial_NPts=100,
        blade_root_cellwidth=0.02, blade_tip_cellwidth=1e-3,
        airfoil_NPts_top=99, airfoil_NPts_bottom=99,
        airfoil_LeadingEdge_width_relative2chord=0.001,
        airfoil_TrailingEdge_width_relative2chord=0.01,
        airfoil_LeadingEdge_abscissa=0.49,
        airfoil_stacking_point_relative2chord=0.25,
        ):

    if isinstance(LiftingLine, str):
        LiftingLine = C.convertFile2PyTree(LiftingLine)
    if isinstance(AirfoilsPolars, str):
        AirfoilsPolars = C.convertFile2PyTree(AirfoilsPolars)

    LiftingLine = I.getZones(LiftingLine)[0]
    AirfoilsPolars = I.getZones(AirfoilsPolars)
    Span, = J.getVars(LiftingLine, ['Span'])
    RadialRelativeDiscretization = dict( N=blade_radial_NPts, kind='tanhTwoSides',
                                 FirstCellHeight=blade_root_cellwidth/Span.max(),
                                 LastCellHeight=blade_tip_cellwidth/Span.max() )

    FoilDistribution=[dict(N=airfoil_NPts_bottom,
                           BreakPoint=airfoil_LeadingEdge_abscissa,
                           kind='tanhTwoSides',
                           FirstCellHeight=airfoil_TrailingEdge_width_relative2chord,
                           LastCellHeight=airfoil_LeadingEdge_width_relative2chord),
                      dict(N=airfoil_NPts_top,
                           BreakPoint=1.0,
                           kind='tanhTwoSides',
                           FirstCellHeight=airfoil_LeadingEdge_width_relative2chord,
                           LastCellHeight=airfoil_TrailingEdge_width_relative2chord),]

    LiftingLine = remapLiftingLine(LiftingLine, RadialRelativeDiscretization)
    blade = postLiftingLine2Surface(LiftingLine, AirfoilsPolars,
                                       ChordRelRef = airfoil_stacking_point_relative2chord,
                                       FoilDistribution=FoilDistribution)
    blade[0] = 'blade'

    return blade

def postLiftingLine2Surface(LiftingLine, PyZonePolars, Variables=[],
                            ChordRelRef=0.25, FoilDistribution=None):
    '''
    Post-process a **LiftingLine** element using enhanced **PyZonePolars** data
    in order to build surface fields (like ``Cp``, ``theta``...) from a BEMT, 
    VPM or BodyForce solution.

    Parameters
    ----------

        LiftingLine : zone
            Result of a BEMT, VPM or BodyForce computation.

        PyZonePolars : :py:func:`list` of :py:func:`zone` or :py:class:`str`
            Enhanced **PyZonePolars** for each airfoil, containing also foilwise
            distributions fields (``Cp``, ``theta``, ``delta1``...).

            .. note::
              if input type is a :py:class:`str`, then **PyZonePolars** is
              interpreted as a CGNS file name containing the airfoil polars data

        Variables : :py:class:`list` of :py:class:`str`
            The variables to be built on the newly created surface.
            For example:

            >>> Variables = ['Cp', 'theta']

        ChordRelRef : float
            Reference chordwise used for stacking the sections.

        FoilDistribution : zone or :py:obj:`None`
            As established in :py:func:`MOLA.Wireframe.useEqualNumberOfPointsOrSameDiscretization`


    Returns
    -------

        Surface : zone
            structured surface containing fields at ``FlowSolution``, where the
            variables requested by the user are interpolated.
    '''
    import scipy.interpolate as si
    def _applyInterpolationFunctionToSpanwiseVariableAtLiftingLine(VariableArray, Var, InterpolationLaw):
        '''
        Perform spanwise interpolation of PyZonePolar data
        contained in AllValues dictionary. For this, use
        the general-purpose macro interpolation function:
        Define a general-purpose macro interpolation function.
        '''
        InterpAxis = 2
        if 'interp1d' in InterpolationLaw.lower():
            ScipyLaw = InterpolationLaw.split('_')[1]
            try:
                interp = si.interp1d( Abscissa, VariableArray, axis=InterpAxis,
                                      kind=ScipyLaw, bounds_error=False,
                                      fill_value='extrapolate', assume_sorted=True)
            except ValueError:
                ErrMsg = 'FATAL ERROR during _applyInterpolationFunctionToSpanwiseVariableAtLiftingLine() call with Var=%s\n'%Var
                ErrMsg+= 'Shapes x and y = %d and %d\n'%(len(x),len(y))
                raise ValueError(ErrMsg)

        elif 'pchip' == InterpolationLaw.lower():
            interp = si.PchipInterpolator(Abscissa, VariableArray, axis=InterpAxis, extrapolate=True)

        elif 'akima' == InterpolationLaw.lower():
            interp = si.Akima1DInterpolator(Abscissa, VariableArray, axis=InterpAxis)

        elif 'cubic' == InterpolationLaw.lower():
            interp = si.CubicSpline(Abscissa, VariableArray, axis=InterpAxis, extrapolate=True)

        else:
            raise AttributeError('applyPolarOnLiftingLine(): InterpolationLaw %s not recognized.'%InterpolationLaw)

        # Prepare data and ship them to SurfVars
        Res = interp(s) # Apply spanwise interpolation function
        ResultSlices = [Res[:,isec,isec] for isec in range(len(s))]
        MyArr = np.vstack(ResultSlices)
        MyArr = MyArr.T
        SurfVars[Var][:] = MyArr

    Surfs = []
    LiftingLines = getLiftingLines(LiftingLine)
    for LiftingLine in LiftingLines:
        v = J.getAllVars(LiftingLine)
        x,y,z = J.getxyz(LiftingLine)
        

        # recover the airfoils at each node of the LiftingLine
        PolarInfoNode = getAirfoilsNodeOfLiftingLine(LiftingLine)
        Abscissa = I.getValue(I.getNodeFromName1(PolarInfoNode, 'Abscissa'))
        PyZonePolarNames = I.getValue(I.getNodeFromName1(PolarInfoNode, 'PyZonePolarNames')).split(' ')
        InterpLaw = I.getValue(I.getNodeFromName1(PolarInfoNode, 'InterpolationLaw'))
        order = J._inferOrderFromInterpLawName(InterpLaw)
        AirfoilsGeom = pyZonePolar2AirfoilZone(PyZonePolarNames,PyZonePolars)
        AirfoilsGeom = W.useEqualNumberOfPointsOrSameDiscretization(AirfoilsGeom, FoilDistribution)
        AirfoilsGeom = W.interpolateAirfoils(AirfoilsGeom, Abscissa, v['s'], order=order)

        # position and resize airfoils
        AirfoilFrame = [(1,0,0),(0,1,0),(0,0,1)]
        for i, foil in enumerate(AirfoilsGeom):
            foil_x = J.getx(foil)
            foil_y = J.gety(foil)
            
            # center at stacking point
            foil_x -= ChordRelRef

            # resize using chord
            foil_x *= v['Chord'][i]
            foil_y *= v['Chord'][i]

            # rotate to match the actual position in the lifting-line
            LLframe = [(v['ChordwiseX'][i],v['ChordwiseY'][i],v['ChordwiseZ'][i]),
                    (v['ThickwiseX'][i],v['ThickwiseY'][i],v['ThickwiseZ'][i]),
                    (-v['SpanwiseX'][i],-v['SpanwiseY'][i],-v['SpanwiseZ'][i])]
            T._rotate(foil, (0,0,0), AirfoilFrame, LLframe)

            # center at the actual Lifting-Line node
            T._translate(foil,(x[i],y[i],z[i]))
        
        # stack all sections
        Surf = G.stack(AirfoilsGeom)
        Surf[0] = LiftingLine[0]+'_surf'
        Surfs.append( Surf )



    if len(Variables) == 0:
        if len(Surfs) == 1: return Surfs[0]
        else: return Surfs

    for Surf, LiftingLine in zip(Surfs, LiftingLines):
        s = W.gets(LiftingLine)
        # Invoke the new variables in surface
        SurfVars = J.invokeFieldsDict(Surf,Variables)

        # Build interpolator functions and store them as dict:
        # usage: InterpDict[<PyZonePolarName>](AoA,Mach,Reynolds,[])
        InterpDict = buildPolarsInterpolatorDict(PyZonePolars,InterpFields=Variables)

        AoA, Mach, Reynolds = J.getVars(LiftingLine,["AoA", "Mach", "Reynolds"])


        # Apply polar interpolations and store them in a dict
        AllFoilNPts = C.getNPts(AirfoilsGeom[0])
        RefCurvAbs = W.gets(AirfoilsGeom[0])
        AllValues = {}
        for pzn in PyZonePolarNames:
            InterpolatedSet = InterpDict[pzn](AoA,Mach,Reynolds)
            # NOTA BENE: InterpolatedSet is a list of arrays.
            # Each element is a (FoilNPts x LLNpts) array of the
            # interpolated variable in the same order as contained
            # in list Variables.

            # Adapt the interpolated data if necessary (adaptedSet)
            adaptedSet = []
            for v in range(len(Variables)):
                InterpolatedArray = InterpolatedSet[v]
                IntArrayShape = InterpolatedArray.shape
                print('Variable %s at polar %s has shape: %s'%(Variables[v],pzn,str(IntArrayShape)))
                if len(IntArrayShape)==2:

                    # Compute the PyZonePolar foilwise abscissa
                    # For that, build an auxiliar foil and
                    # compute its abcissa coordinate
                    AuxFoil = pyZonePolar2AirfoilZone(pzn,PyZonePolars)
                    CurrentCurvAbs = W.gets(AuxFoil)

                    interpFoilwise = si.interp1d(CurrentCurvAbs, InterpolatedArray,
                                        kind='cubic', copy=False, axis=0,
                                        assume_sorted=True, fill_value="extrapolate")
                    NewInterpArray = interpFoilwise(RefCurvAbs)

                    # TODO: Check orientation of foil and data
                    adaptedSet += [NewInterpArray]

                elif len(IntArrayShape)==1:
                    # Integral data. Simply broadcast.
                    print('BROADCAST')
                    adaptedSet += [np.broadcast_to(InterpolatedArray,(AllFoilNPts,IntArrayShape[0]))]
                else:
                    raise ValueError('Interpolated data for variable %s yields not supported shape %s'%(v,str(IntArrayShape)))

            # Store dimensionally-coherent interpolated data
            AllValues[pzn] = adaptedSet

        for v in range(len(Variables)):
            # Build a 3D matrix containing all data.
            # 1st dimension: Foilwise data
            # 2nd dimension: Spanwise data
            # 3rd dimension: slices corresponding to PyZonePolars
            AllValues3D = np.dstack([AllValues[pzn][v] for pzn in PyZonePolarNames])
            _applyInterpolationFunctionToSpanwiseVariableAtLiftingLine(AllValues3D, Variables[v], 'interp1d_linear')

    if len(Surfs) == 1: return Surfs[0]
    else: return Surfs

def _findOptimumAngleOfAttackOnLiftLine(LiftingLine, PolarsInterpolatorDict,
        Aim='Cl', AimValue=0.5, AoASearchBounds=(-2,6),
        SpecificSections=None, ListOfEquations=[]):
    """
    Update ``AoA`` field with the optimum angle of attack based on a given
    **Aim**, using the provided **PolarsInterpolatorDict** as well as the existing
    ``Reynolds`` and ``Mach`` number values contained in ``FlowSolution`` container.

    Parameters
    ----------

        LiftingLine : zone
            the lifting line where ``AoA`` field will be updated

            .. note:: zone **LiftingLine** is modified

        PolarsInterpolatorDict : dict
            dictionary of
            interpolator functions of 2D polars, as obtained from
            :py:func:`buildPolarsInterpolatorDict` function.

        Aim : str
            can be one of:

            * ``'Cl'``
                aims a requested ``Cl`` (:math:`c_l`) value (provided by argument
                **AimValue**) throughout the entire lifting line

            * ``'minCd'``
                aims the minimum ``Cd`` value, :math:`\min (c_d)`

            * ``'maxClCd'``
                aims the maximum ``Cl/Cd`` value, :math:`\max (c_l / c_d)`

        AimValue : float
            Specifies the aimed value for corresponding relevant
            Aim types.

            .. note:: currently, only relevant for **Aim** = ``'Cl'``

        AoASearchBounds : :py:class:`tuple` of 2 :py:class:`float`
            Since there may exist multiple angle-of-attack (*AoA*) values
            verifying the requested conditions, this argument constraints the
            research interval of angle-of-attack of valid candidates.

        SpecificSections : :py:class:`list` of :py:class:`int`
            If specified (not :py:obj:`None`), only the
            indices corresponding to the user-provided sections are updated.

        ListOfEquations : :py:class:`list` of :py:class:`str`
            list of equations compatible with
            the syntax allowed in :py:func:`Converter.PyTree.initVars`
            (``FlowSolution`` located at vertex), in order to tune or correct
            the aerodynamic coefficients.
    """

    import scipy.optimize as so
    AoA, Cl, Cd, Mach, Reynolds = J.getVars(LiftingLine,['AoA','Cl','Cd','Mach', 'Reynolds'])

    if SpecificSections is None: SpecificSections = range(len(AoA))

    if Aim == 'Cl':
        for i in SpecificSections:
            def searchAoA(x,i):
                AoA[i] = x
                _applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict)
                [C._initVars(LiftingLine, eq) for eq in ListOfEquations]

                Residual = Cl[i]-AimValue
                return Residual

            sol=so.root_scalar(searchAoA, bracket=AoASearchBounds, x0=AoA[i], args=(i),  method='toms748')

            if sol.converged:
                searchAoA(sol.root,i)
            else:
                print ("Not found optimum AoA at section %d"%i)
                print (sol)
                continue
    elif Aim == 'maxClCd':
        for i in SpecificSections:
            def searchAoA(x,i):
                AoA[i] = x
                _applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict)
                [C._initVars(LiftingLine, eq) for eq in ListOfEquations]
                MinimizeThis = -Cl[i]/Cd[i]
                return MinimizeThis

            sol=so.minimize_scalar(searchAoA, bracket=[0,2], args=(i),  method='Golden',
                options=dict(xtol=0.01))


            if not sol.success:
                print ("Not found optimum AoA at section %d"%i)
                print (sol)
                continue
    elif Aim == 'minCd':
        for i in SpecificSections:
            def searchAoA(x,i):
                AoA[i] = x
                _applyPolarOnLiftingLine(LiftingLine,PolarsInterpolatorDict)
                [C._initVars(LiftingLine, eq) for eq in ListOfEquations]

                MinimizeThis = Cd[i]
                return MinimizeThis

            sol=so.minimize_scalar(searchAoA, bracket=AoASearchBounds, args=(i),  method='Golden',
                options=dict(xtol=0.01))

            if not sol.success:
                print ("Not found optimum AoA at section %d"%i)
                print (sol)
                continue

def addAccurateSectionArea2LiftingLine(LiftingLine, PyZonePolars):
    '''
    Add a field named ``SectionArea`` info the **LiftingLine** using the airfoil's
    geometry contained in **PyZonePolars**.

    Parameters
    ----------

        LiftingLine : zone
            the lifting line zone where ``SectionArea`` field is added

            .. note:: zone **LiftingLine** is modified

        PyZonePolars : :py:class:`list` of zone
            list of special **PyZonePolar** zones
            containing 2D polar information as well as the geometry. Specifically,
            the polars must contain the node ``.Polar#FoilGeometry`` with the
            children nodes ``CoordinateX`` and ``CoordinateY``.

    Returns
    -------

        LiftingLineSurface : zone
            the corresponding surface of the lifting line, that has been
            generated as an auxiliar item during the function call
    '''
    Surf = postLiftingLine2Surface(LiftingLine, PyZonePolars)
    SectionArea, = J.invokeFields(LiftingLine, ['SectionArea'])
    x = J.getx(Surf)
    NbOfSections = x.shape[1]
    for isec in range(NbOfSections):
        Section = GSD.getBoundary(Surf, 'jmin', layer=isec)
        Section = W.closeCurve(Section,NPts4closingGap=3, tol=1e-10)
        SectSurf = G.T3mesher2D(Section, triangulateOnly=1)
        G._getVolumeMap(SectSurf)
        PatchesAreas, = J.getVars(SectSurf, ['vol'],
                                    Container='FlowSolution#Centers')
        CurrentSectionArea = np.sum(PatchesAreas)
        SectionArea[isec] = CurrentSectionArea

    return SectSurf

def _makeFormat__(current, other):
    # current and other are axes
    def format_coord(x, y):
        # x, y are data coordinates
        # convert to display coords
        display_coord = current.transData.transform((x,y))
        inv = other.transData.inverted()
        # convert back to data coords with respect to ax
        ax_coord = inv.transform(display_coord)
        coords = [ax_coord, (x, y)]
        return ('Left: {:<40}    Right: {:<}'
                .format(*['({:.3f}, {:.3f})'.format(x, y) for x,y in coords]))
    return format_coord

def _plotChordAndTwist(LiftingLines, savefigname=None):
    '''
    Convenient function for interactively plot chord and twist
    geometrical laws of a list of Lifting Lines
    '''

    # TODO remove "_"

    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    Markers = ('o','+','s','.','v','^','x')
    for LiftingLine,m in zip(LiftingLines,Markers):
        r, Chord, Twist =  J.getVars(LiftingLine,['Span','Chord','Twist'])
        ax1.plot(r/r.max(),Chord,'-',marker=m,mfc='None',label=LiftingLine[0])
        ax2.plot(r/r.max(),Twist,'--',marker=m,mfc='None')

    ax1.set_xlabel('$r/R$')
    ax1.set_ylabel('Chord (m) (-)')
    ax2.set_ylabel('Twist (deg) (- - -)')

    ax2.format_coord = _makeFormat__(ax2, ax1)
    ax1.legend(loc='best')
    plt.tight_layout()
    if savefigname is not None: plt.savefig(savefigname)
    plt.show()

def _plotAoAAndCl(LiftingLines, savefigname=None):
    '''
    Convenient function for interactively plot AoA and Cl
    geometrical laws of a list of Lifting Lines
    '''

    # TODO remove "_"

    import matplotlib.pyplot as plt

    LiftingLines = I.getZones(LiftingLines)

    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    Markers = ('o','+','s','.','v','^','x')
    for LiftingLine in LiftingLines:
        if not checkComponentKind(LiftingLine, kind='LiftingLine'): continue
        r, AoA, Cl =  J.getVars(LiftingLine,['Span','AoA','Cl'])
        ax1.plot(r/r.max(),AoA,'-',marker='.',label=LiftingLine[0])
        ax2.plot(r/r.max(),Cl,'--',marker='.')

    ax1.set_xlabel('$r/R$')
    ax1.set_ylabel('AoA (deg) (-)')
    ax2.set_ylabel('Cl (- - -)')

    ax2.format_coord = _makeFormat__(ax2, ax1)
    ax1.legend(loc='upper left')
    plt.tight_layout()
    if savefigname is not None: plt.savefig(savefigname)
    plt.show()
