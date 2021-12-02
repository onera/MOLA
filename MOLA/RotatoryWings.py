'''
MOLA - RotatoryWings.py

This module proposes macro functionalities for rapid creation
and assembly of CFD simple cases of Rotatory Wings (Propellers,
Helicopter Rotors, Ducted Fans, etc)

This module makes use of Cassiopee modules.

File history:
19/03/2019 - v1.0 - L. Bernardos - Creation.
'''

# System modules
import sys
import os
import numpy as np
import numpy.linalg as npla
from scipy.spatial.transform import Rotation as ScipyRotation


# Cassiopee
import Converter.PyTree as C
import Converter.Internal as I
import Geom.PyTree as D
import Post.PyTree as P
import Generator.PyTree as G
import Transform.PyTree as T
import Connector.PyTree as X
import Intersector.PyTree as XOR


# Generative modules
from . import InternalShortcuts as J
from . import Wireframe as W
from . import GenerativeShapeDesign as GSD
from . import GenerativeVolumeDesign as GVD

def addPitchAndAdjustPositionOfBladeSurface(blade,
                                            root_window='jmin',
                                            delta_pitch_angle=0.0,
                                            pitch_center_adjust_relative2chord=0.5,
                                            pitch_axis=(0,0,-1),
                                            pitch_center=(0,0,0)):
    # TODO doc

    pitch_axis = np.array(pitch_axis,dtype=np.float)
    pitch_axis /= np.sqrt(pitch_axis.dot(pitch_axis))

    root = GSD.getBoundary(blade, root_window)
    root_camber = W.buildCamber(root)
    x,y,z = J.getxyz(root_camber)
    adjust_point = [x[0] + pitch_center_adjust_relative2chord * (x[-1] - x[0]),
                    y[0] + pitch_center_adjust_relative2chord * (y[-1] - y[0]),
                    z[0] + pitch_center_adjust_relative2chord * (z[-1] - z[0])]
    adjust_point = np.array(adjust_point, dtype=np.float)
    pitch_center = np.array(pitch_center, dtype=np.float)

    center2adjust_point = adjust_point - pitch_center
    distanceAlongAxis = center2adjust_point.dot(pitch_axis)
    pointAlongAxis = pitch_center + pitch_axis * distanceAlongAxis
    translationVector = pointAlongAxis - adjust_point

    T._translate(blade, translationVector)
    T._rotate(blade, pitch_center, pitch_axis, delta_pitch_angle)



def adjustSpinnerAzimutRelativeToBlade(spinner, blade, RotationAxis=(0,1,0),
                                        RotationCenter=(0,0,0)):
    # TODO doc
    spinnerSurfs = [c for c in I.getZones(spinner) if I.getZoneDim(c)[-1] == 2]

    spinnerTRI = C.convertArray2Tetra(spinnerSurfs)
    spinnerTRI = T.join(spinnerTRI)
    spinner_contour = P.exteriorFaces(spinnerTRI)

    blade = C.convertArray2Tetra(blade)
    blade = T.join(blade)
    blade_root = P.exteriorFaces(blade)
    blade_root = C.convertBAR2Struct(blade_root)
    blade_root = W.discretize(blade_root, N=500)
    root_barycenter = G.barycenter(blade_root)

    n = np.array(list(RotationAxis))
    Pt = np.array(list(root_barycenter))
    PlaneCoefs = n[0],n[1],n[2],-n.dot(Pt)
    C._initVars(spinner_contour,'Slice=%0.12g*{CoordinateX}+%0.12g*{CoordinateY}+%0.12g*{CoordinateZ}+%0.12g'%PlaneCoefs)
    zone = P.isoSurfMC(spinner_contour, 'Slice', 0.0)
    spinner_pt1, spinner_pt2 = T.splitConnexity(zone)


    def residual(angle):
        pt1, pt2 = T.rotate([spinner_pt1, spinner_pt2], RotationCenter, RotationAxis, angle)
        distance1 = W.distance(pt1,root_barycenter)
        distance2 = W.distance(pt2,root_barycenter)
        # print("angle = %g | dist1 = %g  dist2 = %g   diff = %g"%(angle,distance1, distance2,distance1-distance2))
        return distance1 - distance2

    solution = J.secant(residual, x0=-10., x1=-20., ftol=1e-6, #bounds=(50.,-50.),
                        maxiter=20,)

    if not solution['converged']:
        import pprint
        print(J.WARN+pprint.pformat(solution)+J.ENDC)

    T._rotate(spinner, RotationCenter, RotationAxis, solution['root'][0])


def joinSpinnerCurves(curves, LeadingEdgeNPts=20, TrailingEdgeNPts=20,
                              SpinnerFrontNPts=100, RearNPts=100,
                              RootCellLength=0.001,TrailingEdgeCellLength=0.02):
    '''
    Join the different spinner curves produced by :py:func:`makeSpinnerCurves`,
    resulting in a spinner profile that can be used by :py:func:`makeHub`.

    Parameters
    ----------

        curves : :py:class:`list` of zones
            result of :py:func:`makeSpinnerCurves`

        LeadingEdgeNPts : int
            desired number of points used to discretize the leading-edge arc

        TrailingEdgeNPts : int
            desired number of points used to discretize the trailing-edge arc,
            if it exists

        SpinnerFrontNPts : int
            desired number of points used to discretize the spinner's front side
            curve

        RearNPts : int
            desired number of points used to discretize the spinner's rear side
            curve

        RootCellLength : float
            desired size of the cell used to discretize the spinner's maximum
            width location, where the blade ought to be positioned

        TrailingEdgeCellLength : float
            if no trailing edge arc exists, then this value determines the
            length of the cell size located at the rear extremum

    Returns
    -------

        Profile : zone
            spinner profile curve ready to be used in :py:func:`makeHub`

    '''

    ArcFront = W.discretize(curves[0], LeadingEdgeNPts)
    ArcFront_x, ArcFront_y, ArcFront_z = J.getxyz(ArcFront)
    ArcJoinCellLength = np.sqrt( (ArcFront_x[1]-ArcFront_x[0])**2+
                                 (ArcFront_y[1]-ArcFront_y[0])**2+
                                 (ArcFront_z[1]-ArcFront_z[0])**2)

    SpinnerFront = W.discretize(curves[1],SpinnerFrontNPts,dict(kind='tanhTwoSides',
                                               FirstCellHeight=ArcJoinCellLength,
                                               LastCellHeight=RootCellLength))

    NumberOfCurves = len(curves)
    if NumberOfCurves == 3:
        Rear = W.discretize(curves[2],RearNPts,dict(kind='tanhTwoSides',
                                               FirstCellHeight=RootCellLength,
                                               LastCellHeight=TrailingEdgeCellLength))

        Profile = T.join(ArcFront, SpinnerFront)
        Profile = T.join(Profile,Rear)
    elif NumberOfCurves == 4:
        ArcRear = W.discretize(curves[2], TrailingEdgeNPts)
        ArcRear_x, ArcRear_y, ArcRear_z = J.getxyz(ArcRear)
        ArcJoinCellLengthRear = np.sqrt( (ArcRear_x[1]-ArcRear_x[0])**2+
                                         (ArcRear_y[1]-ArcRear_y[0])**2+
                                         (ArcRear_z[1]-ArcRear_z[0])**2)
        Rear = W.discretize(curves[3],RearNPts,dict(kind='tanhTwoSides',
                                               LastCellHeight=RootCellLength,
                                               FirstCellHeight=ArcJoinCellLengthRear))
        Profile = T.join(ArcFront, SpinnerFront)
        Profile = T.join(Profile,Rear)
        Profile = T.join(Profile,ArcRear)

    return Profile

def makeSpinnerCurves(LengthFront=0.2, LengthRear=1, Width=0.15,
                      RelativeArcRadiusFront=0.01, ArcAngleFront=40.,
                      RelativeTensionArcFront=0.1, RelativeTensionRootFront=0.5,
                      NPtsArcFront=200, NPtsSpinnerFront=5000,
                      TopologyRear='arc',
                      RelativeArcRadiusRear=0.0025, ArcAngleRear=70.,
                      RelativeTensionArcRear=0.1, RelativeTensionRootRear=0.5,
                      NPtsArcRear=200, NPtsSpinnerRear=5000):
    '''
    Construct the curves of the spinner profile corresponding to the front
    and rear sides, depending on the chosen topology.

    Most paramaters are equivalent as py:func:`makeFrontSpinnerCurves`, only
    the words *Front* or *Rear* are added to the parameter name in order to
    make the difference between the two parts of the spinner.

    .. note:: if **TopologyRear** = ``'arc'``, then the rear part of the spinner
        is consctructed calling py:func:`makeFrontSpinnerCurves` followed by
        a mirror operation.

    Parameters
    ----------

        TopologyRear : str
            if ``'arc'``, then the same topology as the front is used. If
            ``'line'``, then the rear of the spinner is built using a single
            line which extends from the root at the maximum diameter location

    Returns
    -------

        curves : :py:class:`list` of zones
            list of curves of the different parts of the spinner. These
            can be joined using :py:func:`joinSpinnerCurves`

    '''

    front = makeFrontSpinnerCurves(Length=LengthFront , Width=Width,
                         RelativeArcRadius=RelativeArcRadiusFront, ArcAngle=ArcAngleFront,
                         RelativeTensionArc=RelativeTensionArcFront, RelativeTensionRoot=RelativeTensionRootFront,
                         NPtsArc=NPtsArcFront, NPtsSpinner=NPtsSpinnerFront )

    if TopologyRear == 'arc':
        rear = makeFrontSpinnerCurves(Length=LengthRear, Width=Width,
                             RelativeArcRadius=RelativeArcRadiusRear, ArcAngle=ArcAngleRear,
                             RelativeTensionArc=RelativeTensionArcRear, RelativeTensionRoot=RelativeTensionRootRear,
                             NPtsArc=NPtsArcRear, NPtsSpinner=NPtsSpinnerRear)
        C._initVars(rear,'CoordinateY=-{CoordinateY}')

    elif TopologyRear == 'line':
        line = D.line((Width*0.5,0,0),(Width*0.5,-LengthRear,0),NPtsSpinnerRear)
        line[0] = 'rear'
        rear = [line]
    else:
        raise ValueError("TopologyRear='%s' not supported"%TopologyRear)

    curves = front + rear
    I._correctPyTree(curves,level=3)

    return curves

def makeFrontSpinnerCurves(Length=1., Width=0.6, RelativeArcRadius=0.01, ArcAngle=40.,
                           RelativeTensionArc=0.1, RelativeTensionRoot=0.5,
                           NPtsArc=200, NPtsSpinner=5000):
    '''
    Construct the profile curves of the front side of a spinner, which includes
    an arc in the leading-edge region, and a tangent curve which extends until
    the root position.

    Parameters
    ----------

        Length : float
            Distance (in the rotation axis direction) between the root position
            (corresponding to the maximum width of the spinner, where the blade
            ought to be located) and the leading edge of the spinner

        Width : float
            Maximum diameter of the spinner, which takes place at the root
            position, where the blade ought to be located.

        RelativeArcRadius : float
            radius of the leading edge arc normalized with respect to **Length**

        ArcAngle : float
            angle (in degree) of the leading edge arc

        RelativeTensionArc : float
            tension (normalized using **Length**) of the tangent point between
            the leading edge arc and the spinner arc

        RelativeTensionRoot : float
            tension (normalized using **Length**) of the tangent point at
            the spinner's maximum width location (blade location)

        NPtsArc : int
            number of points used to densely discretize the leading-edge arc

        NPtsSpinner : int
            number of points used to densely discretize the spinner curve

    Returns
    -------

        Arc : zone
            structured curve of the leading-edge arc

        SpinnerCurve : zone
            structured curved of the spinner curve

    '''
    ArcRadius = RelativeArcRadius * Length
    ArcCenter = Length - ArcRadius

    Arc = D.circle((0,ArcCenter,0), ArcRadius, 90., 90.-ArcAngle, N=NPtsArc)
    Arc[0] = 'LeadingEdgeArc'
    Arc_x, Arc_y = J.getxy(Arc)
    dir_y = -np.sin(np.deg2rad(ArcAngle))
    dir_x =  np.cos(np.deg2rad(ArcAngle))

    CtrlPt_1 = (Arc_x[-1]+dir_x*RelativeTensionArc*Length,
                Arc_y[-1]+dir_y*RelativeTensionArc*Length,0)
    CtrlPt_3 = (Width*0.5,0,0)
    CtrlPt_2 = (CtrlPt_3[0], RelativeTensionRoot*Length,0)

    CtrlPts_bezier = D.polyline([(Arc_x[-1],Arc_y[-1],0),
                                  CtrlPt_1,CtrlPt_2,CtrlPt_3])
    SpinnerCurve = D.bezier(CtrlPts_bezier,N=NPtsSpinner)
    SpinnerCurve[0] = 'SpinnerCurve'

    return [Arc, SpinnerCurve]


def makeHub(Profile, AxeCenter=(0,0,0), AxeDir=(1,0,0),
            NumberOfAzimutalPoints=359,
            BladeNumberForPeriodic=None, LeadingEdgeAbscissa=0.05,
            TrailingEdgeAbscissa=0.95, SmoothingParameters={'eps':0.50,
                'niter':300,'type':2}):
    '''
    This user-level function constructs a hub (Propeller's spinner) geometry
    from a user-provided profile (a curve).

    Parameters
    ----------

        Profile : zone
            curve defining the hub's generator line. It does not
            need to be coplanar. BEWARE ! indexing of curve must be oriented from
            leading edge towards trailing edge

        AxeCenter : :py:class:`list` of 3 :py:class:`float`
            coordinates of the axis center for the revolution operation [m]

        AxeDir : :py:class:`list` of 3 :py:class:`float`
            unitary vector pointing towards the direction of revolution

        NumberOfAzimutalPoints : int
            number of points discretizing the hub in the azimut direction

        BladeNumberForPeriodic : int
            If provided, then only an angular
            portion of the hub is constructed, corresponding to the blade number
            specified by this argument.

        LeadingEdgeAbscissa : float
            dimensionless abscissa indicating the point
            at which leading edge is "cut" in order to perform the diamond join

        TrailingEdgeAbscissa : float
            dimensionless abscissa indicating the point
            at which trailing edge is "cut" in order to perform the diamond join

        SmoothingParameters : dict
            literally, the parameters passed to :py:func:`Transform.PyTree.smooth`
            function

    Returns
    -------

        t : PyTree
            surface of the hub

        PeriodicProfiles : :py:class:`list`
            curves (structured zones) defining the periodic profiles boundaries
    '''

    Px, Py, Pz = J.getxyz(Profile)
    NPsi = NumberOfAzimutalPoints
    FineHub = D.axisym(Profile, AxeCenter,AxeDir, angle=360., Ntheta=360*3); FineHub[0]='FineHub'
    BigLength=1.0e6
    AxeLine = D.line((AxeCenter[0]+BigLength*AxeDir[0],AxeCenter[1]+BigLength*AxeDir[1],AxeCenter[2]+BigLength*AxeDir[2]),
        (AxeCenter[0]-BigLength*AxeDir[0],AxeCenter[1]-BigLength*AxeDir[1],AxeCenter[2]-BigLength*AxeDir[2]),2)
    s = W.gets(Profile)

    SplitLEind = np.where(s>LeadingEdgeAbscissa)[0][0]
    LEjonctCell = W.distance((Px[SplitLEind], Py[SplitLEind], Pz[SplitLEind]),
        (Px[SplitLEind+1], Py[SplitLEind+1], Pz[SplitLEind+1]))

    if TrailingEdgeAbscissa is not None:
        SplitTEind = np.where(s>TrailingEdgeAbscissa)[0][0]
        TEjonctCell = W.distance((Px[SplitTEind], Py[SplitTEind], Pz[SplitTEind]),
                                 (Px[SplitTEind+1], Py[SplitTEind+1], Pz[SplitTEind+1]))
    else:
        SplitTEind = len(Px)

    RevolutionProfile = T.subzone(Profile,(SplitLEind,1,1),(SplitTEind,1,1)); RevolutionProfile[0] = 'RevolutionProfile'
    PeriodicProfiles = []
    if BladeNumberForPeriodic is None:
        if NPsi%2==0: raise ValueError('makeHub: NumberOfAzimutalPoints shall be ODD.')
        MainBody = D.axisym(RevolutionProfile, AxeCenter,AxeDir, angle=360., Ntheta=NPsi); MainBody[0]='MainBody'
        ExtFaces = P.exteriorFacesStructured(MainBody)
        ExtFaces = I.getNodesFromType(ExtFaces,'Zone_t')

        # Close Leading Edge
        LEBound, _ = J.getNearestZone(ExtFaces, (Px[0],Py[0],Pz[0]));  LEBound[0]='LEBound'

        tT = C.newPyTree(['bMainBody',MainBody,'bLEBound',LEBound])

        LETFI = G.TFIO(LEBound) # TODO replace ?
        GSD.prepareGlue(LETFI,[LEBound])
        LETFI = T.projectDir(LETFI,FineHub,dir=AxeDir)
        T._smooth(LETFI,eps=SmoothingParameters['eps'], niter=SmoothingParameters['niter'], type=SmoothingParameters['type'], fixedConstraints=[LEBound])
        # T._projectDir(LETFI,FineHub,dir=AxeDir)
        T._projectOrthoSmooth(LETFI,FineHub, niter=3)
        GSD.applyGlue(LETFI,[LEBound])

        LETFIzones = I.getNodesFromType(LETFI,'Zone_t')
        LESingle, LESingleIndex = J.getNearestZone(LETFIzones, (Px[0],Py[0],Pz[0]))
        LEjoinIndices = [i for i in range(len(LETFIzones)) if i != LESingleIndex]
        LEjoin = LETFIzones[LEjoinIndices[0]]
        for i in LEjoinIndices[1:]:
            LEjoin = T.join(LEjoin,LETFIzones[i])
        MainBody[0] = 'hub'
        HubZones = [MainBody, LESingle, LEjoin]

        if TrailingEdgeAbscissa is not None:
            # Close Trailing Edge
            TEBound, _ = J.getNearestZone(ExtFaces, (Px[-1],Py[-1],Pz[-1])); TEBound[0]='TEBound'
            TETFI = G.TFIO(TEBound)
            TETFI = T.projectDir(TETFI,FineHub,dir=(AxeDir[0],-AxeDir[1],AxeDir[2]))
            SmoothingParameters['fixedConstraints'] = [TEBound]
            T._smooth(TETFI, **SmoothingParameters)
            GSD.prepareGlue(TETFI,[TEBound])
            # T._projectDir(TETFI,FineHub,dir=(AxeDir[0],-AxeDir[1],AxeDir[2]))
            T._projectOrthoSmooth(TETFI,FineHub,niter=3)
            GSD.applyGlue(TETFI,[TEBound])

            TETFIzones = I.getNodesFromType(TETFI,'Zone_t')
            TESingle, TESingleIndex = J.getNearestZone(TETFIzones, (Px[-1],Py[-1],Pz[-1]))
            TEjoinIndices = [i for i in range(len(TETFIzones)) if i != TESingleIndex]
            TEjoin = TETFIzones[TEjoinIndices[0]]
            for i in TEjoinIndices[1:]:
                TEjoin = T.join(TEjoin,TETFIzones[i])
            TEjoin[0] = 'hub'
            HubZones += [TESingle, TEjoin]

        # Build PyTree
        t = C.newPyTree(['Hub',HubZones])


    else:
        BladeNumberForPeriodic = float(BladeNumberForPeriodic)
        RevolutionAngle = 360./BladeNumberForPeriodic
        T._rotate(RevolutionProfile,AxeCenter,AxeDir,-0.5*RevolutionAngle)

        MainBody = D.axisym(RevolutionProfile, AxeCenter,AxeDir, angle=RevolutionAngle, Ntheta=NPsi);
        MainBody[0]='MainBody'


        # Close Leading Edge
        LEarc = T.subzone(MainBody,(1,1,1),(1,NPsi,1)); LEarc[0]='LEarc'
        LEarcX, LEarcY, LEarcZ = J.getxyz(LEarc)
        LEptInAxe = T.projectOrtho(D.point((LEarcX[0], LEarcY[0], LEarcZ[0])),AxeLine)
        LEptInAxeList = J.getxyz(LEptInAxe)


        Conn1 = D.line((LEarcX[0], LEarcY[0], LEarcZ[0]),
                       LEptInAxeList, 200)
        Conn1 = T.projectDir(Conn1,FineHub,dir=[-AxeDir[0],AxeDir[1],AxeDir[2]])
        Conn1X,Conn1Y,Conn1Z = J.getxyz(Conn1)
        Conn1X[0],Conn1Y[0],Conn1Z[0]=LEarcX[0], LEarcY[0], LEarcZ[0]
        Conn2 = D.line((LEarcX[-1], LEarcY[-1], LEarcZ[-1]),
                       LEptInAxeList, 200)
        Conn2 = T.projectDir(Conn2,FineHub,dir=[-AxeDir[0],AxeDir[1],AxeDir[2]])
        Conn2X,Conn2Y,Conn2Z = J.getxyz(Conn2)
        Conn2X[0],Conn2Y[0],Conn2Z[0]=LEarcX[-1], LEarcY[-1], LEarcZ[-1]

        # Re-discretize connection curves
        ApproxNconn = int(0.5*(SplitLEind+NPsi))
        NPsiNew, Nconn1, Nconn2 = GSD.getSuitableSetOfPointsForTFITri(NPsi,ApproxNconn,ApproxNconn,choosePriority=['N1,N2=N3'], QtySearch=4, tellMeWhatYouDo=False)
        if NPsiNew != NPsi:
            raise ValueError('Could not find appropriate TFITri values for NPsi=%d. Increase QtySearch or change NPsi.'%NPsi)
        Conn1 = W.discretize(Conn1,Nconn1,
            dict(kind='tanhOneSide',FirstCellHeight=LEjonctCell))
        Conn2 = W.discretize(Conn2,Nconn2,
            dict(kind='tanhOneSide',FirstCellHeight=LEjonctCell))
        LETFI = G.TFITri(LEarc,Conn1,Conn2)
        LETFI = I.getNodesFromType(LETFI,'Zone_t')
        Conn1LE = Conn1; Conn1LE[0] = 'Conn1LE'
        Conn2LE = Conn2; Conn2LE[0] = 'Conn2LE'

        if TrailingEdgeAbscissa is not None:
            # Close Trailing Edge
            MainBodyX = J.getx(MainBody)
            TEarc = T.subzone(MainBody,(len(MainBodyX),1,1),(len(MainBodyX),NPsi,1)); TEarc[0]='TEarc'
            TEarcX, TEarcY, TEarcZ = J.getxyz(TEarc)
            TEptInAxe = T.projectOrtho(D.point((TEarcX[0], TEarcY[0], TEarcZ[0])),AxeLine)
            TEptInAxeList = J.getxyz(TEptInAxe)

            Conn1 = D.line((TEarcX[0], TEarcY[0], TEarcZ[0]),
                           TEptInAxeList, 200)
            Conn1 = T.projectDir(Conn1,FineHub,dir=AxeDir)
            Conn1X,Conn1Y,Conn1Z = J.getxyz(Conn1)
            Conn1X[0],Conn1Y[0],Conn1Z[0]=TEarcX[0], TEarcY[0], TEarcZ[0]
            Conn2 = D.line((TEarcX[-1], TEarcY[-1], TEarcZ[-1]),
                           TEptInAxeList, 200)
            Conn2 = T.projectDir(Conn2,FineHub,dir=AxeDir)
            Conn2X,Conn2Y,Conn2Z = J.getxyz(Conn2)
            Conn2X[0],Conn2Y[0],Conn2Z[0]=TEarcX[-1], TEarcY[-1], TEarcZ[-1]
            # Re-discretize connection curves
            ApproxNconn = int(0.5*(len(Px)-SplitTEind+NPsi))
            NPsiNew, Nconn1, Nconn2 = GSD.getSuitableSetOfPointsForTFITri(NPsi,ApproxNconn,ApproxNconn,choosePriority=['N1,N2=N3'], QtySearch=4, tellMeWhatYouDo=False)
            if NPsiNew != NPsi:
                raise ValueError('Could not find appropriate TFITri values for NPsi=%d. Increase QtySearch or change NPsi.'%NPsi)
            Conn1 = W.discretize(Conn1,Nconn1,
                dict(kind='tanhOneSide',FirstCellHeight=TEjonctCell))
            Conn2 = W.discretize(Conn2,Nconn2,
                dict(kind='tanhOneSide',FirstCellHeight=TEjonctCell))
            TETFI = G.TFITri(TEarc,Conn1,Conn2)
            TETFI = I.getNodesFromType(TETFI,'Zone_t')
            tTemp = C.newPyTree(['Base',MainBody,'TETFI',TETFI,'LETFI',LETFI])

            Conn1TE = Conn1; Conn1TE[0] = 'Conn1TE'
            Conn2TE = Conn2; Conn2TE[0] = 'Conn2TE'

            GSD.prepareGlue(LETFI,[Conn1LE, Conn2LE, LEarc])
            # T._projectDir(LETFI,FineHub,dir=AxeDir)
            T._projectOrthoSmooth(LETFI,FineHub,niter=3)
            GSD.applyGlue(LETFI,[Conn1LE, Conn2LE, LEarc])

            GSD.prepareGlue(TETFI,[Conn1TE, Conn2TE, TEarc])
            # T._projectDir(TETFI,FineHub,dir=(AxeDir[0],-AxeDir[1],AxeDir[2]))
            T._projectOrthoSmooth(TETFI,FineHub,niter=3)
            GSD.applyGlue(TETFI,[Conn1TE, Conn2TE, TEarc])

        # Get the profiles
        FirstProfile=GSD.getBoundary(MainBody,'jmin')
        FirstProfileZones = [Conn1LE,FirstProfile]
        if TrailingEdgeAbscissa is not None: FirstProfileZones += [Conn1TE]
        I._rmNodesByType(FirstProfileZones,'FlowSolution_t')
        FirstProfile = T.join(FirstProfileZones)
        FirstProfile[0] = 'FirstProfile'
        SecondProfile=GSD.getBoundary(MainBody,'jmax')
        SecondProfileZones = [Conn2LE,SecondProfile]
        if TrailingEdgeAbscissa is not None:
            SecondProfileZones += [Conn2TE]
        I._rmNodesByType(SecondProfileZones,'FlowSolution_t')
        SecondProfile = T.join(SecondProfileZones)
        SecondProfile[0] = 'SecondProfile'
        PeriodicProfiles += [FirstProfile,SecondProfile]

        # REDUCE THE NUMBER OF ZONES BY JOINING
        # Join Leading Edge Elements
        LESingle, LESingleIndex = J.getNearestZone(LETFI, (Px[0],Py[0],Pz[0]))
        LEjoinIndices = [i for i in range(len(LETFI)) if i != LESingleIndex]
        LEjoin = LETFI[LEjoinIndices[0]]
        for i in LEjoinIndices[1:]:
            LEjoin = T.join(LEjoin,LETFI[i])
        I._rmNodesByType([MainBody,LEjoin],'FlowSolution_t')
        # Join result with Main body
        MainBody = T.join(MainBody,LEjoin)

        if TrailingEdgeAbscissa is not None:
            # Join Trailing Edge Elements
            TESingle, TESingleIndex = J.getNearestZone(TETFI, (Px[-1],Py[-1],Pz[-1]))
            TEjoinIndices = [i for i in range(len(TETFI)) if i != TESingleIndex]
            TEjoin = TETFI[TEjoinIndices[0]]
            for i in TEjoinIndices[1:]:
                TEjoin = T.join(TEjoin,TETFI[i])
            # Join result with Main body
            MainBody = T.join(MainBody,TEjoin)

        MainBody[0]='hub'
        FinalZones = [LESingle,MainBody]
        ConstraintZones = PeriodicProfiles
        if TrailingEdgeAbscissa is not None: FinalZones += [TESingle]
        else:
            ConstraintZones += [GSD.getBoundary(MainBody,window='imin',layer=0)]
        t = C.newPyTree(['Hub',FinalZones])

        SmoothingParameters['fixedConstraints'] = ConstraintZones
        T._smooth(t, **SmoothingParameters)
        T._projectOrtho(t,FineHub)

        # Determine if reordering is necessary in order to
        # guarantee an outwards-pointing normal for each zone
        G._getNormalMap(t)
        # Compute Revolution profile normal direction (outwards)
        rpX, rpY, rpZ = J.getxyz(RevolutionProfile)
        sizeRevProf = len(rpX)
        MidPt = (rpX[int(sizeRevProf/2)], rpY[int(sizeRevProf/2)], rpZ[int(sizeRevProf/2)])
        ptInAxe = T.projectOrtho(D.point(MidPt),AxeLine)
        ptAxe = J.getxyz(ptInAxe)
        OutwardsDir = (MidPt[0]-ptAxe[0], MidPt[1]-ptAxe[1], MidPt[2]-ptAxe[2])

        for z in I.getNodesFromType(t,'Zone_t'):
            sx=C.getMeanValue(z,'centers:sx')
            sy=C.getMeanValue(z,'centers:sy')
            sz=C.getMeanValue(z,'centers:sz')
            ndotProp = sx*OutwardsDir[0]+sy*OutwardsDir[1]+sz*OutwardsDir[2]
            mustReorder = True if ndotProp < 0 else False

            if mustReorder: T._reorder(z,(-1,2,3))

    return t, PeriodicProfiles


def extrudePeriodicProfiles(PeriodicProfiles,
        Distributions, Constraints=[], AxeDir=(1,0,0), RotationCenter=(0,0,0),
        NBlades=4,
        extrudeOptions={}, AxisProjectionConstraint=False):
    '''
    This function is used to peform the extrusion of the periodic profiles,
    in order to guarantee that there is exact axi-symmetric periodic matching.

    Parameters
    ----------

    PeriodicProfiles : :py:class:`list` of zone
            the periodic profiles boundaries to extrude, as obtained from
            :py:func:`makeHub` function.

        Distributions : :py:class:`list` of zone
            the set of distributions to apply during the extrusion of the profile

            .. note:: this is the same input attribute as in
                :py:func:`MOLA.GenerativeVolumeDesign.extrude` function

        Constraints : :py:class:`list` of :py:class:`dict`
            the set of constraints to apply
            during the extrusion of the profile

            .. note:: this is the same input attribute as in
                :py:func:`MOLA.GenerativeVolumeDesign.extrude` function


        AxeDir : :py:class:`list` of 3 :py:class:`float`
            indicates the rotation axis direction

        NBlades : int
            number of blades

        extrudeOptions : dict
            literally, the extrusion options to pass
            to the function :py:func:`MOLA.GenerativeVolumeDesign.extrude`

        AxisProjectionConstraint : bool
            if :py:obj:`True`, force the extrema of the
            profile boundaries to be projected onto the rotation axis.

    Returns
    -------

        FirstPeriodicSurf : zone
            first periodic surface

        SecondPeriodicSurf : zone
            second periodic surface
    '''

    # ~~~~~~~ PERFORM EXTRUSION OF THE FIRST PROFILE ~~~~~~~ #
    FirstProfile, SecondProfile = I.getZones(PeriodicProfiles)[:2]

    # Prepare imposed normals Constraints
    FPx, FPy, FPz = J.getxyz(FirstProfile)

    """
    # This approach is bugged:
    # Proj1 = FPx[0]*AxeDir[0] + FPy[0]*AxeDir[1] + FPz[0]*AxeDir[2]
    # Proj2 = FPx[-1]*AxeDir[0] + FPy[-1]*AxeDir[1] + FPz[-1]*AxeDir[2]
    # LEpos = np.argmin([Proj1,Proj2])
    """
    Point1 = D.point((FPx[0],FPy[0],FPz[0]));
    Point2 = D.point((FPx[-1],FPy[-1],FPz[-1]))
    Extrema = [Point1, Point2]
    LeadingEdge = Point1;LeadingEdge[0]='LeadingEdge'
    TrailingEdge = Point2;TrailingEdge[0]='TrailingEdge'
    # TrailingEdgeProjection = D.point((AxeDir[0]*FPx[-1],
    #                                   AxeDir[1]*FPy[-1],
    #                                   AxeDir[2]*FPz[-1]))
    LongAxis = D.line((-1e6*AxeDir[0],-1e6*AxeDir[1],-1e6*AxeDir[2]),
                      (+1e6*AxeDir[0],+1e6*AxeDir[1],+1e6*AxeDir[2]),2  )
    LongAxis[0] ='LongAxis'
    TrailingEdgeProjection = T.projectOrtho(TrailingEdge,LongAxis)
    TrailingEdgeProjection[0]='TrailingEdgeProjection'

    TEx,   TEy,  TEz = J.getxyz(TrailingEdge)
    TEPx, TEPy, TEPz = J.getxyz(TrailingEdgeProjection)

    sx = TEx[0]-TEPx[0]
    sy = TEy[0]-TEPy[0]
    sz = TEz[0]-TEPz[0]
    Distance = (sx**2 + sy**2 + sz**2)**0.5

    tol = 1.e-6

    if Distance > tol:
        sx /= Distance
        sy /= Distance
        sz /= Distance
    else:
        sx = AxeDir[0]
        sy = AxeDir[1]
        sz = AxeDir[2]

    C._initVars( LeadingEdge,'sx', -AxeDir[0])
    C._initVars( LeadingEdge,'sy', -AxeDir[1])
    C._initVars( LeadingEdge,'sz', -AxeDir[2])
    C._initVars(TrailingEdge,'sx',  sx)
    C._initVars(TrailingEdge,'sy',  sy)
    C._initVars(TrailingEdge,'sz',  sz)

    Constraints += [dict(kind='Projected',curve=FirstProfile,
                         ProjectionMode='CylinderRadial',
                         ProjectionCenter=RotationCenter,
                         ProjectionAxis=AxeDir),
                    dict(kind='Imposed',curve=LeadingEdge),
                    dict(kind='Imposed',curve=TrailingEdge)]

    if AxisProjectionConstraint:
        FirstProfileAux = I.copyTree(FirstProfile)
        FirstProfileAux = W.extrapolate(FirstProfileAux,0.01)
        FirstProfileAux = W.extrapolate(FirstProfileAux,0.01,opposedExtremum=True)
        FirstProfileAux[0] = 'FirstProfileAux'
        AxisSym1=D.axisym(FirstProfileAux,J.getxyz(LeadingEdge),AxeDir,0.1,5)
        AxisSym1[0]='AxisSym1'
        AxisSym2=D.axisym(FirstProfileAux,J.getxyz(LeadingEdge),AxeDir,-0.1,5)
        AxisSym2[0]='AxisSym2'
        AxisSym = T.join(AxisSym1,AxisSym2)
        a1 = C.convertArray2Hexa(AxisSym1)
        a2 = C.convertArray2Hexa(AxisSym2)
        a = T.join(a1,a2)
        G._close(a)
        a = T.reorder(a,(-1,))
        G._getNormalMap(a)
        C.center2Node__(a,'centers:sx',cellNType=0)
        C.center2Node__(a,'centers:sy',cellNType=0)
        C.center2Node__(a,'centers:sz',cellNType=0)
        I._rmNodesByName(a,'FlowSolution#Centers')
        C._normalize(a, ['sx','sy','sz'])
        T._smoothField(a, 0.9, 100, 0, ['sx','sy','sz'])
        C._normalize(a, ['sx','sy','sz'])

        '''
            # TODO old "dualing" method to be fully removed:
            C._normalize(a, ['centers:sx','centers:sy','centers:sz'])
            C._initVars(a,'centers:sxP={centers:sx}')
            C._initVars(a,'centers:syP={centers:sy}')
            C._initVars(a,'centers:szP={centers:sz}')
            C.center2Node__(a,'centers:sxP',cellNType=0)
            C.center2Node__(a,'centers:syP',cellNType=0)
            C.center2Node__(a,'centers:szP',cellNType=0)
            for i in range(1000):
                C.node2Center__(a,'nodes:sxP')
                C.node2Center__(a,'nodes:syP')
                C.node2Center__(a,'nodes:szP')
                C.center2Node__(a,'centers:sxP',cellNType=0)
                C.center2Node__(a,'centers:syP',cellNType=0)
                C.center2Node__(a,'centers:szP',cellNType=0)
                C._initVars(a,'nodes:sx={nodes:sx}+100.*{nodes:sxP}')
                C._initVars(a,'nodes:sy={nodes:sy}+100.*{nodes:syP}')
                C._initVars(a,'nodes:sz={nodes:sz}+100.*{nodes:szP}')
                C._normalize(a,['nodes:sx','nodes:sy','nodes:sz'])
                C._initVars(a,'nodes:sxP={nodes:sx}')
                C._initVars(a,'nodes:syP={nodes:sy}')
                C._initVars(a,'nodes:szP={nodes:sz}')
            C._initVars(a,'centers:sx={centers:sxP}')
            C._initVars(a,'centers:sy={centers:syP}')
            C._initVars(a,'centers:sz={centers:szP}')
        '''

        FirstProfileAux = P.extractMesh(a,FirstProfileAux)
        FirstProfileAux = T.subzone(FirstProfileAux,(2,1,1),(C.getNPts(FirstProfileAux)-1,1,1))
        C._normalize(FirstProfileAux, ['sx','sy','sz'])
        AuxConstraints =  [dict(kind='Imposed',curve=FirstProfileAux)] + Constraints

        ProjectionExtrusionDistance = np.array([D.getLength(d) for d in Distributions]).max()

        # Main
        ExtrusionDistr = D.line((0,0,0),(ProjectionExtrusionDistance*1.5,0,0),2)
        J._invokeFields(ExtrusionDistr,['normalfactor','growthfactor','normaliters','growthiters','expansionfactor',])
        ProjectionSurfTree = GVD.extrude(FirstProfileAux,[ExtrusionDistr],AuxConstraints,**extrudeOptions)
        ProjectionSurfAux = I.getZones(I.getNodeFromName1(ProjectionSurfTree,'ExtrudedVolume'))[0]

        # Lower
        ExtrusionDistr = D.line((0,0,0),(ProjectionExtrusionDistance*1.5,0,0),2)
        J._invokeFields(ExtrusionDistr,['normalfactor','growthfactor','normaliters','growthiters','expansionfactor',])
        ProjectionSurfTree = GVD.extrude(FirstProfileAux,[ExtrusionDistr],AuxConstraints,**extrudeOptions)
        ProjectionSurfAux = I.getZones(I.getNodeFromName1(ProjectionSurfTree,'ExtrudedVolume'))[0]

        ProjectionSurfAux[0] = 'ProjectionSurfAux'
        Constraints += [dict(kind='Projected',curve=FirstProfile, surface=ProjectionSurfAux)]
        C.convertPyTree2File(ProjectionSurfAux,'ProjectionSurfAux.cgns')

    # Make extrusion
    PeriodicSurf = GVD.extrude(FirstProfile,Distributions,Constraints,**extrudeOptions)

    ExtrudeLayer = I.getNodeFromName3(PeriodicSurf,'ExtrudeLayer')

    FirstPeriodicSurf = I.getNodeFromName2(PeriodicSurf,'ExtrudedVolume')[2][0]
    FirstPeriodicSurf[0] = 'FirstPeriodicSurf'
    RevolutionAngle = -360./float(NBlades)
    SecondPeriodicSurf = T.rotate(FirstPeriodicSurf,(0,0,0),AxeDir,RevolutionAngle)
    SecondPeriodicSurf[0] = 'SecondPeriodicSurf'


    return FirstPeriodicSurf, SecondPeriodicSurf

def makeSimpleSpinner(Height, Width, Length, TensionLeadingEdge=0.05,
        TensionRoot=0.8, TensionTrailingEdge=0.0, NptsTop=100, NptsBottom=150,
        NptsAzimut=180):
    """
    This function is used to make a simple spinner given general geometrical
    paramaters.

    Parameters
    ----------

        Height : float
            total height of the spinner [m]

        Width : float
            total width of the spinner [m]

        Length : float
            total length of the spinner [m]

        TensionLeadingEdge : float
            dimensionless parameter controling the
            tension at the leading edge

        TensionRoot : float
            dimensionless parameter controling the
            tension at the blade's root

        TensionTrailingEdge : float
            dimensionless parameter controling the
            tension at the trailing edge

        NptsTop : int
            number of points on top

        NptsBottom : int
            number of points on bottom

        NptsAzimut : int
            number of points in azimut

    Returns
    -------

        SpinnerProfile : zone
            spinner profile

            .. hint:: can be employed by :py:func:`makeHub`

        SpinnerUnstr : zone
            unstructured spinner surface composed of QUAD
    """

    Poly1 = D.polyline([(Height,0,0),
               (Height,TensionLeadingEdge*Width,0),
               (TensionRoot*Height,Width,0),
               (0,Width,0)])
    Poly1[0] = 'Poly1'


    Poly2 = D.polyline([(0,Width,0),
               (-TensionRoot*Height,Width,0),
               (-(Length-Height),TensionTrailingEdge*Width,0),
               (-(Length-Height),0,0)])
    Poly2[0] = 'Poly2'

    Bezier1 = D.bezier(Poly1,NptsTop)
    Bezier1[0] = 'Bezier1'

    Bezier2 = D.bezier(Poly2,NptsBottom)
    Bezier2[0] = 'Bezier2'

    SpinnerProfile = T.join(Bezier1,Bezier2)
    SpinnerUnstr   = D.axisym(SpinnerProfile,(0,0,0),(1,0,0),360.,NptsAzimut)
    SpinnerUnstr = C.convertArray2Hexa(SpinnerUnstr)
    G._close(SpinnerUnstr)
    SpinnerUnstr[0] = 'spinner'
    SpinnerProfile[0] = 'profile'


    return SpinnerProfile, SpinnerUnstr


def getFrenetFromRotationAxisAndPhaseDirection(RotationAxis, PhaseDirection):
    '''
    Get the Frenet's frame from a rotation axis and a phase direction.

    Parameters
    ----------

        RotationAxis : array of 3 :py:class:`float`
            the rotation axis unitary vector

        PhaseDirection : array of 3 :py:class:`float`
            the phase direction unitary vector.

            .. warning:: It must not be aligned with **RotationAxis**

    Returns
    -------

        FrenetFrame : 3 :py:class:`tuple`  of 3 :py:class:`float` :py:class:`tuple`
            The frenet frame as follows:

            >>> (tuple(PhaseDir), tuple(Binormal), tuple(RotAxis))
    '''
    RotAxis  = np.array(list(RotationAxis),dtype=np.float)
    RotAxis /= np.sqrt(RotAxis.dot(RotAxis))
    PhaseDir = np.array(list(PhaseDirection),dtype=np.float)
    PhaseDir /= np.sqrt(PhaseDir.dot(PhaseDir))

    PhaseAligned = np.allclose(np.abs(PhaseDir),np.abs(RotAxis))
    if PhaseAligned:
        for i in range(3):
            PhaseDir[i] += 1.
            PhaseDir /= np.sqrt(PhaseDir.dot(PhaseDir))
            PhaseAligned = np.allclose(np.abs(PhaseDir),np.abs(RotAxis))
            if not PhaseAligned: break
    if PhaseAligned: raise ValueError('could not displace phase')

    # compute projected PhaseDir into RotAxis plane at (0,0,0)
    aux = np.cross(RotAxis, PhaseDir)
    aux /= np.sqrt(aux.dot(aux))
    PhaseDir = np.cross(aux, RotAxis)
    PhaseDir /= np.sqrt(PhaseDir.dot(PhaseDir))

    # get Binormal, which defines Frenet frame
    Binormal = np.cross(RotAxis, PhaseDir)
    Binormal /= np.sqrt(Binormal.dot(Binormal))

    FrenetFrame = (tuple(PhaseDir), tuple(Binormal), tuple(RotAxis))

    return FrenetFrame


def getEulerAngles4PUMA(RotationAxis, PhaseDirection=(0,1,0)):
    '''
    Given a RotationAxis and a Phase Direction, produce the Euler angles that
    can be provided to PUMA in order to position the propeller.

    Parameters
    ----------

        RotationAxis : array of 3 :py:class:`float`
            the rotation axis unitary vector

        PhaseDirection : array of 3 :py:class:`float`
            the phase direction unitary vector.

            .. warning:: It must not be aligned with **RotationAxis**


    Returns
    -------

        EulerAngles : array of 3 :py:class:`float`
            the `intrinsic XYZ <https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.as_euler.html>`_
            transformation angles [degree]
    '''

    # TODO propagate PhaseDirection up to BodyForceInputData (user-level)
    FrenetDEST = getFrenetFromRotationAxisAndPhaseDirection(RotationAxis,PhaseDirection)


    FrenetPUMA = np.array([[1.,0.,0.],  # Rotation Axis
                           [0.,1.,0.],  # Phase Dir
                           [0.,0.,1.]]) # Binormal

    FrenetDEST = np.array([list(FrenetDEST[2]),
                           list(FrenetDEST[0]),
                           list(FrenetDEST[1])])

    RotationMatrix = FrenetDEST.T.dot(npla.inv(FrenetPUMA.T))

    Rotator = ScipyRotation.from_dcm(RotationMatrix)
    EulerAngles = Rotator.as_euler('XYZ', degrees=True)

    return EulerAngles
