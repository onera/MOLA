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

def extrudeBladeSupportedOnSpinner(blade_surface, spinner_profile, rotation_center,
        rotation_axis, wall_cell_height=2e-6, root_to_transition_distance=0.1,
        root_to_transition_number_of_points=100,
        maximum_number_of_points_in_normal_direction=200, distribution_law='ratio',
        distribution_growth_rate=1.05, last_extrusion_cell_height=1e-3,
        maximum_extrusion_distance_at_spinner=4e-2,
        smoothing_start_at_layer=80,
        smoothing_normals_iterations=1,
        smoothing_normals_subiterations=[5,200,'distance'],
        smoothing_growth_iterations=2,
        smoothing_growth_subiterations=120,
        smoothing_growth_coefficient=[0,0.03,'distance'],
        smoothing_expansion_factor=0.1,
        expand_distribution_radially=False,
        ):
    '''
    Produce the volume mesh of a blade supported onto the surface of a spinner
    defined by its revolution profile.

    Parameters
    ----------

        blade_surface : PyTree, base, zone, list of zones
            the surface of the blade. See the following important note:

            .. important:: **blade_surface** must respect the following
                requirements:

                * the main blade surface defining the contour around the airfoil
                  must be composed of a unique surface zone

                * the main blade surface defining the contour around the airfoil
                  must be the zone with highest number of points (shall yield
                  more points than surfaces defining e.g. the tip surface)

                * the blade surface index ordering must be such that the
                  root section is situated at ``'jmin'`` window (and therefore
                  the tip must be situated at ``'jmax'`` window)

                * the blade surface must *completely* intersect the spinner

                * all surfaces must have a :math:`(i,j)` ordering such that
                  the normals point towards the exterior of the blade

        spinner_profile : zone
            the structured curve of the spinner profile. It must be structured
            and oriented from leading-edge towards trailing-edge.

        rotation_center : 3-:py:class:`float` :py:class:`list` or numpy array
            indicates the rotation center :math:`(x,y,z)` coordinates of the
            blade and spinner

        rotation_axis : 3-:py:class:`float` :py:class:`list` or numpy array
            indicates the rotation axis unitary direction vector

        wall_cell_height : float
            the cell height to verify in wall-adjacent cells

        root_to_transition_distance : float
            radial distance between the spinner wall and the blade location
            up to where the radial boundary-layer is defined.

        root_to_transition_number_of_points : int
            number of points being used to discretize the radial boundary-layer
            located between the spinner wall and the user-specified blade location
            (**root_to_transition_distance**).

        maximum_number_of_points_in_normal_direction : int
            indicates the maximum authorized number of points used for
            the normal extrusion of the blade, as defined by
            **maximum_number_of_points** parameter of function
            :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        distribution_law : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        distribution_growth_rate : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        last_extrusion_cell_height : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        maximum_extrusion_distance_at_spinner : float
            indicates the maximum authorized extrusion distance used for
            the normal extrusion of the blade at the spinner (root), as defined by
            **maximum_length** parameter of function
            :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        smoothing_start_at_layer : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        smoothing_normals_iterations : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        smoothing_normals_subiterations : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        smoothing_growth_iterations : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        smoothing_growth_subiterations : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        smoothing_growth_coefficient : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        smoothing_expansion_factor : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        expand_distribution_radially : bool
            not implemented yet

    Returns
    -------

        blade_extruded : base
            CGNS Base containing the volume grid of the blade

    '''

    rotation_axis = np.array(rotation_axis, dtype=np.float, order='F')
    rotation_center = np.array(rotation_center, dtype=np.float, order='F')

    inds = _getRootAndTransitionIndices(blade_surface, spinner_profile, rotation_center,
                                    rotation_axis, root_to_transition_distance)
    root_section_index, transition_section_index = inds

    intersection_bar, spinner_surface = _computeIntersectionContourBetweenBladeAndSpinner(
            blade_surface, rotation_center, rotation_axis, spinner_profile,
            maximum_extrusion_distance_at_spinner, transition_section_index)



    supported_profile = _convertIntersectionContour2Structured(blade_surface,
                            spinner_surface, root_section_index, intersection_bar)


    Distributions = _computeBladeDistributions(blade_surface, rotation_axis,
            supported_profile, transition_section_index, distribution_law,
            maximum_extrusion_distance_at_spinner, maximum_number_of_points_in_normal_direction,
            wall_cell_height, last_extrusion_cell_height,
            distribution_growth_rate, smoothing_start_at_layer,
            smoothing_normals_iterations, smoothing_normals_subiterations,
            smoothing_growth_iterations, smoothing_growth_subiterations,
            smoothing_growth_coefficient, smoothing_expansion_factor, expand_distribution_radially)


    supported_match = _extrudeBladeRootOnSpinner(blade_surface, spinner_surface,
                        root_section_index, Distributions[0], supported_profile)

    blade_root2trans_extrusion = _extrudeBladeFromTransition(blade_surface, root_section_index,
                                                                  Distributions)

    ExtrusionResult = _buildAndJoinCollarGrid(blade_surface, blade_root2trans_extrusion, transition_section_index,
            root_section_index, supported_profile, supported_match, wall_cell_height,
            root_to_transition_number_of_points, CollarLaw='interp1d_cubic')

    base, = I.getBases(C.newPyTree(['BLADE',ExtrusionResult]))

    return base

def extrudeSpinner(Spinner, periodic_curves, rotation_center, rotation_axis,
        blade_number, maximum_length, blade_distribution,
        maximum_number_of_points_in_normal_direction=500, distribution_law='ratio',
        distribution_growth_rate=1.05, last_cell_height=1.,
        smoothing_start_at_layer=80, smoothing_normals_iterations=1,
        smoothing_normals_subiterations=5, smoothing_growth_iterations=2,
        smoothing_growth_subiterations=120,smoothing_expansion_factor=0.1,
        smoothing_growth_coefficient=[0,0.03,'distance'],
        nb_of_constrained_neighbors=3):
    '''
    extrude a spinner surface verifying periodic connectivity.

    Parameters
    ----------

        Spinner : PyTree, base, zone or list of zone
            The surface of the spinner to be extruded. All surfaces must yield
            :math:`(i,j)` ordering such that the normals point toward the
            exterior.

            .. important:: the spinner surface must be composed of only an angular sector.
                If you wish to extrude a watertight 360 spinner, then you should
                use directly :py:func:`MOLA.GenerativeVolumeDesign.extrude`

        periodic_curves : PyTree, base or list of zone
            container of curves defining the periodic profiles and possibly the
            rear-end trailing-edge boundary of the spinner

        rotation_center : 3-:py:class:`float` :py:class:`list` or numpy array
            indicates the rotation center :math:`(x,y,z)` coordinates of the
            blade and spinner

        rotation_axis : 3-:py:class:`float` :py:class:`list` or numpy array
            indicates the rotation axis unitary direction vector

        blade_number : int
            number of blades being considered by the spinner.

            .. important:: it must be coherent with the provided spinner
                angular sector

        maximum_length : same as same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        blade_distribution : zone
            structured curve yielding the radial absolute distribution of the
            blade, used for the first region of extrusion of the spinner.

        maximum_number_of_points_in_normal_direction : int
            indicates the maximum authorized number of points used for
            the normal extrusion of the blade, as defined by
            **maximum_number_of_points** parameter of function
            :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        distribution_law : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        distribution_growth_rate : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        last_extrusion_cell_height : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        maximum_extrusion_distance_at_spinner : float
            indicates the maximum authorized extrusion distance used for
            the normal extrusion of the blade at the spinner (root), as defined by
            **maximum_length** parameter of function
            :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        smoothing_start_at_layer : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        smoothing_normals_iterations : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        smoothing_normals_subiterations : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        smoothing_growth_iterations : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        smoothing_growth_subiterations : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        smoothing_growth_coefficient : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        smoothing_expansion_factor : same as :py:func:`MOLA.GenerativeVolumeDesign.newExtrusionDistribution`

        nb_of_constrained_neighbors : int
            the number of contours around surfaces patches whose normals are
            constrained

    Returns
    -------

        spinner_extruded : base
            CGNS Base containing the volume grid of the spinner
    '''


    distribution_near_blade = I.copyTree(blade_distribution)
    W.addDistanceRespectToLine(distribution_near_blade,rotation_center, rotation_axis,
                                FieldNameToAdd='Span')
    x,y,z = J.getxyz(distribution_near_blade)
    NPts_distribution_blade = len(x)
    Span, = J.getVars(distribution_near_blade,['Span'])
    y[:] = z[:] = 0
    x[0] = 0
    for i in range(1,NPts_distribution_blade):
        x[i] = x[i-1] + Span[i] - Span[i-1]

    if maximum_number_of_points_in_normal_direction > NPts_distribution_blade:
        distribution_farfield = W.linelaw(P1=(x[-1],0.,0.),
                                 P2=(maximum_length-x[-1],0.,0.),
                                 N=maximum_number_of_points_in_normal_direction-NPts_distribution_blade,
                                 Distribution=dict(
                                     kind=distribution_law,
                                     growth=distribution_growth_rate,
                                     FirstCellHeight=x[-1]-x[-2],
                                     LastCellHeight=last_cell_height))
        I._rmNodesByType([distribution_near_blade, distribution_farfield],'FlowSolution_t')
        total_distribution = T.join(distribution_near_blade, distribution_farfield)
    else:
        total_distribution = distribution_near_blade

    GVD._setExtrusionSmoothingParameters(total_distribution,
                                        smoothing_start_at_layer,
                                        smoothing_normals_iterations,
                                        smoothing_normals_subiterations,
                                        smoothing_growth_iterations,
                                        smoothing_growth_subiterations,
                                        smoothing_growth_coefficient,
                                        smoothing_expansion_factor)

    if maximum_number_of_points_in_normal_direction < NPts_distribution_blade:
        total_distribution = T.subzone(total_distribution,(1,1,1),
                            (maximum_number_of_points_in_normal_direction,1,1))


    Constraints = []
    for z in I.getZones(Spinner):
        for n in range(nb_of_constrained_neighbors+1):
            for w, l in zip(['imin','imax','jmin','jmax'],[n,-n,n,-n]):
                Constraints.append(dict(kind='Projected',
                                        curve=GSD.getBoundary(z,w,l),
                                        ProjectionMode='CylinderRadial',
                                        ProjectionCenter=rotation_center,
                                        ProjectionAxis=rotation_axis))


    PeriodicCurves = I.getZones(periodic_curves)

    LeadingEdgePoint = GSD.getBoundary(PeriodicCurves[0],'imin')
    LeadingEdgePoint[0] = 'LeadingEdgePoint'
    TrailingEdgePoint = GSD.getBoundary(PeriodicCurves[0],'imax')
    TrailingEdgePoint[0] = 'TrailingEdgePoint'

    if W.distanceOfPointToLine(LeadingEdgePoint,rotation_axis,rotation_center) < 1e-8:
        sx, sy, sz = J.invokeFields(LeadingEdgePoint,['sx','sy','sz'])
        sx[:] = rotation_axis[0]
        sy[:] = rotation_axis[1]
        sz[:] = rotation_axis[2]

        Constraints.append(dict(kind='Imposed', curve=LeadingEdgePoint))

    if W.distanceOfPointToLine(TrailingEdgePoint,rotation_axis,rotation_center) < 1e-8:
        sx, sy, sz = J.invokeFields(TrailingEdgePoint,['sx','sy','sz'])
        sx[:] = -rotation_axis[0]
        sy[:] = -rotation_axis[1]
        sz[:] = -rotation_axis[2]

        Constraints.append(dict(kind='Imposed', curve=TrailingEdgePoint))
    else:
        sx, sy, sz = J.invokeFields(PeriodicCurves[2],['sx','sy','sz'])
        Constraints.append(dict(kind='Initial', curve=PeriodicCurves[2]))




    Constraints.extend([
    dict(kind='CopyAndRotate',curve=PeriodicCurves[1], pointsToCopy=PeriodicCurves[0],
         RotationCenter=rotation_center,
         RotationAxis=rotation_axis,
         RotationAngle=360./float(blade_number),),
    ])

    SpinnerExtrusionTree = GVD.extrude(Spinner,[total_distribution],Constraints,
                                       starting_message=J.WARN+'spinner'+J.ENDC,
                                       printIters=True)
    spinner_extruded = I.getZones(I.getNodeFromName3(SpinnerExtrusionTree,'ExtrudedVolume'))
    for z in I.getZones(spinner_extruded): z[0] = 'spinner'
    I._correctPyTree(spinner_extruded,level=3)

    base, = I.getBases(C.newPyTree(['SPINNER',spinner_extruded]))

    return base




def addPitchAndAdjustPositionOfBladeSurface(blade,
                                            root_window='jmin',
                                            delta_pitch_angle=0.0,
                                            pitch_center_adjust_relative2chord=0.5,
                                            pitch_axis=(0,0,-1),
                                            pitch_center=(0,0,0)):
    '''
    Adjust the position of a blade surface and apply a rotation for setting
    a relative pitch angle.

    Parameters
    ----------

        blade : PyTree, base, zone or list of zone
            must contain the blade structured surface (zone with highest number
            of points)

            .. note:: **blade** is modified

        root_window : str
            indicates the window where root is situated, must be one of:
            ``'imin'``, ``'imax'``, ``'jmin'``, ``'jmax'``

        delta_pitch_angle : float
            the angle (in degrees) to apply to the blade

        pitch_center_adjust_relative2chord : float
            chordwise relative position (at root) used for readjusting the
            location of the blade in order to align the blade along **pitch_axis**
            passing through point **pitch_center**

        pitch_axis : 3-:py:class:`float` :py:class:`list` or numpy array
            unitary vector pointing towards the rotation axis of the pitch
            command

        pitch_center : 3-:py:class:`float` :py:class:`list` or numpy array
            coordinates :math:`(x,y,z)` of the point where **pitch_axis** passes

    Returns
    -------
        None : None
            **blade** is modified
    '''

    pitch_axis = np.array(pitch_axis,dtype=np.float)
    pitch_axis /= np.sqrt(pitch_axis.dot(pitch_axis))

    blade_main_surface = J.selectZoneWithHighestNumberOfPoints(blade)
    root = GSD.getBoundary(blade_main_surface, root_window)
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
    '''
    Rotate the spinner such that the blade is located approximately at the
    middle of the spinner surface

    Parameters
    ----------

        spinner : PyTree, base, zone, list of zone
            container with spinner surfaces

            .. note:: spinner surfaces are modified

        blade : PyTree, base, zone, list of zone
            contains the blade wall surface. It must be open at root.

        RotationAxis : 3-:py:class:`float` :py:class:`list` or numpy array
            indicates the rotation axis unitary direction vector

        RotationCenter : 3-:py:class:`float` :py:class:`list` or numpy array
            indicates the rotation center :math:`(x,y,z)` coordinates of the
            blade and spinner

    Returns
    -------
        None : None
            **spinner** is modified

    '''

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
        T._projectOrthoSmooth(LETFI,FineHub, niter=SmoothingParameters['niter'])
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
            T._projectOrthoSmooth(TETFI,FineHub,niter=SmoothingParameters['niter'])
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

        ProfileFineHub = W.discretize(Profile,N=10000)
        ProfileFineHub = T.rotate(Profile,AxeCenter,AxeDir,-0.5*RevolutionAngle)
        FineHub = D.axisym(ProfileFineHub, AxeCenter,AxeDir, angle=RevolutionAngle, Ntheta=NPsi*30); FineHub[0]='FineHub'

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

        SmoothingParameters['fixedConstraints'] = [P.exteriorFaces(LESingle)]
        SmoothingParameters['niter'] = 50
        T._smooth(LESingle, **SmoothingParameters)
        T._projectOrtho(LESingle,FineHub)


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

    try:
        Rotator = ScipyRotation.from_matrix(RotationMatrix) # new scipy
    except AttributeError:
        Rotator = ScipyRotation.from_dcm(RotationMatrix) # old scipy

    EulerAngles = Rotator.as_euler('XYZ', degrees=True)

    return EulerAngles

def extractNearestSectionIndexAtRadius(blade_surface, requested_radius,
        rotation_axis, rotation_center, search_index='jmin',
        strictlyPositive=False):
    _,Ni,Nj,Nk,_ = I.getZoneDim(blade_surface)
    previous_margin = 1e10
    for j in range(Nj):
        Section = GSD.getBoundary(blade_surface,search_index,j)
        barycenter = G.barycenter(Section)
        radius = W.distanceOfPointToLine(barycenter, rotation_axis,
                                         rotation_center)
        margin = radius-requested_radius
        if margin>0: break
        previous_margin = margin

    if strictlyPositive:
        section_index = j
    else:
        if abs(margin) > abs(previous_margin):
            section_index = j
        else:
            section_index = j-1

    if section_index <= 0:
        raise ValueError('it seems that blade does not intersects the spinner sufficiently')

    return section_index

def _getRootAndTransitionIndices(blade_surface, spinner_profile, rotation_center,
                                rotation_axis, root_to_transition_distance):
    spinner_profile = I.copyRef(spinner_profile)
    W.addDistanceRespectToLine(spinner_profile, rotation_center, rotation_axis,
                               FieldNameToAdd='radius')
    approximate_root_radius = C.getMaxValue(spinner_profile,'radius')

    blade_main_surface = J.selectZoneWithHighestNumberOfPoints(blade_surface)

    root_section_index = extractNearestSectionIndexAtRadius(blade_main_surface,
                            approximate_root_radius, rotation_axis, rotation_center,
                            strictlyPositive=True)
    transition_section_index = extractNearestSectionIndexAtRadius(blade_main_surface,
                            approximate_root_radius+root_to_transition_distance,
                            rotation_axis, rotation_center,
                            strictlyPositive=False)

    _,Ni,Nj,Nk,_ = I.getZoneDim(blade_main_surface)
    if root_section_index >= transition_section_index:
        ERROR_MESSAGE = ('detected transition section index ({}) is too close '
          'to detected approximate root index ({}). Try increasing '
          '"root_to_transition_distance" value.').format(transition_section_index,
                                                         root_section_index)
        raise ValueError(ERROR_MESSAGE)
    elif transition_section_index == Nj-1:
        raise ValueError('transition section is located at tip. Try decreasing "root_to_transition_distance" value')

    return root_section_index, transition_section_index


def _computeIntersectionContourBetweenBladeAndSpinner(blade_surface,
        rotation_center, rotation_axis, spinner_profile,
        maximum_extrusion_distance_at_spinner, transition_section_index,
        geometry_npts_azimut=361):

    blade_main_surface = J.selectZoneWithHighestNumberOfPoints(blade_surface)
    _,Ni,Nj,Nk,_ = I.getZoneDim(blade_main_surface)
    trimmed_blade_root = T.subzone(blade_main_surface,(1,1,1),
                                                (Ni,transition_section_index,1))
    trimmed_blade_root_closed = GSD.buildWatertightBodyFromSurfaces([trimmed_blade_root])

    blade_barycenter = np.array(list(G.barycenter(trimmed_blade_root)))
    cut_point_1 = blade_barycenter + 3 * rotation_axis * maximum_extrusion_distance_at_spinner
    cut_point_2 = blade_barycenter - 3 * rotation_axis * maximum_extrusion_distance_at_spinner
    trimmed_spinner_profile = W.trimCurveAlongDirection(spinner_profile,
                                                        rotation_axis,
                                                        cut_point_1,
                                                        cut_point_2)

    nb_azimut_pts = geometry_npts_azimut if geometry_npts_azimut%2==0 else geometry_npts_azimut + 1
    spinner_surface = D.axisym(trimmed_spinner_profile, rotation_center,
                               rotation_axis, angle=360., Ntheta=nb_azimut_pts)
    spinner_surface_closed = GSD.buildWatertightBodyFromSurfaces([spinner_surface])
    intersection_bar = XOR.intersection(spinner_surface_closed, trimmed_blade_root_closed)
    intersection_bar = C.convertBAR2Struct(intersection_bar)
    T._projectOrtho(intersection_bar,trimmed_blade_root)

    return intersection_bar, spinner_surface


def _convertIntersectionContour2Structured(blade_surface, spinner_surface,
                                    root_section_index, intersection_contour):
    blade_main_surface = J.selectZoneWithHighestNumberOfPoints(blade_surface)
    _,Ni,Nj,Nk,_ = I.getZoneDim(blade_main_surface)
    supported_profile = GSD.getBoundary(blade_main_surface,'jmin',root_section_index-1)
    supported_profile[0] = 'supported_profile'
    x,y,z = J.getxyz(supported_profile)

    CassiopeeVersionIsGreaterThan3dot3 = int(C.__version__.split('.')[1]) > 3
    useApproximate=False
    if CassiopeeVersionIsGreaterThan3dot3:
        for i in range(Ni):
            spanwise_curve = GSD.getBoundary(blade_main_surface,'imin',i)
            spanwise_curve = C.convertArray2Tetra(spanwise_curve)
            intersection_contour = C.convertArray2Tetra(intersection_contour)
            # see #9599
            try:
                IntersectingPoint = XOR.intersection(spanwise_curve, intersection_contour,
                                                 tol=1e-6)
            except:
                print(J.WARN+'XOR.intersection failed at point i=%d\nWill use APPROXIMATE METHOD'%i+J.ENDC)
                useApproximate=True
                break
            xPt, yPt, zPt = J.getxyz(IntersectingPoint)
            x[i] = xPt[0]
            y[i] = yPt[0]
            z[i] = zPt[0]
        T._projectOrtho(supported_profile,spinner_surface)
    else:
        useApproximate=True

    if useApproximate:
        supported_profile = GSD.getBoundary(blade_main_surface,'jmin',root_section_index-1)
        supported_profile[0] = 'supported_profile'
        x,y,z = J.getxyz(supported_profile)

        Projections, Distances = [], []
        Section = GSD.getBoundary(blade_main_surface,'jmin',root_section_index)
        dx, dy, dz = J.invokeFields(supported_profile,['dx','dy','dz'])
        Sx, Sy, Sz = J.getxyz(Section)
        dx[:] = x-Sx
        dy[:] = y-Sy
        dz[:] = z-Sz

        T._projectAllDirs(supported_profile,spinner_surface,vect=['dx','dy','dz'],oriented=0)
        I._rmNodesByType(supported_profile,'FlowSolution_t')

    return supported_profile

def _computeTransitionDistanceAndCellWidth(blade_surface, supported_profile,
                                           transition_section_index):
    blade_main_surface = J.selectZoneWithHighestNumberOfPoints(blade_surface)
    x3,y3,z3=J.getxyz(supported_profile)
    transition_section = GSD.getBoundary(blade_main_surface,'jmin',
                                         layer=transition_section_index)
    x1,y1,z1=J.getxyz(transition_section)
    transition_previous_section = GSD.getBoundary(blade_main_surface,'jmin',
                                         layer=transition_section_index-1)
    x2,y2,z2=J.getxyz(transition_previous_section)
    transition_cell_width = np.sqrt((x1[0]-x2[0])**2+(y1[0]-y2[0])**2+(z1[0]-z2[0])**2)
    transition_distance = np.sqrt((x1[0]-x3[0])**2+(y1[0]-y3[0])**2+(z1[0]-z3[0])**2)

    return transition_distance, transition_cell_width

def _computeBladeDistributions(blade_surface, rotation_axis,
        supported_profile, transition_section_index, distribution_law,
        maximum_extrusion_distance_at_spinner, maximum_number_of_points_in_normal_direction,
        wall_cell_height, last_extrusion_cell_height,
        distribution_growth_rate, smoothing_start_at_layer,
        smoothing_normals_iterations, smoothing_normals_subiterations,
        smoothing_growth_iterations, smoothing_growth_subiterations,
        smoothing_growth_coefficient, smoothing_expansion_factor,
        expand_distribution_radially):

    blade_main_surface = J.selectZoneWithHighestNumberOfPoints(blade_surface)
    x3,y3,z3=J.getxyz(supported_profile)
    transition_section = GSD.getBoundary(blade_main_surface,'jmin',
                                         layer=transition_section_index)
    x1,y1,z1=J.getxyz(transition_section)
    transition_previous_section = GSD.getBoundary(blade_main_surface,'jmin',
                                         layer=transition_section_index-1)
    x2,y2,z2=J.getxyz(transition_previous_section)
    transition_cell_width = np.sqrt((x1[0]-x2[0])**2+(y1[0]-y2[0])**2+(z1[0]-z2[0])**2)
    transition_distance = np.sqrt((x1[0]-x3[0])**2+(y1[0]-y3[0])**2+(z1[0]-z3[0])**2)

    BladeExtrusionDistribution = GVD.newExtrusionDistribution(maximum_extrusion_distance_at_spinner,
         maximum_number_of_points=maximum_number_of_points_in_normal_direction,
         distribution_law=distribution_law,
         first_cell_height=wall_cell_height,
         last_cell_height=last_extrusion_cell_height,
         ratio_growth=distribution_growth_rate,
         smoothing_start_at_layer=smoothing_start_at_layer,
         smoothing_normals_iterations=smoothing_normals_iterations,
         smoothing_normals_subiterations=smoothing_normals_subiterations,
         smoothing_growth_iterations=smoothing_growth_iterations,
         smoothing_growth_subiterations=smoothing_growth_subiterations,
         smoothing_growth_coefficient=smoothing_growth_coefficient,
         smoothing_expansion_factor=smoothing_expansion_factor,
         start_point=(x3[0], y3[0], z3[0]), direction=-rotation_axis)

    Distributions = [BladeExtrusionDistribution]

    if expand_distribution_radially:
        raise ValueError('expand_distribution_radially=True to be implemented')
        if distribution_law == 'ratio':
            raise ValueError('cannot expand radially if distribution_law=="ratio", please switch to "tanh"')
        tip_section = GSD.getBoundary(blade_main_surface,'jmax')
        x4, y4, z4 = J.getxyz(tip_section)
        radius_root = W.distanceOfPointToLine((x3[0],y3[0],z3[0]), rotation_axis, rotation_center)
        radius_tip = W.distanceOfPointToLine((x4[0],y4[0],z4[0]), rotation_axis, rotation_center)
        extrusion_distance_tip = radius_tip*maximum_extrusion_distance_at_spinner/radius_root
        TipExtrusionDistribution = GVD.newExtrusionDistribution(extrusion_distance_tip,
             maximum_number_of_points=maximum_number_of_points_in_normal_direction,
             distribution_law=kind,
             first_cell_height=wall_cell_height,
             last_cell_height=last_extrusion_cell_height,
             ratio_growth=distribution_growth_rate,
             smoothing_start_at_layer=smoothing_start_at_layer,
             smoothing_normals_iterations=smoothing_normals_iterations,
             smoothing_normals_subiterations=smoothing_normals_subiterations,
             smoothing_growth_iterations=smoothing_growth_iterations,
             smoothing_growth_subiterations=smoothing_growth_subiterations,
             smoothing_growth_coefficient=smoothing_growth_coefficient,
             smoothing_expansion_factor=smoothing_expansion_factor,
             start_point=(x4[0], y4[0], z4[0]), direction=-rotation_axis)
        Distributions.append(TipExtrusionDistribution)

    return Distributions


def _extrudeBladeRootOnSpinner(blade_surface, spinner_surface, root_section_index,
                              Distribution, supported_profile):
    blade_main_surface = J.selectZoneWithHighestNumberOfPoints(blade_surface)
    Constraints = [dict(kind='Projected',ProjectionMode='ortho',
                        curve=supported_profile, surface=spinner_surface)]

    trimmed_blade_no_intersect = T.subzone(blade_main_surface,
                                          (1,root_section_index+1,1), (-1,-1,-1))

    supported_profile_bar = C.convertArray2Tetra(supported_profile)
    G._close(supported_profile_bar)

    support_match_extrusion = GVD.extrude(supported_profile_bar, [Distribution],
                                    Constraints,  printIters=True,
                                    starting_message=J.CYAN+'root support'+J.ENDC)
    supported_match = I.getNodeFromName3(support_match_extrusion,'ExtrudedVolume')
    supported_match, = I.getZones(supported_match)

    return supported_match


def _extrudeBladeFromTransition(blade_surface, root_section_index, Distributions):
    blade_main_surface = J.selectZoneWithHighestNumberOfPoints(blade_surface)
    trimmed_blade_no_intersect = T.subzone(blade_main_surface,
                                          (1,root_section_index+1,1), (-1,-1,-1))
    copy_profile = GSD.getBoundary(trimmed_blade_no_intersect,'jmin',1)
    copy_profile[0]='copy_profile'
    J._invokeFields(copy_profile,['sx','sy','sz','dH'])
    pts2copy = GSD.getBoundary(trimmed_blade_no_intersect,'jmin',2)
    J._invokeFields(pts2copy,['sx','sy','sz','dH'])
    pts2copy[0]='pts2copy'

    copy_profile2 = GSD.getBoundary(trimmed_blade_no_intersect,'jmin',0)
    copy_profile2[0]='copy_profile2'
    J._invokeFields(copy_profile2,['sx','sy','sz','dH'])
    pts2copy2 = GSD.getBoundary(trimmed_blade_no_intersect,'jmin',2)
    J._invokeFields(pts2copy2,['sx','sy','sz','dH'])
    pts2copy2[0]='pts2copy2'

    Constraints = [dict(kind='Copy', curve=copy_profile, pointsToCopy=pts2copy),
                   dict(kind='Copy', curve=copy_profile2, pointsToCopy=pts2copy2)]

    new_blade_surface_with_tip = [trimmed_blade_no_intersect]
    new_blade_surface_with_tip.extend(J.selectZonesExceptThatWithHighestNumberOfPoints(blade_surface))

    blade_root2trans_extrusion = GVD.extrude(new_blade_surface_with_tip,
                                             Distributions,
                                             Constraints, printIters=True,
                                             starting_message=J.GREEN+'blade'+J.ENDC,
                                             closeExtrusionLayer=True)

    return blade_root2trans_extrusion


def _buildAndJoinCollarGrid(blade_surface, blade_root2trans_extrusion, transition_section_index,
        root_section_index, supported_profile, supported_match, wall_cell_height,
        root_to_transition_number_of_points, CollarLaw='interp1d_cubic'):

    transition_distance, transition_cell_width = _computeTransitionDistanceAndCellWidth(blade_surface,
                                        supported_profile, transition_section_index)
    blade_main_surface = J.selectZoneWithHighestNumberOfPoints(blade_surface)
    _,Ni,Nj,Nk,_ = I.getZoneDim(blade_main_surface)

    extruded_blade_with_tip = I.getNodeFromName1(blade_root2trans_extrusion,'ExtrudedVolume')
    extruded_blade_main = J.selectZoneWithHighestNumberOfPoints(extruded_blade_with_tip)

    extruded_blade_root2trans = T.subzone(extruded_blade_main, (1,1,1),
                        (Ni,transition_section_index-root_section_index+1,-1))
    extruded_blade_trans2tip = T.subzone(extruded_blade_main,
                            (1,transition_section_index-root_section_index+1,1),
                            (-1,-1,-1))
    _,Ni2,Nj2,Nk2,_=I.getZoneDim(extruded_blade_root2trans)

    transition_sections = [supported_match]
    transition_sections.extend([T.subzone(extruded_blade_root2trans,(1,j+1,1),(Ni2,j+1,Nk2)) for j in range(Nj2)])
    transition_distribution = J.getDistributionFromHeterogeneousInput__(W.linelaw(
                                   P2=(transition_distance,0,0), N=root_to_transition_number_of_points,
                                   Distribution=dict(kind='tanhTwoSides',
                                   FirstCellHeight=wall_cell_height,
                                   LastCellHeight=transition_cell_width)))[1]
    extruded_transition = GVD.multiSections(transition_sections, transition_distribution,
                        InterpolationData={'InterpolationLaw':CollarLaw})[0]
    T._reorder(extruded_transition,(1,3,2))
    # extruded_blade = T.join(extruded_transition,extruded_blade_trans2tip) # bug #9653
    extruded_transition = T.subzone(extruded_transition,(1,1,1),(-1,-2,-1)) # use this strategy instead
    x1,y1,z1 = J.getxyz(extruded_transition)
    x2,y2,z2 = J.getxyz(extruded_blade_trans2tip)
    x = np.concatenate((x1,x2),axis=1)
    y = np.concatenate((y1,y2),axis=1)
    z = np.concatenate((z1,z2),axis=1)
    allzones = [J.createZone('blade',[x,y,z],['x','y','z'])]

    allzones.extend(J.selectZonesExceptThatWithHighestNumberOfPoints(extruded_blade_with_tip))

    return allzones

def makeBladeAndSpinnerTreeForChecking(blade_extruded, spinner_extruded,
                                        rotation_center, rotation_axis):
    '''
    make a light CGNS tree of the spinner and blade extrusion result
    for visualization and checking purposes

    Parameters
    ----------

        blade_extruded : base
            as returned by :py:func:`extrudeBladeSupportedOnSpinner`

        spinner_extruded : base
            as returned by :py:func:`extrudeSpinner`

        rotation_center : 3-:py:class:`float` :py:class:`list` or numpy array
            indicates the rotation center :math:`(x,y,z)` coordinates of the
            blade and spinner

        rotation_axis : 3-:py:class:`float` :py:class:`list` or numpy array
            indicates the rotation axis unitary direction vector

    Returns
    -------

        t : PyTree
            visualization PyTree composed of external faces and middle slices
    '''
    t = C.newPyTree(['Blade', P.exteriorFacesStructured(blade_extruded),
                     'Spinner', P.exteriorFacesStructured(spinner_extruded)])
    pt1 = np.array(list(G.barycenter(blade_extruded)),dtype=np.float)
    c = np.array(rotation_center,dtype=np.float)
    a = np.array(rotation_axis,dtype=np.float)
    pt2 = c + a*(pt1-c).dot(a)
    pt3 = c
    n = np.cross(pt1-pt2,pt3-pt2)
    n/=np.sqrt(n.dot(n))
    Pt = pt2
    PlaneCoefs = n[0],n[1],n[2],-n.dot(Pt)
    tAux = C.newPyTree(['BLADE',J.getZones(blade_extruded),
                        'SPINNER',J.getZones(spinner_extruded),])
    C._initVars(tAux,'Slice=%0.12g*{CoordinateX}+%0.12g*{CoordinateY}+%0.12g*{CoordinateZ}+%0.12g'%PlaneCoefs)
    slicezones = P.isoSurfMC(tAux, 'Slice', 0.0)
    t2 = C.newPyTree(['SLICE',slicezones])
    t = I.merge([t,t2])

    return t
