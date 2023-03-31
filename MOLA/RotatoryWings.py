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
MOLA - RotatoryWings.py

This module proposes macro functionalities for rapid creation
and assembly of CFD simple cases of Rotatory Wings (Propellers,
Helicopter Rotors, Ducted Fans, etc)

This module makes use of Cassiopee modules.

File history:
19/03/2019 - v1.0 - L. Bernardos - Creation.
'''

import MOLA

if not MOLA.__ONLY_DOC__:

    # System modules
    import sys
    import os
    from timeit import default_timer as Tok
    import numpy as np
    import numpy.linalg as npla
    norm = np.linalg.norm
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

def extrudeBladeSupportedOnSpinner(blade_surface, spinner, rotation_center,
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
        intersection_method='conformize',
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

        spinner : PyTree, base, zone or list of zone
            the structured curve of the spinner surface

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

    rotation_axis = np.array(rotation_axis, dtype=np.float64, order='F')
    rotation_center = np.array(rotation_center, dtype=np.float64, order='F')

    inds = _getRootAndTransitionIndices(blade_surface, spinner, rotation_center,
                                    rotation_axis, root_to_transition_distance)
    root_section_index, transition_section_index = inds

    intersection_bar, spinner_surface = _computeIntersectionContourBetweenBladeAndSpinner(
            blade_surface, rotation_center, rotation_axis, spinner,
            transition_section_index,method=intersection_method)

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
    for zone in I.getZones(base):
        C._addBC2Zone(zone,'blade_wall','FamilySpecified:BLADE', 'kmin')
    blade = J.selectZoneWithHighestNumberOfPoints(I.getZones(base))
    C._addBC2Zone(blade,'spinner_wall','FamilySpecified:SPINNER', 'jmin')


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

    pitch_axis = np.array(pitch_axis,dtype=np.float64)
    pitch_axis /= np.sqrt(pitch_axis.dot(pitch_axis))

    blade_main_surface = J.selectZoneWithHighestNumberOfPoints(blade)
    root = GSD.getBoundary(blade_main_surface, root_window)
    root_camber = W.buildCamber(root)
    x,y,z = J.getxyz(root_camber)
    adjust_point = [x[0] + pitch_center_adjust_relative2chord * (x[-1] - x[0]),
                    y[0] + pitch_center_adjust_relative2chord * (y[-1] - y[0]),
                    z[0] + pitch_center_adjust_relative2chord * (z[-1] - z[0])]
    adjust_point = np.array(adjust_point, dtype=np.float64)
    pitch_center = np.array(pitch_center, dtype=np.float64)

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
    curves = W.reorderAndSortCurvesSequentially(curves)

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


def makeHub(profile, blade_number, number_of_points_azimut=31,
                profile_axis=[0,1,0]):

    if number_of_points_azimut%2 == 0:
        raise ValueError(J.FAIL+'number_of_points_azimut must be odd'+J.ENDC)

    if blade_number<2:
        raise ValueError(J.FAIL+'blade_number must be greater than 1'+J.ENDC)

    center = np.array([0,0,0],dtype=np.float64)
    axis = np.array(profile_axis,dtype=np.float64)
    tol = 1e-8


    if W.distanceOfPointToLine(W.point(profile),axis,center) > tol:
        pt = W.point(profile)
        print(pt)
        print(_distance2Axis(pt))
        C.convertPyTree2File(profile,'debug.cgns')
        MSG='Leading-edge of spinner profile does not lie on rotation axis. Check debug.cgns'
        raise ValueError(J.FAIL+MSG+J.ENDC)

    REAR_CLOSED = True if W.distanceOfPointToLine(W.point(profile,-1),axis,center) < tol else False



    def makeBulb(profile, axis, center, number_of_points_azimut, reverse=False):

        azimut_central_index = int((number_of_points_azimut - 1) / 2)
        bulb_npts = azimut_central_index + 1


        support = D.axisym(profile,center,axis, 360./float(blade_number), number_of_points_azimut)

        if reverse:
            d = -1
            support = T.reorder(support,(-1,2,3))
            profile = W.reverse(profile)
        else:
            d = 1

        azimut_central_index = int((number_of_points_azimut - 1) / 2)
        mid_profile = GSD.getBoundary(support,'jmin',azimut_central_index)
        mid_profile[0] = 'mid_profile'


        bulb_profile,_ = W.splitAt(profile, bulb_npts-1)

        cut_at = [bulb_npts-1, 2*np.sqrt(2)*D.getLength(bulb_profile)]

        profile_split = W.splitAt(profile, cut_at, 'length')

        front_end_region = C.getNPts(profile_split[:2])-2

        azm_front = GSD.getBoundary(support,'imin',front_end_region)
        azm_front_0, azm_front_1 = T.splitNParts(azm_front,2)


        mid_profile_split = W.splitAt(mid_profile, [np.sqrt(2)*D.getLength(bulb_profile),
                                            front_end_region],'length')
        mid_profile_split[1] = W.discretize(mid_profile_split[1],
            N=C.getNPts(profile_split[1]),
            Distribution=dict( kind='tanhTwoSides',
            FirstCellHeight=W.segment(mid_profile_split[1]),
            LastCellHeight=W.segment(mid_profile_split[1],-1)))



        support_front = T.subzone(support,(C.getNPts(profile_split[0])-1,1,1),
                       (C.getNPts(mid_profile_split[0])+1,azimut_central_index+1,1))



        bulb_union_0 = D.line(tuple(W.point(profile_split[0],-1)),
                              tuple(W.point(mid_profile_split[0],-1)),
                              azimut_central_index+1)
        T._projectOrtho(bulb_union_0, support_front)
        bulb_union_0 = W.discretize(bulb_union_0,Distribution=dict(kind='tanhTwoSides',
            FirstCellHeight=W.segment(profile_split[0],-1),
            LastCellHeight=W.segment(mid_profile_split[0],-1)))
        T._projectOrtho(bulb_union_0, support_front)




        profile_splitB=[]
        for p in profile_split:
            profile_splitB.append(T.rotate(p,tuple(center),tuple(axis),
                                      360./float(blade_number)))

        I._correctPyTree(profile_split+profile_splitB,level=3)

        T._rotate(support_front,tuple(center),tuple(axis),
                                  180./float(blade_number))


        bulb_union_1 = D.line(tuple(W.point(mid_profile_split[0],-1)),
                              tuple(W.point(profile_splitB[0],-1)),
                              azimut_central_index+1)
        T._projectOrtho(bulb_union_1, support_front)
        bulb_union_1 = W.discretize(bulb_union_1,Distribution=dict(kind='tanhTwoSides',
            FirstCellHeight=W.segment(mid_profile_split[0],-1),
            LastCellHeight=W.segment(profile_splitB[0],-1)))
        T._projectOrtho(bulb_union_1, support_front)

        front_bulb = G.TFI([W.reverse(profile_split[0]), bulb_union_1,
                            bulb_union_0, profile_splitB[0]])
        front_bulb[0] = 'bulb'
        if reverse: T._reorder(front_bulb,(-1,2,3))


        front_0 = G.TFI([bulb_union_0, azm_front_0,profile_split[1], mid_profile_split[1]])
        front_1 = G.TFI([bulb_union_1, azm_front_1,mid_profile_split[1],profile_splitB[1]])
        front_join = T.join(front_0, front_1)
        front_join[0] = 'join'
        T._reorder(front_join,(2,1,3))
        if reverse: T._reorder(front_join,(-1,2,3))


        support_front = T.subzone(support,(1,1,1),
                       (C.getNPts(profile_split[:2]),2*azimut_central_index+1,1))

        surfs2proj = [front_bulb,front_join]
        fixed = P.exteriorFaces(surfs2proj)+[mid_profile_split[1]]
        GSD.prepareGlue(surfs2proj, fixed)

        # mirror point used for ray-projectio
        OP = W.point(mid_profile_split[0],-1)
        CM = (OP-center).dot(axis) * axis
        OM = center + CM
        MP = OP - OM
        MPp = - MP
        OPp = OM + MPp
        T._projectRay(surfs2proj, support_front, OPp)
        GSD.applyGlue(surfs2proj, fixed)
        NPts_close = C.getNPts(profile_split[:2])
        hub = T.subzone(support,(NPts_close-1,1,1), (-1,-1,-1))
        hub[0] = 'hub'

        return front_bulb, front_join, hub

    front_bulb, front_join, hub = makeBulb(profile, axis, center, number_of_points_azimut)
    front_bulb[0] = 'front.'+front_bulb[0]
    hub = T.join(front_join,hub)
    zones = [front_bulb, hub]
    if REAR_CLOSED:
        rear_bulb, rear_join, _ = makeBulb(profile, axis, center, number_of_points_azimut, True)
        rear_bulb[0] = 'rear.'+rear_bulb[0]
        Ni = I.getZoneDim(rear_join)[1] + int((number_of_points_azimut-1)/2)
        hub = T.subzone(hub,(1,1,1),(-Ni,-1,-1))
        hub = T.join(hub,rear_join)
        zones = [front_bulb, hub, rear_bulb]
    hub[0] = 'hub'

    return zones


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
    RotAxis  = np.array(list(RotationAxis),dtype=np.float64)
    RotAxis /= np.sqrt(RotAxis.dot(RotAxis))
    PhaseDir = np.array(list(PhaseDirection),dtype=np.float64)
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

def _getRootAndTransitionIndices(blade_surface, spinner, rotation_center,
                                rotation_axis, root_to_transition_distance):
    spinner_profile = I.copyRef(spinner)
    W.addDistanceRespectToLine(spinner, rotation_center, rotation_axis,
                               FieldNameToAdd='radius')
    approximate_root_radius = C.getMaxValue(spinner,'radius')

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
        rotation_center, rotation_axis, spinner,
        transition_section_index, geometry_npts_azimut=361,
        method='conformize'):

    blade_main_surface = J.selectZoneWithHighestNumberOfPoints( blade_surface )
    _,Ni,Nj,Nk,_ = I.getZoneDim(blade_main_surface)
    trimmed_blade_root = T.subzone(blade_main_surface,(1,1,1),
                                                (Ni,transition_section_index,1))
    spinner_surface = _splitSpinnerHgrid(spinner, blade_surface, rotation_axis, rotation_center,
                                        spinner_axial_indexing='i')[1]

    if method=='intersection':
        trimmed_blade_root_closed = GSD.buildWatertightBodyFromSurfaces([trimmed_blade_root])

        ref_root = T.subzone(blade_main_surface,(1,1,1),(Ni,2,1))
        bbox = np.array(G.bbox(ref_root))
        bbox = np.reshape(bbox,(2,3))
        bbox_diag_vector = np.diff(bbox,axis=0).flatten()
        bbox_diag = np.sqrt(bbox_diag_vector.dot(bbox_diag_vector))

        contour = P.exteriorFacesStructured(spinner_surface)
        spinner_closed_contours = [spinner_surface]
        for c in contour:
            c_proj = I.copyTree(c)
            W.projectOnAxis(c_proj, rotation_axis, rotation_center)
            spinner_closed_contours += [G.stack([c,c_proj])]

        spinner_surface_closed = GSD.buildWatertightBodyFromSurfaces(spinner_closed_contours)
        try:
            # WARNING https://elsa.onera.fr/issues/10482
            intersection_bar = XOR.intersection(spinner_surface_closed, trimmed_blade_root_closed)
        except:
            C.convertPyTree2File([spinner_surface_closed, trimmed_blade_root_closed],'debug.cgns')
            MSG = ('could not compute blade-spinner intersection.\n '
                   'BEWARE of https://elsa.onera.fr/issues/10482\n '
                   'CHECK debug.cgns (elements MUST BE intersecting)\n '
                   'Employed method was "%s". Please try "%s" ')%(method,'conformize')
            raise ValueError(J.FAIL+MSG+J.ENDC)

    elif method=='conformize':
        try:
            intersection_bar = GSD.surfacesIntersection(trimmed_blade_root, spinner_surface)
        except:
            C.convertPyTree2File([trimmed_blade_root, spinner_surface],'debug.cgns')
            MSG = ('could not compute blade-spinner intersection.\n '
                   'BEWARE of https://elsa.onera.fr/issues/10482\n '
                   'CHECK debug.cgns (elements MUST BE intersecting)\n '
                   'Employed method was "%s". Please try "%s" ')%(method,'intersection')
            raise ValueError(J.FAIL+MSG+J.ENDC)

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
        intersection_contour = I.getZones(C.convertArray2Tetra(intersection_contour))[0]
        for i in range(Ni):
            spanwise_curve = GSD.getBoundary(blade_main_surface,'imin',i)
            spanwise_curve = I.getZones(C.convertArray2Tetra(spanwise_curve))[0]
            try:
                # see #9599
                IntersectingPoint = XOR.intersection(spanwise_curve,
                                                intersection_contour, tol=1e-10)
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
    pt1 = np.array(list(G.barycenter(blade_extruded)),dtype=np.float64)
    c = np.array(rotation_center,dtype=np.float64)
    a = np.array(rotation_axis,dtype=np.float64)
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


def buildMatchMesh(spinner, blade, blade_number, rotation_axis=[-1,0,0],
                   rotation_center=[0,0,0], H_grid_interior_points=61,
                   relax_relative_length=0.5, distance=10., number_of_points=200,
                   farfield_cell_height=1., tip_axial_scaling_at_farfield=0.5,
                   normal_tension=0.05, DIRECTORY_CHECKME='CHECK_ME'):

    # NOTE implement so that blade can be a surface (2D) or a volume (3D)
    spinner_front, spinner_middle, spinner_rear = _splitSpinnerHgrid(spinner,
                                          blade, rotation_axis, rotation_center)
    external_surfaces, npts_azimut, \
    central_first_cell, central_last_cell = _buildExternalSurfacesHgrid(
                        blade, spinner_middle, rotation_axis, rotation_center)

    try: bulb_front, = [z for z in I.getZones(spinner) if z[0]=='front.bulb']
    except: bulb_front = None
    try: bulb_rear, = [z for z in I.getZones(spinner) if z[0]=='rear.bulb']
    except: bulb_rear = None
    front_surface, = [ z for z in external_surfaces if z[0] == 'front.surf' ]
    rear_surface, = [ z for z in external_surfaces if z[0] == 'rear.surf' ]

    wires_front, surfs_front, grids_front = _buildHubWallAdjacentSector(
                            front_surface, spinner_front,
                            bulb_front, blade_number, rotation_center,
                            rotation_axis, central_first_cell, central_last_cell,
                            'front')

    C.convertPyTree2File(wires_front+ surfs_front,
                         os.path.join(DIRECTORY_CHECKME,'6_front_near_topo.cgns'))

    if bulb_rear:
        wires_rear, surfs_rear, grids_rear = _buildHubWallAdjacentSector(
                                rear_surface, spinner_rear,
                                bulb_rear, blade_number, rotation_center,
                                rotation_axis, central_first_cell,
                                central_last_cell,'rear')
    else:
        wires_rear, surfs_rear, grids_rear = _buildHubWallAdjacentSectorWithoutBulb(
                                rear_surface, spinner_rear, blade_number, rotation_center,
                                rotation_axis, central_last_cell)
    C.convertPyTree2File(wires_rear+ surfs_rear,
                         os.path.join(DIRECTORY_CHECKME,'7_rear_near_topo.cgns'))


    Hgrids = _buildHgridAroundBlade(external_surfaces, blade, rotation_center,
                   rotation_axis, H_grid_interior_points, relax_relative_length)

    profile = _extractWallAdjacentSectorFullProfile(wires_front, wires_rear,
                                                        external_surfaces)


    C.convertPyTree2File(P.exteriorFacesStructured(Hgrids),
                         os.path.join(DIRECTORY_CHECKME,'8_Hgrid_faces.cgns'))


    sector_bnds = _gatherSectorBoundaries(Hgrids, blade, surfs_rear, surfs_front)

    fargrids, farfaces = _buildFarfieldSector(blade, sector_bnds, profile,
                         blade_number, npts_azimut, H_grid_interior_points,
                         rotation_center, rotation_axis,
                         distance=distance, number_of_points=number_of_points,
                         farfield_cell_height=farfield_cell_height,
                         tip_axial_scaling_at_farfield=tip_axial_scaling_at_farfield,
                         normal_tension=normal_tension)

    C.convertPyTree2File(farfaces,
                         os.path.join(DIRECTORY_CHECKME,'9_farfaces.cgns'))

    t = C.newPyTree(['Base',grids_front+Hgrids+grids_rear+fargrids+I.getZones(blade)])
    base, = I.getBases(t)
    J.set(base,'.MeshingParameters',blade_number=blade_number,
               RightHandRuleRotation=True, # TODO CAVEAT
               rotation_axis=rotation_axis, rotation_center=rotation_center)

    C.convertPyTree2File(P.exteriorFacesStructured(t),
                     os.path.join(DIRECTORY_CHECKME,'10_final_grid_faces.cgns'))


    return t


def _splitSpinnerHgrid(spinner, blade, rotation_axis, rotation_center,
                       spinner_axial_indexing='i'):
    blade_main_surface = J.selectZoneWithHighestNumberOfPoints( blade )
    spinner_main_surface = J.selectZoneWithHighestNumberOfPoints( spinner )
    spinner_azimut_indexing = 'j' if spinner_axial_indexing == 'i' else 'i'
    spinner_profile = GSD.getBoundary(spinner_main_surface,
                                      spinner_azimut_indexing+'min')

    N_segs_azimut = C.getNCells( GSD.getBoundary( spinner_main_surface, spinner_axial_indexing+'min' ) )
    N_segs_foil = I.getZoneDim( blade_main_surface )[1] - 1
    N_segs_axial = int(N_segs_foil/2 - N_segs_azimut)

    blade_root = GSD.getBoundary( blade_main_surface, 'jmin')
    bary = G.barycenter( blade_root )
    bary_index,_ = D.getNearestPointIndex( spinner_profile, tuple(bary) )

    if N_segs_axial % 2 == 0:
        cut_indices = [int(bary_index-N_segs_axial/2), int(bary_index+N_segs_axial/2)]
    else:
        N_segs_axial -= 1
        cut_indices = [int(bary_index-N_segs_axial/2 +1), int(bary_index+N_segs_axial/2)]


    _,Ni,Nj,_,_ = I.getZoneDim( spinner_main_surface )
    if spinner_axial_indexing == 'i':
        front = T.subzone(spinner_main_surface, (1,1,1),
                                                (cut_indices[0]+1,Nj,1))
        middle = T.subzone(spinner_main_surface, (cut_indices[0]+1,1,1),
                                                (cut_indices[1]+1,Nj,1))
        rear = T.subzone(spinner_main_surface, (cut_indices[1]+1,1,1),
                                                (Ni,Nj,1))

    else:
        front = T.subzone(spinner_main_surface, (1 ,1,1),
                                                (Ni,cut_indices[0]+1,1))
        middle = T.subzone(spinner_main_surface, (1 ,cut_indices[0]+1,1),
                                                (Ni,cut_indices[1]+1,1))
        rear = T.subzone(spinner_main_surface, (1 ,cut_indices[1]+1,1),
                                                (Ni,Nj,1))
    front[0] = 'hub.front'
    middle[0] = 'hub.middle'
    rear[0] = 'hub.rear'

    return front, middle, rear


def _getSpineFromBlade( blade ):
    blade_main_surface = J.selectZoneWithHighestNumberOfPoints( blade )
    dim = I.getZoneDim( blade_main_surface )[-1]
    if dim == 3:
        external = GSD.getBoundary( blade_main_surface, 'kmax')
        spine = GSD.getBoundary( external, 'imin')
    else:
        spine = GSD.getBoundary( blade_main_surface, 'imin')

    return spine


def _buildExternalSurfacesHgrid(blade, spinner_split, rotation_axis,
                                rotation_center, spreading=2.):
    c = np.array(rotation_center,dtype=np.float64)
    a = np.array(rotation_axis,dtype=np.float64)
    spine = _getSpineFromBlade( blade )
    distribution = D.getDistribution( spine )

    W.addDistanceRespectToLine( spine , c, a, 'span')
    Length = C.getMaxValue( spine, 'span' ) - C.getMinValue( spine, 'span' )

    bary = D.point(G.barycenter(spinner_split))
    W.projectOnAxis(bary, rotation_axis, rotation_center)
    bary = W.point(bary)

    front = GSD.getBoundary(spinner_split,'imin')
    front[0] = 'front'
    rear = GSD.getBoundary(spinner_split,'imax')
    rear[0] = 'rear'
    sideA = GSD.getBoundary(spinner_split,'jmin')
    sideA[0] = 'sideA'
    sideB = GSD.getBoundary(spinner_split,'jmax')
    sideB[0] = 'sideB'
    spinner_contours = [ front, rear, sideA, sideB ]

    external_surfaces = []
    for contour in spinner_contours:
        lines = []
        x, y, z = J.getxyz( contour )
        for i in range( len(x) ):
            OX = np.array([ x[i], y[i], z[i] ])
            CX = OX - c
            OP = c  + a * CX.dot( a )
            PX = OX - OP
            PX_v = PX / np.sqrt( PX.dot( PX ) )
            OL = OX + PX_v * Length - spreading*a*(OP-bary)

            line = D.line( tuple(OX), tuple(OL), 2)
            line = G.map( line, distribution )
            lines += [ line ]

        stack = G.stack(lines)
        stack[0] = contour[0]+'.surf'
        external_surfaces += [ stack ]

    for surf in external_surfaces:
        if surf[0].startswith('sideA'):
            edge = GSD.getBoundary(surf,'imax')
            central_first_cell = W.distance(W.point(edge),W.point(edge,1))
            central_last_cell = W.distance(W.point(edge,-2),W.point(edge,-1))

    npts_azimut = C.getNPts(front)

    return external_surfaces, npts_azimut, central_first_cell, central_last_cell


def _buildSupportFromBoundaries(boundaries, rotation_center, rotation_axis):
    c = np.array(rotation_center,dtype=np.float64)
    a = np.array(rotation_axis,dtype=np.float64)
    boundaries = W.reorderAndSortCurvesSequentially( boundaries )
    alignment = [ a.dot( W.tangentExtremum( b ) ) for b in boundaries ]
    best_aligned = np.argmax( alignment )
    best_aligned_boundary = boundaries[ best_aligned ]
    azimut_boundary = boundaries[ best_aligned - 1 ]
    azimut_boundary = T.reorder(azimut_boundary, (-1,2,3))
    tangent_start = W.tangentExtremum( azimut_boundary )
    tangent_end = W.tangentExtremum(azimut_boundary, opposite_extremum=True)
    sector_angle = np.rad2deg(np.arccos( tangent_start.dot( tangent_end ) ))
    if sector_angle < 0: sector_angle += 180
    sector_angle = 360 / float(int(np.round(360/sector_angle)))
    proj_pt = W.point(azimut_boundary, as_pytree_point=True)
    W.projectOnAxis(proj_pt, a, c)
    start_to_proj = W.point(proj_pt) - W.point(azimut_boundary)
    rotation_sign = np.sign( a.dot ( np.cross(tangent_start,start_to_proj)))
    support = D.axisym(best_aligned_boundary,tuple(c),tuple(rotation_sign*a),
                       sector_angle, C.getNPts(azimut_boundary))
    support[0] = 'support'
    outter_cell_size = W.distance(W.point(azimut_boundary),W.point(azimut_boundary,1))
    return support, outter_cell_size

def _getInnerContour(blade,index,increasing_span_indexing='jmin'):
    blade_main_surface = J.selectZoneWithHighestNumberOfPoints(blade)
    slice = GSD.getBoundary(blade_main_surface, increasing_span_indexing, index)
    dim = I.getZoneDim( slice )[-1]
    if dim == 2:
        exterior = GSD.getBoundary(slice,'jmax')
        neighbor = GSD.getBoundary(slice,'jmax',-1)
        inner_cell_size = W.pointwiseDistances(exterior, neighbor)[2]
        W.pointwiseVectors(exterior, neighbor, reverse=True)
        # normals = W.getVisualizationNormals(exterior, length=inner_cell_size)
        return exterior, inner_cell_size
    return slice, None



def _buildHgridAroundBlade(external_surfaces, blade, rotation_center,
                           rotation_axis, H_grid_interior_points, relax_relative_length):
    spine = _getSpineFromBlade( blade )
    wall_cell_height = W.distance( W.point( spine ), W.point( spine, 1 ) )
    W.addDistanceRespectToLine(spine, rotation_center, rotation_axis, 'span')
    span, = J.getVars(spine, ['span'])
    MinSpan = np.min( span )
    MaxSpan = np.max( span )
    s = W.gets( spine )
    span_pts = len(s)

    Tik = Tok()
    nbOfDigits = int(np.ceil(np.log10(span_pts+1)))
    LastLayer = ('{:0%d}'%nbOfDigits).format(span_pts)
    All_surfs = []

    bnds_to_stack = []
    walls_to_stack = []
    bnds_stack = []
    walls_stack = []
    blks_stack = []

    all_ref_index = []

    for i in range( span_pts ):

        boundaries = [ GSD.getBoundary(e, 'imin', i) for e in external_surfaces ]
        inner_contour, inner_cell_size = _getInnerContour(blade,i)
        bnds_split, inner_split, ref_index = W.splitInnerContourFromOutterBoundariesTopology(
                                                    boundaries, inner_contour)
        all_ref_index += [ ref_index ]

    ref_index = int(np.median(all_ref_index))


    for i in range( span_pts ):

        boundaries = [ GSD.getBoundary(e, 'imin', i) for e in external_surfaces ]
        inner_contour, inner_cell_size = _getInnerContour(blade,i)
        bnds_split, inner_split,_ = W.splitInnerContourFromOutterBoundariesTopology(
                                                    boundaries, inner_contour,ref_index)

        if i>0 and i<span_pts-1:
            for j in range( len(bnds_split) ):
                bnds_to_stack[j] += [bnds_split[j]]
                walls_to_stack[j] += [inner_split[j]]
            continue

        elif i==0:
            for j in range( len(bnds_split) ):
                bnds_to_stack += [[]]
                walls_to_stack += [[]]
                bnds_stack += [[]]
                walls_stack += [[]]
                blks_stack += [[]]

            for j in range( len(bnds_split) ):
                bnds_to_stack[j] = [bnds_split[j]]
                walls_to_stack[j] = [inner_split[j]]

        else:

            for j in range( len(bnds_split) ):
                bnds_to_stack[j] += [bnds_split[j]]
                bnds_stack[j] += [G.stack(bnds_to_stack[j])]
                walls_to_stack[j] += [inner_split[j]]
                walls_stack[j] += [G.stack(walls_to_stack[j])]
                bnds_to_stack[j] = [bnds_to_stack[j][-1]]
                walls_to_stack[j] = [walls_to_stack[j][-1]]

        currentLayer = ('{:0%d}'%nbOfDigits).format(i+1)
        Message = J.MAGE+'H-grid'+J.ENDC+' %s/%s | cost: %0.5f s'%(currentLayer,LastLayer,Tok()-Tik)
        print(Message)
        Tik = Tok()

        all_ref_index += [ ref_index ]

        support, outter_cell_size = _buildSupportFromBoundaries(boundaries,
                                                rotation_center, rotation_axis)
        W.computeBarycenterDirectionalField(boundaries, support)
        W.projectNormals(boundaries, support, normal_projection_length=1e-4)
        W.projectNormals(inner_contour, support, normal_projection_length=1e-4)
        if inner_cell_size is None: inner_cell_size = wall_cell_height

        min_distance = W.pointwiseDistances(bnds_split, inner_split)[0]
        local_relax_length = relax_relative_length * min_distance if i>0 else 0

        surfs = GSD.makeH(boundaries, inner_contour,
                          inner_cell_size=inner_cell_size,
                          outter_cell_size=outter_cell_size,
                          number_of_points_union=H_grid_interior_points,
                          inner_normal_tension=0.3, outter_normal_tension=0.3,
                          projection_support=support,
                          global_projection_relaxation=0,
                          local_projection_relaxation_length=local_relax_length,
                          forced_split_index=ref_index)
        T._reorder(surfs, (2,1,3))
        All_surfs += [ surfs ]

    # zones = [s[0] for s in All_surfs]
    # I._correctPyTree(zones,level=3)
    # C.convertPyTree2File(zones,'test.cgns');exit()

    Tik = Tok()
    nb_stacks = len(All_surfs)
    nb_dgs = int(np.ceil(np.log10(nb_stacks)))
    LastLayer = ('{:0%d}'%nb_dgs).format(nb_stacks-1)
    for i in range( nb_stacks - 1):
        surfs = All_surfs[i]
        next_surfs = All_surfs[i+1]
        currentLayer = ('{:0%d}'%nb_dgs).format(i+1)
        Message = J.CYAN+'TFI...'+J.ENDC+' %s/%s | cost: %0.5f s'%(currentLayer,LastLayer,Tok()-Tik)
        print(Message)
        Tik = Tok()

        for j in range( len(bnds_split) ):
            wall_curve = GSD.getBoundary(walls_stack[j][i],'imin')
            bnd_curve = GSD.getBoundary(bnds_stack[j][i],'imin')
            surfs_curve = GSD.getBoundary(surfs[j],'imin')
            next_surfs_curve = GSD.getBoundary(next_surfs[j],'imin')
            curves = [wall_curve,bnd_curve,surfs_curve,next_surfs_curve]
            imin_surf = G.TFI( curves )


            wall_curve = GSD.getBoundary(walls_stack[j][i],'imax')
            bnd_curve = GSD.getBoundary(bnds_stack[j][i],'imax')
            surfs_curve = GSD.getBoundary(surfs[j],'imax')
            next_surfs_curve = GSD.getBoundary(next_surfs[j],'imax')
            curves = [wall_curve,bnd_curve,surfs_curve,next_surfs_curve]
            imax_surf = G.TFI( curves )

            blk = G.TFI([imin_surf, imax_surf,
                        walls_stack[j][i], bnds_stack[j][i],
                        surfs[j], next_surfs[j] ])

            blks_stack[j] += [blk]


    print('joining...')
    grids = []
    for j in range( len(bnds_split) ):
        blk = blks_stack[j][0]
        for blk2 in blks_stack[j][1:]:
            blk = T.join(blk,blk2)
        blk[0] = blks_stack[j][0][0]
        grids += [ blk ]
    print('joining... ok')

    for grid in grids:
        C._addBC2Zone(grid,'spinner_wall','FamilySpecified:SPINNER', 'kmin')

    return grids


def _buildHubWallAdjacentSector(surface, spinner, bulb, blade_number,
        rotation_center, rotation_axis,central_first_cell, central_last_cell,
        topo):

    central_cell = central_first_cell if topo =='front' else central_last_cell

    def getBinormal(curve, opposite=False):
        t = W.tangentExtremum(curve,True)
        n = np.cross(t,a)
        b = np.cross(t,n)
        b /= np.sqrt(b.dot(b))
        if opposite: b*=-1
        return b

    def getPseudoNormal(curve, opposite=False):
        t = W.tangentExtremum(curve,True)
        n = np.cross(a,t)
        n /= np.sqrt(n.dot(n))
        if opposite: n*=-1
        return n


    def makeBinormalUnion(curve1, curve2, NPts, tension=0.4, reverse=False):
        ext1 = W.extremum(curve1, True)
        ext2 = W.extremum(curve2, True)
        s1 = getBinormal(curve1, reverse)
        s2 = getBinormal(curve2, not reverse)
        d = W.distance(ext1,ext2)
        poly = D.polyline([tuple(ext1), tuple(ext1+d*tension*s1),
                           tuple(ext2+d*tension*s2), tuple(ext2)])
        union = D.bezier(poly,N=NPts)
        union = W.discretize(union,N=NPts, Distribution=dict(kind='tanhTwoSides',
            FirstCellHeight=last_cell_height, LastCellHeight=central_cell))
        return union

    def makeSemiBinormalUnion(curve1, curve2, NPts, tension=0.4, reverse=False):
        ext1 = W.extremum(curve1, True)
        ext2 = W.extremum(curve2, True)
        s1 = getBinormal(curve1, not reverse)
        d = W.distance(ext1,ext2)
        poly = D.polyline([tuple(ext1), tuple(ext1+d*tension*s1), tuple(ext2)])
        union = D.bezier(poly,N=NPts)
        union = W.discretize(union,N=NPts, Distribution=Distribution_edge)
        return union

    def makePseudoNormalUnion(curve1, curve2, NPts, tension=0.4):
        ext1 = W.extremum(curve1, True)
        ext2 = W.extremum(curve2, True)
        s1 = getPseudoNormal(curve1)
        s2 = getPseudoNormal(curve2,True)
        d = W.distance(ext1,ext2)
        poly = D.polyline([tuple(ext1), tuple(ext1+d*tension*s1),
                           tuple(ext2+d*tension*s2), tuple(ext2)])
        union = D.bezier(poly,N=NPts)
        union = W.discretize(union,N=NPts, Distribution=Distribution_edge)
        return union

    def makeSharpPseudoNormalUnion(curve1, curve2, NPts, tension=0.4, reverse_normal=False):
        ext1 = W.extremum(curve1, True)
        ext2 = W.extremum(curve2, True)
        s1 = getPseudoNormal(curve1, reverse_normal)
        d = W.distance(ext1,ext2)
        v = ext1 - ext2
        v /= np.sqrt( v.dot(v) )
        poly = D.polyline([tuple(ext1), tuple(ext1+d*tension*s1),
                           # tuple(ext2+d*tension*v),
                           tuple(ext2)])
        union = D.bezier(poly,N=NPts)
        union = W.discretize(union,N=NPts, Distribution=Distribution_edge)
        return union


    c = np.array(rotation_center,dtype=np.float64)
    a = np.array(rotation_axis,dtype=np.float64)

    surface_dist = GSD.getBoundary(surface,'jmin')
    wall_cell_height = W.distance(W.point(surface_dist),W.point(surface_dist,1))
    Length = D.getLength(surface_dist)

    ext_surf_edge  = GSD.getBoundary(surface,'imax')
    ext_surf_edge[0] = 'ext_surf_edge'

    surf_edge = GSD.getBoundary(surface,'jmin')
    surf_edge[0] = 'surf_edge'


    last_cell_height = W.distance(W.point(ext_surf_edge),W.point(ext_surf_edge,1))

    Distribution_wall=dict(kind='tanhTwoSides', FirstCellHeight=wall_cell_height,
                                               LastCellHeight=last_cell_height)
    Distribution_edge=dict(kind='tanhTwoSides', FirstCellHeight=last_cell_height,
                                               LastCellHeight=last_cell_height)

    NPts_surface_dist = C.getNPts(surface_dist)
    spinner = I.copyRef( spinner )
    G._getNormalMap(spinner)
    C.center2Node__(spinner,'centers:sx',cellNType=0)
    C.center2Node__(spinner,'centers:sy',cellNType=0)
    C.center2Node__(spinner,'centers:sz',cellNType=0)
    I._rmNodesByName(spinner,I.__FlowSolutionCenters__)
    C._normalize(spinner,['sx','sy','sz'])
    if topo == 'front':  spinner_edge = GSD.getBoundary(spinner,'imin')
    elif topo == 'rear': spinner_edge = GSD.getBoundary(spinner,'imax')

    # bulb
    bulb = I.copyRef(bulb)
    W.addDistanceRespectToLine(bulb, c, a, FieldNameToAdd='Distance2Line')
    d,= J.getVars(bulb,['Distance2Line'])
    x, y, z = J.getxyz(bulb)
    d = d.ravel(order='F')
    x = x.ravel(order='F')
    y = y.ravel(order='F')
    z = z.ravel(order='F')

    i = np.argmin(d)
    p = np.array([x[i],y[i],z[i]],dtype=float)
    cp = p-c
    P1 = p + np.sign( cp.dot(a) ) * a * Length
    axis_line = W.linelaw(tuple(p),tuple(P1), NPts_surface_dist, Distribution_wall)
    axis_line[0] = 'axis_line'



    if topo == 'front':
        bulb_side_0 = GSD.getBoundary(bulb,'imin')
        bulb_side_0[0] = 'bulb_side_0'
        bulb_side_1 = GSD.getBoundary(bulb,'jmax')
        bulb_side_1[0] = 'bulb_side_1'
        spinner_bulb_side_0 = GSD.getBoundary(bulb,'imax')
        spinner_bulb_side_0[0] = 'spinner_bulb_side_0'
        spinner_bulb_side_1 = GSD.getBoundary(bulb,'jmin')
        spinner_bulb_side_1[0] = 'spinner_bulb_side_1'

    elif topo == 'rear':
        bulb_side_0 = GSD.getBoundary(bulb,'imax')
        bulb_side_0[0] = 'bulb_side_0'
        bulb_side_1 = GSD.getBoundary(bulb,'jmax')
        bulb_side_1[0] = 'bulb_side_1'
        spinner_bulb_side_0 = GSD.getBoundary(bulb,'imin')
        spinner_bulb_side_0[0] = 'spinner_bulb_side_0'
        spinner_bulb_side_1 = GSD.getBoundary(bulb,'jmin')
        spinner_bulb_side_1[0] = 'spinner_bulb_side_1'




    GSD._alignNormalsWithRadialCylindricProjection(spinner_edge, c, a)
    x,y,z = J.getxyz(spinner_edge)
    spinner_edge_NPts = len(x)
    sx,sy,sz = J.getVars(spinner_edge,['sx','sy','sz'])
    middle_index = int((spinner_edge_NPts-1)/2)
    pt = np.array([x[0],y[0],z[0]],dtype=float)
    n  = np.array([sx[0],sy[0],sz[0]],dtype=float)
    P0 = pt
    P1 = pt + n * Length
    line_0 = W.linelaw(tuple(P0),tuple(P1), NPts_surface_dist, Distribution_wall)
    line_0[0] = 'line_0'

    reverse = True if topo == 'rear' else False
    bulb_union_0 = makeSemiBinormalUnion(line_0, axis_line,
                                             C.getNPts(bulb_side_0), reverse=reverse)
    bulb_union_0[0] = 'bulb_union_0'


    surf_edge_half = GSD.getBoundary(surface,'jmin',middle_index)
    surf_edge_half[0] = 'surf_edge_half'
    surf_edge_1 = GSD.getBoundary(surface,'jmax')
    surf_edge_1[0] = 'surf_edge_1'
    spinner_wall_edge_0 = GSD.getBoundary(spinner,'jmin')
    spinner_wall_edge_0[0] = 'spinner_wall_edge_0'
    spinner_wall_edge_half = GSD.getBoundary(spinner,'jmin',middle_index)
    spinner_wall_edge_half[0] = 'spinner_wall_edge_half'
    spinner_wall_edge_1 = GSD.getBoundary(spinner,'jmax')
    spinner_wall_edge_1[0] = 'spinner_wall_edge_1'
    tension = 0.4 if topo == 'front' else 0.25 # TODO make parameter
    spinner_union_0 = makeBinormalUnion(line_0, surf_edge, C.getNPts(spinner_wall_edge_0),
                                        tension=tension, reverse=reverse)
    spinner_union_0[0] = 'spinner_union_0'


    profile = T.join(spinner_union_0, bulb_union_0)
    profile[0] = 'profile'
    proj_support = D.axisym(profile,tuple(c),tuple(a),
            angle=360./float(blade_number), Ntheta=C.getNPts(ext_surf_edge))
    proj_support[0] = 'proj_support'
    G._getNormalMap(proj_support)
    C.center2Node__(proj_support,'centers:sx',cellNType=0)
    C.center2Node__(proj_support,'centers:sy',cellNType=0)
    C.center2Node__(proj_support,'centers:sz',cellNType=0)
    I._rmNodesByName(proj_support,I.__FlowSolutionCenters__)
    C._normalize(proj_support,['sx','sy','sz'])



    bulb_union_2, spinner_union_2, line_2 = T.rotate([bulb_union_0,
        spinner_union_0, line_0], tuple(c),tuple(a),360./float(blade_number))
    line_2[0] = 'line_2'
    bulb_union_2[0] = 'bulb_union_2'
    spinner_union_2[0] = 'spinner_union_2'



    proj_half = GSD.getBoundary(proj_support,'jmin',middle_index)
    proj_half[0] = 'proj_half'
    T._reorder(proj_half,(-1,2,3))
    s = W.gets(proj_half)
    s *= D.getLength(proj_half)
    L_diag = D.getLength(bulb_union_0) * np.sqrt(2)
    diag_cut_index = np.argmin( np.abs(s - L_diag) )
    spinner_union_1 = T.subzone(proj_half,(diag_cut_index,1,1),(-1,-1,-1))
    spinner_union_1 = W.discretize(spinner_union_1, C.getNPts(spinner_wall_edge_half),
                                   dict(kind='tanhTwoSides', FirstCellHeight=last_cell_height,
                                            LastCellHeight=central_cell))
    spinner_union_1[0] = 'spinner_union_1'
    T._reorder(spinner_union_1,(-1,2,3))



    ext_union_azm_0 = makeSharpPseudoNormalUnion(line_0,spinner_union_1,
                                    C.getNPts(spinner_bulb_side_0),0.5)
    ext_union_azm_0[0] = 'ext_union_azm_0'
    T._projectOrtho(ext_union_azm_0,proj_support)
    ext_union_azm_0 = W.discretize(ext_union_azm_0, C.getNPts(ext_union_azm_0),
                                   Distribution_edge)
    T._projectOrtho(ext_union_azm_0,proj_support)

    ext_union_azm_1 = makeSharpPseudoNormalUnion(line_2,spinner_union_1,
                                    C.getNPts(spinner_bulb_side_1),0.5,True)
    ext_union_azm_1[0] = 'ext_union_azm_1'
    T._projectOrtho(ext_union_azm_1,proj_support)
    ext_union_azm_1 = W.discretize(ext_union_azm_1, C.getNPts(ext_union_azm_1),
                                   Distribution_edge)
    T._projectOrtho(ext_union_azm_1,proj_support)



    x,y,z = J.getxyz(spinner_edge)
    spinner_edge_NPts = len(x)
    sx,sy,sz = J.getVars(spinner_edge,['sx','sy','sz'])
    i = middle_index
    P0 = np.array([x[i],y[i],z[i]],dtype=float)
    n  = np.array([sx[i],sy[i],sz[i]],dtype=float)
    ext2 = W.extremum(spinner_union_1,True)
    n_ext = np.array( P.extractPoint(proj_support, tuple(ext2)),dtype=float)
    if topo == 'rear': n_ext *= -1
    d = W.distance(P0, ext2)
    tension = 0.2 # TODO make parameter
    poly = D.polyline([tuple(P0), tuple(P0+tension*d*n),
                       tuple(ext2+tension*d*n_ext), tuple(ext2)])
    P1 = P0 + n * Length
    line_1 = D.bezier(poly,N=C.getNPts(line_0))
    line_1 = W.discretize(line_1,N=C.getNPts(line_0), Distribution=Distribution_wall)
    line_1[0] = 'line_1'


    wires = [ line_0, axis_line, bulb_union_0, surf_edge,
              spinner_wall_edge_0, spinner_wall_edge_1,
              spinner_bulb_side_0, spinner_bulb_side_1,
              spinner_union_0, bulb_union_2, line_2, spinner_union_2,
              spinner_union_1, ext_union_azm_0, ext_union_azm_1,
              surf_edge_1, bulb_side_0, bulb_side_1, line_1,
              spinner_wall_edge_half, surf_edge_half ]

    # existing surfaces:
    _,Ni,Nj,_,_=I.getZoneDim(spinner)
    spinner_wall_0 = T.subzone(spinner,(1,1,1),(Ni,middle_index+1,1))
    spinner_wall_0[0] = 'spinner_wall_0'
    spinner_wall_1 = T.subzone(spinner,(1,middle_index+1,1),(Ni,Nj,1))
    spinner_wall_1[0] = 'spinner_wall_1'

    _,Ni,Nj,_,_=I.getZoneDim(surface)
    surface_inter_0 = T.subzone(surface,(1,1,1),(Ni,middle_index+1,1))
    surface_inter_0[0] = 'surface_inter_0'
    surface_inter_1 = T.subzone(surface,(1,middle_index+1,1),(Ni,Nj,1))
    surface_inter_1[0] = 'surface_inter_1'

    ext_edge_surface_inter_0 = GSD.getBoundary(surface_inter_0,'imax')
    ext_edge_surface_inter_0[0] = 'ext_edge_surface_inter_0'
    ext_edge_surface_inter_1 = GSD.getBoundary(surface_inter_1,'imax')
    ext_edge_surface_inter_1[0] = 'ext_edge_surface_inter_1'
    wires.extend([ext_edge_surface_inter_0,ext_edge_surface_inter_1])

    # NEW SURFACES

    # correct extremum
    x,y,z = J.getxyz(spinner_union_1)
    ext = W.extremum(surf_edge_half,True)
    x[0] = ext[0]
    y[0] = ext[1]
    z[0] = ext[2]

    TFI2_inter = G.TFI([surf_edge_half, line_1,
                        spinner_wall_edge_half, spinner_union_1])
    TFI2_inter[0] = 'TFI2_inter'

    T._reorder(spinner_union_0,(-1,2,3))
    TFI2_inter_side_1 = G.TFI([surf_edge, line_0,
                               spinner_wall_edge_0, spinner_union_0])
    TFI2_inter_side_1[0] = 'TFI2_inter_side_1'

    TFI2_inter_side_2 = T.rotate(TFI2_inter_side_1,tuple(c),tuple(a),
                                 360./float(blade_number))
    TFI2_inter_side_2[0] = 'TFI2_inter_side_2'

    TFI2_inter_join_1 = G.TFI([line_0, axis_line,
                               bulb_side_0, bulb_union_0])
    TFI2_inter_join_1[0] = 'TFI2_inter_join_1'
    TFI2_inter_join_2 = T.rotate(TFI2_inter_join_1,tuple(c),tuple(a),
                                 360./float(blade_number))
    TFI2_inter_join_2[0] = 'TFI2_inter_join_2'

    TFI2_bulb_0 = G.TFI([line_0, line_1,
                         spinner_bulb_side_1, ext_union_azm_0])
    TFI2_bulb_0[0] = 'TFI2_bulb_0'

    TFI2_bulb_1 = G.TFI([line_1, line_2,
                         spinner_bulb_side_0, ext_union_azm_1])
    TFI2_bulb_1[0] = 'TFI2_bulb_1'

    TFI2_bulb = G.TFI([bulb_union_0, ext_union_azm_1,
                       bulb_union_2, ext_union_azm_0])
    TFI2_bulb[0] = 'TFI2_bulb'

    TFI2_spinner_1 = G.TFI([spinner_union_0, spinner_union_1,
                            ext_union_azm_0, ext_edge_surface_inter_0])
    TFI2_spinner_1[0] = 'TFI2_spinner_1'

    TFI2_spinner_2 = G.TFI([spinner_union_1, spinner_union_2,
                            ext_union_azm_1, ext_edge_surface_inter_1])
    TFI2_spinner_2[0] = 'TFI2_spinner_2'

    # project exterior faces on revolution support surface
    proj_surfs = [TFI2_bulb, TFI2_spinner_1, TFI2_spinner_2]
    fixed_bnds = [P.exteriorFaces(surf) for surf in proj_surfs ]
    GSD.prepareGlue(proj_surfs, fixed_bnds)
    T._projectOrtho(proj_surfs, proj_support)
    GSD.applyGlue(proj_surfs, fixed_bnds)


    surfs = [ bulb, spinner_wall_0, spinner_wall_1,
              surface_inter_0, surface_inter_1, TFI2_inter,
              TFI2_inter_side_1, TFI2_inter_side_2,
              TFI2_inter_join_1, TFI2_inter_join_2,
              TFI2_bulb_0, TFI2_bulb_1, TFI2_bulb,
              TFI2_spinner_1, TFI2_spinner_2]

    # VOLUME GRIDS
    print('making near-blade 3D TFI at %s...'%topo)
    TFI3_bulb = G.TFI([TFI2_inter_join_1, TFI2_bulb_1,
                       TFI2_bulb_0, TFI2_inter_join_2,
                       bulb, TFI2_bulb ])

    TFI3_spinner_1 = G.TFI([TFI2_bulb_0, surface_inter_0,
                            TFI2_inter_side_1, TFI2_inter,
                            spinner_wall_0, TFI2_spinner_1,])

    TFI3_spinner_2 = G.TFI([TFI2_bulb_1, surface_inter_1,
                            TFI2_inter, TFI2_inter_side_2,
                            spinner_wall_1, TFI2_spinner_2])
    TFI3_spinner = T.join(TFI3_spinner_1,TFI3_spinner_2)
    if topo == 'rear': T._reorder([TFI3_bulb, TFI3_spinner],(2,1,3))

    TFI3_bulb[0] = 'TFI3_bulb.'+topo
    TFI3_spinner_1[0] = 'TFI3_spinner_1'
    TFI3_spinner_2[0] = 'TFI3_spinner_2'
    TFI3_spinner[0] = 'TFI3_spinner.'+topo
    print('making near-blade 3D TFI at %s... ok'%topo)

    C._addBC2Zone(TFI3_bulb,'spinner_wall','FamilySpecified:SPINNER', 'kmin')
    C._addBC2Zone(TFI3_spinner,'spinner_wall','FamilySpecified:SPINNER', 'kmin')

    grids = [TFI3_bulb, TFI3_spinner]

    return wires, surfs, grids

def _buildHubWallAdjacentSectorWithoutBulb(surface, spinner, blade_number,
                            rotation_center, rotation_axis,central_last_cell):

    c = np.array(rotation_center,dtype=np.float64)
    a = np.array(rotation_axis,dtype=np.float64)

    surface_dist = GSD.getBoundary(surface,'jmin')
    surface_dist[0] =  'surface_dist'
    wall_cell_height = W.distance(W.point(surface_dist),W.point(surface_dist,1))
    Length = D.getLength(surface_dist)

    ext_surf_edge  = GSD.getBoundary(surface,'imax')
    ext_surf_edge[0] = 'ext_surf_edge'

    spinner_axial_edge = GSD.getBoundary(spinner,'jmin')
    spinner_axial_edge[0] = 'spinner_axial_edge'

    last_cell_height = W.distance(W.point(ext_surf_edge),W.point(ext_surf_edge,1))

    Distribution_wall=dict(kind='tanhTwoSides', FirstCellHeight=wall_cell_height,
                                               LastCellHeight=last_cell_height)
    Distribution_edge=dict(kind='tanhTwoSides', FirstCellHeight=last_cell_height,
                                               LastCellHeight=last_cell_height)

    NPts_surface_dist = C.getNPts(surface_dist)
    spinner = I.copyRef( spinner )
    G._getNormalMap(spinner)
    C.center2Node__(spinner,'centers:sx',cellNType=0)
    C.center2Node__(spinner,'centers:sy',cellNType=0)
    C.center2Node__(spinner,'centers:sz',cellNType=0)
    I._rmNodesByName(spinner,I.__FlowSolutionCenters__)
    C._normalize(spinner,['sx','sy','sz'])
    spinner_edge = GSD.getBoundary(spinner,'imax')
    spinner_edge[0] = 'spinner_edge'
    GSD._alignNormalsWithRadialCylindricProjection(spinner_edge, c, a)

    x,y,z = J.getxyz( spinner_edge )
    sx, sy, sz = J.getVars( spinner_edge, ['sx', 'sy', 'sz'] )

    rear_edge = W.linelaw(P1=(x[0], y[0], z[0]),
        P2=(x[0]+Length*sx[0], y[0]+Length*sy[0], z[0]+Length*sz[0]),
        N=C.getNPts(surface_dist))
    rear_edge = W.discretize(rear_edge,C.getNPts(surface_dist),W.copyDistribution(surface_dist))
    rear_edge[0] = 'rear_edge'

    ext_surf_edge_segment = W.distance(W.point(ext_surf_edge),W.point(ext_surf_edge,1))
    TE_segment = W.distance(W.point(spinner_axial_edge,-1),W.point(spinner_axial_edge,-2))

    spinner_union_0 = W.linelaw(P1=W.point(rear_edge,-1),P2=W.point(ext_surf_edge),
                           N=C.getNPts(spinner_axial_edge),
                           Distribution=dict(kind='tanhTwoSides',
                                            FirstCellHeight=TE_segment,
                                            LastCellHeight=central_last_cell))
    spinner_union_0[0] = 'spinner_union_0'

    TFI2_inter_side_1 = G.TFI([ surface_dist, rear_edge,
                                spinner_axial_edge, spinner_union_0])
    TFI2_inter_side_1[0] = 'TFI2_inter_side_1'
    TFI2_inter_side_2,rear_edge_2,spinner_union_1 = T.rotate([TFI2_inter_side_1,
            rear_edge,spinner_union_0],tuple(c),tuple(a),360./float(blade_number))
    TFI2_inter_side_2[0] = 'TFI2_inter_side_2'
    rear_edge_2[0] = 'rear_edge_2'
    spinner_union_1[0] = 'spinner_union_1'

    dir1 = np.cross(a, W.tangentExtremum(rear_edge,True))
    dir2 = np.cross(a, W.tangentExtremum(rear_edge_2,True))
    pt1 = W.point(rear_edge,-1)
    pt2 = W.point(rear_edge_2,-1)
    d = W.distance(pt1, pt2)
    TE_azm = W.discretize(D.bezier(D.polyline([
                    tuple(pt1), tuple(pt1 + 0.5*d*dir1),
                    tuple(pt2 - 0.5*d*dir2), tuple(pt2),
                    ]), N=500), N=C.getNPts(ext_surf_edge),
                    Distribution=W.copyDistribution(ext_surf_edge))
    TE_azm[0] = 'TE_azm'

    T._reorder([spinner_union_0, spinner_union_1],(-1,2,3))
    TFI2_spinner = G.TFI([spinner_union_0, spinner_union_1,
                          ext_surf_edge, TE_azm])
    TFI2_spinner[0] = 'TFI2_spinner'

    TFI2_spinner_1, TFI2_spinner_2 = T.splitNParts(TFI2_spinner,2,dirs=[1])
    TFI2_spinner_1[0] = 'TFI2_spinner_1'
    TFI2_spinner_2[0] = 'TFI2_spinner_2'
    TFI2_TE = G.TFI([rear_edge, rear_edge_2, TE_azm, spinner_edge])
    TFI2_TE[0] = 'TFI2_TE'

    wires = [spinner_edge,ext_surf_edge,rear_edge, rear_edge_2, surface_dist,
            spinner_axial_edge,spinner_union_0,spinner_union_1]

    surfs = [spinner,surface, TFI2_inter_side_1, TFI2_inter_side_2, TE_azm,
            TFI2_spinner_1,TFI2_spinner_2, TFI2_TE]

    print('making open near-blade 3D TFI at rear...')
    TFI3_spinner = G.TFI([TFI2_TE, surface,
                          TFI2_inter_side_2, TFI2_inter_side_1,
                          spinner, TFI2_spinner])
    TFI3_spinner[0] = 'TFI3_spinner'
    print('making open near-blade 3D TFI at rear... ok')
    C._addBC2Zone(TFI3_spinner,'spinner_wall','FamilySpecified:SPINNER', 'kmin')
    grids = [ TFI3_spinner ]

    return wires, surfs, grids



def _gatherSectorBoundaries(Hgrids, blade, surfs_rear, surfs_front):
    selected_names = ['TFI2_spinner_1','TFI2_spinner_2','TFI2_bulb']
    surfaces = [GSD.getBoundary(z,'kmax') for z in Hgrids]
    surfs_rear = T.reorder(surfs_rear,(1,-2,3))
    surfaces.extend([s for s in surfs_front+surfs_rear if s[0] in selected_names])
    surfaces.extend([GSD.getBoundary(z,'kmax') for z in I.getZones(blade) if z[0].startswith('tip')])
    I._correctPyTree(surfaces,level=3)
    uns = C.convertArray2Hexa(surfaces)
    uns = T.merge(uns)
    G._getNormalMap(uns)
    for s in I.getZones(uns):
        C.center2Node__(s,'centers:sx',cellNType=0)
        C.center2Node__(s,'centers:sy',cellNType=0)
        C.center2Node__(s,'centers:sz',cellNType=0)
    I._rmNodesByName(uns,'FlowSolution#Centers')
    C._normalize(uns, ['sx','sy','sz'])
    T._smoothField(uns, 0.9, 100, 0, ['sx','sy','sz']) # TODO externalize param?
    C._normalize(uns, ['sx','sy','sz'])
    J.migrateFields(uns, surfaces)

    return surfaces


def _extractWallAdjacentSectorFullProfile(wires_front, wires_rear, external_surfaces):
    profile_curves =      [c for c in wires_front if c[0]=='bulb_union_0']
    profile_curves.extend([c for c in wires_front if c[0]=='spinner_union_0'])
    profile_curves.extend([GSD.getBoundary(s,'imax') for s in external_surfaces if s[0]=='sideA.surf'])
    profile_curves.extend([c for c in wires_rear  if c[0]=='spinner_union_0'])
    profile_curves.extend([c for c in wires_rear  if c[0]=='bulb_union_0'])
    I._correctPyTree(profile_curves,level=3)
    return profile_curves


def _buildFarfieldSupport(profile, blade_number, npts_azimut, distance,
                               rotation_center=[0,0,0], rotation_axis=[-1,0,0]):
    c = np.array(rotation_center,dtype=np.float64)
    a = np.array(rotation_axis,dtype=np.float64)

    T._reorder(profile[0],(-1,2,3))
    profile = W.reorderCurvesSequentially(profile)
    profile_joined = profile[0]
    for p in profile[1:]: profile_joined = T.join(profile_joined,p)
    profile_joined[0] = 'profile_joined'

    original_distribution = W.copyDistribution(profile_joined)

    profile_farfield = I.copyTree(profile_joined)
    profile_farfield[0] = 'profile_farfield'
    W.addNormals(profile_farfield)
    GSD._alignNormalsWithRadialCylindricProjection(profile_farfield, c, a)

    x,y,z = J.getxyz(profile_farfield)
    sx,sy,sz = J.getVars(profile_farfield,['sx','sy','sz'])
    if W.distanceOfPointToLine(W.extremum(profile_farfield),a,c) < 1e-10:
        sx[0], sy[0], sz[0] = a
    if W.distanceOfPointToLine(W.extremum(profile_farfield,True),a,c) < 1e-10:
        sx[-1], sy[-1], sz[-1] = -a
    x += distance * sx
    y += distance * sy
    z += distance * sz

    split_points = []
    cumul = 0
    for p in profile[:-1]:
        cumul += C.getNPts(p) - 1
        split_points += [ tuple(W.point(profile_farfield, cumul)) ]


    def getTangent(curve):
        x,y,z = J.getxyz(curve)
        dx = np.diff(x)
        dy = np.diff(y)
        dz = np.diff(z)
        norm = np.sqrt(dx*dx + dy*dy + dz*dz)
        tx = np.hstack((0.,dx/norm))
        ty = np.hstack((0.,dy/norm))
        tz = np.hstack((0.,dz/norm))
        return tx, ty, tz

    profile_rev = I.copyTree(profile_farfield)
    profile_rev[0] = 'profile_rev'
    all_valid = False
    print('optimizing farfield revolution profile...')
    while not all_valid:
        x,y,z = J.getxyz(profile_rev)
        tx, ty, tz = getTangent(profile_rev)
        tjx, tjy, tjz = getTangent(profile_joined)
        scalar = tx * tjx + ty * tjy + tz * tjz
        valid = scalar > 0
        for i in range(1,len(valid)):
            if not valid[i] and valid[i-1] and i>0:
                valid[i-1] = False
        valid[0] = True
        valid[-1] = True
        all_valid = np.all(valid)
        x = x[valid]
        y = y[valid]
        z = z[valid]

        profile_rev = J.createZone('profile_rev',[x,y,z],['x','y','z'])
        profile_rev = W.discretize(profile_rev,N=C.getNPts(profile_joined))
    print('optimizing farfield revolution profile... ok')

    subprofiles = []
    indices = [r[0] for r in D.getNearestPointIndex(profile_rev, split_points)]
    subprofile = T.subzone(profile_rev,(1,1,1),(indices[0]+1,1,1))
    subprofile = W.discretize(subprofile,N=C.getNPts(profile[0]),Distribution=
                              W.copyDistribution(profile[0]))
    subprofiles += [ subprofile ]
    for i in range(1,len(indices)):
        subprofile = T.subzone(profile_rev,(indices[i-1]+1,1,1),(indices[i]+1,1,1))
        subprofile = W.discretize(subprofile,N=C.getNPts(profile[i]),Distribution=
                                  W.copyDistribution(profile[i]))
        subprofiles += [ subprofile ]
    subprofile = T.subzone(profile_rev,(indices[i]+1,1,1),(-1,-1,-1))
    subprofile = W.discretize(subprofile,N=C.getNPts(profile[-1]),Distribution=
                              W.copyDistribution(profile[-1]))
    subprofiles += [ subprofile ]

    profile_rev = subprofiles[0]
    for sp in subprofiles[1:]: profile_rev = T.join(profile_rev,sp)

    proj_support = D.axisym(profile_rev,tuple(c),tuple(a),
            angle=360./float(blade_number), Ntheta=npts_azimut)
    proj_support[0] = 'proj_support'

    return proj_support, profile_rev

def _extractBladeTipTopology(sector_bnds, rotation_axis):
    a = np.array(rotation_axis,dtype=np.float64)


    tip_curves = [GSD.getBoundary(s,'jmin') for s in sector_bnds if s[0].startswith('tfi')]

    # compute mean oscullatory plane (binormal)
    contours = W.reorderAndSortCurvesSequentially(tip_curves)
    for s in sector_bnds:
        if s[0].startswith('tip'):
            tip_curves.extend(P.exteriorFacesStructured(s))
    contour = contours[0]
    for c in contours[1:]: contour = T.join(contour,c)
    cx, cy, cz = J.getxyz(contour)
    NPts = len(cx)
    cxyz = np.vstack((cx, cy, cz)).T
    fT = np.zeros((NPts,3),order='F')
    fT[1:-1,:] = 0.5*(np.diff(cxyz[:-1,:],axis=0)+np.diff(cxyz[1:,:],axis=0))
    fT[0,:] = (cxyz[1,:]-cxyz[0,:])
    fT[-1,:] = (cxyz[-1,:]-cxyz[-2,:])
    fT /= np.sqrt(np.sum(fT*fT, axis=1)).reshape((NPts,1),order='F')
    binormal = np.mean(np.cross(fT[1:],fT[:-1]),axis=0) # pseudo-blade
    binormal /= (binormal[0]**2+binormal[1]**2+binormal[2]**2)**0.5
    normal = np.cross(a,binormal)
    normal /= np.sqrt(normal.dot(normal))
    tangent = np.cross(binormal,normal) # psuedo-axial

    bary = np.array(G.barycenter(contour))
    frenet = (tuple(binormal), tuple(tangent), tuple(normal))

    return tip_curves, bary, frenet

def _buildFarfieldSector(blade, sector_bnds, profile, blade_number, npts_azimut,
                         H_grid_interior_points,
                         rotation_center=[0,0,0], rotation_axis=[-1,0,0],
                         distance=10., number_of_points=200,
                         farfield_cell_height=1.,
                         tip_axial_scaling_at_farfield = 0.5,
                         normal_tension=0.05):

    c = np.array(rotation_center,dtype=np.float64)
    a = np.array(rotation_axis,dtype=np.float64)

    if len(profile) == 5:
        HAS_REAR_BULB = True
    elif len(profile) == 4:
        HAS_REAR_BULB = False
    else:
        C.convertPyTree2File(profile,'debug.cgns')
        raise ValueError('unsuported profile topology. Check debug.cgns')

    spine = _getSpineFromBlade( blade )
    tip_cell_length = W.distance(W.point(spine,-1),W.point(spine,-2))
    support, profile_rev = _buildFarfieldSupport(profile, blade_number,
                          npts_azimut, distance, rotation_center, rotation_axis)

    _,Ni,Nj,_,_=I.getZoneDim(support)
    support_central = T.subzone(support, (C.getNPts(profile[:2])-1,1,1),
                                         (C.getNPts(profile[:3])-1,Nj,1) )

    profile_sideB = T.rotate(profile,tuple(c),tuple(a),360./float(blade_number))

    profile_rev_sectors = []
    first_index = 1
    for p in profile:
        last_index = first_index + C.getNPts(p) - 1
        subpart = T.subzone(profile_rev,(first_index,1,1),(last_index,1,1))
        subpart[0]=p[0]+'.far'
        profile_rev_sectors += [ subpart ]
        first_index = last_index

    # rediscretization of profile_rev_sectors
    profile_rev_sectors[0] = W.discretize(profile_rev_sectors[0],
                                            N=C.getNPts(profile_rev_sectors[0]))
    profile_rev_sectors[2] = W.discretize(profile_rev_sectors[2],
                                            N=C.getNPts(profile_rev_sectors[2]))
    first_cell = W.segment(profile_rev_sectors[0],-1)
    second_cell = W.segment(profile_rev_sectors[2])
    third_cell = W.segment(profile_rev_sectors[3],-1)
    if HAS_REAR_BULB:
        profile_rev_sectors[-1] = W.discretize(profile_rev_sectors[-1],
                                            N=C.getNPts(profile_rev_sectors[-1]))
        third_cell = W.segment(profile_rev_sectors[-1])
    profile_rev_sectors[1] = W.discretize(profile_rev_sectors[1],
            N=C.getNPts(profile_rev_sectors[1]), Distribution=dict(
            kind='tanhTwoSides',
            FirstCellHeight=first_cell,
            LastCellHeight=second_cell))
    profile_rev_sectors[3] = W.discretize(profile_rev_sectors[3],
            N=C.getNPts(profile_rev_sectors[3]), Distribution=dict(
            kind='tanhTwoSides',
            FirstCellHeight=second_cell,
            LastCellHeight=third_cell))


    profile_rev_sectors_sideB = T.rotate(profile_rev_sectors,tuple(c),tuple(a),
                                         360./float(blade_number))
    for p in profile_rev_sectors_sideB: p[0] += '.B'

    middle_index = int((npts_azimut-1)/2)
    support_half_edge = GSD.getBoundary(support,'jmin',middle_index)
    support_half_edge[0] = 'support_half_edge'

    cell_height = W.distance(W.point(profile_rev_sectors[0]),
                             W.point(profile_rev_sectors[0],1))
    Distribution_edge=dict(kind='tanhTwoSides', FirstCellHeight=cell_height,
                                               LastCellHeight=cell_height)



    s = W.gets(support_half_edge)
    s *= D.getLength(support_half_edge)
    L_diag = D.getLength(profile_rev_sectors[0]) * np.sqrt(2)
    diag_cut_index = np.argmin( np.abs(s - L_diag) )
    far_union_1 = T.subzone(support_half_edge,(diag_cut_index,1,1),
      (C.getNPts(profile_rev_sectors[0])+C.getNPts(profile_rev_sectors[1])-1,1,1))
    FirstCell = W.segment(profile_rev_sectors[1])
    LastCell = W.segment(profile_rev_sectors[1],-1)
    far_union_1 = W.discretize(far_union_1,N=C.getNPts(profile_rev_sectors[1]),
        Distribution=dict(kind='tanhTwoSides',
                          FirstCellHeight=FirstCell,LastCellHeight=LastCell))
    far_union_1[0] = 'far_union_1'

    # azimutal
    to_split = GSD.getBoundary(support,'imin',
        C.getNPts(profile_rev_sectors[0])-1+C.getNPts(profile_rev_sectors[1])-1)
    H_azm_0 = T.subzone(to_split, (1,1,1),(middle_index+1,1,1))
    H_azm_0[0] = 'H_azm_0'
    H_azm_1 = T.subzone(to_split, (middle_index+1,1,1),(-1,-1,-1))
    H_azm_1[0] = 'H_azm_1'

    to_split = GSD.getBoundary(support,'imin',
        C.getNPts(profile_rev_sectors[0])-1 + \
        C.getNPts(profile_rev_sectors[1])-1 +
        C.getNPts(profile_rev_sectors[2])-1)
    H_azm_low_0 = T.subzone(to_split, (1,1,1),(middle_index+1,1,1))
    H_azm_low_0[0] = 'H_azm_low_0'
    H_azm_low_1 = T.subzone(to_split, (middle_index+1,1,1),(-1,-1,-1))
    H_azm_low_1[0] = 'H_azm_low_1'
    H_azm_low = T.join(H_azm_low_0,H_azm_low_1)
    H_azm_low[0] = 'H_azm_low'


    # union of bulb with projection on support
    # first side
    ext_1 = W.extremum(profile_rev_sectors[0],True)
    ext_2 = W.extremum(far_union_1)
    t = W.tangentExtremum(profile_rev_sectors[0],True)
    v_az = np.cross(a,t)
    v_az /= np.sqrt(v_az.dot(v_az))
    d = W.distance(ext_1, ext_2)
    tension = 0.6
    poly = D.polyline([tuple(ext_1), tuple(ext_1+d*tension*v_az),tuple(ext_2)])
    T._projectOrtho(poly, support)
    bezier = D.bezier(poly,N=C.getNPts(profile_rev_sectors[0]))
    T._projectOrtho(bezier, support)
    far_bulb_union_0 = W.discretize(bezier,N=C.getNPts(bezier),
                                    Distribution=Distribution_edge)
    far_bulb_union_0[0] = 'far_bulb_union_0'
    T._projectOrtho(far_bulb_union_0, support)
    # second side
    ext_1 = W.extremum(profile_rev_sectors_sideB[0],True)
    ext_2 = W.extremum(far_union_1)
    t = W.tangentExtremum(profile_rev_sectors_sideB[0],True)
    v_az = -np.cross(a,t)
    v_az /= np.sqrt(v_az.dot(v_az))
    d = W.distance(ext_1, ext_2)
    tension = 0.6
    poly = D.polyline([tuple(ext_1), tuple(ext_1+d*tension*v_az),tuple(ext_2)])
    T._projectOrtho(poly, support)
    bezier = D.bezier(poly,N=C.getNPts(profile_rev_sectors[0]))
    T._projectOrtho(bezier, support)
    far_bulb_union_1 = W.discretize(bezier,N=C.getNPts(bezier),
                                    Distribution=Distribution_edge)
    far_bulb_union_1[0] = 'far_bulb_union_1'
    T._projectOrtho(far_bulb_union_1, support)
    front_tfi_0 = G.TFI([profile_rev_sectors[1],far_union_1,
                         far_bulb_union_0, H_azm_0])
    front_tfi_0[0]='front_tfi_0'
    front_tfi_1 = G.TFI([far_union_1, profile_rev_sectors_sideB[1],
                         far_bulb_union_1, H_azm_1])
    front_tfi_1[0]='front_tfi_1'
    main_front_tfi = T.join(front_tfi_0,front_tfi_1)
    main_front_tfi[0] = 'main_front_tfi'
    far_bulb_tfi = G.TFI([profile_rev_sectors[0],far_bulb_union_1,
                          profile_rev_sectors_sideB[0],far_bulb_union_0])
    far_bulb_tfi[0] = 'far_bulb_tfi'



    if HAS_REAR_BULB:
        reversed_support_half_edge = T.reorder(support_half_edge,(-1,2,3))
        s = W.gets(reversed_support_half_edge)
        s *= D.getLength(reversed_support_half_edge)
        L_diag = D.getLength(profile_rev_sectors[-1]) * np.sqrt(2)
        diag_cut_index = np.argmin( np.abs(s - L_diag) )
        far_union_2 = T.subzone(reversed_support_half_edge,(diag_cut_index,1,1),
          (C.getNPts(profile_rev_sectors[-1])+C.getNPts(profile_rev_sectors[-2])-1,1,1))
        far_union_2 = W.discretize(far_union_2,N=C.getNPts(profile_rev_sectors[-2]),
            Distribution=dict(kind='tanhTwoSides',
                LastCellHeight=W.segment(profile_rev_sectors[-2]),
                FirstCellHeight=W.segment(profile_rev_sectors[-2],-1)))
        far_union_2[0] = 'far_union_2'


        # REAR union of bulb with projection on support
        # first side
        ext_1 = W.extremum(profile_rev_sectors[-1])
        ext_2 = W.extremum(far_union_2)
        t = W.tangentExtremum(profile_rev_sectors[-1])
        v_az = np.cross(a,t)
        v_az /= np.sqrt(v_az.dot(v_az))
        d = W.distance(ext_1, ext_2)
        tension = 0.6
        poly = D.polyline([tuple(ext_1), tuple(ext_1+d*tension*v_az),tuple(ext_2)])
        T._projectOrtho(poly, support)
        bezier = D.bezier(poly,N=C.getNPts(profile_rev_sectors[0]))
        T._projectOrtho(bezier, support)
        far_rear_bulb_union_0 = W.discretize(bezier,N=C.getNPts(bezier),
                                        Distribution=Distribution_edge)
        far_rear_bulb_union_0[0] = 'far_rear_bulb_union_0'
        T._projectOrtho(far_rear_bulb_union_0, support)
        # second side
        ext_1 = W.extremum(profile_rev_sectors_sideB[-1])
        ext_2 = W.extremum(far_union_2)
        t = W.tangentExtremum(profile_rev_sectors_sideB[-1])
        v_az = -np.cross(a,t)
        v_az /= np.sqrt(v_az.dot(v_az))
        d = W.distance(ext_1, ext_2)
        tension = 0.6
        poly = D.polyline([tuple(ext_1), tuple(ext_1+d*tension*v_az),tuple(ext_2)])
        T._projectOrtho(poly, support)
        bezier = D.bezier(poly,N=C.getNPts(profile_rev_sectors[0]))
        T._projectOrtho(bezier, support)
        far_rear_bulb_union_1 = W.discretize(bezier,N=C.getNPts(bezier),
                                        Distribution=Distribution_edge)
        far_rear_bulb_union_1[0] = 'far_rear_bulb_union_1'
        T._projectOrtho(far_rear_bulb_union_1, support)

        far_rear_bulb_union = T.join(far_rear_bulb_union_0,far_rear_bulb_union_1)


        main_rear_tfi_0 = G.TFI([profile_rev_sectors[3], far_union_2,
                                   H_azm_low_0,far_rear_bulb_union_0])
        main_rear_tfi_1 = G.TFI([far_union_2,profile_rev_sectors_sideB[3],
                               H_azm_low_1,far_rear_bulb_union_1])
        main_rear_tfi = T.join(main_rear_tfi_0, main_rear_tfi_1)
        main_rear_tfi[0] = 'main_rear_tfi'

    else:
        cut_i = np.sum([C.getNPts(profile_rev_sectors[i]) for i in range(4)])-3
        # in practice this should be replaced by 'imax' instead of using cut_i
        far_rear_bulb_union = GSD.getBoundary(support,'imin',cut_i-1)


    central_index_i = C.getNPts(profile_rev_sectors[0]) + \
        C.getNPts(profile_rev_sectors[1])+int((C.getNPts(profile_rev_sectors[2])-1)/2) -2

    xs, ys, zs = J.getxyz(support)
    pseudo_axial = np.array([
        xs[central_index_i-1,middle_index]-xs[central_index_i,middle_index],
        ys[central_index_i-1,middle_index]-ys[central_index_i,middle_index],
        zs[central_index_i-1,middle_index]-zs[central_index_i,middle_index]])
    pseudo_axial /= np.sqrt(pseudo_axial.dot(pseudo_axial))
    pseudo_front = np.array([
        xs[central_index_i,middle_index+1]-xs[central_index_i,middle_index],
        ys[central_index_i,middle_index+1]-ys[central_index_i,middle_index],
        zs[central_index_i,middle_index+1]-zs[central_index_i,middle_index]])
    pseudo_front /= np.sqrt(pseudo_front.dot(pseudo_front))
    pseudo_blade = np.cross(pseudo_front,pseudo_axial)
    pseudo_blade /= np.sqrt(pseudo_blade.dot(pseudo_blade))
    frenet_topo = (pseudo_blade, pseudo_axial, pseudo_front)
    topo_center = np.array([xs[central_index_i,middle_index],
                   ys[central_index_i,middle_index],
                   zs[central_index_i,middle_index]])

    tip_curves, bary, frenet = _extractBladeTipTopology(sector_bnds, rotation_axis)
    t_tip_curves = C.newPyTree(['Base',tip_curves])
    tip_curves_topo = I.getZones(I.copyTree(t_tip_curves))
    T._rotate(tip_curves_topo, bary, frenet, frenet_topo)
    T._translate(tip_curves_topo, topo_center-bary)

    central_width = W.distance(W.point(profile_rev_sectors[2],0),
                               W.point(profile_rev_sectors[2],-1))
    charact_length = W.getCharacteristicLength(tip_curves_topo)
    factor = tip_axial_scaling_at_farfield*central_width/charact_length
    T._homothety(tip_curves_topo, topo_center, factor)
    T._projectOrtho(tip_curves_topo, support)

    I._rmNodesByType(tip_curves_topo,'FlowSolution_t')
    [C._initVars(tip_curves_topo,'s'+i,0) for i in ('x','y','z')]
    J.migrateFields(support,tip_curves_topo)
    C._normalize(tip_curves_topo,['sx','sy','sz'])

    for crv in tip_curves:
        sx,sy,sz = J.getVars(crv,['sx','sy','sz'])
        sx[:]=frenet[0][0]
        sy[:]=frenet[0][1]
        sz[:]=frenet[0][2]

    GSD._alignNormalsWithRadialCylindricProjection(tip_curves,rotation_center,rotation_axis)

    TFI2_blends = []
    for c1, c2 in zip(tip_curves,tip_curves_topo):
        TFI2_blend = W.fillWithBezier(c1, c2, number_of_points,
                        tension1=normal_tension, tension2=0., # TODO parameter ?
                        length1=tip_cell_length, length2=farfield_cell_height)
        TFI2_blend[0] = c1[0]+'.blend'
        TFI2_blends += [ TFI2_blend ]


    all_topos = []
    for i in range(10):
        topos = [cr for cr in tip_curves_topo if cr[0].startswith('tip.%d'%i)]
        if not topos: break
        topos = W.reorderAndSortCurvesSequentially(topos)
        all_topos += [ topos ]

    All_TFI2_far = [G.TFI([cr[0],cr[2],cr[1],cr[3]]) for cr in all_topos]

    TFI_H_group_topo = [cr for cr in tip_curves_topo if cr[0].startswith('tfi')]
    I._rmNodesByType(TFI_H_group_topo,'FlowSolution_t')
    W.addNormals(TFI_H_group_topo)

    TFI_H_group_bnd = [T.join(H_azm_0, H_azm_1), profile_rev_sectors_sideB[2],
                       T.join(H_azm_low_0, H_azm_low_1),profile_rev_sectors[2]]
    I._rmNodesByType(TFI_H_group_bnd,'FlowSolution_t')
    TFI_H_group_bnd = W.reorderAndSortCurvesSequentially(TFI_H_group_bnd)
    proj_dir = np.cross(pseudo_blade, a)
    proj_dir = np.cross(a, proj_dir)


    portion_length = D.getLength(TFI_H_group_bnd[1]) * 0.75 # TODO parameter here or make it smart
    focus = W.splitAt(TFI_H_group_bnd[0],[portion_length,
                        D.getLength(TFI_H_group_bnd[0])-portion_length],'length')[1]
    focus = W.discretize(focus,N=C.getNPts(TFI_H_group_bnd[0]))
    T._translate(focus,tuple(0.5*(W.point(TFI_H_group_bnd[2],-1)-W.point(TFI_H_group_bnd[0]))))
    T._projectOrtho(focus,support_central)

    xf, yf, zf = J.getxyz( focus )
    curve = TFI_H_group_bnd[0]
    sx, sy, sz = J.invokeFields(curve,['sx','sy','sz'])
    x, y, z = J.getxyz( curve )
    sx[:] = xf - x
    sy[:] = yf - y
    sz[:] = zf - z

    curve = TFI_H_group_bnd[1]
    sx, sy, sz = J.invokeFields(curve,['sx','sy','sz'])
    x, y, z = J.getxyz( curve )
    sx[:] = xf[-1] - x
    sy[:] = yf[-1] - y
    sz[:] = zf[-1] - z

    curve = TFI_H_group_bnd[2]
    sx, sy, sz = J.invokeFields(curve,['sx','sy','sz'])
    x, y, z = J.getxyz( curve )
    sx[:] = xf[::-1] - x
    sy[:] = yf[::-1] - y
    sz[:] = zf[::-1] - z

    curve = TFI_H_group_bnd[3]
    sx, sy, sz = J.invokeFields(curve,['sx','sy','sz'])
    x, y, z = J.getxyz( curve )
    sx[:] = xf[0] - x
    sy[:] = yf[0] - y
    sz[:] = zf[0] - z


    C._normalize(TFI_H_group_bnd,['sx','sy','sz'])
    C._normalize(TFI_H_group_bnd,['sx','sy','sz'])
    W.projectNormals(TFI_H_group_bnd, support_central, projection_direction=proj_dir)



    print('fill farfield H with bezier...')
    topo_cell = W.meanSegmentLength(TFI_H_group_topo)
    i = 0
    TFI2_bnd_blends = []
    for c1, c2 in zip(TFI_H_group_topo,TFI_H_group_bnd):
        i+=1
        TFI2_bnd_blend = W.fillWithBezier(c1, c2, H_grid_interior_points,
                        tension1=normal_tension*central_width, tension2=0.15, # TODO parameter ?
                        tension1_is_absolute=True,
                        tension2_is_absolute=False,
                        length1 = topo_cell, length2=cell_height,
                        support=support_central,
                        projection_direction=proj_dir
                        )
        TFI2_bnd_blend[0] = 'bl.%d.blend'%i
        TFI2_bnd_blends += [ TFI2_bnd_blend ]
    print('fill farfield H with bezier... ok')

    print('smoothing farfield H...')
    fixedZones = TFI_H_group_topo+TFI_H_group_bnd
    fixedZones.extend( [ GSD.getBoundary(z,'imin',1) for z in TFI2_bnd_blends ] )
    fixedZones.extend( [ GSD.getBoundary(z,'imin',2) for z in TFI2_bnd_blends ] )
    GSD.prepareGlue(TFI2_bnd_blends,fixedZones)
    for i in range( 5 ):
        T._smooth(TFI2_bnd_blends,eps=0.8,niter=200,type=2,
            fixedConstraints=fixedZones)
        T._projectRay(TFI2_bnd_blends,support_central, [0,0,0])
    GSD.applyGlue(TFI2_bnd_blends,fixedZones)
    print('smoothing farfield H... ok')


    profile_rev_sectors = W.reorderAndSortCurvesSequentially(profile_rev_sectors)
    for p in profile+profile_rev_sectors: W.addNormals(p)

    GSD._alignNormalsWithRadialCylindricProjection(profile+profile_rev_sectors,
                                                rotation_center, rotation_axis)

    for p in [profile[0], profile_rev_sectors[0]]:
        if W.distanceOfPointToLine(W.point(p,0),a,c) < 1e-8:
            sx,sy,sz = J.getVars(p,['sx','sy','sz'])
            sx[0] = a[0]
            sy[0] = a[1]
            sz[0] = a[2]

    for p in [profile[-1], profile_rev_sectors[-1]]:
        if W.distanceOfPointToLine(W.point(p,-1),a,c) < 1e-8:
            sx,sy,sz = J.getVars(p,['sx','sy','sz'])
            sx[-1] = -a[0]
            sy[-1] = -a[1]
            sz[-1] = -a[2]


    join_cell_length = W.segment(profile[0])
    union_curves = []
    i = -1
    for c1, c2 in zip(profile,profile_rev_sectors):
        i+=1
        length1 = tip_cell_length if i in (2,3) else join_cell_length
        union_curve = W.fillWithBezier(c1,c2,number_of_points, length1=length1,
                    length2=farfield_cell_height,tension2=0.,tension1=normal_tension,
                    only_at_indices=[0])
        union_curves += [ union_curve ]
    union_curve = W.fillWithBezier(c1,c2,number_of_points, length1=length1,
                length2=farfield_cell_height, only_at_indices=[-1],tension2=0.,
                tension1=normal_tension,)
    union_curves += [ union_curve ]

    tfi_unions = []
    i = -1
    for c1, c2 in zip(profile,profile_rev_sectors):
        i+=1
        tfi_unions += [ G.TFI([c1,c2,union_curves[i],union_curves[i+1]]) ]

    tfi_unions_sideB = T.rotate(tfi_unions,c,a,360./float(blade_number))
    for cr in tfi_unions_sideB:
        cr[0]+='.B'


    TFI2_bulbs = [s for s in sector_bnds if s[0].startswith('TFI2_bulb')]
    TFI2_bulb_front = TFI2_bulbs[0]
    TFI2_bulb_front_inner_0 = GSD.getBoundary(TFI2_bulb_front,'jmin')
    TFI2_bulb_front_inner_0[0] = 'TFI2_bulb_front_inner_0'
    TFI2_bulb_front_inner_1 = GSD.getBoundary(TFI2_bulb_front,'imax')
    TFI2_bulb_front_inner_1[0] = 'TFI2_bulb_front_inner_1'

    J._invokeFields(far_bulb_union_0,['sx','sy','sz'])

    inner_union_bulb_front = W.fillWithBezier(TFI2_bulb_front_inner_0,
        far_bulb_union_0,number_of_points, length1=join_cell_length,
        length2=farfield_cell_height, only_at_indices=[-1])
    inner_union_bulb_front[0] = 'inner_union_bulb_front'

    inner_union_bulb_front_sideA = GSD.getBoundary(tfi_unions[0],'jmax')
    inner_union_bulb_front_sideA[0] = 'inner_union_bulb_front_sideA'
    inner_union_bulb_front_sideB = GSD.getBoundary(tfi_unions_sideB[0],'jmax')
    inner_union_bulb_front_sideB[0] = 'inner_union_bulb_front_sideB'
    tfi_inner_bulb_front_sideA = G.TFI([
        inner_union_bulb_front_sideA,inner_union_bulb_front,
        TFI2_bulb_front_inner_0,far_bulb_union_0])
    tfi_inner_bulb_front_sideA[0] = 'tfi_inner_bulb_front_sideA'
    tfi_inner_bulb_front_sideB = G.TFI([
        inner_union_bulb_front,inner_union_bulb_front_sideB,
        TFI2_bulb_front_inner_1,far_bulb_union_1])
    tfi_inner_bulb_front_sideB[0] = 'tfi_inner_bulb_front_sideB'


    inner_union_main_front_sideA = GSD.getBoundary(tfi_unions[1],'jmax')
    inner_union_main_front_sideA[0] = 'inner_union_main_front_sideA'
    inner_union_main_front_sideB = GSD.getBoundary(tfi_unions_sideB[1],'jmax')
    inner_union_main_front_sideB[0] = 'inner_union_main_front_sideB'
    H_azm_front = T.join(H_azm_0, H_azm_1)
    H_azm_front[0] = 'H_azm_front'

    TFI2_spinners = [s for s in sector_bnds if s[0].startswith('TFI2_spinner')]
    TFI2_spinner_join = T.join(*TFI2_spinners[:2])
    TFI2_spinner_join[0] = 'TFI2_spinner_join'
    TFI2_spinner_join_lowedge = GSD.getBoundary(TFI2_spinner_join,'jmin')
    TFI2_spinner_join_lowedge[0] = 'TFI2_spinner_join_lowedge'
    TFI2_bulb_join = T.join(tfi_inner_bulb_front_sideA,tfi_inner_bulb_front_sideB)
    TFI2_bulb_join[0] = 'TFI2_bulb_join'

    # rear
    TFI2_spinner_rear = T.join(*TFI2_spinners[2:])
    TFI2_spinner_rear[0] = 'TFI2_spinner_rear'
    TFI2_spinner_rear_lowedge = GSD.getBoundary(TFI2_spinner_rear,'jmax')
    TFI2_spinner_rear_lowedge[0] = 'TFI2_spinner_rear_lowedge'

    rear_near_low = GSD.getBoundary(TFI2_spinner_rear,'jmin')
    rear_near_low[0] = 'rear_near_low'
    rear_near_low_0,rear_near_low_1=T.splitNParts(rear_near_low,2,dirs=[1])
    rear_near_low_0[0] = 'rear_near_low_0'
    rear_near_low_1[0] = 'rear_near_low_1'


    I._rmNodesByType([far_bulb_union_0, far_bulb_union_1],'FlowSolution_t')
    join_bulb = T.join(far_bulb_union_0, far_bulb_union_1)
    join_bulb[0] = 'join_bulb'

    tfi_front_top = T.join(tfi_inner_bulb_front_sideA,tfi_inner_bulb_front_sideB)
    tfi_front_top[0] = 'tfi_front_top'
    tfi_front_bot = G.TFI([inner_union_main_front_sideA,inner_union_main_front_sideB,
                        TFI2_spinner_join_lowedge,H_azm_front])
    tfi_front_bot[0] = 'tfi_front_bot'

    # 1 of 4
    for s in TFI2_bnd_blends:
        if W.isSubzone(H_azm_front, s):
            tfi2_H_front = s
            tfi2_H_front[0] = 'tfi2_H_front'
            curve_H_far_top = GSD.getBoundary(tfi2_H_front,'jmin')
            curve_H_far_top[0] = 'curve_H_far_top'
            break
    for s in sector_bnds:
        if W.isSubzone(TFI2_spinner_join_lowedge,s):
            tfi2_H_front_spinner = s
            tfi2_H_front_spinner[0] = 'tfi2_H_front_spinner'
            curve_H_near_top = GSD.getBoundary(tfi2_H_front_spinner,'imin')
            curve_H_near_top[0] = 'curve_H_near_top'
            break

    # 2 of 4
    for s in TFI2_bnd_blends:
        if W.isSubzone(profile_rev_sectors[2], s):
            tfi2_H_sideA = s
            tfi2_H_sideA[0] = 'tfi2_H_sideA'
            curve_H_far_sideA = GSD.getBoundary(tfi2_H_sideA,'jmin')
            curve_H_far_sideA[0] = 'curve_H_far_sideA'
            break
    try:
        FOUR = len(curve_H_far_sideA)
    except:
        C.convertPyTree2File(TFI2_bnd_blends+[profile_rev_sectors[2]],'debug.cgns')
        raise ValueError('could not retrieve curve_H_far_sideA')

    for s in sector_bnds:
        if W.isSubzone(profile[2],s):
            tfi2_H_sideA_spinner = s
            tfi2_H_sideA_spinner[0] = 'tfi2_H_sideA_spinner'
            curve_H_near_sideA = GSD.getBoundary(tfi2_H_sideA_spinner,'imin')
            curve_H_near_sideA[0] = 'curve_H_near_sideA'
            break

    # 3 of 4
    for s in TFI2_bnd_blends:
        if W.isSubzone(profile_rev_sectors_sideB[2], s):
            tfi2_H_sideB = s
            tfi2_H_sideB[0] = 'tfi2_H_sideB'
            curve_H_far_sideB = GSD.getBoundary(tfi2_H_sideB,'jmin')
            curve_H_far_sideB[0] = 'curve_H_far_sideB'
            break
    for s in sector_bnds:
        if W.isSubzone(profile_sideB[2],s):
            tfi2_H_sideB_spinner = s
            tfi2_H_sideB_spinner[0] = 'tfi2_H_sideB_spinner'
            curve_H_near_sideB = GSD.getBoundary(tfi2_H_sideB_spinner,'imin')
            curve_H_near_sideB[0] = 'curve_H_near_sideB'
            break

    all_blend_unions = [GSD.getBoundary(s,'jmin') for s in TFI2_blends]

    # 4 of 4
    for s in TFI2_bnd_blends:
        if W.isSubzone(H_azm_low, s):
            tfi2_H_rear = s
            tfi2_H_rear[0] = 'tfi2_H_rear'
            curve_H_far_rear = GSD.getBoundary(tfi2_H_rear,'jmin')
            curve_H_far_rear[0] = 'curve_H_far_rear'
            break

    for s in sector_bnds:
        if W.isSubzone(TFI2_spinner_rear_lowedge,s):
            tfi2_H_rear_spinner = s
            tfi2_H_rear_spinner[0] = 'tfi2_H_rear_spinner'
            curve_H_near_rear = GSD.getBoundary(tfi2_H_rear_spinner,'imin')
            curve_H_near_rear[0] = 'curve_H_near_rear'
            break

    all_blend_unions = []
    for i,s in enumerate(TFI2_blends):
        crv = GSD.getBoundary(s,'jmin')
        crv[0] = 'blend.%d'%i
        all_blend_unions += [ crv ]

    sides_curves = [GSD.getBoundary(s,'jmax') for s in tfi_unions+tfi_unions_sideB]
    candidates_to_connect = sides_curves + all_blend_unions

    c1, c2 = W.getConnectingCurves(curve_H_near_top, candidates_to_connect)
    tfi_H1 = G.TFI([curve_H_near_top, curve_H_far_top, c1,c2])
    tfi_H1[0] = 'tfi_H1'

    c1, c2 = W.getConnectingCurves(curve_H_near_sideA, candidates_to_connect)
    tfi_H2 = G.TFI([curve_H_near_sideA, curve_H_far_sideA, c1,c2])
    tfi_H2[0] = 'tfi_H2'

    c1, c2 = W.getConnectingCurves(curve_H_near_sideB, candidates_to_connect)
    tfi_H3 = G.TFI([curve_H_near_sideB, curve_H_far_sideB, c1,c2])
    tfi_H3[0] = 'tfi_H3'

    c1, c2 = W.getConnectingCurves(curve_H_near_rear, candidates_to_connect)
    tfi_H4 = G.TFI([curve_H_near_rear, curve_H_far_rear, c1,c2])
    tfi_H4[0] = 'tfi_H4'

    if not HAS_REAR_BULB:
        main_rear_tfi = G.TFI([profile_rev_sectors[3],profile_rev_sectors_sideB[3],
                               H_azm_low,far_rear_bulb_union])
        main_rear_tfi[0] = 'main_rear_tfi'

    c1, c2 = W.getConnectingCurves(TFI2_spinner_rear_lowedge, candidates_to_connect)
    tfi_rear_top = G.TFI([TFI2_spinner_rear_lowedge, H_azm_low, c1,c2])
    tfi_rear_top[0] = 'tfi_rear_top'

    c1, c2 = W.getConnectingCurves(rear_near_low, candidates_to_connect)
    tfi_rear_low = G.TFI([rear_near_low, far_rear_bulb_union, c1,c2])
    tfi_rear_low[0] = 'tfi_rear_low'

    FACES_BULB_FRONT = [tfi_unions[0],tfi_inner_bulb_front_sideB,
                        tfi_inner_bulb_front_sideA,tfi_unions_sideB[0],
                        TFI2_bulb_front, far_bulb_tfi]

    FACES_MAIN_FRONT = [tfi_front_top,tfi_front_bot,
                        tfi_unions[1],tfi_unions_sideB[1],
                        TFI2_spinner_join,main_front_tfi]


    lower_side = GSD.selectConnectingSurface([tfi_H1,tfi_H3],TFI2_blends)
    inner_side = GSD.selectConnectingSurface([tfi_H1,tfi_H3],sector_bnds)

    FACES_H_FRONT = [tfi_H1,tfi_H3,
                     tfi_front_bot, lower_side,
                     inner_side,tfi2_H_front]

    right_side = GSD.selectConnectingSurface([tfi_H3,tfi_H4],TFI2_blends)
    left_side = GSD.selectConnectingSurface([tfi_H3,tfi_H4],tfi_unions_sideB)
    ant_side = GSD.selectConnectingSurface([tfi_H3,tfi_H4],sector_bnds)
    post_side = GSD.selectConnectingSurface([tfi_H3,tfi_H4],TFI2_bnd_blends)

    FACES_H_sideA = [tfi_H3,tfi_H4,
                     right_side, left_side,
                     ant_side,post_side]

    right = GSD.selectConnectingSurface([tfi_H1,tfi_H2],TFI2_blends)
    left = GSD.selectConnectingSurface([tfi_H1,tfi_H2],tfi_unions)
    ant = GSD.selectConnectingSurface([tfi_H1,tfi_H2],sector_bnds)
    post = GSD.selectConnectingSurface([tfi_H1,tfi_H2],TFI2_bnd_blends)

    FACES_H_sideB = [tfi_H1,tfi_H2,
                      right,  left,
                          ant,post]


    top = GSD.selectConnectingSurface([tfi_H2,tfi_H4],TFI2_blends)
    bot = tfi_rear_top
    ant = GSD.selectConnectingSurface([tfi_H2,tfi_H4],sector_bnds)
    post = GSD.selectConnectingSurface([tfi_H2,tfi_H4],TFI2_bnd_blends)

    FACES_H_REAR = [tfi_H2,tfi_H4,
                    top, bot,
                    ant,post]

    left = tfi_unions[3]
    right = tfi_unions_sideB[3]
    top = tfi_rear_top
    bottom = tfi_rear_low
    ant = TFI2_spinner_rear
    post = main_rear_tfi

    FACES_MAIN_REAR = [left,right,top,bottom,ant,post]

    GROUP_TFI_CENTRAL = []
    for post in All_TFI2_far:
        neighbors = GSD.selectConnectingSurface([post], TFI2_blends, mode='all')
        ant = GSD.selectConnectingSurface(neighbors, sector_bnds, mode='first')
        GROUP_TFI_CENTRAL += [ neighbors+[ant,post] ]

    if HAS_REAR_BULB:
        rear_bulb_far = G.TFI([profile_rev_sectors[-1],far_rear_bulb_union_1,
                              profile_rev_sectors_sideB[-1],far_rear_bulb_union_0])
        rear_bulb_far[0] = 'rear_bulb_far'
        left = tfi_unions[-1]
        top, right = T.splitNParts(tfi_rear_low,2,dirs=[2])
        bottom = tfi_unions_sideB[-1]
        ant = GSD.selectConnectingSurface([left,right],sector_bnds)
        post = rear_bulb_far

        FACES_BULB_REAR = [left,right,top,bottom,ant,post]

    BaseZonesList = ['bulb_front',FACES_BULB_FRONT,
                     'main_front',FACES_MAIN_FRONT,
                     'h_front',FACES_H_FRONT,
                     'h_sidea',FACES_H_sideA,
                     'h_sideb',FACES_H_sideB,
                     'h_rear',FACES_H_REAR,
                     'main_rear',FACES_MAIN_REAR]

    if HAS_REAR_BULB: BaseZonesList.extend(['bulb_rear',FACES_BULB_REAR])

    for i, zones in enumerate(GROUP_TFI_CENTRAL):
        BaseZonesList.extend(['tip_%d'%i, zones])

    farfield_faces = C.newPyTree(BaseZonesList)

    grids = []
    print('making farfield 3D TFI...')
    for base in I.getBases(farfield_faces):
        tfi3D = G.TFI(I.getZones(base))
        tfi3D[0] = base[0]
        grids += [ tfi3D ]
    print('making farfield 3D TFI... ok')
    T._makeDirect(grids)

    return grids, farfield_faces


def angleBetweenVectors(a, b):
    return np.abs(np.rad2deg( np.arccos( a.dot(b) / (norm(a)*norm(b)) ) ))


def projectApproximateBladeDirectionOnRotationPlane(RotationAxis,
        RequestedBladeDirection, misalignment_tolerance_in_degree=5.0):
    
    RotAxis = np.array(RotationAxis,dtype=float)
    BladeDir = np.array(RequestedBladeDirection,dtype=float)
    
    RotAxis /= norm(RotAxis)
    BladeDir /= norm(BladeDir)


    angle = angleBetweenVectors(BladeDir, RotAxis)
    while angle < 5.:
        BladeDir[0] += 0.01
        BladeDir[1] += 0.02
        BladeDir[2] += 0.03
        BladeDir/= norm(BladeDir)
        angle = angleBetweenVectors(BladeDir, RotAxis)

    # Force initial azimut direction to be on the Rotation plane
    CoplanarBinormalVector = np.cross(BladeDir, RotAxis)
    BladeDir = np.cross(RotAxis, CoplanarBinormalVector)
    BladeDir /= norm(BladeDir)

    return BladeDir

def placeRotorAndDuplicateBlades(InitialMesh, InitialRotationCenter,
        InitialRotationAxis, InitialBladeDirection, InitialRightHandRuleRotation,
        FinalRotationCenter, FinalRotationAxis, FinalBladeDirection, 
        FinalRightHandRuleRotation=True, AzimutalDuplicationNumber=1,
        orthonormal_tolerance_in_degree=0.5):

    O0 = np.array(InitialRotationCenter, dtype=float)
    a0 = np.array(InitialRotationAxis, dtype=float)
    b0 = np.array(InitialBladeDirection, dtype=float)
    d0 = 1 if InitialRightHandRuleRotation else -1

    O1 = np.array(FinalRotationCenter, dtype=float)
    a1 = np.array(FinalRotationAxis, dtype=float)
    b1 = np.array(FinalBladeDirection, dtype=float)
    d1 = 1 if FinalRightHandRuleRotation else -1

    a0 /= norm(a0)
    b0 /= norm(b0)
    a1 /= norm(a1)
    b1 /= norm(b1)
    
    if abs(angleBetweenVectors(a0,b0) - 90) > orthonormal_tolerance_in_degree:
        msg = 'InitialRotationAxis and InitialBladeDirection must form 90 deg'
        raise AttributeError(J.FAIL+msg+J.ENDC)

    if abs(angleBetweenVectors(a1,b1) - 90) > orthonormal_tolerance_in_degree:
        b1 = projectApproximateBladeDirectionOnRotationPlane(a1, b1, misalignment_tolerance_in_degree=5.0)
        msg = 'warning: FinalRotationAxis and FinalBladeDirection must form 90 deg'
        msg+= '\nFinalBladeDirection is projected on rotation plane and becomes: %s'%str(b1)
        print(J.WARN+msg+J.ENDC)

    FinalMeshes = [I.copyTree(InitialMesh)]
    if AzimutalDuplicationNumber > 1:
        for i in range(1, AzimutalDuplicationNumber):
            AzPos = i*d0*(360.0/float(AzimutalDuplicationNumber))
            FinalMeshes += [ T.rotate(InitialMesh,tuple(O0),tuple(a0),AzPos) ]


    InitialFrame = (tuple(b0),                 # blade-wise
                    tuple(d0*np.cross(a0,b0)), # sweep-wise
                    tuple(a0))                 # rotation axis

    FinalFrame   = (tuple(b1),                 # blade-wise
                    tuple(d1*np.cross(a1,b1)), # sweep-wise
                    tuple(a1))                 # rotation axis

    T._rotate(FinalMeshes,(0,0,0),InitialFrame,arg2=FinalFrame)
    T._translate(FinalMeshes,tuple(O1))
    I._correctPyTree(FinalMeshes,level=2)
    I._correctPyTree(FinalMeshes,level=3)
    
    if InitialRightHandRuleRotation != FinalRightHandRuleRotation:
        for t in FinalMeshes:
            for z in I.getZones(t):
                T._reorder(z,(-1,2,3))

    return FinalMeshes