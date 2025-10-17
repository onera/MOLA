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
Creation by recycling RotatoryWings.py of v1.18.1
'''

import os
import pprint
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


from mola.math_tools import interpolate__
# Generative modules
from mola.pytree import InternalShortcuts as J
from mola.mesh import curve as W
from mola.mesh import surface as GSD
from mola.mesh import volume as GVD
from mola.cfd.postprocess.interpolation import migrateFields

maxRadius = W.maxRadius

def extrudeBladeSupportedOnSpinner(blade_surface, spinner, rotation_center,
        rotation_axis, blade_wall_cell_height=2e-6, spinner_wall_cell_height=2e-6,
        root_to_transition_distance=0.1,
        root_to_transition_number_of_points=11,
        maximum_number_of_points_in_normal_direction=500, distribution_law='ratio',
        distribution_growth_rate=1.15, last_extrusion_cell_height=1e-3,
        maximum_extrusion_distance_at_spinner=5e-3,
        smoothing_start_at_layer=10,
        smoothing_normals_iterations=3,
        smoothing_normals_subiterations=[2,30,'distance'],
        smoothing_growth_iterations=2,
        smoothing_growth_subiterations=50,
        smoothing_growth_coefficient=[0.1,0.5,'distance'],
        smoothing_expansion_factor=[0.05,0.2,'index'],
        expand_distribution_radially=False,
        intersection_method='conformize',
        DIRECTORY_CHECKME='CHECKME',
        raise_error_if_negative_volume_cells=False,

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

        blade_wall_cell_height : float
            the cell height to verify in wall-adjacent cells of the blade

        spinner_wall_cell_height : float
            the cell height to verify in wall-adjacent cells of the spinner

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
    if DIRECTORY_CHECKME:
        J.save(J.tree(INTERSECTION_BAR_4=intersection_bar,
                      SPINNER_SURFACE_4=spinner_surface),
               os.path.join(DIRECTORY_CHECKME,'4_blade_hub_intersection_contour.cgns'))

    supported_profile = _convertIntersectionContour2Structured(blade_surface,
                            spinner_surface, root_section_index, intersection_bar)


    Distributions = _computeBladeDistributions(blade_surface, rotation_axis,
            supported_profile, transition_section_index, distribution_law,
            maximum_extrusion_distance_at_spinner, maximum_number_of_points_in_normal_direction,
            blade_wall_cell_height, last_extrusion_cell_height,
            distribution_growth_rate, smoothing_start_at_layer,
            smoothing_normals_iterations, smoothing_normals_subiterations,
            smoothing_growth_iterations, smoothing_growth_subiterations,
            smoothing_growth_coefficient, smoothing_expansion_factor, expand_distribution_radially)

    supported_match = _extrudeBladeRootOnSpinner(spinner_surface,
                        Distributions[0], supported_profile)

    if DIRECTORY_CHECKME:
        J.save(J.tree(SUPPORTED_MATCH_5=supported_match),
               os.path.join(DIRECTORY_CHECKME,'5_blade_root_extrusion.cgns'))
    
    if GSD.hasSelfIntersectingFaces(supported_match):
        msg = ("Blade root projected at hub is self-intersecting. "
            "Adjust blade extrusion parameters or input geometry accordingly. "
            "Check 5_blade_root_extrusion.cgns")
        raise ValueError(J.FAIL+msg+J.ENDC)

    blade_root2trans_extrusion = _extrudeBladeFromTransition(blade_surface,
                                    root_section_index, Distributions)

    ExtrusionResult = _buildAndJoinCollarGrid(blade_surface, blade_root2trans_extrusion, transition_section_index,
            root_section_index, supported_profile, supported_match, spinner_wall_cell_height,
            root_to_transition_number_of_points, CollarLaw='interp1d_cubic')

    base, = I.getBases(C.newPyTree(['BLADE',ExtrusionResult]))
    for zone in I.getZones(base):
        C._addBC2Zone(zone,'blade_wall','FamilySpecified:BLADE', 'kmin')
    blade = J.selectZoneWithHighestNumberOfPoints(I.getZones(base))
    C._addBC2Zone(blade,'spinner_wall','FamilySpecified:SPINNER', 'jmin')

    if DIRECTORY_CHECKME:
        J.save(J.tree(BLADE_FINAL_MAIN_WALL_5=GSD.getBoundary(blade,'kmin')),
               os.path.join(DIRECTORY_CHECKME,'5_blade_final_main_wall.cgns'))


    print('checking negative volume cells in blade... ',end='')
    negative_volume_cells = GVD.checkNegativeVolumeCells(base, volume_threshold=0)
    if negative_volume_cells and DIRECTORY_CHECKME:
        try: os.makedirs(DIRECTORY_CHECKME)
        except: pass
        J.save(negative_volume_cells, os.path.join(DIRECTORY_CHECKME,'negative_volume_cells_in_blade.cgns'))
        if raise_error_if_negative_volume_cells:
            raise ValueError(J.FAIL+"stopped since detected negative cells in blade"+J.ENDC)

    else:
        print(J.GREEN+'ok'+J.ENDC)


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

def makeSpinnerCurves(LengthFront=0.2, LengthRear=10, Width=0.15,
                      RelativeArcRadiusFront=0.008, ArcAngleFront=40.,
                      RelativeTensionArcFront=0.1, RelativeTensionRootFront=0.5,
                      NPtsArcFront=200, NPtsSpinnerFront=5000,
                      TopologyRear='line',
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
    profile = W.joinSequentially(curves)
    profile[0] = 'HubProfile'
    T._rotate(profile,(0,0,0),(0,0,1),90.0)

    return profile

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

def getEulerAngles(RotationAxis, PhaseDirection=(0,1,0)):
    '''
    Given a RotationAxis and a Phase Direction, produce the Euler angles.

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

    # TODO propagate PhaseDirection up to BodyForceModeling (user-level)
    FrenetDEST = getFrenetFromRotationAxisAndPhaseDirection(RotationAxis,PhaseDirection)


    FrenetEULER = np.array([[1.,0.,0.],  # Rotation Axis
                           [0.,1.,0.],  # Phase Dir
                           [0.,0.,1.]]) # Binormal

    FrenetDEST = np.array([list(FrenetDEST[2]),
                           list(FrenetDEST[0]),
                           list(FrenetDEST[1])])

    RotationMatrix = FrenetDEST.T.dot(npla.inv(FrenetEULER.T))

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
    # TODO: replace search_index and j increasing with automatic detection 
    # of spanwise index direction
    W.addDistanceRespectToLine(blade_surface, rotation_center, rotation_axis,
                        FieldNameToAdd='radius')
    for j in range(Nj):
        Section = GSD.getBoundary(blade_surface,search_index,j)
        if strictlyPositive:
            radius = C.getMinValue(Section,'radius')
        else:
            radius = C.getMaxValue(Section,'radius')
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
        errmsg = 'it seems that blade does not intersects the spinner sufficiently\n'
        errmsg+= 'or maybe your blade surface root is not at "jmin" (you may need to reorder indices)'
        raise ValueError(errmsg)

    return section_index

def _getRootAndTransitionIndices(blade_surface, spinner, rotation_center,
                                rotation_axis, root_to_transition_distance):

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
    
    

    _,_,Nj,_,_ = I.getZoneDim(blade_main_surface)
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
        transition_section_index, method='conformize'):

    blade_main_surface = J.selectZoneWithHighestNumberOfPoints( blade_surface )
    _,Ni,Nj,Nk,_ = I.getZoneDim(blade_main_surface)
    trimmed_blade_root = T.subzone(blade_main_surface,(1,1,1),
                                                (Ni,transition_section_index,1))
    spinner_surface = _splitSpinnerHgrid(spinner, blade_surface, spinner_axial_indexing='i')[1]

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

def _extrudeBladeRootOnSpinner(spinner_surface, Distribution, supported_profile):
    
    Constraints = [dict(kind='Projected',ProjectionMode='ortho',
                        curve=supported_profile, surface=spinner_surface)]

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
    transition_sections = discard_intersecting_sections(transition_sections)
    transition_distribution = W.getDistributionFromHeterogeneousInput__(W.linelaw(
                                   P2=(transition_distance,0,0), N=root_to_transition_number_of_points,
                                   Distribution=dict(kind='tanhTwoSides',
                                   FirstCellHeight=wall_cell_height,
                                   LastCellHeight=transition_cell_width)))[1]

    nb_sections = len(transition_sections)
    if nb_sections == 2:
        CollarLaw = 'interp1d_linear'
    elif nb_sections == 3 and CollarLaw == 'interp1d_cubic':
        CollarLaw = 'interp1d_quadratic'
    elif nb_sections == 1:
        raise ValueError("got only 1 section, cannot continue."
            " This may be provoked by presence of negative cells in blade root remesh region")


    extruded_transition, spine_curve_for_debugging = GVD.multiSections(
        transition_sections, transition_distribution,
        InterpolationData={'InterpolationLaw':CollarLaw})
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
                   H_grid_interior_spreading_angles=[-10,10],
                   relax_relative_length=0.5, distance=10., max_radius=np.inf,
                   number_of_points=200,
                   farfield_cell_height=1., tip_axial_scaling_at_farfield=0.5,
                   normal_tension=0.05, RightHandRuleRotation=True,
                   radial_H_compromise=0.25,
                   tip_radial_tension=0.03,
                   h_inner_normal_tension=0.3,
                   h_outter_normal_tension=0.3,
                   HgridXlocations=[],
                   radial_breakpoints=[0.5],
                   FarfieldProfileAbscissaDeltas=[],
                   FarfieldTipSmoothIterations=500,
                   H_front_blade_reference=None,
                   H_rear_blade_reference=None,
                   front_support=None,
                   rear_support=None,
                   shift=0,
                   DIRECTORY_CHECKME='CHECK_ME',
                   raise_error_if_negative_volume_cells=False):
    print('building match mesh...')
    
    spinner_front, spinner_middle, spinner_rear = _splitHubAtHgrid(spinner, HgridXlocations)

    if DIRECTORY_CHECKME:
        t_check = J.tree(SPINNER_SUBPARTS_5=[spinner_front, spinner_middle, spinner_rear])
        J.save(t_check, os.path.join(DIRECTORY_CHECKME,'5_spinner_subparts.cgns'))

    
    # NEW APPROACH
    external_surfaces = buildCurvedExternalSurfacesHgrid(blade, blade_number,
                            spinner_middle, H_front_blade_reference,
                            H_rear_blade_reference, 
                            H_grid_interior_spreading_angles)
    npts_azimut, \
    central_first_cell, \
    central_last_cell, \
    tip_cell_length = _getCentralH_azimutpts_cell_sizes(external_surfaces)

    if DIRECTORY_CHECKME:
        t_check = J.tree(CENTRAL_H_6=external_surfaces)
        J.save(t_check, os.path.join(DIRECTORY_CHECKME,'6_central_H.cgns'))

    try: bulb_front, = [z for z in I.getZones(spinner) if z[0]=='hub.front']
    except: bulb_front = None
    try: bulb_rear, = [z for z in I.getZones(spinner) if z[0]=='hub.rear']
    except: bulb_rear = None
    front_surface, = [ z for z in external_surfaces if z[0] == 'front.surf' ]
    rear_surface, = [ z for z in external_surfaces if z[0] == 'rear.surf' ]

    if bulb_front:
        wires_front, surfs_front, grids_front = _buildHubWallAdjacentSector(
                            front_surface, spinner_front,
                            bulb_front, blade_number, rotation_center,
                            rotation_axis, central_first_cell, central_last_cell,
                            'front')
    else:

        if front_support:
            # NEW
            wires_front, surfs_front, grids_front = buildFrontMonoblockSector(blade_number,
                                    external_surfaces, spinner_front, front_support)


        else:
            # OLD
            spinner_front_jmin = GSD.getBoundary(spinner_front,'jmin')
            start_pt = W.point(spinner_front_jmin,-1)
            end_pt = W.point(spinner_front_jmin)
            front_surface = T.translate(front_surface,end_pt-start_pt)
            wires_front, surfs_front, grids_front = _buildHubWallAdjacentSectorWithoutBulb(
                                front_surface, spinner_front, blade_number, rotation_center,
                                rotation_axis, central_last_cell)

    if DIRECTORY_CHECKME:
        t_check = J.tree(WIRES_FRONT_7=wires_front, SURFS_FRONT_7=surfs_front)
        J.save(t_check, os.path.join(DIRECTORY_CHECKME,'7_front_near_topo.cgns'))

    if bulb_rear:
        wires_rear, surfs_rear, grids_rear = _buildHubWallAdjacentSector(
                                rear_surface, spinner_rear,
                                bulb_rear, blade_number, rotation_center,
                                rotation_axis, central_first_cell,
                                central_last_cell,'rear')
        I._renameNode(surfs_rear,'TFI2_spinner_1','TFI2_spinner_3')
        I._renameNode(surfs_rear,'TFI2_spinner_2','TFI2_spinner_4')
    else:
        if rear_support:
            # NEW
            wires_rear, surfs_rear, grids_rear = buildRearMonoblockSector(blade_number,
                                    external_surfaces, spinner_rear, rear_support)
        else:
            wires_rear, surfs_rear, grids_rear = _buildHubWallAdjacentSectorWithoutBulb(
                                    rear_surface, spinner_rear, blade_number, rotation_center,
                                    rotation_axis, central_last_cell)
            
    if DIRECTORY_CHECKME:
        t_check = J.tree(WIRES_REAR_8=wires_rear, SURFS_REAR_8=surfs_rear)
        J.save(t_check, os.path.join(DIRECTORY_CHECKME,'8_rear_near_topo.cgns'))


    Hgrids = _buildHgridAroundBlade(external_surfaces, blade, spinner_middle, rotation_center,
                   rotation_axis, H_grid_interior_points, relax_relative_length,
                   radial_H_compromise, radial_breakpoints=radial_breakpoints,
                   h_inner_normal_tension=h_inner_normal_tension,
                   h_outter_normal_tension=h_outter_normal_tension,
                   shift=shift,
                   DIR_CHECK=DIRECTORY_CHECKME)

    if DIRECTORY_CHECKME:
        t_check = J.tree(HGRID_FACES_9=P.exteriorFacesStructured(Hgrids))
        J.save(t_check, os.path.join(DIRECTORY_CHECKME,'9_Hgrid_faces.cgns'))

    profile = _extractWallAdjacentSectorFullProfile(wires_front, wires_rear,
                                                        external_surfaces)

    if DIRECTORY_CHECKME:
        t_check = J.tree(PROFILE_SUBPARTS_10=profile)
        J.save(t_check, os.path.join(DIRECTORY_CHECKME,'10_profile_subparts.cgns'))

    sector_bnds = _gatherSectorBoundaries(Hgrids, blade, surfs_rear, surfs_front)

    # print(f"{blade_number=}")
    # print(f"{npts_azimut=}")
    # print(f"{H_grid_interior_points=}")
    # print(f"{rotation_center=}")
    # print(f"{rotation_axis=}")
    # print(f"{distance=}")
    # print(f"{max_radius=}")
    # print(f"{number_of_points=}")
    # print(f"{farfield_cell_height=}")
    # print(f"{tip_axial_scaling_at_farfield=}")
    # print(f"{tip_cell_length=}")
    # print(f"{normal_tension=}")
    # print(f"{tip_radial_tension=}")
    # print(f"{FarfieldTipSmoothIterations=}")
    # print(f"{FarfieldProfileAbscissaDeltas=}")
    # J.save(J.tree(
    #     SECTOR_BNDS=sector_bnds,
    #     PROFILE=profile,
    #     REAR_SUPPORT=rear_support,
    # ),'debug_farfield_sector.cgns')

    fargrids, farfaces = _buildFarfieldSector(sector_bnds, profile,
                         blade_number, npts_azimut, H_grid_interior_points,
                         rotation_center, rotation_axis,
                         distance=distance, max_radius=max_radius,
                         number_of_points=number_of_points,
                         farfield_cell_height=farfield_cell_height,
                         tip_axial_scaling_at_farfield=tip_axial_scaling_at_farfield,
                         tip_cell_length=tip_cell_length,
                         normal_tension=normal_tension,
                         tip_radial_tension=tip_radial_tension,
                         FarfieldTipSmoothIterations=FarfieldTipSmoothIterations,
                         FarfieldProfileAbscissaDeltas=FarfieldProfileAbscissaDeltas,
                         front_support=front_support,
                         rear_support=rear_support,
                         DIRECTORY_CHECKME=DIRECTORY_CHECKME)

    if DIRECTORY_CHECKME:
        J.save(farfaces, os.path.join(DIRECTORY_CHECKME,'11_farfaces.cgns'))

    t = C.newPyTree(['Base',grids_front+Hgrids+grids_rear+fargrids+I.getZones(blade)])
    I._correctPyTree(t,level=3)
    base, = I.getBases(t)
    J.set(base,'.MeshingParameters',blade_number=blade_number,
               RightHandRuleRotation=RightHandRuleRotation, 
               rotation_axis=rotation_axis, rotation_center=rotation_center)

    if DIRECTORY_CHECKME:
        J.save(P.exteriorFacesStructured(t),
                     os.path.join(DIRECTORY_CHECKME,'12_final_grid_faces.cgns'))

    print(f'Total number of cells in mesh: {J.CYAN}{C.getNCells(t):,}{J.ENDC}')


    print('checking negative volume cells... ',end='')
    negative_volume_cells = GVD.checkNegativeVolumeCells(t, volume_threshold=0)
    if negative_volume_cells:
        J.save(negative_volume_cells, os.path.join(DIRECTORY_CHECKME,'negative_volume_cells_match_mesh.cgns'))
        if raise_error_if_negative_volume_cells:
            raise ValueError(J.FAIL+"stopped since detected negative cells in match mesh"+J.ENDC)
    else:
        print(J.GREEN+'ok'+J.ENDC)


    return t

def _splitSpinnerHgrid(spinner, blade, spinner_axial_indexing='i'):
    blade_main_surface = J.selectZoneWithHighestNumberOfPoints( blade )
    spinner_main_surface = J.selectZoneWithHighestNumberOfPoints( spinner )
    spinner_azimut_indexing = 'j' if spinner_axial_indexing == 'i' else 'i'
    ij = dict(i=1,j=2)
    NiNj = I.getZoneDim(spinner_main_surface)
    mid_layer = NiNj[ij[spinner_azimut_indexing]]//2
    spinner_profile = GSD.getBoundary(spinner_main_surface,
                                      spinner_azimut_indexing+'min',
                                      mid_layer)

    N_segs_azimut = NiNj[ij[spinner_azimut_indexing]] - 1
    N_segs_foil = I.getZoneDim( blade_main_surface )[1] - 1
    N_segs_axial = N_segs_foil//2 - N_segs_azimut

    blade_root = GSD.getBoundary( blade_main_surface, 'jmin')
    bary = np.array(G.barycenter( blade_root ))
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
    _,Ni,_,_,dim = I.getZoneDim( blade_main_surface )
    if dim == 3:
        external = GSD.getBoundary( blade_main_surface, 'kmax')
    else:
        external = blade_main_surface

    imin = GSD.getBoundary( external, 'imin')
    imid = GSD.getBoundary( external, 'imin', int(Ni/2))
    spine = imin if W.getLength(imin) > W.getLength(imid) else imid

    return spine

def _buildExternalSurfacesHgrid(blade, spinner_split, rotation_axis,
                                rotation_center, spreading=2.):
    c = np.array(rotation_center,dtype=np.float64)
    a = np.array(rotation_axis,dtype=np.float64)
    spine = _getSpineFromBlade( blade )
    distribution = D.getDistribution( spine )
    d_x = J.getx(distribution)
    W.addDistanceRespectToLine( spine , c, a, 'radius')
    r = J.getVars(spine,['radius'])[0]
    Length = r.max() - r.min()
    d_x[:] = (r - r.min())/Length

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
    
    # compute dir
    x_sideA = J.getx(sideA)
    dir = np.sign(x_sideA[1]-x_sideA[0])*np.sign(a[0])

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
            OL = OX + PX_v * Length + dir*spreading*a*(OP-bary)

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

    support = G.TFI(boundaries)
    outter_cell_size = W.segment(boundaries[-1], -1)

    # OLD:
    # c = np.array(rotation_center,dtype=np.float64)
    # a = np.array(rotation_axis,dtype=np.float64)
    # boundaries = W.reorderAndSortCurvesSequentially( boundaries )
    # alignment = [ a.dot( W.tangentExtremum( b ) ) for b in boundaries ]
    # best_aligned = np.argmax( alignment )
    # best_aligned_boundary = boundaries[ best_aligned ]
    # azimut_boundary = boundaries[ best_aligned - 1 ]
    # azimut_boundary = T.reorder(azimut_boundary, (-1,2,3))
    # tangent_start = W.tangentExtremum( azimut_boundary )
    # tangent_end = W.tangentExtremum(azimut_boundary, opposite_extremum=True)
    # sector_angle = np.rad2deg(np.arccos( tangent_start.dot( tangent_end ) ))
    # if sector_angle < 0: sector_angle += 180
    # sector_angle = 360 / float(int(np.round(360/sector_angle)))
    # proj_pt = W.point(azimut_boundary, as_pytree_point=True)
    # W.projectOnAxis(proj_pt, a, c)
    # start_to_proj = W.point(proj_pt) - W.point(azimut_boundary)
    # rotation_sign = np.sign( a.dot ( np.cross(tangent_start,start_to_proj)))
    # support = D.axisym(best_aligned_boundary,tuple(c),tuple(rotation_sign*a),
    #                    sector_angle, C.getNPts(azimut_boundary))
    # support[0] = 'support'

    # azimut_segment = np.mean([W.segment(azimut_boundary),
    #                           W.segment(azimut_boundary,-1)])
    # axial_segment = np.mean([W.segment(best_aligned_boundary),
    #                          W.segment(best_aligned_boundary,-1)])

    # outter_cell_size = axial_segment

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

def _buildHgridAroundBlade(external_surfaces, blade, hub, rotation_center,
                           rotation_axis, H_grid_interior_points, relax_relative_length,
                           radial_H_compromise, radial_breakpoints=[0.5],
                           h_inner_normal_tension=0.3, h_outter_normal_tension=0.3,
                           shift=0,
                           DIR_CHECK='CHECK_ME'):
    from scipy.interpolate import interp1d

    spine = _getSpineFromBlade( blade )
    wall_cell_height = W.distance( W.point( spine ), W.point( spine, 1 ) )
    W.addDistanceRespectToLine(spine, rotation_center, rotation_axis, 'span')
    span, = J.getVars(spine, ['span'])
    s = W.gets( spine )
    span_pts = len(s)

    i_compromise = int(np.round(np.interp(radial_H_compromise, s, np.arange(0,span_pts))))
    i_breakpoints = [int(i) for i in np.round(np.interp(radial_breakpoints, s, np.arange(0,span_pts)))]


    Tik = Tok()
    nbOfDigits = int(np.ceil(np.log10(span_pts+1)))
    LastLayer = ('{:0%d}'%nbOfDigits).format(span_pts)
    All_surfs = []

    bnds_to_stack = []
    walls_to_stack = []
    bnds_stack = []
    walls_stack = []
    blks_stack = []


    boundaries = [ GSD.getBoundary(e, 'imin', i_compromise) for e in external_surfaces ]
    inner_contour, inner_cell_size = _getInnerContour(blade,i_compromise)
    bnds_split, inner_split, ref_index = W.splitInnerContourFromOutterBoundariesTopology(
                                                boundaries, inner_contour)
    ref_index += shift

    for i in range( span_pts ):


        boundaries = [ GSD.getBoundary(e, 'imin', i) for e in external_surfaces ]
        inner_contour, inner_cell_size = _getInnerContour(blade,i)
        bnds_split, inner_split,_ = W.splitInnerContourFromOutterBoundariesTopology(
                                                    boundaries, inner_contour,ref_index)

        if i>0 and i<span_pts-1 and i not in i_breakpoints:
            for j in range( len(bnds_split) ):
                bnds_to_stack[j] += [bnds_split[j]]
                walls_to_stack[j] += [inner_split[j]]
            continue

        elif i==0:
            is_interior_point = False
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
            if i == span_pts-1:
                is_interior_point = False
            else:
                is_interior_point = True

            for j in range( len(bnds_split) ):
                bnds_to_stack[j] += [bnds_split[j]]
                bnds_stack[j] += [G.stack(bnds_to_stack[j])]
                walls_to_stack[j] += [inner_split[j]]
                walls_stack[j] += [G.stack(walls_to_stack[j])]
                bnds_to_stack[j] = [bnds_to_stack[j][-1]]
                walls_to_stack[j] = [walls_to_stack[j][-1]]

        currentLayer = ('{:0%d}'%nbOfDigits).format(i+1)
        print(J.MAGE+'H-grid boundaries'+J.ENDC+' %s/%s | '%(currentLayer,LastLayer), end='')
        Tik = Tok()

        support, outter_cell_size = _buildSupportFromBoundaries(boundaries,
                                                rotation_center, rotation_axis)
        if is_interior_point:
            inner_normal_tension, outter_normal_tension = 0, 0
            support = None
        else:
            if i==0:
                T._projectOrtho(support,hub)
            inner_normal_tension, outter_normal_tension = h_inner_normal_tension, h_outter_normal_tension
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
                          inner_normal_tension=inner_normal_tension,
                          outter_normal_tension=outter_normal_tension,
                          projection_support=support,
                          global_projection_relaxation=0,
                          local_projection_relaxation_length=local_relax_length,
                          forced_split_index=ref_index,
                          debug=True if currentLayer == LastLayer else False)
        T._reorder(surfs, (2,1,3))

        if i>0:
            fixedZones = I.getZones(boundaries)+I.getZones(inner_contour)
            GSD.prepareGlue(fixedZones,surfs)
            T._smooth(surfs,eps=0.8, type=0,
                niter=20, # FIXME niter make param            
                fixedConstraints=fixedZones)            
            GSD.applyGlue(surfs,fixedZones)

        # J.save(J.tree(**{"SURFS_%d"%i:surfs,"BNDS_%d"%i:boundaries}),'surfs_%d.cgns'%i)
        All_surfs += [ surfs ]
        print('cost: %0.5f s'%(Tok()-Tik))
    Tik = Tok()
    nb_stacks = len(All_surfs)
    nb_dgs = int(np.ceil(np.log10(nb_stacks)))
    LastLayer = ('{:0%d}'%nb_dgs).format(nb_stacks-1)
    for i in range( nb_stacks - 1):
        surfs = All_surfs[i]
        next_surfs = All_surfs[i+1]
        currentLayer = ('{:0%d}'%nb_dgs).format(i+1)
        print(J.CYAN+'H-grid TFI'+J.ENDC+' %s/%s | '%(currentLayer,LastLayer), end='')
        Tik = Tok()

        for j in range( len(bnds_split) ):
            wall_curve = GSD.getBoundary(walls_stack[j][i],'imin')
            wall_curve[0] = 'wall_curve'
            bnd_curve = GSD.getBoundary(bnds_stack[j][i],'imin')
            bnd_curve[0] = 'bnd_curve'
            surfs_curve = GSD.getBoundary(surfs[j],'imin')
            surfs_curve[0] = 'surfs_curve'
            next_surfs_curve = GSD.getBoundary(next_surfs[j],'imin')
            next_surfs_curve[0] = 'next_surfs_curve'
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
        print('cost: %0.5f s'%(Tok()-Tik))

    print('joining H-grids TFI... ',end='')
    grids = []
    for j in range( len(bnds_split) ):
        nb_blks = len(blks_stack[j])
        blk = blks_stack[j][0]
        for blk2 in blks_stack[j][1:]:
            blk = T.join(blk,blk2)
        blk[0] = blks_stack[j][0][0]
        if nb_blks > 1: T._reorder(blk,(3,1,2))
        grids += [ blk ]
    print(J.GREEN+'ok'+J.ENDC)

    for grid in grids:
        C._addBC2Zone(grid,'spinner_wall','FamilySpecified:SPINNER', 'kmin')

    print('checking negative volume cells in H-grid regions... ',end='')
    negative_volume_cells = GVD.checkNegativeVolumeCells(grids, volume_threshold=0)
    if negative_volume_cells:
        print(J.WARN)
        if DIR_CHECK:
            try: os.makedirs(DIR_CHECK)
            except: pass
            J.save(grids, os.path.join(DIR_CHECK,'H-grids.cgns'))
            J.save(negative_volume_cells, os.path.join(DIR_CHECK,'negative_volume_cells.cgns'))
        print(J.ENDC)
    else:
        print(J.GREEN+'ok'+J.ENDC)

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

    def makeSemiBinormalUnion(curve1, curve2, NPts, tension=0.4, reverse=False,
            first_cell_length=None, last_cell_length=None):
        ext1 = W.extremum(curve1, True)
        ext2 = W.extremum(curve2, True)
        s1 = getBinormal(curve1, not reverse)
        d = W.distance(ext1,ext2)
        poly = D.polyline([tuple(ext1), tuple(ext1+d*tension*s1), tuple(ext2)])
        union = D.bezier(poly,N=NPts)
        union = W.discretize(union,N=NPts, Distribution=dict(kind='tanhTwoSides',
            FirstCellHeight=last_cell_height if first_cell_length is None else first_cell_length,
            LastCellHeight=last_cell_height if last_cell_length is None else last_cell_length))
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
        bulb_side_0 = GSD.getBoundary(bulb,'jmin')
        bulb_side_0[0] = 'bulb_side_0'
        bulb_side_1 = GSD.getBoundary(bulb,'imin')
        bulb_side_1[0] = 'bulb_side_1'
        spinner_bulb_side_0 = GSD.getBoundary(bulb,'jmax')
        spinner_bulb_side_0[0] = 'spinner_bulb_side_0'
        spinner_bulb_side_1 = GSD.getBoundary(bulb,'imax')
        spinner_bulb_side_1[0] = 'spinner_bulb_side_1'

    elif topo == 'rear':
        bulb_side_0 = GSD.getBoundary(bulb,'jmin')
        bulb_side_0[0] = 'bulb_side_0'
        bulb_side_1 = GSD.getBoundary(bulb,'imin')
        bulb_side_1[0] = 'bulb_side_1'
        spinner_bulb_side_0 = GSD.getBoundary(bulb,'jmax')
        spinner_bulb_side_0[0] = 'spinner_bulb_side_0'
        spinner_bulb_side_1 = GSD.getBoundary(bulb,'imax')
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
    W.reverse(bulb_union_0,True)
    bulb_union_0[0] = 'bulb_union_0'
    length_bulb_union_0 = W.getLength(bulb_union_0)
    length_bulb_side_0 = W.getLength(bulb_side_0)
    thales_bulb = length_bulb_union_0/length_bulb_side_0
    l1 = W.distance(W.point(bulb_side_0,-2),W.point(bulb_side_0,-1))
    cell_top = np.minimum(l1*thales_bulb,last_cell_height)
    bulb_union_0 = makeSemiBinormalUnion(line_0, axis_line, C.getNPts(bulb_side_0), 
                        first_cell_length=last_cell_height,
                        last_cell_length=cell_top, reverse=reverse)
    W.reverse(bulb_union_0,True)
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
    W.reverse(spinner_union_0,True)
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
    L_bulb_union_0 = W.getLength(bulb_union_0)
    L_proj_half = W.getLength(proj_half)

    L_diag=np.minimum(L_bulb_union_0*1.25, (L_proj_half-L_bulb_union_0)*0.25+L_bulb_union_0)

    spinner_union_1 = W.splitAt(proj_half,L_diag,'length')[1]
    spinner_union_1[0] = 'spinner_union_1'
    W.reverse(spinner_union_1,True)


    azimuthal_relative_tension = 0.1
    ext_union_azm_0 = W.linelaw(W.point(line_0,-1), W.point(spinner_union_1,-1),
        N=C.getNPts(spinner_bulb_side_1))
    ext_union_azm_0[0] = 'ext_union_azm_0'
    unif_azm_seg = W.getLength(ext_union_azm_0)/(C.getNPts(ext_union_azm_0)-1)
    azm_cell_segment = np.minimum(last_cell_height, unif_azm_seg)
    T._projectOrtho(ext_union_azm_0,proj_support)
    W.discretizeInPlace(ext_union_azm_0,Distribution=dict(kind='tanhTwoSides',
        FirstCellHeight=azm_cell_segment, LastCellHeight=azm_cell_segment))
    T._projectOrtho(ext_union_azm_0,proj_support)

    ext_union_azm_1 = W.linelaw(W.point(line_2,-1), W.point(spinner_union_1,-1),
        N=C.getNPts(spinner_bulb_side_1), Distribution=dict(kind='tanhTwoSides',
        FirstCellHeight=azm_cell_segment, LastCellHeight=azm_cell_segment))
    ext_union_azm_1[0] = 'ext_union_azm_1'
    T._projectOrtho(ext_union_azm_1,proj_support)
    W.discretizeInPlace(ext_union_azm_1,Distribution=dict(kind='tanhTwoSides',
        FirstCellHeight=azm_cell_segment, LastCellHeight=azm_cell_segment))
    T._projectOrtho(ext_union_azm_1,proj_support)


    spinner_union_1 = W.discretize(spinner_union_1, C.getNPts(spinner_wall_edge_half),
    Distribution = dict(
        kind='tanhTwoSides',
        FirstCellHeight=central_cell,
        LastCellHeight=azm_cell_segment))



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
              spinner_wall_edge_half, surf_edge_half, ext_surf_edge ]


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


    if topo == 'rear':
        TFI2_inter_join_1_wires = [ line_0, axis_line,
                                    bulb_side_1, bulb_union_0]
    else:
        TFI2_inter_join_1_wires = [ line_0, axis_line,
                                    bulb_side_0, bulb_union_0]
    TFI2_inter_join_1 = G.TFI(TFI2_inter_join_1_wires)
    TFI2_inter_join_1[0] = 'TFI2_inter_join_1'
    TFI2_inter_join_2 = T.rotate(TFI2_inter_join_1,tuple(c),tuple(a),
                                 360./float(blade_number))
    TFI2_inter_join_2[0] = 'TFI2_inter_join_2'

    if topo == 'rear':
        TFI2_bulb_0_wires = [line_0, line_1,
                             spinner_bulb_side_0, ext_union_azm_0]
    else:
        TFI2_bulb_0_wires = [line_0, line_1,
                             spinner_bulb_side_1, ext_union_azm_0]

    TFI2_bulb_0 = G.TFI(TFI2_bulb_0_wires)
    TFI2_bulb_0[0] = 'TFI2_bulb_0'

    if topo == 'rear':
        TFI2_bulb_1_wires =[line_1, line_2,
                            spinner_bulb_side_1, ext_union_azm_1] 
    else:

        TFI2_bulb_1_wires =[line_1, line_2,
                            spinner_bulb_side_0, ext_union_azm_1]

    TFI2_bulb_1 = G.TFI(TFI2_bulb_1_wires)
    TFI2_bulb_1[0] = 'TFI2_bulb_1'

    TFI2_bulb = G.TFI([bulb_union_0, ext_union_azm_1,
                       bulb_union_2, ext_union_azm_0])
    T._reorder(TFI2_bulb,(1,-2,3))
    TFI2_bulb[0] = 'TFI2_bulb'

    TFI2_spinner_1 = G.TFI([spinner_union_0, spinner_union_1,
                            ext_union_azm_0, ext_edge_surface_inter_0])
    T._reorder(TFI2_spinner_1,(1,-2,3))
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
    print('making near-blade 3D TFI at %s... '%topo,end='')
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
    print(J.GREEN+'ok'+J.ENDC)

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
    Length = D.getLength(surface_dist)

    ext_surf_edge  = GSD.getBoundary(surface,'imax')
    ext_surf_edge[0] = 'ext_surf_edge'

    spinner_axial_edge = GSD.getBoundary(spinner,'jmin')
    spinner_axial_edge[0] = 'spinner_axial_edge'

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

    T._reorder([spinner_union_0, spinner_union_1],(-1,2,3))

    pt1 = W.point(rear_edge,-1)

    channel_angle = 360/blade_number
    n_pts_azim = C.getNPts(ext_surf_edge)
    R = W.distanceOfPointToLine(pt1,a,c)
    TE_azm = D.circle((0,0,0),R,-channel_angle/2,channel_angle/2,n_pts_azim)
    T._rotate(TE_azm,(0,0,0),((1,0,0),(0,1,0),(0,0,1)),
                             ((0,1,0),(0,0,-1),(1,0,0)))
    T._translate(TE_azm,(pt1[0], 0,0))
    TE_azm[0] = 'TE_azm'
    wires = [spinner_edge,ext_surf_edge,rear_edge, rear_edge_2, surface_dist,
            spinner_axial_edge,spinner_union_0,spinner_union_1, TE_azm]

    TFI2_spinner = G.TFI([spinner_union_0, spinner_union_1,
                          ext_surf_edge, TE_azm])
    TFI2_spinner[0] = 'TFI2_spinner'

    TFI2_spinner_1, TFI2_spinner_2 = T.splitNParts(TFI2_spinner,2,dirs=[1])
    TFI2_spinner_1[0] = 'TFI2_spinner_1'
    TFI2_spinner_2[0] = 'TFI2_spinner_2'
    TFI2_TE = G.TFI([rear_edge, rear_edge_2, TE_azm, spinner_edge])
    TFI2_TE[0] = 'TFI2_TE'


    surfs = [spinner,surface, TFI2_inter_side_1, TFI2_inter_side_2, TE_azm,
            TFI2_spinner_1,TFI2_spinner_2, TFI2_TE]

    print('making open near-blade 3D TFI at rear... ', end='')
    TFI3_spinner = G.TFI([TFI2_TE, surface,
                          TFI2_inter_side_2, TFI2_inter_side_1,
                          spinner, TFI2_spinner])
    TFI3_spinner[0] = 'TFI3_spinner'
    T._reorder(TFI3_spinner,(2,1,3))
    print(J.GREEN+'ok'+J.ENDC)
    C._addBC2Zone(TFI3_spinner,'spinner_wall','FamilySpecified:SPINNER', 'kmin')
    grids = [ TFI3_spinner ]

    return wires, surfs, grids

def _gatherSectorBoundaries(Hgrids, blade, surfs_rear, surfs_front):
    selected_names = ['TFI2_spinner_0','TFI2_spinner_1','TFI2_spinner_2','TFI2_spinner_3',
                      'TFI2_spinner_4','TFI2_bulb']
    surfaces = [GSD.getBoundary(z,'kmax') for z in Hgrids]
    surfs_rear = T.reorder(surfs_rear,(1,-2,3))
    surfaces.extend([s for s in surfs_front+surfs_rear if s[0] in selected_names])
    for i,z in enumerate(J.selectZonesExceptThatWithHighestNumberOfPoints(blade)):
        surf = GSD.getBoundary(z,'kmax')
        surf[0] = 'tip.%d'%i
        surfaces += [surf]

    _putSmoothedNormalsAtSurfaces(surfaces, eps=0.9, niter=100, mode=0)

    return surfaces

def _putSmoothedNormalsAtSurfaces(surfaces, eps=0.9, niter=100, mode=0):
    uns = C.convertArray2Hexa(surfaces)
    uns = T.merge(uns)
    G._getNormalMap(uns)
    for s in I.getZones(uns):
        C.center2Node__(s,'centers:sx',cellNType=0)
        C.center2Node__(s,'centers:sy',cellNType=0)
        C.center2Node__(s,'centers:sz',cellNType=0)
    I._rmNodesByName(uns,'FlowSolution#Centers')
    C._normalize(uns, ['sx','sy','sz'])
    T._smoothField(uns, eps, niter, mode, ['sx','sy','sz']) # TODO externalize param?
    C._normalize(uns, ['sx','sy','sz'])
    migrateFields(uns, surfaces)

def _extractWallAdjacentSectorFullProfile(wires_front, wires_rear, external_surfaces):
    profile_curves =      [c for c in wires_front if c[0]=='bulb_union_0']
    profile_curves.extend([c for c in wires_front if c[0]=='spinner_union_0'])
    profile_curves.extend([GSD.getBoundary(s,'imax') for s in external_surfaces if s[0]=='sideA.surf'])
    profile_curves.extend([c for c in wires_rear  if c[0]=='spinner_union_0'])
    profile_curves.extend([c for c in wires_rear  if c[0]=='bulb_union_0'])
    I._correctPyTree(profile_curves,level=3)
    profile_curves = W.reorderAndSortCurvesSequentially(profile_curves)

    curves_touching_profile = W.getCurvesInContact(profile_curves,wires_front+wires_rear)
    radial_curves = [c for c in curves_touching_profile if 'ext' not in c[0] \
                                                     and 'union' not in c[0] \
                                                    and 'TE_azm' not in c[0]]
    W.transferTouchingSegmentsAndDirections(profile_curves, radial_curves)
    GSD._alignNormalsWithRadialCylindricProjection(profile_curves)

    return profile_curves

def _buildFarfieldProfile(profile, blade_number, distance, max_radius=np.inf,
                          rotation_center=[0,0,0], rotation_axis=[-1,0,0],
                          FarfieldProfileAbscissaDeltas=[],
                          front_support=None, rear_support=None):
    c = np.array(rotation_center,dtype=np.float64)
    a = np.array(rotation_axis,dtype=np.float64)

    leading_edge_distance_to_axis = W.distanceOfPointToLine(W.extremum(profile[0]),a,c)
    HAS_FRONT_BULB = True if leading_edge_distance_to_axis < 1e-6 else False

    trailing_edge_distance_to_axis = W.distanceOfPointToLine(W.extremum(profile[-1],True),a,c)
    HAS_REAR_BULB = True if trailing_edge_distance_to_axis < 1e-6 else False

    profile_joined = profile[0]
    for p in profile[1:]: profile_joined = T.join(profile_joined,p)
    profile_joined[0] = 'profile_joined'

    profile_farfield = I.copyTree(profile_joined)
    profile_farfield[0] = 'profile_farfield'

    W.addNormals(profile_farfield)    
    
    GSD._alignNormalsWithRadialCylindricProjection(profile_farfield, c, a)
    W.forceVectorPointOutwards(profile_farfield)

    x,y,z = J.getxyz(profile_farfield)
    sx,sy,sz = J.getVars(profile_farfield,['sx','sy','sz'])
    if W.distanceOfPointToLine(W.extremum(profile_farfield),a,c) < 1e-10:
        sx[0], sy[0], sz[0] = a
    if W.distanceOfPointToLine(W.extremum(profile_farfield,True),a,c) < 1e-10:
        sx[-1], sy[-1], sz[-1] = -a

    x += distance * sx
    y += distance * sy
    z += distance * sz

    profile_rev = I.copyTree(profile_farfield)
    profile_rev[0] = 'profile_rev'
    all_valid = False
    print('optimizing farfield revolution profile... ',end='')
    identical_invalid_count = 0
    previous_count = 0
    smoothing_iteration = 0
    tx, ty, tz = getTangent(profile_rev)
    while not all_valid:
        smoothing_iteration += 1
        x,y,z = J.getxyz(profile_rev)
        
        i_xmin = np.argmin(x)
        i_xmax = np.argmax(x)
        x = x[i_xmin:i_xmax]
        y = y[i_xmin:i_xmax]
        z = z[i_xmin:i_xmax]

        # we must be on OXY plane
        z *= 0 

        valid = tx[i_xmin:i_xmax] > 0 
        

        valid[i_xmin] = True
        valid[i_xmax-1] = True
        all_valid = np.all(valid)
        new_count = np.count_nonzero(valid)
        if new_count == previous_count: identical_invalid_count += 1
        previous_count = new_count
        x = x[valid]
        y = y[valid]
        z = z[valid]

        if np.isfinite(max_radius):
            y[:] = np.minimum(max_radius, y)

        if front_support:
            y[0] = max_radius
        
        if rear_support:
            y[-1] = max_radius

        y[-2] = np.maximum(y[-1], y[-2])

        if HAS_FRONT_BULB: y[0] = 0
        
        if HAS_REAR_BULB: y[-1] = 0

        profile_rev = J.createZone('profile_rev',[x,y,z],['x','y','z'])
        profile_rev = W.discretize(profile_rev,N=10)
        tx, ty, tz = getTangent(profile_rev)
        all_valid = np.all(tx >= 0)

        is_stuck = identical_invalid_count >=500
        if is_stuck:
            msg = '\nWARNING: could not optimize rev. profile properly\n'
            msg+= f'kept getting {new_count} invalid tangents after {identical_invalid_count} trys\n'
            msg+= '-> check debug_rev_profile.cgns for reversed segments'
            print(J.WARN+msg+J.ENDC)
            J.save(profile_rev, 'debug_rev_profile.cgns')
            break

    if not is_stuck: print(J.GREEN+'ok'+J.ENDC)

    y = J.gety(profile_rev)
    y[:] = np.minimum(max_radius, y)


    if rear_support:
        profile_rev = W.adjustUpToGeometry(profile_rev, rear_support, direction=[1,0,0])
        y = J.gety(profile_rev)
        if np.isfinite(max_radius): y[-5:]=max_radius

    if front_support:
        W.reverse(profile_rev,True)
        profile_rev = W.adjustUpToGeometry(profile_rev, front_support, direction=[-1,0,0])
        W.reverse(profile_rev,True)
        y = J.gety(profile_rev)
        if np.isfinite(max_radius): y[:4]=max_radius

    T._rotate(profile_rev,(0,0,0),(1,0,0),0.5*(360/blade_number))

    # required to perfectly match interface, clipped at max_radius and after rotation
    if rear_support:
        profile_rev = W.adjustUpToGeometry(profile_rev, rear_support, direction=[1,0,0])

    if front_support:
        W.reverse(profile_rev,True)
        profile_rev = W.adjustUpToGeometry(profile_rev, front_support, direction=[-1,0,0])
        W.reverse(profile_rev,True)

    
    W.discretizeInPlace(profile_rev, N=3000)

    profile_rev_sectors = W.splitAndDiscretizeCurveAsProvidedReferenceCurves(profile_rev,
        profile, FarfieldProfileAbscissaDeltas)
    
    for p, pr in zip(profile, profile_rev_sectors): pr[0] = p[0]+'.far'

    profile_rev = W.joinSequentially(profile_rev_sectors)

    return profile_rev, profile_rev_sectors


def _buildFarfieldCapsule(profile, blade_number, distance, max_radius=np.inf,
                          FarfieldProfileAbscissaDeltas=[],
                          front_support=None, rear_support=None):
    rmax_profile = getMaxRadius(profile)
    xmin = C.getMinValue(profile,'CoordinateX')
    xmax = C.getMaxValue(profile,'CoordinateX')
    capsule_radius = distance+rmax_profile
    if np.isfinite(max_radius): capsule_radius = np.minimum(capsule_radius,max_radius)

    leading_edge_distance_to_axis = W.distanceOfPointToLine(W.extremum(profile[0]),[1,0,0],[0,0,0])
    HAS_FRONT_BULB = True if leading_edge_distance_to_axis < 1e-6 else False

    trailing_edge_distance_to_axis = W.distanceOfPointToLine(W.extremum(profile[-1],True),[1,0,0],[0,0,0])
    HAS_REAR_BULB = True if trailing_edge_distance_to_axis < 1e-6 else False

    if HAS_FRONT_BULB:
        profile_rev = front_arc = D.circle((xmin+rmax_profile,0,0),capsule_radius,90.0,180.0,90)
        W.reverse(front_arc,True)

    if HAS_REAR_BULB:
        profile_rev = rear_arc = D.circle((xmax-rmax_profile,0,0),capsule_radius,0.0,90.0,90)
        W.reverse(rear_arc,True)

    if HAS_FRONT_BULB and HAS_REAR_BULB:
        profile_rev = W.concatenate([front_arc, rear_arc])
    
    if not HAS_FRONT_BULB and not HAS_REAR_BULB:
        profile_rev = D.line((xmin,capsule_radius,0),
                             (xmax,capsule_radius,0),2)
        
    if rear_support:
        profile_rev = W.adjustUpToGeometry(profile_rev, rear_support, direction=[1,0,0])
        y = J.gety(profile_rev)
        if np.isfinite(max_radius): y[-5:]=max_radius

    if front_support:
        W.reverse(profile_rev,True)
        profile_rev = W.adjustUpToGeometry(profile_rev, front_support, direction=[-1,0,0])
        W.reverse(profile_rev,True)
        y = J.gety(profile_rev)
        if np.isfinite(max_radius): y[:4]=max_radius

    T._rotate(profile_rev,(0,0,0),(1,0,0),0.5*(360/blade_number))

    # required to perfectly match interface, clipped at max_radius and after rotation
    if rear_support:
        profile_rev = W.adjustUpToGeometry(profile_rev, rear_support, direction=[1,0,0])

    if front_support:
        W.reverse(profile_rev,True)
        profile_rev = W.adjustUpToGeometry(profile_rev, front_support, direction=[-1,0,0])
        W.reverse(profile_rev,True)

    
    W.discretizeInPlace(profile_rev, N=3000)

    profile_rev_sectors = W.splitAndDiscretizeCurveAsProvidedReferenceCurves(profile_rev,
        profile, FarfieldProfileAbscissaDeltas)
    
    for p, pr in zip(profile, profile_rev_sectors): pr[0] = p[0]+'.far'

    profile_rev = W.joinSequentially(profile_rev_sectors)

    return profile_rev, profile_rev_sectors



def _extractBladeTipTopology(sector_bnds, rotation_axis):
    a = np.array(rotation_axis,dtype=np.float64)

    tip_curves = [GSD.getBoundary(s,'jmin') for s in sector_bnds if s[0].startswith('tfi')]

    # compute mean oscullatory plane (binormal)
    contours = W.reorderAndSortCurvesSequentially(tip_curves)
    for s in sector_bnds:
        if s[0].startswith('tip'):
            bnds = P.exteriorFacesStructured(s)
            for i,b in enumerate(bnds):
                b[0] = s[0]+'.%d'%i
            tip_curves.extend(bnds)
    
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

def _buildFarfieldSector(sector_bnds, profile, blade_number, npts_azimut,
                         H_grid_interior_points,
                         rotation_center=[0,0,0], rotation_axis=[-1,0,0],
                         distance=10.0, max_radius=np.inf, number_of_points=200,
                         farfield_cell_height=1.,
                         tip_axial_scaling_at_farfield = 0.5,
                         tip_cell_length=None,
                         normal_tension=0.05,
                         tip_radial_tension=0.03,
                         FarfieldTipSmoothIterations=500,
                         FarfieldProfileAbscissaDeltas=[],
                         front_support=None,
                         rear_support=None,
                         DIRECTORY_CHECKME="CHECK_ME"
                         ):

    c = np.array(rotation_center,dtype=np.float64)
    a = np.array(rotation_axis,dtype=np.float64)


    leading_edge_distance_to_axis = W.distanceOfPointToLine(W.extremum(profile[0]),a,c)
    HAS_FRONT_BULB = True if leading_edge_distance_to_axis < 1e-6 else False

    trailing_edge_distance_to_axis = W.distanceOfPointToLine(W.extremum(profile[-1],True),a,c)
    HAS_REAR_BULB = True if trailing_edge_distance_to_axis < 1e-6 else False

    element_index = 2 if HAS_FRONT_BULB else 1

    n_parts_profile = len(profile)
    I._correctPyTree(profile,level=3)


    if HAS_FRONT_BULB:
        if HAS_REAR_BULB and n_parts_profile != 5: raise ValueError(f'got {n_parts_profile} with front bulb')
        elif not HAS_REAR_BULB and n_parts_profile != 4: raise ValueError(f'got {n_parts_profile} with front bulb')
    else:
        if HAS_REAR_BULB and n_parts_profile != 4: raise ValueError(f'got {n_parts_profile} without front bulb')
        elif not HAS_REAR_BULB and n_parts_profile != 3: raise ValueError(f'got {n_parts_profile} without front bulb')




    # profile_rev, profile_rev_sectors = _buildFarfieldProfile(profile, blade_number,
    #     distance, max_radius, rotation_center, rotation_axis, FarfieldProfileAbscissaDeltas,
    #     front_support, rear_support)

    profile_rev, profile_rev_sectors = _buildFarfieldCapsule(profile, blade_number,
        distance, max_radius, FarfieldProfileAbscissaDeltas,
        front_support, rear_support)


    support = D.axisym(profile_rev,tuple(c),tuple(a),
            angle=360./float(blade_number), Ntheta=npts_azimut)
    support[0] = 'proj_support'

    _,_,Nj,_,_=I.getZoneDim(support)

    if HAS_FRONT_BULB:
        imin_support_central = C.getNCells(profile[:2])+1
        imax_support_central = imin_support_central + C.getNCells(profile[2])
    else:
        imin_support_central = C.getNPts(profile[0])
        imax_support_central = imin_support_central + C.getNCells(profile[1])

    support_central = T.subzone(support, (imin_support_central,1,1),
                                         (imax_support_central,Nj,1) )
    
    profile_sideB = T.rotate(profile,tuple(c),tuple(a),360./float(blade_number))

    profile_rev_sectors_sideB = T.rotate(profile_rev_sectors,tuple(c),tuple(a),
                                         360./float(blade_number))
    for p in profile_rev_sectors_sideB: p[0] += '.B'

    middle_index = int((npts_azimut-1)/2)

    if HAS_FRONT_BULB or HAS_REAR_BULB:
        support_half_edge = GSD.getBoundary(support,'jmin',middle_index)
        support_half_edge[0] = 'support_half_edge'

        s = W.gets(support_half_edge)
        s *= D.getLength(support_half_edge)
        L_diag = D.getLength(profile_rev_sectors[0]) * 1.25
        diag_cut_index = np.argmin( np.abs(s - L_diag) )
        start_index_of_third_profile_rev_sector = C.getNPts(profile_rev_sectors[0])+C.getNPts(profile_rev_sectors[1])-1

        far_union_1 = T.subzone(support_half_edge,(diag_cut_index,1,1),
                                (start_index_of_third_profile_rev_sector,1,1))

        LastCell = W.segment(profile_rev_sectors[1],-1)

        # azimutal
        to_split = GSD.getBoundary(support,'imin',
            C.getNPts(profile_rev_sectors[0])-1+C.getNPts(profile_rev_sectors[1])-1)
        H_azm_0 = T.subzone(to_split, (1,1,1),(middle_index+1,1,1))
        H_azm_0[0] = 'H_azm_0'
        cell_azimut = W.segment(H_azm_0)
        
        far_union_1 = W.discretize(far_union_1,N=C.getNPts(profile_rev_sectors[1]),
            Distribution=dict(kind='tanhTwoSides',
                            FirstCellHeight=cell_azimut,LastCellHeight=LastCell))
        far_union_1[0] = 'far_union_1'


        H_azm_1 = T.subzone(to_split, (middle_index+1,1,1),(-1,-1,-1))
        H_azm_1[0] = 'H_azm_1'

        if HAS_FRONT_BULB:
            to_split = GSD.getBoundary(support,'imin',
                C.getNPts(profile_rev_sectors[0])-1 + \
                C.getNPts(profile_rev_sectors[1])-1 +
                C.getNPts(profile_rev_sectors[2])-1)
        else:
            to_split = GSD.getBoundary(support,'imin',
                C.getNPts(profile_rev_sectors[0])-1 + \
                C.getNPts(profile_rev_sectors[1])-1)

        H_azm_low_0 = T.subzone(to_split, (1,1,1),(middle_index+1,1,1))
        H_azm_low_0[0] = 'H_azm_low_0'
        H_azm_low_1 = T.subzone(to_split, (middle_index+1,1,1),(-1,-1,-1))
        H_azm_low_1[0] = 'H_azm_low_1'
        H_azm_low = T.join(H_azm_low_0,H_azm_low_1)
        H_azm_low[0] = 'H_azm_low'


    if HAS_FRONT_BULB:
        # union of bulb with projection on support
        # first side
        ext_1 = W.extremum(profile_rev_sectors[0],True)
        ext_2 = W.extremum(far_union_1)
        t = W.tangentExtremum(profile_rev_sectors[0],True)
        v_az = np.cross(a,t)
        v_az /= np.sqrt(v_az.dot(v_az))
        d = W.distance(ext_1, ext_2)
        tension = 0.1
        poly = D.polyline([tuple(ext_1), tuple(ext_1+d*tension*v_az),tuple(ext_2)])
        T._projectOrtho(poly, support)
        bezier = D.bezier(poly,N=C.getNPts(profile_rev_sectors[0]))
        T._projectOrtho(bezier, support)
        
        center_cell_seg = np.minimum(cell_azimut,W.getLength(bezier)/C.getNCells(bezier))
        far_bulb_union_0 = W.discretize(bezier,N=C.getNPts(bezier),
            Distribution=dict(kind='tanhTwoSides',
                FirstCellHeight=center_cell_seg,LastCellHeight=center_cell_seg))
        far_bulb_union_0[0] = 'far_bulb_union_0'
        T._projectOrtho(far_bulb_union_0, support)

        W.discretizeInPlace(far_union_1,Distribution=dict(kind='tanhTwoSides',
            FirstCellHeight=center_cell_seg, LastCellHeight=W.segment(far_union_1,-1)))

        # second side
        ext_1 = W.extremum(profile_rev_sectors_sideB[0],True)
        ext_2 = W.extremum(far_union_1)
        t = W.tangentExtremum(profile_rev_sectors_sideB[0],True)
        v_az = -np.cross(a,t)
        v_az /= np.sqrt(v_az.dot(v_az))
        d = W.distance(ext_1, ext_2)
        tension = 0.1
        poly = D.polyline([tuple(ext_1), tuple(ext_1+d*tension*v_az),tuple(ext_2)])
        T._projectOrtho(poly, support)
        bezier = D.bezier(poly,N=C.getNPts(profile_rev_sectors[0]))
        T._projectOrtho(bezier, support)


        far_bulb_union_1 = W.discretize(bezier,N=C.getNPts(bezier),
            Distribution=dict(kind='tanhTwoSides',
                FirstCellHeight=center_cell_seg,LastCellHeight=center_cell_seg))
        far_bulb_union_1[0] = 'far_bulb_union_1'
        T._projectOrtho(far_bulb_union_1, support)

        wires = [profile_rev_sectors[1],far_union_1,
                far_bulb_union_0, H_azm_0]
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

        proj_surfs = [main_front_tfi, far_bulb_tfi]
        fixed_bnds = [P.exteriorFaces(surf) for surf in proj_surfs ]
        GSD.prepareGlue(proj_surfs, fixed_bnds)
        T._projectOrtho(proj_surfs, support)
        GSD.applyGlue(proj_surfs, fixed_bnds)
    else:
        Nj_support = I.getZoneDim(support)[2]
        main_front_tfi = T.subzone(support,(1,1,1),(C.getNPts(profile[0]),Nj_support,1))
        T._reorder(main_front_tfi,(2,1,3))
        main_front_tfi[0] = 'main_front_tfi'


    if HAS_REAR_BULB:
        reversed_support_half_edge = T.reorder(support_half_edge,(-1,2,3))
        s = W.gets(reversed_support_half_edge)
        s *= D.getLength(reversed_support_half_edge)
        L_diag = D.getLength(profile_rev_sectors[-1]) * np.sqrt(2)
        diag_cut_index = np.argmin( np.abs(s - L_diag) )
        far_union_2 = T.subzone(reversed_support_half_edge,(diag_cut_index,1,1),
          (C.getNPts(profile_rev_sectors[-1])+C.getNPts(profile_rev_sectors[-2])-1,1,1))
        cell_middle_front = W.segment(profile_rev_sectors[-2])
        cell_middle_rear = W.segment(profile_rev_sectors[-2],-1)
        far_union_2 = W.discretize(far_union_2,N=C.getNPts(profile_rev_sectors[-2]),
            Distribution=dict(kind='tanhTwoSides',
                LastCellHeight=cell_middle_front,
                FirstCellHeight=cell_middle_rear))
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
        bezier = D.bezier(poly,N=C.getNPts(profile_rev_sectors[-1]))
        T._projectOrtho(bezier, support)
        far_rear_bulb_union_0 = W.discretize(bezier,N=C.getNPts(bezier),
            Distribution=dict(kind='tanhTwoSides',
                FirstCellHeight=cell_azimut,
                LastCellHeight=cell_middle_rear))
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
        bezier = D.bezier(poly,N=C.getNPts(profile_rev_sectors[-1]))
        T._projectOrtho(bezier, support)
        far_rear_bulb_union_1 = W.discretize(bezier,N=C.getNPts(bezier),
            Distribution=dict(kind='tanhTwoSides',
                FirstCellHeight=cell_azimut,
                LastCellHeight=cell_middle_rear))
        far_rear_bulb_union_1[0] = 'far_rear_bulb_union_1'
        T._projectOrtho(far_rear_bulb_union_1, support)

        far_rear_bulb_union = T.join(far_rear_bulb_union_0,far_rear_bulb_union_1)



        main_rear_tfi_0 = G.TFI([profile_rev_sectors[element_index+1], far_union_2,
                                   H_azm_low_0,far_rear_bulb_union_0])




        main_rear_tfi_1 = G.TFI([far_union_2,profile_rev_sectors_sideB[element_index+1],
                               H_azm_low_1,far_rear_bulb_union_1])
        main_rear_tfi = T.join(main_rear_tfi_0, main_rear_tfi_1)
        main_rear_tfi[0] = 'main_rear_tfi'

        proj_surfs = [main_rear_tfi]
        fixed_bnds = [P.exteriorFaces(surf) for surf in proj_surfs ]
        GSD.prepareGlue(proj_surfs, fixed_bnds)
        T._projectOrtho(proj_surfs, support)
        GSD.applyGlue(proj_surfs, fixed_bnds)

    else:
        far_rear_bulb_union = GSD.getBoundary(support,'imax')




    # CONSTRUCT THE MIDDLE TOPOLOGY

    if HAS_FRONT_BULB:
        central_index_i = C.getNPts(profile_rev_sectors[0]) + \
            C.getNPts(profile_rev_sectors[1])+int((C.getNPts(profile_rev_sectors[2])-1)/2) -2
        central_width = W.distance(W.point(profile_rev_sectors[2],0),
                                   W.point(profile_rev_sectors[2],-1))

    else:
        central_index_i = C.getNPts(profile_rev_sectors[0]) + \
            int((C.getNPts(profile_rev_sectors[1])-1)/2) -2
        central_width = W.distance(W.point(profile_rev_sectors[1],0),
                                   W.point(profile_rev_sectors[1],-1))



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

    charact_length = W.getCharacteristicLength(tip_curves_topo)
    factor = tip_axial_scaling_at_farfield*central_width/charact_length
    T._homothety(tip_curves_topo, topo_center, factor)
    T._projectOrtho(tip_curves_topo, support)
    # W.deformWidth(tip_curves_topo, factor=3.0) # TODO connect factor as a paremeter ?
    T._projectOrtho(tip_curves_topo, support)

    I._rmNodesByType(tip_curves_topo,'FlowSolution_t')
    [C._initVars(tip_curves_topo,'s'+i,0) for i in ('x','y','z')]
    migrateFields(support,tip_curves_topo)
    C._normalize(tip_curves_topo,['sx','sy','sz'])

    tip_sectors =  [ s for s in sector_bnds if s[0].startswith('tip')]
    _putSmoothedNormalsAtSurfaces(tip_sectors, eps=0.9, niter=100, mode=0)
    migrateFields(tip_sectors,tip_curves)
    C._normalize(tip_curves,['sx','sy','sz'])





    all_topos = []
    for i in range(100):
        topos = [cr for cr in tip_curves_topo if cr[0].startswith('tip.%d'%i)]
        if not topos: break
        topos = W.reorderAndSortCurvesSequentially(topos)
        all_topos += [ topos ]


    
    All_TFI2_far = [] # this will contain blade-tip region at farfield
    for cr in all_topos:
        try:
            All_TFI2_far += [G.TFI([cr[0],cr[2],cr[1],cr[3]])]
        except BaseException as e:
            J.save(cr,'debug.cgns')
            if W.isSubzone(cr[0],cr[2]) or W.isSubzone(cr[1],cr[3]):
                raise ValueError(J.FAIL+'TFI bounds cannot be coincident. Check debug.cgns and modify tip topology'+J.ENDC)
            raise e



    TFI_H_group_topo = [cr for cr in tip_curves_topo if cr[0].startswith('tfi')]
    I._rmNodesByType(TFI_H_group_topo,'FlowSolution_t')
    W.addNormals(TFI_H_group_topo)


    if HAS_FRONT_BULB:
        TFI_H_group_bnd = [T.join(H_azm_0, H_azm_1), profile_rev_sectors_sideB[2],
                           T.join(H_azm_low_0, H_azm_low_1),profile_rev_sectors[2]]
    else:
        H_azm =  GSD.getBoundary(support,'imin', C.getNPts(profile_rev_sectors[0])-1)
        H_azm[0] = 'H_azm'
        H_azm_low =  GSD.getBoundary(support,'imin', C.getNPts(profile_rev_sectors[0])-1 + \
                                                     C.getNPts(profile_rev_sectors[1])-1)
        H_azm_low[0] = 'H_azm_low'
        TFI_H_group_bnd = [H_azm, profile_rev_sectors_sideB[1],
                           H_azm_low, profile_rev_sectors[1]]




    I._rmNodesByType(TFI_H_group_bnd,'FlowSolution_t')
    TFI_H_group_bnd = W.reorderAndSortCurvesSequentially(TFI_H_group_bnd)

    proj_dir = np.cross(pseudo_blade, a)
    proj_dir = np.cross(a, proj_dir)


    portion_length = D.getLength(TFI_H_group_bnd[1]) * 0.50 # TODO parameter here or make it smart
    length0 = D.getLength(TFI_H_group_bnd[0])



    focus = W.splitAt(TFI_H_group_bnd[0],[portion_length, length0-portion_length],'length')[1]
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



    print('fill farfield H with bezier... ',end='')
    topo_cell = W.meanSegmentLength(TFI_H_group_topo)  *0.50 # TODO important parametrize ?
    if HAS_FRONT_BULB:
        ref_axial_curve = profile_rev_sectors[2]
    else:
        ref_axial_curve = profile_rev_sectors[1]
    axial_cell_length_front = W.distance(W.point(ref_axial_curve,-2),
                                         W.point(ref_axial_curve,-1))
    axial_cell_length_rear = W.distance(W.point(ref_axial_curve,0),
                                        W.point(ref_axial_curve,1))
    axial_cell_length = 0.5*(axial_cell_length_rear+axial_cell_length_front)




    i = 0
    TFI2_bnd_blends = []
    for c1, c2 in zip(TFI_H_group_topo,TFI_H_group_bnd):
        i+=1
        TFI2_bnd_blend = W.fillWithBezier(c1, c2, H_grid_interior_points,
                        tension1=normal_tension, tension2=0.15, # TODO parameter ?
                        tension1_is_absolute=True,
                        tension2_is_absolute=False,
                        length1 = topo_cell, length2=axial_cell_length,
                        support=support_central,
                        projection_direction=proj_dir
                        )
        TFI2_bnd_blend[0] = 'bl.%d.blend'%i
        TFI2_bnd_blends += [ TFI2_bnd_blend ]
    print(J.GREEN+'ok'+J.ENDC)


    print('smoothing farfield H... ',end='')
    fixedZones = TFI_H_group_bnd 
    surfaces_to_be_smoothed = TFI2_bnd_blends+All_TFI2_far
    GSD.prepareGlue(surfaces_to_be_smoothed,fixedZones)
    GSD.prepareGlue(tip_curves_topo,surfaces_to_be_smoothed)
    for i in range( 5 ):
        T._smooth(surfaces_to_be_smoothed,eps=0.8, type=0,
            niter=FarfieldTipSmoothIterations,
            fixedConstraints=fixedZones)
        T._projectRay(surfaces_to_be_smoothed,support_central, [G.barycenter(surfaces_to_be_smoothed)[0],0,0])
        GSD.applyGlue(surfaces_to_be_smoothed,fixedZones)
    GSD.applyGlue(tip_curves_topo,surfaces_to_be_smoothed)

    print(J.GREEN+'ok'+J.ENDC)


    # double check tip_curves_topo  and tip_curves have the same orientation:
    for bot, top in zip(tip_curves, tip_curves_topo):
        tangent_bot = W.point(bot,-1) - W.point(bot,0)
        tangent_top = W.point(top,-1) - W.point(top,0)
        if tangent_bot.dot(tangent_top) < 0:
            W.reverse(top,True)

    TFI2_blends = []
    for c1, c2 in zip(tip_curves,tip_curves_topo):
        TFI2_blend = W.fillWithBezier(c1, c2, number_of_points,
                        tension1=tip_radial_tension, tension2=0., # BEWARE tip_radial_tension is relative
                        length1=tip_cell_length, length2=farfield_cell_height)
        TFI2_blend[0] = c1[0]+'.blend'
        TFI2_blends += [ TFI2_blend ]

    print('building farfield connectors... ', end='')
    profile_rev_sectors = W.reorderAndSortCurvesSequentially(profile_rev_sectors)

    for p in profile_rev_sectors:
        sx, sy, sz = J.invokeFields(p,['sx','sy','sz'])
        sx[:] = pseudo_blade[0]
        sy[:] = pseudo_blade[1]
        sz[:] = pseudo_blade[2]

    GSD._alignNormalsWithRadialCylindricProjection(profile_rev_sectors,
                                                rotation_center, rotation_axis)

    for p in [ profile_rev_sectors[0] ]:
        if HAS_FRONT_BULB: 
            sx,sy,sz = J.getVars(p,['sx','sy','sz'])
            sx[0] = a[0]
            sy[0] = a[1]
            sz[0] = a[2]

    for p in [ profile_rev_sectors[-1] ]:
        if HAS_REAR_BULB:
            sx,sy,sz = J.getVars(p,['sx','sy','sz'])
            sx[-1] = -a[0]
            sy[-1] = -a[1]
            sz[-1] = -a[2]


    join_cell_length = W.segment(profile[0],-1)
    union_curves = []
    for c1, c2 in zip(profile,profile_rev_sectors):
        length1 = J.getVars(c1,['segment'])[0][0]
        union_curve = W.fillWithBezier(c1,c2,number_of_points, length1=length1,
                    length2=farfield_cell_height,tension2=0.,
                    tension1=normal_tension, tension1_is_absolute=True,
                    only_at_indices=[0])
        union_curves += [ union_curve ]
    length1 = J.getVars(c1,['segment'])[0][-1]
    union_curve = W.fillWithBezier(c1,c2,number_of_points, length1=length1,
                length2=farfield_cell_height,
                only_at_indices=[-1],tension2=0.,
                tension1=normal_tension,tension1_is_absolute=True)
    union_curves += [ union_curve ]
    if rear_support: T._projectDir(union_curves[-1],rear_support,dir=[1,0,0])

    if front_support: T._projectDir(union_curves[0],front_support,dir=[-1,0,0])

    print(J.GREEN+'ok'+J.ENDC)


    print('creating farfield surfacic domains... ',end='')
    tfi_unions = []
    i = -1
    for c1, c2 in zip(profile,profile_rev_sectors):
        i+=1
        wires = [c1,c2,union_curves[i],union_curves[i+1]]
        tfi = G.TFI(wires)
        tfi_unions += [ tfi ]

    tfi_unions_sideB = T.rotate(tfi_unions,c,a,360./float(blade_number))
    for cr in tfi_unions_sideB:
        cr[0]+='.B'


    if HAS_FRONT_BULB:
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

        tfi_inner_bulb_front_sideA = G.TFI([inner_union_bulb_front_sideA,inner_union_bulb_front,
                 TFI2_bulb_front_inner_0,far_bulb_union_0])
        tfi_inner_bulb_front_sideA[0] = 'tfi_inner_bulb_front_sideA'
        tfi_inner_bulb_front_sideB = G.TFI([
            inner_union_bulb_front,inner_union_bulb_front_sideB,
            TFI2_bulb_front_inner_1,far_bulb_union_1])
        tfi_inner_bulb_front_sideB[0] = 'tfi_inner_bulb_front_sideB'

        TFI2_bulb_join = T.join(tfi_inner_bulb_front_sideA,tfi_inner_bulb_front_sideB)
        TFI2_bulb_join[0] = 'TFI2_bulb_join'

    if HAS_FRONT_BULB:
        inner_union_main_front_sideA = GSD.getBoundary(tfi_unions[1],'jmax')
        inner_union_main_front_sideB = GSD.getBoundary(tfi_unions_sideB[1],'jmax')
        H_azm_front = T.join(H_azm_0, H_azm_1)
    else:
        inner_union_main_front_sideA = GSD.getBoundary(tfi_unions[0],'jmax')
        inner_union_main_front_sideB = GSD.getBoundary(tfi_unions_sideB[0],'jmax')
        H_azm_front = H_azm
    inner_union_main_front_sideA[0] = 'inner_union_main_front_sideA'
    inner_union_main_front_sideB[0] = 'inner_union_main_front_sideB'
    H_azm_front[0] = 'H_azm_front'

    TFI2_spinners = [s for s in sector_bnds if s[0].startswith('TFI2_spinner')]

    if not HAS_FRONT_BULB and not HAS_REAR_BULB:
        rear_spinner = [s for s in TFI2_spinners if s[0] in ['TFI2_spinner_1', 'TFI2_spinner_2']]
        front_spinner = [s for s in sector_bnds if s[0] in ['TFI2_spinner_0','TFI2_spinner_3', 'TFI2_spinner_4']]
    else:
        front_spinner = [s for s in TFI2_spinners if s[0] in ['TFI2_spinner_0','TFI2_spinner_1', 'TFI2_spinner_2']]
        rear_spinner = [s for s in sector_bnds if s[0] in ['TFI2_spinner_3', 'TFI2_spinner_4']]


    TFI2_spinner_join = T.join(*front_spinner)
    TFI2_spinner_join[0] = 'TFI2_spinner_join'
    if W.tangentExtremum(GSD.getBoundary(TFI2_spinner_join,'imin')).dot(a) < 0:
        T._reorder(TFI2_spinner_join,(1,-2,3))
    TFI2_spinner_join_lowedge = GSD.getBoundary(TFI2_spinner_join,'jmin')
    TFI2_spinner_join_lowedge[0] = 'TFI2_spinner_join_lowedge'
    

    TFI2_spinner_rear = T.join(*rear_spinner)
    rear_near_low = GSD.getBoundary(TFI2_spinner_rear,'jmin')
    rear_near_low[0] = 'rear_near_low'
    rear_near_low_0,rear_near_low_1=T.splitNParts(rear_near_low,2,dirs=[1])
    rear_near_low_0[0] = 'rear_near_low_0'
    rear_near_low_1[0] = 'rear_near_low_1'

    TFI2_spinner_rear_lowedge = GSD.getBoundary(TFI2_spinner_rear,'jmax')
    TFI2_spinner_rear_lowedge[0] = 'TFI2_spinner_rear_lowedge'

    if HAS_FRONT_BULB:
        tfi_front_top = T.join(tfi_inner_bulb_front_sideA,
                               tfi_inner_bulb_front_sideB)
    else:
        rmin_front = GSD.getBoundary(TFI2_spinner_join,'jmax')
        rmin_front[0] = 'rmin_front'

        theta_min_front = GSD.getBoundary(tfi_unions[0],'jmin')
        theta_min_front[0] = 'theta_min_front'

        theta_max_front = GSD.getBoundary(tfi_unions_sideB[0],'jmin')
        theta_max_front[0] = 'theta_max_front'

        rmax_front = GSD.getBoundary(support,'imin')
        rmax_front[0] = 'rmax_front'

        tfi_wires = [theta_min_front, theta_max_front,
                               rmin_front, rmax_front]

        tfi_front_top = G.TFI(tfi_wires)

    tfi_front_top[0] = 'tfi_front_top'


    tfi_front_bot_wires = [inner_union_main_front_sideA,
                           inner_union_main_front_sideB,
                           TFI2_spinner_join_lowedge,
                           H_azm_front]
    

    tfi_front_bot = G.TFI(tfi_front_bot_wires)
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
        if W.isSubzone(profile_rev_sectors[element_index], s):
            tfi2_H_sideA = s
            tfi2_H_sideA[0] = 'tfi2_H_sideA'
            curve_H_far_sideA = GSD.getBoundary(tfi2_H_sideA,'jmin')
            curve_H_far_sideA[0] = 'curve_H_far_sideA'
            break

    for s in sector_bnds:
        if W.isSubzone(profile[element_index],s):
            tfi2_H_sideA_spinner = s
            tfi2_H_sideA_spinner[0] = 'tfi2_H_sideA_spinner'
            curve_H_near_sideA = GSD.getBoundary(tfi2_H_sideA_spinner,'imin')
            curve_H_near_sideA[0] = 'curve_H_near_sideA'
            break

    # 3 of 4
    for s in TFI2_bnd_blends:
        if W.isSubzone(profile_rev_sectors_sideB[element_index], s):
            tfi2_H_sideB = s
            tfi2_H_sideB[0] = 'tfi2_H_sideB'
            curve_H_far_sideB = GSD.getBoundary(tfi2_H_sideB,'jmin')
            curve_H_far_sideB[0] = 'curve_H_far_sideB'
            break
    for s in sector_bnds:
        if W.isSubzone(profile_sideB[element_index],s):
            tfi2_H_sideB_spinner = s
            tfi2_H_sideB_spinner[0] = 'tfi2_H_sideB_spinner'
            curve_H_near_sideB = GSD.getBoundary(tfi2_H_sideB_spinner,'imin')
            curve_H_near_sideB[0] = 'curve_H_near_sideB'
            break

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
        crv[0] = 'blend.%dm'%i
        all_blend_unions += [ crv ]
        crv = GSD.getBoundary(s,'jmax')
        crv[0] = 'blend.%dM'%i
        all_blend_unions += [ crv ]

    sides_curves = [GSD.getBoundary(s,'jmax') for s in tfi_unions+tfi_unions_sideB]
    candidates_to_connect = sides_curves + all_blend_unions

    c1, c2 = W.getConnectingCurves(curve_H_near_top, candidates_to_connect)
    wires = [curve_H_near_top, curve_H_far_top, c1,c2]
    tfi_H1 = G.TFI(wires)
    tfi_H1[0] = 'tfi_H1'

    c1, c2 = W.getConnectingCurves(curve_H_near_sideA, candidates_to_connect)
    tfi_H2 = G.TFI([curve_H_near_sideA, curve_H_far_sideA, c1,c2])
    tfi_H2[0] = 'tfi_H2'

    c1, c2 = W.getConnectingCurves(curve_H_near_sideB, candidates_to_connect)
    tfi_H3 = G.TFI([curve_H_near_sideB, curve_H_far_sideB, c1,c2])
    tfi_H3[0] = 'tfi_H3'

    c1, c2 = W.getConnectingCurves(curve_H_near_rear, candidates_to_connect)
    tfi_H4_wires = [curve_H_near_rear, curve_H_far_rear, c1,c2]
    tfi_H4 = G.TFI(tfi_H4_wires)
    tfi_H4[0] = 'tfi_H4'


    if not HAS_REAR_BULB:
        main_rear_tfi = G.TFI([profile_rev_sectors[element_index+1],
                               profile_rev_sectors_sideB[element_index+1],
                               H_azm_low,
                               far_rear_bulb_union])
        main_rear_tfi[0] = 'main_rear_tfi'
        

    c1, c2 = W.getConnectingCurves(TFI2_spinner_rear_lowedge, candidates_to_connect)
    tfi_rear_top = G.TFI([TFI2_spinner_rear_lowedge, H_azm_low, c1,c2])
    tfi_rear_top[0] = 'tfi_rear_top'

    c1, c2 = W.getConnectingCurves(rear_near_low, candidates_to_connect)
    if rear_support: T._projectDir(c1,rear_support,dir=[1,0,0])


    c2 = T.rotate(c1,(0,0,0),(-1,0,0),360/blade_number)
    tfi_rear_low = G.TFI([rear_near_low, far_rear_bulb_union, c1,c2])
    tfi_rear_low[0] = 'tfi_rear_low'



    if HAS_FRONT_BULB:
        FACES_BULB_FRONT = [tfi_unions[0],tfi_inner_bulb_front_sideB,
                            tfi_inner_bulb_front_sideA,tfi_unions_sideB[0],
                            TFI2_bulb_front, far_bulb_tfi]


    FACES_MAIN_FRONT = [tfi_front_top,tfi_front_bot,
                        tfi_unions[element_index-1],tfi_unions_sideB[element_index-1],
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

    left = tfi_unions[element_index+1]
    left[0] = 'left'
    right = tfi_unions_sideB[element_index+1]
    right[0] = 'right'
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
        rear_bulb_far_wires = [profile_rev_sectors[-1],far_rear_bulb_union_1,
                              profile_rev_sectors_sideB[-1],far_rear_bulb_union_0]

        rear_bulb_far = G.TFI(rear_bulb_far_wires)
        rear_bulb_far[0] = 'rear_bulb_far'

        proj_surfs = [rear_bulb_far]
        fixed_bnds = [P.exteriorFaces(surf) for surf in proj_surfs ]
        GSD.prepareGlue(proj_surfs, fixed_bnds)
        T._projectOrtho(proj_surfs, support)
        GSD.applyGlue(proj_surfs, fixed_bnds)

        left = tfi_unions[-1]
        top, right = T.splitNParts(tfi_rear_low,2,dirs=[2])
        bottom = tfi_unions_sideB[-1]
        ant = GSD.selectConnectingSurface([left,right],sector_bnds)
        post = rear_bulb_far

        FACES_BULB_REAR = [left,right,top,bottom,ant,post]


    if HAS_FRONT_BULB:
        BaseZonesList = ['bulb_front',FACES_BULB_FRONT]
    else:
        BaseZonesList = []
    BaseZonesList.extend(['main_front',FACES_MAIN_FRONT,
                          'h_front',FACES_H_FRONT,
                          'h_sidea',FACES_H_sideA,
                          'h_sideb',FACES_H_sideB,
                          'h_rear',FACES_H_REAR,
                          'main_rear',FACES_MAIN_REAR])

    if HAS_REAR_BULB: BaseZonesList.extend(['bulb_rear',FACES_BULB_REAR])

    for i, zones in enumerate(GROUP_TFI_CENTRAL):
        BaseZonesList.extend(['tip_%d'%i, zones])

    farfield_faces = C.newPyTree(BaseZonesList)

    def getUniqueSurfaces(list_of_surfaces):
        repeated = []
        for s1 in list_of_surfaces:
            for s2 in list_of_surfaces: 
                if s1[0]!=s2[0] and s1[0] not in repeated and W.isSubzone(s1,s2):
                    repeated += [s2[0]]
        unique = [s for s in list_of_surfaces if s[0] not in repeated]
        return unique

    # remove duplicates
    for base in I.getBases(farfield_faces):
        base[2] = getUniqueSurfaces(I.getZones(base))
    print(J.GREEN+'ok'+J.ENDC) # 'creating farfield surfacic domains... '


    grids = []
    print('making farfield 3D TFI... ', end='')
    for base in I.getBases(farfield_faces):
        try:
            surfs_for_tfi = I.getZones(base)
            nb_of_surfs = len(surfs_for_tfi)
            if nb_of_surfs != 6:
                raise ValueError(f'requires 6 surfaces exactly for TFI, but got {nb_of_surfs} at base {base[0]}')
            tfi3D = G.TFI(I.getZones(base))
        except BaseException as e:
            db = J.tree(SURFS_FOR_TFI=surfs_for_tfi)
            J.save(db,'debug.cgns')
            raise ValueError(f'TFI failed for base {base[0]}, check debug.cgns') from e
        tfi3D[0] = base[0]
        if base[0] == "h_front":
            # HACK https://elsa.onera.fr/issues/11849
            T._reorder(tfi3D,(1,2,-3))
        else:
            T._makeDirect(tfi3D)


        grids += [ tfi3D ]
    print(J.GREEN+'ok'+J.ENDC)

    print('checking negative volume cells in farfield 3D TFI... ',end='')
    negative_volume_cells = GVD.checkNegativeVolumeCells(grids, volume_threshold=0)
    if negative_volume_cells:
        try: os.makedirs(DIRECTORY_CHECKME)
        except: pass
        J.save(negative_volume_cells, os.path.join(DIRECTORY_CHECKME,'negative_volume_cells.cgns'))
    else:
        print(J.GREEN+'ok'+J.ENDC)



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

def makeHub(profile, blade_number=4, rotation_center=[0,0,0],
            rotation_axis=[1,0,0], number_of_cells_azimuth=20,
            front_bulb_shrink_ratio=0.25,
            rear_bulb_shrink_ratio=0.25,
            support_azimuthal_resolution=200,
            support_axial_resolution=200):
    '''
    Build a hub structured surface from a **profile** curve using azimuthal
    revolution avoiding degenerated cells at extrema that lie on rotation axis.

    Parameters
    ----------

        profile : zone
            Structured curve defining the profile used to make the revolution.
            The first index is supposed to be the front (or leading-edge) of the
            hub, and it must lie on the rotation axis.

            .. important::
                first point of **profile** must lie on the rotation axis
            
            .. hint::
                if last point of **profile** does not lie on the rotation axis,
                then the final hub result will be open. If it lies on the 
                rotation axis, then a "bulb" topology is used to avoid the 
                cell degeneration at axis, like in the front part of the hub
            
            .. note::
                the resulting hub axial discretization will exactly follow the 
                discretization of curve **profile**. You may want to control 
                such discretization by applying :py:func:`MOLA.Wireframe.polyDiscretize`
                to **profile** prior to constructing the hub.

        blade_number : :py:class:`int` or :py:class:`str`
            If you provide an :py:class:`int` greater than 1, then only a 
            sector of the hub will be generated, according to the provided
            number of blades. Alternatively, you can set ``blade_number='full'``
            and then a full 360 degree hub is constructed.

        rotation_center : :py:class:`list` or :py:class:`numpy.ndarray` of 3 `float` 
            coordinates :math:`(x,y,z)` of the rotation center of the revolution

        rotation_axis : :py:class:`list` or :py:class:`numpy.ndarray` of 3 `float` 
            unitary vector indicating the rotation axis of the revolution
    
        number_of_cells_azimuth : int
            number of cells discretizing the hub in the azimuthal direction.
            It also corresponds to the number of points of **profile** extrema 
            used to construct the bulbs. It must be even. If ``blade_number=='full'`` then it must be 
            exactly divisible by 4.

        support_azimuthal_resolution : int
            A very fine auxiliar surface is generated for projecting the front 
            and rear bulbs in the hub geometry. This parameter controls the
            number of azimuthal points of such surface. Usually 4 times 
            **number_of_cells_azimuth** is sufficient. High values will make
            take longer to construct the hub. Low values will create low quality
            of hub with "facetted" spots on the bulb regions.

        support_azimuthal_resolution : int
            A very fine auxiliar surface is generated for projecting the front 
            and rear bulbs in the hub geometry. This parameter controls the
            number of axial points of such surface. Usually 4 times 
            **number_of_cells_azimuth** is sufficient. High values will make
            take longer to construct the hub. Low values will create low quality
            of hub with "facetted" spots on the bulb regions.

    Returns
    -------

        hub : :py:class:`list` of zone
            list of structured surfaces corresponding to the hub 

    Example
    -------

    ::

        import MOLA.RotatoryWings as RW
        import Geom.PyTree as D

        # if trailing edge lies on axis, spinner will have two bulbs (front & rear)
        profile = D.bezier(D.polyline([(2,0,0), (1,1,0), (-1,1,0), (-2,0,0)]), 101)
        hub = RW.makeHub(profile, blade_number=4)
        RW.J.save(hub,'hub_closed.cgns')

        # if trailing edge does not lie on axis, spinner will have only front bulb
        profile = D.bezier(D.polyline([(2,0,0), (1,1,0), (-1,1,0), (-2,1,0)]), 101)
        hub = RW.makeHub(profile, blade_number=4)
        RW.J.save(hub,'hub_open.cgns')



    '''


    if blade_number==2:
        hub = makeHub(profile, blade_number=4, rotation_center=rotation_center,
            rotation_axis=rotation_axis,
            number_of_cells_azimuth=int(number_of_cells_azimuth/2),
            front_bulb_shrink_ratio=front_bulb_shrink_ratio,
            rear_bulb_shrink_ratio=rear_bulb_shrink_ratio,
            support_azimuthal_resolution=support_azimuthal_resolution,
            support_axial_resolution=support_axial_resolution)
        hub_rot = T.rotate(hub,rotation_center,rotation_axis,90.0)
        joined_zones = []
        for z, zr in zip(I.getZones(hub), I.getZones(hub_rot)):
            joined = T.join(z,zr)
            joined[0] = z[0]
            joined_zones += [ joined ]
        T._reorder(joined_zones[0],(-1,-2,3))
        T._reorder(joined_zones[1],(2,1,3))
        hub = joined_zones

        return hub

    elif blade_number=='full':
        if number_of_cells_azimuth%4!=0:
            raise ValueError('for full spinner, number_of_cells_azimuth%4 == 0')

        hub = makeHub(profile, blade_number=4, rotation_center=rotation_center,
            rotation_axis=rotation_axis,
            number_of_cells_azimuth=int(number_of_cells_azimuth/4),
            front_bulb_shrink_ratio=front_bulb_shrink_ratio,
            rear_bulb_shrink_ratio=rear_bulb_shrink_ratio,
            support_azimuthal_resolution=support_azimuthal_resolution,
            support_axial_resolution=support_axial_resolution)
        hub_rot = T.rotate(hub,rotation_center,rotation_axis,90.0)
        joined_zones = []
        for z, zr in zip(I.getZones(hub), I.getZones(hub_rot)):
            joined = T.join(z,zr)
            joined[0] = z[0]
            joined_zones += [ joined ]
        T._reorder(joined_zones[0],(-1,-2,3))
        T._reorder(joined_zones[1],(2,1,3))
        hub = joined_zones

        hub_rot = T.rotate(joined_zones,rotation_center,rotation_axis,180.0)
        for z in hub_rot: z[0] += '.rot'
        joined_zones = []
        for z, zr in zip(I.getZones(hub), I.getZones(hub_rot)):
            joined = T.join(z,zr)
            joined[0] = z[0]
            joined_zones += [ joined ]
        T._reorder(joined_zones[0],(-1,-2,3))
        T._reorder(joined_zones[1],(2,1,3))
        hub = joined_zones

        return joined_zones
    
    elif blade_number <= 1:
        raise ValueError(J.FAIL+'blade_number must be greater than 2'+J.ENDC)
    elif blade_number > 2:
        pass
    else:
        raise NotADirectoryError(J.FAIL+f'blade_number={blade_number} not implemented'+J.ENDC)

    if number_of_cells_azimuth%2!=0:
        raise ValueError(J.FAIL+'number_of_cells_azimuth must be even'+J.ENDC)
    elif number_of_cells_azimuth<8:
        raise ValueError(J.FAIL+'insufficient number_of_cells_azimuth. Increase its value.'+J.ENDC)

    number_of_points_azimuth = number_of_cells_azimuth + 1


    angle_sector = 360./float(blade_number)

    number_of_points_leading_edge_tfi=number_of_points_azimuth
    number_of_points_trailing_edge_tfi=number_of_points_azimuth

    profile = I.getZones(profile)[0]

    center = np.array(rotation_center,dtype=np.float64)
    axis = np.array(rotation_axis,dtype=np.float64)
    tol = 1e-8

    leading_edge = W.point(profile)
    trailing_edge = W.point(profile,-1)

    leading_edge_distance_to_axis = W.distanceOfPointToLine(leading_edge,axis,center)
    trailing_edge_distance_to_axis = W.distanceOfPointToLine(trailing_edge,axis,center)

    if leading_edge_distance_to_axis > tol:
        if trailing_edge_distance_to_axis < tol:
            profile = W.reverse(profile)
            hub = makeHub(profile,
                          blade_number=blade_number,
                          rotation_center=rotation_center,
                          rotation_axis=-axis,
                          number_of_cells_azimuth=number_of_cells_azimuth,
                          front_bulb_shrink_ratio=front_bulb_shrink_ratio,
                          rear_bulb_shrink_ratio=rear_bulb_shrink_ratio,
                          support_azimuthal_resolution=support_azimuthal_resolution,
                          support_axial_resolution=support_axial_resolution)
            I._renameNode(hub,'hub.front','hub.rear')
            
        else:
            profile = T.rotate(profile,(0,0,0),(1,0,0),0.5*angle_sector)
            hub = D.axisym(profile,(0,0,0),(1,0,0),-angle_sector,number_of_points_azimuth)
        
        return hub # TODO check/fix order and subparts names


    has_trailing_edge_bulb = True  if trailing_edge_distance_to_axis < tol else False
    if not has_trailing_edge_bulb: number_of_points_trailing_edge_tfi = 1
    profile_min = T.rotate(profile,tuple(rotation_center),tuple(rotation_axis),
                           -angle_sector/2)
    profile_min[0] = 'profile_min'

    npts_profile = C.getNPts(profile_min)
    main_profile = T.subzone(profile_min,(number_of_points_leading_edge_tfi,1,1),
                             (npts_profile-number_of_points_trailing_edge_tfi+1,1,1))
    main_profile[0] = 'main_profile'

    main_hub = D.axisym(main_profile, tuple(rotation_center), tuple(rotation_axis),
        angle_sector, number_of_points_azimuth)
    main_hub[0] = 'main_hub'

    # leading edge
    le_bulb_profile = T.subzone(profile_min,(1,1,1),
                             (number_of_points_leading_edge_tfi,1,1))
    le_bulb_profile[0] = 'le_bulb_profile'

    le_bulb_profile_front, le_bulb_profile_rear = W.splitAt(le_bulb_profile, 
                                                int(number_of_cells_azimuth/2))
    le_bulb_profile_front[0] = 'le_bulb_profile_front'
    le_bulb_profile_rear[0] = 'le_bulb_profile_rear'

    le_bulb_profile_front_rot = T.rotate(le_bulb_profile_front,
        tuple(rotation_center),tuple(rotation_axis), angle_sector)
    le_bulb_profile_front_rot[0] = 'le_bulb_profile_front_rot'

    le_bulb_profile_rear_rot = T.rotate(le_bulb_profile_rear,
        tuple(rotation_center),tuple(rotation_axis), angle_sector)
    le_bulb_profile_rear_rot[0] = 'le_bulb_profile_rear_rot'

    profile_support = W.discretize(le_bulb_profile, support_axial_resolution)
    fine_le_support = D.axisym(profile_support, tuple(rotation_center), tuple(rotation_axis),
        angle_sector, support_azimuthal_resolution)
    fine_le_support[0] = 'fine_le_support'

    le_azimuth_bnd = GSD.getBoundary(main_hub, 'imin')
    le_azimuth_bnd[0] = 'le_azimuth_bnd'
    le_azimuth_bnd_A, le_azimuth_bnd_B = W.splitAt(le_azimuth_bnd,0.5)
    le_azimuth_bnd_A[0] = 'le_azimuth_bnd_A'
    le_azimuth_bnd_B[0] = 'le_azimuth_bnd_B'

    le_middle = T.rotate(le_bulb_profile_rear, tuple(rotation_center),
                         tuple(rotation_axis), angle_sector/2)
    le_middle[0] = 'le_middle'
    le_middle = W.discretize(le_middle,200)
    le_middle = W.splitAt(le_middle, W.getLength(le_bulb_profile_front)/(np.sqrt(2)*2),
                          'length')[1]
    segment = W.segment(le_bulb_profile_rear,-1)
    le_middle = W.discretize(le_middle,C.getNPts(le_bulb_profile_rear),
        Distribution=dict(kind='tanhTwoSides',FirstCellHeight=0.5*segment,
                    LastCellHeight=segment))
    T._projectOrtho(le_middle, fine_le_support)

    union_le_A = D.line(W.point(le_bulb_profile_rear,0),W.point(le_middle,0),C.getNPts(le_azimuth_bnd_A))
    union_le_A[0] = 'union_le_A'
    union_le_B = D.line(W.point(le_middle,0), W.point(le_bulb_profile_rear_rot,0),C.getNPts(le_azimuth_bnd_B))
    union_le_B[0] = 'union_le_B'
    fixed = W.extremaAsZones(union_le_A) + W.extremaAsZones(union_le_B)
    T._projectOrtho([union_le_A, union_le_B], fine_le_support)
    
    le_middle = W.discretize(le_middle,C.getNPts(le_bulb_profile_rear),
        Distribution=dict(kind='tanhTwoSides',
                    FirstCellHeight=W.segment(union_le_A,-1),
                    LastCellHeight=segment))
    T._projectOrtho(le_middle, fine_le_support)

    # glueing is allways required after projections onto surfaces
    W.glueCurvesAtExtrema(union_le_A, le_bulb_profile_rear, 'start','start','1-towards-2')
    W.glueCurvesAtExtrema(union_le_A, le_middle, 'end','start','1-towards-2')
    W.glueCurvesAtExtrema(union_le_B, le_middle, 'start','start','1-towards-2')
    W.glueCurvesAtExtrema(union_le_B, le_bulb_profile_rear_rot, 'end','start','1-towards-2')
    W.glueCurvesAtExtrema(le_middle, le_azimuth_bnd_A, 'end','end','1-towards-2')

    tfi_le_A = G.TFI([le_bulb_profile_rear, le_middle,
                      union_le_A, le_azimuth_bnd_A])
    tfi_le_A[0] = 'tfi_le_A'
    
    tfi_le_B = G.TFI([le_middle, le_bulb_profile_rear_rot,
                      union_le_B, le_azimuth_bnd_B])
    tfi_le_B[0] = 'tfi_le_B'

    tfi_le_union = T.join(tfi_le_A,tfi_le_B)
    tfi_le_union[0] = 'tfi_le_union'

    tfi_le_bulb = G.TFI([le_bulb_profile_front, union_le_B,
                         union_le_A, le_bulb_profile_front_rot])
    tfi_le_bulb[0] = 'tfi_le_bulb'
    T._reorder(tfi_le_bulb,(2,1,3))

    surfs2proj = [tfi_le_bulb, tfi_le_union] # result front
    fixed = P.exteriorFaces(surfs2proj)
    GSD.prepareGlue(surfs2proj, fixed)
    T._projectOrtho(surfs2proj, fine_le_support)
    GSD.applyGlue(surfs2proj, fixed)

    T._reorder(tfi_le_union, (2,1,3))


    

    if has_trailing_edge_bulb: 
        te_bulb_profile = T.subzone(profile_min,
            (npts_profile-number_of_points_trailing_edge_tfi+1,1,1), (npts_profile,1,1))
        te_bulb_profile[0] = 'te_bulb_profile'

        te_bulb_profile_rear, te_bulb_profile_front = W.splitAt(te_bulb_profile,
                                                int(number_of_cells_azimuth/2))
        te_bulb_profile_front[0] = 'te_bulb_profile_front'
        te_bulb_profile_rear[0] = 'te_bulb_profile_rear'

        te_bulb_profile_front_rot = T.rotate(te_bulb_profile_front,
            tuple(rotation_center),tuple(rotation_axis), angle_sector)
        te_bulb_profile_front_rot[0] = 'te_bulb_profile_front_rot'

        te_bulb_profile_rear_rot = T.rotate(te_bulb_profile_rear,
            tuple(rotation_center),tuple(rotation_axis), angle_sector)
        te_bulb_profile_rear_rot[0] = 'te_bulb_profile_rear_rot'

        profile_support = W.discretize(te_bulb_profile, support_axial_resolution)
        fine_te_support = D.axisym(profile_support, tuple(rotation_center), tuple(rotation_axis),
            angle_sector, support_azimuthal_resolution)
        fine_te_support[0] = 'fine_te_support'

        te_azimuth_bnd = GSD.getBoundary(main_hub, 'imax')
        te_azimuth_bnd[0] = 'te_azimuth_bnd'
        te_azimuth_bnd_A, te_azimuth_bnd_B = W.splitAt(te_azimuth_bnd,0.5)
        te_azimuth_bnd_A[0] = 'te_azimuth_bnd_A'
        te_azimuth_bnd_B[0] = 'te_azimuth_bnd_B'

        te_middle = T.rotate(te_bulb_profile_rear, tuple(rotation_center),
                            tuple(rotation_axis), angle_sector/2)
        te_middle[0] = 'te_middle'
        te_middle = W.discretize(te_middle,200)

        length_union = W.getLength(te_bulb_profile_front)/(np.sqrt(2)*2)
        length_te_middle = W.getLength(te_middle)
        te_middle = W.splitAt(te_middle, length_te_middle-length_union, 'length')[0]
        segment = W.segment(te_bulb_profile_rear,0)
        te_middle = W.discretize(te_middle,C.getNPts(te_bulb_profile_rear),
            Distribution=dict(kind='tanhTwoSides',FirstCellHeight=segment,
                        LastCellHeight=0.5*segment))
        T._projectOrtho(te_middle, fine_te_support)

        union_te_A = D.line(W.point(te_bulb_profile_rear,-1),W.point(te_middle,-1),C.getNPts(le_azimuth_bnd_A))
        union_te_A[0] = 'union_te_A'
        union_te_B = D.line(W.point(te_middle,-1), W.point(te_bulb_profile_rear_rot,-1),C.getNPts(le_azimuth_bnd_B))
        union_te_B[0] = 'union_te_B'
        T._projectOrtho([union_te_A, union_te_B], fine_te_support)

        te_middle = W.discretize(te_middle,C.getNPts(te_bulb_profile_rear),
            Distribution=dict(kind='tanhTwoSides',
                              FirstCellHeight=segment,
                              LastCellHeight=W.segment(union_te_A,-1)))
        T._projectOrtho(te_middle, fine_te_support)

        # glueing is allways required after projections onto surfaces
        W.glueCurvesAtExtrema(union_te_A, te_bulb_profile_rear, 'start','end','1-towards-2')
        W.glueCurvesAtExtrema(union_te_A, te_middle, 'end','end','1-towards-2')
        W.glueCurvesAtExtrema(union_te_B, te_middle, 'start','end','1-towards-2')
        W.glueCurvesAtExtrema(union_te_B, te_bulb_profile_rear_rot, 'end','end','1-towards-2')
        W.glueCurvesAtExtrema(te_middle, te_azimuth_bnd_A, 'start','end','1-towards-2')

        tfi_te_A = G.TFI([te_bulb_profile_rear, te_middle,
                        union_te_A, te_azimuth_bnd_A])
        tfi_te_A[0] = 'tfi_te_A'
        tfi_te_B = G.TFI([te_middle, te_bulb_profile_rear_rot,
                        union_te_B, te_azimuth_bnd_B])
        tfi_te_B[0] = 'tfi_te_B'
        tfi_te_union = T.join(tfi_te_A,tfi_te_B)
        tfi_te_union[0] = 'tfi_te_union'

        tfi_te_bulb = G.TFI([te_bulb_profile_front, union_te_B,
                             union_te_A, te_bulb_profile_front_rot])
        tfi_te_bulb[0] = 'tfi_te_bulb'
        T._reorder(tfi_te_bulb,(2,1,3))


        surfs2proj = [tfi_te_bulb, tfi_te_union] # result front
        fixed = P.exteriorFaces(surfs2proj)
        GSD.prepareGlue(surfs2proj, fixed)
        T._projectOrtho(surfs2proj, fine_te_support)
        GSD.applyGlue(surfs2proj, fixed)

        T._reorder(tfi_te_union, (2,1,3))

        main_hub = T.join(tfi_le_union, main_hub)
        main_hub = T.join(main_hub, tfi_te_union)

        surfaces = [ tfi_le_bulb, main_hub, tfi_te_bulb ]
        main_hub[0] = 'hub'
        tfi_le_bulb[0] = 'hub.front'
        tfi_te_bulb[0] = 'hub.rear'


    else:
        main_hub = T.join(tfi_le_union, main_hub)
        main_hub[0] = 'hub'
        tfi_le_bulb[0] = 'hub.front'
        surfaces = [ tfi_le_bulb, main_hub ]

    # at this stage, all surfaces shall have coherent +i indexing following
    # same direction as the profile curve. However, j indexing may require
    # reversing in order to have outwards-pointing face normals.
    # this last step eventually reverses j to have good normals
    ref_main_hub = I.copyRef(main_hub)
    G._getNormalMap(ref_main_hub)
    C.center2Node__(ref_main_hub,'centers:sx',cellNType=0)
    C.center2Node__(ref_main_hub,'centers:sy',cellNType=0)
    C.center2Node__(ref_main_hub,'centers:sz',cellNType=0)
    I._rmNodesByName(ref_main_hub,I.__FlowSolutionCenters__)
    C._normalize(ref_main_hub,['sx','sy','sz'])
    sx,sy,sz = J.getVars(ref_main_hub,['sx','sy','sz'])
    x,y,z = J.getxyz(ref_main_hub)
    prev_normal = np.array([sx[0,0],sy[0,0],sz[0,0]])
    bary = np.array(G.barycenter(ref_main_hub))
    right_direction = np.array([x[0,0],y[0,0],z[0,0]])-bary

    GSD._alignNormalsWithRadialCylindricProjection(ref_main_hub,
                                        rotation_center, rotation_axis)
    normal = np.array([sx[0,0],sy[0,0],sz[0,0]])
    if normal.dot(right_direction) < 1:
        for surface in surfaces:
            T._reorder(surface,(1,-2,3))
    T._reorder(tfi_le_bulb,(2,-1,3))

    return surfaces
    
def computeOptimumProfileHgridSegment(profile, number_of_blades, npts_azimuth,
        BreakPoint, BreakPoint_coordinate='x', rotation_center=[0,0,0],
        rotation_axis=[1,0,0]):
    subparts = W.splitAt(profile, BreakPoint, 'Coordinate'+BreakPoint_coordinate.upper())
    if len(subparts) != 2: return ValueError('unexpected number of profile parts')
    subparts = W.reorderAndSortCurvesSequentially(subparts)
    RelevantPoint = W.point(subparts[0],-1)
    distance_to_axis = W.distanceOfPointToLine(RelevantPoint, rotation_axis,
                                               rotation_center)
    sector_angle = 2*np.pi/number_of_blades
    azimuthal_angle = sector_angle / (npts_azimuth-1)
    arc = azimuthal_angle * distance_to_axis
    return arc

    
    

def buildOpenRotorAndStatorMesh(RotorBlade, StatorBlade, HubProfile,
        FarfieldRadius : float = 4.0,
        InterfaceRadialTensionRelativeToRotorRadius : float = 0.10,
        InterfaceRelativePosition : float = 0.5,

        # ------------------------ ROTOR parameters ------------------------ #
        RotorBuild = True,
        RotorNumberOfBlades : float = 13,

        RotorDeltaPitch : float = 0.0,
        RotorPitchCenter : float = 0.0,
        RotorThetaAdjustmentInDegrees : float = 0.0,
        
        RotorHubProfileReDiscretization = [],

        RotorAzimutalCellAngle : float = 1.0,

        RotorBladeWallCellHeight : float = 1e-5,
        RotorHubWallCellHeight : float = 1e-2,
        RotorBladeWallGrowthRate : float = 1.15,
        RotorBladeRootWallNormalDistanceRelativeToRootChord : float = 0.01, 
        RotorRootWallRemeshRadialDistanceRelativeToMaxRadius : float = 0.1,
        RotorRootRemeshRadialNbOfPoints : int = 11,

        RotorRadialExtrusionNbOfPoints : int = 20,
        RotorHgridXlocations=(-1.50, -0.75),
        RotorHgridNbOfPoints : int = 21,
        RotorHspreadingAngles : list = [-10, 0],
        RotorTipScaleFactorAtRadialFarfield : float = 0.25,
        RotorFarfieldRadialCellLengthRelativeToFarfieldRadius : float = 0.05, 
        RotorFarfieldTipSmoothIterations : int = 50,
        RotorRadialTensionRelativeToRotorRadius : float = 0.10,
        RotorFarfieldProfileAbscissaDeltas : list = [],
        
        RotorBuildMatchMeshAdditionalParams : dict = {},
        RotorBladeExtrusionParams : dict = {},

        # ------------------------ STATOR parameters ------------------------ #
        StatorBuild = True,
        StatorNumberOfBlades : float = 11,

        StatorDeltaPitch : float = 0.0,
        StatorPitchCenter : float = 0.0,
        StatorThetaAdjustmentInDegrees : float = 0.0,
        
        StatorHubProfileReDiscretization = [],

        StatorAzimutalCellAngle : float = 1.0,

        StatorBladeWallCellHeight : float = 1e-5,
        StatorHubWallCellHeight : float = 1e-2,
        StatorBladeWallGrowthRate : float = 1.15,
        StatorBladeRootWallNormalDistanceRelativeToRootChord : float = 0.01, 
        StatorRootWallRemeshRadialDistanceRelativeToMaxRadius : float = 0.11,
        StatorRootRemeshRadialNbOfPoints : int = 11,

        StatorRadialExtrusionNbOfPoints : int = 20,
        StatorHgridXlocations=(-0.45, 0.20),
        StatorHgridNbOfPoints : int = 21,
        StatorHspreadingAngles : list = [0, 10],
        StatorTipScaleFactorAtRadialFarfield : float = 0.25,
        StatorFarfieldRadialCellLengthRelativeToFarfieldRadius : float = 0.05, 
        StatorFarfieldTipSmoothIterations : int = 50,
        StatorRadialTensionRelativeToRotorRadius : float = 0.10,
        StatorFarfieldProfileAbscissaDeltas : list = [],
        
        StatorBuildMatchMeshAdditionalParams : dict = {},
        StatorBladeExtrusionParams : dict = {},

        # ------------------------------- misc ------------------------------- #
        LOCAL_DIRECTORY_CHECKME = 'CHECK_ME',
        raise_error_if_negative_volume_cells=False,
        ):
    '''
    Creates the open rotor and stator single channel case for use with a 
    sliding mesh (elsA's RNA) strategy at interface between domains.
    Flow direction goes in +X direction (rotation axis is +/-X) and rotation
    center is located at (0,0,0).

    Parameters
    ----------

        RotorBlade : PyTree
            Rotor blade, with closed tip, intersecting HubProfile, positionned
            on OXY plane (spanwise direction is +Y)
    
        StatorBlade : PyTree
            Stator blade, with closed tip, intersecting HubProfile, positionned
            on OXY plane (spanwise direction is +Y)

        HubProfile : zone
            1D curve on OXY plane representing the hub. Indexing must start from
            leading edge (increasing i-index goes in +X direction).

        CoordinateOfRotorStatorInterfaceAtHub : float
            Sets the rotor/stator interface location following X coordinate. It
            uses the nearest point of HubProfile to the provided value (it is
            better if HubProfile has a point in the requested location).

        DeltaPitchOfRotor : float
            The change in pitch from StatorBlade position around +Y axis and 
            passing through point **PitchCenterOfRotor** that user must provide

        PitchCenterOfRotor : 3-float list
            Point around which **DeltaPitchOfRotor** is applied in +Y direction

        DeltaPitchOfStator : float
            The change in pitch from StatorBlade position around +Y axis and 
            passing through point **PitchCenterStator** that user must provide

        PitchCenterStator 3-float list
            Point around which **DeltaPitchStator** is applied in +Y direction
            
        ReDiscretizationOfRotorHubProfile : list of dict
            If provided, will rediscretize the rotor portion of the hub profile,
            following a set of dictionnaries as used by :py:func:`polyDiscretize`
    
        ReDiscretizationOfStatorHubProfile : list of dict
            If provided, will rediscretize the stator portion of the hub profile,
            following a set of dictionnaries as used by :py:func:`polyDiscretize`

    '''

    # TODO list : 
    # 1. re-eval nb of pts in azimut (strangely requests too high nb of points in airfoil)
    # 2. save rediscretized blade
    # 3. include option stator_only and rotor_only (useful during meshing iterations)
    # 4. investigate indirect mesh in INPRO case when stator nbptsTop2Bottom=9


    RotorBlade, StatorBlade = adjustPitchAndAzimutPositionBladesORAS(
        RotorBlade, RotorPitchCenter, RotorDeltaPitch,RotorThetaAdjustmentInDegrees,
        StatorBlade, StatorPitchCenter, StatorDeltaPitch, StatorThetaAdjustmentInDegrees)

    RotorMaxRadius = getMaxRadius(RotorBlade)

    saveInputGeometryORAS(LOCAL_DIRECTORY_CHECKME, RotorBlade, StatorBlade, HubProfile)    

    interface_support, rotor_edge, stator_edge = \
        prepareInterfaceORAS(RotorNumberOfBlades, StatorNumberOfBlades,
        RotorBlade, StatorBlade, InterfaceRelativePosition, FarfieldRadius,
        InterfaceRadialTensionRelativeToRotorRadius*RotorMaxRadius, LOCAL_DIRECTORY_CHECKME)

    profile_rotor, profile_stator = prepareSeparatedRotorAndStatorHubProfile(
        HubProfile, interface_support,
        RotorBlade, RotorNumberOfBlades, RotorAzimutalCellAngle, 
        RotorHgridXlocations, RotorHubProfileReDiscretization,
        StatorBlade, StatorNumberOfBlades, StatorAzimutalCellAngle,
        StatorHgridXlocations, StatorHubProfileReDiscretization, LOCAL_DIRECTORY_CHECKME)

    if RotorBuild:
        rotor_mesh = buildMeshStageORAS('ROTOR', RotorNumberOfBlades, RotorAzimutalCellAngle,
            RotorBlade, profile_rotor, RotorBladeWallCellHeight, RotorHubWallCellHeight,
            RotorRootWallRemeshRadialDistanceRelativeToMaxRadius, RotorRootRemeshRadialNbOfPoints,
            RotorBladeWallGrowthRate,RotorBladeRootWallNormalDistanceRelativeToRootChord,
            RotorBladeExtrusionParams, FarfieldRadius,
            raise_error_if_negative_volume_cells,
            RotorHgridNbOfPoints,
            RotorHspreadingAngles,
            RotorRadialExtrusionNbOfPoints,
            RotorTipScaleFactorAtRadialFarfield,
            RotorFarfieldRadialCellLengthRelativeToFarfieldRadius*FarfieldRadius,
            RotorRadialTensionRelativeToRotorRadius*RotorMaxRadius,
            RotorHgridXlocations,
            RotorFarfieldProfileAbscissaDeltas,
            RotorFarfieldTipSmoothIterations,
            None, stator_edge, None, interface_support,
            RotorBuildMatchMeshAdditionalParams, LOCAL_DIRECTORY_CHECKME)
    else:
        rotor_mesh = None

    if StatorBuild:
        stator_mesh = buildMeshStageORAS('STATOR', StatorNumberOfBlades, StatorAzimutalCellAngle,
            StatorBlade, profile_stator, StatorBladeWallCellHeight, StatorHubWallCellHeight,
            StatorRootWallRemeshRadialDistanceRelativeToMaxRadius, StatorRootRemeshRadialNbOfPoints,
            StatorBladeWallGrowthRate,StatorBladeRootWallNormalDistanceRelativeToRootChord,
            StatorBladeExtrusionParams, FarfieldRadius,
            raise_error_if_negative_volume_cells,
            StatorHgridNbOfPoints,
            StatorHspreadingAngles,
            StatorRadialExtrusionNbOfPoints,
            StatorTipScaleFactorAtRadialFarfield,
            StatorFarfieldRadialCellLengthRelativeToFarfieldRadius*FarfieldRadius,
            StatorRadialTensionRelativeToRotorRadius*RotorMaxRadius,
            StatorHgridXlocations,
            StatorFarfieldProfileAbscissaDeltas,
            StatorFarfieldTipSmoothIterations,
            rotor_edge, None, interface_support, None,
            StatorBuildMatchMeshAdditionalParams, LOCAL_DIRECTORY_CHECKME)
    else:
        stator_mesh = None

    t = setFamiliesAndConnectionToORASmesher(rotor_mesh, stator_mesh,
                                     RotorNumberOfBlades, StatorNumberOfBlades)

    return t

def discard_intersecting_sections(transition_sections):
    
    root_section = transition_sections[0]
    new_transition_sections = [root_section]
    for section in transition_sections[1:]:
        if not structured_surfaces_with_same_dimensions_intersect(new_transition_sections[-1], section):
            new_transition_sections += [ section ]
    return new_transition_sections

def structured_surfaces_with_same_dimensions_intersect(surface1, surface2):
    grid = G.stack([surface1, surface2])
    T._makeDirect(grid)
    G._getVolumeMap(grid)
    minimum_cell_volume = C.getMinValue(grid,'centers:vol')
    return minimum_cell_volume < 0

def getTangent(curve):
    x,y,z = J.getxyz(curve)
    tx, ty, tz = J.invokeFields(curve, ['tx','ty','tz'])
    dx = np.diff(x)
    dy = np.diff(y)
    dz = np.diff(z)
    norm = np.sqrt(dx*dx + dy*dy + dz*dz)
    tx[:] = np.hstack((0.,dx/norm))
    ty[:] = np.hstack((0.,dy/norm))
    tz[:] = np.hstack((0.,dz/norm))
    return tx, ty, tz

def maxRadius(zone, rotation_center, rotation_axis):
    zoneR = I.copyRef(zone)
    c = np.array(rotation_center,dtype=np.float64)
    a = np.array(rotation_axis,dtype=np.float64)
    W.addDistanceRespectToLine( zoneR , c, a, 'radius')
    return C.getMaxValue(zoneR,'radius')

def minRadius(zone, rotation_center, rotation_axis):
    zoneR = I.copyRef(zone)
    c = np.array(rotation_center,dtype=np.float64)
    a = np.array(rotation_axis,dtype=np.float64)
    W.addDistanceRespectToLine( zoneR , c, a, 'radius')
    return C.getMinValue(zoneR,'radius')

def makeORASinterfaceProfile(RotorBlade, StatorBlade, InterfaceRelativePosition,
                             FarfieldRadius, InterfaceRadialTension):
    center_and_axis = [0,0,0], [1,0,0] # in context of ORAS meshing

    trailing_edge_rotor = extractBladeEdge( RotorBlade,'x','max')
    leading_edge_stator = extractBladeEdge(StatorBlade,'x','min')

    W.extrapolateCurvesUpToSameRadius(trailing_edge_rotor, leading_edge_stator,
                                      *center_and_axis)
    bisector = W.bisector(trailing_edge_rotor, leading_edge_stator,
                          weight=InterfaceRelativePosition, N=500)
    z = J.getz(bisector)
    z[:] = 0
    
    bezier = W.addTangentCurveAtExtremumUpToRadius(bisector, FarfieldRadius*1.05, *center_and_axis,
                                          relative_tension=InterfaceRadialTension/FarfieldRadius)
    bisector,bezier = W.reDiscretizeCurvesWithSmoothTransitions([bisector,bezier])
    interface_profile = T.join(bisector, bezier)
    interface_profile[0] = 'interface_profile'

    return interface_profile, trailing_edge_rotor, leading_edge_stator


def makeORASinterfaceSupport(interface_profile, RotorNumberOfBlades,
                             StatorNumberOfBlades, delta_azimut=1.0, margin_azimut=20.0):
    azimut_angle = 360/np.minimum(RotorNumberOfBlades,StatorNumberOfBlades)+margin_azimut
    n_pts_azimut = int(azimut_angle//delta_azimut)
    surface = D.axisym(interface_profile, (0,0,0),(1,0,0),azimut_angle, n_pts_azimut)
    surface[0] = 'interface_surface'
    T._rotate(surface,(0,0,0),(1,0,0),-0.5*azimut_angle)

    return surface


def extractBladeEdge(blade, field_criterion='x', value_criterion='max'):
    '''
    Blade sections should be stacked following j indexing
    '''
    blade_main_surface = J.selectZoneWithHighestNumberOfPoints(blade)
    Nj = I.getZoneDim(blade_main_surface)[2]
    
    edge_indices = []
    for j in range(Nj):
        blade_section = GSD.getBoundary(blade_main_surface,'jmin',layer=j)
        field = J.getFieldOrCoordinate(blade_section, field_criterion)

        if value_criterion == 'max':
            index_of_section_at_blade_edge = np.argmax(field)
            
        elif value_criterion == 'min':
            index_of_section_at_blade_edge = np.argmin(field)

        else:
            raise AttributeError('unsupported value_criterion=%s'%value_criterion)

        edge_indices = [index_of_section_at_blade_edge]

    i_mean = (np.max(edge_indices) + np.min(edge_indices))//2

    BladeEdge = GSD.getBoundary(blade_main_surface,'imin',i_mean)
    BladeEdge[0] = 'BladeEdge'+value_criterion.capitalize()
    I._rmNodesByType(BladeEdge,'ZoneBC_t')

    return BladeEdge


def buildCurvedExternalSurfacesHgrid(blade, blade_number, spinner_middle, H_front_blade_reference,
        H_rear_blade_reference, H_grid_interior_spreading_angles,
        front_edge_extrapolation_factor=1.03, # FIXME parametrize
        rear_edge_extrapolation_factor=1.03, # FIXME parametrize
        ):

    front_edge, rear_edge = buildFrontAndRearEdgesUsingReferences(blade,
                spinner_middle, H_front_blade_reference, H_rear_blade_reference)

    original_distribution = I.copyTree(front_edge)

    W.prolongate(front_edge,front_edge_extrapolation_factor, True)
    W.prolongate(rear_edge, rear_edge_extrapolation_factor, True)
    W.discretizeInPlace(front_edge, Distribution=original_distribution)
    W.discretizeInPlace(rear_edge, Distribution=original_distribution)

    setVectorAtSurfaceUsingRotatedCurveTangentExtremum(spinner_middle,
        [front_edge, rear_edge], boundaries=['i','i'], indices=[0,-1])

    
    axialDeformationOfNormalsAtHubBoundaries(spinner_middle, H_grid_interior_spreading_angles)
    

    extrapolating_radius = getRadiusOfBladeTipCountourAtLastLayer(blade)

    sideB_surf = GSD.buildLateralFaceFromEdgesAndSupportSurface(spinner_middle,
                                 front_edge, rear_edge, 'jmax',
                                 extrapolating_radius, original_distribution)
    sideB_surf[0] = 'sideB.surf'

    sideA_surf = T.rotate(sideB_surf,[0,0,0],[1,0,0],360/blade_number)
    sideA_surf[0] = 'sideA.surf'

    front_surf = GSD.fillByTFIfromThreeSurfaces(spinner_middle, sideA_surf, sideB_surf,
                                                'imin','jmin','jmin', True)
    front_surf[0] = 'front.surf'

    rear_surf = GSD.fillByTFIfromThreeSurfaces(spinner_middle, sideA_surf, sideB_surf,
                                                   'imax', 'jmax', 'jmax', True)
    rear_surf[0] = 'rear.surf'

    return [front_surf, rear_surf, sideA_surf, sideB_surf]



def setVectorAtSurfaceUsingRotatedCurveTangentExtremum(surface, reference_curves,
                                            boundaries=['i','i'],
                                            indices=[0,-1],
                                            center=[0,0,0],
                                            axis=[1,0,0],
                                            vector_components=['sx','sy','sz']):
    reference_curves = I.getZones(reference_curves)
    if len(reference_curves) != len(boundaries) != len(indices):
        raise AttributeError('reference_curves, boundaries and indices  must be bijective')
    
    sx, sy, sz = J.invokeFields(surface,vector_components)
    x, y, z = J.getxyz(surface)
    aux_curves = []
    for direction, index, curve in zip(boundaries, indices, reference_curves):
        if direction == 'i':
            for j in range(sx.shape[1]):
                surface_X = np.array([x[index,j],y[index,j],z[index,j]])
                surface_AX = W.vectorOfLineToPoint(surface_X,axis,center)
                curve_AX = W.vectorOfLineToPoint(W.point(curve),axis,center)
                α = W.azimutal_angle_between_vectors(curve_AX, surface_AX, axis)

                aux_curve =I.copyTree(curve)
                T._rotate(aux_curve,center,axis, α)
                aux_curves += [aux_curve]
                t1 = W.tangentExtremum(aux_curve)

                sx[index,j] = t1[0]
                sy[index,j] = t1[1]
                sz[index,j] = t1[2]

        elif direction == 'j':
            
            for i in range(sx.shape[0]):
                P = np.array([x[i,index],y[i,index],z[i,index]])
                AP = W.vectorOfLineToPoint(P,axis,center)
                t0 = W.tangentExtremum(curve)
                α = W.azimutal_angle_between_vectors(AP, t0, axis)

                aux_curve =I.copyTree(curve)
                T._rotate(aux_curve,center,axis, α)
                t1 = W.tangentExtremum(aux_curve)

                sx[i,index] = t1[0]
                sy[i,index] = t1[1]
                sz[i,index] = t1[2]

        else:
            raise AttributeError('direction (1st item of each boundary) must be "i" or "j"')


def axialDeformationOfNormalsAtHubBoundaries(spinner_middle, H_grid_interior_spreading_angles):
    sx, sy = J.getVars(spinner_middle,['sx','sy'])
    for i in (0,-1):
        θ = -np.deg2rad(H_grid_interior_spreading_angles[i])
        cosθ = np.cos(θ)
        sinθ = np.sin(θ)
        sx[i,:] =  cosθ * sx[i,:] - sinθ * sy[i,:]
        sy[i,:] =  sinθ * sx[i,:] + cosθ * sy[i,:]

def _getCentralH_azimutpts_cell_sizes(external_surfaces):
    npts_azimut = I.getZoneDim(external_surfaces[0])[2]
    for surf in external_surfaces:
        if surf[0].startswith('sideA'):
            axial_edge = GSD.getBoundary(surf,'imax')
            central_first_cell = W.segment(axial_edge)
            central_last_cell  = W.segment(axial_edge,-1)
            radial_edge = GSD.getBoundary(surf,'jmin')
            tip_cell_length  = W.segment(radial_edge,-1)


    return npts_azimut, central_first_cell, central_last_cell, tip_cell_length


def getBisectorCurveAtSameRadiusWeightedByPoint(curve1, curve2, weighting_point,
                                                center=[0,0,0], axis=[1,0,0]):
    W.addDistanceRespectToLine([curve1,curve2],center,axis,'radius')
    W.addDistanceRespectToLine(curve1,center,axis,'radius')
    Rmin = C.getMinValue(curve1,'radius')
    Rmax = C.getMaxValue(curve1,'radius')



    split_curve = W.splitAtValue(curve2,'radius',Rmax)[0]
    split_curve = W.splitAtValue(split_curve,'radius',Rmin)
    nb_of_parts = len(split_curve)
    if nb_of_parts == 1:
        split_curve = split_curve[0]
    elif nb_of_parts == 2:
        split_curve = split_curve[1]
    else:
        J.save(split_curve,'debug.cgns')
        raise ValueError('unexpected number of split points when building reference front part, check debug.cgns')


    distance_between_curves = W.distance(W.point(split_curve), W.point(curve1))
    distance_to_point = W.distance(weighting_point, W.point(curve1))
    relative_distance_to_weighting_point = distance_to_point/distance_between_curves

    bisector_curve = W.bisector(curve1, split_curve, weight=relative_distance_to_weighting_point)
    bisector_curve = W.discretize(bisector_curve, Distribution=curve1)

    return bisector_curve

def buildFrontAndRearEdgesUsingReferences(blade, spinner_middle, H_front_blade_reference,
                                     H_rear_blade_reference):
    main_blade = J.selectZoneWithHighestNumberOfPoints(blade)
    blade_exterior = GSD.getBoundary(main_blade,'kmin') # kmin prevents from geom defaults

    front_edge = extractBladeEdge( blade_exterior,'x','min') 
    rear_edge = extractBladeEdge( blade_exterior,'x','max') 

    xs, ys, zs = J.getxyz(spinner_middle)
    spinner_mid_j = (xs.shape[1]-1)//2 

    if H_front_blade_reference:
        i=0
        spinner_front_point = np.array([xs[i,spinner_mid_j],
                                        ys[i,spinner_mid_j],
                                        zs[i,spinner_mid_j]], dtype=float)
        
        front_edge = getBisectorCurveAtSameRadiusWeightedByPoint(front_edge,
                        H_front_blade_reference, spinner_front_point)

    if H_rear_blade_reference:
        i=-1
        spinner_rear_point = np.array([xs[i,spinner_mid_j],
                                       ys[i,spinner_mid_j],
                                       zs[i,spinner_mid_j]], dtype=float)
        rear_edge = getBisectorCurveAtSameRadiusWeightedByPoint(rear_edge,
                        H_rear_blade_reference, spinner_rear_point)
        
    yf, zf  = J.getyz(front_edge)
    yr, zr  = J.getyz(rear_edge)
    yf[:] = 0.5*(yf+yr)
    zf[:] = 0.5*(zf+zr)
    yr[:] = yf
    zr[:] = zf

    front_edge[0] = 'front_edge'
    rear_edge[0] = 'rear_edge'

    return front_edge, rear_edge

def getMaxRadiusOfBladeWallFromExtrudedGrid(blade, center=[0,0,0], axis=[1,0,0]):

    blade_ref = I.copyRef(blade)
    walls = [GSD.getBoundary(z,'kmin') for z in I.getZones(blade_ref)]
    return getMaxRadius(walls, center=center, axis=axis)

def getRadiusOfBladeTipCountourAtLastLayer(blade, center=[0,0,0], axis=[1,0,0]):
    
    main_component = J.selectZoneWithHighestNumberOfPoints(blade)
    last_layer = GSD.getBoundary(main_component,'kmax')
    contour = GSD.getBoundary(last_layer,'jmax')
    barycenter = G.barycenter(contour)
    radius = W.distanceOfPointToLine(barycenter,axis,center)
    
    return radius
    
def getMaxRadius(t, center=[0,0,0], axis=[1,0,0]):
    t_ref = I.copyRef(t)
    W.addDistanceRespectToLine(t_ref,center,axis,'radius')
    return C.getMaxValue(t_ref,'radius')

def getBladeApproximateChordAtRoot(blade,profile):
    root_contour = getBladeApproximateContourAtRoot(blade, profile)
    x,y,z = J.getxyz(root_contour)
    npts = len(x)
    half = npts//2
    return np.sqrt( (x[half]-x[0])**2 + (y[half]-y[0])**2 + (z[half]-z[0])**2 )

def getBladeApproximateContourAtRoot(blade, profile):
    main_blade = J.selectZoneWithHighestNumberOfPoints(blade)
    Ni = I.getZoneDim(main_blade)[1]
    trailing_edge = GSD.getBoundary(main_blade,'imin')
    xTE, yTE = J.getxy(trailing_edge)
    leading_edge = GSD.getBoundary(main_blade,'imin',Ni//2)
    xLE, yLE = J.getxy(leading_edge)
    xmax_blade = xTE.max()
    xmin_blade = xLE.min()
    xp, yp = J.getxy(profile)
    root_region = (xp>xmin_blade) * (xp<xmax_blade)
    root_radius = 0.5*(yp[root_region].max()+yp[root_region].min())
    jTE = np.argmin(np.abs(yTE-root_radius))
    jLE = np.argmin(np.abs(yLE-root_radius))
    j_root = int(0.5*(jTE+jLE))
    return GSD.getBoundary(main_blade,'jmin',j_root)


def buildRearMonoblockSector(blade_number, external_surfaces, spinner_rear, rear_support):

    central_lateral_face = J.getZoneFromListByName(external_surfaces,'sideA.surf')
    central_rear_surf = J.getZoneFromListByName(external_surfaces,'rear.surf')
    central_top_curve = GSD.getBoundary(central_lateral_face,'imax')
    central_top_curve[0] = 'central_top_curve'

    rear_bottom_curve = GSD.getBoundary(spinner_rear,'jmin') 
    rear_bottom_curve[0] = 'rear_bottom_curve'
    rear_top_curve = W.getExtrapolationUpToGeometry(central_top_curve, rear_support,
                        direction=[1,0,0], relative_tension=0.5,
                        Distribution=rear_bottom_curve)
    W.discretizeInPlace(rear_top_curve,
        Distribution=dict(
            kind='tanhTwoSides',
            FirstCellHeight=W.segment(central_top_curve,-1),
            LastCellHeight=W.segment(rear_top_curve,-1)))
    rear_top_curve[0] = 'rear_top_curve'
    
    rear_front_curve = GSD.getBoundary(central_lateral_face,'jmax') 
    rear_front_curve[0] = 'rear_front_curve'
    rear_rear_curve = I.copyTree(rear_front_curve)
    rear_rear_curve[0] = 'rear_rear_curve'
    W.putCurveBetweenTwoPoints(rear_rear_curve, W.point(rear_bottom_curve,-1),
                                                W.point(rear_top_curve,-1))
    projected_rear = I.copyTree(rear_rear_curve)
    projected_rear[0] = 'projected_rear'
    T._projectOrtho(projected_rear, rear_support)
    W.discretizeInPlace(projected_rear, Distribution=rear_front_curve)
    T._projectOrtho(projected_rear, rear_support)
    W.matchExtremaOfCurveToExtremaOfOtherCurve(projected_rear, rear_rear_curve)


    rear_lateral_face = G.TFI([rear_front_curve, projected_rear,
                               rear_bottom_curve, rear_top_curve])
    rear_lateral_face[0] = 'rear_lateral_face'

    rear_lateral_face2 = T.rotate(rear_lateral_face,(0,0,0),(-1,0,0),360/blade_number)
    rear_lateral_face2[0]='rear_lateral_face2'

    rear_face = GSD.fillByTFIfromThreeSurfaces(spinner_rear, rear_lateral_face, rear_lateral_face2,
                                                'imax','imax','imax', True)
    rear_face[0] = 'rear_face'

    top_face = G.TFI([
        GSD.getBoundary(rear_lateral_face,'jmax'),
        GSD.getBoundary(rear_lateral_face2,'jmax'),
        GSD.getBoundary(central_rear_surf,'imax'),
        GSD.getBoundary(rear_face,'imax'),
    ])
    top_face[0] = 'top_face'

    surfs = [
        rear_lateral_face,rear_lateral_face2,
        central_rear_surf, rear_face,
        spinner_rear, top_face,
        ]

    grid_rear = G.TFI(surfs)
    grid_rear[0] = 'grid_rear'
    T._reorder(grid_rear,(1,-2,3))
    
    rear_top_curve[0] = 'spinner_union_0' # required by _extractWallAdjacentSectorFullProfile
    top_face[0] = 'TFI2_spinner_3' # required by _gatherSectorBoundaries
    wires = [rear_front_curve, projected_rear, rear_bottom_curve, rear_top_curve]
    grids = [grid_rear]

    return wires, surfs, grids


def buildFrontMonoblockSector(blade_number, external_surfaces, spinner_front, front_support):

    central_lateral_face = J.getZoneFromListByName(external_surfaces,'sideA.surf')
    central_front_surf = J.getZoneFromListByName(external_surfaces,'front.surf')
    central_top_curve = GSD.getBoundary(central_lateral_face,'imax')
    central_top_curve[0] = 'central_top_curve'
    W.reverse(central_top_curve,True)

    front_bottom_curve = GSD.getBoundary(spinner_front,'jmin') 
    front_bottom_curve[0] = 'front_bottom_curve'

    front_top_curve = W.getExtrapolationUpToGeometry(central_top_curve, front_support,
                        direction=[-1,0,0], relative_tension=0.5,
                        Distribution=front_bottom_curve)
    W.discretizeInPlace(front_top_curve,
        Distribution=dict(
            kind='tanhTwoSides',
            LastCellHeight=W.segment(front_top_curve,0),
            FirstCellHeight=W.segment(central_top_curve,0)))
    front_top_curve[0] = 'front_top_curve'
    W.reverse(front_top_curve, in_place=True)
    
    front_front_curve = GSD.getBoundary(central_lateral_face,'jmin') 
    front_front_curve[0] = 'front_front_curve'
    projected_front_ortho = I.copyTree(front_front_curve)
    projected_front_ortho[0] = 'projected_front_ortho'
    W.putCurveBetweenTwoPoints(projected_front_ortho, W.point(front_bottom_curve,0),
                                                W.point(front_top_curve,0))
    projected_front = I.copyTree(projected_front_ortho)
    projected_front[0] = 'projected_front'
    T._projectOrtho(projected_front, front_support)
    W.discretizeInPlace(projected_front, Distribution=front_front_curve)
    T._projectOrtho(projected_front, front_support)
    W.matchExtremaOfCurveToExtremaOfOtherCurve(projected_front, projected_front_ortho)


    front_lateral_face_wires = [front_front_curve, projected_front,
                                front_bottom_curve, front_top_curve]
    front_lateral_face = G.TFI(front_lateral_face_wires)
    front_lateral_face[0] = 'front_lateral_face'

    front_lateral_face2 = T.rotate(front_lateral_face,(0,0,0),(-1,0,0),360/blade_number)
    front_lateral_face2[0]='front_lateral_face2'


    front_face = GSD.fillByTFIfromThreeSurfaces(spinner_front, front_lateral_face, front_lateral_face2,
                                                'imin','imax','imax', True)
    front_face[0] = 'front_face'

    top_face = G.TFI([
        GSD.getBoundary(front_lateral_face,'jmax'),
        GSD.getBoundary(front_lateral_face2,'jmax'),
        GSD.getBoundary(central_front_surf,'imax'),
        GSD.getBoundary(front_face,'imax'),
    ])
    top_face[0] = 'top_face'

    surfs = [
        front_lateral_face,front_lateral_face2,
        central_front_surf, front_face,
        spinner_front, top_face,
        ]

    grid_front = G.TFI(surfs)
    grid_front[0] = 'grid_front'
    # T._reorder(grid_front,(1,-2,3))
    
    front_top_curve[0] = 'spinner_union_0' # required by _extractWallAdjacentSectorFullProfile
    top_face[0] = 'TFI2_spinner_0' # required by _gatherSectorBoundaries
    wires = front_lateral_face_wires
    grids = [grid_front]

    return wires, surfs, grids


def setFamiliesAndConnectionToORASmesher(rotor_mesh, stator_mesh,
                                     RotorNumberOfBlades, StatorNumberOfBlades):
    
    if rotor_mesh:
        rotor = _setFamiliesAndConnectionToRotorComponent(rotor_mesh, RotorNumberOfBlades)
        rotor_zones = I.getZones(rotor)
        for i, zone in enumerate(rotor_zones):
            zonename = zone[0]
            I._renameNode(rotor,zonename,f'rotor-{i}')
        C._tagWithFamily(rotor, 'Rotor')
    else:
        rotor_zones = []

    if stator_mesh:
        stator = _setFamiliesAndConnectionToStatorComponent(stator_mesh, StatorNumberOfBlades)
        stator_zones = I.getZones(stator)
        for i, zone in enumerate(stator_zones):
            zonename = zone[0]
            I._renameNode(stator,zonename,f'stator-{i}')
        C._tagWithFamily(stator, 'Stator')
    else:
        stator_zones = []

    t = C.newPyTree(['Base',rotor_zones+stator_zones])

    for stage in ['Rotor', 'Stator']:
        if stage == 'Rotor' and not rotor_mesh: continue
        if stage == 'Stator' and not stator_mesh: continue
        C._addFamily2Base(t, stage)
        for bc_family in ['Interface','Exterior','Hub','Blade']:
            C._addFamily2Base(t, stage+bc_family, bndType='UserDefined')

    return t

def _setFamiliesAndConnectionToRotorComponent(rotor_mesh, RotorNumberOfBlades):
  
    print('setting family tags to Rotor component...', end='')
    zones = I.getZones(rotor_mesh)

    def zonesStartingWith(names):
        picked_zones = []
        for name in names:
            for z in zones:
                if z[0].startswith(name):
                    picked_zones.append(z)
        return picked_zones

    t = C.newPyTree(['Base', zones])

    C._rmBCOfName(t, 'FamilySpecified:SPINNER')

    # Rotor/stator interface
    main_rear = J.getZoneFromListByName(zones,'main_rear')
    C._addBC2Zone(main_rear, 'RotorInterface', 'FamilySpecified:RotorInterface',
                             'kmax')

    grid_rear = J.getZoneFromListByName(zones,'grid_rear')
    C._addBC2Zone(grid_rear, 'RotorInterface', 'FamilySpecified:RotorInterface',
                             'jmin')
    
    # Exterior BC
    jmax_exterior_zone_startnames = ["main_rear","bulb_front","h_", "tip_"]
    for zone in zonesStartingWith(jmax_exterior_zone_startnames):
        C._addBC2Zone(zone, 'RotorExterior', 'FamilySpecified:RotorExterior', 'jmax')

    main_front = J.getZoneFromListByName(zones,'main_front')
    C._addBC2Zone(main_front, 'RotorExterior', 'FamilySpecified:RotorExterior', 'kmax')

    if not zonesStartingWith(['bulb']):
        main_rear = J.getZoneFromListByName(zones,'main_rear')
        C._addBC2Zone(main_rear, 'RotorExterior', 'FamilySpecified:RotorExterior', 'kmax')

        TFI3_spinner = J.getZoneFromListByName(zones,'TFI3_spinner')
        C._addBC2Zone(TFI3_spinner, 'RotorExterior', 'FamilySpecified:RotorExterior', 'jmin')


    # Hub
    kmin_hub_zone_startnames = ["grid_rear","TFI3_spinner","TFI3_bulb", "tfi."]
    for zone in zonesStartingWith(kmin_hub_zone_startnames):
        C._addBC2Zone(zone, 'RotorHub', 'FamilySpecified:RotorHub', 'kmin')

    blade = J.getZoneFromListByName(zones,'blade')
    C._addBC2Zone(blade, 'RotorHub', 'FamilySpecified:RotorHub', 'jmin')

    I._renameNode(t,'BLADE','RotorBlade')

    print(J.GREEN+'ok'+J.ENDC)

    print('setting grid connectivity Rotor component...', end='')
    t = X.connectMatch(t, tol=1e-3, dim=3)

    t = X.connectMatchPeriodic(t,[0,0,0],[360/RotorNumberOfBlades,0,0], tol=1e-8, dim=3)
    print(J.GREEN+'ok'+J.ENDC)

    return t


def _setFamiliesAndConnectionToStatorComponent(stator_mesh, StatorNumberOfBlades):
  
    print('setting family tags to Stator component...', end='')

    zones = I.getZones(stator_mesh)

    def zonesStartingWith(names):
        picked_zones = []
        for name in names:
            for z in zones:
                if z[0].startswith(name):
                    picked_zones.append(z)
        return picked_zones

    t = C.newPyTree(['Base', zones])

    C._rmBCOfName(t, 'FamilySpecified:SPINNER')

    # Rotor/stator interface
    main_front = J.getZoneFromListByName(zones,'main_front')
    C._addBC2Zone(main_front, 'StatorInterface', 'FamilySpecified:StatorInterface',
                             'imin')

    grid_front = J.getZoneFromListByName(zones,'grid_front')
    C._addBC2Zone(grid_front, 'StatorInterface', 'FamilySpecified:StatorInterface',
                             'jmax')
    
    # Exterior BC
    jmax_exterior_zone_startnames = ["main_rear","bulb_rear","h_", "tip_"]
    for zone in zonesStartingWith(jmax_exterior_zone_startnames):
        C._addBC2Zone(zone, 'StatorExterior', 'FamilySpecified:StatorExterior', 'jmax')

    C._addBC2Zone(main_front, 'StatorExterior', 'FamilySpecified:StatorExterior', 'kmax')

    if not zonesStartingWith(['bulb']):
        main_rear = J.getZoneFromListByName(zones,'main_rear')
        C._addBC2Zone(main_rear, 'StatorExterior', 'FamilySpecified:StatorExterior', 'kmax')

        TFI3_spinner = J.getZoneFromListByName(zones,'TFI3_spinner')
        C._addBC2Zone(TFI3_spinner, 'StatorExterior', 'FamilySpecified:StatorExterior', 'jmin')

    # Hub
    kmin_hub_zone_startnames = ["grid_front","TFI3_spinner","TFI3_bulb", "tfi."]
    for zone in zonesStartingWith(kmin_hub_zone_startnames):
        C._addBC2Zone(zone, 'StatorHub', 'FamilySpecified:StatorHub', 'kmin')

    blade = J.getZoneFromListByName(zones,'blade')
    C._addBC2Zone(blade, 'StatorHub', 'FamilySpecified:StatorHub', 'jmin')

    I._renameNode(t,'BLADE','StatorBlade')

    print(J.GREEN+'ok'+J.ENDC)


    print('setting grid connectivity Stator component...', end='')

    t = X.connectMatch(t, tol=1e-3, dim=3)

    t = X.connectMatchPeriodic(t,[0,0,0],[360/StatorNumberOfBlades,0,0], tol=1e-8, dim=3)

    print(J.GREEN+'ok'+J.ENDC)


    return t


def designBlade(
        RadiusTip = 0.60,
        RadiusRoot = 0.05,

        RightHandRuleRotation = True,


        BladeStackPointPositionInXaxis = 0.0,
        BladePitchAxisPositionInXaxis = 0.0,
        PitchAngle = 0.0,
        ZeroPitchAngleRelativeRadius = None, # if None, uses construction reference

        # Radial discretization of the blade geometry:
        RadialNbOfPoints = 51,
        RadialCellLengthAtTip = 0.0005,
        RadialCellLengthAtRoot = 0.01,

        # Geometrical Laws
        ChordDistribution = dict(
            RelativeSpan = [0.05/0.60,   0.45,  0.6,  1.0],
            Chord        = [0.07,  0.10, 0.10, 0.02],
            InterpolationLaw = 'akima'),

        TwistDistribution = dict(
            RelativeSpan = [0.05/0.60,  0.6,  1.0],
            Twist        = [30.0,  6.0, -1.0],
            InterpolationLaw = 'akima'),

        DihedralDistribution = dict(
            RelativeSpan = [0.05/0.60,    1.0],
            Dihedral        = [0.0, 0.0],
            InterpolationLaw = 'interp1d_linear'),

        SweepDistribution = dict(
            RelativeSpan = [0.05/0.60,    1.0],
            Sweep        = [0.0, 0.0],
            InterpolationLaw = 'interp1d_linear'),

        # Airfoil distributions
        SectionsDistribution = dict(
            RelativeSpan =   [0.05/0.60,    1.0],
            AirfoilZonesOrNACAstringsOrFilenames = ['NACA4416' , 'NACA4416'],
            TrailingEdgeSegmentLengthRelativeToChord = [0.004, 0.004],
            LeadingEdgeSegmentLengthRelativeToChord = [0.004, 0.004],
            LeadingEdgeAbscissa = [0.49, 0.49],
            StackingPointRelativeToChord = 0.25,
            TopSideNumberOfPoints = 67, # must be odd
            BottomSideNumberOfPoints = 67, # must be odd
            TopToBottomAtTipNumberOfPoints = 9,
            InterpolationLaw = 'interp1d_linear',
            ),
        ):
    '''
    Design a blade for propeller or ORAS meshing
    '''
    
    WingLikeAirfoilDist = getWingLikeAirfoilDistribution(SectionsDistribution)

    BladeDiscretization = dict(P1=(RadiusRoot,0,0),P2=(RadiusTip,0,0),
                            N=RadialNbOfPoints, kind='tanhTwoSides',
                            FirstCellHeight=RadialCellLengthAtRoot,
                            LastCellHeight=RadialCellLengthAtTip)

    blade_main_surface = GSD.wing(BladeDiscretization,
        ChordRelRef = SectionsDistribution['StackingPointRelativeToChord'],
        NPtsTrailingEdge = SectionsDistribution['TopToBottomAtTipNumberOfPoints'],
        AvoidAirfoilModification = True,
        Chord = ChordDistribution,
        Twist =  TwistDistribution,
        Dihedral =  DihedralDistribution,
        Sweep =  SweepDistribution,
        Airfoil =  WingLikeAirfoilDist)[1]
    blade_main_surface[0] = 'blade'

    if not RightHandRuleRotation:
        x = J.getx(blade_main_surface)
        x *= -1
        T._reorder(blade_main_surface,(-1,2,3))

    blade = GSD.closeWingTipAndRoot(blade_main_surface, tip_window='jmax',close_root=False,
            airfoil_top2bottom_NPts=SectionsDistribution['TopToBottomAtTipNumberOfPoints'])

    blade_input_frenet = ((0, 0,-1), (0, 1, 0),(1, 0, 0))
    final_frenet = ((0,1,0), (-1,0,0), (0,0,1))
    T._rotate(blade, (0,0,0), blade_input_frenet, final_frenet)
    T._translate(blade, (BladeStackPointPositionInXaxis,0,0))
    
    pitch_sign = 1 if RightHandRuleRotation else -1
    if ZeroPitchAngleRelativeRadius is None:
        if not PitchAngle: return blade
        T._rotate(blade, (BladePitchAxisPositionInXaxis,0,0), (0,1,0), pitch_sign*PitchAngle)
    else:
        twist_ref = interpolate__(ZeroPitchAngleRelativeRadius, TwistDistribution['RelativeSpan'],
                TwistDistribution['Twist'], 'interp1d_linear')
        T._rotate(blade, (BladePitchAxisPositionInXaxis,0,0), (0,1,0), pitch_sign*(PitchAngle-twist_ref))
   
    
    return blade




def getWingLikeAirfoilDistribution(SectionsDistribution):
    checkAirfoilDistributionCoherency(SectionsDistribution)

    nb_of_sections = len(SectionsDistribution['RelativeSpan'])
    
    AirfoilZones = []
    for i in range(nb_of_sections):
        RequestedAirfoil = SectionsDistribution['AirfoilZonesOrNACAstringsOrFilenames'][i]
        AirfoilZone = W.loadAirfoilInSafeMode(RequestedAirfoil)

        AirfoilDistribution = [
            dict(N=SectionsDistribution['BottomSideNumberOfPoints'],
                BreakPoint=SectionsDistribution['LeadingEdgeAbscissa'][i],
                kind='tanhTwoSides',
                FirstCellHeight=SectionsDistribution['TrailingEdgeSegmentLengthRelativeToChord'][i],
                LastCellHeight=SectionsDistribution['LeadingEdgeSegmentLengthRelativeToChord'][i]),
            dict(N=SectionsDistribution['TopSideNumberOfPoints'],
                BreakPoint=1.0,
                kind='tanhTwoSides',
                FirstCellHeight=SectionsDistribution['LeadingEdgeSegmentLengthRelativeToChord'][i],
                LastCellHeight=SectionsDistribution['TrailingEdgeSegmentLengthRelativeToChord'][i])]

        AirfoilZone = W.polyDiscretize(AirfoilZone, AirfoilDistribution)
        AirfoilZones += [ AirfoilZone ]

    AirfoilDict = dict(
        RelativeSpan=SectionsDistribution['RelativeSpan'],
        Airfoil = AirfoilZones,
        InterpolationLaw = SectionsDistribution['InterpolationLaw'])
    
    return AirfoilDict



def checkAirfoilDistributionCoherency(AirfoilDistribution):

    same_length_keys = ['RelativeSpan', 'AirfoilZonesOrNACAstringsOrFilenames',
        'TrailingEdgeSegmentLengthRelativeToChord',
        'LeadingEdgeSegmentLengthRelativeToChord',
        'LeadingEdgeAbscissa']

    mandatory_keys = same_length_keys+['StackingPointRelativeToChord',
        'TopSideNumberOfPoints', 'BottomSideNumberOfPoints']
    
    for k in mandatory_keys:
        if k not in AirfoilDistribution:
            raise KeyError(f'User shall provide key {k} into AirfoilDistribution dict')
        
    nb_of_sections = len(AirfoilDistribution['RelativeSpan'])
    if nb_of_sections < 2:
        raise AttributeError(f'User shall provide at least 2 sections in AirfoilDistribution (found {nb_of_sections})')
    
    for k in same_length_keys:
        len_key = len(AirfoilDistribution[k])
        if nb_of_sections != len_key:
            raise AttributeError(f'User specified {nb_of_sections} sections in RelativeSpan, so it is expected also {nb_of_sections} items for {k}, but got {len_key}')


def proposeHgridXlocations(blade, profile, x_chord_ratio=0.25):
    main_blade = J.selectZoneWithHighestNumberOfPoints(blade)
    root = getBladeApproximateContourAtRoot(main_blade, profile)
    x = J.getx(root)
    xmin = x.min()
    xmax = x.max()
    xchord = xmax-xmin
    x_loc_min = xmin - x_chord_ratio*xchord
    x_loc_max = xmax + x_chord_ratio*xchord
    return x_loc_min, x_loc_max


def getHgridNPts(blade, NumberOfBlades, AzimutalCellAngle, label=''):

    ncell_azimut = int((360/NumberOfBlades)/AzimutalCellAngle)
    if ncell_azimut %2 != 0: ncell_azimut+=1

    blade_main_surface = J.selectZoneWithHighestNumberOfPoints( blade )
    _,Ni,_,_,_=I.getZoneDim(blade_main_surface)
    Nb_segments_airfoil = Ni - 1
    nb_seg_side = Nb_segments_airfoil//2
    Hgrid_NPts = (nb_seg_side - ncell_azimut) +1

    if Hgrid_NPts < 9:
        msg = "insuficient number of airfoil segments "
        if label: msg += f'in {label} '
        msg += f'({Nb_segments_airfoil} total, {nb_seg_side} per side)\ncompared to azimut points ({ncell_azimut})\n'
        airfoil_min_npts = 7+ncell_azimut +1
        if airfoil_min_npts%2==0: airfoil_min_npts+=1
        msg += f'you must set at least {airfoil_min_npts} points per airfoil side, or reduce the nb of azimut pts'

        raise ValueError(msg)
    
    return Hgrid_NPts

def getSimpleORASHubProfileDiscretizations(rotor, stator,
        RotorNumberOfBlades=13, StatorNumberOfBlades=11,

        AzimutalCellAngle=1.0,
        InterfaceAxialCellLength=1.5e-2,
        BreakPointsAxialCellLength=1.5e-2,
        
        # rotor hub profile discretization
        RotorHgridXlocations=(-1.50, -0.75),
        RotorFrontNPts=30,
        RotorRearNPts=9,
        RotorFrontSegmentLength=1e-2,
        
        # stator hub profile discretization
        StatorHgridXlocations=(-0.45, 0.20),
        StatorFrontNPts=9,
        StatorRearNPts=60,
        StatorRearSegmentLength=1e-2
        ):
    
    RotorHgridNPts = getHgridNPts(rotor, RotorNumberOfBlades, AzimutalCellAngle, 'rotor')

    RotorHubProfileReDiscretization = [
        {'N':RotorFrontNPts,
         'BreakPoint(x)':RotorHgridXlocations[0],
         'FirstCellHeight':RotorFrontSegmentLength,
         'LastCellHeight':BreakPointsAxialCellLength,
         'kind':'tanhTwoSides'},

        {'N':RotorHgridNPts,
         'BreakPoint(x)':RotorHgridXlocations[1],
         'FirstCellHeight':BreakPointsAxialCellLength,
         'LastCellHeight':BreakPointsAxialCellLength,
         'kind':'tanhTwoSides'},

        {'N':RotorRearNPts,
         'BreakPoint':1,
         'FirstCellHeight':BreakPointsAxialCellLength,
         'LastCellHeight':InterfaceAxialCellLength,
         'kind':'tanhTwoSides'}]

    StatorHgridNPts = getHgridNPts(stator, StatorNumberOfBlades, AzimutalCellAngle, 'stator')

    StatorHubProfileReDiscretization = [
        {'N':StatorFrontNPts,
         'BreakPoint(x)':StatorHgridXlocations[0],
         'FirstCellHeight':InterfaceAxialCellLength,
         'LastCellHeight':BreakPointsAxialCellLength,
         'kind':'tanhTwoSides'},

        {'N':StatorHgridNPts,
         'BreakPoint(x)':StatorHgridXlocations[1],
         'FirstCellHeight':BreakPointsAxialCellLength,
         'LastCellHeight':BreakPointsAxialCellLength,
         'kind':'tanhTwoSides'},

        {'N':StatorRearNPts,
         'BreakPoint':1,
         'FirstCellHeight':BreakPointsAxialCellLength,
         'LastCellHeight':StatorRearSegmentLength,
         'kind':'tanhTwoSides'}]
    

    return RotorHubProfileReDiscretization, StatorHubProfileReDiscretization

def splitAndDiscretizeProfileForHgridRegion(profile, HGridNPts, HgridXlocations):
    front_original_dist = W.splitAt(profile, HgridXlocations[0], 'CoordinateX')[0]
    rear_original_dist = W.splitAt(profile, HgridXlocations[1], 'CoordinateX')[1]
    front, rest = W.splitAtValue(profile, 'CoordinateX', HgridXlocations[0])
    middle, rear = W.splitAtValue(rest, 'CoordinateX', HgridXlocations[1])


    W.discretizeInPlace(front, Distribution=front_original_dist)
    W.discretizeInPlace(rear, Distribution=rear_original_dist)

    middle_first_segment = W.segment(front,-1)
    middle_last_segment = W.segment(rear,0)
    W.discretizeInPlace(middle, N=HGridNPts, Distribution=dict(
        kind='tanhTwoSides', FirstCellHeight=middle_first_segment, LastCellHeight=middle_last_segment))
    HubProfile = W.joinSequentially([front,middle,rear])
    HubProfile[0] = 'HubProfile'

    return HubProfile


def _splitHubAtHgrid(spinner, HgridXlocations):
    main_surf = J.selectZoneWithHighestNumberOfPoints(spinner)


    _,Ni,Nj,_,_ = I.getZoneDim(main_surf)
    bnd = GSD.getBoundary(main_surf,'jmin')
    x = J.getx(bnd)
    try:
        imin_cut = np.argmin((x-HgridXlocations[0])**2) + 1
    except TypeError as e:
        raise TypeError(J.FAIL+f"{HgridXlocations=}"+J.ENDC) from e
    imax_cut = np.argmin((x-HgridXlocations[1])**2) + 1
    
    reverse_order = imin_cut > imax_cut
    if reverse_order:
        imin_cut, imax_cut = imax_cut, imin_cut

        rear = T.subzone(main_surf,(1,1,1),(imin_cut,Nj,1))
        middle = T.subzone(main_surf,(imin_cut,1,1),(imax_cut,Nj,1))
        front = T.subzone(main_surf,(imax_cut,1,1),(Ni,Nj,1))
        T._reorder([front,middle,rear],(-1,-2,3))

    else:
        front = T.subzone(main_surf,(1,1,1),(imin_cut,Nj,1))
        middle = T.subzone(main_surf,(imin_cut,1,1),(imax_cut,Nj,1))
        rear = T.subzone(main_surf,(imax_cut,1,1),(Ni,Nj,1))

    front[0] = 'main.hub.front'
    middle[0] = 'main.hub.middle'
    rear[0] = 'main.hub.rear'

    return front, middle, rear

def getBladesORAS_ONERA_SE(
        RotorRadialNbOfPoints = 51,
        RotorRadialCellLengthAtTip = 0.0005,
        RotorRadialCellLengthAtRoot = 0.05,
        RotorAirfoilSideNumberOfPoints=67,

        StatorRadialNbOfPoints = 51,
        StatorRadialCellLengthAtTip = 0.0005,
        StatorRadialCellLengthAtRoot = 0.05,
        StatorAirfoilSideNumberOfPoints=67):
    '''
    Build the blades geometry of the ORAS configuration extracted from: 

    A. Dumont "Design of the open-rotor engine ONERA-SE for research
    application in aero-acoustic optimization" (2024) ISABE conference paper
    '''


    RotorRmax = 4.267 * 0.5  # FIXME double Check
    
    RotorRootExtrapolationCoefficientForIntersectingHubProfile = 1.1
    RotorRmin = 0.275*RotorRmax /RotorRootExtrapolationCoefficientForIntersectingHubProfile

    rotor = designBlade(
        RadiusTip = RotorRmax,
        RadiusRoot = RotorRmin,

        RightHandRuleRotation = True,


        BladeStackPointPositionInXaxis = -1.25, # FIXME check with Antoine
        BladePitchAxisPositionInXaxis = -1.25,  # FIXME check with Antoine
        PitchAngle = 65.9, # FIXME set # 65.9° cruise 43.6° take-off
        ZeroPitchAngleRelativeRadius = None, # if None, uses construction reference

        # Radial discretization of the blade geometry:
        RadialNbOfPoints = RotorRadialNbOfPoints,
        RadialCellLengthAtTip = RotorRadialCellLengthAtTip,
        RadialCellLengthAtRoot = RotorRadialCellLengthAtRoot,


        # Geometrical Laws
        ChordDistribution = dict(
            RelativeSpan = np.array([0.03839114, 0.34114505, 0.61861204, 0.93517084,
                1.30873036, 1.60835724, 1.85638663, 2.02534841, 2.18088164])/RotorRmax,
            Chord        = [0.45008235, 0.52006151, 0.5508826,  0.545311,
                0.49570133, 0.43044444, 0.35901636, 0.29877573, 0.2259112],
            InterpolationLaw = 'interp1d_quadratic'),

        TwistDistribution = dict(
            RelativeSpan = np.array([0.58349825, 0.83149217, 1.090716, 1.2321457,
                1.50072779, 1.74455578, 1.87524668, 2.0030926, 2.12628958])/RotorRmax,
            Twist        = [18.90041593, 14.91054365, 9.98450021, 7.30688174,
                2.08891397, -2.95217354, -6.05264908, -9.5406416, -13.6099662],
            InterpolationLaw = 'akima'),

        DihedralDistribution = dict(
            RelativeSpan = [RotorRmin/RotorRmax,    1.0],
            Dihedral        = [0.0, 0.0],
            InterpolationLaw = 'interp1d_linear'),

        SweepDistribution = dict(
            RelativeSpan = np.array([0.58056285, 0.74403625, 0.94612246, 1.09738395,
                1.19428129, 1.34516226, 1.49912986, 1.62988249, 1.83635547,
                1.99665936, 2.12979761])/RotorRmax,
            Sweep        = -np.array([0.00162137, 0.09976991, 0.17632323, 0.19922474,
                0.19894292, 0.18075015, 0.14496677, 0.09952667, -0.00020738,
                -0.10158806, -0.2005527]),
            InterpolationLaw = 'cubic'),

        # Airfoil distributions
        SectionsDistribution = dict(
            RelativeSpan =   [0.27130046, 0.3499613, 0.54454126, 0.68016831, 0.7756235, 1],
            AirfoilZonesOrNACAstringsOrFilenames = [
                W.getAirfoil_NASA_SC_2_0412(),
                W.getAirfoil_NASA_SC_2_0410(),
                W.getAirfoil_NASA_SC_2_0406(),
                W.getAirfoil_NASA_SC_2_0404(),
                W.getAirfoil_NASA_SC_2_0403(),
                W.getAirfoil_NASA_SC_2_0403()],
            TrailingEdgeSegmentLengthRelativeToChord = 6*[0.004],
            LeadingEdgeSegmentLengthRelativeToChord = [
                0.0010, 0.0008, 0.0005, 0.0002, 0.0001, 0.0001],
            LeadingEdgeAbscissa = [
                0.4985, 0.4990, 0.4990, 0.4990, 0.5000, 0.5000],

            StackingPointRelativeToChord = 0.25,
            TopSideNumberOfPoints = RotorAirfoilSideNumberOfPoints, # must be odd
            BottomSideNumberOfPoints = RotorAirfoilSideNumberOfPoints, # must be odd
            TopToBottomAtTipNumberOfPoints = 9,
            InterpolationLaw = 'interp1d_linear',
            )
    )  


    StatorRootExtrapolationCoefficientForIntersectingHubProfile = 1.05
    StatorRmax = 3.631 * 0.5  # FIXME double Check
    StatorRmin = 0.275 * RotorRmax/StatorRootExtrapolationCoefficientForIntersectingHubProfile

    stator = designBlade(
        RadiusTip = StatorRmax,
        RadiusRoot = StatorRmin,

        RightHandRuleRotation = True,


        BladeStackPointPositionInXaxis = -0.25, # FIXME check with Antoine
        BladePitchAxisPositionInXaxis = -0.25,  # FIXME check with Antoine
        PitchAngle = 82.7, # FIXME set # 82.7° cruise 80.7° take-off
        ZeroPitchAngleRelativeRadius = None, # if None, uses construction reference

        # Radial discretization of the blade geometry:
        RadialNbOfPoints = StatorRadialNbOfPoints,
        RadialCellLengthAtTip = StatorRadialCellLengthAtTip,
        RadialCellLengthAtRoot = StatorRadialCellLengthAtRoot,


        # Geometrical Laws
        ChordDistribution = dict(
            RelativeSpan = np.array([0.6283272, 0.78723261, 1.00159194, 1.18042266,
                1.35664294, 1.50451435, 1.64031922, 1.81237597])/StatorRmax,
            Chord        = [0.45214494, 0.49009969, 0.51048502, 0.50000218,
                0.4624581, 0.40999585, 0.34652996, 0.22613599],
            InterpolationLaw = 'interp1d_quadratic'),

        TwistDistribution = dict(
            RelativeSpan = np.array([0.61842972, 0.71605893, 0.82777895, 0.94352492,
                0.99888168, 1.07537467, 1.19105244, 1.28690501, 1.36478522,
                1.42589122, 1.47621382, 1.53612167, 1.58595418, 1.65506201,
                1.72278769, 1.81262787])/StatorRmax,
            Twist        = [ 0.67999885, 0.43298977,  0.22550214,  0.09047051,
                0.06082942, 0.06082942, 0.08184258, 0.06027904, -0.00833222,
                -0.1141896,  -0.2572931,  -0.48665075, -0.74253671, -1.13601443,
                -1.58376494, -2.24860661],
            InterpolationLaw = 'akima'),

        DihedralDistribution = dict(
            RelativeSpan = [StatorRmin/StatorRmax,    1.0],
            Dihedral        = [0.0, 0.0],
            InterpolationLaw = 'interp1d_linear'),

        SweepDistribution = dict(
            RelativeSpan = np.array([0.61924101, 0.7577755,  0.85066569, 0.96517686,
                1.07278269, 1.23108786, 1.37065794, 1.4999471,
                1.63526293, 1.81336165])/StatorRmax,
            Sweep        = -np.array([0.00061484, 0.06610651, 0.08928049, 0.10162315,
                0.09344366, 0.06723515, 0.02792832, -0.02290787,
                -0.08985667, -0.20082872]),
            InterpolationLaw = 'cubic'),

        # Airfoil distributions
        SectionsDistribution = dict(
            RelativeSpan =   [0.27130046, 0.3499613, 0.54454126, 0.68016831, 0.7756235, 1],
            AirfoilZonesOrNACAstringsOrFilenames = [
                W.getAirfoil_NASA_SC_2_0410(), # FIXME double check
                W.getAirfoil_NASA_SC_2_0410(),
                W.getAirfoil_NASA_SC_2_0406(),
                W.getAirfoil_NASA_SC_2_0404(),
                W.getAirfoil_NASA_SC_2_0403(),
                W.getAirfoil_NASA_SC_2_0403()],
            TrailingEdgeSegmentLengthRelativeToChord = 6*[0.004],
            LeadingEdgeSegmentLengthRelativeToChord = [
                0.0008, 0.0008, 0.0005, 0.0002, 0.0001, 0.0001],
            LeadingEdgeAbscissa = [
                0.4990, 0.4990, 0.4990, 0.4990, 0.5000, 0.5000],

            StackingPointRelativeToChord = 0.25,
            TopSideNumberOfPoints = StatorAirfoilSideNumberOfPoints, # must be odd
            BottomSideNumberOfPoints = StatorAirfoilSideNumberOfPoints, # must be odd
            TopToBottomAtTipNumberOfPoints = 9,
            InterpolationLaw = 'interp1d_linear',
            )
    )  

    return rotor, stator

def getHubProfileORAS_ONERA_SE():

    bezier_pts = [
        (-2.7000000000000000,0.0,0),
        (-2.6976262207607693,0.5003924846883171,0),
        (-2.229549849879404,0.5809392845644833,0),
        (-1.6778309560438065,0.5808324846960423,0),

        (-1.6778309560438065,0.5808324846960423,0),
        (-1.4307118882078287,0.5816481614489555,0),
        (-1.502325057810188,0.551365667106162,0),
        (-1.2473574798293745,0.551365667106162,0),

        (-1.2473574798293745,0.551365667106162,0),
        (-1.0967228817002745,0.551365667106162,0),
        (-0.8981130129034307,0.6206294118001975,0),
        (-0.6963471318324306,0.6206294118001975,0),

        (-0.6963471318324306,0.6206294118001975,0),
        (-0.3981910727598996,0.6206294118001975,0),
        (-0.451287357252268,0.6010038239297818,0),
        (-0.25278893984233664,0.6014126903437489,0),

        (-0.25278893984233664,0.6014126903437489,0),
        (-0.0568255764847736,0.6009770063552681,0),
        (0.09593117914870675,0.6517117720686091,0),
        (0.250115568012407,0.6511758414448765,0),

        (0.250115568012407,0.6511758414448765,0),
        (0.6516740694594296,0.6509959425706399,0),
        (0.9020416781260892,0.8222244732381354,0),
        (1.5001298098941005,0.8216327768603116,0),

        (1.5001298098941005,0.8216327768603116,0),
        (2.504222073695696,0.822016667345834,0),
        (3.929648935342521,0.07179152820182805,0),
        (4,0,0)]

    # not working due to https://github.com/onera/Cassiopee/issues/210
    # polyline = D.polyline(bezier_pts)
    # bezier = D.bezier(polyline,N=100)
    # bezier[0] = 'HubProfileONERA_SE'

    nb_bz_pts = len(bezier_pts)

    xy_pts = [ [p[0], p[1]] for p in bezier_pts ]

    curves = []
    for i in range(0,nb_bz_pts,4):
        x,y = W.bezier_curve_2D(xy_pts[i:i+4])
        bezier = J.createZone('HubProfile',[x,y,x*0],['x','y','z'])
        W.reverse(bezier, in_place=True)
        curves += [bezier]

    bezier = W.joinSequentially(curves)

    return bezier


def saveInputGeometryORAS(DIR_CHECKME, RotorBlade, StatorBlade, HubProfile):
    if not DIR_CHECKME: return
    for stage in ['ROTOR', 'STATOR']:
        try: os.makedirs(os.path.join(DIR_CHECKME,stage))
        except: pass

    t_in = J.tree(ROTOR_0=RotorBlade, STATOR_0=StatorBlade, HUB_PROFILE_0=HubProfile)
    J.save(t_in, os.path.join(DIR_CHECKME,'0_input_geometry.cgns'))


def prepareInterfaceORAS(RotorNumberOfBlades, StatorNumberOfBlades,
        RotorBlade, StatorBlade, InterfaceRelativePosition, FarfieldRadius,
        InterfaceRadialTension, DIR_CHECKME):
    interface_profile, rotor_edge, stator_edge = makeORASinterfaceProfile(
        RotorBlade, StatorBlade, InterfaceRelativePosition, FarfieldRadius,
        InterfaceRadialTension)

    interface_support = makeORASinterfaceSupport(interface_profile, RotorNumberOfBlades,
                                                 StatorNumberOfBlades)
    
    if DIR_CHECKME:
        t_int = J.tree(INTERFACE_SUPPORT_1=interface_support,
                       ROTOR_STATOR_EDGES_1=[rotor_edge, stator_edge])
        J.save(t_int, os.path.join(DIR_CHECKME,'1_interface_support.cgns'))


    return interface_support, rotor_edge, stator_edge

def adjustPitchAndAzimutPositionBladesORAS(
        RotorBlade, RotorPitchCenter, RotorDeltaPitch,RotorThetaAdjustmentInDegrees,
        StatorBlade, StatorPitchCenter, StatorDeltaPitch,StatorThetaAdjustmentInDegrees):
    
    # pitch
    RotorBlade = T.rotate(RotorBlade,(RotorPitchCenter,0,0),(0,1,0),RotorDeltaPitch)
    StatorBlade = T.rotate(StatorBlade,(StatorPitchCenter,0,0),(0,1,0),StatorDeltaPitch)
    
    # azimut
    T._rotate(RotorBlade,(0,0,0),(1,0,0),RotorThetaAdjustmentInDegrees)
    T._rotate(StatorBlade,(0,0,0),(1,0,0),StatorThetaAdjustmentInDegrees)

    return RotorBlade, StatorBlade

def prepareSeparatedRotorAndStatorHubProfile(HubProfile, interface_support,
        RotorBlade, RotorNumberOfBlades, RotorAzimutalCellAngle, 
        RotorHgridXlocations, RotorHubProfileReDiscretization,
        StatorBlade, StatorNumberOfBlades, StatorAzimutalCellAngle,
        StatorHgridXlocations, StatorHubProfileReDiscretization, DIR_CHECK):
    profile_rotor, profile_stator = W.cut(HubProfile, interface_support)
    profile_rotor[0] = 'profile_rotor'
    profile_stator[0] = 'profile_stator'


    profile_rotor = W.polyDiscretize( profile_rotor, RotorHubProfileReDiscretization )
    RotorHgridNPts = getHgridNPts(RotorBlade, RotorNumberOfBlades, RotorAzimutalCellAngle)
    print(f'RotorHgridNPts={RotorHgridNPts}')
    profile_rotor = splitAndDiscretizeProfileForHgridRegion(profile_rotor, RotorHgridNPts, RotorHgridXlocations)
    
    profile_stator = W.polyDiscretize( profile_stator, StatorHubProfileReDiscretization )
    StatorHgridNPts = getHgridNPts(StatorBlade, StatorNumberOfBlades, StatorAzimutalCellAngle)
    print(f'StatorHgridNPts={StatorHgridNPts}')
    profile_stator = splitAndDiscretizeProfileForHgridRegion(profile_stator, StatorHgridNPts, StatorHgridXlocations)


    if DIR_CHECK:
        t_prof = J.tree(PROFILES_2=[profile_rotor,profile_stator])
        J.save(t_prof, os.path.join(DIR_CHECK,'2_profiles.cgns'))

    return profile_rotor, profile_stator

def buildMeshStageORAS(StageName, NumberOfBlades, AzimutalCellAngle,
        Blade, profile, BladeWallCellHeight, HubWallCellHeight,
        RootWallRemeshRadialDistanceRelativeToMaxRadius, RootRemeshRadialNbOfPoints,
        BladeWallGrowthRate,BladeRootWallNormalDistanceRelativeToRootChord,
        BladeExtrusionParams, FarfieldRadius,
        raise_error_if_negative_volume_cells,
        HgridNbOfPoints,
        HspreadingAngles,
        RadialExtrusionNbOfPoints,
        TipScaleFactorAtRadialFarfield,
        FarfieldRadialCellLength,
        RadialTension,
        HgridXlocations,
        FarfieldProfileAbscissaDeltas,
        FarfieldTipSmoothIterations,
        front_edge,
        rear_edge,
        front_support,
        rear_support,
        BuildMatchMeshAdditionalParams,
        DIR_CHECK):


    ncell_azimut = int((360/NumberOfBlades)/AzimutalCellAngle)
    if ncell_azimut %2 != 0: ncell_azimut+=1
    print(f'{StageName} number of cells in azimut direction: {ncell_azimut}')

    Rmax = getMaxRadius(Blade)
    print(f'{StageName} computed maximum radius: {Rmax}')

    RootChord = getBladeApproximateChordAtRoot(Blade,profile)
    print(f'{StageName} approximated chord at root: {RootChord}')

    hub = makeHub(profile, NumberOfBlades,
                    rotation_axis=(1,0,0),
                    number_of_cells_azimuth=ncell_azimut)

    if DIR_CHECK:
        t_hub = J.tree(**{"HUB_%s_3"%StageName:hub})
        J.save(t_hub, os.path.join(DIR_CHECK,StageName,'3_hub.cgns'))
        checkme_dir = os.path.join(DIR_CHECK,StageName)
    else:
        checkme_dir = ''
    
    extruded = extrudeBladeSupportedOnSpinner(Blade, hub,
        (0,0,0), (1,0,0), BladeWallCellHeight,
        spinner_wall_cell_height=HubWallCellHeight,
        root_to_transition_distance=RootWallRemeshRadialDistanceRelativeToMaxRadius*Rmax,
        root_to_transition_number_of_points=RootRemeshRadialNbOfPoints,
        distribution_growth_rate=BladeWallGrowthRate,
        maximum_extrusion_distance_at_spinner=BladeRootWallNormalDistanceRelativeToRootChord*RootChord,
        DIRECTORY_CHECKME=checkme_dir,
        raise_error_if_negative_volume_cells=raise_error_if_negative_volume_cells,
        **BladeExtrusionParams)
    
    blade_last_layer = [GSD.getBoundary(z,'kmax') for z in I.getZones(extruded) ]

    if DIR_CHECK:
        t_check = J.tree(BLADE_LAST_LAYER_4=blade_last_layer)
        J.save(t_check, os.path.join(DIR_CHECK,StageName,'4_blade_last_layer.cgns'))

    max_radius = maxRadius(extruded,(0,0,0),(-1,0,0))
    RadialExtrusionDistance = FarfieldRadius - max_radius
    if RadialExtrusionDistance < 0:
        raise AttributeError(f'FarfieldRadius must be greater than the {StageName} max radius ({max_radius})')

    stage_mesh = buildMatchMesh(hub, extruded, NumberOfBlades,
        rotation_axis=(-1,0,0),
        rotation_center=(0,0,0),
        distance=RadialExtrusionDistance,
        max_radius=FarfieldRadius,
        H_grid_interior_points=HgridNbOfPoints,
        H_grid_interior_spreading_angles=HspreadingAngles,
        number_of_points=RadialExtrusionNbOfPoints,
        tip_axial_scaling_at_farfield=TipScaleFactorAtRadialFarfield,
        farfield_cell_height=FarfieldRadialCellLength,
        normal_tension=RadialTension,
        HgridXlocations=HgridXlocations,
        FarfieldProfileAbscissaDeltas=FarfieldProfileAbscissaDeltas,
        FarfieldTipSmoothIterations=FarfieldTipSmoothIterations,
        H_front_blade_reference=front_edge,
        H_rear_blade_reference=rear_edge,
        front_support=front_support,
        rear_support=rear_support,
        DIRECTORY_CHECKME=checkme_dir,
        raise_error_if_negative_volume_cells=raise_error_if_negative_volume_cells,
        **BuildMatchMeshAdditionalParams)

    return stage_mesh


def rediscretizeBlade(blade,
        RadialNbOfPoints = 51,
        RadialCellLengthAtTip = 0.0005,
        RadialCellLengthAtRoot = 0.02,
        SectionsDistribution = dict(
            RelativeAbscissa =   [0.0,    1.0],
            TrailingEdgeSegmentLength = [0.004, 0.0005],
            LeadingEdgeSegmentLength = [0.0001, 0.00005],
            LeadingEdgeAbscissa = [0.49, 0.49],
            TopSideNumberOfPoints = 67, # must be odd
            BottomSideNumberOfPoints = 67, # must be odd
            TopToBottomAtTipNumberOfPoints = 9,
            InterpolationLaw = 'interp1d_linear',),
        CloseTip=True,
        CloseRoot=False,
        ):
    
    blade = J.selectZoneWithHighestNumberOfPoints(I.getZones(blade))
    NbOfOriginalBladeSections = I.getZoneDim(blade)[2]

    Span = GSD.getBoundary(blade,'imin')
    s = W.gets(Span)
    W.discretizeInPlace(Span, N=RadialNbOfPoints,
        Distribution=dict(
            kind='tanhTwoSides',
            FirstCellHeight=RadialCellLengthAtRoot,
            LastCellHeight=RadialCellLengthAtTip))


    # scan the original blade sections, and simply rediscretize them contourwise:
    sections = []
    for j in range(NbOfOriginalBladeSections):

        section = GSD.getBoundary(blade,'jmin',j)

        TrailingEdgeSegmentLength = float(interpolate__(s[j],
            SectionsDistribution['RelativeAbscissa'],
            SectionsDistribution['TrailingEdgeSegmentLength'],
            Law=SectionsDistribution['InterpolationLaw']))

        LeadingEdgeSegmentLength = float(interpolate__(s[j],
            SectionsDistribution['RelativeAbscissa'],
            SectionsDistribution['LeadingEdgeSegmentLength'],
            Law=SectionsDistribution['InterpolationLaw']))

        LeadingEdgeAbscissa = float(interpolate__(s[j],
            SectionsDistribution['RelativeAbscissa'],
            SectionsDistribution['LeadingEdgeAbscissa'],
            Law=SectionsDistribution['InterpolationLaw']))


        SectionDistribution = [
            dict(N=SectionsDistribution["BottomSideNumberOfPoints"],
                BreakPoint=LeadingEdgeAbscissa,
                kind='tanhTwoSides',
                FirstCellHeight=TrailingEdgeSegmentLength,
                LastCellHeight=LeadingEdgeSegmentLength),
            dict(N=SectionsDistribution['TopSideNumberOfPoints'],
                BreakPoint=1.0,
                kind='tanhTwoSides',
                FirstCellHeight=LeadingEdgeSegmentLength,
                LastCellHeight=TrailingEdgeSegmentLength)]
        section = W.polyDiscretize(section, SectionDistribution)
        sections += [ section ]
    

    new_blade = GSD.multiSections(sections, Span,
        InterpolationData={'InterpolationLaw':SectionsDistribution['InterpolationLaw']})[0]
    new_blade[0] = blade[0]


    if CloseTip:
        new_blade = GSD.closeWingTipAndRoot(new_blade, tip_window='jmax',close_root=CloseRoot,
            airfoil_top2bottom_NPts=SectionsDistribution['TopToBottomAtTipNumberOfPoints'])


    return new_blade
            
