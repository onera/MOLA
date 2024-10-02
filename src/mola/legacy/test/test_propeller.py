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

import pytest
import os

@pytest.mark.user_case
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_3
def test_oras_mesher(tmp_path):
    return True
    import numpy as np
    import Converter.PyTree as C
    import Converter.Internal as I

    import mola.legacy.InternalShortcuts as J
    import mola.legacy.curve as W
    import mola.legacy.surface as GSD
    import mola.legacy.volume as GVD
    import mola.legacy.propeller_mesher as RW


    spinner_rear_topology = 'line' # 'line' or 'arc'

    blade_number = 5
    delta_pitch_angle = +55.0 # deg
    wall_cell_height = 10e-6
    blade_radial_NPts = 40
    airfoil_NPts_top = airfoil_NPts_bottom = 67 #171 # must be ODD
    spinner_azimut_NPts = 31#151

    # BLADE DISCRETIZATION

    RightHandRuleRotation = True
    blade_root_cellwidth = 0.01
    blade_tip_cellwidth = 0.0005
    blade_tip_NPts_top2Bottom = 9
    blade_input_spanwise_direction   = (0, 0,-1)
    blade_input_axial_direction      = (0, 1, 0)
    blade_input_tangential_direction = (1, 0, 0)
    blade_input_pitch_center = (0,0,0)

    airfoil_LeadingEdge_abscissa = 0.49
    airfoil_LeadingEdge_width_relative2chord =  0.004
    airfoil_TrailingEdge_width_relative2chord = 0.004
    airfoil_stacking_point_relative2chord = 0.25


    if spinner_rear_topology == 'line':
        spinner_LengthRear = 10.0
        spinner_LeadingEdge_diamond_abscissa = 0.004/spinner_LengthRear
        spinner_TrailingEdge_diamond_abscissa = None
        spinner_TrailingEdgeCellLength = spinner_LengthRear * 0.015
        spinner_RearNPts = 150
    elif spinner_rear_topology == 'arc':
        spinner_LengthRear = 1.0
        spinner_LeadingEdge_diamond_abscissa = 0.004/spinner_LengthRear
        spinner_TrailingEdge_diamond_abscissa = 0.998/spinner_LengthRear
        spinner_TrailingEdgeCellLength = 3e-5
        spinner_RearNPts = 80
    spinner_azimut_adjust = 'auto' # float or 'auto'

    spinner_profile_input_spanwise_direction   = (1, 0, 0)
    spinner_profile_input_axial_direction      = (0, 1, 0)
    spinner_profile_input_tangential_direction = (0, 0, 1)

    rotation_center = (0, 0, 0)
    final_spanwise_direction   = ( 0, 1, 0)
    final_axial_direction      = (-1, 0, 0) # this is the final rotation axis
    final_tangential_direction = ( 0, 0, 1)


    DIRECTORY_CHECKME = os.path.join(tmp_path,'CHECK_ME')
    try: os.makedirs(DIRECTORY_CHECKME)
    except: pass
    RW.DIRECTORY_CHECKME = DIRECTORY_CHECKME


    # ----------------------- SPINNER AND BLADE SURFACES ----------------------- #
    toc = GVD.tic()


    r = 0.05 # minimum radius of blade
    R = 0.6  # maximum radius of blade

    BladeDiscretization = dict(P1=(r,0,0),P2=(R,0,0),
                            N=blade_radial_NPts,
                            kind='tanhTwoSides',
                            FirstCellHeight=blade_root_cellwidth,
                            LastCellHeight=blade_tip_cellwidth)

    ChordDict = dict(RelativeSpan = [r/R,   0.45,  0.6,  1.0],
                    Chord        = [0.07,  0.10, 0.10, 0.02],
                    InterpolationLaw = 'akima',)

    TwistDict = dict(RelativeSpan = [r/R,  0.6,  1.0],
                    Twist        = [30.,  6.0, -1.0],
                    InterpolationLaw = 'akima',)

    DihedralDict = dict(RelativeSpan = [r/R,    1.0],
                    Dihedral        = [0., 0.],
                    InterpolationLaw = 'interp1d_linear',)

    SweepDict = dict(RelativeSpan = [r/R,    1.0],
                    Sweep        = [0., 0.],
                    InterpolationLaw = 'interp1d_linear',)

    # front root to tip
    Airfoils = [W.airfoil('NACA4416'),
                W.airfoil('NACA4416')]
    # Airfoils = [J.load('/stck/lbernard/AIRFOIL_DATABASE/OA/OA309.cgns'),
    #             J.load('/stck/lbernard/AIRFOIL_DATABASE/OA/OA309.cgns')]

    AirfoilDistribution=[dict(N=airfoil_NPts_bottom,
                        BreakPoint=airfoil_LeadingEdge_abscissa,
                        kind='tanhTwoSides',
                        FirstCellHeight=airfoil_TrailingEdge_width_relative2chord,
                        LastCellHeight=airfoil_LeadingEdge_width_relative2chord),
                        dict(N=airfoil_NPts_top,
                            BreakPoint=1.0,
                            kind='tanhTwoSides',
                            FirstCellHeight=airfoil_LeadingEdge_width_relative2chord,
                            LastCellHeight=airfoil_TrailingEdge_width_relative2chord),]

    Airfoils = [W.polyDiscretize(I.getZones(a)[0], AirfoilDistribution) for a in Airfoils]

    AirfoilsDict = dict(RelativeSpan     = [r/R,  1.000],
                        Airfoil = [Airfoils[0], Airfoils[1]],
                        InterpolationLaw = 'interp1d_linear',)


    blade = GSD.wing(BladeDiscretization,
                    ChordRelRef = airfoil_stacking_point_relative2chord,
                    NPtsTrailingEdge = blade_tip_NPts_top2Bottom,
                    AvoidAirfoilModification = True,
                    Chord = ChordDict,
                    Dihedral =  DihedralDict,
                    Sweep =  SweepDict,
                    Twist =  TwistDict,
                    Airfoil =  AirfoilsDict,)[1]
    blade[0] = 'blade'
    if not RightHandRuleRotation:
        x = J.getx(blade)
        x *= -1
        GVD.T._reorder(blade,(-1,2,3))


    J.save(blade,os.path.join(DIRECTORY_CHECKME,'0_blade_geometry.cgns'))

    RW.addPitchAndAdjustPositionOfBladeSurface(blade, root_window='jmin',
        delta_pitch_angle= delta_pitch_angle if RightHandRuleRotation else -delta_pitch_angle,
        pitch_center_adjust_relative2chord=0.50,
        pitch_axis=blade_input_spanwise_direction,
        pitch_center=blade_input_pitch_center)

    blade = GSD.closeWingTipAndRoot(blade, tip_window='jmax', close_root=False,
                                airfoil_top2bottom_NPts=blade_tip_NPts_top2Bottom)

    J.save(blade,os.path.join(DIRECTORY_CHECKME,'1_blade_surface.cgns'))

    # stator 
    BladeDiscretization = dict(P1=(r,0,0),P2=(0.85*R,0,0),
                            N=int(blade_radial_NPts*0.6),
                            kind='tanhTwoSides',
                            FirstCellHeight=blade_root_cellwidth,
                            LastCellHeight=blade_tip_cellwidth)

    ChordDict = dict(RelativeSpan = [r/R,   0.45,  0.6,  1.0],
                    Chord        = 0.7*np.array([0.07,  0.10, 0.10, 0.03]),
                    InterpolationLaw = 'akima',)

    TwistDict = dict(RelativeSpan = [r/R,  0.6,  1.0],
                    Twist        = [30.,  6.0, -1.0],
                    InterpolationLaw = 'akima',)

    DihedralDict = dict(RelativeSpan = [r/R,    1.0],
                    Dihedral        = [0., 0.],
                    InterpolationLaw = 'interp1d_linear',)

    SweepDict = dict(RelativeSpan = [r/R,    1.0],
                    Sweep        = [0., 0.],
                    InterpolationLaw = 'interp1d_linear',)

    # front root to tip
    Airfoils = [W.airfoil('NACA4416'),
                W.airfoil('NACA4416')]
    # Airfoils = [J.load('/stck/lbernard/AIRFOIL_DATABASE/OA/OA309.cgns'),
    #             J.load('/stck/lbernard/AIRFOIL_DATABASE/OA/OA309.cgns')]

    AirfoilDistribution=[dict(N=airfoil_NPts_bottom,
                        BreakPoint=airfoil_LeadingEdge_abscissa,
                        kind='tanhTwoSides',
                        FirstCellHeight=airfoil_TrailingEdge_width_relative2chord,
                        LastCellHeight=airfoil_LeadingEdge_width_relative2chord),
                        dict(N=airfoil_NPts_top,
                            BreakPoint=1.0,
                            kind='tanhTwoSides',
                            FirstCellHeight=airfoil_LeadingEdge_width_relative2chord,
                            LastCellHeight=airfoil_TrailingEdge_width_relative2chord),]

    Airfoils = [W.polyDiscretize(I.getZones(a)[0], AirfoilDistribution) for a in Airfoils]

    AirfoilsDict = dict(RelativeSpan     = [r/R,  1.000],
                        Airfoil = [Airfoils[0], Airfoils[1]],
                        InterpolationLaw = 'interp1d_linear',)


    stator = GSD.wing(BladeDiscretization,
                    ChordRelRef = airfoil_stacking_point_relative2chord,
                    NPtsTrailingEdge = blade_tip_NPts_top2Bottom,
                    AvoidAirfoilModification = True,
                    Chord = ChordDict,
                    Dihedral =  DihedralDict,
                    Sweep =  SweepDict,
                    Twist =  TwistDict,
                    Airfoil =  AirfoilsDict,)[1]
    stator[0] = 'stator'
    if RightHandRuleRotation:
        x = J.getx(stator)
        x *= -1
        GVD.T._reorder(stator,(-1,2,3))



    J.save(stator,os.path.join(DIRECTORY_CHECKME,'0_stator_geometry.cgns'))

    RW.addPitchAndAdjustPositionOfBladeSurface(stator, root_window='jmin',
        delta_pitch_angle= delta_pitch_angle if not RightHandRuleRotation else -delta_pitch_angle,
        pitch_center_adjust_relative2chord=0.50,
        pitch_axis=blade_input_spanwise_direction,
        pitch_center=blade_input_pitch_center)
    
    GVD.T._translate(stator,(0,-0.15,0))

    stator = GSD.closeWingTipAndRoot(stator, tip_window='jmax', close_root=False,
                                airfoil_top2bottom_NPts=blade_tip_NPts_top2Bottom)

    J.save(stator,os.path.join(DIRECTORY_CHECKME,'1_stator_surface.cgns'))

    curves  = RW.makeSpinnerCurves(LengthFront=0.2, LengthRear=spinner_LengthRear,
                        Width=0.15,
                        RelativeArcRadiusFront=0.008, ArcAngleFront=40.,
                        RelativeTensionArcFront=0.1, RelativeTensionRootFront=0.5,
                        TopologyRear=spinner_rear_topology,
                        RelativeArcRadiusRear=0.0025, ArcAngleRear=70.,
                        RelativeTensionArcRear=0.1, RelativeTensionRootRear=0.5)

    profile = curves[0]
    for c in curves[1:]: profile = RW.T.join(profile, c)

    RW.T._rotate(profile,(0,0,0),(0,0,1),90)
    J.save(profile,os.path.join(DIRECTORY_CHECKME,'2_profile_geometry.cgns'))

    profile_input_frenet = (spinner_profile_input_spanwise_direction,
                            spinner_profile_input_axial_direction,
                            spinner_profile_input_tangential_direction)

    blade_input_frenet = (blade_input_spanwise_direction,
                        blade_input_axial_direction,
                        blade_input_tangential_direction)

    final_frenet = (final_spanwise_direction,
                    final_axial_direction,
                    final_tangential_direction)


    RW.T._rotate(blade, rotation_center, blade_input_frenet, final_frenet)
    RW.T._rotate(stator, rotation_center, blade_input_frenet, final_frenet)

    ###########################################################################
    #                     REQUIRED DATA STARTS FROM HERE                      #
    ###########################################################################
    # (see buildOpenRotorAndStatorMesh for details)
    # profile: entire hub profile, densely discretized, passing through blades, on OXY plane
    # blade: blade intersecting profile, closed at tip and at TE, with root at jmin, imin starts at bottom at real position (except pitch)
    # stator: same as blade, but concerning the stator

    RotorNumberOfBlades = 9
    RotorAzimutalCellAngle = 1.0 # deg
    ncell_azimut_rotor = int((360/RotorNumberOfBlades)/RotorAzimutalCellAngle)

    blade_main_surface = J.selectZoneWithHighestNumberOfPoints( blade )
    _,Ni,_,_,_=I.getZoneDim(blade_main_surface)
    Nb_segments_airfoil = Ni - 1
    Hgrid_NPts = Nb_segments_airfoil//2 - ncell_azimut_rotor

    if Hgrid_NPts < 9:
        raise ValueError('insuficient number of airfoil segments compared to azimut points')
    Hgrid_cell = 3e-3
    law = 'tanhTwoSides'

    interface_cell_length_axially = 0.002

    RotorHubProfileReDiscretization = [
    # reference spinner leading edge arc discretization:
    {'N':32, 'BreakPoint(x)':-0.1919, 'kind':law,'FirstCellHeight':1e-4,'LastCellHeight':1.8e-3},

    # from spinner leading edge to blade root H-grid region:
    {'N':40, 'BreakPoint(x)':-0.063, 'kind':law,'FirstCellHeight':1.8e-3,'LastCellHeight':Hgrid_cell},

    # blade root H-grid region:
    {'N':Hgrid_NPts, 'BreakPoint(x)':+0.06, 'kind':law,'FirstCellHeight':Hgrid_cell,'LastCellHeight':Hgrid_cell},

    # rear
    {'N':10, 'BreakPoint':  1.0, 'kind':law,
    'FirstCellHeight':Hgrid_cell, 'LastCellHeight':interface_cell_length_axially},
    ]


    RotorBladeExtrusionParams = dict(
        root_to_transition_distance=0.1,
        root_to_transition_number_of_points=11,
        
        maximum_extrusion_distance_at_spinner=5e-3,
        maximum_number_of_points_in_normal_direction=500,
        distribution_law='ratio',
        distribution_growth_rate=1.15,
        last_extrusion_cell_height=1e-3,
        
        smoothing_start_at_layer=10,
        smoothing_normals_iterations=3,
        smoothing_normals_subiterations=[2,30,'distance'],
        smoothing_growth_iterations=2,
        smoothing_growth_subiterations=50,
        smoothing_growth_coefficient=[0.1,0.5,'index'],
        smoothing_expansion_factor=[0.05,0.2,'index'],
        intersection_method='conformize')



    ############################## Stator Params ##############################
    StatorNumberOfBlades = 9
    StatorAzimutalCellAngle = 1.0 # deg
    ncell_azimut_Stator = int((360/StatorNumberOfBlades)/StatorAzimutalCellAngle)

    stator_main_surface = J.selectZoneWithHighestNumberOfPoints( stator )
    _,Ni,_,_,_=I.getZoneDim(stator_main_surface)
    Nb_segments_airfoil = Ni - 1
    Hgrid_NPts = Nb_segments_airfoil//2 - ncell_azimut_Stator

    if Hgrid_NPts < 9:
        raise ValueError('insuficient number of airfoil segments compared to azimut points')
    Hgrid_cell = 0.002
    law = 'tanhTwoSides'

    StatorHubProfileReDiscretization = [
    {'N':10, 'BreakPoint(x)':0.1, 'kind':law,'FirstCellHeight':interface_cell_length_axially,'LastCellHeight':Hgrid_cell},
    {'N':Hgrid_NPts, 'BreakPoint(x)':0.2, 'kind':law,'FirstCellHeight':Hgrid_cell,'LastCellHeight':Hgrid_cell},
    {'N':100, 'BreakPoint':1, 'kind':law,'FirstCellHeight':Hgrid_cell,'LastCellHeight':spinner_TrailingEdgeCellLength},
      ]


    StatorBladeExtrusionParams = dict(
        root_to_transition_distance=0.1,
        root_to_transition_number_of_points=11,
        
        maximum_extrusion_distance_at_spinner=5e-3,
        maximum_number_of_points_in_normal_direction=500,
        distribution_law='ratio',
        distribution_growth_rate=1.15,
        last_extrusion_cell_height=1e-3,
        
        smoothing_start_at_layer=10,
        smoothing_normals_iterations=3,
        smoothing_normals_subiterations=[2,30,'distance'],
        smoothing_growth_iterations=2,
        smoothing_growth_subiterations=50,
        smoothing_growth_coefficient=[0.1,0.5,'index'],
        smoothing_expansion_factor=[0.05,0.2,'index'],
        intersection_method='conformize')


    t = RW.buildOpenRotorAndStatorMesh(blade,stator,profile,
            CoordinateOfRotorStatorInterfaceAtHub=0.082,
            RotorHubProfileReDiscretization = RotorHubProfileReDiscretization,
            RotorAzimutalCellAngle = RotorAzimutalCellAngle,
            RotorNumberOfBlades=RotorNumberOfBlades,
            RotorBladeWallCellHeight=1e-5,
            RotorHubWallCellHeight=0.005,
            RotorBladeExtrusionParams = RotorBladeExtrusionParams,

            StatorHubProfileReDiscretization = StatorHubProfileReDiscretization,
            StatorAzimutalCellAngle = StatorAzimutalCellAngle,
            StatorNumberOfBlades=StatorNumberOfBlades,
            StatorBladeWallCellHeight=1e-5,
            StatorHubWallCellHeight=0.005,
            StatorBladeExtrusionParams = StatorBladeExtrusionParams,
            )

    J.save(t,os.path.join(tmp_path,'mesh.cgns'))
    J.printElapsedTime('total meshing time was:', previous_timer=toc)


if __name__ == '__main__':
    # test_propeller_mesher_light('test_propeller_mesher_light')
    test_oras_mesher('test_oras_mesher')