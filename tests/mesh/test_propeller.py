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

import numpy as np
from treelab import cgns
import pytest
import os

@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_1
def test_designBlade(tmp_path):

    import mola.mesh.propeller_mesher as RW
    blade = RW.designBlade(RightHandRuleRotation=False)


@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_0
def test_rediscretizeBlade():

    import mola.mesh.propeller_mesher as RW
    blade = RW.J.selectZoneWithHighestNumberOfPoints( RW.designBlade() )
    new_blade = RW.rediscretizeBlade(blade)
    

@pytest.mark.user_case
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_1
def test_getBladesORAS_ONERA_SE(tmp_path):

    import mola.mesh.propeller_mesher as RW
    rotor, stator = RW.getBladesORAS_ONERA_SE()


@pytest.mark.user_case
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_3
def test_oras_mesher_single(tmp_path):

    import mola.pytree.InternalShortcuts as J
    import mola.mesh.propeller_mesher as RW

    test_mode_else_debug = True # True:testing False:debugging

    if test_mode_else_debug:
        check_dir = ''
        raise_error_if_negative_volume_cells = True
    else:
        check_dir = os.path.join(tmp_path,'CHECK_ME')
        raise_error_if_negative_volume_cells = False

    RotorNumberOfBlades = 13
    StatorNumberOfBlades = 11
    AzimutalCellAngleInDegrees = 1.0 

    toc = J.tic()

    profile = RW.getHubProfileORAS_ONERA_SE()
    rotor, stator = RW.getBladesORAS_ONERA_SE()

    RotorHgridXlocations = RW.proposeHgridXlocations(rotor,profile) # (-1.50, -0.75)
    StatorHgridXlocations = RW.proposeHgridXlocations(stator, profile) # (-0.45, 0.20)

    discretizations = RW.getSimpleORASHubProfileDiscretizations(rotor, stator,
        RotorNumberOfBlades=RotorNumberOfBlades,
        StatorNumberOfBlades=StatorNumberOfBlades,
        RotorHgridXlocations=RotorHgridXlocations,
        StatorHgridXlocations=StatorHgridXlocations)

    t = RW.buildOpenRotorAndStatorMesh(rotor,stator,profile,
            RotorNumberOfBlades=RotorNumberOfBlades,
            RotorHubProfileReDiscretization = discretizations[0],
            RotorAzimutalCellAngle = AzimutalCellAngleInDegrees,
            RotorHgridXlocations=RotorHgridXlocations,

            StatorNumberOfBlades=StatorNumberOfBlades,
            StatorHubProfileReDiscretization = discretizations[1],
            StatorAzimutalCellAngle = AzimutalCellAngleInDegrees,
            StatorHgridXlocations=StatorHgridXlocations,
            
            LOCAL_DIRECTORY_CHECKME=check_dir,
            raise_error_if_negative_volume_cells=raise_error_if_negative_volume_cells,
            )

    J.printElapsedTime('total meshing time was:', previous_timer=toc)
    
    toc = J.tic()

    if not test_mode_else_debug:
        print('will save mesh')
        J.save(t,os.path.join(tmp_path,'mesh.cgns'))
        J.printElapsedTime('saving mesh took:', previous_timer=toc)



@pytest.mark.user_case
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_3
def test_oras_mesher_ultracoarse(tmp_path):

    import mola.pytree.InternalShortcuts as J
    import mola.mesh.propeller_mesher as RW

    test_mode_else_debug = True # True:testing False:debugging

    if test_mode_else_debug:
        check_dir = ''
        raise_error_if_negative_volume_cells = True
    else:
        check_dir = os.path.join(tmp_path,'CHECK_ME')
        raise_error_if_negative_volume_cells = False

    RotorNumberOfBlades = 13
    StatorNumberOfBlades = 11
    AzimutalCellAngleInDegrees = 2.50

    toc = J.tic()

    profile = RW.getHubProfileORAS_ONERA_SE()
    rotor, stator = RW.getBladesORAS_ONERA_SE(
        RotorRadialNbOfPoints = 25,
        RotorRadialCellLengthAtTip = 0.005,
        RotorRadialCellLengthAtRoot = 0.05,
        RotorAirfoilSideNumberOfPoints=37,

        StatorRadialNbOfPoints = 27,
        StatorRadialCellLengthAtTip = 0.005,
        StatorRadialCellLengthAtRoot = 0.05,
        StatorAirfoilSideNumberOfPoints=37
    )

    RotorHgridXlocations = RW.proposeHgridXlocations(rotor,profile) # (-1.50, -0.75)
    StatorHgridXlocations = RW.proposeHgridXlocations(stator, profile) # (-0.45, 0.20)

    discretizations = RW.getSimpleORASHubProfileDiscretizations(rotor, stator,
        RotorNumberOfBlades=RotorNumberOfBlades,
        StatorNumberOfBlades=StatorNumberOfBlades,
        AzimutalCellAngle=1.0,
        InterfaceAxialCellLength=1.5e-2,
        BreakPointsAxialCellLength=1.5e-2,
        
        # rotor hub profile discretization
        RotorHgridXlocations=RotorHgridXlocations,
        RotorFrontNPts=31,
        RotorRearNPts=9,
        RotorFrontSegmentLength=0.05,
        
        # stator hub profile discretization
        StatorHgridXlocations=StatorHgridXlocations,
        StatorFrontNPts=9,
        StatorRearNPts=31,
        StatorRearSegmentLength=0.10
            )


    t = RW.buildOpenRotorAndStatorMesh(rotor,stator,profile,
            FarfieldRadius = 3,
            RotorRadialExtrusionNbOfPoints=9,
            RotorBladeWallCellHeight = 2e-3,
            RotorHubWallCellHeight = 3e-2,
            RotorBladeRootWallNormalDistanceRelativeToRootChord = 0.05,
            RotorNumberOfBlades=RotorNumberOfBlades,
            RotorHubProfileReDiscretization = discretizations[0],
            RotorAzimutalCellAngle = AzimutalCellAngleInDegrees,
            RotorHgridXlocations=RotorHgridXlocations,
            RotorHgridNbOfPoints=9,
            RotorRootRemeshRadialNbOfPoints=9,
            RotorRootWallRemeshRadialDistanceRelativeToMaxRadius=0.25,

            StatorRadialExtrusionNbOfPoints=15,
            StatorBladeWallCellHeight = 2e-3,
            StatorHubWallCellHeight = 3e-2,
            StatorBladeRootWallNormalDistanceRelativeToRootChord = 0.03,
            StatorNumberOfBlades=StatorNumberOfBlades,
            StatorHubProfileReDiscretization = discretizations[1],
            StatorAzimutalCellAngle = AzimutalCellAngleInDegrees,
            StatorHgridXlocations=StatorHgridXlocations,
            StatorHgridNbOfPoints=9,
            StatorRootRemeshRadialNbOfPoints=9,
            StatorRootWallRemeshRadialDistanceRelativeToMaxRadius=0.25,
            
            LOCAL_DIRECTORY_CHECKME=check_dir,
            raise_error_if_negative_volume_cells=raise_error_if_negative_volume_cells,
            )

    J.printElapsedTime('total meshing time was:', previous_timer=toc)
    
    toc = J.tic()

    if not test_mode_else_debug:
        print('will save mesh')
        J.save(t,os.path.join(tmp_path,'mesh.cgns'))
        J.printElapsedTime('saving mesh took:', previous_timer=toc)


@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_3
def test_oras_mesher_designer_ultracoarse(tmp_path):

    import mola.pytree.InternalShortcuts as J
    import mola.mesh.propeller_mesher as RW

    test_mode_else_debug = True # True:testing (no file write) False:debugging (file write)

    if test_mode_else_debug:
        check_dir = ''
        raise_error_if_negative_volume_cells = True
    else:
        check_dir = os.path.join(tmp_path,'CHECK_ME')
        raise_error_if_negative_volume_cells = False

    RotorNumberOfBlades = 9
    StatorNumberOfBlades = 11
    AzimutalCellAngle = 3.5 # deg


    toc = J.tic()

    profile = RW.makeSpinnerCurves(LengthFront=0.2, LengthRear=0.5, Width=0.15,
                      RelativeArcRadiusFront=0.008, ArcAngleFront=40.0,
                      RelativeTensionArcFront=0.1, RelativeTensionRootFront=0.5,
                      NPtsArcFront=200, NPtsSpinnerFront=5000,
                      TopologyRear='line')

    rotor = RW.designBlade(
        RadiusTip = 0.60,
        RadiusRoot = 0.05,

        RightHandRuleRotation = True,


        BladeStackPointPositionInXaxis = 0.0,
        BladePitchAxisPositionInXaxis = 0.0,
        PitchAngle = 55.0,
        ZeroPitchAngleRelativeRadius = None, # if None, uses construction reference

        # Radial discretization of the blade geometry:
        RadialNbOfPoints = 31,
        RadialCellLengthAtTip = 0.005,
        RadialCellLengthAtRoot = 0.03,

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
            TopSideNumberOfPoints = 31, # must be odd
            BottomSideNumberOfPoints = 31, # must be odd
            TopToBottomAtTipNumberOfPoints = 9,
            InterpolationLaw = 'interp1d_linear',
            ),
    )

    stator = RW.designBlade(
        RadiusTip = 0.50,
        RadiusRoot = 0.05,

        RightHandRuleRotation = False,

        BladeStackPointPositionInXaxis = 0.20,
        BladePitchAxisPositionInXaxis = 0.20,
        PitchAngle = 55.0,
        ZeroPitchAngleRelativeRadius = None, # if None, uses construction reference

        # Radial discretization of the blade geometry:
        RadialNbOfPoints = 31,
        RadialCellLengthAtTip = 0.005,
        RadialCellLengthAtRoot = 0.03,

        # Geometrical Laws
        ChordDistribution=dict(
            RelativeSpan = [0.05/0.6,   0.45,  0.6,  1.0],
            Chord        = 0.7*np.array([0.07,  0.10, 0.10, 0.03]),
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
            TopSideNumberOfPoints = 31, # must be odd
            BottomSideNumberOfPoints = 31, # must be odd
            TopToBottomAtTipNumberOfPoints = 9,
            InterpolationLaw = 'interp1d_linear',
            ))

    RotorHgridXlocations = RW.proposeHgridXlocations(rotor,profile, 0.3) # (-0.04, 0.075)
    StatorHgridXlocations = RW.proposeHgridXlocations(stator, profile,0.3) # (0.12, 0.20)

    discretizations = RW.getSimpleORASHubProfileDiscretizations(rotor, stator,
        RotorNumberOfBlades=RotorNumberOfBlades,
        StatorNumberOfBlades=StatorNumberOfBlades,

        AzimutalCellAngle=AzimutalCellAngle,
        InterfaceAxialCellLength=6e-3,
        BreakPointsAxialCellLength=6e-3,
        
        # rotor hub profile discretization
        RotorHgridXlocations=RotorHgridXlocations,
        RotorFrontNPts=31,
        RotorRearNPts=8,
        RotorFrontSegmentLength=3e-5,
        
        # stator hub profile discretization
        StatorHgridXlocations=StatorHgridXlocations,
        StatorFrontNPts=10,
        StatorRearNPts=15,
        StatorRearSegmentLength=0.03)

    t = RW.buildOpenRotorAndStatorMesh(rotor,stator,profile,
        FarfieldRadius= 0.9,

        # ------------------------ ROTOR parameters ------------------------ #
        RotorNumberOfBlades= RotorNumberOfBlades,
        RotorThetaAdjustmentInDegrees= -2.0,
        RotorHubProfileReDiscretization = discretizations[0],
        RotorAzimutalCellAngle= AzimutalCellAngle,
        RotorBladeWallCellHeight= 5e-4,
        RotorBladeRootWallNormalDistanceRelativeToRootChord=0.05,
        RotorHubWallCellHeight= 1e-2,
        RotorRootWallRemeshRadialDistanceRelativeToMaxRadius=0.15,
        RotorRootRemeshRadialNbOfPoints= 9,
        RotorRadialExtrusionNbOfPoints=9,
        RotorHgridXlocations = RotorHgridXlocations,
        RotorHgridNbOfPoints = 9,
        RotorHspreadingAngles= [-10, +3],
        RotorTipScaleFactorAtRadialFarfield= 0.25,
        RotorFarfieldRadialCellLengthRelativeToFarfieldRadius=0.1,
        RotorBladeExtrusionParams=dict(smoothing_start_at_layer=0), 

        # ------------------------ STATOR parameters ------------------------ #
        StatorNumberOfBlades=StatorNumberOfBlades,
        StatorThetaAdjustmentInDegrees= 1.0,
        StatorHubProfileReDiscretization = discretizations[1],
        StatorAzimutalCellAngle= AzimutalCellAngle,
        StatorBladeWallCellHeight= 5e-4,
        StatorBladeRootWallNormalDistanceRelativeToRootChord=0.05,
        StatorHubWallCellHeight= 1e-2,
        StatorRootWallRemeshRadialDistanceRelativeToMaxRadius=0.25,
        StatorRootRemeshRadialNbOfPoints= 9,
        StatorRadialExtrusionNbOfPoints=9,
        StatorHgridXlocations = StatorHgridXlocations,
        StatorHgridNbOfPoints = 9,
        StatorHspreadingAngles= [0, 10],
        StatorTipScaleFactorAtRadialFarfield= 0.25,
        StatorFarfieldRadialCellLengthRelativeToFarfieldRadius=0.1,
        StatorBladeExtrusionParams=dict(smoothing_start_at_layer=0), 

        # ------------------------------- misc ------------------------------- #
        LOCAL_DIRECTORY_CHECKME = check_dir,
        raise_error_if_negative_volume_cells=raise_error_if_negative_volume_cells,
            )

    J.printElapsedTime('total meshing time was:', previous_timer=toc)
    
    toc = J.tic()

    if not test_mode_else_debug:
        print('will save mesh')
        J.save(t,os.path.join(tmp_path,'mesh.cgns'))
        J.printElapsedTime('saving mesh took:', previous_timer=toc)



@pytest.mark.user_case
@pytest.mark.restricted_user_case
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_3
def test_oras_mesher_inpro_ultracoarse(tmp_path):

    import mola.pytree.InternalShortcuts as J
    import mola.mesh.propeller_mesher as RW

    test_mode_else_debug = True # True:testing False:debugging
    msg_restricted = J.WARN+"WARNING: RESTRICTED USER CASE - SHARING INPUT/OUTPUT DATA IS FORBIDDEN"+J.ENDC

    if test_mode_else_debug:
        check_dir = ''
        raise_error_if_negative_volume_cells = True
    else:
        print(msg_restricted)
        check_dir = os.path.join(tmp_path,'CHECK_ME')
        raise_error_if_negative_volume_cells = False

    RotorNumberOfBlades = 14
    StatorNumberOfBlades = 12
    AzimutalCellAngleInDegrees = 2.25

    toc = J.tic()


    profile = J.load("/stck/mola/data/restricted/geometry/oras/inpro/hub_profile.cgns",
                     return_type='zone')
    rotor = J.load("/stck/mola/data/restricted/geometry/oras/inpro/rotor_structured_surface.cgns")
    stator = J.load("/stck/mola/data/restricted/geometry/oras/inpro/stator_structured_surface.cgns")

    # extrapolate at root in order to guarantee that blade fully intersects hub
    rotor_blade = J.selectZoneWithHighestNumberOfPoints(rotor)
    rotor_blade = RW.GSD.extrapolateSurface(rotor_blade,'jmin',0.05)
    stator_blade = J.selectZoneWithHighestNumberOfPoints(stator)
    stator_blade = RW.GSD.extrapolateSurface(stator_blade,'jmin',0.05)


    rotor = RW.rediscretizeBlade(rotor_blade,
        RadialNbOfPoints = 29,
        RadialCellLengthAtTip = 2e-3,
        RadialCellLengthAtRoot = 0.05,
        SectionsDistribution = dict(
            RelativeAbscissa =   [0.0,  0.4,  1.0],
            TrailingEdgeSegmentLength = [5e-4, 3e-4, 1.75e-4],
            LeadingEdgeSegmentLength = [8e-4, 5e-4, 2.5e-4],
            LeadingEdgeAbscissa = [0.508, 0.5005, 0.5005],
            TopSideNumberOfPoints = 41, # must be odd
            BottomSideNumberOfPoints = 41, # must be odd
            TopToBottomAtTipNumberOfPoints = 9,
            InterpolationLaw = 'interp1d_linear'))

    stator = RW.rediscretizeBlade(stator_blade,
        RadialNbOfPoints = 25,
        RadialCellLengthAtTip = 2e-3,
        RadialCellLengthAtRoot = 0.05,
        SectionsDistribution = dict(
            RelativeAbscissa =   [0.0,  0.5,  0.75, 1.0],
            TrailingEdgeSegmentLength = [5e-4, 3e-4, 2.2e-4, 1.75e-4],
            LeadingEdgeSegmentLength = [8e-4, 5e-4, 3e-4, 2.5e-4],
            LeadingEdgeAbscissa = [0.495, 0.498, 0.499, 0.495],
            TopSideNumberOfPoints = 41, # must be odd
            BottomSideNumberOfPoints = 41, # must be odd
            TopToBottomAtTipNumberOfPoints = 9,
            InterpolationLaw = 'interp1d_linear'))


    RotorHgridXlocations = RW.proposeHgridXlocations(rotor,profile,0.3) # (-1.50, -0.75)
    StatorHgridXlocations = RW.proposeHgridXlocations(stator, profile,0.3) # (-0.45, 0.20)

    discretizations = RW.getSimpleORASHubProfileDiscretizations(rotor, stator,
        RotorNumberOfBlades=RotorNumberOfBlades,
        StatorNumberOfBlades=StatorNumberOfBlades,
        AzimutalCellAngle=AzimutalCellAngleInDegrees,
        InterfaceAxialCellLength=1.5e-2,
        BreakPointsAxialCellLength=1.5e-2,
        
        # rotor hub profile discretization
        RotorHgridXlocations=RotorHgridXlocations,
        RotorFrontNPts=21,
        RotorRearNPts=9,
        RotorFrontSegmentLength=0.075,
        
        # stator hub profile discretization
        StatorHgridXlocations=StatorHgridXlocations,
        StatorFrontNPts=9,
        StatorRearNPts=15,
        StatorRearSegmentLength=1.0
            )


    t = RW.buildOpenRotorAndStatorMesh(rotor,stator,profile,
            FarfieldRadius = 3,

            RotorNumberOfBlades=RotorNumberOfBlades,
            RotorDeltaPitch= -0.10,
            RotorPitchCenter= 0.0,
            RotorRadialExtrusionNbOfPoints=9,
            RotorBladeWallCellHeight = 2e-3,
            RotorHubWallCellHeight = 3e-2,
            RotorBladeRootWallNormalDistanceRelativeToRootChord = 0.05,
            RotorHubProfileReDiscretization = discretizations[0],
            RotorAzimutalCellAngle = AzimutalCellAngleInDegrees,
            RotorHgridXlocations=RotorHgridXlocations,
            RotorHgridNbOfPoints=9,
            RotorRootRemeshRadialNbOfPoints=9,
            RotorRootWallRemeshRadialDistanceRelativeToMaxRadius=0.25,

            # # NOTE that rotor requires specific optimum indexing H grid closer to root:
            RotorBuildMatchMeshAdditionalParams=dict(radial_H_compromise=0),

            StatorNumberOfBlades=StatorNumberOfBlades,
            StatorDeltaPitch = 0.0,
            StatorPitchCenter = 0.9955,
            StatorRadialExtrusionNbOfPoints=15,
            StatorBladeWallCellHeight = 2e-3,
            StatorHubWallCellHeight = 3e-2,
            StatorBladeRootWallNormalDistanceRelativeToRootChord = 0.03,
            StatorHubProfileReDiscretization = discretizations[1],
            StatorAzimutalCellAngle = AzimutalCellAngleInDegrees,
            StatorHgridXlocations=StatorHgridXlocations,
            StatorHgridNbOfPoints=9,
            StatorRootRemeshRadialNbOfPoints=9,
            StatorRootWallRemeshRadialDistanceRelativeToMaxRadius=0.25,
            
            LOCAL_DIRECTORY_CHECKME=check_dir,
            raise_error_if_negative_volume_cells=raise_error_if_negative_volume_cells,
            )

    J.printElapsedTime('total meshing time was:', previous_timer=toc)
    
    toc = J.tic()

    if not test_mode_else_debug:
        print('will save mesh')
        J.save(t,os.path.join(tmp_path,'mesh.cgns'))
        J.printElapsedTime('saving mesh took:', previous_timer=toc)
        print(msg_restricted)


@pytest.mark.user_case
@pytest.mark.restricted_user_case
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_3
def test_oras_mesher_inpro(tmp_path):

    import mola.pytree.InternalShortcuts as J
    import mola.mesh.propeller_mesher as RW

    test_mode_else_debug = True # True:testing False:debugging
    msg_restricted = J.WARN+"WARNING: RESTRICTED USER CASE - SHARING INPUT/OUTPUT DATA IS FORBIDDEN"+J.ENDC

    if test_mode_else_debug:
        check_dir = ''
        raise_error_if_negative_volume_cells = True
    else:
        print(msg_restricted)
        check_dir = os.path.join(tmp_path,'CHECK_ME')
        raise_error_if_negative_volume_cells = False

    RotorNumberOfBlades = 14
    StatorNumberOfBlades = 12
    AzimutalCellAngleInDegrees = 0.50 # TODO use 0.15 (acoustic)

    toc = J.tic()

    profile = J.load("/stck/mola/data/restricted/geometry/oras/inpro/hub_profile.cgns",
                        return_type='zone')
    x = J.getx(profile)
    x[-1] = 21 # prolongation



    rotor = J.load("/stck/mola/data/restricted/geometry/oras/inpro/rotor_structured_surface.cgns")
    stator = J.load("/stck/mola/data/restricted/geometry/oras/inpro/stator_structured_surface.cgns")


    # extrapolate at root in order to guarantee that blade fully intersects hub
    rotor_blade = J.selectZoneWithHighestNumberOfPoints(rotor)
    rotor_blade = RW.GSD.extrapolateSurface(rotor_blade,'jmin',0.05)
    stator_blade = J.selectZoneWithHighestNumberOfPoints(stator)
    stator_blade = RW.GSD.extrapolateSurface(stator_blade,'jmin',0.05)

    rotor = RW.rediscretizeBlade(rotor_blade,
        RadialNbOfPoints = 80,
        RadialCellLengthAtTip = 2e-3,
        RadialCellLengthAtRoot = 0.05,
        SectionsDistribution = dict(
            RelativeAbscissa =   [0.0,  0.4,  1.0],
            TrailingEdgeSegmentLength = [2e-4, 1.25e-4, 0.62e-4],
            LeadingEdgeSegmentLength = [2e-4, 1.25e-4, 0.62e-4],
            LeadingEdgeAbscissa = [0.508, 0.5005, 0.5005],
            TopSideNumberOfPoints = 101, # must be odd
            BottomSideNumberOfPoints = 101, # must be odd
            TopToBottomAtTipNumberOfPoints = 15,
            InterpolationLaw = 'interp1d_linear'))



    stator = RW.rediscretizeBlade(stator_blade,
        RadialNbOfPoints = 80,
        RadialCellLengthAtTip = 1.5e-3,
        RadialCellLengthAtRoot = 0.05,
        SectionsDistribution = dict(
            RelativeAbscissa =   [0.0,  0.5,  0.75, 1.0],
            TrailingEdgeSegmentLength = [1e-4, 0.75e-4, 0.65e-4, 0.5e-4],
            LeadingEdgeSegmentLength = [1e-4, 0.75e-4, 0.65e-4, 0.5e-4],
            LeadingEdgeAbscissa = [0.495, 0.498, 0.499, 0.495],
            TopSideNumberOfPoints = 101, # must be odd
            BottomSideNumberOfPoints = 101, # must be odd
            TopToBottomAtTipNumberOfPoints = 15, 
            InterpolationLaw = 'interp1d_linear'))


    RotorHgridXlocations = RW.proposeHgridXlocations(rotor,profile,0.3) # (-1.50, -0.75)
    StatorHgridXlocations = RW.proposeHgridXlocations(stator, profile,0.3) # (-0.45, 0.20)


    discretizations = RW.getSimpleORASHubProfileDiscretizations(rotor, stator,
        RotorNumberOfBlades=RotorNumberOfBlades,
        StatorNumberOfBlades=StatorNumberOfBlades,
        AzimutalCellAngle=AzimutalCellAngleInDegrees,
        InterfaceAxialCellLength=5e-3,
        BreakPointsAxialCellLength=5e-3,
        
        # rotor hub profile discretization
        RotorHgridXlocations=RotorHgridXlocations,
        RotorFrontNPts=130,
        RotorRearNPts=27,
        RotorFrontSegmentLength=0.033,
        
        # stator hub profile discretization
        StatorHgridXlocations=StatorHgridXlocations,
        StatorFrontNPts=27,
        StatorRearNPts=150,
        StatorRearSegmentLength=1.0
            )


    t = RW.buildOpenRotorAndStatorMesh(rotor,stator,profile,
            FarfieldRadius = 20,

            RotorNumberOfBlades=RotorNumberOfBlades,
            RotorDeltaPitch= -0.10,
            RotorPitchCenter= 0.0,
            RotorRadialExtrusionNbOfPoints=70,
            RotorBladeWallCellHeight = 2e-6,
            RotorHubWallCellHeight = 2e-6,
            RotorBladeRootWallNormalDistanceRelativeToRootChord = 0.08,
            RotorHubProfileReDiscretization = discretizations[0],
            RotorAzimutalCellAngle = AzimutalCellAngleInDegrees,
            RotorHgridXlocations=RotorHgridXlocations,
            RotorHgridNbOfPoints=27,
            RotorRootRemeshRadialNbOfPoints=80,
            RotorRootWallRemeshRadialDistanceRelativeToMaxRadius=0.10,
            RotorFarfieldProfileAbscissaDeltas = [0.0,-0.05,0.0],
            RotorFarfieldTipSmoothIterations = 10,
            RotorBladeExtrusionParams = dict(
                smoothing_start_at_layer=30,
                smoothing_normals_iterations=3,
                smoothing_normals_subiterations=[2,30,'distance'],
                smoothing_growth_iterations=2,
                smoothing_growth_subiterations=50,
                smoothing_growth_coefficient=[0.1,0.5,'distance'],
                smoothing_expansion_factor=[0.05,0.2,'index'],),

            # # NOTE that rotor requires specific optimum indexing H grid closer to root:
            RotorBuildMatchMeshAdditionalParams=dict(radial_H_compromise=0),

            StatorNumberOfBlades=StatorNumberOfBlades,
            StatorDeltaPitch = 0.0,
            StatorPitchCenter = 0.9955,
            StatorRadialExtrusionNbOfPoints=70,
            StatorBladeWallCellHeight = 2e-6,
            StatorHubWallCellHeight = 2e-6,
            StatorBladeRootWallNormalDistanceRelativeToRootChord = 0.08,
            StatorHubProfileReDiscretization = discretizations[1],
            StatorAzimutalCellAngle = AzimutalCellAngleInDegrees,
            StatorHgridXlocations=StatorHgridXlocations,
            StatorHgridNbOfPoints=27,
            StatorRootRemeshRadialNbOfPoints=80,
            StatorRootWallRemeshRadialDistanceRelativeToMaxRadius=0.10,
            StatorFarfieldProfileAbscissaDeltas = [0.10,0.25],
            StatorFarfieldTipSmoothIterations = 10,
            StatorBladeExtrusionParams = dict(
                smoothing_start_at_layer=30,
                smoothing_normals_iterations=3,
                smoothing_normals_subiterations=[2,30,'distance'],
                smoothing_growth_iterations=2,
                smoothing_growth_subiterations=50,
                smoothing_growth_coefficient=[0.1,0.5,'distance'],
                smoothing_expansion_factor=[0.05,0.2,'index'],),

            LOCAL_DIRECTORY_CHECKME=check_dir,
            raise_error_if_negative_volume_cells=raise_error_if_negative_volume_cells,
            )

    J.printElapsedTime('total meshing time was:', previous_timer=toc)
    
    toc = J.tic()

    if not test_mode_else_debug:
        print('will save mesh')
        J.save(t,os.path.join(tmp_path,'mesh.cgns'))
        J.printElapsedTime('saving mesh took:', previous_timer=toc)
        print(msg_restricted)


@pytest.mark.user_case
@pytest.mark.restricted_user_case
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_4
def test_oras_mesher_barrier_ultracoarse(tmp_path):

    if os.getenv('USER') != 'lbernard': return # confidential case

    import mola.pytree.InternalShortcuts as J
    import mola.mesh.propeller_mesher as RW

    test_mode_else_debug = True # True:testing False:debugging
    msg_restricted = J.WARN+"WARNING: RESTRICTED USER CASE - SHARING INPUT/OUTPUT DATA IS FORBIDDEN"+J.ENDC

    if test_mode_else_debug:
        check_dir = ''
        raise_error_if_negative_volume_cells = True
    else:
        print(msg_restricted)
        check_dir = os.path.join(tmp_path,'CHECK_ME')
        raise_error_if_negative_volume_cells = False

    RotorNumberOfBlades = 14
    StatorNumberOfBlades = 12
    AzimutalCellAngleInDegrees = 2.25

    toc = J.tic()

    profile = J.load("/stck/lbernard/PROJETS/CLEOPATRA/TEST_GEOM_RAPHAEL/hub_profile.cgns", return_type='zone')
    rotor = J.load("/stck/lbernard/PROJETS/CLEOPATRA/TEST_GEOM_RAPHAEL/rotor_blade.cgns")
    stator = J.load("/stck/lbernard/PROJETS/CLEOPATRA/TEST_GEOM_RAPHAEL/stator_blade.cgns")

    # extrapolate at root in order to guarantee that blade fully intersects hub
    rotor_blade = J.selectZoneWithHighestNumberOfPoints(rotor)
    rotor_blade = RW.GSD.extrapolateSurface(rotor_blade,'jmin',0.05)
    stator_blade = J.selectZoneWithHighestNumberOfPoints(stator)
    stator_blade = RW.GSD.extrapolateSurface(stator_blade,'jmin',0.05)


    rotor = RW.rediscretizeBlade(rotor_blade,
        RadialNbOfPoints = 25,
        RadialCellLengthAtTip = 5e-3,
        RadialCellLengthAtRoot = 0.05,
        SectionsDistribution = dict(
            RelativeAbscissa =   [0.0,  0.2, 0.65, 1.0],
            TrailingEdgeSegmentLength = [5e-4, 3e-4, 2e-4, 1.75e-4],
            LeadingEdgeSegmentLength = [8e-4, 5e-4, 3e-4, 2.5e-4],
            LeadingEdgeAbscissa = [0.517, 0.503, 0.500, 0.5015],
            TopSideNumberOfPoints = 41, # must be odd
            BottomSideNumberOfPoints = 41, # must be odd
            TopToBottomAtTipNumberOfPoints = 11,
            InterpolationLaw = 'interp1d_linear'))

    stator = RW.rediscretizeBlade(stator_blade,
        RadialNbOfPoints = 25,
        RadialCellLengthAtTip = 2e-3,
        RadialCellLengthAtRoot = 0.05,
        SectionsDistribution = dict(
            RelativeAbscissa =   [0.0,  0.2, 0.65, 1.0],
            TrailingEdgeSegmentLength = [5e-4, 3e-4, 2e-4, 1.75e-4],
            LeadingEdgeSegmentLength = [8e-4, 5e-4, 3e-4, 2.5e-4],
            LeadingEdgeAbscissa = [0.485, 0.497, 0.499, 0.499],
            TopSideNumberOfPoints = 41, # must be odd
            BottomSideNumberOfPoints = 41, # must be odd
            TopToBottomAtTipNumberOfPoints = 9,
            InterpolationLaw = 'interp1d_linear'))



    RotorHgridXlocations = RW.proposeHgridXlocations(rotor,profile,0.40) # (-1.50, -0.75)
    StatorHgridXlocations = RW.proposeHgridXlocations(stator, profile,0.40) # (-0.45, 0.20)

    discretizations = RW.getSimpleORASHubProfileDiscretizations(rotor, stator,
        RotorNumberOfBlades=RotorNumberOfBlades,
        StatorNumberOfBlades=StatorNumberOfBlades,
        AzimutalCellAngle=1.0,
        InterfaceAxialCellLength=1.5e-2,
        BreakPointsAxialCellLength=1.5e-2,
        
        # rotor hub profile discretization
        RotorHgridXlocations=RotorHgridXlocations,
        RotorFrontNPts=21,
        RotorRearNPts=9,
        RotorFrontSegmentLength=0.1,
        
        # stator hub profile discretization
        StatorHgridXlocations=StatorHgridXlocations,
        StatorFrontNPts=9,
        StatorRearNPts=15,
        StatorRearSegmentLength=1.0
            )


    t = RW.buildOpenRotorAndStatorMesh(rotor,stator,profile,
            FarfieldRadius = 3,

            RotorNumberOfBlades=RotorNumberOfBlades,
            RotorDeltaPitch=  0.0,
            RotorPitchCenter= 0.0,
            RotorRadialExtrusionNbOfPoints=9,
            RotorBladeWallCellHeight = 2e-3,
            RotorHubWallCellHeight = 3e-2,
            RotorBladeWallGrowthRate=1.1,
            RotorTipScaleFactorAtRadialFarfield = 0.01,
            RotorBladeRootWallNormalDistanceRelativeToRootChord = 0.05,
            RotorHubProfileReDiscretization = discretizations[0],
            RotorAzimutalCellAngle = AzimutalCellAngleInDegrees,
            RotorHgridXlocations=RotorHgridXlocations,
            RotorHgridNbOfPoints=9,
            RotorHspreadingAngles = [-10, +2],
            RotorRootRemeshRadialNbOfPoints=9,
            RotorFarfieldProfileAbscissaDeltas=[0,-0.006,0],
            RotorFarfieldTipSmoothIterations=30,
            RotorRootWallRemeshRadialDistanceRelativeToMaxRadius=0.25,

            # NOTE complex rotor tip region requires specific optimum indexing H
            # grid closer to root (low radial_H_compromise) highly smoothed support
            # of H at tip region (high relax_relative_length ) as well as higher
            # tip normal tension (tip_radial_tension)
            RotorBuildMatchMeshAdditionalParams=dict(
                radial_H_compromise=0.05,
                relax_relative_length=1,
                tip_radial_tension=0.30),

            StatorNumberOfBlades=StatorNumberOfBlades,
            StatorDeltaPitch = 0.0,
            StatorPitchCenter = 0.9955,
            StatorRadialExtrusionNbOfPoints=15,
            StatorBladeWallCellHeight = 2e-3,
            StatorHubWallCellHeight = 3e-2,
            StatorBladeRootWallNormalDistanceRelativeToRootChord = 0.03,
            StatorHubProfileReDiscretization = discretizations[1],
            StatorAzimutalCellAngle = AzimutalCellAngleInDegrees,
            StatorHgridXlocations=StatorHgridXlocations,
            StatorHgridNbOfPoints=9,
            StatorFarfieldProfileAbscissaDeltas=[0.02,0.05],
            StatorFarfieldTipSmoothIterations=10,
            StatorHspreadingAngles = [-2.5, 5],
            StatorRootRemeshRadialNbOfPoints=9,
            StatorRootWallRemeshRadialDistanceRelativeToMaxRadius=0.25,

            StatorBuildMatchMeshAdditionalParams=dict(
                radial_H_compromise=0,
                relax_relative_length=1,
                tip_radial_tension=0.05),



            LOCAL_DIRECTORY_CHECKME=check_dir,
            raise_error_if_negative_volume_cells=raise_error_if_negative_volume_cells,
            )

    J.printElapsedTime('total meshing time was:', previous_timer=toc)
    
    toc = J.tic()

    if not test_mode_else_debug:
        print('will save mesh')
        J.save(t,os.path.join(tmp_path,'mesh.cgns'))
        J.printElapsedTime('saving mesh took:', previous_timer=toc)
        print(msg_restricted)


@pytest.mark.user_case
@pytest.mark.restricted_user_case
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_4
def test_oras_mesher_barrier_acoustic(tmp_path):

    if os.getenv('USER') != 'lbernard': return # confidential case

    import mola.pytree.InternalShortcuts as J
    import mola.mesh.propeller_mesher as RW

    test_mode_else_debug = True # True:testing False:debugging
    msg_restricted = J.WARN+"WARNING: RESTRICTED USER CASE - SHARING INPUT/OUTPUT DATA IS FORBIDDEN"+J.ENDC

    if test_mode_else_debug:
        check_dir = ''
        raise_error_if_negative_volume_cells = True
    else:
        print(msg_restricted)
        check_dir = os.path.join(tmp_path,'CHECK_ME')
        raise_error_if_negative_volume_cells = False

    RotorNumberOfBlades = 14
    StatorNumberOfBlades = 12
    AzimutalCellAngleInDegrees = 0.53

    toc = J.tic()

    profile = J.load("/stck/lbernard/PROJETS/CLEOPATRA/TEST_GEOM_RAPHAEL/hub_profile.cgns", return_type='zone')
    x = J.getx(profile)
    x[-1] = 20 # prolongation
    rotor = J.load("/stck/lbernard/PROJETS/CLEOPATRA/TEST_GEOM_RAPHAEL/rotor_blade.cgns")
    stator = J.load("/stck/lbernard/PROJETS/CLEOPATRA/TEST_GEOM_RAPHAEL/stator_blade.cgns")

    # extrapolate at root in order to guarantee that blade fully intersects hub
    rotor_blade = J.selectZoneWithHighestNumberOfPoints(rotor)
    rotor_blade = RW.GSD.extrapolateSurface(rotor_blade,'jmin',0.05)
    stator_blade = J.selectZoneWithHighestNumberOfPoints(stator)
    stator_blade = RW.GSD.extrapolateSurface(stator_blade,'jmin',0.05)

    rotor = RW.rediscretizeBlade(rotor_blade,
        RadialNbOfPoints = 190,
        RadialCellLengthAtTip = 8e-4,
        RadialCellLengthAtRoot = 0.02,
        SectionsDistribution = dict(
            RelativeAbscissa =   [0.0,  0.2, 0.65, 1.0],
            TrailingEdgeSegmentLength = [2e-4, 1.5e-4, 1e-4, 9e-5],
            LeadingEdgeSegmentLength = [7e-4, 5e-4, 2e-4, 1.6e-4],
            LeadingEdgeAbscissa = [0.517, 0.503, 0.500, 0.5015],
            TopSideNumberOfPoints = 173, # must be odd
            BottomSideNumberOfPoints = 173, # must be odd
            TopToBottomAtTipNumberOfPoints = 51,
            InterpolationLaw = 'interp1d_linear'))

    stator = RW.rediscretizeBlade(stator_blade,
        RadialNbOfPoints = 133,
        RadialCellLengthAtTip = 5e-4,
        RadialCellLengthAtRoot = 0.02,
        SectionsDistribution = dict(
            RelativeAbscissa =   [0.0,  0.2, 0.65, 1.0],
            TrailingEdgeSegmentLength = [2e-4, 1.5e-4, 1e-4, 8e-5],
            LeadingEdgeSegmentLength = [6.5e-4, 5e-4, 2e-4, 2e-4],
            LeadingEdgeAbscissa = [0.485, 0.497, 0.499, 0.499],
            TopSideNumberOfPoints = 157, # must be odd
            BottomSideNumberOfPoints = 157, # must be odd
            TopToBottomAtTipNumberOfPoints = 31,
            InterpolationLaw = 'interp1d_linear'))



    RotorHgridXlocations = RW.proposeHgridXlocations(rotor,profile,0.40) # (-1.50, -0.75)
    StatorHgridXlocations = RW.proposeHgridXlocations(stator, profile,0.40) # (-0.45, 0.20)

    discretizations = RW.getSimpleORASHubProfileDiscretizations(rotor, stator,
        RotorNumberOfBlades=RotorNumberOfBlades,
        StatorNumberOfBlades=StatorNumberOfBlades,
        AzimutalCellAngle=AzimutalCellAngleInDegrees,
        InterfaceAxialCellLength=4e-3,
        BreakPointsAxialCellLength=4e-3,

        # rotor hub profile discretization
        RotorHgridXlocations=RotorHgridXlocations,
        RotorFrontNPts=300,
        RotorRearNPts=50,
        RotorFrontSegmentLength=7e-3,
        
        # stator hub profile discretization
        StatorHgridXlocations=StatorHgridXlocations,
        StatorFrontNPts=50,
        StatorRearNPts=165,
        StatorRearSegmentLength=2.0
            )

    t = RW.buildOpenRotorAndStatorMesh(rotor,stator,profile,
            FarfieldRadius = 20,

            RotorNumberOfBlades=RotorNumberOfBlades,
            RotorThetaAdjustmentInDegrees=1.0,
            RotorDeltaPitch=  0.0,
            RotorPitchCenter= 0.0,
            RotorRadialExtrusionNbOfPoints=77,
            RotorBladeWallCellHeight = 2.5e-5,
            RotorHubWallCellHeight = 2.5e-5,
            RotorBladeWallGrowthRate=1.12,
            RotorBladeRootWallNormalDistanceRelativeToRootChord = 0.05,
            RotorHubProfileReDiscretization = discretizations[0],
            RotorAzimutalCellAngle = AzimutalCellAngleInDegrees,
            RotorHgridXlocations=RotorHgridXlocations,
            RotorHgridNbOfPoints=31,
            RotorHspreadingAngles = [-10, +2],
            RotorRootRemeshRadialNbOfPoints=50,
            RotorFarfieldProfileAbscissaDeltas=[0,-0.050,0],
            RotorFarfieldRadialCellLengthRelativeToFarfieldRadius=0.10,
            RotorRootWallRemeshRadialDistanceRelativeToMaxRadius=0.05,

            RotorBuildMatchMeshAdditionalParams=dict(
                radial_H_compromise=0.05,
                relax_relative_length=1.0,
                tip_radial_tension=0.0175,
                ),


            StatorNumberOfBlades=StatorNumberOfBlades,
            StatorDeltaPitch = 0.0,
            StatorPitchCenter = 0.0,
            StatorRadialExtrusionNbOfPoints=77,
            StatorBladeWallCellHeight = 2.5e-5,
            StatorHubWallCellHeight = 2.5e-5,
            StatorBladeWallGrowthRate=1.12,
            StatorBladeRootWallNormalDistanceRelativeToRootChord = 0.05,
            StatorHubProfileReDiscretization = discretizations[1],
            StatorAzimutalCellAngle = AzimutalCellAngleInDegrees,
            StatorHgridXlocations=StatorHgridXlocations,
            StatorHgridNbOfPoints=31,
            StatorHspreadingAngles = [-2.5, 5],
            StatorRootRemeshRadialNbOfPoints=50,
            StatorFarfieldProfileAbscissaDeltas=[+0.10,+0.30],
            StatorFarfieldRadialCellLengthRelativeToFarfieldRadius=0.10,
            StatorRootWallRemeshRadialDistanceRelativeToMaxRadius=0.05,

            StatorBuildMatchMeshAdditionalParams=dict(
                radial_H_compromise=0.05,
                relax_relative_length=1.0,
                tip_radial_tension=0.005,
                ),


            LOCAL_DIRECTORY_CHECKME=check_dir,
            raise_error_if_negative_volume_cells=raise_error_if_negative_volume_cells,
            )

    J.printElapsedTime('total meshing time was:', previous_timer=toc)
    
    toc = J.tic()

    if not test_mode_else_debug:
        print('will save mesh')
        J.save(t,os.path.join(tmp_path,'mesh.cgns'))
        J.printElapsedTime('saving mesh took:', previous_timer=toc)
        print(msg_restricted)



if __name__ == '__main__':

    test_oras_mesher_designer_ultracoarse('test_oras_mesher_designer_ultracoarse')
    # test_oras_mesher_ultracoarse('test_oras_mesher_ultracoarse')
    # test_oras_mesher_inpro_ultracoarse("test_oras_mesher_inpro_ultracoarse")    
    # test_oras_mesher_barrier_ultracoarse("test_oras_mesher_barrier_ultracoarse")
    # test_oras_mesher_single('test_oras_mesher_single')
    # test_oras_mesher_inpro('test_oras_mesher_inpro')
    # test_oras_mesher_barrier_acoustic("test_oras_mesher_barrier_acoustic")
