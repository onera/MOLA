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
def test_design_blade(tmp_path):

    import mola.legacy.propeller_mesher as RW
    blade = RW.designBlade(RightHandRuleRotation=False)


@pytest.mark.user_case
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_1
def test_getBladesORAS_ONERA_SE(tmp_path):

    import mola.legacy.propeller_mesher as RW
    rotor, stator = RW.getBladesORAS_ONERA_SE()


@pytest.mark.user_case
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_3
def test_oras_mesher(tmp_path):

    import mola.legacy.InternalShortcuts as J
    import mola.legacy.propeller_mesher as RW

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

    import mola.legacy.InternalShortcuts as J
    import mola.legacy.propeller_mesher as RW

    test_mode_else_debug = True # True:testing False:debugging

    if test_mode_else_debug:
        check_dir = ''
        raise_error_if_negative_volume_cells = True
    else:
        check_dir = os.path.join(tmp_path,'CHECK_ME')
        raise_error_if_negative_volume_cells = False

    RotorNumberOfBlades = 13
    StatorNumberOfBlades = 11
    AzimutalCellAngleInDegrees = 3.0 

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
        RotorFrontNPts=21,
        RotorRearNPts=9,
        RotorFrontSegmentLength=0.1,
        
        # stator hub profile discretization
        StatorHgridXlocations=StatorHgridXlocations,
        StatorFrontNPts=9,
        StatorRearNPts=21,
        StatorRearSegmentLength=0.15
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

    import mola.legacy.InternalShortcuts as J
    import mola.legacy.propeller_mesher as RW

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
        RotorFrontNPts=27,
        RotorRearNPts=8,
        RotorFrontSegmentLength=5e-4,
        
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
        RotorFarfieldAxialSpreadingAngles= [-15,-20,-6],

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
        StatorFarfieldAxialSpreadingAngles= [2, 26],

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



if __name__ == '__main__':
    # test_oras_mesher_designer_ultracoarse('test_oras_mesher_designer_ultracoarse')
    # test_oras_mesher('test_oras_mesher')
    test_oras_mesher_ultracoarse('test_oras_mesher_ultracoarse')
    # test_design_blade('test_design_blade')
    # test_getBladesORAS_ONERA_SE('test_getBladesORAS_ONERA_SE')

