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
    import mola.legacy.InternalShortcuts as J
    import mola.legacy.propeller_mesher as RW

    blade = RW.designBlade(RightHandRuleRotation=False)


@pytest.mark.user_case
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_3
def test_oras_mesher(tmp_path):

    import mola.legacy.InternalShortcuts as J
    import mola.legacy.propeller_mesher as RW

    test_mode_else_debug = False # True:debugging False:testing

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


    if test_mode_else_debug:
        check_dir = ''
        raise_error_if_negative_volume_cells = True
    else:
        check_dir = os.path.join(tmp_path,'CHECK_ME')
        raise_error_if_negative_volume_cells = False


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
def test_oras_mesher_straight_blades(tmp_path):

    import mola.legacy.InternalShortcuts as J
    import mola.legacy.propeller_mesher as RW

    RotorNumberOfBlades = 9
    StatorNumberOfBlades = 11
    AzimutalCellAngle = 1.0 # deg

    RotorHgridXlocations=(-0.04, 0.075)
    StatorHgridXlocations=(0.12, 0.20)

    profile = RW.makeSpinnerCurves(LengthFront=0.2, LengthRear=10, Width=0.15,
                      RelativeArcRadiusFront=0.008, ArcAngleFront=40.,
                      RelativeTensionArcFront=0.1, RelativeTensionRootFront=0.5,
                      NPtsArcFront=200, NPtsSpinnerFront=5000,
                      TopologyRear='line')


    DIRECTORY_CHECKME = os.path.join(tmp_path,'CHECK_ME')
    try: os.makedirs(DIRECTORY_CHECKME)
    except: pass
    RW.DIRECTORY_CHECKME = DIRECTORY_CHECKME


    toc = J.tic()

    rotor = RW.designBlade(
        RadiusTip = 0.60,
        RadiusRoot = 0.05,

        RightHandRuleRotation = True,


        BladeStackPointPositionInXaxis = 0.0,
        BladePitchAxisPositionInXaxis = 0.0,
        PitchAngle = 55.0,
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
    )

    stator = RW.designBlade(
        RadiusTip = 0.5,
        RadiusRoot = 0.05,

        RightHandRuleRotation = False,

        BladeStackPointPositionInXaxis = 0.15,
        BladePitchAxisPositionInXaxis = 0.15,
        PitchAngle = 55.0,
        ZeroPitchAngleRelativeRadius = None, # if None, uses construction reference

        # Radial discretization of the blade geometry:
        RadialNbOfPoints = 51,
        RadialCellLengthAtTip = 0.0005,
        RadialCellLengthAtRoot = 0.01,

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
            TopSideNumberOfPoints = 67, # must be odd
            BottomSideNumberOfPoints = 67, # must be odd
            TopToBottomAtTipNumberOfPoints = 9,
            InterpolationLaw = 'interp1d_linear',
            ))


    discretizations = RW.getSimpleORASHubProfileDiscretizations(rotor, stator,
        RotorNumberOfBlades=RotorNumberOfBlades,
        StatorNumberOfBlades=StatorNumberOfBlades,

        AzimutalCellAngle=1.0,
        InterfaceAxialCellLength=2e-3,
        BreakPointsAxialCellLength=2e-3,
        
        # rotor hub profile discretization
        RotorHgridXlocations=RotorHgridXlocations,
        RotorFrontNPts=72,
        RotorRearNPts=12,
        RotorFrontSegmentLength=2e-3,
        
        # stator hub profile discretization
        StatorHgridXlocations=StatorHgridXlocations,
        StatorFrontNPts=10,
        StatorRearNPts=100,
        StatorRearSegmentLength=0.15)

    t = RW.buildOpenRotorAndStatorMesh(rotor,stator,profile,
        # signature
        FarfieldRadius= 2.0,
        InterfaceRadialTension= 1.0,
        InterfaceRelativePosition= 0.5,

        # ------------------------ ROTOR parameters ------------------------ #
        RotorNumberOfBlades= RotorNumberOfBlades,

        RotorDeltaPitch= 0.0,
        RotorPitchCenter= 0.0,
        RotorThetaAdjustmentInDegrees= -2.0,
        
        RotorHubProfileReDiscretization = discretizations[0],

        RotorAzimutalCellAngle= AzimutalCellAngle,

        RotorBladeWallCellHeight= 1e-5,
        RotorHubWallCellHeight= 1e-2,

        RotorBladeRootWallNormalDistance= 5e-3, 
        RotorRadialExtrusionNbOfPoints= 20,
        RotorHgridXlocations = RotorHgridXlocations,
        RotorHgridNbOfPoints= 21,
        RotorHspreadingAngles= [-10, 0],
        RotorTipScaleFactorAtRadialFarfield= 0.25,
        RotorFarfieldRadialCellLength= 0.25,
        RotorRadialTension= 0.05, # FIXME make completely normal front_near_topo
        RotorRelativeLengthOfRelaxation= 0.5,
        RotorFarfieldAxialSpreadingAngles= [-15,-20,-6],
        RotorBuildMatchMeshAdditionalParams= {},

        RotorBladeExtrusionParams= {},

        # ------------------------ STATOR parameters ------------------------ #

        StatorNumberOfBlades=StatorNumberOfBlades,

        StatorDeltaPitch= 0.0,
        StatorPitchCenter= 0.0,
        StatorThetaAdjustmentInDegrees= 1.0,
        
        StatorHubProfileReDiscretization = discretizations[1],

        StatorAzimutalCellAngle= AzimutalCellAngle,

        StatorBladeWallCellHeight= 1e-5,
        StatorHubWallCellHeight= 1e-2,

        StatorBladeRootWallNormalDistance= 5e-3, 
        StatorRadialExtrusionNbOfPoints= 20,
        StatorHgridXlocations = StatorHgridXlocations,
        StatorHgridNbOfPoints= 21,
        StatorHspreadingAngles= [0, 10],
        StatorTipScaleFactorAtRadialFarfield= 0.25,
        StatorFarfieldRadialCellLength= 0.25,
        StatorRadialTension= 0.1, # FIXME make completely normal rear_near_topo
        StatorRelativeLengthOfRelaxation= 0.5,
        StatorFarfieldAxialSpreadingAngles= [2, 26],
        StatorBuildMatchMeshAdditionalParams= {},

        StatorBladeExtrusionParams= {},



            )

    J.printElapsedTime('total meshing time was:', previous_timer=toc)
    
    toc = J.tic()
    print('will save mesh')
    J.save(t,os.path.join(tmp_path,'mesh.cgns'))
    J.printElapsedTime('saving mesh cost:', previous_timer=toc)


@pytest.mark.user_case
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_3
def test_getBladesORAS_ONERA_SE(tmp_path):

    import mola.legacy.propeller_mesher as RW
    import mola.legacy.InternalShortcuts as J

    try: os.makedirs(tmp_path)
    except: pass


    rotor, stator = RW.getBladesORAS_ONERA_SE()
    # t = J.tree(ROTOR=rotor, STATOR=stator)
    # RW.J.save(t,os.path.join(tmp_path,'blades_onera_se.cgns'))



if __name__ == '__main__':
    # test_propeller_mesher_light('test_propeller_mesher_light')
    test_oras_mesher('test_oras_mesher')
    # test_oras_mesher_straight_blades('test_oras_mesher_straight_blades')
    # test_design_blade('test_design_blade')
    # test_getBladesORAS_ONERA_SE('test_getBladesORAS_ONERA_SE')

    # import Generator.PyTree as G
    # import Transform.PyTree as T
    # c = G.cart((0,0,0),(5,2,1),(2,2,2))
    # T.rotate(c,(0,0,0),(0,0,1),30)
    # T.rotate(c,(0,0,0),(0,1,0),45)
    # bbox = G.BB(c)
    # bbox[0] ='bbox'
    
    # import mola.legacy.InternalShortcuts as J
    # J.save([c,bbox],'out.cgns')
