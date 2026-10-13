# 41k cells
ultra_light_sharp = dict(

    # farfield parameters
    extension=2.0,
    external_cell_size=0.2,
    RadialNbOfPointsInFarfield=11,
    FarfieldProfileAbscissaDeltas=[0.08,0.13,0.14],

    # spinner parameters
    hub_radius=0.3,
    AzimutalCellAngle=4.0,
    spinner_wall_slip_cell_size = 0.012,
    spinner_profile_nb_pts_front = 21,
    spinner_profile_nb_pts_rear = 21,
    spinner_adjust_rotation = 6.0,

    # near-blade H grid region parameters
    H_grid_axial_cell_length_transitions = 0.015,
    H_grid_X_locations = [-0.15, 0.26],
    H_grid_interior_spreading_angles=[-15,15],
    H_interior_connecting_nb_points = 11,
    H_inner_normal_tension = 0.3,
    H_outter_normal_tension = 0.1,
    topological_radial_compromise_for_H_region = 0.0,

    # blade parameters
    max_radius=1.0,
    pitch=50.0,
    nb_blades=8,
    virtual_nb_blades_if_two_bladed = 6,
    RadialNbOfPoints = 21,
    RadialCellLengthAtTip = 0.01,
    blade_wall_noslip_cell_size = 0.001,
    boundary_layer_growth_rate = 1.20,
    max_boundary_layer_extension_relative_to_chord=0.04,
    reconstructed_blade_from_root_portion = 0.25,
    reconstructed_blade_from_root_nb_pts = 11,

    ChordDistribution = dict(RelativeSpan=[0,1],
                                Chord=[0.21,0.21],
                                InterpolationLaw='interp1d_linear'),
    TwistDistribution = dict(RelativeSpan=[0,1],
                                Twist=[30.0,-5.0],
                                InterpolationLaw='interp1d_linear'),
    DihedralDistribution = dict(RelativeSpan=[0,1],
                                Dihedral=[0.0,0.0],
                                InterpolationLaw='interp1d_linear'),
    SweepDistribution = dict(RelativeSpan=[0,1],
                                Sweep=[0.0,0.0],
                                InterpolationLaw='interp1d_linear'),
    SectionsDistribution = dict(
        RelativeSpan =   [0.0,1.0],
        AirfoilZonesOrNACAstringsOrFilenames = ['NACA4412','NACA4412'],
        TrailingEdgeSegmentLengthRelativeToChord = 0.01, # can be a list
        LeadingEdgeSegmentLengthRelativeToChord =  0.005, # can be a list
        LeadingEdgeAbscissa = 0.494, # can be a list
        StackingPointRelativeToChord = 0.25,
        TopSideNumberOfPoints = 31, # must be odd
        BottomSideNumberOfPoints = 31, # must be odd
        TopToBottomAtTipNumberOfPoints = 15,
        InterpolationLaw = 'interp1d_linear',
        ),
    sharp_TrailingEdge_TFI_chord_ref=0.2,
    raise_error_if_negative_volume_cells=True,
)

# 8M cell
two_blade = dict(
    # farfield parameters
    extension=10.0,
    external_cell_size=1.0,
    RadialNbOfPointsInFarfield=61,
    FarfieldProfileAbscissaDeltas=[0.2,0.4,0.5],

    # spinner parameters
    hub_radius=0.3,
    AzimutalCellAngle=1.5,
    spinner_wall_slip_cell_size = 0.012,
    spinner_profile_nb_pts_front = 80,
    spinner_profile_nb_pts_rear = 121,
    spinner_adjust_rotation = 6.0,

    # near-blade H grid region parameters
    H_grid_axial_cell_length_transitions = 4e-3,
    H_grid_X_locations = [-0.15, 0.26],
    H_grid_interior_spreading_angles=[-15,15],
    H_interior_connecting_nb_points = 30,
    H_inner_normal_tension = 0.3,
    H_outter_normal_tension = 0.1,
    topological_radial_compromise_for_H_region = 0.0,

    # blade parameters
    max_radius=1.0,
    pitch=50.0,
    nb_blades=2,
    virtual_nb_blades_if_two_bladed = 6,
    RadialNbOfPoints = 121,
    RadialCellLengthAtTip = 8e-3,
    blade_wall_noslip_cell_size = 2e-6,
    boundary_layer_growth_rate = 1.15,
    max_boundary_layer_extension_relative_to_chord=0.10,
    reconstructed_blade_from_root_portion = 0.20,
    reconstructed_blade_from_root_nb_pts = 21,

    ChordDistribution = dict(RelativeSpan=[0,1],
                                Chord=[0.21,0.21],
                                InterpolationLaw='interp1d_linear'),
    TwistDistribution = dict(RelativeSpan=[0,1],
                                Twist=[30.0,-5.0],
                                InterpolationLaw='interp1d_linear'),
    DihedralDistribution = dict(RelativeSpan=[0,1],
                                Dihedral=[0.0,0.0],
                                InterpolationLaw='interp1d_linear'),
    SweepDistribution = dict(RelativeSpan=[0,1],
                                Sweep=[0.0,0.0],
                                InterpolationLaw='interp1d_linear'),
    SectionsDistribution = dict(
        RelativeSpan =   [0.0,1.0],
        AirfoilZonesOrNACAstringsOrFilenames = ['NACA4412','NACA4412'],
        TrailingEdgeSegmentLengthRelativeToChord = 2e-3, # can be a list
        LeadingEdgeSegmentLengthRelativeToChord =  3e-3, # can be a list
        LeadingEdgeAbscissa = 0.494, # can be a list
        StackingPointRelativeToChord = 0.25,
        TopSideNumberOfPoints = 131, # must be odd
        BottomSideNumberOfPoints = 131, # must be odd
        TopToBottomAtTipNumberOfPoints = 21,
        InterpolationLaw = 'interp1d_linear',
        ),
    sharp_TrailingEdge_TFI_chord_ref=0.1,
    raise_error_if_negative_volume_cells=True,
)

# 6M cell
six_blade_swept_thick_foil = dict(
    # farfield parameters
    extension=10.0,
    external_cell_size=1.0,
    RadialNbOfPointsInFarfield=121,
    FarfieldProfileAbscissaDeltas=[0.2,0.45,0.55],

    # spinner parameters
    hub_radius=0.3,
    AzimutalCellAngle=1.5,
    spinner_wall_slip_cell_size = 8e-3,
    spinner_profile_nb_pts_front = 80,
    spinner_profile_nb_pts_rear = 121,
    spinner_adjust_rotation = 0.0,

    # near-blade H grid region parameters
    H_grid_axial_cell_length_transitions = 4e-3,
    H_grid_X_locations = [-0.2218, 0.278056],
    H_grid_interior_spreading_angles=[-25,20],
    H_interior_connecting_nb_points = 30,
    H_inner_normal_tension = 0.3,
    H_outter_normal_tension = 0.1,
    topological_radial_compromise_for_H_region = 0.0,

    # blade parameters
    max_radius=1.18,
    pitch=38.0,
    nb_blades=6,
    virtual_nb_blades_if_two_bladed = 6,
    RadialNbOfPoints = 121,
    RadialCellLengthAtTip = 8e-3,
    blade_wall_noslip_cell_size = 2e-6,
    boundary_layer_growth_rate = 1.10,
    max_boundary_layer_extension_relative_to_chord=0.05,
    reconstructed_blade_from_root_portion = 0.2,
    reconstructed_blade_from_root_nb_pts = 21,

    ChordDistribution = dict(RelativeSpan=[0.25, 0.60, 0.92, 1.00],
                                    Chord=[0.20, 0.24, 0.20, 0.15],
                                InterpolationLaw='interp1d_quadratic'),
    TwistDistribution = dict(RelativeSpan=[0.25, 0.75, 1.00],
                                Twist    =[  24, 0.00,-8.0],
                                InterpolationLaw='interp1d_quadratic'),
    DihedralDistribution = dict(RelativeSpan=[0.25,1.00],
                                Dihedral=[0.00,0.01],
                                InterpolationLaw='interp1d_linear'),
    SweepDistribution = dict(RelativeSpan=[ 0.25, 0.60, 0.92, 1.00],
                                Sweep=[-0.02,-0.05,-0.01, 0.03],
                                InterpolationLaw='interp1d_quadratic'),
    SectionsDistribution = dict(
        RelativeSpan = [0.25,
                        0.50,
                        1.0],
        AirfoilZonesOrNACAstringsOrFilenames = [
            '/stck/mola/data/open/airfoils/dummy/root_thick.dat',
            '/stck/mola/data/open/airfoils/dummy/mid_thick.dat',
            '/stck/mola/data/open/airfoils/dummy/tip_thick.dat',
            ],
        TrailingEdgeSegmentLengthRelativeToChord = 2e-3, # can be a list
        LeadingEdgeSegmentLengthRelativeToChord =  [3e-3, 0.7e-3, 0.5e-3], # can be a list
        LeadingEdgeAbscissa = [0.480, 0.495, 0.497], # can be a list
        StackingPointRelativeToChord = 0.25,
        TopSideNumberOfPoints = 131, # must be odd
        BottomSideNumberOfPoints = 131, # must be odd
        TopToBottomAtTipNumberOfPoints = 21,
        InterpolationLaw = 'interp1d_linear',
        ),
    raise_error_if_negative_volume_cells=True,
)

