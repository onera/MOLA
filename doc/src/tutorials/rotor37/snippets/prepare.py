from mola.workflow.rotating_component import turbomachinery
from treelab import cgns

w = turbomachinery.Workflow( 
    RawMeshComponents=[
        dict(
            Name='rotor37',
            Source = '/stck/mola/data/mesh/rotor37/rotor37.cgns',
            Unit = 'cm',
            )
    ],

    ApplicationContext = dict(
        # Shaft speed
        ShaftRotationSpeed = -1800.,  # given in rad/s, so it needs to be specified
        ShaftRotationSpeedUnit = 'rad/s',  # otherwise default unit is 'rpm'
        # Hub rotation speed
        # List of tuples. Each tuple (xmin, xmax) corresponds to a CoordinateX
        # interval where the speed at hub wall is ShaftRotationSpeed. It is zero
        # outsides these intervals.
        HubRotationIntervals = [(-999.0, 999.0)], # Here the whole hub is rotating (no stator part)

        # This dictionary has one entry for each row domain.
        # The key names must be the family names in the CGNS Tree.
        Rows = dict(
            R37 = dict(
                IsRotating = True, 
                # The number of blades in the row (on 360 degrees)
                NumberOfBlades = 36,
                # The number of blades in the computational domain
                # set to <NumberOfBlades> for a full 360 simulation
                # The default value is 1
                # If the value is >1, the mesh will be duplicated if it is not already
                # NumberOfBladesSimulated = 2,
            )
        )
    ),

    Flow = dict(
        MassFlow              = 20.5114,  # for the 360 degrees section, even it is simulated entirely
        TemperatureStagnation = 288.15,
        PressureStagnation    = 101330.,
    ),

    Turbulence = dict(
        Level = 0.03,
        Viscosity_EddyMolecularRatio = 0.1,
        # Possible values for the Model are :
        # 'SA', 'BSL','BSL-V','SST-2003','SST','SST-V','Wilcox2006-klim',
        # 'SST-2003-LM2009', 'SSG/LRR-RSM-w2012', 'smith'
        # Most names follow the NASA convention: https://turbmodels.larc.nasa.gov/
        Model='SA',
    ),

    Numerics = dict(
        NumberOfIterations = 10000,
        CFL = dict(EndIteration=300, StartValue=1., EndValue=30.)
    ),

    BoundaryConditions = [
        dict(Family='R37_INFLOW', Type='InflowStagnation'),
        dict(Family='R37_OUTFLOW', Type='OutflowPressure', Pressure=0.95e5), #0.9936*1e5),
        # dict(Family='R37_OUTFLOW', Type='OutflowRadialEquilibrium', Pressure=1e5),
    ],

    Initialization = dict(
        Method = 'uniform',
        ParametrizeWithHeight = True,
        # ComputeWallDistanceAtPreprocess = True, 
    ),

    # The list Extractions gathers all the extractions that will be done in co-processing.
    # Each element is a dictionary that triggers one type of extraction.
    Extractions = [
        dict(Type='BC', Source='BCWall*', Fields=['Pressure', 'BoundaryLayer', 'yPlus']),
        dict(Type='BC', Source='R37_INFLOW', Fields=['Pressure']),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=-0.03, OtherOptions=dict(tag='InletPlane', ReferenceRow='R37')),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=0.07, OtherOptions=dict(tag='OutletPlane', ReferenceRow='R37')),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateY', IsoSurfaceValue=0.23)
    ],    

    # ConvergenceCriteria = [
    #     dict(
    #         ExtractionName = 'R37_INFLOW',
    #         Variable  = 'rsd-MassFlow',
    #         Threshold = 1e-3,
    #     ),
    #     dict(
    #         Family    = 'PERFOS_R37',
    #         Variable  = 'rsd-PressureStagnationRatio',
    #         Threshold = 1e-5,
    #     ),
    # ],

    RunManagement=dict(
        JobName = 'rotor37',
        NumberOfProcessors = 4,
        # RunDirectory = '/tmp_user/sator/$USER/test/',
        # AER='XXXXXXXXX',
        # TimeLimit = '00:30:00',
        ),

    )


w.prepare()
w.write_cfd_files()
w.submit()

