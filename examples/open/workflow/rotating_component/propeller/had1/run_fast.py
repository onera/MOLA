from mola.workflow.rotating_component import propeller

solver = 'fast'
example_name = 'had1_'+solver

w = propeller.Workflow( 

    Solver=solver,

    Mesh='/stck/mola/data/open/mesh/had1/mesh.cgns',

    Flow = dict(
        Velocity = 340.294*0.3,
        Density = 1.225,
        Temperature = 288.15,
    ),

    ApplicationContext = dict(
        ShaftRotationSpeed = -2030.0,
        NumberOfBlades = 3,
        Surface = 1.0,
        Length = 1.0
    ),

    Turbulence = dict(
        Level = 0.1 * 0.01,
        Viscosity_EddyMolecularRatio = 0.1,
        TurbulenceCutOffRatio = 1e-8,
        Model = 'SA',
    ),

    Numerics = dict(
        NumberOfIterations=4000,
        MinimumNumberOfIterations=3,
        CFL=dict(StartIteration =    1, StartValue =  1.0,
                 EndIteration   = 1000,   EndValue = 10.0),
    ),

    BoundaryConditions = [
        dict(Family='SPINNER', Type='WallInviscid'),
        dict(Family='BLADE', Type='WallViscous'),
        dict(Family='FARFIELD', Type='Farfield')
    ],

    Extractions = [
        dict(Type="IsoSurface", IsoSurfaceField="CoordinateX", IsoSurfaceValue=0.0, Fields=['MomentumX','MomentumY','MomentumZ','Viscosity_EddyMolecularRatio']),
        dict(Type="IsoSurface", IsoSurfaceField="CoordinateY", IsoSurfaceValue=0.3, Fields=['MomentumX','MomentumY','MomentumZ','Viscosity_EddyMolecularRatio']),
        dict(Type="IsoSurface", IsoSurfaceField="CoordinateZ", IsoSurfaceValue=0.0, Fields=['MomentumX','MomentumY','MomentumZ','Viscosity_EddyMolecularRatio']),
        dict(Type="BC", Source="BLADE", Fields=['Pressure']),
    ],

    ConvergenceCriteria = [
        dict(
            ExtractionName = 'BLADE',
            Variable = "std-Thrust",
            Threshold = 1.0,
        ),
        dict(
            ExtractionName = 'BLADE',
            Variable = "Thrust",
            Threshold = float('-inf'), # HINT just for showing Variable progress in coprocess.log
        ),
        dict(
            ExtractionName = 'BLADE',
            Variable = "Power",
            Threshold = float('-inf'), # HINT just for showing Variable progress in coprocess.log
        )
    ],

    RunManagement = dict(
        NumberOfProcessors = 96,
        JobName=example_name,
        RunDirectory = '/tmp_user/sator/$USER/mola_examples/'+example_name,
        TimeLimit = '15:00:00',
        AER='37753003F',
    ),
)

# w.prepare()
# w.write_cfd_files()
w.submit()

# BUG excluded from examples -> leads to NaN