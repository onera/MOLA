from mola.workflow.rotating_component import propeller

solver = 'sonics'
example_name = 'prop_light_'+solver

w = propeller.Workflow( 

    Solver=solver,

    Mesh='/stck/mola/data/open/mesh/light_propeller/mesh.cgns',
    
    Flow = dict(
        Velocity = 10.0,
        Density = 1.225,
        Temperature = 288.15,
    ),

    ApplicationContext = dict(
        ShaftRotationSpeed = -2000.0, # please NOTE the minus sign
        NumberOfBlades = 5,
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
        NumberOfIterations=2000,
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
        dict(Type="IsoSurface", IsoSurfaceField="CoordinateY", IsoSurfaceValue=0.45, Fields=['MomentumX','MomentumY','MomentumZ','Viscosity_EddyMolecularRatio']),
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
        NumberOfProcessors = 8,
        JobName=example_name,
        RunDirectory = '/tmp_user/juno/$USER/mola_examples/'+example_name,
        TimeLimit = '3:00:00',
    ),
)
w.ApplicationContext['_BladeRadius'] = 0.6 # FIXME BUG https://gitlab.onera.net/numerics/mola/-/work_items/324

# w.prepare()
# w.write_cfd_files()
w.submit()

# MOLA_CI_MODE: "mono"