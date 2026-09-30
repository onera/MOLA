from mola.workflow.fixed.airplane.workflow import WorkflowAirplane

solver = 'sonics'
example_name = 'isol_wing_'+solver

w = WorkflowAirplane(

    Solver=solver,

    RawMeshComponents=[
        dict(
            Name='Wing',
            Source='/stck/mola/data/open/mesh/isolated_wing/raw_mesh.cgns',
            Families=[
                dict(Name='WING',     Location='kmin'),
                dict(Name='FARFIELD', Location='kmax'),
                dict(Name='SYMMETRY', Location='planeXZ'),
            ],
            Connection = [dict(Type='Match', Tolerance=1e-8)]
        )
    ],


    Flow = dict(
        Velocity = 50.0,
        Density = 1.225,
        Temperature = 288.0,
    ),

    ApplicationContext = dict(
        AngleOfAttackDeg = 4.0,
        Surface = 0.15,
        Length = 0.15,
    ),

    Turbulence = dict(
        Level = 0.1 * 0.01,
        Viscosity_EddyMolecularRatio = 0.1,
        Model = 'SA',
    ),

    Numerics = dict(
        NumberOfIterations=3000,
        MinimumNumberOfIterations=3,
        CFL=dict(StartIteration =    1, StartValue =  1.0,
                 EndIteration   = 1000,   EndValue = 10.0),
    ),

    BoundaryConditions = [
        dict(Family='WING', Type='Wall'),
        dict(Family='FARFIELD', Type='Farfield'),
        dict(Family='SYMMETRY', Type='SymmetryPlane'),
    ],

    ConvergenceCriteria = [
        dict(
            ExtractionName = 'WING',
            Variable = "rsd-CD",
            Threshold = 1e-4,
        ),
        dict(
            ExtractionName = 'WING',
            Variable = "CL",
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

# w.prepare()
# w.write_cfd_files()
w.submit()

# MOLA_CI_MODE: "mono"