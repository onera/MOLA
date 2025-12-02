from mola.workflow.fixed.airplane.workflow import WorkflowAirplane

# WorkflowAirplane().print_interface();exit()

w = WorkflowAirplane(

    Solver='elsa',

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
        NumberOfIterations=2000,
        CFL=dict(StartIteration =    1, StartValue =  1.0,
                 EndIteration   = 1000,   EndValue = 10.0),
    ),

    BoundaryConditions = [
        dict(Family='WING', Type='Wall'),
        dict(Family='FARFIELD', Type='Farfield'),
        dict(Family='SYMMETRY', Type='SymmetryPlane'),
    ],

    SplittingAndDistribution = dict (
        Strategy = 'AtComputation',
        Splitter = 'PyPart',
        Distributor = 'PyPart',
    ),

    RunManagement = dict(
        NumberOfProcessors = 1,
        RunDirectory = '/tmp_user/sator/$USER/.test_user_case/isolated_wing',
        Machine='sator',
        TimeLimit = '00:30:00',
        AER='34790003F',
    ),
)

w.prepare()
w.write_cfd_files()
w.submit()
