from mola.workflow.rotating_component.propeller.workflow import WorkflowPropeller, solver
import numpy as np

w = WorkflowPropeller(

    RawMeshComponents=[
        dict(
            Name='LIGHT',
            Source='/stck/mola/data/open/mesh/light_propeller/mesh.cgns',
        )
    ],


    Flow = dict(
        Velocity = 1.0,
        Density = 1.225,
        Temperature = 288.15,
    ),

    ApplicationContext = dict(
        ShaftRotationSpeed = 2000.0,
        NumberOfBlades = 5,
        Surface = 1.0,
        Length = 1.0
    ),

    Turbulence = dict(
        Level = 0.1 * 0.01,
        Viscosity_EddyMolecularRatio = 0.1,
        Model = 'SA',
    ),

    # Initialization = dict(WallDistanceComputingTool='cassiopee'),

    Numerics = dict(
        NumberOfIterations=10,
        MinimumNumberOfIterations=3,
        CFL=dict(StartIteration =    1, StartValue =  1.0,
                 EndIteration   = 1000,   EndValue = 10.0),
    ),

    BoundaryConditions = [
        dict(Family='SPINNER', Type='WallInviscid'),
        dict(Family='BLADE', Type='WallViscous'),
        dict(Family='FARFIELD', Type='Farfield')
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
            Threshold = -np.inf, # HINT just for showing Variable progress in coprocess.log
        ),
        dict(
            ExtractionName = 'BLADE',
            Variable = "Power",
            Threshold = -np.inf, # HINT just for showing Variable progress in coprocess.log
        )
    ],

    RunManagement = dict(
        NumberOfProcessors = 8,
        RunDirectory = f'example_{solver}',
        Scheduler = 'local',
    ),

    # HINT if running in local LD, avoid using PyPart since it provokes huge overhead https://elsa.onera.fr/issues/11440
    SplittingAndDistribution = dict(
        Strategy='AtPreprocess',
        Splitter='Cassiopee',
        Distributor='Cassiopee',
    ),
)

w.prepare()
w.write_cfd_files()
w.submit()
w.assert_completed_without_errors()
