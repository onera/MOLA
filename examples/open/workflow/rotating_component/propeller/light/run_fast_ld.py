from mola.workflow.rotating_component import propeller, solver
import numpy as np

assert solver == 'fast'

w = propeller.Workflow(
    
    RawMeshComponents=[
        dict(
            Name='LIGHT',
            Source='/stck/mola/data/open/mesh/light_propeller/mesh.cgns',
        )
    ],

    Flow = dict(
        Velocity = 10.0,
        Density = 1.225,
        Temperature = 288.15,
    ),

    ApplicationContext = dict(
        ShaftRotationSpeed = -2000.0,
        NumberOfBlades = 5,
        Surface = 1.0,
        Length = 1.0
    ),

    Turbulence = dict(
        Level = 0.1 * 0.01,
        Viscosity_EddyMolecularRatio = 0.1,
        Model = 'SA',
    ),

    Numerics = dict(
        NumberOfIterations=2000,
        MinimumNumberOfIterations=20,
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

    Initialization = dict(WallDistanceComputingTool='cassiopee'),
    
    RunManagement = dict(
        NumberOfProcessors = 1,
        NumberOfThreads = 8,
        RunDirectory = f'example_{solver}',
        
    ),
)

w.prepare()
w.write_cfd_files()
w.submit()
