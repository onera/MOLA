import os
from mola.workflow.rotating_component import propeller, solver
import numpy as np

assert solver == 'fast'

w = propeller.Workflow( 

    RawMeshComponents=[
        dict(
            Name='HAD1',
            Source='/stck/mola/data/open/mesh/had1/mesh.cgns',
        )
    ],

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
        NumberOfProcessors = 1,
        NumberOfThreads = 96,
        Machine='juno',
        RunDirectory = os.path.join(os.getcwd(),f'example_{solver}'),
        TimeLimit = '24:00:00',
    ),
)

w.prepare()
w.write_cfd_files()
w.submit()
