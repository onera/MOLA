import os
from mola.workflow.rotating_component.propeller.workflow import WorkflowPropeller, solver


w = WorkflowPropeller(

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
        ShaftRotationSpeed = 2030.0,
        NumberOfBlades = 3,
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
        NumberOfIterations=3000,
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
            Variable = "ForceX",
            Threshold = -1e9, # HINT just for showing Variable progress in coprocess.log
        )
    ],

    RunManagement = dict(
        NumberOfProcessors = 17,
        Machine='juno',
        RunDirectory = os.path.join(os.getcwd(),f'example_{solver}'),
        TimeLimit = '5:00:00',
    ),
)

w.prepare()
w.write_cfd_files()
w.submit()
