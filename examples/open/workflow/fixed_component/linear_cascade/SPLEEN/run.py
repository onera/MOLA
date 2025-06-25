from mola.workflow.fixed import linear_cascade
import os
solver = os.getenv('MOLA_SOLVER')

w = linear_cascade.Workflow(

    RawMeshComponents=[
        dict(
            Name='SPLEEN_Base',
            Source='/stck/mola/data/open/mesh/spleen/SPLEEN.cgns',
            )
    ],

    Flow = dict(
        Mach = 0.45,
        TemperatureStagnation = 285.,
        PressureStagnation = 8883.,
    ),

    ApplicationContext = dict(
        AngleOfAttackDeg = -37.3,
    ),

    Turbulence = dict(
        Level = 0.025,
        Viscosity_EddyMolecularRatio = 0.1,
        Model = 'SST-V2003',
    ),

    Numerics = dict(
        NumberOfIterations=2000,
        CFL=dict(EndIteration=300, StartValue=1., EndValue=30.),
    ),

    BoundaryConditions = [
        dict(Family='SPLEEN_INFLOW', Type='InflowStagnation'),
        dict(Family='SPLEEN_OUTFLOW', Type='OutflowPressure', Pressure=8883./1.6913),
        dict(Family='SPLEEN_BLADE', Type='WallViscous'),
        dict(Family='HUB', Type='WallInviscid'),
        dict(Family='SHROUD', Type='WallInviscid'),
    ],

    Extractions = [
        dict(Type='BC', Source='SPLEEN_BLADE', Fields=['Pressure']),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateY', IsoSurfaceValue=0.001, 
             Fields=['Conservatives', 'Entropy', 'PressureStagnation', 'Pressure', 'Mach']), # midspan
        # dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=-0.05328, Fields=['Conservatives'], OtherOptions=dict(tag='Plan01')),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=-0.023807, Fields=['Conservatives'], OtherOptions=dict(tag='Plan02')),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue= 0.0,      Fields=['Conservatives'], OtherOptions=dict(tag='Plan03')),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue= 0.071421, Fields=['Conservatives'], OtherOptions=dict(tag='Plan06')),
    ],

    RunManagement = dict(
        NumberOfProcessors = 4,
        RunDirectory = f'/tmp/mola_test_cases/SPLEEN/example_{solver}',
        Scheduler = 'local',
        TimeLimit = '3:00:00',
    )

)

w.prepare()
w.write_cfd_files()
w.submit()
