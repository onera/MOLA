from mola.workflow.fixed import linear_cascade
import os
solver = os.getenv('MOLA_SOLVER')

w = linear_cascade.Workflow(

    Mesh = '/stck/mola/data/open/mesh/spleen/SPLEEN.cgns',

    Flow = dict(
        Mach = 0.45,
        TemperatureStagnation = 300.,
        PressureStagnation = 9500.,
    ),

    ApplicationContext = dict(
        AngleOfAttackDeg = -37.3,
    ),

    Turbulence = dict(
        Level = 0.025,
        Viscosity_EddyMolecularRatio = 100.,
        Model = 'SST-V2003',
    ),

    Numerics = dict(
        NumberOfIterations=3000,
        CFL=dict(EndIteration=300, StartValue=1., EndValue=30.),
    ),

    BoundaryConditions = [
        dict(Family='SPLEEN_INFLOW', Type='InflowStagnation'),
        dict(Family='SPLEEN_OUTFLOW', Type='OutflowPressure', Pressure=5617.),
        dict(Family='SPLEEN_BLADE', Type='WallViscous'),
        dict(Family='HUB', Type='WallInviscid'),
        dict(Family='SHROUD', Type='WallInviscid'),
    ],

    Extractions = [
        dict(Type='BC', Source='SPLEEN_BLADE', Fields=['Pressure']),
        # dict(Type='IsoSurface', Name='MidSpan', IsoSurfaceField='CoordinateY', IsoSurfaceValue=0.001, Fields=['Conservatives']), # midspan
        dict(Type='IsoSurface', Name='MidSpan', IsoSurfaceField='ChannelHeight', IsoSurfaceValue=0.501, Fields=['Conservatives']), # approximately midspan (issue for 0.5 because exactly on a mesh layer)
        # dict(Type='IsoSurface', Name='Plane01', IsoSurfaceField='CoordinateX', IsoSurfaceValue=-0.05328, Fields=['Conservatives']), 
        dict(Type='IsoSurface', Name='Plane02', IsoSurfaceField='CoordinateX', IsoSurfaceValue=-0.023807, Fields=['Conservatives'], OtherOptions=dict(tag='InletPlane', ReferenceRow='SPLEEN')),  
        dict(Type='IsoSurface', Name='Plane03', IsoSurfaceField='CoordinateX', IsoSurfaceValue= 0.0,      Fields=['Conservatives']),  
        dict(Type='IsoSurface', Name='Plane06', IsoSurfaceField='CoordinateX', IsoSurfaceValue= 0.071421, Fields=['Conservatives'], OtherOptions=dict(tag='OutletPlane', ReferenceRow='SPLEEN')), 
    ],

    RunManagement = dict(
        NumberOfProcessors = 8,
        RunDirectory = f'example_{solver}',
        Scheduler = 'local',
        TimeLimit = '3:00:00',
    )

)

w.prepare()
w.write_cfd_files()
w.submit()
