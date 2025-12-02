import numpy as np
from mola.workflow.rotating_component import turbomachinery

Fields = ['Conservatives', 'PressureStagnation', 'Pressure', 'TemperatureStagnation', 'Entropy', 'Mach', 'VelocityX', 'VelocityY', 'VelocityZ']

def hub_rotation_function(CoordinateX):  #, CoordinateY, CoordinateZ):
    # CoordinateR = (CoordinateY**2 + CoordinateZ**2)**0.5
    xmin = ...
    xmax = ...
    omega = np.zeros(CoordinateX.shape, dtype=float)
    omega[(xmin<=CoordinateX) & (CoordinateX<=xmax)] = ... # ShaftRotationSpeed in rad/s
    return np.asfortranarray(omega).ravel(order='K')



w = turbomachinery.Workflow( 
    RawMeshComponents=[
        dict(
            Name='SRV2',
            Source = '/stck/mola/data/open/mesh/SRV2/mesh_autogrid/SRV2.cgns',  # 2.1M points
            Unit='mm',
            ) 
    ],

    ApplicationContext = dict(
        ShaftRotationSpeed = 40000., 
        # HubRotationIntervals = [dict(xmin=..., xmax=...)],
        # HubRotationIntervals = [(..., ...)],
        # HubRotationIntervals = hub_rotation_function,
        Rows = dict(
            Impeller = dict(
                IsRotating = True,
                NumberOfBlades = 13,
                FlowAngleAtRootDeg = -35.,
                FlowAngleAtTipDeg = -35.,
            )
        )
    ),

    Flow = dict(
        MassFlow              = 2.,  # for the 360 degrees section, even it is simulated entirely
        TemperatureStagnation = 288.15,
        PressureStagnation    = 101325.,
    ),

    Turbulence = dict(
        Level = 0.01,
        Viscosity_EddyMolecularRatio = 0.1,
        Model = 'smith',
    ),

    Numerics = dict(
        NumberOfIterations = 10000,
        CFL = dict(EndIteration=1000, StartValue=1., EndValue=5.)
    ),

    BoundaryConditions = [
        dict(Family='Impeller_INFLOW', Type='InflowStagnation'),
        # dict(Family='Impeller_OUTFLOW', Type='OutflowPressure', Pressure=3e5),
        dict(Family='Impeller_OUTFLOW', Type='OutflowMassFlow'),
    ],

    Initialization = dict(
        ParametrizeWithHeight = 'turbo',
    ),

    Extractions = [
        dict(Type='IsoSurface', IsoSurfaceField='ChannelHeight', IsoSurfaceValue=0.9, Fields=Fields),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=-0.02, Fields=Fields, OtherOptions=dict(tag='InletPlane', ReferenceRow='Impeller')),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateR', IsoSurfaceValue=0.2, Fields=Fields, OtherOptions=dict(tag='OutletPlane', ReferenceRow='Impeller')),
    ],

    ConvergenceCriteria = [
        dict(
            ExtractionName = 'Impeller_INFLOW',
            Variable  = 'rsd-MassFlow',
            Threshold = 1e-4,
        ),
    ],

    RunManagement=dict(
        JobName='SRV2',
        RunDirectory=f'/tmp_user/sator/$USER/.test_user_case/SRV2_init_antoine/',
        NumberOfProcessors=48,
        RemovePreviousRunDirectory = True,
        AER = '34790003F', # PDEV MOLA 2025
        ),

    )

w.prepare()
w.write_cfd_files()
w.submit()

# w.prepare_and_submit_remotely()


