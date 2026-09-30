from mola.workflow.rotating_component import turbomachinery
import numpy as np

solver = 'sonics'
example_name = 'srv2_'+solver

# Define the section where the hub rotates
def hub_rotation_function(CoordinateX, CoordinateY, CoordinateZ):
    CoordinateR = (CoordinateY**2 + CoordinateZ**2)**0.5
    xmin = -0.056
    rmax = 0.1137
    omega = np.zeros(CoordinateX.shape, dtype=float)
    omega[(xmin<=CoordinateX) & (CoordinateR<=rmax)] = 40000.0/30*np.pi # ShaftRotationSpeed in rad/s
    return np.asfortranarray(omega).ravel(order='K')

w = turbomachinery.Workflow( 

    Solver=solver,
    
    RawMeshComponents=[
        dict(
            Name='SRV2',
            Source = '/stck/mola/data/open/mesh/SRV2/mesh_autogrid/SRV2.cgns',  # 2.1M points
            Unit='mm',
            ) 
    ],

    ApplicationContext = dict(
        ShaftRotationSpeed = 40000.0, 

        # if enabled, user-specified function; else apply uniform rot speed
        # HubRotationIntervals = hub_rotation_function, 
        
        Rows = dict(
            Impeller = dict(
                IsRotating = True,
                NumberOfBlades = 13,
                FlowAngleAtRootDeg = -35.0,
                FlowAngleAtTipDeg = -35.0,
            )
        )
    ),

    Flow = dict(
        MassFlow              = 2.0,  # for the 360 degrees section, even it is simulated entirely
        TemperatureStagnation = 288.15,
        PressureStagnation    = 101325.0,
    ),

    Turbulence = dict(
        Level = 0.01,
        Viscosity_EddyMolecularRatio = 0.1,
        Model = 'smith',
    ),

    Numerics = dict(
        NumberOfIterations = 10000,
        CFL = dict(EndIteration=300, StartValue=1., EndValue=30.)
    ),

    BoundaryConditions = [
        dict(Family='Impeller_INFLOW', Type='InflowStagnation'),
        dict(Family='Impeller_OUTFLOW', Type='OutflowPressure', Pressure=3e5),
    ],

    Initialization = dict(
        Method = 'copy',
        Source = '/stck/mola/data/open/mesh/SRV2/init_field/init.cgns',
        ParametrizeWithHeight = 'turbo',
    ),

    Extractions = [
        dict(Type='IsoSurface', IsoSurfaceField='ChannelHeight', IsoSurfaceValue=0.5, Fields=['Conservatives']),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=-0.02, Fields=['Conservatives'], OtherOptions=dict(tag='InletPlane', ReferenceRow='Impeller')),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateR', IsoSurfaceValue=0.2, Fields=['Conservatives'], OtherOptions=dict(tag='OutletPlane', ReferenceRow='Impeller')),
    ],

    ConvergenceCriteria = [
        dict(
            ExtractionName = 'Impeller_INFLOW',
            Variable  = 'rsd-MassFlow',
            Threshold = 1e-5,
        ),
    ],

    RunManagement=dict(
        NumberOfProcessors = 48,
        JobName=example_name,
        RunDirectory = '/tmp_user/sator/$USER/mola_examples/'+example_name,
        TimeLimit = '15:00:00',
        AER='37753003F',
        ),

    )

# w.prepare()
# w.write_cfd_files()
w.submit()

# MOLA_CI_MODE: "mono"