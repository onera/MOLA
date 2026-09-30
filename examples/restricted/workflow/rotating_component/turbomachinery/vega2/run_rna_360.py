import numpy as np
from mola.workflow.rotating_component import turbomachinery

exit() # FIXME this case is not operational yet

solver = 'elsa'
example_name = 'vega2_rna360_'+solver


# Compute time step based on rotor Blade Passage Period (BPP)
RPM = 13000.0
Omega = RPM * 2 * np.pi / 60 # in rad/s
T_rotation = 2*np.pi/Omega
BPP = T_rotation/37
Niter_BPP = 100
dt = BPP/Niter_BPP
Niter_rotation = 37*Niter_BPP

workflow = turbomachinery.Workflow( 

    Solver=solver,

    Mesh = dict(
        Source = '/stck/mola/data/restricted/mesh/vega2/vega2.cgns',  # 236k cells -> 7.3m cells after duplication over 360°
        DefaultToleranceForConnection = 1e-6,
    ),

    ApplicationContext = dict(
        ShaftRotationSpeed = RPM, 
        HubRotationIntervals = [dict(xmin=0.069)],  # The hub rotates for x > xmin, and it is fixed for x < xmin
        Rows = dict(
            stator = dict(NumberOfBlades=23, NumberOfBladesSimulated=23),
            rotor = dict(NumberOfBlades=37, NumberOfBladesSimulated=37, IsRotating=True), 
        )
    ),

    Flow = dict(
        MassFlow              = 3.06,  
        TemperatureStagnation = 400.7,
        PressureStagnation    = 8e4,
    ),

    Turbulence = dict(
        Model='SST-V2003',
    ),

    Numerics = dict(
        TimeMarching = 'Unsteady',
        NumberOfIterations = 2*Niter_rotation,
        TimeStep = dt,
    ),

    BoundaryConditions = [
        dict(Family='stator_INFLOW', Type='InflowStagnation'),
        dict(Family='rotor_OUTFLOW', Type='OutflowPressure', Pressure=2.2e4), 
        dict(Family='Rotor_stator_10_left', LinkedFamily='Rotor_stator_10_right', Type='UnsteadyRotorStatorInterface'),
    ],

    Initialization = dict(
        Method = 'interpolate',
        Source = '/stck/mola/data/restricted/mesh/vega2/init_field/init_RANS.cgns',
        # ParametrizeWithHeight = 'turbo'
    ),

    Extractions = [
        # dict(Type='3D', 
        #      Fields=['Mach', 'Pressure', 'PressureStagnation', 'Entropy'], 
        #      ExtractionPeriod=500, SavePeriod=500, ExtractAtEndOfRun=True),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', 
             Fields=['Conservatives'], IsoSurfaceValue = -0.004, 
             OtherOptions=dict(tag='InletPlane', ReferenceRow='stator')), 
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', 
             Fields=['Conservatives'], IsoSurfaceValue = 0.065, 
             OtherOptions=dict(tag='OutletPlane', ReferenceRow='stator')),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', 
             Fields=['Conservatives'], IsoSurfaceValue = 0.073, 
             OtherOptions=dict(tag='InletPlane', ReferenceRow='rotor')),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', 
             Fields=['Conservatives'], IsoSurfaceValue = 0.13, 
             OtherOptions=dict(tag='OutletPlane', ReferenceRow='rotor')),
        # dict(Type='BC', Source='BCWall'),
        dict(Type='IsoSurface', IsoSurfaceField='ChannelHeight', 
             Fields=['Conservatives'], IsoSurfaceValue=0.9),
    ],

    ConvergenceCriteria = [
        # HINT just for monitoring, without stopping simulation
        dict(ExtractionName='stator_INFLOW', Variable='avg-MassFlow', Threshold=None), 
        dict(ExtractionName='stator_INFLOW', Variable='rsd-avg-MassFlow', Threshold=None), 
        dict(ExtractionName='rotor_OUTFLOW', Variable='avg-MassFlow', Threshold=None), 
        dict(ExtractionName='rotor_OUTFLOW', Variable='rsd-avg-MassFlow', Threshold=None), 
    ],

    RunManagement=dict(
        NumberOfProcessors = 16,
        JobName=example_name,
        RunDirectory = '/tmp_user/sator/$USER/mola_examples/'+example_name,
        AER='37753003F',
        ),
    
    SolverParameters = dict(
        model = dict(
            walldistcompute='gridline'
        )
    )
    )

# workflow.prepare()
# workflow.write_cfd_files()
workflow.submit()

# ___MOLA_CI_MODE: "mono"___ # FIXME this case is not operational yet