import numpy as np
from mola.workflow.rotating_component import turbomachinery

solver = 'elsa'
example_name = 'vega2_choro_'+solver


# Compute time step based on rotor Blade Passage Period (BPP)
RPM = 13000.
Omega = RPM * 2 * np.pi / 60 # in rad/s
T_rotation = 2*np.pi/Omega
BPP = T_rotation/37
Niter_BPP = 100
dt = BPP/Niter_BPP
Niter_rotation = 37*Niter_BPP
# Develop an interface to set: 
#   NumberOfIterationsPerBladePassage = ('rotor', 37)
#   NumberOfRotations = 2
# to compute TimeStep and NumberOfIterations ?

workflow = turbomachinery.Workflow( 

    Solver=solver,

    Mesh = '/stck/mola/data/restricted/mesh/vega2/vega2.cgns',  # 236k cells

    ApplicationContext = dict(
        ShaftRotationSpeed = RPM, 
        HubRotationIntervals = [dict(xmin=0.069)],  # The hub rotates for x > xmin, and it is fixed for x < xmin
        Rows = dict(
            stator = dict(NumberOfBlades=23),
            rotor = dict(NumberOfBlades=37, IsRotating=True), 
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
        dict(Family='Rotor_stator_10_left', LinkedFamily='Rotor_stator_10_right', Type='ChorochronicInterface'),
    ],

    Initialization = dict(
        Method = 'copy',
        Source = '/stck/mola/data/restricted/mesh/vega2/init_field/init_RANS.cgns',
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
        dict(ExtractionName='stator_INFLOW', Variable='avg-MassFlow', Threshold=-1.), 
        dict(ExtractionName='stator_INFLOW', Variable='rsd-avg-MassFlow', Threshold=-1.), 
        dict(ExtractionName='rotor_OUTFLOW', Variable='avg-MassFlow', Threshold=-1.), 
        dict(ExtractionName='rotor_OUTFLOW', Variable='rsd-avg-MassFlow', Threshold=-1.), 
    ],

    RunManagement=dict(
        NumberOfProcessors = 16,
        JobName=example_name,
        RunDirectory = '/tmp_user/sator/$USER/mola_examples/'+example_name,
        AER='37753003F',
        ),
    )

# workflow.prepare()
# workflow.write_cfd_files()
workflow.submit()

# MOLA_CI_MODE: "mono"