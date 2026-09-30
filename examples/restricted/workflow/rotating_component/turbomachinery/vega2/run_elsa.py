from mola.workflow.rotating_component import turbomachinery

solver = 'elsa'
example_name = 'vega2_'+solver


workflow = turbomachinery.Workflow( 

    Solver=solver,

    Mesh = '/stck/mola/data/restricted/mesh/vega2/vega2.cgns',  # 236k cells

    ApplicationContext = dict(
        ShaftRotationSpeed = 13000., 
        HubRotationIntervals = [dict(xmin=0.069)],  # The hub rotates for x > xmin, and it is fixed for x < xmin
        Rows = dict(
            stator = dict(NumberOfBlades=23),
            rotor = dict(NumberOfBlades=37, IsRotating=True), 
        ), 
        RowType = 'turbine',
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
        NumberOfIterations = 5000,
        CFL = dict(EndIteration=300, StartValue=1., EndValue=30.),
    ),

    BoundaryConditions = [
        dict(Family='stator_INFLOW', Type='InflowStagnation'),
        dict(Family='rotor_OUTFLOW', Type='OutflowPressure', Pressure=2.2e4), 
        dict(Family='Rotor_stator_10_left', LinkedFamily='Rotor_stator_10_right', Type='MixingPlane'),
    ],

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
        dict(
            ExtractionName = 'rotor_OUTFLOW',
            Variable  = 'rsd-MassFlow',
            Threshold = 1e-4,
        ),
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