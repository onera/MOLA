from mola.workflow.rotating_component import turbomachinery

solver = 'sonics'
example_name = 'ecl5'

w = turbomachinery.Workflow(

    Solver=solver,

    RawMeshComponents=[
        dict(
            Name='ECL5',
            Source='/stck/mola/data/restricted/mesh/ECL5/ECL5_N100_with_ChannelHeight.cgns',
            Unit='mm',
            )
    ],

    ApplicationContext = dict(
        ShaftRotationSpeed = -11000., 
        HubRotationIntervals = [dict(xmax=0.0742685)],
        Rows = dict(
            FAN = dict(IsRotating=True, NumberOfBlades=16, NumberOfBladesSimulated=1), 
            OGV = dict(NumberOfBlades=31, NumberOfBladesSimulated=1),
        )
    ),

    Flow = dict(
        MassFlow              = 36.0,
        TemperatureStagnation = 288.15,
        PressureStagnation    = 101325.,
    ),

    Turbulence = dict(
        Model='smith', #'SST-V2003',
        Level = 0.005,
        Viscosity_EddyMolecularRatio = 0.1,
    ),

    Numerics = dict(
        NumberOfIterations = 10000,
        CFL = dict(EndIteration=1000, StartValue=1., EndValue=30.),
    ),

    BoundaryConditions = [
        dict(Family='FAN_INFLOW', Type='InflowStagnation'),
        # dict(Family='OGV_OUTFLOW', Type='OutflowRadialEquilibrium', MassFlow=36),
        dict(Family='OGV_OUTFLOW', Type='OutflowRadialEquilibrium', ValveLaw = dict(Type='Quadratic', ValveCoefficient=0.3)), 
        dict(Family='Rotor_stator_10_left', LinkedFamily='Rotor_stator_10_right', Type='MixingPlane'),
    ],

    Initialization = dict(
        Method = 'copy',
        Source = '/stck/mola/data/restricted/mesh/ECL5/init_fields/init.cgns',
        # Method = 'uniform',
        ParametrizeWithHeight = 'maia',
    ),

    Extractions = [
        # Plane 110, Pt and Tt rakes, Ps annular rake
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=-0.2432, Name='Plane110', OtherOptions=dict(tag='InletPlane', ReferenceRow='FAN'),
             Fields=['Conservatives']),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=0.0398, Name='DownstreamFAN', OtherOptions=dict(tag='OutletPlane', ReferenceRow='FAN'),
             Fields=['Conservatives']),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=0.1, Name='UpstreamOGV', OtherOptions=dict(tag='InletPlane', ReferenceRow='OGV'),
             Fields=['Conservatives']),
        # Plane 150, Pt and Tt rakes, Ps annular rake
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=0.3418, Name='Plane150', OtherOptions=dict(tag='OutletPlane', ReferenceRow='OGV'),
             Fields=['Conservatives']),
        dict(
            Type='IsoSurface', IsoSurfaceField='ChannelHeight', IsoSurfaceValue=0.9, IsoSurfaceContainer='FlowSolution#Height',
            Fields=['Conservatives']),
    ],

    ConvergenceCriteria = [
        dict(
            ExtractionName = 'FAN_INFLOW',
            Variable  = 'rsd-MassFlow',
            Threshold = 1e-5,
        ),
    ],

    RunManagement=dict(
        NumberOfProcessors = 48,
        JobName=example_name,
        RunDirectory = '/tmp_user/sator/$USER/mola_examples/'+example_name,
        AER='37752011F',  # RGC ECL5 2026
        ),

    )

# w.prepare()
# w.write_cfd_files()
# w.submit()

manager = turbomachinery.WorkflowManager(w, root_directory='/tmp_user/sator/$USER/mola_examples/'+example_name)
valve_coeffs = [0.1,0.3,0.5]

for solver in ['elsa',
               'sonics']:
    for model in ['SST-V2003',
                  'smith']:
        manager.new_job(f"{solver}_{model}")
        for throttle in valve_coeffs:
            manager.add_variations(
                [
                    ('RunManagement|JobName', f'{solver}_{model}_{throttle:.2f}'),
                    ('RunManagement|RunDirectory', f'ValveCoefficient_{throttle:.2f}'),
                    ('Solver', solver),
                    ('Turbulence|Model', model),
                    (f'BoundaryConditions|Family=OGV_OUTFLOW|ValveLaw|ValveCoefficient', throttle),
                ], 
                initialize_from_previous = True
                )

manager.prepare()
manager.submit()

# MOLA_CI_MODE: "multi"
