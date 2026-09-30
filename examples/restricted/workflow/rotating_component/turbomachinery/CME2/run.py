from mola.workflow.rotating_component import turbomachinery

solver = 'elsa'
example_name = 'cme2_'+solver


w = turbomachinery.Workflow( 

    Solver=solver,

    RawMeshComponents=[
        dict(
            Name='CME2',
            Source='/stck/mola/data/restricted/mesh/CME2/CME2.cgns',
            )
    ],

    ApplicationContext = dict(
        ShaftRotationSpeed = -6400.,
        HubRotationIntervals = [dict(xmax=0.0742685)],

        Rows = dict(
            Rotor = dict(IsRotating=True, NumberOfBlades=30),
            Stator = dict(NumberOfBlades=40),
        )
    ),

    Flow = dict(
        MassFlow              = 12.3,  
        TemperatureStagnation = 288.15,
        PressureStagnation    = 101325.,
    ),

    Turbulence = dict(
        # Level = 0.01,  # Tu
        # Viscosity_EddyMolecularRatio = 0.1,  # µt/µ
        Model = 'smith',  #'SST-V2003',
    ),

    Numerics = dict(
        NumberOfIterations = 10000,
        CFL = dict(EndIteration=1000, StartValue=1., EndValue=10.),
    ),

    BoundaryConditions = [
        dict(Family='Rotor_INFLOW', Type='InflowStagnation'),
        dict(Family='Stator_OUTFLOW', Type='OutflowRadialEquilibrium', ValveLaw=dict(Type='Quadratic', ValveCoefficient=0.3)),  
        dict(Type='MixingPlane', Family='Rotor_stator_10_left', LinkedFamily='Rotor_stator_10_right'),
    ],

    Initialization = dict(
        Method = 'uniform',
        ParametrizeWithHeight = 'turbo',
    ),

    # The list Extractions gathers all the extractions that will be done in co-processing.
    # Each element is a dictionary that triggers one type of extraction.
    Extractions = [
        # # dict(Type='BC', Source='BCWall*', Fields=['Pressure', 'BoundaryLayer', 'yPlus']),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=-0.01, Fields=['Conservatives'], OtherOptions=dict(tag='InletPlane', ReferenceRow='Rotor')),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=0.0675,Fields=['Conservatives'], OtherOptions=dict(tag='OutletPlane', ReferenceRow='Rotor')),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=0.076, Fields=['Conservatives'], OtherOptions=dict(tag='InletPlane', ReferenceRow='Stator')),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=0.17,  Fields=['Conservatives'], OtherOptions=dict(tag='OutletPlane', ReferenceRow='Stator')),
        dict(Type='IsoSurface', IsoSurfaceField='ChannelHeight', IsoSurfaceValue=0.1, Fields=['Conservatives']),
        dict(Type='IsoSurface', IsoSurfaceField='ChannelHeight', IsoSurfaceValue=0.5, Fields=['Conservatives']),
        dict(Type='IsoSurface', IsoSurfaceField='ChannelHeight', IsoSurfaceValue=0.9, Fields=['Conservatives']),
    ],

    ConvergenceCriteria = [
        dict(
            ExtractionName = 'Stator_OUTFLOW',
            Variable  = 'rsd-MassFlow',
            Threshold = 1e-4,
        ),
    ],

    RunManagement=dict(
        NumberOfProcessors = 96,
        JobName=example_name,
        RunDirectory = '/tmp_user/sator/$USER/mola_examples/'+example_name,
        TimeLimit = '15:00:00',
        AER='37753003F',
        ),
    )

# w.prepare()
# w.write_cfd_files()
w.submit()

# manager = turbomachinery.WorkflowManager(w, root_directory='/tmp_user/sator/tbontemp/.test_user_case/test_CME2_multi')
# manager.add_isospeed_line(throttles=[1e5, 1.1e5, 1.2e5])
# manager.prepare()
# manager.submit()

# manager = turbomachinery.WorkflowManager('workflow_manager.cgns')
# perfo = manager.gather_performance('Rotor')
# manager.plot_isospeed_lines(perfo)

# MOLA_CI_MODE: "mono"