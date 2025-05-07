from mola.workflow.rotating_component import turbomachinery

w = turbomachinery.Workflow( 
    RawMeshComponents=[
        dict(
            Name='rotor37',
            Source = '/stck/mola/data/open/mesh/rotor37/rotor37.cgns',
            Unit = 'cm',
            )
    ],

    ApplicationContext = dict(
        ShaftRotationSpeed = -1800., 

        Rows = dict(
            R37 = dict(
                IsRotating = True,
                NumberOfBlades = 36,
            )
        )
    ),

    Flow = dict(
        MassFlow              = 20.5114,  # for the 360 degrees section, even it is simulated entirely
        TemperatureStagnation = 288.15,
        PressureStagnation    = 101330.,
    ),

    Turbulence = dict(
        Level = 0.03,
        Viscosity_EddyMolecularRatio = 0.1,
        Model = 'SA',
    ),

    Numerics = dict(
        NumberOfIterations = 10000,
        CFL = dict(EndIteration=300, StartValue=1., EndValue=30.)
    ),

    BoundaryConditions = [
        dict(Family='R37_INFLOW', Type='InflowStagnation'),
        dict(Family='R37_OUTFLOW', Type='OutflowRadialEquilibrium', 
             valve_type=4, #'BCValveLawQHyperbolic',
             valve_ref_pres=0.75*101325., 
             Pressure=0.75*101325., 
             valve_relax=0.1#*101325.
             )
    ],

    Extractions = [
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=-0.03, Fields=['Conservatives'], OtherOptions=dict(tag='InletPlane', ReferenceRow='R37')),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=0.07, Fields=['Conservatives'], OtherOptions=dict(tag='OutletPlane', ReferenceRow='R37')),
        dict(Type='IsoSurface', IsoSurfaceField='ChannelHeight', IsoSurfaceValue=0.9, Fields=['Conservatives']),
    ],    

    ConvergenceCriteria = [
        dict(
            ExtractionName = 'R37_INFLOW',
            Variable  = 'rsd-MassFlow',
            Threshold = 1e-4,
        ),
    ],

    # Initialization = dict(
    #     ParametrizeWithHeight = 'turbo'
    # ),

    RunManagement=dict(
        JobName='rotor37',
        NumberOfProcessors=24,
        RunDirectory='/tmp_user/sator/tbontemp/.test_user_case/rotor37',
        #RemovePreviousRunDirectory = True,
        # AER='34790003F', # PDEV MOLA 2025
        #TimeLimit = '00:30:00',
        ),

    )

w.prepare()
w.write_cfd_files()
w.submit()
w.assert_completed_without_errors()

# import numpy as np
# manager = turbomachinery.WorkflowManager(w, root_directory='/tmp_user/sator/tbontemp/rotor37_multi_elsa_none_cf_v5.3.03')
# manager.add_isospeed_line(throttles=101325. * np.arange(0.2, 0.61, 0.05))
# manager.prepare()
# manager.submit()

# manager = turbomachinery.WorkflowManager('workflow_manager.cgns')
# perfo = manager.gather_performance('R37')
# manager.plot_isospeed_lines(perfo)
