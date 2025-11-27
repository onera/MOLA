from mola.workflow.rotating_component import turbomachinery

w = turbomachinery.Workflow( 
    
    Mesh = dict(
        Source = '/stck/mola/data/open/mesh/rotor37/rotor37.cgns',
        Unit = 'cm',
    ),

    ApplicationContext = dict(
        ShaftRotationSpeed = -1800., 
        ShaftRotationSpeedUnit = 'rad/s',

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
        Model = 'smith',
    ),

    Numerics = dict(
        NumberOfIterations = 10000,
        CFL = dict(EndIteration=300, StartValue=1., EndValue=30.)
    ),

    BoundaryConditions = [
        dict(Family='R37_INFLOW', Type='InflowStagnation'),
        dict(Family='R37_OUTFLOW', Type='OutflowRadialEquilibrium', 
             ValveLaw=dict(Type='Quadratic', ValveCoefficient=0.1)
            # MassFlow=20.,
             )
    ],

    Initialization = dict(
        ParametrizeWithHeight = 'turbo'
    ),

    Extractions = [
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=-0.03, Fields=['Conservatives'], OtherOptions=dict(tag='InletPlane', ReferenceRow='R37')),
        dict(Type='IsoSurface', IsoSurfaceField='CoordinateX', IsoSurfaceValue=0.07, Fields=['Conservatives'], OtherOptions=dict(tag='OutletPlane', ReferenceRow='R37')),
        dict(Type='IsoSurface', IsoSurfaceField='ChannelHeight', IsoSurfaceValue=0.9, Fields=['Conservatives']),
    ],    

    ConvergenceCriteria = [
        dict(
            ExtractionName = 'R37_INFLOW',
            Variable  = 'rsd-MassFlow',
            Threshold = 1e-5,
        ),
    ],

    RunManagement=dict(
        JobName='rotor37',
        NumberOfProcessors=24,
        RunDirectory='/tmp_user/sator/$USER/.test_user_case/rotor37',
        AER='34790003F', # PDEV MOLA 2025
        ),

    )


import numpy as np
manager = turbomachinery.WorkflowManager(w, root_directory='/tmp_user/sator/$USER/.test_user_case/rotor37_multi')
manager.add_isospeed_line(np.arange(0.2, 0.61, 0.05))
manager.prepare()
manager.submit()

## To plot the result, use the following lines in the same directory where you executed lines above
# manager = turbomachinery.WorkflowManager('workflow_manager.cgns')
# perfo = manager.gather_performance('R37')
# manager.plot_isospeed_lines(perfo)
