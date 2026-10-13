#imports@start
from mola.workflow import Workflow
#imports@end

from mola.workflow.test.test_workflow import get_workflow_sphere_struct
w = get_workflow_sphere_struct('.')
w.print_interface()  # to redirect in file print_interface_output.txt
exit()


w = Workflow(
    RawMeshComponents=[
        dict(
            Name='sphere',
            Source='/stck/mola/data/mesh/sphere/sphere_struct.cgns',
            Families=[
                dict(Name='Wall', Location='kmin'),
                dict(Name='Farfield', Location='remaining'),
            ],
            )
    ],

    # SplittingAndDistribution=dict(
    #     Strategy='AtComputation', # "AtPreprocess" or "AtComputation"
    #     Splitter='PyPart', # or 'maia', 'PyPart' etc..
    #     ),

    Flow=dict(
        Density = 0.2,
        Temperature = 100.,
        Velocity = 50.,
    ),

    Turbulence = dict(
        Model = 'SA',
    ),

    Numerics = dict(
        NumberOfIterations=10,
        CFL=1.,
        Scheme = 'Roe',
    ),

    BoundaryConditions=[
        dict(Family='Wall', Type='Wall'),
        dict(Family='Farfield', Type='Farfield'),
    ],

    Extractions=[
        # dict(Type='BC', Source='*', Name='ByFamily', Fields=['Pressure'], ExtractAtEndOfRun=True),
        # dict(Type='BC', Source='BCWall*', Name='ByFamily', Fields=['NormalVector', 'SkinFriction', 'BoundaryLayer'], ExtractAtEndOfRun=True),
        # dict(Type='IsoSurface', IsoSurfaceField='CoordinateZ', IsoSurfaceValue=1e-6, ExtractAtEndOfRun=True),
        dict(Type='3D', Fields=['PressureStagnation', 'Pressure', 'Mach', 'Entropy'], ExtractAtEndOfRun=True),
        ],

    RunManagement=dict(
        NumberOfProcessors=4,
        # RunDirectory = '/tmp_user/juno/tbontemp/TEST/sphere_elsa',
        ),
    )

# w.set_workflow_parameters_in_tree()
# w.write_tree('workflow.cgns')
# w = Workflow(tree='workflow.cgns')
# w.prepare()  

# w.set_workflow_parameters_in_tree()
# w.write_tree_remote()

#print_interface@start
w.print_interface()
#print_interface@end

# w.prepare()
# w.write_cfd_files()
# w.submit()

