#    Copyright 2023 ONERA - contact luis.bernardos@onera.fr
#
#    This file is part of MOLA.
#
#    MOLA is free software: you can redistribute it and/or modify
#    it under the terms of the GNU Lesser General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    MOLA is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU Lesser General Public License for more details.
#
#    You should have received a copy of the GNU Lesser General Public License
#    along with MOLA.  If not, see <http://www.gnu.org/licenses/>.

import pytest

import os
import shutil
import numpy as np

import treelab.cgns as cgns

import mola.naming_conventions as names
from mola.logging import mola_logger, MolaException, MolaUserError, mute_stdout
from mola import server as SV
from mola.cfd.preprocess.run_manager import run_manager
from mola.workflow import Workflow


def get_workflow_dist():
    from mpi4py import MPI
    import maia
    

    MPI.COMM_WORLD.barrier()
    mesh = None
    if MPI.COMM_WORLD.Get_rank() == 0:
        x, y, z = np.meshgrid( np.linspace(0,1,21),
                            np.linspace(0,1,21),
                            np.linspace(0,1,21), indexing='ij')
        mesh = cgns.newZoneFromArrays( 'block', ['x','y','z'],
                                                [ x,  y,  z ])
        mesh = cgns.add(mesh) # mesh must be CGNSTree_t for full_to_dist_tree


    MPI.COMM_WORLD.barrier()
    mesh = maia.factory.full_to_dist_tree(mesh, MPI.COMM_WORLD, owner=0)
    mesh = cgns.castNode(mesh) 
    MPI.COMM_WORLD.barrier()

    w = Workflow(
        RawMeshComponents=[
            dict(
                Name='cartesian',
                Source=mesh,
                Families=[
                    dict(Name='Ground',
                         Location='kmin'),
                    dict(Name='Farfield',
                         Location='remaining'),
                ],
                Positioning=[
                    dict(
                        Type='TranslationAndRotation',
                        InitialFrame=dict(
                            Point=[0,0,0],
                            Axis1=[1,0,0],
                            Axis2=[0,1,0],
                            Axis3=[0,0,1]),
                        RequestedFrame=dict(
                            Point=[0,0,0],
                            Axis1=[1,0,0],
                            Axis2=[0,1,0],
                            Axis3=[0,0,1]),
                        ),
                ],
                Connection = [
                    # dict(Type='Match', Tolerance=1e-8),
                ],
                )
        ],

        SplittingAndDistribution=dict(
            Strategy='AtPreprocess', # "AtPreprocess" or "AtComputation"
            Splitter='maia', # or 'maia', 'PyPart' etc..
            Distributor='maia', 
            ComponentsToSplit='all', # 'all', or None or ['first', 'second'...]
            ),

        Flow=dict(
            Velocity = 100.,
        ),

        Turbulence = dict(
            Model = 'SA',
        ),

        Solver=os.environ.get('MOLA_SOLVER'),

        Numerics = dict(
            CFL=1.,
        ),

        BoundaryConditions=[
            dict(Family='Ground', Type='Wall'),
            dict(Family='Farfield', Type='Farfield'),
        ],

        ExtractionsDefaults=[dict(ReferenceParameter='File',File='signals.cgns',SavePeriod=69)],

        Extractions=[
            dict(Type='Integral', Name='Loads', Fields=['Force', 'Torque'], Source="BCWall*"),
            # dict(Type='Probe', Name='probe1', Fields=['std-Pressure'], SavePeriod=5),
            # dict(Type='Probe', Name='probe2', Fields=['std-Density'], SavePeriod=5),
            dict(Type='3D', Fields=['Mach', 'q_criterion']),
            dict(Type='BC', Source='BCWall*', Name='ByFamily', Fields=['normalvector', 'frictionvector']),
            dict(Type='BC', Source='*', Name='ByFamily', Fields=['Pressure']),
            dict(Type='IsoSurface', Name='MySlice', IsoSurfaceField='CoordinateY', IsoSurfaceValue=1.e-6, Fields=['Mach','cellN']),
            ],

        RunManagement=dict(Scheduler='local'),

        )
    return w

def get_workflow2_parameters():

    x, y, z = np.meshgrid( np.linspace(0,1,21),
                           np.linspace(0,1,21),
                           np.linspace(0,1,21), indexing='ij')
    mesh = cgns.newZoneFromArrays( 'block', ['x','y','z'], [ x,  y,  z ])

    params = dict(

        RawMeshComponents=[
            dict(
                Name='cartesian',
                Source=mesh,
                Families=[
                    dict(Name='Ground',
                         Location='kmin'),
                    dict(Name='Farfield',
                         Location='remaining'),
                ],
                Positioning=[
                    dict(
                        Type='TranslationAndRotation',
                        InitialFrame=dict(
                            Point=[0,0,0],
                            Axis1=[1,0,0],
                            Axis2=[0,1,0],
                            Axis3=[0,0,1]),
                        RequestedFrame=dict(
                            Point=[0,0,0],
                            Axis1=[1,0,0],
                            Axis2=[0,1,0],
                            Axis3=[0,0,1]),
                        ),
                ],
                Connection = [
                    dict(Type='Match', Tolerance=1e-8),
                ],
                )
        ],

        SplittingAndDistribution=dict(
            Strategy='AtPreprocess', # "AtPreprocess" or "AtComputation"
            Splitter='Cassiopee', # or 'maia', 'PyPart' etc..
            Distributor='Cassiopee', 
            ComponentsToSplit='all', # 'all', or None or ['first', 'second'...]
            NumberOfParts=4,
            ),

        Flow=dict(
            Velocity = 100.,
        ),

        Turbulence = dict(
            Model = 'SA',
        ),

        Solver=os.environ.get('MOLA_SOLVER'),

        Numerics = dict(
            NumberOfIterations = 2,
            CFL=1.0,
        ),

        BoundaryConditions=[
            dict(Family='Ground', Type='Wall'),
            dict(Family='Farfield', Type='Farfield'),
        ],

        ExtractionsDefaults=[dict(ReferenceParameter='File',File='signals.cgns',SavePeriod=69)],

        Extractions=[
            dict(Type='Integral', Name='Loads', Fields=['Force'], Source="WallViscous"),
            dict(Type='BC', Fields=['Pressure'], Source="WallViscous",
                 File='separated.cgns' # FIXME BUG, requires separated file, otherwise it does not write extract.cgns 
                 ),
            ],

        RunManagement=dict(Scheduler='local'),

        )
    return params

def get_workflow2():
    params = get_workflow2_parameters()
    w = Workflow(**params)
    return w

def get_workflow_sphere_struct(RunDirectory):
    w = Workflow(
        RawMeshComponents=[
            dict(
                Name='sphere',
                Source='/stck/mola/data/open/mesh/sphere/sphere_struct_not_connected.cgns',
                Families=[
                    dict(Name='Wall', Location='kmin'),
                    dict(Name='Farfield', Location='kmax'),
                ],
                Connection = [ dict(Type='Match', Tolerance=1e-8),],
                )
        ],

        SplittingAndDistribution=dict(
            Strategy='AtPreprocess', # "AtPreprocess" or "AtComputation"
            Splitter='Cassiopee', # or 'maia', 'PyPart' etc..
            Distributor='Cassiopee', 
            ComponentsToSplit=None, # 'all', or None or ['first', 'second'...]
            ),

        Flow=dict(
            Density = 0.2,
            Temperature = 100.,
            Velocity = 50.,
                 ),

        Turbulence = dict(
            Model = 'SA',
        ),

        Solver=os.environ.get('MOLA_SOLVER'),

        Numerics = dict(
            NumberOfIterations=2,
            CFL=1.,
        ),

        BoundaryConditions=[
            dict(Family='Wall', Type='Wall'),
            dict(Family='Farfield', Type='Farfield'),
        ],

        Extractions=[
            # dict(Type='BC', Source='*', Name='ByFamily', Fields=['Pressure']),
            dict(Type='BC', Source='BCWall*', Name='ByFamily', Fields=['NormalVector', 'SkinFriction', 'BoundaryLayer']),
            dict(Type='IsoSurface', Name='MySurface', IsoSurfaceField='CoordinateZ',
                 IsoSurfaceValue=1.e-6),
            dict(Type='3D', Fields=['Density','MomentumX','MomentumY','MomentumZ'],
                 GridLocation='Vertex', GhostCells = False),
            ],

        RunManagement=dict(
            NumberOfProcessors=1,
            RunDirectory=RunDirectory,
            ),
        )
    
    return w

def get_workflow_sphere_struct_cassiopee_mpi_to_connect(RunDirectory):
    from mpi4py import MPI
    w = Workflow(
        RawMeshComponents=[
            dict(
                Name='sphere',
                Source='/stck/mola/data/open/mesh/sphere/sphere_struct_not_connected.cgns',
                Families=[
                    dict(Name='Wall', Location='kmin'),
                    dict(Name='Farfield', Location='kmax'),
                ],
                Connection = [ dict(Type='Match', Tolerance=1e-8),],
                )
        ],

        SplittingAndDistribution=dict(
            Strategy='AtPreprocess', # "AtPreprocess" or "AtComputation"
            Splitter='Cassiopee', 
            Distributor='Cassiopee', 
            ComponentsToSplit='all', # 'all', or None or ['first', 'second'...]
            ),

        Flow=dict(
            Density = 0.2,
            Temperature = 100.,
            Velocity = 50.,
                 ),

        Turbulence = dict(
            Model = 'SA',
        ),

        Solver=os.environ.get('MOLA_SOLVER'),

        Numerics = dict(
            NumberOfIterations=2,
            CFL=1.,
        ),

        BoundaryConditions=[
            dict(Family='Wall', Type='Wall'),
            dict(Family='Farfield', Type='Farfield'),
        ],

        Extractions=[
            # dict(Type='BC', Source='*', Name='ByFamily', Fields=['Pressure']),
            dict(Type='BC', Source='BCWall*', Name='ByFamily', Fields=['NormalVector', 'SkinFriction', 'BoundaryLayer']),
            dict(Type='IsoSurface', Name='MySurface', IsoSurfaceField='CoordinateZ',
                 IsoSurfaceValue=1.e-6),
            dict(Type='3D', Fields=['Density','MomentumX','MomentumY','MomentumZ'],
                 GridLocation='Vertex', GhostCells = False),
            ],

        RunManagement=dict(
            NumberOfProcessors=MPI.COMM_WORLD.Get_size(),
            RunDirectory=RunDirectory,
            ),
        )
    
    return w

def get_workflow_sphere_struct_dist(RunDirectory):
    from mpi4py import MPI

    w = Workflow(
        RawMeshComponents=[
            dict(
                Name='sphere',
                Source='/stck/mola/data/open/mesh/sphere/sphere_struct.cgns',
                Families=[
                    dict(Name='Wall', Location='kmin'),
                    dict(Name='Farfield', Location='kmax'),
                ],
                )
        ],

        SplittingAndDistribution=dict(
            Strategy='AtPreprocess', # "AtPreprocess" or "AtComputation"
            Splitter='maia', # or 'maia', 'PyPart' etc..
            Distributor='maia', 
            ComponentsToSplit='all', # 'all', or None or ['first', 'second'...]
            ),

        Flow=dict(
            Density = 0.2,
            Temperature = 100.,
            Velocity = 50.,
                 ),

        Turbulence = dict(
            Model = 'SA',
        ),

        Solver=os.environ.get('MOLA_SOLVER'),

        Numerics = dict(
            NumberOfIterations=10,
            CFL=1.,
        ),

        BoundaryConditions=[
            dict(Family='Wall', Type='Wall'),
            dict(Family='Farfield', Type='Farfield'),
        ],

        Extractions=[
            dict(Type='BC', Source='*', Name='ByFamily', Fields=['Pressure']),
            dict(Type='BC', Source='BCWall*', Name='ByFamily', Fields=['NormalVector', 'SkinFriction', 'BoundaryLayer']),
            dict(Type='IsoSurface', Name='MySurface', IsoSurfaceField='CoordinateZ', IsoSurfaceValue=1.e-6),
            dict(Type='3D', Fields=['Density','MomentumX','MomentumY','MomentumZ'],
                 GridLocation='CellCenter', GhostCells = False),
            ],

        RunManagement=dict(
            NumberOfProcessors=MPI.COMM_WORLD.Get_size(), 
            RunDirectory=RunDirectory,
            ),
        )
    
    return w

def get_workflow_sphere_hybrid(RunDirectory):
    w = Workflow(
        RawMeshComponents=[
            dict(
                Name='sphere',
                Source='/stck/mola/data/open/mesh/sphere/sphere_hybrid.cgns',
                Positioning=[dict(Type='Scale', Scale=1e-3)], # since Pointwise mesh is in mm
                )
        ],

        SplittingAndDistribution=dict(
            Strategy='AtComputation', # "AtPreprocess" or "AtComputation"
            Splitter='PyPart', # or 'maia', 'PyPart' etc..
            ),

        Flow=dict(
            Density = 0.2,
            Temperature = 100.,
            Velocity = 50.,
        ),

        Turbulence = dict(
            Model = 'SA',
        ),

        Solver=os.environ.get('MOLA_SOLVER'),

        Numerics = dict(
            NumberOfIterations=2,
            CFL=1.,
        ),

        BoundaryConditions=[
            dict(Family='Wall', Type='Wall'),
            dict(Family='Farfield', Type='Farfield'),
        ],

        Extractions=[
            # dict(Type='BC', Source='*', Name='ByFamily', Fields=['Pressure'], ExtractAtEndOfRun=True),
            # dict(Type='BC', Source='BCWall*', Name='ByFamily', Fields=['NormalVector', 'SkinFriction', 'BoundaryLayer'], ExtractAtEndOfRun=True),
            # dict(Type='IsoSurface', IsoSurfaceField='CoordinateZ', IsoSurfaceValue=1e-6, ExtractAtEndOfRun=True),
            # dict(Type='3D', Fields=['PressureStagnation', 'Pressure', 'Mach', 'Entropy'], ExtractAtEndOfRun=True),
            ],

        RunManagement=dict(
            NumberOfProcessors=1,
            RunDirectory=RunDirectory,
            ),
        )

    return w

def get_workflow_sphere_unstruct(RunDirectory):
    w = Workflow(
        RawMeshComponents=[
            dict(
                Name='sphere',
                Source='/stck/mola/data/open/mesh/sphere/sphere_unstructured.cgns',
                Positioning=[dict(Type='Scale', Scale=1e-3)], # since Pointwise mesh is in mm
                )
        ],

        SplittingAndDistribution=dict(
            Strategy='AtComputation', # "AtPreprocess" or "AtComputation"
            Splitter='PyPart', # or 'maia', 'PyPart' etc..
            ),

        Flow=dict(
            Density = 0.2,
            Temperature = 100.,
            Velocity = 50.,
        ),

        Turbulence = dict(
            Model = 'SA',
        ),

        Solver=os.environ.get('MOLA_SOLVER'),

        Numerics = dict(
            NumberOfIterations=2,
            CFL=1.,
        ),

        BoundaryConditions=[
            dict(Family='Wall', Type='Wall'),
            dict(Family='Farfield', Type='Farfield'),
        ],

        Extractions=[
            # dict(Type='BC', Source='*', Name='ByFamily', Fields=['Pressure'], ExtractAtEndOfRun=True),
            # dict(Type='BC', Source='WallViscous', Name='ByFamily', Fields=['Pressure'], ExtractAtEndOfRun=True), # FIXME BUG if active
            # dict(Type='IsoSurface', IsoSurfaceField='CoordinateZ', IsoSurfaceValue=1e-6, ExtractAtEndOfRun=True),
            dict(Type='3D', Fields=['PressureStagnation', 'Pressure', 'Mach', 'Entropy'], ExtractAtEndOfRun=True),
            ],

        RunManagement=dict(
            NumberOfProcessors=1,
            RunDirectory=RunDirectory,
            ),
        )

    return w

def get_workflow1():

    x, y, z = np.meshgrid( np.linspace(0,1,21),
                           np.linspace(0,1,21),
                           np.linspace(0,1,21), indexing='ij')
    mesh = cgns.newZoneFromArrays( 'block', ['x','y','z'],
                                            [ x,  y,  z ])

    w = Workflow(
        RawMeshComponents=[
            dict(
                Name='cartesian',
                Source=mesh,
                Mesher=None,
                CleaningMacro=None,
                Families=[
                    dict(Name='Ground',
                         Location='kmin'),
                    dict(Name='Farfield',
                         Location='remaining'),
                ],
                Positioning=[
                    dict(
                        Type='TranslationAndRotation',
                        InitialFrame=dict(
                            Point=[0,0,0],
                            Axis1=[1,0,0],
                            Axis2=[0,1,0],
                            Axis3=[0,0,1]),
                        RequestedFrame=dict(
                            Point=[0,0,0],
                            Axis1=[1,0,0],
                            Axis2=[0,1,0],
                            Axis3=[0,0,1]),
                        ),
                    # dict(
                    #     Type='DuplicateByRotation',
                    #     RotationPoint=[0,0,0],
                    #     RotationAxis=[0,0,1],
                    #     RightHandRuleRotation=True,
                    #     NumberOfInstances=4,
                    #     AddInstancesAsNewComponents=True,
                    #     ),
                ],
                Connection = [
                    dict(Type='Match', Tolerance=1e-8),
                ],
                )
        ],

        SplittingAndDistribution=dict(
            Strategy='AtPreprocess', # "AtPreprocess" or "AtComputation"
            Splitter='Cassiopee', # or 'maia', 'PyPart' etc..
            Distributor='Cassiopee', 
            ComponentsToSplit='all', # 'all', or None or ['first', 'second'...]
            ),


        )

    return w

def get_workflow_cart_monoproc(RunDirectory):

    n_pts_dir = 14
    assert n_pts_dir > 13 # otherwise RSD_L2_rh == 0 and elsa stops at it=1
    x, y, z = np.meshgrid( np.linspace(0,1,n_pts_dir),
                           np.linspace(0,1,n_pts_dir),
                           np.linspace(0,1,n_pts_dir), indexing='ij')
    mesh = cgns.newZoneFromArrays( 'block', ['x','y','z'], [ x,  y,  z ])


    w = Workflow(
        RawMeshComponents=[
            dict(
                Name='cart',
                Source=mesh,
                Families=[
                    dict(Name='Ground',
                         Location='kmin'),
                    dict(Name='Inlet',
                         Location='imin'),
                    dict(Name='Farfield',
                         Location='remaining'),
                ],
                )
        ],

        SplittingAndDistribution=dict(
            Strategy='AtPreprocess', # "AtPreprocess" or "AtComputation"
            Splitter='Cassiopee', # or 'maia', 'PyPart' etc..
            Distributor='Cassiopee', 
            ComponentsToSplit=None, # 'all', or None or ['first', 'second'...]
            ),

        Flow=dict(
            Density = 0.2,
            Temperature = 100.,
            Velocity = 50.,
                 ),

        Turbulence = dict(
            Model = 'SA',
        ),

        Solver=os.environ.get('MOLA_SOLVER'),

        Numerics = dict(
            NumberOfIterations=2,
            CFL=1.0,
        ),

        BoundaryConditions=[
            dict(Family='Ground',   Type='Wall'),
            dict(Family='Inlet',    Type='Farfield'),
            dict(Family='Farfield', Type='Farfield'),
        ],

        RunManagement=dict(
            NumberOfProcessors=1,
            RunDirectory=RunDirectory,
            ),
        )
    
    return w


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_init():
    w = Workflow()


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_print_interface_1():
    w = get_workflow1()
    w.print_interface()


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_submit(tmp_path):
    test_dir = str(tmp_path)
    w = Workflow(RunManagement=dict(
        RunDirectory=test_dir, 
        # Force launching with bash, whatever the machine, this is a simple test
        LauncherCommand=f"cd {test_dir}; bash {names.FILE_JOB}"
        ))
    run_manager.set_default(w.RunManagement)
    os.makedirs(test_dir, exist_ok=True)
    with open(os.path.join(test_dir,'job.sh'),'w') as f:
        f.write('hostname > test.txt')
    w.submit()
    if not os.path.exists(os.path.join(test_dir,'test.txt')):
        raise MolaException('submit test failed')
    shutil.rmtree(test_dir)
    

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_write_tree(tmp_path):
    w = Workflow()
    w.write_tree(str(tmp_path/'main.cgns'))
    try: os.unlink(tmp_path/'main.cgns')
    except: pass


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.mpi
def test_set_workflow_parameters_in_tree_mpi(tmp_path, filename=''):
    w = get_workflow_sphere_struct_dist(tmp_path)
    w.set_workflow_parameters_in_tree()
    if filename: w.write_tree(filename)


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_set_workflow_parameters_in_tree(filename=''):
    w = Workflow()
    w.set_workflow_parameters_in_tree()
    if filename: w.write_tree(filename)

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_get_workflow_parameters_from_tree(filename=''):
    w = Workflow()
    w.set_workflow_parameters_in_tree()
    w.write_tree('test.cgns')
    w = Workflow(tree='test.cgns')
    try: os.unlink('test.cgns')
    except: pass
    if filename: w.write_tree(filename) 

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_equality_between_workflows():
    w = get_workflow1()
    w2 = get_workflow1()
    assert w == w2

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_prepare_assemble_1():
    w = get_workflow1()
    w.assemble()

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_init_with_arg_Mesh():
    w = Workflow(Mesh='mesh.cgns')
    assert w.RawMeshComponents[0]['Source'] == 'mesh.cgns'
    assert not hasattr(w, 'Mesh')
    del w

    w = Workflow(Mesh=dict(Source='mesh.cgns', Name='MyMesh'))
    assert w.RawMeshComponents[0]['Source'] == 'mesh.cgns'
    assert w.RawMeshComponents[0]['Name'] == 'MyMesh'
    assert not hasattr(w, 'Mesh')

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_prepare_assemble_2():
    w = get_workflow2()
    w.assemble()


@pytest.mark.unit
@pytest.mark.cost_level_0
@pytest.mark.mpi
def test_prepare_assemble_dist():
    w = get_workflow_dist()
    w.assemble()


@pytest.mark.cost_level_1
@pytest.mark.integration
def test_process_mesh_workflow1():
    w = get_workflow1()
    w.process_mesh()
    w.write_tree('test.cgns')
    try: os.unlink('test.cgns')
    except: pass

@pytest.mark.integration
@pytest.mark.cost_level_1
def test_prepare_workflow2(tmp_path):
    w = get_workflow2()
    w.RunManagement["RunDirectory"] = str(tmp_path)
    w.prepare()
    w.write_cfd_files()


@pytest.mark.integration
@pytest.mark.cost_level_1
def test_prepare_workflow2_comp2(tmp_path):
    params = get_workflow2_parameters()
    params["RunManagement"]["RunDirectory"] = str(tmp_path)
    params["RunManagement"]["NumberOfProcessors"] = 2
    w = Workflow(**params)
    w.prepare()
    w.write_cfd_files()

# @pytest.mark.integration
# @pytest.mark.cost_level_1
# @pytest.mark.mpi
# def test_prepare_workflow_dist():
#     w = get_workflow_dist()
#     if w.Solver == 'sonics':
#         from mola.cfd.preprocess.boundary_conditions.solver_sonics import adapt_workflow_for_sonics
#         adapt_workflow_for_sonics(w)
#     w.prepare()
#     w.write_cfd_files()


@pytest.mark.integration
@pytest.mark.cost_level_1
def test_workflow_cart_monoproc(tmp_path):
    w = get_workflow_cart_monoproc(tmp_path)
    w.RunManagement['Scheduler'] = 'local'
    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()


@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.cost_level_3
def test_workflow_2_presplit(tmp_path):
    params = get_workflow2_parameters()

    run_management = params["RunManagement"]
    run_management["RunDirectory"] = str(tmp_path)
    run_management["NumberOfProcessors"] = 2
    run_management['Scheduler'] = 'local'

    w = Workflow(**params)
    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()


@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.cost_level_3
def test_workflow_2_cosplit(tmp_path):
    params = get_workflow2_parameters()

    run_management = params["RunManagement"]
    run_management["RunDirectory"] = str(tmp_path)
    run_management["NumberOfProcessors"] = 2
    run_management['Scheduler'] = 'local'
    
    splitting = params["SplittingAndDistribution"]
    splitting["Strategy"] = "AtComputation"
    splitting["Splitter"] = "PyPart"
    splitting["Distributor"] = "PyPart"

    w = Workflow(**params)

    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    
    w.assert_completed_without_errors()



@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.cost_level_3
@pytest.mark.skip(reason="FIXME BUG elsa unstructured+extractionBC+pypart") # FIXME BUG
def test_workflow_2_unstr_cosplit(tmp_path):
    
    params = get_workflow2_parameters()
    
    mesh_comp = params["RawMeshComponents"][0]
    mesh_comp["Source"] = unstructured_cart_grid()
    mesh_comp["Families"] = None
    mesh_comp["Connection"] = None

    run_management = params["RunManagement"]
    run_management["RunDirectory"] = str(tmp_path)
    run_management["NumberOfProcessors"] = 2
    run_management['Scheduler'] = 'local'
    
    splitting = params["SplittingAndDistribution"]
    splitting["Strategy"] = "AtComputation"
    splitting["Splitter"] = "PyPart"
    splitting["Distributor"] = "PyPart"

    w = Workflow(**params)

    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    
    w.assert_completed_without_errors()


@pytest.mark.integration
@pytest.mark.cost_level_3
def test_workflow_sphere_struct_local_monoproc(tmp_path):
    w = get_workflow_sphere_struct(tmp_path)
    w.RunManagement['Scheduler'] = 'local'
    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()

@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.cost_level_3
def test_workflow_sphere_struct_local_monoproc_pypart(tmp_path):
    w = get_workflow_sphere_struct(tmp_path)
    w.RunManagement['Scheduler'] = 'local'
    w.SplittingAndDistribution = dict(
        Splitter = 'PyPart',
        Strategy = 'AtComputation',
    )
    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()

@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.cost_level_3
def test_workflow_sphere_struct_local_monoproc_maia(tmp_path):
    w = get_workflow_sphere_struct(tmp_path)
    w.RunManagement['Scheduler'] = 'local'
    w.SplittingAndDistribution = dict(
        Splitter = 'maia',
        Strategy = 'AtComputation',
    )
    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()

@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.sonics
@pytest.mark.cost_level_3
def test_workflow_sphere_unstruct_local_euler(tmp_path):
    w = get_workflow_sphere_unstruct(tmp_path)
    w.Turbulence['Model'] = 'Euler'
    w.RunManagement['Scheduler'] = 'local'
    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()

@pytest.mark.integration
@pytest.mark.cost_level_3
@pytest.mark.mpi
def test_workflow_sphere_struct_local_cassiopee_mpi(tmp_path):
    w = get_workflow_sphere_struct_cassiopee_mpi_to_connect(tmp_path)
    w.RunManagement['Scheduler'] = 'local'
    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()

@pytest.mark.elsa
@pytest.mark.fast
@pytest.mark.integration
@pytest.mark.cost_level_3
@pytest.mark.mpi
def test_workflow_sphere_struct_local_dist(tmp_path):
    w = get_workflow_sphere_struct_dist(tmp_path)
    w.RunManagement['Scheduler'] = 'local'
    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()

@pytest.mark.integration
@pytest.mark.cost_level_4
def test_workflow_sphere_unstruct_local(tmp_path):
    w = get_workflow_sphere_unstruct(tmp_path)
    try:
        w.prepare()
    except MolaUserError as e:
        if "The mesh should be structured to be used with the solver Fast." in str(e): return
        raise MolaUserError(e)
    w.RunManagement['Scheduler'] = 'local'

    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')
    w.assert_completed_without_errors()

@pytest.mark.integration
@pytest.mark.elsa
@pytest.mark.cost_level_4
@pytest.mark.skip(reason="FIXME BUG elsa unstructured+extractionBC+pypart") # FIXME BUG
def test_workflow_sphere_unstruct_pypart(tmp_path):
    w = get_workflow_sphere_unstruct(tmp_path)
    w.RunManagement['Scheduler'] = 'local'
    w.SplittingAndDistribution["Strategy"] = "AtComputation"
    w.SplittingAndDistribution["Splitter"] = "PyPart"
    w.SplittingAndDistribution["Distributor"] = "PyPart"

    w.prepare()
    w.write_cfd_files()
    w.submit(f'cd {tmp_path}; bash job.sh')

    w.assert_completed_without_errors()

# @pytest.mark.integration
# @pytest.mark.cost_level_3
# def test_workflow_sphere_hybrid_local(tmp_path):
#     # Tested with elsA v5.3.03 and Maia, it works. Other cases are to debug
#     w = get_workflow_sphere_hybrid(tmp_path)
#     if w.Solver == 'sonics':
#         from mola.cfd.preprocess.boundary_conditions.solver_sonics import adapt_workflow_for_sonics
#         adapt_workflow_for_sonics(w)
#     # w.SplittingAndDistribution=dict(Strategy='AtComputation', Splitter='maia')
#     w.prepare()
#     w.write_cfd_files()
#     w.submit(f'cd {tmp_path}; bash job.sh')
#     w.assert_completed_without_errors()

@pytest.mark.network_onera
@pytest.mark.integration
@pytest.mark.cost_level_4
def test_workflow_sphere_struct_remote_sator():
    w = get_workflow_sphere_struct(
        RunDirectory=f'/tmp_user/sator/$USER/.test_workflow_sphere_struct_remote_sator_{os.getenv("MOLA_SOLVER")}/'
    )
    scheduler_defaults = SV.get_scheduler_defaults('sator')
    w.RunManagement['AER'] = scheduler_defaults.AER_FOR_TEST
    w.RunManagement['TimeLimit'] = '00:30:00'

    SV.remove_path(w.RunManagement['RunDirectory'], machine='sator', file_only=False)

    w.prepare()
    w.write_cfd_files()
    w.submit()

    # TODO: 
    # w.simulation_status( wait_until_simulation_end=True ) # TODO implement option
    # SV.remove_path(w.RunManagement['RunDirectory'], machine='sator', file_only=False)

    # NOTE: do not wait for job to end, since that approach would provoke
    # too important delays (waiting for resources of SLURM)
    # COMPLETED_PATH = os.path.join(w.RunManagement['RunDirectory'], names.FILE_JOB_COMPLETED)
    # SV.wait_until(SV.is_existing_path, path=COMPLETED_PATH, machine='sator', timeout=30)
    # SV.remove_path(w.RunManagement['RunDirectory'], machine='sator', file_only=False)

def unstructured_cart_grid():
    from mola.cfd.preprocess.mesh.io.unstructured.solver_sonics import make_mesh_unstructured
    import Converter.PyTree as C

    x, y, z = np.meshgrid( np.linspace(0,1,21),
                           np.linspace(0,1,21),
                           np.linspace(0,1,21), indexing='ij')
    zone = cgns.newZoneFromArrays( 'block', ['x','y','z'], [ x,  y,  z ])
    t = cgns.Tree(Base=[zone])
    C._addBC2Zone(t, 'Ground', 'FamilySpecified:Ground', 'kmin')
    for loc in ['imin','imax','jmin','jmax','kmax']:
        C._addBC2Zone(t, 'Farfield', 'FamilySpecified:Farfield', loc)

    t = cgns.castNode(t)
    t = make_mesh_unstructured(t)

    return t




if __name__ == '__main__':
    # test_workflow_sphere_struct_local_dist()
    # test_prepare_workflow2()
    test_workflow_cart_monoproc('test_workflow_cart_monoproc')
