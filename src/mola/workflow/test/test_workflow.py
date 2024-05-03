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
from mola.workflow import Workflow
from mola.logging import mola_logger, MolaException, mute_stdout
from mola import server as SV
from mola.cfd.preprocess.write_cfd_files.write_cfd_files import set_default


def get_workflow2():

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
            NumberOfProcessors=4, 
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
            dict(Type='Integral', Name='AeroCoefs', Fields=['CL', 'std-CL']),
            dict(Type='Probe', Name='probe1', Fields=['std-Pressure'], SavePeriod=5),
            dict(Type='Probe', Name='probe2', Fields=['std-Density'], SavePeriod=5),
            dict(Type='3D', Fields=['Mach', 'q_criterion']),
            dict(Type='BC', Source='BCWall*', Name='ByFamily', Fields=['normalvector', 'frictionvector']),
            dict(Type='BC', Source='*', Name='ByFamily', Fields=['Pressure']),
            dict(Type='IsoSurface', Name='MySurface', IsoSurfaceField='CoordinateY', IsoSurfaceValue=1.e-6, Fields=['Mach','cellN']),
            ],


        )
    return w

def get_workflow_sphere_struct():
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

        SplittingAndDistribution=dict(
            Strategy='AtPreprocess', # "AtPreprocess" or "AtComputation"
            Splitter='Cassiopee', # or 'maia', 'PyPart' etc..
            Distributor='Cassiopee', 
            ComponentsToSplit='all', # 'all', or None or ['first', 'second'...]
            NumberOfProcessors=1, 
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
            dict(Type='BC', Source='BCWall*', Name='ByFamily', Fields=['NormalVector', 'Friction', 'BoundaryLayer']),
            dict(Type='IsoSurface', Name='MySurface', IsoSurfaceField='CoordinateZ', IsoSurfaceValue=1.e-6),
            ],

        RunManagement=dict(
            NumberOfProcessors=1,
            RunDirectory=os.path.dirname(os.path.realpath(__file__)),
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
                    dict(
                        Type='DuplicateByRotation',
                        RotationPoint=[0,0,0],
                        RotationAxis=[0,0,1],
                        RightHandRuleRotation=True,
                        NumberOfInstances=4,
                        AddInstancesAsNewComponents=True,
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
            NumberOfProcessors=4, 
            # MinimumAllowedNodes=1,
            # MaximumAllowedNodes=20,
            # MaximumNumberOfPointsPerNode=1e9,
            # CoresPerNode=48,
            # DistributeExclusivelyOnFullNodes=True,
            ),


        )

    return w

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_init():
    w = Workflow()


@pytest.mark.unit
@pytest.mark.cost_level_0
def test_submit():
    test_dir = 'test_submit_dir'
    os.makedirs(test_dir, exist_ok=True)
    w = Workflow(RunManagement=dict(RunDirectory=test_dir))
    set_default(w.RunManagement)
    SV.job_writer.set_launcher_command(w.RunManagement)
    with open(os.path.join(test_dir,'job.sh'),'w') as f:
        f.write('hostname > test.txt')
    w.submit()
    if not os.path.exists(os.path.join(test_dir,'test.txt')):
        raise MolaException('submit test failed')
    shutil.rmtree(test_dir)
    

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_write_tree():
    w = Workflow()
    w.write_tree('main.cgns')
    os.unlink('main.cgns')


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
    w.tree = 'test.cgns'
    w.get_workflow_parameters_from_tree()
    os.unlink('test.cgns')
    if filename: w.write_tree(filename)    


@pytest.mark.cost_level_1
@pytest.mark.integration
def test_prepare_workflow1():
    w = get_workflow1()
    w.assemble()
    w.positioning()
    w.connect()
    w.define_families()
    w.split_and_distribute()
    w.tree.save('test.cgns')
    os.unlink('test.cgns')
    
@pytest.mark.integration
@pytest.mark.cost_level_1
def test_prepare_workflow2():
    w = get_workflow2()
    w.prepare()
    w.write_cfd_files()
    w.remove_cfd_files()

@pytest.mark.integration
@pytest.mark.cost_level_3
def test_workflow_sphere_struct_local():
    w = get_workflow_sphere_struct()
    w.prepare()
    w.write_cfd_files()
    w.submit()
    w.simulation_status()
    w.remove_cfd_files()


@pytest.mark.network_onera
@pytest.mark.integration
@pytest.mark.cost_level_4
def test_workflow_sphere_struct_remote_sator():
    w = get_workflow_sphere_struct()
    w.RunManagement['RunDirectory'] = f'/tmp_user/sator/$USER/.test/test_workflow_sphere_struct_remote_sator/'
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

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_print_interface_1():
    w = get_workflow1()
    w.print_interface()

@pytest.mark.unit
@pytest.mark.cost_level_0
def test_submit():
    test_dir = 'test_submit_dir'
    os.makedirs(test_dir, exist_ok=True)
    w = Workflow(RunManagement=dict(RunDirectory=test_dir))
    set_default(w.RunManagement)
    SV.job_writer.set_launcher_command(w.RunManagement)
    with open(os.path.join(test_dir,names.FILE_JOB),'w') as f:
        f.write('hostname > test.txt')
    w.submit()
    if not os.path.exists(os.path.join(test_dir,'test.txt')):
        raise MolaException('submit test failed')
    shutil.rmtree(test_dir)
    


def test_wip():
    import inspect

    def repack_kwargs_only(**kwargs):
        # Get the current frame (frame where this function is called)
        frame = inspect.currentframe().f_back
        # Get the arguments from the calling frame
        locals_dict = frame.f_locals
        # Remove 'self' if this is a method in a class
        locals_dict.pop("self", None)
        # Remove 'kwargs' if it exists
        locals_dict.pop("kwargs", None)
        # Repack only kwargs
        kwargs = {key: locals_dict[key] for key in locals_dict if key not in locals_dict.get("args", [])}
        return kwargs

    # Example usage:
    def example_function(a, b, c, d=1, e=2, *, f=None, g=None):
        kwargs = repack_kwargs_only()
        return kwargs

    result = example_function(1, 2, 3, g='value')
    print("Keyword arguments:", result)
    
    

if __name__ == '__main__':
    test_workflow_sphere_struct_remote_sator()
