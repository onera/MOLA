import pytest

import os
import shutil
import copy
import numpy as np
from dataclasses import dataclass

from treelab import cgns

from mola.workflow import Workflow
import mola.workflow.workflow_manager as WM
from mola.logging import check_error_message, MolaException

def get_fake():
    @dataclass
    class Fake(Workflow):

        BoundaryConditions = [
            dict(Family='INFLOW'),
            dict(Family='OUTFLOW', type='OutflowPressure', Pressure=10),
        ]

        RunManagement = dict(
            RunDirectory = 'test',
            other = 3,
        )
    return Fake()

def test_set_value_on_leaf_1():
    d = dict(x=1, y=dict(z=3))
    WM.set_value_on_leaf(d, ['y', 'z'], 2)
    assert d['y']['z'] == 2

def test_set_value_on_leaf_2():
    l = [dict(x=1), dict(y=2, z=3)]
    WM.set_value_on_leaf(l, ['y=2', 'z'], 5)
    assert l[1]['z'] == 5

def test_set_value_on_leaf_3():
    fake = get_fake()
    WM.set_value_on_leaf(fake, ['RunManagement', 'RunDirectory'], 'other_test')
    assert fake.RunManagement['RunDirectory'] == 'other_test'

def test_set_value_on_leaf_4():
    fake = get_fake()
    WM.set_value_on_leaf(fake, ['BoundaryConditions', 'Family=OUTFLOW', 'Pressure'], 20)
    assert fake.BoundaryConditions[1]['Pressure'] == 20

def test_get_value_on_leaf():
    fake = get_fake()
    value = WM.get_value_on_leaf(fake, ['BoundaryConditions', 'Family=OUTFLOW', 'Pressure'])
    assert value == 10


def get_fake_workflow():
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
            ),
        ],

        Flow=dict(
            Density = 0.2,
            Temperature = 100.,
            Velocity = 50.,

        ),

        Solver='elsa',

        Numerics = dict(
            NumberOfIterations=10,
            CFL=1.,
        ),

        BoundaryConditions=[
            dict(Family='Wall', type='Wall'),
            dict(Family='Farfield', type='Farfield', Pressure=10),
        ],

        RunManagement=dict(
            RunDirectory='test_10',
            NumberOfProcessors=4,
            ),

        )

    return w


def test_dispatcher_error_new_job_empty():

    w = object()
    dispatcher = WM.WorkflowDispatcher(w)
    err_msg = 'Before calling `add_variations`, `new_job` must be called first to declare directory.'
    check_error_message(err_msg, dispatcher.add_variations, ['fake'])

def test_dispatcher_directories():

    w = get_fake_workflow()
    dispatcher = WM.WorkflowDispatcher(w)

    dispatcher.new_job('root')
    for index in [10, 20, 30]:
        dispatcher.add_variations([('RunManagement|RunDirectory', f'test_{index}')])

    current_directories = dispatcher.get_directories_in_current_job()
    assert current_directories == ['test_10', 'test_20', 'test_30']

    dispatcher.new_job('root2')
    for index in [40, 50]:
        dispatcher.add_variations([('RunManagement|RunDirectory', f'test_{index}')])

    current_directories = dispatcher.get_directories_in_current_job()
    assert current_directories == ['test_40', 'test_50']
    directories = dispatcher.get_directories()
    assert directories == ['root/test_10', 'root/test_20', 'root/test_30', 'root2/test_40', 'root2/test_50']


def test_WorkflowParallelScheduler_prepare():

    w = get_fake_workflow()
    dispatcher = WM.WorkflowDispatcher(w)

    for model in ['model1', 'model2']:
        dispatcher.new_job(model)
        for pressure in [10, 20, 30]:
            dispatcher.add_variations([('RunManagement|RunDirectory', f'test_{pressure}')])

    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tmp_test_WorkflowParallelScheduler_prepare_root')
    try:
        # remove this directory in case it exists already (e.g. because of a previous error)
        shutil.rmtree(test_dir)
    except FileNotFoundError:
        pass

    scheduler = WM.WorkflowParallelScheduler(dispatcher, test_dir)
    scheduler.prepare()

    root_dirs = []
    files_list = []
    for root, dirs, files in os.walk(test_dir):
        root_dirs.append(root)
        files_list.append(files)

    assert root_dirs == [test_dir, f'{test_dir}/model1', f'{test_dir}/model1/test_10', f'{test_dir}/model1/test_30', f'{test_dir}/model1/test_20', f'{test_dir}/model2', f'{test_dir}/model2/test_10', f'{test_dir}/model2/test_30', f'{test_dir}/model2/test_20']
    assert files_list == [[], ['job_sequence.sh'], ['workflow.cgns'], ['workflow.cgns'], ['workflow.cgns'], ['job_sequence.sh'], ['workflow.cgns'], ['workflow.cgns'], ['workflow.cgns']]
    
    shutil.rmtree(test_dir)


def test_WorkflowParallelScheduler_sphere():

    from mola.workflow.test.test_workflow import get_workflow_sphere_struct
    w = get_workflow_sphere_struct()

    dispatcher = WM.WorkflowDispatcher(w)
    for BCWall in ['WallViscous', 'WallInviscid']:
        dispatcher.new_job(BCWall)
        for velocity in [50., 20., 80.]:
            dispatcher.add_variations(
                [
                    ('RunManagement|RunDirectory', f'Velocity_{velocity}'),
                    ('Flow|Velocity', velocity),
                    ('BoundaryConditions|Family=Wall|type', BCWall),
                ], 
                initialize_from_previous=False
                )
    
    test_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'tmp_test_WorkflowParallelScheduler_root')
    try:
        # remove this directory in case it exists already (e.g. because of a previous error)
        shutil.rmtree(test_dir)
    except FileNotFoundError:
        pass
        
    scheduler = WM.WorkflowParallelScheduler(dispatcher, test_dir)
    scheduler.prepare()
    scheduler.submit()

    for BCWall in ['WallViscous', 'WallInviscid']:
        for velocity in [50., 20., 80.]:
            COMPLETED_PATH = os.path.join(scheduler.root_directory, BCWall, f'Velocity_{velocity}', 'COMPLETED')
            if not os.path.exists(COMPLETED_PATH):
                raise MolaException(f'simulation did not ended as expected: unable to found file {COMPLETED_PATH}')

    shutil.rmtree(test_dir)

