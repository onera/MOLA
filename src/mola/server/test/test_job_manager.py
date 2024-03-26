import pytest
import copy
import numpy as np
from dataclasses import dataclass

from treelab import cgns

from mola.workflow import Workflow
import mola.server.job_manager as JM
from mola.logging import check_error_message

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
    JM.set_value_on_leaf(d, ['y', 'z'], 2)
    assert d['y']['z'] == 2

def test_set_value_on_leaf_2():
    l = [dict(x=1), dict(y=2, z=3)]
    JM.set_value_on_leaf(l, ['y=2', 'z'], 5)
    assert l[1]['z'] == 5

def test_set_value_on_leaf_3():
    fake = get_fake()
    JM.set_value_on_leaf(fake, ['RunManagement', 'RunDirectory'], 'other_test')
    assert fake.RunManagement['RunDirectory'] == 'other_test'

def test_set_value_on_leaf_4():
    fake = get_fake()
    JM.set_value_on_leaf(fake, ['BoundaryConditions', 'Family=OUTFLOW', 'Pressure'], 20)
    assert fake.BoundaryConditions[1]['Pressure'] == 20

def test_get_value_on_leaf():
    fake = get_fake()
    value = JM.get_value_on_leaf(fake, ['BoundaryConditions', 'Family=OUTFLOW', 'Pressure'])
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
    dispatcher = JM.WorkflowDispatcher(w)
    err_msg = 'Before calling `add_variations`, `new_job` must be called first to declare directory.'
    check_error_message(err_msg, dispatcher.add_variations, ['fake'])

def test_dispatcher_directories():

    w = get_fake_workflow()
    dispatcher = JM.WorkflowDispatcher(w)

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



# w = get_fake_workflow()
# dispatcher = JM.WorkflowDispatcher(w)

# for model in ['model1', 'model2']:
#     dispatcher.new_job(model)
#     for pressure in [10, 20, 30]:
#         dispatcher.add_variations([('RunManagement|RunDirectory', f'test_{pressure}')])

# print(dispatcher.root_directories)
# print(dispatcher.get_directories())
# scheduler = JM.WorkflowParallelScheduler(dispatcher, 'root_parallel')
# scheduler.prepare()
# # scheduler.submit()